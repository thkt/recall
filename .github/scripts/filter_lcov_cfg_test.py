#!/usr/bin/env python3
"""Remove Rust `#[cfg(test)]` item ranges from an lcov tracefile.

`diff-cover` works on changed lines from the lcov `DA:` entries. Inline Rust
test modules live in production source files, so filename-based exclusions do
not remove them. This filter drops coverage records for any source range guarded
by `#[cfg(test)]`, matching the repo policy that test code is excluded from the
diff coverage gate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _scan_line(line: str, state):
    """Count code-context braces in one line, carrying string state across lines.

    `state` is None in code context, "str" inside a normal "..." string, or an
    int n inside a raw string with n hashes (carried across lines so multi-line
    strings are handled). Braces inside string literals, char literals, and line
    comments are ignored so a stray `{` in `write!("{{")` or `'{'` does not
    corrupt the cfg(test) range scan. Block comments (`/* */`) are intentionally
    unhandled: no recall source uses them inside `#[cfg(test)]` items. A brace in
    one is counted as code, so an unbalanced `/* { */` over-extends a range
    exactly as the pre-refactor raw counter did -- never worse, but not fixed.
    Returns (delta, state).
    """
    delta = 0
    i = 0
    n = len(line)
    while i < n:
        c = line[i]
        if state == "str":
            if c == "\\":
                i += 2
            elif c == '"':
                state = None
                i += 1
            else:
                i += 1
            continue
        if isinstance(state, int):  # raw string with `state` hashes
            if c == '"' and line[i + 1 : i + 1 + state] == "#" * state:
                i += 1 + state
                state = None
            else:
                i += 1
            continue
        if c == '"':  # normal string opens
            state = "str"
            i += 1
        elif c == "r" and i + 1 < n and line[i + 1] in '#"':  # raw string?
            j = i + 1
            while j < n and line[j] == "#":
                j += 1
            if j < n and line[j] == '"':
                state = j - (i + 1)  # hash count (0 for r"...")
                i = j + 1
            else:
                i += 1  # raw identifier r#ident, not a raw string
        elif c == "'":
            if i + 1 < n and line[i + 1] == "\\":  # escaped char '\n' '\u{7b}'
                j = i + 3
                while j < n and line[j] != "'":
                    j += 1
                i = j + 1
            elif i + 2 < n and line[i + 2] == "'":  # simple char 'X' incl '{'
                i += 3
            else:  # lifetime 'a / 'static: treat ' as ordinary
                i += 1
        elif c == "/" and i + 1 < n and line[i + 1] == "/":  # line comment
            break
        elif c == "{":
            delta += 1
            i += 1
        elif c == "}":
            delta -= 1
            i += 1
        else:
            i += 1
    return delta, state


def brace_delta(line: str) -> int:
    # Single-line net brace count in code context (no carried state). Thin
    # wrapper over the stateful core for callers that scan a line in isolation.
    delta, _ = _scan_line(line, None)
    return delta


def cfg_test_ranges(path: Path) -> set[int]:
    if path.suffix != ".rs" or not path.exists():
        return set()

    lines = path.read_text(encoding="utf-8").splitlines()
    ignored: set[int] = set()
    idx = 0
    while idx < len(lines):
        if lines[idx].strip() != "#[cfg(test)]":
            idx += 1
            continue

        start = idx + 1
        item_idx = idx + 1
        while item_idx < len(lines) and not lines[item_idx].strip():
            item_idx += 1
        if item_idx >= len(lines):
            ignored.add(start)
            break

        balance = 0
        end_idx = item_idx
        saw_block = False
        state = None
        while end_idx < len(lines):
            delta, state = _scan_line(lines[end_idx], state)
            balance += delta
            if balance > 0:
                saw_block = True
            if saw_block and balance <= 0:
                break
            if not saw_block:
                break
            end_idx += 1

        ignored.update(range(start, end_idx + 2))
        idx = end_idx + 1

    return ignored


# The lcov record parsers below are tightly coupled to cargo llvm-cov's line
# format (DA:<line>,<hit>  FN:<line>,<name>  FNDA:<count>,<name>). A parse failure
# means that format contract is violated (tool upgrade, truncation, corruption),
# so the filter's output can no longer be trusted. Fail loud rather than default
# silently: a None/0 default would skew the gate in an unverifiable direction (a
# dropped line number leaves test code in; a defaulted hit count marks a covered
# line uncovered). This differs from ENC-002/OPS-006, which guard against
# gate-weakening specifically; here the contract itself is the invariant.
def parse_line_number(record: str) -> int:
    try:
        return int(record.split(":", 1)[1].split(",", 1)[0])
    except (IndexError, ValueError) as e:
        raise ValueError(f"malformed lcov record: {record!r}") from e


def parse_fn(record: str) -> tuple[int, str]:
    try:
        payload = record.split(":", 1)[1]
        line, name = payload.split(",", 1)
        return int(line), name
    except (IndexError, ValueError) as e:
        raise ValueError(f"malformed lcov FN record: {record!r}") from e


def parse_fnda(record: str) -> tuple[int, str]:
    try:
        payload = record.split(":", 1)[1]
        count, name = payload.split(",", 1)
        return int(count), name
    except (IndexError, ValueError) as e:
        raise ValueError(f"malformed lcov FNDA record: {record!r}") from e


def render_record(record: list[str], ignored: set[int]) -> list[str]:
    output: list[str] = []
    skipped_functions: set[str] = set()
    fn_count = fn_hit = line_count = line_hit = branch_count = branch_hit = 0
    saw_fn = saw_branch = False

    for entry in record:
        if entry.startswith(("LF:", "LH:", "FNF:", "FNH:", "BRF:", "BRH:")):
            continue

        if entry.startswith("DA:"):
            line_no = parse_line_number(entry)
            if line_no in ignored:
                continue
            line_count += 1
            try:
                hit_count = int(entry.split(",", 2)[1])
            except (IndexError, ValueError) as e:
                raise ValueError(f"malformed lcov DA record: {entry!r}") from e
            if hit_count > 0:
                line_hit += 1
            output.append(entry)
            continue

        if entry.startswith("BRDA:"):
            line_no = parse_line_number(entry)
            if line_no in ignored:
                continue
            saw_branch = True
            branch_count += 1
            taken = entry.rsplit(",", 1)[-1]
            if taken not in {"-", "0"}:
                branch_hit += 1
            output.append(entry)
            continue

        if entry.startswith("FN:"):
            line_no, name = parse_fn(entry)
            if line_no in ignored:
                skipped_functions.add(name)
                continue
            saw_fn = True
            fn_count += 1
            output.append(entry)
            continue

        if entry.startswith("FNDA:"):
            hit_count, name = parse_fnda(entry)
            if name in skipped_functions:
                continue
            if hit_count > 0:
                fn_hit += 1
            output.append(entry)
            continue

        output.append(entry)

    if saw_fn:
        output.extend([f"FNF:{fn_count}", f"FNH:{fn_hit}"])
    output.extend([f"LF:{line_count}", f"LH:{line_hit}"])
    if saw_branch:
        output.extend([f"BRF:{branch_count}", f"BRH:{branch_hit}"])
    output.append("end_of_record")
    return output


def _is_under(path: Path, root: Path) -> bool:
    try:
        return path.resolve().is_relative_to(root.resolve())
    except (OSError, ValueError):
        return False


def _check_source_resolvable(source: Path, repo_root: Path) -> bool:
    # Decide whether `source` should be scanned for cfg(test) ranges.
    #   non-.rs          -> scan (cfg_test_ranges returns set() anyway).
    #   .rs outside root -> skip WITHOUT reading. An external/registry dep cannot
    #                       affect the gate (diff-cover scores only this repo's
    #                       changed lines), and skipping avoids reading an
    #                       out-of-tree path even when it exists (a traversed SF).
    #   .rs under root,
    #     missing        -> OPS-006: SF path resolution is broken and test code
    #                       would leak into the gate. Fail loud.
    if source.suffix != ".rs":
        return True
    if not _is_under(source, repo_root):
        print(f"warning: skipping out-of-root source: {source}", file=sys.stderr)
        return False
    if not source.exists():
        raise FileNotFoundError(
            f"SF source under repo root does not exist: {source} (repo_root={repo_root}). "
            "Path resolution is broken; test code would leak into the coverage gate."
        )
    return True


def _validate_filtered(filtered: list[str], sf_count: int, input_path: Path) -> None:
    # ENC-002: diff-cover treats an empty / record-less tracefile as 100%
    # covered, so an over-aggressive filter that drops every record would
    # silently disable the gate. Require at least one SF and one end_of_record
    # per SF. The mismatch branch is defense-in-depth: it backstops the invariant
    # that the filtered output carries exactly one end_of_record per SF, catching
    # any upstream record-accounting bug regardless of cause (the orphan check
    # above already handles a truncated trailing record).
    if sf_count == 0:
        raise ValueError(
            f"filtered lcov has no SF records; coverage input {input_path} "
            "was empty or unparsable"
        )
    eor_count = filtered.count("end_of_record")
    if eor_count != sf_count:
        raise ValueError(
            f"filtered lcov record mismatch for {input_path}: "
            f"{sf_count} SF but {eor_count} end_of_record"
        )


def filter_lcov(input_path: Path, output_path: Path, repo_root: Path) -> None:
    filtered: list[str] = []
    record: list[str] = []
    ignored: set[int] = set()
    sf_count = 0

    for line in input_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("SF:"):
            sf_count += 1
            record = [line]
            source = Path(line[3:])
            if not source.is_absolute():
                source = repo_root / source
            if _check_source_resolvable(source, repo_root):
                ignored = cfg_test_ranges(source)
            else:
                ignored = set()
            continue

        if line == "end_of_record":
            filtered.extend(render_record(record, ignored))
            record = []
            ignored = set()
            continue

        if record:
            record.append(line)
        else:
            filtered.append(line)

    if record:
        raise ValueError(
            f"truncated lcov: SF record not terminated by end_of_record in {input_path}"
        )
    _validate_filtered(filtered, sf_count, input_path)
    output_path.write_text("\n".join(filtered) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    filter_lcov(args.input, args.output, args.repo_root)


if __name__ == "__main__":
    main()
