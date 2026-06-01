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
from pathlib import Path


def brace_delta(line: str) -> int:
    # NOTE: counts raw `{`/`}` including those inside string literals, char
    # literals, and comments. This miscounts lines like `write!(buf, "{{")`, but
    # no current recall source triggers it (multi-line raw-string JSON braces in
    # the tests are balanced and net to zero). A correct fix needs a Rust
    # tokenizer; tracked as a followup. See test_filter_lcov_cfg_test.py xfails.
    return line.count("{") - line.count("}")


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
        while end_idx < len(lines):
            balance += brace_delta(lines[end_idx])
            if "{" in lines[end_idx]:
                saw_block = True
            if saw_block and balance <= 0:
                break
            if not saw_block:
                break
            end_idx += 1

        ignored.update(range(start, end_idx + 2))
        idx = end_idx + 1

    return ignored


def parse_line_number(record: str) -> int | None:
    try:
        return int(record.split(":", 1)[1].split(",", 1)[0])
    except (IndexError, ValueError):
        return None


def parse_fn(record: str) -> tuple[int | None, str]:
    try:
        payload = record.split(":", 1)[1]
        line, name = payload.split(",", 1)
        return int(line), name
    except (IndexError, ValueError):
        return None, ""


def parse_fnda(record: str) -> tuple[int, str]:
    try:
        payload = record.split(":", 1)[1]
        count, name = payload.split(",", 1)
        return int(count), name
    except (IndexError, ValueError):
        return 0, ""


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
            except (IndexError, ValueError):
                hit_count = 0
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


def filter_lcov(input_path: Path, output_path: Path, repo_root: Path) -> None:
    filtered: list[str] = []
    record: list[str] = []
    ignored: set[int] = set()

    for line in input_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("SF:"):
            record = [line]
            source = Path(line[3:])
            if not source.is_absolute():
                source = repo_root / source
            ignored = cfg_test_ranges(source)
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
