"""Unit tests for filter_lcov_cfg_test.py (the coverage-gate cfg(test) filter).

Run: python3 -m pytest .github/scripts/test_filter_lcov_cfg_test.py

The filter sits between `cargo llvm-cov` and `diff-cover`, so a brace-counting
bug that misclassifies production code as test code silently weakens the 95%
gate. These tests pin the token-aware brace scanner (string / char / line-comment
and multi-line raw-string braces are excluded) and the fail-loud guards on
unresolved SF paths and empty filtered output.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from filter_lcov_cfg_test import (  # noqa: E402
    brace_delta,
    cfg_test_ranges,
    filter_lcov,
)


def test_brace_delta_counts_structural_braces():
    assert brace_delta("mod tests {") == 1
    assert brace_delta("}") == -1
    assert brace_delta("fn f() {}") == 0


def test_brace_delta_ignores_braces_in_string_literals():
    assert brace_delta('    write!(buf, "{{")?;') == 0
    assert brace_delta('    let s = "}";') == 0


def test_brace_delta_ignores_char_literal_and_line_comment_braces():
    assert brace_delta("    let c = '{';") == 0
    assert brace_delta("    foo(); // a trailing { brace") == 0


def test_cfg_test_ranges_brace_in_string_does_not_swallow_following_production(tmp_path):
    # An unbalanced brace in a test string literal must not extend the ignored
    # range into the production fn that follows. The token-aware scanner ignores
    # the string-literal brace, so the range ends at the test module's close.
    src = tmp_path / "x.rs"
    src.write_text(
        "#[cfg(test)]\n"  # 1
        "mod tests {\n"  # 2
        '    fn t() { assert_eq!(x, "{"); }\n'  # 3  unbalanced `{` in string
        "}\n"  # 4
        "pub fn prod() -> i32 { 1 }\n",  # 5  production AFTER
        encoding="utf-8",
    )
    ranges = cfg_test_ranges(src)
    assert {1, 2, 3, 4} <= ranges
    assert 5 not in ranges


def test_cfg_test_ranges_excludes_balanced_test_module(tmp_path):
    # The common case in recall: a `#[cfg(test)] mod tests` whose braces are
    # balanced (no stray string-literal brace). Raw-count handles this correctly
    # and is what keeps real source files (embedder/indexer/main) at 100%.
    src = tmp_path / "x.rs"
    src.write_text(
        "pub fn prod() -> i32 {\n"  # 1 production
        "    1\n"  # 2
        "}\n"  # 3
        "\n"  # 4
        "#[cfg(test)]\n"  # 5
        "mod tests {\n"  # 6
        "    fn t() {}\n"  # 7
        "}\n",  # 8
        encoding="utf-8",
    )
    ranges = cfg_test_ranges(src)
    assert 1 not in ranges and 2 not in ranges and 3 not in ranges
    assert {6, 7, 8} <= ranges


def test_cfg_test_ranges_balanced_multiline_raw_string(tmp_path):
    # Regression guard for the indexer.rs breakage: a multi-line raw string with
    # balanced JSON braces inside the test module must NOT truncate the range.
    # Raw-count passes because the JSON braces net to zero.
    src = tmp_path / "x.rs"
    src.write_text(
        "#[cfg(test)]\n"  # 1
        "mod tests {\n"  # 2
        "    fn t() {\n"  # 3
        '        let s = r#"{"a":{"b":1}}\n'  # 4  multi-line raw string opens
        '{"c":2}"#;\n'  # 5  ...and closes; JSON braces balanced
        "        assert!(s.len() > 0);\n"  # 6
        "    }\n"  # 7
        "}\n",  # 8
        encoding="utf-8",
    )
    ranges = cfg_test_ranges(src)
    assert {6, 7, 8} <= ranges, "balanced multi-line raw string must not truncate the range"


def test_cfg_test_ranges_non_rs_file_returns_empty(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("#[cfg(test)]\nmod tests {}\n", encoding="utf-8")
    assert cfg_test_ranges(p) == set()


def test_cfg_test_ranges_missing_file_returns_empty(tmp_path):
    assert cfg_test_ranges(tmp_path / "nonexistent.rs") == set()


def test_brace_delta_lifetime_not_treated_as_char():
    # 'a and 'static are lifetimes, not char literals; the scanner must not
    # consume the following code as a char and must still count real braces.
    assert brace_delta("fn f<'a>(x: &'a str) {") == 1
    assert brace_delta("impl<'a> Foo<'a> {}") == 0


def test_brace_delta_char_escape_with_brace_in_codepoint():
    # '\u{7b}' is the codepoint escape for '{'; the brace lives inside the char
    # escape and must not be counted, while a real trailing brace is.
    assert brace_delta(r"    let c = '\u{7b}'; if x {") == 1
    assert brace_delta(r"    if c == '\x1b' { f(); }") == 0


def test_cfg_test_ranges_unbalanced_brace_in_multiline_raw_string(tmp_path):
    # An unbalanced brace inside a multi-line raw string must not leak the
    # ignored range into the production fn that follows (raw-string state is
    # carried across lines).
    src = tmp_path / "x.rs"
    src.write_text(
        "#[cfg(test)]\n"  # 1
        "mod tests {\n"  # 2
        "    fn t() {\n"  # 3
        '        let s = r#"{{{ unbalanced\n'  # 4  raw string opens, stray {
        '        still raw }"#;\n'  # 5  ...closes; braces were inside the raw string
        "        assert!(s.len() > 0);\n"  # 6
        "    }\n"  # 7
        "}\n"  # 8
        "pub fn prod() -> i32 { 1 }\n",  # 9  production AFTER
        encoding="utf-8",
    )
    ranges = cfg_test_ranges(src)
    assert {1, 2, 3, 4, 5, 6, 7, 8} <= ranges
    assert 9 not in ranges


def test_filter_lcov_raises_on_unresolved_repo_source(tmp_path):
    # A `.rs` SF path under repo_root that does not exist is a path
    # resolution bug; fail loud rather than silently under-filtering.
    lcov = tmp_path / "in.info"
    lcov.write_text("SF:src/ghost.rs\nDA:1,1\nend_of_record\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        filter_lcov(lcov, tmp_path / "out.info", tmp_path)


def test_filter_lcov_warns_and_skips_external_source(tmp_path, capsys):
    # An absolute `.rs` outside repo_root (external dep) cannot
    # affect the gate (diff-cover scores only this repo's changed lines); it is
    # skipped without reading and the record is passed through unfiltered.
    repo = tmp_path / "repo"
    repo.mkdir()
    lcov = repo / "in.info"
    lcov.write_text(
        "SF:/nonexistent/external/dep.rs\nDA:1,1\nend_of_record\n", encoding="utf-8"
    )
    out = repo / "out.info"
    filter_lcov(lcov, out, repo)
    assert "skipping out-of-root source" in capsys.readouterr().err
    text = out.read_text(encoding="utf-8")
    assert "SF:/nonexistent/external/dep.rs" in text  # record preserved, not dropped
    assert "DA:1,1" in text  # skip-not-drop: line kept unfiltered
    assert text.count("end_of_record") == 1


def test_filter_lcov_raises_on_empty_input(tmp_path):
    # An input with no SF records would yield a trivially-passing
    # filtered file; fail loud.
    lcov = tmp_path / "in.info"
    lcov.write_text("TN:\n", encoding="utf-8")
    with pytest.raises(ValueError):
        filter_lcov(lcov, tmp_path / "out.info", tmp_path)


def test_filter_lcov_emits_one_end_of_record_per_sf(tmp_path):
    # Well-formedness: one end_of_record per SF, production lines kept.
    (tmp_path / "real.rs").write_text(
        "pub fn f() -> i32 {\n    1\n}\n", encoding="utf-8"
    )
    lcov = tmp_path / "in.info"
    lcov.write_text(
        "SF:real.rs\nDA:1,1\nDA:2,1\nDA:3,1\nend_of_record\n", encoding="utf-8"
    )
    out = tmp_path / "out.info"
    filter_lcov(lcov, out, tmp_path)
    text = out.read_text(encoding="utf-8")
    assert text.count("end_of_record") == 1
    assert "SF:real.rs" in text
    assert "DA:1,1" in text  # production line retained


def test_filter_lcov_skips_existing_out_of_root_source(tmp_path, capsys):
    # SEC: an EXISTING `.rs` resolved outside repo_root (e.g. a traversed SF
    # path) is skipped without reading. The pre-fix code read it because
    # _is_under was consulted only on the missing-file branch.
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret.rs"
    secret.write_text("#[cfg(test)]\nmod tests {\n}\n", encoding="utf-8")
    lcov = repo / "in.info"
    lcov.write_text(f"SF:{secret}\nDA:1,1\nend_of_record\n", encoding="utf-8")
    out = repo / "out.info"
    filter_lcov(lcov, out, repo)
    assert "skipping out-of-root source" in capsys.readouterr().err
    # record passes through unfiltered (file was not read for cfg(test) ranges)
    assert "DA:1,1" in out.read_text(encoding="utf-8")


def test_filter_lcov_raises_on_orphan_record(tmp_path):
    # Item 2 (EOF-orphan): an SF record with no terminating end_of_record
    # (truncated input) leaves a non-empty record after the loop. The explicit
    # orphan check fires -- earlier and with a clearer message than _validate's
    # SF/end_of_record mismatch guard (which now only backs render_record's
    # one-end_of_record-per-SF invariant). The match pins the orphan branch.
    (tmp_path / "real.rs").write_text("pub fn f() -> i32 { 1 }\n", encoding="utf-8")
    lcov = tmp_path / "in.info"
    lcov.write_text("SF:real.rs\nDA:1,1\n", encoding="utf-8")  # no end_of_record
    with pytest.raises(ValueError, match="not terminated by end_of_record"):
        filter_lcov(lcov, tmp_path / "out.info", tmp_path)


def test_filter_lcov_raises_on_mid_stream_orphan(tmp_path):
    # _validate defense-in-depth (reachable): a record lacking end_of_record
    # before the next SF (mid-stream orphan, not a trailing truncation) is missed
    # by the orphan check -- the new SF resets record -- but trips _validate's
    # SF / end_of_record count mismatch (2 SF, 1 end_of_record). Pins the backstop
    # branch as reachable, refuting a "dead code" reading.
    (tmp_path / "a.rs").write_text("pub fn a() -> i32 { 1 }\n", encoding="utf-8")
    (tmp_path / "b.rs").write_text("pub fn b() -> i32 { 1 }\n", encoding="utf-8")
    lcov = tmp_path / "in.info"
    lcov.write_text(
        "SF:a.rs\nDA:1,1\nSF:b.rs\nDA:1,1\nend_of_record\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="record mismatch"):
        filter_lcov(lcov, tmp_path / "out.info", tmp_path)


def test_brace_delta_byte_strings_ignore_braces():
    # Byte string b"{" and raw byte string br#"}"# keep their brace inside the
    # string; the scanner consumes `b` as ordinary then enters string state.
    assert brace_delta('    let b = b"{"; foo() {') == 1
    assert brace_delta('    let r = br#"}"#; bar() }') == -1


def test_brace_delta_escaped_quote_char():
    # '\'' is an escaped-single-quote char literal; the scanner must skip it and
    # still count a real trailing brace.
    assert brace_delta(r"    let q = '\''; baz() {") == 1


def _cfg_test_src_with_ignored_line5(tmp_path: Path) -> Path:
    # A source whose cfg(test) block (lines 3-6) makes line 5 an ignored line,
    # while line 1 is production. Shared by the render_record FN/BRDA tests.
    src = tmp_path / "real.rs"
    src.write_text(
        "pub fn prod() -> i32 { 1 }\n"  # 1 production
        "\n"  # 2
        "#[cfg(test)]\n"  # 3
        "mod tests {\n"  # 4
        "    fn t() {}\n"  # 5 ignored
        "}\n",  # 6
        encoding="utf-8",
    )
    return src


def test_render_record_drops_fn_and_fnda_for_ignored_function(tmp_path):
    # render_record FN/FNDA path: a FN declared on an ignored (cfg(test)) line is
    # dropped and its name recorded in skipped_functions, so the matching FNDA is
    # also dropped. FNF/FNH reflect only the surviving production function.
    _cfg_test_src_with_ignored_line5(tmp_path)
    lcov = tmp_path / "in.info"
    lcov.write_text(
        "SF:real.rs\n"
        "FN:1,prod\n"
        "FN:5,t\n"  # line 5 ignored -> recorded in skipped_functions
        "FNDA:3,prod\n"
        "FNDA:0,t\n"  # name in skipped_functions -> dropped
        "DA:1,3\n"
        "DA:5,0\n"  # ignored line -> dropped
        "end_of_record\n",
        encoding="utf-8",
    )
    out = tmp_path / "out.info"
    filter_lcov(lcov, out, tmp_path)
    text = out.read_text(encoding="utf-8")
    assert "FN:1,prod" in text and "FNDA:3,prod" in text
    assert "FN:5,t" not in text  # ignored-line FN dropped
    assert "FNDA:0,t" not in text  # FNDA for a skipped function dropped
    assert "FNF:1" in text and "FNH:1" in text  # one production fn, hit
    assert "DA:5,0" not in text  # ignored DA dropped


def test_render_record_branch_coverage_taken_and_ignored(tmp_path):
    # render_record BRDA path: taken "-"/"0" is not a hit, any other value is; a
    # BRDA on an ignored line is dropped. BRF/BRH count only surviving branches.
    _cfg_test_src_with_ignored_line5(tmp_path)
    lcov = tmp_path / "in.info"
    lcov.write_text(
        "SF:real.rs\n"
        "DA:1,3\n"
        "BRDA:1,0,0,3\n"  # taken 3 -> hit
        "BRDA:1,0,1,0\n"  # taken 0 -> not hit
        "BRDA:1,0,2,-\n"  # taken - -> not hit
        "BRDA:5,0,0,1\n"  # line 5 ignored -> dropped
        "end_of_record\n",
        encoding="utf-8",
    )
    out = tmp_path / "out.info"
    filter_lcov(lcov, out, tmp_path)
    text = out.read_text(encoding="utf-8")
    assert "BRDA:1,0,0,3" in text and "BRDA:1,0,1,0" in text and "BRDA:1,0,2,-" in text
    assert "BRDA:5,0,0,1" not in text  # ignored-line branch dropped
    assert "BRF:3" in text  # three surviving branches on line 1
    assert "BRH:1" in text  # only the taken=3 branch is a hit


def test_filter_lcov_raises_on_malformed_da_line_number(tmp_path):
    # Item 1 (format-contract): a DA record whose line field is non-numeric means
    # the cargo llvm-cov format assumption is violated; fail loud instead of
    # silently passing the record through with a None line number.
    (tmp_path / "real.rs").write_text("pub fn f() -> i32 { 1 }\n", encoding="utf-8")
    lcov = tmp_path / "in.info"
    lcov.write_text("SF:real.rs\nDA:abc,1\nend_of_record\n", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed lcov record"):
        filter_lcov(lcov, tmp_path / "out.info", tmp_path)


def test_filter_lcov_raises_on_malformed_fn_line_number(tmp_path):
    # Item 1: a FN record with a non-numeric line field is a format violation.
    (tmp_path / "real.rs").write_text("pub fn f() -> i32 { 1 }\n", encoding="utf-8")
    lcov = tmp_path / "in.info"
    lcov.write_text("SF:real.rs\nFN:xyz,f\nDA:1,1\nend_of_record\n", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed lcov FN record"):
        filter_lcov(lcov, tmp_path / "out.info", tmp_path)


def test_filter_lcov_raises_on_malformed_fnda_count(tmp_path):
    # Item 1: a FNDA record with a non-numeric hit count is a format violation.
    (tmp_path / "real.rs").write_text("pub fn f() -> i32 { 1 }\n", encoding="utf-8")
    lcov = tmp_path / "in.info"
    lcov.write_text(
        "SF:real.rs\nFN:1,f\nFNDA:foo,f\nDA:1,1\nend_of_record\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="malformed lcov FNDA record"):
        filter_lcov(lcov, tmp_path / "out.info", tmp_path)


def test_filter_lcov_raises_on_malformed_da_hit_count(tmp_path):
    # Item 1: a DA record with a non-numeric hit count (second field) is a format
    # violation; render_record must fail loud rather than default the count to 0.
    (tmp_path / "real.rs").write_text("pub fn f() -> i32 { 1 }\n", encoding="utf-8")
    lcov = tmp_path / "in.info"
    lcov.write_text("SF:real.rs\nDA:1,bar\nend_of_record\n", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed lcov DA record"):
        filter_lcov(lcov, tmp_path / "out.info", tmp_path)
