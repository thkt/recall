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
    # OPS-006: a `.rs` SF path under repo_root that does not exist is a path
    # resolution bug; fail loud rather than silently under-filtering.
    lcov = tmp_path / "in.info"
    lcov.write_text("SF:src/ghost.rs\nDA:1,1\nend_of_record\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        filter_lcov(lcov, tmp_path / "out.info", tmp_path)


def test_filter_lcov_warns_and_skips_external_source(tmp_path, capsys):
    # OPS-006 / SEC: an absolute `.rs` outside repo_root (external dep) cannot
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
    # ENC-002: an input with no SF records would yield a trivially-passing
    # filtered file; fail loud.
    lcov = tmp_path / "in.info"
    lcov.write_text("TN:\n", encoding="utf-8")
    with pytest.raises(ValueError):
        filter_lcov(lcov, tmp_path / "out.info", tmp_path)


def test_filter_lcov_emits_one_end_of_record_per_sf(tmp_path):
    # ENC-002 well-formedness: one end_of_record per SF, production lines kept.
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


def test_filter_lcov_raises_on_sf_eor_mismatch(tmp_path):
    # ENC-002 mismatch branch: an SF record with no end_of_record (truncated
    # input) yields sf_count=1 but eor_count=0; the guard must fire (distinct
    # from the empty-input branch).
    (tmp_path / "real.rs").write_text("pub fn f() -> i32 { 1 }\n", encoding="utf-8")
    lcov = tmp_path / "in.info"
    lcov.write_text("SF:real.rs\nDA:1,1\n", encoding="utf-8")  # no end_of_record
    with pytest.raises(ValueError):
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
