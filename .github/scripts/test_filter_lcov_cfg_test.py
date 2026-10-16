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


@pytest.mark.parametrize("header", ["#![cfg(test)]", "#![cfg(not(test))]", 'const S: &str = r#"\n#![cfg(test)]\n"#;'])
def test_file_guard_only_excludes_actual_test_only_source(tmp_path, header):
    src = tmp_path / "guarded.rs"
    text = header + "\nfn example() {}\n"
    src.write_text(text)
    expected = set(range(1, len(text.splitlines()) + 1)) if header == "#![cfg(test)]" else set()
    assert cfg_test_ranges(src) == expected


def test_brace_delta_ignores_braces_in_string_literals():
    assert brace_delta('    write!(buf, "{{")?;') == 0
    assert brace_delta('    let s = "}";') == 0


def test_brace_delta_ignores_char_literal_and_line_comment_braces():
    assert brace_delta("    let c = '{';") == 0
    assert brace_delta("    foo(); // a trailing { brace") == 0


def test_cfg_test_ranges_brace_in_string_does_not_swallow_following_production(tmp_path):
    src = tmp_path / "x.rs"
    src.write_text(
        "#[cfg(test)]\n"
        "mod tests {\n"
        '    fn t() { assert_eq!(x, "{"); }\n'
        "}\n"
        "pub fn prod() -> i32 { 1 }\n",
        encoding="utf-8",
    )
    ranges = cfg_test_ranges(src)
    assert {1, 2, 3, 4} <= ranges
    assert 5 not in ranges


def test_cfg_test_ranges_excludes_balanced_test_module(tmp_path):
    src = tmp_path / "x.rs"
    src.write_text(
        "pub fn prod() -> i32 {\n"
        "    1\n"
        "}\n"
        "\n"
        "#[cfg(test)]\n"
        "mod tests {\n"
        "    fn t() {}\n"
        "}\n",
        encoding="utf-8",
    )
    ranges = cfg_test_ranges(src)
    assert 1 not in ranges and 2 not in ranges and 3 not in ranges
    assert {6, 7, 8} <= ranges


def test_cfg_test_ranges_balanced_multiline_raw_string(tmp_path):
    src = tmp_path / "x.rs"
    src.write_text(
        "#[cfg(test)]\n"
        "mod tests {\n"
        "    fn t() {\n"
        '        let s = r#"{"a":{"b":1}}\n'
        '{"c":2}"#;\n'
        "        assert!(s.len() > 0);\n"
        "    }\n"
        "}\n",
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
    assert brace_delta("fn f<'a>(x: &'a str) {") == 1
    assert brace_delta("impl<'a> Foo<'a> {}") == 0


def test_brace_delta_char_escape_with_brace_in_codepoint():
    assert brace_delta(r"    let c = '\u{7b}'; if x {") == 1
    assert brace_delta(r"    if c == '\x1b' { f(); }") == 0


def test_cfg_test_ranges_unbalanced_brace_in_multiline_raw_string(tmp_path):
    src = tmp_path / "x.rs"
    src.write_text(
        "#[cfg(test)]\n"
        "mod tests {\n"
        "    fn t() {\n"
        '        let s = r#"{{{ unbalanced\n'
        '        still raw }"#;\n'
        "        assert!(s.len() > 0);\n"
        "    }\n"
        "}\n"
        "pub fn prod() -> i32 { 1 }\n",
        encoding="utf-8",
    )
    ranges = cfg_test_ranges(src)
    assert {1, 2, 3, 4, 5, 6, 7, 8} <= ranges
    assert 9 not in ranges


def test_filter_lcov_raises_on_unresolved_repo_source(tmp_path):
    lcov = tmp_path / "in.info"
    lcov.write_text("SF:src/ghost.rs\nDA:1,1\nend_of_record\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        filter_lcov(lcov, tmp_path / "out.info", tmp_path)


def test_filter_lcov_warns_and_skips_external_source(tmp_path, capsys):
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
    assert "SF:/nonexistent/external/dep.rs" in text
    assert "DA:1,1" in text
    assert text.count("end_of_record") == 1


def test_filter_lcov_raises_on_empty_input(tmp_path):
    lcov = tmp_path / "in.info"
    lcov.write_text("TN:\n", encoding="utf-8")
    with pytest.raises(ValueError):
        filter_lcov(lcov, tmp_path / "out.info", tmp_path)


def test_filter_lcov_emits_one_end_of_record_per_sf(tmp_path):
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
    assert "DA:1,1" in text


def test_filter_lcov_skips_existing_out_of_root_source(tmp_path, capsys):
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
    assert "DA:1,1" in out.read_text(encoding="utf-8")


def test_filter_lcov_raises_on_orphan_record(tmp_path):
    (tmp_path / "real.rs").write_text("pub fn f() -> i32 { 1 }\n", encoding="utf-8")
    lcov = tmp_path / "in.info"
    lcov.write_text("SF:real.rs\nDA:1,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not terminated by end_of_record"):
        filter_lcov(lcov, tmp_path / "out.info", tmp_path)


def test_filter_lcov_raises_on_mid_stream_orphan(tmp_path):
    (tmp_path / "a.rs").write_text("pub fn a() -> i32 { 1 }\n", encoding="utf-8")
    (tmp_path / "b.rs").write_text("pub fn b() -> i32 { 1 }\n", encoding="utf-8")
    lcov = tmp_path / "in.info"
    lcov.write_text(
        "SF:a.rs\nDA:1,1\nSF:b.rs\nDA:1,1\nend_of_record\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="record mismatch"):
        filter_lcov(lcov, tmp_path / "out.info", tmp_path)


def test_brace_delta_byte_strings_ignore_braces():
    assert brace_delta('    let b = b"{"; foo() {') == 1
    assert brace_delta('    let r = br#"}"#; bar() }') == -1


def test_brace_delta_escaped_quote_char():
    assert brace_delta(r"    let q = '\''; baz() {") == 1


def _cfg_test_src_with_ignored_line5(tmp_path: Path) -> Path:
    src = tmp_path / "real.rs"
    src.write_text(
        "pub fn prod() -> i32 { 1 }\n"
        "\n"
        "#[cfg(test)]\n"
        "mod tests {\n"
        "    fn t() {}\n"
        "}\n",
        encoding="utf-8",
    )
    return src


def test_render_record_drops_fn_and_fnda_for_ignored_function(tmp_path):
    _cfg_test_src_with_ignored_line5(tmp_path)
    lcov = tmp_path / "in.info"
    lcov.write_text(
        "SF:real.rs\n"
        "FN:1,prod\n"
        "FN:5,t\n"
        "FNDA:3,prod\n"
        "FNDA:0,t\n"
        "DA:1,3\n"
        "DA:5,0\n"
        "end_of_record\n",
        encoding="utf-8",
    )
    out = tmp_path / "out.info"
    filter_lcov(lcov, out, tmp_path)
    text = out.read_text(encoding="utf-8")
    assert "FN:1,prod" in text and "FNDA:3,prod" in text
    assert "FN:5,t" not in text
    assert "FNDA:0,t" not in text
    assert "FNF:1" in text and "FNH:1" in text
    assert "DA:5,0" not in text


def test_render_record_branch_coverage_taken_and_ignored(tmp_path):
    _cfg_test_src_with_ignored_line5(tmp_path)
    lcov = tmp_path / "in.info"
    lcov.write_text(
        "SF:real.rs\n"
        "DA:1,3\n"
        "BRDA:1,0,0,3\n"
        "BRDA:1,0,1,0\n"
        "BRDA:1,0,2,-\n"
        "BRDA:5,0,0,1\n"
        "end_of_record\n",
        encoding="utf-8",
    )
    out = tmp_path / "out.info"
    filter_lcov(lcov, out, tmp_path)
    text = out.read_text(encoding="utf-8")
    assert "BRDA:1,0,0,3" in text and "BRDA:1,0,1,0" in text and "BRDA:1,0,2,-" in text
    assert "BRDA:5,0,0,1" not in text
    assert "BRF:3" in text
    assert "BRH:1" in text


def test_filter_lcov_raises_on_malformed_da_line_number(tmp_path):
    (tmp_path / "real.rs").write_text("pub fn f() -> i32 { 1 }\n", encoding="utf-8")
    lcov = tmp_path / "in.info"
    lcov.write_text("SF:real.rs\nDA:abc,1\nend_of_record\n", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed lcov record"):
        filter_lcov(lcov, tmp_path / "out.info", tmp_path)


def test_filter_lcov_raises_on_malformed_fn_line_number(tmp_path):
    (tmp_path / "real.rs").write_text("pub fn f() -> i32 { 1 }\n", encoding="utf-8")
    lcov = tmp_path / "in.info"
    lcov.write_text("SF:real.rs\nFN:xyz,f\nDA:1,1\nend_of_record\n", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed lcov FN record"):
        filter_lcov(lcov, tmp_path / "out.info", tmp_path)


def test_filter_lcov_raises_on_malformed_fnda_count(tmp_path):
    (tmp_path / "real.rs").write_text("pub fn f() -> i32 { 1 }\n", encoding="utf-8")
    lcov = tmp_path / "in.info"
    lcov.write_text(
        "SF:real.rs\nFN:1,f\nFNDA:foo,f\nDA:1,1\nend_of_record\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="malformed lcov FNDA record"):
        filter_lcov(lcov, tmp_path / "out.info", tmp_path)


def test_filter_lcov_raises_on_malformed_da_hit_count(tmp_path):
    (tmp_path / "real.rs").write_text("pub fn f() -> i32 { 1 }\n", encoding="utf-8")
    lcov = tmp_path / "in.info"
    lcov.write_text("SF:real.rs\nDA:1,bar\nend_of_record\n", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed lcov DA record"):
        filter_lcov(lcov, tmp_path / "out.info", tmp_path)
