"""Unit tests for filter_lcov_cfg_test.py (the coverage-gate cfg(test) filter).

Run: python3 -m pytest .github/scripts/test_filter_lcov_cfg_test.py

The filter sits between `cargo llvm-cov` and `diff-cover`, so a brace-counting
bug that misclassifies production code as test code silently weakens the 95%
gate. These tests pin the current raw-count behavior and document its known
limitation (string-literal braces) as xfail until a Rust tokenizer replaces the
heuristic.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from filter_lcov_cfg_test import brace_delta, cfg_test_ranges  # noqa: E402

# Known limitation: brace_delta counts raw `{`/`}` including those inside string
# literals / char literals / comments. No current recall source triggers a wrong
# coverage range (multi-line raw-string JSON braces are balanced), so a proper
# fix (a Rust tokenizer) is deferred. These xfails document the gap honestly
# without forcing a regex masking implementation that broke multi-line raw
# strings in src/indexer.rs.
_TOKENIZER_LIMITATION = "brace counting miscounts string-literal braces; needs a Rust tokenizer (followup)"


def test_brace_delta_counts_structural_braces():
    assert brace_delta("mod tests {") == 1
    assert brace_delta("}") == -1
    assert brace_delta("fn f() {}") == 0


@pytest.mark.xfail(reason=_TOKENIZER_LIMITATION, strict=False)
def test_brace_delta_ignores_braces_in_string_literals():
    assert brace_delta('    write!(buf, "{{")?;') == 0
    assert brace_delta('    let s = "}";') == 0


@pytest.mark.xfail(reason=_TOKENIZER_LIMITATION, strict=False)
def test_brace_delta_ignores_char_literal_and_line_comment_braces():
    assert brace_delta("    let c = '{';") == 0
    assert brace_delta("    foo(); // a trailing { brace") == 0


@pytest.mark.xfail(reason=_TOKENIZER_LIMITATION, strict=False)
def test_cfg_test_ranges_brace_in_string_does_not_swallow_following_production(tmp_path):
    # Aspirational: an unbalanced brace in a test string literal should not
    # extend the ignored range into the production fn that follows. Raw-count
    # cannot do this; documented as the followup tokenizer fix.
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
