use super::*;

fn run(text: &str) -> String {
    normalize_for_fts(text, &QueryNormalizationConfig::default())
}

#[test]
fn pre_phase_5_disabled_matches_disabled() {
    assert_eq!(pre_phase_5_disabled(), QueryNormalizationConfig::disabled());
}

#[test]
fn fullwidth_latin_to_halfwidth() {
    assert_eq!(run("ＡＢＣ"), "abc");
}

#[test]
fn fullwidth_digits_to_halfwidth() {
    assert_eq!(run("１２３"), "123");
}

#[test]
fn ascii_uppercase_lowered() {
    assert_eq!(run("React Hooks"), "react hooks");
}

#[test]
fn hiragana_unchanged() {
    assert_eq!(run("ひらがな"), "ひらがな");
}

#[test]
fn katakana_unchanged() {
    // NFKC does not map Hiragana ↔ Katakana — only halfwidth katakana
    // (`ｱ` = U+FF71) folds to fullwidth katakana (`ア` = U+30A2).
    assert_eq!(run("カタカナ"), "カタカナ");
}

#[test]
fn halfwidth_katakana_to_fullwidth() {
    assert_eq!(run("ｱｲｳｴｵ"), "アイウエオ");
}

#[test]
fn ideographic_space_collapsed() {
    // U+3000 (ideographic space) NFKC-folds to U+0020, then collapses.
    assert_eq!(run("foo　bar"), "foo bar");
}

#[test]
fn whitespace_runs_collapsed() {
    assert_eq!(run("foo   bar\t\nbaz"), "foo bar baz");
}

#[test]
fn leading_trailing_whitespace_trimmed() {
    assert_eq!(run("  hello  "), "hello");
}

#[test]
fn empty_input_stays_empty() {
    assert_eq!(run(""), "");
}

#[test]
fn whitespace_only_collapses_to_empty() {
    assert_eq!(run("   "), "");
}

#[test]
fn ascii_passthrough_idempotent() {
    let s = "react hooks";
    assert_eq!(run(s), s);
    assert_eq!(run(&run(s)), run(s));
}

#[test]
fn idempotent_on_japanese_drift() {
    let s = "ＲｅａｃｔのHooks";
    let once = run(s);
    let twice = run(&once);
    assert_eq!(once, twice, "normalize must be a fixed point");
}

#[test]
fn idempotent_on_mixed_whitespace() {
    let s = "foo　　bar  baz";
    let once = run(s);
    let twice = run(&once);
    assert_eq!(once, twice);
}

#[test]
fn disabled_returns_input_unchanged() {
    let s = "ＡＢＣ　foo";
    let result = normalize_for_fts(s, &QueryNormalizationConfig::disabled());
    assert_eq!(result, s);
}

#[test]
fn nfkc_only_skips_lowercase_and_whitespace() {
    let config = QueryNormalizationConfig {
        nfkc: true,
        ascii_lowercase: false,
        collapse_whitespace: false,
    };
    assert_eq!(normalize_for_fts("ＡＢＣ", &config), "ABC");
    assert_eq!(normalize_for_fts("foo  bar", &config), "foo  bar");
}

#[test]
fn lowercase_only_skips_nfkc() {
    let config = QueryNormalizationConfig {
        nfkc: false,
        ascii_lowercase: true,
        collapse_whitespace: false,
    };
    assert_eq!(normalize_for_fts("React", &config), "react");
    assert_eq!(normalize_for_fts("ＡＢＣ", &config), "ＡＢＣ");
}

#[test]
fn whitespace_only_skips_nfkc_and_lowercase() {
    let config = QueryNormalizationConfig {
        nfkc: false,
        ascii_lowercase: false,
        collapse_whitespace: true,
    };
    assert_eq!(normalize_for_fts("  React   App  ", &config), "React App");
}

#[test]
fn config_round_trips_through_serde() {
    let original = QueryNormalizationConfig::default();
    let json = serde_json::to_string(&original).expect("serialise");
    let parsed: QueryNormalizationConfig = serde_json::from_str(&json).expect("round-trip");
    assert_eq!(parsed, original);
}

#[test]
fn fullwidth_punctuation_folds_under_nfkc() {
    // Fullwidth `！` (U+FF01) NFKC-folds to ASCII `!` (U+0021).
    assert_eq!(run("hello！"), "hello!");
}
