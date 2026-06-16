use super::*;

// -- T-003..T-008: strip_noise_tags -----------------------------------------------

#[test]
fn test_strip_noise_tags_table() {
    // Data-driven table: (label, input, expected)
    let cases: &[(&str, &str, &str)] = &[
        // T-003: command-name tag stripped
        (
            "T-003 command-name stripped",
            "before<command-name>/clear</command-name>after",
            "beforeafter",
        ),
        // T-004: local-command-caveat tag stripped
        (
            "T-004 local-command-caveat stripped",
            "start<local-command-caveat>Caveat: some warning text</local-command-caveat>end",
            "startend",
        ),
        // T-005: system-reminder tag stripped
        (
            "T-005 system-reminder stripped",
            "hello<system-reminder>internal instructions here</system-reminder>world",
            "helloworld",
        ),
        // T-006: local-command-stdout preserved
        (
            "T-006 local-command-stdout preserved",
            "prefix<local-command-stdout>output data</local-command-stdout>suffix",
            "prefix<local-command-stdout>output data</local-command-stdout>suffix",
        ),
        // T-007: plain text unchanged
        ("T-007 plain text unchanged", "hello world", "hello world"),
        // T-008: unclosed tag left intact
        (
            "T-008 unclosed tag left intact",
            "<command-name>no closing tag",
            "<command-name>no closing tag",
        ),
    ];

    for (label, input, expected) in cases {
        assert_eq!(
            strip_noise_tags(input.to_string()),
            *expected,
            "FAILED: {label} | input: {input:?}"
        );
    }
}

#[test]
fn test_parse_iso_timestamp_basics() {
    // Numeric passthrough
    let val = Value::Number(serde_json::Number::from(1709251200000_i64));
    assert_eq!(parse_iso_timestamp(&val), Some(1709251200000));

    // Rejection cases
    assert_eq!(parse_iso_timestamp(&Value::Null), None);
    assert_eq!(parse_iso_timestamp(&Value::String("2026".into())), None);
    assert_eq!(
        parse_iso_timestamp(&Value::String("2026-03-01T00:00:00Ü".into())),
        None
    );
    assert_eq!(
        parse_iso_timestamp(&Value::String("2026-03-01T24:00:00Z".into())),
        None
    );
    assert_eq!(
        parse_iso_timestamp(&Value::String("2026-03-01T12:60:00Z".into())),
        None
    );
    assert_eq!(
        parse_iso_timestamp(&Value::String("2026-03-01T12:00:60Z".into())),
        None
    );

    // No millis
    let ts = parse_iso_timestamp(&Value::String("2026-03-01T00:00:00Z".into())).unwrap();
    let day_ms = date::days_from_civil(2026, 3, 1).unwrap() * date::MS_PER_DAY;
    assert_eq!(ts, day_ms);
}

#[test]
fn test_parse_iso_timestamp_fractional_millis() {
    let day_ms = date::days_from_civil(2026, 3, 1).unwrap() * date::MS_PER_DAY;
    let base = day_ms + 12 * 3_600_000 + 30 * 60_000 + 45 * 1000;
    for (frac, expected_ms) in [(".123", 123), (".5", 500), (".12", 120)] {
        let s = format!("2026-03-01T12:30:45{frac}Z");
        let ts = parse_iso_timestamp(&Value::String(s.clone())).unwrap();
        assert_eq!(ts, base + expected_ms, "input: {s}");
    }
}

#[test]
fn test_parse_iso_timestamp_offsets() {
    let utc = parse_iso_timestamp(&Value::String("2026-03-01T12:00:00Z".into())).unwrap();
    // (offset_str, expected_diff_ms): positive = UTC is ahead, negative = UTC is behind
    for (offset_str, diff_ms) in [
        ("+09:00", 9 * 3_600_000_i64),
        ("+0900", 9 * 3_600_000),
        ("-05:00", -5 * 3_600_000),
    ] {
        let s = format!("2026-03-01T12:00:00{offset_str}");
        let with_offset = parse_iso_timestamp(&Value::String(s.clone())).unwrap();
        assert_eq!(utc - with_offset, diff_ms, "input: {s}");
    }
    // Offset with millis
    let utc_ms = parse_iso_timestamp(&Value::String("2026-03-01T12:00:00.500Z".into())).unwrap();
    let off_ms =
        parse_iso_timestamp(&Value::String("2026-03-01T12:00:00.500+09:00".into())).unwrap();
    assert_eq!(utc_ms - off_ms, 9 * 3_600_000);
}

#[test]
fn test_role_from_db_known_values() {
    assert_eq!(Role::from_db("user"), Some(Role::User));
    assert_eq!(Role::from_db("assistant"), Some(Role::Assistant));
}

#[test]
fn test_role_from_db_unknown_returns_none() {
    assert_eq!(Role::from_db("system"), None);
    assert_eq!(Role::from_db(""), None);
    assert_eq!(Role::from_db("USER"), None);
}

// ADR-0002 Confirmation: every `Source` variant round-trips through its DB token,
// so `from_db(x.as_str()) == Some(x)`. Driven off `ValueEnum::value_variants()`
// so a newly added variant is covered without editing this test.
#[test]
fn test_source_from_db_roundtrips_every_variant() {
    use clap::ValueEnum;
    for &variant in Source::value_variants() {
        assert_eq!(
            Source::from_db(variant.as_str()),
            Some(variant),
            "Source::from_db({:?}) must round-trip back to {variant:?}",
            variant.as_str(),
        );
    }
}

#[test]
fn test_source_from_db_unknown_returns_none() {
    assert_eq!(Source::from_db("gemini"), None);
    assert_eq!(Source::from_db(""), None);
    assert_eq!(Source::from_db("Claude"), None);
}
