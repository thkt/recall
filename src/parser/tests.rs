use std::fs;

use serde_json::json;

use super::*;

fn claude_session(lines: &[String]) -> ParseResult {
    let lines: Vec<&str> = lines.iter().map(String::as_str).collect();
    let file = write_test_jsonl(&lines);
    parse_session_including_empty(file.path(), Source::Claude)
        .unwrap()
        .unwrap()
}

fn user_line(content: &str) -> String {
    json!({"type": "user", "message": {"role": "user", "content": content}}).to_string()
}

/// The session time a Claude log gets when its only entry carries `timestamp`.
fn session_timestamp(timestamp: &Value) -> Option<i64> {
    let line = json!({
        "type": "user",
        "timestamp": timestamp,
        "message": {"role": "user", "content": "q"},
    });
    claude_session(&[line.to_string()]).metadata.timestamp
}

#[test]
fn noise_tags_are_removed_from_messages_but_command_output_and_plain_text_stay() {
    let session = claude_session(&[
        user_line("before<command-name>/clear</command-name>after"),
        user_line("start<local-command-caveat>Caveat: some warning</local-command-caveat>end"),
        user_line("hello<system-reminder>internal instructions</system-reminder>world"),
        user_line("prefix<local-command-stdout>output data</local-command-stdout>suffix"),
        user_line("hello world"),
        user_line("<command-name>no closing tag"),
    ]);

    assert_eq!(
        session
            .messages
            .iter()
            .map(|m| m.text.as_str())
            .collect::<Vec<_>>(),
        [
            "beforeafter",
            "startend",
            "helloworld",
            "prefix<local-command-stdout>output data</local-command-stdout>suffix",
            "hello world",
            "<command-name>no closing tag",
        ]
    );
}

#[test]
fn iso_timestamps_set_the_session_time_in_utc_milliseconds() {
    assert_eq!(
        session_timestamp(&json!("2026-03-01T00:00:00Z")),
        Some(1_772_323_200_000)
    );
    assert_eq!(
        session_timestamp(&json!("2026-03-01T12:30:45.123Z")),
        Some(1_772_368_245_123)
    );
    assert_eq!(
        session_timestamp(&json!("2026-03-01T12:30:45.5Z")),
        Some(1_772_368_245_500)
    );
    assert_eq!(
        session_timestamp(&json!("2026-03-01T12:30:45.12Z")),
        Some(1_772_368_245_120)
    );
    assert_eq!(
        session_timestamp(&json!("2026-03-01T12:00:00+09:00")),
        Some(1_772_334_000_000)
    );
    assert_eq!(
        session_timestamp(&json!("2026-03-01T12:00:00+0900")),
        Some(1_772_334_000_000)
    );
    assert_eq!(
        session_timestamp(&json!("2026-03-01T12:00:00-05:00")),
        Some(1_772_384_400_000)
    );
    assert_eq!(
        session_timestamp(&json!("2026-03-01T12:00:00.500+09:00")),
        Some(1_772_334_000_500)
    );
}

#[test]
fn numeric_timestamps_are_taken_as_epoch_milliseconds() {
    assert_eq!(
        session_timestamp(&json!(1_709_251_200_000_i64)),
        Some(1_709_251_200_000)
    );
}

#[test]
fn malformed_timestamps_leave_the_session_without_a_time() {
    assert_eq!(session_timestamp(&Value::Null), None);
    assert_eq!(session_timestamp(&json!("2026")), None);
    assert_eq!(session_timestamp(&json!("2026-03-01T00:00:00Ü")), None);
    assert_eq!(session_timestamp(&json!("2026-03-01T24:00:00Z")), None);
    assert_eq!(session_timestamp(&json!("2026-03-01T12:60:00Z")), None);
    assert_eq!(session_timestamp(&json!("2026-03-01T12:00:60Z")), None);
}

#[test]
fn stored_role_names_map_back_to_roles() {
    assert_eq!(Role::from_db("user"), Some(Role::User));
    assert_eq!(Role::from_db("assistant"), Some(Role::Assistant));
    assert_eq!(Role::from_db("system"), None);
    assert_eq!(Role::from_db(""), None);
    assert_eq!(Role::from_db("USER"), None);
}

#[test]
fn every_source_round_trips_through_its_stored_name() {
    use clap::ValueEnum;
    // DR 0002: a variant added later must round-trip too, so this walks every variant.
    for &variant in Source::value_variants() {
        assert_eq!(Source::from_db(variant.as_str()), Some(variant));
    }
    assert_eq!(Source::from_db("gemini"), None);
    assert_eq!(Source::from_db(""), None);
    assert_eq!(Source::from_db("Claude"), None);
}

/// Line diagnostics `[invalid JSON, invalid UTF-8, incomplete tail]` for a log holding `bytes`,
/// read as `[Claude, Codex]`. The classification reader must report the same.
fn diagnostics_of(bytes: &[u8]) -> [[i64; 3]; 2] {
    let file = tempfile::NamedTempFile::new().unwrap();
    fs::write(file.path(), bytes).unwrap();
    let claude = parse_session_including_empty(file.path(), Source::Claude)
        .unwrap()
        .unwrap();
    let codex = parse_session_including_empty(file.path(), Source::Codex)
        .unwrap()
        .unwrap();
    assert_classification_matches(file.path(), Source::Claude, &claude);
    assert_classification_matches(file.path(), Source::Codex, &codex);
    [claude.diagnostics, codex.diagnostics].map(|d| {
        [
            d.invalid_json_lines,
            d.invalid_utf8_lines,
            d.incomplete_tail_lines,
        ]
    })
}

#[test]
fn line_diagnostics_tell_corrupt_lines_from_a_possibly_unfinished_last_line() {
    assert_eq!(
        diagnostics_of(b"{\"type\":\"progress\"}\n \t\r\n"),
        [[0, 0, 0]; 2]
    );
    assert_eq!(diagnostics_of(b"{\"type\":\"progress\"}"), [[0, 0, 0]; 2]);
    assert_eq!(diagnostics_of(b"{\"type\":"), [[0, 0, 1]; 2]);
    assert_eq!(diagnostics_of(b"{\"type\":\n"), [[1, 0, 0]; 2]);
    assert_eq!(diagnostics_of(b"{broken"), [[1, 0, 0]; 2]);
    assert_eq!(diagnostics_of(b"\xff\n"), [[0, 1, 0]; 2]);
    assert_eq!(diagnostics_of(b"\xff"), [[0, 1, 0]; 2]);
    assert_eq!(diagnostics_of(b"{\"text\":\"\xe3\x81"), [[0, 0, 1]; 2]);
    assert_eq!(diagnostics_of(b"{\"text\":\"\xe3\x81\n"), [[0, 1, 0]; 2]);
}

const CLAUDE_SIDECHAIN: &str =
    r#"{"type":"user","sessionId":"sample","isSidechain":true,"message":"question"}"#;
const CLAUDE_OTHER_ID: &str = r#"{"role":"assistant","sessionId":"other"}"#;
const CODEX_GUARDIAN: &str =
    r#"{"type":"session_meta","payload":{"id":"sample","source":{"internal":"guardian"}}}"#;
const CODEX_OTHER_ID: &str = r#"{"type":"session_meta","payload":{"id":"other"}}"#;

/// `(identity conflict, automated, line diagnostics)` for a log whose first line marks it
/// automated and whose second line is `tail`. The classification reader must report the same.
fn origin_after(source: Source, first: &str, tail: &[u8]) -> (bool, bool, [i64; 3]) {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("sample.jsonl");
    let mut input = format!("{first}\n").into_bytes();
    input.extend_from_slice(tail);
    fs::write(&path, input).unwrap();
    let full = parse_session_including_empty(&path, source)
        .unwrap()
        .unwrap();
    assert_classification_matches(&path, source, &full);
    let d = full.diagnostics;
    (
        full.origin.identity_conflict,
        full.origin.automated,
        [
            d.invalid_json_lines,
            d.invalid_utf8_lines,
            d.incomplete_tail_lines,
        ],
    )
}

#[test]
fn a_conflicting_session_id_cancels_automated_evidence_but_bad_lines_do_not() {
    assert_eq!(
        origin_after(Source::Claude, CLAUDE_SIDECHAIN, CLAUDE_OTHER_ID.as_bytes()),
        (true, false, [0, 0, 0])
    );
    assert_eq!(
        origin_after(Source::Claude, CLAUDE_SIDECHAIN, b"{bad\n"),
        (false, true, [1, 0, 0])
    );
    assert_eq!(
        origin_after(Source::Claude, CLAUDE_SIDECHAIN, b"\xff\n"),
        (false, true, [0, 1, 0])
    );
    assert_eq!(
        origin_after(Source::Claude, CLAUDE_SIDECHAIN, b"{\"type\":"),
        (false, true, [0, 0, 1])
    );
    assert_eq!(
        origin_after(Source::Codex, CODEX_GUARDIAN, CODEX_OTHER_ID.as_bytes()),
        (true, false, [0, 0, 0])
    );
    assert_eq!(
        origin_after(Source::Codex, CODEX_GUARDIAN, b"{bad\n"),
        (false, true, [1, 0, 0])
    );
    assert_eq!(
        origin_after(Source::Codex, CODEX_GUARDIAN, b"\xff\n"),
        (false, true, [0, 1, 0])
    );
    assert_eq!(
        origin_after(Source::Codex, CODEX_GUARDIAN, b"{\"type\":"),
        (false, true, [0, 0, 1])
    );
}
