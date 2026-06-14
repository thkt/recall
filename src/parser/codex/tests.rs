use std::fs;

use super::*;
use crate::parser::write_test_jsonl as write_jsonl;

#[test]
fn test_codex_response_item_extraction() {
    let tmp = write_jsonl(&[
        r#"{"timestamp":"2026-01-17T16:39:33Z","type":"session_meta","payload":{"id":"abc-123","cwd":"/home/me/codex-proj"}}"#,
        r#"{"timestamp":"2026-01-17T16:40:00Z","type":"response_item","payload":{"role":"user","content":[{"type":"input_text","text":"hello from codex"}]}}"#,
        r#"{"timestamp":"2026-01-17T16:40:01Z","type":"response_item","payload":{"role":"assistant","content":[{"type":"output_text","text":"hi back"}]}}"#,
    ]);
    let result = parse_codex_session(tmp.path()).unwrap().unwrap();
    assert_eq!(result.messages.len(), 2);
    assert_eq!(result.messages[0].text, "hello from codex");
    assert_eq!(result.messages[1].text, "hi back");
}

#[test]
fn test_codex_skips_developer_role() {
    let tmp = write_jsonl(&[
        r#"{"timestamp":"2026-01-17T16:39:33Z","type":"response_item","payload":{"role":"developer","content":[{"type":"input_text","text":"system stuff"}]}}"#,
        r#"{"timestamp":"2026-01-17T16:40:00Z","type":"response_item","payload":{"role":"user","content":[{"type":"input_text","text":"real msg"}]}}"#,
    ]);
    let result = parse_codex_session(tmp.path()).unwrap().unwrap();
    assert_eq!(result.messages.len(), 1);
    assert_eq!(result.messages[0].text, "real msg");
}

#[test]
fn test_codex_skips_markers() {
    let tmp = write_jsonl(&[
        r#"{"timestamp":"2026-01-17T16:40:00Z","type":"response_item","payload":{"role":"user","content":[{"type":"input_text","text":"<user_instructions>system instructions</user_instructions>"}]}}"#,
        r#"{"timestamp":"2026-01-17T16:40:01Z","type":"response_item","payload":{"role":"user","content":[{"type":"input_text","text":"actual question"}]}}"#,
    ]);
    let result = parse_codex_session(tmp.path()).unwrap().unwrap();
    assert_eq!(result.messages.len(), 1);
    assert_eq!(result.messages[0].text, "actual question");
}

#[test]
fn test_codex_session_meta_extraction() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir
        .path()
        .join("rollout-2026-01-17T16-39-33-019bccd3-8798-7e11-b1d1-c61959201d0b.jsonl");
    fs::write(
            &path,
            concat!(
                r#"{"timestamp":"2026-01-17T16:39:33Z","type":"session_meta","payload":{"id":"019bccd3-8798-7e11-b1d1-c61959201d0b","cwd":"/home/me/codex-proj"}}"#,
                "\n",
                r#"{"timestamp":"2026-01-17T16:40:00Z","type":"response_item","payload":{"role":"user","content":[{"type":"input_text","text":"hello"}]}}"#,
            ),
        )
        .unwrap();
    let result = parse_codex_session(&path).unwrap().unwrap();
    assert_eq!(
        result.metadata.session_id,
        "019bccd3-8798-7e11-b1d1-c61959201d0b"
    );
    assert_eq!(result.metadata.project, "/home/me/codex-proj");
    assert_eq!(result.metadata.source, Source::Codex);
}

#[test]
fn test_codex_date_slug_from_path() {
    assert_eq!(
        extract_date_from_path("/home/.codex/sessions/2026/01/18/rollout-xxx.jsonl"),
        Some("2026-01-18".to_owned())
    );
    assert_eq!(extract_date_from_path("/some/other/path/file.jsonl"), None);
    // Windows-style backslash path
    assert_eq!(
        extract_date_from_path("C:\\Users\\me\\.codex\\sessions\\2026\\01\\18\\rollout-xxx.jsonl"),
        Some("2026-01-18".to_owned())
    );
}

#[test]
fn test_uuid_short_extraction() {
    assert_eq!(
        extract_uuid_short("rollout-2026-01-18T01-37-09-019bccd1-564a-73b3-b3b0-f6b12671ed24"),
        Some("019bccd1".to_owned())
    );
    assert_eq!(extract_uuid_short("no-uuid-here"), None);
}

#[test]
fn test_codex_empty_messages_returns_none() {
    let tmp = write_jsonl(&[
        r#"{"timestamp":"2026-01-17T16:39:33Z","type":"session_meta","payload":{"id":"abc","cwd":"/proj"}}"#,
        r#"{"timestamp":"2026-01-17T16:40:00Z","type":"event_msg","payload":{}}"#,
    ]);
    assert!(parse_codex_session(tmp.path()).unwrap().is_none());
}
