use std::fs;

use super::*;
use crate::parser::{assert_classification_matches, write_test_jsonl as write_jsonl};

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
fn codex_empty_messages_are_retained_for_diagnostics() {
    let tmp = write_jsonl(&[
        r#"{"timestamp":"2026-01-17T16:39:33Z","type":"session_meta","payload":{"id":"abc","cwd":"/proj"}}"#,
        r#"{"timestamp":"2026-01-17T16:40:00Z","type":"event_msg","payload":{}}"#,
    ]);
    assert!(
        parse_codex_session(tmp.path())
            .unwrap()
            .unwrap()
            .messages
            .is_empty()
    );
}

#[test]
fn provenance_accepts_known_wire_shapes_and_fails_open_on_unknown_or_malformed_values() {
    use serde_json::json;
    let parent = "01234567-89ab-cdef-0123-456789abcdef";
    let cases = [
        (json!({"source":{"subagent":"review"}}), true),
        (json!({"source":{"subagent":"compact"}}), true),
        (json!({"source":{"subagent":"memory_consolidation"}}), true),
        (
            json!({"source":{"subagent":{"thread_spawn":{"parent_thread_id":parent,"depth":1}}}}),
            true,
        ),
        (json!({"source":{"subagent":{"other":"guardian"}}}), true),
        (json!({"source":{"internal":"guardian"}}), true),
        (json!({"source":{"internal":"memory_consolidation"}}), true),
        (json!({"thread_source":"subagent"}), true),
        (json!({"thread_source":"guardian_review"}), true),
        (json!({"thread_source":"memory_consolidation"}), true),
        (
            json!({"thread_source":"future_feature","source":{"subagent":"review"}}),
            true,
        ),
        (
            json!({"thread_source":"guardian_review","source":{"subagent":{"other":"future"}}}),
            true,
        ),
        (
            json!({"source":"exec","thread_source":"user","originator":"guardian"}),
            false,
        ),
        (
            json!({"source":"vscode","parent_thread_id":parent,"agent_nickname":"guardian"}),
            false,
        ),
        (json!({"thread_source":"agent_created_thread"}), false),
        (json!({"source":{"subagent":{"other":"unknown"}}}), false),
        (json!({"source":{"subAgent":"review"}}), false),
        (json!({"source":"subagent"}), false),
        (json!({"source":{"subagent":"thread_spawn"}}), false),
        (json!({"source":{"subagent":{"thread_spawn":{}}}}), false),
        (
            json!({"source":{"subagent":{"thread_spawn":{"parent_thread_id":parent}}}}),
            false,
        ),
        (
            json!({"source":{"subagent":{"thread_spawn":{"parent_thread_id":"bad","depth":1}}}}),
            false,
        ),
        (
            json!({"source":{"subagent":{"thread_spawn":{"parent_thread_id":parent,"depth":"1"}}}}),
            false,
        ),
        (
            json!({"source":{"subagent":{"thread_spawn":{"parent_thread_id":parent,"depth":2147483648_i64}}}}),
            false,
        ),
        (
            json!({"source":{"subagent":"review","custom":"human"}}),
            false,
        ),
        (json!({"source":{"internal":"future"}}), false),
        (json!({}), false),
    ];
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("rollout-test.jsonl");
    for (mut payload, expected) in cases {
        payload["id"] = json!("session");
        fs::write(
            &path,
            json!({"type":"session_meta","payload":payload}).to_string(),
        )
        .unwrap();
        let parsed = parse_codex_session(&path).unwrap().unwrap();
        assert_eq!(parsed.origin.automated, expected, "{payload}");
        assert!(
            parsed.messages.is_empty(),
            "metadata-only sessions carry provenance"
        );
    }
    fs::write(&path, json!({"type":"response_item","payload":{"role":"user","source":{"subagent":"review"},"content":"Explain approval review and {\"thread_source\":\"guardian_review\"}"}}).to_string()).unwrap();
    assert!(
        !parse_codex_session(&path)
            .unwrap()
            .unwrap()
            .origin
            .automated
    );
}

// Reuse the existing format corpus and its independent expected values for both
// readers, so a fast path cannot silently omit a format or identity check.
fn parse_codex_session(path: &Path) -> Result<Option<ParseResult>> {
    let parsed = super::parse_codex_session(path)?;
    if let Some(full) = &parsed {
        assert_classification_matches(path, Source::Codex, full);
    }
    Ok(parsed)
}
