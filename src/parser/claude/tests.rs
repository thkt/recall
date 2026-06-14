use super::*;
use crate::parser::write_test_jsonl as write_jsonl;

#[test]
fn test_claude_text_string_content() {
    let tmp = write_jsonl(&[
        r#"{"type":"user","cwd":"/home/me/proj","message":{"role":"user","content":"hello world"},"timestamp":"2026-03-01T00:00:00Z"}"#,
    ]);
    let result = parse_claude_session(tmp.path()).unwrap().unwrap();
    assert_eq!(result.messages.len(), 1);
    assert_eq!(result.messages[0].role, Role::User);
    assert_eq!(result.messages[0].text, "hello world");
}

#[test]
fn test_claude_block_array_extracts_text_only() {
    let tmp = write_jsonl(&[
        r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"answer"},{"type":"tool_use","id":"t1","name":"Bash","input":{}},{"type":"thinking","thinking":"hmm"}]},"timestamp":"2026-03-01T00:00:00Z"}"#,
    ]);
    let result = parse_claude_session(tmp.path()).unwrap().unwrap();
    assert_eq!(result.messages.len(), 1);
    assert_eq!(result.messages[0].text, "answer");
}

#[test]
fn test_claude_skips_non_message_types() {
    let tmp = write_jsonl(&[
        r#"{"type":"file-history-snapshot","messageId":"abc","snapshot":{}}"#,
        r#"{"type":"progress","data":{"type":"hook_progress"}}"#,
        r#"{"type":"user","message":{"role":"user","content":"real message"},"timestamp":"2026-03-01T00:00:00Z"}"#,
    ]);
    let result = parse_claude_session(tmp.path()).unwrap().unwrap();
    assert_eq!(result.messages.len(), 1);
    assert_eq!(result.messages[0].text, "real message");
}

#[test]
fn test_claude_metadata_extraction() {
    let tmp = write_jsonl(&[
        r#"{"type":"user","cwd":"/home/me/project","slug":"my-session","message":{"role":"user","content":"hello"},"timestamp":"2026-03-01T12:00:00Z"}"#,
    ]);
    let result = parse_claude_session(tmp.path()).unwrap().unwrap();
    assert_eq!(result.metadata.source, Source::Claude);
    assert_eq!(result.metadata.project, "/home/me/project");
    assert_eq!(result.metadata.slug, "my-session");
    assert!(result.metadata.timestamp.is_some_and(|ts| ts > 0));
}

#[test]
fn test_claude_invalid_json_skipped() {
    let tmp = write_jsonl(&[
        "not valid json",
        r#"{"type":"user","message":{"role":"user","content":"ok"},"timestamp":"2026-03-01T00:00:00Z"}"#,
        "{broken",
    ]);
    let result = parse_claude_session(tmp.path()).unwrap().unwrap();
    assert_eq!(result.messages.len(), 1);
}

#[test]
fn test_claude_slug_fallback_to_session_id() {
    let tmp = write_jsonl(&[
        r#"{"type":"user","message":{"role":"user","content":"hi"},"timestamp":"2026-03-01T00:00:00Z"}"#,
    ]);
    let result = parse_claude_session(tmp.path()).unwrap().unwrap();
    assert!(!result.metadata.slug.is_empty());
}

#[test]
fn test_claude_empty_messages_returns_none() {
    let tmp = write_jsonl(&[
        r#"{"type":"progress","data":{"type":"hook_progress"}}"#,
        r#"{"type":"file-history-snapshot","messageId":"abc","snapshot":{}}"#,
    ]);
    assert!(parse_claude_session(tmp.path()).unwrap().is_none());
}

// T-001: isMeta entries are skipped
#[test]
fn test_claude_is_meta_skipped() {
    let tmp = write_jsonl(&[
        r#"{"type":"user","isMeta":true,"message":{"role":"user","content":"<local-command-caveat>Caveat: ...</local-command-caveat>"},"timestamp":"2026-03-01T00:00:00Z"}"#,
        r#"{"type":"user","message":{"role":"user","content":"real message"},"timestamp":"2026-03-01T00:00:00Z"}"#,
    ]);
    let result = parse_claude_session(tmp.path()).unwrap().unwrap();
    assert_eq!(result.messages.len(), 1);
    assert_eq!(result.messages[0].text, "real message");
}
