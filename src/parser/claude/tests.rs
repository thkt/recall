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

// U-002 dedupe: a write-target path repeated across tool_use blocks — within one
// assistant turn and again in a later turn — collapses to a single scanned_files
// entry, preserving first-seen order. T-002/T-003 never repeat a path, so this is
// the only cover for the contract's "セッション内 dedupe" clause. The two assistant
// turns carry no text block, so this also pins that a tool_use-only turn still
// contributes paths (extraction runs before the empty-text early return).
#[test]
fn test_claude_scanned_files_deduped_within_session() {
    let tmp = write_jsonl(&[
        r#"{"type":"user","message":{"role":"user","content":"edit the file twice"},"timestamp":"2026-03-01T00:00:00Z"}"#,
        r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"Edit","input":{"file_path":"/proj/same.rs","old_string":"a","new_string":"b"}},{"type":"tool_use","id":"t2","name":"Write","input":{"file_path":"/proj/other.rs","content":"c"}}]},"timestamp":"2026-03-01T00:00:01Z"}"#,
        r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","id":"t3","name":"Edit","input":{"file_path":"/proj/same.rs","old_string":"b","new_string":"d"}}]},"timestamp":"2026-03-01T00:00:02Z"}"#,
    ]);
    let result = parse_claude_session(tmp.path()).unwrap().unwrap();
    assert_eq!(
        result.scanned_files,
        vec!["/proj/same.rs".to_owned(), "/proj/other.rs".to_owned()],
        "the repeated path is recorded once, in first-seen order, across both turns"
    );
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
