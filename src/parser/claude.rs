use std::path::Path;

use anyhow::Result;
use serde_json::Value;

use super::{
    Message, ParseResult, Role, SessionData, Source, extract_text, parse_iso_timestamp,
    parse_jsonl_entries, session_id_from_path, update_earliest,
};

struct ClaudeParseState {
    project: String,
    slug: String,
    earliest_ts: Option<i64>,
}

fn process_claude_entry(entry: &Value, state: &mut ClaudeParseState) -> Option<Message> {
    if entry
        .get("isMeta")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
    {
        return None;
    }

    if state.project.is_empty()
        && let Some(cwd) = entry.get("cwd").and_then(|v| v.as_str())
        && !cwd.is_empty()
    {
        state.project = cwd.to_string();
    }

    if state.slug.is_empty()
        && let Some(s) = entry
            .get("slug")
            .or_else(|| entry.get("leafName"))
            .and_then(|v| v.as_str())
        && !s.is_empty()
    {
        state.slug = s.to_string();
    }

    if let Some(ts_val) = entry.get("timestamp")
        && let Some(ts) = parse_iso_timestamp(ts_val)
    {
        update_earliest(&mut state.earliest_ts, ts);
    }

    let entry_type = entry.get("type").and_then(|v| v.as_str()).unwrap_or("");
    let role_field = entry.get("role").and_then(|v| v.as_str()).unwrap_or("");

    let role = if role_field == "user" || entry_type == "user" || entry_type == "human" {
        Role::User
    } else if role_field == "assistant" || entry_type == "assistant" {
        Role::Assistant
    } else {
        return None;
    };

    let msg_content = match entry.get("message") {
        Some(Value::Object(msg)) => msg.get("content"),
        Some(Value::String(_)) => entry.get("message"),
        _ => entry.get("content"),
    };

    let text = extract_text(msg_content);
    if text.is_empty() {
        return None;
    }
    Some(Message { role, text })
}

pub fn parse_claude_session(path: &Path) -> Result<Option<ParseResult>> {
    let Some(session_id) = session_id_from_path(path) else {
        return Ok(None);
    };

    let mut state = ClaudeParseState {
        project: String::new(),
        slug: String::new(),
        earliest_ts: None,
    };

    let (messages, skipped_lines) =
        parse_jsonl_entries(path, |entry| process_claude_entry(entry, &mut state))?;

    if messages.is_empty() {
        return Ok(None);
    }

    if state.slug.is_empty() {
        state.slug = session_id.chars().take(12).collect();
    }

    Ok(Some(ParseResult {
        metadata: SessionData {
            session_id,
            source: Source::Claude,
            file_path: path.to_string_lossy().to_string(),
            project: state.project,
            slug: state.slug,
            timestamp: state.earliest_ts,
        },
        messages,
        skipped_lines,
    }))
}

#[cfg(test)]
mod tests {
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
}
