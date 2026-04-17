use std::path::Path;

use anyhow::Result;
use serde_json::Value;

use super::{
    Message, ParseResult, Role, SessionData, Source, extract_text, parse_iso_timestamp,
    parse_jsonl_entries, session_id_from_path, update_earliest,
};

const CODEX_SKIP_MARKERS: &[&str] = &[
    "<user_instructions>",
    "<environment_context>",
    "<permissions instructions>",
    "# AGENTS.md instructions",
];

fn extract_codex_message(payload: &Value) -> Option<Message> {
    let role = match payload.get("role").and_then(|v| v.as_str()) {
        Some("user") => Role::User,
        Some("assistant") => Role::Assistant,
        _ => return None,
    };
    let text = extract_text(payload.get("content"));
    if text.is_empty() || is_codex_boilerplate(&text) {
        return None;
    }
    Some(Message { role, text })
}

fn extract_cwd_from_content(content: Option<&Value>) -> Option<String> {
    if let Some(Value::Array(blocks)) = content {
        for block in blocks {
            if let Some(text) = block.get("text").and_then(|v| v.as_str())
                && let Some(cwd) = extract_cwd_from_env_context(text)
            {
                return Some(cwd);
            }
        }
    }
    None
}

struct CodexParseState {
    session_id: String,
    project: String,
    earliest_ts: Option<i64>,
}

fn process_codex_entry(
    entry: &Value,
    entry_type: &str,
    state: &mut CodexParseState,
) -> Option<Message> {
    if let Some(ts_val) = entry.get("timestamp")
        && let Some(ts) = parse_iso_timestamp(ts_val)
    {
        update_earliest(&mut state.earliest_ts, ts);
    }

    match entry_type {
        "session_meta" => {
            let payload = entry.get("payload").unwrap_or(&Value::Null);
            if let Some(id) = payload.get("id").and_then(|v| v.as_str())
                && !id.is_empty()
                && state.session_id.starts_with("rollout-")
            {
                state.session_id = id.to_owned();
            }
            if state.project.is_empty()
                && let Some(cwd) = payload.get("cwd").and_then(|v| v.as_str())
                && !cwd.is_empty()
            {
                state.project = cwd.to_owned();
            }
            None
        }
        "response_item" => {
            let payload = entry.get("payload").unwrap_or(&Value::Null);
            extract_codex_message(payload)
        }
        "event_msg" | "turn_context" => None,
        _ => {
            match entry.get("role").and_then(|v| v.as_str()) {
                Some("user" | "assistant") => {}
                _ => return None,
            }
            if state.project.is_empty()
                && let Some(cwd) = extract_cwd_from_content(entry.get("content"))
            {
                state.project = cwd;
            }
            extract_codex_message(entry)
        }
    }
}

pub fn parse_codex_session(path: &Path) -> Result<Option<ParseResult>> {
    let Some(initial_session_id) = session_id_from_path(path) else {
        return Ok(None);
    };

    let mut state = CodexParseState {
        session_id: initial_session_id,
        project: String::new(),
        earliest_ts: None,
    };

    let path_str = path.to_string_lossy();
    let date_slug = extract_date_from_path(&path_str);
    let uuid_short = extract_uuid_short(&state.session_id);

    let (messages, skipped_lines) = parse_jsonl_entries(path, |entry| {
        let entry_type = entry.get("type").and_then(|v| v.as_str()).unwrap_or("");
        process_codex_entry(entry, entry_type, &mut state)
    })?;

    if messages.is_empty() {
        return Ok(None);
    }

    let slug = match (&date_slug, &uuid_short) {
        (Some(d), Some(u)) => format!("{d}-{u}"),
        (Some(d), None) => d.clone(),
        (None, Some(u)) => u.clone(),
        (None, None) => state.session_id.chars().take(8).collect(),
    };

    Ok(Some(ParseResult {
        metadata: SessionData {
            session_id: state.session_id,
            source: Source::Codex,
            file_path: path_str.into_owned(),
            project: state.project,
            slug,
            timestamp: state.earliest_ts,
        },
        messages,
        skipped_lines,
    }))
}

fn extract_date_from_path(path: &str) -> Option<String> {
    let segments: Vec<&str> = path.split(['/', '\\']).collect();
    for window in segments.windows(4) {
        if window[0] == "sessions"
            && window[1].len() == 4
            && window[1].chars().all(|c| c.is_ascii_digit())
            && window[2].len() == 2
            && window[3].len() == 2
        {
            return Some(format!("{}-{}-{}", window[1], window[2], window[3]));
        }
    }
    None
}

fn extract_uuid_short(filename: &str) -> Option<String> {
    let parts: Vec<&str> = filename.split('-').collect();
    for window in parts.windows(5) {
        if window[0].len() == 8
            && window[1].len() == 4
            && window[2].len() == 4
            && window[3].len() == 4
            && window[4].len() >= 12
            && window[0].chars().all(|c| c.is_ascii_hexdigit())
            && window[1].chars().all(|c| c.is_ascii_hexdigit())
        {
            return Some(window[0].to_owned());
        }
    }
    None
}

fn is_codex_boilerplate(text: &str) -> bool {
    CODEX_SKIP_MARKERS.iter().any(|m| text.starts_with(m))
}

fn extract_cwd_from_env_context(text: &str) -> Option<String> {
    const MARKER: &str = "Current working directory:";
    for line in text.lines() {
        if let Some(pos) = line.find(MARKER) {
            let after = &line[pos + MARKER.len()..];
            let trimmed = after.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_owned());
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
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
            extract_date_from_path(
                "C:\\Users\\me\\.codex\\sessions\\2026\\01\\18\\rollout-xxx.jsonl"
            ),
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
}
