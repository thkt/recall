//! JSONL session log parser for Claude Code and Codex CLI.
//!
//! Claude format: one JSON object per line with `type`, `message.content`, `cwd`, `timestamp`.
//! Codex format: structured with `session_meta`, `response_item`, and legacy fallback entries.

use std::io::{BufRead, BufReader};
use std::path::Path;

use anyhow::{Context, Result};
use serde_json::Value;

use crate::date;

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Source {
    Claude,
    Codex,
}

impl Source {
    pub fn as_str(&self) -> &'static str {
        match self {
            Source::Claude => "claude",
            Source::Codex => "codex",
        }
    }

    pub fn from_db(s: &str) -> Option<Source> {
        match s {
            "claude" => Some(Source::Claude),
            "codex" => Some(Source::Codex),
            _ => None,
        }
    }
}

impl std::fmt::Display for Source {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
}

impl Role {
    pub fn as_str(&self) -> &'static str {
        match self {
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }
}

#[derive(Clone)]
pub struct SessionData {
    pub session_id: String,
    pub source: Source,
    pub file_path: String,
    pub project: String,
    pub slug: String,
    pub timestamp: i64,
}

pub struct Message {
    pub role: Role,
    pub text: String,
}

pub struct ParseResult {
    pub metadata: SessionData,
    pub messages: Vec<Message>,
    pub skipped_lines: usize,
}

const TEXT_BLOCK_TYPES: &[&str] = &["text", "input_text", "output_text"];
const CODEX_SKIP_MARKERS: &[&str] = &[
    "<user_instructions>",
    "<environment_context>",
    "<permissions instructions>",
    "# AGENTS.md instructions",
];

fn extract_text(content: Value) -> String {
    match content {
        Value::String(s) => s,
        Value::Array(blocks) => {
            let mut result = String::new();
            for block in &blocks {
                let Some(block_type) = block.get("type").and_then(|v| v.as_str()) else {
                    continue;
                };
                if TEXT_BLOCK_TYPES.contains(&block_type)
                    && let Some(text) = block.get("text").and_then(|v| v.as_str())
                {
                    if !result.is_empty() {
                        result.push('\n');
                    }
                    result.push_str(text);
                }
            }
            result
        }
        _ => String::new(),
    }
}

fn parse_iso_timestamp(val: &Value) -> Option<i64> {
    match val {
        Value::String(s) => {
            let s = s.trim();
            if s.len() < 19 || !s.is_ascii() {
                return None;
            }
            let b = s.as_bytes();
            if b[4] != b'-' || b[7] != b'-' || b[13] != b':' || b[16] != b':' {
                return None;
            }
            let year: i64 = s[0..4].parse().ok()?;
            let month: i64 = s[5..7].parse().ok()?;
            let day: i64 = s[8..10].parse().ok()?;
            if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
                return None;
            }
            let hour: i64 = s[11..13].parse().ok()?;
            let min: i64 = s[14..16].parse().ok()?;
            let sec: i64 = s[17..19].parse().ok()?;

            let millis = if s.len() > 20 && b[19] == b'.' {
                let end = s[20..]
                    .find(|c: char| !c.is_ascii_digit())
                    .unwrap_or(s.len() - 20);
                let frac = &s[20..20 + end];
                let padded = format!("{:0<3}", &frac[..frac.len().min(3)]);
                padded.parse::<i64>().unwrap_or(0)
            } else {
                0
            };

            let days = date::days_from_civil(year, month, day)?;
            let epoch_ms = days * 86_400_000 + hour * 3_600_000 + min * 60_000 + sec * 1000 + millis;
            Some(epoch_ms)
        }
        Value::Number(n) => n.as_i64(),
        _ => None,
    }
}

fn update_earliest(earliest: &mut Option<i64>, ts: i64) {
    *earliest = Some(match *earliest {
        Some(prev) => prev.min(ts),
        None => ts,
    });
}

pub fn parse_claude_session(path: &Path) -> Result<Option<ParseResult>> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("Failed to open {}", path.display()))?;
    let reader = BufReader::new(file);

    let session_id = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_string();

    let mut project = String::new();
    let mut slug = String::new();
    let mut earliest_ts: Option<i64> = None;
    let mut messages = Vec::new();
    let mut skipped_lines = 0usize;

    for line_result in reader.lines() {
        let Ok(raw_line) = line_result else {
            skipped_lines += 1;
            continue;
        };
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        let entry: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => {
                skipped_lines += 1;
                continue;
            }
        };

        if project.is_empty()
            && let Some(cwd) = entry.get("cwd").and_then(|v| v.as_str())
            && !cwd.is_empty()
        {
            project = cwd.to_string();
        }

        if slug.is_empty()
            && let Some(s) = entry
                .get("slug")
                .or_else(|| entry.get("leafName"))
                .and_then(|v| v.as_str())
            && !s.is_empty()
        {
            slug = s.to_string();
        }

        if let Some(ts_val) = entry.get("timestamp")
            && let Some(ts) = parse_iso_timestamp(ts_val)
        {
            update_earliest(&mut earliest_ts, ts);
        }

        let entry_type = entry.get("type").and_then(|v| v.as_str()).unwrap_or("");
        let role_field = entry.get("role").and_then(|v| v.as_str()).unwrap_or("");

        let role = if role_field == "user" || entry_type == "user" || entry_type == "human" {
            Role::User
        } else if role_field == "assistant" || entry_type == "assistant" {
            Role::Assistant
        } else {
            continue;
        };

        let msg_content = match entry.get("message") {
            Some(Value::Object(msg)) => msg
                .get("content")
                .cloned()
                .unwrap_or(Value::String(String::new())),
            Some(Value::String(s)) => Value::String(s.clone()),
            _ => entry
                .get("content")
                .cloned()
                .unwrap_or(Value::String(String::new())),
        };

        let text = extract_text(msg_content);
        if !text.is_empty() {
            messages.push(Message { role, text });
        }
    }

    if messages.is_empty() {
        return Ok(None);
    }

    if slug.is_empty() {
        slug = session_id.chars().take(12).collect();
    }

    Ok(Some(ParseResult {
        metadata: SessionData {
            session_id,
            source: Source::Claude,
            file_path: path.to_string_lossy().to_string(),
            project,
            slug,
            timestamp: earliest_ts.unwrap_or(0),
        },
        messages,
        skipped_lines,
    }))
}

pub fn parse_codex_session(path: &Path) -> Result<Option<ParseResult>> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("Failed to open {}", path.display()))?;
    let reader = BufReader::new(file);

    let mut session_id = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_string();

    let mut project = String::new();
    let mut earliest_ts: Option<i64> = None;
    let mut messages = Vec::new();
    let mut skipped_lines = 0usize;

    let path_str = path.to_string_lossy();
    let date_slug = extract_date_from_path(&path_str);
    let uuid_short = extract_uuid_short(&session_id);

    for line_result in reader.lines() {
        let Ok(raw_line) = line_result else {
            skipped_lines += 1;
            continue;
        };
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        let entry: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => {
                skipped_lines += 1;
                continue;
            }
        };

        if let Some(ts_val) = entry.get("timestamp")
            && let Some(ts) = parse_iso_timestamp(ts_val)
        {
            update_earliest(&mut earliest_ts, ts);
        }

        let entry_type = entry.get("type").and_then(|v| v.as_str()).unwrap_or("");

        match entry_type {
            "session_meta" => {
                let payload = entry.get("payload").unwrap_or(&Value::Null);
                if let Some(id) = payload.get("id").and_then(|v| v.as_str())
                    && !id.is_empty()
                    && session_id.starts_with("rollout-")
                {
                    session_id = id.to_string();
                }
                if project.is_empty()
                    && let Some(cwd) = payload.get("cwd").and_then(|v| v.as_str())
                    && !cwd.is_empty()
                {
                    project = cwd.to_string();
                }
            }
            "response_item" => {
                let payload = entry.get("payload").unwrap_or(&Value::Null);
                let role = match payload.get("role").and_then(|v| v.as_str()) {
                    Some("user") => Role::User,
                    Some("assistant") => Role::Assistant,
                    _ => continue,
                };
                let content_val = payload
                    .get("content")
                    .cloned()
                    .unwrap_or(Value::String(String::new()));
                let text = extract_text(content_val);
                if text.is_empty() {
                    continue;
                }
                if is_codex_boilerplate(&text) {
                    continue;
                }
                messages.push(Message { role, text });
            }
            "event_msg" | "turn_context" => continue,
            _ => {
                // Legacy format
                let role = match entry.get("role").and_then(|v| v.as_str()) {
                    Some("user") => Role::User,
                    Some("assistant") => Role::Assistant,
                    _ => continue,
                };
                let content_val = entry
                    .get("content")
                    .cloned()
                    .unwrap_or(Value::String(String::new()));

                if project.is_empty()
                    && let Value::Array(blocks) = &content_val
                {
                    for block in blocks {
                        if let Some(text) = block.get("text").and_then(|v| v.as_str())
                            && let Some(cwd) = extract_cwd_from_env_context(text)
                        {
                            project = cwd;
                            break;
                        }
                    }
                }

                let text = extract_text(content_val);
                if text.is_empty() {
                    continue;
                }
                if is_codex_boilerplate(&text) {
                    continue;
                }
                messages.push(Message { role, text });
            }
        }
    }

    if messages.is_empty() {
        return Ok(None);
    }

    let slug = match (&date_slug, &uuid_short) {
        (Some(d), Some(u)) => format!("{d}-{u}"),
        (Some(d), None) => d.clone(),
        (None, Some(u)) => u.clone(),
        (None, None) => session_id.chars().take(8).collect(),
    };

    Ok(Some(ParseResult {
        metadata: SessionData {
            session_id,
            source: Source::Codex,
            file_path: path.to_string_lossy().to_string(),
            project,
            slug,
            timestamp: earliest_ts.unwrap_or(0),
        },
        messages,
        skipped_lines,
    }))
}

fn extract_date_from_path(path: &str) -> Option<String> {
    let segments: Vec<&str> = path.split('/').collect();
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
            return Some(window[0].to_string());
        }
    }
    None
}

fn is_codex_boilerplate(text: &str) -> bool {
    CODEX_SKIP_MARKERS.iter().any(|m| text.contains(m))
}

fn extract_cwd_from_env_context(text: &str) -> Option<String> {
    const MARKER: &str = "Current working directory:";
    for line in text.lines() {
        if let Some(pos) = line.find(MARKER) {
            let after = &line[pos + MARKER.len()..];
            let trimmed = after.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn write_jsonl(lines: &[&str]) -> NamedTempFile {
        let mut tmp = NamedTempFile::new().unwrap();
        for line in lines {
            writeln!(tmp, "{}", line).unwrap();
        }
        tmp.flush().unwrap();
        tmp
    }

    // --- T-001: Claude JSONL パース ---

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
        assert!(result.metadata.timestamp > 0);
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

    // --- T-002: Codex JSONL パース ---

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
        std::fs::write(
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
            Some("2026-01-18".to_string())
        );
        assert_eq!(extract_date_from_path("/some/other/path/file.jsonl"), None);
    }

    #[test]
    fn test_uuid_short_extraction() {
        assert_eq!(
            extract_uuid_short("rollout-2026-01-18T01-37-09-019bccd1-564a-73b3-b3b0-f6b12671ed24"),
            Some("019bccd1".to_string())
        );
        assert_eq!(extract_uuid_short("no-uuid-here"), None);
    }

    // --- T-003 supplement: Timestamp parsing edge cases ---

    #[test]
    fn test_parse_iso_timestamp_with_millis() {
        let val = Value::String("2026-03-01T12:30:45.123Z".to_string());
        let ts = parse_iso_timestamp(&val).unwrap();
        let expected_days = date::days_from_civil(2026, 3, 1).unwrap();
        let expected = expected_days * 86_400_000 + 12 * 3_600_000 + 30 * 60_000 + 45 * 1000 + 123;
        assert_eq!(ts, expected);
    }

    #[test]
    fn test_parse_iso_timestamp_numeric() {
        let val = Value::Number(serde_json::Number::from(1709251200000_i64));
        assert_eq!(parse_iso_timestamp(&val), Some(1709251200000));
    }

    #[test]
    fn test_parse_iso_timestamp_too_short() {
        let val = Value::String("2026".to_string());
        assert_eq!(parse_iso_timestamp(&val), None);
    }

    #[test]
    fn test_parse_iso_timestamp_null() {
        assert_eq!(parse_iso_timestamp(&Value::Null), None);
    }

    #[test]
    fn test_parse_iso_timestamp_no_millis() {
        let val = Value::String("2026-03-01T00:00:00Z".to_string());
        let ts = parse_iso_timestamp(&val).unwrap();
        let expected_days = date::days_from_civil(2026, 3, 1).unwrap();
        assert_eq!(ts, expected_days * 86_400_000);
    }

    #[test]
    fn test_parse_iso_timestamp_non_ascii_rejected() {
        let val = Value::String("2026-03-01T00:00:00Ü".to_string());
        assert_eq!(parse_iso_timestamp(&val), None);
    }

    #[test]
    fn test_claude_empty_messages_returns_none() {
        let tmp = write_jsonl(&[
            r#"{"type":"progress","data":{"type":"hook_progress"}}"#,
            r#"{"type":"file-history-snapshot","messageId":"abc","snapshot":{}}"#,
        ]);
        assert!(parse_claude_session(tmp.path()).unwrap().is_none());
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
