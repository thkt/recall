use std::path::Path;

use anyhow::Result;
use serde_json::Value;

use super::{
    Message, ParseResult, Role, SessionData, Source, extract_text, extract_tool_use_path,
    is_valid_scanned_path, parse_iso_timestamp, parse_jsonl_entries, session_id_from_path,
    update_earliest,
};

struct ClaudeParseState {
    project: String,
    slug: String,
    earliest_ts: Option<i64>,
    /// Distinct write-target paths seen across the session, in first-seen order.
    scanned_files: Vec<String>,
}

fn process_claude_entry(entry: &Value, state: &mut ClaudeParseState) -> Option<Message> {
    if entry
        .get("isMeta")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
    {
        return None;
    }

    if state.project.is_empty()
        && let Some(cwd) = entry.get("cwd").and_then(|v| v.as_str())
        && !cwd.is_empty()
    {
        state.project = cwd.to_owned();
    }

    if state.slug.is_empty()
        && let Some(s) = entry
            .get("slug")
            .or_else(|| entry.get("leafName"))
            .and_then(|v| v.as_str())
        && !s.is_empty()
    {
        state.slug = s.to_owned();
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

    // Collect write-target paths before the empty-text early return: an assistant
    // turn that is only tool_use (no text block) still carries scanned files.
    if let Some(Value::Array(blocks)) = msg_content {
        for block in blocks {
            if let Some(path) = extract_tool_use_path(block)
                && is_valid_scanned_path(path)
                && !state.scanned_files.iter().any(|p| p == path)
            {
                state.scanned_files.push(path.to_owned());
            }
        }
    }

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
        scanned_files: Vec::new(),
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
        scanned_files: state.scanned_files,
        skipped_lines,
    }))
}

#[cfg(test)]
mod tests;
