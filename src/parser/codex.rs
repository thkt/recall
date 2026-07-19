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
            // Legacy fallback for pre-session_meta rollouts: untyped entries with
            // a top-level `role` that inline cwd in an <environment_context> block.
            // Modern files carry cwd in the session_meta payload (the first line),
            // so `project` is already set before any response_item is reached and
            // the response_item arm needs no cwd extraction. Verified in issue #232:
            // 925/925 audit-corpus files had session_meta.cwd populated.
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
        // Codex rollouts carry no Claude Code write-tool metadata (contract U-002).
        scanned_files: Vec::new(),
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
mod tests;
