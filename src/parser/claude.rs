use std::path::Path;

use anyhow::Result;
use serde_json::Value;

use super::provenance::{SessionOrigin, claude_agent_id, claude_parent, is_uuid};

use super::{
    Message, ParseResult, Role, SessionData, Source, extract_text, extract_tool_use_path,
    is_valid_scanned_path, parse_iso_timestamp, parse_jsonl_entries, session_id_from_path,
    update_earliest,
};

struct ClaudeParseState {
    origin: SessionOrigin,
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

    state.origin.has_records = true;
    state.origin.automated |= entry.get("isSidechain").and_then(Value::as_bool) == Some(true);

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

/// Retain empty reads and their diagnostics for ingestion and path backfill.
pub fn parse_claude_session(path: &Path) -> Result<Option<ParseResult>> {
    let Some(session_id) = session_id_from_path(path) else {
        return Ok(None);
    };

    let agent_id = claude_agent_id(path);
    let parent = agent_id.and_then(|_| claude_parent(path));
    let mut legacy_identity_matches = parent.is_none() && agent_id.is_some();
    let mut legacy_parent = None;
    let mut state = ClaudeParseState {
        origin: SessionOrigin {
            automated: parent.is_some(),
            ..Default::default()
        },
        project: String::new(),
        slug: String::new(),
        earliest_ts: None,
        scanned_files: Vec::new(),
    };

    let (messages, diagnostics) = parse_jsonl_entries(path, |entry| {
        let message = process_claude_entry(entry, &mut state);
        if matches!(
            entry.get("type").and_then(Value::as_str),
            Some("user" | "human" | "assistant")
        ) || matches!(
            entry.get("role").and_then(Value::as_str),
            Some("user" | "assistant")
        ) {
            let id = entry.get("sessionId").and_then(Value::as_str);
            if let Some(id) = id {
                state.origin.identity_conflict |= id != session_id && Some(id) != parent;
            }
            // Outside the canonical layout, every message must attest to this
            // agent and the same UUID parent. One good line cannot authenticate
            // a mixed or incomplete transcript, nor override a directory parent.
            legacy_identity_matches = legacy_identity_matches
                && entry.get("isMeta").and_then(Value::as_bool) != Some(true)
                && entry.get("isSidechain").and_then(Value::as_bool) == Some(true)
                && entry.get("agentId").and_then(Value::as_str) == agent_id
                && id.is_some_and(is_uuid);
            if legacy_identity_matches && let Some(id) = id {
                let expected = legacy_parent.get_or_insert_with(|| id.to_owned());
                legacy_identity_matches &= expected == id;
            }
        }
        message
    })?;
    state.origin.identity_conflict &= !legacy_identity_matches;
    state.origin.automated &= !state.origin.identity_conflict;

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
        diagnostics,
        origin: state.origin,
    }))
}

#[cfg(test)]
mod tests;
