use std::path::Path;

use anyhow::Result;
use serde_json::Value;

use super::provenance::ClaudeOrigin;

use super::{
    Message, ParseResult, Role, SessionData, Source, extract_text, extract_tool_use_path,
    parse_iso_timestamp, parse_jsonl_entries, session_id_from_path, update_earliest,
};

pub(super) struct ClaudeEntry<'a> {
    pub(super) role: Option<Role>,
    pub(super) is_meta: bool,
    pub(super) sidechain: bool,
    pub(super) session_id: Option<&'a str>,
    pub(super) agent_id: Option<&'a str>,
    cwd: Option<&'a str>,
    slug: Option<&'a str>,
    timestamp: Option<i64>,
    content: Option<&'a Value>,
}

impl<'a> ClaudeEntry<'a> {
    pub(super) fn from_json(entry: &'a Value) -> Self {
        let entry_type = entry.get("type").and_then(Value::as_str).unwrap_or("");
        let role_field = entry.get("role").and_then(Value::as_str).unwrap_or("");
        let role = if role_field == "user" || matches!(entry_type, "user" | "human") {
            Some(Role::User)
        } else if role_field == "assistant" || entry_type == "assistant" {
            Some(Role::Assistant)
        } else {
            None
        };
        Self {
            role,
            is_meta: entry.get("isMeta").and_then(Value::as_bool) == Some(true),
            sidechain: entry.get("isSidechain").and_then(Value::as_bool) == Some(true),
            session_id: entry.get("sessionId").and_then(Value::as_str),
            agent_id: entry.get("agentId").and_then(Value::as_str),
            cwd: entry
                .get("cwd")
                .and_then(Value::as_str)
                .filter(|cwd| !cwd.is_empty()),
            slug: entry
                .get("slug")
                .or_else(|| entry.get("leafName"))
                .and_then(Value::as_str)
                .filter(|slug| !slug.is_empty()),
            timestamp: entry.get("timestamp").and_then(parse_iso_timestamp),
            content: match entry.get("message") {
                Some(Value::Object(msg)) => msg.get("content"),
                message @ Some(Value::String(_)) => message,
                _ => entry.get("content"),
            },
        }
    }
}

pub fn parse_claude_session(path: &Path) -> Result<Option<ParseResult>> {
    let Some(session_id) = session_id_from_path(path) else {
        return Ok(None);
    };

    let mut origin = ClaudeOrigin::new(path, &session_id);
    let mut project = String::new();
    let mut slug = String::new();
    let mut earliest_ts = None;
    let mut scanned_files = Vec::new();

    let (messages, diagnostics) = parse_jsonl_entries(path, |entry| {
        let entry = ClaudeEntry::from_json(entry);
        origin.observe(&entry);
        if entry.is_meta {
            return None;
        }
        if project.is_empty()
            && let Some(cwd) = entry.cwd
        {
            project = cwd.to_owned();
        }
        if slug.is_empty()
            && let Some(value) = entry.slug
        {
            slug = value.to_owned();
        }
        if let Some(ts) = entry.timestamp {
            update_earliest(&mut earliest_ts, ts);
        }
        let role = entry.role?;
        if let Some(Value::Array(blocks)) = entry.content {
            for block in blocks {
                if let Some(path) = extract_tool_use_path(block)
                    && !scanned_files.iter().any(|p| p == path)
                {
                    scanned_files.push(path.to_owned());
                }
            }
        }
        let text = extract_text(entry.content);
        (!text.is_empty()).then_some(Message { role, text })
    })?;
    let origin = origin.finish();

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
            timestamp: earliest_ts,
        },
        messages,
        scanned_files,
        diagnostics,
        origin,
    }))
}

#[cfg(test)]
mod tests;
