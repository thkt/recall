use std::path::Path;

use anyhow::Result;
use serde_json::Value;

use super::provenance::{CodexOrigin, codex_automated};

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

pub(super) enum CodexEntry<'a> {
    Metadata {
        id: Option<&'a str>,
        cwd: Option<&'a str>,
        automated: bool,
    },
    Message {
        role: Role,
        content: Option<&'a Value>,
        legacy: bool,
    },
    Ignored,
}

impl<'a> CodexEntry<'a> {
    pub(super) fn from_json(entry: &'a Value) -> Self {
        let entry_type = entry.get("type").and_then(Value::as_str).unwrap_or("");
        let payload = entry.get("payload").unwrap_or(&Value::Null);
        match entry_type {
            "session_meta" => Self::Metadata {
                id: payload
                    .get("id")
                    .and_then(Value::as_str)
                    .filter(|id| !id.is_empty()),
                cwd: payload
                    .get("cwd")
                    .and_then(Value::as_str)
                    .filter(|cwd| !cwd.is_empty()),
                automated: codex_automated(payload),
            },
            "event_msg" | "turn_context" => Self::Ignored,
            _ => {
                let legacy = entry_type != "response_item";
                let message = if legacy { entry } else { payload };
                let Some(role) = message
                    .get("role")
                    .and_then(Value::as_str)
                    .and_then(Role::from_db)
                else {
                    return Self::Ignored;
                };
                Self::Message {
                    role,
                    content: message.get("content"),
                    legacy,
                }
            }
        }
    }
}

pub fn parse_codex_session(path: &Path) -> Result<Option<ParseResult>> {
    let Some(initial_session_id) = session_id_from_path(path) else {
        return Ok(None);
    };

    let mut origin = CodexOrigin::new(initial_session_id);
    let mut project = String::new();
    let mut earliest_ts = None;

    let path_str = path.to_string_lossy();
    let date_slug = extract_date_from_path(&path_str);
    let uuid_short = extract_uuid_short(&origin.session_id);

    let (messages, diagnostics) = parse_jsonl_entries(path, |entry| {
        if let Some(ts) = entry.get("timestamp").and_then(parse_iso_timestamp) {
            update_earliest(&mut earliest_ts, ts);
        }
        let entry = CodexEntry::from_json(entry);
        origin.observe(&entry);
        match entry {
            CodexEntry::Metadata { cwd, .. } => {
                if project.is_empty()
                    && let Some(cwd) = cwd
                {
                    project = cwd.to_owned();
                }
                None
            }
            CodexEntry::Message {
                role,
                content,
                legacy,
            } => {
                if legacy
                    && project.is_empty()
                    && let Some(Value::Array(blocks)) = content
                {
                    project = blocks
                        .iter()
                        .find_map(|block| {
                            block
                                .get("text")
                                .and_then(Value::as_str)
                                .and_then(extract_cwd_from_env_context)
                        })
                        .unwrap_or_default();
                }
                let text = extract_text(content);
                if text.is_empty() || CODEX_SKIP_MARKERS.iter().any(|m| text.starts_with(m)) {
                    None
                } else {
                    Some(Message { role, text })
                }
            }
            CodexEntry::Ignored => None,
        }
    })?;

    let (session_id, origin) = origin.finish();

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
            file_path: path_str.into_owned(),
            project,
            slug,
            timestamp: earliest_ts,
        },
        messages,
        scanned_files: Vec::new(),
        diagnostics,
        origin,
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
