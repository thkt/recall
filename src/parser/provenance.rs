//! Source-specific evidence, not keywords from conversation text. Wire formats
//! and their pinned sources are recorded in docs/research/session-provenance-classification.md.
use std::path::Path;

use serde_json::Value;

#[derive(Debug, Default, PartialEq, Eq)]
pub struct SessionOrigin {
    pub automated: bool,
    pub identity_conflict: bool,
    /// Distinguish an unknown origin on a session from an unrelated JSON file.
    pub has_records: bool,
}

pub(super) fn is_uuid(s: &str) -> bool {
    s.len() == 36
        && s.bytes().enumerate().all(|(i, b)| match i {
            8 | 13 | 18 | 23 => b == b'-',
            _ => b.is_ascii_hexdigit(),
        })
}

pub(super) fn claude_agent_id(path: &Path) -> Option<&str> {
    if path.extension()?.to_str()? != "jsonl" {
        return None;
    }
    path.file_stem()?
        .to_str()?
        .strip_prefix("agent-")
        .filter(|id| !id.is_empty())
}

/// Check the <parent UUID>/subagents directory after `claude_agent_id` validates the filename.
pub(super) fn claude_parent(path: &Path) -> Option<&str> {
    let dir = path.parent()?;
    if dir.file_name()?.to_str()? != "subagents" {
        return None;
    }
    let parent = dir.parent()?.file_name()?.to_str()?;
    is_uuid(parent).then_some(parent)
}

pub(super) fn codex_automated(payload: &Value) -> bool {
    // Either independently known field suffices, even with an unknown feature
    // in the other field. Arbitrary `other`/feature strings are not evidence.
    if matches!(
        payload.get("thread_source").and_then(Value::as_str),
        Some("subagent" | "guardian_review" | "memory_consolidation")
    ) {
        return true;
    }
    let Some(source) = payload.get("source").and_then(Value::as_object) else {
        return false;
    };
    if source.len() != 1 {
        return false;
    }
    if matches!(
        source.get("internal").and_then(Value::as_str),
        Some("guardian" | "memory_consolidation")
    ) {
        return true;
    }
    match source.get("subagent") {
        Some(Value::String(kind)) => {
            matches!(kind.as_str(), "review" | "compact" | "memory_consolidation")
        }
        Some(Value::Object(kind)) if kind.len() == 1 => {
            if kind.get("other").and_then(Value::as_str) == Some("guardian") {
                return true;
            }
            let Some(spawn) = kind.get("thread_spawn").and_then(Value::as_object) else {
                return false;
            };
            spawn
                .get("parent_thread_id")
                .and_then(Value::as_str)
                .is_some_and(is_uuid)
                && spawn
                    .get("depth")
                    .and_then(Value::as_i64)
                    .is_some_and(|n| i32::try_from(n).is_ok())
        }
        _ => false,
    }
}

/// The same all-record identity/provenance scan is used by ingestion and classify.
pub(super) struct ClaudeOrigin<'a> {
    session_id: &'a str,
    agent_id: Option<&'a str>,
    parent: Option<&'a str>,
    legacy_identity_matches: bool,
    legacy_parent: Option<String>,
    origin: SessionOrigin,
}

impl<'a> ClaudeOrigin<'a> {
    pub fn new(path: &'a Path, session_id: &'a str) -> Self {
        let agent_id = claude_agent_id(path);
        let parent = agent_id.and_then(|_| claude_parent(path));
        Self {
            session_id,
            agent_id,
            parent,
            legacy_identity_matches: parent.is_none() && agent_id.is_some(),
            legacy_parent: None,
            origin: SessionOrigin {
                automated: parent.is_some(),
                ..Default::default()
            },
        }
    }

    pub fn observe(&mut self, entry: &Value) {
        if !matches!(
            entry.get("type").and_then(Value::as_str),
            Some("user" | "human" | "assistant")
        ) && !matches!(
            entry.get("role").and_then(Value::as_str),
            Some("user" | "assistant")
        ) {
            return;
        }
        let is_meta = entry.get("isMeta").and_then(Value::as_bool) == Some(true);
        let sidechain = entry.get("isSidechain").and_then(Value::as_bool) == Some(true);
        if !is_meta {
            self.origin.has_records = true;
            self.origin.automated |= sidechain;
        }
        let id = entry.get("sessionId").and_then(Value::as_str);
        if let Some(id) = id {
            self.origin.identity_conflict |= id != self.session_id && Some(id) != self.parent;
        }
        // Every message must attest to the same agent and UUID parent outside
        // the canonical layout. Meta/role-only records cannot bypass identity.
        self.legacy_identity_matches = self.legacy_identity_matches
            && !is_meta
            && sidechain
            && entry.get("agentId").and_then(Value::as_str) == self.agent_id
            && id.is_some_and(is_uuid);
        if self.legacy_identity_matches
            && let Some(id) = id
        {
            let expected = self.legacy_parent.get_or_insert_with(|| id.to_owned());
            self.legacy_identity_matches &= expected == id;
        }
    }

    pub fn finish(mut self) -> SessionOrigin {
        self.origin.identity_conflict &= !self.legacy_identity_matches;
        self.origin.automated &= !self.origin.identity_conflict;
        self.origin
    }
}

pub(super) struct CodexOrigin {
    pub session_id: String,
    metadata_id: Option<String>,
    origin: SessionOrigin,
}

impl CodexOrigin {
    pub fn new(session_id: String) -> Self {
        Self {
            session_id,
            metadata_id: None,
            origin: SessionOrigin::default(),
        }
    }

    pub fn observe(&mut self, entry: &Value) {
        match entry.get("type").and_then(Value::as_str).unwrap_or("") {
            "session_meta" => {
                let payload = entry.get("payload").unwrap_or(&Value::Null);
                if let Some(id) = payload.get("id").and_then(Value::as_str)
                    && !id.is_empty()
                {
                    self.origin.has_records = true;
                    self.origin.automated |= codex_automated(payload);
                    self.origin.identity_conflict |=
                        self.metadata_id.as_deref().is_some_and(|prev| prev != id);
                    self.metadata_id = Some(id.to_owned());
                    if self.session_id.starts_with("rollout-") {
                        self.session_id = id.to_owned();
                    }
                }
            }
            "response_item" => {
                self.origin.has_records |= matches!(
                    entry
                        .get("payload")
                        .and_then(|p| p.get("role"))
                        .and_then(Value::as_str),
                    Some("user" | "assistant")
                );
            }
            "event_msg" | "turn_context" => {}
            _ => {
                self.origin.has_records |= matches!(
                    entry.get("role").and_then(Value::as_str),
                    Some("user" | "assistant")
                );
            }
        }
    }

    pub fn finish(mut self) -> (String, SessionOrigin) {
        self.origin.identity_conflict |= self
            .metadata_id
            .as_deref()
            .is_some_and(|id| id != self.session_id);
        self.origin.automated &= !self.origin.identity_conflict;
        (self.session_id, self.origin)
    }
}
