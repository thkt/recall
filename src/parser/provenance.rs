use std::path::Path;

use serde_json::Value;

use super::claude::ClaudeEntry;
use super::codex::CodexEntry;

#[derive(Debug, Default, PartialEq, Eq)]
pub struct SessionOrigin {
    pub automated: bool,
    pub identity_conflict: bool,
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

pub(super) fn claude_parent(path: &Path) -> Option<&str> {
    let dir = path.parent()?;
    if dir.file_name()?.to_str()? != "subagents" {
        return None;
    }
    let parent = dir.parent()?.file_name()?.to_str()?;
    is_uuid(parent).then_some(parent)
}

pub(super) fn codex_automated(payload: &Value) -> bool {
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

    pub fn observe(&mut self, entry: &ClaudeEntry<'_>) {
        if entry.role.is_none() {
            return;
        }
        if !entry.is_meta {
            self.origin.has_records = true;
            self.origin.automated |= entry.sidechain;
        }
        let id = entry.session_id;
        if let Some(id) = id {
            self.origin.identity_conflict |= id != self.session_id && Some(id) != self.parent;
        }
        self.legacy_identity_matches = self.legacy_identity_matches
            && !entry.is_meta
            && entry.sidechain
            && entry.agent_id == self.agent_id
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

    pub fn observe(&mut self, entry: &CodexEntry<'_>) {
        match entry {
            CodexEntry::Metadata {
                id: Some(id),
                automated,
                ..
            } => {
                self.origin.has_records = true;
                self.origin.automated |= automated;
                self.origin.identity_conflict |=
                    self.metadata_id.as_deref().is_some_and(|prev| prev != *id);
                self.metadata_id = Some((*id).to_owned());
                if self.session_id.starts_with("rollout-") {
                    self.session_id = (*id).to_owned();
                }
            }
            CodexEntry::Message { .. } => self.origin.has_records = true,
            _ => {}
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
