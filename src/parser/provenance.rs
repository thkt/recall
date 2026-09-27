//! Source-specific evidence, not keywords from conversation text. Wire formats
//! and their pinned sources are recorded in docs/research/session-provenance-classification.md.
use std::path::Path;

use serde_json::Value;

#[derive(Default)]
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
