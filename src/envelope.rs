//! Output envelopes for `--json` mode (ADR-0060, #67 Phase 2).
//!
//! Mirrors `sae/src/envelope.rs`. [`CommandOutput`] is the canonical runner
//! return type, pairing the human-facing `markdown` with the machine `data`
//! plus `degraded`/`notes` signals. [`SuccessEnvelope`] and [`ErrorEnvelope`]
//! are the wire shapes serialized to stdout/stderr under `--json`.
//!
//! recall has no `CantCreat` (73) class — DB/IO failures map to `IoError` (see
//! [`crate::error::ErrorCode`]).

use serde::Serialize;

use crate::error::ErrorCode;

/// Canonical return type for command runners (#67 Phase 2).
///
/// `markdown` is the human-facing rendering surfaced when `--json` is absent.
/// `data` is the machine payload mirrored into [`SuccessEnvelope::data`] when
/// `--json` is set; runners build it with `serde_json::json!` so the public JSON
/// schema stays decoupled from internal types ([`crate::search::SearchResult`] /
/// [`crate::parser::SessionData`] deliberately stay non-`Serialize`). `degraded`
/// and `notes` surface deviations from the ideal path so agents can react —
/// currently set only by `search`, when the embedding model fails to load and
/// ranking falls back to FTS-only.
#[derive(Debug)]
pub(crate) struct CommandOutput {
    pub markdown: String,
    pub data: serde_json::Value,
    pub degraded: bool,
    pub notes: Vec<String>,
}

impl CommandOutput {
    /// An ideal-path result: not degraded, no notes.
    pub(crate) fn ok(markdown: String, data: serde_json::Value) -> Self {
        Self {
            markdown,
            data,
            degraded: false,
            notes: Vec::new(),
        }
    }

    /// A result that may have taken a fallback path; `degraded` + `notes` record it.
    pub(crate) fn with_notes(
        markdown: String,
        data: serde_json::Value,
        degraded: bool,
        notes: Vec<String>,
    ) -> Self {
        Self {
            markdown,
            data,
            degraded,
            notes,
        }
    }
}

/// Serialized to stdout when `--json` is set.
#[derive(Debug, Serialize)]
pub(crate) struct SuccessEnvelope {
    pub data: serde_json::Value,
    pub degraded: bool,
    pub notes: Vec<String>,
}

/// Serialized to stderr when `--json` is set and the command failed. Wrapping the
/// payload under `error` lets consumers branch on the root key.
#[derive(Debug, Serialize)]
pub(crate) struct ErrorEnvelope {
    pub error: ErrorPayload,
}

/// Error payload nested under [`ErrorEnvelope::error`].
///
/// `next_step` and `candidates` are omitted from JSON when absent/empty, keeping
/// the envelope compact when no structured guidance applies.
#[derive(Debug, Serialize)]
pub(crate) struct ErrorPayload {
    pub code: ErrorCode,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_step: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub candidates: Vec<String>,
    pub retryable: bool,
}

/// Serializes a successful result to the wire envelope.
pub(crate) fn render_json_success(out: &CommandOutput) -> String {
    let env = SuccessEnvelope {
        data: out.data.clone(),
        degraded: out.degraded,
        notes: out.notes.clone(),
    };
    serde_json::to_string(&env).expect("SuccessEnvelope is always serializable")
}

/// Serializes a prepared error envelope to the wire format. The [`ErrorEnvelope`]
/// is assembled by the caller (via [`crate::error::RecallError::to_error_envelope`]),
/// keeping this module a pure serializer that never constructs a `RecallError`.
pub(crate) fn render_json_error(env: &ErrorEnvelope) -> String {
    serde_json::to_string(env).expect("ErrorEnvelope is always serializable")
}

#[cfg(test)]
mod tests;
