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
mod tests {
    use super::{CommandOutput, ErrorEnvelope, ErrorPayload, SuccessEnvelope};
    use super::{render_json_error, render_json_success};
    use crate::error::ErrorCode;

    // T-EN001: error_payload_omits_optional_next_step_when_none
    // Perspective: boundary (None is the empty/min case for the optional field).
    // next_step=None must not emit the key, keeping the envelope compact.
    #[test]
    fn error_payload_omits_optional_next_step_when_none() {
        let payload = ErrorPayload {
            code: ErrorCode::UsageError,
            message: "A search query is required.".into(),
            next_step: None,
            candidates: vec![],
            retryable: false,
        };
        let json = serde_json::to_string(&payload).unwrap();
        assert!(
            !json.contains("next_step"),
            "next_step should be omitted when None, got: {json}"
        );
    }

    // T-EN002: error_payload_omits_candidates_when_empty
    // Perspective: boundary (empty Vec is the min case). An empty candidates list
    // must not emit the key.
    #[test]
    fn error_payload_omits_candidates_when_empty() {
        let payload = ErrorPayload {
            code: ErrorCode::UsageError,
            message: "No session found matching 'dead'.".into(),
            next_step: None,
            candidates: vec![],
            retryable: false,
        };
        let json = serde_json::to_string(&payload).unwrap();
        assert!(
            !json.contains("candidates"),
            "candidates should be omitted when empty, got: {json}"
        );
    }

    // T-EN003: error_payload_serializes_present_optional_fields
    // Perspective: equivalence (the "guidance present" class — both optional
    // fields populated). next_step=Some and a non-empty candidates list both
    // appear with their values.
    #[test]
    fn error_payload_serializes_present_optional_fields() {
        let payload = ErrorPayload {
            code: ErrorCode::UsageError,
            message: "Multiple sessions match 'de'.".into(),
            next_step: Some("Narrow the session ID prefix.".into()),
            candidates: vec!["deadbeef".into(), "decade01".into()],
            retryable: false,
        };
        let json = serde_json::to_string(&payload).unwrap();
        assert!(
            json.contains(r#""next_step":"Narrow the session ID prefix.""#),
            "next_step value should appear, got: {json}"
        );
        assert!(
            json.contains(r#""candidates":["deadbeef","decade01"]"#),
            "candidates values should appear, got: {json}"
        );
    }

    // T-EN004: error_payload_renders_code_in_context
    // Perspective: equivalence (the code field renders the enum to its
    // SCREAMING_SNAKE_CASE string when nested in a payload). The exhaustive
    // variant table lives in error.rs (T-ERR002); this pins the in-context
    // rendering this module owns, plus the retryable=true class.
    #[test]
    fn error_payload_renders_code_in_context() {
        let payload = ErrorPayload {
            code: ErrorCode::TempFailure,
            message: "embedding model probe failed".into(),
            next_step: Some("Retry; the probe may succeed.".into()),
            candidates: vec![],
            retryable: true,
        };
        let json = serde_json::to_string(&payload).unwrap();
        assert!(
            json.contains(r#""code":"TEMP_FAILURE""#),
            "code should render in SCREAMING_SNAKE_CASE, got: {json}"
        );
        assert!(
            json.contains(r#""retryable":true"#),
            "retryable should render as a bool, got: {json}"
        );
    }

    // T-EN005: error_envelope_wraps_payload_under_error_key
    // Perspective: equivalence (the wrap shape). The envelope is a single object
    // whose only key is `error`, and the nested payload carries the code.
    #[test]
    fn error_envelope_wraps_payload_under_error_key() {
        let env = ErrorEnvelope {
            error: ErrorPayload {
                code: ErrorCode::UsageError,
                message: "A search query is required.".into(),
                next_step: None,
                candidates: vec![],
                retryable: false,
            },
        };
        let json = serde_json::to_string(&env).unwrap();
        assert!(
            json.starts_with(r#"{"error":"#),
            "envelope must start with `{{\"error\":`, got: {json}"
        );
        assert!(
            json.contains(r#""code":"USAGE_ERROR""#),
            "payload must contain the code, got: {json}"
        );
    }

    // T-EN006: success_envelope_serializes_required_fields
    // Perspective: equivalence (the ideal-path shape — not degraded, no notes).
    // data, degraded, and notes are always present, even when notes is empty.
    #[test]
    fn success_envelope_serializes_required_fields() {
        let env = SuccessEnvelope {
            data: serde_json::json!({"results": []}),
            degraded: false,
            notes: vec![],
        };
        let json = serde_json::to_string(&env).unwrap();
        assert!(
            json.contains(r#""data":{"results":[]}"#),
            "data should appear, got: {json}"
        );
        assert!(
            json.contains(r#""degraded":false"#),
            "degraded should appear, got: {json}"
        );
        assert!(
            json.contains(r#""notes":[]"#),
            "notes should appear, got: {json}"
        );
    }

    // T-EN007: success_envelope_surfaces_degradation_with_notes
    // Perspective: equivalence (the degraded class — semantic search unavailable,
    // FTS fallback taken). degraded=true plus a populated note both surface.
    #[test]
    fn success_envelope_surfaces_degradation_with_notes() {
        let env = SuccessEnvelope {
            data: serde_json::json!({"results": []}),
            degraded: true,
            notes: vec!["semantic search unavailable, falling back to FTS".into()],
        };
        let json = serde_json::to_string(&env).unwrap();
        assert!(
            json.contains(r#""degraded":true"#),
            "degraded should surface, got: {json}"
        );
        assert!(
            json.contains(r#""notes":["semantic search unavailable, falling back to FTS"]"#),
            "the degradation note should surface, got: {json}"
        );
    }

    // T-EN008: render_json_success_serializes_command_output
    // Perspective: equivalence (the renderer maps a non-degraded CommandOutput to
    // the success wire shape). The data payload, degraded=false, and empty notes
    // all round-trip through the renderer.
    #[test]
    fn render_json_success_serializes_command_output() {
        let out = CommandOutput::ok(
            "Sessions: 5".into(),
            serde_json::json!({"sessions": 5, "qa_chunks": 12, "embedded": 0}),
        );
        let json = render_json_success(&out);
        assert!(
            json.contains(r#""sessions":5"#),
            "data should carry the runner-built keys, got: {json}"
        );
        assert!(
            json.contains(r#""degraded":false"#),
            "degraded should be false for ok(), got: {json}"
        );
        assert!(
            json.contains(r#""notes":[]"#),
            "notes should be empty, got: {json}"
        );
    }

    // T-EN009: render_json_success_surfaces_notes
    // Perspective: equivalence (the renderer carries degradation through). A
    // with_notes CommandOutput surfaces degraded=true and the note text.
    #[test]
    fn render_json_success_surfaces_notes() {
        let out = CommandOutput::with_notes(
            "No matching sessions found.".into(),
            serde_json::json!({"results": []}),
            true,
            vec!["semantic search unavailable, falling back to FTS".into()],
        );
        let json = render_json_success(&out);
        assert!(
            json.contains(r#""degraded":true"#),
            "degraded should surface, got: {json}"
        );
        assert!(
            json.contains(r#""semantic search unavailable, falling back to FTS""#),
            "the note text should surface, got: {json}"
        );
    }

    // T-EN010: render_json_error_wraps_payload
    // Perspective: equivalence (the error renderer maps a prepared envelope to the
    // wire shape). The `{"error":...}` wrap, the code string, and retryable=true
    // all survive serialization.
    #[test]
    fn render_json_error_wraps_payload() {
        let env = ErrorEnvelope {
            error: ErrorPayload {
                code: ErrorCode::TempFailure,
                message: "embedding model probe failed".into(),
                next_step: Some("Retry; the probe may succeed.".into()),
                candidates: vec![],
                retryable: true,
            },
        };
        let json = render_json_error(&env);
        assert!(
            json.starts_with(r#"{"error":"#),
            "the error envelope must start with `{{\"error\":`, got: {json}"
        );
        assert!(
            json.contains(r#""code":"TEMP_FAILURE""#),
            "the code should appear, got: {json}"
        );
        assert!(
            json.contains(r#""retryable":true"#),
            "retryable should appear, got: {json}"
        );
    }

    // T-EN011: render_json_error_marks_non_retryable
    // Perspective: condition (the false side of retryable). A usage error renders
    // retryable=false — the complement of T-EN010's true case, so the field is
    // proven to vary, not hard-coded.
    #[test]
    fn render_json_error_marks_non_retryable() {
        let env = ErrorEnvelope {
            error: ErrorPayload {
                code: ErrorCode::UsageError,
                message: "A search query is required.".into(),
                next_step: Some("Run `recall search \"query\"`.".into()),
                candidates: vec![],
                retryable: false,
            },
        };
        let json = render_json_error(&env);
        assert!(
            json.contains(r#""code":"USAGE_ERROR""#),
            "the code should appear, got: {json}"
        );
        assert!(
            json.contains(r#""retryable":false"#),
            "a usage error must render retryable=false, got: {json}"
        );
    }
}
