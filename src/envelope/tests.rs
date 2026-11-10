use super::{CommandOutput, ErrorEnvelope, ErrorPayload, render_json_error, render_json_success};
use crate::error::ErrorCode;

#[test]
fn a_success_envelope_carries_data_degraded_and_notes() {
    let out = CommandOutput::ok("Sessions: 5".into(), serde_json::json!({"sessions": 5}));

    assert_eq!(
        render_json_success(&out),
        r#"{"data":{"sessions":5},"degraded":false,"notes":[]}"#
    );
}

#[test]
fn a_degraded_success_envelope_carries_its_notes() {
    let out = CommandOutput::with_notes(
        "No matching sessions found.".into(),
        serde_json::json!({"results": []}),
        true,
        vec!["semantic search unavailable, falling back to FTS".into()],
    );

    assert_eq!(
        render_json_success(&out),
        r#"{"data":{"results":[]},"degraded":true,"notes":["semantic search unavailable, falling back to FTS"]}"#
    );
}

#[test]
fn an_error_envelope_leaves_out_an_absent_next_step_and_empty_candidates() {
    let env = ErrorEnvelope {
        error: ErrorPayload {
            code: ErrorCode::UsageError,
            message: "A search query is required.".into(),
            next_step: None,
            candidates: vec![],
            retryable: false,
        },
    };

    assert_eq!(
        render_json_error(&env),
        r#"{"error":{"code":"USAGE_ERROR","message":"A search query is required.","retryable":false}}"#
    );
}

#[test]
fn an_error_envelope_carries_next_step_and_candidates_when_present() {
    let env = ErrorEnvelope {
        error: ErrorPayload {
            code: ErrorCode::UsageError,
            message: "Multiple sessions match 'de'.".into(),
            next_step: Some("Narrow the session ID prefix.".into()),
            candidates: vec!["deadbeef".into(), "decade01".into()],
            retryable: false,
        },
    };

    assert_eq!(
        render_json_error(&env),
        r#"{"error":{"code":"USAGE_ERROR","message":"Multiple sessions match 'de'.","next_step":"Narrow the session ID prefix.","candidates":["deadbeef","decade01"],"retryable":false}}"#
    );
}
