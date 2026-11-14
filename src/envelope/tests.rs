use super::{
    CommandOutput, ErrorEnvelope, ErrorPayload, Note, render_json_error, render_json_success,
    render_text_notes,
};
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
        vec![Note::new(
            "semantic search unavailable, falling back to FTS",
        )],
    );

    assert_eq!(
        render_json_success(&out),
        r#"{"data":{"results":[]},"degraded":true,"notes":["semantic search unavailable, falling back to FTS"]}"#
    );
}

#[test]
fn a_success_envelope_strips_control_characters_from_every_string_value_and_note() {
    let out = CommandOutput::with_notes(
        String::new(),
        serde_json::json!({
            "session": { "slug": "fix-\u{1b}[1mlogin\u{1b}[0m", "lines": ["a\u{7}", "tab\there\nnext"] },
            "count": 2,
        }),
        true,
        vec![Note::new("see /work/\u{1b}]0;PROJ\u{7}app")],
    );

    assert_eq!(
        render_json_success(&out),
        r#"{"data":{"count":2,"session":{"lines":["a","tab here\nnext"],"slug":"fix-login"}},"degraded":true,"notes":["see /work/app"]}"#
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

#[test]
fn a_text_run_prints_only_the_notes_not_yet_shown_without_control_characters() {
    let out = CommandOutput::with_notes(
        String::new(),
        serde_json::json!({}),
        true,
        vec![
            Note::shown("already on stderr"),
            Note::new("see /work/\u{1b}]0;PROJ\u{7}app"),
            Note::new("first line\nsecond line"),
        ],
    );

    assert_eq!(
        render_text_notes(&out),
        "note: see /work/app\nnote: first line second line\n"
    );
}
