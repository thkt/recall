use super::*;

#[test]
fn error_code_numbers_match_sysexits_baseline() {
    assert_eq!(ErrorCode::UsageError.code(), 64);
    assert_eq!(ErrorCode::DataError.code(), 65);
    assert_eq!(ErrorCode::Internal.code(), 70);
    assert_eq!(ErrorCode::IoError.code(), 74);
    assert_eq!(ErrorCode::TempFailure.code(), 75);
    assert_eq!(ErrorCode::Unknown.code(), 104);
}

#[test]
fn recall_error_classifies_per_variant() {
    assert_eq!(
        RecallError::Usage("x".into()).error_code(),
        ErrorCode::UsageError
    );
    assert_eq!(
        RecallError::DataError("x".into()).error_code(),
        ErrorCode::DataError
    );
    assert_eq!(
        RecallError::Internal("x".into()).error_code(),
        ErrorCode::Internal
    );
    assert_eq!(
        RecallError::TempFailure("x".into()).error_code(),
        ErrorCode::TempFailure
    );
}

#[test]
fn classify_recovers_typed_recall_error_through_anyhow() {
    let err = anyhow::Error::new(RecallError::DataError("bad query".into()));
    assert_eq!(classify(&err), ErrorCode::DataError);
}

#[test]
fn classify_recovers_recall_error_under_context() {
    // A typed error wrapped in extra anyhow context still classifies.
    let err = anyhow::Error::new(RecallError::Usage("no query".into())).context("while searching");
    assert_eq!(classify(&err), ErrorCode::UsageError);
}

#[test]
fn classify_maps_sanitize_vocab_lookup_to_io() {
    let err = anyhow::Error::new(SanitizeError::VocabLookupFailed("disk".into()));
    assert_eq!(classify(&err), ErrorCode::IoError);
}

#[test]
fn classify_maps_invalid_vocab_table_to_internal() {
    let err = anyhow::Error::new(SanitizeError::InvalidVocabTable("1bad".into()));
    assert_eq!(classify(&err), ErrorCode::Internal);
}

#[test]
fn classify_maps_empty_query_to_data_error() {
    assert_eq!(
        classify(&anyhow::Error::new(SanitizeError::EmptyInput)),
        ErrorCode::DataError
    );
    assert_eq!(
        classify(&anyhow::Error::new(SanitizeError::NoSearchableTerms)),
        ErrorCode::DataError
    );
}

// The public ExitCode wrappers (classify_exit_code, CliError::exit_code)
// must agree with the ErrorCode::code() table they delegate to.
#[test]
fn exit_code_wrappers_agree_with_code_table() {
    assert_eq!(
        classify_exit_code(&anyhow::Error::new(RecallError::Usage("x".into()))),
        ExitCode::from(codes::USAGE)
    );
    assert_eq!(
        RecallError::DataError("x".into()).exit_code(),
        ExitCode::from(codes::DATA_ERROR)
    );
}

#[test]
fn classify_maps_raw_io_error() {
    let err = anyhow::Error::new(io::Error::other("boom"));
    assert_eq!(classify(&err), ErrorCode::IoError);
}

#[test]
fn classify_falls_back_to_unknown_for_unmapped_anyhow() {
    let err = anyhow::anyhow!("something unclassified");
    assert_eq!(classify(&err), ErrorCode::Unknown);
}

// A permanent backend mismatch must not be retryable; only network/probe
// failures are transient.
#[test]
fn download_error_classifies_backend_as_permanent() {
    assert_eq!(
        download_error(&ModelDownloadError::BackendUnavailable).error_code(),
        ErrorCode::Internal
    );
    assert_eq!(
        download_error(&ModelDownloadError::DownloadFailed("net".into())).error_code(),
        ErrorCode::TempFailure
    );
    assert_eq!(
        download_error(&ModelDownloadError::ProbeFailed(Some("corrupt".into()))).error_code(),
        ErrorCode::TempFailure
    );
}

// -- #67 Phase 2 (--json envelope) --
//
// These tests pin the `--json` error surface: the SCREAMING_SNAKE_CASE serde
// table for `ErrorCode` and the `to_error_envelope` / `error_envelope`
// mapping (code, message, next_step, retryable) per `RecallError` variant.

// T-ERR002: error_code_serializes_screaming_snake_case
// Perspective: equivalence (one representative per variant). The canonical
// serde table — owned here because the derive lives on this enum. envelope.rs
// only asserts the code rendering in-context, not this exhaustive list.
#[test]
fn error_code_serializes_screaming_snake_case() {
    let pairs = [
        (ErrorCode::UsageError, r#""USAGE_ERROR""#),
        (ErrorCode::DataError, r#""DATA_ERROR""#),
        (ErrorCode::Internal, r#""INTERNAL""#),
        (ErrorCode::IoError, r#""IO_ERROR""#),
        (ErrorCode::TempFailure, r#""TEMP_FAILURE""#),
        (ErrorCode::Unknown, r#""UNKNOWN""#),
    ];
    for (code, expected) in pairs {
        let actual = serde_json::to_string(&code).unwrap();
        assert_eq!(
            actual, expected,
            "code {code:?} should serialize as {expected}"
        );
    }
}

// T-ERR003: to_error_envelope_marks_usage_non_retryable_with_next_step
// Perspective: equivalence (the Usage class). A usage error is not retryable
// and carries actionable next_step guidance (the agent fixes the invocation).
#[test]
fn to_error_envelope_marks_usage_non_retryable_with_next_step() {
    let env = RecallError::Usage("A search query is required.".into()).to_error_envelope();
    assert_eq!(env.error.code, ErrorCode::UsageError);
    assert!(
        !env.error.retryable,
        "a usage error must not invite a retry"
    );
    assert!(
        env.error.next_step.is_some(),
        "a usage error should carry next_step guidance, got: {:?}",
        env.error.next_step
    );
    assert_eq!(env.error.message, "A search query is required.");
}

// T-ERR004: to_error_envelope_marks_temp_failure_retryable
// Perspective: equivalence (the transient class). A TempFailure is retryable
// and carries next_step guidance. This is the retryable=true complement of
// the Usage / Internal / DataError cases. Asserts the behavior (retryable +
// guidance present), not the literal wording — matching T-ERR003, so Phase 3
// stays free to phrase the retry hint however it likes.
#[test]
fn to_error_envelope_marks_temp_failure_retryable() {
    let env = RecallError::TempFailure("embedding model probe failed".into()).to_error_envelope();
    assert_eq!(env.error.code, ErrorCode::TempFailure);
    assert!(env.error.retryable, "a transient failure must be retryable");
    assert!(
        env.error.next_step.is_some(),
        "a transient failure should carry next_step guidance, got: {:?}",
        env.error.next_step
    );
}

// T-ERR005: to_error_envelope_marks_internal_and_data_non_retryable
// Perspective: condition (the false side of retryable for the permanent
// classes). Internal (programmer/environment fault) and DataError (malformed
// input) are both terminal — a retry of the same call cannot succeed.
#[test]
fn to_error_envelope_marks_internal_and_data_non_retryable() {
    let internal = RecallError::Internal("MLX backend unavailable".into()).to_error_envelope();
    assert_eq!(internal.error.code, ErrorCode::Internal);
    assert!(
        !internal.error.retryable,
        "an internal fault must not be retryable"
    );

    let data = RecallError::DataError("unparseable query".into()).to_error_envelope();
    assert_eq!(data.error.code, ErrorCode::DataError);
    assert!(
        !data.error.retryable,
        "a malformed-input error must not be retryable"
    );
}

// T-ERR006: error_envelope falls back to classify() for an untyped anyhow
// error (no RecallError downcast), rendering UNKNOWN without guidance — the
// JSON twin of classify_falls_back_to_unknown_for_unmapped_anyhow.
#[test]
fn error_envelope_classifies_untyped_anyhow() {
    let env = error_envelope(&anyhow::anyhow!("something unclassified"));
    assert_eq!(env.error.code, ErrorCode::Unknown);
    assert!(
        !env.error.retryable,
        "an unclassified error must not be retryable"
    );
    assert!(
        env.error.next_step.is_none(),
        "an untyped error carries no structured next_step"
    );
}

// T-ERR007: error_envelope assembles the IO_ERROR envelope for a raw io::Error
// (the untyped path's IoError branch). The JSON twin of classify_maps_raw_io_error,
// which pins only the code, not the assembled envelope (code + retryable + next_step).
#[test]
fn error_envelope_assembles_io_error_for_raw_io() {
    let env = error_envelope(&anyhow::Error::new(io::Error::other("disk failure")));
    assert_eq!(env.error.code, ErrorCode::IoError);
    assert!(!env.error.retryable, "an I/O error must not be retryable");
    assert!(
        env.error.next_step.is_none(),
        "an untyped I/O error carries no structured next_step"
    );
}
