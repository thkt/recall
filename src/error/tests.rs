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
    assert!(
        data.error
            .next_step
            .as_deref()
            .is_some_and(|hint| hint.contains("query")),
        "malformed queries must retain query repair guidance"
    );
}

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
