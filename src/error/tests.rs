use super::*;
use crate::envelope::render_json_error;

/// What a failing command reports: its exit code and the `--json` error envelope.
fn reported(err: &anyhow::Error) -> (ExitCode, String) {
    (
        classify_exit_code(err),
        render_json_error(&error_envelope(err)),
    )
}

#[test]
fn a_usage_error_exits_64_and_points_at_the_help() {
    assert_eq!(
        reported(&RecallError::Usage("A search query is required.".into()).into()),
        (
            ExitCode::from(64),
            r#"{"error":{"code":"USAGE_ERROR","message":"A search query is required.","next_step":"See `recall --help`; if the index is missing, run `recall index` first.","retryable":false}}"#.to_owned()
        )
    );
}

#[test]
fn a_rejected_query_exits_65_and_asks_to_revise_it() {
    assert_eq!(
        reported(&RecallError::DataError("unparseable query".into()).into()),
        (
            ExitCode::from(65),
            r#"{"error":{"code":"DATA_ERROR","message":"unparseable query","next_step":"Revise the query: drop bare AND/OR/NOT operators and recheck the syntax.","retryable":false}}"#.to_owned()
        )
    );
}

#[test]
fn an_unreadable_index_exits_65_without_a_query_hint() {
    assert_eq!(
        reported(&RecallError::DatabaseRead("database schema is out of date".into()).into()),
        (
            ExitCode::from(65),
            r#"{"error":{"code":"DATA_ERROR","message":"database schema is out of date","retryable":false}}"#.to_owned()
        )
    );
}

#[test]
fn an_internal_fault_exits_70_and_is_not_retryable() {
    assert_eq!(
        reported(&RecallError::Internal("backend unavailable".into()).into()),
        (
            ExitCode::from(70),
            r#"{"error":{"code":"INTERNAL","message":"backend unavailable","retryable":false}}"#
                .to_owned()
        )
    );
}

#[test]
fn a_transient_failure_exits_75_and_invites_a_retry() {
    assert_eq!(
        reported(&RecallError::TempFailure("embedding model probe failed".into()).into()),
        (
            ExitCode::from(75),
            r#"{"error":{"code":"TEMP_FAILURE","message":"embedding model probe failed","next_step":"Retry the command; the failure may be transient.","retryable":true}}"#.to_owned()
        )
    );
}

#[test]
fn context_added_on_the_way_up_keeps_the_original_route_and_message() {
    let err = anyhow::Error::new(RecallError::Usage("no query".into())).context("while searching");

    assert_eq!(
        reported(&err),
        (
            ExitCode::from(64),
            r#"{"error":{"code":"USAGE_ERROR","message":"no query","next_step":"See `recall --help`; if the index is missing, run `recall index` first.","retryable":false}}"#.to_owned()
        )
    );
}

#[test]
fn an_io_failure_exits_74() {
    assert_eq!(
        reported(&io::Error::other("disk failure").into()),
        (
            ExitCode::from(74),
            r#"{"error":{"code":"IO_ERROR","message":"disk failure","retryable":false}}"#
                .to_owned()
        )
    );
}

#[test]
fn an_unclassified_error_exits_104() {
    assert_eq!(
        reported(&anyhow::anyhow!("something unclassified")),
        (
            ExitCode::from(104),
            r#"{"error":{"code":"UNKNOWN","message":"something unclassified","retryable":false}}"#
                .to_owned()
        )
    );
}

#[test]
fn query_sanitizer_failures_exit_by_cause() {
    let exit = |e: SanitizeError| classify_exit_code(&e.into());

    assert_eq!(
        exit(SanitizeError::VocabLookupFailed("disk".into())),
        ExitCode::from(74)
    );
    assert_eq!(
        exit(SanitizeError::InvalidVocabTable("1bad".into())),
        ExitCode::from(70)
    );
    assert_eq!(exit(SanitizeError::EmptyInput), ExitCode::from(65));
    assert_eq!(exit(SanitizeError::NoSearchableTerms), ExitCode::from(65));
}

#[test]
fn a_failed_model_download_is_retryable_but_a_missing_backend_is_not() {
    let exit = |e: ModelDownloadError| classify_exit_code(&download_error(&e).into());

    assert_eq!(
        exit(ModelDownloadError::DownloadFailed("net".into())),
        ExitCode::from(75)
    );
    assert_eq!(
        exit(ModelDownloadError::ProbeFailed(Some("corrupt".into()))),
        ExitCode::from(75)
    );
    assert_eq!(
        exit(ModelDownloadError::BackendUnavailable),
        ExitCode::from(70)
    );
}
