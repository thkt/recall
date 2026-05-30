//! Exit-code classification per ADR-0066 Group 2 baseline.
//!
//! recall stays anyhow-based internally. `RecallError` types only the explicit
//! failures recall raises, so a classification survives to the CLI boundary
//! where [`classify_exit_code`] maps the anyhow error chain to a sysexits exit
//! code (via [`amici::cli::exit_code::codes`]). This replaces the former
//! `USER_ERROR_MARKERS` stderr-string match, which silently broke whenever a
//! `bail!` message was reworded (#26 / #82 / #83).
//!
//! The ADR-0066 Group 2 baseline also defines `CANT_CREAT` (73); recall has no
//! distinct "cannot create output" path (DB/IO failures map to `IoError`), so
//! that class is intentionally absent until a path needs it.

use std::io;
use std::process::ExitCode;

use amici::cli::exit_code::{CliError, codes};
use amici::model::ModelDownloadError;
use amici::model::embedder::DegradedReason;
use rurico::storage::SanitizeError;
use serde::Serialize;

use crate::envelope::{ErrorEnvelope, ErrorPayload};

/// sysexits-derived exit-code classes recall distinguishes (ADR-0066 Group 2).
///
/// Serializes to its SCREAMING_SNAKE_CASE name for the `--json` error envelope
/// (#67 Phase 2), so agents branch on the concept name (`"USAGE_ERROR"`), not
/// the bare sysexits number.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub(crate) enum ErrorCode {
    /// Command used wrong: missing query, unresolved or ambiguous id, no index.
    UsageError,
    /// Malformed input data: an unparseable search query.
    DataError,
    /// Invariant violation or programmer error.
    Internal,
    /// I/O or SQLite failure.
    IoError,
    /// Transient failure; a retry may succeed.
    TempFailure,
    /// Unclassified: an anyhow error with no known mapping.
    Unknown,
}

impl ErrorCode {
    pub(crate) fn code(self) -> u8 {
        match self {
            Self::UsageError => codes::USAGE,
            Self::DataError => codes::DATA_ERROR,
            Self::Internal => codes::INTERNAL,
            Self::IoError => codes::IO_ERR,
            Self::TempFailure => codes::TEMP_FAIL,
            Self::Unknown => codes::UNKNOWN,
        }
    }
}

/// The explicit, classifiable failures recall raises. Internal functions keep
/// returning `anyhow::Result`; these variants are constructed at the few sites
/// that know the right exit class and recovered by downcast at the boundary.
#[derive(Debug, thiserror::Error)]
pub(crate) enum RecallError {
    /// User invoked the command wrong (maps to `USAGE`, 64).
    #[error("{0}")]
    Usage(String),
    /// User-supplied data was malformed (maps to `DATA_ERROR`, 65).
    #[error("{0}")]
    DataError(String),
    /// Invariant violation or permanent environment fault, e.g. the MLX backend
    /// is missing on non-Apple-Silicon hardware (maps to `INTERNAL`, 70).
    #[error("{0}")]
    Internal(String),
    /// Transient failure; retry may succeed (maps to `TEMP_FAILURE`, 75).
    #[error("{0}")]
    TempFailure(String),
}

impl RecallError {
    pub(crate) fn error_code(&self) -> ErrorCode {
        match self {
            Self::Usage(_) => ErrorCode::UsageError,
            Self::DataError(_) => ErrorCode::DataError,
            Self::Internal(_) => ErrorCode::Internal,
            Self::TempFailure(_) => ErrorCode::TempFailure,
        }
    }

    /// Build the `--json` error envelope for this failure (#67 Phase 2).
    ///
    /// `retryable` is true only for [`Self::TempFailure`]: retrying the same
    /// call cannot clear a usage, data, or internal fault. `next_step` carries
    /// variant-level guidance (the specific cause stays in `message`), derived
    /// from the variant rather than the message text so a reworded `bail!` never
    /// silently drops the hint (the #82 lesson that retired `USER_ERROR_MARKERS`).
    pub(crate) fn to_error_envelope(&self) -> ErrorEnvelope {
        let next_step = match self {
            Self::Usage(_) => Some(
                "See `recall --help`; if the index is missing, run `recall index` first."
                    .to_owned(),
            ),
            Self::DataError(_) => Some(
                "Revise the query: drop bare AND/OR/NOT operators and recheck the syntax."
                    .to_owned(),
            ),
            Self::TempFailure(_) => {
                Some("Retry the command; the failure may be transient.".to_owned())
            }
            Self::Internal(_) => None,
        };
        ErrorEnvelope {
            error: ErrorPayload {
                code: self.error_code(),
                message: self.to_string(),
                next_step,
                candidates: Vec::new(),
                retryable: matches!(self, Self::TempFailure(_)),
            },
        }
    }
}

impl CliError for RecallError {
    fn exit_code(&self) -> ExitCode {
        ExitCode::from(self.error_code().code())
    }
}

/// Classify an error that propagated to the CLI boundary into an exit code.
///
/// Resolution order: a typed [`RecallError`] wins; then rurico's
/// [`SanitizeError`] (a failed vocab lookup is I/O, a bad vocab table is a
/// programmer bug); then raw I/O and SQLite errors; otherwise `UNKNOWN` (104),
/// which signals an `anyhow` path that should be promoted to a typed variant.
pub(crate) fn classify_exit_code(err: &anyhow::Error) -> ExitCode {
    ExitCode::from(classify(err).code())
}

/// Build the `--json` error envelope for any error reaching the CLI boundary
/// (#67 Phase 2). A typed [`RecallError`] carries its own `next_step` /
/// `retryable` via [`RecallError::to_error_envelope`]; an untyped `anyhow` error
/// is classified by [`classify`] and rendered without structured guidance
/// (`next_step: None`), mirroring the exit-code path so the JSON and the exit
/// code never disagree.
pub(crate) fn error_envelope(err: &anyhow::Error) -> ErrorEnvelope {
    if let Some(e) = err.downcast_ref::<RecallError>() {
        return e.to_error_envelope();
    }
    let code = classify(err);
    ErrorEnvelope {
        error: ErrorPayload {
            code,
            message: format!("{err:#}"),
            next_step: None,
            candidates: Vec::new(),
            retryable: matches!(code, ErrorCode::TempFailure),
        },
    }
}

fn classify(err: &anyhow::Error) -> ErrorCode {
    if let Some(e) = err.downcast_ref::<RecallError>() {
        return e.error_code();
    }
    if let Some(e) = err.downcast_ref::<SanitizeError>() {
        return match e {
            SanitizeError::VocabLookupFailed(_) => ErrorCode::IoError,
            SanitizeError::InvalidVocabTable(_) => ErrorCode::Internal,
            SanitizeError::EmptyInput | SanitizeError::NoSearchableTerms => ErrorCode::DataError,
        };
    }
    if err.downcast_ref::<io::Error>().is_some() || err.downcast_ref::<rusqlite::Error>().is_some()
    {
        return ErrorCode::IoError;
    }
    ErrorCode::Unknown
}

/// Classify a `recall model download` failure. A failed HTTP download or a
/// re-downloadable probe failure is transient (retry may help); a missing MLX
/// backend is a permanent hardware mismatch, so it must not invite a retry.
pub(crate) fn download_error(e: &ModelDownloadError) -> RecallError {
    let msg = e.to_string();
    match e {
        ModelDownloadError::DownloadFailed(_) | ModelDownloadError::ProbeFailed(_) => {
            RecallError::TempFailure(msg)
        }
        ModelDownloadError::BackendUnavailable => RecallError::Internal(msg),
    }
}

/// Classify why the embedder failed to load for a command that requires it
/// (`recall embed`). A missing model is a usage error the user resolves with
/// `recall model download`; a missing backend is a permanent fault; a probe
/// failure may clear on retry.
pub(crate) fn embedder_error(reason: DegradedReason) -> RecallError {
    match reason {
        DegradedReason::NotInstalled => RecallError::Usage(
            "embedding model not installed; run `recall model download`".to_owned(),
        ),
        DegradedReason::ProbeFailed => {
            RecallError::TempFailure("embedding model probe failed".to_owned())
        }
        DegradedReason::BackendUnavailable => {
            RecallError::Internal("MLX backend unavailable".to_owned())
        }
        DegradedReason::Disabled => RecallError::Internal("embedder disabled".to_owned()),
    }
}

#[cfg(test)]
mod tests {
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
        let err =
            anyhow::Error::new(RecallError::Usage("no query".into())).context("while searching");
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
            download_error(&ModelDownloadError::ProbeFailed("corrupt".into())).error_code(),
            ErrorCode::TempFailure
        );
    }

    // `recall embed` without the model installed is a usage error the user
    // fixes with `recall model download`, not an Unknown(104) catch-all.
    #[test]
    fn embedder_error_classifies_not_installed_as_usage() {
        assert_eq!(
            embedder_error(DegradedReason::NotInstalled).error_code(),
            ErrorCode::UsageError
        );
        assert_eq!(
            embedder_error(DegradedReason::BackendUnavailable).error_code(),
            ErrorCode::Internal
        );
        assert_eq!(
            embedder_error(DegradedReason::ProbeFailed).error_code(),
            ErrorCode::TempFailure
        );
        assert_eq!(
            embedder_error(DegradedReason::Disabled).error_code(),
            ErrorCode::Internal
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
        let env =
            RecallError::TempFailure("embedding model probe failed".into()).to_error_envelope();
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
}
