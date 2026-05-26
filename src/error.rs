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

/// sysexits-derived exit-code classes recall distinguishes (ADR-0066 Group 2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ErrorCode {
    /// Command used wrong: missing query, unresolved or ambiguous id, no index.
    Usage,
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
            Self::Usage => codes::USAGE,
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
            Self::Usage(_) => ErrorCode::Usage,
            Self::DataError(_) => ErrorCode::DataError,
            Self::Internal(_) => ErrorCode::Internal,
            Self::TempFailure(_) => ErrorCode::TempFailure,
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
        assert_eq!(ErrorCode::Usage.code(), 64);
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
            ErrorCode::Usage
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
        assert_eq!(classify(&err), ErrorCode::Usage);
    }

    #[test]
    fn classify_maps_sanitize_vocab_lookup_to_io() {
        let err = anyhow::Error::new(SanitizeError::VocabLookupFailed("disk".into()));
        assert_eq!(classify(&err), ErrorCode::IoError);
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
            ErrorCode::Usage
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
}
