//! Exit-code classification per recall ADR-0006 (extending scout ADR-0066 Group 2).
//!
//! recall stays anyhow-based internally. `RecallError` types only the explicit
//! failures recall raises, so a classification survives to the CLI boundary
//! where [`classify_exit_code`] maps the anyhow error chain to a sysexits exit
//! code (via [`amici::cli::exit_code::codes`]). This replaces the former
//! `USER_ERROR_MARKERS` stderr-string match, which silently broke whenever a
//! `bail!` message was reworded (#26 / #82 / #83).
//!
//! The scout ADR-0066 Group 2 baseline also defines `CANT_CREAT` (73); recall has no
//! distinct "cannot create output" path (DB/IO failures map to `IoError`), so
//! that class is intentionally absent until a path needs it.

use std::io;
use std::process::ExitCode;

use amici::cli::exit_code::{CliError, codes};
use amici::model::ModelDownloadError;
use rurico::storage::SanitizeError;
use serde::Serialize;

use crate::envelope::{ErrorEnvelope, ErrorPayload};

/// sysexits-derived exit-code classes recall distinguishes (ADR-0006).
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

#[cfg(test)]
mod tests;
