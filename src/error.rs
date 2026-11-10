use std::io;
use std::process::ExitCode;

use crate::model::ModelDownloadError;
use crate::runtime::codes;
use crate::storage::{MIN_TRIGRAM_TERM_CHARS, SanitizeError};
use serde::Serialize;

use crate::envelope::{ErrorEnvelope, ErrorPayload};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub(crate) enum ErrorCode {
    UsageError,
    DataError,
    Internal,
    IoError,
    TempFailure,
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

#[derive(Debug, thiserror::Error)]
pub(crate) enum RecallError {
    #[error("{0}")]
    Usage(String),
    #[error("{0}")]
    DataError(String),
    #[error("{0}")]
    QuerySyntax(String),
    #[error("{0}")]
    DatabaseRead(String),
    #[error("{0}")]
    Internal(String),
    #[error("{0}")]
    TempFailure(String),
}

impl RecallError {
    pub(crate) fn error_code(&self) -> ErrorCode {
        match self {
            Self::Usage(_) => ErrorCode::UsageError,
            Self::DataError(_) | Self::QuerySyntax(_) | Self::DatabaseRead(_) => {
                ErrorCode::DataError
            }
            Self::Internal(_) => ErrorCode::Internal,
            Self::TempFailure(_) => ErrorCode::TempFailure,
        }
    }

    pub(crate) fn retryable(&self) -> bool {
        matches!(self, Self::TempFailure(_))
    }

    pub(crate) fn to_error_envelope(&self) -> ErrorEnvelope {
        let next_step = match self {
            Self::Usage(_) => Some(
                "See `recall --help`; if the index is missing, run `recall index` first."
                    .to_owned(),
            ),
            Self::QuerySyntax(_) => Some(format!(
                "Supported query syntax: words, \"quoted phrases\", uppercase AND / OR / NOT between terms, and parentheses for grouping. NOT needs a left term of at least {MIN_TRIGRAM_TERM_CHARS} characters."
            )),
            Self::TempFailure(_) => {
                Some("Retry the command; the failure may be transient.".to_owned())
            }
            Self::DataError(_) | Self::Internal(_) | Self::DatabaseRead(_) => None,
        };
        ErrorEnvelope {
            error: ErrorPayload {
                code: self.error_code(),
                message: self.to_string(),
                next_step,
                candidates: Vec::new(),
                retryable: self.retryable(),
            },
        }
    }
}

pub(crate) fn classify_exit_code(err: &anyhow::Error) -> ExitCode {
    ExitCode::from(classify(err).code())
}

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
