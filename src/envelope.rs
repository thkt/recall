use serde::Serialize;

use crate::error::ErrorCode;

#[derive(Debug)]
pub(crate) struct CommandOutput {
    pub markdown: String,
    pub data: serde_json::Value,
    pub degraded: bool,
    pub notes: Vec<String>,
}

impl CommandOutput {
    pub(crate) fn ok(markdown: String, data: serde_json::Value) -> Self {
        Self {
            markdown,
            data,
            degraded: false,
            notes: Vec::new(),
        }
    }

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

#[derive(Debug, Serialize)]
pub(crate) struct SuccessEnvelope {
    pub data: serde_json::Value,
    pub degraded: bool,
    pub notes: Vec<String>,
}

#[derive(Debug, Serialize)]
pub(crate) struct ErrorEnvelope {
    pub error: ErrorPayload,
}

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

pub(crate) fn render_json_success(out: &CommandOutput) -> String {
    let env = SuccessEnvelope {
        data: out.data.clone(),
        degraded: out.degraded,
        notes: out.notes.clone(),
    };
    serde_json::to_string(&env).expect("SuccessEnvelope is always serializable")
}

pub(crate) fn render_json_error(env: &ErrorEnvelope) -> String {
    serde_json::to_string(env).expect("ErrorEnvelope is always serializable")
}

#[cfg(test)]
mod tests;
