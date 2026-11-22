use std::borrow::Cow;

use serde::{Serialize, Serializer};

use crate::ansi;
use crate::error::ErrorCode;

#[derive(Debug)]
pub(crate) struct CommandOutput {
    pub markdown: String,
    pub data: serde_json::Value,
    pub degraded: bool,
    pub notes: Vec<Note>,
}

#[derive(Debug)]
pub(crate) struct Note {
    pub text: String,
    pub shown: bool,
}

impl Note {
    pub(crate) fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            shown: false,
        }
    }

    pub(crate) fn shown(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            shown: true,
        }
    }
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
        notes: Vec<Note>,
    ) -> Self {
        Self {
            markdown,
            data,
            degraded,
            notes,
        }
    }
}

#[derive(Serialize)]
struct SuccessEnvelope<'a> {
    data: StringValuesStripped<'a>,
    degraded: bool,
    notes: Vec<Cow<'a, str>>,
}

struct StringValuesStripped<'a>(&'a serde_json::Value);

impl Serialize for StringValuesStripped<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self.0 {
            serde_json::Value::String(s) => serializer.serialize_str(&ansi::strip_control_chars(s)),
            serde_json::Value::Array(items) => {
                serializer.collect_seq(items.iter().map(StringValuesStripped))
            }
            serde_json::Value::Object(map) => serializer.collect_map(
                map.iter()
                    .map(|(key, value)| (key, StringValuesStripped(value))),
            ),
            other => other.serialize(serializer),
        }
    }
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
        data: StringValuesStripped(&out.data),
        degraded: out.degraded,
        notes: out
            .notes
            .iter()
            .map(|note| ansi::strip_control_chars(&note.text))
            .collect(),
    };
    serde_json::to_string(&env).expect("SuccessEnvelope is always serializable")
}

pub(crate) fn render_text_notes(out: &CommandOutput) -> String {
    out.notes
        .iter()
        .filter(|note| !note.shown)
        .map(|note| {
            format!(
                "note: {}\n",
                ansi::strip_control_chars(&note.text).replace('\n', " ")
            )
        })
        .collect()
}

pub(crate) fn render_json_error(env: &ErrorEnvelope) -> String {
    serde_json::to_string(env).expect("ErrorEnvelope is always serializable")
}

#[cfg(test)]
mod tests;
