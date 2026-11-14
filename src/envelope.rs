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

/// A note on a command's result. `--json` carries every note; a text run prints on stderr the
/// notes the command has not shown already (#403).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Note {
    pub text: String,
    /// The command already put this in its text output, on stdout or stderr.
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
    data: Stripped<'a>,
    degraded: bool,
    notes: Vec<Cow<'a, str>>,
}

/// Serializes a JSON value with control characters removed from every string in it. Object keys
/// stay as they are, since the commands name them.
struct Stripped<'a>(&'a serde_json::Value);

impl Serialize for Stripped<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self.0 {
            serde_json::Value::String(s) => serializer.serialize_str(&ansi::strip_control_chars(s)),
            serde_json::Value::Array(items) => serializer.collect_seq(items.iter().map(Stripped)),
            serde_json::Value::Object(map) => {
                serializer.collect_map(map.iter().map(|(key, value)| (key, Stripped(value))))
            }
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

/// Session content reaches these strings, so the envelope strips every one of them (DR 0001).
pub(crate) fn render_json_success(out: &CommandOutput) -> String {
    let env = SuccessEnvelope {
        data: Stripped(&out.data),
        degraded: out.degraded,
        notes: out
            .notes
            .iter()
            .map(|note| ansi::strip_control_chars(&note.text))
            .collect(),
    };
    serde_json::to_string(&env).expect("SuccessEnvelope is always serializable")
}

/// The stderr lines a text run adds for the notes its command has not shown, control characters
/// removed as in the envelope.
pub(crate) fn render_text_notes(out: &CommandOutput) -> String {
    out.notes
        .iter()
        .filter(|note| !note.shown)
        .map(|note| format!("note: {}\n", ansi::strip_control_chars(&note.text)))
        .collect()
}

pub(crate) fn render_json_error(env: &ErrorEnvelope) -> String {
    serde_json::to_string(env).expect("ErrorEnvelope is always serializable")
}

#[cfg(test)]
mod tests;
