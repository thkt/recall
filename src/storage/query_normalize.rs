use serde::{Deserialize, Serialize};
use unicode_normalization::UnicodeNormalization;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueryNormalizationConfig {
    pub nfkc: bool,
    pub ascii_lowercase: bool,
    pub collapse_whitespace: bool,
}

impl Default for QueryNormalizationConfig {
    fn default() -> Self {
        Self {
            nfkc: true,
            ascii_lowercase: true,
            collapse_whitespace: true,
        }
    }
}

impl QueryNormalizationConfig {
    #[must_use]
    pub const fn disabled() -> Self {
        Self {
            nfkc: false,
            ascii_lowercase: false,
            collapse_whitespace: false,
        }
    }
}

#[must_use]
#[cfg(test)]
pub fn pre_phase_5_disabled() -> QueryNormalizationConfig {
    QueryNormalizationConfig::disabled()
}

#[must_use]
pub fn normalize_for_fts(text: &str, config: &QueryNormalizationConfig) -> String {
    if !config.nfkc && !config.ascii_lowercase && !config.collapse_whitespace {
        return text.to_owned();
    }
    let mut buf = if config.nfkc {
        text.nfkc().collect::<String>()
    } else {
        text.to_owned()
    };
    if config.ascii_lowercase {
        buf.make_ascii_lowercase();
    }
    if config.collapse_whitespace {
        buf = buf.split_whitespace().collect::<Vec<_>>().join(" ");
    }
    buf
}

#[cfg(test)]
mod tests;
