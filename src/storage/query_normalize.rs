//! Query normalization for FTS5 indexing and retrieval (Phase 5, Issue #69).
//!
//! Resolves common Japanese/Latin notation drift before sanitization and
//! short-term expansion. Applied to **both** indexed text and query text so
//! the FTS5 token streams agree — applying only to one side leaves the index
//! holding the un-normalized form and silently misses matches.
//!
//! # Pipeline order
//!
//! 1. NFKC compatibility composition (`unicode-normalization`).
//!    Folds full-width Latin/digits to half-width and unifies compatibility
//!    characters. Hiragana ↔ Katakana is intentionally **not** mapped — that
//!    requires a separate Issue once evidence justifies it.
//! 2. ASCII lowercase. Japanese characters are left untouched (case is a
//!    Latin-only concept here); Unicode case folding is deferred to a future
//!    Issue if measurements show a need.
//! 3. Whitespace collapse. Trims leading/trailing whitespace and collapses
//!    runs to a single space so trigram boundaries stay deterministic across
//!    full-width spaces (`U+3000`) NFKC-folded into ASCII spaces.

use serde::{Deserialize, Serialize};
use unicode_normalization::UnicodeNormalization;

/// Per-step toggles for the [`normalize_for_fts`] pipeline.
///
/// Every step defaults to **on** at runtime — the pipeline is meant to be
/// applied transparently. The `serde` `Deserialize` path uses
/// [`pre_phase_5_disabled`] so historical baseline files (captured before
/// Phase 5 existed) round-trip with normalization disabled, preserving the
/// numbers they were captured under.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueryNormalizationConfig {
    /// Apply NFKC compatibility composition (full-width → half-width Latin).
    pub nfkc: bool,
    /// Apply ASCII lowercase (`A-Z` → `a-z`). Non-ASCII characters untouched.
    pub ascii_lowercase: bool,
    /// Trim and collapse runs of whitespace to a single space.
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
    /// All steps off — the literal pre-Phase-5 behaviour.
    ///
    /// Used by the serde-default path on `amici::eval::baseline::BaselineSnapshot`
    /// (eval harness migrated to amici per ADR 0006) so a baseline file written
    /// before normalization existed round-trips with the same metric values it
    /// was captured under.
    #[must_use]
    pub const fn disabled() -> Self {
        Self {
            nfkc: false,
            ascii_lowercase: false,
            collapse_whitespace: false,
        }
    }
}

/// Serde-default factory: pre-Phase-5 baselines lacked this field, so a
/// missing field must resolve to all-OFF, **not** to runtime [`Default`].
#[must_use]
#[cfg(test)]
pub fn pre_phase_5_disabled() -> QueryNormalizationConfig {
    QueryNormalizationConfig::disabled()
}

/// Apply the configured normalization pipeline to `text`.
///
/// Idempotent for any config: `normalize_for_fts(normalize_for_fts(x, c), c) ==
/// normalize_for_fts(x, c)`. Callers can layer this over already-normalized
/// input from downstream consumers without breaking the fixed point.
///
/// Returns the input unchanged when every step is disabled (avoids the NFKC
/// allocation on the hot path when callers explicitly opt out).
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
        buf = collapse_whitespace(&buf);
    }
    buf
}

/// Trim and collapse runs of whitespace to a single ASCII space.
fn collapse_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests;
