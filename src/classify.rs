//! Session classification (#24): label a session interactive or automated from
//! its first user turn, so `recall search` can exclude hook/script-generated
//! noise by default. Markers are programmatic prefixes intrinsic to the
//! Claude/Codex ecosystem (slash-command wrappers, local-command I/O, synthetic
//! actions, interrupt/continuation strings). A false positive would hide a real
//! session from default search, so only prefixes a human is very unlikely to open
//! a session with belong here. Matched literally — no regex/config dependency.

/// Classification of a session, derived from its first user turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionType {
    /// Human-driven session. The default whenever no marker matches.
    Interactive,
    /// Hook / script / slash-command / agent-generated session.
    Automated,
}

impl SessionType {
    /// The value stored in `sessions.session_type`.
    pub fn as_str(self) -> &'static str {
        match self {
            SessionType::Interactive => "interactive",
            SessionType::Automated => "automated",
        }
    }
}

/// Programmatic prefixes that mark a session as automated. Each is a string
/// tooling injects verbatim at the start of a turn but a human is very unlikely
/// to open a session with: slash-command wrappers, synthetic actions, local-command
/// I/O, the interrupt notice, and the compaction-continuation opener. Verified
/// against ~/.recall.db (5899 sessions). Keep entries unambiguous — a false match
/// hides a real session from default search.
const AUTOMATED_MARKERS: &[&str] = &[
    "<command-message>",
    "<user_action>",
    "<local-command-stdout>",
    "<local-command-caveat>",
    "<bash-input>",
    "[Request interrupted by user",
    "This session is being continued",
];

/// Classify a session from its first user turn. Leading whitespace is trimmed
/// (tooling often injects a newline + indent before the marker), then the turn is
/// matched against [`AUTOMATED_MARKERS`] at its start. Anchoring at the start means
/// a marker appearing mid-sentence in a genuine human turn does not trip the match,
/// and the scan is inherently bounded to the marker lengths (well within the first
/// 120 characters). Returns [`SessionType::Interactive`] when nothing matches —
/// including an empty or whitespace-only turn — so an unmatched session is never
/// hidden by default.
pub fn classify_first_turn(first_turn: &str) -> SessionType {
    let head = first_turn.trim_start();
    if AUTOMATED_MARKERS
        .iter()
        .any(|marker| head.starts_with(marker))
    {
        SessionType::Automated
    } else {
        SessionType::Interactive
    }
}

#[cfg(test)]
mod tests;
