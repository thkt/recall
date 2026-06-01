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
mod tests {
    use super::*;

    // T-001 (#24/FR-001): XML-wrapper markers classify as automated, even with the
    // leading newline + indent that tooling injects before the tag.
    #[test]
    fn test_xml_wrapper_with_leading_whitespace_is_automated() {
        for turn in [
            "\n            <command-message>clear</command-message>",
            "<user_action>\n  <context>review</context>",
            "<local-command-stdout>Goodbye!</local-command-stdout>",
            "<bash-input>git push</bash-input>",
        ] {
            assert_eq!(
                classify_first_turn(turn),
                SessionType::Automated,
                "turn: {turn:?}"
            );
        }
    }

    // T-002 (#24/FR-001): the compaction continuation opener classifies as automated.
    #[test]
    fn test_continuation_marker_is_automated() {
        let turn =
            "This session is being continued from a previous conversation that ran out of context.";
        assert_eq!(classify_first_turn(turn), SessionType::Automated);
    }

    // T-003 (#24/FR-002, FR-010): a normal human turn and an empty/whitespace turn
    // (a session with no user turn) both classify as interactive, so they are never
    // hidden from default search.
    #[test]
    fn test_human_and_empty_turn_is_interactive() {
        for turn in ["how do I implement authentication?", "", "   \n  "] {
            assert_eq!(
                classify_first_turn(turn),
                SessionType::Interactive,
                "turn: {turn:?}"
            );
        }
    }
}
