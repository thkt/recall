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
