use super::*;

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

#[test]
fn test_continuation_marker_is_automated() {
    let turn =
        "This session is being continued from a previous conversation that ran out of context.";
    assert_eq!(classify_first_turn(turn), SessionType::Automated);
}

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

#[test]
fn quoted_and_mid_sentence_markers_do_not_hide_human_turns() {
    for marker in AUTOMATED_MARKERS {
        for turn in [
            format!("Explain {marker} in this log"),
            format!("`{marker}` is printed; why?"),
            format!("\"{marker}\" is printed; why?"),
            format!("```text\n{marker}\n```\nWhat does this mean?"),
        ] {
            assert_eq!(classify_session(&turn, false), SessionType::Interactive);
        }
    }
}
