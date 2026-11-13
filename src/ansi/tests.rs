use super::*;

#[test]
fn text_without_control_characters_is_returned_unchanged() {
    assert_eq!(strip_control_chars("normal text"), "normal text");
    assert_eq!(strip_control_chars("line1\nline2"), "line1\nline2");
}

#[test]
fn terminal_escape_sequences_are_removed() {
    assert_eq!(
        strip_control_chars("hello \x1b[31mred\x1b[0m world"),
        "hello red world"
    );
    assert_eq!(
        strip_control_chars("before\x1b]0;title\x07after"),
        "beforeafter"
    );
    assert_eq!(
        strip_control_chars("before\x1b]0;title\x1b\\after"),
        "beforeafter"
    );
}

#[test]
fn an_unterminated_escape_sequence_drops_the_rest_of_the_text() {
    assert_eq!(strip_control_chars("before\x1b]0;title"), "before");
    assert_eq!(strip_control_chars("before\x1b[31"), "before");
}

#[test]
fn tabs_become_spaces_and_other_control_characters_are_removed_except_newlines() {
    assert_eq!(strip_control_chars("foo\tbar"), "foo bar");
    assert_eq!(strip_control_chars("a\t\tb"), "a  b");
    assert_eq!(strip_control_chars("foo\rbar"), "foobar");
    assert_eq!(strip_control_chars("a\x1bb"), "ab");
    assert_eq!(strip_control_chars("a\u{1}b\u{7f}c\u{85}d\ne"), "abcd\ne");
}

#[test]
fn an_escape_sequence_ending_in_a_final_byte_other_than_a_letter_keeps_the_text_after_it() {
    assert_eq!(
        strip_control_chars("\x1b[200~こんにちは。テストです\x1b[201~"),
        "こんにちは。テストです"
    );
    assert_eq!(
        strip_control_chars("a \x1b[2~123 456 done"),
        "a 123 456 done"
    );
}

#[test]
fn an_escape_sequence_broken_by_a_character_it_cannot_contain_keeps_that_character() {
    assert_eq!(strip_control_chars("\x1b[3こんにちは"), "こんにちは");
    assert_eq!(strip_control_chars("before\x1b[31\nafter"), "before\nafter");
}

#[test]
fn an_unterminated_osc_sequence_drops_only_the_rest_of_its_line() {
    assert_eq!(
        strip_control_chars("before\x1b]0;title\nafter"),
        "before\nafter"
    );
}
