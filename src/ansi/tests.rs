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
