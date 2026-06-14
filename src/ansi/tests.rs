use super::*;

#[test]
fn test_strip_control_chars() {
    for (input, expected) in [
        ("normal text", "normal text"),
        ("line1\nline2", "line1\nline2"),
        ("hello \x1b[31mred\x1b[0m world", "hello red world"),
        ("before\x1b]0;title\x07after", "beforeafter"),
        ("before\x1b]0;title\x1b\\after", "beforeafter"),
        ("before\x1b]0;title", "before"), // unterminated OSC
        ("before\x1b[31", "before"),      // unterminated CSI
    ] {
        assert_eq!(strip_control_chars(input), expected, "input: {input:?}");
    }
}
