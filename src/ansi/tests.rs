use super::*;

#[test]
fn test_strip_control_chars() {
    for (input, expected) in [
        ("normal text", "normal text"),
        ("line1\nline2", "line1\nline2"),
        ("hello \x1b[31mred\x1b[0m world", "hello red world"),
        ("before\x1b]0;title\x07after", "beforeafter"),
        ("before\x1b]0;title\x1b\\after", "beforeafter"),
        ("before\x1b]0;title", "before"),
        ("before\x1b[31", "before"),
        ("foo\tbar", "foo bar"),
        ("a\t\tb", "a  b"),
        ("foo\rbar", "foobar"),
    ] {
        assert_eq!(strip_control_chars(input), expected, "input: {input:?}");
    }
}

#[test]
fn test_fast_path_and_char_loop_agree_on_control_chars() {
    for cp in 0u32..=0x9F {
        let Some(c) = char::from_u32(cp) else {
            continue;
        };
        let clean = format!("a{c}b");
        let forcing = format!("a{c}b\u{1}");
        let isolated = strip_control_chars(&clean);
        let forced = strip_control_chars(&forcing);
        assert_eq!(isolated, forced, "paths diverge on {cp:#04x}");
        if c == '\t' {
            assert_eq!(isolated, "a b", "tab must normalize to a space");
        } else if c.is_control() && c != '\n' {
            assert_eq!(isolated, "ab", "control char {cp:#04x} must be stripped");
        } else if c == '\n' {
            assert_eq!(isolated, "a\nb", "newline must be preserved");
        }
    }
}
