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

// Divergence guard (#191): the byte fast-path (`strip_control_chars` line ~26,
// returns `Cow::Borrowed` when nothing needs stripping) and the char loop (line
// ~44, decides each char's fate) must agree on which chars survive. If they
// drift, a control char's fate becomes context-dependent -- kept when the
// fast-path borrows, stripped when some other control byte forces the slow path.
// Both now share the predicate `!c.is_control() || c == '\n'`, so this test pins
// that every is_control char except '\n' is stripped in isolation (fast-path
// eligible) exactly as it is alongside a forcing byte (slow path).
#[test]
fn test_fast_path_and_char_loop_agree_on_control_chars() {
    // 0x00..=0x9F spans C0 controls, printable ASCII, DEL (0x7F), and C1
    // controls (0x80..=0x9F, e.g. 0x9B is a single-char CSI introducer).
    for cp in 0u32..=0x9F {
        let Some(c) = char::from_u32(cp) else {
            continue;
        };
        let clean = format!("a{c}b"); // fast-path eligible
        let forcing = format!("a{c}b\u{1}"); // 0x01 forces the slow path
        let isolated = strip_control_chars(&clean);
        let forced = strip_control_chars(&forcing);
        assert_eq!(isolated, forced, "paths diverge on {cp:#04x}");
        if c.is_control() && c != '\n' {
            assert_eq!(isolated, "ab", "control char {cp:#04x} must be stripped");
        } else if c == '\n' {
            assert_eq!(isolated, "a\nb", "newline must be preserved");
        }
    }
}
