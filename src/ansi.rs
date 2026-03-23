use std::borrow::Cow;

fn skip_csi(chars: &mut std::iter::Peekable<std::str::Chars>) {
    for next in chars.by_ref() {
        if next.is_ascii_alphabetic() {
            break;
        }
    }
}

fn skip_osc(chars: &mut std::iter::Peekable<std::str::Chars>) {
    while let Some(next) = chars.next() {
        if next == '\x07' {
            break;
        }
        if next == '\x1b' && chars.peek() == Some(&'\\') {
            chars.next();
            break;
        }
    }
}

pub fn strip_control_chars(s: &str) -> Cow<'_, str> {
    if s.bytes().all(|b| b >= 0x20 || b == b'\n') {
        return Cow::Borrowed(s);
    }
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            match chars.peek() {
                Some(&'[') => {
                    chars.next();
                    skip_csi(&mut chars);
                }
                Some(&']') => {
                    chars.next();
                    skip_osc(&mut chars);
                }
                _ => {}
            }
        } else if !c.is_control() || c == '\n' {
            out.push(c);
        }
    }
    Cow::Owned(out)
}

#[cfg(test)]
mod tests {
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
}
