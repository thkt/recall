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
                Some(&'[') => { chars.next(); skip_csi(&mut chars); }
                Some(&']') => { chars.next(); skip_osc(&mut chars); }
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
    fn test_strip_control_chars_ansi() {
        assert_eq!(
            strip_control_chars("hello \x1b[31mred\x1b[0m world"),
            "hello red world"
        );
    }

    #[test]
    fn test_strip_control_chars_clean() {
        assert_eq!(strip_control_chars("normal text"), "normal text");
    }

    #[test]
    fn test_strip_control_chars_preserves_newlines() {
        assert_eq!(strip_control_chars("line1\nline2"), "line1\nline2");
    }

    #[test]
    fn test_strip_control_chars_osc() {
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
    fn test_strip_control_chars_osc_unterminated() {
        assert_eq!(strip_control_chars("before\x1b]0;title"), "before");
    }

    #[test]
    fn test_strip_control_chars_csi_unterminated() {
        // CSI sequence with no terminating alphabetic char — consumes rest of string
        assert_eq!(strip_control_chars("before\x1b[31"), "before");
    }
}
