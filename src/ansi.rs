use std::borrow::Cow;
use std::iter::Peekable;
use std::str::Chars;

/// A CSI sequence holds bytes 0x20–0x3F and ends at a final byte in 0x40–0x7E (ECMA-48 § 5.4).
/// Any other character breaks the sequence and is kept, so session text after it is not lost.
fn skip_csi(chars: &mut Peekable<Chars>) {
    while let Some(&next) = chars.peek() {
        if !(' '..='~').contains(&next) {
            break;
        }
        chars.next();
        if ('@'..='~').contains(&next) {
            break;
        }
    }
}

/// An OSC sequence ends at BEL or ESC \. One left unterminated ends at the end of its line, so it
/// drops the rest of that line and not the rest of the message.
fn skip_osc(chars: &mut Peekable<Chars>) {
    while let Some(&next) = chars.peek() {
        if next == '\n' {
            break;
        }
        chars.next();
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
    if s.chars().all(|c| !c.is_control() || c == '\n') {
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
        } else if c == '\t' {
            out.push(' ');
        } else if !c.is_control() || c == '\n' {
            out.push(c);
        }
    }
    Cow::Owned(out)
}

#[cfg(test)]
mod tests;
