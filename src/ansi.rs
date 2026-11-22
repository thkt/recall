use std::borrow::Cow;
use std::iter::Peekable;
use std::str::Chars;

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
