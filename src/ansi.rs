use std::borrow::Cow;
use std::iter::Peekable;
use std::str::Chars;

fn skip_csi(chars: &mut Peekable<Chars>) {
    for next in chars.by_ref() {
        if next.is_ascii_alphabetic() {
            break;
        }
    }
}

fn skip_osc(chars: &mut Peekable<Chars>) {
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
    // Fast-path keep predicate. MUST stay in sync with the char loop below: both
    // gate on `!c.is_control() || c == '\n'`. If they drift, a control char's
    // fate turns context-dependent -- kept when this borrows, stripped once some
    // other control char forces the loop. A byte check (`b >= 0x20`) cannot
    // express this: it treats DEL (0x7F) and C1 controls (0x80..=0x9F, e.g. the
    // 0x9B CSI introducer) as printable, leaking terminal-control bytes that the
    // loop strips. Iterating chars keeps the two predicates a single source of
    // truth. See tests::test_fast_path_and_char_loop_agree_on_control_chars.
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
        } else if !c.is_control() || c == '\n' {
            // Keep-in-sync anchor: this predicate is mirrored by the fast-path
            // above. ESC (0x1b) is handled by the branch above; every other
            // control char except '\n' is dropped here.
            out.push(c);
        }
    }
    Cow::Owned(out)
}

#[cfg(test)]
mod tests;
