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
    // Fast-path "no rewrite needed" predicate. MUST stay in sync with the char
    // loop below: both gate on `!c.is_control() || c == '\n'`. A char failing it
    // is rewritten by the loop -- most control chars are dropped, but '\t' maps
    // to a space (#239). The predicate already excludes '\t' (it is a control
    // char), so any string containing a tab takes the slow path and gets
    // normalized; the fast-path needs no tab-specific clause. If the two drift, a
    // control char's fate turns context-dependent -- kept when this borrows,
    // rewritten once some other control char forces the loop. A byte check
    // (`b >= 0x20`) cannot express this: it treats DEL (0x7F) and C1 controls
    // (0x80..=0x9F, e.g. the 0x9B CSI introducer) as printable, leaking
    // terminal-control bytes that the loop strips. Iterating chars keeps the two
    // predicates a single source of truth. See
    // tests::test_fast_path_and_char_loop_agree_on_control_chars.
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
            // #239: an inter-word tab fuses neighboring tokens (`foo\tbar` ->
            // `foobar`), erasing the boundary FTS5 trigram and embedding rely on.
            // Normalize to a space so space-delimited queries still match.
            out.push(' ');
        } else if !c.is_control() || c == '\n' {
            // Keep-in-sync anchor: this predicate is mirrored by the fast-path
            // above. ESC (0x1b) and '\t' are handled by the branches above; every
            // other control char except '\n' (e.g. '\r', CRLF -> LF) is dropped
            // here.
            out.push(c);
        }
    }
    Cow::Owned(out)
}

#[cfg(test)]
mod tests;
