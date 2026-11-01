use std::fmt;

pub(crate) mod classify;
pub(crate) mod doctor;
pub(crate) mod index;
pub(crate) mod search;
pub(crate) mod show;
pub(crate) mod status;

/// Builds a command's Markdown output. Writing to a `String` cannot fail, so
/// the builder's `fmt::Result` is only there to let it use `?`.
pub(crate) fn build_markdown(build: impl FnOnce(&mut String) -> fmt::Result) -> String {
    let mut buf = String::new();
    build(&mut buf).expect("writing to a String cannot fail");
    buf
}
