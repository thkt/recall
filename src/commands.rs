use std::fmt;

pub(crate) mod classify;
pub(crate) mod doctor;
pub(crate) mod index;
pub(crate) mod search;
pub(crate) mod show;
pub(crate) mod status;

/// `String` never fails a write, and the values formatted here only propagate
/// the writer's error, so an `Err` would be a bug.
pub(crate) fn build_markdown(build: impl FnOnce(&mut String) -> fmt::Result) -> String {
    let mut buf = String::new();
    build(&mut buf).expect("writing to a String cannot fail");
    buf
}
