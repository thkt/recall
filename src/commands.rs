use std::fmt;

pub(crate) mod classify;
pub(crate) mod doctor;
pub(crate) mod index;
mod read_index;
pub(crate) mod search;
pub(crate) mod show;
pub(crate) mod status;

pub(crate) use read_index::ReadIndex;

pub(crate) fn build_markdown(build: impl FnOnce(&mut String) -> fmt::Result) -> String {
    let mut buf = String::new();
    build(&mut buf).expect("writing to a String cannot fail");
    buf
}
