use std::io::{self, ErrorKind};

use super::{WriteOutcome, classify_write, write_output, write_result};

// T-W001: write_output appends a newline when the input lacks one.
#[test]
fn write_output_appends_newline_when_missing() {
    let mut buf = Vec::new();
    write_output(&mut buf, "hello").unwrap();
    assert_eq!(&buf, b"hello\n");
}

// T-W002: write_output does not double an existing trailing newline.
#[test]
fn write_output_preserves_existing_newline() {
    let mut buf = Vec::new();
    write_output(&mut buf, "hello\n").unwrap();
    assert_eq!(&buf, b"hello\n");
}

// T-W003: write_output preserves an existing trailing blank line; it appends
// only when missing and never collapses extras (search results end in a
// blank line by design, so that line must survive being written).
#[test]
fn write_output_preserves_trailing_blank_line() {
    let mut buf = Vec::new();
    write_output(&mut buf, "hello\n\n").unwrap();
    assert_eq!(&buf, b"hello\n\n");
}

// T-W004: a successful write classifies as Written.
#[test]
fn classify_write_maps_ok_to_written() {
    assert_eq!(classify_write(Ok(())).unwrap(), WriteOutcome::Written);
}

// T-W005: a consumer that closed the pipe classifies as PipeClosed, not an error.
#[test]
fn classify_write_maps_broken_pipe_to_pipe_closed() {
    let result = Err(io::Error::from(ErrorKind::BrokenPipe));
    assert_eq!(classify_write(result).unwrap(), WriteOutcome::PipeClosed);
}

// T-W006: a non-BrokenPipe write failure propagates as an error.
#[test]
fn classify_write_propagates_other_error() {
    let result = Err(io::Error::other("disk full"));
    assert_eq!(classify_write(result).unwrap_err().kind(), ErrorKind::Other);
}

// T-W007: write_result writes the bytes (with newline) and reports Written,
// exercising the write_output + classify_write composition end to end.
#[test]
fn write_result_writes_to_sink_and_reports_written() {
    let mut buf = Vec::new();
    assert_eq!(write_result(&mut buf, "hi").unwrap(), WriteOutcome::Written);
    assert_eq!(&buf, b"hi\n");
}
