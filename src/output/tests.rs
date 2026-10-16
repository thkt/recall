use std::io::{self, ErrorKind};

use super::{WriteOutcome, classify_write, write_output, write_result};

#[test]
fn write_output_appends_newline_when_missing() {
    let mut buf = Vec::new();
    write_output(&mut buf, "hello").unwrap();
    assert_eq!(&buf, b"hello\n");
}

#[test]
fn write_output_preserves_existing_newline() {
    let mut buf = Vec::new();
    write_output(&mut buf, "hello\n").unwrap();
    assert_eq!(&buf, b"hello\n");
}

#[test]
fn write_output_preserves_trailing_blank_line() {
    let mut buf = Vec::new();
    write_output(&mut buf, "hello\n\n").unwrap();
    assert_eq!(&buf, b"hello\n\n");
}

#[test]
fn classify_write_maps_ok_to_written() {
    assert_eq!(classify_write(Ok(())).unwrap(), WriteOutcome::Written);
}

#[test]
fn classify_write_maps_broken_pipe_to_pipe_closed() {
    let result = Err(io::Error::from(ErrorKind::BrokenPipe));
    assert_eq!(classify_write(result).unwrap(), WriteOutcome::PipeClosed);
}

#[test]
fn classify_write_propagates_other_error() {
    let result = Err(io::Error::other("disk full"));
    assert_eq!(classify_write(result).unwrap_err().kind(), ErrorKind::Other);
}

#[test]
fn write_result_writes_to_sink_and_reports_written() {
    let mut buf = Vec::new();
    assert_eq!(write_result(&mut buf, "hi").unwrap(), WriteOutcome::Written);
    assert_eq!(&buf, b"hi\n");
}
