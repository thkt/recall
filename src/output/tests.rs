use std::io::{self, ErrorKind, Write};

use super::{WriteOutcome, write_output, write_result};

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

struct FailingWriter(ErrorKind);

impl Write for FailingWriter {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> {
        Err(io::Error::from(self.0))
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[test]
fn write_result_maps_broken_pipe_to_pipe_closed() {
    assert_eq!(
        write_result(&mut FailingWriter(ErrorKind::BrokenPipe), "hi").unwrap(),
        WriteOutcome::PipeClosed
    );
}

#[test]
fn write_result_propagates_other_error() {
    assert_eq!(
        write_result(&mut FailingWriter(ErrorKind::Other), "hi")
            .unwrap_err()
            .kind(),
        ErrorKind::Other
    );
}

#[test]
fn write_result_writes_to_sink_and_reports_written() {
    let mut buf = Vec::new();
    assert_eq!(write_result(&mut buf, "hi").unwrap(), WriteOutcome::Written);
    assert_eq!(&buf, b"hi\n");
}
