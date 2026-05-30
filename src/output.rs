//! Stdout result writing with a single SIGPIPE policy (ADR-0060, #67 Phase 1).
//!
//! `recall` renders a command's result into a string, then writes it through
//! [`write_result`]. A consumer that closes the pipe early (`recall search foo
//! | head`) surfaces as [`WriteOutcome::PipeClosed`] instead of the panic a
//! bare `println!` raises, so the caller stops cleanly with exit 0. The
//! search/status result paths share this one policy; `recall show` keeps its
//! own (to be unified in Phase 2).

use std::io::{self, ErrorKind, Write};

/// Write `output` to `w`, appending a trailing newline only when `output` does
/// not already end with one. Newlines already present (such as a renderer's
/// trailing blank line) are preserved, not collapsed.
pub fn write_output<W: Write>(w: &mut W, output: &str) -> io::Result<()> {
    w.write_all(output.as_bytes())?;
    if !output.ends_with('\n') {
        w.write_all(b"\n")?;
    }
    Ok(())
}

/// Whether a rendered result reached the consumer or the pipe closed early.
#[derive(Debug, PartialEq, Eq)]
pub enum WriteOutcome {
    Written,
    PipeClosed,
}

/// Write fully-rendered `output` to `w`, classifying the result via
/// [`classify_write`]: a closed pipe is reported, any other write failure
/// propagates.
pub fn write_result<W: Write>(w: &mut W, output: &str) -> io::Result<WriteOutcome> {
    classify_write(write_output(w, output))
}

/// Map a stdout write result to a [`WriteOutcome`]. A closed pipe
/// (`BrokenPipe`) becomes [`WriteOutcome::PipeClosed`] (a clean stop, not a
/// failure); success becomes [`WriteOutcome::Written`]; any other error
/// propagates to the caller. Kept separate from the I/O so the classification
/// is testable without a writer.
fn classify_write(result: io::Result<()>) -> io::Result<WriteOutcome> {
    match result {
        Ok(()) => Ok(WriteOutcome::Written),
        Err(e) if e.kind() == ErrorKind::BrokenPipe => Ok(WriteOutcome::PipeClosed),
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod tests {
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
}
