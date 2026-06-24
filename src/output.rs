//! Stdout result writing with a single SIGPIPE policy (scout ADR-0060, external; #67 Phase 1).
//!
//! `recall` renders a command's result into a string, then writes it through
//! [`write_result`]. A consumer that closes the pipe early (`recall search foo
//! | head`) surfaces as [`WriteOutcome::PipeClosed`] instead of the panic a
//! bare `println!` raises, so the caller stops cleanly with exit 0. The
//! search/status/show result paths all share this one policy, routed through
//! this `write_result` boundary (show unified in #67 Phase 2).

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
mod tests;
