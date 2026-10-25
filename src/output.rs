use std::io::{self, ErrorKind, Write};

pub fn write_output<W: Write>(w: &mut W, output: &str) -> io::Result<()> {
    w.write_all(output.as_bytes())?;
    if !output.ends_with('\n') {
        w.write_all(b"\n")?;
    }
    Ok(())
}

#[derive(Debug, PartialEq, Eq)]
pub enum WriteOutcome {
    Written,
    PipeClosed,
}

pub fn write_result<W: Write>(w: &mut W, output: &str) -> io::Result<WriteOutcome> {
    match write_output(w, output) {
        Ok(()) => Ok(WriteOutcome::Written),
        Err(e) if e.kind() == ErrorKind::BrokenPipe => Ok(WriteOutcome::PipeClosed),
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod tests;
