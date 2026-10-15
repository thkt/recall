//! Small CLI primitives owned by recall; no shared application runtime.
#[cfg(test)]
use std::process::ExitCode;
use std::{ffi::OsString, io};
use tracing_subscriber::EnvFilter;

pub mod codes {
    pub const USAGE: u8 = 64;
    pub const DATA_ERROR: u8 = 65;
    pub const INTERNAL: u8 = 70;
    pub const IO_ERR: u8 = 74;
    pub const TEMP_FAIL: u8 = 75;
    pub const UNKNOWN: u8 = 104;
}
#[cfg(test)]
pub trait CliError {
    fn exit_code(&self) -> ExitCode;
}
pub fn done(message: &str) {
    eprintln!("{message}");
}
pub fn info(message: &str) {
    eprintln!("{message}");
}
pub fn warning(message: &str) {
    eprintln!("warning: {message}");
}
pub fn exit_error(message: &str) {
    eprintln!("error: {message}");
}
pub fn init_subscriber(filter: &str) {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| filter.into()))
        .with_writer(io::stderr)
        .try_init();
}
pub fn notify_schema_change(app: &str, what: &str, count: usize, remedy: &str) {
    warning(&format!(
        "{app}: {what} schema changed ({count} cached entries); run `{remedy}`"
    ));
}

/// Preserve global options (including their separate values) before shorthand.
pub fn try_expand_shorthand(
    args: &[OsString],
    commands: &[&str],
    flags: &[&str],
) -> Option<Vec<OsString>> {
    let mut i = 1;
    while i < args.len() {
        let s = args[i].to_str()?;
        if matches!(s, "--db-path" | "--model-manifest") {
            i += 2;
            continue;
        }
        if s.starts_with("--db-path=") || s.starts_with("--model-manifest=") || flags.contains(&s) {
            i += 1;
            continue;
        }
        if s == "--help" || s == "-h" || commands.contains(&s) {
            return None;
        }
        if s.starts_with('-') {
            return None;
        }
        let mut result = args.to_vec();
        result.insert(i, "search".into());
        return Some(result);
    }
    None
}
