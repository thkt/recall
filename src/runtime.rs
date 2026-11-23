use std::{env, io, path::PathBuf};
use tracing_subscriber::EnvFilter;

pub fn home_dir() -> Option<PathBuf> {
    env::home_dir().filter(|path| !path.as_os_str().is_empty())
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
