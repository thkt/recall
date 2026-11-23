use std::env;
use std::ffi::OsStr;
use std::fmt;
use std::fs::{OpenOptions, create_dir_all};
use std::io::{self, ErrorKind};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use rusqlite::Connection;

use crate::db;
use crate::embedding::Embed;
use crate::error::RecallError;
use crate::model::{self, DegradedReason, EmbedderDegraded, record_degraded};
use crate::runtime::{home_dir, info as cli_info};

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

fn create_db_file(path: &Path) -> io::Result<()> {
    create_dir_all(path.parent().unwrap_or(Path::new("")))?;
    let mut opts = OpenOptions::new();
    opts.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    opts.open(path)?;
    Ok(())
}

pub(crate) fn open_or_create_db(path: &Path) -> Result<Connection> {
    match create_db_file(path) {
        Ok(_) => {}
        Err(e) if e.kind() == ErrorKind::AlreadyExists => {}
        Err(e) => return Err(e).context("Failed to create database file"),
    }
    {
        let (existing, _) = db::open_db_readonly(path)?;
        model::check_index(&existing, model::selected())
            .map_err(|e| RecallError::DatabaseRead(e.to_string()))?;
    }
    let mut conn = db::open_db(path).map_err(|error| {
        if error.is::<RecallError>() {
            error
        } else {
            error.context("Failed to open database")
        }
    })?;
    model::bind_index(&mut conn, model::selected())?;
    Ok(conn)
}

pub(crate) fn resolve_db_path(db_path: &Option<PathBuf>) -> Result<PathBuf> {
    match db_path {
        Some(p) => Ok(p.clone()),
        None => resolve_default_db_path(
            env::var_os("XDG_DATA_HOME").as_deref(),
            home_dir().as_deref(),
        ),
    }
}

fn resolve_default_db_path(xdg_data_home: Option<&OsStr>, home: Option<&Path>) -> Result<PathBuf> {
    let data_home = match xdg_data_home.map(Path::new) {
        Some(p) if p.is_absolute() => p.to_path_buf(),
        _ => home
            .context("Could not determine home directory")?
            .join(".local")
            .join("share"),
    };
    Ok(data_home
        .join("recall")
        .join("indexes")
        .join(model::selected().identity())
        .join("recall.db"))
}

pub(crate) fn try_load_embedder_cached_with<F>(
    open_embedder: F,
) -> Result<Arc<dyn Embed>, DegradedReason>
where
    F: FnOnce() -> Result<Arc<dyn Embed>, DegradedReason>,
{
    try_load_embedder_cached_with_reporter(open_embedder, cli_info)
}

pub(crate) fn try_load_embedder_cached_with_reporter<F, I>(
    open_embedder: F,
    mut report_info: I,
) -> Result<Arc<dyn Embed>, DegradedReason>
where
    F: FnOnce() -> Result<Arc<dyn Embed>, DegradedReason>,
    I: FnMut(&str),
{
    open_embedder().inspect_err(|&reason| {
        if let Some(note) = search_degraded_note(reason) {
            report_info(&note);
        }
        if reason != DegradedReason::NotInstalled {
            record_degraded(reason);
        }
    })
}

pub(crate) fn search_degraded_note(reason: DegradedReason) -> Option<String> {
    EmbedderDegraded(reason)
        .user_note()
        .map(|note| format!("note: {note}."))
}

#[cfg(test)]
mod tests;
