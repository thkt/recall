use std::collections::HashSet;
use std::ffi::OsString;
use std::os::unix::ffi::{OsStrExt, OsStringExt};
use std::path::{Path, PathBuf};
use std::result::Result as StdResult;

use anyhow::Result;
use rusqlite::types::Value;
use rusqlite::{Connection, Transaction};

use crate::ansi;
use crate::parser::LineDiagnostics;
use crate::parser::Source;

/// The last unresolved observation of a file, including files without sessions.
#[derive(Debug)]
pub struct FileParseDiagnostics {
    pub file_path: PathBuf,
    pub source: Source,
    pub lines: LineDiagnostics,
    pub read_error: bool,
}

impl FileParseDiagnostics {
    pub fn display_path(&self) -> String {
        ansi::strip_control_chars(&self.file_path.to_string_lossy())
            .chars()
            .filter(|c| !c.is_control())
            .take(240)
            .collect()
    }

    pub fn notes(&self) -> Vec<String> {
        let path = self.display_path();
        let mut notes = Vec::new();
        if self.lines.invalid_json_lines + self.lines.invalid_utf8_lines > 0 {
            notes.push(format!(
                "{path}: skipped {} invalid JSON line(s) and {} invalid UTF-8 line(s); repair the source file, then rerun `recall index` (use `recall rebuild` if size and mtime are unchanged)",
                self.lines.invalid_json_lines, self.lines.invalid_utf8_lines,
            ));
        }
        if self.lines.incomplete_tail_lines > 0 {
            notes.push(format!(
                "{path}: {} incomplete final line(s), possibly still being written; wait for the next append, then rerun `recall index`; if the writer has stopped, repair the source file",
                self.lines.incomplete_tail_lines,
            ));
        }
        if self.read_error {
            notes.push(format!(
                "{path}: 1 file could not be read; check file access and UTF-8 path encoding, then rerun `recall index`; existing indexed data was retained",
            ));
        }
        notes
    }
}

// Keep UTF-8 keys compatible with existing rows and sessions.file_path. SQLite
// stores non-UTF-8 paths as BLOBs without applying the column's TEXT affinity.
// Lossy conversion belongs only at the display boundary, never in identity.
fn path_key(path: &Path) -> Value {
    match path.to_str() {
        Some(path) => Value::Text(path.to_owned()),
        None => Value::Blob(path.as_os_str().as_bytes().to_vec()),
    }
}

pub(super) fn save_parse_diagnostics(
    tx: &Transaction,
    path: &str,
    source: Source,
    lines: LineDiagnostics,
) -> Result<()> {
    if lines.is_empty() {
        tx.execute("DELETE FROM parse_diagnostics WHERE file_path = ?", [path])?;
    } else {
        tx.execute(
            "INSERT OR REPLACE INTO parse_diagnostics
             (file_path, source, invalid_json_lines, invalid_utf8_lines, incomplete_tail_lines)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            rusqlite::params![
                path,
                source.as_str(),
                lines.invalid_json_lines,
                lines.invalid_utf8_lines,
                lines.incomplete_tail_lines
            ],
        )?;
    }
    Ok(())
}

pub(super) fn save_read_error(tx: &Transaction, path: &Path, source: Source) -> Result<()> {
    // Keep previously observed line loss until a successful read replaces it.
    tx.execute(
        "INSERT INTO parse_diagnostics (file_path, source, read_error) VALUES (?1, ?2, 1)
         ON CONFLICT(file_path) DO UPDATE SET read_error = 1",
        rusqlite::params![path_key(path), source.as_str()],
    )?;
    Ok(())
}

pub(crate) fn load_parse_diagnostics(conn: &Connection) -> Result<Vec<FileParseDiagnostics>> {
    let mut stmt = conn.prepare(
        "SELECT CAST(file_path AS BLOB), source, invalid_json_lines, invalid_utf8_lines,
                incomplete_tail_lines, read_error FROM parse_diagnostics ORDER BY file_path",
    )?;
    let rows = stmt.query_map([], |row| {
        let source: String = row.get(1)?;
        let file_path = PathBuf::from(OsString::from_vec(row.get(0)?));
        let lines = LineDiagnostics {
            invalid_json_lines: row.get(2)?,
            invalid_utf8_lines: row.get(3)?,
            incomplete_tail_lines: row.get(4)?,
        };
        let read_error = row.get(5)?;
        Ok(Source::from_db(&source).map(|source| FileParseDiagnostics {
            file_path,
            source,
            lines,
            read_error,
        }))
    })?;
    rows.filter_map(StdResult::transpose)
        .collect::<StdResult<_, _>>()
        .map_err(Into::into)
}

pub(super) fn cleanup_parse_diagnostics(
    tx: &Transaction,
    source_paths: &HashSet<&Path>,
    scanned: &HashSet<Source>,
) -> Result<Vec<FileParseDiagnostics>> {
    let mut remaining = Vec::new();
    for diagnostic in load_parse_diagnostics(tx)? {
        if scanned.contains(&diagnostic.source)
            && !source_paths.contains(diagnostic.file_path.as_path())
        {
            // Orphan cleanup has already run. A session still pointing here
            // retains its old body (e.g. a deferred replacement at another path),
            // so its unresolved loss must survive along with that body.
            let removed = tx.execute(
                "DELETE FROM parse_diagnostics WHERE file_path = ?1
                 AND NOT EXISTS (SELECT 1 FROM sessions WHERE file_path = ?1)",
                [path_key(&diagnostic.file_path)],
            )?;
            if removed > 0 {
                continue;
            }
        }
        remaining.push(diagnostic);
    }
    Ok(remaining)
}
