use std::fmt::{self, Write};
use std::path::PathBuf;

use anyhow::Result;
use rusqlite::{Connection, OptionalExtension};
use tracing::warn;

use crate::ansi;
use crate::commands::build_markdown;
use crate::compat::filter::escape_like;
use crate::envelope::{CommandOutput, Note};
use crate::error::RecallError;
use crate::parser::Source;
use crate::{db, parser};
use crate::{format_timestamp, require_current_schema, resolve_db_path};

fn write_show_header(
    buf: &mut String,
    session: &parser::SessionData,
    verbose: bool,
    extra: &[String],
) -> fmt::Result {
    let slug = ansi::strip_control_chars(&session.slug);
    let session_id = ansi::strip_control_chars(&session.session_id);
    let project = ansi::strip_control_chars(&session.project);
    writeln!(buf, "# {slug}")?;
    writeln!(buf, "- session_id: {session_id}")?;
    writeln!(buf, "- date: {}", format_timestamp(session.timestamp))?;
    writeln!(buf, "- project: {project}")?;
    writeln!(buf, "- source: {}", session.source)?;
    for line in extra {
        writeln!(buf, "{line}")?;
    }
    if verbose {
        let file = ansi::strip_control_chars(&session.file_path);
        writeln!(buf, "- file: {file}")?;
    }
    writeln!(buf)
}

fn show_session(conn: &Connection, session_id: &str, verbose: bool) -> Result<CommandOutput> {
    let session = resolve_session(conn, session_id)?;

    let mut msg_stmt =
        conn.prepare("SELECT role, text FROM messages WHERE session_id = ?1 ORDER BY rowid")?;
    let rows = msg_stmt.query_map([&session.session_id], |row| {
        Ok((row.get(0)?, shown_text(&row.get::<_, String>(1)?)))
    })?;
    let messages: Vec<(String, String)> = rows.collect::<rusqlite::Result<Vec<_>>>()?;

    let markdown = build_markdown(|buf| {
        write_show_header(buf, &session, verbose, &[])?;
        for (role, text) in &messages {
            writeln!(buf, "## [{role}]")?;
            writeln!(buf, "{text}")?;
            writeln!(buf)?;
        }
        Ok(())
    });

    let data = serde_json::json!({
        "session_id": session.session_id,
        "slug": session.slug,
        "project": session.project,
        "source": session.source.to_string(),
        "timestamp": session.timestamp,
        "messages": messages
            .iter()
            .map(|(role, text)| serde_json::json!({ "role": role, "text": text }))
            .collect::<Vec<_>>(),
    });

    Ok(CommandOutput::ok(markdown, data))
}

pub(crate) const DEFAULT_WINDOW: i64 = 5;
const MAX_WINDOW: i64 = 50;
const MAX_MSG_CHARS: usize = 4000;

/// A stored message or chunk body as both outputs print it. It is stripped before `cap_message`
/// cuts it, since an escape sequence cut in half would swallow the `…` that marks the cut.
fn shown_text(stored: &str) -> String {
    ansi::strip_control_chars(stored).into_owned()
}

fn resolve_session(conn: &Connection, session_id: &str) -> Result<parser::SessionData> {
    let mut stmt = conn.prepare(
        "SELECT session_id, source, file_path, project, slug, timestamp \
         FROM sessions WHERE session_id LIKE ?1 ESCAPE '\\' ORDER BY session_id",
    )?;
    let pattern = format!("{}%", escape_like(session_id));
    let rows = stmt.query_map([&pattern], |row| {
        let source_str: String = row.get(1)?;
        Ok(parser::SessionData {
            session_id: row.get(0)?,
            source: Source::from_db(&source_str).unwrap_or_else(|| {
                warn!(source = %source_str, "unknown source, defaulting to claude");
                Source::Claude
            }),
            file_path: row.get(2)?,
            project: row.get(3)?,
            slug: row.get(4)?,
            timestamp: row.get(5)?,
        })
    })?;
    let matches: Vec<parser::SessionData> = rows.collect::<rusqlite::Result<Vec<_>>>()?;

    if matches.is_empty() {
        return Err(RecallError::Usage(format!("No session found matching '{session_id}'")).into());
    }
    if matches.len() > 1 {
        let mut msg = format!("Multiple sessions match '{session_id}':\n");
        for s in &matches {
            msg.push_str(&format!(
                "  {}  {}\n",
                ansi::strip_control_chars(&s.session_id),
                ansi::strip_control_chars(&s.slug)
            ));
        }
        msg.push_str("Narrow the session ID prefix to match exactly one session");
        return Err(RecallError::Usage(msg).into());
    }
    Ok(matches.into_iter().next().expect("exactly one match"))
}

fn cap_message(text: &str) -> String {
    if text.chars().count() <= MAX_MSG_CHARS {
        return text.to_owned();
    }
    let cut: String = text.chars().take(MAX_MSG_CHARS).collect();
    format!("{cut}…")
}

fn show_session_windowed(
    conn: &Connection,
    session_id: &str,
    verbose: bool,
    chunk: Option<i64>,
    window: i64,
) -> Result<CommandOutput> {
    let Some(chunk_id) = chunk else {
        return show_session(conn, session_id, verbose);
    };

    let session = resolve_session(conn, session_id)?;

    let anchor = conn
        .query_row(
            "SELECT session_id, content, src_rowid_lo, src_rowid_hi \
             FROM qa_chunks WHERE id = ?1",
            [chunk_id],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, Option<i64>>(2)?,
                    row.get::<_, Option<i64>>(3)?,
                ))
            },
        )
        .optional()?
        .ok_or_else(|| RecallError::Usage(format!("No chunk found with id {chunk_id}")))?;
    let (chunk_session, content, lo, hi) = anchor;

    if chunk_session != session.session_id {
        return Err(RecallError::Usage(format!(
            "chunk {chunk_id} belongs to session '{}', not '{}'",
            ansi::strip_control_chars(&chunk_session),
            ansi::strip_control_chars(&session.session_id)
        ))
        .into());
    }

    match (lo, hi) {
        (Some(lo), Some(hi)) => render_windowed(conn, &session, chunk_id, lo, hi, window, verbose),
        _ => Ok(render_chunk_fallback(&session, chunk_id, &content, verbose)),
    }
}

fn render_windowed(
    conn: &Connection,
    session: &parser::SessionData,
    chunk_id: i64,
    lo: i64,
    hi: i64,
    window: i64,
    verbose: bool,
) -> Result<CommandOutput> {
    let n = window.clamp(0, MAX_WINDOW) as usize;

    let mut stmt = conn
        .prepare("SELECT rowid, role, text FROM messages WHERE session_id = ?1 ORDER BY rowid")?;
    let rows = stmt.query_map([&session.session_id], |row| {
        Ok((
            row.get::<_, i64>(0)?,
            row.get::<_, String>(1)?,
            shown_text(&row.get::<_, String>(2)?),
        ))
    })?;
    let msgs: Vec<(i64, String, String)> = rows.collect::<rusqlite::Result<Vec<_>>>()?;

    let first_body = msgs.iter().position(|(rid, ..)| *rid >= lo);
    let last_body = msgs.iter().rposition(|(rid, ..)| *rid <= hi);
    let slice: &[(i64, String, String)] = match (first_body, last_body) {
        (Some(f), Some(l)) if f <= l => {
            let start = f.saturating_sub(n);
            let end = (l + n).min(msgs.len() - 1);
            &msgs[start..=end]
        }
        _ => &[],
    };

    let markdown = build_markdown(|buf| {
        write_show_header(
            buf,
            session,
            verbose,
            &[format!("- chunk: {chunk_id}"), format!("- window: {n}")],
        )?;
        for (rid, role, text) in slice {
            let marker = if *rid >= lo && *rid <= hi {
                " (chunk)"
            } else {
                ""
            };
            writeln!(buf, "## [{role}]{marker}")?;
            writeln!(buf, "{}", cap_message(text))?;
            writeln!(buf)?;
        }
        Ok(())
    });

    let data = serde_json::json!({
        "session_id": session.session_id,
        "slug": session.slug,
        "project": session.project,
        "source": session.source.to_string(),
        "timestamp": session.timestamp,
        "chunk": chunk_id,
        "window": n,
        "messages": slice
            .iter()
            .map(|(rid, role, text)| serde_json::json!({
                "rowid": rid,
                "role": role,
                "text": cap_message(text),
                "in_chunk": *rid >= lo && *rid <= hi,
            }))
            .collect::<Vec<_>>(),
    });

    Ok(CommandOutput::ok(markdown, data))
}

fn render_chunk_fallback(
    session: &parser::SessionData,
    chunk_id: i64,
    content: &str,
    verbose: bool,
) -> CommandOutput {
    let hint = format!(
        "This chunk predates message-range tracking; run `recall show {}` for the full conversation.",
        ansi::strip_control_chars(&session.session_id)
    );
    let body = cap_message(&shown_text(content));

    let markdown = build_markdown(|buf| {
        write_show_header(buf, session, verbose, &[format!("- chunk: {chunk_id}")])?;
        writeln!(buf, "{body}")?;
        writeln!(buf)?;
        writeln!(buf, "{hint}")
    });

    let data = serde_json::json!({
        "session_id": session.session_id,
        "slug": session.slug,
        "project": session.project,
        "source": session.source.to_string(),
        "timestamp": session.timestamp,
        "chunk": chunk_id,
        "content": body,
        "windowed": false,
        "next_step": hint,
    });

    CommandOutput::ok(markdown, data)
}

pub(crate) fn run_show(
    session_id: &str,
    verbose: bool,
    chunk: Option<i64>,
    window: i64,
    db_path: &Option<PathBuf>,
) -> Result<CommandOutput> {
    let path = resolve_db_path(db_path)?;
    if !path.exists() {
        return Err(
            RecallError::Usage("Database not found. Run `recall index` first.".to_owned()).into(),
        );
    }

    let (conn, tier) = db::open_db_readonly(&path)?;
    let wal_note = db::stale_wal_note(&path, tier);
    let shown = match db::schema_state(&conn)? {
        db::SchemaState::Empty => Err(RecallError::Usage(
            "Database not found. Run `recall index` first.".to_owned(),
        )
        .into()),
        state => match require_current_schema(state) {
            Ok(()) => show_session_windowed(&conn, session_id, verbose, chunk, window),
            Err(e) => Err(e.into()),
        },
    };
    match (shown, wal_note) {
        (Ok(mut out), Some(note)) => {
            out.degraded = true;
            out.notes.push(Note::new(note));
            Ok(out)
        }
        (Ok(out), None) => Ok(out),
        // The missing schema, session or chunk may sit in the log this snapshot cannot read.
        (Err(e), Some(note)) => Err(RecallError::DatabaseRead(format!("{e}; {note}")).into()),
        (Err(e), None) => Err(e),
    }
}

#[cfg(test)]
mod tests;
