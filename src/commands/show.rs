use std::io::Write;
use std::path::PathBuf;

use anyhow::Result;
use rusqlite::{Connection, OptionalExtension};
use tracing::warn;

use crate::compat::filter::escape_like;
use crate::envelope::CommandOutput;
use crate::error::RecallError;
use crate::parser::Source;
use crate::{db, parser};
use crate::{format_timestamp, require_current_schema, resolve_db_path};

fn write_show_header(
    buf: &mut Vec<u8>,
    session: &parser::SessionData,
    verbose: bool,
    extra: &[String],
) {
    let _ = writeln!(buf, "# {}", session.slug);
    let _ = writeln!(buf, "- session_id: {}", session.session_id);
    let _ = writeln!(buf, "- date: {}", format_timestamp(session.timestamp));
    let _ = writeln!(buf, "- project: {}", session.project);
    let _ = writeln!(buf, "- source: {}", session.source);
    for line in extra {
        let _ = writeln!(buf, "{line}");
    }
    if verbose {
        let _ = writeln!(buf, "- file: {}", session.file_path);
    }
    let _ = writeln!(buf);
}

fn show_session(conn: &Connection, session_id: &str, verbose: bool) -> Result<CommandOutput> {
    let session = resolve_session(conn, session_id)?;

    let mut msg_stmt =
        conn.prepare("SELECT role, text FROM messages WHERE session_id = ?1 ORDER BY rowid")?;
    let rows = msg_stmt.query_map([&session.session_id], |row| Ok((row.get(0)?, row.get(1)?)))?;
    let messages: Vec<(String, String)> = rows.collect::<rusqlite::Result<Vec<_>>>()?;

    let mut buf = Vec::new();
    write_show_header(&mut buf, &session, verbose, &[]);
    for (role, text) in &messages {
        let _ = writeln!(buf, "## [{role}]");
        let _ = writeln!(buf, "{text}");
        let _ = writeln!(buf);
    }
    let markdown = String::from_utf8_lossy(&buf).into_owned();

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
            msg.push_str(&format!("  {}  {}\n", s.session_id, s.slug));
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
            "chunk {chunk_id} belongs to session '{chunk_session}', not '{}'",
            session.session_id
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
            row.get::<_, String>(2)?,
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

    let mut buf = Vec::new();
    write_show_header(
        &mut buf,
        session,
        verbose,
        &[format!("- chunk: {chunk_id}"), format!("- window: {n}")],
    );
    for (rid, role, text) in slice {
        let marker = if *rid >= lo && *rid <= hi {
            " (chunk)"
        } else {
            ""
        };
        let _ = writeln!(buf, "## [{role}]{marker}");
        let _ = writeln!(buf, "{}", cap_message(text));
        let _ = writeln!(buf);
    }
    let markdown = String::from_utf8_lossy(&buf).into_owned();

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
        session.session_id
    );

    let mut buf = Vec::new();
    write_show_header(
        &mut buf,
        session,
        verbose,
        &[format!("- chunk: {chunk_id}")],
    );
    let _ = writeln!(buf, "{}", cap_message(content));
    let _ = writeln!(buf);
    let _ = writeln!(buf, "{hint}");
    let markdown = String::from_utf8_lossy(&buf).into_owned();

    let data = serde_json::json!({
        "session_id": session.session_id,
        "slug": session.slug,
        "project": session.project,
        "source": session.source.to_string(),
        "timestamp": session.timestamp,
        "chunk": chunk_id,
        "content": cap_message(content),
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

    let (conn, _tier) = db::open_db_readonly(&path)?;
    match db::schema_state(&conn)? {
        db::SchemaState::Empty => Err(RecallError::Usage(
            "Database not found. Run `recall index` first.".to_owned(),
        )
        .into()),
        state => {
            require_current_schema(state)?;
            show_session_windowed(&conn, session_id, verbose, chunk, window)
        }
    }
}

#[cfg(test)]
mod tests;
