use std::env;
use std::fmt::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use clap::Args;
use tracing::warn;

use crate::commands::build_markdown;
use crate::embedding::Embed;
use crate::envelope::CommandOutput;
use crate::error::RecallError;
use crate::model::{self, DegradedReason, EmbedderDegraded};
use crate::parser::Source;
use crate::runtime::done;
use crate::{ansi, db, search};
use crate::{
    format_timestamp, require_current_schema, resolve_db_path, try_load_embedder_cached_with,
};

#[derive(Debug, Args)]
pub(crate) struct SearchArgs {
    /// Search query. Optional when `--file` is given: a query-less
    /// `--file <path>` lists the sessions that touched the path, newest
    /// first. A call with neither a query nor `--file` is a usage error —
    /// rejected declaratively here, with `file_list_outcome` keeping the
    /// same guard for direct (non-CLI) callers.
    #[arg(required_unless_present = "file")]
    pub(crate) query: Option<String>,

    /// Filter by project path (prefix match)
    #[arg(long)]
    pub(crate) project: Option<String>,

    /// Only sessions from last N days
    #[arg(long, value_parser = clap::value_parser!(i64).range(1..))]
    pub(crate) days: Option<i64>,

    /// Filter by source
    #[arg(long, value_enum)]
    pub(crate) source: Option<Source>,

    /// Restrict to sessions that touched this file (exact path, or a
    /// basename / tail sub-path suffix). While Codex file extraction is
    /// unsupported, Codex sessions record no files and are never matched.
    #[arg(long)]
    pub(crate) file: Option<String>,

    /// Max results (1..=100)
    #[arg(long, default_value_t = 10, value_parser = clap::value_parser!(u16).range(1..=100))]
    pub(crate) limit: u16,

    /// Exclude the invoking Claude Code session from results. Default when
    /// CLAUDE_CODE_SESSION_ID is set (recall run from inside a session).
    #[arg(long, conflicts_with_all = ["include_current", "only_current"])]
    pub(crate) exclude_current: bool,

    /// Include the invoking session even when CLAUDE_CODE_SESSION_ID is set.
    #[arg(long, conflicts_with_all = ["exclude_current", "only_current"])]
    pub(crate) include_current: bool,

    /// Return only the invoking session (requires CLAUDE_CODE_SESSION_ID).
    #[arg(long, conflicts_with_all = ["exclude_current", "include_current"])]
    pub(crate) only_current: bool,

    /// Include sessions classified automated (hook/script/agent-generated).
    /// Default excludes them.
    #[arg(long)]
    pub(crate) include_automated: bool,
}

fn search_fallback_notes(reason: DegradedReason) -> Vec<String> {
    EmbedderDegraded(reason)
        .user_note("recall model download")
        .map(|note| vec![format!("semantic search unavailable: {note}")])
        .unwrap_or_default()
}

fn search_degraded_state(load: &Result<Arc<dyn Embed>, DegradedReason>) -> (bool, Vec<String>) {
    match load {
        Ok(_) => (false, Vec::new()),
        Err(reason) => {
            let notes = search_fallback_notes(*reason);
            (!notes.is_empty(), notes)
        }
    }
}

fn search_idle_output() -> CommandOutput {
    done("No sessions indexed. Run `recall index` first.");
    CommandOutput::ok(String::new(), serde_json::json!({ "results": [] }))
}

#[derive(Clone, Copy, Debug)]
struct CurrentSessionFlags {
    exclude: bool,
    include: bool,
    only: bool,
}

fn resolve_current_session(
    flags: CurrentSessionFlags,
    env_session_id: Option<String>,
) -> Result<search::CurrentSession, RecallError> {
    use search::CurrentSession;
    let env_session_id = env_session_id.filter(|s| !s.trim().is_empty());
    if flags.only {
        return match env_session_id {
            Some(id) => Ok(CurrentSession::Only(id)),
            None => Err(RecallError::Usage(
                "--only-current requires CLAUDE_CODE_SESSION_ID; run recall from inside a Claude Code session"
                    .to_owned(),
            )),
        };
    }
    if flags.include {
        return Ok(CurrentSession::Ignore);
    }
    match env_session_id {
        Some(id) => Ok(CurrentSession::Exclude(id)),
        None => {
            if flags.exclude {
                warn!("--exclude-current has no effect: CLAUDE_CODE_SESSION_ID is not set");
            }
            Ok(CurrentSession::Ignore)
        }
    }
}

pub(crate) fn run_search(args: SearchArgs, db_path: &Option<PathBuf>) -> Result<CommandOutput> {
    run_search_in(args, db_path, &model::HOST)
}

/// Picks the probe-skipping loader; the model host is the only injected input.
pub(crate) fn run_search_in(
    args: SearchArgs,
    db_path: &Option<PathBuf>,
    host: &model::ModelHost<'_>,
) -> Result<CommandOutput> {
    run_search_with(args, db_path, || {
        try_load_embedder_cached_with(|| host.load_for_search())
    })
}

pub(crate) fn run_search_with<F>(
    args: SearchArgs,
    db_path: &Option<PathBuf>,
    load_embedder: F,
) -> Result<CommandOutput>
where
    F: FnOnce() -> Result<Arc<dyn Embed>, DegradedReason>,
{
    let SearchArgs {
        query,
        project,
        days,
        source,
        file,
        limit,
        exclude_current,
        include_current,
        only_current,
        include_automated,
    } = args;

    let current = resolve_current_session(
        CurrentSessionFlags {
            exclude: exclude_current,
            include: include_current,
            only: only_current,
        },
        env::var("CLAUDE_CODE_SESSION_ID").ok(),
    )?;

    let path = resolve_db_path(db_path)?;

    if !path.exists() {
        return Ok(search_idle_output());
    }

    let (conn, tier) = db::open_db_readonly(&path)?;
    match db::schema_state(&conn)? {
        db::SchemaState::Empty => return Ok(search_idle_output()),
        state => require_current_schema(state)?,
    }
    let wal_note = db::stale_wal_note(&path, tier);

    let session_count: i64 = conn.query_row("SELECT COUNT(*) FROM sessions", [], |r| r.get(0))?;
    if session_count == 0 {
        return Ok(search_idle_output());
    }

    let identity_error = if query.is_some() {
        model::check_index(&conn, model::selected()).err()
    } else {
        None
    };
    let (mut degraded, mut notes, embedder) = if let Some(error) = identity_error {
        (
            true,
            vec![format!(
                "semantic search unavailable: {error}; results are text-only"
            )],
            None,
        )
    } else if query.is_some() {
        let embedder = load_embedder();
        let (degraded, notes) = search_degraded_state(&embedder);
        (degraded, notes, embedder.ok())
    } else {
        (false, Vec::new(), None)
    };

    let options = search::SearchOptions {
        project,
        days,
        source,
        file,
        limit: limit.into(),
        current,
        include_automated,
        now_ms: None,
    };
    let outcome =
        search::search_with_embedder(&conn, query.as_deref(), &options, embedder.as_deref())?;
    let results = outcome.results;

    if outcome.vec_degraded {
        degraded = true;
        notes.push(
            "semantic search unavailable: vector query failed at runtime; results are text-only"
                .to_owned(),
        );
    }

    if outcome.results_incomplete {
        degraded = true;
        notes.push(
            "results may be incomplete: some matched sessions that could have ranked in these results were excluded because their index rows were unreadable or carry an unrecognized source; for unreadable rows, run `recall index` to rebuild"
                .to_owned(),
        );
    }

    let automated =
        search::count_excluded_automated(&conn, query.as_deref(), &options).unwrap_or(0);
    if automated > 0 {
        notes.push(format!(
            "{automated} automated session(s) matching this search were excluded by default; use --include-automated to include them"
        ));
    }

    let markdown = render_results(&results);
    let data = serde_json::json!({
        "results": results.iter().map(result_to_json).collect::<Vec<_>>(),
    });

    if let Some(note) = wal_note {
        degraded = true;
        notes.push(note);
    }

    Ok(CommandOutput::with_notes(markdown, data, degraded, notes))
}

fn extract_parent_session(file_path: &str) -> Option<&str> {
    let idx = file_path.find("/subagents/")?;
    let prefix = &file_path[..idx];
    let parent = prefix.rsplit('/').next()?;
    if parent.is_empty() {
        None
    } else {
        Some(parent)
    }
}

fn format_result(w: &mut impl Write, i: usize, r: &search::SearchResult) -> fmt::Result {
    let s = &r.session;
    let date = format_timestamp(s.timestamp);
    let proj_name = if s.project.is_empty() {
        "unknown"
    } else {
        Path::new(&s.project)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
    };
    let slug = ansi::strip_control_chars(&s.slug);
    let project = ansi::strip_control_chars(&s.project);
    writeln!(
        w,
        "[{}] {} | {} | {} [{}]",
        i + 1,
        date,
        slug,
        proj_name,
        s.source
    )?;
    if !project.is_empty() {
        writeln!(w, "    {project}")?;
    }
    let session_id = ansi::strip_control_chars(&s.session_id);
    writeln!(w, "    ID: {session_id}")?;
    let file_path = ansi::strip_control_chars(&s.file_path);
    if let Some(parent) = extract_parent_session(&file_path) {
        writeln!(w, "    Parent: {parent}")?;
    }
    if !r.excerpt.is_empty() {
        let excerpt = ansi::strip_control_chars(&r.excerpt);
        for line in excerpt.trim().lines() {
            writeln!(w, "    > {line}")?;
        }
    }
    writeln!(w, "    inspect: {}", inspect_hint(&session_id, r.chunk_id))?;
    writeln!(w)?;
    Ok(())
}

fn render_results(results: &[search::SearchResult]) -> String {
    if results.is_empty() {
        return "No matching sessions found.".to_owned();
    }

    build_markdown(|buf| {
        writeln!(buf, "Found {} sessions:\n", results.len())?;
        for (i, r) in results.iter().enumerate() {
            format_result(buf, i, r)?;
        }
        Ok(())
    })
}

fn inspect_hint(session_id: &str, chunk_id: Option<i64>) -> String {
    match chunk_id {
        Some(id) => format!("recall show {session_id} --chunk {id} --window 3"),
        None => format!("recall show {session_id}"),
    }
}

fn result_to_json(r: &search::SearchResult) -> serde_json::Value {
    let s = &r.session;
    serde_json::json!({
        "session_id": s.session_id,
        "project": s.project,
        "slug": s.slug,
        "source": s.source.to_string(),
        "timestamp": s.timestamp,
        "excerpt": r.excerpt,
        "chunk": r.chunk_id,
        "inspect": inspect_hint(&s.session_id, r.chunk_id),
    })
}

#[cfg(test)]
mod tests;
