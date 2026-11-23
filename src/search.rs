use std::collections::{HashMap, HashSet};
use std::slice;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::compat::filter::{
    append_eq_filter, append_like_prefix_filter, append_timestamp_cutoff_filter, escape_like,
};
use crate::compat::{anon_placeholders, clean_for_trigram};
use crate::embedding::Embed;
use crate::storage::{
    MIN_TRIGRAM_TERM_CHARS, SanitizeError, prepare_match_query, prepare_operator_query,
};
use anyhow::{Context, Result};
use rusqlite::Connection;
use rusqlite::OptionalExtension;
use rusqlite::types::ToSql;
use tracing::{debug, warn};

use crate::date::MS_PER_DAY;
use crate::db::table_def;
use crate::embedder::f32_as_bytes;
use crate::error::RecallError;
use crate::hybrid::{self, MAX_RECENCY_FACTOR, rrf_merge_sessions};
use crate::model;
use crate::parser::{SessionData, Source};

pub(crate) const MAX_LIMIT: u16 = 100;
const VEC_CANDIDATE_COUNT: usize = 3 * MAX_LIMIT as usize;
const SQLITE_VEC_VEC0_K_MAX: usize = 4096;
const _: () = assert!(VEC_CANDIDATE_COUNT <= SQLITE_VEC_VEC0_K_MAX);

pub struct SearchResult {
    pub session: SessionData,
    pub excerpt: String,
    pub chunk_id: Option<i64>,
}

pub struct SearchOutcome {
    pub results: Vec<SearchResult>,
    pub vec_degraded: bool,
    pub results_incomplete: bool,
}

impl SearchOutcome {
    fn nominal(results: Vec<SearchResult>) -> Self {
        Self {
            results,
            vec_degraded: false,
            results_incomplete: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum CurrentSession {
    #[default]
    Ignore,
    Exclude(String),
    Only(String),
}

#[derive(Clone)]
pub struct SearchOptions {
    pub project: Option<String>,
    pub days: Option<i64>,
    pub source: Option<Source>,
    pub file: Option<String>,
    pub limit: usize,
    pub current: CurrentSession,
    pub include_automated: bool,
    pub now_ms: Option<i64>,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            project: None,
            days: None,
            source: None,
            file: None,
            limit: 10,
            current: CurrentSession::Ignore,
            include_automated: false,
            now_ms: None,
        }
    }
}

fn append_search_filters(
    sql: &mut String,
    params: &mut Vec<Box<dyn ToSql>>,
    column: &'static str,
    opts: &SearchOptions,
    now_ms: i64,
) {
    if opts.project.is_some() || opts.days.is_some() || opts.source.is_some() {
        sql.push_str(" AND ");
        sql.push_str(column);
        sql.push_str(" IN (SELECT s2.session_id FROM sessions s2 WHERE 1 = 1");
        if let Some(ref project) = opts.project {
            append_like_prefix_filter(sql, params, "s2.project", slice::from_ref(project));
        }
        let cutoff = opts.days.map(|d| now_ms - d * MS_PER_DAY);
        append_timestamp_cutoff_filter(sql, params, "s2.timestamp", cutoff);
        let source_str = opts.source.as_ref().map(Source::as_str);
        append_eq_filter(sql, params, "s2.source", source_str);
        sql.push(')');
    }
    if let Some(ref file) = opts.file {
        sql.push_str(" AND ");
        sql.push_str(column);
        sql.push_str(
            " IN (SELECT sf.session_id FROM session_files sf \
             WHERE sf.path = ? OR sf.path LIKE ? ESCAPE '\\')",
        );
        params.push(Box::new(file.clone()));
        let mut suffix = String::from("%/");
        suffix.push_str(&escape_like(file));
        params.push(Box::new(suffix));
    }
    let current = match &opts.current {
        CurrentSession::Ignore => None,
        CurrentSession::Exclude(id) => Some((" <> ?", id)),
        CurrentSession::Only(id) => Some((" = ?", id)),
    };
    if let Some((op, id)) = current {
        sql.push_str(" AND ");
        sql.push_str(column);
        sql.push_str(op);
        params.push(Box::new(id.clone()));
    }
    if !opts.include_automated {
        sql.push_str(" AND ");
        sql.push_str(column);
        sql.push_str(
            " IN (SELECT s3.session_id FROM sessions s3 \
             WHERE COALESCE(s3.session_type, 'interactive') <> 'automated')",
        );
    }
}

fn candidate_query_error(err: &rusqlite::Error) -> anyhow::Error {
    if is_fts5_query_syntax_error(err) {
        RecallError::QuerySyntax(format!(
            "Invalid search query: {err}. Use words, \"quoted phrases\", or AND/OR/NOT operators."
        ))
    } else {
        RecallError::DataError(format!("Search failed: {err}"))
    }
    .into()
}

fn is_fts5_query_syntax_error(err: &rusqlite::Error) -> bool {
    matches!(
        err,
        rusqlite::Error::SqliteFailure(e, Some(msg))
            if e.code == rusqlite::ErrorCode::Unknown && msg.starts_with("fts5")
    )
}

enum DecodedSession {
    Known(SessionData),
    UnknownSource {
        session_id: String,
        timestamp: Option<i64>,
    },
}

fn decode_session(row: &rusqlite::Row) -> rusqlite::Result<Option<SessionData>> {
    Ok(match decode_session_at(row, 0)? {
        DecodedSession::Known(session) => Some(session),
        DecodedSession::UnknownSource { .. } => None,
    })
}

fn decode_session_at(row: &rusqlite::Row, first_column: usize) -> rusqlite::Result<DecodedSession> {
    let session_id: String = row.get(first_column)?;
    let source_str: String = row.get(first_column + 1)?;
    let timestamp = row.get(first_column + 5)?;
    let Some(source) = Source::from_db(&source_str) else {
        return Ok(DecodedSession::UnknownSource {
            session_id,
            timestamp,
        });
    };
    Ok(DecodedSession::Known(SessionData {
        session_id,
        source,
        file_path: row.get(first_column + 2)?,
        project: row.get(first_column + 3)?,
        slug: row.get(first_column + 4)?,
        timestamp,
    }))
}

fn build_keyword_session_query(
    fts_query: &str,
    opts: &SearchOptions,
    now_ms: i64,
) -> (String, Vec<Box<dyn ToSql>>) {
    let mut inner =
        "SELECT session_id, MIN(rank) AS best_rank FROM messages WHERE messages MATCH ?".to_owned();
    let mut params: Vec<Box<dyn ToSql>> = vec![Box::new(fts_query.to_owned())];
    append_search_filters(&mut inner, &mut params, "session_id", opts, now_ms);
    inner.push_str(" GROUP BY session_id");
    let sql = format!(
        "SELECT m.session_id, m.best_rank, \
         s.session_id, s.source, s.file_path, s.project, s.slug, s.timestamp \
         FROM ({inner}) m LEFT JOIN sessions s ON s.session_id = m.session_id \
         ORDER BY m.best_rank, m.session_id"
    );
    (sql, params)
}

struct SkippedSession {
    session_id: String,
    rank: f64,
    best_factor: f64,
}

#[derive(Default)]
struct KeywordSessions {
    order: Vec<(String, f64)>,
    ranked: Vec<(f64, SessionData)>,
    skipped: Vec<SkippedSession>,
    unreadable_rows: bool,
}

fn find_keyword_sessions(
    conn: &Connection,
    opts: &SearchOptions,
    fts_query: &str,
    now_ms: i64,
) -> Result<KeywordSessions> {
    let (sql, params) = build_keyword_session_query(fts_query, opts, now_ms);
    debug_assert_eq!(
        params.len(),
        sql.matches('?').count(),
        "param count must match SQL placeholder count"
    );
    let mut stmt = conn
        .prepare(&sql)
        .context("Failed to prepare search query")?;
    let rows = stmt
        .query_map(rusqlite::params_from_iter(params.iter()), |row| {
            let session_id: String = row.get(0)?;
            let best_rank: f64 = row.get(1)?;
            let session = match row.get::<_, Option<String>>(2)? {
                Some(_) => decode_session_at(row, 2).map(Some),
                None => Ok(None),
            };
            Ok((session_id, best_rank, session))
        })
        .context("Search query failed")?;

    let mut order = Vec::new();
    let mut ranked = Vec::new();
    let mut first_error = None;
    let mut row_errors = 0;
    let mut skipped = Vec::new();
    let (mut missing_row, mut unknown_source, mut unreadable_metadata) = (0, 0, 0);
    let mut first_metadata_error = None;
    let mut rows_read = 0;
    for r in rows {
        match r {
            Ok((session_id, best_rank, session)) => {
                rows_read += 1;
                order.push((session_id.clone(), best_rank));
                let best_factor = match session {
                    Ok(Some(DecodedSession::Known(session))) => {
                        ranked.push((best_rank, session));
                        continue;
                    }
                    Ok(Some(DecodedSession::UnknownSource { timestamp, .. })) => {
                        unknown_source += 1;
                        hybrid::recency_factor(now_ms, timestamp)
                    }
                    Ok(None) => {
                        missing_row += 1;
                        MAX_RECENCY_FACTOR
                    }
                    Err(e) => {
                        first_metadata_error.get_or_insert(e);
                        unreadable_metadata += 1;
                        MAX_RECENCY_FACTOR
                    }
                };
                skipped.push(SkippedSession {
                    session_id,
                    rank: best_rank,
                    best_factor,
                });
            }
            Err(e) => {
                first_error.get_or_insert(e);
                row_errors += 1;
            }
        }
    }
    if let Some(err) = first_error {
        if rows_read == 0 {
            return Err(candidate_query_error(&err));
        }
        warn!(row_errors, %err, "rows skipped during search");
    }
    // why-not: a warn line per skipped session would repeat on every search, since every match is read.
    if !skipped.is_empty() {
        let first_error = first_metadata_error
            .map(|e| e.to_string())
            .unwrap_or_default();
        warn!(
            missing_row,
            unknown_source, unreadable_metadata, first_error, "matching sessions skipped"
        );
    }
    Ok(KeywordSessions {
        order,
        ranked,
        skipped,
        unreadable_rows: row_errors > 0,
    })
}

#[derive(Default)]
struct SessionRows {
    sessions: HashMap<String, SessionData>,
    unknown_source_dates: HashMap<String, Option<i64>>,
}

fn fetch_session_metadata(conn: &Connection, session_ids: &[&str]) -> Result<SessionRows> {
    if session_ids.is_empty() {
        return Ok(SessionRows::default());
    }
    let placeholders = anon_placeholders(session_ids.len());
    let sql = format!(
        "SELECT session_id, source, file_path, project, slug, timestamp \
         FROM sessions WHERE session_id IN ({placeholders})"
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(rusqlite::params_from_iter(session_ids), |row| {
        decode_session_at(row, 0)
    })?;

    let mut found = SessionRows::default();
    let mut db_errors = 0;
    for r in rows {
        match r {
            Ok(DecodedSession::Known(session)) => {
                found.sessions.insert(session.session_id.clone(), session);
            }
            Ok(DecodedSession::UnknownSource {
                session_id,
                timestamp,
            }) => {
                found.unknown_source_dates.insert(session_id, timestamp);
            }
            Err(e) => {
                if db_errors == 0 {
                    warn!(error = %e, "metadata query error");
                }
                db_errors += 1;
            }
        }
    }
    if db_errors > 1 {
        warn!(db_errors, "metadata rows failed");
    }
    if !found.unknown_source_dates.is_empty() {
        warn!(
            unknown_source = found.unknown_source_dates.len(),
            "vector matches skipped"
        );
    }
    Ok(found)
}

struct RankedSessions {
    sessions: Vec<SessionData>,
    limit_rank: Option<f64>,
}

fn rank_sessions(sessions: Vec<(f64, SessionData)>, now_ms: i64, limit: usize) -> RankedSessions {
    let mut scored: Vec<(f64, SessionData)> = sessions
        .into_iter()
        .map(|(rank, session)| {
            debug_assert!(
                rank < 0.0,
                "FTS bm25 rank must be negative; rank={rank} would nullify the recency blend or invert ordering"
            );
            (rank * hybrid::recency_factor(now_ms, session.timestamp), session)
        })
        .collect();
    scored.sort_by(|a, b| {
        a.0.total_cmp(&b.0)
            .then_with(|| a.1.session_id.cmp(&b.1.session_id))
    });
    let limit_rank = at_limit(&scored, limit).map(|s| s.0);
    scored.truncate(limit);
    RankedSessions {
        sessions: scored.into_iter().map(|(_, session)| session).collect(),
        limit_rank,
    }
}

fn at_limit<T>(items: &[T], limit: usize) -> Option<&T> {
    limit.checked_sub(1).and_then(|i| items.get(i))
}

fn left_out_session_could_rank(
    best_scores: impl IntoIterator<Item = f64>,
    limit_score: Option<f64>,
) -> bool {
    let mut best_scores = best_scores.into_iter();
    match limit_score {
        Some(cut) => best_scores.any(|s| s >= cut),
        None => best_scores.next().is_some(),
    }
}

fn row_or_default<T>(result: rusqlite::Result<T>, session_id: &str) -> Option<T> {
    match result {
        Ok(s) => Some(s),
        Err(rusqlite::Error::QueryReturnedNoRows) => None,
        Err(e) => {
            warn!(error = %e, session_id, "snippet extraction failed");
            None
        }
    }
}

// why-not: `messages.session_id` is UNINDEXED, so a MATCH per session reads every matching row each time.
fn fts_rows_per_session<T>(
    conn: &Connection,
    fts_query: &str,
    session_ids: &[&str],
    pick: &str,
    decode: impl Fn(&rusqlite::Row) -> rusqlite::Result<T>,
) -> Result<HashMap<String, T>> {
    if session_ids.is_empty() {
        return Ok(HashMap::new());
    }
    let sql = format!(
        "SELECT session_id, {pick} FROM messages \
         WHERE messages MATCH ? AND session_id IN ({}) GROUP BY session_id",
        anon_placeholders(session_ids.len())
    );
    let mut params: Vec<&dyn ToSql> = vec![&fts_query];
    params.extend(session_ids.iter().map(|sid| sid as &dyn ToSql));
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(params.as_slice(), |row| {
        Ok((row.get::<_, String>(0)?, decode(row)?))
    });
    let mut picked = HashMap::new();
    match rows.and_then(|rows| rows.collect::<rusqlite::Result<Vec<_>>>()) {
        Ok(rows) => picked.extend(rows),
        Err(e) => warn!(error = %e, "snippet extraction failed"),
    }
    Ok(picked)
}

// why-not: `MIN(rank)` with bare columns leaves a rank tie to SQLite, and a window sorts every matching row.
fn best_matching_messages(
    conn: &Connection,
    fts_query: &str,
    session_ids: &[&str],
) -> Result<HashMap<String, i64>> {
    if session_ids.is_empty() {
        return Ok(HashMap::new());
    }
    let sql = format!(
        "SELECT session_id, rowid, rank FROM messages \
         WHERE messages MATCH ? AND session_id IN ({})",
        anon_placeholders(session_ids.len())
    );
    let mut params: Vec<&dyn ToSql> = vec![&fts_query];
    params.extend(session_ids.iter().map(|sid| sid as &dyn ToSql));
    let mut stmt = conn.prepare(&sql)?;
    let mut best: HashMap<String, (f64, i64)> = HashMap::new();
    let read = stmt
        .query_map(params.as_slice(), |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, f64>(2)?,
            ))
        })
        .and_then(|rows| {
            for row in rows {
                let (session_id, rowid, rank) = row?;
                let kept = best.entry(session_id).or_insert((rank, rowid));
                if rank.total_cmp(&kept.0).then(rowid.cmp(&kept.1)).is_lt() {
                    *kept = (rank, rowid);
                }
            }
            Ok(())
        });
    if let Err(e) = read {
        warn!(error = %e, "best matching message lookup failed");
        return Ok(HashMap::new());
    }
    Ok(best
        .into_iter()
        .map(|(session_id, (_, rowid))| (session_id, rowid))
        .collect())
}

fn fetch_chunks(
    conn: &Connection,
    fts_query: Option<&str>,
    sessions: Vec<SessionData>,
    vec_chunks: &HashMap<String, i64>,
) -> Result<Vec<SearchResult>> {
    let session_ids: Vec<&str> = sessions.iter().map(|s| s.session_id.as_str()).collect();
    let (best_hits, first_hits) = match fts_query {
        Some(q) => (
            best_matching_messages(conn, q, &session_ids)?,
            fts_rows_per_session(conn, q, &session_ids, "MIN(rowid)", |row| {
                row.get::<_, i64>(1)
            })?,
        ),
        None => (HashMap::new(), HashMap::new()),
    };
    let rowid_match_sql = "SELECT c.id, MAX(c.content) FROM qa_chunks c \
        WHERE c.session_id = ?1 AND c.src_rowid_lo IS NOT NULL \
          AND ?2 BETWEEN c.src_rowid_lo AND c.src_rowid_hi HAVING COUNT(*) = 1";
    let mut rowid_match_stmt = conn.prepare(rowid_match_sql)?;
    let instr_match_sql = "SELECT MAX(c.content) FROM qa_chunks c \
        WHERE c.session_id = ?1 AND c.src_rowid_lo IS NULL \
          AND instr(c.content, (SELECT text FROM messages WHERE rowid = ?2)) > 0 \
        HAVING COUNT(*) = 1";
    let mut instr_match_stmt = conn.prepare(instr_match_sql)?;
    let snippet_sql = "SELECT snippet(messages, 2, '**', '**', '...', 20) \
               FROM messages WHERE messages MATCH ?1 AND rowid = ?2";
    let mut snippet_stmt = conn.prepare(snippet_sql)?;
    let vec_chunk_sql = "SELECT content FROM qa_chunks WHERE id = ?1";
    let mut vec_chunk_stmt = conn.prepare(vec_chunk_sql)?;
    let mut excerpt_of = |sid: &str| -> (String, Option<i64>) {
        if let Some(rowid) = best_hits.get(sid) {
            if let Some((chunk_id, content)) = row_or_default(
                rowid_match_stmt.query_row(rusqlite::params![sid, rowid], |row| {
                    Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
                }),
                sid,
            ) {
                return (content, Some(chunk_id));
            }
            if let Some(content) = row_or_default(
                instr_match_stmt.query_row(rusqlite::params![sid, rowid], |row| row.get(0)),
                sid,
            ) {
                return (content, None);
            }
        }
        if let (Some(fts_query), Some(rowid)) = (fts_query, first_hits.get(sid))
            && let Some(snippet) = row_or_default(
                snippet_stmt.query_row(rusqlite::params![fts_query, rowid], |row| row.get(0)),
                sid,
            )
        {
            return (snippet, None);
        }
        if let Some(&chunk_id) = vec_chunks.get(sid)
            && let Some(content) = row_or_default(
                vec_chunk_stmt.query_row(rusqlite::params![chunk_id], |row| row.get(0)),
                sid,
            )
        {
            return (content, Some(chunk_id));
        }
        (String::new(), None)
    };
    Ok(sessions
        .into_iter()
        .map(|session| {
            let (excerpt, chunk_id) = excerpt_of(&session.session_id);
            SearchResult {
                session,
                excerpt,
                chunk_id,
            }
        })
        .collect())
}

fn resolve_now_ms(opts: &SearchOptions) -> Result<i64> {
    match opts.now_ms {
        Some(ms) => Ok(ms),
        None => i64::try_from(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .context("system clock before UNIX epoch")?
                .as_millis(),
        )
        .context("timestamp overflow"),
    }
}

fn has_vec_data(conn: &Connection) -> Result<bool> {
    Ok(conn
        .query_row("SELECT 1 FROM vec_chunks LIMIT 1", [], |_| Ok(()))
        .optional()
        .context("Failed to probe vec_chunks for embeddings")?
        .is_some())
}

struct VecHit {
    session_id: String,
    chunk_id: i64,
    distance: f64,
}

fn vec_search(
    conn: &Connection,
    embedder: &dyn Embed,
    query: &str,
    opts: &SearchOptions,
    now_ms: i64,
) -> Result<Vec<VecHit>> {
    let embedding = model::timed("query embedding", || embedder.embed_query(query))?;
    let embedding_bytes: &[u8] = f32_as_bytes(&embedding);

    let mut sql = String::from(
        "SELECT c.session_id, c.id, v.distance AS d, row_number() OVER ( \
         PARTITION BY c.session_id ORDER BY v.distance, c.id) AS n \
         FROM qa_chunks c \
         JOIN ( \
             SELECT chunk_id, distance FROM vec_chunks \
             WHERE embedding MATCH ? \
             ORDER BY distance \
             LIMIT ? \
         ) v ON c.id = v.chunk_id \
         WHERE 1 = 1",
    );
    let mut filter_params: Vec<Box<dyn ToSql>> = Vec::new();
    append_search_filters(&mut sql, &mut filter_params, "c.session_id", opts, now_ms);
    let sql = format!("SELECT session_id, id, d FROM ({sql}) WHERE n = 1 ORDER BY d, session_id");

    let limit_i64 = VEC_CANDIDATE_COUNT as i64;
    let mut refs: Vec<&dyn ToSql> = Vec::with_capacity(2 + filter_params.len());
    refs.push(&embedding_bytes);
    refs.push(&limit_i64);
    for p in &filter_params {
        refs.push(p.as_ref());
    }
    debug_assert_eq!(
        refs.len(),
        sql.matches('?').count(),
        "param count must match SQL placeholder count"
    );

    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(refs.as_slice(), |row| {
        Ok(VecHit {
            session_id: row.get(0)?,
            chunk_id: row.get(1)?,
            distance: row.get(2)?,
        })
    })?;

    let mut hits = Vec::new();
    for r in rows {
        hits.push(r?);
    }
    Ok(hits)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Script {
    Han,
    Hiragana,
    Katakana,
    Latin,
    Other,
}

fn classify_script(c: char) -> Script {
    match c {
        'a'..='z' | 'A'..='Z' | '0'..='9' => Script::Latin,
        '\u{3040}'..='\u{309F}' => Script::Hiragana,
        '\u{30A0}'..='\u{30FF}' | '\u{31F0}'..='\u{31FF}' | '\u{FF66}'..='\u{FF9D}' => {
            Script::Katakana
        }
        '\u{3005}'
        | '\u{3400}'..='\u{4DBF}'
        | '\u{4E00}'..='\u{9FFF}'
        | '\u{F900}'..='\u{FAFF}' => Script::Han,
        _ => Script::Other,
    }
}

fn segment_script_boundaries(query: &str) -> String {
    let mut out = String::with_capacity(query.len() + 8);
    let mut in_quotes = false;
    let mut prev: Option<Script> = None;
    for c in query.chars() {
        if c == '"' {
            in_quotes = !in_quotes;
            out.push(c);
            prev = None;
            continue;
        }
        if in_quotes {
            out.push(c);
            continue;
        }
        let cur = classify_script(c);
        if cur != Script::Other
            && let Some(p) = prev
            && p != cur
        {
            out.push(' ');
        }
        out.push(c);
        prev = (cur != Script::Other).then_some(cur);
    }
    out
}

struct KeywordQuery {
    fts: Option<String>,
    not_operands: Vec<String>,
    embed_text: Option<String>,
}

fn keyword_query(conn: &Connection, query: &str) -> Result<Option<KeywordQuery>> {
    if let Some(operator) =
        prepare_operator_query(query).map_err(|e| RecallError::QuerySyntax(e.to_string()))?
    {
        if operator.embed_text.is_empty() {
            debug!(%query, "operator query holds no term");
            return Ok(None);
        }
        return Ok(Some(KeywordQuery {
            fts: operator.match_query,
            not_operands: operator.not_operands,
            embed_text: Some(operator.embed_text),
        }));
    }
    let segmented = segment_script_boundaries(query);
    let fts_input = if segmented
        .split_whitespace()
        .any(|t| t.chars().count() >= MIN_TRIGRAM_TERM_CHARS)
    {
        segmented
    } else {
        query.to_owned()
    };
    match prepare_match_query(conn, &fts_input, "messages_vocab") {
        Ok(m) => Ok(Some(KeywordQuery {
            fts: clean_for_trigram(&m),
            not_operands: Vec::new(),
            embed_text: None,
        })),
        Err(SanitizeError::EmptyInput | SanitizeError::NoSearchableTerms) => {
            debug!(%query, "query produced no searchable terms");
            Ok(None)
        }
        Err(e @ (SanitizeError::InvalidVocabTable(_) | SanitizeError::VocabLookupFailed(_))) => {
            Err(e.into())
        }
    }
}

fn drop_vec_only_sessions_matching_not(
    conn: &Connection,
    not_operands: &[String],
    fts_ids: &HashSet<&str>,
    merged: &mut Vec<(String, f64)>,
) -> bool {
    if not_operands.is_empty() {
        return false;
    }
    let vec_only: Vec<&str> = merged
        .iter()
        .map(|(sid, _)| sid.as_str())
        .filter(|sid| !fts_ids.contains(sid))
        .collect();
    if vec_only.is_empty() {
        return false;
    }
    match sessions_matching_any(conn, not_operands, &vec_only) {
        Ok(excluded) => {
            merged.retain(|(sid, _)| !excluded.contains(sid));
            false
        }
        Err(e) => {
            warn!(error = %e, "NOT filter on vector results failed, using text search only");
            merged.retain(|(sid, _)| fts_ids.contains(sid.as_str()));
            true
        }
    }
}

fn sessions_matching_any(
    conn: &Connection,
    operands: &[String],
    session_ids: &[&str],
) -> Result<HashSet<String>> {
    let operands = operands.join(" OR ");
    let sql = format!(
        "SELECT DISTINCT session_id FROM messages \
         WHERE messages MATCH ? AND session_id IN ({})",
        anon_placeholders(session_ids.len())
    );
    let mut params: Vec<&dyn ToSql> = vec![&operands];
    params.extend(session_ids.iter().map(|sid| sid as &dyn ToSql));
    let mut stmt = conn
        .prepare(&sql)
        .context("Failed to prepare NOT filter query")?;
    stmt.query_map(params.as_slice(), |row| row.get::<_, String>(0))
        .and_then(|rows| rows.collect::<rusqlite::Result<_>>())
        .context("NOT filter query failed")
}

pub fn count_excluded_automated(
    conn: &Connection,
    query: Option<&str>,
    opts: &SearchOptions,
) -> Result<usize> {
    if opts.include_automated {
        return Ok(0);
    }
    let now_ms = resolve_now_ms(opts)?;
    let unfiltered = SearchOptions {
        include_automated: true,
        ..opts.clone()
    };
    let mut params: Vec<Box<dyn ToSql>> = Vec::new();
    let (mut sql, column) = match query {
        Some(query) => {
            let Some(fts_query) = keyword_query(conn, query)?.and_then(|k| k.fts) else {
                return Ok(0);
            };
            params.push(Box::new(fts_query));
            (
                "SELECT COUNT(DISTINCT session_id) FROM messages WHERE messages MATCH ?".to_owned(),
                "session_id",
            )
        }
        None if opts.file.is_some() => (
            "SELECT COUNT(*) FROM sessions s WHERE 1 = 1".to_owned(),
            "s.session_id",
        ),
        None => return Ok(0),
    };
    append_search_filters(&mut sql, &mut params, column, &unfiltered, now_ms);
    sql.push_str(" AND ");
    sql.push_str(column);
    sql.push_str(
        " IN (SELECT s4.session_id FROM sessions s4 \
         WHERE COALESCE(s4.session_type, 'interactive') = 'automated')",
    );
    debug_assert_eq!(
        params.len(),
        sql.matches('?').count(),
        "param count must match SQL placeholder count"
    );
    let count: i64 = conn.query_row(&sql, rusqlite::params_from_iter(params.iter()), |r| {
        r.get(0)
    })?;
    Ok(usize::try_from(count).unwrap_or(0))
}

#[cfg(test)]
pub fn search(conn: &Connection, query: &str, opts: &SearchOptions) -> Result<Vec<SearchResult>> {
    Ok(search_with_embedder(conn, query, opts, None)?.results)
}

pub fn search_with_embedder<'q>(
    conn: &Connection,
    query: impl Into<Option<&'q str>>,
    opts: &SearchOptions,
    embedder: Option<&dyn Embed>,
) -> Result<SearchOutcome> {
    if opts.limit == 0 {
        return Ok(SearchOutcome::nominal(Vec::new()));
    }

    if opts.file.is_some() && table_def(conn, "session_files")?.is_none() {
        return Err(RecallError::DataError(
            "`--file` needs the scanned-files index; run `recall index` once to record it."
                .to_owned(),
        )
        .into());
    }

    let now_ms = resolve_now_ms(opts)?;

    let Some(query) = query.into() else {
        return file_list_outcome(conn, opts, now_ms);
    };
    let Some(keyword) = keyword_query(conn, query)? else {
        return Ok(SearchOutcome::nominal(Vec::new()));
    };
    let fts_query = keyword.fts.as_deref();
    let found = match fts_query {
        Some(q) => find_keyword_sessions(conn, opts, q, now_ms)?,
        None => KeywordSessions::default(),
    };

    let run_hybrid = match embedder {
        Some(_) => match has_vec_data(conn) {
            Ok(has_data) => has_data,
            Err(e) => {
                warn!(error = %e, "vec_chunks probe failed, using text search only");
                return keyword_only_outcome(conn, fts_query, found, opts, now_ms, true);
            }
        },
        None => false,
    };

    let Some(embedder) = embedder.filter(|_| run_hybrid) else {
        return keyword_only_outcome(conn, fts_query, found, opts, now_ms, false);
    };

    let incomplete = found.unreadable_rows;

    let vec_hits = match vec_search(
        conn,
        embedder,
        keyword.embed_text.as_deref().unwrap_or(query),
        opts,
        now_ms,
    ) {
        Ok(hits) => hits,
        Err(e) => {
            warn!(error = %e, "vector search failed, using text search only");
            return keyword_only_outcome(conn, fts_query, found, opts, now_ms, true);
        }
    };

    let vec_chunks: HashMap<String, i64> = vec_hits
        .iter()
        .map(|hit| (hit.session_id.clone(), hit.chunk_id))
        .collect();
    let mut merged = rrf_merge_sessions(
        found.order.iter().map(|(sid, rank)| (sid.as_str(), *rank)),
        vec_hits
            .iter()
            .map(|hit| (hit.session_id.as_str(), hit.distance)),
    );
    let fts_ids: HashSet<&str> = found.order.iter().map(|(sid, _)| sid.as_str()).collect();
    let vec_degraded =
        drop_vec_only_sessions_matching_not(conn, &keyword.not_operands, &fts_ids, &mut merged);

    if merged.is_empty() {
        return Ok(SearchOutcome {
            results: Vec::new(),
            vec_degraded,
            results_incomplete: incomplete,
        });
    }

    let vec_only: Vec<&str> = merged
        .iter()
        .map(|(sid, _)| sid.as_str())
        .filter(|sid| !fts_ids.contains(sid))
        .collect();
    let rows = fetch_session_metadata(conn, &vec_only)?;
    let mut meta_map = rows.sessions;
    meta_map.extend(
        found
            .ranked
            .into_iter()
            .map(|(_, session)| (session.session_id.clone(), session)),
    );

    let (kept, pruned): (Vec<_>, Vec<_>) = merged
        .into_iter()
        .partition(|(sid, _)| meta_map.contains_key(sid));
    merged = kept;
    let skipped_factors: HashMap<&str, f64> = found
        .skipped
        .iter()
        .map(|s| (s.session_id.as_str(), s.best_factor))
        .chain(
            rows.unknown_source_dates
                .iter()
                .map(|(sid, ts)| (sid.as_str(), hybrid::recency_factor(now_ms, *ts))),
        )
        .collect();
    let pruned_best: Vec<f64> = pruned
        .iter()
        .map(|(sid, score)| {
            score
                * skipped_factors
                    .get(sid.as_str())
                    .copied()
                    .unwrap_or(MAX_RECENCY_FACTOR)
        })
        .collect();

    if merged.is_empty() {
        return Ok(SearchOutcome {
            results: Vec::new(),
            vec_degraded,
            results_incomplete: incomplete || left_out_session_could_rank(pruned_best, None),
        });
    }

    hybrid::apply_recency_boost(
        &mut merged,
        |sid| meta_map.get(sid).and_then(|sd| sd.timestamp),
        now_ms,
    );
    let limit_score = at_limit(&merged, opts.limit).map(|(_, score)| *score);
    let incomplete = incomplete || left_out_session_could_rank(pruned_best, limit_score);
    merged.truncate(opts.limit);

    let sessions: Vec<SessionData> = merged
        .into_iter()
        .filter_map(|(sid, _)| meta_map.remove(&sid))
        .collect();
    Ok(SearchOutcome {
        results: fetch_chunks(conn, fts_query, sessions, &vec_chunks)?,
        vec_degraded,
        results_incomplete: incomplete,
    })
}

fn keyword_only_outcome(
    conn: &Connection,
    fts_query: Option<&str>,
    found: KeywordSessions,
    opts: &SearchOptions,
    now_ms: i64,
    vec_degraded: bool,
) -> Result<SearchOutcome> {
    let ranking = rank_sessions(found.ranked, now_ms, opts.limit);
    let results_incomplete = found.unreadable_rows
        || left_out_session_could_rank(
            found.skipped.iter().map(|s| -(s.rank * s.best_factor)),
            ranking.limit_rank.map(|r| -r),
        );
    if ranking.sessions.is_empty() {
        return Ok(SearchOutcome {
            results: Vec::new(),
            vec_degraded,
            results_incomplete,
        });
    }
    Ok(SearchOutcome {
        results: fetch_chunks(conn, fts_query, ranking.sessions, &HashMap::new())?,
        vec_degraded,
        results_incomplete,
    })
}

fn file_list_outcome(
    conn: &Connection,
    opts: &SearchOptions,
    now_ms: i64,
) -> Result<SearchOutcome> {
    if opts.file.is_none() {
        return Err(RecallError::Usage("A search query or --file is required.".to_owned()).into());
    }

    let mut sql = "SELECT s.session_id, s.source, s.file_path, s.project, s.slug, s.timestamp \
                   FROM sessions s WHERE 1 = 1"
        .to_owned();
    let mut params: Vec<Box<dyn ToSql>> = Vec::new();
    append_search_filters(&mut sql, &mut params, "s.session_id", opts, now_ms);
    sql.push_str(" ORDER BY s.timestamp DESC, s.session_id");
    debug_assert_eq!(
        params.len(),
        sql.matches('?').count(),
        "param count must match SQL placeholder count"
    );

    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(rusqlite::params_from_iter(params.iter()), decode_session)?;

    let mut results = Vec::new();
    let mut skipped = 0;
    let mut first_error = None;
    for r in rows {
        if results.len() >= opts.limit {
            break;
        }
        match r {
            Ok(Some(session)) => results.push(SearchResult {
                session,
                excerpt: String::new(),
                chunk_id: None,
            }),
            Ok(None) => skipped += 1,
            Err(e) => {
                if first_error.is_none() {
                    first_error = Some(e.to_string());
                }
                skipped += 1;
            }
        }
    }
    if skipped > 0 {
        let err_msg = first_error.as_deref().unwrap_or("unknown source");
        warn!(skipped, err_msg, "rows dropped in --file list");
    }
    Ok(SearchOutcome {
        results,
        vec_degraded: false,
        results_incomplete: skipped > 0,
    })
}

#[cfg(test)]
mod syntax_tests;
#[cfg(test)]
mod tests;
