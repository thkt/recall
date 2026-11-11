use std::collections::{HashMap, HashSet};
use std::slice;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::compat::filter::{
    append_eq_filter, append_like_prefix_filter, append_timestamp_cutoff_filter, escape_like,
};
use crate::compat::{anon_placeholders, clean_for_trigram};
use crate::embedding::Embed;
use crate::storage::{
    MIN_TRIGRAM_TERM_CHARS, QueryNormalizationConfig, SanitizeError, prepare_match_query,
    prepare_operator_query,
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
use crate::hybrid::{self, RECENCY_BOOST_WEIGHT, rrf_merge_sessions};
use crate::model;
use crate::parser::{SessionData, Source};

const VEC_CANDIDATE_MULTIPLIER: usize = 3;
/// Floor of the vector candidate count, so a small `limit` still reads past the
/// few nearest chunks and the recency blend can lift a newer session into view.
const VEC_CANDIDATE_FLOOR: usize = 300;
/// Most neighbours a single vec0 query can return.
const VEC_CANDIDATE_CEILING: usize = 4096;

/// A skipped session's date is unknown, so it is judged with the largest
/// recency boost a session can get (`recency_decay` of a session dated now).
const MAX_BOOST_FACTOR: f64 = 1.0 + RECENCY_BOOST_WEIGHT;

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

/// The MATCH strings search renders quote every term, so the FTS5 parse errors
/// they can reach are plain `SQLITE_ERROR`s whose message starts with `fts5`,
/// such as `fts5: syntax error near` or `fts5: parser stack overflow`.
/// Corruption also starts with `fts5:` but carries `SQLITE_CORRUPT_VTAB`, and a
/// busy or locked database carries its own code.
fn is_fts5_query_syntax_error(err: &rusqlite::Error) -> bool {
    matches!(
        err,
        rusqlite::Error::SqliteFailure(e, Some(msg))
            if e.code == rusqlite::ErrorCode::Unknown && msg.starts_with("fts5")
    )
}

fn decode_session(row: &rusqlite::Row) -> rusqlite::Result<Option<SessionData>> {
    decode_session_at(row, 0)
}

/// Reads the six `sessions` columns that start at column `first`.
fn decode_session_at(row: &rusqlite::Row, first: usize) -> rusqlite::Result<Option<SessionData>> {
    let session_id: String = row.get(first)?;
    let source_str: String = row.get(first + 1)?;
    let Some(source) = Source::from_db(&source_str) else {
        warn!(source = %source_str, session_id, "unknown source");
        return Ok(None);
    };
    Ok(Some(SessionData {
        session_id,
        source,
        file_path: row.get(first + 2)?,
        project: row.get(first + 3)?,
        slug: row.get(first + 4)?,
        timestamp: row.get(first + 5)?,
    }))
}

/// Search reads every matching session, with its `sessions` row joined in,
/// because the recency blend can move any of them to the front. The cut to
/// `limit` comes after the blend, in `rank_sessions` or the hybrid fusion.
fn build_keyword_session_query(
    fts_query: &str,
    opts: &SearchOptions,
    now_ms: i64,
) -> (String, Vec<Box<dyn ToSql>>) {
    let mut sql = "SELECT messages.session_id, MIN(messages.rank) AS best_rank, \
                   s.session_id, s.source, s.file_path, s.project, s.slug, s.timestamp \
                   FROM messages LEFT JOIN sessions s ON s.session_id = messages.session_id \
                   WHERE messages MATCH ?"
        .to_owned();
    let mut params: Vec<Box<dyn ToSql>> = vec![Box::new(fts_query.to_owned())];
    append_search_filters(&mut sql, &mut params, "messages.session_id", opts, now_ms);
    sql.push_str(" GROUP BY messages.session_id ORDER BY best_rank, messages.session_id");
    (sql, params)
}

struct KeywordSessions {
    /// Every matching session id in `(best_rank, session_id)` order, skipped
    /// ones included, so fusion gives a skipped session the rank it holds.
    order: Vec<String>,
    ranked: Vec<(f64, SessionData)>,
    /// Best FTS rank of each session skipped for a missing `sessions` row, an
    /// unknown source, or unreadable metadata.
    skipped_ranks: Vec<f64>,
    /// A row that could not be read at all has no rank to estimate from.
    unreadable_rows: bool,
}

impl KeywordSessions {
    fn empty() -> Self {
        Self {
            order: Vec::new(),
            ranked: Vec::new(),
            skipped_ranks: Vec::new(),
            unreadable_rows: false,
        }
    }
}

/// A session with no `sessions` row or an unknown source is skipped, so it
/// never takes one of the `limit` slots. `keyword_only_outcome` flags the
/// results incomplete when that session could have ranked within `limit`.
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
                Some(_) => decode_session_at(row, 2),
                None => Ok(None),
            };
            Ok((session_id, best_rank, session))
        })
        .context("Search query failed")?;

    let mut order = Vec::new();
    let mut ranked = Vec::new();
    let mut first_error = None;
    let mut row_errors = 0;
    let mut skipped_ranks = Vec::new();
    let mut rows_read = 0;
    for r in rows {
        match r {
            Ok((session_id, best_rank, session)) => {
                rows_read += 1;
                order.push(session_id.clone());
                match session {
                    Ok(Some(session)) => ranked.push((best_rank, session)),
                    Ok(None) => {
                        warn!(
                            session_id,
                            "session skipped: no sessions row or unknown source"
                        );
                        skipped_ranks.push(best_rank);
                    }
                    Err(e) => {
                        warn!(error = %e, session_id, "session metadata unreadable");
                        skipped_ranks.push(best_rank);
                    }
                }
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
    Ok(KeywordSessions {
        order,
        ranked,
        skipped_ranks,
        unreadable_rows: row_errors > 0,
    })
}

fn fetch_session_metadata(
    conn: &Connection,
    session_ids: &[&str],
) -> Result<HashMap<String, SessionData>> {
    if session_ids.is_empty() {
        return Ok(HashMap::new());
    }
    let placeholders = anon_placeholders(session_ids.len());
    let sql = format!(
        "SELECT session_id, source, file_path, project, slug, timestamp \
         FROM sessions WHERE session_id IN ({placeholders})"
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(rusqlite::params_from_iter(session_ids), decode_session)?;

    let mut map = HashMap::new();
    let mut db_errors = 0;
    for r in rows {
        match r {
            Ok(Some(session)) => {
                map.insert(session.session_id.clone(), session);
            }
            Ok(None) => {}
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
    Ok(map)
}

fn build_candidates(
    ranked: Vec<(String, f64)>,
    meta_map: &mut HashMap<String, SessionData>,
    mut get_excerpt: impl FnMut(&str) -> (String, Option<i64>),
) -> Vec<(f64, SearchResult)> {
    ranked
        .into_iter()
        .filter_map(|(session_id, rank)| {
            let session = meta_map.remove(&session_id)?;
            let (excerpt, chunk_id) = get_excerpt(&session_id);
            Some((
                rank,
                SearchResult {
                    session,
                    excerpt,
                    chunk_id,
                },
            ))
        })
        .collect()
}

struct RankedSessions {
    ranked: Vec<(String, f64)>,
    meta_map: HashMap<String, SessionData>,
    /// Blended rank of the `limit`-th session, or `None` when fewer than
    /// `limit` sessions were ranked.
    limit_rank: Option<f64>,
}

/// Applies the recency blend to every session, sorts, and only then cuts to
/// `limit`, so a session the blend lifts to the front is not cut beforehand.
fn rank_sessions(sessions: Vec<(f64, SessionData)>, now_ms: i64, limit: usize) -> RankedSessions {
    let mut scored: Vec<(f64, SessionData)> = sessions
        .into_iter()
        .map(|(rank, session)| {
            debug_assert!(
                rank < 0.0,
                "FTS bm25 rank must be negative; rank={rank} would nullify the recency blend or invert ordering"
            );
            let recency_boost = hybrid::recency_decay(now_ms, session.timestamp);
            (rank * (1.0 + RECENCY_BOOST_WEIGHT * recency_boost), session)
        })
        .collect();
    scored.sort_by(|a, b| {
        a.0.total_cmp(&b.0)
            .then_with(|| a.1.session_id.cmp(&b.1.session_id))
    });
    let limit_rank = limit
        .checked_sub(1)
        .and_then(|i| scored.get(i))
        .map(|s| s.0);
    scored.truncate(limit);
    let ranked = scored
        .iter()
        .map(|(rank, session)| (session.session_id.clone(), *rank))
        .collect();
    let meta_map = scored
        .into_iter()
        .map(|(_, session)| (session.session_id.clone(), session))
        .collect();
    RankedSessions {
        ranked,
        meta_map,
        limit_rank,
    }
}

/// A skipped session scored with the largest boost reaches the results when it
/// ties or beats the `limit`-th blended rank (lower is better), or when the
/// results still have a free slot.
fn skipped_keyword_session_could_rank(skipped_ranks: &[f64], limit_rank: Option<f64>) -> bool {
    match limit_rank {
        Some(cut) => skipped_ranks.iter().any(|r| r * MAX_BOOST_FACTOR <= cut),
        None => !skipped_ranks.is_empty(),
    }
}

/// Same check after fusion (higher is better).
fn pruned_hybrid_session_could_rank(pruned_scores: &[f64], limit_score: Option<f64>) -> bool {
    match limit_score {
        Some(cut) => pruned_scores.iter().any(|s| s * MAX_BOOST_FACTOR >= cut),
        None => !pruned_scores.is_empty(),
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

/// `messages.session_id` is UNINDEXED, so a per-session MATCH still reads every
/// matching row in the index; grouping by session reads them once for all.
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

fn fetch_chunks(
    conn: &Connection,
    fts_query: Option<&str>,
    ranked: Vec<(String, f64)>,
    meta_map: &mut HashMap<String, SessionData>,
    vec_chunks: &HashMap<String, i64>,
) -> Result<Vec<(f64, SearchResult)>> {
    let session_ids: Vec<&str> = ranked.iter().map(|(sid, _)| sid.as_str()).collect();
    let (best_hits, first_hits) = match fts_query {
        Some(q) => (
            fts_rows_per_session(conn, q, &session_ids, "rowid, text, MIN(rank)", |row| {
                Ok((row.get::<_, i64>(1)?, row.get::<_, String>(2)?))
            })?,
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
        WHERE c.session_id = ?1 AND c.src_rowid_lo IS NULL AND instr(c.content, ?2) > 0 \
        HAVING COUNT(*) = 1";
    let mut instr_match_stmt = conn.prepare(instr_match_sql)?;
    let snippet_sql = "SELECT snippet(messages, 2, '**', '**', '...', 20) \
               FROM messages WHERE messages MATCH ?1 AND rowid = ?2";
    let mut snippet_stmt = conn.prepare(snippet_sql)?;
    let vec_chunk_sql = "SELECT content FROM qa_chunks WHERE id = ?1";
    let mut vec_chunk_stmt = conn.prepare(vec_chunk_sql)?;
    Ok(build_candidates(ranked, meta_map, |sid| {
        if let Some((rowid, text)) = best_hits.get(sid) {
            if let Some((chunk_id, content)) = row_or_default(
                rowid_match_stmt.query_row(rusqlite::params![sid, rowid], |row| {
                    Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
                }),
                sid,
            ) {
                return (content, Some(chunk_id));
            }
            if let Some(content) = row_or_default(
                instr_match_stmt.query_row(rusqlite::params![sid, text], |row| row.get(0)),
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
    }))
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

fn vec_candidate_count(limit: usize) -> usize {
    limit
        .saturating_mul(VEC_CANDIDATE_MULTIPLIER)
        .clamp(VEC_CANDIDATE_FLOOR, VEC_CANDIDATE_CEILING)
}

fn vec_search(
    conn: &Connection,
    embedder: &dyn Embed,
    query: &str,
    limit: usize,
    opts: &SearchOptions,
    now_ms: i64,
) -> Result<Vec<(String, i64)>> {
    let embedding = model::timed("query embedding", || embedder.embed_query(query))?;
    let embedding_bytes: &[u8] = f32_as_bytes(&embedding);

    let mut sql = String::from(
        "SELECT c.session_id, c.id, MIN(v.distance) AS d \
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
    sql.push_str(" GROUP BY c.session_id ORDER BY d, c.session_id");

    let limit_i64 = limit as i64;
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
        Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
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

/// What the keyword leg and the vector leg search for. `search_with_embedder`
/// and `count_excluded_automated` each build it with `keyword_query`, so both
/// agree on which sessions a query matches.
struct KeywordQuery {
    /// Final FTS5 MATCH string; `None` when no term is long enough for the trigram index.
    fts: Option<String>,
    /// MATCH strings for the right side of every NOT.
    not_operands: Vec<String>,
    /// Embedding input when it differs from the raw query.
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
    match prepare_match_query(
        conn,
        &fts_input,
        "messages_vocab",
        &QueryNormalizationConfig::disabled(),
    ) {
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

/// A session found only by the vector search never went through the keyword
/// MATCH, so it is dropped here when one of its messages matches a NOT operand.
/// Sessions the keyword search returned already satisfy the whole expression.
/// When the NOT query fails, every vector-only session is dropped and the
/// function returns true, so the caller reports the vector leg as degraded.
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

/// Matches reachable only through the vector leg are not counted, so the count
/// can fall short but never includes a session that does not match the search.
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

    let run_hybrid = match embedder {
        Some(_) => match has_vec_data(conn) {
            Ok(has_data) => has_data,
            Err(e) => {
                warn!(error = %e, "vec_chunks probe failed, using text search only");
                return keyword_only_outcome(conn, fts_query, opts, now_ms, true);
            }
        },
        None => false,
    };

    let Some(embedder) = embedder.filter(|_| run_hybrid) else {
        return keyword_only_outcome(conn, fts_query, opts, now_ms, false);
    };

    let found = match fts_query {
        Some(q) => find_keyword_sessions(conn, opts, q, now_ms)?,
        None => KeywordSessions::empty(),
    };
    let incomplete = found.unreadable_rows;

    let (vec_hits, vec_degraded) = match vec_search(
        conn,
        embedder,
        keyword.embed_text.as_deref().unwrap_or(query),
        vec_candidate_count(opts.limit),
        opts,
        now_ms,
    ) {
        Ok(hits) => (hits, false),
        Err(e) => {
            warn!(error = %e, "vector search failed, using text search only");
            (Vec::new(), true)
        }
    };

    let vec_chunks: HashMap<String, i64> = vec_hits.iter().cloned().collect();
    let mut merged = rrf_merge_sessions(
        found.order.iter().map(String::as_str),
        vec_hits.iter().map(|(sid, _)| sid.as_str()),
    );
    let fts_ids: HashSet<&str> = found.order.iter().map(String::as_str).collect();
    let not_filter_failed =
        drop_vec_only_sessions_matching_not(conn, &keyword.not_operands, &fts_ids, &mut merged);
    let vec_degraded = vec_degraded || not_filter_failed;

    if merged.is_empty() {
        return Ok(SearchOutcome {
            results: Vec::new(),
            vec_degraded,
            results_incomplete: incomplete,
        });
    }

    // The keyword read already carries each session's row; only sessions the
    // vector search alone found need a metadata query.
    let vec_only: Vec<&str> = merged
        .iter()
        .map(|(sid, _)| sid.as_str())
        .filter(|sid| !fts_ids.contains(sid))
        .collect();
    let mut meta_map = fetch_session_metadata(conn, &vec_only)?;
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
    let pruned_scores: Vec<f64> = pruned.into_iter().map(|(_, score)| score).collect();

    if merged.is_empty() {
        return Ok(SearchOutcome {
            results: Vec::new(),
            vec_degraded,
            results_incomplete: incomplete || !pruned_scores.is_empty(),
        });
    }

    hybrid::apply_recency_boost(
        &mut merged,
        |sid| meta_map.get(sid).and_then(|sd| sd.timestamp),
        now_ms,
        RECENCY_BOOST_WEIGHT,
    );
    let limit_score = merged.get(opts.limit - 1).map(|(_, score)| *score);
    let incomplete = incomplete || pruned_hybrid_session_could_rank(&pruned_scores, limit_score);
    merged.truncate(opts.limit);

    let candidates = fetch_chunks(conn, fts_query, merged, &mut meta_map, &vec_chunks)?;
    Ok(SearchOutcome {
        results: candidates.into_iter().map(|(_, r)| r).collect(),
        vec_degraded,
        results_incomplete: incomplete,
    })
}

fn keyword_only_outcome(
    conn: &Connection,
    fts_query: Option<&str>,
    opts: &SearchOptions,
    now_ms: i64,
    vec_degraded: bool,
) -> Result<SearchOutcome> {
    let Some(q) = fts_query else {
        return Ok(SearchOutcome {
            results: Vec::new(),
            vec_degraded,
            results_incomplete: false,
        });
    };
    let found = find_keyword_sessions(conn, opts, q, now_ms)?;
    let incomplete = |limit_rank| {
        found.unreadable_rows
            || skipped_keyword_session_could_rank(&found.skipped_ranks, limit_rank)
    };
    let mut ranking = rank_sessions(found.ranked, now_ms, opts.limit);
    let results_incomplete = incomplete(ranking.limit_rank);
    let candidates = fetch_chunks(
        conn,
        fts_query,
        ranking.ranked,
        &mut ranking.meta_map,
        &HashMap::new(),
    )?;
    Ok(SearchOutcome {
        results: candidates.into_iter().map(|(_, r)| r).collect(),
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
    sql.push_str(" ORDER BY s.timestamp DESC");
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
