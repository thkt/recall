use std::collections::HashMap;
use std::slice;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::compat::filter::{
    append_eq_filter, append_like_prefix_filter, append_timestamp_cutoff_filter, escape_like,
};
use crate::compat::{anon_placeholders, clean_for_trigram};
use crate::embedding::Embed;
use crate::storage::{QueryNormalizationConfig, SanitizeError, prepare_match_query};
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
use crate::parser::{SessionData, Source};

const CANDIDATE_LIMIT_MULTIPLIER: usize = 3;

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

fn build_fts_candidate_query(
    fts_query: &str,
    opts: &SearchOptions,
    now_ms: i64,
) -> (String, Vec<Box<dyn ToSql>>) {
    let mut sql =
        "SELECT session_id, MIN(rank) as best_rank FROM messages WHERE messages MATCH ?".to_owned();
    let mut params: Vec<Box<dyn ToSql>> = vec![Box::new(fts_query.to_owned())];
    append_search_filters(&mut sql, &mut params, "session_id", opts, now_ms);
    sql.push_str(" GROUP BY session_id ORDER BY best_rank LIMIT ?");
    let candidate_limit = opts.limit * CANDIDATE_LIMIT_MULTIPLIER;
    params.push(Box::new(candidate_limit as i64));
    (sql, params)
}

fn find_candidate_sessions(
    conn: &Connection,
    opts: &SearchOptions,
    fts_query: &str,
    now_ms: i64,
) -> Result<(Vec<(String, f64)>, usize)> {
    let (sql, params) = build_fts_candidate_query(fts_query, opts, now_ms);
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
            Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?))
        })
        .context("Search query failed")?;

    let mut ranked = Vec::new();
    let mut first_error = None;
    let mut skipped = 0;
    for r in rows {
        match r {
            Ok(row) => ranked.push(row),
            Err(e) => {
                if first_error.is_none() {
                    first_error = Some(e.to_string());
                }
                skipped += 1;
            }
        }
    }
    if let Some(ref err_msg) = first_error {
        if ranked.is_empty() {
            return Err(RecallError::DataError(format!(
                "Invalid search query: {err_msg}. Use words, \"quoted phrases\", or AND/OR/NOT operators."
            ))
            .into());
        }
        warn!(skipped, %err_msg, "rows skipped during search");
    }
    Ok((ranked, skipped))
}

fn decode_session(row: &rusqlite::Row) -> rusqlite::Result<Option<SessionData>> {
    let session_id: String = row.get(0)?;
    let source_str: String = row.get(1)?;
    let Some(source) = Source::from_db(&source_str) else {
        warn!(source = %source_str, session_id, "unknown source");
        return Ok(None);
    };
    Ok(Some(SessionData {
        session_id,
        source,
        file_path: row.get(2)?,
        project: row.get(3)?,
        slug: row.get(4)?,
        timestamp: row.get(5)?,
    }))
}

fn fetch_session_metadata(
    conn: &Connection,
    ranked: &[(String, f64)],
) -> Result<HashMap<String, SessionData>> {
    let placeholders = anon_placeholders(ranked.len());
    let sql = format!(
        "SELECT session_id, source, file_path, project, slug, timestamp \
         FROM sessions WHERE session_id IN ({placeholders})"
    );
    let params = rusqlite::params_from_iter(ranked.iter().map(|(sid, _)| sid));
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(params, decode_session)?;

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

fn score_and_sort(
    candidates: Vec<(f64, SearchResult)>,
    now_ms: i64,
    limit: usize,
) -> Vec<SearchResult> {
    let mut results: Vec<(SearchResult, f64)> = candidates
        .into_iter()
        .map(|(rank, result)| {
            debug_assert!(
                rank < 0.0,
                "FTS bm25 rank must be negative; rank={rank} would nullify the recency blend or invert ordering"
            );
            let recency_boost = hybrid::recency_decay(now_ms, result.session.timestamp);
            let blended_rank = rank * (1.0 + RECENCY_BOOST_WEIGHT * recency_boost);
            (result, blended_rank)
        })
        .collect();
    results.sort_by(|a, b| a.1.total_cmp(&b.1));
    results.truncate(limit);
    results.into_iter().map(|(r, _)| r).collect()
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

fn fetch_chunks(
    conn: &Connection,
    fts_query: Option<&str>,
    ranked: Vec<(String, f64)>,
    meta_map: &mut HashMap<String, SessionData>,
    vec_chunks: &HashMap<String, i64>,
) -> Result<Vec<(f64, SearchResult)>> {
    let rowid_match_sql = "SELECT c.id, MAX(c.content) FROM qa_chunks c \
        JOIN (SELECT rowid FROM messages WHERE messages MATCH ?1 AND session_id = ?2 ORDER BY rank LIMIT 1) m \
          ON m.rowid BETWEEN c.src_rowid_lo AND c.src_rowid_hi \
        WHERE c.session_id = ?2 AND c.src_rowid_lo IS NOT NULL HAVING COUNT(*) = 1";
    let mut rowid_match_stmt = conn.prepare(rowid_match_sql)?;
    let instr_match_sql = "SELECT MAX(c.content) FROM qa_chunks c \
        JOIN (SELECT text FROM messages WHERE messages MATCH ?1 AND session_id = ?2 ORDER BY rank LIMIT 1) m \
          ON instr(c.content, m.text) > 0 \
        WHERE c.session_id = ?2 AND c.src_rowid_lo IS NULL HAVING COUNT(*) = 1";
    let mut instr_match_stmt = conn.prepare(instr_match_sql)?;
    let snippet_sql = "SELECT snippet(messages, 2, '**', '**', '...', 20) \
               FROM messages WHERE messages MATCH ?1 AND session_id = ?2 LIMIT 1";
    let mut snippet_stmt = conn.prepare(snippet_sql)?;
    let vec_chunk_sql = "SELECT content FROM qa_chunks WHERE id = ?1";
    let mut vec_chunk_stmt = conn.prepare(vec_chunk_sql)?;
    Ok(build_candidates(ranked, meta_map, |sid| {
        if let Some(fts_query) = fts_query {
            if let Some((chunk_id, content)) = row_or_default(
                rowid_match_stmt.query_row(rusqlite::params![fts_query, sid], |row| {
                    Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
                }),
                sid,
            ) {
                return (content, Some(chunk_id));
            }
            if let Some(content) = row_or_default(
                instr_match_stmt.query_row(rusqlite::params![fts_query, sid], |row| row.get(0)),
                sid,
            ) {
                return (content, None);
            }
            if let Some(snippet) = row_or_default(
                snippet_stmt.query_row(rusqlite::params![fts_query, sid], |row| row.get(0)),
                sid,
            ) {
                return (snippet, None);
            }
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

fn vec_search(
    conn: &Connection,
    embedder: &dyn Embed,
    query: &str,
    limit: usize,
    opts: &SearchOptions,
    now_ms: i64,
) -> Result<Vec<(String, i64)>> {
    let embedding = embedder.embed_query(query)?;
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
    sql.push_str(" GROUP BY c.session_id ORDER BY d");

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
    let segmented = segment_script_boundaries(query);
    let fts_input = if segmented.split_whitespace().any(|t| t.chars().count() >= 3) {
        segmented
    } else {
        query.to_owned()
    };
    let matched = match prepare_match_query(
        conn,
        &fts_input,
        "messages_vocab",
        &QueryNormalizationConfig::disabled(),
    ) {
        Ok(m) => m,
        Err(SanitizeError::EmptyInput | SanitizeError::NoSearchableTerms) => {
            debug!(%query, "query produced no searchable terms");
            return Ok(SearchOutcome::nominal(Vec::new()));
        }
        Err(e @ (SanitizeError::InvalidVocabTable(_) | SanitizeError::VocabLookupFailed(_))) => {
            return Err(e.into());
        }
    };
    let fts_query = clean_for_trigram(&matched);
    let fts_query = fts_query.as_deref();
    let (fts_ranked, fts_skipped) = match fts_query {
        Some(q) => find_candidate_sessions(conn, opts, q, now_ms)?,
        None => (Vec::new(), 0),
    };
    let incomplete = fts_skipped > 0;

    let run_hybrid = match embedder {
        Some(_) => match has_vec_data(conn) {
            Ok(has_data) => has_data,
            Err(e) => {
                warn!(error = %e, "vec_chunks probe failed, using text search only");
                return fts_only_outcome(
                    conn, fts_query, fts_ranked, opts, now_ms, true, incomplete,
                );
            }
        },
        None => false,
    };

    let Some(embedder) = embedder.filter(|_| run_hybrid) else {
        return fts_only_outcome(conn, fts_query, fts_ranked, opts, now_ms, false, incomplete);
    };

    let candidate_limit = opts.limit * CANDIDATE_LIMIT_MULTIPLIER;

    let (vec_hits, vec_degraded) =
        match vec_search(conn, embedder, query, candidate_limit, opts, now_ms) {
            Ok(hits) => (hits, false),
            Err(e) => {
                warn!(error = %e, "vector search failed, using text search only");
                (Vec::new(), true)
            }
        };

    let vec_chunks: HashMap<String, i64> = vec_hits.iter().cloned().collect();
    let mut merged = rrf_merge_sessions(
        fts_ranked.iter().map(|(sid, _)| sid.as_str()),
        vec_hits.iter().map(|(sid, _)| sid.as_str()),
    );

    if merged.is_empty() {
        return Ok(SearchOutcome {
            results: Vec::new(),
            vec_degraded,
            results_incomplete: incomplete,
        });
    }

    let mut meta_map = fetch_session_metadata(conn, &merged)?;

    let before_prune = merged.len();
    merged.retain(|(sid, _)| meta_map.contains_key(sid));
    let incomplete = incomplete || merged.len() < before_prune;

    if merged.is_empty() {
        return Ok(SearchOutcome {
            results: Vec::new(),
            vec_degraded,
            results_incomplete: incomplete,
        });
    }

    hybrid::apply_recency_boost(
        &mut merged,
        |sid| meta_map.get(sid).and_then(|sd| sd.timestamp),
        now_ms,
        RECENCY_BOOST_WEIGHT,
    );
    merged.truncate(opts.limit);

    let candidates = fetch_chunks(conn, fts_query, merged, &mut meta_map, &vec_chunks)?;
    Ok(SearchOutcome {
        results: candidates.into_iter().map(|(_, r)| r).collect(),
        vec_degraded,
        results_incomplete: incomplete,
    })
}

fn fts_only_outcome(
    conn: &Connection,
    fts_query: Option<&str>,
    fts_ranked: Vec<(String, f64)>,
    opts: &SearchOptions,
    now_ms: i64,
    vec_degraded: bool,
    incomplete: bool,
) -> Result<SearchOutcome> {
    if fts_ranked.is_empty() {
        return Ok(SearchOutcome {
            results: Vec::new(),
            vec_degraded,
            results_incomplete: incomplete,
        });
    }
    let requested = fts_ranked.len();
    let mut meta_map = fetch_session_metadata(conn, &fts_ranked)?;
    let incomplete = incomplete || meta_map.len() < requested;
    let candidates = fetch_chunks(conn, fts_query, fts_ranked, &mut meta_map, &HashMap::new())?;
    Ok(SearchOutcome {
        results: score_and_sort(candidates, now_ms, opts.limit),
        vec_degraded,
        results_incomplete: incomplete,
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
mod tests;
