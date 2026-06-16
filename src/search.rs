use std::collections::HashMap;
use std::slice;
use std::time::{SystemTime, UNIX_EPOCH};

use amici::storage::filter::{
    append_eq_filter, append_like_prefix_filter, append_timestamp_cutoff_filter,
};
use amici::storage::{anon_placeholders, fts::clean_for_trigram};
use anyhow::{Context, Result};
use rurico::embed::Embed;
use rurico::storage::{QueryNormalizationConfig, SanitizeError, prepare_match_query};
use rusqlite::Connection;
use rusqlite::types::ToSql;
use tracing::{debug, warn};

use crate::embedder::f32_as_bytes;
use crate::error::RecallError;
use crate::hybrid::rrf_merge_strings;

use crate::date::MS_PER_DAY;
use crate::hybrid::{self, RECENCY_BOOST_WEIGHT};
use crate::parser::{SessionData, Source};

/// Candidate over-sample factor: each path fetches `limit * CANDIDATE_LIMIT_MULTIPLIER`
/// candidates before the RRF merge and recency boost truncate the union back to
/// `limit`. Shared by the FTS and vec paths so the two stay in lockstep (#54).
const CANDIDATE_LIMIT_MULTIPLIER: usize = 3;

pub struct SearchResult {
    pub session: SessionData,
    /// Excerpt for display: the full containing qa_chunk for an FTS hit, else the
    /// FTS5 20-token snippet (with `**` highlight markers), else the closest vector
    /// chunk, else empty.
    pub excerpt: String,
}

/// How `recall search` treats the session that invoked it.
///
/// Resolved by the CLI layer from `CLAUDE_CODE_SESSION_ID` and the
/// `--{exclude,include,only}-current` flags. The library default is `Ignore`, so
/// existing search behavior is unchanged unless a caller opts in — the env-based
/// exclude policy lives only in the CLI (`resolve_current_session`), never here.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum CurrentSession {
    /// No current-session filtering.
    #[default]
    Ignore,
    /// Drop results whose session id equals this value (the invoking session).
    Exclude(String),
    /// Return only results whose session id equals this value.
    Only(String),
}

pub struct SearchOptions {
    pub project: Option<String>,
    pub days: Option<i64>,
    pub source: Option<Source>,
    pub limit: usize,
    /// How to treat the invoking session (see `CurrentSession`).
    pub current: CurrentSession,
    /// Include sessions tagged automated. Default false: automated sessions
    /// (hook/script/agent-generated) are excluded from results (#24).
    pub include_automated: bool,
    /// Override current time (epoch ms) for deterministic testing.
    pub now_ms: Option<i64>,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            project: None,
            days: None,
            source: None,
            limit: 10,
            current: CurrentSession::Ignore,
            include_automated: false,
            now_ms: None,
        }
    }
}

/// Append ` AND {column} IN (...)` to `sql` and push matching params.
///
/// `column` lets FTS pass `"session_id"` and vec pass `"c.session_id"` — both
/// paths emit identical filter SQL, keeping yomu #103's single-source strategy.
/// `&'static str` restricts `column` to compile-time literals so runtime input
/// cannot reach the SQL string (matches amici filter helper convention).
///
/// Precondition: `sql` ends inside an open WHERE clause — callers anchor with
/// `MATCH ?` (FTS path) or a trailing `WHERE 1 = 1` (vec path).
fn append_session_filter(
    sql: &mut String,
    params: &mut Vec<Box<dyn ToSql>>,
    column: &'static str,
    opts: &SearchOptions,
    now_ms: i64,
) {
    if opts.project.is_none() && opts.days.is_none() && opts.source.is_none() {
        return;
    }
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

/// Append ` AND {column} <> ?` (Exclude) or ` AND {column} = ?` (Only) for the
/// invoking session; `Ignore` emits nothing. Called unconditionally on both the
/// FTS and vec candidate queries so an excluded session cannot slip back in via
/// the path that skipped filtering — the RRF merge re-unions both candidate sets,
/// so a single-path filter would leak. Precondition matches
/// `append_session_filter`: `sql` ends inside an open WHERE clause.
fn append_current_session_filter(
    sql: &mut String,
    params: &mut Vec<Box<dyn ToSql>>,
    column: &'static str,
    current: &CurrentSession,
) {
    let (op, id) = match current {
        CurrentSession::Ignore => return,
        CurrentSession::Exclude(id) => (" <> ?", id),
        CurrentSession::Only(id) => (" = ?", id),
    };
    sql.push_str(" AND ");
    sql.push_str(column);
    sql.push_str(op);
    params.push(Box::new(id.clone()));
}

/// Append a NULL-safe automated-session exclusion. `session_type` lives on
/// `sessions` (not on the messages / qa_chunks candidate tables), so the filter is
/// a subquery on session_id. `COALESCE` treats a NULL / unclassified session_type
/// as interactive, so sessions indexed before classification are never hidden by
/// default. `include_automated` skips it. Called on both candidate paths — the RRF
/// merge re-unions both sets, so a single-path filter would leak — matching
/// `append_current_session_filter`. Emits no `?` placeholders (the labels are
/// compile-time literals), so callers' param counts are unaffected.
fn append_automated_filter(sql: &mut String, column: &'static str, include_automated: bool) {
    if include_automated {
        return;
    }
    sql.push_str(" AND ");
    sql.push_str(column);
    sql.push_str(
        " IN (SELECT s3.session_id FROM sessions s3 \
         WHERE COALESCE(s3.session_type, 'interactive') <> 'automated')",
    );
}

fn build_fts_candidate_query(
    fts_query: &str,
    opts: &SearchOptions,
    now_ms: i64,
) -> (String, Vec<Box<dyn ToSql>>) {
    let mut sql =
        "SELECT session_id, MIN(rank) as best_rank FROM messages WHERE messages MATCH ?".to_owned();
    let mut params: Vec<Box<dyn ToSql>> = vec![Box::new(fts_query.to_owned())];
    append_session_filter(&mut sql, &mut params, "session_id", opts, now_ms);
    append_current_session_filter(&mut sql, &mut params, "session_id", &opts.current);
    append_automated_filter(&mut sql, "session_id", opts.include_automated);
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
) -> Result<Vec<(String, f64)>> {
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
    Ok(ranked)
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
    let rows = stmt.query_map(params, |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, String>(3)?,
            row.get::<_, String>(4)?,
            row.get::<_, Option<i64>>(5)?,
        ))
    })?;

    let mut map = HashMap::new();
    let mut db_errors = 0;
    for r in rows {
        match r {
            Ok((session_id, source_str, file_path, project, slug, timestamp)) => {
                let source = match Source::from_db(&source_str) {
                    Some(s) => s,
                    None => {
                        warn!(source = %source_str, session_id, "unknown source");
                        continue;
                    }
                };
                map.insert(
                    session_id.clone(),
                    SessionData {
                        session_id,
                        source,
                        file_path,
                        project,
                        slug,
                        timestamp,
                    },
                );
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
    Ok(map)
}

fn build_candidates(
    ranked: Vec<(String, f64)>,
    meta_map: &mut HashMap<String, SessionData>,
    mut get_snippet: impl FnMut(&str) -> String,
) -> Vec<(f64, SearchResult)> {
    ranked
        .into_iter()
        .filter_map(|(session_id, rank)| {
            let session = meta_map.remove(&session_id)?;
            let snippet = get_snippet(&session_id);
            Some((
                rank,
                SearchResult {
                    session,
                    excerpt: snippet,
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
            // bm25 rank is negative (lower = better), so the (1 + w*boost)
            // multiplier amplifies magnitude and recent sessions sort earlier
            // (ascending). Intentionally asymmetric with the hybrid path
            // (apply_recency_boost: positive RRF scores, descending). A rank of
            // exactly 0.0 would nullify the blend but is unreachable with
            // trigram FTS (#119: needs a trigram in exactly N/2 of all rows;
            // verified absent on the live index).
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

fn snippet_or_default(result: rusqlite::Result<String>, session_id: &str) -> Option<String> {
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
    // Full chunk for an FTS hit (#8, #192): find the message that matched, then the
    // qa_chunk it was generated from. The primary path links by the source message
    // rowid range stored on the chunk (#192): the FTS top-hit's rowid is resolved to
    // its owning chunk via `m.rowid BETWEEN c.src_rowid_lo AND c.src_rowid_hi`. Ranges
    // are disjoint per group, so a short hit (e.g. "yes") that the old instr substring
    // guess collided with now lands on its true home. `ORDER BY rank LIMIT 1` picks the
    // most relevant matched message; scoped to the session (idx_qa_chunks_session).
    // Uniqueness guard (#118): `HAVING COUNT(*) = 1` keeps split-siblings (which share
    // one range → COUNT=N) collapsing to the snippet instead of an arbitrary sibling.
    // Restricted to `src_rowid_lo IS NOT NULL` so only linked (current or backfilled)
    // chunks take this path; legacy NULL rows fall through to the instr guess below.
    let rowid_match_sql = "SELECT MAX(c.content) FROM qa_chunks c \
        JOIN (SELECT rowid FROM messages WHERE messages MATCH ?1 AND session_id = ?2 ORDER BY rank LIMIT 1) m \
          ON m.rowid BETWEEN c.src_rowid_lo AND c.src_rowid_hi \
        WHERE c.session_id = ?2 AND c.src_rowid_lo IS NOT NULL HAVING COUNT(*) = 1";
    let mut rowid_match_stmt = conn.prepare(rowid_match_sql)?;
    // Fallback for legacy chunks indexed before #192 (src_rowid_lo NULL) that a
    // count-mismatched session left unbackfilled: the pre-#192 instr substring guess.
    // `instr` runs on the already-matched message text (not the raw query), so FTS5
    // operators (AND/OR/quotes) are resolved by MATCH and never reach `instr`.
    // `HAVING COUNT(*) = 1` keeps the same uniqueness guard; the `IS NULL` scope means
    // that in a fully-linked DB the `instr` join matches no chunk and returns no row.
    // (The inner MATCH subquery still runs on a rowid-tier miss before the outer
    // `IS NULL` prunes it; harmless on this single-user path, and the instr tier is
    // skipped entirely whenever the rowid tier already resolved the hit.)
    let instr_match_sql = "SELECT MAX(c.content) FROM qa_chunks c \
        JOIN (SELECT text FROM messages WHERE messages MATCH ?1 AND session_id = ?2 ORDER BY rank LIMIT 1) m \
          ON instr(c.content, m.text) > 0 \
        WHERE c.session_id = ?2 AND c.src_rowid_lo IS NULL HAVING COUNT(*) = 1";
    let mut instr_match_stmt = conn.prepare(instr_match_sql)?;
    // Fallback when no single chunk contains the matched message (e.g. the message
    // was split across chunks, or the session has no qa_chunks): the 20-token snippet.
    let snippet_sql = "SELECT snippet(messages, 2, '**', '**', '...', 20) \
               FROM messages WHERE messages MATCH ?1 AND session_id = ?2 LIMIT 1";
    let mut snippet_stmt = conn.prepare(snippet_sql)?;
    // Vector-only hit: no FTS match exists, so use the closest chunk by id (#36).
    let vec_chunk_sql = "SELECT content FROM qa_chunks WHERE id = ?1";
    let mut vec_chunk_stmt = conn.prepare(vec_chunk_sql)?;
    Ok(build_candidates(ranked, meta_map, |sid| {
        // No trigram-indexable FTS query (#164): a short-only query reaches here on
        // the vector path. Skip the three MATCH-based snippet tiers (MATCH "" errors
        // and logs a warning per session) and use the vec-chunk content directly.
        if let Some(fts_query) = fts_query {
            if let Some(content) = snippet_or_default(
                rowid_match_stmt.query_row(rusqlite::params![fts_query, sid], |row| row.get(0)),
                sid,
            ) {
                return content;
            }
            if let Some(content) = snippet_or_default(
                instr_match_stmt.query_row(rusqlite::params![fts_query, sid], |row| row.get(0)),
                sid,
            ) {
                return content;
            }
            if let Some(snippet) = snippet_or_default(
                snippet_stmt.query_row(rusqlite::params![fts_query, sid], |row| row.get(0)),
                sid,
            ) {
                return snippet;
            }
        }
        if let Some(&chunk_id) = vec_chunks.get(sid)
            && let Some(content) = snippet_or_default(
                vec_chunk_stmt.query_row(rusqlite::params![chunk_id], |row| row.get(0)),
                sid,
            )
        {
            return content;
        }
        String::new()
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

fn has_vec_data(conn: &Connection) -> bool {
    conn.query_row("SELECT 1 FROM vec_chunks LIMIT 1", [], |_| Ok(true))
        .unwrap_or(false)
}

fn vec_search(
    conn: &Connection,
    embedder: &dyn Embed,
    query: &str,
    limit: usize,
    opts: &SearchOptions,
    now_ms: i64,
) -> Result<Vec<(String, i64)>> {
    let embedding = embedder
        .embed_query(query)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let embedding_bytes: &[u8] = f32_as_bytes(&embedding);

    // Subquery pushes the knn LIMIT down to the vec0 virtual table.
    // GROUP BY deduplicates sessions; ORDER BY MIN(distance) preserves rank.
    // The explicit MIN(v.distance) in the SELECT binds the bare `c.id` to the
    // MIN-distance row (SQLite single-min/max rule) — the chunk closest to the
    // query, used as the excerpt source for vector-only hits (no FTS snippet).
    // `WHERE 1 = 1` anchors the optional session_filter without branching on
    // its presence; SQLite folds the constant at plan time.
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
    append_session_filter(&mut sql, &mut filter_params, "c.session_id", opts, now_ms);
    append_current_session_filter(&mut sql, &mut filter_params, "c.session_id", &opts.current);
    append_automated_filter(&mut sql, "c.session_id", opts.include_automated);
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

/// Script class of a character for trigram-boundary segmentation (#167).
/// ASCII letters and digits share one class so technical tokens (`utf8`, `v2`,
/// `sha256`) are never split at a letter↔digit transition.
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
        // CJK ideographs (incl. ext A / compatibility) and the 々 iteration mark.
        '\u{3005}'
        | '\u{3400}'..='\u{4DBF}'
        | '\u{4E00}'..='\u{9FFF}'
        | '\u{F900}'..='\u{FAFF}' => Script::Han,
        _ => Script::Other,
    }
}

/// Insert a space at every script-run transition so a no-space natural-language
/// query reaches the same sessions a space-separated query would (#167).
///
/// In degraded (FTS-only) mode the trigram tokenizer matches a no-space query as
/// a single phrase requiring a contiguous substring; a query whose 助詞 were
/// dropped (`textlint日本語ドキュメント校正`) never matches indexed text that
/// keeps them (`textlintで日本語ドキュメントを校正する`). Splitting at the
/// latin↔han↔katakana↔hiragana transitions produces the implicit-AND of
/// per-word phrases — byte-equivalent to the spaced query the user could have
/// typed — so the no-space form inherits the spaced form's recall.
///
/// Only transitions among {Latin, Han, Hiragana, Katakana} split. A pure
/// single-script run (e.g. a kanji compound with no transition) is left intact:
/// splitting it requires a dictionary, which is out of scope. Text inside double
/// quotes is preserved verbatim to keep an explicit phrase-match intent.
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
    search_with_embedder(conn, query, opts, None)
}

pub fn search_with_embedder(
    conn: &Connection,
    query: &str,
    opts: &SearchOptions,
    embedder: Option<&dyn Embed>,
) -> Result<Vec<SearchResult>> {
    if opts.limit == 0 {
        return Ok(Vec::new());
    }

    let now_ms = resolve_now_ms(opts)?;
    // Normalization disabled to stay symmetric with the index: the indexer writes
    // raw message text into FTS (no normalize_for_fts pass), so query-side NFKC /
    // lowercase / whitespace folding would silently miss full-width and CJK matches.
    // Adopting normalization means normalizing both sides and re-indexing (#51).
    // Split a no-space CJK query at script boundaries so the trigram FTS path
    // sees the same per-word tokens a spaced query would (#167). The raw `query`
    // still feeds vec_search below — embedding wants the natural text.
    //
    // Only adopt the split when it yields an anchor term (≥3 chars). Without one,
    // an all-short split (e.g. `校正する` → `校正 する`) vocab-expands into two
    // OR-groups whose product can exceed amici's MAX_COMBOS, dropping to
    // `fixed_only()` with no fixed term → no results — a regression over the raw
    // contiguous-phrase match. An anchor term survives as a `fixed` phrase, so
    // segmenting is then a strict recall superset of the raw query.
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
            return Ok(Vec::new());
        }
        // A bad vocab-table name or a failed SQLite vocab lookup is a real
        // fault, not an empty query — surface it instead of returning no hits.
        Err(e @ (SanitizeError::InvalidVocabTable(_) | SanitizeError::VocabLookupFailed(_))) => {
            return Err(e.into());
        }
    };
    // A short query (e.g. "UI") whose terms are all <3 chars yields no trigram-
    // indexable FTS query. That must not skip vector search when embeddings exist
    // (#164): fall through with no FTS candidates and let the vec path answer by
    // meaning. The empty/whitespace-only case already returned above (NoSearchableTerms).
    let fts_query = clean_for_trigram(&matched);
    let fts_query = fts_query.as_deref();
    let fts_ranked = match fts_query {
        Some(q) => find_candidate_sessions(conn, opts, q, now_ms)?,
        None => Vec::new(),
    };

    if let Some(embedder) = embedder
        && has_vec_data(conn)
    {
        let candidate_limit = opts.limit * CANDIDATE_LIMIT_MULTIPLIER;

        let fts_hits: Vec<(String, f64)> = fts_ranked
            .iter()
            .map(|(sid, _)| (sid.clone(), 0.0))
            .collect();

        let vec_hits = match vec_search(conn, embedder, query, candidate_limit, opts, now_ms) {
            Ok(hits) => hits,
            Err(e) => {
                warn!(error = %e, "vector search failed, using text search only");
                Vec::new()
            }
        };

        // Map each vector-hit session to its closest chunk so fetch_chunks can use
        // that chunk's content as the excerpt when there is no FTS snippet (#36).
        let vec_chunks: HashMap<String, i64> = vec_hits.iter().cloned().collect();
        let vec_for_rrf: Vec<(String, f64)> =
            vec_hits.iter().map(|(sid, _)| (sid.clone(), 0.0)).collect();

        let mut merged = rrf_merge_strings(&fts_hits, &vec_for_rrf);

        if merged.is_empty() {
            return Ok(Vec::new());
        }

        let mut meta_map = fetch_session_metadata(conn, &merged)?;

        // Drop hits whose `sessions` row is missing or has an unknown source;
        // `fetch_session_metadata` skips those, and without pruning here they
        // would consume `limit` slots during truncate and cause `fetch_chunks`
        // to under-fill the result set.
        merged.retain(|(sid, _)| meta_map.contains_key(sid));

        if merged.is_empty() {
            return Ok(Vec::new());
        }

        hybrid::apply_recency_boost(
            &mut merged,
            |sid| meta_map.get(sid).and_then(|sd| sd.timestamp),
            now_ms,
            RECENCY_BOOST_WEIGHT,
        );
        merged.truncate(opts.limit);

        let candidates = fetch_chunks(conn, fts_query, merged, &mut meta_map, &vec_chunks)?;
        Ok(candidates.into_iter().map(|(_, r)| r).collect())
    } else {
        if fts_ranked.is_empty() {
            return Ok(Vec::new());
        }
        let mut meta_map = fetch_session_metadata(conn, &fts_ranked)?;
        // FTS-only path: every session matched FTS, so there is always a snippet;
        // pass an empty vec-chunk map (no vector-only fallback needed here).
        let candidates = fetch_chunks(conn, fts_query, fts_ranked, &mut meta_map, &HashMap::new())?;
        Ok(score_and_sort(candidates, now_ms, opts.limit))
    }
}

#[cfg(test)]
mod tests;
