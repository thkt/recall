use std::collections::HashMap;
use std::slice;
use std::time::{SystemTime, UNIX_EPOCH};

use amici::storage::filter::{
    append_eq_filter, append_like_prefix_filter, append_timestamp_cutoff_filter, escape_like,
};
use amici::storage::{anon_placeholders, fts::clean_for_trigram};
use anyhow::{Context, Result};
use rurico::embed::Embed;
use rurico::storage::{QueryNormalizationConfig, SanitizeError, prepare_match_query};
use rusqlite::Connection;
use rusqlite::OptionalExtension;
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
    /// The `qa_chunks.id` the excerpt resolved to, for a chunk-backed hit (FTS
    /// rowid-tier or vector-tier). `None` for a snippet-tier hit, whose excerpt is an
    /// fts5 snippet with no backing chunk; the caller then guides to the whole
    /// conversation (`recall show <id>`) rather than a `--chunk` drill-down (#282).
    pub chunk_id: Option<i64>,
}

/// The ranked hits plus two independent degradation signals the caller forwards
/// to the `--json` envelope so a `degraded:false` is never reported over an
/// internally-truncated result (#204, #249).
///
/// `vec_degraded` covers the semantic (vector) leg silently falling back to
/// FTS-only: a loaded embedder can still fail at query time (e.g. `embed_query`
/// under memory pressure) or the `vec_chunks` probe can error, and `vec_search`
/// swallows the former into a text-only result. It is the runtime counterpart to
/// the load-time degradation derived in `search_degraded_state`.
///
/// `results_incomplete` covers the FTS leg silently dropping matched sessions:
/// a candidate row that fails to deserialize (`find_candidate_sessions`) or a
/// matched session pruned because its `sessions` metadata is missing or has an
/// unknown source (`fetch_session_metadata` retain / `build_candidates`). Either
/// thins the result below the true match set, so the envelope must flag it rather
/// than present the partial set as complete (#249 RC-002/RC-003).
pub struct SearchOutcome {
    pub results: Vec<SearchResult>,
    pub vec_degraded: bool,
    pub results_incomplete: bool,
}

impl SearchOutcome {
    /// A search that never touched a degraded vector path nor dropped a matched
    /// session: the empty-query, limit-zero, and FTS-only branches with complete
    /// coverage.
    fn nominal(results: Vec<SearchResult>) -> Self {
        Self {
            results,
            vec_degraded: false,
            results_incomplete: false,
        }
    }
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
    /// Restrict to sessions that touched this file path. Matched against
    /// `session_files.path` by exact equality or a `%/<suffix>` LIKE, so a
    /// basename or tail sub-path resolves an absolute stored path (#283).
    pub file: Option<String>,
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
            file: None,
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
    // File axis (#283): restrict to sessions that touched the path, joined from
    // `session_files` (Codex sessions have no rows here, so `--file` excludes
    // them while Codex extraction is unsupported). Exact equality catches an
    // absolute path; the `%/<suffix>` LIKE catches a basename or tail sub-path.
    // The suffix is `escape_like`-escaped so LIKE metacharacters in the input
    // (e.g. `_`) match literally under `ESCAPE '\'`.
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

/// FTS candidate sessions paired with the count of candidate rows that failed to
/// deserialize. A non-zero `skipped` means the returned ranking is a strict subset
/// of the true FTS matches, which the caller threads into
/// `SearchOutcome::results_incomplete` so the envelope never reports a thinned
/// result as complete (#249 RC-002). `skipped` is only non-zero on a corrupt row;
/// the all-rows-failed case still hard-errors below.
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

/// Decode one `sessions` row shaped (session_id, source, file_path, project,
/// slug, timestamp) into `SessionData`. `None` on an unrecognized `source`
/// string, with a warning — the shared row mapping for every reader of this
/// projection (`fetch_session_metadata`, `file_list_outcome`).
fn session_data_from_row(
    session_id: String,
    source_str: &str,
    file_path: String,
    project: String,
    slug: String,
    timestamp: Option<i64>,
) -> Option<SessionData> {
    let Some(source) = Source::from_db(source_str) else {
        warn!(source = %source_str, session_id, "unknown source");
        return None;
    };
    Some(SessionData {
        session_id,
        source,
        file_path,
        project,
        slug,
        timestamp,
    })
}

/// The `(session_id, source, file_path, project, slug, timestamp)` projection
/// shared by `fetch_session_metadata` and `file_list_outcome`. Positional
/// decode, so keeping it in one place makes a column-order drift break both
/// readers at once instead of silently diverging.
type SessionRow = (String, String, String, String, String, Option<i64>);

fn decode_session_row(row: &rusqlite::Row) -> rusqlite::Result<SessionRow> {
    Ok((
        row.get(0)?,
        row.get(1)?,
        row.get(2)?,
        row.get(3)?,
        row.get(4)?,
        row.get(5)?,
    ))
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
    let rows = stmt.query_map(params, decode_session_row)?;

    let mut map = HashMap::new();
    let mut db_errors = 0;
    for r in rows {
        match r {
            Ok((session_id, source_str, file_path, project, slug, timestamp)) => {
                if let Some(session) = session_data_from_row(
                    session_id,
                    &source_str,
                    file_path,
                    project,
                    slug,
                    timestamp,
                ) {
                    map.insert(session.session_id.clone(), session);
                }
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

/// Map a per-session excerpt lookup to `Option`: a hit is `Some`, a
/// `QueryReturnedNoRows` (no matching chunk/snippet in this tier) is a quiet `None`
/// so the next tier runs, and any other SQLite error is logged and treated as a
/// miss. Generic over the row shape so the rowid tier can return `(id, content)`
/// while the other tiers return the content string alone.
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
    // `HAVING COUNT(*) = 1` forces the group to a single chunk, so the bare `c.id`
    // is unambiguous alongside `MAX(c.content)` (it is that one row's id) — the
    // drill-down anchor the excerpt resolved to (#282).
    let rowid_match_sql = "SELECT c.id, MAX(c.content) FROM qa_chunks c \
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
            // rowid tier resolves the exact backing chunk, so return its id as the
            // drill-down anchor.
            if let Some((chunk_id, content)) = row_or_default(
                rowid_match_stmt.query_row(rusqlite::params![fts_query, sid], |row| {
                    Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
                }),
                sid,
            ) {
                return (content, Some(chunk_id));
            }
            // instr tier (legacy NULL-src_rowid chunk) and snippet tier carry no
            // stable chunk anchor, so the caller guides to the full-conversation show.
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
        // Vector-only hit: the closest chunk id came from the knn ranking
        // (`vec_chunks`), so it is the drill-down anchor directly.
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

/// Whether the `vec_chunks` table holds at least one embedding.
///
/// Three outcomes the caller must tell apart (#249 RC-001): `Ok(true)` has data
/// (run the vector leg), `Ok(false)` is an empty table — the normal indexed-but-
/// not-yet-embedded state, never a degradation — and `Err` is a real probe fault
/// (e.g. a corrupt or missing virtual table) that silently bypasses the vector
/// leg. `.optional()` maps `QueryReturnedNoRows` (empty table) to `None`, so only
/// a genuine SQLite error reaches `Err` and flips `vec_degraded`.
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
    // The FTS-only path (no embedder) can never be vector-degraded, so the
    // signal is discarded here; only the CLI envelope path needs it.
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

    let now_ms = resolve_now_ms(opts)?;

    // Query-less mode (#284/U-005): with no query, `--file` alone lists the
    // sessions that touched the path, newest first, via a dedicated SELECT that
    // bypasses the FTS/vector MATCH path. `Some("")` (an explicit empty query) is
    // distinct — it flows on to the MATCH path and its NoSearchableTerms handling.
    let Some(query) = query.into() else {
        return file_list_outcome(conn, opts, now_ms);
    };
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
            return Ok(SearchOutcome::nominal(Vec::new()));
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
    let (fts_ranked, fts_skipped) = match fts_query {
        Some(q) => find_candidate_sessions(conn, opts, q, now_ms)?,
        None => (Vec::new(), 0),
    };
    // A skipped FTS candidate row thins the match set; carry it so a partial
    // result is never reported as complete (#249 RC-002).
    let incomplete = fts_skipped > 0;

    // Decide whether the vector leg runs. The probe has three outcomes (#249
    // RC-001): has data -> hybrid; empty table -> FTS-only by design (not a
    // degradation); probe error -> FTS-only but flag `vec_degraded`, since the
    // semantic leg is silently unavailable.
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
        // No embedder or an empty vector table: the FTS-only path by design, not a
        // runtime degradation, so `vec_degraded` stays false.
        return fts_only_outcome(conn, fts_query, fts_ranked, opts, now_ms, false, incomplete);
    };

    let candidate_limit = opts.limit * CANDIDATE_LIMIT_MULTIPLIER;

    let fts_hits: Vec<(String, f64)> = fts_ranked
        .iter()
        .map(|(sid, _)| (sid.clone(), 0.0))
        .collect();

    // A runtime `vec_search` failure (e.g. `embed_query` under memory
    // pressure) silently drops semantic coverage. Record it so the caller can
    // mark the envelope `degraded:true` instead of returning a thinned or
    // empty result as if it were complete (#204).
    let (vec_hits, vec_degraded) =
        match vec_search(conn, embedder, query, candidate_limit, opts, now_ms) {
            Ok(hits) => (hits, false),
            Err(e) => {
                warn!(error = %e, "vector search failed, using text search only");
                (Vec::new(), true)
            }
        };

    // Map each vector-hit session to its closest chunk so fetch_chunks can use
    // that chunk's content as the excerpt when there is no FTS snippet (#36).
    let vec_chunks: HashMap<String, i64> = vec_hits.iter().cloned().collect();
    let vec_for_rrf: Vec<(String, f64)> =
        vec_hits.iter().map(|(sid, _)| (sid.clone(), 0.0)).collect();

    let mut merged = rrf_merge_strings(&fts_hits, &vec_for_rrf);

    if merged.is_empty() {
        return Ok(SearchOutcome {
            results: Vec::new(),
            vec_degraded,
            results_incomplete: incomplete,
        });
    }

    let mut meta_map = fetch_session_metadata(conn, &merged)?;

    // Drop hits whose `sessions` row is missing or has an unknown source;
    // `fetch_session_metadata` skips those, and without pruning here they
    // would consume `limit` slots during truncate and cause `fetch_chunks`
    // to under-fill the result set. A prune thins the match set, so flag it (#249
    // RC-003).
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

/// The FTS-only outcome shared by the no-embedder / empty-vector path and the
/// probe-error fallback (#249 RC-001). `vec_degraded` is set by the probe-error
/// caller (the semantic leg was silently unavailable) and stays false on the
/// by-design path. A matched session pruned by `fetch_session_metadata` (missing
/// row or unknown source) thins the set, so it folds into `results_incomplete`
/// alongside any FTS skip the caller already counted (#249 RC-003).
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
    // `fts_ranked` is GROUP BY session_id (unique ids), and `meta_map` keys are a
    // subset, so a smaller map means `build_candidates` will silently drop the
    // missing sessions.
    let incomplete = incomplete || meta_map.len() < requested;
    // FTS-only path: every session matched FTS, so there is always a snippet;
    // pass an empty vec-chunk map (no vector-only fallback needed here).
    let candidates = fetch_chunks(conn, fts_query, fts_ranked, &mut meta_map, &HashMap::new())?;
    Ok(SearchOutcome {
        results: score_and_sort(candidates, now_ms, opts.limit),
        vec_degraded,
        results_incomplete: incomplete,
    })
}

/// Query-less `--file` list mode (#284/U-005): with no query, list the sessions
/// that touched `opts.file`, newest first, via a dedicated SELECT that bypasses
/// the FTS/vector MATCH path entirely. Reuses the same session/current/automated
/// filters as the search paths, so `--project` / `--days` / `--source` and the
/// current/automated policies narrow the list identically to a query search. Each
/// result carries an empty excerpt (no MATCH hit to resolve a snippet or chunk
/// from) and no chunk anchor. A query-less call with no `--file` is a usage error:
/// there is no term and no path, so nothing to list.
///
/// Row decoding follows the same contract as the MATCH paths (#249 RC-003): a
/// row that fails to decode (NULL in a nullable `sessions` column) or carries an
/// unknown `source` is dropped and counted, folding into `results_incomplete`
/// instead of aborting the command or presenting the thinned list as complete.
/// The limit is applied in Rust after the drops, so a dropped row never
/// under-fills the requested count while later matches exist.
fn file_list_outcome(
    conn: &Connection,
    opts: &SearchOptions,
    now_ms: i64,
) -> Result<SearchOutcome> {
    if opts.file.is_none() {
        return Err(RecallError::Usage("A search query or --file is required.".to_owned()).into());
    }

    // `append_session_filter` appends the `--file` subquery (exact path or
    // `%/<suffix>` LIKE) plus any project/days/source filter; with `--file` set it
    // never early-returns. The subquery aliases (s2/s3/sf) do not collide with the
    // outer `s`.
    let mut sql = "SELECT s.session_id, s.source, s.file_path, s.project, s.slug, s.timestamp \
                   FROM sessions s WHERE 1 = 1"
        .to_owned();
    let mut params: Vec<Box<dyn ToSql>> = Vec::new();
    append_session_filter(&mut sql, &mut params, "s.session_id", opts, now_ms);
    append_current_session_filter(&mut sql, &mut params, "s.session_id", &opts.current);
    append_automated_filter(&mut sql, "s.session_id", opts.include_automated);
    // Pure timestamp DESC (newest first), not the recency-boosted bm25 blend the
    // MATCH paths use: there is no relevance score to blend here. NULL timestamps
    // sort last under SQLite's DESC ordering. No SQL LIMIT: the cap is applied in
    // Rust below so a dropped row never under-fills the requested count.
    sql.push_str(" ORDER BY s.timestamp DESC");
    debug_assert_eq!(
        params.len(),
        sql.matches('?').count(),
        "param count must match SQL placeholder count"
    );

    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(
        rusqlite::params_from_iter(params.iter()),
        decode_session_row,
    )?;

    let mut results = Vec::new();
    let mut skipped = 0;
    let mut first_error = None;
    for r in rows {
        if results.len() >= opts.limit {
            break;
        }
        match r {
            Ok((session_id, source_str, file_path, project, slug, timestamp)) => {
                match session_data_from_row(
                    session_id,
                    &source_str,
                    file_path,
                    project,
                    slug,
                    timestamp,
                ) {
                    // `session_data_from_row` already warned on the unknown source.
                    Some(session) => results.push(SearchResult {
                        session,
                        excerpt: String::new(),
                        chunk_id: None,
                    }),
                    None => skipped += 1,
                }
            }
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
