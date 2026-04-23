use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use amici::storage::{anon_placeholders, fts::clean_for_trigram};
use anyhow::{Context, Result};
use rurico::embed::Embed;
use rurico::storage::{SanitizeError, f32_as_bytes, prepare_match_query, rrf_merge};
use rusqlite::Connection;
use rusqlite::types::{ToSql, ToSqlOutput};
use tracing::{debug, warn};

use crate::date::MS_PER_DAY;
use crate::hybrid::{self, RECENCY_BOOST_WEIGHT};
use crate::parser::{SessionData, Source};

pub struct SearchResult {
    pub session: SessionData,
    /// Snippet with `**` highlight markers from FTS5.
    pub excerpt: String,
}

pub struct SearchOptions {
    pub project: Option<String>,
    pub days: Option<i64>,
    pub source: Option<Source>,
    pub limit: usize,
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
            now_ms: None,
        }
    }
}

enum Param {
    Str(String),
    Int(i64),
}

impl ToSql for Param {
    fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
        match self {
            Param::Str(s) => s.to_sql(),
            Param::Int(i) => i.to_sql(),
        }
    }
}

/// Escape LIKE metacharacters (`%`, `_`, `\`) for use with `ESCAPE '\'`.
pub(crate) fn escape_like(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('%', "\\%")
        .replace('_', "\\_")
}

/// Append ` AND {column} IN (...)` to `sql` and push matching params.
///
/// `column` lets FTS pass `"session_id"` and vec pass `"c.session_id"` — both
/// paths emit identical filter SQL, keeping yomu #103's single-source strategy.
/// `&'static str` restricts `column` to compile-time literals so runtime input
/// cannot reach the SQL string (matches amici filter helper convention).
fn append_session_filter(
    sql: &mut String,
    params: &mut Vec<Param>,
    column: &'static str,
    opts: &SearchOptions,
    now_ms: i64,
) {
    let mut conditions = Vec::new();
    if let Some(ref project) = opts.project {
        let escaped = escape_like(project);
        conditions.push("s2.project LIKE ? || '%' ESCAPE '\\'".to_owned());
        params.push(Param::Str(escaped));
    }
    if let Some(days) = opts.days {
        let cutoff = now_ms - days * MS_PER_DAY;
        conditions.push("s2.timestamp >= ?".to_owned());
        params.push(Param::Int(cutoff));
    }
    if let Some(source) = opts.source {
        conditions.push("s2.source = ?".to_owned());
        params.push(Param::Str(source.as_str().to_owned()));
    }
    if conditions.is_empty() {
        return;
    }
    sql.push_str(&format!(
        " AND {column} IN (SELECT s2.session_id FROM sessions s2 WHERE {})",
        conditions.join(" AND ")
    ));
}

fn build_fts_candidate_query(
    fts_query: &str,
    opts: &SearchOptions,
    now_ms: i64,
) -> (String, Vec<Param>) {
    let mut sql =
        "SELECT session_id, MIN(rank) as best_rank FROM messages WHERE messages MATCH ?".to_owned();
    let mut params = vec![Param::Str(fts_query.to_owned())];
    append_session_filter(&mut sql, &mut params, "session_id", opts, now_ms);
    sql.push_str(" GROUP BY session_id ORDER BY best_rank LIMIT ?");
    let candidate_limit = opts.limit * 3;
    params.push(Param::Int(candidate_limit as i64));
    (sql, params)
}

fn find_candidate_sessions(
    conn: &Connection,
    opts: &SearchOptions,
    fts_query: &str,
    now_ms: i64,
) -> Result<Vec<(String, f64)>> {
    let (sql, params) = build_fts_candidate_query(fts_query, opts, now_ms);
    let refs: Vec<&dyn ToSql> = params.iter().map(|p| p as &dyn ToSql).collect();
    debug_assert_eq!(
        refs.len(),
        sql.matches('?').count(),
        "param count must match SQL placeholder count"
    );
    let mut stmt = conn
        .prepare(&sql)
        .context("Failed to prepare search query")?;
    let rows = stmt
        .query_map(refs.as_slice(), |row| {
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
            anyhow::bail!(
                "Invalid search query: {err_msg}. Use words, \"quoted phrases\", or AND/OR/NOT operators."
            );
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
    let refs: Vec<&dyn ToSql> = ranked.iter().map(|(sid, _)| sid as &dyn ToSql).collect();

    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(refs.as_slice(), |row| {
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
            let recency_boost = hybrid::recency_decay(now_ms, result.session.timestamp);
            let blended_rank = if rank == 0.0 {
                -recency_boost
            } else {
                rank * (1.0 + RECENCY_BOOST_WEIGHT * recency_boost)
            };
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

fn fetch_snippets(
    conn: &Connection,
    fts_query: &str,
    ranked: Vec<(String, f64)>,
    meta_map: &mut HashMap<String, SessionData>,
) -> Result<Vec<(f64, SearchResult)>> {
    let sql = "SELECT snippet(messages, 2, '**', '**', '...', 20) \
               FROM messages WHERE messages MATCH ?1 AND session_id = ?2 LIMIT 1";
    let mut stmt = conn.prepare(sql)?;
    Ok(build_candidates(ranked, meta_map, |sid| {
        snippet_or_default(
            stmt.query_row(rusqlite::params![fts_query, sid], |row| row.get(0)),
            sid,
        )
        .unwrap_or_default()
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
) -> Result<Vec<(String, f64)>> {
    let embedding = embedder
        .embed_query(query)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let embedding_bytes: &[u8] = f32_as_bytes(&embedding);

    // Subquery pushes the knn LIMIT down to the vec0 virtual table.
    // GROUP BY deduplicates sessions; ORDER BY MIN(distance) preserves rank.
    // `WHERE 1 = 1` anchors the optional session_filter without branching on
    // its presence; SQLite folds the constant at plan time.
    let mut sql = String::from(
        "SELECT c.session_id \
         FROM qa_chunks c \
         JOIN ( \
             SELECT chunk_id, distance FROM vec_chunks \
             WHERE embedding MATCH ? \
             ORDER BY distance \
             LIMIT ? \
         ) v ON c.id = v.chunk_id \
         WHERE 1 = 1",
    );
    let mut filter_params: Vec<Param> = Vec::new();
    append_session_filter(&mut sql, &mut filter_params, "c.session_id", opts, now_ms);
    sql.push_str(" GROUP BY c.session_id ORDER BY MIN(v.distance)");

    let limit_i64 = limit as i64;
    let mut refs: Vec<&dyn ToSql> = Vec::with_capacity(2 + filter_params.len());
    refs.push(&embedding_bytes);
    refs.push(&limit_i64);
    for p in &filter_params {
        refs.push(p as &dyn ToSql);
    }
    debug_assert_eq!(
        refs.len(),
        sql.matches('?').count(),
        "param count must match SQL placeholder count"
    );

    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(refs.as_slice(), |row| row.get::<_, String>(0))?;

    let mut hits = Vec::new();
    for r in rows {
        hits.push((r?, 0.0));
    }
    Ok(hits)
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
    let matched = match prepare_match_query(conn, query, "messages_vocab") {
        Ok(m) => m,
        Err(SanitizeError::EmptyInput | SanitizeError::NoSearchableTerms) => {
            debug!(%query, "query produced no searchable terms");
            return Ok(Vec::new());
        }
    };
    let Some(fts_query) = clean_for_trigram(&matched) else {
        return Ok(Vec::new());
    };
    let fts_ranked = find_candidate_sessions(conn, opts, &fts_query, now_ms)?;

    if let Some(embedder) = embedder
        && has_vec_data(conn)
    {
        let candidate_limit = opts.limit * 3;

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

        let mut merged = rrf_merge(&fts_hits, &vec_hits);

        if merged.is_empty() {
            return Ok(Vec::new());
        }

        let mut meta_map = fetch_session_metadata(conn, &merged)?;

        // Drop hits whose `sessions` row is missing or has an unknown source;
        // `fetch_session_metadata` skips those, and without pruning here they
        // would consume `limit` slots during truncate and cause `fetch_snippets`
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

        let candidates = fetch_snippets(conn, &fts_query, merged, &mut meta_map)?;
        Ok(candidates.into_iter().map(|(_, r)| r).collect())
    } else {
        if fts_ranked.is_empty() {
            return Ok(Vec::new());
        }
        let mut meta_map = fetch_session_metadata(conn, &fts_ranked)?;
        let candidates = fetch_snippets(conn, &fts_query, fts_ranked, &mut meta_map)?;
        Ok(score_and_sort(candidates, now_ms, opts.limit))
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use amici::testing::hybrid::assert_filter_symmetric;

    use super::*;
    use crate::db::setup_test_db;
    use crate::embedder::MockEmbedder;

    fn make_result(sid: &str, ts: i64) -> SearchResult {
        SearchResult {
            session: SessionData {
                session_id: sid.to_owned(),
                source: Source::Claude,
                file_path: String::new(),
                project: String::new(),
                slug: String::new(),
                timestamp: Some(ts),
            },
            excerpt: String::new(),
        }
    }

    fn insert_session(conn: &Connection, sid: &str, source: &str, project: &str, ts: i64) {
        conn.execute(
            "INSERT INTO sessions (session_id, source, file_path, project, slug, timestamp, mtime) VALUES (?1, ?2, '', ?3, ?4, ?5, 0.0)",
            rusqlite::params![sid, source, project, sid, ts],
        ).unwrap();
    }

    fn insert_message(conn: &Connection, sid: &str, role: &str, text: &str) {
        conn.execute(
            "INSERT INTO messages (session_id, role, text) VALUES (?1, ?2, ?3)",
            rusqlite::params![sid, role, text],
        )
        .unwrap();
    }

    fn insert_chunk_with_embedding(
        conn: &Connection,
        chunk_id: i64,
        sid: &str,
        text: &str,
        ts: i64,
    ) {
        conn.execute(
            "INSERT INTO qa_chunks (id, session_id, user_text, assistant_text, content, timestamp, chunk_hash) \
             VALUES (?1, ?2, 'q', 'a', ?3, ?4, ?5)",
            rusqlite::params![chunk_id, sid, text, ts, format!("hash{chunk_id}")],
        )
        .unwrap();
        let embedding = MockEmbedder::deterministic_vector(text);
        let embedding_bytes: &[u8] = f32_as_bytes(&embedding);
        conn.execute(
            "INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?1, ?2)",
            rusqlite::params![chunk_id, embedding_bytes],
        )
        .unwrap();
    }

    #[test]
    fn test_basic_search_returns_match() {
        let (_dir, conn) = setup_test_db();
        insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
        insert_message(&conn, "s1", "user", "how to implement authentication");

        let results = search(&conn, "authentication", &SearchOptions::default()).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].session.session_id, "s1");
        assert!(!results[0].excerpt.is_empty());
    }

    #[test]
    fn test_search_no_match() {
        let (_dir, conn) = setup_test_db();
        insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
        insert_message(&conn, "s1", "user", "hello world");

        let results = search(&conn, "nonexistent_term_xyz", &SearchOptions::default()).unwrap();

        assert!(results.is_empty());
    }

    #[test]
    fn test_recency_boost_favors_newer() {
        let (_dir, conn) = setup_test_db();

        let now_ms = 1_750_000_000_000_i64; // fixed reference point

        insert_session(&conn, "old", "claude", "/proj", now_ms - 365 * MS_PER_DAY);
        insert_message(&conn, "old", "user", "testing the search feature");

        insert_session(&conn, "new", "claude", "/proj", now_ms);
        insert_message(&conn, "new", "user", "testing the search feature");

        let results = search(
            &conn,
            "testing search",
            &SearchOptions {
                now_ms: Some(now_ms),
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].session.session_id, "new");
    }

    #[test]
    fn test_project_filter() {
        let (_dir, conn) = setup_test_db();
        insert_session(&conn, "s1", "claude", "/home/me/proj-a", 1709251200000);
        insert_message(&conn, "s1", "user", "error handling patterns");
        insert_session(&conn, "s2", "claude", "/home/me/proj-b", 1709251200000);
        insert_message(&conn, "s2", "user", "error handling patterns");

        let results = search(
            &conn,
            "error handling",
            &SearchOptions {
                project: Some("/home/me/proj-a".to_owned()),
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].session.project, "/home/me/proj-a");
    }

    #[test]
    fn test_source_filter() {
        let (_dir, conn) = setup_test_db();
        insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
        insert_message(&conn, "s1", "user", "debugging tips");
        insert_session(&conn, "s2", "codex", "/proj", 1709251200000);
        insert_message(&conn, "s2", "user", "debugging tips");

        let results = search(
            &conn,
            "debugging",
            &SearchOptions {
                source: Some(Source::Codex),
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].session.source, Source::Codex);
    }

    #[test]
    fn test_days_filter() {
        let (_dir, conn) = setup_test_db();

        let now_ms = 1_750_000_000_000_i64; // fixed reference point

        insert_session(&conn, "recent", "claude", "/proj", now_ms - MS_PER_DAY); // 1 day ago
        insert_message(&conn, "recent", "user", "rust compiler optimization");
        insert_session(&conn, "old", "claude", "/proj", now_ms - 60 * MS_PER_DAY); // 60 days ago
        insert_message(&conn, "old", "user", "rust compiler optimization");

        let results = search(
            &conn,
            "rust compiler",
            &SearchOptions {
                days: Some(7),
                now_ms: Some(now_ms),
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].session.session_id, "recent");
    }

    #[test]
    fn test_project_filter_escapes_wildcards() {
        let (_dir, conn) = setup_test_db();
        insert_session(&conn, "s1", "claude", "/home/me/proj-a", 1709251200000);
        insert_message(&conn, "s1", "user", "wildcard test content");
        insert_session(&conn, "s2", "claude", "/other/path", 1709251200000);
        insert_message(&conn, "s2", "user", "wildcard test content");

        let results = search(
            &conn,
            "wildcard",
            &SearchOptions {
                project: Some("%".to_owned()),
                ..Default::default()
            },
        )
        .unwrap();

        assert!(results.is_empty());
    }

    #[test]
    fn test_cjk_japanese_2char() {
        let (_dir, conn) = setup_test_db();
        insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
        insert_message(&conn, "s1", "user", "型安全についての議論");

        let results = search(&conn, "型安", &SearchOptions::default()).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].session.session_id, "s1");
        assert!(!results[0].excerpt.is_empty());
    }

    #[test]
    fn test_cjk_single_char() {
        let (_dir, conn) = setup_test_db();
        insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
        insert_message(&conn, "s1", "user", "Rust言語の型システム");

        let results = search(&conn, "型", &SearchOptions::default()).unwrap();

        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_cjk_no_match() {
        let (_dir, conn) = setup_test_db();
        insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
        insert_message(&conn, "s1", "user", "hello world");

        let results = search(&conn, "没有", &SearchOptions::default()).unwrap();

        assert!(results.is_empty());
    }

    #[test]
    fn test_mixed_cjk_ascii_query() {
        let (_dir, conn) = setup_test_db();
        insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
        insert_message(&conn, "s1", "user", "Rustの型安全について");
        insert_session(&conn, "s2", "claude", "/proj", 1709251200000);
        insert_message(&conn, "s2", "user", "Pythonのパフォーマンス");

        let results = search(&conn, "Rust 型安全", &SearchOptions::default()).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].session.session_id, "s1");
    }

    #[test]
    fn test_cjk_with_project_filter() {
        let (_dir, conn) = setup_test_db();
        insert_session(&conn, "s1", "claude", "/home/me/proj-a", 1709251200000);
        insert_message(&conn, "s1", "user", "型安全の設計");
        insert_session(&conn, "s2", "claude", "/home/me/proj-b", 1709251200000);
        insert_message(&conn, "s2", "user", "型安全の設計");

        let results = search(
            &conn,
            "型安",
            &SearchOptions {
                project: Some("/home/me/proj-a".to_owned()),
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].session.project, "/home/me/proj-a");
    }

    #[test]
    fn test_cjk_recency_favors_newer() {
        let (_dir, conn) = setup_test_db();
        let now_ms = 1_750_000_000_000_i64;

        insert_session(&conn, "old", "claude", "/proj", now_ms - 365 * MS_PER_DAY);
        insert_message(&conn, "old", "user", "型安全についての議論");

        insert_session(&conn, "new", "claude", "/proj", now_ms);
        insert_message(&conn, "new", "user", "型安全の設計パターン");

        let results = search(
            &conn,
            "型安全",
            &SearchOptions {
                now_ms: Some(now_ms),
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].session.session_id, "new");
    }

    #[test]
    fn test_cjk_recency_with_candidate_overflow() {
        let (_dir, conn) = setup_test_db();
        let now_ms = 1_750_000_000_000_i64;
        let limit = 5;
        let candidate_limit = limit * 3; // 15

        // Insert 20 old sessions (exceeds candidate_limit)
        for i in 0..20 {
            let sid = format!("old-{i:02}");
            let ts = now_ms - (365 - i) * MS_PER_DAY;
            insert_session(&conn, &sid, "claude", "/proj", ts);
            insert_message(&conn, &sid, "user", &format!("型安全の議論 パート{i}"));
        }

        // Insert the newest session last (highest rowid)
        insert_session(&conn, "newest", "claude", "/proj", now_ms);
        insert_message(&conn, "newest", "user", "型安全の最新設計");

        let results = search(
            &conn,
            "型安全",
            &SearchOptions {
                limit,
                now_ms: Some(now_ms),
                ..Default::default()
            },
        )
        .unwrap();

        assert!(
            results.len() <= limit,
            "should respect limit: got {} results",
            results.len()
        );
        assert_eq!(
            results[0].session.session_id, "newest",
            "newest session must appear first despite {} candidates exceeding candidate_limit {}",
            21, candidate_limit
        );
    }

    #[test]
    fn test_korean_search() {
        let (_dir, conn) = setup_test_db();
        insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
        insert_message(&conn, "s1", "user", "한국어 테스트 메시지입니다");

        let results = search(&conn, "테스트", &SearchOptions::default()).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].session.session_id, "s1");
    }

    #[test]
    fn test_unbalanced_quote_auto_balanced() {
        let (_dir, conn) = setup_test_db();
        insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
        insert_message(&conn, "s1", "user", "hello world");

        let result = search(&conn, "\"hello", &SearchOptions::default());
        assert!(
            result.is_ok(),
            "auto-balanced quote should not cause FTS5 error"
        );
    }

    #[test]
    fn test_slash_command_query_matches_body() {
        let (_dir, conn) = setup_test_db();
        insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
        insert_message(&conn, "s1", "user", "used /challenge to review the plan");

        let results = search(&conn, "/challenge", &SearchOptions::default()).unwrap();
        assert_eq!(
            results.len(),
            1,
            "quoted slash token should match body text"
        );
        assert_eq!(results[0].session.session_id, "s1");
    }

    #[test]
    fn test_score_and_sort_respects_limit() {
        let now_ms = 1_750_000_000_000_i64;
        let candidates: Vec<_> = (0..5)
            .map(|i| (-1.0, make_result(&format!("s{i}"), now_ms)))
            .collect();
        let results = score_and_sort(candidates, now_ms, 3);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_search_limit_zero_returns_empty() {
        let (_dir, conn) = setup_test_db();
        insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
        insert_message(&conn, "s1", "user", "hello world");

        let results = search(
            &conn,
            "hello",
            &SearchOptions {
                limit: 0,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_index_then_search_roundtrip() {
        use crate::indexer::{IndexOptions, index_from_dirs};

        let (_dir, mut conn) = setup_test_db();
        let tmp = tempfile::TempDir::new().unwrap();

        let claude_dir = tmp.path().join("claude_projects");
        fs::create_dir_all(&claude_dir).unwrap();
        fs::write(
            claude_dir.join("sess-alpha.jsonl"),
            concat!(
                r#"{"type":"user","cwd":"/home/me/alpha","slug":"alpha-session","message":{"role":"user","content":"how to implement authentication in Rust"},"timestamp":"2026-03-01T12:00:00Z"}"#,
                "\n",
                r#"{"type":"assistant","message":{"role":"assistant","content":"Use the argon2 crate for password hashing"},"timestamp":"2026-03-01T12:00:01Z"}"#,
            ),
        ).unwrap();
        fs::write(
            claude_dir.join("sess-beta.jsonl"),
            r#"{"type":"user","cwd":"/home/me/beta","slug":"beta-session","message":{"role":"user","content":"explain quicksort algorithm"},"timestamp":"2026-03-02T10:00:00Z"}"#,
        ).unwrap();

        let codex_dir = tmp.path().join("codex_sessions");

        let stats = index_from_dirs(
            &mut conn,
            &IndexOptions {
                force: false,
                claude_dir: &claude_dir,
                codex_dir: &codex_dir,
            },
        )
        .unwrap();
        assert_eq!(stats.indexed, 2);
        assert_eq!(stats.total_sessions, 2);

        let results = search(&conn, "authentication", &SearchOptions::default()).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].session.session_id, "sess-alpha");
        assert_eq!(results[0].session.project, "/home/me/alpha");
        assert!(!results[0].excerpt.is_empty());

        let results = search(&conn, "quicksort", &SearchOptions::default()).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].session.session_id, "sess-beta");

        // Project filter — alpha-only. rurico quotes FTS5 operators as literal
        // terms, so test each side separately instead of a single OR query.
        let results = search(
            &conn,
            "authentication",
            &SearchOptions {
                project: Some("/home/me/alpha".to_owned()),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].session.project, "/home/me/alpha");

        let results = search(
            &conn,
            "algorithm",
            &SearchOptions {
                project: Some("/home/me/alpha".to_owned()),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(
            results.is_empty(),
            "algorithm lives in beta; alpha filter should exclude it"
        );

        let stats2 = index_from_dirs(
            &mut conn,
            &IndexOptions {
                force: false,
                claude_dir: &claude_dir,
                codex_dir: &codex_dir,
            },
        )
        .unwrap();
        assert_eq!(stats2.indexed, 0);
        assert_eq!(stats2.total_sessions, 2);

        let results = search(&conn, "authentication", &SearchOptions::default()).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_column_filter_blocked_in_search() {
        let (_dir, conn) = setup_test_db();
        insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
        insert_message(&conn, "s1", "user", "some content about admin");

        let results = search(&conn, "role:user", &SearchOptions::default()).unwrap();
        assert!(
            results.is_empty(),
            "role:user should not match as column filter"
        );
    }

    #[test]
    fn test_012_search_without_embedder_fts5_only() {
        let (_dir, conn) = setup_test_db();
        insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
        insert_message(&conn, "s1", "user", "authentication flow discussion");

        let results =
            search_with_embedder(&conn, "authentication", &SearchOptions::default(), None).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].session.session_id, "s1");
    }

    #[test]
    fn test_hybrid_search_merges_fts_and_vector() {
        let (_dir, conn) = setup_test_db();
        let now_ms = 1_750_000_000_000_i64;

        // s1: matches FTS for "authentication"
        insert_session(&conn, "s1", "claude", "/proj", now_ms);
        insert_message(&conn, "s1", "user", "authentication flow discussion");

        // s2: no FTS match, but has a vec_chunk with embedding close to query
        insert_session(&conn, "s2", "claude", "/proj", now_ms);
        insert_message(&conn, "s2", "user", "unrelated topic about weather");

        insert_chunk_with_embedding(&conn, 1, "s2", "authentication", now_ms);

        assert!(has_vec_data(&conn));

        let embedder = MockEmbedder::new();
        let results = search_with_embedder(
            &conn,
            "authentication",
            &SearchOptions {
                now_ms: Some(now_ms),
                ..Default::default()
            },
            Some(&embedder),
        )
        .unwrap();

        let ids: Vec<&str> = results
            .iter()
            .map(|r| r.session.session_id.as_str())
            .collect();
        assert!(
            ids.contains(&"s1"),
            "s1 should appear (FTS match), got: {ids:?}"
        );
        assert!(
            ids.contains(&"s2"),
            "s2 should appear (vector match), got: {ids:?}"
        );
    }

    #[test]
    fn test_hybrid_recency_boost_favors_newer() {
        let (_dir, conn) = setup_test_db();
        let now_ms = 1_750_000_000_000_i64;

        // Two sessions with identical FTS-matching text and identical embedding content.
        // Only the timestamp differs; hybrid path must rank the newer one first.
        for (sid, ts) in [("old", now_ms - 365 * MS_PER_DAY), ("new", now_ms)] {
            insert_session(&conn, sid, "claude", "/proj", ts);
            insert_message(&conn, sid, "user", "authentication flow discussion");
        }

        for (chunk_id, sid) in [(1_i64, "old"), (2, "new")] {
            insert_chunk_with_embedding(
                &conn,
                chunk_id,
                sid,
                "authentication flow discussion",
                now_ms,
            );
        }

        let embedder = MockEmbedder::new();
        let results = search_with_embedder(
            &conn,
            "authentication",
            &SearchOptions {
                now_ms: Some(now_ms),
                ..Default::default()
            },
            Some(&embedder),
        )
        .unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(
            results[0].session.session_id, "new",
            "hybrid path must apply recency boost so newer session ranks first"
        );
    }

    #[test]
    fn test_hybrid_search_falls_back_on_vector_error() {
        let (_dir, conn) = setup_test_db();
        let now_ms = 1_750_000_000_000_i64;

        insert_session(&conn, "s1", "claude", "/proj", now_ms);
        insert_message(&conn, "s1", "user", "authentication flow discussion");

        // Need vec_chunks to trigger hybrid path
        insert_chunk_with_embedding(&conn, 1, "s1", "content", now_ms);

        let embedder = MockEmbedder::failing_after(0);
        let results = search_with_embedder(
            &conn,
            "authentication",
            &SearchOptions {
                now_ms: Some(now_ms),
                ..Default::default()
            },
            Some(&embedder),
        )
        .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].session.session_id, "s1");
    }

    fn hybrid_search_ids(
        conn: &Connection,
        embedder: &MockEmbedder,
        opts: &SearchOptions,
    ) -> Vec<String> {
        search_with_embedder(conn, "authentication", opts, Some(embedder))
            .unwrap()
            .into_iter()
            .map(|r| r.session.session_id)
            .collect()
    }

    #[test]
    fn test_hybrid_project_filter_excludes_vec_only_mismatch() {
        let (_dir, conn) = setup_test_db();
        let now_ms = 1_750_000_000_000_i64;
        let embedder = MockEmbedder::new();

        assert_filter_symmetric(
            || {
                insert_session(&conn, "s1", "claude", "/home/me/proj-a", now_ms);
                insert_message(&conn, "s1", "user", "authentication flow discussion");
                insert_session(&conn, "s2", "claude", "/home/me/proj-b", now_ms);
                insert_message(&conn, "s2", "user", "unrelated topic about weather");
                insert_chunk_with_embedding(&conn, 1, "s2", "authentication", now_ms);
                ("s1".to_owned(), "s2".to_owned())
            },
            || {
                hybrid_search_ids(
                    &conn,
                    &embedder,
                    &SearchOptions {
                        project: Some("/home/me/proj-a".to_owned()),
                        now_ms: Some(now_ms),
                        ..Default::default()
                    },
                )
            },
            || {
                hybrid_search_ids(
                    &conn,
                    &embedder,
                    &SearchOptions {
                        now_ms: Some(now_ms),
                        ..Default::default()
                    },
                )
            },
        );
    }

    #[test]
    fn test_hybrid_days_filter_excludes_vec_only_old() {
        let (_dir, conn) = setup_test_db();
        let now_ms = 1_750_000_000_000_i64;
        let embedder = MockEmbedder::new();

        assert_filter_symmetric(
            || {
                insert_session(&conn, "s1", "claude", "/proj", now_ms - MS_PER_DAY);
                insert_message(&conn, "s1", "user", "authentication flow discussion");
                insert_session(&conn, "s2", "claude", "/proj", now_ms - 60 * MS_PER_DAY);
                insert_message(&conn, "s2", "user", "unrelated topic about weather");
                insert_chunk_with_embedding(&conn, 1, "s2", "authentication", now_ms);
                ("s1".to_owned(), "s2".to_owned())
            },
            || {
                hybrid_search_ids(
                    &conn,
                    &embedder,
                    &SearchOptions {
                        days: Some(7),
                        now_ms: Some(now_ms),
                        ..Default::default()
                    },
                )
            },
            || {
                hybrid_search_ids(
                    &conn,
                    &embedder,
                    &SearchOptions {
                        now_ms: Some(now_ms),
                        ..Default::default()
                    },
                )
            },
        );
    }

    #[test]
    fn test_hybrid_project_filter_case_insensitive_match() {
        // SQLite `LIKE` is ASCII case-insensitive. Both FTS and vec paths now
        // share the same SQL filter, so a case-mismatched project must still
        // match on the vec path.
        let (_dir, conn) = setup_test_db();
        let now_ms = 1_750_000_000_000_i64;

        insert_session(&conn, "s1", "claude", "/Home/me/Proj-A", now_ms);
        insert_message(&conn, "s1", "user", "authentication flow discussion");
        insert_chunk_with_embedding(&conn, 1, "s1", "authentication", now_ms);

        let embedder = MockEmbedder::new();
        let ids = hybrid_search_ids(
            &conn,
            &embedder,
            &SearchOptions {
                project: Some("/home/me/proj-a".to_owned()),
                now_ms: Some(now_ms),
                ..Default::default()
            },
        );

        assert!(
            ids.iter().any(|id| id == "s1"),
            "case-mismatched project should match (SQLite LIKE parity), got: {ids:?}"
        );
    }

    #[test]
    fn test_hybrid_source_filter_excludes_vec_only_mismatch() {
        let (_dir, conn) = setup_test_db();
        let now_ms = 1_750_000_000_000_i64;
        let embedder = MockEmbedder::new();

        assert_filter_symmetric(
            || {
                insert_session(&conn, "s1", "claude", "/proj", now_ms);
                insert_message(&conn, "s1", "user", "authentication flow discussion");
                insert_session(&conn, "s2", "codex", "/proj", now_ms);
                insert_message(&conn, "s2", "user", "unrelated topic about weather");
                insert_chunk_with_embedding(&conn, 1, "s2", "authentication", now_ms);
                ("s1".to_owned(), "s2".to_owned())
            },
            || {
                hybrid_search_ids(
                    &conn,
                    &embedder,
                    &SearchOptions {
                        source: Some(Source::Claude),
                        now_ms: Some(now_ms),
                        ..Default::default()
                    },
                )
            },
            || {
                hybrid_search_ids(
                    &conn,
                    &embedder,
                    &SearchOptions {
                        now_ms: Some(now_ms),
                        ..Default::default()
                    },
                )
            },
        );
    }

    #[test]
    fn test_hybrid_unknown_source_does_not_crowd_out_valid_hit() {
        // Regression: if `fetch_session_metadata` skips a merged hit (e.g. the
        // `sessions.source` value is unknown to `Source::from_db`), that hit
        // must be pruned before `truncate(limit)`, or a higher-rank invalid
        // session could consume the slot a valid one should have used.
        let (_dir, conn) = setup_test_db();
        let now_ms = 1_750_000_000_000_i64;

        insert_session(&conn, "s_valid", "claude", "/proj", now_ms);
        insert_message(&conn, "s_valid", "user", "authentication flow discussion");

        insert_session(&conn, "s_unknown", "mystery_source_v99", "/proj", now_ms);
        insert_message(&conn, "s_unknown", "user", "unrelated topic");
        insert_chunk_with_embedding(&conn, 1, "s_unknown", "authentication", now_ms);

        let embedder = MockEmbedder::new();
        let results = search_with_embedder(
            &conn,
            "authentication",
            &SearchOptions {
                limit: 1,
                now_ms: Some(now_ms),
                ..Default::default()
            },
            Some(&embedder),
        )
        .unwrap();

        assert_eq!(
            results.len(),
            1,
            "s_valid must fill the limit slot even if s_unknown ranks above it"
        );
        assert_eq!(results[0].session.session_id, "s_valid");
    }

    #[test]
    fn test_search_cjk_short_term_with_special_chars() {
        // The actual bug scenario: 2-char CJK term where vocab contains
        // punctuation-laden trigrams (e.g. "方針：", "方針*", "方針（")
        let (_dir, conn) = setup_test_db();
        insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
        insert_message(
            &conn,
            "s1",
            "user",
            "方針：まとめ 方針*概要 方針（案） ディレクトリ整理の方針",
        );

        // "方針" is 2 chars → triggers vocab expansion with special-char trigrams
        let results = search(&conn, "ディレクトリ整理 方針", &SearchOptions::default())
            .expect("CJK short term + long term should not cause FTS5 error");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_escape_like_basic() {
        assert_eq!(escape_like("hello"), "hello");
        assert_eq!(escape_like("100%"), "100\\%");
        assert_eq!(escape_like("a_b"), "a\\_b");
        assert_eq!(escape_like("c:\\path"), "c:\\\\path");
        assert_eq!(escape_like("%_\\"), "\\%\\_\\\\");
    }
}
