use std::collections::HashMap;

use anyhow::{Context, Result};
use rusqlite::Connection;

use crate::date::MS_PER_DAY;
use crate::parser::{SessionData, Source};

const SNIPPET_CONTEXT_BEFORE: usize = 60;
const SNIPPET_MAX_LEN: usize = 200;

/// Sessions older than 30 days lose half their recency boost — balances "recent" vs "ancient".
const RECENCY_HALF_LIFE_DAYS: f64 = 30.0;
/// Mild recency boost (0.2) avoids overriding strong BM25 matches while favoring newer sessions.
const RECENCY_BOOST_WEIGHT: f64 = 0.2;

pub struct SearchResult {
    pub session: SessionData,
    /// Snippet with `**` highlight markers (FTS5) or a raw substring window (CJK instr).
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

fn extract_search_words(query: &str) -> Vec<String> {
    query
        .split_whitespace()
        .filter(|w| !matches!(w.to_ascii_uppercase().as_str(), "AND" | "OR" | "NOT"))
        .map(|w| {
            w.trim_matches(|c: char| c == '"' || c == '*' || c == '(' || c == ')')
                .to_string()
        })
        .filter(|w| !w.is_empty())
        .collect()
}

fn has_cjk(s: &str) -> bool {
    s.chars().any(|c| {
        matches!(c,
            '\u{1100}'..='\u{11FF}'     // Hangul Jamo
            | '\u{2E80}'..='\u{9FFF}'   // CJK Radicals .. Unified Ideographs
            | '\u{A960}'..='\u{A97F}'   // Hangul Jamo Extended-A
            | '\u{AC00}'..='\u{D7FF}'   // Hangul Syllables + Jamo Extended-B
            | '\u{F900}'..='\u{FAFF}'   // CJK Compatibility Ideographs
            | '\u{FE30}'..='\u{FE4F}'   // CJK Compatibility Forms
            | '\u{FF00}'..='\u{FFEF}'   // Halfwidth and Fullwidth Forms
            | '\u{20000}'..='\u{2FA1F}' // CJK Extensions B..
        )
    })
}

enum Param {
    Str(String),
    Int(i64),
}

impl rusqlite::types::ToSql for Param {
    fn to_sql(&self) -> rusqlite::Result<rusqlite::types::ToSqlOutput<'_>> {
        match self {
            Param::Str(s) => s.to_sql(),
            Param::Int(i) => i.to_sql(),
        }
    }
}

struct SearchQuery {
    raw_query: String,
    use_instr: bool,
    terms: Vec<String>,
    session_filter: String,
    params: Vec<Param>,
}

/// Neutralize FTS5 special syntax in user queries: column-filters (`role:admin`),
/// start-of-column (`^term`), `NEAR()` grouping, `+` required-match prefix,
/// and unbalanced quotes (auto-balanced to prevent syntax errors).
fn sanitize_fts_query(query: &str) -> String {
    let result = query
        .split_whitespace()
        .filter(|w| {
            let upper = w.to_ascii_uppercase();
            !upper.starts_with("NEAR(") && !upper.starts_with("NEAR/")
        })
        .map(|w| {
            let w = w.trim_start_matches(['^', '+', '-']);
            let w = w.trim_matches(['(', ')']);
            if w.contains(':') && !w.starts_with('"') {
                let clean = w.replace('"', "");
                format!("\"{clean}\"")
            } else {
                w.to_string()
            }
        })
        .filter(|w| !w.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    let quote_count = result.chars().filter(|&c| c == '"').count();
    if quote_count % 2 != 0 {
        format!("{result}\"")
    } else {
        result
    }
}

fn build_session_filter(
    opts: &SearchOptions,
    now_ms: i64,
    params: &mut Vec<Param>,
) -> String {
    let mut conditions = Vec::new();
    if let Some(ref project) = opts.project {
        let escaped = project
            .replace('\\', "\\\\")
            .replace('%', "\\%")
            .replace('_', "\\_");
        conditions.push("s2.project LIKE ? || '%' ESCAPE '\\'".to_string());
        params.push(Param::Str(escaped));
    }
    if let Some(days) = opts.days {
        let cutoff = now_ms - days * MS_PER_DAY;
        conditions.push("s2.timestamp >= ?".to_string());
        params.push(Param::Int(cutoff));
    }
    if let Some(source) = opts.source {
        conditions.push("s2.source = ?".to_string());
        params.push(Param::Str(source.as_str().to_string()));
    }
    if conditions.is_empty() {
        String::new()
    } else {
        format!(
            " AND session_id IN (SELECT s2.session_id FROM sessions s2 WHERE {})",
            conditions.join(" AND ")
        )
    }
}

fn build_search_query(query: &str, opts: &SearchOptions, now_ms: i64) -> SearchQuery {
    let terms = extract_search_words(query);
    let use_instr = terms.iter().any(|t| has_cjk(t));

    let mut params = Vec::new();
    let sanitized_query = sanitize_fts_query(query);
    if use_instr {
        for term in &terms {
            params.push(Param::Str(term.clone()));
        }
    } else {
        params.push(Param::Str(sanitized_query.clone()));
    }

    let session_filter = build_session_filter(opts, now_ms, &mut params);

    let candidate_limit = opts.limit * 3;
    params.push(Param::Int(candidate_limit as i64));

    SearchQuery {
        raw_query: sanitized_query,
        use_instr,
        terms,
        session_filter,
        params,
    }
}

fn build_candidate_sql(sq: &SearchQuery) -> String {
    if sq.use_instr {
        let instr_conds: Vec<&str> = sq.terms.iter().map(|_| "instr(text, ?) > 0").collect();
        format!(
            "SELECT session_id, 0.0 as best_rank FROM messages WHERE {conds}{filter} GROUP BY session_id ORDER BY MIN(rowid) LIMIT ?",
            conds = instr_conds.join(" AND "),
            filter = sq.session_filter
        )
    } else {
        format!(
            "SELECT session_id, MIN(rank) as best_rank FROM messages WHERE messages MATCH ?{filter} GROUP BY session_id ORDER BY best_rank LIMIT ?",
            filter = sq.session_filter
        )
    }
}

fn find_candidate_sessions(conn: &Connection, sq: &SearchQuery) -> Result<Vec<(String, f64)>> {
    let sql = build_candidate_sql(sq);
    let refs: Vec<&dyn rusqlite::types::ToSql> = sq.params.iter().map(|p| p as &dyn rusqlite::types::ToSql).collect();
    debug_assert_eq!(
        refs.len(),
        sql.matches('?').count(),
        "param count must match SQL placeholder count"
    );
    let mut stmt = conn.prepare(&sql).context("Failed to prepare search query")?;
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
                if first_error.is_none() { first_error = Some(e.to_string()); }
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
        eprintln!("Warning: {skipped} row(s) skipped during search: {err_msg}");
    }
    Ok(ranked)
}

fn fetch_session_metadata(
    conn: &Connection,
    ranked: &[(String, f64)],
) -> Result<HashMap<String, SessionData>> {
    let placeholders = ranked.iter().map(|_| "?").collect::<Vec<_>>().join(", ");
    let sql = format!(
        "SELECT session_id, source, file_path, project, slug, timestamp \
         FROM sessions WHERE session_id IN ({placeholders})"
    );
    let refs: Vec<&dyn rusqlite::types::ToSql> =
        ranked.iter().map(|(sid, _)| sid as &dyn rusqlite::types::ToSql).collect();

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
                        eprintln!("Warning: unknown source {source_str:?} for session {session_id}");
                        continue;
                    }
                };
                map.insert(session_id.clone(), SessionData {
                    session_id, source, file_path, project, slug, timestamp,
                });
            }
            Err(e) => {
                if db_errors == 0 { eprintln!("Warning: metadata query error: {e}"); }
                db_errors += 1;
            }
        }
    }
    if db_errors > 1 { eprintln!("Warning: {db_errors} metadata row(s) failed"); }
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
            Some((rank, SearchResult { session, excerpt: snippet }))
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
            let recency_boost = if let Some(ts) = result.session.timestamp {
                let age_days = ((now_ms as f64 - ts as f64) / MS_PER_DAY as f64).max(0.0);
                (-std::f64::consts::LN_2 * age_days / RECENCY_HALF_LIFE_DAYS).exp()
            } else {
                0.0
            };
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

fn snippet_or_default(result: rusqlite::Result<String>, sid: &str) -> Option<String> {
    match result {
        Ok(s) => Some(s),
        Err(rusqlite::Error::QueryReturnedNoRows) => None,
        Err(e) => {
            eprintln!("Warning: snippet extraction failed for session {sid}: {e}");
            None
        }
    }
}

fn fetch_snippets(
    conn: &Connection,
    sq: &SearchQuery,
    ranked: Vec<(String, f64)>,
    meta_map: &mut HashMap<String, SessionData>,
) -> Result<Vec<(f64, SearchResult)>> {
    if sq.use_instr {
        let sql = &format!("SELECT substr(text, max(1, instr(text, ?1) - {SNIPPET_CONTEXT_BEFORE}), {SNIPPET_MAX_LEN}) FROM messages WHERE instr(text, ?1) > 0 AND session_id = ?2 LIMIT 1");
        let mut stmt = conn.prepare(sql)?;
        Ok(build_candidates(ranked, meta_map, |sid| {
            sq.terms
                .iter()
                .find_map(|t| {
                    snippet_or_default(
                        stmt.query_row(rusqlite::params![t, sid], |row| row.get(0)),
                        sid,
                    )
                })
                .unwrap_or_default()
        }))
    } else {
        let sql = "SELECT snippet(messages, 2, '**', '**', '...', 20) FROM messages WHERE messages MATCH ?1 AND session_id = ?2 LIMIT 1";
        let mut stmt = conn.prepare(sql)?;
        Ok(build_candidates(ranked, meta_map, |sid| {
            snippet_or_default(
                stmt.query_row(rusqlite::params![&sq.raw_query, sid], |row| row.get(0)),
                sid,
            )
            .unwrap_or_default()
        }))
    }
}

/// Search indexed sessions. FTS5 MATCH for non-CJK, `instr()` fallback for CJK.
/// Results ranked by BM25 blended with exponential recency boost.
pub fn search(conn: &Connection, query: &str, opts: &SearchOptions) -> Result<Vec<SearchResult>> {
    if opts.limit == 0 {
        return Ok(Vec::new());
    }

    let now_ms = match opts.now_ms {
        Some(ms) => ms,
        None => i64::try_from(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .context("system clock before UNIX epoch")?
                .as_millis(),
        )
        .context("timestamp overflow")?,
    };

    let sq = build_search_query(query, opts, now_ms);
    let ranked = find_candidate_sessions(conn, &sq)?;
    if ranked.is_empty() {
        return Ok(Vec::new());
    }

    let mut meta_map = fetch_session_metadata(conn, &ranked)?;
    let candidates = fetch_snippets(conn, &sq, ranked, &mut meta_map)?;
    Ok(score_and_sort(candidates, now_ms, opts.limit))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::setup_test_db;

    fn make_result(sid: &str, ts: i64) -> SearchResult {
        SearchResult {
            session: SessionData {
                session_id: sid.to_string(),
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
                project: Some("/home/me/proj-a".to_string()),
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
                project: Some("%".to_string()),
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
                project: Some("/home/me/proj-a".to_string()),
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

        // Unbalanced quotes are now auto-balanced — graceful degradation instead of error
        let result = search(&conn, "\"hello", &SearchOptions::default());
        assert!(result.is_ok(), "auto-balanced quote should not cause FTS5 error");
    }

    #[test]
    fn test_score_and_sort_recency_tiebreak() {
        let now_ms = 1_750_000_000_000_i64;
        let candidates = vec![
            (-1.0, make_result("old", now_ms - 365 * MS_PER_DAY)),
            (-1.0, make_result("new", now_ms)),
        ];
        let results = score_and_sort(candidates, now_ms, 10);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].session.session_id, "new");
    }

    #[test]
    fn test_score_and_sort_cjk_rank_zero() {
        let now_ms = 1_750_000_000_000_i64;
        let candidates = vec![
            (0.0, make_result("old", now_ms - 365 * MS_PER_DAY)),
            (0.0, make_result("new", now_ms)),
        ];
        let results = score_and_sort(candidates, now_ms, 10);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].session.session_id, "new");
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
    fn test_extract_search_words_basic() {
        assert_eq!(extract_search_words("hello world"), vec!["hello", "world"]);
    }

    #[test]
    fn test_extract_search_words_strips_operators() {
        assert_eq!(extract_search_words("rust AND error NOT warning"), vec!["rust", "error", "warning"]);
        assert_eq!(extract_search_words("foo OR bar"), vec!["foo", "bar"]);
    }

    #[test]
    fn test_extract_search_words_strips_punctuation() {
        assert_eq!(extract_search_words("\"quoted phrase\" wild*"), vec!["quoted", "phrase", "wild"]);
        assert_eq!(extract_search_words("(group)"), vec!["group"]);
    }

    #[test]
    fn test_extract_search_words_empty() {
        assert!(extract_search_words("AND OR NOT").is_empty());
        assert!(extract_search_words("\"\"").is_empty());
        assert!(extract_search_words("   ").is_empty());
    }

    #[test]
    fn test_search_limit_zero_returns_empty() {
        let (_dir, conn) = setup_test_db();
        insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
        insert_message(&conn, "s1", "user", "hello world");

        let results = search(
            &conn,
            "hello",
            &SearchOptions { limit: 0, ..Default::default() },
        )
        .unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_sanitize_fts_query_quotes_colons() {
        assert_eq!(sanitize_fts_query("role:admin"), "\"role:admin\"");
        assert_eq!(sanitize_fts_query("hello world"), "hello world");
        assert_eq!(sanitize_fts_query("foo role:user bar"), "foo \"role:user\" bar");
    }

    #[test]
    fn test_sanitize_fts_query_preserves_quoted() {
        assert_eq!(sanitize_fts_query("\"role:admin\""), "\"role:admin\"");
        assert_eq!(sanitize_fts_query("AND OR NOT"), "AND OR NOT");
    }

    #[test]
    fn test_sanitize_fts_query_strips_caret_prefix() {
        assert_eq!(sanitize_fts_query("^hello world"), "hello world");
        assert_eq!(sanitize_fts_query("^foo ^bar"), "foo bar");
        assert_eq!(sanitize_fts_query("^"), "");
    }

    #[test]
    fn test_sanitize_fts_query_filters_near_tokens() {
        // NEAR( prefix tokens removed; trailing parentheses stripped from remaining tokens
        assert_eq!(sanitize_fts_query("NEAR(a b)"), "b");
        assert_eq!(sanitize_fts_query("hello NEAR(a b) world"), "hello b world");
        assert_eq!(sanitize_fts_query("hello NEAR/3 world"), "hello world");
        assert_eq!(sanitize_fts_query("near(x y)"), "y");
        assert_eq!(sanitize_fts_query("Near/2 test"), "test");
    }

    #[test]
    fn test_sanitize_fts_query_strips_plus_prefix() {
        assert_eq!(sanitize_fts_query("+required term"), "required term");
    }

    #[test]
    fn test_sanitize_fts_query_strips_minus_prefix() {
        assert_eq!(sanitize_fts_query("-excluded term"), "excluded term");
        assert_eq!(sanitize_fts_query("keep -this"), "keep this");
    }

    #[test]
    fn test_sanitize_fts_query_balances_quotes() {
        assert_eq!(sanitize_fts_query("\"unbalanced"), "\"unbalanced\"");
        assert_eq!(sanitize_fts_query("\"balanced\""), "\"balanced\"");
    }

    #[test]
    fn test_index_then_search_roundtrip() {
        use crate::indexer::{IndexOptions, index_from_dirs};

        let (_dir, mut conn) = setup_test_db();
        let tmp = tempfile::TempDir::new().unwrap();

        let claude_dir = tmp.path().join("claude_projects");
        std::fs::create_dir_all(&claude_dir).unwrap();
        std::fs::write(
            claude_dir.join("sess-alpha.jsonl"),
            concat!(
                r#"{"type":"user","cwd":"/home/me/alpha","slug":"alpha-session","message":{"role":"user","content":"how to implement authentication in Rust"},"timestamp":"2026-03-01T12:00:00Z"}"#,
                "\n",
                r#"{"type":"assistant","message":{"role":"assistant","content":"Use the argon2 crate for password hashing"},"timestamp":"2026-03-01T12:00:01Z"}"#,
            ),
        ).unwrap();
        std::fs::write(
            claude_dir.join("sess-beta.jsonl"),
            r#"{"type":"user","cwd":"/home/me/beta","slug":"beta-session","message":{"role":"user","content":"explain quicksort algorithm"},"timestamp":"2026-03-02T10:00:00Z"}"#,
        ).unwrap();

        let codex_dir = tmp.path().join("codex_sessions");

        let stats = index_from_dirs(&mut conn, &IndexOptions {
            force: false, verbose: false, claude_dir: &claude_dir, codex_dir: &codex_dir,
        }).unwrap();
        assert_eq!(stats.indexed, 2);
        assert_eq!(stats.total_sessions, 2);
        assert!(stats.total_messages >= 3);

        // Search for "authentication" — should find alpha only
        let results = search(&conn, "authentication", &SearchOptions::default()).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].session.session_id, "sess-alpha");
        assert_eq!(results[0].session.project, "/home/me/alpha");
        assert!(!results[0].excerpt.is_empty());

        // Search for "quicksort" — should find beta only
        let results = search(&conn, "quicksort", &SearchOptions::default()).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].session.session_id, "sess-beta");

        // Project filter — only alpha
        let results = search(&conn, "algorithm OR authentication", &SearchOptions {
            project: Some("/home/me/alpha".to_string()),
            ..Default::default()
        }).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].session.project, "/home/me/alpha");

        // Re-index should skip (unchanged files)
        let stats2 = index_from_dirs(&mut conn, &IndexOptions {
            force: false, verbose: false, claude_dir: &claude_dir, codex_dir: &codex_dir,
        }).unwrap();
        assert_eq!(stats2.indexed, 0);
        assert_eq!(stats2.total_sessions, 2);

        // Search still works after skip
        let results = search(&conn, "authentication", &SearchOptions::default()).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_column_filter_blocked_in_search() {
        let (_dir, conn) = setup_test_db();
        insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
        insert_message(&conn, "s1", "user", "some content about admin");

        // "role:user" should NOT act as a column filter — should search for the literal text
        let results = search(&conn, "role:user", &SearchOptions::default()).unwrap();
        assert!(results.is_empty(), "role:user should not match as column filter");
    }
}
