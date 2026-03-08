use anyhow::{Context, Result};
use rusqlite::Connection;

use crate::parser::{SessionData, Source};

const MS_PER_DAY: i64 = 86_400_000;
const RECENCY_HALF_LIFE_DAYS: f64 = 30.0;
const RECENCY_BOOST_WEIGHT: f64 = 0.2;

pub struct SearchResult {
    pub session: SessionData,
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
            '\u{2E80}'..='\u{9FFF}'
            | '\u{F900}'..='\u{FAFF}'
            | '\u{FE30}'..='\u{FE4F}'
            | '\u{FF00}'..='\u{FFEF}'
            | '\u{20000}'..='\u{2FA1F}'
        )
    })
}

/// Full-text search over indexed session messages.
///
/// Uses FTS5 MATCH with porter stemming for non-CJK queries.
/// Falls back to `instr()`-based substring matching for CJK queries
/// (porter unicode61 cannot segment CJK text).
/// Results are ranked by BM25 (FTS5) or recency-only (instr fallback).
pub fn search(conn: &Connection, query: &str, opts: &SearchOptions) -> Result<Vec<SearchResult>> {
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

    let terms = extract_search_words(query);
    let use_instr = terms.iter().any(|t| has_cjk(t));

    let mut params: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
    if use_instr {
        if terms.is_empty() {
            return Ok(Vec::new());
        }
        for term in &terms {
            params.push(Box::new(term.clone()));
        }
    } else {
        params.push(Box::new(query.to_string()));
    }
    let mut conditions = Vec::new();

    if let Some(ref project) = opts.project {
        let escaped = project
            .replace('\\', "\\\\")
            .replace('%', "\\%")
            .replace('_', "\\_");
        conditions.push("s2.project LIKE ? || '%' ESCAPE '\\'".to_string());
        params.push(Box::new(escaped));
    }
    if let Some(days) = opts.days {
        let cutoff = now_ms - days * MS_PER_DAY;
        conditions.push("s2.timestamp >= ?".to_string());
        params.push(Box::new(cutoff));
    }
    if let Some(source) = opts.source {
        conditions.push("s2.source = ?".to_string());
        params.push(Box::new(source.as_str().to_string()));
    }

    let session_filter = if conditions.is_empty() {
        String::new()
    } else {
        format!(
            " AND session_id IN (SELECT s2.session_id FROM sessions s2 WHERE {})",
            conditions.join(" AND ")
        )
    };

    let candidate_limit = opts.limit * 3;
    params.push(Box::new(candidate_limit as i64));

    // Pass 1: FTS5 ranking (porter) or instr() substring fallback for CJK.
    let pass1_sql = if use_instr {
        let instr_conds: Vec<String> = terms
            .iter()
            .map(|_| "instr(text, ?) > 0".to_string())
            .collect();
        format!(
            "SELECT session_id, 0.0 as best_rank
             FROM messages
             WHERE {}{session_filter}
             GROUP BY session_id
             LIMIT ?",
            instr_conds.join(" AND ")
        )
    } else {
        format!(
            "SELECT session_id, MIN(rank) as best_rank
             FROM messages
             WHERE messages MATCH ?{session_filter}
             GROUP BY session_id
             ORDER BY best_rank
             LIMIT ?"
        )
    };

    let param_refs: Vec<&dyn rusqlite::types::ToSql> = params.iter().map(|p| p.as_ref()).collect();
    let mut stmt = conn
        .prepare(&pass1_sql)
        .context("Failed to prepare search query")?;
    let rows = stmt
        .query_map(param_refs.as_slice(), |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?))
        })
        .context("Search query failed")?;

    let mut ranked: Vec<(String, f64)> = Vec::new();
    let mut first_error: Option<String> = None;
    for r in rows {
        match r {
            Ok(row) => ranked.push(row),
            Err(e) => {
                if first_error.is_none() {
                    first_error = Some(e.to_string());
                }
            }
        }
    }
    if ranked.is_empty()
        && let Some(err_msg) = first_error
    {
        anyhow::bail!(
            "Invalid search query: {err_msg}. Use words, \"quoted phrases\", or AND/OR/NOT operators."
        );
    }

    if ranked.is_empty() {
        return Ok(Vec::new());
    }

    // Pass 2: Batch metadata lookup (1 query instead of N).
    let placeholders: Vec<&str> = ranked.iter().map(|_| "?").collect();
    let meta_sql = format!(
        "SELECT session_id, source, file_path, project, slug, timestamp FROM sessions WHERE session_id IN ({})",
        placeholders.join(", ")
    );
    let meta_params: Vec<Box<dyn rusqlite::types::ToSql>> = ranked
        .iter()
        .map(|(sid, _)| Box::new(sid.clone()) as Box<dyn rusqlite::types::ToSql>)
        .collect();
    let meta_refs: Vec<&dyn rusqlite::types::ToSql> =
        meta_params.iter().map(|p| p.as_ref()).collect();

    let mut meta_stmt = conn.prepare(&meta_sql)?;
    let mut meta_map: std::collections::HashMap<String, SessionData> = meta_stmt
        .query_map(meta_refs.as_slice(), |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, i64>(5)?,
            ))
        })?
        .filter_map(|r| match r {
            Ok(row) => Some(row),
            Err(e) => {
                eprintln!("Warning: skipped metadata row: {e}");
                None
            }
        })
        .filter_map(|(sid, source_str, fp, proj, slug, ts)| {
            let source = match Source::from_db(&source_str) {
                Some(s) => s,
                None => {
                    eprintln!("Warning: unknown source '{source_str}' for session {sid}");
                    return None;
                }
            };
            Some((
                sid.clone(),
                SessionData {
                    session_id: sid,
                    source,
                    file_path: fp,
                    project: proj,
                    slug,
                    timestamp: ts,
                },
            ))
        })
        .collect();

    // Pass 3: Per-result snippet.
    let snippet_term: &str = if use_instr { &terms[0] } else { query };
    let snippet_sql = if use_instr {
        "SELECT substr(text, max(1, instr(text, ?1) - 60), 200) FROM messages WHERE instr(text, ?1) > 0 AND session_id = ?2 LIMIT 1"
    } else {
        "SELECT snippet(messages, 2, '**', '**', '...', 20) FROM messages WHERE messages MATCH ?1 AND session_id = ?2 LIMIT 1"
    };
    let mut snippet_stmt = conn.prepare(snippet_sql)?;
    let candidates: Vec<_> = ranked
        .into_iter()
        .filter_map(|(session_id, rank)| {
            let session = meta_map.remove(&session_id)?;
            let snippet: String = snippet_stmt
                .query_row(rusqlite::params![snippet_term, &session_id], |row| {
                    row.get(0)
                })
                .unwrap_or_default();
            let ts = session.timestamp;
            Some((
                rank,
                ts,
                SearchResult {
                    session,
                    excerpt: snippet,
                },
            ))
        })
        .collect();

    let mut results: Vec<(SearchResult, f64)> = candidates
        .into_iter()
        .map(|(rank, timestamp, result)| {
            let recency_boost = if timestamp > 0 {
                let age_days = ((now_ms as f64 - timestamp as f64) / MS_PER_DAY as f64).max(0.0);
                (-std::f64::consts::LN_2 * age_days / RECENCY_HALF_LIFE_DAYS).exp()
            } else {
                0.0
            };
            // rank is negative (more negative = better match in FTS5 BM25).
            // To boost recent sessions, increase the magnitude (make more negative).
            let blended_rank = rank * (1.0 + RECENCY_BOOST_WEIGHT * recency_boost);
            (result, blended_rank)
        })
        .collect();

    results.sort_by(|a, b| a.1.total_cmp(&b.1));
    results.truncate(opts.limit);

    Ok(results.into_iter().map(|(r, _)| r).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::open_db;
    use tempfile::TempDir;

    fn setup_search_db() -> (TempDir, Connection) {
        let dir = TempDir::new().unwrap();
        let db_path = dir.path().join("test.db");
        let conn = open_db(&db_path).unwrap();
        (dir, conn)
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

    // T-004: Search & Ranking

    #[test]
    fn test_basic_search_returns_match() {
        let (_dir, conn) = setup_search_db();
        insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
        insert_message(&conn, "s1", "user", "how to implement authentication");

        let results = search(
            &conn,
            "authentication",
            &SearchOptions {
                project: None,
                days: None,
                source: None,
                limit: 10,
                now_ms: None,
            },
        )
        .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].session.session_id, "s1");
        assert!(!results[0].excerpt.is_empty());
    }

    #[test]
    fn test_search_no_match() {
        let (_dir, conn) = setup_search_db();
        insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
        insert_message(&conn, "s1", "user", "hello world");

        let results = search(
            &conn,
            "nonexistent_term_xyz",
            &SearchOptions {
                project: None,
                days: None,
                source: None,
                limit: 10,
                now_ms: None,
            },
        )
        .unwrap();

        assert!(results.is_empty());
    }

    #[test]
    fn test_recency_boost_favors_newer() {
        let (_dir, conn) = setup_search_db();

        let now_ms = 1_750_000_000_000_i64; // fixed reference point

        // Old session (1 year ago)
        insert_session(&conn, "old", "claude", "/proj", now_ms - 365 * MS_PER_DAY);
        insert_message(&conn, "old", "user", "testing the search feature");

        // New session (today)
        insert_session(&conn, "new", "claude", "/proj", now_ms);
        insert_message(&conn, "new", "user", "testing the search feature");

        let results = search(
            &conn,
            "testing search",
            &SearchOptions {
                project: None,
                days: None,
                source: None,
                limit: 10,
                now_ms: Some(now_ms),
            },
        )
        .unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].session.session_id, "new");
    }

    #[test]
    fn test_project_filter() {
        let (_dir, conn) = setup_search_db();
        insert_session(&conn, "s1", "claude", "/home/me/proj-a", 1709251200000);
        insert_message(&conn, "s1", "user", "error handling patterns");
        insert_session(&conn, "s2", "claude", "/home/me/proj-b", 1709251200000);
        insert_message(&conn, "s2", "user", "error handling patterns");

        let results = search(
            &conn,
            "error handling",
            &SearchOptions {
                project: Some("/home/me/proj-a".to_string()),
                days: None,
                source: None,
                limit: 10,
                now_ms: None,
            },
        )
        .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].session.project, "/home/me/proj-a");
    }

    #[test]
    fn test_source_filter() {
        let (_dir, conn) = setup_search_db();
        insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
        insert_message(&conn, "s1", "user", "debugging tips");
        insert_session(&conn, "s2", "codex", "/proj", 1709251200000);
        insert_message(&conn, "s2", "user", "debugging tips");

        let results = search(
            &conn,
            "debugging",
            &SearchOptions {
                project: None,
                days: None,
                source: Some(Source::Codex),
                limit: 10,
                now_ms: None,
            },
        )
        .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].session.source, Source::Codex);
    }

    #[test]
    fn test_days_filter() {
        let (_dir, conn) = setup_search_db();

        let now_ms = 1_750_000_000_000_i64; // fixed reference point

        insert_session(&conn, "recent", "claude", "/proj", now_ms - MS_PER_DAY); // 1 day ago
        insert_message(&conn, "recent", "user", "rust compiler optimization");
        insert_session(&conn, "old", "claude", "/proj", now_ms - 60 * MS_PER_DAY); // 60 days ago
        insert_message(&conn, "old", "user", "rust compiler optimization");

        let results = search(
            &conn,
            "rust compiler",
            &SearchOptions {
                project: None,
                days: Some(7),
                source: None,
                limit: 10,
                now_ms: Some(now_ms),
            },
        )
        .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].session.session_id, "recent");
    }

    // SEC-001: LIKE wildcard escape

    #[test]
    fn test_project_filter_escapes_wildcards() {
        let (_dir, conn) = setup_search_db();
        insert_session(&conn, "s1", "claude", "/home/me/proj-a", 1709251200000);
        insert_message(&conn, "s1", "user", "wildcard test content");
        insert_session(&conn, "s2", "claude", "/other/path", 1709251200000);
        insert_message(&conn, "s2", "user", "wildcard test content");

        // "%" without escaping would match everything
        let results = search(
            &conn,
            "wildcard",
            &SearchOptions {
                project: Some("%".to_string()),
                days: None,
                source: None,
                limit: 10,
                now_ms: None,
            },
        )
        .unwrap();

        // With escaping, "%" is literal — no project starts with "%"
        assert!(results.is_empty());
    }

    // Short query fallback (instr)

    #[test]
    fn test_short_query_japanese_2char() {
        let (_dir, conn) = setup_search_db();
        insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
        insert_message(&conn, "s1", "user", "型安全についての議論");

        let results = search(
            &conn,
            "型安",
            &SearchOptions {
                project: None,
                days: None,
                source: None,
                limit: 10,
                now_ms: None,
            },
        )
        .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].session.session_id, "s1");
        assert!(!results[0].excerpt.is_empty());
    }

    #[test]
    fn test_short_query_single_char() {
        let (_dir, conn) = setup_search_db();
        insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
        insert_message(&conn, "s1", "user", "Rust言語の型システム");

        let results = search(
            &conn,
            "型",
            &SearchOptions {
                project: None,
                days: None,
                source: None,
                limit: 10,
                now_ms: None,
            },
        )
        .unwrap();

        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_short_query_no_match() {
        let (_dir, conn) = setup_search_db();
        insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
        insert_message(&conn, "s1", "user", "hello world");

        let results = search(
            &conn,
            "没有",
            &SearchOptions {
                project: None,
                days: None,
                source: None,
                limit: 10,
                now_ms: None,
            },
        )
        .unwrap();

        assert!(results.is_empty());
    }

    #[test]
    fn test_mixed_cjk_ascii_query() {
        let (_dir, conn) = setup_search_db();
        insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
        insert_message(&conn, "s1", "user", "Rustの型安全について");
        insert_session(&conn, "s2", "claude", "/proj", 1709251200000);
        insert_message(&conn, "s2", "user", "Pythonのパフォーマンス");

        let results = search(
            &conn,
            "Rust 型安全",
            &SearchOptions {
                project: None,
                days: None,
                source: None,
                limit: 10,
                now_ms: None,
            },
        )
        .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].session.session_id, "s1");
    }

    #[test]
    fn test_short_query_with_project_filter() {
        let (_dir, conn) = setup_search_db();
        insert_session(&conn, "s1", "claude", "/home/me/proj-a", 1709251200000);
        insert_message(&conn, "s1", "user", "型安全の設計");
        insert_session(&conn, "s2", "claude", "/home/me/proj-b", 1709251200000);
        insert_message(&conn, "s2", "user", "型安全の設計");

        let results = search(
            &conn,
            "型安",
            &SearchOptions {
                project: Some("/home/me/proj-a".to_string()),
                days: None,
                source: None,
                limit: 10,
                now_ms: None,
            },
        )
        .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].session.project, "/home/me/proj-a");
    }

    #[test]
    fn test_invalid_fts5_query_returns_friendly_error() {
        let (_dir, conn) = setup_search_db();
        insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
        insert_message(&conn, "s1", "user", "hello world");

        let result = search(
            &conn,
            "\"unbalanced quote",
            &SearchOptions {
                project: None,
                days: None,
                source: None,
                limit: 10,
                now_ms: None,
            },
        );

        let err_msg = match result {
            Err(e) => e.to_string(),
            Ok(_) => panic!("Expected error for invalid FTS5 query"),
        };
        assert!(
            err_msg.contains("Invalid search query") || err_msg.contains("syntax error"),
            "Expected user-friendly error, got: {err_msg}"
        );
    }
}
