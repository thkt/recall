use std::fs;

use crate::compat::assert_filter_symmetric;

use super::*;
use crate::db::{seed_session_file, setup_test_db};
use crate::embedder::MockEmbedder;
use crate::indexer::index_chunks;

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
        chunk_id: None,
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

fn insert_session_file(conn: &Connection, sid: &str, path: &str) {
    seed_session_file(conn, sid, path);
}

fn insert_chunk_with_embedding(conn: &Connection, chunk_id: i64, sid: &str, text: &str, ts: i64) {
    conn.execute(
        "INSERT INTO qa_chunks (id, session_id, content, timestamp) VALUES (?1, ?2, ?3, ?4)",
        rusqlite::params![chunk_id, sid, text, ts],
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

fn insert_chunk_with_rowid_range(
    conn: &Connection,
    chunk_id: i64,
    sid: &str,
    text: &str,
    ts: i64,
    lo: i64,
    hi: i64,
) {
    conn.execute(
        "INSERT INTO qa_chunks (id, session_id, content, timestamp, src_rowid_lo, src_rowid_hi) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        rusqlite::params![chunk_id, sid, text, ts, lo, hi],
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

fn message_rowid(conn: &Connection, sid: &str, text: &str) -> i64 {
    conn.query_row(
        "SELECT rowid FROM messages WHERE session_id = ?1 AND text = ?2",
        rusqlite::params![sid, text],
        |r| r.get(0),
    )
    .unwrap()
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
fn test_fts_hit_excerpt_is_full_chunk_not_snippet() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
    insert_message(&conn, "s1", "user", "how to implement authentication");
    insert_chunk_with_embedding(
        &conn,
        1,
        "s1",
        "how to implement authentication — use JWT with short-lived access tokens and rotating refresh tokens",
        1709251200000,
    );

    let results = search(&conn, "authentication", &SearchOptions::default()).unwrap();

    assert_eq!(results.len(), 1);
    let excerpt = &results[0].excerpt;
    assert!(
        excerpt.contains("rotating refresh tokens"),
        "FTS-hit excerpt must be the full chunk, not a 20-token snippet: {excerpt:?}"
    );
}

#[test]
fn test_fts_operator_query_returns_containing_chunk() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
    insert_message(
        &conn,
        "s1",
        "user",
        "authentication middleware configuration",
    );
    insert_chunk_with_embedding(
        &conn,
        1,
        "s1",
        "authentication middleware configuration — mount it before the route handlers",
        1709251200000,
    );

    let results = search(
        &conn,
        "authentication middleware",
        &SearchOptions::default(),
    )
    .unwrap();

    assert_eq!(results.len(), 1);
    let excerpt = &results[0].excerpt;
    assert!(
        excerpt.contains("mount it before the route handlers"),
        "operator query must return the full containing chunk: {excerpt:?}"
    );
}

#[test]
fn test_fts_hit_without_chunk_falls_back_to_snippet() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
    insert_message(&conn, "s1", "user", "authentication flow discussion");

    let results = search(&conn, "authentication", &SearchOptions::default()).unwrap();

    assert_eq!(results.len(), 1);
    let excerpt = &results[0].excerpt;
    assert!(
        excerpt.contains("**"),
        "fallback must be the FTS snippet (** markers), not a chunk: {excerpt:?}"
    );
}

#[test]
fn test_fts_hit_selects_highest_ranked_message_chunk() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
    insert_message(
        &conn,
        "s1",
        "user",
        "authentication alpha padded with many extra filler words here to lower its relevance",
    );
    insert_message(&conn, "s1", "user", "authentication beta");
    insert_chunk_with_embedding(
        &conn,
        1,
        "s1",
        "authentication alpha padded with many extra filler words here to lower its relevance — ALPHA_CHUNK_BODY",
        1709251200000,
    );
    insert_chunk_with_embedding(
        &conn,
        2,
        "s1",
        "authentication beta — BETA_CHUNK_BODY",
        1709251200000,
    );

    let results = search(&conn, "authentication", &SearchOptions::default()).unwrap();

    assert_eq!(results.len(), 1);
    let excerpt = &results[0].excerpt;
    assert!(
        excerpt.contains("BETA_CHUNK_BODY"),
        "must select the highest-ranked message's chunk, not the first by rowid: {excerpt:?}"
    );
}

#[test]
fn test_fts_short_hit_ambiguous_chunk_falls_back_to_snippet() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
    insert_message(&conn, "s1", "user", "yes");
    insert_chunk_with_embedding(&conn, 1, "s1", "yes — COLLIDING_BODY_A", 1709251200000);
    insert_chunk_with_embedding(&conn, 2, "s1", "yes — COLLIDING_BODY_B", 1709251200000);

    let results = search(&conn, "yes", &SearchOptions::default()).unwrap();

    assert_eq!(results.len(), 1);
    let excerpt = &results[0].excerpt;
    assert!(
        excerpt.contains("**")
            && !excerpt.contains("COLLIDING_BODY_A")
            && !excerpt.contains("COLLIDING_BODY_B"),
        "ambiguous short-text hit must fall back to the match-centered snippet, \
         not either arbitrary colliding chunk: {excerpt:?}"
    );
}

#[test]
fn test_fts_hit_returns_unique_chunk_among_multiple() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
    insert_message(&conn, "s1", "user", "needle phrase");
    insert_chunk_with_embedding(&conn, 1, "s1", "needle phrase — HOME_BODY", 1709251200000);
    insert_chunk_with_embedding(&conn, 2, "s1", "unrelated topic NOISE_BODY", 1709251200000);

    let results = search(&conn, "needle", &SearchOptions::default()).unwrap();

    assert_eq!(results.len(), 1);
    let excerpt = &results[0].excerpt;
    assert!(
        excerpt.contains("HOME_BODY") && !excerpt.contains("**"),
        "count=1 must return the full home chunk, not over-suppress to a snippet: {excerpt:?}"
    );
}

#[test]
fn test_192a_short_hit_rowid_range_disambiguates_to_full_chunk() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
    insert_message(&conn, "s1", "user", "yes");
    insert_message(&conn, "s1", "user", "did it say yes or no");
    let yes_rowid = message_rowid(&conn, "s1", "yes");
    let other_rowid = message_rowid(&conn, "s1", "did it say yes or no");

    let top_hit_rowid: i64 = conn
        .query_row(
            "SELECT rowid FROM messages WHERE messages MATCH ?1 AND session_id = ?2 ORDER BY rank LIMIT 1",
            ("yes", "s1"),
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(
        top_hit_rowid, yes_rowid,
        "fixture precondition: the FTS top-hit for 'yes' is the 'yes' message"
    );

    insert_chunk_with_rowid_range(
        &conn,
        1,
        "s1",
        "yes — OWNER_BODY confirming the deploy",
        1709251200000,
        yes_rowid,
        yes_rowid,
    );
    insert_chunk_with_rowid_range(
        &conn,
        2,
        "s1",
        "did it say yes or no — COLLIDER_BODY",
        1709251200000,
        other_rowid,
        other_rowid,
    );

    let results = search(&conn, "yes", &SearchOptions::default()).unwrap();

    assert_eq!(results.len(), 1);
    let excerpt = &results[0].excerpt;
    assert!(
        excerpt.contains("OWNER_BODY") && !excerpt.contains("COLLIDER_BODY"),
        "rowid range must scope the hit to its owning chunk and return it full, \
         not degrade to a snippet under instr collision: {excerpt:?}"
    );
    assert!(
        !excerpt.contains("**"),
        "the disambiguated hit returns a full chunk, not the snippet fallback: {excerpt:?}"
    );
}

#[test]
fn test_192b_assistant_hit_at_range_hi_returns_full_chunk() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
    insert_message(&conn, "s1", "user", "what is the brightest object");
    insert_message(&conn, "s1", "assistant", "a quasar outshines its galaxy");
    let user_rowid = message_rowid(&conn, "s1", "what is the brightest object");
    let assistant_rowid = message_rowid(&conn, "s1", "a quasar outshines its galaxy");
    assert_eq!(
        assistant_rowid,
        user_rowid + 1,
        "fixture precondition: assistant rowid is the group's hi"
    );

    insert_chunk_with_rowid_range(
        &conn,
        1,
        "s1",
        "what is the brightest object\na quasar outshines its galaxy — QUASAR_BODY",
        1709251200000,
        user_rowid,
        assistant_rowid,
    );

    let results = search(&conn, "quasar", &SearchOptions::default()).unwrap();

    assert_eq!(results.len(), 1);
    let excerpt = &results[0].excerpt;
    assert!(
        excerpt.contains("QUASAR_BODY") && !excerpt.contains("**"),
        "an assistant hit at the range hi must return the full owning chunk; a \
         user-rowid-only link would miss it and fall to a snippet: {excerpt:?}"
    );
}

#[test]
fn test_192c_split_group_shared_range_falls_back_to_snippet() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
    insert_message(&conn, "s1", "user", "explain consensus");
    let user_rowid = message_rowid(&conn, "s1", "explain consensus");

    insert_chunk_with_rowid_range(
        &conn,
        1,
        "s1",
        "explain consensus — SPLIT_BODY_A part one",
        1709251200000,
        user_rowid,
        user_rowid,
    );
    insert_chunk_with_rowid_range(
        &conn,
        2,
        "s1",
        "explain consensus — SPLIT_BODY_B part two",
        1709251200000,
        user_rowid,
        user_rowid,
    );

    let results = search(&conn, "consensus", &SearchOptions::default()).unwrap();

    assert_eq!(results.len(), 1);
    let excerpt = &results[0].excerpt;
    assert!(
        excerpt.contains("**")
            && !excerpt.contains("SPLIT_BODY_A")
            && !excerpt.contains("SPLIT_BODY_B"),
        "a hit matching N sibling chunks must reject (COUNT>1) and return the \
         match-centered snippet, not an arbitrary sibling: {excerpt:?}"
    );
}

#[test]
fn test_192d_null_rowid_range_falls_back_to_instr_guard() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
    insert_message(&conn, "s1", "user", "legacy authentication question");
    insert_chunk_with_embedding(
        &conn,
        1,
        "s1",
        "legacy authentication question — LEGACY_BODY answered before #192",
        1709251200000,
    );

    let results = search(&conn, "authentication", &SearchOptions::default()).unwrap();

    assert_eq!(results.len(), 1);
    let excerpt = &results[0].excerpt;
    assert!(
        excerpt.contains("LEGACY_BODY") && !excerpt.contains("**"),
        "a NULL-rowid legacy chunk must still resolve via the instr fallback and \
         return the full chunk, preserving pre-#192 behavior: {excerpt:?}"
    );
}

#[test]
fn test_192g_indexed_chunk_resolves_full_chunk_via_rowid_join() {
    let (_dir, mut conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
    insert_message(&conn, "s1", "user", "define perihelion precisely");
    insert_message(
        &conn,
        "s1",
        "assistant",
        "the closest orbital point — APSIS_REPLY_BODY",
    );

    let stats = index_chunks(&mut conn, None).unwrap();
    assert_eq!(
        stats.chunks_created, 1,
        "one user+assistant pair → one chunk"
    );

    let results = search(&conn, "perihelion", &SearchOptions::default()).unwrap();

    assert_eq!(results.len(), 1);
    let excerpt = &results[0].excerpt;
    assert!(
        excerpt.contains("APSIS_REPLY_BODY") && !excerpt.contains("**"),
        "an index_chunks-written chunk must resolve to the full user+assistant body \
         via the rowid JOIN; an inverted lo/hi range would degrade to a snippet: \
         {excerpt:?}"
    );
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

    let now_ms = 1_750_000_000_000_i64;

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

    let now_ms = 1_750_000_000_000_i64;

    insert_session(&conn, "recent", "claude", "/proj", now_ms - MS_PER_DAY);
    insert_message(&conn, "recent", "user", "rust compiler optimization");
    insert_session(&conn, "old", "claude", "/proj", now_ms - 60 * MS_PER_DAY);
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
fn クエリ_file_が_and_で絞り込む() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
    insert_message(&conn, "s1", "user", "authentication token rotation");
    insert_session_file(&conn, "s1", "/proj/auth.rs");

    insert_session(&conn, "s2", "claude", "/proj", 1709251200000);
    insert_message(&conn, "s2", "user", "unrelated weather notes");
    insert_session_file(&conn, "s2", "/proj/auth.rs");

    insert_session(&conn, "s3", "claude", "/proj", 1709251200000);
    insert_message(&conn, "s3", "user", "authentication token rotation");
    insert_session_file(&conn, "s3", "/proj/unrelated.rs");

    let results = search(
        &conn,
        "authentication",
        &SearchOptions {
            file: Some("/proj/auth.rs".to_owned()),
            ..Default::default()
        },
    )
    .unwrap();

    let ids: Vec<&str> = results
        .iter()
        .map(|r| r.session.session_id.as_str())
        .collect();
    assert_eq!(
        ids,
        vec!["s1"],
        "only the session matching both the query and --file must return: {ids:?}"
    );
}

#[test]
fn suffix_一致が機能する() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "hit", "claude", "/proj", 1709251200000);
    insert_message(&conn, "hit", "user", "authentication token rotation");
    insert_session_file(&conn, "hit", "/home/me/proj/src/a_b.rs");

    insert_session(&conn, "decoy", "claude", "/proj", 1709251200000);
    insert_message(&conn, "decoy", "user", "authentication token rotation");
    insert_session_file(&conn, "decoy", "/home/me/proj/src/aXb.rs");

    let results = search(
        &conn,
        "authentication",
        &SearchOptions {
            file: Some("a_b.rs".to_owned()),
            ..Default::default()
        },
    )
    .unwrap();

    let ids: Vec<&str> = results
        .iter()
        .map(|r| r.session.session_id.as_str())
        .collect();
    assert_eq!(
        ids,
        vec!["hit"],
        "suffix match must return the literal-underscore path only, not the wildcard decoy: {ids:?}"
    );
}

#[test]
fn クエリなし_file_は新しい順に返る() {
    let (_dir, conn) = setup_test_db();
    let now_ms = 1_750_000_000_000_i64;

    insert_session(&conn, "same_old", "claude", "/proj", 1000);
    insert_message(&conn, "same_old", "user", "alpha notes");
    insert_session_file(&conn, "same_old", "/proj/shared.rs");

    insert_session(&conn, "same_new", "claude", "/proj", 3000);
    insert_message(&conn, "same_new", "user", "beta notes");
    insert_session_file(&conn, "same_new", "/proj/shared.rs");

    insert_session(&conn, "same_mid", "claude", "/proj", 2000);
    insert_message(&conn, "same_mid", "user", "gamma notes");
    insert_session_file(&conn, "same_mid", "/proj/shared.rs");

    insert_session(&conn, "decoy", "claude", "/proj", 4000);
    insert_message(&conn, "decoy", "user", "delta notes");
    insert_session_file(&conn, "decoy", "/proj/unrelated.rs");

    let results = search_with_embedder(
        &conn,
        None,
        &SearchOptions {
            file: Some("/proj/shared.rs".to_owned()),
            now_ms: Some(now_ms),
            ..Default::default()
        },
        None,
    )
    .unwrap()
    .results;

    let ids: Vec<&str> = results
        .iter()
        .map(|r| r.session.session_id.as_str())
        .collect();
    assert_eq!(
        ids,
        vec!["same_new", "same_mid", "same_old"],
        "query-less --file must return the file's sessions in timestamp-descending order, \
         excluding the other-path decoy: {ids:?}"
    );
    assert!(
        results.iter().all(|r| r.excerpt.is_empty()),
        "query-less --file list mode carries no excerpt (MATCH bypassed, no FTS/vector snippet resolution)"
    );
}

#[test]
fn file_リストは_null_行を落として_incomplete_を立てる() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "ok", "claude", "/proj", 2000);
    insert_session_file(&conn, "ok", "/proj/shared.rs");
    conn.execute(
        "INSERT INTO sessions (session_id, source, file_path, project, slug, timestamp, mtime) \
         VALUES ('null_src', NULL, '', '/proj', 'null_src', 1000, 0.0)",
        [],
    )
    .unwrap();
    insert_session_file(&conn, "null_src", "/proj/shared.rs");

    let outcome = search_with_embedder(
        &conn,
        None,
        &SearchOptions {
            file: Some("/proj/shared.rs".to_owned()),
            ..Default::default()
        },
        None,
    )
    .unwrap();

    let ids: Vec<&str> = outcome
        .results
        .iter()
        .map(|r| r.session.session_id.as_str())
        .collect();
    assert_eq!(
        ids,
        vec!["ok"],
        "the NULL-source row must be dropped, not abort the whole list: {ids:?}"
    );
    assert!(
        outcome.results_incomplete,
        "a dropped row must be flagged as results_incomplete, not presented as complete"
    );
}

#[test]
fn file_リストは_未知_source_行を落として_incomplete_を立てる() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "ok", "claude", "/proj", 2000);
    insert_session_file(&conn, "ok", "/proj/shared.rs");
    insert_session(&conn, "mystery", "perplexity", "/proj", 1000);
    insert_session_file(&conn, "mystery", "/proj/shared.rs");

    let outcome = search_with_embedder(
        &conn,
        None,
        &SearchOptions {
            file: Some("/proj/shared.rs".to_owned()),
            ..Default::default()
        },
        None,
    )
    .unwrap();

    let ids: Vec<&str> = outcome
        .results
        .iter()
        .map(|r| r.session.session_id.as_str())
        .collect();
    assert_eq!(
        ids,
        vec!["ok"],
        "the unknown-source row must be dropped: {ids:?}"
    );
    assert!(
        outcome.results_incomplete,
        "an unknown-source drop must be flagged as results_incomplete"
    );
}

#[test]
fn file_リストは_落ちた行の分まで_limit_を埋める() {
    let (_dir, conn) = setup_test_db();
    conn.execute(
        "INSERT INTO sessions (session_id, source, file_path, project, slug, timestamp, mtime) \
         VALUES ('null_src', NULL, '', '/proj', 'null_src', 2000, 0.0)",
        [],
    )
    .unwrap();
    insert_session_file(&conn, "null_src", "/proj/shared.rs");
    insert_session(&conn, "ok", "claude", "/proj", 1000);
    insert_session_file(&conn, "ok", "/proj/shared.rs");

    let outcome = search_with_embedder(
        &conn,
        None,
        &SearchOptions {
            file: Some("/proj/shared.rs".to_owned()),
            limit: 1,
            ..Default::default()
        },
        None,
    )
    .unwrap();

    let ids: Vec<&str> = outcome
        .results
        .iter()
        .map(|r| r.session.session_id.as_str())
        .collect();
    assert_eq!(
        ids,
        vec!["ok"],
        "a dropped newest row must not under-fill the limit: {ids:?}"
    );
    assert!(outcome.results_incomplete);
}

#[test]
fn file_指定は_session_files_不在の_db_で_index_案内エラーになる() {
    let (_dir, conn) = setup_test_db();
    conn.execute_batch("DROP TABLE session_files;").unwrap();
    let opts = SearchOptions {
        file: Some("/proj/a.rs".to_owned()),
        ..Default::default()
    };

    let Err(query_less) = search_with_embedder(&conn, None, &opts, None) else {
        panic!("query-less --file on a legacy DB must error");
    };
    assert!(
        query_less.to_string().contains("recall index"),
        "query-less --file on a legacy DB must name `recall index`, got: {query_less}"
    );

    let Err(with_query) = search_with_embedder(&conn, "auth", &opts, None) else {
        panic!("query + --file on a legacy DB must error");
    };
    assert!(
        with_query.to_string().contains("recall index"),
        "query + --file on a legacy DB must name `recall index`, got: {with_query}"
    );
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
    let candidate_limit = limit * CANDIDATE_LIMIT_MULTIPLIER;

    for i in 0..20 {
        let sid = format!("old-{i:02}");
        let ts = now_ms - (365 - i) * MS_PER_DAY;
        insert_session(&conn, &sid, "claude", "/proj", ts);
        insert_message(&conn, &sid, "user", &format!("型安全の議論 パート{i}"));
    }

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
fn test_score_and_sort_prefers_recent_on_equal_rank() {
    let now_ms = 1_750_000_000_000_i64;
    let old_ts = now_ms - 90 * MS_PER_DAY;
    let candidates = vec![
        (-1.0, make_result("old", old_ts)),
        (-1.0, make_result("recent", now_ms)),
    ];
    let results = score_and_sort(candidates, now_ms, 10);
    assert_eq!(results[0].session.session_id, "recent");
    assert_eq!(results[1].session.session_id, "old");
}

#[test]
fn test_score_and_sort_sorts_none_timestamp_last_on_equal_rank() {
    let now_ms = 1_750_000_000_000_i64;
    let mut no_ts = make_result("no-ts", now_ms);
    no_ts.session.timestamp = None;
    let candidates = vec![(-1.0, no_ts), (-1.0, make_result("dated", now_ms))];
    let results = score_and_sort(candidates, now_ms, 10);
    assert_eq!(results[0].session.session_id, "dated");
    assert_eq!(results[1].session.session_id, "no-ts");
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
    use crate::indexer::{IndexOptions, index_chunks, index_from_dirs};

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
        true,
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

    assert_eq!(index_chunks(&mut conn, None).unwrap().chunks_created, 2);

    let stats2 = index_from_dirs(
        &mut conn,
        &IndexOptions {
            force: false,
            claude_dir: &claude_dir,
            codex_dir: &codex_dir,
        },
        true,
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

    let results = search_with_embedder(&conn, "authentication", &SearchOptions::default(), None)
        .unwrap()
        .results;

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].session.session_id, "s1");
}

#[test]
fn test_hybrid_search_merges_fts_and_vector() {
    let (_dir, conn) = setup_test_db();
    let now_ms = 1_750_000_000_000_i64;

    insert_session(&conn, "s1", "claude", "/proj", now_ms);
    insert_message(&conn, "s1", "user", "authentication flow discussion");

    insert_session(&conn, "s2", "claude", "/proj", now_ms);
    insert_message(&conn, "s2", "user", "unrelated topic about weather");

    insert_chunk_with_embedding(&conn, 1, "s2", "authentication", now_ms);

    assert!(has_vec_data(&conn).unwrap());

    let embedder = MockEmbedder::new();
    let outcome = search_with_embedder(
        &conn,
        "authentication",
        &SearchOptions {
            now_ms: Some(now_ms),
            ..Default::default()
        },
        Some(&embedder),
    )
    .unwrap();
    assert!(
        !outcome.vec_degraded,
        "a successful vector search must not flag degradation"
    );
    let results = outcome.results;

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

    let s2 = results
        .iter()
        .find(|r| r.session.session_id == "s2")
        .unwrap();
    assert!(
        s2.excerpt.contains("authentication"),
        "vector-only excerpt must contain matched chunk content, got: {:?}",
        s2.excerpt
    );
}

#[test]
fn test_short_query_runs_vector_search_when_fts_has_no_trigram_term() {
    let (_dir, conn) = setup_test_db();
    let now_ms = 1_750_000_000_000_i64;

    insert_session(&conn, "s1", "claude", "/proj", now_ms);
    insert_message(&conn, "s1", "user", "completely separate gardening notes");

    insert_session(&conn, "s2", "claude", "/proj", now_ms);
    insert_chunk_with_embedding(&conn, 1, "s2", "ui", now_ms);

    let matched = prepare_match_query(
        &conn,
        "ui",
        "messages_vocab",
        &QueryNormalizationConfig::disabled(),
    )
    .unwrap();
    assert!(
        clean_for_trigram(&matched).is_none(),
        "test precondition: short query must yield no trigram FTS query"
    );

    let opts = SearchOptions {
        now_ms: Some(now_ms),
        ..Default::default()
    };

    let fts_only = search(&conn, "ui", &opts).unwrap();
    assert!(
        fts_only.is_empty(),
        "FTS-only short query must return no results, got: {:?}",
        fts_only
            .iter()
            .map(|r| r.session.session_id.as_str())
            .collect::<Vec<_>>()
    );

    let embedder = MockEmbedder::new();
    let results = search_with_embedder(&conn, "ui", &opts, Some(&embedder))
        .unwrap()
        .results;
    let ids: Vec<&str> = results
        .iter()
        .map(|r| r.session.session_id.as_str())
        .collect();
    assert!(
        ids.contains(&"s2"),
        "short query must reach vector search and find s2, got: {ids:?}"
    );

    let s2 = results
        .iter()
        .find(|r| r.session.session_id == "s2")
        .unwrap();
    assert!(
        s2.excerpt.contains("ui"),
        "vector-only excerpt must contain chunk content, got: {:?}",
        s2.excerpt
    );
}

#[test]
fn test_empty_and_whitespace_queries_return_empty_even_with_embedder_and_vec_data() {
    let (_dir, conn) = setup_test_db();
    let now_ms = 1_750_000_000_000_i64;

    insert_session(&conn, "s1", "claude", "/proj", now_ms);
    insert_chunk_with_embedding(&conn, 1, "s1", "some indexed content", now_ms);

    let opts = SearchOptions {
        now_ms: Some(now_ms),
        ..Default::default()
    };
    let embedder = MockEmbedder::new();

    for query in ["", "   "] {
        let results = search_with_embedder(&conn, query, &opts, Some(&embedder))
            .unwrap()
            .results;
        assert!(
            results.is_empty(),
            "query {query:?} must return no results, got: {:?}",
            results
                .iter()
                .map(|r| r.session.session_id.as_str())
                .collect::<Vec<_>>()
        );
    }

    let punct = search_with_embedder(&conn, "...", &opts, Some(&embedder))
        .unwrap()
        .results;
    assert!(
        !punct.is_empty(),
        "non-empty punctuation-only query must reach vector search, not return empty"
    );
}

#[test]
fn test_vector_only_excerpt_uses_closest_chunk() {
    let (_dir, conn) = setup_test_db();
    let now_ms = 1_750_000_000_000_i64;

    insert_session(&conn, "s1", "claude", "/proj", now_ms);
    insert_message(&conn, "s1", "user", "authentication flow discussion");

    insert_session(&conn, "s2", "claude", "/proj", now_ms);
    insert_message(&conn, "s2", "user", "unrelated weather topic");
    insert_chunk_with_embedding(&conn, 1, "s2", "totally unrelated weather forecast", now_ms);
    insert_chunk_with_embedding(&conn, 2, "s2", "authentication", now_ms);

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
    .unwrap()
    .results;

    let s2 = results
        .iter()
        .find(|r| r.session.session_id == "s2")
        .unwrap();
    assert!(
        s2.excerpt.contains("authentication"),
        "excerpt must come from the closest chunk (chunk 2), got: {:?}",
        s2.excerpt
    );
    assert!(
        !s2.excerpt.contains("weather"),
        "excerpt must be the MIN-distance chunk, not an arbitrary one, got: {:?}",
        s2.excerpt
    );
}

#[test]
fn test_hybrid_recency_boost_favors_newer() {
    let (_dir, conn) = setup_test_db();
    let now_ms = 1_750_000_000_000_i64;

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
    .unwrap()
    .results;

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

    insert_chunk_with_embedding(&conn, 1, "s1", "content", now_ms);

    let embedder = MockEmbedder::failing_after(0);
    let outcome = search_with_embedder(
        &conn,
        "authentication",
        &SearchOptions {
            now_ms: Some(now_ms),
            ..Default::default()
        },
        Some(&embedder),
    )
    .unwrap();

    assert_eq!(outcome.results.len(), 1);
    assert_eq!(outcome.results[0].session.session_id, "s1");
    assert!(
        outcome.vec_degraded,
        "a runtime vector-search failure must set vec_degraded even when FTS yields hits"
    );
}

#[test]
fn test_runtime_vector_failure_with_no_fts_match_flags_degraded_on_empty() {
    let (_dir, conn) = setup_test_db();
    let now_ms = 1_750_000_000_000_i64;

    insert_session(&conn, "s1", "claude", "/proj", now_ms);
    insert_message(&conn, "s1", "user", "totally unrelated weather forecast");
    insert_chunk_with_embedding(&conn, 1, "s1", "weather forecast", now_ms);

    let embedder = MockEmbedder::failing_after(0);
    let outcome = search_with_embedder(
        &conn,
        "authentication",
        &SearchOptions {
            now_ms: Some(now_ms),
            ..Default::default()
        },
        Some(&embedder),
    )
    .unwrap();

    assert!(
        outcome.results.is_empty(),
        "FTS finds nothing and the vector leg failed, so the result is empty"
    );
    assert!(
        outcome.vec_degraded,
        "an empty result from a runtime vector failure must still flag degradation (#204)"
    );
}

#[test]
fn test_runtime_vector_failure_with_unknown_source_hits_flags_degraded_on_empty() {
    let (_dir, conn) = setup_test_db();
    let now_ms = 1_750_000_000_000_i64;

    insert_session(&conn, "weird", "gemini", "/proj", now_ms);
    insert_message(&conn, "weird", "user", "authentication flow design");
    insert_chunk_with_embedding(&conn, 1, "weird", "authentication flow", now_ms);

    let embedder = MockEmbedder::failing_after(0);
    let outcome = search_with_embedder(
        &conn,
        "authentication",
        &SearchOptions {
            now_ms: Some(now_ms),
            ..Default::default()
        },
        Some(&embedder),
    )
    .unwrap();

    assert!(
        outcome.results.is_empty(),
        "the only FTS hit had an unknown source and was pruned, so the result is empty"
    );
    assert!(
        outcome.vec_degraded,
        "the pruned-to-empty early-return must still flag the runtime vector failure (#204)"
    );
}

#[test]
fn test_vec_probe_error_flags_vec_degraded_falls_back_to_fts() {
    let (_dir, conn) = setup_test_db();
    let now_ms = 1_750_000_000_000_i64;

    insert_session(&conn, "s1", "claude", "/proj", now_ms);
    insert_message(&conn, "s1", "user", "authentication flow discussion");
    conn.execute_batch("DROP TABLE vec_chunks;").unwrap();

    let embedder = MockEmbedder::new();
    let outcome = search_with_embedder(
        &conn,
        "authentication",
        &SearchOptions {
            now_ms: Some(now_ms),
            ..Default::default()
        },
        Some(&embedder),
    )
    .unwrap();

    assert_eq!(
        outcome.results.len(),
        1,
        "FTS still answers from the messages table"
    );
    assert!(
        outcome.vec_degraded,
        "a vec_chunks probe error must flag the semantic leg as degraded (#249 RC-001)"
    );
}

#[test]
fn test_empty_vec_table_with_embedder_is_not_degraded() {
    let (_dir, conn) = setup_test_db();
    let now_ms = 1_750_000_000_000_i64;

    insert_session(&conn, "s1", "claude", "/proj", now_ms);
    insert_message(&conn, "s1", "user", "authentication flow discussion");

    let embedder = MockEmbedder::new();
    let outcome = search_with_embedder(
        &conn,
        "authentication",
        &SearchOptions {
            now_ms: Some(now_ms),
            ..Default::default()
        },
        Some(&embedder),
    )
    .unwrap();

    assert_eq!(outcome.results.len(), 1);
    assert!(
        !outcome.vec_degraded,
        "an empty vec table is the normal not-embedded state, never degraded"
    );
    assert!(!outcome.results_incomplete);
}

#[test]
fn test_metadata_prune_flags_results_incomplete_without_vec_degraded() {
    let (_dir, conn) = setup_test_db();
    let now_ms = 1_750_000_000_000_i64;

    insert_session(&conn, "s1", "claude", "/proj", now_ms);
    insert_message(&conn, "s1", "user", "authentication flow design");
    insert_chunk_with_embedding(&conn, 1, "s1", "authentication flow", now_ms);
    insert_session(&conn, "weird", "gemini", "/proj", now_ms);
    insert_message(&conn, "weird", "user", "authentication retry logic");
    insert_chunk_with_embedding(&conn, 2, "weird", "authentication retry", now_ms);

    let embedder = MockEmbedder::new();
    let outcome = search_with_embedder(
        &conn,
        "authentication",
        &SearchOptions {
            now_ms: Some(now_ms),
            ..Default::default()
        },
        Some(&embedder),
    )
    .unwrap();

    assert!(
        !outcome.vec_degraded,
        "the vector leg succeeded, so vec_degraded must stay false"
    );
    assert!(
        outcome.results_incomplete,
        "a pruned unknown-source match must flag results_incomplete (#249 RC-003)"
    );
    assert!(
        outcome
            .results
            .iter()
            .all(|r| r.session.session_id != "weird"),
        "the unknown-source session must not appear in results"
    );
}

#[test]
fn test_fts_only_metadata_prune_flags_results_incomplete() {
    let (_dir, conn) = setup_test_db();
    let now_ms = 1_750_000_000_000_i64;

    insert_session(&conn, "s1", "claude", "/proj", now_ms);
    insert_message(&conn, "s1", "user", "authentication flow design");
    insert_session(&conn, "weird", "gemini", "/proj", now_ms);
    insert_message(&conn, "weird", "user", "authentication retry logic");

    let outcome = search_with_embedder(
        &conn,
        "authentication",
        &SearchOptions {
            now_ms: Some(now_ms),
            ..Default::default()
        },
        None,
    )
    .unwrap();

    assert!(
        outcome.results_incomplete,
        "the FTS-only path must also flag a pruned unknown-source match (#249 RC-003)"
    );
    assert!(
        !outcome.vec_degraded,
        "no embedder, so no vector leg to degrade"
    );
    assert_eq!(outcome.results.len(), 1);
}

fn insert_null_session_id_message(conn: &Connection, text: &str) {
    conn.execute(
        "INSERT INTO messages (session_id, role, text) VALUES (NULL, 'user', ?1)",
        rusqlite::params![text],
    )
    .unwrap();
}

#[test]
fn test_undeserializable_candidate_row_reports_skip() {
    let (_dir, conn) = setup_test_db();
    let now_ms = 1_750_000_000_000_i64;
    insert_session(&conn, "s1", "claude", "/proj", now_ms);
    insert_message(&conn, "s1", "user", "authentication flow");
    insert_null_session_id_message(&conn, "authentication token");

    let (ranked, skipped) = find_candidate_sessions(
        &conn,
        &SearchOptions {
            now_ms: Some(now_ms),
            include_automated: true,
            ..Default::default()
        },
        "authentication",
        now_ms,
    )
    .unwrap();

    assert_eq!(
        skipped, 1,
        "the NULL-session_id row must be counted as skipped"
    );
    assert_eq!(ranked.len(), 1, "the one valid candidate survives");
    assert_eq!(ranked[0].0, "s1");
}

#[test]
fn test_skipped_candidate_row_flags_results_incomplete() {
    let (_dir, conn) = setup_test_db();
    let now_ms = 1_750_000_000_000_i64;
    insert_session(&conn, "s1", "claude", "/proj", now_ms);
    insert_message(&conn, "s1", "user", "authentication flow");
    insert_null_session_id_message(&conn, "authentication token");

    let outcome = search_with_embedder(
        &conn,
        "authentication",
        &SearchOptions {
            now_ms: Some(now_ms),
            include_automated: true,
            ..Default::default()
        },
        None,
    )
    .unwrap();

    assert!(
        outcome.results_incomplete,
        "a skipped FTS candidate row must flag results_incomplete (#249 RC-002)"
    );
    assert!(
        !outcome.vec_degraded,
        "no embedder, no vector leg to degrade"
    );
    assert_eq!(outcome.results.len(), 1);
}

#[test]
fn test_find_candidate_sessions_reports_no_skips_on_clean_rows() {
    let (_dir, conn) = setup_test_db();
    let now_ms = 1_750_000_000_000_i64;
    insert_session(&conn, "s1", "claude", "/proj", now_ms);
    insert_message(&conn, "s1", "user", "authentication flow design");

    let (ranked, skipped) = find_candidate_sessions(
        &conn,
        &SearchOptions {
            now_ms: Some(now_ms),
            ..Default::default()
        },
        "authentication",
        now_ms,
    )
    .unwrap();

    assert_eq!(skipped, 0, "clean rows skip nothing");
    assert_eq!(ranked.len(), 1);
}

fn hybrid_search_ids(
    conn: &Connection,
    embedder: &MockEmbedder,
    opts: &SearchOptions,
) -> Vec<String> {
    search_with_embedder(conn, "authentication", opts, Some(embedder))
        .unwrap()
        .results
        .into_iter()
        .map(|r| r.session.session_id)
        .collect()
}

fn set_session_type(conn: &Connection, sid: &str, session_type: &str) {
    conn.execute(
        "UPDATE sessions SET session_type = ?2 WHERE session_id = ?1",
        rusqlite::params![sid, session_type],
    )
    .unwrap();
}

fn fts_ids(conn: &Connection, opts: &SearchOptions) -> Vec<String> {
    search(conn, "authentication", opts)
        .unwrap()
        .into_iter()
        .map(|r| r.session.session_id)
        .collect()
}

#[test]
fn test_default_search_excludes_automated_keeps_null() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "auto", "claude", "/proj", 1709251200000);
    insert_message(&conn, "auto", "user", "authentication flow");
    set_session_type(&conn, "auto", "automated");
    insert_session(&conn, "null", "claude", "/proj", 1709251200000);
    insert_message(&conn, "null", "user", "authentication flow");

    let ids = fts_ids(&conn, &SearchOptions::default());

    assert!(
        !ids.contains(&"auto".to_owned()),
        "automated session must be excluded by default: {ids:?}"
    );
    assert!(
        ids.contains(&"null".to_owned()),
        "NULL session_type must stay visible (interactive): {ids:?}"
    );
}

#[test]
fn test_include_automated_surfaces_automated() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "auto", "claude", "/proj", 1709251200000);
    insert_message(&conn, "auto", "user", "authentication flow");
    set_session_type(&conn, "auto", "automated");

    let ids = fts_ids(
        &conn,
        &SearchOptions {
            include_automated: true,
            ..Default::default()
        },
    );

    assert!(
        ids.contains(&"auto".to_owned()),
        "include_automated must surface automated: {ids:?}"
    );
}

#[test]
fn test_default_hybrid_excludes_automated_on_vec_path() {
    let (_dir, conn) = setup_test_db();
    let now_ms = 1_750_000_000_000_i64;
    let embedder = MockEmbedder::new();
    insert_session(&conn, "keep", "claude", "/proj", now_ms);
    insert_message(&conn, "keep", "user", "authentication flow discussion");
    insert_session(&conn, "auto", "claude", "/proj", now_ms);
    insert_chunk_with_embedding(&conn, 1, "auto", "authentication", now_ms);
    set_session_type(&conn, "auto", "automated");

    let ids = hybrid_search_ids(
        &conn,
        &embedder,
        &SearchOptions {
            now_ms: Some(now_ms),
            ..Default::default()
        },
    );

    assert!(
        !ids.contains(&"auto".to_owned()),
        "automated vec-only session must be excluded by default: {ids:?}"
    );
    assert!(
        ids.contains(&"keep".to_owned()),
        "non-automated session must remain: {ids:?}"
    );
}

#[test]
fn test_hybrid_exclude_current_drops_session_from_both_paths() {
    let (_dir, conn) = setup_test_db();
    let now_ms = 1_750_000_000_000_i64;
    let embedder = MockEmbedder::new();

    assert_filter_symmetric(
        || {
            insert_session(&conn, "keep", "claude", "/proj", now_ms);
            insert_message(&conn, "keep", "user", "authentication flow discussion");
            insert_session(&conn, "cur", "claude", "/proj", now_ms);
            insert_message(&conn, "cur", "user", "authentication flow discussion");
            insert_chunk_with_embedding(&conn, 1, "cur", "authentication", now_ms);
            ("keep".to_owned(), "cur".to_owned())
        },
        || {
            hybrid_search_ids(
                &conn,
                &embedder,
                &SearchOptions {
                    current: CurrentSession::Exclude("cur".to_owned()),
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
fn test_hybrid_file_filter_excludes_vec_only_mismatch() {
    let (_dir, conn) = setup_test_db();
    let now_ms = 1_750_000_000_000_i64;
    let embedder = MockEmbedder::new();

    assert_filter_symmetric(
        || {
            insert_session(&conn, "keep", "claude", "/proj", now_ms);
            insert_message(&conn, "keep", "user", "authentication flow discussion");
            insert_session_file(&conn, "keep", "/proj/auth.rs");
            insert_session(&conn, "other", "claude", "/proj", now_ms);
            insert_chunk_with_embedding(&conn, 1, "other", "authentication", now_ms);
            insert_session_file(&conn, "other", "/proj/unrelated.rs");
            ("keep".to_owned(), "other".to_owned())
        },
        || {
            hybrid_search_ids(
                &conn,
                &embedder,
                &SearchOptions {
                    file: Some("/proj/auth.rs".to_owned()),
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
fn test_hybrid_only_current_keeps_only_invoking_session() {
    let (_dir, conn) = setup_test_db();
    let now_ms = 1_750_000_000_000_i64;
    let embedder = MockEmbedder::new();

    assert_filter_symmetric(
        || {
            insert_session(&conn, "cur", "claude", "/proj", now_ms);
            insert_message(&conn, "cur", "user", "authentication flow discussion");
            insert_session(&conn, "keep", "claude", "/proj", now_ms);
            insert_message(&conn, "keep", "user", "authentication flow discussion");
            insert_chunk_with_embedding(&conn, 1, "keep", "authentication", now_ms);
            ("cur".to_owned(), "keep".to_owned())
        },
        || {
            hybrid_search_ids(
                &conn,
                &embedder,
                &SearchOptions {
                    current: CurrentSession::Only("cur".to_owned()),
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
    .unwrap()
    .results;

    assert_eq!(
        results.len(),
        1,
        "s_valid must fill the limit slot even if s_unknown ranks above it"
    );
    assert_eq!(results[0].session.session_id, "s_valid");
}

#[test]
fn test_search_cjk_short_term_with_special_chars() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
    insert_message(
        &conn,
        "s1",
        "user",
        "方針：まとめ 方針*概要 方針（案） ディレクトリ整理の方針",
    );

    let results = search(&conn, "ディレクトリ整理 方針", &SearchOptions::default())
        .expect("CJK short term + long term should not cause FTS5 error");
    assert_eq!(results.len(), 1);
}

#[test]
fn segment_script_boundaries_splits_multi_script_run() {
    assert_eq!(
        segment_script_boundaries("textlint日本語ドキュメント校正"),
        "textlint 日本語 ドキュメント 校正"
    );
    assert_eq!(segment_script_boundaries("校正する"), "校正 する");
    assert_eq!(segment_script_boundaries("Reactのhooks"), "React の hooks");
}

#[test]
fn segment_script_boundaries_keeps_ascii_alnum_whole() {
    for q in ["utf8", "v2", "sha256", "h264", "fix the bug"] {
        assert_eq!(
            segment_script_boundaries(q),
            q,
            "ASCII/space query must be byte-identical (precision guard, AC2)"
        );
    }
}

#[test]
fn segment_script_boundaries_is_noop_on_single_script_and_quotes() {
    assert_eq!(
        segment_script_boundaries("機械学習日本語処理"),
        "機械学習日本語処理"
    );
    assert_eq!(
        segment_script_boundaries("日本語 ドキュメント"),
        "日本語 ドキュメント"
    );
    assert_eq!(
        segment_script_boundaries("\"日本語ドキュメント\""),
        "\"日本語ドキュメント\""
    );
}

#[test]
fn degraded_no_space_cjk_query_reaches_session() {
    let (_dir, conn) = setup_test_db();
    let now_ms = 1_750_000_000_000_i64;
    insert_session(&conn, "s1", "claude", "/proj", now_ms);
    insert_message(
        &conn,
        "s1",
        "user",
        "textlintで日本語ドキュメントを校正する",
    );
    insert_session(&conn, "s2", "claude", "/proj", now_ms);
    insert_message(&conn, "s2", "user", "completely unrelated gardening notes");

    let opts = SearchOptions {
        now_ms: Some(now_ms),
        ..Default::default()
    };
    let ids = |q: &str| -> Vec<String> {
        search(&conn, q, &opts)
            .unwrap()
            .into_iter()
            .map(|r| r.session.session_id)
            .collect()
    };

    let no_space = ids("textlint日本語ドキュメント校正");
    assert!(
        no_space.iter().any(|s| s == "s1"),
        "degraded no-space query must reach s1, got: {no_space:?}"
    );
    assert!(
        !no_space.iter().any(|s| s == "s2"),
        "no-space query must not match the unrelated session, got: {no_space:?}"
    );

    let spaced = ids("textlint 日本語 ドキュメント 校正");
    assert_eq!(
        no_space, spaced,
        "no-space query must reach parity with the space-separated form (AC1)"
    );
}
#[test]
fn segment_falls_back_to_raw_when_no_anchor_term() {
    let (_dir, conn) = setup_test_db();
    let now_ms = 1_750_000_000_000_i64;
    insert_session(&conn, "s1", "claude", "/proj", now_ms);
    insert_message(&conn, "s1", "user", "これを校正する手順をまとめた");
    let following = "案版中後前用例集表記号点字形面";
    for (i, c) in following.chars().enumerate() {
        insert_session(&conn, &format!("k{i}"), "claude", "/proj", now_ms);
        insert_message(
            &conn,
            &format!("k{i}"),
            "user",
            &format!("校正{c}についてのメモ"),
        );
        insert_session(&conn, &format!("u{i}"), "claude", "/proj", now_ms);
        insert_message(
            &conn,
            &format!("u{i}"),
            "user",
            &format!("する{c}に関する記録"),
        );
    }
    let opts = SearchOptions {
        now_ms: Some(now_ms),
        ..Default::default()
    };
    let r = search(&conn, "校正する", &opts).unwrap();
    assert!(
        r.iter().any(|x| x.session.session_id == "s1"),
        "all-short query must keep raw-phrase recall (no MAX_COMBOS overflow), got: {:?}",
        r.iter()
            .map(|x| x.session.session_id.as_str())
            .collect::<Vec<_>>()
    );
}

#[test]
fn segment_prolonged_sound_mark_does_not_strand_hiragana() {
    assert_eq!(segment_script_boundaries("うーん"), "う ー ん");
    let (_dir, conn) = setup_test_db();
    let now_ms = 1_750_000_000_000_i64;
    insert_session(&conn, "s1", "claude", "/proj", now_ms);
    insert_message(&conn, "s1", "user", "それはうーんと悩ましい問題だ");
    let opts = SearchOptions {
        now_ms: Some(now_ms),
        ..Default::default()
    };
    let r = search(&conn, "うーん", &opts).unwrap();
    assert!(
        r.iter().any(|x| x.session.session_id == "s1"),
        "hiragana word with ー must keep raw recall, got: {:?}",
        r.iter()
            .map(|x| x.session.session_id.as_str())
            .collect::<Vec<_>>()
    );
}
