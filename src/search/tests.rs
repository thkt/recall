use std::fs;
use std::path::Path;

use crate::compat::assert_filter_symmetric;

use super::*;
use crate::db::{seed_session_file, setup_test_db};
use crate::embedder::MockEmbedder;
use crate::indexer::{IndexOptions, index_chunks, index_from_dirs};

const NOW_MS: i64 = 1_750_000_000_000;
const NONE: [&str; 0] = [];

fn insert_session(conn: &Connection, sid: &str, source: &str, project: &str, ts: i64) {
    conn.execute(
            "INSERT INTO sessions (session_id, source, file_path, project, slug, timestamp, mtime) VALUES (?1, ?2, '', ?3, ?4, ?5, 0.0)",
            rusqlite::params![sid, source, project, sid, ts],
        ).unwrap();
}

fn insert_undated_session(conn: &Connection, sid: &str) {
    conn.execute(
        "INSERT INTO sessions (session_id, source, file_path, project, slug, timestamp, mtime) \
         VALUES (?1, 'claude', '', '/proj', ?1, NULL, 0.0)",
        [sid],
    )
    .unwrap();
}

fn insert_sourceless_session(conn: &Connection, sid: &str, ts: i64) {
    conn.execute(
        "INSERT INTO sessions (session_id, source, file_path, project, slug, timestamp, mtime) \
         VALUES (?1, NULL, '', '/proj', ?1, ?2, 0.0)",
        rusqlite::params![sid, ts],
    )
    .unwrap();
}

/// Returns the message rowid, which a chunk's message range refers to.
fn insert_message(conn: &Connection, sid: &str, role: &str, text: &str) -> i64 {
    conn.execute(
        "INSERT INTO messages (session_id, role, text) VALUES (?1, ?2, ?3)",
        rusqlite::params![sid, role, text],
    )
    .unwrap();
    conn.last_insert_rowid()
}

fn insert_null_session_id_message(conn: &Connection, text: &str) {
    conn.execute(
        "INSERT INTO messages (session_id, role, text) VALUES (NULL, 'user', ?1)",
        rusqlite::params![text],
    )
    .unwrap();
}

fn embed_chunk(conn: &Connection, chunk_id: i64, text: &str) {
    let embedding = MockEmbedder::deterministic_vector(text);
    let embedding_bytes: &[u8] = f32_as_bytes(&embedding);
    conn.execute(
        "INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?1, ?2)",
        rusqlite::params![chunk_id, embedding_bytes],
    )
    .unwrap();
}

/// A chunk without a message range, as chunks were written before #192.
fn insert_chunk(conn: &Connection, chunk_id: i64, sid: &str, text: &str) {
    conn.execute(
        "INSERT INTO qa_chunks (id, session_id, content, timestamp) VALUES (?1, ?2, ?3, 0)",
        rusqlite::params![chunk_id, sid, text],
    )
    .unwrap();
    embed_chunk(conn, chunk_id, text);
}

/// A chunk built from the messages with rowids `lo..=hi`.
fn insert_chunk_over(conn: &Connection, chunk_id: i64, sid: &str, text: &str, lo: i64, hi: i64) {
    conn.execute(
        "INSERT INTO qa_chunks (id, session_id, content, timestamp, src_rowid_lo, src_rowid_hi) \
         VALUES (?1, ?2, ?3, 0, ?4, ?5)",
        rusqlite::params![chunk_id, sid, text, lo, hi],
    )
    .unwrap();
    embed_chunk(conn, chunk_id, text);
}

fn set_session_type(conn: &Connection, sid: &str, session_type: &str) {
    conn.execute(
        "UPDATE sessions SET session_type = ?2 WHERE session_id = ?1",
        rusqlite::params![sid, session_type],
    )
    .unwrap();
}

fn at_now() -> SearchOptions {
    SearchOptions {
        now_ms: Some(NOW_MS),
        ..SearchOptions::default()
    }
}

fn ids(results: &[SearchResult]) -> Vec<&str> {
    results
        .iter()
        .map(|r| r.session.session_id.as_str())
        .collect()
}

/// (session id, excerpt, chunk id) for each result, in result order.
fn found(results: &[SearchResult]) -> Vec<(&str, &str, Option<i64>)> {
    results
        .iter()
        .map(|r| {
            (
                r.session.session_id.as_str(),
                r.excerpt.as_str(),
                r.chunk_id,
            )
        })
        .collect()
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

#[test]
fn a_match_without_a_chunk_shows_a_highlighted_snippet() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
    insert_message(&conn, "s1", "user", "authentication flow discussion");

    let results = search(&conn, "authentication", &SearchOptions::default()).unwrap();

    assert_eq!(
        found(&results),
        [("s1", "**authentication** flow di...", None)]
    );
}

#[test]
fn a_chunk_without_a_message_range_is_found_by_the_message_text() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
    insert_message(&conn, "s1", "user", "legacy authentication question");
    insert_chunk(
        &conn,
        1,
        "s1",
        "legacy authentication question — LEGACY_BODY answered before #192",
    );

    let results = search(&conn, "authentication", &SearchOptions::default()).unwrap();

    assert_eq!(
        found(&results),
        [(
            "s1",
            "legacy authentication question — LEGACY_BODY answered before #192",
            None
        )]
    );
}

#[test]
fn a_multi_word_query_shows_the_chunk_holding_the_matched_message() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
    insert_message(
        &conn,
        "s1",
        "user",
        "authentication middleware configuration",
    );
    insert_chunk(
        &conn,
        1,
        "s1",
        "authentication middleware configuration — mount it before the route handlers",
    );

    let results = search(
        &conn,
        "authentication middleware",
        &SearchOptions::default(),
    )
    .unwrap();

    assert_eq!(
        found(&results),
        [(
            "s1",
            "authentication middleware configuration — mount it before the route handlers",
            None
        )]
    );
}

#[test]
fn the_excerpt_is_the_chunk_of_the_best_matching_message() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
    insert_message(
        &conn,
        "s1",
        "user",
        "authentication alpha padded with many extra filler words here to lower its relevance",
    );
    insert_message(&conn, "s1", "user", "authentication beta");
    insert_chunk(
        &conn,
        1,
        "s1",
        "authentication alpha padded with many extra filler words here to lower its relevance — ALPHA_CHUNK_BODY",
    );
    insert_chunk(&conn, 2, "s1", "authentication beta — BETA_CHUNK_BODY");

    let results = search(&conn, "authentication", &SearchOptions::default()).unwrap();

    assert_eq!(
        found(&results),
        [("s1", "authentication beta — BETA_CHUNK_BODY", None)]
    );
}

#[test]
fn the_one_chunk_containing_the_message_is_shown_in_full() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
    insert_message(&conn, "s1", "user", "needle phrase");
    insert_chunk(&conn, 1, "s1", "needle phrase — HOME_BODY");
    insert_chunk(&conn, 2, "s1", "unrelated topic NOISE_BODY");

    let results = search(&conn, "needle", &SearchOptions::default()).unwrap();

    assert_eq!(found(&results), [("s1", "needle phrase — HOME_BODY", None)]);
}

#[test]
fn a_message_found_in_two_chunks_shows_a_snippet_instead_of_either() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
    insert_message(&conn, "s1", "user", "yes");
    insert_chunk(&conn, 1, "s1", "yes — COLLIDING_BODY_A");
    insert_chunk(&conn, 2, "s1", "yes — COLLIDING_BODY_B");

    let results = search(&conn, "yes", &SearchOptions::default()).unwrap();

    assert_eq!(found(&results), [("s1", "**yes**", None)]);
}

#[test]
fn a_message_range_picks_the_owning_chunk_when_its_text_is_in_several() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
    let yes = insert_message(&conn, "s1", "user", "yes");
    let other = insert_message(&conn, "s1", "user", "did it say yes or no");
    insert_chunk_over(
        &conn,
        1,
        "s1",
        "yes — OWNER_BODY confirming the deploy",
        yes,
        yes,
    );
    insert_chunk_over(
        &conn,
        2,
        "s1",
        "did it say yes or no — COLLIDER_BODY",
        other,
        other,
    );

    let results = search(&conn, "yes", &SearchOptions::default()).unwrap();

    assert_eq!(
        found(&results),
        [("s1", "yes — OWNER_BODY confirming the deploy", Some(1))]
    );
}

#[test]
fn an_assistant_reply_at_the_end_of_a_range_shows_the_whole_chunk() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
    let question = insert_message(&conn, "s1", "user", "what is the brightest object");
    let reply = insert_message(&conn, "s1", "assistant", "a quasar outshines its galaxy");
    insert_chunk_over(
        &conn,
        1,
        "s1",
        "what is the brightest object\na quasar outshines its galaxy — QUASAR_BODY",
        question,
        reply,
    );

    let results = search(&conn, "quasar", &SearchOptions::default()).unwrap();

    assert_eq!(
        found(&results),
        [(
            "s1",
            "what is the brightest object\na quasar outshines its galaxy — QUASAR_BODY",
            Some(1)
        )]
    );
}

#[test]
fn a_message_split_across_chunks_shows_a_snippet() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
    let question = insert_message(&conn, "s1", "user", "explain consensus");
    insert_chunk_over(
        &conn,
        1,
        "s1",
        "explain consensus — SPLIT_BODY_A part one",
        question,
        question,
    );
    insert_chunk_over(
        &conn,
        2,
        "s1",
        "explain consensus — SPLIT_BODY_B part two",
        question,
        question,
    );

    let results = search(&conn, "consensus", &SearchOptions::default()).unwrap();

    assert_eq!(found(&results), [("s1", "explain **consensus**", None)]);
}

#[test]
fn chunks_built_by_the_indexer_show_the_question_and_reply() {
    let (_dir, mut conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
    insert_message(&conn, "s1", "user", "define perihelion precisely");
    insert_message(
        &conn,
        "s1",
        "assistant",
        "the closest orbital point — APSIS_REPLY_BODY",
    );
    index_chunks(&mut conn, None).unwrap();

    let results = search(&conn, "perihelion", &SearchOptions::default()).unwrap();

    assert_eq!(
        found(&results),
        [(
            "s1",
            "define perihelion precisely\nthe closest orbital point — APSIS_REPLY_BODY",
            Some(1)
        )]
    );
}

#[test]
fn each_session_gets_its_own_best_chunk_when_several_sessions_match() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", NOW_MS);
    insert_session(&conn, "s2", "claude", "/proj", NOW_MS - 365 * MS_PER_DAY);
    let s1_low = insert_message(
        &conn,
        "s1",
        "user",
        "authentication one padded with filler words to lower rank",
    );
    let s2_low = insert_message(
        &conn,
        "s2",
        "user",
        "authentication two padded with filler words to lower rank",
    );
    let s1_best = insert_message(&conn, "s1", "user", "authentication s1 best");
    let s2_best = insert_message(&conn, "s2", "user", "authentication s2 best");
    insert_chunk_over(
        &conn,
        1,
        "s1",
        "authentication one padded with filler words to lower rank — S1_LOW",
        s1_low,
        s1_low,
    );
    insert_chunk_over(
        &conn,
        2,
        "s2",
        "authentication two padded with filler words to lower rank — S2_LOW",
        s2_low,
        s2_low,
    );
    insert_chunk_over(
        &conn,
        3,
        "s1",
        "authentication s1 best — S1_BEST",
        s1_best,
        s1_best,
    );
    insert_chunk_over(
        &conn,
        4,
        "s2",
        "authentication s2 best — S2_BEST",
        s2_best,
        s2_best,
    );

    let results = search(&conn, "authentication", &at_now()).unwrap();

    assert_eq!(
        found(&results),
        [
            ("s1", "authentication s1 best — S1_BEST", Some(3)),
            ("s2", "authentication s2 best — S2_BEST", Some(4)),
        ]
    );
}

#[test]
fn snippet_fallback_comes_from_the_first_matching_message_of_the_session() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
    insert_session(&conn, "s2", "claude", "/proj", 1709251200000);
    insert_message(&conn, "s2", "user", "authentication other session");
    insert_message(
        &conn,
        "s1",
        "user",
        "authentication earliest padded with many extra filler words to lower its rank",
    );
    insert_message(&conn, "s1", "user", "authentication later");

    let results = search(&conn, "authentication", &SearchOptions::default()).unwrap();

    assert_eq!(
        found(&results),
        [
            ("s1", "**authentication** earlies...", None),
            ("s2", "**authentication** other s...", None),
        ]
    );
}

#[test]
fn a_query_matching_no_message_returns_nothing() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
    insert_message(&conn, "s1", "user", "hello world");

    let results = search(&conn, "nonexistent_term_xyz", &SearchOptions::default()).unwrap();

    assert_eq!(ids(&results), NONE);
}

#[test]
fn a_limit_of_zero_returns_nothing() {
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

    assert_eq!(ids(&results), NONE);
}

/// Three year-old sessions and one new one match "authentication". By text alone
/// they rank old-short, old-medium, new, old-long, since shorter messages score
/// higher. "old-short" scores 16% above "new", less than the 20% that recency
/// adds to a same-day session.
fn seed_new_session_ranked_third_by_text(conn: &Connection) {
    let year_ago = NOW_MS - 365 * MS_PER_DAY;
    insert_session(conn, "old-short", "claude", "/proj", year_ago);
    insert_message(conn, "old-short", "user", "authentication notes");
    insert_session(conn, "old-medium", "claude", "/proj", year_ago);
    insert_message(conn, "old-medium", "user", "authentication notes kept");
    insert_session(conn, "new", "claude", "/proj", NOW_MS);
    insert_message(conn, "new", "user", "authentication notes kept here");
    insert_session(conn, "old-long", "claude", "/proj", year_ago);
    insert_message(
        conn,
        "old-long",
        "user",
        "authentication notes kept here for later use",
    );
}

#[test]
fn recency_lifts_a_new_session_over_somewhat_better_older_text_matches() {
    let (_dir, conn) = setup_test_db();
    seed_new_session_ranked_third_by_text(&conn);

    let results = search(&conn, "authentication", &at_now()).unwrap();

    assert_eq!(
        ids(&results),
        ["new", "old-short", "old-medium", "old-long"]
    );
}

#[test]
fn sessions_that_match_equally_at_the_same_time_are_listed_in_session_id_order() {
    let (_dir, conn) = setup_test_db();
    for sid in ["m", "z", "a", "q"] {
        insert_session(&conn, sid, "claude", "/proj", NOW_MS);
        insert_message(&conn, sid, "user", "authentication flow");
    }
    // A limit of 1 reads three candidates, so the tie also decides which
    // session is left out before the final ranking.
    let one = SearchOptions {
        limit: 1,
        ..at_now()
    };

    assert_eq!(
        ids(&search(&conn, "authentication", &at_now()).unwrap()),
        ["a", "m", "q", "z"]
    );
    assert_eq!(ids(&search(&conn, "authentication", &one).unwrap()), ["a"]);
}

#[test]
fn the_new_session_takes_the_only_slot_even_when_it_ranks_third_by_text() {
    let (_dir, conn) = setup_test_db();
    seed_new_session_ranked_third_by_text(&conn);

    let results = search(
        &conn,
        "authentication",
        &SearchOptions {
            limit: 1,
            ..at_now()
        },
    )
    .unwrap();

    assert_eq!(ids(&results), ["new"]);
}

#[test]
fn a_session_without_a_timestamp_ranks_below_an_equally_matching_dated_one() {
    let (_dir, conn) = setup_test_db();
    insert_undated_session(&conn, "a-undated");
    insert_message(&conn, "a-undated", "user", "authentication flow");
    insert_session(
        &conn,
        "z-dated",
        "claude",
        "/proj",
        NOW_MS - 365 * MS_PER_DAY,
    );
    insert_message(&conn, "z-dated", "user", "authentication flow");

    let results = search(&conn, "authentication", &at_now()).unwrap();

    assert_eq!(ids(&results), ["z-dated", "a-undated"]);
}

#[test]
fn recency_lifts_a_new_session_over_a_somewhat_better_older_match_in_hybrid_search() {
    let (_dir, conn) = setup_test_db();
    // "old" ranks first in both legs, which puts it 1.7% ahead after fusion.
    // At 90 days "new" gains 2.5% from recency, enough to pass it; half is not.
    insert_session(&conn, "old", "claude", "/proj", NOW_MS - 365 * MS_PER_DAY);
    insert_message(&conn, "old", "user", "authentication");
    insert_chunk(&conn, 1, "old", "authentication");
    insert_session(&conn, "new", "claude", "/proj", NOW_MS - 90 * MS_PER_DAY);
    insert_message(&conn, "new", "user", "authentication flow discussion");
    insert_chunk(&conn, 2, "new", "authentication flow discussion");

    let results = search_with_embedder(
        &conn,
        "authentication",
        &at_now(),
        Some(&MockEmbedder::new()),
    )
    .unwrap()
    .results;

    assert_eq!(ids(&results), ["new", "old"]);
}

#[test]
fn a_percent_sign_in_the_project_filter_is_matched_literally() {
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

    assert_eq!(ids(&results), NONE);
}

#[test]
fn a_query_with_file_returns_only_sessions_matching_both() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
    insert_message(&conn, "s1", "user", "authentication token rotation");
    seed_session_file(&conn, "s1", "/proj/auth.rs");
    insert_session(&conn, "s2", "claude", "/proj", 1709251200000);
    insert_message(&conn, "s2", "user", "unrelated weather notes");
    seed_session_file(&conn, "s2", "/proj/auth.rs");
    insert_session(&conn, "s3", "claude", "/proj", 1709251200000);
    insert_message(&conn, "s3", "user", "authentication token rotation");
    seed_session_file(&conn, "s3", "/proj/unrelated.rs");

    let results = search(
        &conn,
        "authentication",
        &SearchOptions {
            file: Some("/proj/auth.rs".to_owned()),
            ..Default::default()
        },
    )
    .unwrap();

    assert_eq!(ids(&results), ["s1"]);
}

#[test]
fn file_matches_a_path_suffix_and_treats_underscore_literally() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "hit", "claude", "/proj", 1709251200000);
    insert_message(&conn, "hit", "user", "authentication token rotation");
    seed_session_file(&conn, "hit", "/home/me/proj/src/a_b.rs");
    insert_session(&conn, "decoy", "claude", "/proj", 1709251200000);
    insert_message(&conn, "decoy", "user", "authentication token rotation");
    seed_session_file(&conn, "decoy", "/home/me/proj/src/aXb.rs");

    let results = search(
        &conn,
        "authentication",
        &SearchOptions {
            file: Some("a_b.rs".to_owned()),
            ..Default::default()
        },
    )
    .unwrap();

    assert_eq!(ids(&results), ["hit"]);
}

#[test]
fn file_without_a_query_lists_its_sessions_newest_first_without_excerpts() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "same_old", "claude", "/proj", 1000);
    insert_message(&conn, "same_old", "user", "alpha notes");
    seed_session_file(&conn, "same_old", "/proj/shared.rs");
    insert_session(&conn, "same_new", "claude", "/proj", 3000);
    insert_message(&conn, "same_new", "user", "beta notes");
    seed_session_file(&conn, "same_new", "/proj/shared.rs");
    insert_session(&conn, "same_mid", "claude", "/proj", 2000);
    insert_message(&conn, "same_mid", "user", "gamma notes");
    seed_session_file(&conn, "same_mid", "/proj/shared.rs");
    insert_session(&conn, "decoy", "claude", "/proj", 4000);
    insert_message(&conn, "decoy", "user", "delta notes");
    seed_session_file(&conn, "decoy", "/proj/unrelated.rs");

    let outcome = search_with_embedder(
        &conn,
        None,
        &SearchOptions {
            file: Some("/proj/shared.rs".to_owned()),
            ..at_now()
        },
        None,
    )
    .unwrap();

    assert_eq!(
        found(&outcome.results),
        [
            ("same_new", "", None),
            ("same_mid", "", None),
            ("same_old", "", None),
        ]
    );
    assert!(!outcome.results_incomplete);
}

#[test]
fn file_list_drops_a_session_without_a_source_and_flags_incomplete() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "ok", "claude", "/proj", 2000);
    seed_session_file(&conn, "ok", "/proj/shared.rs");
    insert_sourceless_session(&conn, "null_src", 1000);
    seed_session_file(&conn, "null_src", "/proj/shared.rs");

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

    assert_eq!(ids(&outcome.results), ["ok"]);
    assert!(outcome.results_incomplete);
}

#[test]
fn file_list_drops_a_session_with_an_unknown_source_and_flags_incomplete() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "ok", "claude", "/proj", 2000);
    seed_session_file(&conn, "ok", "/proj/shared.rs");
    insert_session(&conn, "mystery", "perplexity", "/proj", 1000);
    seed_session_file(&conn, "mystery", "/proj/shared.rs");

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

    assert_eq!(ids(&outcome.results), ["ok"]);
    assert!(outcome.results_incomplete);
}

#[test]
fn file_list_fills_the_limit_past_a_dropped_session() {
    let (_dir, conn) = setup_test_db();
    insert_sourceless_session(&conn, "null_src", 2000);
    seed_session_file(&conn, "null_src", "/proj/shared.rs");
    insert_session(&conn, "ok", "claude", "/proj", 1000);
    seed_session_file(&conn, "ok", "/proj/shared.rs");

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

    assert_eq!(ids(&outcome.results), ["ok"]);
    assert!(outcome.results_incomplete);
}

#[test]
fn file_on_an_index_without_scanned_files_asks_to_run_recall_index() {
    let (_dir, conn) = setup_test_db();
    conn.execute_batch("DROP TABLE session_files;").unwrap();
    let opts = SearchOptions {
        file: Some("/proj/a.rs".to_owned()),
        ..Default::default()
    };

    let Err(query_less) = search_with_embedder(&conn, None, &opts, None) else {
        panic!("query-less --file on an index without session_files must fail");
    };
    let Err(with_query) = search_with_embedder(&conn, "auth", &opts, None) else {
        panic!("query + --file on an index without session_files must fail");
    };

    assert_eq!(
        query_less.to_string(),
        "`--file` needs the scanned-files index; run `recall index` once to record it."
    );
    assert_eq!(
        with_query.to_string(),
        "`--file` needs the scanned-files index; run `recall index` once to record it."
    );
}

#[test]
fn a_two_character_japanese_query_finds_the_session() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
    insert_message(&conn, "s1", "user", "型安全についての議論");

    let results = search(&conn, "型安", &SearchOptions::default()).unwrap();

    assert_eq!(found(&results), [("s1", "**型安全**についての議論", None)]);
}

#[test]
fn a_one_character_kanji_query_finds_the_session() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
    insert_message(&conn, "s1", "user", "Rust言語の型システム");

    let results = search(&conn, "型", &SearchOptions::default()).unwrap();

    assert_eq!(ids(&results), ["s1"]);
}

#[test]
fn a_cjk_query_found_in_no_message_returns_nothing() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
    insert_message(&conn, "s1", "user", "hello world");

    let results = search(&conn, "没有", &SearchOptions::default()).unwrap();

    assert_eq!(ids(&results), NONE);
}

#[test]
fn a_query_mixing_latin_and_kanji_terms_requires_both() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
    insert_message(&conn, "s1", "user", "Rustの型安全について");
    insert_session(&conn, "s2", "claude", "/proj", 1709251200000);
    insert_message(&conn, "s2", "user", "Pythonのパフォーマンス");

    let results = search(&conn, "Rust 型安全", &SearchOptions::default()).unwrap();

    assert_eq!(ids(&results), ["s1"]);
}

#[test]
fn a_cjk_query_with_a_short_and_a_long_term_finds_the_session() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
    insert_message(
        &conn,
        "s1",
        "user",
        "方針：まとめ 方針*概要 方針（案） ディレクトリ整理の方針",
    );

    let results = search(&conn, "ディレクトリ整理 方針", &SearchOptions::default()).unwrap();

    assert_eq!(ids(&results), ["s1"]);
}

#[test]
fn a_korean_word_finds_the_session() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
    insert_message(&conn, "s1", "user", "한국어 테스트 메시지입니다");

    let results = search(&conn, "테스트", &SearchOptions::default()).unwrap();

    assert_eq!(ids(&results), ["s1"]);
}

#[test]
fn a_slash_command_query_matches_the_command_in_a_message() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
    insert_message(&conn, "s1", "user", "used /challenge to review the plan");

    let results = search(&conn, "/challenge", &SearchOptions::default()).unwrap();

    assert_eq!(ids(&results), ["s1"]);
}

#[test]
fn a_no_space_query_splits_between_kanji_and_hiragana() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "both", "claude", "/proj", NOW_MS);
    insert_message(&conn, "both", "user", "日本語の文章でひらがなを使う");
    insert_session(&conn, "kanji-only", "claude", "/proj", NOW_MS);
    insert_message(&conn, "kanji-only", "user", "日本語だけの文章");

    let results = search(&conn, "日本語ひらがな", &at_now()).unwrap();

    assert_eq!(ids(&results), ["both"]);
}

#[test]
fn a_letter_digit_token_is_matched_whole() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "whole", "claude", "/proj", NOW_MS);
    insert_message(&conn, "whole", "user", "the sha256 digest of the file");
    insert_session(&conn, "apart", "claude", "/proj", NOW_MS);
    insert_message(&conn, "apart", "user", "sha1 hashes computed on 256 cores");

    let results = search(&conn, "sha256", &at_now()).unwrap();

    assert_eq!(ids(&results), ["whole"]);
}

#[test]
fn a_kanji_compound_in_a_query_is_matched_whole() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "whole", "claude", "/proj", NOW_MS);
    insert_message(&conn, "whole", "user", "Rustで機械学習を試す");
    insert_session(&conn, "apart", "claude", "/proj", NOW_MS);
    insert_message(&conn, "apart", "user", "Rustで機械の学習を試す");
    // "Rust" is a run of at least three characters, so the query is searched
    // split at script boundaries. Without it the query would be searched as one
    // phrase whether or not the kanji run were split.

    let results = search(&conn, "Rust機械学習", &at_now()).unwrap();

    assert_eq!(ids(&results), ["whole"]);
}

#[test]
fn a_quoted_cjk_span_is_not_split_into_separate_terms() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "together", "claude", "/proj", NOW_MS);
    insert_message(&conn, "together", "user", "日本語ドキュメントを校正した");
    insert_session(&conn, "apart", "claude", "/proj", NOW_MS);
    insert_message(&conn, "apart", "user", "日本語の技術ドキュメントを校正した");

    let results = search(&conn, "\"日本語ドキュメント\"", &at_now()).unwrap();

    assert_eq!(ids(&results), ["together"]);
}

#[test]
fn degraded_no_space_cjk_query_reaches_session() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", NOW_MS);
    insert_message(
        &conn,
        "s1",
        "user",
        "textlintで日本語ドキュメントを校正する",
    );
    insert_session(&conn, "s2", "claude", "/proj", NOW_MS);
    insert_message(&conn, "s2", "user", "completely unrelated gardening notes");

    let no_space = search(&conn, "textlint日本語ドキュメント校正", &at_now()).unwrap();
    let spaced = search(&conn, "textlint 日本語 ドキュメント 校正", &at_now()).unwrap();

    assert_eq!(ids(&no_space), ["s1"]);
    assert_eq!(ids(&spaced), ["s1"]);
}

#[test]
fn segment_falls_back_to_raw_when_no_anchor_term() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", NOW_MS);
    insert_message(&conn, "s1", "user", "これを校正する手順をまとめた");
    // These messages add fifteen trigrams starting with "校正" and fifteen
    // starting with "する", sixteen of each with s1. Split into "校正 する", the
    // query would expand to 16 × 16 = 256 combinations, over MAX_COMBINATIONS
    // (100, src/compat.rs), and find nothing. The test depends on that cap
    // staying below 256. With a cap of 256 or more the split query also returns
    // only s1, and the test passes without the gate.
    insert_session(&conn, "kousei", "claude", "/proj", NOW_MS);
    insert_message(
        &conn,
        "kousei",
        "user",
        "校正案 校正版 校正中 校正後 校正前 校正用 校正例 校正集 校正表 校正記 校正号 校正点 校正字 校正形 校正面",
    );
    insert_session(&conn, "suru", "claude", "/proj", NOW_MS);
    insert_message(
        &conn,
        "suru",
        "user",
        "する案 する版 する中 する後 する前 する用 する例 する集 する表 する記 する号 する点 する字 する形 する面",
    );

    let results = search(&conn, "校正する", &at_now()).unwrap();

    assert_eq!(ids(&results), ["s1"]);
}

#[test]
fn a_hiragana_word_with_a_prolonged_sound_mark_finds_the_session() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", NOW_MS);
    insert_message(&conn, "s1", "user", "それはうーんと悩ましい問題だ");
    // ー is Katakana, so the query splits into three one-character runs. This
    // message adds five trigrams starting with う, five starting with ー and
    // four starting with ん, since んじ ends the message. The split form would
    // expand to 6 × 6 × 5 = 180 combinations. Over MAX_COMBINATIONS (100,
    // src/compat.rs) it finds nothing. With a cap of 180 or more it finds this
    // message too. Either way the test fails without the gate.
    insert_session(&conn, "decoy", "claude", "/proj", NOW_MS);
    insert_message(
        &conn,
        "decoy",
        "user",
        "うみ うた うし うま うち ーる ーと ーな ーき ーす んご んぶ んか んぼ んじ",
    );

    let results = search(&conn, "うーん", &at_now()).unwrap();

    assert_eq!(ids(&results), ["s1"]);
}

/// Writes two Claude logs: "sess-alpha" in /home/me/alpha asks about
/// authentication, and "sess-beta" in /home/me/beta asks about quicksort.
fn write_two_claude_logs(claude_dir: &Path) {
    fs::create_dir_all(claude_dir).unwrap();
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
}

fn index_claude_logs(conn: &mut Connection, claude_dir: &Path) {
    index_from_dirs(
        conn,
        &IndexOptions {
            force: false,
            claude_dir,
            codex_dir: &claude_dir.join("no-codex-sessions"),
            codex_archived_dir: None,
        },
        true,
    )
    .unwrap();
}

#[test]
fn indexed_logs_are_found_by_their_text_and_project() {
    let (_dir, mut conn) = setup_test_db();
    let logs = tempfile::TempDir::new().unwrap();
    write_two_claude_logs(logs.path());
    index_claude_logs(&mut conn, logs.path());
    let alpha = SearchOptions {
        project: Some("/home/me/alpha".to_owned()),
        ..Default::default()
    };

    let authentication = search(&conn, "authentication", &SearchOptions::default()).unwrap();
    let quicksort = search(&conn, "quicksort", &SearchOptions::default()).unwrap();
    let authentication_in_alpha = search(&conn, "authentication", &alpha).unwrap();
    let algorithm_in_alpha = search(&conn, "algorithm", &alpha).unwrap();

    assert_eq!(ids(&authentication), ["sess-alpha"]);
    assert_eq!(ids(&quicksort), ["sess-beta"]);
    assert_eq!(ids(&authentication_in_alpha), ["sess-alpha"]);
    assert_eq!(ids(&algorithm_in_alpha), NONE);
}

#[test]
fn reindexing_unchanged_logs_keeps_one_result_with_its_chunk() {
    let (_dir, mut conn) = setup_test_db();
    let logs = tempfile::TempDir::new().unwrap();
    write_two_claude_logs(logs.path());
    index_claude_logs(&mut conn, logs.path());
    index_chunks(&mut conn, None).unwrap();
    index_claude_logs(&mut conn, logs.path());

    let results = search(&conn, "authentication", &SearchOptions::default()).unwrap();

    assert_eq!(
        found(&results),
        [(
            "sess-alpha",
            "how to implement authentication in Rust\nUse the argon2 crate for password hashing",
            Some(1)
        )]
    );
}

#[test]
fn hybrid_search_returns_text_and_vector_matches_with_their_excerpts() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", NOW_MS);
    insert_message(&conn, "s1", "user", "authentication flow discussion");
    insert_session(&conn, "s2", "claude", "/proj", NOW_MS);
    insert_message(&conn, "s2", "user", "unrelated topic about weather");
    insert_chunk(&conn, 1, "s2", "authentication");

    let outcome = search_with_embedder(
        &conn,
        "authentication",
        &at_now(),
        Some(&MockEmbedder::new()),
    )
    .unwrap();

    assert_eq!(
        found(&outcome.results),
        [
            ("s1", "**authentication** flow di...", None),
            ("s2", "authentication", Some(1)),
        ]
    );
    assert!(!outcome.vec_degraded);
    assert!(!outcome.results_incomplete);
}

#[test]
fn a_text_match_supplies_the_excerpt_when_the_vector_leg_also_finds_the_session() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", NOW_MS);
    insert_message(&conn, "s1", "user", "authentication flow discussion");
    insert_chunk(&conn, 1, "s1", "authentication");

    let results = search_with_embedder(
        &conn,
        "authentication",
        &at_now(),
        Some(&MockEmbedder::new()),
    )
    .unwrap()
    .results;

    assert_eq!(
        found(&results),
        [("s1", "**authentication** flow di...", None)]
    );
}

#[test]
fn a_vector_match_shows_its_closest_chunk() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", NOW_MS);
    insert_message(&conn, "s1", "user", "authentication flow discussion");
    insert_session(&conn, "s2", "claude", "/proj", NOW_MS);
    insert_message(&conn, "s2", "user", "unrelated weather topic");
    insert_chunk(&conn, 1, "s2", "totally unrelated weather forecast");
    insert_chunk(&conn, 2, "s2", "authentication");

    let results = search_with_embedder(
        &conn,
        "authentication",
        &at_now(),
        Some(&MockEmbedder::new()),
    )
    .unwrap()
    .results;

    assert_eq!(
        found(&results),
        [
            ("s1", "**authentication** flow di...", None),
            ("s2", "authentication", Some(2)),
        ]
    );
}

#[test]
fn a_two_letter_query_is_answered_by_the_vector_leg() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", NOW_MS);
    insert_message(&conn, "s1", "user", "completely separate gardening notes");
    insert_session(&conn, "s2", "claude", "/proj", NOW_MS);
    insert_chunk(&conn, 1, "s2", "ui");

    let results = search_with_embedder(&conn, "ui", &at_now(), Some(&MockEmbedder::new()))
        .unwrap()
        .results;

    assert_eq!(found(&results), [("s2", "ui", Some(1))]);
}

#[test]
fn an_empty_or_blank_query_returns_nothing_even_with_embeddings() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", NOW_MS);
    insert_chunk(&conn, 1, "s1", "some indexed content");
    let embedder = MockEmbedder::new();

    let empty = search_with_embedder(&conn, "", &at_now(), Some(&embedder)).unwrap();
    let blank = search_with_embedder(&conn, "   ", &at_now(), Some(&embedder)).unwrap();

    assert_eq!(ids(&empty.results), NONE);
    assert_eq!(ids(&blank.results), NONE);
}

#[test]
fn a_punctuation_only_query_is_answered_by_the_vector_leg() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", NOW_MS);
    insert_chunk(&conn, 1, "s1", "some indexed content");

    let results = search_with_embedder(&conn, "...", &at_now(), Some(&MockEmbedder::new()))
        .unwrap()
        .results;

    assert_eq!(ids(&results), ["s1"]);
}

#[test]
fn a_failing_vector_leg_falls_back_to_text_matches_and_flags_degraded() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", NOW_MS);
    insert_message(&conn, "s1", "user", "authentication flow discussion");
    insert_chunk(&conn, 1, "s1", "content");

    let outcome = search_with_embedder(
        &conn,
        "authentication",
        &at_now(),
        Some(&MockEmbedder::failing_after(0)),
    )
    .unwrap();

    assert_eq!(ids(&outcome.results), ["s1"]);
    assert!(outcome.vec_degraded);
    assert!(!outcome.results_incomplete);
}

#[test]
fn a_failing_vector_leg_flags_degraded_even_with_no_text_match() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", NOW_MS);
    insert_message(&conn, "s1", "user", "totally unrelated weather forecast");
    insert_chunk(&conn, 1, "s1", "weather forecast");

    let outcome = search_with_embedder(
        &conn,
        "authentication",
        &at_now(),
        Some(&MockEmbedder::failing_after(0)),
    )
    .unwrap();

    assert_eq!(ids(&outcome.results), NONE);
    assert!(outcome.vec_degraded);
}

#[test]
fn a_failing_vector_leg_flags_degraded_when_every_text_match_is_dropped() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "weird", "gemini", "/proj", NOW_MS);
    insert_message(&conn, "weird", "user", "authentication flow design");
    insert_chunk(&conn, 1, "weird", "authentication flow");

    let outcome = search_with_embedder(
        &conn,
        "authentication",
        &at_now(),
        Some(&MockEmbedder::failing_after(0)),
    )
    .unwrap();

    assert_eq!(ids(&outcome.results), NONE);
    assert!(outcome.vec_degraded);
    assert!(outcome.results_incomplete);
}

#[test]
fn an_unreadable_vector_table_falls_back_to_text_matches_and_flags_degraded() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", NOW_MS);
    insert_message(&conn, "s1", "user", "authentication flow discussion");
    conn.execute_batch("DROP TABLE vec_chunks;").unwrap();

    let outcome = search_with_embedder(
        &conn,
        "authentication",
        &at_now(),
        Some(&MockEmbedder::new()),
    )
    .unwrap();

    assert_eq!(ids(&outcome.results), ["s1"]);
    assert!(outcome.vec_degraded);
}

#[test]
fn an_index_without_embeddings_answers_from_text_without_flags() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", NOW_MS);
    insert_message(&conn, "s1", "user", "authentication flow discussion");

    let outcome = search_with_embedder(
        &conn,
        "authentication",
        &at_now(),
        Some(&MockEmbedder::new()),
    )
    .unwrap();

    assert_eq!(ids(&outcome.results), ["s1"]);
    assert!(!outcome.vec_degraded);
    assert!(!outcome.results_incomplete);
}

#[test]
fn hybrid_search_drops_an_unknown_source_and_flags_incomplete_only() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", NOW_MS);
    insert_message(&conn, "s1", "user", "authentication flow design");
    insert_chunk(&conn, 1, "s1", "authentication flow");
    insert_session(&conn, "weird", "gemini", "/proj", NOW_MS);
    insert_message(&conn, "weird", "user", "authentication retry logic");
    insert_chunk(&conn, 2, "weird", "authentication retry");

    let outcome = search_with_embedder(
        &conn,
        "authentication",
        &at_now(),
        Some(&MockEmbedder::new()),
    )
    .unwrap();

    assert_eq!(ids(&outcome.results), ["s1"]);
    assert!(!outcome.vec_degraded);
    assert!(outcome.results_incomplete);
}

#[test]
fn text_search_drops_an_unknown_source_and_flags_incomplete() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", NOW_MS);
    insert_message(&conn, "s1", "user", "authentication flow design");
    insert_session(&conn, "weird", "gemini", "/proj", NOW_MS);
    insert_message(&conn, "weird", "user", "authentication retry logic");

    let outcome = search_with_embedder(&conn, "authentication", &at_now(), None).unwrap();

    assert_eq!(ids(&outcome.results), ["s1"]);
    assert!(!outcome.vec_degraded);
    assert!(outcome.results_incomplete);
}

#[test]
fn an_unreadable_message_row_flags_incomplete_and_keeps_the_rest() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", NOW_MS);
    insert_message(&conn, "s1", "user", "authentication flow");
    insert_null_session_id_message(&conn, "authentication token");

    // With automated sessions excluded, the session-type filter would drop the
    // NULL row before it is read; including them lets the row reach the reader.
    let outcome = search_with_embedder(
        &conn,
        "authentication",
        &SearchOptions {
            include_automated: true,
            ..at_now()
        },
        None,
    )
    .unwrap();

    assert_eq!(ids(&outcome.results), ["s1"]);
    assert!(!outcome.vec_degraded);
    assert!(outcome.results_incomplete);
}

#[test]
fn automated_sessions_are_hidden_by_default_and_unclassified_ones_stay() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "auto", "claude", "/proj", 1709251200000);
    insert_message(&conn, "auto", "user", "authentication flow");
    set_session_type(&conn, "auto", "automated");
    insert_session(&conn, "null", "claude", "/proj", 1709251200000);
    insert_message(&conn, "null", "user", "authentication flow");

    let results = search(&conn, "authentication", &SearchOptions::default()).unwrap();

    assert_eq!(ids(&results), ["null"]);
}

#[test]
fn include_automated_returns_automated_sessions() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "auto", "claude", "/proj", 1709251200000);
    insert_message(&conn, "auto", "user", "authentication flow");
    set_session_type(&conn, "auto", "automated");

    let results = search(
        &conn,
        "authentication",
        &SearchOptions {
            include_automated: true,
            ..Default::default()
        },
    )
    .unwrap();

    assert_eq!(ids(&results), ["auto"]);
}

#[test]
fn automated_sessions_reached_only_by_vector_are_hidden_by_default() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "keep", "claude", "/proj", NOW_MS);
    insert_message(&conn, "keep", "user", "authentication flow discussion");
    insert_session(&conn, "auto", "claude", "/proj", NOW_MS);
    insert_chunk(&conn, 1, "auto", "authentication");
    set_session_type(&conn, "auto", "automated");

    let ids = hybrid_search_ids(&conn, &MockEmbedder::new(), &at_now());

    assert_eq!(ids, ["keep"]);
}

#[test]
fn excluding_the_current_session_drops_it_from_both_legs() {
    let (_dir, conn) = setup_test_db();
    let embedder = MockEmbedder::new();

    assert_filter_symmetric(
        || {
            insert_session(&conn, "keep", "claude", "/proj", NOW_MS);
            insert_message(&conn, "keep", "user", "authentication flow discussion");
            insert_session(&conn, "cur", "claude", "/proj", NOW_MS);
            insert_message(&conn, "cur", "user", "authentication flow discussion");
            insert_chunk(&conn, 1, "cur", "authentication");
            ("keep".to_owned(), "cur".to_owned())
        },
        || {
            hybrid_search_ids(
                &conn,
                &embedder,
                &SearchOptions {
                    current: CurrentSession::Exclude("cur".to_owned()),
                    ..at_now()
                },
            )
        },
        || hybrid_search_ids(&conn, &embedder, &at_now()),
    );
}

#[test]
fn only_the_current_session_keeps_it_and_drops_the_rest_from_both_legs() {
    let (_dir, conn) = setup_test_db();
    let embedder = MockEmbedder::new();

    assert_filter_symmetric(
        || {
            insert_session(&conn, "cur", "claude", "/proj", NOW_MS);
            insert_message(&conn, "cur", "user", "authentication flow discussion");
            insert_session(&conn, "keep", "claude", "/proj", NOW_MS);
            insert_message(&conn, "keep", "user", "authentication flow discussion");
            insert_chunk(&conn, 1, "keep", "authentication");
            ("cur".to_owned(), "keep".to_owned())
        },
        || {
            hybrid_search_ids(
                &conn,
                &embedder,
                &SearchOptions {
                    current: CurrentSession::Only("cur".to_owned()),
                    ..at_now()
                },
            )
        },
        || hybrid_search_ids(&conn, &embedder, &at_now()),
    );
}

#[test]
fn file_filter_drops_a_vector_only_match_on_another_file() {
    let (_dir, conn) = setup_test_db();
    let embedder = MockEmbedder::new();

    assert_filter_symmetric(
        || {
            insert_session(&conn, "keep", "claude", "/proj", NOW_MS);
            insert_message(&conn, "keep", "user", "authentication flow discussion");
            seed_session_file(&conn, "keep", "/proj/auth.rs");
            insert_session(&conn, "other", "claude", "/proj", NOW_MS);
            insert_chunk(&conn, 1, "other", "authentication");
            seed_session_file(&conn, "other", "/proj/unrelated.rs");
            ("keep".to_owned(), "other".to_owned())
        },
        || {
            hybrid_search_ids(
                &conn,
                &embedder,
                &SearchOptions {
                    file: Some("/proj/auth.rs".to_owned()),
                    ..at_now()
                },
            )
        },
        || hybrid_search_ids(&conn, &embedder, &at_now()),
    );
}

#[test]
fn project_filter_drops_a_vector_only_match_in_another_project() {
    let (_dir, conn) = setup_test_db();
    let embedder = MockEmbedder::new();

    assert_filter_symmetric(
        || {
            insert_session(&conn, "s1", "claude", "/home/me/proj-a", NOW_MS);
            insert_message(&conn, "s1", "user", "authentication flow discussion");
            insert_session(&conn, "s2", "claude", "/home/me/proj-b", NOW_MS);
            insert_message(&conn, "s2", "user", "unrelated topic about weather");
            insert_chunk(&conn, 1, "s2", "authentication");
            ("s1".to_owned(), "s2".to_owned())
        },
        || {
            hybrid_search_ids(
                &conn,
                &embedder,
                &SearchOptions {
                    project: Some("/home/me/proj-a".to_owned()),
                    ..at_now()
                },
            )
        },
        || hybrid_search_ids(&conn, &embedder, &at_now()),
    );
}

#[test]
fn days_filter_drops_an_older_vector_only_match() {
    let (_dir, conn) = setup_test_db();
    let embedder = MockEmbedder::new();

    assert_filter_symmetric(
        || {
            insert_session(&conn, "s1", "claude", "/proj", NOW_MS - MS_PER_DAY);
            insert_message(&conn, "s1", "user", "authentication flow discussion");
            insert_session(&conn, "s2", "claude", "/proj", NOW_MS - 60 * MS_PER_DAY);
            insert_message(&conn, "s2", "user", "unrelated topic about weather");
            insert_chunk(&conn, 1, "s2", "authentication");
            ("s1".to_owned(), "s2".to_owned())
        },
        || {
            hybrid_search_ids(
                &conn,
                &embedder,
                &SearchOptions {
                    days: Some(7),
                    ..at_now()
                },
            )
        },
        || hybrid_search_ids(&conn, &embedder, &at_now()),
    );
}

#[test]
fn source_filter_drops_a_vector_only_match_from_another_source() {
    let (_dir, conn) = setup_test_db();
    let embedder = MockEmbedder::new();

    assert_filter_symmetric(
        || {
            insert_session(&conn, "s1", "claude", "/proj", NOW_MS);
            insert_message(&conn, "s1", "user", "authentication flow discussion");
            insert_session(&conn, "s2", "codex", "/proj", NOW_MS);
            insert_message(&conn, "s2", "user", "unrelated topic about weather");
            insert_chunk(&conn, 1, "s2", "authentication");
            ("s1".to_owned(), "s2".to_owned())
        },
        || {
            hybrid_search_ids(
                &conn,
                &embedder,
                &SearchOptions {
                    source: Some(Source::Claude),
                    ..at_now()
                },
            )
        },
        || hybrid_search_ids(&conn, &embedder, &at_now()),
    );
}

#[test]
fn project_filter_ignores_case_for_a_vector_only_match() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "mixed-case", "claude", "/Home/me/Proj-A", NOW_MS);
    insert_chunk(&conn, 1, "mixed-case", "authentication");
    insert_session(&conn, "other", "claude", "/home/me/proj-b", NOW_MS);
    insert_chunk(&conn, 2, "other", "authentication");

    let ids = hybrid_search_ids(
        &conn,
        &MockEmbedder::new(),
        &SearchOptions {
            project: Some("/home/me/proj-a".to_owned()),
            ..at_now()
        },
    );

    assert_eq!(ids, ["mixed-case"]);
}

#[test]
fn an_unknown_source_vector_match_does_not_take_the_limit_slot() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s_valid", "claude", "/proj", NOW_MS);
    insert_message(&conn, "s_valid", "user", "authentication flow discussion");
    insert_session(&conn, "s_unknown", "mystery_source_v99", "/proj", NOW_MS);
    insert_message(&conn, "s_unknown", "user", "unrelated topic");
    insert_chunk(&conn, 1, "s_unknown", "authentication");

    let outcome = search_with_embedder(
        &conn,
        "authentication",
        &SearchOptions {
            limit: 1,
            ..at_now()
        },
        Some(&MockEmbedder::new()),
    )
    .unwrap();

    assert_eq!(ids(&outcome.results), ["s_valid"]);
    assert!(outcome.results_incomplete);
}

/// One session reachable by both text and vector search.
fn seed_filter_case(
    conn: &Connection,
    chunk_id: i64,
    sid: &str,
    source: &str,
    project: &str,
    ts: i64,
    file: &str,
) {
    insert_session(conn, sid, source, project, ts);
    insert_message(conn, sid, "user", "authentication flow");
    insert_chunk(conn, chunk_id, sid, "authentication");
    seed_session_file(conn, sid, file);
}

/// "keep" passes every filter in `every_filter`, and each other session fails
/// exactly one. All are reachable by both legs, so a filter missing from either
/// leg lets its decoy back in.
fn seed_one_decoy_per_filter(conn: &Connection) {
    seed_filter_case(
        conn,
        1,
        "keep",
        "claude",
        "/proj",
        NOW_MS,
        "/proj/shared.rs",
    );
    seed_filter_case(
        conn,
        2,
        "project",
        "claude",
        "/other",
        NOW_MS,
        "/proj/shared.rs",
    );
    seed_filter_case(
        conn,
        3,
        "old",
        "claude",
        "/proj",
        NOW_MS - 8 * MS_PER_DAY,
        "/proj/shared.rs",
    );
    seed_filter_case(
        conn,
        4,
        "source",
        "codex",
        "/proj",
        NOW_MS,
        "/proj/shared.rs",
    );
    seed_filter_case(conn, 5, "file", "claude", "/proj", NOW_MS, "/proj/other.rs");
    seed_filter_case(
        conn,
        6,
        "current",
        "claude",
        "/proj",
        NOW_MS,
        "/proj/shared.rs",
    );
    seed_filter_case(
        conn,
        7,
        "automated",
        "claude",
        "/proj",
        NOW_MS,
        "/proj/shared.rs",
    );
    set_session_type(conn, "automated", "automated");
}

fn every_filter() -> SearchOptions {
    SearchOptions {
        project: Some("/proj".to_owned()),
        days: Some(7),
        source: Some(Source::Claude),
        file: Some("shared.rs".to_owned()),
        current: CurrentSession::Exclude("current".to_owned()),
        ..at_now()
    }
}

#[test]
fn combined_filters_apply_to_text_search() {
    let (_dir, conn) = setup_test_db();
    seed_one_decoy_per_filter(&conn);

    let outcome = search_with_embedder(&conn, "authentication", &every_filter(), None).unwrap();

    assert_eq!(ids(&outcome.results), ["keep"]);
    assert!(!outcome.vec_degraded);
    assert!(!outcome.results_incomplete);
}

#[test]
fn combined_filters_apply_to_hybrid_search() {
    let (_dir, conn) = setup_test_db();
    seed_one_decoy_per_filter(&conn);

    let outcome = search_with_embedder(
        &conn,
        "authentication",
        &every_filter(),
        Some(&MockEmbedder::new()),
    )
    .unwrap();

    assert_eq!(ids(&outcome.results), ["keep"]);
    assert!(!outcome.vec_degraded);
    assert!(!outcome.results_incomplete);
}

#[test]
fn combined_filters_apply_to_the_file_list() {
    let (_dir, conn) = setup_test_db();
    seed_one_decoy_per_filter(&conn);

    let outcome = search_with_embedder(&conn, None, &every_filter(), None).unwrap();

    assert_eq!(ids(&outcome.results), ["keep"]);
    assert!(!outcome.vec_degraded);
    assert!(!outcome.results_incomplete);
}

#[test]
fn excluded_automated_count_follows_the_file_filter_without_a_query() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "auto-a", "claude", "/proj", 1709251200000);
    set_session_type(&conn, "auto-a", "automated");
    seed_session_file(&conn, "auto-a", "/proj/src/a.rs");
    insert_session(&conn, "auto-b", "claude", "/proj", 1709251200000);
    set_session_type(&conn, "auto-b", "automated");
    seed_session_file(&conn, "auto-b", "/proj/src/b.rs");
    insert_session(&conn, "human-a", "claude", "/proj", 1709251200000);
    seed_session_file(&conn, "human-a", "/proj/src/a.rs");
    insert_session(&conn, "human-a2", "claude", "/proj", 1709251200000);
    seed_session_file(&conn, "human-a2", "/proj/src/a.rs");
    let opts = SearchOptions {
        file: Some("a.rs".to_owned()),
        ..SearchOptions::default()
    };

    let count = count_excluded_automated(&conn, None, &opts).unwrap();

    assert_eq!(
        count, 1,
        "only auto-a is an automated session that touched a.rs"
    );
}

#[test]
fn excluded_automated_count_is_zero_when_automated_sessions_are_included() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "auto", "claude", "/proj", 1709251200000);
    insert_message(&conn, "auto", "user", "authentication flow");
    set_session_type(&conn, "auto", "automated");
    let opts = SearchOptions {
        include_automated: true,
        ..SearchOptions::default()
    };

    let count = count_excluded_automated(&conn, Some("authentication"), &opts).unwrap();

    assert_eq!(
        count, 0,
        "nothing is excluded when automated sessions are included"
    );
}

#[test]
fn excluded_automated_count_is_zero_without_a_query_or_file_filter() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "auto", "claude", "/proj", 1709251200000);
    set_session_type(&conn, "auto", "automated");

    let count = count_excluded_automated(&conn, None, &SearchOptions::default()).unwrap();

    assert_eq!(
        count, 0,
        "with nothing to match, no automated session was left out"
    );
}

#[test]
fn excluded_automated_count_is_zero_when_the_query_has_no_searchable_terms() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "auto", "claude", "/proj", 1709251200000);
    insert_message(&conn, "auto", "user", "authentication flow");
    set_session_type(&conn, "auto", "automated");

    let count = count_excluded_automated(&conn, Some("!!"), &SearchOptions::default()).unwrap();

    assert_eq!(count, 0, "a query with no searchable terms matches nothing");
}

#[test]
fn excluded_automated_count_reports_a_broken_vocab_table_as_an_error() {
    let (_dir, conn) = setup_test_db();
    conn.execute_batch("DROP TABLE messages_vocab; CREATE TABLE messages_vocab (term TEXT);")
        .unwrap();

    let result = count_excluded_automated(&conn, Some("ab"), &SearchOptions::default());

    assert!(
        result.is_err(),
        "a vocab table without the cnt column must surface as an error, got {result:?}"
    );
}
