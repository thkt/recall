use std::path::Path;

use super::*;
use crate::date::MS_PER_DAY;
use crate::db::{open_db, open_db_readonly, seed_session_file, setup_test_db};
use crate::embedder::MockEmbedder;
use crate::embedding::{ChunkedEmbedding, EmbedError, EmbedOptions};

const NOW_MS: i64 = 1_750_000_000_000;

fn add_session(conn: &Connection, sid: &str, source: &str, project: &str, ts: i64) {
    conn.execute(
        "INSERT INTO sessions (session_id, source, file_path, project, slug, timestamp, mtime) \
         VALUES (?1, ?2, '', ?3, ?1, ?4, 0.0)",
        rusqlite::params![sid, source, project, ts],
    )
    .unwrap();
}

fn add_message(conn: &Connection, sid: &str, text: &str) -> i64 {
    conn.execute(
        "INSERT INTO messages (session_id, role, text) VALUES (?1, 'user', ?2)",
        rusqlite::params![sid, text],
    )
    .unwrap();
    conn.last_insert_rowid()
}

fn add_chunk(conn: &Connection, chunk_id: i64, sid: &str, text: &str, over: Option<i64>) {
    conn.execute(
        "INSERT INTO qa_chunks (id, session_id, content, timestamp, src_rowid_lo, src_rowid_hi) \
         VALUES (?1, ?2, ?3, 0, ?4, ?4)",
        rusqlite::params![chunk_id, sid, text, over],
    )
    .unwrap();
    let embedding = MockEmbedder::deterministic_vector(text);
    conn.execute(
        "INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?1, vec_int8(?2))",
        rusqlite::params![chunk_id, quantize_int8(&embedding)],
    )
    .unwrap();
}

fn set_range(conn: &Connection, sid: &str, lo: Option<i64>, hi: Option<i64>) {
    conn.execute(
        "UPDATE sessions SET message_rowid_lo = ?2, message_rowid_hi = ?3 WHERE session_id = ?1",
        rusqlite::params![sid, lo, hi],
    )
    .unwrap();
}

fn record_message_ranges(conn: &Connection) {
    conn.execute_batch(
        "UPDATE sessions SET message_rowid_lo = r.lo, message_rowid_hi = r.hi \
             FROM (SELECT session_id, min(rowid) AS lo, max(rowid) AS hi \
                   FROM messages GROUP BY session_id) AS r \
             WHERE r.session_id = sessions.session_id; \
         UPDATE sessions SET message_rowid_lo = 1, message_rowid_hi = 0 \
             WHERE message_rowid_lo IS NULL;",
    )
    .unwrap();
    assert_reads_ranges(conn);
}

fn assert_reads_ranges(conn: &Connection) {
    assert!(
        matches!(
            MessageSessions::load(conn).unwrap(),
            MessageSessions::RowidRanges(_)
        ),
        "the search must resolve sessions from the stored ranges"
    );
}

type Seen = (Vec<(String, String, Option<i64>)>, bool, bool);

fn seen(outcome: SearchOutcome) -> Seen {
    (
        outcome
            .results
            .into_iter()
            .map(|r| (r.session.session_id, r.excerpt, r.chunk_id))
            .collect(),
        outcome.vec_degraded,
        outcome.results_incomplete,
    )
}

fn sorted_ids(seen: &Seen) -> Vec<&str> {
    let mut ids: Vec<&str> = seen.0.iter().map(|(sid, _, _)| sid.as_str()).collect();
    ids.sort_unstable();
    ids
}

fn run(conn: &Connection, query: &str, opts: &SearchOptions, hybrid: bool) -> Seen {
    let embedder = MockEmbedder::new();
    let embedder: Option<&dyn Embed> = if hybrid { Some(&embedder) } else { None };
    seen(search_with_embedder(conn, query, opts, embedder).unwrap())
}

fn at_now() -> SearchOptions {
    SearchOptions {
        now_ms: Some(NOW_MS),
        ..SearchOptions::default()
    }
}

fn seed_corpus(conn: &Connection) {
    add_session(conn, "chunked", "claude", "/proj/a", NOW_MS);
    let padded = add_message(
        conn,
        "chunked",
        "authentication alpha padded with many extra filler words here to lower its relevance",
    );
    let best = add_message(conn, "chunked", "authentication beta");
    add_chunk(conn, 1, "chunked", "ALPHA_CHUNK_BODY", Some(padded));
    add_chunk(conn, 2, "chunked", "BETA_CHUNK_BODY", Some(best));

    add_session(
        conn,
        "snippet",
        "codex",
        "/proj/b",
        NOW_MS - 30 * MS_PER_DAY,
    );
    add_message(conn, "snippet", "authentication flow discussion");

    add_session(conn, "automated", "claude", "/proj/a", NOW_MS);
    add_message(conn, "automated", "authentication automated run");
    conn.execute(
        "UPDATE sessions SET session_type = 'automated' WHERE session_id = 'automated'",
        [],
    )
    .unwrap();

    add_session(conn, "touched", "claude", "/proj/a", NOW_MS);
    add_message(conn, "touched", "authentication with gateway notes");
    seed_session_file(conn, "touched", "/proj/a/src/lib.rs");

    add_session(conn, "vector", "claude", "/proj/a", NOW_MS);
    add_message(conn, "vector", "unrelated weather topic");
    add_chunk(conn, 10, "vector", "authentication", None);

    add_session(conn, "vector-gateway", "claude", "/proj/a", NOW_MS);
    add_message(conn, "vector-gateway", "weather gateway topic");
    add_chunk(conn, 11, "vector-gateway", "authentication", None);
}

fn corpus_searches() -> Vec<(&'static str, SearchOptions, bool)> {
    let with = |f: fn(&mut SearchOptions)| {
        let mut opts = at_now();
        f(&mut opts);
        opts
    };
    let mut searches = Vec::new();
    for hybrid in [false, true] {
        searches.extend([
            ("authentication", at_now(), hybrid),
            (
                "authentication",
                with(|o| o.include_automated = true),
                hybrid,
            ),
            (
                "authentication",
                with(|o| o.project = Some("/proj/b".to_owned())),
                hybrid,
            ),
            ("authentication", with(|o| o.days = Some(7)), hybrid),
            (
                "authentication",
                with(|o| o.source = Some(Source::Codex)),
                hybrid,
            ),
            (
                "authentication",
                with(|o| o.file = Some("lib.rs".to_owned())),
                hybrid,
            ),
            (
                "authentication",
                with(|o| o.current = CurrentSession::Exclude("snippet".to_owned())),
                hybrid,
            ),
            (
                "authentication",
                with(|o| o.current = CurrentSession::Only("chunked".to_owned())),
                hybrid,
            ),
            ("authentication NOT gateway", at_now(), hybrid),
        ]);
    }
    searches
}

#[test]
fn stored_ranges_return_the_same_sessions_order_and_excerpts_for_every_filter_and_not() {
    let (_dir, conn) = setup_test_db();
    seed_corpus(&conn);
    let searches = corpus_searches();
    let without_ranges: Vec<Seen> = searches
        .iter()
        .map(|(q, opts, hybrid)| run(&conn, q, opts, *hybrid))
        .collect();

    record_message_ranges(&conn);
    let with_ranges: Vec<Seen> = searches
        .iter()
        .map(|(q, opts, hybrid)| run(&conn, q, opts, *hybrid))
        .collect();

    assert_eq!(with_ranges, without_ranges);
    let keyword: Vec<Vec<&str>> = with_ranges[..9].iter().map(sorted_ids).collect();
    assert_eq!(
        keyword,
        [
            vec!["chunked", "snippet", "touched"],
            vec!["automated", "chunked", "snippet", "touched"],
            vec!["snippet"],
            vec!["chunked", "touched"],
            vec!["snippet"],
            vec!["touched"],
            vec!["chunked", "touched"],
            vec!["chunked"],
            vec!["chunked", "snippet"],
        ]
    );
    assert_eq!(
        sorted_ids(&with_ranges[9]),
        ["chunked", "snippet", "touched", "vector", "vector-gateway"]
    );
    assert_eq!(
        sorted_ids(&with_ranges[17]),
        ["chunked", "snippet", "vector"]
    );
    let excerpt = |sid: &str| {
        with_ranges[0]
            .0
            .iter()
            .find(|(s, _, _)| s == sid)
            .map(|(_, excerpt, chunk)| (excerpt.clone(), *chunk))
            .unwrap()
    };
    assert_eq!(excerpt("chunked"), ("BETA_CHUNK_BODY".to_owned(), Some(2)));
    assert_eq!(
        excerpt("snippet"),
        ("**authentication** flow di...".to_owned(), None)
    );
}

#[test]
fn stored_ranges_count_the_same_excluded_automated_sessions() {
    let (_dir, conn) = setup_test_db();
    seed_corpus(&conn);
    let in_b = SearchOptions {
        project: Some("/proj/b".to_owned()),
        ..at_now()
    };
    let count = |opts: &SearchOptions| {
        count_excluded_automated(&conn, Some("authentication"), opts).unwrap()
    };
    let without_ranges = (count(&at_now()), count(&in_b));

    record_message_ranges(&conn);

    assert_eq!((count(&at_now()), count(&in_b)), without_ranges);
    assert_eq!(without_ranges, (1, 0));
}

#[test]
fn a_row_that_lies_only_in_another_sessions_range_is_taken_as_that_sessions_row() {
    let (_dir, conn) = setup_test_db();
    add_session(&conn, "holder", "claude", "/proj", NOW_MS);
    add_session(&conn, "writer", "claude", "/proj", NOW_MS);
    let first = add_message(&conn, "holder", "plain words");
    let stray = add_message(&conn, "writer", "zebra crossing");
    set_range(&conn, "holder", Some(first), Some(stray));
    set_range(&conn, "writer", Some(1), Some(0));
    assert_reads_ranges(&conn);

    let seen = run(&conn, "zebra", &at_now(), false);

    assert_eq!(
        sorted_ids(&seen),
        ["holder"],
        "a broken range premise attributes the row to the range's session without an error"
    );
}

#[test]
fn a_row_in_two_overlapping_ranges_belongs_to_its_stored_session() {
    let (_dir, conn) = setup_test_db();
    add_session(&conn, "wide", "claude", "/proj", NOW_MS);
    add_session(&conn, "narrow", "claude", "/proj", NOW_MS);
    let first = add_message(&conn, "wide", "plain words");
    let shared = add_message(&conn, "narrow", "zebra crossing");
    set_range(&conn, "wide", Some(first), Some(shared));
    set_range(&conn, "narrow", Some(shared), Some(shared));
    assert_reads_ranges(&conn);

    let seen = run(&conn, "zebra", &at_now(), false);

    assert_eq!(sorted_ids(&seen), ["narrow"]);
}

#[test]
fn a_row_of_a_session_without_a_range_inside_another_sessions_range_belongs_to_its_stored_session()
{
    let (_dir, conn) = setup_test_db();
    add_session(&conn, "ranged", "claude", "/proj", NOW_MS);
    add_session(&conn, "unranged", "claude", "/proj", NOW_MS);
    let first = add_message(&conn, "ranged", "plain words");
    let inside = add_message(&conn, "unranged", "zebra crossing");
    set_range(&conn, "ranged", Some(first), Some(inside));

    let seen = run(&conn, "zebra", &at_now(), false);

    assert_eq!(sorted_ids(&seen), ["unranged"]);
}

#[test]
fn a_match_without_a_session_row_outside_every_range_still_marks_the_results_incomplete() {
    let (_dir, conn) = setup_test_db();
    add_session(&conn, "valid", "claude", "/proj", NOW_MS - 365 * MS_PER_DAY);
    add_message(&conn, "valid", "authentication notes");
    record_message_ranges(&conn);
    add_message(&conn, "ghost", "authentication notes kept");

    let seen = run(
        &conn,
        "authentication",
        &SearchOptions {
            limit: 1,
            include_automated: true,
            ..at_now()
        },
        false,
    );

    assert_eq!(sorted_ids(&seen), ["valid"]);
    assert!(
        seen.2,
        "the ghost session could rank first, so results are incomplete"
    );
}

#[test]
fn an_index_without_the_range_columns_opened_read_only_returns_the_same_results() {
    let (dir, conn) = setup_test_db();
    seed_corpus(&conn);
    let searches = corpus_searches();
    let expected: Vec<Seen> = searches
        .iter()
        .map(|(q, opts, hybrid)| run(&conn, q, opts, *hybrid))
        .collect();
    conn.execute_batch(
        "ALTER TABLE sessions DROP COLUMN message_rowid_lo; \
         ALTER TABLE sessions DROP COLUMN message_rowid_hi;",
    )
    .unwrap();
    drop(conn);

    let (conn, _) = open_db_readonly(&dir.path().join("test.db")).unwrap();
    assert!(
        !table_def(&conn, "sessions")
            .unwrap()
            .unwrap()
            .contains("message_rowid_lo")
    );
    let actual: Vec<Seen> = searches
        .iter()
        .map(|(q, opts, hybrid)| run(&conn, q, opts, *hybrid))
        .collect();

    assert_eq!(actual, expected);
}

struct WritesDuringEmbedding<'a> {
    db_path: &'a Path,
    inner: MockEmbedder,
}

impl Embed for WritesDuringEmbedding<'_> {
    fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        let writer = open_db(self.db_path).unwrap();
        let rowid = add_message(&writer, "vector", "beta reference");
        set_range(&writer, "vector", Some(rowid - 1), Some(rowid));
        self.inner.embed_query(text)
    }

    fn embed_document(&self, text: &str) -> Result<ChunkedEmbedding, EmbedError> {
        self.inner.embed_document(text)
    }

    fn embed_documents_batch(
        &self,
        texts: &[&str],
        options: &EmbedOptions,
    ) -> Result<Vec<ChunkedEmbedding>, EmbedError> {
        self.inner.embed_documents_batch(texts, options)
    }
}

#[test]
fn a_search_reads_the_ranges_and_every_match_from_the_index_as_it_was_when_it_began() {
    let (dir, conn) = setup_test_db();
    add_session(&conn, "text-hit", "claude", "/proj", NOW_MS);
    add_message(&conn, "text-hit", "alpha notes");
    add_session(&conn, "vector", "claude", "/proj", NOW_MS);
    add_message(&conn, "vector", "weather topic");
    add_chunk(&conn, 1, "vector", "alpha", None);
    record_message_ranges(&conn);
    let embedder = WritesDuringEmbedding {
        db_path: &dir.path().join("test.db"),
        inner: MockEmbedder::new(),
    };

    let outcome =
        search_with_embedder(&conn, "alpha NOT beta", &at_now(), Some(&embedder)).unwrap();

    let mut ids: Vec<&str> = outcome
        .results
        .iter()
        .map(|r| r.session.session_id.as_str())
        .collect();
    ids.sort_unstable();
    assert_eq!(
        ids,
        ["text-hit", "vector"],
        "a NOT match committed after the search began must not drop the vector-only session"
    );
    let after: i64 = conn
        .query_row(
            "SELECT count(*) FROM messages WHERE messages MATCH 'beta'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(after, 1, "the write the embedder made is committed");
}

const NOT_UTF8_TEXT_SESSION_IDS: [&str; 4] = ["42", "4.5", "x'0102'", "CAST(x'ff' AS TEXT)"];

fn add_message_with_stored_session_id(conn: &Connection, literal: &str, text: &str) {
    conn.execute(
        &format!("INSERT INTO messages (session_id, role, text) VALUES ({literal}, 'user', ?1)"),
        [text],
    )
    .unwrap();
}

fn seed_automated_and_valid(conn: &Connection) {
    add_session(conn, "valid", "claude", "/proj", NOW_MS);
    add_message(conn, "valid", "authentication notes");
    add_session(conn, "automated", "claude", "/proj", NOW_MS);
    add_message(conn, "automated", "authentication automated run");
    conn.execute(
        "UPDATE sessions SET session_type = 'automated' WHERE session_id = 'automated'",
        [],
    )
    .unwrap();
}

#[test]
fn a_match_whose_session_id_is_not_utf8_text_marks_the_results_incomplete_with_or_without_the_automated_filter()
 {
    for literal in NOT_UTF8_TEXT_SESSION_IDS {
        let (_dir, conn) = setup_test_db();
        seed_automated_and_valid(&conn);
        add_message_with_stored_session_id(&conn, literal, "authentication stray");
        let included = SearchOptions {
            include_automated: true,
            ..at_now()
        };
        let observe = |conn: &Connection| {
            (
                run(conn, "authentication", &at_now(), false),
                run(conn, "authentication", &included, false),
                count_excluded_automated(conn, Some("authentication"), &at_now())
                    .map_err(|e| e.to_string()),
            )
        };

        let without_ranges = observe(&conn);
        record_message_ranges(&conn);
        let with_ranges = observe(&conn);

        assert_eq!(sorted_ids(&without_ranges.0), ["valid"], "{literal}");
        assert!(without_ranges.0.2, "{literal} under the default filters");
        assert_eq!(
            sorted_ids(&without_ranges.1),
            ["automated", "valid"],
            "{literal}"
        );
        assert!(without_ranges.1.2, "{literal} with --include-automated");
        assert_eq!(
            without_ranges.2,
            Ok(1),
            "the automated count leaves out the {literal} row"
        );
        assert_eq!(with_ranges, without_ranges, "{literal}");
    }
}

#[test]
fn a_search_whose_only_match_has_a_session_id_that_is_not_utf8_text_succeeds_and_marks_the_results_incomplete()
 {
    for literal in NOT_UTF8_TEXT_SESSION_IDS {
        let (_dir, conn) = setup_test_db();
        add_session(&conn, "valid", "claude", "/proj", NOW_MS);
        add_message(&conn, "valid", "unrelated notes");
        add_message_with_stored_session_id(&conn, literal, "authentication stray");

        let outcome = search_with_embedder(&conn, "authentication", &at_now(), None);

        assert_eq!(
            outcome.map(seen).map_err(|e| e.to_string()),
            Ok((Vec::new(), false, true)),
            "{literal}"
        );
    }
}

#[test]
fn a_not_operand_matching_a_row_whose_session_id_is_not_utf8_text_keeps_the_vector_only_session() {
    for literal in NOT_UTF8_TEXT_SESSION_IDS {
        let (_dir, conn) = setup_test_db();
        add_session(&conn, "text-hit", "claude", "/proj", NOW_MS);
        add_message(&conn, "text-hit", "alpha notes");
        add_session(&conn, "vector", "claude", "/proj", NOW_MS);
        add_message(&conn, "vector", "weather topic");
        add_chunk(&conn, 1, "vector", "alpha", None);
        add_message_with_stored_session_id(&conn, literal, "beta stray");

        let seen = run(&conn, "alpha NOT beta", &at_now(), true);

        assert_eq!(sorted_ids(&seen), ["text-hit", "vector"], "{literal}");
        assert!(
            !seen.1,
            "the NOT filter on {literal} must not degrade the search"
        );
    }
}

#[test]
fn a_session_row_that_cannot_hold_a_range_makes_the_search_read_every_session_id() {
    let session_rows = [
        "INSERT INTO sessions (session_id, message_rowid_lo, message_rowid_hi) VALUES (x'0102', 1, 0)",
        "INSERT INTO sessions (session_id, message_rowid_lo, message_rowid_hi) VALUES ('text-range', 'one', 'zero')",
        "INSERT INTO sessions (session_id, message_rowid_lo, message_rowid_hi) VALUES ('real-range', 1.5, 2.5)",
    ];
    for session_row in session_rows {
        let (_dir, conn) = setup_test_db();
        seed_corpus(&conn);
        let searches = corpus_searches();
        let observe = |conn: &Connection| {
            let seen: Vec<Seen> = searches
                .iter()
                .map(|(q, opts, hybrid)| run(conn, q, opts, *hybrid))
                .collect();
            let count = count_excluded_automated(conn, Some("authentication"), &at_now())
                .map_err(|e| e.to_string());
            (seen, count)
        };
        let without_ranges = observe(&conn);
        record_message_ranges(&conn);

        conn.execute(session_row, []).unwrap();

        assert!(
            matches!(
                MessageSessions::load(&conn).unwrap(),
                MessageSessions::StoredPerRow
            ),
            "{session_row}"
        );
        assert_eq!(observe(&conn), without_ranges, "{session_row}");
        assert_eq!(without_ranges.1, Ok(1));
    }
}
