use super::*;
use crate::db::setup_test_db;
use crate::embedder::MockEmbedder;

const TS: i64 = 1709251200000;

fn db_with_messages(messages: &[(&str, &str)]) -> (tempfile::TempDir, Connection) {
    let (dir, conn) = setup_test_db();
    for (sid, text) in messages {
        conn.execute(
            "INSERT INTO sessions (session_id, source, file_path, project, slug, timestamp, mtime) VALUES (?1, 'claude', '', '/proj', ?1, ?2, 0.0)",
            rusqlite::params![sid, TS],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO messages (session_id, role, text) VALUES (?1, 'user', ?2)",
            rusqlite::params![sid, text],
        )
        .unwrap();
    }
    (dir, conn)
}

fn matched_sessions(conn: &Connection, query: &str) -> Vec<String> {
    let mut ids: Vec<String> = search(conn, query, &SearchOptions::default())
        .unwrap()
        .into_iter()
        .map(|r| r.session.session_id)
        .collect();
    ids.sort();
    ids
}

#[test]
fn lowercase_and_or_not_in_a_query_are_matched_as_ordinary_words() {
    let (_dir, conn) = db_with_messages(&[
        ("with-and", "tokio and async runtime"),
        ("no-and", "tokio async runtime"),
        ("with-not", "please do not panic here"),
        ("no-not", "please do panic here"),
        ("with-or", "pick red or blue paint"),
        ("no-or", "pick red blue paint"),
    ]);

    assert_eq!(matched_sessions(&conn, "tokio and async"), ["with-and"]);
    assert_eq!(matched_sessions(&conn, "do not panic"), ["with-not"]);
    // "or" is under 3 characters, so it is dropped like any other short word
    // and both messages match on the remaining words.
    assert_eq!(matched_sessions(&conn, "red or blue"), ["no-or", "with-or"]);
}

#[test]
fn a_short_query_term_matches_the_indexed_words_that_start_with_it() {
    let (_dir, conn) = db_with_messages(&[
        ("kube", "kubernetes cluster setup"),
        ("swarm", "docker swarm setup"),
    ]);

    assert_eq!(matched_sessions(&conn, "ku"), ["kube"]);
    assert_eq!(matched_sessions(&conn, "ku setup"), ["kube"]);
}

#[test]
fn a_short_query_term_matches_an_indexed_word_that_holds_a_quote_character() {
    let (_dir, conn) = db_with_messages(&[
        ("quoted", "she said q\"uick replies"),
        ("plain", "nothing to see here"),
    ]);

    assert_eq!(matched_sessions(&conn, "q"), ["quoted"]);
}

#[test]
fn a_trailing_uppercase_or_with_nothing_after_it_is_ignored() {
    let (_dir, conn) = db_with_messages(&[
        ("hello", "hello world everyone"),
        ("other", "something unrelated entirely"),
    ]);

    assert_eq!(matched_sessions(&conn, "hello OR"), ["hello"]);
}

#[test]
fn a_term_with_hyphens_finds_the_message_containing_it() {
    let (_dir, conn) = db_with_messages(&[
        ("hyphen", "we added a rate-limit to the gateway"),
        ("apart", "the rate was raised and the limit was lifted"),
    ]);

    assert_eq!(matched_sessions(&conn, "rate-limit"), ["hyphen"]);
}

#[test]
fn a_term_with_colons_finds_the_message_containing_it() {
    let (_dir, conn) = db_with_messages(&[
        ("colon", "import std::collections for the map"),
        ("apart", "std was imported and collections were listed"),
    ]);

    assert_eq!(matched_sessions(&conn, "std::collections"), ["colon"]);
}

#[test]
fn a_quoted_phrase_finds_the_message_with_those_words_adjacent_and_not_one_with_them_apart() {
    let (_dir, conn) = db_with_messages(&[
        ("adjacent", "restart the connection pool before deploy"),
        ("apart", "the pool needs a connection restart before deploy"),
    ]);

    assert_eq!(matched_sessions(&conn, "\"connection pool\""), ["adjacent"]);
}

#[test]
fn a_quoted_phrase_keeps_the_and_or_not_words_inside_it() {
    let (_dir, conn) = db_with_messages(&[
        ("hamlet", "to be or not to be, that is the question"),
        ("repeated", "to be to be continued"),
        ("whether", "whether or not it works today"),
        ("whether-short", "whether it works today"),
    ]);

    assert_eq!(
        matched_sessions(&conn, "\"to be or not to be\""),
        ["hamlet"]
    );
    assert_eq!(
        matched_sessions(&conn, "\"whether or not it works\""),
        ["whether"]
    );
}

#[test]
fn a_quoted_phrase_inside_parentheses_leaves_no_parenthesis_term_behind() {
    let (_dir, conn) = db_with_messages(&[
        ("adjacent", "restart the connection pool before deploy"),
        ("parens", "call init(pool) twice"),
    ]);

    assert_eq!(
        matched_sessions(&conn, "(\"connection pool\")"),
        ["adjacent"]
    );
}

#[test]
fn a_quoted_phrase_attached_to_a_colon_term_keeps_its_boundaries() {
    let (_dir, conn) = db_with_messages(&[
        ("some-value", "set key:\"some value\" in the foo config"),
        ("other-value", "set key:\"other value\" in the foo config"),
    ]);

    assert_eq!(
        matched_sessions(&conn, "key:\"some value\" (foo)"),
        ["some-value"]
    );
}

#[test]
fn an_unclosed_quote_finds_the_message_containing_the_quoted_word() {
    let (_dir, conn) = db_with_messages(&[
        ("quoted", "debugging the tokio scheduler"),
        ("other", "debugging the rayon scheduler"),
    ]);

    assert_eq!(matched_sessions(&conn, "\"tokio"), ["quoted"]);
}

#[test]
fn an_explicit_and_between_two_words_finds_the_message_containing_both() {
    let (_dir, conn) = db_with_messages(&[
        ("both", "alpha release with beta tag"),
        ("alpha-only", "alpha release without the other"),
        ("beta-only", "beta tag without the other"),
    ]);

    assert_eq!(matched_sessions(&conn, "alpha AND beta"), ["both"]);
}

#[test]
fn an_or_finds_a_message_containing_either_word() {
    let (_dir, conn) = db_with_messages(&[
        ("alpha-only", "alpha release notes"),
        ("beta-only", "beta tag notes"),
        ("neither", "gamma release notes"),
    ]);

    assert_eq!(
        matched_sessions(&conn, "alpha OR beta"),
        ["alpha-only", "beta-only"]
    );
}

#[test]
fn parentheses_group_an_or_before_the_and_that_follows_it() {
    let (_dir, conn) = db_with_messages(&[
        ("alpha-gamma", "alpha with gamma"),
        ("beta-gamma", "beta with gamma"),
        ("alpha-beta", "alpha with beta"),
        ("gamma-only", "gamma on its own"),
    ]);

    assert_eq!(
        matched_sessions(&conn, "(alpha OR beta) AND gamma"),
        ["alpha-gamma", "beta-gamma"]
    );
}

/// The `RecallError` a rejected query fails with, in its `Debug` form so a test
/// can compare the variant and the message as one literal.
fn rejection(conn: &Connection, query: &str) -> String {
    match search(conn, query, &SearchOptions::default()) {
        Ok(results) => panic!(
            "query {query:?} should be rejected but matched {} sessions",
            results.len()
        ),
        Err(err) => format!(
            "{:?}",
            err.downcast::<RecallError>()
                .expect("a rejected query fails with a RecallError")
        ),
    }
}

#[test]
fn a_not_excludes_messages_containing_the_word_after_it() {
    let (_dir, conn) = db_with_messages(&[
        ("alpha-only", "alpha release notes"),
        ("alpha-beta", "alpha release with beta tag"),
        ("beta-only", "beta tag notes"),
    ]);

    assert_eq!(matched_sessions(&conn, "alpha NOT beta"), ["alpha-only"]);
}

#[test]
fn a_short_word_after_not_is_ignored_and_the_rest_of_the_query_still_matches() {
    let (_dir, conn) = db_with_messages(&[
        ("with-short", "alpha release zz marker"),
        ("without-short", "alpha release notes"),
        ("unrelated", "gamma release notes"),
    ]);

    assert_eq!(
        matched_sessions(&conn, "alpha NOT zz"),
        ["with-short", "without-short"]
    );
}

#[test]
fn a_short_word_before_not_is_rejected_as_a_query_syntax_error() {
    let (_dir, conn) = db_with_messages(&[
        ("short-only", "zz marker notes"),
        ("short-beta", "zz marker with beta tag"),
    ]);

    assert_eq!(
        rejection(&conn, "zz NOT beta"),
        r#"QuerySyntax("NOT needs a left term of at least 3 characters")"#
    );
}

#[test]
fn a_not_with_nothing_before_it_is_rejected_as_a_query_syntax_error() {
    let (_dir, conn) = db_with_messages(&[
        ("alpha-only", "alpha release notes"),
        ("alpha-beta", "alpha release with beta tag"),
    ]);

    assert_eq!(
        rejection(&conn, "NOT beta"),
        r#"QuerySyntax("NOT needs a term on its left")"#
    );
}

fn add_embedded_chunk(conn: &Connection, chunk_id: i64, sid: &str, content: &str) {
    conn.execute(
        "INSERT INTO qa_chunks (id, session_id, content, timestamp) VALUES (?1, ?2, ?3, ?4)",
        rusqlite::params![chunk_id, sid, content, TS],
    )
    .unwrap();
    let embedding = MockEmbedder::deterministic_vector(content);
    conn.execute(
        "INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?1, ?2)",
        rusqlite::params![chunk_id, f32_as_bytes(&embedding)],
    )
    .unwrap();
}

fn hybrid_matched_sessions(conn: &Connection, query: &str) -> Vec<String> {
    let embedder = MockEmbedder::new();
    let opts = SearchOptions {
        now_ms: Some(TS),
        ..Default::default()
    };
    let mut ids: Vec<String> = search_with_embedder(conn, query, &opts, Some(&embedder))
        .unwrap()
        .results
        .into_iter()
        .map(|r| r.session.session_id)
        .collect();
    ids.sort();
    ids
}

#[test]
fn a_session_found_only_by_the_vector_search_is_dropped_when_it_matches_the_word_after_not() {
    let (_dir, conn) = db_with_messages(&[
        ("text-hit", "alpha release notes"),
        ("vec-with-beta", "beta tag notes"),
    ]);
    add_embedded_chunk(&conn, 1, "vec-with-beta", "beta tag notes");

    assert_eq!(
        hybrid_matched_sessions(&conn, "alpha NOT beta"),
        ["text-hit"]
    );
}

#[test]
fn a_session_found_only_by_the_vector_search_stays_when_it_does_not_match_the_word_after_not() {
    let (_dir, conn) = db_with_messages(&[
        ("text-hit", "alpha release notes"),
        ("vec-without-beta", "gamma tag notes"),
    ]);
    add_embedded_chunk(&conn, 1, "vec-without-beta", "gamma tag notes");

    assert_eq!(
        hybrid_matched_sessions(&conn, "alpha NOT beta"),
        ["text-hit", "vec-without-beta"]
    );
}

#[test]
fn a_session_found_only_by_the_vector_search_stays_when_the_not_sits_under_an_or() {
    let (_dir, conn) = db_with_messages(&[
        ("text-hit", "alpha release notes"),
        ("vec-with-beta", "beta tag notes"),
    ]);
    add_embedded_chunk(&conn, 1, "vec-with-beta", "beta tag notes");

    assert_eq!(
        hybrid_matched_sessions(&conn, "(alpha NOT beta) OR gamma"),
        ["text-hit", "vec-with-beta"]
    );
}

#[test]
fn a_session_found_only_by_the_vector_search_stays_when_it_matches_a_not_nested_after_another_not()
{
    let (_dir, conn) = db_with_messages(&[
        ("text-hit", "alpha release notes"),
        ("vec-with-gamma", "gamma tag notes"),
    ]);
    add_embedded_chunk(&conn, 1, "vec-with-gamma", "gamma tag notes");

    assert_eq!(
        hybrid_matched_sessions(&conn, "alpha NOT (beta NOT gamma)"),
        ["text-hit", "vec-with-gamma"]
    );
}

fn hybrid_result_count_and_embed_calls(conn: &Connection, query: &str) -> (usize, usize) {
    let embedder = MockEmbedder::new();
    let opts = SearchOptions {
        now_ms: Some(TS),
        ..Default::default()
    };
    let results = search_with_embedder(conn, query, &opts, Some(&embedder))
        .unwrap()
        .results;
    (results.len(), embedder.calls())
}

#[test]
fn a_query_of_only_operators_matches_no_session_and_embeds_nothing() {
    let (_dir, conn) = db_with_messages(&[("vec-hit", "alpha release notes")]);
    add_embedded_chunk(&conn, 1, "vec-hit", "alpha release notes");

    assert_eq!(hybrid_result_count_and_embed_calls(&conn, "AND"), (0, 0));
    assert_eq!(hybrid_result_count_and_embed_calls(&conn, "OR"), (0, 0));
    assert_eq!(
        hybrid_result_count_and_embed_calls(&conn, "( AND )"),
        (0, 0)
    );
}

#[test]
fn the_excluded_automated_count_includes_an_automated_session_matching_an_operator_query() {
    let (_dir, conn) = db_with_messages(&[
        ("auto-alpha", "alpha release notes"),
        ("auto-alpha-beta", "alpha release with beta tag"),
        ("auto-gamma", "gamma release notes"),
        ("human-alpha", "alpha deployment notes"),
    ]);
    for sid in ["auto-alpha", "auto-alpha-beta", "auto-gamma"] {
        conn.execute(
            "UPDATE sessions SET session_type = 'automated' WHERE session_id = ?1",
            [sid],
        )
        .unwrap();
    }

    let count =
        count_excluded_automated(&conn, Some("alpha NOT beta"), &SearchOptions::default()).unwrap();

    assert_eq!(
        count, 1,
        "only auto-alpha matches `alpha NOT beta` among the automated sessions"
    );
}
