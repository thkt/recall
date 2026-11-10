use super::*;
use crate::db::setup_test_db;

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
    assert_eq!(
        matched_sessions(&conn, "red or blue"),
        ["no-or", "with-or"]
    );
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
fn a_trailing_uppercase_or_with_nothing_after_it_is_ignored() {
    let (_dir, conn) = db_with_messages(&[
        ("hello", "hello world everyone"),
        ("other", "something unrelated entirely"),
    ]);

    assert_eq!(matched_sessions(&conn, "hello OR"), ["hello"]);
}
