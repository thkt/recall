use super::*;

fn seed_classify_db() -> (tempfile::TempDir, Connection) {
    let (dir, conn) = db::setup_test_db();
    conn.execute(
            "INSERT INTO sessions (session_id, source, file_path, project, slug, timestamp, mtime, session_type) VALUES ('auto', 'claude', '/f', '/p', 'a', 0, 0.0, NULL)",
            [],
        )
        .unwrap();
    conn.execute(
            "INSERT INTO messages (session_id, role, text) VALUES ('auto', 'user', '<command-message>clear</command-message>')",
            [],
        )
        .unwrap();
    conn.execute(
            "INSERT INTO sessions (session_id, source, file_path, project, slug, timestamp, mtime, session_type) VALUES ('human', 'claude', '/f', '/p', 'h', 0, 0.0, NULL)",
            [],
        )
        .unwrap();
    conn.execute(
            "INSERT INTO messages (session_id, role, text) VALUES ('human', 'user', 'how do I implement auth')",
            [],
        )
        .unwrap();
    (dir, conn)
}

fn session_type_of(conn: &Connection, sid: &str) -> Option<String> {
    conn.query_row(
        "SELECT session_type FROM sessions WHERE session_id = ?1",
        [sid],
        |r| r.get(0),
    )
    .unwrap()
}

#[test]
fn classify_all_labels_every_session_from_its_first_turn() {
    let (_dir, mut conn) = seed_classify_db();
    let outcomes = reclassify_sessions(&mut conn, true, false).unwrap();
    assert_eq!(session_type_of(&conn, "auto").as_deref(), Some("automated"));
    assert_eq!(
        session_type_of(&conn, "human").as_deref(),
        Some("interactive")
    );
    assert_eq!(
        outcomes.len(),
        2,
        "both sessions with a user turn are classified"
    );
}

#[test]
fn classify_without_all_labels_only_unlabeled_sessions() {
    let (_dir, mut conn) = db::setup_test_db();
    conn.execute(
            "INSERT INTO sessions (session_id, source, file_path, project, slug, timestamp, mtime, session_type) VALUES ('fresh', 'claude', '/f', '/p', 'f', 0, 0.0, NULL)",
            [],
        )
        .unwrap();
    conn.execute(
            "INSERT INTO messages (session_id, role, text) VALUES ('fresh', 'user', '<command-message>clear</command-message>')",
            [],
        )
        .unwrap();
    conn.execute(
            "INSERT INTO sessions (session_id, source, file_path, project, slug, timestamp, mtime, session_type) VALUES ('tagged', 'claude', '/f', '/p', 't', 0, 0.0, 'interactive')",
            [],
        )
        .unwrap();
    conn.execute(
            "INSERT INTO messages (session_id, role, text) VALUES ('tagged', 'user', '<command-message>clear</command-message>')",
            [],
        )
        .unwrap();

    let outcomes = reclassify_sessions(&mut conn, false, false).unwrap();

    assert_eq!(
        session_type_of(&conn, "fresh").as_deref(),
        Some("automated"),
        "an unclassified session is classified"
    );
    assert_eq!(
        session_type_of(&conn, "tagged").as_deref(),
        Some("interactive"),
        "an already-tagged session is untouched without --all"
    );
    assert_eq!(
        outcomes.len(),
        1,
        "only the unclassified session is processed"
    );
}
