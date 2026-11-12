use crate::open_or_create_db;
use crate::tests::bare_empty_db;

use super::*;

fn setup_show_db() -> (tempfile::TempDir, Connection) {
    let (dir, conn) = db::setup_test_db();
    conn.execute(
            "INSERT INTO sessions (session_id, source, file_path, project, slug, timestamp, mtime) VALUES ('abc-123', 'claude', '/path/f.jsonl', '/proj', 'my-slug', 1709251200000, 0.0)",
            [],
        )
        .unwrap();
    conn.execute(
            "INSERT INTO sessions (session_id, source, file_path, project, slug, timestamp, mtime) VALUES ('abc-456', 'claude', '/path/g.jsonl', '/proj', 'other-slug', 1709251200000, 0.0)",
            [],
        )
        .unwrap();
    conn.execute(
        "INSERT INTO messages (session_id, role, text) VALUES ('abc-123', 'user', 'hello world')",
        [],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO messages (session_id, role, text) VALUES ('abc-123', 'assistant', 'hi there')",
        [],
    )
    .unwrap();
    (dir, conn)
}

#[test]
fn test_show_session_exact_match() {
    let (_dir, conn) = setup_show_db();
    let out = show_session(&conn, "abc-123", false).unwrap();
    assert!(out.markdown.contains("# my-slug"));
    assert!(out.markdown.contains("- session_id: abc-123"));
    assert!(out.markdown.contains("- date: 2024-03-01"));
    assert!(out.markdown.contains("## [user]\nhello world"));
    assert!(out.markdown.contains("## [assistant]\nhi there"));
    assert_eq!(out.data["session_id"], "abc-123");
    assert_eq!(out.data["messages"][0]["role"], "user");
    assert_eq!(out.data["messages"][0]["text"], "hello world");
}

#[test]
fn test_show_session_prefix_match() {
    let (_dir, conn) = setup_show_db();
    let out = show_session(&conn, "abc-1", false).unwrap();
    assert!(out.markdown.contains("- session_id: abc-123"));
}

#[test]
fn test_show_session_ambiguous_prefix() {
    let (_dir, conn) = setup_show_db();
    let err = show_session(&conn, "abc", false).unwrap_err();
    assert!(err.to_string().contains("Multiple sessions match"));
}

#[test]
fn test_show_session_not_found() {
    let (_dir, conn) = setup_show_db();
    let err = show_session(&conn, "zzz", false).unwrap_err();
    assert!(err.to_string().contains("No session found matching"));
}

#[test]
fn test_show_session_verbose_includes_file() {
    let (_dir, conn) = setup_show_db();
    let out = show_session(&conn, "abc-123", true).unwrap();
    assert!(out.markdown.contains("- file: /path/f.jsonl"));
}

#[test]
fn test_show_session_message_order() {
    let (_dir, conn) = setup_show_db();
    let out = show_session(&conn, "abc-123", false).unwrap();
    let user_pos = out.markdown.find("## [user]").unwrap();
    let assistant_pos = out.markdown.find("## [assistant]").unwrap();
    assert!(
        user_pos < assistant_pos,
        "user message should come before assistant"
    );
}

#[test]
fn show_session_unknown_source_falls_back_to_claude() {
    let (_dir, conn) = db::setup_test_db();
    conn.execute(
            "INSERT INTO sessions (session_id, source, file_path, project, slug, timestamp, mtime) VALUES ('xyz-1', 'bogus', '/f', '/p', 'odd-slug', 1709251200000, 0.0)",
            [],
        )
        .unwrap();
    let out = show_session(&conn, "xyz-1", false).unwrap();
    assert!(
        out.markdown.contains("- source: claude"),
        "an unknown stored source must fall back to claude, got: {}",
        out.markdown
    );
}

#[test]
fn window_returns_only_same_session_neighbors_within_n() {
    let (_dir, conn) = db::setup_test_db();
    db::seed_session(&conn, "sessA");
    db::seed_session(&conn, "sessB");
    for (sess, text) in [
        ("sessA", "AMSG0"),
        ("sessB", "BMSG0"),
        ("sessA", "AMSG1"),
        ("sessB", "BMSG1"),
        ("sessA", "AMSG2"),
        ("sessB", "BMSG2"),
        ("sessA", "AMSG3_BODY"),
        ("sessB", "BMSG3"),
        ("sessA", "AMSG4"),
        ("sessB", "BMSG4"),
        ("sessA", "AMSG5"),
        ("sessB", "BMSG5"),
        ("sessA", "AMSG6"),
        ("sessB", "BMSG6"),
    ] {
        conn.execute(
            "INSERT INTO messages (session_id, role, text) VALUES (?1, 'user', ?2)",
            rusqlite::params![sess, text],
        )
        .unwrap();
    }
    let body: i64 = conn
        .query_row(
            "SELECT rowid FROM messages WHERE session_id='sessA' AND text='AMSG3_BODY'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    conn.execute(
        "INSERT INTO qa_chunks (id, session_id, content, timestamp, src_rowid_lo, src_rowid_hi) \
             VALUES (100, 'sessA', 'AMSG3_BODY', 0, ?1, ?1)",
        [body],
    )
    .unwrap();

    let out = show_session_windowed(&conn, "sessA", false, Some(100), 2).unwrap();

    for included in ["AMSG1", "AMSG2", "AMSG3_BODY", "AMSG4", "AMSG5"] {
        assert!(
            out.markdown.contains(included),
            "window=2 must include same-session {included}, got: {}",
            out.markdown
        );
    }
    assert!(
        !out.markdown.contains("AMSG0"),
        "before-cap: the 3rd-earlier same-session message must be excluded, got: {}",
        out.markdown
    );
    assert!(
        !out.markdown.contains("AMSG6"),
        "after-cap: the 3rd-later same-session message must be excluded, got: {}",
        out.markdown
    );
    assert!(
        !out.markdown.contains("BMSG"),
        "must not cross into the adjacent session sharing interleaved rowids, got: {}",
        out.markdown
    );
}

#[test]
fn null_src_rowid_chunk_falls_back_to_body_plus_full_show_hint() {
    let (_dir, conn) = db::setup_test_db();
    db::seed_session(&conn, "s1");
    db::seed_chunk(&conn, 200, "legacy chunk body TEXTMARK");

    let out = show_session_windowed(&conn, "s1", false, Some(200), 3).unwrap();

    assert!(
        out.markdown.contains("legacy chunk body TEXTMARK"),
        "the legacy chunk's own body must be shown, got: {}",
        out.markdown
    );
    assert!(
        out.markdown.contains("recall show"),
        "must steer to the full `recall show` for the whole conversation, got: {}",
        out.markdown
    );
}

fn seed_window_db(count: i64) -> (tempfile::TempDir, Connection) {
    let (dir, conn) = db::setup_test_db();
    db::seed_session(&conn, "w1");
    for i in 0..count {
        conn.execute(
            "INSERT INTO messages (session_id, role, text) VALUES ('w1', 'user', ?1)",
            [format!("M{i}")],
        )
        .unwrap();
    }
    (dir, conn)
}

#[test]
fn windowed_show_errors_when_chunk_id_is_absent() {
    let (_dir, conn) = seed_window_db(3);
    let err = show_session_windowed(&conn, "w1", false, Some(999), 2).unwrap_err();
    assert!(
        err.to_string().contains("No chunk found with id 999"),
        "got: {err}"
    );
}

#[test]
fn windowed_show_errors_when_chunk_belongs_to_another_session() {
    let (_dir, conn) = seed_window_db(3);
    db::seed_session(&conn, "w2");
    conn.execute(
        "INSERT INTO qa_chunks (id, session_id, content, timestamp, src_rowid_lo, src_rowid_hi) \
             VALUES (50, 'w2', 'body', 0, 1, 1)",
        [],
    )
    .unwrap();

    let err = show_session_windowed(&conn, "w1", false, Some(50), 2).unwrap_err();

    assert!(
        err.to_string().contains("belongs to session 'w2'"),
        "cross-session --chunk must name the mismatched session, got: {err}"
    );
}

#[test]
fn windowed_show_truncates_a_message_longer_than_the_cap() {
    let (_dir, conn) = db::setup_test_db();
    db::seed_session(&conn, "w1");
    let long = "X".repeat(MAX_MSG_CHARS + 1);
    conn.execute(
        "INSERT INTO messages (session_id, role, text) VALUES ('w1', 'user', ?1)",
        [&long],
    )
    .unwrap();
    let rowid: i64 = conn
        .query_row(
            "SELECT rowid FROM messages WHERE session_id='w1'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    conn.execute(
        "INSERT INTO qa_chunks (id, session_id, content, timestamp, src_rowid_lo, src_rowid_hi) \
             VALUES (60, 'w1', 'body', 0, ?1, ?1)",
        [rowid],
    )
    .unwrap();

    let out = show_session_windowed(&conn, "w1", false, Some(60), 0).unwrap();

    assert!(
        out.markdown.contains('…'),
        "an over-cap message must be marked truncated"
    );
    assert!(
        !out.markdown.contains(&long),
        "the full over-cap message must not be rendered verbatim"
    );
}

#[test]
fn windowed_show_renders_metadata_only_when_range_matches_no_message() {
    let (_dir, conn) = seed_window_db(2);
    conn.execute(
        "INSERT INTO qa_chunks (id, session_id, content, timestamp, src_rowid_lo, src_rowid_hi) \
             VALUES (70, 'w1', 'orphan-range', 0, 100000, 100001)",
        [],
    )
    .unwrap();

    let out = show_session_windowed(&conn, "w1", false, Some(70), 2).unwrap();

    assert!(
        out.markdown.contains("- chunk: 70"),
        "metadata header must still render, got: {}",
        out.markdown
    );
    assert!(
        !out.markdown.contains("## ["),
        "no message body should render for an out-of-range chunk, got: {}",
        out.markdown
    );
}

#[test]
fn windowed_show_json_payload_lists_anchored_messages_with_in_chunk_flag() {
    let (_dir, conn) = seed_window_db(5);
    conn.execute(
        "INSERT INTO qa_chunks (id, session_id, content, timestamp, src_rowid_lo, src_rowid_hi) \
             VALUES (80, 'w1', 'M2', 0, 3, 3)",
        [],
    )
    .unwrap();

    let out = show_session_windowed(&conn, "w1", false, Some(80), 1).unwrap();

    assert_eq!(out.data["chunk"], 80);
    assert_eq!(out.data["window"], 1);
    let msgs = out.data["messages"].as_array().unwrap();
    assert_eq!(
        msgs.len(),
        3,
        "window=1 around a one-message body yields 3 messages"
    );
    assert_eq!(msgs[1]["text"], "M2");
    assert_eq!(
        msgs[1]["in_chunk"], true,
        "the body message must be flagged in_chunk"
    );
    assert_eq!(
        msgs[0]["in_chunk"], false,
        "a neighbor must not be flagged in_chunk"
    );
    assert_eq!(msgs[2]["in_chunk"], false);
}

#[test]
fn run_show_windowed_reads_a_real_db_and_includes_file_when_verbose() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    let conn = open_or_create_db(&db_path).unwrap();
    db::seed_session(&conn, "s1");
    for text in ["m0", "m1", "m2"] {
        conn.execute(
            "INSERT INTO messages (session_id, role, text) VALUES ('s1', 'user', ?1)",
            [text],
        )
        .unwrap();
    }
    let rowid: i64 = conn
        .query_row("SELECT rowid FROM messages WHERE text='m1'", [], |r| {
            r.get(0)
        })
        .unwrap();
    conn.execute(
        "INSERT INTO qa_chunks (id, session_id, content, timestamp, src_rowid_lo, src_rowid_hi) \
             VALUES (1, 's1', 'm1', 0, ?1, ?1)",
        [rowid],
    )
    .unwrap();
    drop(conn);

    let out = run_show("s1", true, Some(1), 1, &Some(db_path)).unwrap();

    assert!(out.markdown.contains("- chunk: 1"));
    assert!(
        out.markdown.contains("- file: /f"),
        "verbose windowed show must include the file path, got: {}",
        out.markdown
    );
    for m in ["m0", "m1", "m2"] {
        assert!(out.markdown.contains(m), "window=1 must include {m}");
    }
}

#[test]
fn windowed_show_fallback_includes_file_when_verbose() {
    let (_dir, conn) = db::setup_test_db();
    db::seed_session(&conn, "s1");
    db::seed_chunk(&conn, 300, "legacy body");

    let out = show_session_windowed(&conn, "s1", true, Some(300), 3).unwrap();

    assert!(
        out.markdown.contains("- file: /f"),
        "verbose fallback must include the file path, got: {}",
        out.markdown
    );
}

#[test]
fn run_show_on_existing_empty_db_is_not_found_usage_error() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = bare_empty_db(db_dir.path());

    let err = run_show("deadbeef", false, None, DEFAULT_WINDOW, &Some(db_path)).unwrap_err();

    assert!(
        err.to_string().contains("Database not found"),
        "empty DB show must report not-found, got: {err}"
    );
}
