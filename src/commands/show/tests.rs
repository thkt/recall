use std::process::ExitCode;

use crate::envelope::{render_json_error, render_json_success};
use crate::error::{classify_exit_code, error_envelope};
use crate::open_or_create_db;
use crate::tests::bare_empty_db;

use super::*;

/// The `--json` envelope `recall show` prints for an output.
fn printed_envelope(out: &CommandOutput) -> serde_json::Value {
    serde_json::from_str(&render_json_success(out)).unwrap()
}

/// What a failing `recall show` reports: its exit code and the `--json` error envelope.
fn reported(err: &anyhow::Error) -> (ExitCode, String) {
    (
        classify_exit_code(err),
        render_json_error(&error_envelope(err)),
    )
}

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
fn show_prints_every_message_of_an_exactly_matching_session_under_its_metadata() {
    let (dir, conn) = setup_show_db();
    drop(conn);

    let out = run_show(
        "abc-123",
        false,
        None,
        DEFAULT_WINDOW,
        &Some(dir.path().join("test.db")),
    )
    .unwrap();

    assert_eq!(
        out.markdown,
        concat!(
            "# my-slug\n",
            "- session_id: abc-123\n",
            "- date: 2024-03-01\n",
            "- project: /proj\n",
            "- source: claude\n",
            "\n",
            "## [user]\n",
            "hello world\n",
            "\n",
            "## [assistant]\n",
            "hi there\n",
            "\n",
        )
    );
    assert_eq!(
        printed_envelope(&out),
        serde_json::json!({
            "data": {
                "messages": [
                    { "role": "user", "text": "hello world" },
                    { "role": "assistant", "text": "hi there" },
                ],
                "project": "/proj",
                "session_id": "abc-123",
                "slug": "my-slug",
                "source": "claude",
                "timestamp": 1709251200000_i64,
            },
            "degraded": false,
            "notes": [],
        })
    );
}

#[test]
fn show_of_a_unique_id_prefix_prints_that_session() {
    let (dir, conn) = setup_show_db();
    drop(conn);

    let out = run_show(
        "abc-1",
        false,
        None,
        DEFAULT_WINDOW,
        &Some(dir.path().join("test.db")),
    )
    .unwrap();

    assert_eq!(
        out.markdown,
        concat!(
            "# my-slug\n",
            "- session_id: abc-123\n",
            "- date: 2024-03-01\n",
            "- project: /proj\n",
            "- source: claude\n",
            "\n",
            "## [user]\n",
            "hello world\n",
            "\n",
            "## [assistant]\n",
            "hi there\n",
            "\n",
        )
    );
    assert_eq!(
        printed_envelope(&out),
        serde_json::json!({
            "data": {
                "messages": [
                    { "role": "user", "text": "hello world" },
                    { "role": "assistant", "text": "hi there" },
                ],
                "project": "/proj",
                "session_id": "abc-123",
                "slug": "my-slug",
                "source": "claude",
                "timestamp": 1709251200000_i64,
            },
            "degraded": false,
            "notes": [],
        })
    );
}

#[test]
fn show_of_a_prefix_shared_by_two_sessions_is_a_usage_error_listing_both() {
    let (dir, conn) = setup_show_db();
    drop(conn);

    let err = run_show(
        "abc",
        false,
        None,
        DEFAULT_WINDOW,
        &Some(dir.path().join("test.db")),
    )
    .unwrap_err();

    assert_eq!(
        reported(&err),
        (
            ExitCode::from(64),
            r#"{"error":{"code":"USAGE_ERROR","message":"Multiple sessions match 'abc':\n  abc-123  my-slug\n  abc-456  other-slug\nNarrow the session ID prefix to match exactly one session","next_step":"See `recall --help`; if the index is missing, run `recall index` first.","retryable":false}}"#.to_owned()
        )
    );
}

#[test]
fn show_of_an_unknown_id_is_a_usage_error_naming_it() {
    let (dir, conn) = setup_show_db();
    drop(conn);

    let err = run_show(
        "zzz",
        false,
        None,
        DEFAULT_WINDOW,
        &Some(dir.path().join("test.db")),
    )
    .unwrap_err();

    assert_eq!(
        reported(&err),
        (
            ExitCode::from(64),
            r#"{"error":{"code":"USAGE_ERROR","message":"No session found matching 'zzz'","next_step":"See `recall --help`; if the index is missing, run `recall index` first.","retryable":false}}"#.to_owned()
        )
    );
}

#[test]
fn verbose_show_adds_the_session_file_to_the_header() {
    let (dir, conn) = setup_show_db();
    drop(conn);

    let out = run_show(
        "abc-123",
        true,
        None,
        DEFAULT_WINDOW,
        &Some(dir.path().join("test.db")),
    )
    .unwrap();

    assert_eq!(
        out.markdown,
        concat!(
            "# my-slug\n",
            "- session_id: abc-123\n",
            "- date: 2024-03-01\n",
            "- project: /proj\n",
            "- source: claude\n",
            "- file: /path/f.jsonl\n",
            "\n",
            "## [user]\n",
            "hello world\n",
            "\n",
            "## [assistant]\n",
            "hi there\n",
            "\n",
        )
    );
    assert_eq!(
        printed_envelope(&out),
        serde_json::json!({
            "data": {
                "messages": [
                    { "role": "user", "text": "hello world" },
                    { "role": "assistant", "text": "hi there" },
                ],
                "project": "/proj",
                "session_id": "abc-123",
                "slug": "my-slug",
                "source": "claude",
                "timestamp": 1709251200000_i64,
            },
            "degraded": false,
            "notes": [],
        })
    );
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
fn show_prints_an_unknown_stored_source_as_claude() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    open_or_create_db(&db_path)
        .unwrap()
        .execute(
            "INSERT INTO sessions (session_id, source, file_path, project, slug, timestamp, mtime)
                 VALUES ('xyz-1', 'bogus', '/f', '/p', 'odd-slug', 1709251200000, 0.0)",
            [],
        )
        .unwrap();

    let out = run_show("xyz-1", false, None, DEFAULT_WINDOW, &Some(db_path)).unwrap();

    assert_eq!(
        out.markdown,
        concat!(
            "# odd-slug\n",
            "- session_id: xyz-1\n",
            "- date: 2024-03-01\n",
            "- project: /p\n",
            "- source: claude\n",
            "\n",
        )
    );
    assert_eq!(
        printed_envelope(&out),
        serde_json::json!({
            "data": {
                "messages": [],
                "project": "/p",
                "session_id": "xyz-1",
                "slug": "odd-slug",
                "source": "claude",
                "timestamp": 1709251200000_i64,
            },
            "degraded": false,
            "notes": [],
        })
    );
}

#[test]
fn windowed_show_prints_only_same_session_messages_within_the_window() {
    // The two sessions' messages interleave, so a rowid window that ignored the
    // session would pull in sessB's messages.
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    let conn = open_or_create_db(&db_path).unwrap();
    db::seed_session(&conn, "sessA");
    db::seed_session(&conn, "sessB");
    conn.execute_batch(
        "INSERT INTO messages (rowid, session_id, role, text) VALUES
             (1, 'sessA', 'user', 'AMSG0'), (2, 'sessB', 'user', 'BMSG0'),
             (3, 'sessA', 'user', 'AMSG1'), (4, 'sessB', 'user', 'BMSG1'),
             (5, 'sessA', 'user', 'AMSG2'), (6, 'sessB', 'user', 'BMSG2'),
             (7, 'sessA', 'user', 'AMSG3_BODY'), (8, 'sessB', 'user', 'BMSG3'),
             (9, 'sessA', 'user', 'AMSG4'), (10, 'sessB', 'user', 'BMSG4'),
             (11, 'sessA', 'user', 'AMSG5'), (12, 'sessB', 'user', 'BMSG5'),
             (13, 'sessA', 'user', 'AMSG6'), (14, 'sessB', 'user', 'BMSG6');
         INSERT INTO qa_chunks (id, session_id, content, timestamp, src_rowid_lo, src_rowid_hi)
             VALUES (100, 'sessA', 'AMSG3_BODY', 0, 7, 7);",
    )
    .unwrap();
    drop(conn);

    let out = run_show("sessA", false, Some(100), 2, &Some(db_path)).unwrap();

    assert_eq!(
        out.markdown,
        concat!(
            "# slug\n",
            "- session_id: sessA\n",
            "- date: 1970-01-01\n",
            "- project: /p\n",
            "- source: claude\n",
            "- chunk: 100\n",
            "- window: 2\n",
            "\n",
            "## [user]\n",
            "AMSG1\n",
            "\n",
            "## [user]\n",
            "AMSG2\n",
            "\n",
            "## [user] (chunk)\n",
            "AMSG3_BODY\n",
            "\n",
            "## [user]\n",
            "AMSG4\n",
            "\n",
            "## [user]\n",
            "AMSG5\n",
            "\n",
        )
    );
    assert_eq!(
        printed_envelope(&out),
        serde_json::json!({
            "data": {
                "chunk": 100,
                "messages": [
                    { "in_chunk": false, "role": "user", "rowid": 3, "text": "AMSG1" },
                    { "in_chunk": false, "role": "user", "rowid": 5, "text": "AMSG2" },
                    { "in_chunk": true, "role": "user", "rowid": 7, "text": "AMSG3_BODY" },
                    { "in_chunk": false, "role": "user", "rowid": 9, "text": "AMSG4" },
                    { "in_chunk": false, "role": "user", "rowid": 11, "text": "AMSG5" },
                ],
                "project": "/p",
                "session_id": "sessA",
                "slug": "slug",
                "source": "claude",
                "timestamp": 0,
                "window": 2,
            },
            "degraded": false,
            "notes": [],
        })
    );
}

#[test]
fn verbose_show_of_a_chunk_without_a_message_range_prints_its_body_and_points_to_the_full_show() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    let conn = open_or_create_db(&db_path).unwrap();
    db::seed_session(&conn, "s1");
    db::seed_chunk(&conn, 200, "legacy chunk body");
    drop(conn);

    let out = run_show("s1", true, Some(200), 3, &Some(db_path)).unwrap();

    assert_eq!(
        out.markdown,
        concat!(
            "# slug\n",
            "- session_id: s1\n",
            "- date: 1970-01-01\n",
            "- project: /p\n",
            "- source: claude\n",
            "- chunk: 200\n",
            "- file: /f\n",
            "\n",
            "legacy chunk body\n",
            "\n",
            "This chunk predates message-range tracking; run `recall show s1` for the full conversation.\n",
        )
    );
    assert_eq!(
        printed_envelope(&out),
        serde_json::json!({
            "data": {
                "chunk": 200,
                "content": "legacy chunk body",
                "next_step": "This chunk predates message-range tracking; run `recall show s1` for the full conversation.",
                "project": "/p",
                "session_id": "s1",
                "slug": "slug",
                "source": "claude",
                "timestamp": 0,
                "windowed": false,
            },
            "degraded": false,
            "notes": [],
        })
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
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    db::seed_session(&open_or_create_db(&db_path).unwrap(), "w1");

    let err = run_show("w1", false, Some(999), 2, &Some(db_path)).unwrap_err();

    assert_eq!(
        reported(&err),
        (
            ExitCode::from(64),
            r#"{"error":{"code":"USAGE_ERROR","message":"No chunk found with id 999","next_step":"See `recall --help`; if the index is missing, run `recall index` first.","retryable":false}}"#.to_owned()
        )
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
fn show_on_a_database_file_without_tables_is_a_usage_error_asking_to_run_index() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = bare_empty_db(db_dir.path());

    let err = run_show("deadbeef", false, None, DEFAULT_WINDOW, &Some(db_path)).unwrap_err();

    assert!(
        err.to_string().contains("Database not found"),
        "empty DB show must report not-found, got: {err}"
    );
}
