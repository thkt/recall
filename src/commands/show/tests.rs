use std::process::ExitCode;

use crate::open_or_create_db;
use crate::tests::{bare_empty_db, printed_envelope, reported};

use super::*;

/// An index where abc-123 holds a user and an assistant message and abc-456 shares its prefix.
fn setup_show_db() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::TempDir::new().unwrap();
    let db_path = dir.path().join("recall.db");
    open_or_create_db(&db_path)
        .unwrap()
        .execute_batch(
            "INSERT INTO sessions (session_id, source, file_path, project, slug, timestamp, mtime) VALUES
                 ('abc-123', 'claude', '/path/f.jsonl', '/proj', 'my-slug', 1709251200000, 0.0),
                 ('abc-456', 'claude', '/path/g.jsonl', '/proj', 'other-slug', 1709251200000, 0.0);
             INSERT INTO messages (session_id, role, text) VALUES
                 ('abc-123', 'user', 'hello world'),
                 ('abc-123', 'assistant', 'hi there');",
        )
        .unwrap();
    (dir, db_path)
}

#[test]
fn show_prints_every_message_of_an_exactly_matching_session_under_its_metadata() {
    let (_dir, db_path) = setup_show_db();

    let out = run_show("abc-123", false, None, DEFAULT_WINDOW, &Some(db_path)).unwrap();

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
    let (_dir, db_path) = setup_show_db();

    let out = run_show("abc-1", false, None, DEFAULT_WINDOW, &Some(db_path)).unwrap();

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
    let (_dir, db_path) = setup_show_db();

    let err = run_show("abc", false, None, DEFAULT_WINDOW, &Some(db_path)).unwrap_err();

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
    let (_dir, db_path) = setup_show_db();

    let err = run_show("zzz", false, None, DEFAULT_WINDOW, &Some(db_path)).unwrap_err();

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
    let (_dir, db_path) = setup_show_db();

    let out = run_show("abc-123", true, None, DEFAULT_WINDOW, &Some(db_path)).unwrap();

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
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    let conn = open_or_create_db(&db_path).unwrap();
    db::seed_session(&conn, "w1");
    db::seed_session(&conn, "w2");
    conn.execute_batch(
        "INSERT INTO messages (rowid, session_id, role, text) VALUES (1, 'w2', 'user', 'body');
         INSERT INTO qa_chunks (id, session_id, content, timestamp, src_rowid_lo, src_rowid_hi)
             VALUES (50, 'w2', 'body', 0, 1, 1);",
    )
    .unwrap();
    drop(conn);

    let err = run_show("w1", false, Some(50), 2, &Some(db_path)).unwrap_err();

    assert_eq!(
        reported(&err),
        (
            ExitCode::from(64),
            r#"{"error":{"code":"USAGE_ERROR","message":"chunk 50 belongs to session 'w2', not 'w1'","next_step":"See `recall --help`; if the index is missing, run `recall index` first.","retryable":false}}"#.to_owned()
        )
    );
}

#[test]
fn windowed_show_cuts_a_message_at_4000_characters_and_marks_the_cut() {
    // 界 takes three bytes, so both messages exceed 4000 bytes.
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    let conn = open_or_create_db(&db_path).unwrap();
    db::seed_session(&conn, "w1");
    conn.execute(
        "INSERT INTO messages (rowid, session_id, role, text) VALUES
             (1, 'w1', 'user', ?1),
             (2, 'w1', 'user', ?2)",
        [
            format!("{}{}", "a".repeat(3990), "界".repeat(10)),
            format!("{}{}", "a".repeat(3990), "界".repeat(11)),
        ],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO qa_chunks (id, session_id, content, timestamp, src_rowid_lo, src_rowid_hi)
             VALUES (60, 'w1', 'body', 0, 2, 2)",
        [],
    )
    .unwrap();
    drop(conn);

    let out = run_show("w1", false, Some(60), 1, &Some(db_path)).unwrap();

    assert_eq!(
        out.markdown,
        format!(
            concat!(
                "# slug\n",
                "- session_id: w1\n",
                "- date: 1970-01-01\n",
                "- project: /p\n",
                "- source: claude\n",
                "- chunk: 60\n",
                "- window: 1\n",
                "\n",
                "## [user]\n",
                "{}{}\n",
                "\n",
                "## [user] (chunk)\n",
                "{}{}…\n",
                "\n",
            ),
            "a".repeat(3990),
            "界".repeat(10),
            "a".repeat(3990),
            "界".repeat(10),
        )
    );
    assert_eq!(
        printed_envelope(&out),
        serde_json::json!({
            "data": {
                "chunk": 60,
                "messages": [
                    {
                        "in_chunk": false,
                        "role": "user",
                        "rowid": 1,
                        "text": format!("{}{}", "a".repeat(3990), "界".repeat(10)),
                    },
                    {
                        "in_chunk": true,
                        "role": "user",
                        "rowid": 2,
                        "text": format!("{}{}…", "a".repeat(3990), "界".repeat(10)),
                    },
                ],
                "project": "/p",
                "session_id": "w1",
                "slug": "slug",
                "source": "claude",
                "timestamp": 0,
                "window": 1,
            },
            "degraded": false,
            "notes": [],
        })
    );
}

#[test]
fn windowed_show_renders_metadata_only_when_range_matches_no_message() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    let conn = open_or_create_db(&db_path).unwrap();
    db::seed_session(&conn, "w1");
    conn.execute_batch(
        "INSERT INTO messages (rowid, session_id, role, text) VALUES
             (1, 'w1', 'user', 'M0'),
             (2, 'w1', 'user', 'M1');
         INSERT INTO qa_chunks (id, session_id, content, timestamp, src_rowid_lo, src_rowid_hi)
             VALUES (70, 'w1', 'orphan-range', 0, 100000, 100001);",
    )
    .unwrap();
    drop(conn);

    let out = run_show("w1", false, Some(70), 2, &Some(db_path)).unwrap();

    assert_eq!(
        out.markdown,
        concat!(
            "# slug\n",
            "- session_id: w1\n",
            "- date: 1970-01-01\n",
            "- project: /p\n",
            "- source: claude\n",
            "- chunk: 70\n",
            "- window: 2\n",
            "\n",
        )
    );
    assert_eq!(
        printed_envelope(&out),
        serde_json::json!({
            "data": {
                "chunk": 70,
                "messages": [],
                "project": "/p",
                "session_id": "w1",
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
fn verbose_windowed_show_prints_the_file_and_flags_the_chunk_messages() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    let conn = open_or_create_db(&db_path).unwrap();
    db::seed_session(&conn, "s1");
    conn.execute_batch(
        "INSERT INTO messages (rowid, session_id, role, text) VALUES
             (1, 's1', 'user', 'm0'),
             (2, 's1', 'assistant', 'm1'),
             (3, 's1', 'user', 'm2'),
             (4, 's1', 'assistant', 'm3'),
             (5, 's1', 'user', 'm4');
         INSERT INTO qa_chunks (id, session_id, content, timestamp, src_rowid_lo, src_rowid_hi)
             VALUES (1, 's1', 'm1 m2', 0, 2, 3);",
    )
    .unwrap();
    drop(conn);

    let out = run_show("s1", true, Some(1), 1, &Some(db_path)).unwrap();

    assert_eq!(
        out.markdown,
        concat!(
            "# slug\n",
            "- session_id: s1\n",
            "- date: 1970-01-01\n",
            "- project: /p\n",
            "- source: claude\n",
            "- chunk: 1\n",
            "- window: 1\n",
            "- file: /f\n",
            "\n",
            "## [user]\n",
            "m0\n",
            "\n",
            "## [assistant] (chunk)\n",
            "m1\n",
            "\n",
            "## [user] (chunk)\n",
            "m2\n",
            "\n",
            "## [assistant]\n",
            "m3\n",
            "\n",
        )
    );
    assert_eq!(
        printed_envelope(&out),
        serde_json::json!({
            "data": {
                "chunk": 1,
                "messages": [
                    { "in_chunk": false, "role": "user", "rowid": 1, "text": "m0" },
                    { "in_chunk": true, "role": "assistant", "rowid": 2, "text": "m1" },
                    { "in_chunk": true, "role": "user", "rowid": 3, "text": "m2" },
                    { "in_chunk": false, "role": "assistant", "rowid": 4, "text": "m3" },
                ],
                "project": "/p",
                "session_id": "s1",
                "slug": "slug",
                "source": "claude",
                "timestamp": 0,
                "window": 1,
            },
            "degraded": false,
            "notes": [],
        })
    );
}

#[test]
fn show_on_a_database_file_without_tables_is_a_usage_error_asking_to_run_index() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = bare_empty_db(db_dir.path());

    let err = run_show("deadbeef", false, None, DEFAULT_WINDOW, &Some(db_path)).unwrap_err();

    assert_eq!(
        reported(&err),
        (
            ExitCode::from(64),
            r#"{"error":{"code":"USAGE_ERROR","message":"Database not found. Run `recall index` first.","next_step":"See `recall --help`; if the index is missing, run `recall index` first.","retryable":false}}"#.to_owned()
        )
    );
}

/// An index whose session `abc-123` carries control characters in its id, file, project, slug,
/// messages and chunks. Chunk 1 covers both messages, and chunk 2 has no message range.
fn index_with_control_characters() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::TempDir::new().unwrap();
    let db_path = dir.path().join("recall.db");
    open_or_create_db(&db_path)
        .unwrap()
        .execute_batch(
            "INSERT INTO sessions (session_id, source, file_path, project, slug, timestamp, mtime)
                 VALUES ('abc' || char(7) || '-123', 'claude', '/logs/' || char(27) || '[2Jf.jsonl',
                         '/work/' || char(27) || ']0;PROJ' || char(7) || 'app',
                         'fix-' || char(27) || '[1mlogin' || char(27) || '[0m', 1709251200000, 0.0);
             INSERT INTO messages (rowid, session_id, role, text) VALUES
                 (1, 'abc' || char(7) || '-123', 'user', 'run ' || char(27) || '[31mmake' || char(27) || '[0m'),
                 (2, 'abc' || char(7) || '-123', 'assistant', 'done' || char(7));
             INSERT INTO qa_chunks (id, session_id, content, timestamp, src_rowid_lo, src_rowid_hi) VALUES
                 (1, 'abc' || char(7) || '-123', 'run make done', 1709251200000, 1, 2),
                 (2, 'abc' || char(7) || '-123', 'legacy ' || char(27) || '[31mbody' || char(27) || '[0m', 1709251200000, NULL, NULL);",
        )
        .unwrap();
    (dir, db_path)
}

#[test]
fn show_strips_control_characters_from_the_session_fields_and_messages_it_prints() {
    let (_dir, db_path) = index_with_control_characters();

    let out = run_show("abc", true, None, DEFAULT_WINDOW, &Some(db_path)).unwrap();

    assert_eq!(
        out.markdown,
        concat!(
            "# fix-login\n",
            "- session_id: abc-123\n",
            "- date: 2024-03-01\n",
            "- project: /work/app\n",
            "- source: claude\n",
            "- file: /logs/f.jsonl\n",
            "\n",
            "## [user]\n",
            "run make\n",
            "\n",
            "## [assistant]\n",
            "done\n",
            "\n",
        )
    );
    assert_eq!(
        printed_envelope(&out),
        serde_json::json!({
            "data": {
                "messages": [
                    { "role": "user", "text": "run make" },
                    { "role": "assistant", "text": "done" },
                ],
                "project": "/work/app",
                "session_id": "abc-123",
                "slug": "fix-login",
                "source": "claude",
                "timestamp": 1709251200000_i64,
            },
            "degraded": false,
            "notes": [],
        })
    );
}

#[test]
fn windowed_show_strips_control_characters_from_the_session_fields_and_messages_it_prints() {
    let (_dir, db_path) = index_with_control_characters();

    let out = run_show("abc", false, Some(1), 3, &Some(db_path)).unwrap();

    assert_eq!(
        out.markdown,
        concat!(
            "# fix-login\n",
            "- session_id: abc-123\n",
            "- date: 2024-03-01\n",
            "- project: /work/app\n",
            "- source: claude\n",
            "- chunk: 1\n",
            "- window: 3\n",
            "\n",
            "## [user] (chunk)\n",
            "run make\n",
            "\n",
            "## [assistant] (chunk)\n",
            "done\n",
            "\n",
        )
    );
    assert_eq!(
        printed_envelope(&out),
        serde_json::json!({
            "data": {
                "chunk": 1,
                "messages": [
                    { "in_chunk": true, "role": "user", "rowid": 1, "text": "run make" },
                    { "in_chunk": true, "role": "assistant", "rowid": 2, "text": "done" },
                ],
                "project": "/work/app",
                "session_id": "abc-123",
                "slug": "fix-login",
                "source": "claude",
                "timestamp": 1709251200000_i64,
                "window": 3,
            },
            "degraded": false,
            "notes": [],
        })
    );
}

#[test]
fn show_of_a_chunk_without_a_message_range_strips_control_characters_from_the_session_fields_and_body()
 {
    let (_dir, db_path) = index_with_control_characters();

    let out = run_show("abc", false, Some(2), 3, &Some(db_path)).unwrap();

    assert_eq!(
        out.markdown,
        concat!(
            "# fix-login\n",
            "- session_id: abc-123\n",
            "- date: 2024-03-01\n",
            "- project: /work/app\n",
            "- source: claude\n",
            "- chunk: 2\n",
            "\n",
            "legacy body\n",
            "\n",
            "This chunk predates message-range tracking; run `recall show abc-123` for the full conversation.\n",
        )
    );
    assert_eq!(
        printed_envelope(&out),
        serde_json::json!({
            "data": {
                "chunk": 2,
                "content": "legacy body",
                "next_step": "This chunk predates message-range tracking; run `recall show abc-123` for the full conversation.",
                "project": "/work/app",
                "session_id": "abc-123",
                "slug": "fix-login",
                "source": "claude",
                "timestamp": 1709251200000_i64,
                "windowed": false,
            },
            "degraded": false,
            "notes": [],
        })
    );
}

#[test]
fn show_of_a_prefix_shared_by_two_sessions_strips_control_characters_from_the_listed_candidates() {
    let (_dir, db_path) = index_with_control_characters();
    open_or_create_db(&db_path)
        .unwrap()
        .execute(
            "INSERT INTO sessions (session_id, source, file_path, project, slug, timestamp, mtime)
                 VALUES ('abc' || char(7) || '-456', 'claude', '/f', '/p',
                         char(27) || ']0;TITLE' || char(7) || 'other', 0, 0.0)",
            [],
        )
        .unwrap();

    let err = run_show("abc", false, None, DEFAULT_WINDOW, &Some(db_path)).unwrap_err();

    assert_eq!(
        reported(&err),
        (
            ExitCode::from(64),
            r#"{"error":{"code":"USAGE_ERROR","message":"Multiple sessions match 'abc':\n  abc-123  fix-login\n  abc-456  other\nNarrow the session ID prefix to match exactly one session","next_step":"See `recall --help`; if the index is missing, run `recall index` first.","retryable":false}}"#.to_owned()
        )
    );
}

#[test]
fn windowed_show_of_another_sessions_chunk_strips_control_characters_from_the_owner_id() {
    let (_dir, db_path) = index_with_control_characters();
    db::seed_session(&open_or_create_db(&db_path).unwrap(), "w1");

    let err = run_show("w1", false, Some(1), 2, &Some(db_path)).unwrap_err();

    assert_eq!(
        reported(&err),
        (
            ExitCode::from(64),
            r#"{"error":{"code":"USAGE_ERROR","message":"chunk 1 belongs to session 'abc-123', not 'w1'","next_step":"See `recall --help`; if the index is missing, run `recall index` first.","retryable":false}}"#.to_owned()
        )
    );
}
