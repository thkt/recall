use std::sync::Mutex;
use std::time::{Duration, SystemTime};

use super::*;
use crate::db::{seed_session, setup_test_db};
use crate::embedder::{EMBED_BATCH_SIZE, MockEmbedder, embed_recent_chunks};
use tempfile::TempDir;

#[test]
fn test_index_from_dirs_indexes_and_skips() {
    let (_dir, mut conn) = setup_test_db();
    let tmp = TempDir::new().unwrap();

    let claude_dir = tmp.path().join("claude_projects");
    fs::create_dir_all(&claude_dir).unwrap();
    fs::write(
            claude_dir.join("session1.jsonl"),
            r#"{"type":"user","cwd":"/proj","message":{"role":"user","content":"hello world"},"timestamp":"2026-03-01T00:00:00Z"}"#,
        ).unwrap();

    let codex_dir = tmp.path().join("codex_sessions");

    let stats = index_from_dirs(
        &mut conn,
        &IndexOptions {
            force: false,
            claude_dir: &claude_dir,
            codex_dir: &codex_dir,
        },
    )
    .unwrap();
    assert_eq!(stats.indexed, 1);
    assert_eq!(stats.total_sessions, 1);

    let stats2 = index_from_dirs(
        &mut conn,
        &IndexOptions {
            force: false,
            claude_dir: &claude_dir,
            codex_dir: &codex_dir,
        },
    )
    .unwrap();
    assert_eq!(stats2.indexed, 0);
    assert_eq!(stats2.total_sessions, 1);
}

// TC-003 (#130): the production steady state is a hook re-indexing an
// already-indexed tree. Beyond `indexed == 0` (parse skip, pinned above),
// the DB content must be byte-stable: no duplicate chunks, no FTS growth,
// no re-created sessions.
#[test]
fn test_index_from_dirs_steady_state_keeps_db_stable() {
    let (_dir, mut conn) = setup_test_db();
    let tmp = TempDir::new().unwrap();
    let claude_dir = tmp.path().join("claude_projects");
    fs::create_dir_all(&claude_dir).unwrap();
    fs::write(
            claude_dir.join("s.jsonl"),
            concat!(
                r#"{"type":"user","cwd":"/proj","message":{"role":"user","content":"steady question"},"timestamp":"2026-03-01T00:00:00Z"}"#,
                "\n",
                r#"{"type":"assistant","cwd":"/proj","message":{"role":"assistant","content":"steady answer"},"timestamp":"2026-03-01T00:00:01Z"}"#,
            ),
        )
        .unwrap();
    let codex_dir = tmp.path().join("codex_sessions");
    let opts = IndexOptions {
        force: false,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
    };

    index_from_dirs(&mut conn, &opts).unwrap();
    index_chunks(&mut conn, None).unwrap();
    let counts = |conn: &Connection| -> (i64, i64, i64, i64) {
        let m = conn
            .query_row("SELECT COUNT(*) FROM messages", [], |r| r.get(0))
            .unwrap();
        let c = conn
            .query_row("SELECT COUNT(*) FROM qa_chunks", [], |r| r.get(0))
            .unwrap();
        let s = conn
            .query_row("SELECT COUNT(*) FROM sessions", [], |r| r.get(0))
            .unwrap();
        let v = conn
            .query_row("SELECT COUNT(*) FROM vec_chunks", [], |r| r.get(0))
            .unwrap();
        (m, c, s, v)
    };
    let before = counts(&conn);

    let stats = index_from_dirs(&mut conn, &opts).unwrap();
    index_chunks(&mut conn, None).unwrap();

    assert_eq!(stats.indexed, 0, "an unchanged tree re-parses nothing");
    assert_eq!(
        counts(&conn),
        before,
        "re-index of an unchanged tree must not grow or rewrite the DB"
    );
}

// TC-004 (#130): mtime-forward re-index is the self-heal core — a session
// file appended after indexing (the tail of a live session) must be
// re-parsed, surface the new message, and replace its chunks without
// duplicating the session.
#[test]
fn test_index_from_dirs_reindexes_appended_file() {
    let (_dir, mut conn) = setup_test_db();
    let tmp = TempDir::new().unwrap();
    let claude_dir = tmp.path().join("claude_projects");
    fs::create_dir_all(&claude_dir).unwrap();
    let file = claude_dir.join("s.jsonl");
    fs::write(
            &file,
            concat!(
                r#"{"type":"user","cwd":"/proj","message":{"role":"user","content":"first question"},"timestamp":"2026-03-01T00:00:00Z"}"#,
                "\n",
            ),
        )
        .unwrap();
    let codex_dir = tmp.path().join("codex_sessions");
    let opts = IndexOptions {
        force: false,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
    };
    index_from_dirs(&mut conn, &opts).unwrap();

    // Append a turn and push mtime clearly past the 0.001s freshness epsilon.
    let mut all = fs::read_to_string(&file).unwrap();
    all.push_str(
            r#"{"type":"assistant","cwd":"/proj","message":{"role":"assistant","content":"appended answer"},"timestamp":"2026-03-01T00:00:01Z"}"#,
        );
    fs::write(&file, all).unwrap();
    let f = fs::OpenOptions::new().append(true).open(&file).unwrap();
    f.set_modified(SystemTime::now() + Duration::from_secs(10))
        .unwrap();
    drop(f);

    let stats = index_from_dirs(&mut conn, &opts).unwrap();

    assert_eq!(stats.indexed, 1, "the appended file must be re-parsed");
    let sessions: i64 = conn
        .query_row("SELECT COUNT(*) FROM sessions", [], |r| r.get(0))
        .unwrap();
    assert_eq!(
        sessions, 1,
        "re-index upserts the session, not duplicates it"
    );
    let appended: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM messages WHERE messages MATCH 'appended answer'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(appended, 1, "the appended message must be searchable");
    let original: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM messages WHERE messages MATCH 'first question'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        original, 1,
        "re-parsing the appended file must keep the original message searchable"
    );
}

fn index_one_claude_session(content: &str) -> Option<String> {
    let (_dir, mut conn) = setup_test_db();
    let tmp = TempDir::new().unwrap();
    let claude_dir = tmp.path().join("claude_projects");
    fs::create_dir_all(&claude_dir).unwrap();
    let line = format!(
        r#"{{"type":"user","cwd":"/proj","message":{{"role":"user","content":{content:?}}},"timestamp":"2026-03-01T00:00:00Z"}}"#
    );
    fs::write(claude_dir.join("s.jsonl"), line).unwrap();
    let codex_dir = tmp.path().join("codex_sessions");
    index_from_dirs(
        &mut conn,
        &IndexOptions {
            force: false,
            claude_dir: &claude_dir,
            codex_dir: &codex_dir,
        },
    )
    .unwrap();
    conn.query_row("SELECT session_type FROM sessions", [], |r| r.get(0))
        .unwrap()
}

// T-004 (#24/FR-003): ingest tags session_type from the first user turn — an
// automated marker yields 'automated', a human turn yields 'interactive'.
#[test]
fn test_ingest_tags_session_type_from_first_user_turn() {
    assert_eq!(
        index_one_claude_session("<command-message>clear</command-message>").as_deref(),
        Some("automated"),
        "a slash-command first turn must be tagged automated"
    );
    assert_eq!(
        index_one_claude_session("how do I implement authentication").as_deref(),
        Some("interactive"),
        "a human first turn must be tagged interactive"
    );
}

#[test]
fn test_incremental_scan_picks_up_new_file_in_existing_codex_day_dir() {
    let (_dir, mut conn) = setup_test_db();
    let tmp = TempDir::new().unwrap();
    let claude_dir = tmp.path().join("claude_projects");
    fs::create_dir_all(&claude_dir).unwrap();
    let codex_dir = tmp.path().join("codex_sessions");
    let day_dir = codex_dir.join("2026/04/27");
    fs::create_dir_all(&day_dir).unwrap();

    let s1 = r#"{"timestamp":"2026-04-27T00:00:00Z","type":"session_meta","payload":{"id":"codex-s1","cwd":"/proj"}}
{"timestamp":"2026-04-27T00:00:01Z","type":"response_item","payload":{"role":"user","content":[{"type":"input_text","text":"first codex session"}]}}"#;
    fs::write(day_dir.join("s1.jsonl"), s1).unwrap();

    let stats1 = index_from_dirs(
        &mut conn,
        &IndexOptions {
            force: false,
            claude_dir: &claude_dir,
            codex_dir: &codex_dir,
        },
    )
    .unwrap();
    assert_eq!(stats1.indexed, 1);

    // New session added into the SAME existing day dir. The parent `2026/` mtime
    // does not change, so the old dirs_changed_since optimization skipped the scan
    // and permanently missed the new session (#52 / #70).
    let s2 = r#"{"timestamp":"2026-04-27T00:00:00Z","type":"session_meta","payload":{"id":"codex-s2","cwd":"/proj"}}
{"timestamp":"2026-04-27T00:00:01Z","type":"response_item","payload":{"role":"user","content":[{"type":"input_text","text":"second codex session"}]}}"#;
    fs::write(day_dir.join("s2.jsonl"), s2).unwrap();

    let stats2 = index_from_dirs(
        &mut conn,
        &IndexOptions {
            force: false,
            claude_dir: &claude_dir,
            codex_dir: &codex_dir,
        },
    )
    .unwrap();
    assert_eq!(
        stats2.indexed, 1,
        "a new session in an existing deep dir must be indexed (parent mtime unchanged)"
    );
    assert_eq!(stats2.total_sessions, 2);
}

#[test]
fn test_index_from_dirs_force_reindexes() {
    let (_dir, mut conn) = setup_test_db();
    let tmp = TempDir::new().unwrap();

    let claude_dir = tmp.path().join("claude_projects");
    fs::create_dir_all(&claude_dir).unwrap();
    fs::write(
            claude_dir.join("session1.jsonl"),
            r#"{"type":"user","message":{"role":"user","content":"hello"},"timestamp":"2026-03-01T00:00:00Z"}"#,
        ).unwrap();

    let codex_dir = tmp.path().join("codex_sessions");

    let stats = index_from_dirs(
        &mut conn,
        &IndexOptions {
            force: false,
            claude_dir: &claude_dir,
            codex_dir: &codex_dir,
        },
    )
    .unwrap();
    assert_eq!(stats.indexed, 1);

    // Populate qa_chunks + vec_chunks before force reindex
    index_chunks(&mut conn, None).unwrap();
    let embedder = MockEmbedder::new();
    embed_recent_chunks(&mut conn, &embedder, 100, None).unwrap();
    let qa_before: i64 = conn
        .query_row("SELECT COUNT(*) FROM qa_chunks", [], |r| r.get(0))
        .unwrap();
    let vec_before: i64 = conn
        .query_row("SELECT COUNT(*) FROM vec_chunks", [], |r| r.get(0))
        .unwrap();
    assert!(qa_before > 0, "should have qa_chunks before force reindex");
    assert!(
        vec_before > 0,
        "should have vec_chunks before force reindex"
    );

    let stats2 = index_from_dirs(
        &mut conn,
        &IndexOptions {
            force: true,
            claude_dir: &claude_dir,
            codex_dir: &codex_dir,
        },
    )
    .unwrap();
    assert_eq!(stats2.indexed, 1);

    let qa_after: i64 = conn
        .query_row("SELECT COUNT(*) FROM qa_chunks", [], |r| r.get(0))
        .unwrap();
    let vec_after: i64 = conn
        .query_row("SELECT COUNT(*) FROM vec_chunks", [], |r| r.get(0))
        .unwrap();
    assert_eq!(qa_after, 0, "force reindex should clear qa_chunks");
    assert_eq!(vec_after, 0, "force reindex should clear vec_chunks");
}

#[test]
fn test_orphan_cleanup_removes_deleted_files() {
    let (_dir, mut conn) = setup_test_db();
    let tmp = TempDir::new().unwrap();

    let claude_dir = tmp.path().join("claude_projects");
    fs::create_dir_all(&claude_dir).unwrap();
    let f1 = claude_dir.join("session1.jsonl");
    let f2 = claude_dir.join("session2.jsonl");
    fs::write(&f1, r#"{"type":"user","message":{"role":"user","content":"one"},"timestamp":"2026-03-01T00:00:00Z"}"#).unwrap();
    fs::write(&f2, r#"{"type":"user","message":{"role":"user","content":"two"},"timestamp":"2026-03-01T00:00:00Z"}"#).unwrap();

    let codex_dir = tmp.path().join("codex_sessions");

    let stats = index_from_dirs(
        &mut conn,
        &IndexOptions {
            force: false,
            claude_dir: &claude_dir,
            codex_dir: &codex_dir,
        },
    )
    .unwrap();
    assert_eq!(stats.indexed, 2);
    assert_eq!(stats.total_sessions, 2);

    // Create chunks + embeddings for session2 before deleting the file
    index_chunks(&mut conn, None).unwrap();
    let embedder = MockEmbedder::new();
    embed_recent_chunks(&mut conn, &embedder, 100, None).unwrap();
    let chunks_before: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM qa_chunks WHERE session_id = 'session2'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert!(chunks_before > 0, "session2 should have chunks");
    let vecs_before: i64 = conn
        .query_row("SELECT COUNT(*) FROM vec_chunks", [], |r| r.get(0))
        .unwrap();
    assert!(
        vecs_before > 0,
        "should have vec_chunks before orphan cleanup"
    );

    fs::remove_file(&f2).unwrap();
    // force: false exercises cleanup_orphans (force: true bulk-deletes everything,
    // which would make the cascade assertion a false pass)
    let stats2 = index_from_dirs(
        &mut conn,
        &IndexOptions {
            force: false,
            claude_dir: &claude_dir,
            codex_dir: &codex_dir,
        },
    )
    .unwrap();
    assert_eq!(stats2.total_sessions, 1);

    let chunks_after: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM qa_chunks WHERE session_id = 'session2'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        chunks_after, 0,
        "orphan cleanup should cascade to qa_chunks"
    );

    let vecs_for_session1: i64 = conn
        .query_row("SELECT COUNT(*) FROM vec_chunks", [], |r| r.get(0))
        .unwrap();
    // session1 still exists, so its vec_chunks should remain; session2's should be gone
    assert!(
        vecs_for_session1 < vecs_before,
        "orphan cleanup should remove session2 vec_chunks: before={vecs_before}, after={vecs_for_session1}"
    );
}

// Seeds one claude session and one codex session into a populated DB and
// returns (tmp, claude_dir, codex_dir). Used by the #165 regression tests.
fn seed_claude_and_codex(conn: &mut Connection) -> (TempDir, PathBuf, PathBuf) {
    let tmp = TempDir::new().unwrap();
    let claude_dir = tmp.path().join("claude_projects");
    fs::create_dir_all(&claude_dir).unwrap();
    fs::write(
            claude_dir.join("c.jsonl"),
            r#"{"type":"user","cwd":"/proj","message":{"role":"user","content":"claude turn"},"timestamp":"2026-03-01T00:00:00Z"}"#,
        )
        .unwrap();
    let codex_dir = tmp.path().join("codex_sessions");
    let day_dir = codex_dir.join("2026/04/27");
    fs::create_dir_all(&day_dir).unwrap();
    fs::write(
            day_dir.join("s1.jsonl"),
            "{\"timestamp\":\"2026-04-27T00:00:00Z\",\"type\":\"session_meta\",\"payload\":{\"id\":\"codex-s1\",\"cwd\":\"/proj\"}}\n{\"timestamp\":\"2026-04-27T00:00:01Z\",\"type\":\"response_item\",\"payload\":{\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"codex turn\"}]}}",
        )
        .unwrap();
    let stats = index_from_dirs(
        conn,
        &IndexOptions {
            force: false,
            claude_dir: &claude_dir,
            codex_dir: &codex_dir,
        },
    )
    .unwrap();
    assert_eq!(stats.total_sessions, 2, "seed should index both sessions");
    (tmp, claude_dir, codex_dir)
}

// #165: a refresh where every source root is missing must not be read as
// proof that all indexed files were deleted. The existing rows survive.
#[test]
fn test_missing_source_roots_preserve_existing_index() {
    let (_dir, mut conn) = setup_test_db();
    let (tmp, _claude_dir, _codex_dir) = seed_claude_and_codex(&mut conn);

    let gone_claude = tmp.path().join("missing_claude");
    let gone_codex = tmp.path().join("missing_codex");
    let stats = index_from_dirs(
        &mut conn,
        &IndexOptions {
            force: false,
            claude_dir: &gone_claude,
            codex_dir: &gone_codex,
        },
    )
    .unwrap();

    assert_eq!(
        stats.total_sessions, 2,
        "missing source roots must not delete existing sessions"
    );
    assert_eq!(
        stats.skipped_roots.len(),
        2,
        "both unscanned roots with preserved sessions must be reported"
    );
}

// #165: with one root scanned and the other missing, cleanup is selective —
// a file deleted under the scanned root is removed, while the missing root's
// session is preserved (its absence is unproven).
#[test]
fn test_partial_missing_root_preserves_unscanned_source_only() {
    let (_dir, mut conn) = setup_test_db();
    let (tmp, claude_dir, _codex_dir) = seed_claude_and_codex(&mut conn);

    // Legitimately delete the claude file; point codex at a missing root.
    fs::remove_file(claude_dir.join("c.jsonl")).unwrap();
    let gone_codex = tmp.path().join("missing_codex");
    let stats = index_from_dirs(
        &mut conn,
        &IndexOptions {
            force: false,
            claude_dir: &claude_dir,
            codex_dir: &gone_codex,
        },
    )
    .unwrap();

    assert_eq!(
        stats.total_sessions, 1,
        "the deleted claude file is cleaned, the unscanned codex session is kept"
    );
    let surviving_source: String = conn
        .query_row("SELECT source FROM sessions", [], |r| r.get(0))
        .unwrap();
    assert_eq!(
        surviving_source, "codex",
        "the surviving session must be the one whose root was not scanned"
    );
}

// #165 high-severity variant: a root that exists but cannot be read
// (e.g. permission denied) yields zero sources without proving deletion.
// `is_dir()` is true, so the fix must gate cleanup on read success.
#[cfg(unix)]
#[test]
fn test_unreadable_source_root_preserves_index() {
    use std::os::unix::fs::PermissionsExt;

    let (_dir, mut conn) = setup_test_db();
    let (_tmp, claude_dir, codex_dir) = seed_claude_and_codex(&mut conn);

    // Strip all permissions so read_dir fails while is_dir() still succeeds.
    let locked = fs::metadata(&claude_dir).unwrap().permissions();
    let mut zero = locked.clone();
    zero.set_mode(0o000);
    fs::set_permissions(&claude_dir, zero).unwrap();

    let stats = index_from_dirs(
        &mut conn,
        &IndexOptions {
            force: false,
            claude_dir: &claude_dir,
            codex_dir: &codex_dir,
        },
    );

    // Restore permissions before asserting so TempDir cleanup always works.
    fs::set_permissions(&claude_dir, locked).unwrap();
    let stats = stats.unwrap();

    assert_eq!(
        stats.total_sessions, 2,
        "an unreadable root must preserve its sessions, not wipe them"
    );
}

// #165 (CHX-1): a root whose top level reads fine but holds an unreadable
// nested subdirectory is only partially enumerated. Files under the locked
// subtree appear absent, so the source must be preserved, not deleted.
#[cfg(unix)]
#[test]
fn test_unreadable_child_subdir_preserves_source() {
    use std::os::unix::fs::PermissionsExt;

    let (_dir, mut conn) = setup_test_db();
    let (_tmp, claude_dir, codex_dir) = seed_claude_and_codex(&mut conn);

    // Lock the codex year subdir: read_dir(codex_dir) lists it, but
    // descending into it fails, so codex's tree is not fully enumerated.
    let locked_subdir = codex_dir.join("2026");
    let orig = fs::metadata(&locked_subdir).unwrap().permissions();
    let mut zero = orig.clone();
    zero.set_mode(0o000);
    fs::set_permissions(&locked_subdir, zero).unwrap();

    let stats = index_from_dirs(
        &mut conn,
        &IndexOptions {
            force: false,
            claude_dir: &claude_dir,
            codex_dir: &codex_dir,
        },
    );

    fs::set_permissions(&locked_subdir, orig).unwrap();
    let stats = stats.unwrap();

    assert_eq!(
        stats.total_sessions, 2,
        "an unreadable nested subdir must preserve the whole source's rows"
    );
}

// #165 negative control: with both roots present and readable, deleting a
// file under a scanned root still removes its session AND cascades to its
// messages and qa_chunks — the fix narrows cleanup, it does not disable it.
#[test]
fn test_deleted_file_under_scanned_root_cascades_cleanup() {
    let (_dir, mut conn) = setup_test_db();
    let (_tmp, claude_dir, codex_dir) = seed_claude_and_codex(&mut conn);

    let claude_sid: String = conn
        .query_row(
            "SELECT session_id FROM sessions WHERE source = 'claude'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    let msgs_before: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM messages WHERE session_id = ?",
            [&claude_sid],
            |r| r.get(0),
        )
        .unwrap();
    assert!(
        msgs_before > 0,
        "seed should leave messages for the session"
    );

    // Delete the claude file; both roots stay present and readable.
    fs::remove_file(claude_dir.join("c.jsonl")).unwrap();
    let stats = index_from_dirs(
        &mut conn,
        &IndexOptions {
            force: false,
            claude_dir: &claude_dir,
            codex_dir: &codex_dir,
        },
    )
    .unwrap();

    assert_eq!(
        stats.total_sessions, 1,
        "the deleted file's session must be cleaned when its root was scanned"
    );
    let surviving_source: String = conn
        .query_row("SELECT source FROM sessions", [], |r| r.get(0))
        .unwrap();
    assert_eq!(surviving_source, "codex", "only the codex session survives");

    let msgs_after: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM messages WHERE session_id = ?",
            [&claude_sid],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        msgs_after, 0,
        "cleanup must cascade to the session's messages"
    );
    let chunks_after: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM qa_chunks WHERE session_id = ?",
            [&claude_sid],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        chunks_after, 0,
        "cleanup must cascade to the session's qa_chunks"
    );
    assert!(
        stats.skipped_roots.is_empty(),
        "no root was skipped, so nothing should be reported as preserved"
    );
}

// #165 follow-up: a skipped root that preserved sessions is reported so the
// silent preservation stays visible to the operator.
#[test]
fn test_skipped_root_reports_preserved_count() {
    let (_dir, mut conn) = setup_test_db();
    let (tmp, claude_dir, _codex_dir) = seed_claude_and_codex(&mut conn);

    let gone_codex = tmp.path().join("missing_codex");
    let stats = index_from_dirs(
        &mut conn,
        &IndexOptions {
            force: false,
            claude_dir: &claude_dir,
            codex_dir: &gone_codex,
        },
    )
    .unwrap();

    assert_eq!(
        stats.skipped_roots.len(),
        1,
        "only the missing codex root should be reported"
    );
    let reported = &stats.skipped_roots[0];
    assert_eq!(reported.source, Source::Codex);
    assert_eq!(
        reported.preserved_sessions, 1,
        "the one preserved codex session must be counted"
    );
}

// #165 CHX-1 regression: when an unscanned source holds BOTH a re-indexed
// present file and a preserved absent file, only the absent row is at risk.
// Counting every row of the source would over-report the preserved magnitude.
#[cfg(unix)]
#[test]
fn test_skipped_root_counts_only_file_absent_rows() {
    use std::os::unix::fs::PermissionsExt;

    let (_dir, mut conn) = setup_test_db();
    let (_tmp, claude_dir, codex_dir) = seed_claude_and_codex(&mut conn);

    // Add a second codex session at the root level (outside 2026), so it
    // stays readable — hence re-indexed and present — when 2026 is locked.
    fs::write(
            codex_dir.join("top.jsonl"),
            "{\"timestamp\":\"2026-05-01T00:00:00Z\",\"type\":\"session_meta\",\"payload\":{\"id\":\"codex-top\",\"cwd\":\"/proj\"}}\n{\"timestamp\":\"2026-05-01T00:00:01Z\",\"type\":\"response_item\",\"payload\":{\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"codex top turn\"}]}}",
        )
        .unwrap();
    index_from_dirs(
        &mut conn,
        &IndexOptions {
            force: false,
            claude_dir: &claude_dir,
            codex_dir: &codex_dir,
        },
    )
    .unwrap();

    // Lock the year subdir: top.jsonl is re-read (present in sources),
    // s1.jsonl is not (preserved). The codex tree is not fully enumerated,
    // so codex is unscanned and reported.
    let locked_subdir = codex_dir.join("2026");
    let orig = fs::metadata(&locked_subdir).unwrap().permissions();
    let mut zero = orig.clone();
    zero.set_mode(0o000);
    fs::set_permissions(&locked_subdir, zero).unwrap();

    let stats = index_from_dirs(
        &mut conn,
        &IndexOptions {
            force: false,
            claude_dir: &claude_dir,
            codex_dir: &codex_dir,
        },
    );

    fs::set_permissions(&locked_subdir, orig).unwrap();
    let stats = stats.unwrap();

    let codex = stats
        .skipped_roots
        .iter()
        .find(|s| s.source == Source::Codex)
        .expect("codex tree was not fully scanned, so it must be reported");
    assert_eq!(
        codex.preserved_sessions, 1,
        "only the file-absent s1 row is at risk; the re-indexed top row must not inflate the count"
    );
}

#[test]
fn test_session_id_collision_cleans_old_messages() {
    let (_dir, mut conn) = setup_test_db();
    let tmp = TempDir::new().unwrap();

    let dir_a = tmp.path().join("claude_a");
    let dir_b = tmp.path().join("claude_b");
    fs::create_dir_all(&dir_a).unwrap();
    fs::create_dir_all(&dir_b).unwrap();
    fs::write(
            dir_a.join("collision.jsonl"),
            r#"{"type":"user","message":{"role":"user","content":"from dir a"},"timestamp":"2026-03-01T00:00:00Z"}"#,
        ).unwrap();
    fs::write(
            dir_b.join("collision.jsonl"),
            r#"{"type":"user","message":{"role":"user","content":"from dir b"},"timestamp":"2026-03-02T00:00:00Z"}"#,
        ).unwrap();

    let empty_dir = tmp.path().join("empty");

    let stats = index_from_dirs(
        &mut conn,
        &IndexOptions {
            force: false,
            claude_dir: &dir_a,
            codex_dir: &empty_dir,
        },
    )
    .unwrap();
    assert_eq!(stats.indexed, 1);

    let stats2 = index_from_dirs(
        &mut conn,
        &IndexOptions {
            force: false,
            claude_dir: &dir_b,
            codex_dir: &empty_dir,
        },
    )
    .unwrap();
    assert_eq!(stats2.indexed, 1);

    let msg_count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM messages WHERE session_id = 'collision'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(msg_count, 1);
}

#[test]
fn test_collect_depth_limit() {
    let tmp = TempDir::new().unwrap();

    let mut deep = tmp.path().to_path_buf();
    for i in 0..MAX_DIR_DEPTH {
        deep = deep.join(format!("d{i}"));
    }
    fs::create_dir_all(&deep).unwrap();
    fs::write(deep.join("deep.jsonl"), "{}").unwrap();

    fs::write(tmp.path().join("shallow.jsonl"), "{}").unwrap();

    let mut files = Vec::new();
    collect_jsonl_files(tmp.path(), &mut files, Source::Claude);

    assert_eq!(files.len(), 1);
    assert!(files[0].0.ends_with("shallow.jsonl"));
}

#[cfg(unix)]
#[test]
fn test_collect_skips_symlinks() {
    use std::os::unix::fs::symlink;

    let tmp = TempDir::new().unwrap();

    let real_file = tmp.path().join("real.jsonl");
    fs::write(&real_file, "{}").unwrap();

    let link = tmp.path().join("link.jsonl");
    symlink(&real_file, &link).unwrap();

    let mut files = Vec::new();
    collect_jsonl_files(tmp.path(), &mut files, Source::Claude);

    assert_eq!(files.len(), 1);
    assert!(files[0].0.ends_with("real.jsonl"));
}

#[test]
fn test_011_index_chunks_incremental() {
    let (_dir, mut conn) = setup_test_db();
    let tmp = TempDir::new().unwrap();

    let claude_dir = tmp.path().join("claude_projects");
    fs::create_dir_all(&claude_dir).unwrap();
    fs::write(
            claude_dir.join("s1.jsonl"),
            concat!(
                r#"{"type":"user","cwd":"/proj","message":{"role":"user","content":"hello"},"timestamp":"2026-03-01T00:00:00Z"}"#,
                "\n",
                r#"{"type":"assistant","message":{"role":"assistant","content":"hi there"}}"#,
            ),
        )
        .unwrap();
    let codex_dir = tmp.path().join("codex_sessions");

    index_from_dirs(
        &mut conn,
        &IndexOptions {
            force: false,
            claude_dir: &claude_dir,
            codex_dir: &codex_dir,
        },
    )
    .unwrap();

    let stats1 = index_chunks(&mut conn, None).unwrap();
    assert_eq!(stats1.chunks_created, 1);

    let stats2 = index_chunks(&mut conn, None).unwrap();
    assert_eq!(stats2.chunks_created, 0);
}

#[test]
fn test_index_chunks_progress_callback_counts_sessions() {
    let (_dir, mut conn) = setup_test_db();
    seed_session(&conn, "s1");
    seed_session(&conn, "s2");
    for (role, text) in [("user", "hello"), ("assistant", "hi there")] {
        conn.execute(
            "INSERT INTO messages (session_id, role, text) VALUES ('s1', ?1, ?2)",
            rusqlite::params![role, text],
        )
        .unwrap();
    }

    let calls = Mutex::new(Vec::new());
    let stats = index_chunks(
        &mut conn,
        Some(&|done, total| calls.lock().unwrap().push((done, total))),
    )
    .unwrap();

    // Progress advances per session processed, not per chunk created:
    // s2 has no messages (0 chunks) yet still counts toward done/total.
    assert_eq!(calls.into_inner().unwrap(), vec![(1, 2), (2, 2)]);
    assert_eq!(stats.chunks_created, 1);
}

#[test]
fn test_index_chunks_progress_callback_silent_when_all_chunked() {
    let (_dir, mut conn) = setup_test_db();
    seed_session(&conn, "s1");
    conn.execute(
        "INSERT INTO messages (session_id, role, text) VALUES ('s1', 'user', 'hello')",
        [],
    )
    .unwrap();
    index_chunks(&mut conn, None).unwrap();

    // Every session is chunked, so the empty early-return must not
    // invoke the callback even when one is supplied.
    let calls = Mutex::new(Vec::new());
    let stats = index_chunks(
        &mut conn,
        Some(&|done, total| calls.lock().unwrap().push((done, total))),
    )
    .unwrap();

    assert_eq!(stats.chunks_created, 0);
    assert_eq!(calls.into_inner().unwrap(), Vec::<(usize, usize)>::new());
}

#[test]
fn test_009_recall_claude_dir_overrides_claude_config_dir() {
    let home = Path::new("/fake/home");
    let result = resolve_claude_dir_with(home, |key| match key {
        "RECALL_CLAUDE_DIR" => Some(OsString::from("/custom/recall/dir")),
        "CLAUDE_CONFIG_DIR" => Some(OsString::from("/custom/config/dir")),
        _ => None,
    });
    assert_eq!(
        result,
        PathBuf::from("/custom/recall/dir"),
        "RECALL_CLAUDE_DIR should take priority over CLAUDE_CONFIG_DIR"
    );
}

#[test]
fn test_010_claude_config_dir_used_as_fallback() {
    let home = Path::new("/fake/home");
    let result = resolve_claude_dir_with(home, |key| match key {
        "CLAUDE_CONFIG_DIR" => Some(OsString::from("/custom/config")),
        _ => None,
    });
    assert_eq!(
        result,
        PathBuf::from("/custom/config/projects"),
        "CLAUDE_CONFIG_DIR/projects should be used when RECALL_CLAUDE_DIR is unset"
    );
}

#[test]
fn test_codex_dir_env_override() {
    let home = Path::new("/fake/home");
    let result = resolve_codex_dir_with(home, |key| match key {
        "RECALL_CODEX_DIR" => Some(OsString::from("/custom/codex/sessions")),
        _ => None,
    });
    assert_eq!(result, PathBuf::from("/custom/codex/sessions"));
}

#[test]
fn test_codex_dir_default() {
    let home = Path::new("/fake/home");
    let result = resolve_codex_dir_with(home, |_| None);
    assert_eq!(result, PathBuf::from("/fake/home/.codex/sessions"));
}

#[test]
fn test_embed_recent_chunks_populates_vec_chunks() {
    let (_dir, mut conn) = setup_test_db();
    let tmp = TempDir::new().unwrap();
    let claude_dir = tmp.path().join("claude_projects");
    fs::create_dir_all(&claude_dir).unwrap();
    fs::write(
            claude_dir.join("s1.jsonl"),
            concat!(
                r#"{"type":"user","cwd":"/proj","message":{"role":"user","content":"hello world"},"timestamp":"2026-03-01T00:00:00Z"}"#,
                "\n",
                r#"{"type":"assistant","message":{"role":"assistant","content":"hi there"}}"#,
            ),
        )
        .unwrap();
    let codex_dir = tmp.path().join("codex_sessions");

    index_from_dirs(
        &mut conn,
        &IndexOptions {
            force: false,
            claude_dir: &claude_dir,
            codex_dir: &codex_dir,
        },
    )
    .unwrap();
    index_chunks(&mut conn, None).unwrap();

    let chunk_count: i64 = conn
        .query_row("SELECT COUNT(*) FROM qa_chunks", [], |r| r.get(0))
        .unwrap();
    assert!(chunk_count > 0, "should have chunks to embed");

    let embedder = MockEmbedder::new();
    let result = embed_recent_chunks(&mut conn, &embedder, 100, None).unwrap();

    assert_eq!(result.embedded, usize::try_from(chunk_count).unwrap());
    assert_eq!(result.failed_count, 0);

    let vec_count: i64 = conn
        .query_row("SELECT COUNT(*) FROM vec_chunks", [], |r| r.get(0))
        .unwrap();
    assert_eq!(vec_count, chunk_count, "all chunks should be embedded");

    let result2 = embed_recent_chunks(&mut conn, &embedder, 100, None).unwrap();
    assert_eq!(result2.embedded, 0);
}

// T-007 (FR-001, FR-002): a mid-run batch failure is non-blocking — the run
// continues past it and counts the rest as failed. With failing_after(128) the
// first batch (128 chunks) succeeds and commits; the MockEmbedder counter is not
// reset, so every later batch fails (call_count stays >= 128). Under continue the
// run does not stop at batch 2: it reports embedded == 128 and failed_count ==
// the remaining chunks, and the committed batch survives.
// (Updated from test_embed_chunks_stops_on_error_preserves_progress, whose
// stopped_at_error.is_some() assertion predates the break->continue change.)
// Perspective: branch (the Err arm continues) + boundary (split at EMBED_BATCH_SIZE).
#[test]
fn test_embed_chunks_continues_past_failed_batch_and_counts_remainder() {
    let (_dir, mut conn) = setup_test_db();

    seed_session(&conn, "s1");
    let message_count = EMBED_BATCH_SIZE + 10;
    for i in 0..message_count {
        conn.execute(
            "INSERT INTO messages (session_id, role, text) VALUES ('s1', 'user', ?1)",
            [format!("question number {i}")],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO messages (session_id, role, text) VALUES ('s1', 'assistant', ?1)",
            [format!("answer number {i}")],
        )
        .unwrap();
    }
    index_chunks(&mut conn, None).unwrap();

    let chunk_count: i64 = conn
        .query_row("SELECT COUNT(*) FROM qa_chunks", [], |r| r.get(0))
        .unwrap();
    assert!(
        chunk_count > EMBED_BATCH_SIZE as i64,
        "need enough chunks for multiple batches, got {chunk_count}"
    );

    let embedder = MockEmbedder::failing_after(EMBED_BATCH_SIZE);
    let result = embed_recent_chunks(
        &mut conn,
        &embedder,
        usize::try_from(chunk_count).unwrap(),
        None,
    )
    .unwrap();

    assert_eq!(
        result.embedded, EMBED_BATCH_SIZE,
        "the first batch embeds and commits before the failures start"
    );
    let remaining = usize::try_from(chunk_count).unwrap() - EMBED_BATCH_SIZE;
    assert_eq!(
        result.failed_count, remaining,
        "every batch after the first fails (counter not reset), so all remaining chunks are failed"
    );

    let vec_count: i64 = conn
        .query_row("SELECT COUNT(*) FROM vec_chunks", [], |r| r.get(0))
        .unwrap();
    assert_eq!(
        vec_count, EMBED_BATCH_SIZE as i64,
        "the committed first batch survives the later failures"
    );
}
