use std::sync::Mutex;
use std::time::{Duration, SystemTime};

use super::*;
use crate::db::{seed_session, setup_test_db};
use crate::embedder::{EMBED_BATCH_SIZE, MockEmbedder, embed_recent_chunks, f32_as_bytes};
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
    // force: false exercises cleanup_orphans on the incremental path; force: true
    // takes the same source-keyed cleanup, so either flag drives this cascade.
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
    assert_eq!(
        reported.reason,
        SkippedReason::MissingRoot,
        "an absent root must be classified as missing, not incomplete enumeration"
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
    assert_eq!(
        codex.reason,
        SkippedReason::IncompleteEnumeration,
        "a present root whose subtree could not be read is incomplete enumeration, not a missing root"
    );
}

// #185 T-185-1: a present root that fails to fully enumerate must surface even
// when nothing was preserved (an empty DB / first index). The partial read
// silently dropped files, which is the under-index this signal exists to catch.
#[cfg(unix)]
#[test]
fn test_incomplete_enumeration_surfaces_with_empty_db() {
    use std::os::unix::fs::PermissionsExt;

    let (_dir, mut conn) = setup_test_db();
    let tmp = TempDir::new().unwrap();
    // claude root: present and empty, so it scans cleanly (not skipped).
    let claude_dir = tmp.path().join("claude_projects");
    fs::create_dir_all(&claude_dir).unwrap();
    // codex root: present, but its only subtree is unreadable, so the tree is
    // not fully enumerated and no codex file is indexed (preserved stays 0).
    let codex_dir = tmp.path().join("codex_sessions");
    let locked_subdir = codex_dir.join("2026");
    fs::create_dir_all(&locked_subdir).unwrap();
    fs::write(
            locked_subdir.join("s1.jsonl"),
            "{\"timestamp\":\"2026-04-27T00:00:00Z\",\"type\":\"session_meta\",\"payload\":{\"id\":\"codex-s1\",\"cwd\":\"/proj\"}}",
        )
        .unwrap();
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
        .expect("an incompletely enumerated root must surface even with nothing preserved");
    assert_eq!(
        codex.preserved_sessions, 0,
        "an empty DB has no rows to preserve"
    );
    assert_eq!(codex.reason, SkippedReason::IncompleteEnumeration);
}

// #185 T-185-2: a missing root with nothing preserved is a tool the user never
// ran, so it must stay silent (the suppressed false side of the MissingRoot
// gate, covering the changed branch for the #49/#121 diff-cover gate).
#[test]
fn test_missing_root_with_no_preserved_rows_is_silent() {
    let (_dir, mut conn) = setup_test_db();
    let tmp = TempDir::new().unwrap();
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

    assert!(
        stats.skipped_roots.is_empty(),
        "missing roots with no preserved sessions must not be reported as noise"
    );
}

// #183 T-003: both reasons render distinct remediation wording and machine
// tags. Calling both variants directly keeps every match arm DA>0 without
// constructing an APFS-unreachable failure (the #49/#121 coverage trap).
#[test]
fn test_skipped_reason_note_and_tag_differ_per_variant() {
    let missing = SkippedReason::MissingRoot.note("codex", 3);
    assert!(
        missing.contains("once the root is back"),
        "a missing root must advise waiting for the root to return: {missing}"
    );
    assert!(
        !missing.contains("reconcile any real deletions"),
        "the missing-root note must use forward-looking framing, not past-tense deletions (#187): {missing}"
    );
    // Mirror the IncompleteEnumeration preserved>0 test: the negative guard alone
    // only blocks the one historical phrase, so pin the new forward-looking content
    // positively too, else a revert that drops it (OPS-01 harmonization) goes unseen.
    assert!(
        missing.contains("none deleted this run"),
        "the missing-root note must state nothing was deleted this run (#187): {missing}"
    );
    assert!(
        missing.contains("full scan can reconcile"),
        "the missing-root note must frame reconcile as a future full scan (#187): {missing}"
    );
    let incomplete = SkippedReason::IncompleteEnumeration.note("claude", 2);
    assert!(
        incomplete.contains("permissions and access"),
        "incomplete enumeration must advise fixing access, not waiting: {incomplete}"
    );
    assert!(
        !incomplete.contains("once the root is back"),
        "incomplete enumeration must not reuse the missing-root remedy"
    );
    assert_eq!(SkippedReason::MissingRoot.as_str(), "missing_root");
    assert_eq!(
        SkippedReason::IncompleteEnumeration.as_str(),
        "incomplete_enumeration"
    );
}

// #185 T-185-3: with nothing preserved, the incomplete-enumeration note must
// drop the "preserved N / reconcile deletions" framing (there is nothing to
// reconcile) yet still advise fixing access. Calling the variant directly keeps
// the new preserved==0 match arm DA>0 deterministically (#49/#121 trap).
#[test]
fn test_incomplete_note_at_zero_preserved_omits_reconcile() {
    let note = SkippedReason::IncompleteEnumeration.note("claude", 0);
    assert!(
        note.contains("some sessions may be missing"),
        "a zero-preserved partial read must warn about missing sessions: {note}"
    );
    assert!(
        note.contains("permissions and access"),
        "it must still advise fixing access: {note}"
    );
    assert!(
        !note.contains("reconcile any real deletions"),
        "with nothing preserved there are no deletions to reconcile: {note}"
    );
    assert!(
        !note.contains("preserved 0"),
        "the self-contradictory 'preserved 0' phrasing must not appear: {note}"
    );
}

// #187: with rows preserved, the reconcile clause is forward-looking — it points
// at the next full scan, not deletions that already happened. The note must say
// nothing was deleted this run and frame the reconcile as a future scan, so the
// audit-style misreading ("reconcile any real deletions" = deletions occurred)
// cannot recur. Calling the variant directly keeps the arm DA>0 (#49/#121 trap).
#[test]
fn test_incomplete_note_at_nonzero_preserved_frames_reconcile_as_future() {
    let note = SkippedReason::IncompleteEnumeration.note("claude", 2);
    assert!(
        note.contains("none deleted this run"),
        "a preserved partial read must state nothing was deleted this run: {note}"
    );
    assert!(
        note.contains("next full scan"),
        "the reconcile must be framed as a future scan, not a past deletion: {note}"
    );
    // Mirror the preserved==0 guard (test_incomplete_note_at_zero_preserved_omits_reconcile):
    // a positive-only test would still pass if the old past-tense clause were re-added
    // alongside the new one, so block the exact banned phrase from returning.
    assert!(
        !note.contains("reconcile any real deletions"),
        "the old past-tense deletion framing must not re-appear: {note}"
    );
}

// #183 T-004 (CHX-1 regression): the count-based early-return in cleanup_orphans
// must not mask a real deletion when an unrelated new file enters the same scan.
// existing {s1 codex, claude rows}; on disk the claude file is gone but a new
// claude file appears, so indexed>0 and the early-return guard stays off — the
// absent claude row must be deleted, not preserved by the early return.
#[test]
fn test_cleanup_deletes_real_orphan_when_new_file_added() {
    let (_dir, mut conn) = setup_test_db();
    let (_tmp, claude_dir, codex_dir) = seed_claude_and_codex(&mut conn);

    // The regression this test guards depends on the early-return's count-equality
    // half (sources.len() == existing.len()) being TRUE at cleanup, so only the
    // `indexed == 0` term keeps the guard off (the CHX-1 shape). That requires the
    // seed to hold exactly 2 sessions and the swap below to keep on-disk count at 2.
    // Anchor it so the test fails loudly if the shared seed later drifts, rather
    // than silently degrading into a plain cleanup test.
    let seeded: i64 = conn
        .query_row("SELECT COUNT(*) FROM sessions", [], |r| r.get(0))
        .unwrap();
    assert_eq!(
        seeded, 2,
        "T-004 needs exactly 2 seeded sessions to reach the guard's count-equality branch"
    );

    // The originally-seeded claude file is gone; replace it with a brand-new one
    // so the claude root is fully scanned with a non-empty, different file set.
    for entry in fs::read_dir(&claude_dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|e| e == "jsonl") {
            fs::remove_file(&path).unwrap();
        }
    }
    fs::write(
            claude_dir.join("fresh.jsonl"),
            r#"{"type":"user","cwd":"/proj","message":{"role":"user","content":"fresh claude turn"},"timestamp":"2026-07-01T00:00:00Z"}"#,
        )
        .unwrap();

    let stats = index_from_dirs(
        &mut conn,
        &IndexOptions {
            force: false,
            claude_dir: &claude_dir,
            codex_dir: &codex_dir,
        },
    )
    .unwrap();

    assert!(stats.indexed > 0, "the new claude file must be indexed");
    assert!(
        stats.skipped_roots.is_empty(),
        "both roots are present and fully scanned, so nothing is skipped"
    );
    let claude_remaining: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM sessions WHERE source = 'claude'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        claude_remaining, 1,
        "the absent original claude session must be deleted; only the fresh one remains \
         (if the early-return masked the deletion, the original row would inflate this to 2)"
    );
    assert_eq!(
        stats.total_sessions, 2,
        "codex-s1 is preserved and the claude orphan is swapped for fresh: 1 codex + 1 claude"
    );
}

// #177: `rebuild` (force) with a missing source root must not destroy that
// root's index. The force path used to bulk-delete every table before re-scan,
// so a transient root outage wiped the preserved root's sessions and their
// embeddings unrecoverably. The fix removes the blanket delete and routes force
// deletion through the same source-keyed cleanup as #165: the present root is
// genuinely rebuilt while the missing root's rows survive.
#[test]
fn test_force_rebuild_preserves_missing_source_root() {
    let (_dir, mut conn) = setup_test_db();
    let (tmp, claude_dir, _codex_dir) = seed_claude_and_codex(&mut conn);

    // Build chunks + embeddings for both seeded sessions so the data-loss is
    // observable at the embedding layer, not just the session row.
    index_chunks(&mut conn, None).unwrap();
    let embedder = MockEmbedder::new();
    embed_recent_chunks(&mut conn, &embedder, 100, None).unwrap();
    let codex_vecs_before: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM vec_chunks WHERE chunk_id IN \
             (SELECT id FROM qa_chunks WHERE session_id = 's1')",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert!(
        codex_vecs_before > 0,
        "the codex session must have embeddings before the rebuild"
    );

    // Rebuild (force) while the codex root is missing.
    let gone_codex = tmp.path().join("missing_codex");
    let stats = index_from_dirs(
        &mut conn,
        &IndexOptions {
            force: true,
            claude_dir: &claude_dir,
            codex_dir: &gone_codex,
        },
    )
    .unwrap();

    assert_eq!(
        stats.total_sessions, 2,
        "force rebuild with a missing root must preserve that root's session"
    );
    let codex_vecs_after: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM vec_chunks WHERE chunk_id IN \
             (SELECT id FROM qa_chunks WHERE session_id = 's1')",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        codex_vecs_after, codex_vecs_before,
        "the missing root's embeddings must survive the rebuild unchanged"
    );
    // The present claude root is genuinely rebuilt: its chunks are cleared,
    // awaiting re-embed — force is not silently downgraded to incremental.
    let claude_chunks_after: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM qa_chunks WHERE session_id = 'c'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        claude_chunks_after, 0,
        "the present root is rebuilt: its chunks are cleared for re-embed"
    );
    assert_eq!(
        stats.skipped_roots.len(),
        1,
        "the missing codex root must be reported as preserved"
    );
    assert_eq!(stats.skipped_roots[0].source, Source::Codex);
}

// #177 worst case: a force rebuild where BOTH source roots are missing must not
// wipe the index. With nothing successfully scanned, every session and its
// embeddings survive and both roots are reported as preserved.
#[test]
fn test_force_rebuild_all_roots_missing_preserves_index() {
    let (_dir, mut conn) = setup_test_db();
    let (tmp, _claude_dir, _codex_dir) = seed_claude_and_codex(&mut conn);

    index_chunks(&mut conn, None).unwrap();
    let embedder = MockEmbedder::new();
    embed_recent_chunks(&mut conn, &embedder, 100, None).unwrap();
    let vecs_before: i64 = conn
        .query_row("SELECT COUNT(*) FROM vec_chunks", [], |r| r.get(0))
        .unwrap();
    assert!(vecs_before > 0, "both sessions must have embeddings first");

    let gone_claude = tmp.path().join("missing_claude");
    let gone_codex = tmp.path().join("missing_codex");
    let stats = index_from_dirs(
        &mut conn,
        &IndexOptions {
            force: true,
            claude_dir: &gone_claude,
            codex_dir: &gone_codex,
        },
    )
    .unwrap();

    assert_eq!(
        stats.total_sessions, 2,
        "both sessions must survive when neither root is scannable"
    );
    let vecs_after: i64 = conn
        .query_row("SELECT COUNT(*) FROM vec_chunks", [], |r| r.get(0))
        .unwrap();
    assert_eq!(
        vecs_after, vecs_before,
        "every embedding must survive a force rebuild with both roots missing"
    );
    assert_eq!(
        stats.skipped_roots.len(),
        2,
        "both missing roots must be reported as preserved"
    );
}

// #177 force-path cascade: the two missing-root force tests above exit
// cleanup_orphans early (the absent root is unscanned, so its rows are
// preserved). This pins the other branch — a force rebuild where both roots
// are scanned and a file genuinely vanished must still delete that session and
// cascade to its messages and qa_chunks, exactly as the incremental path does.
#[test]
fn test_force_rebuild_deleted_file_under_scanned_root_cascades_cleanup() {
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

    // Delete the claude file; both roots stay present and readable, so both
    // are scanned and the deleted row is delete-eligible under force.
    fs::remove_file(claude_dir.join("c.jsonl")).unwrap();
    let stats = index_from_dirs(
        &mut conn,
        &IndexOptions {
            force: true,
            claude_dir: &claude_dir,
            codex_dir: &codex_dir,
        },
    )
    .unwrap();

    assert_eq!(
        stats.total_sessions, 1,
        "force must clean the deleted file's session when its root was scanned"
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
        "force cleanup must cascade to the session's messages"
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
        "force cleanup must cascade to the session's qa_chunks"
    );
    assert!(
        stats.skipped_roots.is_empty(),
        "both roots were scanned, so nothing should be reported as preserved"
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

// #181: a per-entry read failure (DirEntry or file_type() error) leaves that
// entry unseen, so the enumeration must report incomplete (false) — otherwise
// the source counts as scanned and cleanup_orphans deletes the unseen file's
// row. APFS never triggers this path (readdir returns d_type), so the only way
// to cover the merged error arm is to inject an Err item directly. The Err is
// chained FIRST so that real.jsonl is read after it: collecting the file proves
// the error arm only suppresses cleanup and lets enumeration continue (a `break`
// or early `return` in place of `continue` would yield an empty `files`).
#[test]
fn test_collect_from_entries_incomplete_on_entry_error_still_collects() {
    use std::{io, iter};

    let tmp = TempDir::new().unwrap();
    fs::write(tmp.path().join("real.jsonl"), "{}").unwrap();

    let entries =
        iter::once(Err(io::Error::other("injected"))).chain(fs::read_dir(tmp.path()).unwrap());

    let mut files = Vec::new();
    let fully_read = collect_from_entries(entries, tmp.path(), &mut files, Source::Claude, 0);

    assert!(
        !fully_read,
        "a per-entry error must mark the tree as not fully enumerated"
    );
    assert_eq!(
        files.len(),
        1,
        "the readable jsonl is still collected despite the failed entry"
    );
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

/// Seeds one message into the default test session `s1`. Returns its rowid so the
/// backfill tests can pin which range a chunk should receive.
fn seed_message(conn: &Connection, role: &str, text: &str) -> i64 {
    conn.execute(
        "INSERT INTO messages (session_id, role, text) VALUES ('s1', ?1, ?2)",
        rusqlite::params![role, text],
    )
    .unwrap();
    conn.query_row(
        "SELECT rowid FROM messages WHERE session_id = 's1' AND text = ?1",
        [text],
        |r| r.get(0),
    )
    .unwrap()
}

/// Seeds a legacy qa_chunk (NULL src_rowid_lo/hi) plus its vec_chunks embedding,
/// mirroring a row written before #192. Backfill must populate lo/hi in place
/// without disturbing id or the embedding.
fn seed_legacy_chunk(conn: &Connection, id: i64, content: &str) {
    conn.execute(
        "INSERT INTO qa_chunks (id, session_id, content, timestamp) VALUES (?1, 's1', ?2, 0)",
        rusqlite::params![id, content],
    )
    .unwrap();
    let embedding = MockEmbedder::deterministic_vector(content);
    conn.execute(
        "INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?1, ?2)",
        rusqlite::params![id, f32_as_bytes(&embedding)],
    )
    .unwrap();
}

// T-192e (#192, FR-6): migration backfill on a count-matching session. A legacy
// session whose stored qa_chunks count equals the re-derived chunk count gets
// its NULL src_rowid_lo/hi populated by re-running the chunker and zipping by
// qa_chunks.id ascending. The embedding rows (vec_chunks) and qa_chunks.id must
// be untouched — no re-embed, no row churn (FR-5). Perspective: State (NULL →
// populated) + Hazard (embedding loss / id reassignment on migration).
#[test]
fn test_192e_backfill_populates_rowid_range_count_match() {
    let (_dir, mut conn) = setup_test_db();
    seed_session(&conn, "s1");
    // One user+assistant pair → the chunker re-derives exactly one chunk.
    let user_rowid = seed_message(&conn, "user", "how to rotate tokens");
    let assistant_rowid = seed_message(&conn, "assistant", "use a refresh grant");
    // Legacy stored chunk: NULL lo/hi, one row → matches the re-derived count of 1.
    seed_legacy_chunk(&conn, 1, "how to rotate tokens\nuse a refresh grant");

    let vec_before: i64 = conn
        .query_row("SELECT COUNT(*) FROM vec_chunks", [], |r| r.get(0))
        .unwrap();

    let updated = backfill_rowid_ranges(&mut conn).unwrap();
    assert_eq!(updated, 1, "the count-matching session is backfilled");

    let (lo, hi): (Option<i64>, Option<i64>) = conn
        .query_row(
            "SELECT src_rowid_lo, src_rowid_hi FROM qa_chunks WHERE id = 1",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    assert_eq!(
        lo,
        Some(user_rowid),
        "backfill sets lo to the user message rowid"
    );
    assert_eq!(
        hi,
        Some(assistant_rowid),
        "backfill sets hi to the adjacent assistant message rowid"
    );

    let vec_after: i64 = conn
        .query_row("SELECT COUNT(*) FROM vec_chunks", [], |r| r.get(0))
        .unwrap();
    assert_eq!(
        vec_after, vec_before,
        "backfill must not delete or re-create embeddings"
    );
    let embedding_kept: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM vec_chunks WHERE chunk_id = 1",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        embedding_kept, 1,
        "the chunk's embedding stays linked by its unchanged id"
    );
}

// T-192f (#192, FR-6): migration safety on a count mismatch. When the re-derived
// chunk count does not equal the stored count, the id-order zip is unsafe (a
// chunk would get the wrong message's rowid), so backfill leaves the rows' lo/hi
// NULL and does not panic. Those rows then resolve via the instr fallback
// (T-192d). Perspective: Error (mismatched cardinality) + Hazard (silent
// mis-stamping). The stored count is 2; the seeded single user message re-derives
// to 1 chunk → mismatch.
#[test]
fn test_192f_backfill_leaves_null_on_count_mismatch() {
    let (_dir, mut conn) = setup_test_db();
    seed_session(&conn, "s1");
    // One user-only message → the chunker re-derives exactly one chunk.
    seed_message(&conn, "user", "single turn");
    // But two legacy chunks are stored → re-derived (1) != stored (2): mismatch.
    seed_legacy_chunk(&conn, 1, "single turn");
    seed_legacy_chunk(&conn, 2, "orphaned extra chunk");

    let updated = backfill_rowid_ranges(&mut conn).unwrap();
    assert_eq!(
        updated, 0,
        "a count-mismatched session is skipped, not backfilled"
    );

    let null_count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM qa_chunks WHERE session_id = 's1' AND src_rowid_lo IS NULL",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        null_count, 2,
        "both rows keep NULL lo/hi when the chunk count cannot be reconciled"
    );
}

// T-192h (#192): read_session_messages skips rows whose role is neither user nor
// assistant (Role::from_db returns None). recall only writes those two roles, so
// this is defensive, but an unknown role must never leak into a chunk's content or
// shift the rowid range. Perspective: Error (invalid role input) + Hazard (foreign
// text contaminating an excerpt). Here a 'system' row sits between two real
// messages; the re-derived chunk must contain only the user/assistant text.
#[test]
fn test_192h_unknown_role_message_is_skipped() {
    let (_dir, mut conn) = setup_test_db();
    seed_session(&conn, "s1");
    seed_message(&conn, "user", "how do vaccines train immunity");
    // 'system' is not a known role; read_session_messages must drop it.
    seed_message(&conn, "system", "INTERNAL_SYSTEM_NOTE do not index");
    seed_message(&conn, "assistant", "they present a harmless antigen");

    let stats = index_chunks(&mut conn, None).unwrap();
    assert_eq!(
        stats.chunks_created, 1,
        "the user message pairs with the adjacent assistant; the system row is dropped"
    );

    let content: String = conn
        .query_row(
            "SELECT content FROM qa_chunks WHERE session_id = 's1'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert!(
        !content.contains("INTERNAL_SYSTEM_NOTE")
            && content.contains("how do vaccines train immunity")
            && content.contains("they present a harmless antigen"),
        "the chunk holds only the user and assistant text, never the skipped system \
         row: {content:?}"
    );
}
