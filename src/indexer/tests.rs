use std::cell::RefCell;
use std::ffi::OsString;
use std::io;
use std::sync::Mutex;
use std::time::{Duration, SystemTime};

use super::*;
use crate::commands::classify::reclassify_sessions;
use crate::db::{open_db, seed_session, seed_session_file, setup_test_db};
use crate::embedder::{EMBED_BATCH_SIZE, MockEmbedder, embed_recent_chunks, f32_as_bytes};
use crate::index_observer::{recording, stable_lines};
use crate::search::{SearchOptions, search};
use tempfile::TempDir;

fn collect_strings(conn: &Connection, sql: &str) -> Vec<String> {
    let mut stmt = conn.prepare(sql).unwrap();
    let rows = stmt.query_map([], |r| r.get::<_, String>(0)).unwrap();
    rows.map(Result::unwrap).collect()
}

#[test]
fn reindexing_unchanged_logs_indexes_nothing_and_leaves_stored_rows_unchanged() {
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
        codex_archived_dir: None,
    };

    index_from_dirs(&mut conn, &opts, true).unwrap();
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

    let stats = index_from_dirs(&mut conn, &opts, true).unwrap();
    index_chunks(&mut conn, None).unwrap();

    assert_eq!(stats.indexed, 0, "an unchanged tree re-parses nothing");
    assert_eq!(
        counts(&conn),
        before,
        "re-index of an unchanged tree must not grow or rewrite the DB"
    );
}

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
        codex_archived_dir: None,
    };
    index_from_dirs(&mut conn, &opts, true).unwrap();

    let mut all = fs::read_to_string(&file).unwrap();
    all.push_str(
            r#"{"type":"assistant","cwd":"/proj","message":{"role":"assistant","content":"appended answer"},"timestamp":"2026-03-01T00:00:01Z"}"#,
        );
    fs::write(&file, all).unwrap();
    let f = fs::OpenOptions::new().append(true).open(&file).unwrap();
    f.set_modified(SystemTime::now() + Duration::from_secs(10))
        .unwrap();
    drop(f);

    let stats = index_from_dirs(&mut conn, &opts, true).unwrap();

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

const FRESHNESS_QUESTION: &str = concat!(
    r#"{"type":"user","cwd":"/proj","message":{"role":"user","content":"original question"}}"#,
    "\n",
);
const FRESHNESS_ANSWER: &str = concat!(
    r#"{"type":"assistant","message":{"role":"assistant","content":"appended answer"}}"#,
    "\n",
);

fn rewrite_with_mtime(path: &Path, body: &str, mtime: SystemTime) {
    fs::write(path, body).unwrap();
    fs::OpenOptions::new()
        .write(true)
        .open(path)
        .unwrap()
        .set_modified(mtime)
        .unwrap();
}

#[test]
fn size_changes_reindex_with_equal_or_submillisecond_mtime() {
    for (source, question, answer, metadata) in [
        (
            Source::Claude,
            FRESHNESS_QUESTION,
            FRESHNESS_ANSWER,
            r#"{"type":"system","cwd":"/proj"}"#,
        ),
        (
            Source::Codex,
            "{\"type\":\"response_item\",\"payload\":{\"role\":\"user\",\"content\":\"original question\"}}\n",
            "{\"type\":\"response_item\",\"payload\":{\"role\":\"assistant\",\"content\":\"appended answer\"}}\n",
            r#"{"type":"session_meta","payload":{"cwd":"/proj"}}"#,
        ),
    ] {
        for offset in [Duration::ZERO, Duration::from_micros(500)] {
            let (_db_dir, mut conn) = setup_test_db();
            let root = TempDir::new().unwrap();
            let file = root.path().join("s.jsonl");
            let absent = root.path().join("absent");
            let opts = IndexOptions {
                force: false,
                claude_dir: if source == Source::Claude {
                    root.path()
                } else {
                    &absent
                },
                codex_dir: if source == Source::Codex {
                    root.path()
                } else {
                    &absent
                },
                codex_archived_dir: None,
            };
            let initial = UNIX_EPOCH + Duration::from_secs(1_700_000_000);
            rewrite_with_mtime(&file, question, initial);
            index_from_dirs(&mut conn, &opts, true).unwrap();

            rewrite_with_mtime(
                &file,
                &question.replace("original", "modified"),
                initial + Duration::from_secs(2),
            );
            assert_eq!(index_from_dirs(&mut conn, &opts, true).unwrap().indexed, 1);
            assert_eq!(
                collect_strings(&conn, "SELECT text FROM messages"),
                ["modified question"]
            );
            rewrite_with_mtime(&file, "{", initial + Duration::from_secs(2));
            assert_eq!(index_from_dirs(&mut conn, &opts, true).unwrap().indexed, 0);
            assert_eq!(
                collect_strings(&conn, "SELECT text FROM messages"),
                ["modified question"]
            );
            let appended = format!("{question}{answer}");
            for (body, expected) in [
                (
                    appended.as_str(),
                    vec!["original question", "appended answer"],
                ),
                (question, vec!["original question"]),
                (metadata, vec![]),
                ("", vec![]),
            ] {
                let old = fs::metadata(&file).unwrap().modified().unwrap();
                rewrite_with_mtime(&file, body, old + offset);
                let observed = fs::metadata(&file).unwrap().modified().unwrap();
                assert_eq!(observed.duration_since(old).unwrap(), offset);
                assert_eq!(index_from_dirs(&mut conn, &opts, true).unwrap().indexed, 1);
                index_chunks(&mut conn, None).unwrap();
                assert_eq!(
                    collect_strings(&conn, "SELECT text FROM messages ORDER BY rowid"),
                    expected
                );
                assert_eq!(index_from_dirs(&mut conn, &opts, true).unwrap().indexed, 0);
            }
            let existing = load_existing_sessions(&conn).unwrap();
            let tx = conn.transaction().unwrap();
            let ctx = IndexContext {
                tx: &tx,
                existing: &existing,
                force: false,
                embedded_sessions: None,
            };
            assert!(matches!(
                index_file_with_parser(&ctx, &file, &source, |_, _| panic!(
                    "unchanged empty body must not be parsed"
                ))
                .unwrap(),
                IndexOutcome::Unchanged
            ));
            drop(tx);
            assert!(collect_strings(&conn, "SELECT content FROM qa_chunks").is_empty());
        }
    }
}

#[test]
fn metadata_only_reindex_applies_provenance_only_to_the_matching_session() {
    for initial_type in ["interactive", "automated"] {
        for replacement_id in ["session-a", "session-b"] {
            let (_db_dir, mut conn) = setup_test_db();
            let root = TempDir::new().unwrap();
            let file = root.path().join("rollout-test.jsonl");
            let absent = root.path().join("absent");
            let opts = IndexOptions {
                force: false,
                claude_dir: &absent,
                codex_dir: root.path(),
                codex_archived_dir: None,
            };
            let initial = format!(
                "{{\"type\":\"session_meta\",\"payload\":{{\"id\":\"session-a\",\"source\":\"exec\"}}}}\n{}",
                r#"{"type":"response_item","payload":{"role":"user","content":"original question"}}"#,
            );
            fs::write(&file, &initial).unwrap();
            index_from_dirs(&mut conn, &opts, true).unwrap();
            index_chunks(&mut conn, None).unwrap();
            conn.execute("UPDATE sessions SET session_type = ?", [initial_type])
                .unwrap();
            let replacement = format!(
                r#"{{"type":"session_meta","payload":{{"id":"{replacement_id}","source":{{"subagent":"review"}}}}}}"#,
            );
            assert_ne!(initial.len(), replacement.len(), "force a freshness miss");
            fs::write(&file, replacement).unwrap();

            let matches = replacement_id == "session-a";
            for attempt in 0..2 {
                let stats = index_from_dirs(&mut conn, &opts, true).unwrap();
                index_chunks(&mut conn, None).unwrap();
                assert_eq!(
                    collect_strings(&conn, "SELECT session_id FROM sessions"),
                    ["session-a"]
                );
                assert_eq!(
                    collect_strings(&conn, "SELECT session_type FROM sessions"),
                    [if matches { "automated" } else { initial_type }],
                    "replacement {replacement_id}, initial {initial_type}, attempt {attempt}"
                );
                assert_eq!(stats.indexed, usize::from(matches && attempt == 0));
                assert_eq!(
                    collect_strings(&conn, "SELECT text FROM messages"),
                    if matches {
                        vec![]
                    } else {
                        vec!["original question"]
                    },
                    "another session's metadata cannot establish that this body is empty"
                );
                reclassify_sessions(&mut conn, true, false).unwrap();
                assert_eq!(
                    collect_strings(&conn, "SELECT session_type FROM sessions"),
                    [if matches { "automated" } else { initial_type }],
                    "explicit reclassification must agree with indexing"
                );
            }
        }
    }
}

#[test]
fn append_around_parsing_is_retried_and_unchanged_body_is_not_parsed() {
    for order in 0..3 {
        let (_db_dir, mut conn) = setup_test_db();
        let root = TempDir::new().unwrap();
        let file = root.path().join("s.jsonl");
        fs::write(&file, FRESHNESS_QUESTION).unwrap();
        let mtime = fs::metadata(&file).unwrap().modified().unwrap();
        let append = || {
            rewrite_with_mtime(
                &file,
                &format!("{FRESHNESS_QUESTION}{FRESHNESS_ANSWER}"),
                mtime,
            )
        };
        let existing = load_existing_sessions(&conn).unwrap();
        let tx = conn.transaction().unwrap();
        let ctx = IndexContext {
            tx: &tx,
            existing: &existing,
            force: false,
            embedded_sessions: None,
        };
        let outcome = index_file_with_parser(&ctx, &file, &Source::Claude, |path, source| {
            if order == 0 {
                append();
            }
            let parsed = parse_session_including_empty(path, source);
            if order == 1 {
                append();
            }
            parsed
        })
        .unwrap();
        assert!(matches!(outcome, IndexOutcome::Indexed(_)));
        if order == 2 {
            append();
        }
        tx.commit().unwrap();
        let saved: Option<i64> = conn
            .query_row("SELECT file_size FROM sessions", [], |r| r.get(0))
            .unwrap();
        assert_eq!(
            saved,
            if order == 2 {
                Some(i64::try_from(FRESHNESS_QUESTION.len()).unwrap())
            } else {
                None
            }
        );

        let absent = root.path().join("absent");
        let opts = IndexOptions {
            force: false,
            claude_dir: root.path(),
            codex_dir: &absent,
            codex_archived_dir: None,
        };
        assert_eq!(index_from_dirs(&mut conn, &opts, true).unwrap().indexed, 1);
        assert_eq!(
            collect_strings(&conn, "SELECT text FROM messages ORDER BY rowid"),
            ["original question", "appended answer"]
        );
        index_chunks(&mut conn, None).unwrap();
        let existing = load_existing_sessions(&conn).unwrap();
        let tx = conn.transaction().unwrap();
        let ctx = IndexContext {
            tx: &tx,
            existing: &existing,
            force: false,
            embedded_sessions: None,
        };
        assert!(matches!(
            index_file_with_parser(&ctx, &file, &Source::Claude, |_, _| panic!(
                "unchanged body must not be parsed"
            ))
            .unwrap(),
            IndexOutcome::Unchanged
        ));
    }
}

#[test]
fn legacy_size_upgrade_preserves_embeddings_and_retries_until_model_returns() {
    let db_dir = TempDir::new().unwrap();
    let db_path = db_dir.path().join("index.db");
    let mut conn = open_db(&db_path).unwrap();
    let root = TempDir::new().unwrap();
    let file = root.path().join("s.jsonl");
    fs::write(&file, FRESHNESS_QUESTION).unwrap();
    let mtime = fs::metadata(&file).unwrap().modified().unwrap();
    let absent = root.path().join("absent");
    let opts = IndexOptions {
        force: false,
        claude_dir: root.path(),
        codex_dir: &absent,
        codex_archived_dir: None,
    };
    index_from_dirs(&mut conn, &opts, true).unwrap();
    index_chunks(&mut conn, None).unwrap();
    embed_recent_chunks(&mut conn, &MockEmbedder::new(), 100, None).unwrap();
    let snapshot = |conn: &Connection| {
        [
            collect_strings(
                conn,
                "SELECT json_array(session_id, mtime, chunks_indexed, files_scanned) FROM sessions",
            ),
            collect_strings(conn, "SELECT json_array(rowid, text) FROM messages"),
            collect_strings(
                conn,
                "SELECT json_array(id, content, src_rowid_lo, src_rowid_hi) FROM qa_chunks",
            ),
            collect_strings(
                conn,
                "SELECT json_array(rowid, chunk_id, hex(embedding)) FROM vec_chunks",
            ),
        ]
    };
    let before = snapshot(&conn);
    assert!(!before[3].is_empty());
    conn.execute_batch("ALTER TABLE sessions DROP COLUMN file_size;")
        .unwrap();
    drop(conn);
    rewrite_with_mtime(
        &file,
        &format!("{FRESHNESS_QUESTION}{FRESHNESS_ANSWER}"),
        mtime,
    );
    for _ in 0..2 {
        let mut conn = open_db(&db_path).unwrap();
        assert_eq!(snapshot(&conn), before);
        let stats = index_from_dirs(&mut conn, &opts, false).unwrap();
        assert_eq!((stats.indexed, stats.preserved_embedded), (0, 1));
        assert_eq!(snapshot(&conn), before);
        let size: Option<i64> = conn
            .query_row("SELECT file_size FROM sessions", [], |r| r.get(0))
            .unwrap();
        assert_eq!(size, None);
    }
    let mut conn = open_db(&db_path).unwrap();
    assert_eq!(index_from_dirs(&mut conn, &opts, true).unwrap().indexed, 1);
    assert_eq!(
        collect_strings(&conn, "SELECT text FROM messages ORDER BY rowid"),
        ["original question", "appended answer"]
    );
    drop(conn);
    let mut conn = open_db(&db_path).unwrap();
    assert_eq!(index_from_dirs(&mut conn, &opts, true).unwrap().indexed, 0);
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
            codex_archived_dir: None,
        },
        true,
    )
    .unwrap();
    conn.query_row("SELECT session_type FROM sessions", [], |r| r.get(0))
        .unwrap()
}

#[test]
fn slash_command_first_turn_is_stored_as_automated_and_human_first_turn_as_interactive() {
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
            codex_archived_dir: None,
        },
        true,
    )
    .unwrap();
    assert_eq!(stats1.indexed, 1);
    index_chunks(&mut conn, None).unwrap();

    let s2 = r#"{"timestamp":"2026-04-27T00:00:00Z","type":"session_meta","payload":{"id":"codex-s2","cwd":"/proj"}}
{"timestamp":"2026-04-27T00:00:01Z","type":"response_item","payload":{"role":"user","content":[{"type":"input_text","text":"second codex session"}]}}"#;
    fs::write(day_dir.join("s2.jsonl"), s2).unwrap();

    let stats2 = index_from_dirs(
        &mut conn,
        &IndexOptions {
            force: false,
            claude_dir: &claude_dir,
            codex_dir: &codex_dir,
            codex_archived_dir: None,
        },
        true,
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
            codex_archived_dir: None,
        },
        true,
    )
    .unwrap();
    assert_eq!(stats.indexed, 1);

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
            codex_archived_dir: None,
        },
        true,
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
            codex_archived_dir: None,
        },
        true,
    )
    .unwrap();
    assert_eq!(stats.total_sessions, 2, "seed should index both sessions");
    (tmp, claude_dir, codex_dir)
}

#[test]
fn deleted_claude_log_is_removed_while_sessions_under_a_missing_codex_root_are_kept_and_reported() {
    let (_dir, mut conn) = setup_test_db();
    let (tmp, claude_dir, _codex_dir) = seed_claude_and_codex(&mut conn);
    fs::remove_file(claude_dir.join("c.jsonl")).unwrap();
    let missing_codex = tmp.path().join("missing_codex");

    let stats = index_from_dirs(
        &mut conn,
        &IndexOptions {
            force: false,
            claude_dir: &claude_dir,
            codex_dir: &missing_codex,
            codex_archived_dir: None,
        },
        true,
    )
    .unwrap();

    assert_eq!(stats.total_sessions, 1);
    assert_eq!(
        collect_strings(&conn, "SELECT source FROM sessions"),
        ["codex"]
    );
    assert_eq!(
        stats
            .skipped_roots
            .iter()
            .map(|s| (s.source, s.reason, s.preserved_sessions))
            .collect::<Vec<_>>(),
        [(Source::Codex, SkippedReason::MissingRoot, 1)]
    );
}

#[test]
fn deleting_a_log_under_a_fully_read_root_removes_its_session_messages_chunks_embeddings_and_touched_files()
 {
    let (_dir, mut conn) = setup_test_db();
    let tmp = TempDir::new().unwrap();
    let claude_dir = tmp.path().join("claude_projects");
    fs::create_dir_all(&claude_dir).unwrap();
    let claude_log = claude_dir.join("c.jsonl");
    fs::write(
        &claude_log,
        concat!(
            r#"{"type":"user","cwd":"/proj","message":{"role":"user","content":"claude turn"},"timestamp":"2026-03-01T00:00:00Z"}"#,
            "\n",
            r#"{"type":"assistant","cwd":"/proj","message":{"role":"assistant","content":[{"type":"text","text":"editing"},{"type":"tool_use","id":"t1","name":"Edit","input":{"file_path":"/proj/c.rs","old_string":"x","new_string":"y"}}]},"timestamp":"2026-03-01T00:00:01Z"}"#,
        ),
    )
    .unwrap();
    let codex_dir = tmp.path().join("codex_sessions");
    let day_dir = codex_dir.join("2026/04/27");
    fs::create_dir_all(&day_dir).unwrap();
    fs::write(
        day_dir.join("s1.jsonl"),
        concat!(
            r#"{"timestamp":"2026-04-27T00:00:00Z","type":"session_meta","payload":{"id":"codex-s1","cwd":"/proj"}}"#,
            "\n",
            r#"{"timestamp":"2026-04-27T00:00:01Z","type":"response_item","payload":{"role":"user","content":[{"type":"input_text","text":"codex turn"}]}}"#,
        ),
    )
    .unwrap();
    let opts = IndexOptions {
        force: false,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
        codex_archived_dir: None,
    };
    index_from_dirs(&mut conn, &opts, true).unwrap();
    index_chunks(&mut conn, None).unwrap();
    assert_eq!(
        embed_recent_chunks(&mut conn, &MockEmbedder::new(), 100, None)
            .unwrap()
            .embedded,
        2
    );
    assert_eq!(
        collect_strings(&conn, "SELECT path FROM session_files"),
        ["/proj/c.rs"]
    );

    fs::remove_file(&claude_log).unwrap();
    let stats = index_from_dirs(&mut conn, &opts, true).unwrap();

    assert_eq!(stats.total_sessions, 1);
    assert!(stats.skipped_roots.is_empty());
    assert_eq!(
        collect_strings(&conn, "SELECT session_id FROM sessions"),
        ["s1"]
    );
    assert_eq!(
        collect_strings(&conn, "SELECT DISTINCT session_id FROM messages"),
        ["s1"]
    );
    assert_eq!(
        collect_strings(&conn, "SELECT DISTINCT session_id FROM qa_chunks"),
        ["s1"]
    );
    let vectors: i64 = conn
        .query_row("SELECT COUNT(*) FROM vec_chunks", [], |r| r.get(0))
        .unwrap();
    assert_eq!(vectors, 1, "the deleted session's embedding is removed");
    assert!(collect_strings(&conn, "SELECT path FROM session_files").is_empty());
}

#[cfg(unix)]
#[test]
fn unreadable_codex_subdirectory_keeps_every_session_and_reports_only_unseen_ones_as_preserved() {
    use std::os::unix::fs::PermissionsExt;

    let (_dir, mut conn) = setup_test_db();
    let (_tmp, claude_dir, codex_dir) = seed_claude_and_codex(&mut conn);
    fs::write(
        codex_dir.join("top.jsonl"),
        concat!(
            r#"{"timestamp":"2026-05-01T00:00:00Z","type":"session_meta","payload":{"id":"codex-top","cwd":"/proj"}}"#,
            "\n",
            r#"{"timestamp":"2026-05-01T00:00:01Z","type":"response_item","payload":{"role":"user","content":[{"type":"input_text","text":"codex top turn"}]}}"#,
        ),
    )
    .unwrap();
    let opts = IndexOptions {
        force: false,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
        codex_archived_dir: None,
    };
    index_from_dirs(&mut conn, &opts, true).unwrap();

    let locked_subdir = codex_dir.join("2026");
    let permissions = fs::metadata(&locked_subdir).unwrap().permissions();
    fs::set_permissions(&locked_subdir, fs::Permissions::from_mode(0o000)).unwrap();
    let stats = index_from_dirs(&mut conn, &opts, true);
    fs::set_permissions(&locked_subdir, permissions).unwrap();
    let stats = stats.unwrap();

    assert_eq!(stats.total_sessions, 3, "the unseen s1 session is kept");
    assert_eq!(
        stats
            .skipped_roots
            .iter()
            .map(|s| (s.source, s.reason, s.preserved_sessions))
            .collect::<Vec<_>>(),
        [(Source::Codex, SkippedReason::IncompleteEnumeration, 1)],
        "only the unseen s1 session counts as preserved; the re-read top session does not"
    );
}

#[cfg(unix)]
#[test]
fn codex_root_that_cannot_be_fully_read_is_reported_even_with_no_stored_sessions() {
    use std::os::unix::fs::PermissionsExt;

    let (_dir, mut conn) = setup_test_db();
    let tmp = TempDir::new().unwrap();
    let claude_dir = tmp.path().join("claude_projects");
    fs::create_dir_all(&claude_dir).unwrap();
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
            codex_archived_dir: None,
        },
        true,
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

#[test]
fn missing_roots_without_stored_sessions_are_not_reported() {
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
            codex_archived_dir: None,
        },
        true,
    )
    .unwrap();

    assert!(
        stats.skipped_roots.is_empty(),
        "missing roots with no preserved sessions must not be reported as noise"
    );
}

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

#[test]
fn incomplete_root_note_without_preserved_sessions_warns_that_sessions_may_be_missing() {
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

#[test]
fn incomplete_root_note_with_preserved_sessions_says_nothing_was_deleted_and_to_fix_access() {
    let note = SkippedReason::IncompleteEnumeration.note("claude", 2);
    assert!(
        note.contains("none deleted this run"),
        "a preserved partial read must state nothing was deleted this run: {note}"
    );
    assert!(
        note.contains("next full scan"),
        "the reconcile must be framed as a future scan, not a past deletion: {note}"
    );
    assert!(
        !note.contains("reconcile any real deletions"),
        "the old past-tense deletion framing must not re-appear: {note}"
    );
}

#[cfg(unix)]
#[test]
fn equal_count_replacement_without_indexing_removes_only_confirmed_orphans() {
    use std::os::unix::fs::PermissionsExt;

    for (replacement, unreadable) in [("", false), ("invalid JSON\n", false), ("", true)] {
        for incomplete_codex in [false, true] {
            let (_dir, mut conn) = setup_test_db();
            let (_tmp, claude_dir, codex_dir) = seed_claude_and_codex(&mut conn);
            index_chunks(&mut conn, None).unwrap();
            let embedder = MockEmbedder::new();
            assert_eq!(
                embed_recent_chunks(&mut conn, &embedder, 100, None)
                    .unwrap()
                    .embedded,
                2
            );
            for sid in ["c", "s1"] {
                seed_session_file(&conn, sid, &format!("/proj/{sid}.rs"));
            }

            let snapshot = |conn: &Connection, sid: &str| {
                [
                    "SELECT json_array(session_id, source, file_path, project, slug, timestamp, mtime, file_size, session_type, files_scanned, chunks_indexed) FROM sessions WHERE session_id = ? ORDER BY session_id",
                    "SELECT json_array(rowid, role, text) FROM messages WHERE session_id = ? ORDER BY rowid",
                    "SELECT json_array(id, content, timestamp, src_rowid_lo, src_rowid_hi, generation) FROM qa_chunks WHERE session_id = ? ORDER BY id",
                    "SELECT json_array(rowid, chunk_id, sub_idx, hex(embedding)) FROM vec_chunks WHERE chunk_id IN (SELECT id FROM qa_chunks WHERE session_id = ?) ORDER BY rowid",
                    "SELECT path FROM session_files WHERE session_id = ? ORDER BY path",
                ]
                .map(|sql| {
                    conn.prepare(sql)
                        .unwrap()
                        .query_map([sid], |row| row.get::<_, String>(0))
                        .unwrap()
                        .map(Result::unwrap)
                        .collect::<Vec<_>>()
                })
            };
            let survivor = snapshot(&conn, "s1");
            assert!(
                survivor.iter().all(|rows| rows.len() == 1),
                "survivor row counts: {:?}",
                survivor.each_ref().map(Vec::len)
            );
            assert!(snapshot(&conn, "c").iter().all(|rows| rows.len() == 1));

            if incomplete_codex {
                let mut deep = codex_dir.clone();
                for _ in 0..MAX_DIR_DEPTH {
                    deep = deep.join("nested");
                }
                fs::create_dir_all(&deep).unwrap();
                fs::rename(codex_dir.join("2026/04/27/s1.jsonl"), deep.join("s1.jsonl")).unwrap();
                fs::write(codex_dir.join("empty.jsonl"), "").unwrap();
            }
            fs::remove_file(claude_dir.join("c.jsonl")).unwrap();
            let fresh = claude_dir.join("fresh.jsonl");
            fs::write(&fresh, replacement).unwrap();
            let opts = IndexOptions {
                force: false,
                claude_dir: &claude_dir,
                codex_dir: &codex_dir,
                codex_archived_dir: None,
            };
            assert_eq!(collect_sources(&opts).sources.len(), 2);

            for _ in 0..2 {
                let permissions = fs::metadata(&fresh).unwrap().permissions();
                if unreadable {
                    fs::set_permissions(&fresh, fs::Permissions::from_mode(0o000)).unwrap();
                }
                let stats = index_from_dirs(&mut conn, &opts, false);
                fs::set_permissions(&fresh, permissions).unwrap();
                let stats = stats.unwrap();
                assert_eq!(stats.indexed, 0);
                assert_eq!(
                    stats
                        .parse_diagnostics
                        .iter()
                        .filter(|d| d.read_error)
                        .count(),
                    usize::from(unreadable)
                );
                assert_eq!(
                    stats.total_sessions, 1,
                    "exactly the Claude orphan is deleted"
                );
                assert_eq!(snapshot(&conn, "s1"), survivor);
                assert!(snapshot(&conn, "c").iter().all(Vec::is_empty));
                for table in ["messages", "qa_chunks", "vec_chunks", "session_files"] {
                    let count: i64 = conn
                        .query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |r| r.get(0))
                        .unwrap();
                    assert_eq!(count, 1, "only the survivor's data remains in {table}");
                }
                assert_eq!(stats.skipped_roots.len(), usize::from(incomplete_codex));
                if incomplete_codex {
                    let skipped = &stats.skipped_roots[0];
                    assert_eq!(skipped.source, Source::Codex);
                    assert_eq!(skipped.reason, SkippedReason::IncompleteEnumeration);
                    assert_eq!(skipped.preserved_sessions, 1);
                }
            }
        }
    }
}

#[test]
fn complete_scan_preserves_sessions_with_unknown_source() {
    let (_dir, mut conn) = setup_test_db();
    seed_session(&conn, "unknown");
    conn.execute("UPDATE sessions SET source = 'unknown'", [])
        .unwrap();
    let root = TempDir::new().unwrap();
    let stats = index_from_dirs(
        &mut conn,
        &IndexOptions {
            force: false,
            claude_dir: root.path(),
            codex_dir: root.path(),
            codex_archived_dir: None,
        },
        true,
    )
    .unwrap();
    assert_eq!(stats.total_sessions, 1);
    assert_eq!(
        collect_strings(&conn, "SELECT session_id FROM sessions"),
        ["unknown"]
    );
}

#[test]
fn rebuild_with_a_missing_codex_root_keeps_its_session_and_embeddings_and_reports_it() {
    let (_dir, mut conn) = setup_test_db();
    let (tmp, claude_dir, _codex_dir) = seed_claude_and_codex(&mut conn);

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

    let gone_codex = tmp.path().join("missing_codex");
    let stats = index_from_dirs(
        &mut conn,
        &IndexOptions {
            force: true,
            claude_dir: &claude_dir,
            codex_dir: &gone_codex,
            codex_archived_dir: None,
        },
        true,
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

#[test]
fn rebuild_with_every_root_missing_keeps_all_sessions_and_embeddings() {
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
            codex_archived_dir: None,
        },
        true,
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

#[test]
fn rebuild_removes_the_session_of_a_log_deleted_under_a_fully_read_root() {
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

    fs::remove_file(claude_dir.join("c.jsonl")).unwrap();
    let stats = index_from_dirs(
        &mut conn,
        &IndexOptions {
            force: true,
            claude_dir: &claude_dir,
            codex_dir: &codex_dir,
            codex_archived_dir: None,
        },
        true,
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
            codex_archived_dir: None,
        },
        true,
    )
    .unwrap();
    assert_eq!(stats.indexed, 1);
    assert_eq!(index_chunks(&mut conn, None).unwrap().chunks_created, 1);

    let stats2 = index_from_dirs(
        &mut conn,
        &IndexOptions {
            force: false,
            claude_dir: &dir_b,
            codex_dir: &empty_dir,
            codex_archived_dir: None,
        },
        true,
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
    assert_eq!(index_chunks(&mut conn, None).unwrap().chunks_created, 0);
    assert_eq!(
        collect_strings(&conn, "SELECT content FROM qa_chunks"),
        ["from dir b"]
    );
}

#[test]
fn model_absent_session_id_collision_preserves_data_until_embedding_is_available() {
    for remove_original in [false, true] {
        let (_dir, mut conn) = setup_test_db();
        let tmp = TempDir::new().unwrap();
        let claude_dir = tmp.path().join("claude");
        let dir_a = claude_dir.join("a");
        let dir_b = claude_dir.join("b");
        fs::create_dir_all(&dir_a).unwrap();
        fs::create_dir_all(&dir_b).unwrap();
        let original = dir_a.join("collision.jsonl");
        fs::write(
            &original,
            concat!(
                r#"{"type":"user","message":{"role":"user","content":"original question"},"timestamp":"2026-03-01T00:00:00Z"}"#,
                "\ninvalid JSON\n",
            ),
        )
        .unwrap();
        let codex_dir = tmp.path().join("codex");
        let opts = IndexOptions {
            force: false,
            claude_dir: &claude_dir,
            codex_dir: &codex_dir,
            codex_archived_dir: None,
        };
        assert_eq!(index_from_dirs(&mut conn, &opts, true).unwrap().indexed, 1);
        assert_eq!(index_chunks(&mut conn, None).unwrap().chunks_created, 1);
        let embedder = MockEmbedder::new();
        assert_eq!(
            embed_recent_chunks(&mut conn, &embedder, 8192, None)
                .unwrap()
                .embedded,
            1
        );

        let snapshot = |conn: &Connection| {
            [
                "SELECT json_array(session_id, file_path, mtime, file_size, chunks_indexed, files_scanned) FROM sessions ORDER BY session_id",
                "SELECT json_array(rowid, session_id, role, text) FROM messages ORDER BY rowid",
                "SELECT json_array(id, session_id, content, timestamp, src_rowid_lo, src_rowid_hi) FROM qa_chunks ORDER BY id",
                "SELECT json_array(rowid, chunk_id, sub_idx, hex(embedding)) FROM vec_chunks ORDER BY rowid",
                "SELECT json_array(session_id, path) FROM session_files ORDER BY session_id, path",
            ]
            .map(|sql| collect_strings(conn, sql))
        };
        seed_session_file(&conn, "collision", "/proj/original.rs");
        let before = snapshot(&conn);
        let replacement = dir_b.join("collision.jsonl");
        fs::write(
            &replacement,
            r#"{"type":"user","message":{"role":"user","content":"replacement question"},"timestamp":"2026-03-02T00:00:00Z"}"#,
        )
        .unwrap();

        assert!(original.is_file());
        if remove_original {
            fs::remove_file(&original).unwrap();
        }
        for _ in 0..2 {
            let stats = index_from_dirs(&mut conn, &opts, false).unwrap();
            assert_eq!(stats.indexed, 0);
            assert_eq!(stats.preserved_embedded, 1);
            assert_eq!(stats.total_sessions, 1);
            assert_eq!(stats.parse_diagnostics.len(), 1);
            assert_eq!(stats.parse_diagnostics[0].file_path, original);
            assert_eq!(stats.parse_diagnostics[0].lines.invalid_json_lines, 1);
            assert_eq!(index_chunks(&mut conn, None).unwrap().sessions_chunked, 0);
            assert_eq!(snapshot(&conn), before);
        }

        let stats = index_from_dirs(&mut conn, &opts, true).unwrap();
        assert_eq!(stats.indexed, 1);
        assert_eq!(stats.preserved_embedded, 0);
        assert_eq!(stats.parse_diagnostics.len(), usize::from(!remove_original));
        assert_eq!(stats.total_sessions, 1);
        assert_eq!(
            collect_strings(&conn, "SELECT file_path FROM sessions"),
            [replacement.to_str().unwrap()]
        );
        assert_eq!(
            collect_strings(&conn, "SELECT text FROM messages ORDER BY rowid"),
            ["replacement question"]
        );
        for table in ["vec_chunks", "session_files"] {
            let count: i64 = conn
                .query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |r| r.get(0))
                .unwrap();
            assert_eq!(count, 0, "old {table} must be removed on replacement");
        }
        assert_eq!(index_chunks(&mut conn, None).unwrap().chunks_created, 0);
        assert_eq!(
            collect_strings(&conn, "SELECT content FROM qa_chunks"),
            ["replacement question"]
        );
        fs::remove_file(&replacement).unwrap();
        if !remove_original {
            fs::remove_file(&original).unwrap();
        }
        let stats = index_from_dirs(&mut conn, &opts, false).unwrap();
        assert_eq!(stats.total_sessions, 0);
        assert!(stats.parse_diagnostics.is_empty());
    }
}

#[test]
fn session_log_nested_ten_directories_deep_is_not_indexed_and_its_root_is_reported_as_incompletely_read()
 {
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
fn symlinked_session_log_is_not_indexed_as_a_second_session() {
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
fn unreadable_directory_entry_marks_the_root_incompletely_read_and_keeps_the_readable_logs() {
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
    let (dir, mut conn) = setup_test_db();
    let tmp = TempDir::new().unwrap();
    let claude_dir = tmp.path().join("claude_projects");
    fs::create_dir_all(&claude_dir).unwrap();
    let path = claude_dir.join("s1.jsonl");
    let assistant = r#"{"type":"assistant","cwd":"/proj","message":{"role":"assistant","content":"hi there"},"timestamp":"2026-03-01T00:00:00Z"}"#;
    fs::write(&path, assistant).unwrap();
    let codex_dir = tmp.path().join("codex_sessions");
    let opts = IndexOptions {
        force: false,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
        codex_archived_dir: None,
    };
    index_from_dirs(&mut conn, &opts, true).unwrap();
    let first = index_chunks(&mut conn, None).unwrap();
    assert_eq!(
        (
            first.chunks_created,
            first.sessions_chunked,
            first.messages_read
        ),
        (0, 1, 1)
    );

    drop(conn);
    let mut conn = open_db(&dir.path().join("test.db")).unwrap();
    assert_eq!(index_from_dirs(&mut conn, &opts, true).unwrap().indexed, 0);
    let unchanged = index_chunks(
        &mut conn,
        Some(&|_, _| panic!("unchanged session processed")),
    )
    .unwrap();
    assert_eq!(
        (
            unchanged.chunks_created,
            unchanged.sessions_chunked,
            unchanged.messages_read,
            unchanged.message_scans,
            unchanged.message_batches
        ),
        (0, 0, 0, 0, 0)
    );

    fs::write(
        &path,
        format!(
            "{assistant}\n{}",
            r#"{"type":"user","message":{"role":"user","content":"hello"}}"#
        ),
    )
    .unwrap();
    conn.execute("UPDATE sessions SET mtime = 0", []).unwrap();
    assert_eq!(index_from_dirs(&mut conn, &opts, true).unwrap().indexed, 1);
    let appended = index_chunks(&mut conn, None).unwrap();
    assert_eq!(
        (
            appended.chunks_created,
            appended.sessions_chunked,
            appended.messages_read
        ),
        (1, 1, 2)
    );
    assert_eq!(
        collect_strings(&conn, "SELECT content FROM qa_chunks"),
        ["hello"]
    );
    assert_eq!(index_chunks(&mut conn, None).unwrap().sessions_chunked, 0);
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
fn recall_claude_dir_takes_priority_over_claude_config_dir() {
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
fn claude_config_dir_projects_is_used_when_recall_claude_dir_is_unset() {
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
fn session_root_that_is_not_an_absolute_path_is_not_read_and_keeps_its_sessions() {
    // `src` exists relative to the package root, where the test runner starts, and holds
    // no session logs: read as a root, it would look like every Claude session was deleted.
    for relative in ["", "src"] {
        let (_dir, mut conn) = setup_test_db();
        let (_tmp, _claude_dir, codex_dir) = seed_claude_and_codex(&mut conn);

        let stats = index_from_dirs(
            &mut conn,
            &IndexOptions {
                force: false,
                claude_dir: Path::new(relative),
                codex_dir: &codex_dir,
                codex_archived_dir: None,
            },
            true,
        )
        .unwrap();

        assert_eq!(
            collect_strings(&conn, "SELECT source FROM sessions ORDER BY source"),
            ["claude", "codex"],
            "{relative:?}"
        );
        assert_eq!(
            stats
                .skipped_roots
                .iter()
                .map(|s| (s.source, s.reason, s.preserved_sessions))
                .collect::<Vec<_>>(),
            [(Source::Claude, SkippedReason::IncompleteEnumeration, 1)],
            "{relative:?}"
        );
    }
}

#[test]
fn recall_codex_dir_replaces_the_default_codex_sessions_dir() {
    let home = Path::new("/fake/home");
    let result = resolve_codex_dir_with(home, |key| match key {
        "RECALL_CODEX_DIR" => Some(OsString::from("/custom/codex/sessions")),
        _ => None,
    });
    assert_eq!(result, PathBuf::from("/custom/codex/sessions"));
}

#[test]
fn codex_sessions_dir_defaults_to_dot_codex_sessions_under_home() {
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
            codex_archived_dir: None,
        },
        true,
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

#[test]
fn test_192e_backfill_populates_rowid_range_count_match() {
    let (_dir, mut conn) = setup_test_db();
    seed_session(&conn, "s1");
    let user_rowid = seed_message(&conn, "user", "how to rotate tokens");
    let assistant_rowid = seed_message(&conn, "assistant", "use a refresh grant");
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

#[test]
fn test_192f_backfill_leaves_null_on_count_mismatch() {
    let (_dir, mut conn) = setup_test_db();
    seed_session(&conn, "s1");
    seed_message(&conn, "user", "single turn");
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

#[test]
fn test_229b_backfill_skips_count_match_with_content_drift() {
    let (_dir, mut conn) = setup_test_db();
    seed_session(&conn, "s1");
    seed_message(&conn, "user", "how do i tune the cache");
    seed_message(&conn, "assistant", "raise the ttl first");
    seed_message(&conn, "assistant", "then widen the shard count");
    seed_legacy_chunk(&conn, 1, "how do i tune the cache\nraise the ttl first");

    let updated = backfill_rowid_ranges(&mut conn).unwrap();
    assert_eq!(
        updated, 0,
        "a count-matching but content-drifted session is skipped, not backfilled"
    );

    let (lo, hi): (Option<i64>, Option<i64>) = conn
        .query_row(
            "SELECT src_rowid_lo, src_rowid_hi FROM qa_chunks WHERE id = 1",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    assert_eq!(
        (lo, hi),
        (None, None),
        "the drifted chunk keeps NULL lo/hi so it resolves via the instr fallback, \
         never inheriting the widened [Q..A2] range over its stale content"
    );
}

#[test]
fn test_229_trailing_assistant_in_multi_turn_is_embedded() {
    let (_dir, mut conn) = setup_test_db();
    seed_session(&conn, "s1");
    seed_message(&conn, "user", "how do i configure the retry backoff");
    seed_message(&conn, "assistant", "first set the base interval");
    seed_message(
        &conn,
        "assistant",
        "then cap it with zylophonemax jitter ceiling",
    );

    index_chunks(&mut conn, None).unwrap();
    let embedder = MockEmbedder::new();
    embed_recent_chunks(&mut conn, &embedder, 100, None).unwrap();

    let embedded: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM qa_chunks c JOIN vec_chunks v ON v.chunk_id = c.id \
             WHERE c.content LIKE '%zylophonemax%'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert!(
        embedded > 0,
        "the trailing assistant's text must reach an embedded qa_chunk so semantic \
         search can find it; orphaning it leaves it FTS-only and vector-invisible"
    );
}

#[test]
fn edit_write_の_file_path_が_session_files_に入る() {
    let (_dir, mut conn) = setup_test_db();
    let tmp = TempDir::new().unwrap();
    let claude_dir = tmp.path().join("claude_projects");
    fs::create_dir_all(&claude_dir).unwrap();
    fs::write(
        claude_dir.join("edit_session.jsonl"),
        concat!(
            r#"{"type":"user","cwd":"/proj","message":{"role":"user","content":"please make the edits"},"timestamp":"2026-03-01T00:00:00Z"}"#,
            "\n",
            r#"{"type":"assistant","cwd":"/proj","message":{"role":"assistant","content":[{"type":"text","text":"doing the edits"},{"type":"tool_use","id":"t1","name":"Edit","input":{"file_path":"/proj/alpha_marker.rs","old_string":"x","new_string":"y"}},{"type":"tool_use","id":"t2","name":"Write","input":{"file_path":"/proj/beta_marker.rs","content":"filecontents"}}]},"timestamp":"2026-03-01T00:00:01Z"}"#,
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
            codex_archived_dir: None,
        },
        true,
    )
    .unwrap();
    index_chunks(&mut conn, None).unwrap();

    let paths = collect_strings(&conn, "SELECT path FROM session_files ORDER BY path");
    assert_eq!(
        paths,
        vec![
            "/proj/alpha_marker.rs".to_owned(),
            "/proj/beta_marker.rs".to_owned(),
        ],
        "Edit and Write file_path values must land in session_files"
    );

    let message_texts = collect_strings(&conn, "SELECT text FROM messages");
    let chunk_contents = collect_strings(&conn, "SELECT content FROM qa_chunks");
    for marker in ["alpha_marker", "beta_marker", "filecontents"] {
        assert!(
            !message_texts.iter().any(|t| t.contains(marker)),
            "tool_use text {marker:?} must not appear in messages"
        );
        assert!(
            !chunk_contents.iter().any(|c| c.contains(marker)),
            "tool_use text {marker:?} must not appear in qa_chunks"
        );
    }
}

#[test]
fn 妥当でない値は捨てられる() {
    let (_dir, mut conn) = setup_test_db();
    let tmp = TempDir::new().unwrap();
    let claude_dir = tmp.path().join("claude_projects");
    fs::create_dir_all(&claude_dir).unwrap();

    let long_path = format!("/proj/{}.rs", "a".repeat(100_000));
    let assistant = serde_json::json!({
        "type": "assistant",
        "cwd": "/proj",
        "message": {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "editing"},
                {"type": "tool_use", "id": "t1", "name": "Edit",
                 "input": {"file_path": "/proj/valid_control.rs", "old_string": "x", "new_string": "y"}},
                {"type": "tool_use", "id": "t2", "name": "Edit",
                 "input": {"file_path": "/proj/bad\nname.rs", "old_string": "x", "new_string": "y"}},
                {"type": "tool_use", "id": "t3", "name": "Write",
                 "input": {"file_path": long_path, "content": "z"}},
                {"type": "tool_use", "id": "t4", "name": "Edit",
                 "input": {"file_path": "", "old_string": "x", "new_string": "y"}}
            ]
        },
        "timestamp": "2026-03-01T00:00:01Z"
    });
    let user_line = r#"{"type":"user","cwd":"/proj","message":{"role":"user","content":"edit these files"},"timestamp":"2026-03-01T00:00:00Z"}"#;
    let contents = format!(
        "{}\n{}\n",
        user_line,
        serde_json::to_string(&assistant).unwrap()
    );
    fs::write(claude_dir.join("filter_session.jsonl"), contents).unwrap();
    let codex_dir = tmp.path().join("codex_sessions");

    index_from_dirs(
        &mut conn,
        &IndexOptions {
            force: false,
            claude_dir: &claude_dir,
            codex_dir: &codex_dir,
            codex_archived_dir: None,
        },
        true,
    )
    .unwrap();

    let paths = collect_strings(&conn, "SELECT path FROM session_files ORDER BY path");
    assert_eq!(
        paths,
        vec!["/proj/valid_control.rs".to_owned()],
        "only the valid file_path is kept; newline / over-length / empty are filtered"
    );
}

#[test]
fn 読めない_jsonl_のセッションはマーカーを立てず次回に再試行する() {
    let (_dir, mut conn) = setup_test_db();
    seed_session(&conn, "gone-session");
    let unmarked_before: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM sessions WHERE files_scanned IS NULL",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(unmarked_before, 1, "precondition: one unmarked session");

    let marked = backfill_session_files(&mut conn, None).unwrap();

    assert_eq!(
        marked, 0,
        "an unreadable session must not be counted as marked"
    );
    let unmarked_after: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM sessions WHERE files_scanned IS NULL",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        unmarked_after, 1,
        "an unreadable JSONL must leave files_scanned NULL so the next index retries"
    );
    let file_rows: i64 = conn
        .query_row("SELECT COUNT(*) FROM session_files", [], |r| r.get(0))
        .unwrap();
    assert_eq!(
        file_rows, 0,
        "an unreadable JSONL yields no session_files rows"
    );

    let diagnostics = load_parse_diagnostics(&conn).unwrap();
    assert_eq!(diagnostics.len(), 1);
    assert!(diagnostics[0].read_error);

    let retried = backfill_session_files(&mut conn, None).unwrap();
    assert_eq!(
        retried, 0,
        "the retry still fails on the vanished file but keeps the session eligible"
    );
}

#[test]
fn 未知_source_と_空_jsonl_のセッションはマーカーを立てて再読しない() {
    let (_dir, mut conn) = setup_test_db();
    conn.execute(
        "INSERT INTO sessions (session_id, source, file_path, project, slug, timestamp, mtime, files_scanned) VALUES ('mystery', 'perplexity', '/f', '/p', 'slug', 0, 0.0, NULL)",
        [],
    )
    .unwrap();
    let tmp = TempDir::new().unwrap();
    let empty = tmp.path().join("empty-session.jsonl");
    fs::write(&empty, "").unwrap();
    conn.execute(
        "INSERT INTO sessions (session_id, source, file_path, project, slug, timestamp, mtime, files_scanned) VALUES ('zero', 'claude', ?1, '/p', 'slug', 0, 0.0, NULL)",
        [empty.to_string_lossy().to_string()],
    )
    .unwrap();

    let marked = backfill_session_files(&mut conn, None).unwrap();

    assert_eq!(marked, 2, "both terminal cases must be counted as marked");
    let unmarked: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM sessions WHERE files_scanned IS NULL",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        unmarked, 0,
        "terminal cases must not stay eligible for re-read"
    );
    let file_rows: i64 = conn
        .query_row("SELECT COUNT(*) FROM session_files", [], |r| r.get(0))
        .unwrap();
    assert_eq!(
        file_rows, 0,
        "neither terminal case yields session_files rows"
    );
}

#[test]
fn chunk_completion_rolls_back_with_chunks_and_retries() {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    for interrupt in [false, true] {
        let (_dir, mut conn) = setup_test_db();
        seed_session(&conn, "empty");
        seed_session(&conn, "s1");
        seed_message(&conn, "user", "question");
        let (observer, lines) = recording();
        if !interrupt {
            conn.execute_batch(
                "CREATE TEMP TRIGGER fail_completion BEFORE UPDATE OF chunks_indexed ON sessions
                 WHEN NEW.session_id = 's1'
                 BEGIN SELECT RAISE(ABORT, 'injected completion failure'); END;",
            )
            .unwrap();
            let error = index_chunks_observed(&mut conn, None, &observer)
                .err()
                .unwrap();
            assert!(error.to_string().contains("injected completion failure"));
        } else {
            assert!(
                catch_unwind(AssertUnwindSafe(|| {
                    index_chunks_observed(
                        &mut conn,
                        Some(&|done, _| {
                            if done == 2 {
                                panic!("injected interruption");
                            }
                        }),
                        &observer,
                    )
                    .unwrap();
                }))
                .is_err()
            );
        }
        assert_eq!(
            conn.query_row("SELECT count(*) FROM qa_chunks", [], |r| r.get::<_, i64>(0))
                .unwrap(),
            0
        );
        assert_eq!(
            conn.query_row(
                "SELECT count(*) FROM sessions WHERE chunks_indexed IS NOT NULL",
                [],
                |r| r.get::<_, i64>(0)
            )
            .unwrap(),
            0
        );
        assert_eq!(
            observer.snapshot()["counts"]["sessions_chunked_committed"],
            0
        );
        assert!(
            lines
                .borrow()
                .iter()
                .any(|line| line.contains("chunking: unfinished"))
        );
        assert!(
            !lines
                .borrow()
                .iter()
                .any(|line| line.contains("chunking: committed"))
        );
        conn.execute_batch("DROP TRIGGER IF EXISTS fail_completion;")
            .unwrap();
        let retried = index_chunks(&mut conn, None).unwrap();
        assert_eq!((retried.chunks_created, retried.sessions_chunked), (1, 2));
        assert_eq!(index_chunks(&mut conn, None).unwrap().sessions_chunked, 0);
    }
}

#[test]
fn chunk_pass_reports_stage_lines_counts_and_rows_with_and_without_pending_sessions() {
    let (_dir, mut conn) = setup_test_db();

    let (observer, lines) = recording();
    let stats = index_chunks_observed(&mut conn, None, &observer).unwrap();
    assert_eq!(
        (
            stats.chunks_created,
            stats.sessions_chunked,
            stats.message_batches,
            stats.message_scans,
            stats.messages_read
        ),
        (0, 0, 0, 0, 0)
    );
    assert_eq!(
        stable_lines(&lines.borrow()),
        [
            "index: chunking: started (not yet complete)",
            "index: chunking: no pending sessions"
        ]
    );
    assert_eq!(
        observer.snapshot()["counts"],
        serde_json::json!({
            "chunks_created_committed": 0,
            "sessions_chunk_pending": 0,
            "sessions_chunk_remaining": 0,
            "sessions_chunked_committed": 0,
            "sessions_chunked_empty": 0,
        })
    );

    seed_session(&conn, "empty");
    seed_session(&conn, "empty2");
    seed_session(&conn, "s1");
    conn.execute(
        "UPDATE sessions SET timestamp = 1234 WHERE session_id = 's1'",
        [],
    )
    .unwrap();
    let question = seed_message(&conn, "user", "question");
    seed_message(&conn, "unknown-role", "dropped text");
    let (observer, lines) = recording();
    let progress = RefCell::new(Vec::new());
    let on_progress = |done, total| progress.borrow_mut().push((done, total));

    let stats = index_chunks_observed(&mut conn, Some(&on_progress), &observer).unwrap();

    assert_eq!(
        (
            stats.chunks_created,
            stats.sessions_chunked,
            stats.message_batches,
            stats.message_scans,
            stats.messages_read
        ),
        (1, 3, 1, 1, 2)
    );
    assert_eq!(*progress.borrow(), [(1, 3), (2, 3), (3, 3)]);
    assert_eq!(
        stable_lines(&lines.borrow()),
        [
            "index: chunking: started (not yet complete)",
            "index: chunking: processed=3/3 unprocessed=0 committed=0",
            "index: chunking: committed"
        ]
    );
    assert_eq!(
        observer.snapshot()["counts"],
        serde_json::json!({
            "chunks_created_committed": 1,
            "sessions_chunk_pending": 3,
            "sessions_chunk_remaining": 0,
            "sessions_chunked_committed": 3,
            "sessions_chunked_empty": 2,
        })
    );
    let chunk: (String, String, i64, i64, i64) = conn
        .query_row(
            "SELECT session_id, content, timestamp, src_rowid_lo, src_rowid_hi FROM qa_chunks",
            [],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)),
        )
        .unwrap();
    assert_eq!(
        chunk,
        (
            "s1".to_owned(),
            "question".to_owned(),
            1234,
            question,
            question
        )
    );
    assert_eq!(
        collect_strings(
            &conn,
            "SELECT session_id || '=' || chunks_indexed FROM sessions ORDER BY session_id"
        ),
        ["empty=0", "empty2=0", "s1=0"]
    );
    assert!(
        !conn
            .prepare("SELECT 1 FROM temp.sqlite_master WHERE name = 'chunk_message_rows'")
            .unwrap()
            .exists([])
            .unwrap(),
        "the staging table must not outlive the pass"
    );
}

#[test]
fn chunk_batches_limit_session_count_and_source_bytes() {
    let small = vec![("s".to_owned(), None, 1); CHUNK_BATCH_SESSIONS + 1];
    assert_eq!(chunk_batch_len(&small), 64);
    let sizes = [
        CHUNK_BATCH_BYTES / 2,
        CHUNK_BATCH_BYTES / 2,
        1,
        CHUNK_BATCH_BYTES + 1,
        0,
    ];
    let sessions: Vec<_> = sizes
        .iter()
        .map(|size| ("s".to_owned(), None, *size))
        .collect();
    assert_eq!(
        chunk_batch_len(&sessions),
        2,
        "exact byte budget is allowed"
    );
    assert_eq!(
        chunk_batch_len(&sessions[2..]),
        1,
        "do not add an oversized session"
    );
    assert_eq!(
        chunk_batch_len(&sessions[3..]),
        1,
        "oversized conversation runs alone"
    );
    assert_eq!(
        chunk_batch_len(&sessions[4..]),
        1,
        "empty conversation still finishes"
    );
}

#[test]
fn chunk_batches_preserve_interleaved_sessions_order_and_search_ranges() {
    let (_dir, mut conn) = setup_test_db();
    for i in 0..65 {
        seed_session(&conn, &format!("s{i:03}"));
    }
    for (turn, role) in [
        (0, "user"),
        (1, "system"),
        (2, "assistant"),
        (3, "assistant"),
    ] {
        for i in (0..65).rev() {
            conn.execute(
                "INSERT INTO messages (session_id, role, text) VALUES (?1, ?2, ?3)",
                rusqlite::params![format!("s{i:03}"), role, format!("session {i} turn {turn}")],
            )
            .unwrap();
        }
    }
    let stats = index_chunks(&mut conn, None).unwrap();
    assert_eq!(
        (
            stats.chunks_created,
            stats.sessions_chunked,
            stats.message_batches,
            stats.messages_read,
            stats.message_scans
        ),
        (65, 65, 2, 260, 1)
    );
    let mut stmt = conn.prepare("SELECT session_id, content, src_rowid_lo, src_rowid_hi FROM qa_chunks ORDER BY session_id").unwrap();
    let actual: Vec<(String, String, i64, i64)> = stmt
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))
        .unwrap()
        .map(Result::unwrap)
        .collect();
    for (i, (sid, content, lo, hi)) in (0_i64..65).zip(actual) {
        assert_eq!(sid, format!("s{i:03}"));
        assert_eq!(
            content,
            format!("session {i} turn 0\nsession {i} turn 2\nsession {i} turn 3")
        );
        assert_eq!((lo, hi), (65 - i, 260 - i));
    }
    conn.execute_batch("CREATE TEMP TABLE chunk_message_rows (session_id TEXT, message_rowid INTEGER, text_bytes INTEGER, PRIMARY KEY(session_id, message_rowid)) WITHOUT ROWID;").unwrap();
    let stage_plan: Vec<String> = conn
        .prepare(&format!("EXPLAIN QUERY PLAN {STAGE_CHUNK_MESSAGES}"))
        .unwrap()
        .query_map([], |r| r.get(3))
        .unwrap()
        .map(Result::unwrap)
        .collect();
    assert!(
        stage_plan[0].contains("SCAN m VIRTUAL TABLE"),
        "FTS must be the single outer scan: {stage_plan:?}"
    );
    assert!(
        stage_plan[1].contains("SEARCH s USING INDEX"),
        "each message probes the session primary key: {stage_plan:?}"
    );
    let plan: Vec<String> = conn
        .prepare(&format!("EXPLAIN QUERY PLAN {READ_CHUNK_BATCH}"))
        .unwrap()
        .query_map(["s000", "s064"], |r| r.get(3))
        .unwrap()
        .map(Result::unwrap)
        .collect();
    assert!(
        plan.iter()
            .any(|line| line.contains("VIRTUAL TABLE INDEX") && line.contains('=')),
        "expected FTS rowid equality lookup: {plan:?}"
    );
}

#[test]
fn parse_diagnostics_survive_freshness_preservation_and_root_outages_until_repaired() {
    let (db_dir, mut conn) = setup_test_db();
    let tmp = TempDir::new().unwrap();
    let claude_dir = tmp.path().join("claude");
    let codex_dir = tmp.path().join("codex");
    fs::create_dir(&claude_dir).unwrap();
    let path = claude_dir.join("partial.jsonl");
    let valid = "{\"type\":\"user\",\"message\":\"retained question\"}\n";
    fs::write(&path, format!("{valid}{{broken\n")).unwrap();
    let opts = IndexOptions {
        force: false,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
        codex_archived_dir: None,
    };
    let stats = index_from_dirs(&mut conn, &opts, true).unwrap();
    assert_eq!(stats.indexed, 1);
    assert_eq!(stats.parse_diagnostics[0].lines.invalid_json_lines, 1);
    index_chunks(&mut conn, None).unwrap();
    embed_recent_chunks(&mut conn, &MockEmbedder::new(), 100, None).unwrap();
    let chunks = collect_strings(
        &conn,
        "SELECT json_array(id, content, generation) FROM qa_chunks",
    );
    let vectors = collect_strings(&conn, "SELECT hex(embedding) FROM vec_chunks");
    drop(conn);
    let mut conn = open_db(&db_dir.path().join("test.db")).unwrap();
    let stats = index_from_dirs(&mut conn, &opts, true).unwrap();
    assert_eq!(stats.indexed, 0);
    assert_eq!(stats.parse_diagnostics[0].lines.invalid_json_lines, 1);
    assert_eq!(
        collect_strings(
            &conn,
            "SELECT json_array(id, content, generation) FROM qa_chunks"
        ),
        chunks
    );

    fs::write(&path, valid).unwrap();
    let stats = index_from_dirs(&mut conn, &opts, false).unwrap();
    assert_eq!(stats.preserved_embedded, 1);
    assert_eq!(stats.parse_diagnostics[0].lines.invalid_json_lines, 1);
    assert_eq!(
        collect_strings(&conn, "SELECT hex(embedding) FROM vec_chunks"),
        vectors
    );

    let hidden = tmp.path().join("hidden");
    fs::rename(&claude_dir, &hidden).unwrap();
    let stats = index_from_dirs(&mut conn, &opts, true).unwrap();
    assert_eq!(stats.skipped_roots.len(), 1);
    assert_eq!(stats.parse_diagnostics.len(), 1);
    fs::rename(&hidden, &claude_dir).unwrap();
    let stats = index_from_dirs(&mut conn, &opts, true).unwrap();
    assert_eq!(stats.indexed, 1);
    assert!(stats.parse_diagnostics.is_empty());
    assert_eq!(
        collect_strings(&conn, "SELECT text FROM messages"),
        ["retained question"]
    );

    let tail = claude_dir.join("tail.jsonl");
    fs::write(
        &tail,
        "{\"type\":\"user\",\"message\":\"completed by append",
    )
    .unwrap();
    let stats = index_from_dirs(&mut conn, &opts, true).unwrap();
    assert_eq!(stats.parse_diagnostics[0].lines.incomplete_tail_lines, 1);
    assert_eq!(stats.total_sessions, 1);
    {
        use std::io::Write;
        let mut file = fs::OpenOptions::new().append(true).open(&tail).unwrap();
        file.write_all(b"\"}\n").unwrap();
    }
    let stats = index_from_dirs(&mut conn, &opts, true).unwrap();
    assert!(stats.parse_diagnostics.is_empty());
    assert_eq!(stats.total_sessions, 2);
    assert!(
        collect_strings(&conn, "SELECT text FROM messages")
            .contains(&"completed by append".to_owned())
    );
    fs::remove_file(tail).unwrap();

    let empty = claude_dir.join("no-messages.jsonl");
    fs::write(&empty, "invalid\n").unwrap();
    let stats = index_from_dirs(&mut conn, &opts, true).unwrap();
    assert_eq!(stats.parse_diagnostics.len(), 1);
    assert_eq!(stats.total_sessions, 1);
    fs::remove_file(empty).unwrap();
    assert!(
        index_from_dirs(&mut conn, &opts, true)
            .unwrap()
            .parse_diagnostics
            .is_empty()
    );
}

#[test]
fn read_failure_retains_body_and_line_diagnostics_then_retries_unchanged_metadata() {
    let (_dir, mut conn) = setup_test_db();
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("s.jsonl");
    fs::write(&path, "{\"type\":\"user\",\"message\":\"kept\"}\ninvalid\n").unwrap();
    let missing = tmp.path().join("no-codex");
    let opts = IndexOptions {
        force: false,
        claude_dir: tmp.path(),
        codex_dir: &missing,
        codex_archived_dir: None,
    };
    index_from_dirs(&mut conn, &opts, true).unwrap();
    let existing = load_existing_sessions(&conn).unwrap();
    let tx = conn.transaction().unwrap();
    let ctx = IndexContext {
        tx: &tx,
        existing: &existing,
        force: true,
        embedded_sessions: None,
    };
    index_file_with_parser(&ctx, &path, &Source::Claude, |_, _| {
        Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "private content must not escape",
        )
        .into())
    })
    .unwrap();
    tx.commit().unwrap();
    let d = load_parse_diagnostics(&conn).unwrap();
    assert!(d[0].read_error);
    assert_eq!(d[0].lines.invalid_json_lines, 1);
    assert_eq!(
        collect_strings(&conn, "SELECT text FROM messages"),
        ["kept"]
    );
    let stats = index_from_dirs(&mut conn, &opts, true).unwrap();
    assert_eq!(stats.indexed, 1);
    assert!(!stats.parse_diagnostics[0].read_error);
    assert_eq!(stats.parse_diagnostics[0].lines.invalid_json_lines, 1);
}

#[cfg(unix)]
#[test]
fn colliding_display_paths_keep_independent_diagnostics_until_each_file_is_resolved() {
    use std::os::unix::ffi::OsStringExt;

    fn observe(conn: &mut Connection, paths: &[PathBuf]) -> Vec<FileParseDiagnostics> {
        let existing = load_existing_sessions(conn).unwrap();
        let tx = conn.transaction().unwrap();
        let ctx = IndexContext {
            tx: &tx,
            existing: &existing,
            force: false,
            embedded_sessions: None,
        };
        let sources: Vec<_> = paths.iter().cloned().map(|p| (p, Source::Claude)).collect();
        index_all(&ctx, &sources, &Observer::default()).unwrap();
        let source_paths = paths.iter().map(PathBuf::as_path).collect();
        let diagnostics = cleanup_parse_diagnostics(
            &tx,
            &source_paths,
            &RemovalGate::fully_read([Source::Claude]),
        )
        .unwrap();
        tx.commit().unwrap();
        diagnostics
    }

    let (db_dir, mut conn) = setup_test_db();
    let tmp = TempDir::new().unwrap();
    let paths = [
        tmp.path()
            .join(OsString::from_vec(b"bad-\xff.jsonl".to_vec())),
        tmp.path()
            .join(OsString::from_vec(b"bad-\xfe.jsonl".to_vec())),
        tmp.path().join("bad-\u{fffd}.jsonl"),
    ];
    fs::write(&paths[2], "invalid JSON\n").unwrap();
    let diagnostics = observe(&mut conn, &paths);
    assert_eq!(diagnostics.len(), 3);
    assert_eq!(diagnostics.iter().filter(|d| d.read_error).count(), 2);
    assert!(
        diagnostics
            .iter()
            .all(|d| { d.display_path() == paths[2].to_str().unwrap() })
    );
    assert_eq!(
        diagnostics
            .iter()
            .map(|d| d.lines.invalid_json_lines)
            .sum::<i64>(),
        1
    );

    drop(conn);
    let mut conn = open_db(&db_dir.path().join("test.db")).unwrap();
    fs::write(&paths[2], "{\"type\":\"user\",\"message\":\"readable\"}\n").unwrap();
    for _ in 0..2 {
        let diagnostics = observe(&mut conn, &paths);

        assert_eq!(diagnostics.len(), 2);
        assert!(diagnostics.iter().all(|d| d.read_error));
        let reported: HashSet<_> = diagnostics.iter().map(|d| &d.file_path).collect();
        assert_eq!(reported, HashSet::from([&paths[0], &paths[1]]));
    }
    let diagnostics = observe(&mut conn, &paths[1..]);
    assert_eq!(diagnostics.len(), 1);
    assert_eq!(diagnostics[0].file_path, paths[1]);
    assert!(diagnostics[0].read_error);
    let diagnostics = observe(&mut conn, &paths[2..]);
    assert!(diagnostics.is_empty());
    assert_eq!(
        collect_strings(&conn, "SELECT text FROM messages"),
        ["readable"]
    );
}

#[test]
fn failed_fts_transaction_never_reports_saved_files() {
    let (_dir, mut conn) = setup_test_db();
    let source = TempDir::new().unwrap();
    fs::write(
        source.path().join("s.jsonl"),
        r#"{"type":"user","message":{"role":"user","content":"question"}}"#,
    )
    .unwrap();
    conn.execute_batch("CREATE TRIGGER fail_session BEFORE INSERT ON sessions BEGIN SELECT RAISE(ABORT, 'injected FTS failure'); END;").unwrap();
    let (observer, lines) = recording();
    let absent = source.path().join("absent");
    let opts = IndexOptions {
        force: false,
        claude_dir: source.path(),
        codex_dir: &absent,
        codex_archived_dir: None,
    };
    let error = index_from_dirs_observed(&mut conn, &opts, true, &observer)
        .err()
        .unwrap();
    assert!(error.to_string().contains("injected FTS failure"));
    assert_eq!(observer.snapshot()["counts"]["files_updated_committed"], 0);
    assert!(collect_strings(&conn, "SELECT session_id FROM sessions").is_empty());
    assert!(
        !lines
            .borrow()
            .iter()
            .any(|line| line.contains("fts_transaction: committed"))
    );
    assert!(
        lines
            .borrow()
            .iter()
            .any(|line| line.contains("fts_transaction: unfinished"))
    );
    conn.execute_batch("DROP TRIGGER fail_session;").unwrap();
    assert_eq!(index_from_dirs(&mut conn, &opts, true).unwrap().indexed, 1);
}

#[test]
fn file_outcome_counts_partition_empty_failed_and_deferred_work() {
    let (_dir, mut conn) = setup_test_db();
    let source = TempDir::new().unwrap();
    let body = r#"{"type":"user","message":{"role":"user","content":"question"}}"#;
    let absent = source.path().join("absent");
    let opts = IndexOptions {
        force: false,
        claude_dir: source.path(),
        codex_dir: &absent,
        codex_archived_dir: None,
    };
    let observer = Observer::default();
    index_from_dirs_observed(&mut conn, &opts, true, &observer).unwrap();
    let counts = observer.snapshot()["counts"].clone();
    for key in [
        "files_discovered",
        "files_updated",
        "files_unchanged",
        "files_empty",
        "files_failed",
        "files_deferred",
        "files_remaining",
        "files_updated_committed",
    ] {
        assert_eq!(counts[key], 0, "empty source: {key}");
    }

    fs::write(source.path().join("good.jsonl"), body).unwrap();
    fs::write(source.path().join("empty.jsonl"), "\n").unwrap();
    fs::write(source.path().join("broken.jsonl"), "{broken}\n").unwrap();
    for pass in 0..3 {
        let observer = Observer::default();
        index_from_dirs_observed(&mut conn, &opts, pass != 2, &observer).unwrap();
        let counts = observer.snapshot()["counts"].clone();
        assert_eq!(counts["files_discovered"], 3);
        assert_eq!(counts["files_empty"], 1);
        assert_eq!(counts["files_failed"], 1);
        assert_eq!(counts["files_updated"], usize::from(pass == 0));
        assert_eq!(counts["files_unchanged"], usize::from(pass == 1));
        assert_eq!(counts["files_deferred"], usize::from(pass == 2));
        assert_eq!(counts["files_remaining"], 0);
        index_chunks(&mut conn, None).unwrap();
        if pass == 1 {
            embed_recent_chunks(&mut conn, &MockEmbedder::new(), usize::MAX, None).unwrap();
            fs::write(
                source.path().join("good.jsonl"),
                format!("{body}\n{body}\n"),
            )
            .unwrap();
        }
    }

    let protected = chunk_vectors(&conn);
    assert_eq!(protected.len(), 1);
    let unembedded = source.path().join("unembedded.jsonl");
    write_reuse_conversation(&unembedded, &[("user", "pending question")]);
    index_from_dirs(&mut conn, &opts, false).unwrap();
    index_chunks(&mut conn, None).unwrap();
    let before = collect_strings(
        &conn,
        "SELECT json_array(id, generation, content) FROM qa_chunks WHERE session_id = 'unembedded'",
    );
    write_reuse_conversation(
        &unembedded,
        &[("user", "pending question"), ("user", "appended question")],
    );
    let observer = Observer::default();
    let stats = index_from_dirs_observed(&mut conn, &opts, false, &observer).unwrap();
    assert_eq!(stats.indexed, 1);
    assert_eq!(stats.preserved_embedded, 1);
    let counts = observer.snapshot()["counts"].clone();
    assert_eq!(counts["chunks_matched_committed"], 1);
    assert_eq!(counts["embeddings_reused_committed"], 0);
    assert_eq!(counts["chunks_reconciled_created_committed"], 1);
    assert_eq!(index_chunks(&mut conn, None).unwrap().sessions_chunked, 0);
    assert_eq!(chunk_vectors(&conn), protected);
    let after = collect_strings(
        &conn,
        "SELECT json_array(id, generation, content) FROM qa_chunks WHERE session_id = 'unembedded' ORDER BY id",
    );
    assert_eq!(after.len(), 2);
    assert_eq!(after[..1], before);
    assert_eq!(
        collect_strings(
            &conn,
            "SELECT text FROM messages WHERE session_id = 'unembedded' ORDER BY rowid"
        ),
        ["pending question", "appended question"]
    );
}

fn write_reuse_conversation(path: &Path, messages: &[(&str, &str)]) {
    let text = messages
        .iter()
        .map(|(role, text)| {
            serde_json::json!({"type": role, "message": {"role": role, "content": text}})
                .to_string()
                + "\n"
        })
        .collect::<String>();
    fs::write(path, text).unwrap();
}

fn chunk_vectors(conn: &Connection) -> Vec<String> {
    collect_strings(
        conn,
        "SELECT json_array(c.content, v.sub_idx, hex(v.embedding)) FROM qa_chunks c JOIN vec_chunks v ON c.id = v.chunk_id ORDER BY c.content, v.sub_idx, hex(v.embedding)",
    )
}

#[test]
fn a_session_indexed_after_enumeration_survives_orphan_cleanup() {
    use rusqlite::{Error as SqliteError, ErrorCode};
    use std::cell::{Cell, RefCell};
    use std::rc::Rc;

    for embed_capable in [true, false] {
        let (db_dir, mut conn) = setup_test_db();
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().to_path_buf();
        let absent = root.join("absent");
        let opts = IndexOptions {
            force: false,
            claude_dir: &root,
            codex_dir: &absent,
            codex_archived_dir: None,
        };
        let writer = open_db(&db_dir.path().join("test.db")).unwrap();
        writer.busy_timeout(Duration::ZERO).unwrap();
        let writer = RefCell::new(writer);
        let writer_root = root.clone();
        let saved = Rc::new(RefCell::new(None));
        let writer_saved = Rc::clone(&saved);
        let index_new = Rc::new(move || -> Result<()> {
            let mut writer = writer.borrow_mut();
            let absent = writer_root.join("absent");
            let opts = IndexOptions {
                force: false,
                claude_dir: &writer_root,
                codex_dir: &absent,
                codex_archived_dir: None,
            };
            index_from_dirs(&mut writer, &opts, true)?;
            index_chunks(&mut writer, None)?;
            assert_eq!(
                embed_recent_chunks(&mut writer, &MockEmbedder::new(), 100, None)?.embedded,
                1
            );
            *writer_saved.borrow_mut() = Some(chunk_vectors(&writer));
            Ok(())
        });
        let attempted = Rc::new(Cell::new(false));
        let observer_attempted = Rc::clone(&attempted);
        let observer_index = Rc::clone(&index_new);
        let path = root.join("new.jsonl");
        let observer = Observer::with_reporter(move |line| {
            if line.starts_with("index: enumeration: complete") {
                assert!(!observer_attempted.replace(true));
                write_reuse_conversation(
                    &path,
                    &[("user", "new question"), ("assistant", "new answer")],
                );
                if let Err(error) = observer_index() {
                    assert_eq!(
                        error
                            .downcast_ref::<SqliteError>()
                            .and_then(SqliteError::sqlite_error_code),
                        Some(ErrorCode::DatabaseBusy),
                        "only writer contention permits a retry: {error:#}"
                    );
                }
            }
        });

        index_from_dirs_observed(&mut conn, &opts, embed_capable, &observer).unwrap();
        assert!(attempted.get());
        if saved.borrow().is_none() {
            index_new().unwrap();
        }
        assert_eq!(
            collect_strings(&conn, "SELECT session_id FROM sessions"),
            ["new"]
        );
        assert_eq!(
            collect_strings(&conn, "SELECT text FROM messages ORDER BY rowid"),
            ["new question", "new answer"]
        );
        assert_eq!(chunk_vectors(&conn).len(), 1);
        assert_eq!(chunk_vectors(&conn), *saved.borrow().as_ref().unwrap());
    }
}

#[test]
fn unchanged_chunks_reuse_vectors_and_changed_groups_match_a_fresh_rebuild() {
    let original = [
        ("user", "duplicate"),
        ("assistant", "same answer"),
        ("user", "duplicate"),
        ("assistant", "same answer"),
        ("user", "tail question"),
        ("assistant", "tail answer"),
    ];
    for scenario in [
        "append",
        "assistant",
        "mtime",
        "truncate",
        "partial-duplicate",
        "replace",
        "rules",
        "pending-rules",
        "pending-old-rules",
        "pending-current-rules",
        "rebuild",
    ] {
        let (_db_dir, mut conn) = setup_test_db();
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("claude");
        fs::create_dir(&root).unwrap();
        let path = root.join("reuse.jsonl");
        let absent = tmp.path().join("absent");
        let mut opts = IndexOptions {
            force: false,
            claude_dir: &root,
            codex_dir: &absent,
            codex_archived_dir: None,
        };
        write_reuse_conversation(&path, &original);
        index_from_dirs(&mut conn, &opts, true).unwrap();
        if scenario == "pending-current-rules" {
            let offline = tmp.path().join("offline");
            fs::rename(&root, &offline).unwrap();
            let mut resumed = open_db(&_db_dir.path().join("test.db")).unwrap();
            assert_eq!(
                index_from_dirs(&mut resumed, &opts, true).unwrap().indexed,
                0
            );
            assert_eq!(
                index_chunks(&mut resumed, None).unwrap().sessions_chunked,
                1
            );
            fs::rename(&offline, &root).unwrap();
            assert_eq!(index_from_dirs(&mut conn, &opts, true).unwrap().indexed, 0);
        }
        index_chunks(&mut conn, None).unwrap();
        let model = MockEmbedder::new();
        assert_eq!(
            embed_recent_chunks(&mut conn, &model, 100, None)
                .unwrap()
                .embedded,
            3
        );
        if scenario == "partial-duplicate" {
            conn.execute(
                "DELETE FROM vec_chunks WHERE chunk_id = (SELECT min(id) FROM qa_chunks)",
                [],
            )
            .unwrap();
        }
        let before = collect_strings(
            &conn,
            "SELECT json_array(c.id, c.generation, c.content, v.rowid, hex(v.embedding)) FROM qa_chunks c JOIN vec_chunks v ON c.id = v.chunk_id ORDER BY c.id",
        );
        conn.execute(
            "INSERT INTO messages (session_id, role, text) VALUES ('other', 'user', 'unrelated')",
            [],
        )
        .unwrap();
        let mut changed = original.to_vec();
        let (reused, inferred) = match scenario {
            "append" => {
                changed.extend([
                    ("user", "appended question"),
                    ("assistant", "appended answer"),
                ]);
                (3, 1)
            }
            "assistant" => {
                changed.push(("assistant", "continuation needle"));
                (2, 1)
            }
            "mtime" | "pending-current-rules" => (3, 0),
            "truncate" | "partial-duplicate" => {
                changed.truncate(2);
                (1, 0)
            }
            "replace" => {
                changed = vec![
                    ("user", "tail question"),
                    ("assistant", "tail answer"),
                    ("user", "replacement needle"),
                ];
                (1, 1)
            }
            "rules" => {
                conn.execute("UPDATE sessions SET chunks_indexed = 99", [])
                    .unwrap();
                (0, 3)
            }
            "pending-rules" | "pending-old-rules" => {
                conn.execute_batch("DELETE FROM vec_chunks; DELETE FROM qa_chunks; UPDATE sessions SET chunks_indexed = NULL; UPDATE messages SET text = 'obsolete parser output' WHERE session_id = 'reuse';").unwrap();
                if scenario == "pending-old-rules" {
                    conn.execute("UPDATE sessions SET chunks_indexed = -99", [])
                        .unwrap();
                }
                let stamp = resolve_file_stamp(&path);
                let offline = tmp.path().join("offline");
                fs::rename(&root, &offline).unwrap();
                let stats = index_from_dirs(&mut conn, &opts, true).unwrap();
                assert_eq!(stats.indexed, 0);
                assert_eq!(stats.total_sessions, 1);
                assert_eq!(stats.skipped_roots[0].preserved_sessions, 1);
                assert_eq!(index_chunks(&mut conn, None).unwrap().sessions_chunked, 1);
                assert_eq!(index_chunks(&mut conn, None).unwrap().sessions_chunked, 0);
                embed_recent_chunks(&mut conn, &MockEmbedder::new(), 100, None).unwrap();
                fs::rename(&offline, &root).unwrap();
                assert!(resolve_file_stamp(&path) == stamp);
                (0, 3)
            }
            "rebuild" => {
                opts.force = true;
                (0, 3)
            }
            _ => unreachable!(),
        };
        if !matches!(scenario, "rules" | "pending-rules" | "pending-old-rules") {
            write_reuse_conversation(&path, &changed);
            conn.execute("UPDATE sessions SET mtime = 0", []).unwrap();
        }
        let observer = Observer::default();
        assert_eq!(
            index_from_dirs_observed(&mut conn, &opts, true, &observer)
                .unwrap()
                .indexed,
            1,
            "{scenario}"
        );
        assert_eq!(
            observer.snapshot()["counts"]["embeddings_reused_committed"],
            reused,
            "{scenario}"
        );
        index_chunks(&mut conn, None).unwrap();
        let retained = collect_strings(
            &conn,
            "SELECT json_array(c.id, c.generation, c.content, v.rowid, hex(v.embedding)) FROM qa_chunks c JOIN vec_chunks v ON c.id = v.chunk_id ORDER BY c.id",
        );
        assert_eq!(retained.len(), reused as usize, "{scenario}");
        assert!(
            retained.iter().all(|row| before.contains(row)),
            "{scenario}: reused vectors changed"
        );
        assert_eq!(
            embed_recent_chunks(&mut conn, &model, 100, None)
                .unwrap()
                .embedded,
            inferred as usize
        );
        assert_eq!(
            model.calls(),
            3 + inferred as usize,
            "{scenario}: redundant inference"
        );
        let tx = conn.transaction().unwrap();
        let messages = read_session_messages(&tx, "reuse").unwrap();
        let mut expected: Vec<_> = chunker::chunk_messages("reuse", &messages, None)
            .into_iter()
            .map(|c| (c.content, c.src_rowid_lo, c.src_rowid_hi))
            .collect();
        let mut actual: Vec<(String, i64, i64)> = tx
            .prepare("SELECT content, src_rowid_lo, src_rowid_hi FROM qa_chunks")
            .unwrap()
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
            .unwrap()
            .map(Result::unwrap)
            .collect();
        expected.sort();
        actual.sort();
        assert_eq!(
            actual, expected,
            "{scenario}: incorrect source rowid mapping"
        );
        tx.commit().unwrap();
        if scenario == "assistant" {
            let results = search(&conn, "continuation", &SearchOptions::default()).unwrap();
            assert_eq!(results.len(), 1);
            assert!(results[0].excerpt.contains("continuation needle"));
            assert!(results[0].chunk_id.is_some());
        }
        let expected_vectors = chunk_vectors(&conn);
        opts.force = true;
        index_from_dirs(&mut conn, &opts, true).unwrap();
        index_chunks(&mut conn, None).unwrap();
        embed_recent_chunks(&mut conn, &MockEmbedder::new(), 100, None).unwrap();
        assert_eq!(
            chunk_vectors(&conn),
            expected_vectors,
            "{scenario}: differs from rebuild"
        );
    }
}

#[test]
fn reconciliation_rollback_and_failed_inference_preserve_reusable_work_for_resume() {
    use crate::embedder::{EmbedOptions, embed_chunks, pending_chunks};
    let (db_dir, mut conn) = setup_test_db();
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("reuse.jsonl");
    let absent = tmp.path().join("absent");
    let opts = IndexOptions {
        force: false,
        claude_dir: tmp.path(),
        codex_dir: &absent,
        codex_archived_dir: None,
    };
    let mut messages = vec![
        ("user", "stable"),
        ("assistant", "answer"),
        ("user", "tail"),
    ];
    write_reuse_conversation(&path, &messages);
    index_from_dirs(&mut conn, &opts, true).unwrap();
    index_chunks(&mut conn, None).unwrap();
    let old_pending = pending_chunks(&conn, 100).unwrap();
    let stable = conn
        .query_row(
            "SELECT id FROM qa_chunks WHERE content = 'stable\nanswer'",
            [],
            |r| r.get::<_, i64>(0),
        )
        .unwrap();
    let bytes = f32_as_bytes(&MockEmbedder::deterministic_vector("stable\nanswer")).to_vec();
    conn.execute(
        "INSERT INTO vec_chunks (chunk_id, sub_idx, embedding) VALUES (?1, 0, ?2)",
        rusqlite::params![stable, bytes],
    )
    .unwrap();
    let before = chunk_vectors(&conn);
    messages.push(("assistant", "new tail"));
    write_reuse_conversation(&path, &messages);
    conn.execute_batch("CREATE TEMP TRIGGER fail_reconcile BEFORE UPDATE OF chunks_indexed ON sessions BEGIN SELECT RAISE(ABORT, 'reconciliation failed'); END;").unwrap();
    assert!(index_from_dirs(&mut conn, &opts, true).is_err());
    assert_eq!(chunk_vectors(&conn), before);
    assert_eq!(
        collect_strings(&conn, "SELECT text FROM messages ORDER BY rowid"),
        ["stable", "answer", "tail"]
    );
    conn.execute_batch("DROP TRIGGER fail_reconcile;").unwrap();
    let mut writer = open_db(&db_dir.path().join("test.db")).unwrap();
    index_from_dirs(&mut writer, &opts, true).unwrap();
    assert_eq!(chunk_vectors(&conn), before);
    let stale = embed_chunks(
        &mut conn,
        &MockEmbedder::new(),
        &old_pending,
        None,
        &EmbedOptions::default(),
    )
    .unwrap();
    assert_eq!(stale.embedded, 1, "only the unchanged input can save");
    assert_eq!(chunk_vectors(&conn), before);
    assert_eq!(
        embed_recent_chunks(&mut conn, &MockEmbedder::failing_after(0), 100, None)
            .unwrap()
            .failed_count,
        1
    );
    assert_eq!(chunk_vectors(&conn), before);
    drop(conn);
    drop(writer);
    let mut conn = open_db(&db_dir.path().join("test.db")).unwrap();
    assert_eq!(index_from_dirs(&mut conn, &opts, true).unwrap().indexed, 0);
    let model = MockEmbedder::new();
    assert_eq!(
        embed_recent_chunks(&mut conn, &model, 100, None)
            .unwrap()
            .embedded,
        1
    );
    assert_eq!(model.calls(), 1);
    assert_eq!(chunk_vectors(&conn).len(), 2);
}

const ROLLOUT_NAME: &str = "rollout-2026-04-27T00-00-00-01900000-0000-7000-8000-000000000001.jsonl";
const ROLLOUT_TURNS: &str = concat!(
    r#"{"timestamp":"2026-04-27T00:00:01Z","type":"response_item","payload":{"role":"user","content":[{"type":"input_text","text":"which flag disables the cache"}]}}"#,
    "\n",
    r#"{"timestamp":"2026-04-27T00:00:02Z","type":"response_item","payload":{"role":"assistant","content":[{"type":"output_text","text":"pass --no-cache"}]}}"#,
    "\n",
);

fn compress_log(plain: &Path) -> PathBuf {
    let compressed = plain.with_file_name(format!(
        "{}.zst",
        plain.file_name().unwrap().to_str().unwrap()
    ));
    let body = fs::read(plain).unwrap();
    fs::write(&compressed, zstd::encode_all(body.as_slice(), 0).unwrap()).unwrap();
    compressed
}

fn session_row(conn: &Connection) -> Vec<String> {
    collect_strings(
        conn,
        "SELECT json_array(session_id, file_path, slug, project, timestamp) FROM sessions",
    )
}

fn stored_vectors(conn: &Connection) -> Vec<String> {
    collect_strings(
        conn,
        "SELECT json_array(c.content, hex(v.embedding)) FROM vec_chunks v JOIN qa_chunks c ON c.id = v.chunk_id ORDER BY c.content, v.sub_idx",
    )
}

#[test]
fn compressing_a_codex_log_keeps_the_indexed_session_and_its_embeddings() {
    let meta = r#"{"timestamp":"2026-04-27T00:00:00Z","type":"session_meta","payload":{"id":"01900000-0000-7000-8000-000000000001","cwd":"/proj"}}"#;
    for body in [format!("{meta}\n{ROLLOUT_TURNS}"), ROLLOUT_TURNS.to_owned()] {
        let (_db, mut conn) = setup_test_db();
        let tmp = TempDir::new().unwrap();
        let claude_dir = tmp.path().join("claude_projects");
        let codex_dir = tmp.path().join("codex_sessions");
        let day_dir = codex_dir.join("2026/04/27");
        fs::create_dir_all(&day_dir).unwrap();
        let plain = day_dir.join(ROLLOUT_NAME);
        fs::write(&plain, &body).unwrap();
        let opts = IndexOptions {
            force: false,
            claude_dir: &claude_dir,
            codex_dir: &codex_dir,
            codex_archived_dir: None,
        };
        index_from_dirs(&mut conn, &opts, true).unwrap();
        index_chunks(&mut conn, None).unwrap();
        embed_recent_chunks(&mut conn, &MockEmbedder::new(), 100, None).unwrap();
        let row_before = session_row(&conn);
        let vectors_before = stored_vectors(&conn);
        assert_eq!(row_before.len(), 1);
        assert!(!vectors_before.is_empty());

        let compressed = compress_log(&plain);
        fs::remove_file(&plain).unwrap();
        let stats = index_from_dirs(&mut conn, &opts, true).unwrap();

        assert_eq!(stats.total_sessions, 1, "{body}");
        let row_after = session_row(&conn);
        assert_eq!(
            row_after,
            [row_before[0].replace(plain.to_str().unwrap(), compressed.to_str().unwrap())],
            "only the file path changes"
        );
        assert_eq!(stored_vectors(&conn), vectors_before);
        assert!(stats.parse_diagnostics.is_empty());
    }
}

#[test]
fn plain_codex_log_is_indexed_once_when_its_compressed_copy_sits_beside_it() {
    let (_db, mut conn) = setup_test_db();
    let tmp = TempDir::new().unwrap();
    let claude_dir = tmp.path().join("claude_projects");
    let codex_dir = tmp.path().join("codex_sessions");
    let day_dir = codex_dir.join("2026/04/27");
    fs::create_dir_all(&day_dir).unwrap();
    let plain = day_dir.join(ROLLOUT_NAME);
    fs::write(&plain, ROLLOUT_TURNS).unwrap();
    compress_log(&plain);
    let opts = IndexOptions {
        force: false,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
        codex_archived_dir: None,
    };

    let first = index_from_dirs(&mut conn, &opts, true).unwrap();
    index_chunks(&mut conn, None).unwrap();
    let second = index_from_dirs(&mut conn, &opts, true).unwrap();

    assert_eq!((first.indexed, first.total_sessions), (1, 1));
    assert_eq!(
        second.indexed, 0,
        "the two copies must not replace each other"
    );
    assert_eq!(
        collect_strings(&conn, "SELECT file_path FROM sessions"),
        [plain.to_str().unwrap()]
    );
}

#[test]
fn compressed_codex_log_keeps_its_session_while_an_empty_plain_copy_sits_beside_it() {
    let (_db, mut conn) = setup_test_db();
    let tmp = TempDir::new().unwrap();
    let claude_dir = tmp.path().join("claude_projects");
    let codex_dir = tmp.path().join("codex_sessions");
    let day_dir = codex_dir.join("2026/04/27");
    fs::create_dir_all(&day_dir).unwrap();
    let plain = day_dir.join(ROLLOUT_NAME);
    fs::write(&plain, ROLLOUT_TURNS).unwrap();
    compress_log(&plain);
    fs::remove_file(&plain).unwrap();
    let opts = IndexOptions {
        force: false,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
        codex_archived_dir: None,
    };
    index_from_dirs(&mut conn, &opts, true).unwrap();
    index_chunks(&mut conn, None).unwrap();
    embed_recent_chunks(&mut conn, &MockEmbedder::new(), 100, None).unwrap();
    let row_before = session_row(&conn);
    let vectors_before = stored_vectors(&conn);
    assert!(!vectors_before.is_empty());

    fs::write(&plain, "").unwrap();
    index_from_dirs(&mut conn, &opts, true).unwrap();

    assert_eq!(session_row(&conn), row_before);
    assert_eq!(stored_vectors(&conn), vectors_before);
}

#[test]
fn compressed_logs_are_indexed_only_from_codex_roots() {
    let tmp = TempDir::new().unwrap();
    let claude_dir = tmp.path().join("claude_projects");
    let codex_dir = tmp.path().join("codex_sessions");
    fs::create_dir_all(&claude_dir).unwrap();
    fs::create_dir_all(&codex_dir).unwrap();
    for dir in [&claude_dir, &codex_dir] {
        fs::write(
            dir.join("a.jsonl.zst"),
            zstd::encode_all(&b"{}\n"[..], 0).unwrap(),
        )
        .unwrap();
        fs::write(dir.join("b.zst"), "").unwrap();
        fs::write(dir.join("c.jsonl.gz"), "").unwrap();
    }

    let mut found: Vec<_> = collect_sources(&IndexOptions {
        force: false,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
        codex_archived_dir: None,
    })
    .sources
    .into_iter()
    .map(|(path, source)| (path.strip_prefix(tmp.path()).unwrap().to_owned(), source))
    .collect();
    found.sort_by(|a, b| a.0.cmp(&b.0));

    assert_eq!(
        found,
        [(PathBuf::from("codex_sessions/a.jsonl.zst"), Source::Codex)]
    );
}

#[test]
fn undecodable_compressed_codex_log_is_reported_as_a_read_error_and_other_sessions_index() {
    let (_db, mut conn) = setup_test_db();
    let tmp = TempDir::new().unwrap();
    let claude_dir = tmp.path().join("claude_projects");
    let codex_dir = tmp.path().join("codex_sessions");
    let day_dir = codex_dir.join("2026/04/27");
    fs::create_dir_all(&day_dir).unwrap();
    fs::write(day_dir.join(ROLLOUT_NAME), ROLLOUT_TURNS).unwrap();
    let corrupt =
        day_dir.join("rollout-2026-04-27T00-00-00-01900000-0000-7000-8000-000000000002.jsonl.zst");
    let mut truncated = zstd::encode_all(ROLLOUT_TURNS.as_bytes(), 0).unwrap();
    truncated.truncate(truncated.len() / 2);
    fs::write(&corrupt, truncated).unwrap();

    let stats = index_from_dirs(
        &mut conn,
        &IndexOptions {
            force: false,
            claude_dir: &claude_dir,
            codex_dir: &codex_dir,
            codex_archived_dir: None,
        },
        true,
    )
    .unwrap();

    assert_eq!((stats.indexed, stats.total_sessions), (1, 1));
    assert_eq!(
        stats
            .parse_diagnostics
            .iter()
            .map(|d| (d.file_path.as_path(), d.source, d.read_error))
            .collect::<Vec<_>>(),
        [(corrupt.as_path(), Source::Codex, true)]
    );
}

fn rollout_name(n: u8) -> String {
    format!("rollout-2026-04-27T00-00-00-01900000-0000-7000-8000-0000000000{n:02}.jsonl")
}

fn rollout_body(n: u8, topic: &str) -> String {
    format!(
        r#"{{"timestamp":"2026-04-27T00:00:00Z","type":"session_meta","payload":{{"id":"01900000-0000-7000-8000-0000000000{n:02}","cwd":"/proj"}}}}
{{"timestamp":"2026-04-27T00:00:01Z","type":"response_item","payload":{{"role":"user","content":[{{"type":"input_text","text":"{topic} question"}}]}}}}
"#
    )
}

struct CodexRoots {
    tmp: TempDir,
    claude: PathBuf,
    sessions: PathBuf,
    day: PathBuf,
    archived: PathBuf,
}

impl CodexRoots {
    fn new() -> Self {
        let tmp = TempDir::new().unwrap();
        let claude = tmp.path().join("claude_projects");
        let sessions = tmp.path().join(".codex/sessions");
        let day = sessions.join("2026/04/27");
        let archived = tmp.path().join(".codex/archived_sessions");
        fs::create_dir_all(&day).unwrap();
        fs::create_dir_all(&archived).unwrap();
        Self {
            tmp,
            claude,
            sessions,
            day,
            archived,
        }
    }

    fn opts(&self) -> IndexOptions<'_> {
        IndexOptions {
            force: false,
            claude_dir: &self.claude,
            codex_dir: &self.sessions,
            codex_archived_dir: Some(&self.archived),
        }
    }
}

fn session_ids(conn: &Connection) -> Vec<String> {
    collect_strings(conn, "SELECT session_id FROM sessions ORDER BY session_id")
}

#[test]
fn archiving_a_codex_log_keeps_the_indexed_session_and_its_embeddings() {
    let meta = r#"{"timestamp":"2026-04-27T00:00:00Z","type":"session_meta","payload":{"id":"01900000-0000-7000-8000-000000000001","cwd":"/proj"}}"#;
    for body in [format!("{meta}\n{ROLLOUT_TURNS}"), ROLLOUT_TURNS.to_owned()] {
        let (_db, mut conn) = setup_test_db();
        let roots = CodexRoots::new();
        let live = roots.day.join(ROLLOUT_NAME);
        fs::write(&live, &body).unwrap();
        index_from_dirs(&mut conn, &roots.opts(), true).unwrap();
        index_chunks(&mut conn, None).unwrap();
        embed_recent_chunks(&mut conn, &MockEmbedder::new(), 100, None).unwrap();
        let row_before = session_row(&conn);
        let vectors_before = stored_vectors(&conn);
        assert!(!vectors_before.is_empty());

        let compressed = compress_log(&live);
        let archived = roots.archived.join(compressed.file_name().unwrap());
        fs::rename(&compressed, &archived).unwrap();
        fs::remove_file(&live).unwrap();
        let stats = index_from_dirs(&mut conn, &roots.opts(), true).unwrap();

        assert_eq!(stats.total_sessions, 1, "{body}");
        assert_eq!(
            session_row(&conn),
            [row_before[0].replace(live.to_str().unwrap(), archived.to_str().unwrap())],
            "only the file path changes; the slug keeps the session's date"
        );
        assert_eq!(stored_vectors(&conn), vectors_before);
        assert!(stats.skipped_roots.is_empty());
    }
}

#[test]
fn absent_archived_root_neither_reports_nor_blocks_codex_cleanup() {
    let (_db, mut conn) = setup_test_db();
    let roots = CodexRoots::new();
    fs::remove_dir(&roots.archived).unwrap();
    fs::write(roots.day.join(rollout_name(1)), rollout_body(1, "kept")).unwrap();
    let deleted = roots.day.join(rollout_name(3));
    fs::write(&deleted, rollout_body(3, "deleted")).unwrap();
    index_from_dirs(&mut conn, &roots.opts(), true).unwrap();

    fs::remove_file(&deleted).unwrap();
    let stats = index_from_dirs(&mut conn, &roots.opts(), true).unwrap();

    assert_eq!(session_ids(&conn), ["01900000-0000-7000-8000-000000000001"]);
    assert!(stats.skipped_roots.is_empty());
}

#[test]
fn missing_archived_root_keeps_its_sessions_and_still_removes_sessions_deleted_elsewhere() {
    let (_db, mut conn) = setup_test_db();
    let roots = CodexRoots::new();
    fs::write(roots.day.join(rollout_name(1)), rollout_body(1, "live")).unwrap();
    fs::write(
        roots.archived.join(rollout_name(2)),
        rollout_body(2, "archived"),
    )
    .unwrap();
    let deleted = roots.day.join(rollout_name(3));
    fs::write(&deleted, rollout_body(3, "deleted")).unwrap();
    assert_eq!(
        index_from_dirs(&mut conn, &roots.opts(), true)
            .unwrap()
            .total_sessions,
        3
    );

    fs::remove_dir_all(&roots.archived).unwrap();
    fs::remove_file(&deleted).unwrap();
    let stats = index_from_dirs(&mut conn, &roots.opts(), true).unwrap();

    assert_eq!(
        session_ids(&conn),
        [
            "01900000-0000-7000-8000-000000000001",
            "01900000-0000-7000-8000-000000000002"
        ]
    );
    assert_eq!(
        stats
            .skipped_roots
            .iter()
            .map(|s| (s.source, s.reason, s.preserved_sessions))
            .collect::<Vec<_>>(),
        [(Source::Codex, SkippedReason::MissingRoot, 1)]
    );
    let note = stats.skipped_roots[0].note();
    assert!(
        note.contains("reconciled the rest") && !note.contains("none deleted this run"),
        "the other Codex root was cleaned, so the note must not claim nothing was deleted: {note}"
    );
}

#[cfg(unix)]
#[test]
fn missing_archived_root_keeps_its_parse_diagnostics_and_still_clears_those_deleted_elsewhere() {
    let (_db, mut conn) = setup_test_db();
    let roots = CodexRoots::new();
    let mut undecodable = zstd::encode_all(rollout_body(2, "broken").as_bytes(), 0).unwrap();
    undecodable.truncate(undecodable.len() / 2);
    let archived = roots.archived.join(format!("{}.zst", rollout_name(2)));
    let deleted = roots.day.join(format!("{}.zst", rollout_name(3)));
    fs::write(&archived, &undecodable).unwrap();
    fs::write(&deleted, &undecodable).unwrap();
    let diagnosed = |stats: &IndexStats| -> Vec<PathBuf> {
        let mut paths: Vec<_> = stats
            .parse_diagnostics
            .iter()
            .map(|d| d.file_path.clone())
            .collect();
        paths.sort();
        paths
    };
    let first = index_from_dirs(&mut conn, &roots.opts(), true).unwrap();
    let mut both = vec![archived.clone(), deleted.clone()];
    both.sort();
    assert_eq!(diagnosed(&first), both);

    fs::remove_dir_all(&roots.archived).unwrap();
    fs::remove_file(&deleted).unwrap();
    let second = index_from_dirs(&mut conn, &roots.opts(), true).unwrap();

    assert_eq!(diagnosed(&second), [archived]);
}

#[test]
fn unreadable_archived_root_keeps_every_codex_session() {
    use std::os::unix::fs::PermissionsExt;

    let (_db, mut conn) = setup_test_db();
    let roots = CodexRoots::new();
    fs::write(roots.day.join(rollout_name(1)), rollout_body(1, "live")).unwrap();
    fs::write(
        roots.archived.join(rollout_name(2)),
        rollout_body(2, "archived"),
    )
    .unwrap();
    let moved = roots.day.join(rollout_name(3));
    fs::write(&moved, rollout_body(3, "moved")).unwrap();
    assert_eq!(
        index_from_dirs(&mut conn, &roots.opts(), true)
            .unwrap()
            .total_sessions,
        3
    );

    fs::rename(&moved, roots.archived.join(rollout_name(3))).unwrap();
    let permissions = fs::metadata(&roots.archived).unwrap().permissions();
    fs::set_permissions(&roots.archived, fs::Permissions::from_mode(0o000)).unwrap();
    let stats = index_from_dirs(&mut conn, &roots.opts(), true);
    fs::set_permissions(&roots.archived, permissions).unwrap();
    let stats = stats.unwrap();

    assert_eq!(
        session_ids(&conn).len(),
        3,
        "a session moved into an unreadable root survives"
    );
    assert_eq!(
        stats
            .skipped_roots
            .iter()
            .map(|s| (s.source, s.reason, s.preserved_sessions))
            .collect::<Vec<_>>(),
        [(Source::Codex, SkippedReason::IncompleteEnumeration, 2)]
    );
    drop(roots.tmp);
}

#[test]
fn archived_root_whose_parent_cannot_be_searched_keeps_every_codex_session() {
    use std::os::unix::fs::PermissionsExt;

    let (_db, mut conn) = setup_test_db();
    let roots = CodexRoots::new();
    let locked = roots.tmp.path().join("locked");
    let archived = locked.join("archived_sessions");
    fs::create_dir_all(&archived).unwrap();
    let opts = IndexOptions {
        codex_archived_dir: Some(&archived),
        ..roots.opts()
    };
    fs::write(roots.day.join(rollout_name(1)), rollout_body(1, "live")).unwrap();
    let moved = roots.day.join(rollout_name(3));
    fs::write(&moved, rollout_body(3, "moved")).unwrap();
    index_from_dirs(&mut conn, &opts, true).unwrap();

    fs::rename(&moved, archived.join(rollout_name(3))).unwrap();
    let permissions = fs::metadata(&locked).unwrap().permissions();
    fs::set_permissions(&locked, fs::Permissions::from_mode(0o000)).unwrap();
    let stats = index_from_dirs(&mut conn, &opts, true);
    fs::set_permissions(&locked, permissions).unwrap();
    let stats = stats.unwrap();

    assert_eq!(
        session_ids(&conn).len(),
        2,
        "a session moved into a root that cannot be inspected survives"
    );
    assert_eq!(
        stats
            .skipped_roots
            .iter()
            .map(|s| (s.source, s.reason, s.preserved_sessions))
            .collect::<Vec<_>>(),
        [(Source::Codex, SkippedReason::IncompleteEnumeration, 1)]
    );
}

fn archived_dir_for(vars: &[(&str, &str)]) -> Option<PathBuf> {
    resolve_codex_archived_dir_with(Path::new("/fake/home"), |key| {
        vars.iter()
            .find(|(k, _)| *k == key)
            .map(|(_, v)| OsString::from(v))
    })
}

#[test]
fn empty_codex_archived_dir_turns_the_archive_off() {
    assert_eq!(archived_dir_for(&[("RECALL_CODEX_ARCHIVED_DIR", "")]), None);
}

#[test]
fn codex_archived_dir_defaults_under_home_unless_only_the_sessions_dir_is_overridden() {
    let resolve = archived_dir_for;

    assert_eq!(
        resolve(&[]),
        Some(PathBuf::from("/fake/home/.codex/archived_sessions"))
    );
    assert_eq!(resolve(&[("RECALL_CODEX_DIR", "/tmp/codex")]), None);
    assert_eq!(
        resolve(&[
            ("RECALL_CODEX_DIR", "/tmp/codex"),
            ("RECALL_CODEX_ARCHIVED_DIR", "/tmp/archived")
        ]),
        Some(PathBuf::from("/tmp/archived"))
    );
    assert_eq!(
        resolve(&[("RECALL_CODEX_ARCHIVED_DIR", "/tmp/archived")]),
        Some(PathBuf::from("/tmp/archived"))
    );
}
