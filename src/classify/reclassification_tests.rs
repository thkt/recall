use std::cell::{Cell, RefCell};
use std::fs;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::process::ExitCode;
use std::time::Duration;

use rusqlite::Connection;
use serde_json::Value;

use crate::db::open_db;
use crate::envelope::render_json_error;
use crate::error::{classify_exit_code, error_envelope};
use crate::{
    ClassifyPhase, read_classification_origin, reclassify_sessions, reclassify_sessions_observed,
};

fn fixture() -> (tempfile::TempDir, Connection, Connection) {
    let root = tempfile::tempdir().unwrap();
    let db = root.path().join("index.db");
    let conn = open_db(&db).unwrap();
    for id in ["a", "b"] {
        let log = root.path().join(format!("{id}.jsonl"));
        fs::write(
            &log,
            r#"{"type":"user","message":"question","isSidechain":true}"#,
        )
        .unwrap();
        conn.execute(
            "INSERT INTO sessions (session_id, source, file_path) VALUES (?1, 'claude', ?2)",
            [id, log.to_str().unwrap()],
        )
        .unwrap();
    }
    let other = open_db(&db).unwrap();
    other.busy_timeout(Duration::ZERO).unwrap();
    (root, conn, other)
}

fn labels(conn: &Connection) -> Vec<(String, Option<String>)> {
    conn.prepare("SELECT session_id, session_type FROM sessions ORDER BY session_id")
        .unwrap()
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
        .unwrap()
        .map(Result::unwrap)
        .collect()
}

fn assert_retryable(error: &anyhow::Error) {
    assert_eq!(classify_exit_code(error), ExitCode::from(75));
    let json: Value = serde_json::from_str(&render_json_error(&error_envelope(error))).unwrap();
    assert_eq!(json["error"]["code"], "TEMP_FAILURE");
    assert_eq!(json["error"]["retryable"], true);
    assert!(
        json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("no classification changes")
    );
}

#[test]
fn database_commits_invalidate_the_whole_attempt_even_when_values_return_to_the_snapshot() {
    // Each case commits through another connection at a controlled boundary.
    // Restored values catch implementations that compare only current row fields.
    for mutation in [
        "label",
        "source_path",
        "first_turn",
        "insert",
        "delete",
        "recreate",
        "restore",
        "unrelated",
    ] {
        let (root, mut conn, other) = fixture();
        other
            .execute_batch(
                "CREATE TABLE unrelated(value INTEGER); INSERT INTO unrelated VALUES (0);",
            )
            .unwrap();
        let mut attempts = 0;
        let mut committed = labels(&other);
        let error = reclassify_sessions_observed(&mut conn, true, false, read_classification_origin, |phase| {
            if !matches!(phase, ClassifyPhase::Prepared) { return; }
            attempts += 1;
            match mutation {
                "label" => { other.execute("UPDATE sessions SET session_type = ?1 WHERE session_id = 'a'", [if attempts % 2 == 1 {"interactive"} else {"automated"}]).unwrap(); }
                "source_path" => {
                    other.execute("UPDATE sessions SET source = ?1, file_path = ?2 WHERE session_id = 'a'", [if attempts % 2 == 1 {"codex"} else {"claude"}, root.path().join(format!("changed-{attempts}.jsonl")).to_str().unwrap()]).unwrap();
                }
                "first_turn" => { other.execute("INSERT INTO messages(session_id, role, text) VALUES ('a', 'user', ?1)", [format!("question {attempts}")]).unwrap(); }
                "insert" => { other.execute("INSERT INTO sessions(session_id, source, file_path) VALUES (?1, 'claude', '/missing')", [format!("new-{attempts}")]).unwrap(); }
                "delete" => { other.execute_batch("INSERT OR REPLACE INTO sessions(session_id, source, file_path) VALUES ('a', 'claude', '/missing'); DELETE FROM sessions WHERE session_id = 'a';").unwrap(); }
                "recreate" => {
                    other.execute("DELETE FROM sessions WHERE session_id = 'a'", []).unwrap();
                    other.execute("INSERT INTO sessions(session_id, source, file_path) VALUES ('a', 'claude', ?1)", [root.path().join("a.jsonl").to_str().unwrap()]).unwrap();
                }
                "restore" => { other.execute_batch("UPDATE sessions SET session_type = 'interactive' WHERE session_id = 'a'; UPDATE sessions SET session_type = NULL WHERE session_id = 'a';").unwrap(); }
                "unrelated" => { other.execute("UPDATE unrelated SET value = value + 1", []).unwrap(); }
                _ => unreachable!(),
            }
            committed = labels(&other);
        }).err().expect("every attempt must conflict");
        assert_retryable(&error);
        assert_eq!(attempts, 3, "{mutation}");
        assert_eq!(
            labels(&conn),
            committed,
            "{mutation}: preserve only the other writer's changes"
        );
        assert_eq!(
            labels(&conn).iter().find(|(id, _)| id == "b").unwrap().1,
            None
        );
    }
}

#[test]
fn an_insert_after_an_empty_snapshot_is_included_on_retry() {
    let (_root, mut conn, other) = fixture();
    conn.execute("DELETE FROM sessions", []).unwrap();
    let mut snapshots = 0;
    let outcomes = reclassify_sessions_observed(&mut conn, false, false, read_classification_origin, |phase| {
        if matches!(phase, ClassifyPhase::Snapshot) {
            snapshots += 1;
            if snapshots == 1 {
                other.execute("INSERT INTO sessions(session_id, source, file_path) VALUES ('new', 'claude', '/missing')", []).unwrap();
            }
        }
    }).unwrap();
    assert_eq!(snapshots, 2);
    assert_eq!(outcomes.len(), 1);
    assert_eq!(
        labels(&conn),
        [("new".to_owned(), Some("interactive".to_owned()))]
    );
}

#[test]
fn source_changes_during_read_and_before_apply_discard_old_candidates() {
    for boundary in ["before_read", "after_read", "before_apply"] {
        for mutation in ["append", "replace", "delete", "recover", "permissions"] {
            // Only replacement/deletion distinguish retaining the opened file
            // from reopening its path; other mutations use the existing hooks.
            if boundary == "before_read" && !matches!(mutation, "replace" | "delete") {
                continue;
            }
            let (root, mut conn, _other) = fixture();
            let log = root.path().join("a.jsonl");
            if mutation == "recover" {
                fs::remove_file(&log).unwrap();
            }
            let mut changed = false;
            let retries = Cell::new(0);
            let mutate = || match mutation {
                "append" => {
                    use std::io::Write;
                    writeln!(
                        fs::OpenOptions::new().append(true).open(&log).unwrap(),
                        "\n{{\"type\":\"user\",\"sessionId\":\"another-session\"}}"
                    )
                    .unwrap();
                }
                "replace" | "recover" => {
                    let replacement = root.path().join("replacement");
                    fs::write(
                        &replacement,
                        "{\"type\":\"user\",\"message\":\"now interactive\"}\n",
                    )
                    .unwrap();
                    fs::rename(replacement, &log).unwrap();
                }
                "delete" => fs::remove_file(&log).unwrap(),
                "permissions" => {
                    let mut permissions = fs::metadata(&log).unwrap().permissions();
                    permissions.set_readonly(true);
                    fs::set_permissions(&log, permissions).unwrap();
                }
                _ => unreachable!(),
            };
            // RefCell coordinates two deterministic hooks without sleeps/threads.
            let mutate_once = RefCell::new(|| {
                if !changed {
                    mutate();
                    changed = true;
                }
            });
            let outcomes = reclassify_sessions_observed(
                &mut conn,
                true,
                false,
                |file, path, source, id| {
                    if boundary == "before_read" {
                        (mutate_once.borrow_mut())();
                    }
                    let origin = read_classification_origin(file, path, source, id);
                    if boundary == "before_read"
                        && id == "a"
                        && retries.get() == 0
                        && matches!(mutation, "replace" | "delete")
                    {
                        // The already-open input remains readable after the path
                        // changes; reopening here would read another file or fail.
                        assert_eq!(origin, Ok(true));
                    }
                    if boundary == "after_read" {
                        (mutate_once.borrow_mut())();
                    }
                    origin
                },
                |phase| {
                    if boundary == "before_apply" && matches!(phase, ClassifyPhase::Prepared) {
                        (mutate_once.borrow_mut())();
                    }
                    if matches!(phase, ClassifyPhase::Retry) {
                        retries.set(retries.get() + 1);
                    }
                },
            )
            .unwrap();
            assert_eq!(retries.get(), 1, "{boundary}/{mutation}");
            assert_eq!(outcomes.len(), 2);
            let expected = if mutation == "permissions" {
                "automated"
            } else {
                "interactive"
            };
            assert_eq!(
                labels(&conn)[0].1.as_deref(),
                Some(expected),
                "{boundary}/{mutation}"
            );
        }
    }
}

#[test]
fn persistent_file_conflicts_and_busy_writer_leave_all_labels_unchanged() {
    for busy in [false, true] {
        let (root, mut conn, other) = fixture();
        conn.busy_timeout(Duration::ZERO).unwrap();
        if busy {
            other.execute_batch("BEGIN IMMEDIATE").unwrap();
        }
        let before = labels(&conn);
        let error = reclassify_sessions_observed(
            &mut conn,
            true,
            false,
            read_classification_origin,
            |phase| {
                if !busy && matches!(phase, ClassifyPhase::Prepared) {
                    use std::io::Write;
                    writeln!(
                        fs::OpenOptions::new()
                            .append(true)
                            .open(root.path().join("a.jsonl"))
                            .unwrap()
                    )
                    .unwrap();
                }
            },
        )
        .err()
        .expect("must exhaust retries");
        assert_retryable(&error);
        assert_eq!(labels(&conn), before);
    }
}

#[test]
fn failed_or_interrupted_apply_rolls_back_earlier_classification_updates() {
    for interrupt in [false, true] {
        let (_root, mut conn, _other) = fixture();
        let before = labels(&conn);
        if interrupt {
            let interrupted = catch_unwind(AssertUnwindSafe(|| {
                reclassify_sessions_observed(
                    &mut conn,
                    true,
                    false,
                    read_classification_origin,
                    |phase| {
                        if matches!(phase, ClassifyPhase::Updated) {
                            panic!("interrupt after the first update");
                        }
                    },
                )
                .unwrap();
            }));
            assert!(interrupted.is_err());
        } else {
            conn.execute_batch("CREATE TEMP TRIGGER reject_second BEFORE UPDATE ON sessions WHEN NEW.session_id = 'b' BEGIN SELECT RAISE(ABORT, 'synthetic apply failure'); END;").unwrap();
            let error = reclassify_sessions(&mut conn, true, false).err().unwrap();
            assert!(error.to_string().contains("synthetic apply failure"));
            assert_ne!(
                classify_exit_code(&error),
                ExitCode::from(75),
                "SQL failures are not input conflicts"
            );
        }
        assert_eq!(labels(&conn), before);
    }
}

#[test]
fn repeated_classification_counts_candidates_without_physical_updates() {
    let (_root, mut conn, _other) = fixture();
    assert_eq!(
        reclassify_sessions(&mut conn, true, false).unwrap().len(),
        2
    );
    conn.execute_batch("CREATE TEMP TRIGGER reject_redundant BEFORE UPDATE ON sessions BEGIN SELECT RAISE(ABORT, 'redundant update'); END;").unwrap();
    let before = conn.total_changes();
    assert_eq!(
        reclassify_sessions(&mut conn, true, false).unwrap().len(),
        2
    );
    assert_eq!(conn.total_changes(), before);
}
