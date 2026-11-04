use std::cell::{Cell, RefCell};
use std::fs;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::process::ExitCode;
use std::time::Duration;

use rusqlite::Connection;
use serde_json::Value;

use crate::commands::classify::{
    ClassifyPhase, read_classification_origin, reclassify_sessions, reclassify_sessions_observed,
    run_classify,
};
use crate::db::open_db;
use crate::envelope::render_json_error;
use crate::error::{classify_exit_code, error_envelope};

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

fn phase_name(phase: ClassifyPhase) -> &'static str {
    match phase {
        ClassifyPhase::Snapshot => "snapshot",
        ClassifyPhase::Prepared => "prepared",
        ClassifyPhase::WriterWaiting => "writer_waiting",
        ClassifyPhase::WriterAcquired => "writer_acquired",
        ClassifyPhase::WriterReleased => "writer_released",
        ClassifyPhase::Updated => "updated",
        ClassifyPhase::Retry => "retry",
    }
}

fn append_blank_line(log: &Path) {
    use std::io::Write;
    writeln!(fs::OpenOptions::new().append(true).open(log).unwrap()).unwrap();
}

#[test]
fn classification_phases_keep_their_order_for_dry_run_apply_and_every_retry_cause() {
    const APPLY: [&str; 7] = [
        "snapshot",
        "prepared",
        "writer_waiting",
        "writer_acquired",
        "updated",
        "updated",
        "writer_released",
    ];
    let retried = |first_attempt: &[&'static str]| {
        let mut phases = first_attempt.to_vec();
        phases.extend(APPLY);
        phases
    };
    let rejected_at_apply = [
        "snapshot",
        "prepared",
        "writer_waiting",
        "writer_acquired",
        "writer_released",
        "retry",
    ];
    let scenarios = [
        ("dry_run", vec!["snapshot", "prepared"]),
        (
            "dry_run_source_changed_during_read",
            vec!["snapshot", "prepared", "retry", "snapshot", "prepared"],
        ),
        ("apply", APPLY.to_vec()),
        (
            "source_changed_during_read",
            retried(&["snapshot", "prepared", "retry"]),
        ),
        (
            "writer_busy",
            retried(&["snapshot", "prepared", "writer_waiting", "retry"]),
        ),
        ("database_changed_before_apply", retried(&rejected_at_apply)),
        ("source_changed_before_apply", retried(&rejected_at_apply)),
    ];

    for (scenario, expected) in scenarios {
        let (root, mut conn, other) = fixture();
        conn.busy_timeout(Duration::ZERO).unwrap();
        let log = root.path().join("a.jsonl");
        let dry_run = scenario.starts_with("dry_run");
        let read_once = Cell::new(false);
        let prepared_once = Cell::new(false);
        let mut phases = Vec::new();

        let outcomes = reclassify_sessions_observed(
            &mut conn,
            true,
            dry_run,
            |file, path, source, id| {
                let origin = read_classification_origin(file, path, source, id);
                if scenario.ends_with("source_changed_during_read") && !read_once.replace(true) {
                    append_blank_line(&log);
                }
                origin
            },
            |phase| {
                phases.push(phase_name(phase));
                if matches!(phase, ClassifyPhase::Prepared) && !prepared_once.replace(true) {
                    match scenario {
                        "writer_busy" => other.execute_batch("BEGIN IMMEDIATE").unwrap(),
                        "database_changed_before_apply" => other
                            .execute_batch("CREATE TABLE unrelated(value INTEGER)")
                            .unwrap(),
                        "source_changed_before_apply" => append_blank_line(&log),
                        _ => {}
                    }
                }
                if matches!(phase, ClassifyPhase::Retry) && scenario == "writer_busy" {
                    other.execute_batch("ROLLBACK").unwrap();
                }
            },
        )
        .unwrap_or_else(|error| panic!("{scenario}: {error}"));

        assert_eq!(phases, expected, "{scenario}");
        assert_eq!(outcomes.len(), 2, "{scenario}");
        let expected_label = (!dry_run).then(|| "automated".to_owned());
        assert_eq!(
            labels(&conn),
            [
                ("a".to_owned(), expected_label.clone()),
                ("b".to_owned(), expected_label),
            ],
            "{scenario}"
        );
    }
}

#[test]
fn unavailable_sources_keep_existing_labels_and_classify_unlabeled_rows_from_the_first_turn() {
    let (root, mut conn, _other) = fixture();
    conn.execute(
        "UPDATE sessions SET session_type = 'automated' WHERE session_id = 'a'",
        [],
    )
    .unwrap();
    for text in [
        "<command-message>synthetic</command-message>",
        "plain question",
    ] {
        conn.execute(
            "INSERT INTO messages (session_id, role, text) VALUES ('b', 'user', ?1)",
            [text],
        )
        .unwrap();
    }
    fs::remove_file(root.path().join("a.jsonl")).unwrap();
    fs::remove_file(root.path().join("b.jsonl")).unwrap();

    let outcomes = reclassify_sessions(&mut conn, true, false).unwrap();

    assert_eq!(outcomes.len(), 1);
    assert_eq!(
        labels(&conn),
        [
            ("a".to_owned(), Some("automated".to_owned())),
            ("b".to_owned(), Some("automated".to_owned())),
        ]
    );
}

#[test]
fn dry_run_excerpt_skips_leading_whitespace_and_keeps_eighty_characters() {
    let (root, conn, other) = fixture();
    let first_turn = format!("  \t{}", "あ".repeat(100));
    conn.execute(
        "INSERT INTO messages (session_id, role, text) VALUES ('a', 'user', ?1)",
        [first_turn],
    )
    .unwrap();
    drop((conn, other));

    let out = run_classify(true, true, &Some(root.path().join("index.db"))).unwrap();

    assert_eq!(
        out.markdown.lines().next().unwrap(),
        format!("a [automated] {}", "あ".repeat(80))
    );
}

#[test]
fn a_missing_messages_table_adds_rebuild_guidance_only_for_dry_run() {
    let (_root, mut conn, _other) = fixture();
    conn.execute_batch("DROP TABLE messages").unwrap();

    let dry_run = reclassify_sessions(&mut conn, true, true)
        .err()
        .expect("the snapshot query must fail");
    let apply = reclassify_sessions(&mut conn, true, false)
        .err()
        .expect("the snapshot query must fail");

    let dry_run = dry_run.to_string();
    assert!(dry_run.contains("classify --dry-run"), "{dry_run}");
    assert!(dry_run.contains("recall rebuild"), "{dry_run}");
    let apply = apply.to_string();
    assert!(apply.contains("no such table"), "{apply}");
    assert!(!apply.contains("recall rebuild"), "{apply}");
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
