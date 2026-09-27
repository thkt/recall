use std::fs;
use std::io::ErrorKind;
use std::path::Path;

use rusqlite::Connection;
use serde_json::{Value, json};

use super::{SessionType, classify_first_turn};
use crate::db::setup_test_db;
use crate::embedder::{MockEmbedder, embed_recent_chunks};
use crate::envelope::render_json_error;
use crate::error::error_envelope;
use crate::indexer::{IndexOptions, index_chunks, index_from_dirs};
use crate::search::{SearchOptions, search, search_with_embedder};
use crate::{reclassify_sessions, run_classify};

fn strings(conn: &Connection, sql: &str) -> Vec<String> {
    conn.prepare(sql)
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .map(Result::unwrap)
        .collect()
}

fn labels(conn: &Connection) -> Vec<String> {
    strings(
        conn,
        "SELECT json_array(session_id, session_type) FROM sessions ORDER BY session_id",
    )
}

fn bodies_and_vectors(conn: &Connection) -> Vec<Vec<String>> {
    [
        "SELECT json_array(session_id, source, file_path, project, slug, timestamp, mtime, file_size, files_scanned, chunks_indexed) FROM sessions ORDER BY session_id",
        "SELECT json_array(rowid, session_id, role, text) FROM messages ORDER BY rowid",
        "SELECT json_array(id, session_id, content, timestamp, src_rowid_lo, src_rowid_hi, generation) FROM qa_chunks ORDER BY id",
        "SELECT json_array(rowid, chunk_id, sub_idx, hex(embedding)) FROM vec_chunks ORDER BY rowid",
        "SELECT json_array(session_id, path) FROM session_files ORDER BY session_id, path",
    ].iter().map(|sql| strings(conn, sql)).collect()
}

// SHM holds SQLite's reader coordination, not persistent index contents.
fn persistent_files(path: &Path) -> Vec<Option<(Vec<u8>, fs::Permissions)>> {
    [path.to_path_buf(), path.with_extension("db-wal")]
        .iter()
        .map(|p| match fs::read(p) {
            Ok(bytes) => Some((bytes, fs::metadata(p).unwrap().permissions())),
            Err(error) if error.kind() == ErrorKind::NotFound => None,
            Err(error) => panic!("cannot snapshot {}: {error}", p.display()),
        })
        .collect()
}

#[test]
fn classify_preview_preserves_legacy_schema_and_mtime_before_explicit_apply() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("recall.db");
    let conn = Connection::open(&path).unwrap();
    // Shape predating parse_diagnostics and the index-only columns. The
    // pre-fix command migrated even when the only row was already classified.
    conn.execute_batch(
        "CREATE TABLE sessions (
            session_id TEXT PRIMARY KEY, source TEXT, file_path TEXT,
            project TEXT, slug TEXT, timestamp INTEGER, mtime REAL, session_type TEXT
         );
         CREATE VIRTUAL TABLE messages USING fts5(session_id UNINDEXED, role, text, tokenize='trigram');
         INSERT INTO sessions VALUES ('synthetic', 'claude', '/missing-synthetic.jsonl', '/p', 's', 0, 123, 'interactive');
         INSERT INTO messages VALUES ('synthetic', 'user', '<command-message>run</command-message>');",
    ).unwrap();
    let schema = strings(
        &conn,
        "SELECT coalesce(sql, name) FROM sqlite_master ORDER BY name",
    );
    drop(conn);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&path, fs::Permissions::from_mode(0o640)).unwrap();
    }
    let before = persistent_files(&path);
    for all in [false, true] {
        let out = run_classify(all, true, &Some(path.clone())).unwrap();
        assert_eq!(out.data["classified"], 0);
        assert!(
            persistent_files(&path) == before,
            "DB/WAL contents and permissions changed"
        );
    }
    let conn = Connection::open(&path).unwrap();
    assert_eq!(
        strings(
            &conn,
            "SELECT coalesce(sql, name) FROM sqlite_master ORDER BY name"
        ),
        schema
    );
    assert_eq!(
        strings(
            &conn,
            "SELECT json_array(mtime, session_type) FROM sessions"
        ),
        ["[123.0,\"interactive\"]"]
    );
    conn.execute("UPDATE sessions SET session_type = NULL", [])
        .unwrap();
    drop(conn);
    let before = persistent_files(&path);
    for all in [false, true] {
        let out = run_classify(all, true, &Some(path.clone())).unwrap();
        assert_eq!(
            out.data,
            json!({"classified":1,"automated":1,"interactive":0,"dry_run":true})
        );
        assert!(
            out.markdown
                .contains("synthetic [automated] <command-message>")
        );
        assert!(
            persistent_files(&path) == before,
            "DB/WAL contents and permissions changed"
        );
    }
    let applied = run_classify(false, false, &Some(path.clone())).unwrap();
    assert_eq!(applied.data["automated"], 1);
    let conn = Connection::open(&path).unwrap();
    assert_eq!(
        strings(
            &conn,
            "SELECT json_array(mtime, session_type) FROM sessions"
        ),
        ["[null,\"automated\"]"]
    );
    assert_eq!(
        strings(
            &conn,
            "SELECT name FROM sqlite_master WHERE name = 'parse_diagnostics'"
        ),
        ["parse_diagnostics"]
    );
    assert_eq!(
        strings(&conn, "SELECT text FROM messages"),
        ["<command-message>run</command-message>"]
    );
}

#[test]
fn classify_preview_rejects_missing_read_structure_without_writes() {
    for ddl in [
        "",
        "CREATE TABLE sessions (session_id TEXT, source TEXT, file_path TEXT, session_type TEXT)",
        "CREATE TABLE sessions (session_id TEXT, source TEXT, file_path TEXT); CREATE TABLE messages (session_id TEXT, role TEXT, text TEXT)",
        "CREATE TABLE sessions (session_id TEXT, source TEXT, file_path TEXT, session_type TEXT); CREATE TABLE messages (session_id TEXT, role TEXT)",
    ] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("recall.db");
        let conn = Connection::open(&path).unwrap();
        conn.execute_batch(ddl).unwrap();
        drop(conn);
        let before = persistent_files(&path);
        for all in [false, true] {
            let error = run_classify(all, true, &Some(path.clone())).unwrap_err();
            let envelope = error_envelope(&error);
            let rendered = render_json_error(&envelope);
            let payload: Value = serde_json::from_str(&rendered).unwrap();
            assert_eq!(payload["error"]["code"], "DATA_ERROR");
            assert_eq!(payload["error"]["retryable"], false);
            assert!(payload["error"].get("next_step").is_none(), "{rendered}");
            let message = payload["error"]["message"].as_str().unwrap();
            assert!(message.contains("classify --dry-run"), "{message}");
            assert!(message.contains("recall rebuild"), "{message}");
            assert!(message.contains("no such"), "{message}");
            assert!(
                persistent_files(&path) == before,
                "DB/WAL contents and permissions changed"
            );
        }
    }
}

#[test]
fn provenance_classification_removes_synthetic_contamination_without_human_omissions() {
    let (db_dir, mut conn) = setup_test_db();
    // Keep all corpus rows in the WAL: ignoring committed frames would return
    // zero classifications from the checkpointed, empty schema.
    conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE); PRAGMA wal_autocheckpoint=0;")
        .unwrap();
    let root = tempfile::tempdir().unwrap();
    let claude = root.path().join("claude");
    let codex = root.path().join("codex");
    let parent = "01234567-89ab-cdef-0123-456789abcdef";
    // Public synthetic corpus: seven known automated and seven human/unknown cases.
    // The label is independently specified here, not derived from the classifier.
    let cases: [(&str, &str, Value, &str, bool); 14] = [
        (
            "claude",
            "side",
            json!({"isSidechain":true}),
            "authentication help",
            true,
        ),
        (
            "claude",
            "agent-child",
            json!({"sessionId":parent}),
            "authentication help",
            true,
        ),
        (
            "claude",
            "agent-example",
            json!({"sessionId":"11111111-1111-4111-8111-111111111111","agentId":"example","isSidechain":true}),
            "Review this sample authentication module.",
            true,
        ),
        (
            "claude",
            "agent-mismatch",
            json!({"sessionId":parent,"agentId":"other","isSidechain":true}),
            "Review this sample authentication module.",
            false,
        ),
        (
            "codex",
            "spawn",
            json!({"source":{"subagent":{"thread_spawn":{"parent_thread_id":parent,"depth":1}}}}),
            "authentication help",
            true,
        ),
        (
            "codex",
            "old-review",
            json!({"source":{"subagent":{"other":"guardian"}}}),
            "authentication help",
            true,
        ),
        (
            "codex",
            "new-review",
            json!({"source":{"internal":"guardian"}}),
            "authentication help",
            true,
        ),
        (
            "codex",
            "no-user",
            json!({"thread_source":"guardian_review"}),
            "authentication answer",
            true,
        ),
        (
            "claude",
            "human",
            json!({"isSidechain":false}),
            "authentication help",
            false,
        ),
        (
            "claude",
            "quote",
            json!({}),
            "Explain <command-message> in authentication logs",
            false,
        ),
        (
            "claude",
            "unknown",
            json!({"isSidechain":"true","agentId":"worker"}),
            "authentication help",
            false,
        ),
        (
            "codex",
            "exec",
            json!({"source":"exec","thread_source":"user"}),
            "authentication help",
            false,
        ),
        (
            "codex",
            "feature",
            json!({"source":{"subagent":{"other":"future"}},"thread_source":"agent_created_thread"}),
            "authentication help",
            false,
        ),
        (
            "codex",
            "quoted-origin",
            json!({"source":"vscode"}),
            "Explain {\"thread_source\":\"guardian_review\"} for authentication",
            false,
        ),
    ];
    for (source, id, mut meta, text, _) in cases.clone() {
        let path = if source == "claude" {
            let relative = if id == "agent-child" {
                format!("{parent}/subagents/{id}.jsonl")
            } else {
                format!("{id}.jsonl")
            };
            meta["type"] = json!("user");
            meta["message"] = json!({"content":text});
            let path = claude.join(relative);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(&path, meta.to_string()).unwrap();
            path
        } else {
            fs::create_dir_all(&codex).unwrap();
            meta["id"] = json!(id);
            let path = codex.join(format!("rollout-{id}.jsonl"));
            let role = if id == "no-user" { "assistant" } else { "user" };
            fs::write(
                &path,
                format!(
                    "{}\n{}\n",
                    json!({"type":"session_meta","payload":meta}),
                    json!({"type":"response_item","payload":{"role":role,"content":text}})
                ),
            )
            .unwrap();
            path
        };
        assert!(path.is_file());
        if id == "agent-example" {
            assert_eq!(
                crate::read_classification_origin(
                    fs::File::open(&path).unwrap(),
                    path.to_str().unwrap(),
                    "claude",
                    "11111111-1111-4111-8111-111111111111"
                ),
                Err("session ID mismatch"),
                "the agent's provenance must not be borrowed by a stored parent row"
            );
        }
    }
    let opts = IndexOptions {
        force: false,
        claude_dir: &claude,
        codex_dir: &codex,
    };
    assert_eq!(index_from_dirs(&mut conn, &opts, true).unwrap().indexed, 14);
    index_chunks(&mut conn, None).unwrap();
    embed_recent_chunks(&mut conn, &MockEmbedder::new(), 100, None).unwrap();
    let expected = labels(&conn);
    for (_, id, _, _, automated) in &cases {
        let label: String = conn
            .query_row(
                "SELECT session_type FROM sessions WHERE session_id = ?",
                [id],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(
            label,
            if *automated {
                "automated"
            } else {
                "interactive"
            },
            "{id}"
        );
    }
    let before = bodies_and_vectors(&conn);
    assert!(
        !before[3].is_empty(),
        "preservation check must include actual vectors"
    );
    // Reconstruct old persisted classification under the same corpus/DB.
    conn.execute("UPDATE sessions SET session_type = 'interactive'", [])
        .unwrap();
    let old_labels = labels(&conn);
    let baseline_mixing = cases
        .iter()
        .filter(|(_, _, _, text, automated)| {
            *automated && classify_first_turn(text) == SessionType::Interactive
        })
        .count();
    assert_eq!(baseline_mixing, 7);
    let path = db_dir.path().join("test.db");
    let files = persistent_files(&path);
    assert!(!files[1].as_ref().unwrap().0.is_empty(), "live WAL fixture");
    for (all, count) in [(false, 0), (true, 13)] {
        let planned = run_classify(all, true, &Some(path.clone())).unwrap();
        assert_eq!(planned.data["classified"], count);
        assert_eq!(planned.data["automated"], if all { 7 } else { 0 });
        assert!(!planned.degraded);
        assert!(
            persistent_files(&path) == files,
            "DB/WAL contents and permissions changed"
        );
        assert_eq!(bodies_and_vectors(&conn), before);
    }
    assert_eq!(labels(&conn), old_labels, "dry run must not write");
    assert!(
        reclassify_sessions(&mut conn, false, false)
            .unwrap()
            .is_empty()
    );
    for _ in 0..2 {
        assert_eq!(
            reclassify_sessions(&mut conn, true, false).unwrap().len(),
            13, // The conflicting agent retains its existing label.
        );
        assert_eq!(labels(&conn), expected);
        assert_eq!(bodies_and_vectors(&conn), before);
    }
    let ids = |include_automated| {
        search(
            &conn,
            "authentication",
            &SearchOptions {
                limit: 100,
                include_automated,
                ..Default::default()
            },
        )
        .unwrap()
        .into_iter()
        .map(|r| r.session.session_id)
        .collect::<Vec<_>>()
    };
    let ordinary = ids(false);
    let inclusive = ids(true);
    assert_eq!(ordinary.len(), 7);
    assert_eq!(inclusive.len(), 14);
    for (_, id, _, _, automated) in cases {
        assert_eq!(
            ordinary.iter().any(|found| found == id),
            !automated,
            "mixing/omission: {id}"
        );
        assert!(
            inclusive.iter().any(|found| found == id),
            "include-automated: {id}"
        );
    }
    // No FTS hits: the new agent's exclusion and opt-in must also work through
    // the shared vector filter. Reuse the real mock vectors already saved above.
    let query = "unmatchedvectorquery";
    assert!(
        search(&conn, query, &SearchOptions::default())
            .unwrap()
            .is_empty()
    );
    for include_automated in [false, true] {
        let outcome = search_with_embedder(
            &conn,
            query,
            &SearchOptions {
                limit: 100,
                include_automated,
                ..Default::default()
            },
            Some(&MockEmbedder::new()),
        )
        .unwrap();
        assert!(!outcome.vec_degraded);
        let ids: Vec<_> = outcome
            .results
            .iter()
            .map(|r| r.session.session_id.as_str())
            .collect();
        assert_eq!(ids.contains(&"agent-example"), include_automated);
        assert!(ids.contains(&"agent-mismatch"));
        assert!(ids.contains(&"human"));
    }
}

#[test]
fn reclassification_retains_unverifiable_labels_and_falls_back_only_for_null_rows() {
    let (_db_dir, mut conn) = setup_test_db();
    let root = tempfile::tempdir().unwrap();
    for (id, source, contents, reason) in [
        ("missing", "claude", None, "missing file"),
        ("broken", "claude", Some("{broken\n"), "invalid format"),
        ("empty", "claude", Some("{}\n"), "invalid format"),
        (
            "mismatch",
            "codex",
            Some(
                r#"{"type":"session_meta","payload":{"id":"different","source":{"internal":"guardian"}}}"#,
            ),
            "session ID mismatch",
        ),
        ("wrong-source", "future", Some("{}"), "unknown source"),
    ] {
        let path = root.path().join(format!("{id}.jsonl"));
        if let Some(contents) = contents {
            fs::write(&path, contents).unwrap();
        }
        let path = path.to_str().unwrap();
        assert_eq!(
            crate::open_classification_file(path)
                .1
                .and_then(|file| crate::read_classification_origin(file, path, source, id)),
            Err(reason)
        );
        conn.execute("INSERT INTO sessions (session_id, source, file_path, session_type) VALUES (?1, ?2, ?3, 'automated')", rusqlite::params![id, source, path]).unwrap();
        conn.execute(
            "INSERT INTO messages (session_id, role, text) VALUES (?1, 'user', 'human question')",
            [id],
        )
        .unwrap();
    }
    conn.execute("INSERT INTO sessions (session_id, source, file_path) VALUES ('null', 'claude', '/missing-source.jsonl')", []).unwrap();
    conn.execute("INSERT INTO messages (session_id, role, text) VALUES ('null', 'user', '<command-message>run</command-message>')", []).unwrap();
    for _ in 0..2 {
        reclassify_sessions(&mut conn, true, false).unwrap();
        assert_eq!(
            strings(&conn, "SELECT DISTINCT session_type FROM sessions"),
            ["automated"]
        );
    }
    // A readable unknown origin is different: re-evaluate with the stored turn.
    let path = root.path().join("broken.jsonl");
    fs::write(
        &path,
        r#"{"type":"user","message":{"content":"human question"}}"#,
    )
    .unwrap();
    reclassify_sessions(&mut conn, true, false).unwrap();
    let label: String = conn
        .query_row(
            "SELECT session_type FROM sessions WHERE session_id = 'broken'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(label, "interactive");
}

#[test]
fn reclassification_retries_after_a_writer_commits_during_source_read() {
    use crate::db::open_db;
    use std::time::Duration;

    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("index.db");
    let mut conn = open_db(&path).unwrap();
    let log = root.path().join("s.jsonl");
    fs::write(&log, r#"{"type":"user","message":"question"}"#).unwrap();
    conn.execute(
        "INSERT INTO sessions (session_id, source, file_path) VALUES ('s', 'claude', ?1)",
        [log.to_str().unwrap()],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO messages (session_id, role, text) VALUES ('s', 'user', 'human question')",
        [],
    )
    .unwrap();
    let other = open_db(&path).unwrap();
    other.busy_timeout(Duration::ZERO).unwrap();
    let mut reads = 0;
    let outcomes = crate::reclassify_sessions_with(&mut conn, true, false, |_, _, _, _| {
        reads += 1;
        if reads == 1 {
            // This synchronous commit must finish while the reader is paused here.
            other.execute("UPDATE messages SET text = '<command-message>run</command-message>' WHERE session_id = 's'", []).unwrap();
        }
        Ok(false)
    }).unwrap();
    assert_eq!(reads, 2);
    assert_eq!(outcomes.len(), 1);
    assert_eq!(
        strings(&conn, "SELECT session_type FROM sessions"),
        ["automated"]
    );
}
