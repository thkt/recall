use std::fs;

use rusqlite::Connection;
use serde_json::{Value, json};

use super::{SessionType, classify_first_turn};
use crate::db::setup_test_db;
use crate::embedder::{MockEmbedder, embed_recent_chunks};
use crate::indexer::{IndexOptions, index_chunks, index_from_dirs};
use crate::reclassify_sessions;
use crate::search::{SearchOptions, search};

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

#[test]
fn provenance_classification_removes_synthetic_contamination_without_human_omissions() {
    let (_db_dir, mut conn) = setup_test_db();
    let root = tempfile::tempdir().unwrap();
    let claude = root.path().join("claude");
    let codex = root.path().join("codex");
    let parent = "01234567-89ab-cdef-0123-456789abcdef";
    // Public synthetic corpus: six known automated and six human/unknown cases.
    // The label is independently specified here, not derived from the classifier.
    let cases: [(&str, &str, Value, &str, bool); 12] = [
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
    }
    let opts = IndexOptions {
        force: false,
        claude_dir: &claude,
        codex_dir: &codex,
    };
    assert_eq!(index_from_dirs(&mut conn, &opts, true).unwrap().indexed, 12);
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
    assert_eq!(baseline_mixing, 6);
    let planned = reclassify_sessions(&mut conn, true, true).unwrap();
    assert_eq!(
        planned
            .iter()
            .filter(|o| o.session_type == SessionType::Automated)
            .count(),
        6
    );
    assert_eq!(labels(&conn), old_labels, "dry run must not write");
    assert!(
        reclassify_sessions(&mut conn, false, false)
            .unwrap()
            .is_empty()
    );
    for _ in 0..2 {
        assert_eq!(
            reclassify_sessions(&mut conn, true, false).unwrap().len(),
            12
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
    assert_eq!(ordinary.len(), 6);
    assert_eq!(inclusive.len(), 12);
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
            crate::read_classification_origin(path, source, id),
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
fn reclassification_reserves_the_writer_before_reading_source_files() {
    use crate::db::open_db;
    use rusqlite::ErrorCode;
    use std::time::Duration;

    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("index.db");
    let mut conn = open_db(&path).unwrap();
    conn.execute(
        "INSERT INTO sessions (session_id, source, file_path) VALUES ('s', 'claude', '/s')",
        [],
    )
    .unwrap();
    let other = open_db(&path).unwrap();
    other.busy_timeout(Duration::ZERO).unwrap();
    let mut reads = 0;
    crate::reclassify_sessions_with(&mut conn, true, false, |_, _, _| {
        reads += 1;
        let error = other
            .execute(
                "UPDATE sessions SET session_type = 'interactive' WHERE session_id = 's'",
                [],
            )
            .unwrap_err();
        assert_eq!(error.sqlite_error_code(), Some(ErrorCode::DatabaseBusy));
        Ok(true)
    })
    .unwrap();
    assert_eq!(reads, 1);
    assert_eq!(
        strings(&conn, "SELECT session_type FROM sessions"),
        ["automated"]
    );
    other
        .execute(
            "UPDATE sessions SET session_type = 'interactive' WHERE session_id = 's'",
            [],
        )
        .unwrap();
}
