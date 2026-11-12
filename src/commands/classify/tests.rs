use std::fs;

use crate::envelope::render_json_success;

use super::*;

/// The `--json` envelope `recall classify` prints for an output.
fn printed_envelope(out: &CommandOutput) -> serde_json::Value {
    serde_json::from_str(&render_json_success(out)).unwrap()
}

/// Every session's stored label, ordered by session id.
fn labels(db_path: &Path) -> Vec<(String, Option<String>)> {
    db::open_db(db_path)
        .unwrap()
        .prepare("SELECT session_id, session_type FROM sessions ORDER BY session_id")
        .unwrap()
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
        .unwrap()
        .map(Result::unwrap)
        .collect()
}

/// An index of two unlabeled sessions whose logs are missing and `tagged`, labeled interactive,
/// whose readable log names no automated origin. Each is labeled from its first turn.
fn seed_classify_db() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::TempDir::new().unwrap();
    let tagged_log = dir.path().join("tagged.jsonl");
    fs::write(&tagged_log, r#"{"type":"user","message":"question"}"#).unwrap();
    let db_path = dir.path().join("recall.db");
    let conn = open_or_create_db(&db_path).unwrap();
    conn.execute_batch(
        "INSERT INTO sessions (session_id, source, file_path, project, slug, timestamp, mtime, session_type) VALUES
             ('auto', 'claude', '/f', '/p', 'a', 0, 0.0, NULL),
             ('human', 'claude', '/f', '/p', 'h', 0, 0.0, NULL);
         INSERT INTO messages (session_id, role, text) VALUES
             ('auto', 'user', '<command-message>clear</command-message>'),
             ('human', 'user', 'how do I implement auth'),
             ('tagged', 'user', '<command-message>clear</command-message>');",
    )
    .unwrap();
    conn.execute(
        "INSERT INTO sessions (session_id, source, file_path, project, slug, timestamp, mtime, session_type)
             VALUES ('tagged', 'claude', ?1, '/p', 't', 0, 0.0, 'interactive')",
        [tagged_log.to_str().unwrap()],
    )
    .unwrap();
    (dir, db_path)
}

#[test]
fn classify_all_labels_every_session_from_its_first_turn() {
    let (_dir, db_path) = seed_classify_db();

    let out = run_classify(true, false, &Some(db_path.clone())).unwrap();

    assert_eq!(
        out.markdown,
        "classified 3 session(s): 2 automated, 1 interactive"
    );
    assert_eq!(
        printed_envelope(&out),
        serde_json::json!({
            "data": { "automated": 2, "classified": 3, "dry_run": false, "interactive": 1 },
            "degraded": false,
            "notes": [],
        })
    );
    assert_eq!(
        labels(&db_path),
        [
            ("auto".to_owned(), Some("automated".to_owned())),
            ("human".to_owned(), Some("interactive".to_owned())),
            ("tagged".to_owned(), Some("automated".to_owned())),
        ]
    );
}

#[test]
fn classify_without_all_labels_only_unlabeled_sessions() {
    let (_dir, db_path) = seed_classify_db();

    let out = run_classify(false, false, &Some(db_path.clone())).unwrap();

    assert_eq!(
        out.markdown,
        "classified 2 session(s): 1 automated, 1 interactive"
    );
    assert_eq!(
        printed_envelope(&out),
        serde_json::json!({
            "data": { "automated": 1, "classified": 2, "dry_run": false, "interactive": 1 },
            "degraded": false,
            "notes": [],
        })
    );
    assert_eq!(
        labels(&db_path),
        [
            ("auto".to_owned(), Some("automated".to_owned())),
            ("human".to_owned(), Some("interactive".to_owned())),
            ("tagged".to_owned(), Some("interactive".to_owned())),
        ]
    );
}
