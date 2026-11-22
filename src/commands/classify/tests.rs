use std::fs;

use crate::db;
use crate::tests::printed_envelope;

use super::*;

fn stored_labels_by_session_id(db_path: &Path) -> Vec<(String, Option<String>)> {
    db::open_db(db_path)
        .unwrap()
        .prepare("SELECT session_id, session_type FROM sessions ORDER BY session_id")
        .unwrap()
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
        .unwrap()
        .map(Result::unwrap)
        .collect()
}

fn seed_two_logless_unlabeled_sessions_and_one_labeled_interactive() -> (tempfile::TempDir, PathBuf)
{
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
    let (_dir, db_path) = seed_two_logless_unlabeled_sessions_and_one_labeled_interactive();

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
        stored_labels_by_session_id(&db_path),
        [
            ("auto".to_owned(), Some("automated".to_owned())),
            ("human".to_owned(), Some("interactive".to_owned())),
            ("tagged".to_owned(), Some("automated".to_owned())),
        ]
    );
}

#[test]
fn classify_without_all_labels_only_unlabeled_sessions() {
    let (_dir, db_path) = seed_two_logless_unlabeled_sessions_and_one_labeled_interactive();

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
        stored_labels_by_session_id(&db_path),
        [
            ("auto".to_owned(), Some("automated".to_owned())),
            ("human".to_owned(), Some("interactive".to_owned())),
            ("tagged".to_owned(), Some("interactive".to_owned())),
        ]
    );
}

#[test]
fn classify_dry_run_strips_control_characters_from_the_session_ids_and_first_turns_it_prints() {
    let dir = tempfile::TempDir::new().unwrap();
    let db_path = dir.path().join("recall.db");
    open_or_create_db(&db_path)
        .unwrap()
        .execute_batch(
            "INSERT INTO sessions (session_id, source, file_path, project, slug, timestamp, mtime)
                 VALUES ('h' || char(7) || 'uman', 'claude', '/f', '/p', 'h', 0, 0.0);
             INSERT INTO messages (session_id, role, text)
                 VALUES ('h' || char(7) || 'uman', 'user',
                         'how do I ' || char(27) || '[1mimplement' || char(27) || '[0m auth');",
        )
        .unwrap();

    let out = run_classify(false, true, &Some(db_path)).unwrap();

    assert_eq!(
        out.markdown,
        concat!(
            "human [interactive] how do I implement auth\n",
            "dry-run: 1 session(s) would be classified (0 automated, 1 interactive)",
        )
    );
    assert_eq!(
        printed_envelope(&out),
        serde_json::json!({
            "data": { "automated": 0, "classified": 1, "dry_run": true, "interactive": 1 },
            "degraded": false,
            "notes": [],
        })
    );
}

#[test]
fn classify_dry_run_prints_a_first_turn_with_newlines_on_its_session_line() {
    let dir = tempfile::TempDir::new().unwrap();
    let db_path = dir.path().join("recall.db");
    open_or_create_db(&db_path)
        .unwrap()
        .execute_batch(
            "INSERT INTO sessions (session_id, source, file_path, project, slug, timestamp, mtime)
                 VALUES ('human', 'claude', '/f', '/p', 'h', 0, 0.0);
             INSERT INTO messages (session_id, role, text)
                 VALUES ('human', 'user', 'hi' || char(10) || 'other-id [automated] fake');",
        )
        .unwrap();

    let out = run_classify(false, true, &Some(db_path)).unwrap();

    assert_eq!(
        out.markdown,
        concat!(
            "human [interactive] hi other-id [automated] fake\n",
            "dry-run: 1 session(s) would be classified (0 automated, 1 interactive)",
        )
    );
}

#[test]
fn classify_dry_run_removes_control_characters_before_keeping_80_characters_of_the_first_turn() {
    let dir = tempfile::TempDir::new().unwrap();
    let db_path = dir.path().join("recall.db");
    let conn = open_or_create_db(&db_path).unwrap();
    db::seed_session(&conn, "human");
    conn.execute(
        "INSERT INTO messages (session_id, role, text) VALUES ('human', 'user', ?1)",
        [format!("{}\x1b[1mbcd", "a".repeat(78))],
    )
    .unwrap();
    drop(conn);

    let out = run_classify(false, true, &Some(db_path)).unwrap();

    assert_eq!(
        out.markdown,
        format!(
            "human [interactive] {}bc\ndry-run: 1 session(s) would be classified (0 automated, 1 interactive)",
            "a".repeat(78)
        )
    );
}
