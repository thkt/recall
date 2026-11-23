use std::fs;

use super::*;
use crate::embedder;

#[cfg(unix)]
#[test]
fn create_db_file_sets_owner_only_0600_mode() {
    use std::os::unix::fs::PermissionsExt;

    let dir = tempfile::TempDir::new().unwrap();
    let db_path = dir.path().join("index.db");

    create_db_file(&db_path).unwrap();

    let mode = fs::metadata(&db_path).unwrap().permissions().mode();
    assert_eq!(
        mode & 0o777,
        0o600,
        "created DB must be owner-only 0o600, got {:o}",
        mode & 0o777,
    );
}

#[test]
fn create_db_file_creates_missing_parent_dirs() {
    let dir = tempfile::TempDir::new().unwrap();
    let db_path = dir.path().join("recall").join("recall.db");

    create_db_file(&db_path).unwrap();

    assert!(
        db_path.exists(),
        "DB file must be created under a fresh parent dir"
    );
}

#[test]
fn create_db_file_errors_when_parent_cannot_be_created() {
    let dir = tempfile::TempDir::new().unwrap();
    let blocker = dir.path().join("not_a_dir");
    create_db_file(&blocker).unwrap();
    let db_path = blocker.join("sub").join("recall.db");

    let err = create_db_file(&db_path).unwrap_err();

    assert_ne!(
        err.kind(),
        ErrorKind::AlreadyExists,
        "expected a parent-creation failure, got {err:?}"
    );
}

#[test]
fn opening_an_index_for_another_model_for_writing_fails_before_changing_its_schema_or_journal_mode()
{
    let dir = tempfile::TempDir::new().unwrap();
    let db_path = dir.path().join("recall.db");
    let other_model = Connection::open(&db_path).unwrap();
    other_model
        .execute_batch(
            "CREATE TABLE embedding_identity (singleton INTEGER PRIMARY KEY, fingerprint TEXT, manifest TEXT);
             INSERT INTO embedding_identity VALUES (1, 'another model', '{}');",
        )
        .unwrap();
    drop(other_model);

    open_or_create_db(&db_path).unwrap_err();

    let conn = Connection::open(&db_path).unwrap();
    let tables = db::collect_strings(&conn, "SELECT name FROM sqlite_master ORDER BY name");
    let journal_mode: String = conn
        .query_row("PRAGMA journal_mode", [], |r| r.get(0))
        .unwrap();
    assert_eq!(
        (tables, journal_mode),
        (vec!["embedding_identity".to_owned()], "delete".to_owned())
    );
}

#[test]
fn resolve_default_db_path_uses_local_share_when_xdg_unset() {
    let home = Path::new("/home/u");

    let path = resolve_default_db_path(None, Some(home)).unwrap();

    assert_eq!(
        path,
        Path::new("/home/u/.local/share/recall/indexes")
            .join(model::selected().identity())
            .join("recall.db")
    );
}

#[test]
fn resolve_default_db_path_prefers_absolute_xdg_data_home() {
    let path =
        resolve_default_db_path(Some(OsStr::new("/xdg/data")), Some(Path::new("/home/u"))).unwrap();

    assert_eq!(
        path,
        Path::new("/xdg/data/recall/indexes")
            .join(model::selected().identity())
            .join("recall.db")
    );
}

#[test]
fn resolve_default_db_path_ignores_relative_xdg_data_home() {
    let path = resolve_default_db_path(
        Some(OsStr::new("relative/data")),
        Some(Path::new("/home/u")),
    )
    .unwrap();

    assert_eq!(
        path,
        Path::new("/home/u/.local/share/recall/indexes")
            .join(model::selected().identity())
            .join("recall.db")
    );
}

#[test]
fn resolve_db_path_passes_through_explicit_path() {
    let explicit = PathBuf::from("/tmp/custom/recall.db");

    let path = resolve_db_path(&Some(explicit.clone())).unwrap();

    assert_eq!(path, explicit);
}

#[test]
fn resolve_db_path_defaults_to_recall_db_under_data_dir() {
    let path = resolve_db_path(&None).unwrap();

    assert!(
        path.ends_with(
            Path::new("recall/indexes")
                .join(model::selected().identity())
                .join("recall.db")
        ),
        "default path must include the selected model identity, got {path:?}"
    );
}

#[test]
fn try_load_embedder_cached_reports_reason_aware_notes_once() {
    for (reason, expected) in [
        (
            DegradedReason::NotInstalled,
            vec![
                "note: embedding model not installed; run `recall model download` to enable semantic search.",
            ],
        ),
        (
            DegradedReason::ProbeFailed,
            vec!["note: embedding model unavailable; results from text search only."],
        ),
        (
            DegradedReason::BackendUnavailable,
            vec!["note: embedding model unavailable; results from text search only."],
        ),
        (DegradedReason::Disabled, vec![]),
    ] {
        let mut notes = Vec::new();
        let result = try_load_embedder_cached_with_reporter(
            || Err(reason),
            |msg| notes.push(msg.to_owned()),
        );

        assert_eq!(result.err(), Some(reason));
        assert_eq!(notes, expected, "reason: {reason:?}");
    }
}

#[test]
fn search_degraded_note_covers_backend_failures() {
    assert_eq!(
        search_degraded_note(DegradedReason::BackendUnavailable).as_deref(),
        Some("note: embedding model unavailable; results from text search only.")
    );
    assert_eq!(search_degraded_note(DegradedReason::Disabled), None);
}

#[test]
fn try_load_embedder_cached_returns_loaded_embedder() {
    let loaded: Arc<dyn Embed> = Arc::new(embedder::MockEmbedder::new());
    let returned = try_load_embedder_cached_with(|| Ok(Arc::clone(&loaded)))
        .expect("a loaded embedder must be returned");
    assert!(
        Arc::ptr_eq(&returned, &loaded),
        "the returned embedder must be the one the loader opened"
    );
}
