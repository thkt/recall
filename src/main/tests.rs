#![cfg(test)]
use std::assert_matches;
use std::fs;

use super::*;
use crate::commands::search::{SearchArgs, run_search_with};
use crate::commands::show::{DEFAULT_WINDOW, run_show};
use crate::commands::status::run_status;

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
fn test_format_timestamp() {
    for (input, expected) in [
        (None, "unknown"),
        (Some(0), "1970-01-01"),
        (Some(1709251200000), "2024-03-01"),
        (Some(-date::MS_PER_DAY), "1969-12-31"),
    ] {
        assert_eq!(format_timestamp(input), expected, "input: {input:?}");
    }
}

#[test]
fn try_load_embedder_cached_records_degraded_non_install_failure() {
    let result = try_load_embedder_cached_with(|| Err(DegradedReason::ProbeFailed));
    assert!(result.is_err());
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
    let result = try_load_embedder_cached_with(|| Ok(loaded));
    assert!(result.is_ok(), "a loaded embedder must be returned");
}

#[test]
fn try_load_embedder_cached_returns_err_when_model_not_installed() {
    let result = try_load_embedder_cached_with(|| Err(DegradedReason::NotInstalled));
    assert!(
        result.is_err(),
        "a not-installed model must degrade (Err), not abort"
    );
}

struct FailingWriter;

impl Write for FailingWriter {
    fn write(&mut self, _buf: &[u8]) -> io::Result<usize> {
        Err(io::Error::other("stdout unavailable"))
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[test]
fn emit_success_propagates_non_pipe_write_error() {
    let out = CommandOutput::ok("hello".to_owned(), serde_json::Value::Null);
    let mut writer = FailingWriter;

    let err = emit_success_to(&out, false, &mut writer).unwrap_err();

    assert_eq!(err.kind(), ErrorKind::Other);
}

#[test]
fn exit_code_for_write_maps_write_failure_to_io_error() {
    let code = exit_code_for_write(Err(io::Error::other("stdout unavailable")), false);
    assert_eq!(code, ExitCode::from(codes::IO_ERR));
}

#[test]
fn exit_code_for_write_maps_written_to_success() {
    let code = exit_code_for_write(Ok(WriteOutcome::Written), false);
    assert_eq!(code, ExitCode::SUCCESS);
}

#[test]
fn test_search_requires_query_or_file_at_parse_time() {
    assert!(Cli::try_parse_from(["recall", "search"]).is_err());
    assert!(Cli::try_parse_from(["recall", "search", "--file", "a.rs"]).is_ok());
    assert!(Cli::try_parse_from(["recall", "search", "auth"]).is_ok());
}

#[test]
fn test_rebuild_is_a_distinct_subcommand() {
    let cli = Cli::try_parse_from(["recall", "rebuild"]).unwrap();
    assert_matches!(cli.command, Some(Command::Rebuild { .. }));
}

#[test]
fn test_index_no_longer_accepts_force_flag() {
    assert!(Cli::try_parse_from(["recall", "index", "--force"]).is_err());
    let cli = Cli::try_parse_from(["recall", "index"]).unwrap();
    assert_matches!(cli.command, Some(Command::Index { .. }));
}

#[test]
fn index_token_budget_flag_clamps_above_256000_ceiling_down() {
    let cli = Cli::try_parse_from(["recall", "index", "--token-budget", "500000"]).unwrap();
    let Some(Command::Index { embed_flags }) = cli.command else {
        panic!("expected Command::Index with a token_budget field");
    };
    assert_eq!(
        embed_flags.token_budget,
        Some(256_000),
        "500_000 exceeds the 256_000 ceiling and must clamp down, never up"
    );
}

#[test]
fn test_rebuild_recognized_by_shorthand_expansion() {
    let args: Vec<OsString> = ["recall", "rebuild"].iter().map(OsString::from).collect();
    assert!(try_expand_shorthand(&args, KNOWN_SUBCOMMANDS, GLOBAL_FLAGS).is_none());
}

#[test]
fn doctor_recognized_by_shorthand_expansion() {
    let args: Vec<OsString> = ["recall", "doctor"].iter().map(OsString::from).collect();
    assert!(try_expand_shorthand(&args, KNOWN_SUBCOMMANDS, GLOBAL_FLAGS).is_none());
    let cli = Cli::try_parse_from(["recall", "doctor"]).unwrap();
    assert_matches!(cli.command, Some(Command::Doctor { fix: false }));
}

#[test]
fn doctor_fix_flag_parses() {
    let cli = Cli::try_parse_from(["recall", "doctor", "--fix"]).unwrap();
    assert_matches!(cli.command, Some(Command::Doctor { fix: true }));
}

#[test]
fn embed_token_shorthand_expands_to_search() {
    let args: Vec<OsString> = ["recall", "embed"].iter().map(OsString::from).collect();
    assert!(try_expand_shorthand(&args, KNOWN_SUBCOMMANDS, GLOBAL_FLAGS).is_some());
}

#[test]
fn known_subcommands_covers_every_clap_subcommand() {
    use clap::CommandFactory;
    for sub in Cli::command().get_subcommands() {
        let name = sub.get_name();
        assert!(
            KNOWN_SUBCOMMANDS.contains(&name),
            "clap subcommand `{name}` is missing from KNOWN_SUBCOMMANDS; \
                 shorthand expansion will rewrite `recall {name}` to a search query"
        );
    }
}

#[test]
fn readme_describes_index_time_embedding_not_post_search_growth() {
    let en = fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/README.md"))
        .expect("README.md must exist at the crate root");
    let ja = fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/README.ja.md"))
        .expect("README.ja.md must exist at the crate root");
    for (name, doc) in [("README.md", en.as_str()), ("README.ja.md", ja.as_str())] {
        for banned in [
            "post-search embedding",
            "Progressive embedding",
            "--no-embed",
        ] {
            assert!(
                !doc.contains(banned),
                "{name} still references the retired model: {banned:?}"
            );
        }
    }
}

fn seed_classify_db() -> (tempfile::TempDir, Connection) {
    let (dir, conn) = db::setup_test_db();
    conn.execute(
            "INSERT INTO sessions (session_id, source, file_path, project, slug, timestamp, mtime, session_type) VALUES ('auto', 'claude', '/f', '/p', 'a', 0, 0.0, NULL)",
            [],
        )
        .unwrap();
    conn.execute(
            "INSERT INTO messages (session_id, role, text) VALUES ('auto', 'user', '<command-message>clear</command-message>')",
            [],
        )
        .unwrap();
    conn.execute(
            "INSERT INTO sessions (session_id, source, file_path, project, slug, timestamp, mtime, session_type) VALUES ('human', 'claude', '/f', '/p', 'h', 0, 0.0, NULL)",
            [],
        )
        .unwrap();
    conn.execute(
            "INSERT INTO messages (session_id, role, text) VALUES ('human', 'user', 'how do I implement auth')",
            [],
        )
        .unwrap();
    (dir, conn)
}

fn session_type_of(conn: &Connection, sid: &str) -> Option<String> {
    conn.query_row(
        "SELECT session_type FROM sessions WHERE session_id = ?1",
        [sid],
        |r| r.get(0),
    )
    .unwrap()
}

#[test]
fn test_010_reclassify_all_tags_sessions() {
    let (_dir, mut conn) = seed_classify_db();
    let outcomes = reclassify_sessions(&mut conn, true, false).unwrap();
    assert_eq!(session_type_of(&conn, "auto").as_deref(), Some("automated"));
    assert_eq!(
        session_type_of(&conn, "human").as_deref(),
        Some("interactive")
    );
    assert_eq!(
        outcomes.len(),
        2,
        "both sessions with a user turn are classified"
    );
}

#[test]
fn reclassify_incremental_skips_already_tagged() {
    let (_dir, mut conn) = db::setup_test_db();
    conn.execute(
            "INSERT INTO sessions (session_id, source, file_path, project, slug, timestamp, mtime, session_type) VALUES ('fresh', 'claude', '/f', '/p', 'f', 0, 0.0, NULL)",
            [],
        )
        .unwrap();
    conn.execute(
            "INSERT INTO messages (session_id, role, text) VALUES ('fresh', 'user', '<command-message>clear</command-message>')",
            [],
        )
        .unwrap();
    conn.execute(
            "INSERT INTO sessions (session_id, source, file_path, project, slug, timestamp, mtime, session_type) VALUES ('tagged', 'claude', '/f', '/p', 't', 0, 0.0, 'interactive')",
            [],
        )
        .unwrap();
    conn.execute(
            "INSERT INTO messages (session_id, role, text) VALUES ('tagged', 'user', '<command-message>clear</command-message>')",
            [],
        )
        .unwrap();

    let outcomes = reclassify_sessions(&mut conn, false, false).unwrap();

    assert_eq!(
        session_type_of(&conn, "fresh").as_deref(),
        Some("automated"),
        "an unclassified session is classified"
    );
    assert_eq!(
        session_type_of(&conn, "tagged").as_deref(),
        Some("interactive"),
        "an already-tagged session is untouched without --all"
    );
    assert_eq!(
        outcomes.len(),
        1,
        "only the unclassified session is processed"
    );
}

pub(crate) fn vec_chunk_count(conn: &Connection) -> i64 {
    conn.query_row("SELECT COUNT(*) FROM vec_chunks", [], |r| r.get(0))
        .unwrap()
}

pub(crate) fn mock_loader() -> Result<Arc<dyn Embed>, DegradedReason> {
    Ok(Arc::new(embedder::MockEmbedder::new()) as Arc<dyn Embed>)
}

pub(crate) fn seed_healthy_index(conn: &Connection) {
    db::seed_session(conn, "s1");
    db::seed_chunk(conn, 1, "healthy chunk");
    let emb = embedder::MockEmbedder::deterministic_vector("healthy chunk");
    conn.execute(
        "INSERT INTO vec_chunks (embedding, chunk_id, sub_idx) VALUES (?1, 1, 0)",
        rusqlite::params![embedder::f32_as_bytes(&emb)],
    )
    .unwrap();
}

pub(crate) fn bare_empty_db(dir: &Path) -> PathBuf {
    let db_path = dir.join("recall.db");
    drop(rusqlite::Connection::open(&db_path).unwrap());
    db_path
}

pub(crate) fn search_command(query: &str) -> SearchArgs {
    SearchArgs {
        query: Some(query.to_owned()),
        project: None,
        days: None,
        source: None,
        file: None,
        limit: 10,
        exclude_current: false,
        include_current: false,
        only_current: false,
        include_automated: false,
    }
}

#[test]
fn read_commands_use_pre_generation_index_without_migrating_it() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("recall.db");
    let conn = open_or_create_db(&path).unwrap();
    conn.execute_batch(
        "DROP TRIGGER qa_chunks_insert_generation;
             DROP TRIGGER qa_chunks_update_generation;
             DROP TABLE qa_chunk_generation;
             ALTER TABLE qa_chunks DROP COLUMN generation;",
    )
    .unwrap();
    db::seed_session(&conn, "s1");
    let content = "synthetic legacy content";
    db::seed_chunk(&conn, 1, content);
    conn.execute(
        "INSERT INTO messages (session_id, role, text) VALUES ('s1', 'user', ?1)",
        [content],
    )
    .unwrap();
    let vector = embedder::MockEmbedder::deterministic_vector(content);
    conn.execute(
        "INSERT INTO vec_chunks (embedding, chunk_id, sub_idx) VALUES (?1, 1, 0)",
        [embedder::f32_as_bytes(&vector)],
    )
    .unwrap();
    let schema_version: i64 = conn
        .query_row("PRAGMA schema_version", [], |r| r.get(0))
        .unwrap();
    drop(conn);

    let db_path = Some(path.clone());
    let status = run_status(false, &db_path).unwrap();
    assert_eq!(status.data["sessions"], 1);
    assert_eq!(status.data["qa_chunks"], 1);
    assert_eq!(status.data["embedded"], 1);
    let show = run_show("s1", false, None, DEFAULT_WINDOW, &db_path).unwrap();
    assert!(show.markdown.contains(content));
    let search = run_search_with(search_command(content), &db_path, mock_loader).unwrap();
    assert!(!search.degraded);
    let results = search.data["results"].as_array().unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0]["session_id"], "s1");
    assert!(search.markdown.contains(content));

    let (conn, _) = db::open_db_readonly(&path).unwrap();
    let after: i64 = conn
        .query_row("PRAGMA schema_version", [], |r| r.get(0))
        .unwrap();
    assert_eq!(
        after, schema_version,
        "read commands must not migrate the DB"
    );
    assert_eq!(vec_chunk_count(&conn), 1);
    let saved: Vec<u8> = conn
        .query_row("SELECT embedding FROM vec_chunks", [], |r| r.get(0))
        .unwrap();
    assert_eq!(saved, embedder::f32_as_bytes(&vector));
}

pub(crate) fn payload_keys(v: &serde_json::Value) -> Vec<String> {
    let mut keys: Vec<String> = v
        .as_object()
        .expect("payload must be a JSON object")
        .keys()
        .cloned()
        .collect();
    keys.sort();
    keys
}

#[test]
fn search_result_and_session_data_stay_non_serialize() {
    fn leading_block_mentions_serialize(src: &str, decl: &str) -> bool {
        let pos = src.find(decl).unwrap_or_else(|| {
            panic!("{decl} not found — the Serialize guard is reading the wrong source")
        });
        let block_start = src[..pos].rfind(['}', ';']).map(|i| i + 1).unwrap_or(0);
        src[block_start..pos].contains("Serialize")
    }

    assert!(
        !leading_block_mentions_serialize(include_str!("../search.rs"), "pub struct SearchResult"),
        "SearchResult must not derive Serialize (ADR-0001): the --json schema is built explicitly"
    );
    assert!(
        !leading_block_mentions_serialize(
            include_str!("../parser/mod.rs"),
            "pub struct SessionData"
        ),
        "SessionData must not derive Serialize (ADR-0001): the --json schema is built explicitly"
    );
}

#[test]
fn require_current_schema_stale_is_data_error_naming_rebuild() {
    let err = require_current_schema(db::SchemaState::Stale).unwrap_err();
    assert!(
        matches!(err, RecallError::DataError(_)),
        "stale schema must be a DataError, got {err:?}"
    );
    assert!(
        err.to_string().contains("recall rebuild"),
        "message must name the fix: {err}"
    );
    assert_eq!(err.error_code().code(), 65);
}

#[test]
fn require_current_schema_current_is_ok() {
    assert!(require_current_schema(db::SchemaState::Current).is_ok());
}

#[test]
fn require_current_schema_empty_is_ok() {
    assert!(require_current_schema(db::SchemaState::Empty).is_ok());
}

#[test]
fn emit_success_preserves_text_newlines_and_json_envelope() {
    for text in ["", "質問と回答", "answer\n", "answer\n\n"] {
        let out = CommandOutput::with_notes(
            text.to_owned(),
            serde_json::json!({"messages": [{"text": "回答\n\"引用\""}]}),
            true,
            vec!["diagnostic\nwith \"quotes\"".to_owned()],
        );
        let mut buf = Vec::new();
        assert_eq!(
            emit_success_to(&out, false, &mut buf).unwrap(),
            WriteOutcome::Written,
        );
        let expected = if text.is_empty() || text.ends_with('\n') {
            text.to_owned()
        } else {
            format!("{text}\n")
        };
        assert_eq!(buf, expected.as_bytes());

        buf.clear();
        assert_eq!(
            emit_success_to(&out, true, &mut buf).unwrap(),
            WriteOutcome::Written,
        );
        let envelope: serde_json::Value = serde_json::from_slice(&buf).unwrap();
        assert_eq!(
            envelope,
            serde_json::json!({"data": out.data, "degraded": true, "notes": out.notes}),
        );
        assert_eq!(buf.last(), Some(&b'\n'));
    }
}
