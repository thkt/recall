#![cfg(test)]
use std::assert_matches;
use std::fs;

use crate::embedding::{ChunkedEmbedding, EmbedError};

use super::*;
use std::cell::Cell;

#[test]
fn doctor_fix_model_mismatch_preserves_index_and_never_suggests_deletion() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("index.db");
    {
        let conn = open_or_create_db(&path).unwrap();
        conn.execute(
            "UPDATE embedding_identity SET fingerprint='previous-model'",
            [],
        )
        .unwrap();
    }
    let out = run_doctor_with(false, true, &Some(path.clone()), mock_loader, || {
        panic!("model artifacts must not be replaced for an index identity mismatch")
    })
    .unwrap();
    assert_eq!(out.data["healthy"], false);
    assert!(out.degraded);
    assert!(
        out.notes
            .iter()
            .any(|n| n.contains("matching --model-manifest"))
    );
    assert!(out.notes.iter().all(|n| !n.contains("remove the database")));
    let (conn, _) = db::open_db_readonly(&path).unwrap();
    assert_eq!(
        conn.query_row("SELECT fingerprint FROM embedding_identity", [], |r| r
            .get::<_, String>(
            0
        ))
        .unwrap(),
        "previous-model"
    );
}

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
fn resolve_current_session_truth_table() {
    use search::CurrentSession;
    let flags = |exclude, include, only| CurrentSessionFlags {
        exclude,
        include,
        only,
    };
    let with_env = || Some("sess-x".to_owned());
    let exclude_x = || CurrentSession::Exclude("sess-x".to_owned());

    assert_eq!(
        resolve_current_session(flags(false, false, false), with_env()).unwrap(),
        exclude_x()
    );
    assert_eq!(
        resolve_current_session(flags(false, false, false), None).unwrap(),
        CurrentSession::Ignore
    );
    assert_eq!(
        resolve_current_session(flags(true, false, false), with_env()).unwrap(),
        exclude_x()
    );
    assert_eq!(
        resolve_current_session(flags(true, false, false), None).unwrap(),
        CurrentSession::Ignore
    );
    assert_eq!(
        resolve_current_session(flags(false, true, false), with_env()).unwrap(),
        CurrentSession::Ignore
    );
    assert_eq!(
        resolve_current_session(flags(false, true, false), None).unwrap(),
        CurrentSession::Ignore
    );
    assert_eq!(
        resolve_current_session(flags(false, false, true), with_env()).unwrap(),
        CurrentSession::Only("sess-x".to_owned())
    );
    assert!(
        resolve_current_session(flags(false, false, true), None).is_err(),
        "--only-current without CLAUDE_CODE_SESSION_ID must be a usage error"
    );
    assert_eq!(
        resolve_current_session(flags(false, false, false), Some("  ".to_owned())).unwrap(),
        CurrentSession::Ignore,
        "blank CLAUDE_CODE_SESSION_ID must not produce a degenerate Exclude"
    );
    assert!(
        resolve_current_session(flags(false, false, true), Some(String::new())).is_err(),
        "--only-current with a blank env value must still be a usage error"
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
fn search_fallback_notes_are_reason_aware() {
    assert_eq!(
        search_fallback_notes(DegradedReason::NotInstalled),
        [
            "semantic search unavailable: embedding model not installed; run `recall model download` to enable semantic search"
        ]
    );
    assert_eq!(
        search_fallback_notes(DegradedReason::ProbeFailed),
        ["semantic search unavailable: embedding model unavailable; results from text search only"]
    );
    assert_eq!(
        search_fallback_notes(DegradedReason::BackendUnavailable),
        ["semantic search unavailable: embedding model unavailable; results from text search only"]
    );
    assert!(
        search_fallback_notes(DegradedReason::Disabled).is_empty(),
        "a disabled model is intentional, not degraded: no note"
    );
}

#[test]
fn search_degraded_state_loaded_embedder_is_not_degraded() {
    let loaded: Arc<dyn Embed> = Arc::new(embedder::MockEmbedder::new());
    let (degraded, notes) = search_degraded_state(&Ok(loaded));
    assert!(!degraded, "a loaded embedder must not be marked degraded");
    assert!(notes.is_empty(), "the ideal path carries no notes");
}

#[test]
fn search_degraded_state_reason_failure_degrades_with_note() {
    let (degraded, notes) = search_degraded_state(&Err(DegradedReason::ProbeFailed));
    assert!(degraded);
    assert_eq!(
        notes,
        ["semantic search unavailable: embedding model unavailable; results from text search only"]
    );
}

#[test]
fn search_degraded_state_disabled_is_not_degraded() {
    let (degraded, notes) = search_degraded_state(&Err(DegradedReason::Disabled));
    assert!(!degraded, "a disabled model is intentional, not degraded");
    assert!(notes.is_empty());
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

fn make_search_result(session_id: &str, project: &str, excerpt: &str) -> search::SearchResult {
    search::SearchResult {
        session: parser::SessionData {
            session_id: session_id.to_owned(),
            source: parser::Source::Claude,
            file_path: "/path/to/file.jsonl".to_owned(),
            project: project.to_owned(),
            slug: "test-slug".to_owned(),
            timestamp: Some(1709251200000),
        },
        excerpt: excerpt.to_owned(),
        chunk_id: None,
    }
}

#[test]
fn test_format_result_basic() {
    let r = make_search_result("s1", "/home/me/project", "some excerpt");
    let mut buf = Vec::new();
    format_result(&mut buf, 0, &r).unwrap();
    let output = String::from_utf8(buf).unwrap();
    assert!(output.contains("[1] 2024-03-01"));
    assert!(output.contains("test-slug"));
    assert!(output.contains("project [claude]"));
    assert!(output.contains("ID: s1"));
    assert!(output.contains("> some excerpt"));
}

#[test]
fn test_render_results_empty_reports_no_match() {
    assert_eq!(render_results(&[]), "No matching sessions found.");
}

#[test]
fn test_render_results_lists_header_and_each_result() {
    let results = [
        make_search_result("abc123", "/proj/a", "hello world"),
        make_search_result("def456", "/proj/b", "another match"),
    ];
    let out = render_results(&results);
    assert!(out.starts_with("Found 2 sessions:\n"), "got: {out}");
    assert!(out.contains("ID: abc123"), "got: {out}");
    assert!(out.contains("ID: def456"), "got: {out}");
}

#[test]
fn test_format_result_empty_project() {
    let r = make_search_result("s1", "", "excerpt");
    let mut buf = Vec::new();
    format_result(&mut buf, 0, &r).unwrap();
    let output = String::from_utf8(buf).unwrap();
    assert!(output.contains("unknown [claude]"));
    assert!(!output.contains("    /"));
}

#[test]
fn test_format_result_excerpt_not_truncated() {
    let long = "a".repeat(300);
    let r = make_search_result("s1", "/proj", &long);
    let mut buf = Vec::new();
    format_result(&mut buf, 0, &r).unwrap();
    let output = String::from_utf8(buf).unwrap();
    assert!(
        output.contains(&format!("> {long}")),
        "the full excerpt must render without truncation"
    );
    assert!(
        !output.contains("> ..."),
        "no truncation marker should be emitted, got: {output}"
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
fn test_format_result_excerpt_strips_control_chars() {
    let r = make_search_result("s1", "/proj", "clean\x1b[31mANSI\x1b[0mhere\x07bell");
    let mut buf = Vec::new();
    format_result(&mut buf, 0, &r).unwrap();
    let output = String::from_utf8(buf).unwrap();
    assert!(
        !output.contains('\x1b') && !output.contains('\x07'),
        "control characters must be stripped from the excerpt, got: {output:?}"
    );
    assert!(
        output.contains("> cleanANSIherebell"),
        "the cleaned excerpt text must render, got: {output:?}"
    );
}

#[test]
fn test_format_result_empty_excerpt() {
    let r = make_search_result("s1", "/proj", "");
    let mut buf = Vec::new();
    format_result(&mut buf, 0, &r).unwrap();
    let output = String::from_utf8(buf).unwrap();
    assert!(!output.contains("> "));
}

#[test]
fn test_search_idle_message_only_when_unindexed() {
    assert_eq!(
        search_idle_message(0),
        Some("No sessions indexed. Run `recall index` first.")
    );
    assert_eq!(search_idle_message(5), None);
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

#[test]
fn test_subagent_file_path_shows_parent_session() {
    let r = search::SearchResult {
        session: parser::SessionData {
            session_id: "agent-a58c408".to_owned(),
            source: parser::Source::Claude,
            file_path: "/home/.claude/projects/proj/abc-def-123/subagents/agent-a58c408.jsonl"
                .to_owned(),
            project: "/proj".to_owned(),
            slug: "agent-slug".to_owned(),
            timestamp: Some(1709251200000),
        },
        excerpt: "some text".to_owned(),
        chunk_id: None,
    };
    let mut buf = Vec::new();
    format_result(&mut buf, 0, &r).unwrap();
    let output = String::from_utf8(buf).unwrap();
    assert!(output.contains("abc-def-123"));
}

#[test]
fn test_normal_session_path_shows_no_parent() {
    let r = make_search_result("s1", "/home/me/project", "excerpt");
    let mut buf = Vec::new();
    format_result(&mut buf, 0, &r).unwrap();
    let output = String::from_utf8(buf).unwrap();
    assert!(!output.contains("Parent:"));
}

fn setup_show_db() -> (tempfile::TempDir, Connection) {
    let (dir, conn) = db::setup_test_db();
    conn.execute(
            "INSERT INTO sessions (session_id, source, file_path, project, slug, timestamp, mtime) VALUES ('abc-123', 'claude', '/path/f.jsonl', '/proj', 'my-slug', 1709251200000, 0.0)",
            [],
        )
        .unwrap();
    conn.execute(
            "INSERT INTO sessions (session_id, source, file_path, project, slug, timestamp, mtime) VALUES ('abc-456', 'claude', '/path/g.jsonl', '/proj', 'other-slug', 1709251200000, 0.0)",
            [],
        )
        .unwrap();
    conn.execute(
        "INSERT INTO messages (session_id, role, text) VALUES ('abc-123', 'user', 'hello world')",
        [],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO messages (session_id, role, text) VALUES ('abc-123', 'assistant', 'hi there')",
        [],
    )
    .unwrap();
    (dir, conn)
}

#[test]
fn test_show_session_exact_match() {
    let (_dir, conn) = setup_show_db();
    let out = show_session(&conn, "abc-123", false).unwrap();
    assert!(out.markdown.contains("# my-slug"));
    assert!(out.markdown.contains("- session_id: abc-123"));
    assert!(out.markdown.contains("- date: 2024-03-01"));
    assert!(out.markdown.contains("## [user]\nhello world"));
    assert!(out.markdown.contains("## [assistant]\nhi there"));
    assert_eq!(out.data["session_id"], "abc-123");
    assert_eq!(out.data["messages"][0]["role"], "user");
    assert_eq!(out.data["messages"][0]["text"], "hello world");
}

#[test]
fn test_show_session_prefix_match() {
    let (_dir, conn) = setup_show_db();
    let out = show_session(&conn, "abc-1", false).unwrap();
    assert!(out.markdown.contains("- session_id: abc-123"));
}

#[test]
fn test_show_session_ambiguous_prefix() {
    let (_dir, conn) = setup_show_db();
    let err = show_session(&conn, "abc", false).unwrap_err();
    assert!(err.to_string().contains("Multiple sessions match"));
}

#[test]
fn test_show_session_not_found() {
    let (_dir, conn) = setup_show_db();
    let err = show_session(&conn, "zzz", false).unwrap_err();
    assert!(err.to_string().contains("No session found matching"));
}

#[test]
fn test_show_session_verbose_includes_file() {
    let (_dir, conn) = setup_show_db();
    let out = show_session(&conn, "abc-123", true).unwrap();
    assert!(out.markdown.contains("- file: /path/f.jsonl"));
}

#[test]
fn test_show_session_message_order() {
    let (_dir, conn) = setup_show_db();
    let out = show_session(&conn, "abc-123", false).unwrap();
    let user_pos = out.markdown.find("## [user]").unwrap();
    let assistant_pos = out.markdown.find("## [assistant]").unwrap();
    assert!(
        user_pos < assistant_pos,
        "user message should come before assistant"
    );
}

#[test]
fn show_session_unknown_source_falls_back_to_claude() {
    let (_dir, conn) = db::setup_test_db();
    conn.execute(
            "INSERT INTO sessions (session_id, source, file_path, project, slug, timestamp, mtime) VALUES ('xyz-1', 'bogus', '/f', '/p', 'odd-slug', 1709251200000, 0.0)",
            [],
        )
        .unwrap();
    let out = show_session(&conn, "xyz-1", false).unwrap();
    assert!(
        out.markdown.contains("- source: claude"),
        "an unknown stored source must fall back to claude, got: {}",
        out.markdown
    );
}

#[test]
fn window_returns_only_same_session_neighbors_within_n() {
    let (_dir, conn) = db::setup_test_db();
    db::seed_session(&conn, "sessA");
    db::seed_session(&conn, "sessB");
    for (sess, text) in [
        ("sessA", "AMSG0"),
        ("sessB", "BMSG0"),
        ("sessA", "AMSG1"),
        ("sessB", "BMSG1"),
        ("sessA", "AMSG2"),
        ("sessB", "BMSG2"),
        ("sessA", "AMSG3_BODY"),
        ("sessB", "BMSG3"),
        ("sessA", "AMSG4"),
        ("sessB", "BMSG4"),
        ("sessA", "AMSG5"),
        ("sessB", "BMSG5"),
        ("sessA", "AMSG6"),
        ("sessB", "BMSG6"),
    ] {
        conn.execute(
            "INSERT INTO messages (session_id, role, text) VALUES (?1, 'user', ?2)",
            rusqlite::params![sess, text],
        )
        .unwrap();
    }
    let body: i64 = conn
        .query_row(
            "SELECT rowid FROM messages WHERE session_id='sessA' AND text='AMSG3_BODY'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    conn.execute(
        "INSERT INTO qa_chunks (id, session_id, content, timestamp, src_rowid_lo, src_rowid_hi) \
             VALUES (100, 'sessA', 'AMSG3_BODY', 0, ?1, ?1)",
        [body],
    )
    .unwrap();

    let out = show_session_windowed(&conn, "sessA", false, Some(100), 2).unwrap();

    for included in ["AMSG1", "AMSG2", "AMSG3_BODY", "AMSG4", "AMSG5"] {
        assert!(
            out.markdown.contains(included),
            "window=2 must include same-session {included}, got: {}",
            out.markdown
        );
    }
    assert!(
        !out.markdown.contains("AMSG0"),
        "before-cap: the 3rd-earlier same-session message must be excluded, got: {}",
        out.markdown
    );
    assert!(
        !out.markdown.contains("AMSG6"),
        "after-cap: the 3rd-later same-session message must be excluded, got: {}",
        out.markdown
    );
    assert!(
        !out.markdown.contains("BMSG"),
        "must not cross into the adjacent session sharing interleaved rowids, got: {}",
        out.markdown
    );
}

#[test]
fn null_src_rowid_chunk_falls_back_to_body_plus_full_show_hint() {
    let (_dir, conn) = db::setup_test_db();
    db::seed_session(&conn, "s1");
    db::seed_chunk(&conn, 200, "legacy chunk body TEXTMARK");

    let out = show_session_windowed(&conn, "s1", false, Some(200), 3).unwrap();

    assert!(
        out.markdown.contains("legacy chunk body TEXTMARK"),
        "the legacy chunk's own body must be shown, got: {}",
        out.markdown
    );
    assert!(
        out.markdown.contains("recall show"),
        "must steer to the full `recall show` for the whole conversation, got: {}",
        out.markdown
    );
}

#[test]
fn show_without_chunk_renders_full_session() {
    let (_dir, conn) = setup_show_db();

    let out = show_session_windowed(&conn, "abc-123", false, None, 3).unwrap();

    assert!(
        out.markdown.contains("hello world"),
        "full render lost the user turn"
    );
    assert!(
        out.markdown.contains("hi there"),
        "full render lost the assistant turn"
    );
    assert!(out.markdown.contains("## [user]"));
    assert!(out.markdown.contains("## [assistant]"));
}

fn seed_window_db(count: i64) -> (tempfile::TempDir, Connection) {
    let (dir, conn) = db::setup_test_db();
    db::seed_session(&conn, "w1");
    for i in 0..count {
        conn.execute(
            "INSERT INTO messages (session_id, role, text) VALUES ('w1', 'user', ?1)",
            [format!("M{i}")],
        )
        .unwrap();
    }
    (dir, conn)
}

#[test]
fn windowed_show_errors_when_chunk_id_is_absent() {
    let (_dir, conn) = seed_window_db(3);
    let err = show_session_windowed(&conn, "w1", false, Some(999), 2).unwrap_err();
    assert!(
        err.to_string().contains("No chunk found with id 999"),
        "got: {err}"
    );
}

#[test]
fn windowed_show_errors_when_chunk_belongs_to_another_session() {
    let (_dir, conn) = seed_window_db(3);
    db::seed_session(&conn, "w2");
    conn.execute(
        "INSERT INTO qa_chunks (id, session_id, content, timestamp, src_rowid_lo, src_rowid_hi) \
             VALUES (50, 'w2', 'body', 0, 1, 1)",
        [],
    )
    .unwrap();

    let err = show_session_windowed(&conn, "w1", false, Some(50), 2).unwrap_err();

    assert!(
        err.to_string().contains("belongs to session 'w2'"),
        "cross-session --chunk must name the mismatched session, got: {err}"
    );
}

#[test]
fn windowed_show_truncates_a_message_longer_than_the_cap() {
    let (_dir, conn) = db::setup_test_db();
    db::seed_session(&conn, "w1");
    let long = "X".repeat(MAX_MSG_CHARS + 1);
    conn.execute(
        "INSERT INTO messages (session_id, role, text) VALUES ('w1', 'user', ?1)",
        [&long],
    )
    .unwrap();
    let rowid: i64 = conn
        .query_row(
            "SELECT rowid FROM messages WHERE session_id='w1'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    conn.execute(
        "INSERT INTO qa_chunks (id, session_id, content, timestamp, src_rowid_lo, src_rowid_hi) \
             VALUES (60, 'w1', 'body', 0, ?1, ?1)",
        [rowid],
    )
    .unwrap();

    let out = show_session_windowed(&conn, "w1", false, Some(60), 0).unwrap();

    assert!(
        out.markdown.contains('…'),
        "an over-cap message must be marked truncated"
    );
    assert!(
        !out.markdown.contains(&long),
        "the full over-cap message must not be rendered verbatim"
    );
}

#[test]
fn windowed_show_renders_metadata_only_when_range_matches_no_message() {
    let (_dir, conn) = seed_window_db(2);
    conn.execute(
        "INSERT INTO qa_chunks (id, session_id, content, timestamp, src_rowid_lo, src_rowid_hi) \
             VALUES (70, 'w1', 'orphan-range', 0, 100000, 100001)",
        [],
    )
    .unwrap();

    let out = show_session_windowed(&conn, "w1", false, Some(70), 2).unwrap();

    assert!(
        out.markdown.contains("- chunk: 70"),
        "metadata header must still render, got: {}",
        out.markdown
    );
    assert!(
        !out.markdown.contains("## ["),
        "no message body should render for an out-of-range chunk, got: {}",
        out.markdown
    );
}

#[test]
fn windowed_show_json_payload_lists_anchored_messages_with_in_chunk_flag() {
    let (_dir, conn) = seed_window_db(5);
    conn.execute(
        "INSERT INTO qa_chunks (id, session_id, content, timestamp, src_rowid_lo, src_rowid_hi) \
             VALUES (80, 'w1', 'M2', 0, 3, 3)",
        [],
    )
    .unwrap();

    let out = show_session_windowed(&conn, "w1", false, Some(80), 1).unwrap();

    assert_eq!(out.data["chunk"], 80);
    assert_eq!(out.data["window"], 1);
    let msgs = out.data["messages"].as_array().unwrap();
    assert_eq!(
        msgs.len(),
        3,
        "window=1 around a one-message body yields 3 messages"
    );
    assert_eq!(msgs[1]["text"], "M2");
    assert_eq!(
        msgs[1]["in_chunk"], true,
        "the body message must be flagged in_chunk"
    );
    assert_eq!(
        msgs[0]["in_chunk"], false,
        "a neighbor must not be flagged in_chunk"
    );
    assert_eq!(msgs[2]["in_chunk"], false);
}

#[test]
fn run_show_windowed_reads_a_real_db_and_includes_file_when_verbose() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    let conn = open_or_create_db(&db_path).unwrap();
    db::seed_session(&conn, "s1");
    for text in ["m0", "m1", "m2"] {
        conn.execute(
            "INSERT INTO messages (session_id, role, text) VALUES ('s1', 'user', ?1)",
            [text],
        )
        .unwrap();
    }
    let rowid: i64 = conn
        .query_row("SELECT rowid FROM messages WHERE text='m1'", [], |r| {
            r.get(0)
        })
        .unwrap();
    conn.execute(
        "INSERT INTO qa_chunks (id, session_id, content, timestamp, src_rowid_lo, src_rowid_hi) \
             VALUES (1, 's1', 'm1', 0, ?1, ?1)",
        [rowid],
    )
    .unwrap();
    drop(conn);

    let out = run_show("s1", true, Some(1), 1, &Some(db_path)).unwrap();

    assert!(out.markdown.contains("- chunk: 1"));
    assert!(
        out.markdown.contains("- file: /f"),
        "verbose windowed show must include the file path, got: {}",
        out.markdown
    );
    for m in ["m0", "m1", "m2"] {
        assert!(out.markdown.contains(m), "window=1 must include {m}");
    }
}

#[test]
fn windowed_show_fallback_includes_file_when_verbose() {
    let (_dir, conn) = db::setup_test_db();
    db::seed_session(&conn, "s1");
    db::seed_chunk(&conn, 300, "legacy body");

    let out = show_session_windowed(&conn, "s1", true, Some(300), 3).unwrap();

    assert!(
        out.markdown.contains("- file: /f"),
        "verbose fallback must include the file path, got: {}",
        out.markdown
    );
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

fn pending_embed_count(conn: &Connection) -> i64 {
    conn.query_row(
        "SELECT COUNT(*) FROM qa_chunks c \
             WHERE NOT EXISTS (SELECT 1 FROM vec_chunks v WHERE v.chunk_id = c.id)",
        [],
        |r| r.get(0),
    )
    .unwrap()
}

fn vec_chunk_count(conn: &Connection) -> i64 {
    conn.query_row("SELECT COUNT(*) FROM vec_chunks", [], |r| r.get(0))
        .unwrap()
}

fn mock_loader() -> Result<Arc<dyn Embed>, DegradedReason> {
    Ok(Arc::new(embedder::MockEmbedder::new()) as Arc<dyn Embed>)
}

fn no_recover() -> Result<(), RecallError> {
    panic!("model recovery must not run")
}

fn absent_source_dirs(root: &Path) -> (PathBuf, PathBuf) {
    (root.join("no_claude"), root.join("no_codex"))
}

#[test]
fn embed_all_pending_embeds_every_pending_chunk() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    let mut conn = open_or_create_db(&db_path).unwrap();
    db::seed_session(&conn, "s1");
    for i in 1..=3 {
        db::seed_chunk(&conn, i, &format!("pending content {i}"));
    }

    let result = embed_all_pending(
        &mut conn,
        &embedder::MockEmbedder::new(),
        &embedder::EmbedOptions::default(),
    )
    .unwrap();

    assert_eq!(result.embedded, 3, "every pending chunk embeds");
    assert_eq!(result.failed_count, 0, "a working embedder fails nothing");
    assert_eq!(pending_embed_count(&conn), 0, "the pending set empties");
    assert_eq!(vec_chunk_count(&conn), 3, "each chunk gains a vec row");
}

#[test]
fn embed_all_pending_partial_failure_reports_failed_count() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    let mut conn = open_or_create_db(&db_path).unwrap();
    db::seed_session(&conn, "s1");
    db::seed_chunk(&conn, 1, "x");
    for i in 2..=3 {
        db::seed_chunk(&conn, i, &format!("pending content {i}"));
    }

    let result = embed_all_pending(
        &mut conn,
        &embedder::MockEmbedder::failing_on_text("x"),
        &embedder::EmbedOptions::default(),
    )
    .unwrap();

    assert_eq!(
        result.embedded, 0,
        "the poisoned single batch embeds nothing"
    );
    assert_eq!(
        result.failed_count, 3,
        "the whole 3-chunk batch is reported failed"
    );
    let err = result.first_error.as_deref().unwrap_or_default();
    assert!(
        err == "batch inference failed",
        "the first batch error is carried up for the --json note, got: {err:?}"
    );
    assert_eq!(
        pending_embed_count(&conn),
        3,
        "the failed batch stays pending for the next run"
    );
}

#[test]
fn embed_all_pending_reembeds_nothing_on_second_run() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    let mut conn = open_or_create_db(&db_path).unwrap();
    db::seed_session(&conn, "s1");
    for (id, content) in [(1, "already embedded"), (2, "new pending")] {
        db::seed_chunk(&conn, id, content);
    }
    let emb = embedder::MockEmbedder::deterministic_vector("already embedded");
    conn.execute(
        "INSERT INTO vec_chunks (embedding, chunk_id, sub_idx) VALUES (?1, 1, 0)",
        [embedder::f32_as_bytes(&emb)],
    )
    .unwrap();

    let first = embed_all_pending(
        &mut conn,
        &embedder::MockEmbedder::new(),
        &embedder::EmbedOptions::default(),
    )
    .unwrap();
    assert_eq!(
        first.embedded, 1,
        "the first run embeds only the pending chunk"
    );
    assert_eq!(vec_chunk_count(&conn), 2, "vec gains exactly the new chunk");

    let second = embed_all_pending(
        &mut conn,
        &embedder::MockEmbedder::new(),
        &embedder::EmbedOptions::default(),
    )
    .unwrap();
    assert_eq!(second.embedded, 0, "the second run re-embeds nothing");
    assert_eq!(
        vec_chunk_count(&conn),
        2,
        "a second run must re-embed nothing (NOT EXISTS vec_chunks gate)"
    );
}

#[test]
fn index_and_report_forwards_index_flags_as_embed_options() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    let claude_dir = db_dir.path().join("claude");
    let codex_dir = db_dir.path().join("codex");
    seed_claude_source(&claude_dir);
    fs::create_dir(&codex_dir).unwrap();
    let opts = indexer::IndexOptions {
        force: false,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
    };

    let embedder = Arc::new(embedder::MockEmbedder::capturing_options());
    let options = embedder::EmbedOptions {
        token_budget: Some(2048),
        forward_pause: Some(Duration::from_millis(700)),
    };

    let result =
        index_and_report_with_options(&Some(db_path), &opts, || Ok(embedder.clone()), options)
            .unwrap();

    assert_eq!(result.embedded, 1);
    assert_eq!(
        embedder.captured_options(),
        Some(options),
        "--token-budget 2048 --forward-pause-ms 700 must forward as \
             EmbedOptions{{token_budget:Some(2048), forward_pause:Some(700ms)}}"
    );
}

#[test]
fn index_and_report_forwards_default_embed_options_when_flags_unset() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    let claude_dir = db_dir.path().join("claude");
    let codex_dir = db_dir.path().join("codex");
    seed_claude_source(&claude_dir);
    fs::create_dir(&codex_dir).unwrap();
    let opts = indexer::IndexOptions {
        force: false,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
    };

    let embedder = Arc::new(embedder::MockEmbedder::capturing_options());

    let result = index_and_report_with(&Some(db_path), &opts, || Ok(embedder.clone())).unwrap();

    assert_eq!(
        result.embedded, 1,
        "default EmbedOptions must preserve current embed behavior"
    );
    assert_eq!(
        embedder.captured_options(),
        Some(embedder::EmbedOptions::default()),
        "default EmbedOptions (None/None) must forward when no flags are set"
    );
}

#[test]
fn run_status_counts_distinct_embedded_chunks_not_subembedding_rows() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    let conn = open_or_create_db(&db_path).unwrap();
    db::seed_session(&conn, "s1");
    db::seed_chunk(&conn, 1, "one chunk, two sub-embeddings");
    let emb = embedder::MockEmbedder::deterministic_vector("one chunk, two sub-embeddings");
    for sub_idx in 0..2 {
        conn.execute(
            "INSERT INTO vec_chunks (embedding, chunk_id, sub_idx) VALUES (?1, 1, ?2)",
            rusqlite::params![embedder::f32_as_bytes(&emb), sub_idx],
        )
        .unwrap();
    }
    drop(conn);

    let out = run_status(false, &Some(db_path)).unwrap();

    assert_eq!(
        out.data["embedded"], 1,
        "one qa_chunk with two sub-embedding rows is one embedded chunk, not two"
    );
    assert_eq!(out.data["qa_chunks"], 1);
    assert!(
        out.markdown.contains("Embedded: 1/1"),
        "human output must not overstate coverage past the qa_chunks total, got: {}",
        out.markdown
    );
}

struct WidthEmbedder(usize);

impl Embed for WidthEmbedder {
    fn embed_query(&self, _text: &str) -> Result<Vec<f32>, EmbedError> {
        Ok(vec![0.0; self.0])
    }
    fn embed_document(&self, _text: &str) -> Result<ChunkedEmbedding, EmbedError> {
        unreachable!("doctor probe only calls embed_query")
    }
    fn embed_documents_batch(&self, _texts: &[&str]) -> Result<Vec<ChunkedEmbedding>, EmbedError> {
        unreachable!("doctor probe only calls embed_query")
    }
    fn embed_text(&self, _text: &str, _prefix: &str) -> Result<Vec<f32>, EmbedError> {
        unreachable!("doctor probe only calls embed_query")
    }
}

struct FailingEmbedder;

impl Embed for FailingEmbedder {
    fn embed_query(&self, _text: &str) -> Result<Vec<f32>, EmbedError> {
        Err(EmbedError::EmptySequence)
    }
    fn embed_document(&self, _text: &str) -> Result<ChunkedEmbedding, EmbedError> {
        unreachable!("doctor probe only calls embed_query")
    }
    fn embed_documents_batch(&self, _texts: &[&str]) -> Result<Vec<ChunkedEmbedding>, EmbedError> {
        unreachable!("doctor probe only calls embed_query")
    }
    fn embed_text(&self, _text: &str, _prefix: &str) -> Result<Vec<f32>, EmbedError> {
        unreachable!("doctor probe only calls embed_query")
    }
}

fn seed_healthy_index(conn: &Connection) {
    db::seed_session(conn, "s1");
    db::seed_chunk(conn, 1, "healthy chunk");
    let emb = embedder::MockEmbedder::deterministic_vector("healthy chunk");
    conn.execute(
        "INSERT INTO vec_chunks (embedding, chunk_id, sub_idx) VALUES (?1, 1, 0)",
        rusqlite::params![embedder::f32_as_bytes(&emb)],
    )
    .unwrap();
}

#[test]
fn doctor_reports_healthy_on_intact_index_and_loadable_model() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    let conn = open_or_create_db(&db_path).unwrap();
    seed_healthy_index(&conn);
    drop(conn);

    let out = run_doctor_with(false, false, &Some(db_path), mock_loader, no_recover).unwrap();

    assert_eq!(out.data["healthy"], true);
    assert!(!out.degraded, "an intact index must not be degraded");
    assert!(
        out.notes.is_empty(),
        "no failing check means no remedy notes"
    );
    assert!(
        out.markdown.contains("Health: healthy"),
        "got: {}",
        out.markdown
    );
}

#[test]
fn doctor_flags_orphan_embedding_referencing_missing_chunk() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    let conn = open_or_create_db(&db_path).unwrap();
    seed_healthy_index(&conn);
    let emb = embedder::MockEmbedder::deterministic_vector("orphan");
    conn.execute(
        "INSERT INTO vec_chunks (embedding, chunk_id, sub_idx) VALUES (?1, 999, 0)",
        rusqlite::params![embedder::f32_as_bytes(&emb)],
    )
    .unwrap();
    drop(conn);

    let out = run_doctor_with(false, false, &Some(db_path), mock_loader, no_recover).unwrap();

    assert_eq!(out.data["healthy"], false);
    assert!(out.degraded);
    let check = out.data["checks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == "orphan_embeddings")
        .unwrap();
    assert_eq!(check["ok"], false);
    assert!(
        out.notes.iter().any(|n| n.contains("recall doctor --fix")),
        "remedy note must point at the --fix repair, got: {:?}",
        out.notes
    );
}

#[test]
fn fix_orphan_embeddings_deletes_only_dangling_rows() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let conn = open_or_create_db(&db_dir.path().join("recall.db")).unwrap();
    seed_healthy_index(&conn);
    let orphan = embedder::MockEmbedder::deterministic_vector("orphan");
    conn.execute(
        "INSERT INTO vec_chunks (embedding, chunk_id, sub_idx) VALUES (?1, 999, 0)",
        rusqlite::params![embedder::f32_as_bytes(&orphan)],
    )
    .unwrap();

    let deleted = fix_orphan_embeddings(&conn).unwrap();

    assert_eq!(deleted, 1, "only the dangling chunk_id=999 row is deleted");
    let remaining: i64 = conn
        .query_row("SELECT COUNT(*) FROM vec_chunks", [], |r| r.get(0))
        .unwrap();
    assert_eq!(remaining, 1, "the linked chunk_id=1 row must survive");
}

#[test]
fn fix_orphan_embeddings_is_noop_when_clean() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let conn = open_or_create_db(&db_dir.path().join("recall.db")).unwrap();
    seed_healthy_index(&conn);

    let deleted = fix_orphan_embeddings(&conn).unwrap();

    assert_eq!(deleted, 0, "a clean index has nothing to repair");
    let remaining: i64 = conn
        .query_row("SELECT COUNT(*) FROM vec_chunks", [], |r| r.get(0))
        .unwrap();
    assert_eq!(remaining, 1, "the linked row must be untouched");
}

#[test]
fn fix_orphan_embeddings_errors_when_table_missing() {
    let conn = Connection::open_in_memory().unwrap();
    assert!(fix_orphan_embeddings(&conn).is_err());
}

#[test]
fn doctor_fix_repairs_orphan_and_rechecks_healthy() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    let conn = open_or_create_db(&db_path).unwrap();
    seed_healthy_index(&conn);
    let orphan = embedder::MockEmbedder::deterministic_vector("orphan");
    conn.execute(
        "INSERT INTO vec_chunks (embedding, chunk_id, sub_idx) VALUES (?1, 999, 0)",
        rusqlite::params![embedder::f32_as_bytes(&orphan)],
    )
    .unwrap();
    drop(conn);

    let out = run_doctor_with(false, true, &Some(db_path), mock_loader, no_recover).unwrap();

    assert_eq!(out.data["healthy"], true, "got: {}", out.data);
    assert!(!out.degraded, "repaired index must not be degraded");
    assert_eq!(
        out.data["repaired"]["orphan_embeddings"], 1,
        "the name-keyed breakdown must report 1 row repaired, got: {}",
        out.data
    );
    let check = out.data["checks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == "orphan_embeddings")
        .unwrap();
    assert_eq!(check["ok"], true, "orphan check must pass post-repair");
    assert!(
        out.markdown
            .contains("Repaired orphan_embeddings: 1 row(s)"),
        "got: {}",
        out.markdown
    );
}

#[test]
fn render_doctor_report_surfaces_repair_error() {
    let path = PathBuf::from("/tmp/recall-test.db");
    let checks = vec![DoctorCheck::fail(
        "orphan_embeddings",
        "1 embedding(s) reference a missing chunk".to_owned(),
        "recall doctor --fix",
    )];
    let repairs = Some(vec![RepairOutcome {
        name: "orphan_embeddings",
        result: Err(anyhow::anyhow!("disk I/O error")),
    }]);

    let out = render_doctor_report(false, &path, &checks, &repairs, &ModelRecovery::Skipped);

    assert!(
        out.markdown
            .contains("Repair failed orphan_embeddings: disk I/O error"),
        "the repair cause must appear in human output, got: {}",
        out.markdown
    );
    assert!(
        out.notes.iter().any(|n| n.contains("repair failed")),
        "the repair cause must appear in notes, got: {:?}",
        out.notes
    );
    assert_eq!(
        out.data["repaired"]["orphan_embeddings"],
        serde_json::Value::Null,
        "a failed repair reports a null value under its name, got: {}",
        out.data
    );
}

#[test]
fn doctor_flags_orphan_chunk_referencing_missing_session() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    let conn = open_or_create_db(&db_path).unwrap();
    conn.execute(
        "INSERT INTO qa_chunks (id, session_id, content, timestamp) VALUES (1, 'ghost', 'x', 0)",
        [],
    )
    .unwrap();
    drop(conn);

    let out = run_doctor_with(false, false, &Some(db_path), mock_loader, no_recover).unwrap();

    let check = out.data["checks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == "orphan_chunks")
        .unwrap();
    assert_eq!(check["ok"], false);
    assert_eq!(out.data["healthy"], false);
}

#[test]
fn doctor_fix_leaves_unrepairable_orphan_chunk_in_place() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    let conn = open_or_create_db(&db_path).unwrap();
    conn.execute(
        "INSERT INTO qa_chunks (id, session_id, content, timestamp) VALUES (1, 'ghost', 'x', 0)",
        [],
    )
    .unwrap();
    drop(conn);

    let out =
        run_doctor_with(false, true, &Some(db_path.clone()), mock_loader, no_recover).unwrap();

    let check = out.data["checks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == "orphan_chunks")
        .unwrap();
    assert_eq!(
        check["ok"], false,
        "--fix must not repair orphan_chunks, got: {}",
        out.data
    );
    assert_eq!(
        out.data["healthy"], false,
        "unrepaired fault stays degraded"
    );
    assert!(
        out.notes.iter().any(|n| n.contains("recall rebuild")),
        "the orphan_chunks remedy must still surface, got: {:?}",
        out.notes
    );
    assert!(
        out.data["repaired"].get("orphan_chunks").is_none(),
        "orphan_chunks is not a repair, so the breakdown must omit it, got: {}",
        out.data
    );
    let conn = open_or_create_db(&db_path).unwrap();
    let remaining: i64 = conn
        .query_row("SELECT COUNT(*) FROM qa_chunks", [], |r| r.get(0))
        .unwrap();
    assert_eq!(
        remaining, 1,
        "the orphan chunk must not be deleted by --fix"
    );
}

#[test]
fn doctor_fix_breakdown_lists_every_repair_even_when_clean() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    let conn = open_or_create_db(&db_path).unwrap();
    seed_healthy_index(&conn);
    drop(conn);

    let out = run_doctor_with(false, true, &Some(db_path), mock_loader, no_recover).unwrap();

    assert_eq!(out.data["healthy"], true, "got: {}", out.data);
    let repaired = out.data["repaired"]
        .as_object()
        .unwrap_or_else(|| panic!("--fix must report an object breakdown, got: {}", out.data));
    assert_eq!(
        repaired.keys().collect::<Vec<_>>(),
        vec!["orphan_embeddings"],
        "the breakdown includes only the orphan_embeddings repair, got: {}",
        out.data
    );
    assert_eq!(
        repaired["orphan_embeddings"], 0,
        "a clean DB repairs zero rows, got: {}",
        out.data
    );
}

#[test]
fn doctor_flags_model_that_fails_to_load() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    let conn = open_or_create_db(&db_path).unwrap();
    seed_healthy_index(&conn);
    drop(conn);

    let failing_loader = || Err(DegradedReason::ProbeFailed);
    let out = run_doctor_with(false, false, &Some(db_path), failing_loader, no_recover).unwrap();

    let check = out.data["checks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == "model")
        .unwrap();
    assert_eq!(check["ok"], false);
    assert_eq!(out.data["healthy"], false);
    assert!(
        out.notes
            .iter()
            .any(|n| n.contains("recall model download")),
        "got: {:?}",
        out.notes
    );
}

#[test]
fn doctor_flags_model_dimension_mismatch() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    let conn = open_or_create_db(&db_path).unwrap();
    seed_healthy_index(&conn);
    drop(conn);

    let wrong_width = || Ok(Arc::new(WidthEmbedder(EMBEDDING_DIMS - 1)) as Arc<dyn Embed>);
    let out = run_doctor_with(false, false, &Some(db_path), wrong_width, no_recover).unwrap();

    let check = out.data["checks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == "model")
        .unwrap();
    assert_eq!(
        check["ok"], false,
        "wrong embedding width must fail the probe"
    );
    assert_eq!(out.data["healthy"], false);
}

#[test]
fn doctor_reports_not_installed_model_as_info_keeping_healthy() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    let conn = open_or_create_db(&db_path).unwrap();
    seed_healthy_index(&conn);
    drop(conn);

    let not_installed = || Err(DegradedReason::NotInstalled);
    let out = run_doctor_with(false, false, &Some(db_path), not_installed, no_recover).unwrap();

    let check = out.data["checks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == "model")
        .unwrap();
    assert_eq!(
        check["ok"], true,
        "a not-installed model is FTS-only, not a broken index"
    );
    assert_eq!(check["remedy"], serde_json::Value::Null);
    assert_eq!(out.data["healthy"], true);
    assert!(
        !out.notes
            .iter()
            .any(|n| n.contains("recall model download")),
        "info-level no-model must not emit the download remedy, got: {:?}",
        out.notes
    );
}

#[test]
fn doctor_fix_recovers_not_installed_model_to_healthy() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    let conn = open_or_create_db(&db_path).unwrap();
    seed_healthy_index(&conn);
    drop(conn);

    let calls = Cell::new(0u32);
    let loader = || {
        let n = calls.get();
        calls.set(n + 1);
        if n == 0 {
            Err(DegradedReason::NotInstalled)
        } else {
            Ok(Arc::new(embedder::MockEmbedder::new()) as Arc<dyn Embed>)
        }
    };
    let recover = || Ok(());
    let out = run_doctor_with(false, true, &Some(db_path), loader, recover).unwrap();

    let check = out.data["checks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == "model")
        .unwrap();
    assert_eq!(check["ok"], true, "post-recovery probe must pass");
    assert_eq!(
        check["detail"],
        serde_json::Value::Null,
        "a recovered model is a clean pass, not the not-installed note"
    );
    assert_eq!(out.data["model_recovered"], true);
    assert_eq!(out.data["healthy"], true);
    assert!(
        out.markdown.contains("Recovered model"),
        "human output must report the re-fetch, got: {}",
        out.markdown
    );
    assert_eq!(
        calls.get(),
        2,
        "recovery must trigger a second, re-probe load"
    );
}

#[test]
fn doctor_fix_recovers_unloadable_model_flipping_healthy() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    let conn = open_or_create_db(&db_path).unwrap();
    seed_healthy_index(&conn);
    drop(conn);

    let calls = Cell::new(0u32);
    let loader = || {
        let n = calls.get();
        calls.set(n + 1);
        if n == 0 {
            Err(DegradedReason::ProbeFailed)
        } else {
            Ok(Arc::new(embedder::MockEmbedder::new()) as Arc<dyn Embed>)
        }
    };
    let recover = || Ok(());
    let out = run_doctor_with(false, true, &Some(db_path), loader, recover).unwrap();

    assert_eq!(out.data["model_recovered"], true);
    assert_eq!(
        out.data["healthy"], true,
        "a recovered fault must clear the degraded flag"
    );
}

#[test]
fn doctor_fix_surfaces_model_recovery_failure() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    let conn = open_or_create_db(&db_path).unwrap();
    seed_healthy_index(&conn);
    drop(conn);

    let not_installed = || Err(DegradedReason::NotInstalled);
    let recover = || Err(RecallError::Internal("MLX backend unavailable".to_owned()));
    let out = run_doctor_with(false, true, &Some(db_path), not_installed, recover).unwrap();

    assert_eq!(out.data["model_recovered"], false);
    assert_eq!(out.data["model_recovery_retryable"], false);
    assert!(
        out.markdown
            .contains("Model recovery failed: MLX backend unavailable"),
        "human output must name the recovery failure, got: {}",
        out.markdown
    );
    assert!(
        out.notes
            .iter()
            .any(|n| n.contains("fix: model: recovery failed: MLX backend unavailable")),
        "recovery failure must reach notes, got: {:?}",
        out.notes
    );
    assert!(
        out.degraded,
        "a failed --fix recovery must mark the envelope degraded even when every check is ok (#249 RC-004)"
    );
}

#[test]
fn doctor_fix_failure_marks_transient_recovery_retryable() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    let conn = open_or_create_db(&db_path).unwrap();
    seed_healthy_index(&conn);
    drop(conn);

    let failing = || Err(DegradedReason::ProbeFailed);
    let recover = || Err(RecallError::TempFailure("download interrupted".to_owned()));
    let out = run_doctor_with(false, true, &Some(db_path), failing, recover).unwrap();

    assert_eq!(out.data["model_recovered"], false);
    assert_eq!(out.data["model_recovery_retryable"], true);
    assert_eq!(
        out.data["healthy"], false,
        "a model that stays unloadable keeps the degraded flag"
    );
    assert!(
        out.markdown
            .contains("Model recovery failed: download interrupted"),
        "human output must name the recovery failure, got: {}",
        out.markdown
    );
}

#[test]
fn doctor_fix_refetch_ok_but_probe_still_faults_reports_failed() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    let conn = open_or_create_db(&db_path).unwrap();
    seed_healthy_index(&conn);
    drop(conn);

    let calls = Cell::new(0u32);
    let loader = || {
        calls.set(calls.get() + 1);
        Err(DegradedReason::ProbeFailed)
    };
    let recover = || Ok(());
    let out = run_doctor_with(false, true, &Some(db_path), loader, recover).unwrap();

    assert_eq!(
        out.data["model_recovered"], false,
        "a re-fetch that leaves the probe failing did not recover the model"
    );
    assert_eq!(
        out.data["model_recovery_retryable"], false,
        "a deterministic post-verify probe failure will not clear on retry"
    );
    assert_eq!(
        out.data["healthy"], false,
        "the model is still unloadable, so the index stays degraded"
    );
    assert_eq!(
        calls.get(),
        2,
        "recovery must trigger a second, re-probe load even when it still faults"
    );
    assert!(
        out.markdown
            .contains("Model recovery failed: re-fetched model still fails its probe"),
        "human output must name the still-faulting re-probe as a failure, got: {}",
        out.markdown
    );
    assert!(
        out.notes
            .iter()
            .any(|n| n
                .contains("fix: model: recovery failed: re-fetched model still fails its probe")),
        "the still-faulting re-probe must reach notes, got: {:?}",
        out.notes
    );
}

#[test]
fn doctor_without_fix_leaves_model_recovered_null() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    let conn = open_or_create_db(&db_path).unwrap();
    seed_healthy_index(&conn);
    drop(conn);

    let failing = || Err(DegradedReason::ProbeFailed);
    let out = run_doctor_with(false, false, &Some(db_path), failing, no_recover).unwrap();

    assert_eq!(out.data["model_recovered"], serde_json::Value::Null);
    assert_eq!(out.data["healthy"], false, "an unloadable model is a fault");
}

#[test]
fn doctor_reports_unhealthy_and_does_not_create_db_when_absent() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("missing.db");

    let out = run_doctor_with(
        false,
        false,
        &Some(db_path.clone()),
        mock_loader,
        no_recover,
    )
    .unwrap();

    assert_eq!(out.data["healthy"], false);
    assert!(out.degraded);
    assert!(
        out.data
            .get("repaired")
            .is_some_and(serde_json::Value::is_null),
        "got: {:?}",
        out.data
    );
    assert!(
        out.notes.iter().any(|n| n.contains("recall index")),
        "got: {:?}",
        out.notes
    );
    assert!(
        !db_path.exists(),
        "doctor must not create the database while diagnosing"
    );
}

#[test]
fn run_status_does_not_create_db_when_absent() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("missing.db");

    let out = run_status(false, &Some(db_path.clone())).unwrap();

    assert_eq!(out.data["sessions"], 0);
    assert_eq!(out.data["qa_chunks"], 0);
    assert_eq!(out.data["embedded"], 0);
    assert!(
        !db_path.exists(),
        "status must not create the database while reporting"
    );
}

#[test]
fn doctor_reports_unhealthy_on_unopenable_database() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    fs::write(&db_path, b"this is not a sqlite database").unwrap();

    let out = run_doctor_with(false, false, &Some(db_path), mock_loader, no_recover).unwrap();

    assert_eq!(out.data["healthy"], false);
    assert!(out.degraded);
    let integrity = out.data["checks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == "integrity")
        .unwrap();
    assert_eq!(integrity["ok"], false);
    assert!(
        integrity["detail"]
            .as_str()
            .unwrap()
            .contains("cannot open database"),
        "got: {integrity}"
    );
    assert!(
        out.notes.iter().any(|n| n.contains("recall index")),
        "remedy must point at re-indexing a removed DB, got: {:?}",
        out.notes
    );
}

#[test]
fn doctor_flags_model_that_fails_to_embed() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    let conn = open_or_create_db(&db_path).unwrap();
    seed_healthy_index(&conn);
    drop(conn);

    let failing_embed = || Ok(Arc::new(FailingEmbedder) as Arc<dyn Embed>);
    let out = run_doctor_with(false, false, &Some(db_path), failing_embed, no_recover).unwrap();

    let check = out.data["checks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == "model")
        .unwrap();
    assert_eq!(check["ok"], false);
    assert!(
        check["detail"]
            .as_str()
            .unwrap()
            .contains("model embed failed"),
        "got: {check}"
    );
    assert_eq!(out.data["healthy"], false);
    assert!(
        out.notes
            .iter()
            .any(|n| n.contains("recall model download")),
        "got: {:?}",
        out.notes
    );
}

#[test]
fn run_db_check_converts_query_error_into_fail_verdict() {
    let check = run_db_check("integrity", || Err(anyhow::anyhow!("disk I/O error")));

    assert!(!check.ok);
    assert!(
        check
            .detail
            .as_deref()
            .unwrap()
            .contains("check could not run"),
        "got: {:?}",
        check.detail
    );
    assert_eq!(check.remedy, Some(CORRUPT_DB_REMEDY));
}

#[test]
fn check_orphan_embeddings_errors_when_table_missing() {
    let conn = Connection::open_in_memory().unwrap();
    assert!(check_orphan_embeddings(&conn).is_err());
}

#[test]
fn check_orphan_chunks_errors_when_table_missing() {
    let conn = Connection::open_in_memory().unwrap();
    assert!(check_orphan_chunks(&conn).is_err());
}

#[test]
fn run_doctor_wrapper_reports_on_healthy_db_with_verbose_output() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    let conn = open_or_create_db(&db_path).unwrap();
    seed_healthy_index(&conn);
    drop(conn);

    let out = run_doctor(true, false, &Some(db_path.clone())).unwrap();

    assert!(out.data["healthy"].is_boolean(), "got: {}", out.data);
    assert!(
        out.markdown.contains(&db_path.display().to_string()),
        "verbose output must show the DB path, got: {}",
        out.markdown
    );
}

#[test]
fn run_doctor_reports_unhealthy_on_never_indexed_db() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    drop(rusqlite::Connection::open(&db_path).unwrap());

    let out = run_doctor(false, false, &Some(db_path.clone())).unwrap();

    assert_eq!(
        out.data["healthy"], false,
        "never-indexed DB must be unhealthy, got: {}",
        out.data
    );
    let index_check = out.data["checks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == "index");
    assert!(
        index_check.is_some_and(|c| c["ok"] == false),
        "expected a failing `index` check, got: {}",
        out.data
    );
}

fn bare_empty_db(dir: &Path) -> PathBuf {
    let db_path = dir.join("recall.db");
    drop(rusqlite::Connection::open(&db_path).unwrap());
    db_path
}

#[test]
fn run_search_on_existing_empty_db_returns_idle() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = bare_empty_db(db_dir.path());

    let out = run_search_with(search_command("anything"), &Some(db_path), mock_loader).unwrap();

    assert_eq!(
        out.data["results"].as_array().unwrap().len(),
        0,
        "an empty DB search must return zero results, got: {}",
        out.data
    );
}

#[test]
fn run_show_on_existing_empty_db_is_not_found_usage_error() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = bare_empty_db(db_dir.path());

    let err = run_show("deadbeef", false, None, DEFAULT_WINDOW, &Some(db_path)).unwrap_err();

    assert!(
        err.to_string().contains("Database not found"),
        "empty DB show must report not-found, got: {err}"
    );
}

#[test]
fn run_status_on_existing_empty_db_reports_zero_counts() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = bare_empty_db(db_dir.path());

    let out = run_status(false, &Some(db_path)).unwrap();

    assert_eq!(out.data["sessions"], 0);
    assert_eq!(out.data["qa_chunks"], 0);
    assert_eq!(out.data["embedded"], 0);
}

#[test]
fn run_doctor_reports_schema_out_of_date_on_stale_db() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    {
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        conn.execute_batch(
            "CREATE TABLE sessions (session_id TEXT PRIMARY KEY, session_type TEXT);",
        )
        .unwrap();
    }

    let out = run_doctor_with(false, false, &Some(db_path), mock_loader, no_recover).unwrap();

    let schema = out.data["checks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == "schema")
        .unwrap_or_else(|| panic!("expected a `schema` check, got: {}", out.data));
    assert_eq!(schema["ok"], false);
    assert!(
        schema["detail"].as_str().unwrap().contains("out of date"),
        "got: {schema}"
    );
    assert_eq!(out.data["healthy"], false);
}

#[test]
fn doctor_on_an_existing_bare_sqlite_db_emits_an_index_fail_with_the_unconfirmed_origin_caveat() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = bare_empty_db(db_dir.path());

    let out = run_doctor_with(
        false,
        false,
        &Some(db_path.clone()),
        mock_loader,
        no_recover,
    )
    .unwrap();

    assert_eq!(out.data["healthy"], false);
    let index_check = out.data["checks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == "index")
        .unwrap_or_else(|| panic!("expected an `index` check, got: {}", out.data));
    assert_eq!(index_check["ok"], false);
    let detail = index_check["detail"].as_str().unwrap();
    assert!(detail.contains("cannot confirm"), "got: {detail}");
    assert!(
        detail.contains(&db_path.display().to_string()),
        "caveat detail must name the resolved DB path to verify; got: {detail}"
    );
    assert_eq!(index_check["remedy"], "run `recall index`");
}

#[test]
fn doctor_fix_refuses_to_graft_recall_tables_into_a_foreign_empty_db() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = bare_empty_db(db_dir.path());

    let out =
        run_doctor_with(false, true, &Some(db_path.clone()), mock_loader, no_recover).unwrap();

    assert_eq!(out.data["healthy"], false);
    let index_check = out.data["checks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == "index")
        .unwrap_or_else(|| panic!("expected an `index` check, got: {}", out.data));
    assert_eq!(index_check["ok"], false);
    assert!(
        index_check["detail"]
            .as_str()
            .unwrap()
            .contains("cannot confirm"),
        "got: {index_check}"
    );
    assert_eq!(out.data["repaired"], serde_json::Value::Null);
    let (conn, _) = db::open_db_readonly(&db_path).unwrap();
    assert!(
        matches!(db::schema_state(&conn), Ok(db::SchemaState::Empty)),
        "expected the foreign DB to stay Empty (ungrafted)"
    );
}

#[test]
fn run_doctor_fix_reports_open_failure_on_unopenable_db() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    fs::write(&db_path, b"this is not a sqlite database").unwrap();

    let out = run_doctor_with(false, true, &Some(db_path), mock_loader, no_recover).unwrap();

    let integrity = out.data["checks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == "integrity")
        .unwrap();
    assert_eq!(integrity["ok"], false);
    assert!(
        integrity["detail"]
            .as_str()
            .unwrap()
            .contains("cannot open database"),
        "got: {integrity}"
    );
}

fn seed_claude_source(claude_dir: &Path) {
    fs::create_dir_all(claude_dir).unwrap();
    fs::write(
            claude_dir.join("s1.jsonl"),
            r#"{"type":"user","cwd":"/proj","message":{"role":"user","content":"quokka rebuild probe"},"timestamp":"2026-03-01T00:00:00Z"}"#,
        )
        .unwrap();
}

#[test]
fn rebuild_reembeds_all_and_rebuilds_fts_index() {
    let src = tempfile::TempDir::new().unwrap();
    let claude_dir = src.path().join("claude");
    let (_, codex_dir) = absent_source_dirs(src.path());
    seed_claude_source(&claude_dir);

    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");

    let opts = indexer::IndexOptions {
        force: false,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
    };
    index_and_report_with(&Some(db_path.clone()), &opts, mock_loader).unwrap();
    {
        let conn = open_or_create_db(&db_path).unwrap();
        assert_eq!(
            pending_embed_count(&conn),
            0,
            "precondition: initial index must leave nothing pending"
        );
    }

    let rebuild_opts = indexer::IndexOptions {
        force: true,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
    };
    index_and_report_with(&Some(db_path.clone()), &rebuild_opts, mock_loader).unwrap();

    let conn = open_or_create_db(&db_path).unwrap();
    assert_eq!(
        pending_embed_count(&conn),
        0,
        "rebuild must re-embed every chunk (pending=0 after full reset)"
    );

    let opts = search::SearchOptions::default();
    let hits = search::search(&conn, "quokka", &opts).unwrap();
    assert_eq!(
        hits.len(),
        1,
        "rebuild must reconstruct FTS: the seeded term must return its session"
    );
}

#[test]
fn index_and_report_with_backfills_legacy_null_rowid_ranges() {
    let src = tempfile::TempDir::new().unwrap();
    let claude_dir = src.path().join("claude");
    let (_, codex_dir) = absent_source_dirs(src.path());
    seed_claude_source(&claude_dir);

    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");

    let opts = indexer::IndexOptions {
        force: false,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
    };
    index_and_report_with(&Some(db_path.clone()), &opts, mock_loader).unwrap();

    {
        let conn = open_or_create_db(&db_path).unwrap();
        let cleared = conn
            .execute(
                "UPDATE qa_chunks SET src_rowid_lo = NULL, src_rowid_hi = NULL",
                [],
            )
            .unwrap();
        assert_eq!(cleared, 1, "precondition: one chunk to un-link");
    }

    index_and_report_with(&Some(db_path.clone()), &opts, mock_loader).unwrap();

    let conn = open_or_create_db(&db_path).unwrap();
    let (lo, hi): (Option<i64>, Option<i64>) = conn
        .query_row(
            "SELECT src_rowid_lo, src_rowid_hi FROM qa_chunks",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    assert!(
        lo.is_some() && hi.is_some(),
        "the index-path backfill must repopulate the legacy NULL rowid range, got lo={lo:?} hi={hi:?}"
    );
}

fn write_backfill_session(
    dir: &Path,
    file: &str,
    user_text: &str,
    edit_path: Option<&str>,
) -> PathBuf {
    fs::create_dir_all(dir).unwrap();
    let user = serde_json::json!({
        "type": "user",
        "cwd": "/proj",
        "message": {"role": "user", "content": user_text},
        "timestamp": "2026-03-01T00:00:00Z"
    });
    let mut content = serde_json::to_string(&user).unwrap();
    if let Some(path) = edit_path {
        let assistant = serde_json::json!({
            "type": "assistant",
            "cwd": "/proj",
            "message": {"role": "assistant", "content": [
                {"type": "text", "text": "editing"},
                {"type": "tool_use", "id": "t1", "name": "Edit",
                 "input": {"file_path": path, "old_string": "x", "new_string": "y"}}
            ]},
            "timestamp": "2026-03-01T00:00:01Z"
        });
        content.push('\n');
        content.push_str(&serde_json::to_string(&assistant).unwrap());
    }
    let full = dir.join(file);
    fs::write(&full, content).unwrap();
    full
}

fn collect_i64(conn: &Connection, sql: &str) -> Vec<i64> {
    let mut stmt = conn.prepare(sql).unwrap();
    let rows = stmt.query_map([], |r| r.get::<_, i64>(0)).unwrap();
    rows.map(Result::unwrap).collect()
}

fn body_snapshot(conn: &Connection) -> (Vec<i64>, Vec<i64>, Vec<i64>) {
    (
        collect_i64(conn, "SELECT rowid FROM messages ORDER BY rowid"),
        collect_i64(conn, "SELECT id FROM qa_chunks ORDER BY id"),
        collect_i64(conn, "SELECT chunk_id FROM vec_chunks ORDER BY chunk_id"),
    )
}

fn session_file_paths(conn: &Connection) -> Vec<String> {
    let mut stmt = conn
        .prepare("SELECT path FROM session_files ORDER BY path")
        .unwrap();
    let rows = stmt.query_map([], |r| r.get::<_, String>(0)).unwrap();
    rows.map(Result::unwrap).collect()
}

fn reset_to_legacy_state(conn: &Connection) {
    conn.execute("UPDATE sessions SET files_scanned = NULL", [])
        .unwrap();
    conn.execute("DELETE FROM session_files", []).unwrap();
}

#[test]
fn backfill_が本文_index_と_embedding_を変えない() {
    let src = tempfile::TempDir::new().unwrap();
    let claude_dir = src.path().join("claude");
    let (_, codex_dir) = absent_source_dirs(src.path());
    let partial = write_backfill_session(
        &claude_dir,
        "a.jsonl",
        "session a probe",
        Some("/proj/marker_a.rs"),
    );
    write_backfill_session(
        &claude_dir,
        "b.jsonl",
        "session b probe",
        Some("/proj/marker_b.rs"),
    );

    let body = fs::read_to_string(&partial).unwrap();
    fs::write(&partial, format!("{body}\ninvalid JSON\n")).unwrap();

    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    let opts = indexer::IndexOptions {
        force: false,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
    };

    let outcome = index_and_report_with(&Some(db_path.clone()), &opts, mock_loader).unwrap();
    assert_eq!(
        index_command_output(&outcome).data["parse_diagnostics"][0]["invalid_json_lines"],
        1
    );
    let before = {
        let conn = open_or_create_db(&db_path).unwrap();
        reset_to_legacy_state(&conn);
        conn.execute("DELETE FROM parse_diagnostics", []).unwrap();
        assert!(
            !collect_i64(&conn, "SELECT chunk_id FROM vec_chunks").is_empty(),
            "precondition: the legacy DB must hold embeddings, or the invariant is vacuous"
        );
        body_snapshot(&conn)
    };

    let outcome = index_and_report_with(&Some(db_path.clone()), &opts, mock_loader).unwrap();
    assert_eq!(
        index_command_output(&outcome).data["parse_diagnostics"][0]["invalid_json_lines"],
        1
    );

    let conn = open_or_create_db(&db_path).unwrap();
    assert_eq!(
        body_snapshot(&conn),
        before,
        "backfill must not change messages / qa_chunks / vec_chunks row counts or id sets"
    );
    assert_eq!(
        session_file_paths(&conn),
        vec![
            "/proj/marker_a.rs".to_owned(),
            "/proj/marker_b.rs".to_owned(),
        ],
        "the backfill must fill session_files from the write-target tool_use paths"
    );
}

#[test]
fn 二回目の_index_で_backfill_が再実行されない() {
    let src = tempfile::TempDir::new().unwrap();
    let claude_dir = src.path().join("claude");
    let (_, codex_dir) = absent_source_dirs(src.path());
    let write_path = write_backfill_session(
        &claude_dir,
        "write.jsonl",
        "write session probe",
        Some("/proj/orig_marker.rs"),
    );
    write_backfill_session(&claude_dir, "zero.jsonl", "zero touch probe", None);

    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    let opts = indexer::IndexOptions {
        force: false,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
    };

    index_and_report_with(&Some(db_path.clone()), &opts, mock_loader).unwrap();
    {
        let conn = open_or_create_db(&db_path).unwrap();
        reset_to_legacy_state(&conn);
    }
    index_and_report_with(&Some(db_path.clone()), &opts, mock_loader).unwrap();

    let original = fs::metadata(&write_path).unwrap();
    let pinned = original.modified().unwrap();
    write_backfill_session(
        &claude_dir,
        "write.jsonl",
        "write session probe",
        Some("/proj/next_marker.rs"),
    );
    let f = OpenOptions::new().write(true).open(&write_path).unwrap();
    f.set_modified(pinned).unwrap();
    drop(f);
    let replaced = fs::metadata(&write_path).unwrap();
    assert_eq!(replaced.len(), original.len(), "canary must preserve size");
    assert_eq!(replaced.modified().unwrap(), pinned);

    index_and_report_with(&Some(db_path.clone()), &opts, mock_loader).unwrap();

    let conn = open_or_create_db(&db_path).unwrap();
    let unmarked: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM sessions WHERE files_scanned IS NULL",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        unmarked, 0,
        "every session (write-target and zero-touch) must carry files_scanned after the backfill"
    );
    assert!(
        !session_file_paths(&conn)
            .iter()
            .any(|p| p == "/proj/next_marker.rs"),
        "a marked session must not be re-read: the canary path must not appear in session_files"
    );

    conn.execute("DELETE FROM session_files WHERE session_id = 'write'", [])
        .unwrap();
    conn.execute(
        "UPDATE sessions SET files_scanned = NULL WHERE session_id = 'write'",
        [],
    )
    .unwrap();
    drop(conn);
    index_and_report_with(&Some(db_path.clone()), &opts, mock_loader).unwrap();
    let conn = open_or_create_db(&db_path).unwrap();
    assert!(session_file_paths(&conn).contains(&"/proj/next_marker.rs".to_owned()));
}

#[test]
fn index_and_report_with_missing_root_threads_skipped_root_into_outcome() {
    let src = tempfile::TempDir::new().unwrap();
    let claude_dir = src.path().join("claude");
    let (_, codex_dir) = absent_source_dirs(src.path());
    seed_claude_source(&claude_dir);

    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");

    let opts = indexer::IndexOptions {
        force: false,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
    };
    index_and_report_with(&Some(db_path.clone()), &opts, mock_loader).unwrap();

    let gone_claude = src.path().join("claude_gone");
    let missing_opts = indexer::IndexOptions {
        force: false,
        claude_dir: &gone_claude,
        codex_dir: &codex_dir,
    };
    let outcome = index_and_report_with(&Some(db_path), &missing_opts, mock_loader).unwrap();

    assert_eq!(
        outcome.skipped_roots.len(),
        1,
        "the missing claude root must be threaded into the outcome"
    );
    assert_eq!(outcome.skipped_roots[0].source, parser::Source::Claude);
    assert_eq!(
        outcome.skipped_roots[0].preserved_sessions, 1,
        "the one preserved claude session must be counted"
    );
}

#[test]
fn index_and_report_with_degraded_loader_completes_fts_and_succeeds() {
    let src = tempfile::TempDir::new().unwrap();
    let claude_dir = src.path().join("claude");
    let (_, codex_dir) = absent_source_dirs(src.path());
    seed_claude_source(&claude_dir);

    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");

    let opts = indexer::IndexOptions {
        force: false,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
    };
    let outcome = index_and_report_with(&Some(db_path.clone()), &opts, || {
        Err(DegradedReason::NotInstalled)
    });
    assert!(
        outcome.is_ok(),
        "a model-less index must complete FTS and exit 0, not abort"
    );
    let outcome = outcome.unwrap();
    let counts = &outcome.observations["counts"];
    assert_eq!(counts["chunks_pending_snapshot"], 1);
    assert_eq!(counts["chunks_unattempted"], 1);
    assert_eq!(counts["chunks_remaining_snapshot"], 1);
    for key in [
        "chunks_saved",
        "chunks_failed",
        "chunks_stale",
        "chunks_save_failed",
    ] {
        assert_eq!(counts[key], 0);
    }

    let conn = open_or_create_db(&db_path).unwrap();
    let search_opts = search::SearchOptions::default();
    let hits = search::search(&conn, "quokka", &search_opts).unwrap();
    assert_eq!(
        hits.len(),
        1,
        "the FTS index must be complete after a degraded index: the seeded term returns its session"
    );
}

#[test]
fn index_and_report_with_degraded_loader_emits_model_download_note() {
    let src = tempfile::TempDir::new().unwrap();
    let claude_dir = src.path().join("claude");
    let (_, codex_dir) = absent_source_dirs(src.path());
    seed_claude_source(&claude_dir);

    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");

    let opts = indexer::IndexOptions {
        force: false,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
    };
    let outcome = index_and_report_with(&Some(db_path.clone()), &opts, || {
        Err(DegradedReason::NotInstalled)
    })
    .expect("a degraded index must succeed");

    let note = outcome
        .degraded_note
        .expect("a model-less index must carry a degraded note");
    assert!(
        note.contains("recall model download"),
        "the degraded note must name `recall model download` to guide the user, got: {note}"
    );
}

#[test]
fn index_and_report_with_unavailable_loader_carries_unavailability_note() {
    for reason in [
        DegradedReason::BackendUnavailable,
        DegradedReason::ProbeFailed,
    ] {
        let src = tempfile::TempDir::new().unwrap();
        let claude_dir = src.path().join("claude");
        let (_, codex_dir) = absent_source_dirs(src.path());
        seed_claude_source(&claude_dir);

        let db_dir = tempfile::TempDir::new().unwrap();
        let db_path = db_dir.path().join("recall.db");

        let opts = indexer::IndexOptions {
            force: false,
            claude_dir: &claude_dir,
            codex_dir: &codex_dir,
        };
        let outcome = index_and_report_with(&Some(db_path), &opts, || Err(reason))
            .expect("a degraded index must succeed");

        let note = outcome
            .degraded_note
            .unwrap_or_else(|| panic!("{reason:?} must carry a degraded note"));
        assert!(
            note.contains("unavailable"),
            "{reason:?} must report unavailability (not a download hint), got: {note}"
        );
        assert!(
            !note.to_lowercase().contains("download"),
            "{reason:?} is not repaired by re-downloading, so the note must not \
                 suggest it (that hint belongs to NotInstalled), got: {note}"
        );
        assert_eq!(outcome.embedded, 0, "{reason:?} skips embedding");
        assert_eq!(outcome.failed_count, 0, "{reason:?} fails no batches");
    }
}

#[test]
fn index_and_report_with_disabled_loader_is_indistinguishable_from_success() {
    let src = tempfile::TempDir::new().unwrap();
    let claude_dir = src.path().join("claude");
    let (_, codex_dir) = absent_source_dirs(src.path());
    seed_claude_source(&claude_dir);

    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");

    let opts = indexer::IndexOptions {
        force: false,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
    };
    let outcome = index_and_report_with(&Some(db_path), &opts, || Err(DegradedReason::Disabled))
        .expect("a disabled-embedder index must succeed");

    assert_eq!(
        outcome.degraded_note, None,
        "opt-out is not a degradation: no note is carried"
    );
    assert_eq!(outcome.embedded, 0, "a disabled embedder embeds nothing");
    assert_eq!(
        outcome.failed_count, 0,
        "a disabled embedder fails no batches"
    );
    let envelope = index_command_output(&outcome);
    assert!(
        !envelope.degraded,
        "the --json envelope reports a disabled embedder as non-degraded"
    );
}

#[test]
fn model_absent_rebuild_preserves_existing_embeddings() {
    let src = tempfile::TempDir::new().unwrap();
    let claude_dir = src.path().join("claude");
    let (_, codex_dir) = absent_source_dirs(src.path());
    seed_claude_source(&claude_dir);

    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");

    let opts = indexer::IndexOptions {
        force: false,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
    };
    index_and_report_with(&Some(db_path.clone()), &opts, mock_loader).unwrap();
    let before = {
        let conn = open_or_create_db(&db_path).unwrap();
        vec_chunk_count(&conn)
    };
    assert!(
        before > 0,
        "precondition: the initial index must embed at least one chunk"
    );

    let rebuild = indexer::IndexOptions {
        force: true,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
    };
    index_and_report_with(&Some(db_path.clone()), &rebuild, || {
        Err(DegradedReason::NotInstalled)
    })
    .unwrap();

    let conn = open_or_create_db(&db_path).unwrap();
    assert_eq!(
        vec_chunk_count(&conn),
        before,
        "a model-absent rebuild must preserve existing embeddings, not destroy them"
    );
}

#[test]
fn load_ok_inference_fail_rebuild_preserves_existing_embeddings() {
    let src = tempfile::TempDir::new().unwrap();
    let claude_dir = src.path().join("claude");
    let (_, codex_dir) = absent_source_dirs(src.path());
    seed_claude_source(&claude_dir);

    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");

    let opts = indexer::IndexOptions {
        force: false,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
    };
    index_and_report_with(&Some(db_path.clone()), &opts, mock_loader).unwrap();
    let before = {
        let conn = open_or_create_db(&db_path).unwrap();
        vec_chunk_count(&conn)
    };
    assert!(
        before > 0,
        "precondition: the initial index must embed at least one chunk"
    );

    let rebuild = indexer::IndexOptions {
        force: true,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
    };
    let outcome = index_and_report_with(&Some(db_path.clone()), &rebuild, || {
        Ok(Arc::new(embedder::MockEmbedder::failing_after(0)) as Arc<dyn Embed>)
    })
    .unwrap();

    let conn = open_or_create_db(&db_path).unwrap();
    assert_eq!(
        vec_chunk_count(&conn),
        before,
        "a load-OK-but-inference-fails rebuild must preserve existing embeddings, not destroy them"
    );
    assert!(
        outcome.preserved_embedded > 0,
        "the preserved-embedding count must ride the degraded arm"
    );
    assert!(
        outcome.degraded_note.is_some(),
        "an inference-incapable model must report a degraded note"
    );
}

#[test]
fn model_present_rebuild_heals_orphan_vec_rows() {
    let src = tempfile::TempDir::new().unwrap();
    let claude_dir = src.path().join("claude");
    let (_, codex_dir) = absent_source_dirs(src.path());
    seed_claude_source(&claude_dir);

    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");

    let opts = indexer::IndexOptions {
        force: false,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
    };
    index_and_report_with(&Some(db_path.clone()), &opts, mock_loader).unwrap();

    const ORPHAN_ID: i64 = 9_999_999;
    {
        let conn = open_or_create_db(&db_path).unwrap();
        let orphan = embedder::MockEmbedder::deterministic_vector("orphan");
        conn.execute(
            "INSERT INTO vec_chunks (embedding, chunk_id, sub_idx) VALUES (?1, ?2, 0)",
            rusqlite::params![embedder::f32_as_bytes(&orphan), ORPHAN_ID],
        )
        .unwrap();
    }

    let rebuild = indexer::IndexOptions {
        force: true,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
    };
    index_and_report_with(&Some(db_path.clone()), &rebuild, mock_loader).unwrap();

    let conn = open_or_create_db(&db_path).unwrap();
    let orphan_remaining: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM vec_chunks WHERE chunk_id = ?1",
            [ORPHAN_ID],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        orphan_remaining, 0,
        "a model-present rebuild must delete the pre-existing orphan vec row"
    );
}

#[test]
fn model_absent_rebuild_heals_orphan_and_preserves_embeddings() {
    let src = tempfile::TempDir::new().unwrap();
    let claude_dir = src.path().join("claude");
    let (_, codex_dir) = absent_source_dirs(src.path());
    seed_claude_source(&claude_dir);

    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");

    let opts = indexer::IndexOptions {
        force: false,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
    };
    index_and_report_with(&Some(db_path.clone()), &opts, mock_loader).unwrap();
    let before = {
        let conn = open_or_create_db(&db_path).unwrap();
        vec_chunk_count(&conn)
    };
    assert!(
        before > 0,
        "precondition: the initial index embedded chunks"
    );

    const ORPHAN_ID: i64 = 9_999_999;
    {
        let conn = open_or_create_db(&db_path).unwrap();
        let orphan = embedder::MockEmbedder::deterministic_vector("orphan");
        conn.execute(
            "INSERT INTO vec_chunks (embedding, chunk_id, sub_idx) VALUES (?1, ?2, 0)",
            rusqlite::params![embedder::f32_as_bytes(&orphan), ORPHAN_ID],
        )
        .unwrap();
    }

    let rebuild = indexer::IndexOptions {
        force: true,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
    };
    index_and_report_with(&Some(db_path.clone()), &rebuild, || {
        Err(DegradedReason::NotInstalled)
    })
    .unwrap();

    let conn = open_or_create_db(&db_path).unwrap();
    let orphan_remaining: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM vec_chunks WHERE chunk_id = ?1",
            [ORPHAN_ID],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        orphan_remaining, 0,
        "the orphan vec row must be deleted on the model-absent rebuild arm"
    );
    assert_eq!(
        vec_chunk_count(&conn),
        before,
        "preserved embeddings (stable qa ids) survive; only the orphan is removed"
    );
}

#[test]
fn model_absent_rebuild_surfaces_preserved_count_and_note() {
    let src = tempfile::TempDir::new().unwrap();
    let claude_dir = src.path().join("claude");
    let (_, codex_dir) = absent_source_dirs(src.path());
    seed_claude_source(&claude_dir);

    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");

    let opts = indexer::IndexOptions {
        force: false,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
    };
    index_and_report_with(&Some(db_path.clone()), &opts, mock_loader).unwrap();

    let rebuild = indexer::IndexOptions {
        force: true,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
    };
    let outcome = index_and_report_with(&Some(db_path.clone()), &rebuild, || {
        Err(DegradedReason::NotInstalled)
    })
    .unwrap();

    assert!(
        outcome.preserved_embedded > 0,
        "the outcome must report the preserved session count, got {}",
        outcome.preserved_embedded
    );

    let out = index_command_output(&outcome);
    assert!(
        out.degraded,
        "a model-absent run with preservation degrades"
    );
    assert!(
        out.notes
            .iter()
            .any(|n| n.contains(&outcome.preserved_embedded.to_string())),
        "the preserve note must name the preserved count, got: {:?}",
        out.notes
    );
    assert!(
        out.notes
            .iter()
            .any(|n| n.contains("recall model download")),
        "the degraded note must carry the download remedy for a not-installed model, got: {:?}",
        out.notes
    );
    assert_eq!(
        out.data["preserved_embedded"], outcome.preserved_embedded,
        "the --json payload carries the preserved magnitude for an agent"
    );
}

#[test]
fn model_absent_index_adds_new_session_and_preserves_embeddings() {
    let src = tempfile::TempDir::new().unwrap();
    let claude_dir = src.path().join("claude");
    let (_, codex_dir) = absent_source_dirs(src.path());
    seed_claude_source(&claude_dir);

    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");

    let opts = indexer::IndexOptions {
        force: false,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
    };
    index_and_report_with(&Some(db_path.clone()), &opts, mock_loader).unwrap();
    let before = {
        let conn = open_or_create_db(&db_path).unwrap();
        vec_chunk_count(&conn)
    };
    assert!(before > 0, "precondition: the first session was embedded");

    fs::write(
            claude_dir.join("s2.jsonl"),
            r#"{"type":"user","cwd":"/proj","message":{"role":"user","content":"platypus newcomer"},"timestamp":"2026-03-02T00:00:00Z"}"#,
        )
        .unwrap();
    let rebuild_opts = indexer::IndexOptions {
        force: true,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
    };
    let outcome = index_and_report_with(&Some(db_path.clone()), &rebuild_opts, || {
        Err(DegradedReason::NotInstalled)
    })
    .unwrap();

    let conn = open_or_create_db(&db_path).unwrap();
    assert_eq!(
        vec_chunk_count(&conn),
        before,
        "the original session's embeddings must be preserved, not destroyed"
    );
    assert_eq!(
        outcome.preserved_embedded, 1,
        "only the one embedded session is preserved; the new one is indexed"
    );
    let search_opts = search::SearchOptions::default();
    let hits = search::search(&conn, "platypus", &search_opts).unwrap();
    assert_eq!(
        hits.len(),
        1,
        "the new session must be FTS-indexed even with the model absent"
    );
}

#[test]
fn model_absent_rebuild_reindexes_cached_session_without_embeddings() {
    let src = tempfile::TempDir::new().unwrap();
    let claude_dir = src.path().join("claude");
    let (_, codex_dir) = absent_source_dirs(src.path());
    fs::create_dir_all(&claude_dir).unwrap();
    let session = claude_dir.join("s1.jsonl");
    fs::write(
            &session,
            r#"{"type":"user","cwd":"/proj","message":{"role":"user","content":"alphaword original"},"timestamp":"2026-03-01T00:00:00Z"}"#,
        )
        .unwrap();

    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");

    let opts = indexer::IndexOptions {
        force: false,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
    };
    index_and_report_with(&Some(db_path.clone()), &opts, || {
        Err(DegradedReason::NotInstalled)
    })
    .unwrap();
    {
        let conn = open_or_create_db(&db_path).unwrap();
        assert_eq!(
            vec_chunk_count(&conn),
            0,
            "precondition: a model-absent index embeds nothing"
        );
    }

    fs::write(
            &session,
            r#"{"type":"user","cwd":"/proj","message":{"role":"user","content":"betaword replaced"},"timestamp":"2026-03-01T00:00:00Z"}"#,
        )
        .unwrap();
    let rebuild = indexer::IndexOptions {
        force: true,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
    };
    let outcome = index_and_report_with(&Some(db_path.clone()), &rebuild, || {
        Err(DegradedReason::NotInstalled)
    })
    .unwrap();

    assert_eq!(
        outcome.preserved_embedded, 0,
        "a cached session without embeddings must not be preserved"
    );
    let conn = open_or_create_db(&db_path).unwrap();
    let search_opts = search::SearchOptions::default();
    assert_eq!(
        search::search(&conn, "betaword", &search_opts)
            .unwrap()
            .len(),
        1,
        "the rebuild must re-index the new content, not freeze the old"
    );
    assert_eq!(
        search::search(&conn, "alphaword", &search_opts)
            .unwrap()
            .len(),
        0,
        "the stale content must be gone, proving a real re-index occurred"
    );
}

#[test]
fn model_absent_incremental_index_preserves_edited_embedded_session() {
    use std::time::{Duration, SystemTime};

    let src = tempfile::TempDir::new().unwrap();
    let claude_dir = src.path().join("claude");
    let (_, codex_dir) = absent_source_dirs(src.path());
    seed_claude_source(&claude_dir);

    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");

    let opts = indexer::IndexOptions {
        force: false,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
    };
    index_and_report_with(&Some(db_path.clone()), &opts, mock_loader).unwrap();
    let before = {
        let conn = open_or_create_db(&db_path).unwrap();
        vec_chunk_count(&conn)
    };
    assert!(before > 0, "precondition: the session was embedded");

    let session = claude_dir.join("s1.jsonl");
    let mut all = fs::read_to_string(&session).unwrap();
    all.push('\n');
    all.push_str(
            r#"{"type":"user","cwd":"/proj","message":{"role":"user","content":"echidna appended"},"timestamp":"2026-03-01T00:00:01Z"}"#,
        );
    fs::write(&session, all).unwrap();
    let f = fs::OpenOptions::new().append(true).open(&session).unwrap();
    f.set_modified(SystemTime::now() + Duration::from_secs(10))
        .unwrap();
    drop(f);

    let outcome = index_and_report_with(&Some(db_path.clone()), &opts, || {
        Err(DegradedReason::NotInstalled)
    })
    .unwrap();

    assert_eq!(
        outcome.preserved_embedded, 1,
        "the edited embedded session is preserved, not destroyed, on `recall index`"
    );
    let conn = open_or_create_db(&db_path).unwrap();
    assert_eq!(
        vec_chunk_count(&conn),
        before,
        "the embeddings survive the incremental re-index"
    );
    let search_opts = search::SearchOptions::default();
    assert_eq!(
        search::search(&conn, "quokka", &search_opts).unwrap().len(),
        1,
        "the preserved session's original FTS rows remain searchable"
    );
    assert_eq!(
        search::search(&conn, "echidna", &search_opts)
            .unwrap()
            .len(),
        0,
        "the appended turn stays out of FTS until the model returns (FR-2 trade-off)"
    );
}

fn embed_stop_outcome() -> IndexOutcome {
    IndexOutcome {
        degraded_note: None,
        embedded: 5,
        failed_count: 3,
        first_error: Some("batch: mock failure".to_owned()),
        skipped_roots: Vec::new(),
        preserved_embedded: 0,
        parse_diagnostics: Vec::new(),
        observations: serde_json::Value::Null,
    }
}

fn model_absent_outcome() -> IndexOutcome {
    IndexOutcome {
        degraded_note: Some("note: run `recall model download` to enable.".to_owned()),
        embedded: 0,
        failed_count: 0,
        first_error: None,
        skipped_roots: Vec::new(),
        preserved_embedded: 0,
        parse_diagnostics: Vec::new(),
        observations: serde_json::Value::Null,
    }
}

fn success_outcome() -> IndexOutcome {
    IndexOutcome {
        degraded_note: None,
        embedded: 8,
        failed_count: 0,
        first_error: None,
        skipped_roots: Vec::new(),
        preserved_embedded: 0,
        parse_diagnostics: Vec::new(),
        observations: serde_json::Value::Null,
    }
}

#[test]
fn index_command_output_surfaces_embed_stop_with_failed_count_and_retry() {
    let out = index_command_output(&embed_stop_outcome());

    assert!(
        out.degraded,
        "an embed-stop index is degraded: --json must not report it as a clean success"
    );
    assert!(
        out.notes[0].contains("3 chunk"),
        "the note must name how many chunks failed, got: {:?}",
        out.notes
    );
    assert!(
        out.notes[0].contains("recall index"),
        "the note must guide the agent to retry via `recall index`, got: {:?}",
        out.notes
    );
    assert!(
        out.notes[0].contains("mock failure"),
        "the note carries the first batch error, not the unknown-error fallback, got: {:?}",
        out.notes
    );
    assert_eq!(
        out.data["failed_count"], 3,
        "the payload carries the failure magnitude, not just a bool"
    );
    assert_eq!(
        out.data["embedded"], 5,
        "the payload reports how many chunks did embed despite the stop"
    );
}

#[test]
fn index_command_output_surfaces_model_absent_download_note() {
    let out = index_command_output(&model_absent_outcome());

    assert!(
        out.degraded,
        "a model-absent index is degraded: embedding was skipped"
    );
    assert!(
        out.notes
            .iter()
            .any(|n| n.contains("recall model download")),
        "the model-absent note must reach --json (not stderr only), got: {:?}",
        out.notes
    );
}

#[test]
fn index_command_output_clean_success_is_not_degraded() {
    let out = index_command_output(&success_outcome());

    assert!(!out.degraded, "a fully embedded index is not degraded");
    assert!(
        out.notes.is_empty(),
        "a clean success carries no notes, got: {:?}",
        out.notes
    );
    assert_eq!(
        out.data["failed_count"], 0,
        "the payload reports zero failures on success"
    );
    assert_eq!(
        out.data["embedded"], 8,
        "the payload reports how many chunks were embedded on success"
    );
}

#[test]
fn index_command_output_degraded_covaries_with_notes() {
    for outcome in [
        embed_stop_outcome(),
        model_absent_outcome(),
        success_outcome(),
    ] {
        let out = index_command_output(&outcome);
        assert_eq!(
            out.degraded,
            !out.notes.is_empty(),
            "degraded must equal notes-non-empty (degraded={}, notes={:?})",
            out.degraded,
            out.notes
        );
    }
}

#[test]
fn parse_diagnostics_degrade_alone_and_coexist_with_embedding_and_root_reasons() {
    for mut outcome in [
        success_outcome(),
        model_absent_outcome(),
        embed_stop_outcome(),
    ] {
        let existing_notes = index_command_output(&outcome).notes;
        outcome
            .parse_diagnostics
            .push(indexer::FileParseDiagnostics {
                file_path: format!("/logs/\x1b[31mfile\n\t{}", "界".repeat(300)).into(),
                source: parser::Source::Claude,
                lines: parser::LineDiagnostics {
                    invalid_json_lines: 2,
                    invalid_utf8_lines: 1,
                    incomplete_tail_lines: 1,
                },
                read_error: true,
            });
        let without_root = index_command_output(&outcome);
        assert!(without_root.degraded);
        assert_eq!(without_root.notes.len(), existing_notes.len() + 3);
        outcome.skipped_roots.push(indexer::SkippedRoot {
            source: parser::Source::Codex,
            preserved_sessions: 1,
            reason: indexer::SkippedReason::MissingRoot,
        });
        let out = index_command_output(&outcome);
        assert!(out.degraded);
        for note in existing_notes {
            assert!(out.notes.contains(&note));
        }
        assert!(out.notes.iter().any(|n| n.contains("root unavailable")));
        let d = &out.data["parse_diagnostics"][0];
        let path = d["file_path"].as_str().unwrap();
        assert_eq!(path.chars().count(), 240);
        assert!(path.starts_with("/logs/file"));
        assert!(!path.chars().any(char::is_control));
        assert_eq!(d["invalid_json_lines"], 2);
        assert_eq!(d["invalid_utf8_lines"], 1);
        assert_eq!(d["incomplete_tail_lines"], 1);
        assert_eq!(d["read_error"], true);
        for note in &out.notes {
            assert!(!note.chars().any(char::is_control));
            assert!(note.chars().count() < 500);
        }
    }
}

#[test]
fn index_command_output_sanitizes_and_bounds_first_error_in_note() {
    let outcome = IndexOutcome {
        degraded_note: None,
        embedded: 0,
        failed_count: 2,
        first_error: Some(format!("batch: \x1b[31mboom\x1b[0m {}", "e".repeat(300))),
        skipped_roots: Vec::new(),
        preserved_embedded: 0,
        parse_diagnostics: Vec::new(),
        observations: serde_json::Value::Null,
    };

    let out = index_command_output(&outcome);

    assert!(
        !out.notes[0].contains('\x1b'),
        "control chars are stripped from the note, got: {:?}",
        out.notes
    );
    assert!(
        out.notes[0].len() < 200,
        "a verbose error is capped, got len {}",
        out.notes[0].len()
    );
    assert!(
        out.notes[0].contains("rerun `recall index`"),
        "the retry guidance survives the capping, got: {:?}",
        out.notes
    );
}

#[test]
fn index_command_output_surfaces_skipped_root_in_json() {
    let outcome = IndexOutcome {
        degraded_note: None,
        embedded: 4,
        failed_count: 0,
        first_error: None,
        skipped_roots: vec![indexer::SkippedRoot {
            source: parser::Source::Codex,
            preserved_sessions: 2,
            reason: indexer::SkippedReason::IncompleteEnumeration,
        }],
        preserved_embedded: 0,
        parse_diagnostics: Vec::new(),
        observations: serde_json::Value::Null,
    };

    let out = index_command_output(&outcome);

    assert!(
        out.degraded,
        "a skipped root is a degradation: --json must not report it as a clean success"
    );
    assert!(
        out.notes
            .iter()
            .any(|n| n.contains("codex") && n.contains("2") && n.contains("recall index")),
        "the note must name the source, the preserved count, and the reconcile command, got: {:?}",
        out.notes
    );
    assert!(
        out.notes
            .iter()
            .any(|n| n.contains("permissions and access")),
        "an incomplete-enumeration root must surface the access remedy, not the missing-root wording, got: {:?}",
        out.notes
    );
    assert_eq!(
        out.data["skipped_roots"][0]["source"], "codex",
        "the payload carries the skipped source, not just prose"
    );
    assert_eq!(
        out.data["skipped_roots"][0]["preserved_sessions"], 2,
        "the payload carries the preserved magnitude an agent can act on"
    );
    assert_eq!(
        out.data["skipped_roots"][0]["reason"], "incomplete_enumeration",
        "the payload carries the machine-readable skip cause so an agent picks the right remedy"
    );
}

#[test]
fn index_command_note_for_zero_preserved_incomplete_root_omits_reconcile() {
    let outcome = IndexOutcome {
        degraded_note: None,
        embedded: 0,
        failed_count: 0,
        first_error: None,
        skipped_roots: vec![indexer::SkippedRoot {
            source: parser::Source::Claude,
            preserved_sessions: 0,
            reason: indexer::SkippedReason::IncompleteEnumeration,
        }],
        preserved_embedded: 0,
        parse_diagnostics: Vec::new(),
        observations: serde_json::Value::Null,
    };

    let out = index_command_output(&outcome);

    assert!(
        out.notes
            .iter()
            .any(|n| n.contains("claude") && n.contains("some sessions may be missing")),
        "a zero-preserved incomplete root must warn that sessions may be missing, got: {:?}",
        out.notes
    );
    assert!(
        !out.notes
            .iter()
            .any(|n| n.contains("reconcile any real deletions")),
        "with nothing preserved the note must not promise to reconcile deletions, got: {:?}",
        out.notes
    );
    assert_eq!(
        out.data["skipped_roots"][0]["reason"], "incomplete_enumeration",
        "the payload still carries the machine-readable cause at zero preserved"
    );
}

fn search_command(query: &str) -> Command {
    Command::Search {
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
fn run_search_with_embedder_present_embeds_nothing() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    {
        let conn = open_or_create_db(&db_path).unwrap();
        db::seed_session(&conn, "s1");
        db::seed_chunk(&conn, 1, "pending content");
    }

    run_search_with(
        search_command("anything"),
        &Some(db_path.clone()),
        mock_loader,
    )
    .unwrap();

    let conn = open_or_create_db(&db_path).unwrap();
    assert_eq!(
        vec_chunk_count(&conn),
        0,
        "search must not embed: vec_chunks stays empty even with an embedder present"
    );
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

fn runtime_failing_loader() -> Result<Arc<dyn Embed>, DegradedReason> {
    Ok(Arc::new(embedder::MockEmbedder::failing_after(0)) as Arc<dyn Embed>)
}

fn no_load() -> Result<Arc<dyn Embed>, DegradedReason> {
    panic!("query-less --file must not load the embedder")
}

#[test]
fn run_search_query_less_file_list_skips_embedder_and_stays_nominal() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    {
        let conn = open_or_create_db(&db_path).unwrap();
        db::seed_session(&conn, "s1");
        db::seed_session_file(&conn, "s1", "/proj/a.rs");
    }

    let out = run_search_with(
        Command::Search {
            query: None,
            project: None,
            days: None,
            source: None,
            file: Some("/proj/a.rs".to_owned()),
            limit: 10,
            exclude_current: false,
            include_current: false,
            only_current: false,
            include_automated: false,
        },
        &Some(db_path),
        no_load,
    )
    .unwrap();

    assert_eq!(
        out.data["results"].as_array().unwrap().len(),
        1,
        "the file list itself must succeed, got: {}",
        out.data
    );
    assert!(
        !out.degraded,
        "a complete query-less file list must not be marked degraded"
    );
    assert!(
        out.notes
            .iter()
            .all(|n| !n.contains("semantic search unavailable")),
        "no semantic-search note may appear on the MATCH-free path, got: {:?}",
        out.notes
    );
}

#[test]
fn run_search_reports_degraded_when_vector_search_fails_at_runtime() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    {
        let mut conn = open_or_create_db(&db_path).unwrap();
        db::seed_session(&conn, "s1");
        db::seed_chunk(&conn, 1, "authentication flow");
        embed_all_pending(
            &mut conn,
            &embedder::MockEmbedder::new(),
            &embedder::EmbedOptions::default(),
        )
        .unwrap();
    }

    let out = run_search_with(
        search_command("authentication"),
        &Some(db_path),
        runtime_failing_loader,
    )
    .unwrap();

    assert!(
        out.degraded,
        "a runtime vector-search failure must mark the envelope degraded"
    );
    assert!(
        out.notes
            .iter()
            .any(|n| n.contains("vector query failed at runtime")),
        "the runtime degradation note must surface for the agent channel, got: {:?}",
        out.notes
    );
}

#[test]
fn run_search_reports_degraded_when_results_incomplete() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    {
        let mut conn = open_or_create_db(&db_path).unwrap();
        db::seed_session(&conn, "s1");
        db::seed_chunk(&conn, 1, "authentication flow");
        conn.execute(
                "INSERT INTO sessions (session_id, source, file_path, project, slug, timestamp, mtime) VALUES ('weird', 'gemini', '/f', '/p', 'slug', 0, 0.0)",
                [],
            )
            .unwrap();
        conn.execute(
                "INSERT INTO messages (session_id, role, text) VALUES ('weird', 'user', 'authentication retry')",
                [],
            )
            .unwrap();
        conn.execute(
                "INSERT INTO qa_chunks (id, session_id, content, timestamp) VALUES (2, 'weird', 'authentication retry', 0)",
                [],
            )
            .unwrap();
        embed_all_pending(
            &mut conn,
            &embedder::MockEmbedder::new(),
            &embedder::EmbedOptions::default(),
        )
        .unwrap();
    }

    let out = run_search_with(
        search_command("authentication"),
        &Some(db_path),
        mock_loader,
    )
    .unwrap();

    assert!(
        out.degraded,
        "a silently pruned match must mark the envelope degraded"
    );
    assert!(
        out.notes
            .iter()
            .any(|n| n.contains("results may be incomplete")),
        "the incompleteness note must surface for the agent channel, got: {:?}",
        out.notes
    );
}

fn payload_keys(v: &serde_json::Value) -> Vec<String> {
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
fn freeze_index_payload_key_set() {
    let outcome = IndexOutcome {
        degraded_note: None,
        embedded: 4,
        failed_count: 0,
        first_error: None,
        skipped_roots: vec![indexer::SkippedRoot {
            source: parser::Source::Codex,
            preserved_sessions: 2,
            reason: indexer::SkippedReason::IncompleteEnumeration,
        }],
        preserved_embedded: 0,
        parse_diagnostics: Vec::new(),
        observations: serde_json::Value::Null,
    };

    let out = index_command_output(&outcome);

    assert_eq!(
        payload_keys(&out.data),
        [
            "embedded",
            "failed_count",
            "observations",
            "parse_diagnostics",
            "preserved_embedded",
            "skipped_roots"
        ],
        "index --json top-level keys are frozen"
    );
    assert_eq!(
        payload_keys(&out.data["skipped_roots"][0]),
        ["preserved_sessions", "reason", "source"],
        "index --json skipped_roots element keys are frozen"
    );
}

#[test]
fn freeze_status_payload_key_set_both_branches() {
    let frozen = ["embedded", "model_ready", "qa_chunks", "sessions"];

    let missing = tempfile::TempDir::new().unwrap();
    let no_db = run_status(false, &Some(missing.path().join("absent.db"))).unwrap();
    assert_eq!(
        payload_keys(&no_db.data),
        frozen,
        "status --json keys are frozen on the no-database branch"
    );

    let dir = tempfile::TempDir::new().unwrap();
    let db_path = dir.path().join("recall.db");
    {
        let conn = open_or_create_db(&db_path).unwrap();
        db::seed_session(&conn, "s1");
    }
    let with_db = run_status(false, &Some(db_path)).unwrap();
    assert_eq!(
        payload_keys(&with_db.data),
        frozen,
        "status --json keys are frozen on the with-database branch"
    );
}

#[test]
fn freeze_search_payload_key_set_both_branches() {
    let empty_dir = tempfile::TempDir::new().unwrap();
    let empty_path = empty_dir.path().join("empty.db");
    open_or_create_db(&empty_path).unwrap();
    let empty = run_search(search_command("anything"), &Some(empty_path)).unwrap();
    assert_eq!(
        payload_keys(&empty.data),
        ["results"],
        "search --json top-level keys are frozen on the empty branch"
    );
    assert_eq!(
        empty.data["results"].as_array().unwrap().len(),
        0,
        "the empty branch emits an empty results array"
    );

    let dir = tempfile::TempDir::new().unwrap();
    let db_path = dir.path().join("recall.db");
    {
        let conn = open_or_create_db(&db_path).unwrap();
        db::seed_session(&conn, "s1");
        conn.execute(
            "INSERT INTO messages (session_id, role, text) \
                 VALUES ('s1', 'user', 'frozen envelope contract')",
            [],
        )
        .unwrap();
    }
    let populated = run_search(search_command("frozen"), &Some(db_path)).unwrap();
    assert_eq!(
        payload_keys(&populated.data),
        ["results"],
        "search --json top-level keys are frozen on the populated branch"
    );
    let results = populated.data["results"].as_array().unwrap();
    assert!(
        !results.is_empty(),
        "the populated branch must return at least one hit, got: {}",
        populated.data
    );
    assert_eq!(
        payload_keys(&results[0]),
        [
            "chunk",
            "excerpt",
            "inspect",
            "project",
            "session_id",
            "slug",
            "source",
            "timestamp"
        ],
        "search --json result element keys are frozen"
    );
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
fn u002_fts_hit_json_carries_chunk_id_and_inspect_guidance() {
    let (_dir, conn) = db::setup_test_db();
    let sid = "a1b2c3d4-e5f6-7890-1234-567890abcdef";
    conn.execute(
        "INSERT INTO sessions (session_id, source, file_path, project, slug, timestamp, mtime) \
             VALUES (?1, 'claude', '', '/proj', ?1, 1709251200000, 0.0)",
        [sid],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO messages (session_id, role, text) \
             VALUES (?1, 'user', 'how to implement authentication')",
        [sid],
    )
    .unwrap();
    conn.execute(
            "INSERT INTO qa_chunks (id, session_id, content, timestamp, src_rowid_lo, src_rowid_hi) \
             VALUES (42, ?1, 'how to implement authentication — use JWT access tokens', 1709251200000, 1, 1000000)",
            [sid],
        )
        .unwrap();

    let results =
        search::search(&conn, "authentication", &search::SearchOptions::default()).unwrap();
    assert_eq!(
        results.len(),
        1,
        "setup guard: the FTS query must return the one session"
    );

    let json = result_to_json(&results[0]);
    assert_eq!(
        json["chunk"],
        serde_json::json!(42),
        "an FTS rowid-tier hit must expose its resolved chunk id as the drill-down anchor, got: {json}"
    );
    let payload = json.to_string();
    assert!(
        payload.contains(&format!("recall show {sid} --chunk 42 --window 3")),
        "the hit must carry the inspect guidance line with the full (un-shortened) session_id, got: {payload}"
    );
}

#[test]
fn u002_vector_hit_json_carries_chunk_id_and_inspect_guidance() {
    let (_dir, conn) = db::setup_test_db();
    let now_ms = 1_750_000_000_000_i64;
    let sid = "0f9e8d7c-6b5a-4321-fedc-ba9876543210";
    conn.execute(
        "INSERT INTO sessions (session_id, source, file_path, project, slug, timestamp, mtime) \
             VALUES (?1, 'claude', '', '/proj', ?1, ?2, 0.0)",
        rusqlite::params![sid, now_ms],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO messages (session_id, role, text) \
             VALUES (?1, 'user', 'gardening and weather notes')",
        [sid],
    )
    .unwrap();
    let embedding = embedder::MockEmbedder::deterministic_vector("authentication");
    let embedding_bytes: &[u8] = embedder::f32_as_bytes(&embedding);
    conn.execute(
        "INSERT INTO qa_chunks (id, session_id, content, timestamp) \
             VALUES (7, ?1, 'authentication', ?2)",
        rusqlite::params![sid, now_ms],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO vec_chunks (chunk_id, embedding) VALUES (7, ?1)",
        rusqlite::params![embedding_bytes],
    )
    .unwrap();

    let mock = embedder::MockEmbedder::new();
    let opts = search::SearchOptions {
        now_ms: Some(now_ms),
        ..Default::default()
    };
    let outcome =
        search::search_with_embedder(&conn, "authentication", &opts, Some(&mock)).unwrap();
    let hit = outcome
        .results
        .iter()
        .find(|r| r.session.session_id == sid)
        .expect("setup guard: the vector leg must surface the session");

    let json = result_to_json(hit);
    assert_eq!(
        json["chunk"],
        serde_json::json!(7),
        "a vector-tier hit must expose its closest chunk id as the drill-down anchor, got: {json}"
    );
    let payload = json.to_string();
    assert!(
        payload.contains(&format!("recall show {sid} --chunk 7 --window 3")),
        "the vector hit must carry the inspect guidance line with the full (un-shortened) session_id, got: {payload}"
    );
}

#[test]
fn u002_snippet_hit_json_guides_full_show_without_inspect_line() {
    let (_dir, conn) = db::setup_test_db();
    let sid = "deadbeef-0000-1111-2222-333344445555";
    conn.execute(
        "INSERT INTO sessions (session_id, source, file_path, project, slug, timestamp, mtime) \
             VALUES (?1, 'claude', '', '/proj', ?1, 1709251200000, 0.0)",
        [sid],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO messages (session_id, role, text) \
             VALUES (?1, 'user', 'authentication flow discussion')",
        [sid],
    )
    .unwrap();

    let results =
        search::search(&conn, "authentication", &search::SearchOptions::default()).unwrap();
    assert_eq!(
        results.len(),
        1,
        "setup guard: the FTS query must return the one session"
    );
    assert!(
        results[0].excerpt.contains("**"),
        "setup guard: the hit must be snippet-tier (** highlight markers), got: {:?}",
        results[0].excerpt
    );

    let json = result_to_json(&results[0]);
    let payload = json.to_string();
    assert!(
        payload.contains(&format!("recall show {sid}")),
        "a snippet-tier hit must guide to the full-conversation show, got: {payload}"
    );
    assert!(
        json["chunk"].is_null(),
        "a snippet-tier hit has no chunk anchor, got: {json}"
    );
    assert!(
        !payload.contains("--chunk"),
        "a snippet-tier hit must not emit a --chunk inspect line, got: {payload}"
    );
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
