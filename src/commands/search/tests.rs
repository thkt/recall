use crate::commands::index::embed_all_pending;
use crate::tests::{bare_empty_db, mock_loader, payload_keys, search_command, vec_chunk_count};
use crate::{embedder, open_or_create_db, parser};

use super::*;

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
fn search_on_indexed_db_without_sessions_returns_idle_without_loading_model() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    drop(open_or_create_db(&db_path).unwrap());

    let out = run_search_with(
        search_command("anything"),
        &Some(db_path),
        || -> Result<Arc<dyn Embed>, DegradedReason> {
            panic!("a DB with no sessions must short-circuit before loading the model")
        },
    )
    .unwrap();

    assert_eq!(out.data, serde_json::json!({ "results": [] }));
    assert!(!out.degraded, "an idle search is not degraded");
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
        SearchArgs {
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
