use std::path::{Path, PathBuf};

use crate::commands::index::embed_all_pending;
use crate::envelope::render_json_success;
use crate::model::tests::FakeHost;
use crate::tests::{
    bare_empty_db, mock_loader, payload_keys, search_command, seed_healthy_index, vec_chunk_count,
};
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

/// The `--json` envelope `recall search` prints for an output.
fn printed_envelope(out: &CommandOutput) -> serde_json::Value {
    serde_json::from_str(&render_json_success(out)).unwrap()
}

/// An index whose only session, `s1`, holds one user message.
fn index_with_one_message(dir: &Path, text: &str) -> PathBuf {
    let db_path = dir.join("recall.db");
    let conn = open_or_create_db(&db_path).unwrap();
    db::seed_session(&conn, "s1");
    conn.execute(
        "INSERT INTO messages (session_id, role, text) VALUES ('s1', 'user', ?1)",
        [text],
    )
    .unwrap();
    db_path
}

#[test]
fn search_without_the_model_installed_returns_text_matches_and_says_how_to_download_it() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = index_with_one_message(db_dir.path(), "authentication flow");

    let out = run_search_with(search_command("authentication"), &Some(db_path), || {
        Err(DegradedReason::NotInstalled)
    })
    .unwrap();

    assert_eq!(
        out.markdown,
        "Found 1 sessions:\n\n[1] 1970-01-01 | slug | p [claude]\n    /p\n    ID: s1\n    > **authentication** flow\n    inspect: recall show s1\n\n"
    );
    assert_eq!(
        printed_envelope(&out),
        serde_json::json!({
            "data": {
                "results": [{
                    "chunk": null,
                    "excerpt": "**authentication** flow",
                    "inspect": "recall show s1",
                    "project": "/p",
                    "session_id": "s1",
                    "slug": "slug",
                    "source": "claude",
                    "timestamp": 0,
                }],
            },
            "degraded": true,
            "notes": ["semantic search unavailable: embedding model not installed; run `recall model download` to enable semantic search"],
        })
    );
}

#[test]
fn search_whose_model_fails_to_load_returns_text_matches_with_an_unavailable_note() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = index_with_one_message(db_dir.path(), "authentication flow");

    let out = run_search_with(search_command("authentication"), &Some(db_path), || {
        Err(DegradedReason::ProbeFailed)
    })
    .unwrap();

    assert_eq!(
        out.markdown,
        "Found 1 sessions:\n\n[1] 1970-01-01 | slug | p [claude]\n    /p\n    ID: s1\n    > **authentication** flow\n    inspect: recall show s1\n\n"
    );
    assert_eq!(
        printed_envelope(&out),
        serde_json::json!({
            "data": {
                "results": [{
                    "chunk": null,
                    "excerpt": "**authentication** flow",
                    "inspect": "recall show s1",
                    "project": "/p",
                    "session_id": "s1",
                    "slug": "slug",
                    "source": "claude",
                    "timestamp": 0,
                }],
            },
            "degraded": true,
            "notes": ["semantic search unavailable: embedding model unavailable; results from text search only"],
        })
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
fn search_lists_each_match_with_its_date_slug_project_id_excerpt_and_inspect_line() {
    // Both sessions hold the same message at the same time, so their scores
    // tie and the session id orders them; the b2c3d4e5 session is inserted first.
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    open_or_create_db(&db_path)
        .unwrap()
        .execute_batch(
            "INSERT INTO sessions (session_id, source, file_path, project, slug, timestamp, mtime) VALUES
                 ('b2c3d4e5-f6a7-8901-2345-67890abcdef1', 'claude', '/home/me/.claude/projects/other/b2c3d4e5-f6a7-8901-2345-67890abcdef1.jsonl', '/home/me/other', 'add-login-tests', 1709251200000, 0.0),
                 ('a1b2c3d4-e5f6-7890-1234-567890abcdef', 'claude', '/home/me/.claude/projects/project/a1b2c3d4-e5f6-7890-1234-567890abcdef.jsonl', '/home/me/project', 'fix-login', 1709251200000, 0.0);
             INSERT INTO messages (rowid, session_id, role, text) VALUES
                 (1, 'b2c3d4e5-f6a7-8901-2345-67890abcdef1', 'user', 'how to implement authentication'),
                 (2, 'a1b2c3d4-e5f6-7890-1234-567890abcdef', 'user', 'how to implement authentication');
             INSERT INTO qa_chunks (id, session_id, content, timestamp, src_rowid_lo, src_rowid_hi) VALUES
                 (7, 'b2c3d4e5-f6a7-8901-2345-67890abcdef1', 'Q: how to implement authentication' || char(10) || 'A: start with the login test', 1709251200000, 1, 1),
                 (42, 'a1b2c3d4-e5f6-7890-1234-567890abcdef', 'Q: how to implement authentication' || char(10) || 'A: use short-lived JWT access tokens, rotate the refresh token on every use, keep it in an httpOnly cookie, and check the signature and the expiry of the access token on every request before the handler runs', 1709251200000, 2, 2);",
        )
        .unwrap();

    let out = run_search_with(
        search_command("authentication"),
        &Some(db_path),
        mock_loader,
    )
    .unwrap();

    assert_eq!(
        out.markdown,
        concat!(
            "Found 2 sessions:\n",
            "\n",
            "[1] 2024-03-01 | fix-login | project [claude]\n",
            "    /home/me/project\n",
            "    ID: a1b2c3d4-e5f6-7890-1234-567890abcdef\n",
            "    > Q: how to implement authentication\n",
            "    > A: use short-lived JWT access tokens, rotate the refresh token on every use, keep it in an httpOnly cookie, and check the signature and the expiry of the access token on every request before the handler runs\n",
            "    inspect: recall show a1b2c3d4-e5f6-7890-1234-567890abcdef --chunk 42 --window 3\n",
            "\n",
            "[2] 2024-03-01 | add-login-tests | other [claude]\n",
            "    /home/me/other\n",
            "    ID: b2c3d4e5-f6a7-8901-2345-67890abcdef1\n",
            "    > Q: how to implement authentication\n",
            "    > A: start with the login test\n",
            "    inspect: recall show b2c3d4e5-f6a7-8901-2345-67890abcdef1 --chunk 7 --window 3\n",
            "\n",
        )
    );
    assert_eq!(
        printed_envelope(&out),
        serde_json::json!({
            "data": {
                "results": [
                    {
                        "chunk": 42,
                        "excerpt": "Q: how to implement authentication\nA: use short-lived JWT access tokens, rotate the refresh token on every use, keep it in an httpOnly cookie, and check the signature and the expiry of the access token on every request before the handler runs",
                        "inspect": "recall show a1b2c3d4-e5f6-7890-1234-567890abcdef --chunk 42 --window 3",
                        "project": "/home/me/project",
                        "session_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
                        "slug": "fix-login",
                        "source": "claude",
                        "timestamp": 1709251200000_i64,
                    },
                    {
                        "chunk": 7,
                        "excerpt": "Q: how to implement authentication\nA: start with the login test",
                        "inspect": "recall show b2c3d4e5-f6a7-8901-2345-67890abcdef1 --chunk 7 --window 3",
                        "project": "/home/me/other",
                        "session_id": "b2c3d4e5-f6a7-8901-2345-67890abcdef1",
                        "slug": "add-login-tests",
                        "source": "claude",
                        "timestamp": 1709251200000_i64,
                    },
                ],
            },
            "degraded": false,
            "notes": [],
        })
    );
}

#[test]
fn search_that_matches_no_session_prints_no_matching_sessions_found() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = index_with_one_message(db_dir.path(), "authentication flow");

    let out = run_search_with(search_command("kubernetes"), &Some(db_path), mock_loader).unwrap();

    assert_eq!(out.markdown, "No matching sessions found.");
    assert_eq!(
        printed_envelope(&out),
        serde_json::json!({ "data": { "results": [] }, "degraded": false, "notes": [] })
    );
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
fn search_prints_unknown_for_a_session_without_a_project_and_omits_the_project_line() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    open_or_create_db(&db_path)
        .unwrap()
        .execute_batch(
            "INSERT INTO sessions (session_id, source, file_path, project, slug, timestamp, mtime)
                 VALUES ('s1', 'claude', '/f', '', 'slug', 1709251200000, 0.0);
             INSERT INTO messages (session_id, role, text) VALUES ('s1', 'user', 'authentication flow');",
        )
        .unwrap();

    let out = run_search_with(
        search_command("authentication"),
        &Some(db_path),
        mock_loader,
    )
    .unwrap();

    assert_eq!(
        out.markdown,
        concat!(
            "Found 1 sessions:\n",
            "\n",
            "[1] 2024-03-01 | slug | unknown [claude]\n",
            "    ID: s1\n",
            "    > **authentication** flow\n",
            "    inspect: recall show s1\n",
            "\n",
        )
    );
    assert_eq!(
        printed_envelope(&out),
        serde_json::json!({
            "data": {
                "results": [{
                    "chunk": null,
                    "excerpt": "**authentication** flow",
                    "inspect": "recall show s1",
                    "project": "",
                    "session_id": "s1",
                    "slug": "slug",
                    "source": "claude",
                    "timestamp": 1709251200000_i64,
                }],
            },
            "degraded": false,
            "notes": [],
        })
    );
}

#[test]
fn test_format_result_excerpt_not_truncated() {
    let long = "a".repeat(300);
    let r = make_search_result("s1", "/proj", &long);
    let mut output = String::new();
    format_result(&mut output, 0, &r).unwrap();
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
fn search_strips_control_characters_from_the_printed_excerpt() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    open_or_create_db(&db_path)
        .unwrap()
        .execute_batch(
            "INSERT INTO sessions (session_id, source, file_path, project, slug, timestamp, mtime)
                 VALUES ('s1', 'claude', '/f', '/p', 'slug', 1709251200000, 0.0);
             INSERT INTO messages (rowid, session_id, role, text)
                 VALUES (1, 's1', 'user', 'authentication failed at login');
             INSERT INTO qa_chunks (id, session_id, content, timestamp, src_rowid_lo, src_rowid_hi)
                 VALUES (1, 's1', 'authentication ' || char(27) || '[31mfailed' || char(27) || '[0m at login' || char(7), 1709251200000, 1, 1);",
        )
        .unwrap();

    let out = run_search_with(
        search_command("authentication"),
        &Some(db_path),
        mock_loader,
    )
    .unwrap();

    assert_eq!(
        out.markdown,
        concat!(
            "Found 1 sessions:\n",
            "\n",
            "[1] 2024-03-01 | slug | p [claude]\n",
            "    /p\n",
            "    ID: s1\n",
            "    > authentication failed at login\n",
            "    inspect: recall show s1 --chunk 1 --window 3\n",
            "\n",
        )
    );
    // The --json excerpt still carries the raw control characters; that gap
    // is tracked apart from this test, so the comparison leaves it out.
    let mut envelope = printed_envelope(&out);
    envelope["data"]["results"][0]
        .as_object_mut()
        .unwrap()
        .remove("excerpt");
    assert_eq!(
        envelope,
        serde_json::json!({
            "data": {
                "results": [{
                    "chunk": 1,
                    "inspect": "recall show s1 --chunk 1 --window 3",
                    "project": "/p",
                    "session_id": "s1",
                    "slug": "slug",
                    "source": "claude",
                    "timestamp": 1709251200000_i64,
                }],
            },
            "degraded": false,
            "notes": [],
        })
    );
}

#[test]
fn test_format_result_empty_excerpt() {
    let r = make_search_result("s1", "/proj", "");
    let mut output = String::new();
    format_result(&mut output, 0, &r).unwrap();
    assert!(!output.contains("> "));
}

#[test]
fn search_names_the_parent_session_of_a_subagent_log() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    open_or_create_db(&db_path)
        .unwrap()
        .execute_batch(
            "INSERT INTO sessions (session_id, source, file_path, project, slug, timestamp, mtime)
                 VALUES ('agent-a58c408', 'claude', '/home/me/.claude/projects/proj/abc-def-123/subagents/agent-a58c408.jsonl', '/home/me/proj', 'agent-slug', 1709251200000, 0.0);
             INSERT INTO messages (session_id, role, text) VALUES ('agent-a58c408', 'user', 'authentication flow');",
        )
        .unwrap();

    let out = run_search_with(
        search_command("authentication"),
        &Some(db_path),
        mock_loader,
    )
    .unwrap();

    assert_eq!(
        out.markdown,
        concat!(
            "Found 1 sessions:\n",
            "\n",
            "[1] 2024-03-01 | agent-slug | proj [claude]\n",
            "    /home/me/proj\n",
            "    ID: agent-a58c408\n",
            "    Parent: abc-def-123\n",
            "    > **authentication** flow\n",
            "    inspect: recall show agent-a58c408\n",
            "\n",
        )
    );
    assert_eq!(
        printed_envelope(&out),
        serde_json::json!({
            "data": {
                "results": [{
                    "chunk": null,
                    "excerpt": "**authentication** flow",
                    "inspect": "recall show agent-a58c408",
                    "project": "/home/me/proj",
                    "session_id": "agent-a58c408",
                    "slug": "agent-slug",
                    "source": "claude",
                    "timestamp": 1709251200000_i64,
                }],
            },
            "degraded": false,
            "notes": [],
        })
    );
}

#[test]
fn test_normal_session_path_shows_no_parent() {
    let r = make_search_result("s1", "/home/me/project", "excerpt");
    let mut output = String::new();
    format_result(&mut output, 0, &r).unwrap();
    assert!(!output.contains("Parent:"));
}

#[test]
fn search_on_a_database_file_without_tables_returns_no_results() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = bare_empty_db(db_dir.path());

    let out = run_search_with(search_command("anything"), &Some(db_path), no_load).unwrap();

    assert_eq!(out.markdown, "");
    assert_eq!(
        printed_envelope(&out),
        serde_json::json!({ "data": { "results": [] }, "degraded": false, "notes": [] })
    );
}

#[test]
fn search_on_an_index_without_sessions_returns_no_results_without_loading_the_model() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    drop(open_or_create_db(&db_path).unwrap());

    let out = run_search_with(search_command("anything"), &Some(db_path), no_load).unwrap();

    assert_eq!(out.markdown, "");
    assert_eq!(
        printed_envelope(&out),
        serde_json::json!({ "data": { "results": [] }, "degraded": false, "notes": [] })
    );
}

#[test]
fn search_with_a_loaded_model_leaves_a_pending_chunk_unembedded() {
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
    panic!("this search must not load the embedder")
}

#[test]
fn search_by_file_without_a_query_lists_its_session_without_loading_the_model() {
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
fn run_search_is_not_degraded_when_a_skipped_session_could_not_reach_the_results() {
    // "weird" has an unknown source and a much longer message, so even the
    // largest recency boost leaves it below "s1" at a limit of one.
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    {
        let conn = open_or_create_db(&db_path).unwrap();
        db::seed_session(&conn, "s1");
        conn.execute(
            "INSERT INTO messages (session_id, role, text) VALUES ('s1', 'user', 'authentication')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO sessions (session_id, source, file_path, project, slug, timestamp, mtime) VALUES ('weird', 'gemini', '/f', '/p', 'slug', 0, 0.0)",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO messages (session_id, role, text) VALUES ('weird', 'user', 'authentication retry logic for the session refresh path in the gateway service')",
            [],
        )
        .unwrap();
    }

    let out = run_search_with(
        SearchArgs {
            limit: 1,
            ..search_command("authentication")
        },
        &Some(db_path),
        mock_loader,
    )
    .unwrap();

    assert_eq!(out.data["results"][0]["session_id"], "s1", "{}", out.data);
    assert_eq!(
        (out.degraded, out.notes),
        (false, Vec::<String>::new()),
        "a skipped session below the results must not mark the envelope degraded"
    );
}

#[test]
fn freeze_search_payload_key_set() {
    let empty_dir = tempfile::TempDir::new().unwrap();
    let empty_path = empty_dir.path().join("empty.db");
    open_or_create_db(&empty_path).unwrap();
    let empty =
        run_search_with(search_command("anything"), &Some(empty_path), mock_loader).unwrap();
    assert_eq!(
        payload_keys(&empty.data),
        ["results"],
        "search --json top-level keys are frozen on the empty branch"
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
    let populated = run_search_with(search_command("frozen"), &Some(db_path), mock_loader).unwrap();
    assert_eq!(
        payload_keys(&populated.data),
        ["results"],
        "search --json top-level keys are frozen on the populated branch"
    );
    assert_eq!(
        payload_keys(&populated.data["results"][0]),
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

#[test]
fn two_searches_in_one_environment_start_the_probe_only_for_the_first() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    seed_healthy_index(&open_or_create_db(&db_path).unwrap());
    let records_dir = db_dir.path().join("probe");
    let host = FakeHost::healthy();

    for _ in 0..2 {
        let out = host
            .serve(db_dir.path(), &records_dir, |model_host| {
                run_search_in(
                    search_command("healthy chunk"),
                    &Some(db_path.clone()),
                    model_host,
                )
            })
            .unwrap();
        assert!(!out.degraded, "{:?}", out.notes);
        assert!(out.notes.is_empty(), "{:?}", out.notes);
        assert_eq!(out.data["results"].as_array().unwrap().len(), 1);
    }

    assert_eq!(host.probes.get(), 1);
    assert_eq!(host.loads.get(), 2);
}
