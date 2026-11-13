use std::path::{Path, PathBuf};

use crate::model::tests::FakeHost;
use crate::tests::{
    bare_empty_db, mock_loader, payload_keys, printed_envelope, search_command, seed_healthy_index,
    vec_chunk_count,
};
use crate::{embedder, open_or_create_db};

use super::*;

/// The markdown of a text-only search for "authentication" in the index of "authentication flow".
const AUTHENTICATION_FLOW_MATCH_MARKDOWN: &str = "Found 1 sessions:\n\n[1] 1970-01-01 | slug | p [claude]\n    /p\n    ID: s1\n    > **authentication** flow\n    inspect: recall show s1\n\n";

/// The `--json` results of the same search.
fn authentication_flow_match_results() -> serde_json::Value {
    serde_json::json!([{
        "chunk": null,
        "excerpt": "**authentication** flow",
        "inspect": "recall show s1",
        "project": "/p",
        "session_id": "s1",
        "slug": "slug",
        "source": "claude",
        "timestamp": 0,
    }])
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

    assert_eq!(out.markdown, AUTHENTICATION_FLOW_MATCH_MARKDOWN);
    assert_eq!(
        printed_envelope(&out),
        serde_json::json!({
            "data": { "results": authentication_flow_match_results() },
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

    assert_eq!(out.markdown, AUTHENTICATION_FLOW_MATCH_MARKDOWN);
    assert_eq!(
        printed_envelope(&out),
        serde_json::json!({
            "data": { "results": authentication_flow_match_results() },
            "degraded": true,
            "notes": ["semantic search unavailable: embedding model unavailable; results from text search only"],
        })
    );
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
                 (42, 'a1b2c3d4-e5f6-7890-1234-567890abcdef', 'Q: how to implement authentication' || char(10) || 'A: use short-lived JWT access tokens, rotate the refresh token on every use, keep it in an httpOnly cookie, and check the signature and the expiry of the access token on every request before the handler runs, then log each refresh with the client id so that a reused refresh token revokes every token issued from the same login', 1709251200000, 2, 2);",
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
            "    > A: use short-lived JWT access tokens, rotate the refresh token on every use, keep it in an httpOnly cookie, and check the signature and the expiry of the access token on every request before the handler runs, then log each refresh with the client id so that a reused refresh token revokes every token issued from the same login\n",
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
                        "excerpt": "Q: how to implement authentication\nA: use short-lived JWT access tokens, rotate the refresh token on every use, keep it in an httpOnly cookie, and check the signature and the expiry of the access token on every request before the handler runs, then log each refresh with the client id so that a reused refresh token revokes every token issued from the same login",
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
    // The --json excerpt still carries the raw control characters (#426), so
    // the comparison leaves it out.
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
            file: Some("/proj/a.rs".to_owned()),
            ..search_command("")
        },
        &Some(db_path),
        no_load,
    )
    .unwrap();

    assert_eq!(
        out.markdown,
        concat!(
            "Found 1 sessions:\n",
            "\n",
            "[1] 1970-01-01 | slug | p [claude]\n",
            "    /p\n",
            "    ID: s1\n",
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
                    "excerpt": "",
                    "inspect": "recall show s1",
                    "project": "/p",
                    "session_id": "s1",
                    "slug": "slug",
                    "source": "claude",
                    "timestamp": 0,
                }],
            },
            "degraded": false,
            "notes": [],
        })
    );
}

#[test]
fn search_whose_vector_query_fails_returns_the_text_match_with_a_text_only_note() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    {
        let conn = open_or_create_db(&db_path).unwrap();
        seed_healthy_index(&conn);
        conn.execute(
            "INSERT INTO messages (session_id, role, text) VALUES ('s1', 'user', 'healthy chunk')",
            [],
        )
        .unwrap();
    }

    let out = run_search_with(
        search_command("healthy"),
        &Some(db_path),
        runtime_failing_loader,
    )
    .unwrap();

    assert_eq!(
        out.markdown,
        concat!(
            "Found 1 sessions:\n",
            "\n",
            "[1] 1970-01-01 | slug | p [claude]\n",
            "    /p\n",
            "    ID: s1\n",
            "    > healthy chunk\n",
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
                    "excerpt": "healthy chunk",
                    "inspect": "recall show s1",
                    "project": "/p",
                    "session_id": "s1",
                    "slug": "slug",
                    "source": "claude",
                    "timestamp": 0,
                }],
            },
            "degraded": true,
            "notes": ["semantic search unavailable: vector query failed at runtime; results are text-only"],
        })
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

    assert_eq!(
        out.markdown,
        concat!(
            "Found 1 sessions:\n",
            "\n",
            "[1] 1970-01-01 | slug | p [claude]\n",
            "    /p\n",
            "    ID: s1\n",
            "    > **authentication**\n",
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
                    "excerpt": "**authentication**",
                    "inspect": "recall show s1",
                    "project": "/p",
                    "session_id": "s1",
                    "slug": "slug",
                    "source": "claude",
                    "timestamp": 0,
                }],
            },
            "degraded": false,
            "notes": [],
        })
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
fn search_shows_a_vector_only_match_with_its_closest_chunk_and_a_chunk_inspect_line() {
    // No message holds the query word, so only the vector search finds the session.
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    {
        let conn = open_or_create_db(&db_path).unwrap();
        conn.execute_batch(
            "INSERT INTO sessions (session_id, source, file_path, project, slug, timestamp, mtime)
                 VALUES ('0f9e8d7c-6b5a-4321-fedc-ba9876543210', 'claude', '/f', '/proj', 'garden-notes', 1709251200000, 0.0);
             INSERT INTO messages (session_id, role, text)
                 VALUES ('0f9e8d7c-6b5a-4321-fedc-ba9876543210', 'user', 'gardening and weather notes');
             INSERT INTO qa_chunks (id, session_id, content, timestamp)
                 VALUES (7, '0f9e8d7c-6b5a-4321-fedc-ba9876543210', 'authentication', 1709251200000);",
        )
        .unwrap();
        let embedding = embedder::MockEmbedder::deterministic_vector("authentication");
        conn.execute(
            "INSERT INTO vec_chunks (embedding, chunk_id, sub_idx) VALUES (?1, 7, 0)",
            [embedder::f32_as_bytes(&embedding)],
        )
        .unwrap();
    }

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
            "[1] 2024-03-01 | garden-notes | proj [claude]\n",
            "    /proj\n",
            "    ID: 0f9e8d7c-6b5a-4321-fedc-ba9876543210\n",
            "    > authentication\n",
            "    inspect: recall show 0f9e8d7c-6b5a-4321-fedc-ba9876543210 --chunk 7 --window 3\n",
            "\n",
        )
    );
    assert_eq!(
        printed_envelope(&out),
        serde_json::json!({
            "data": {
                "results": [{
                    "chunk": 7,
                    "excerpt": "authentication",
                    "inspect": "recall show 0f9e8d7c-6b5a-4321-fedc-ba9876543210 --chunk 7 --window 3",
                    "project": "/proj",
                    "session_id": "0f9e8d7c-6b5a-4321-fedc-ba9876543210",
                    "slug": "garden-notes",
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
fn search_shows_a_match_outside_any_chunk_as_a_snippet_with_a_full_show_inspect_line() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    open_or_create_db(&db_path)
        .unwrap()
        .execute_batch(
            "INSERT INTO sessions (session_id, source, file_path, project, slug, timestamp, mtime)
                 VALUES ('deadbeef-0000-1111-2222-333344445555', 'claude', '/f', '/proj', 'auth-talk', 1709251200000, 0.0);
             INSERT INTO messages (session_id, role, text)
                 VALUES ('deadbeef-0000-1111-2222-333344445555', 'user', 'authentication flow discussion');",
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
            "[1] 2024-03-01 | auth-talk | proj [claude]\n",
            "    /proj\n",
            "    ID: deadbeef-0000-1111-2222-333344445555\n",
            "    > **authentication** flow di...\n",
            "    inspect: recall show deadbeef-0000-1111-2222-333344445555\n",
            "\n",
        )
    );
    assert_eq!(
        printed_envelope(&out),
        serde_json::json!({
            "data": {
                "results": [{
                    "chunk": null,
                    "excerpt": "**authentication** flow di...",
                    "inspect": "recall show deadbeef-0000-1111-2222-333344445555",
                    "project": "/proj",
                    "session_id": "deadbeef-0000-1111-2222-333344445555",
                    "slug": "auth-talk",
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
fn two_searches_in_one_environment_start_the_probe_only_for_the_first() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    seed_healthy_index(&open_or_create_db(&db_path).unwrap());
    let records_dir = db_dir.path().join("probe");
    let host = FakeHost::healthy();

    let first_search = host
        .serve(db_dir.path(), &records_dir, |model_host| {
            run_search_in(
                search_command("healthy chunk"),
                &Some(db_path.clone()),
                model_host,
            )
        })
        .unwrap();
    let second_search = host
        .serve(db_dir.path(), &records_dir, |model_host| {
            run_search_in(
                search_command("healthy chunk"),
                &Some(db_path.clone()),
                model_host,
            )
        })
        .unwrap();

    let expected_markdown = concat!(
        "Found 1 sessions:\n",
        "\n",
        "[1] 1970-01-01 | slug | p [claude]\n",
        "    /p\n",
        "    ID: s1\n",
        "    > healthy chunk\n",
        "    inspect: recall show s1 --chunk 1 --window 3\n",
        "\n",
    );
    let expected_envelope = serde_json::json!({
        "data": {
            "results": [{
                "chunk": 1,
                "excerpt": "healthy chunk",
                "inspect": "recall show s1 --chunk 1 --window 3",
                "project": "/p",
                "session_id": "s1",
                "slug": "slug",
                "source": "claude",
                "timestamp": 0,
            }],
        },
        "degraded": false,
        "notes": [],
    });
    assert_eq!(first_search.markdown, expected_markdown);
    assert_eq!(printed_envelope(&first_search), expected_envelope);
    assert_eq!(second_search.markdown, expected_markdown);
    assert_eq!(printed_envelope(&second_search), expected_envelope);
    assert_eq!(host.probes.get(), 1, "the second search skipped the probe");
    assert_eq!(host.loads.get(), 2, "each search loaded the model");
}
