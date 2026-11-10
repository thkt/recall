use std::fs;

use serde_json::json;

use super::*;
use crate::parser::{assert_classification_matches, write_test_jsonl as write_jsonl};

const USER_TURN: &str = r#"{"timestamp":"2026-01-18T01:40:00Z","type":"response_item","payload":{"role":"user","content":[{"type":"input_text","text":"hello"}]}}"#;
const UUID_NAME: &str = "rollout-2026-01-18T01-37-09-019bccd1-564a-73b3-b3b0-f6b12671ed24.jsonl";

/// Parses with the full reader and checks the classification reader agrees.
fn parse_codex_session(path: &Path) -> Result<Option<ParseResult>> {
    let parsed = super::parse_codex_session(path)?;
    if let Some(full) = &parsed {
        assert_classification_matches(path, Source::Codex, full);
    }
    Ok(parsed)
}

fn turns(parsed: &ParseResult) -> Vec<(Role, &str)> {
    parsed
        .messages
        .iter()
        .map(|m| (m.role, m.text.as_str()))
        .collect()
}

/// The slug of a one-turn Codex log stored at `relative` under a fresh directory.
fn slug_at(relative: &str) -> String {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join(relative);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(&path, USER_TURN).unwrap();
    parse_codex_session(&path).unwrap().unwrap().metadata.slug
}

/// Whether a log whose only record is a `session_meta` carrying `payload` reads as automated.
fn automated_meta(mut payload: Value) -> bool {
    payload["id"] = json!("session");
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("rollout-test.jsonl");
    fs::write(
        &path,
        json!({"type": "session_meta", "payload": payload}).to_string(),
    )
    .unwrap();
    parse_codex_session(&path)
        .unwrap()
        .unwrap()
        .origin
        .automated
}

#[test]
fn user_and_assistant_turns_become_messages() {
    let tmp = write_jsonl(&[
        r#"{"timestamp":"2026-01-17T16:39:33Z","type":"session_meta","payload":{"id":"abc-123","cwd":"/home/me/codex-proj"}}"#,
        r#"{"timestamp":"2026-01-17T16:40:00Z","type":"response_item","payload":{"role":"user","content":[{"type":"input_text","text":"hello from codex"}]}}"#,
        r#"{"timestamp":"2026-01-17T16:40:01Z","type":"response_item","payload":{"role":"assistant","content":[{"type":"output_text","text":"hi back"}]}}"#,
    ]);

    assert_eq!(
        turns(&parse_codex_session(tmp.path()).unwrap().unwrap()),
        [
            (Role::User, "hello from codex"),
            (Role::Assistant, "hi back")
        ]
    );
}

#[test]
fn developer_turns_and_injected_instructions_are_not_messages() {
    let tmp = write_jsonl(&[
        r#"{"type":"response_item","payload":{"role":"developer","content":[{"type":"input_text","text":"system stuff"}]}}"#,
        r#"{"type":"response_item","payload":{"role":"user","content":[{"type":"input_text","text":"<user_instructions>system instructions</user_instructions>"}]}}"#,
        r#"{"type":"response_item","payload":{"role":"user","content":[{"type":"input_text","text":"actual question"}]}}"#,
    ]);

    assert_eq!(
        turns(&parse_codex_session(tmp.path()).unwrap().unwrap()),
        [(Role::User, "actual question")]
    );
}

#[test]
fn session_meta_sets_the_session_id_and_project() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join(UUID_NAME);
    fs::write(
        &path,
        r#"{"timestamp":"2026-01-17T16:39:33Z","type":"session_meta","payload":{"id":"019bccd3-8798-7e11-b1d1-c61959201d0b","cwd":"/home/me/codex-proj"}}"#,
    )
    .unwrap();

    let metadata = parse_codex_session(&path).unwrap().unwrap().metadata;

    assert_eq!(metadata.session_id, "019bccd3-8798-7e11-b1d1-c61959201d0b");
    assert_eq!(metadata.project, "/home/me/codex-proj");
    assert_eq!(metadata.source, Source::Codex);
}

#[test]
fn a_session_without_turns_is_kept_with_no_messages() {
    let tmp = write_jsonl(&[
        r#"{"timestamp":"2026-01-17T16:39:33Z","type":"session_meta","payload":{"id":"abc","cwd":"/proj"}}"#,
        r#"{"timestamp":"2026-01-17T16:40:00Z","type":"event_msg","payload":{}}"#,
    ]);

    assert_eq!(
        turns(&parse_codex_session(tmp.path()).unwrap().unwrap()),
        []
    );
}

#[test]
fn the_slug_joins_the_date_directory_and_the_uuid_from_the_file_name() {
    assert_eq!(
        slug_at(&format!("sessions/2026/01/18/{UUID_NAME}")),
        "2026-01-18-019bccd1"
    );
    assert_eq!(
        slug_at("sessions/2026/01/18/rollout-2026-01-19T00-00-00-x.jsonl"),
        "2026-01-18"
    );
}

#[test]
fn an_archived_log_takes_the_slug_date_from_its_rollout_name() {
    assert_eq!(
        slug_at(&format!("archived_sessions/{UUID_NAME}")),
        "2026-01-18-019bccd1"
    );
}

#[test]
fn a_name_without_a_valid_date_or_uuid_falls_back_to_the_session_id() {
    assert_eq!(
        slug_at("archived_sessions/rollout-x-019bccd1-564a-73b3-b3b0-f6b12671ed24.jsonl"),
        "019bccd1"
    );
    assert_eq!(
        slug_at("archived_sessions/rollout-2026-1-18T01.jsonl"),
        "rollout-"
    );
    assert_eq!(
        slug_at("archived_sessions/rollout-2026-13-01T00-00-00-x.jsonl"),
        "rollout-"
    );
    assert_eq!(
        slug_at("archived_sessions/rollout-2026-02-30T00-00-00-x.jsonl"),
        "rollout-"
    );
    assert_eq!(
        slug_at("archived_sessions/other-2026-01-18.jsonl"),
        "other-20"
    );
}

#[test]
fn known_automated_session_sources_mark_the_session_automated() {
    let parent = "01234567-89ab-cdef-0123-456789abcdef";
    assert!(automated_meta(json!({"source": {"subagent": "review"}})));
    assert!(automated_meta(json!({"source": {"subagent": "compact"}})));
    assert!(automated_meta(
        json!({"source": {"subagent": "memory_consolidation"}})
    ));
    assert!(automated_meta(json!({
        "source": {"subagent": {"thread_spawn": {"parent_thread_id": parent, "depth": 1}}}
    })));
    assert!(automated_meta(
        json!({"source": {"subagent": {"other": "guardian"}}})
    ));
    assert!(automated_meta(json!({"source": {"internal": "guardian"}})));
    assert!(automated_meta(
        json!({"source": {"internal": "memory_consolidation"}})
    ));
    assert!(automated_meta(json!({"thread_source": "subagent"})));
    assert!(automated_meta(json!({"thread_source": "guardian_review"})));
    assert!(automated_meta(
        json!({"thread_source": "memory_consolidation"})
    ));
    assert!(automated_meta(json!({
        "thread_source": "future_feature",
        "source": {"subagent": "review"}
    })));
    assert!(automated_meta(json!({
        "thread_source": "guardian_review",
        "source": {"subagent": {"other": "future"}}
    })));
}

#[test]
fn unknown_or_malformed_session_sources_fail_open_as_human() {
    let parent = "01234567-89ab-cdef-0123-456789abcdef";
    assert!(!automated_meta(json!({
        "source": "exec", "thread_source": "user", "originator": "guardian"
    })));
    assert!(!automated_meta(json!({
        "source": "vscode", "parent_thread_id": parent, "agent_nickname": "guardian"
    })));
    assert!(!automated_meta(
        json!({"thread_source": "agent_created_thread"})
    ));
    assert!(!automated_meta(
        json!({"source": {"subagent": {"other": "unknown"}}})
    ));
    assert!(!automated_meta(json!({"source": {"subAgent": "review"}})));
    assert!(!automated_meta(json!({"source": "subagent"})));
    assert!(!automated_meta(
        json!({"source": {"subagent": "thread_spawn"}})
    ));
    assert!(!automated_meta(
        json!({"source": {"subagent": {"thread_spawn": {}}}})
    ));
    assert!(!automated_meta(json!({
        "source": {"subagent": {"thread_spawn": {"parent_thread_id": parent}}}
    })));
    assert!(!automated_meta(json!({
        "source": {"subagent": {"thread_spawn": {"parent_thread_id": "bad", "depth": 1}}}
    })));
    assert!(!automated_meta(json!({
        "source": {"subagent": {"thread_spawn": {"parent_thread_id": parent, "depth": "1"}}}
    })));
    assert!(!automated_meta(json!({
        "source": {"subagent": {"thread_spawn": {"parent_thread_id": parent, "depth": 2_147_483_648_i64}}}
    })));
    assert!(!automated_meta(
        json!({"source": {"subagent": "review", "custom": "human"}})
    ));
    assert!(!automated_meta(json!({"source": {"internal": "future"}})));
    assert!(!automated_meta(json!({})));
}

#[test]
fn provenance_words_inside_a_message_do_not_mark_the_session_automated() {
    let tmp = write_jsonl(&[
        r#"{"type":"response_item","payload":{"role":"user","source":{"subagent":"review"},"content":"Explain approval review and {\"thread_source\":\"guardian_review\"}"}}"#,
    ]);

    assert!(
        !parse_codex_session(tmp.path())
            .unwrap()
            .unwrap()
            .origin
            .automated
    );
}

#[test]
fn mixed_record_shapes_keep_payload_roles_and_legacy_project_context() {
    let tmp = write_jsonl(&[
        "null",
        r#"{"type":"session_meta","payload":{"id":17,"thread_source":"subagent"}}"#,
        r#"{"type":"event_msg","role":"user","content":"ignored event"}"#,
        r#"{"type":"turn_context","role":"assistant","content":"ignored context"}"#,
        r#"{"type":"response_item","role":"user","payload":{"role":"developer","content":"ignored developer"}}"#,
        r#"{"type":"response_item","payload":{"role":17,"content":"ignored malformed role"}}"#,
        r#"{"type":"legacy","role":"user","content":[{"type":"input_text","text":"<environment_context>\nCurrent working directory: /legacy\n</environment_context>"}]}"#,
        r#"{"type":"response_item","role":"assistant","payload":{"role":"user","content":"question"}}"#,
        r#"{"role":"assistant","content":"answer"}"#,
    ]);

    let parsed = parse_codex_session(tmp.path()).unwrap().unwrap();

    assert_eq!(
        turns(&parsed),
        [(Role::User, "question"), (Role::Assistant, "answer")]
    );
    assert_eq!(parsed.metadata.project, "/legacy");
    assert!(parsed.origin.has_records);
    assert!(!parsed.origin.automated);
    assert!(parsed.diagnostics.is_empty());
}

const ROLLOUT_NAME: &str = "rollout-2026-04-27T00-00-00-01900000-0000-7000-8000-000000000001.jsonl";
const META: &str = r#"{"timestamp":"2026-04-27T00:00:00Z","type":"session_meta","payload":{"id":"01900000-0000-7000-8000-000000000001","cwd":"/proj"}}"#;
const TURNS: &str = concat!(
    r#"{"timestamp":"2026-04-27T00:00:01Z","type":"response_item","payload":{"role":"user","content":[{"type":"input_text","text":"compressed question"}]}}"#,
    "\n",
    r#"{"timestamp":"2026-04-27T00:00:02Z","type":"response_item","payload":{"role":"assistant","content":[{"type":"output_text","text":"compressed answer"}]}}"#,
    "\n",
);

/// Parses `body` from a plain log and from a `.jsonl.zst` copy in a dated directory.
fn plain_and_compressed(body: &str) -> (ParseResult, ParseResult) {
    let root = tempfile::tempdir().unwrap();
    let plain = root
        .path()
        .join("plain/sessions/2026/04/27")
        .join(ROLLOUT_NAME);
    let compressed = root
        .path()
        .join("zst/sessions/2026/04/27")
        .join(format!("{ROLLOUT_NAME}.zst"));
    fs::create_dir_all(plain.parent().unwrap()).unwrap();
    fs::create_dir_all(compressed.parent().unwrap()).unwrap();
    fs::write(&plain, body).unwrap();
    fs::write(&compressed, zstd::encode_all(body.as_bytes(), 0).unwrap()).unwrap();
    (
        parse_codex_session(&plain).unwrap().unwrap(),
        parse_codex_session(&compressed).unwrap().unwrap(),
    )
}

#[test]
fn a_compressed_log_with_session_meta_parses_like_its_plain_copy() {
    let (plain, compressed) = plain_and_compressed(&format!("{META}\n{TURNS}"));

    assert_eq!(
        compressed.metadata.session_id,
        "01900000-0000-7000-8000-000000000001"
    );
    assert_eq!(compressed.metadata.slug, "2026-04-27-01900000");
    assert_eq!(compressed.metadata.session_id, plain.metadata.session_id);
    assert_eq!(compressed.metadata.slug, plain.metadata.slug);
    assert_eq!(compressed.metadata.project, plain.metadata.project);
    assert_eq!(compressed.metadata.timestamp, plain.metadata.timestamp);
    assert_eq!(
        turns(&compressed),
        [
            (Role::User, "compressed question"),
            (Role::Assistant, "compressed answer")
        ]
    );
    assert_eq!(compressed.origin, plain.origin);
    assert!(compressed.diagnostics.is_empty());
}

#[test]
fn a_compressed_log_without_session_meta_takes_its_id_from_the_name_like_its_plain_copy() {
    let (plain, compressed) = plain_and_compressed(TURNS);

    assert_eq!(
        compressed.metadata.session_id,
        "rollout-2026-04-27T00-00-00-01900000-0000-7000-8000-000000000001"
    );
    assert_eq!(compressed.metadata.session_id, plain.metadata.session_id);
    assert_eq!(compressed.metadata.slug, plain.metadata.slug);
    assert_eq!(turns(&compressed), turns(&plain));
    assert_eq!(compressed.origin, plain.origin);
    assert!(compressed.diagnostics.is_empty());
}
