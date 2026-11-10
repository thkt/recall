use std::fs;

use serde_json::json;

use super::*;
use crate::parser::{assert_classification_matches, write_test_jsonl as write_jsonl};

/// Parses with the full reader and checks the classification reader agrees.
fn parse_claude_session(path: &Path) -> Result<Option<ParseResult>> {
    let parsed = super::parse_claude_session(path)?;
    if let Some(full) = &parsed {
        assert_classification_matches(path, Source::Claude, full);
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

/// Whether a Claude log stored at `relative` under a fresh directory, holding only
/// `record`, reads as automated.
fn automated_at(relative: &str, record: &Value) -> bool {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join(relative);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(&path, record.to_string()).unwrap();
    parse_claude_session(&path)
        .unwrap()
        .unwrap()
        .origin
        .automated
}

#[test]
fn string_content_becomes_a_message() {
    let tmp = write_jsonl(&[
        r#"{"type":"user","cwd":"/home/me/proj","message":{"role":"user","content":"hello world"},"timestamp":"2026-03-01T00:00:00Z"}"#,
    ]);

    assert_eq!(
        turns(&parse_claude_session(tmp.path()).unwrap().unwrap()),
        [(Role::User, "hello world")]
    );
}

#[test]
fn only_the_text_blocks_of_an_answer_become_message_text() {
    let tmp = write_jsonl(&[
        r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"answer"},{"type":"tool_use","id":"t1","name":"Bash","input":{}},{"type":"thinking","thinking":"hmm"}]},"timestamp":"2026-03-01T00:00:00Z"}"#,
    ]);

    assert_eq!(
        turns(&parse_claude_session(tmp.path()).unwrap().unwrap()),
        [(Role::Assistant, "answer")]
    );
}

#[test]
fn records_that_are_not_turns_and_meta_turns_are_skipped() {
    let tmp = write_jsonl(&[
        r#"{"type":"file-history-snapshot","messageId":"abc","snapshot":{}}"#,
        r#"{"type":"progress","data":{"type":"hook_progress"}}"#,
        r#"{"type":"user","isMeta":true,"message":{"role":"user","content":"<local-command-caveat>Caveat: ...</local-command-caveat>"},"timestamp":"2026-03-01T00:00:00Z"}"#,
        r#"{"type":"user","message":{"role":"user","content":"real message"},"timestamp":"2026-03-01T00:00:00Z"}"#,
    ]);

    assert_eq!(
        turns(&parse_claude_session(tmp.path()).unwrap().unwrap()),
        [(Role::User, "real message")]
    );
}

#[test]
fn session_metadata_comes_from_the_records() {
    let tmp = write_jsonl(&[
        r#"{"type":"user","cwd":"/home/me/project","slug":"my-session","message":{"role":"user","content":"hello"},"timestamp":"2026-03-01T12:00:00Z"}"#,
    ]);

    let metadata = parse_claude_session(tmp.path()).unwrap().unwrap().metadata;

    assert_eq!(metadata.source, Source::Claude);
    assert_eq!(metadata.project, "/home/me/project");
    assert_eq!(metadata.slug, "my-session");
    assert_eq!(metadata.timestamp, Some(1_772_366_400_000));
}

#[test]
fn invalid_json_lines_are_skipped_and_counted() {
    let tmp = write_jsonl(&[
        "not valid json",
        r#"{"type":"user","message":{"role":"user","content":"ok"},"timestamp":"2026-03-01T00:00:00Z"}"#,
        "{broken",
    ]);

    let parsed = parse_claude_session(tmp.path()).unwrap().unwrap();

    assert_eq!(turns(&parsed), [(Role::User, "ok")]);
    assert_eq!(parsed.diagnostics.invalid_json_lines, 2);
}

#[test]
fn without_a_slug_the_start_of_the_session_id_becomes_the_slug() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("fallback-session-0001.jsonl");
    fs::write(
        &path,
        r#"{"type":"user","message":{"role":"user","content":"hi"},"timestamp":"2026-03-01T00:00:00Z"}"#,
    )
    .unwrap();

    assert_eq!(
        parse_claude_session(&path).unwrap().unwrap().metadata.slug,
        "fallback-ses"
    );
}

#[test]
fn a_session_without_turns_is_kept_with_no_messages() {
    let tmp = write_jsonl(&[
        r#"{"type":"progress","data":{"type":"hook_progress"}}"#,
        r#"{"type":"file-history-snapshot","messageId":"abc","snapshot":{}}"#,
    ]);

    assert_eq!(
        turns(&parse_claude_session(tmp.path()).unwrap().unwrap()),
        []
    );
}

#[test]
fn files_touched_by_tools_are_listed_once_in_first_seen_order() {
    let tmp = write_jsonl(&[
        r#"{"type":"user","message":{"role":"user","content":"edit the file twice"},"timestamp":"2026-03-01T00:00:00Z"}"#,
        r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"Edit","input":{"file_path":"/proj/same.rs","old_string":"a","new_string":"b"}},{"type":"tool_use","id":"t2","name":"Write","input":{"file_path":"/proj/other.rs","content":"c"}}]},"timestamp":"2026-03-01T00:00:01Z"}"#,
        r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","id":"t3","name":"Edit","input":{"file_path":"/proj/same.rs","old_string":"b","new_string":"d"}}]},"timestamp":"2026-03-01T00:00:02Z"}"#,
    ]);

    assert_eq!(
        parse_claude_session(tmp.path())
            .unwrap()
            .unwrap()
            .scanned_files,
        ["/proj/same.rs", "/proj/other.rs"]
    );
}

#[test]
fn a_boolean_sidechain_flag_on_a_turn_or_the_subagent_layout_marks_the_session_automated() {
    let parent = "01234567-89ab-cdef-0123-456789abcdef";
    assert!(automated_at(
        "session.jsonl",
        &json!({"type": "user", "isSidechain": true, "message": {"content": "help"}})
    ));
    assert!(automated_at(
        "session.jsonl",
        &json!({"type": "assistant", "isSidechain": true, "message": {"content": []}})
    ));
    assert!(automated_at(
        &format!("{parent}/subagents/agent-a.jsonl"),
        &json!({"type": "user", "sessionId": parent, "message": {"content": "help"}})
    ));
}

#[test]
fn other_sidechain_values_and_near_miss_layouts_fail_open_as_human() {
    let parent = "01234567-89ab-cdef-0123-456789abcdef";
    assert!(!automated_at(
        "session.jsonl",
        &json!({"type": "user", "isSidechain": "true", "message": {"content": "help"}})
    ));
    assert!(!automated_at(
        "session.jsonl",
        &json!({"type": "summary", "isSidechain": true})
    ));
    assert!(!automated_at(
        "session.jsonl",
        &json!({"type": "user", "agentId": "agent-a", "message": {"content": "Explain {\"isSidechain\":true}"}})
    ));
    assert!(!automated_at(
        "session.jsonl",
        &json!({"role": "user", "sessionId": "unrelated", "isSidechain": true, "content": "human question"})
    ));
    assert!(!automated_at(
        &format!("{parent}/subagents/agent-a.jsonl"),
        &json!({"type": "user", "sessionId": "other", "message": {"content": "help"}})
    ));
    assert!(!automated_at(
        "subagents/agent-a.jsonl",
        &json!({"type": "user"})
    ));
    assert!(!automated_at(
        "project/subagents/agent-a.jsonl",
        &json!({"type": "user"})
    ));
    assert!(!automated_at("agent-a.jsonl", &json!({"type": "user"})));
    assert!(!automated_at(
        &format!("{parent}/subagents/session.jsonl"),
        &json!({"type": "user"})
    ));
    assert!(!automated_at(
        &format!("{parent}/subagents/agent-.jsonl"),
        &json!({"type": "user"})
    ));
}

const AGENT_TURN: &str = r#"{"type":"user","sessionId":"11111111-1111-4111-8111-111111111111","agentId":"example","isSidechain":true,"message":{"content":"Review this sample module."}}"#;
const AGENT_ANSWER: &str = r#"{"sessionId":"11111111-1111-4111-8111-111111111111","agentId":"example","isSidechain":true,"role":"assistant","message":{"content":"The sample looks correct."}}"#;

#[test]
fn a_legacy_agent_log_is_automated_when_its_name_and_records_agree() {
    let root = tempfile::tempdir().unwrap();
    let alone = root.path().join("agent-example.jsonl");
    let nested = root.path().join("project/subagents/agent-example.jsonl");
    fs::create_dir_all(nested.parent().unwrap()).unwrap();
    fs::write(&alone, AGENT_TURN).unwrap();
    fs::write(&nested, format!("{AGENT_TURN}\n{AGENT_ANSWER}\n")).unwrap();

    let alone = parse_claude_session(&alone).unwrap().unwrap();
    let nested = parse_claude_session(&nested).unwrap().unwrap();

    assert_eq!(
        (alone.origin.automated, alone.origin.identity_conflict),
        (true, false)
    );
    assert_eq!(alone.metadata.session_id, "agent-example");
    assert_eq!(alone.messages.len(), 1);
    assert_eq!(
        (nested.origin.automated, nested.origin.identity_conflict),
        (true, false)
    );
    assert_eq!(nested.metadata.session_id, "agent-example");
    assert_eq!(nested.messages.len(), 2);
}

#[test]
fn legacy_agent_provenance_does_not_borrow_mismatched_or_incomplete_evidence() {
    let root = tempfile::tempdir().unwrap();
    let valid = json!({"type":"user","sessionId":"11111111-1111-4111-8111-111111111111","agentId":"example","isSidechain":true,"message":{"content":"Review this sample module."}});
    let mut invalid_records = Vec::new();
    for (field, values) in [
        (
            "agentId",
            vec![json!("other"), json!(""), json!(42), Value::Null],
        ),
        (
            "sessionId",
            vec![
                json!("22222222-2222-4222-8222-222222222222"),
                json!("invalid"),
                json!(""),
                json!(42),
                Value::Null,
            ],
        ),
        (
            "isSidechain",
            vec![json!("true"), json!(false), Value::Null],
        ),
    ] {
        for value in values {
            let mut record = valid.clone();
            if value.is_null() {
                record.as_object_mut().unwrap().remove(field);
            } else {
                record[field] = value;
            }
            invalid_records.push(record);
        }
    }
    let mut meta = valid.clone();
    meta["isMeta"] = json!(true);
    invalid_records.push(meta);
    invalid_records.push(json!({"type":"user","message":{"content":valid.to_string()}}));
    let path = root.path().join("agent-example.jsonl");
    for invalid in invalid_records {
        let mut role_only = invalid.clone();
        role_only.as_object_mut().unwrap().remove("type");
        role_only["role"] = json!("assistant");
        for records in [
            format!("{valid}\n{invalid}"),
            format!("{role_only}\n{valid}"),
        ] {
            fs::write(&path, &records).unwrap();
            let parsed = parse_claude_session(&path).unwrap().unwrap();
            assert!(!parsed.origin.automated, "{records}");
            assert!(parsed.origin.identity_conflict, "{records}");
        }
    }
    for relative in [
        "agent-other.jsonl",
        "agent-.jsonl",
        "session.jsonl",
        "agent-example.txt",
        "22222222-2222-4222-8222-222222222222/subagents/agent-example.jsonl",
    ] {
        let path = root.path().join(relative);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, valid.to_string()).unwrap();
        let parsed = parse_claude_session(&path).unwrap().unwrap();
        assert!(!parsed.origin.automated, "{relative}");
        assert!(parsed.origin.identity_conflict, "{relative}");
    }
}

#[test]
fn mixed_record_shapes_preserve_role_precedence_and_boolean_metadata() {
    let tmp = write_jsonl(&[
        "null",
        r#"{"type":"progress","cwd":"/project","slug":"from-progress","timestamp":100}"#,
        r#"{"type":"assistant","role":"user","isMeta":"true","isSidechain":"true","message":"question"}"#,
        r#"{"type":"human","role":"assistant","message":{"content":"follow-up"}}"#,
        r#"{"role":"assistant","content":[{"type":"text","text":"answer"}]}"#,
        r#"{"type":"user","isMeta":true,"cwd":"/ignored","timestamp":1,"message":"meta"}"#,
    ]);
    let parsed = parse_claude_session(tmp.path()).unwrap().unwrap();
    assert_eq!(
        turns(&parsed),
        [
            (Role::User, "question"),
            (Role::User, "follow-up"),
            (Role::Assistant, "answer")
        ]
    );
    assert_eq!(parsed.metadata.project, "/project");
    assert_eq!(parsed.metadata.slug, "from-progress");
    assert_eq!(parsed.metadata.timestamp, Some(100));
    assert!(parsed.origin.has_records);
    assert!(!parsed.origin.automated);
    assert!(parsed.diagnostics.is_empty());
}
