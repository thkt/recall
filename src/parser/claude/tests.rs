use super::*;
use crate::parser::{assert_classification_matches, write_test_jsonl as write_jsonl};

#[test]
fn test_claude_text_string_content() {
    let tmp = write_jsonl(&[
        r#"{"type":"user","cwd":"/home/me/proj","message":{"role":"user","content":"hello world"},"timestamp":"2026-03-01T00:00:00Z"}"#,
    ]);
    let result = parse_claude_session(tmp.path()).unwrap().unwrap();
    assert_eq!(result.messages.len(), 1);
    assert_eq!(result.messages[0].role, Role::User);
    assert_eq!(result.messages[0].text, "hello world");
}

#[test]
fn test_claude_block_array_extracts_text_only() {
    let tmp = write_jsonl(&[
        r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"answer"},{"type":"tool_use","id":"t1","name":"Bash","input":{}},{"type":"thinking","thinking":"hmm"}]},"timestamp":"2026-03-01T00:00:00Z"}"#,
    ]);
    let result = parse_claude_session(tmp.path()).unwrap().unwrap();
    assert_eq!(result.messages.len(), 1);
    assert_eq!(result.messages[0].text, "answer");
}

#[test]
fn test_claude_skips_non_message_types() {
    let tmp = write_jsonl(&[
        r#"{"type":"file-history-snapshot","messageId":"abc","snapshot":{}}"#,
        r#"{"type":"progress","data":{"type":"hook_progress"}}"#,
        r#"{"type":"user","message":{"role":"user","content":"real message"},"timestamp":"2026-03-01T00:00:00Z"}"#,
    ]);
    let result = parse_claude_session(tmp.path()).unwrap().unwrap();
    assert_eq!(result.messages.len(), 1);
    assert_eq!(result.messages[0].text, "real message");
}

#[test]
fn test_claude_metadata_extraction() {
    let tmp = write_jsonl(&[
        r#"{"type":"user","cwd":"/home/me/project","slug":"my-session","message":{"role":"user","content":"hello"},"timestamp":"2026-03-01T12:00:00Z"}"#,
    ]);
    let result = parse_claude_session(tmp.path()).unwrap().unwrap();
    assert_eq!(result.metadata.source, Source::Claude);
    assert_eq!(result.metadata.project, "/home/me/project");
    assert_eq!(result.metadata.slug, "my-session");
    assert!(result.metadata.timestamp.is_some_and(|ts| ts > 0));
}

#[test]
fn test_claude_invalid_json_skipped() {
    let tmp = write_jsonl(&[
        "not valid json",
        r#"{"type":"user","message":{"role":"user","content":"ok"},"timestamp":"2026-03-01T00:00:00Z"}"#,
        "{broken",
    ]);
    let result = parse_claude_session(tmp.path()).unwrap().unwrap();
    assert_eq!(result.messages.len(), 1);
}

#[test]
fn test_claude_slug_fallback_to_session_id() {
    let tmp = write_jsonl(&[
        r#"{"type":"user","message":{"role":"user","content":"hi"},"timestamp":"2026-03-01T00:00:00Z"}"#,
    ]);
    let result = parse_claude_session(tmp.path()).unwrap().unwrap();
    assert!(!result.metadata.slug.is_empty());
}

#[test]
fn claude_empty_messages_are_retained_for_diagnostics() {
    let tmp = write_jsonl(&[
        r#"{"type":"progress","data":{"type":"hook_progress"}}"#,
        r#"{"type":"file-history-snapshot","messageId":"abc","snapshot":{}}"#,
    ]);
    assert!(
        parse_claude_session(tmp.path())
            .unwrap()
            .unwrap()
            .messages
            .is_empty()
    );
}

// U-002 dedupe: a write-target path repeated across tool_use blocks — within one
// assistant turn and again in a later turn — collapses to a single scanned_files
// entry, preserving first-seen order. T-002/T-003 never repeat a path, so this is
// the only cover for the contract's "セッション内 dedupe" clause. The two assistant
// turns carry no text block, so this also pins that a tool_use-only turn still
// contributes paths (extraction runs before the empty-text early return).
#[test]
fn test_claude_scanned_files_deduped_within_session() {
    let tmp = write_jsonl(&[
        r#"{"type":"user","message":{"role":"user","content":"edit the file twice"},"timestamp":"2026-03-01T00:00:00Z"}"#,
        r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"Edit","input":{"file_path":"/proj/same.rs","old_string":"a","new_string":"b"}},{"type":"tool_use","id":"t2","name":"Write","input":{"file_path":"/proj/other.rs","content":"c"}}]},"timestamp":"2026-03-01T00:00:01Z"}"#,
        r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","id":"t3","name":"Edit","input":{"file_path":"/proj/same.rs","old_string":"b","new_string":"d"}}]},"timestamp":"2026-03-01T00:00:02Z"}"#,
    ]);
    let result = parse_claude_session(tmp.path()).unwrap().unwrap();
    assert_eq!(
        result.scanned_files,
        vec!["/proj/same.rs".to_owned(), "/proj/other.rs".to_owned()],
        "the repeated path is recorded once, in first-seen order, across both turns"
    );
}

// T-001: isMeta entries are skipped
#[test]
fn test_claude_is_meta_skipped() {
    let tmp = write_jsonl(&[
        r#"{"type":"user","isMeta":true,"message":{"role":"user","content":"<local-command-caveat>Caveat: ...</local-command-caveat>"},"timestamp":"2026-03-01T00:00:00Z"}"#,
        r#"{"type":"user","message":{"role":"user","content":"real message"},"timestamp":"2026-03-01T00:00:00Z"}"#,
    ]);
    let result = parse_claude_session(tmp.path()).unwrap().unwrap();
    assert_eq!(result.messages.len(), 1);
    assert_eq!(result.messages[0].text, "real message");
}

#[test]
fn provenance_requires_boolean_message_metadata_or_the_exact_subagent_layout() {
    use serde_json::json;
    use std::fs;
    let root = tempfile::tempdir().unwrap();
    let parent = "01234567-89ab-cdef-0123-456789abcdef";
    for (relative, record, expected) in [
        (
            "session.jsonl".to_owned(),
            json!({"type":"user","isSidechain":true,"message":{"content":"help"}}),
            true,
        ),
        (
            "session.jsonl".to_owned(),
            json!({"type":"assistant","isSidechain":true,"message":{"content":[]}}),
            true,
        ),
        (
            "session.jsonl".to_owned(),
            json!({"type":"user","isSidechain":"true","message":{"content":"help"}}),
            false,
        ),
        (
            "session.jsonl".to_owned(),
            json!({"type":"summary","isSidechain":true}),
            false,
        ),
        (
            "session.jsonl".to_owned(),
            json!({"type":"user","agentId":"agent-a","message":{"content":"Explain {\"isSidechain\":true}"}}),
            false,
        ),
        (
            format!("{parent}/subagents/agent-a.jsonl"),
            json!({"type":"user","sessionId":parent,"message":{"content":"help"}}),
            true,
        ),
        (
            format!("{parent}/subagents/agent-a.jsonl"),
            json!({"type":"user","sessionId":"other","message":{"content":"help"}}),
            false,
        ),
        (
            "subagents/agent-a.jsonl".to_owned(),
            json!({"type":"user"}),
            false,
        ),
        (
            "project/subagents/agent-a.jsonl".to_owned(),
            json!({"type":"user"}),
            false,
        ),
        ("agent-a.jsonl".to_owned(), json!({"type":"user"}), false),
        (
            format!("{parent}/subagents/session.jsonl"),
            json!({"type":"user"}),
            false,
        ),
        (
            format!("{parent}/subagents/agent-.jsonl"),
            json!({"type":"user"}),
            false,
        ),
        (
            "session.jsonl".to_owned(),
            json!({"role":"user","sessionId":"unrelated","isSidechain":true,"content":"human question"}),
            false,
        ),
    ] {
        let path = root.path().join(&relative);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, record.to_string()).unwrap();
        let parsed = parse_claude_session(&path).unwrap().unwrap();
        assert_eq!(parsed.origin.automated, expected, "{relative}: {record}");
    }
}

#[test]
fn legacy_agent_provenance_matches_the_filename_and_one_parent() {
    use serde_json::json;
    use std::fs;

    let root = tempfile::tempdir().unwrap();
    let user = json!({"type":"user","sessionId":"11111111-1111-4111-8111-111111111111","agentId":"example","isSidechain":true,"message":{"content":"Review this sample module."}});
    let mut assistant = user.clone();
    assistant.as_object_mut().unwrap().remove("type");
    assistant["role"] = json!("assistant");
    assistant["message"]["content"] = json!("The sample looks correct.");
    for (relative, contents, count) in [
        ("agent-example.jsonl", user.to_string(), 1),
        (
            "project/subagents/agent-example.jsonl",
            format!("{user}\n{assistant}\n"),
            2,
        ),
    ] {
        let path = root.path().join(relative);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, contents).unwrap();
        let parsed = parse_claude_session(&path).unwrap().unwrap();
        assert!(parsed.origin.automated, "{relative}");
        assert!(!parsed.origin.identity_conflict);
        assert_eq!(parsed.metadata.session_id, "agent-example");
        assert_eq!(parsed.messages.len(), count);
    }
}

#[test]
fn legacy_agent_provenance_does_not_borrow_mismatched_or_incomplete_evidence() {
    use serde_json::json;
    use std::fs;

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
        // Both orders must reject the whole exception, including role-only logs.
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
    // A matching record cannot grant the exception to a different filename,
    // an empty agent suffix, a non-jsonl file, or a conflicting canonical parent.
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

// Reuse the existing format corpus and its independent expected values for both
// readers, so a fast path cannot silently omit a format or identity check.
fn parse_claude_session(path: &Path) -> Result<Option<ParseResult>> {
    let parsed = super::parse_claude_session(path)?;
    if let Some(full) = &parsed {
        assert_classification_matches(path, Source::Claude, full);
    }
    Ok(parsed)
}
