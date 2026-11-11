use std::ffi::OsString;
use std::io;
use std::time::{Duration, SystemTime};

use super::*;
use crate::commands::classify::reclassify_sessions;
use crate::db::{open_db, seed_session, seed_session_file, setup_test_db};
use crate::embedder::{
    EmbedOptions, MockEmbedder, embed_pending_observed, embed_recent_chunks, f32_as_bytes,
};
use crate::index_observer::{recording, stable_lines};
use crate::search::{SearchOptions, search};
use tempfile::TempDir;

fn collect_strings(conn: &Connection, sql: &str) -> Vec<String> {
    let mut stmt = conn.prepare(sql).unwrap();
    let rows = stmt.query_map([], |r| r.get::<_, String>(0)).unwrap();
    rows.map(Result::unwrap).collect()
}

#[test]
fn reindexing_unchanged_logs_indexes_nothing_and_leaves_stored_rows_unchanged() {
    let (_dir, mut conn) = setup_test_db();
    let tmp = TempDir::new().unwrap();
    let claude_dir = tmp.path().join("claude_projects");
    fs::create_dir_all(&claude_dir).unwrap();
    fs::write(
            claude_dir.join("s.jsonl"),
            concat!(
                r#"{"type":"user","cwd":"/proj","message":{"role":"user","content":"steady question"},"timestamp":"2026-03-01T00:00:00Z"}"#,
                "\n",
                r#"{"type":"assistant","cwd":"/proj","message":{"role":"assistant","content":"steady answer"},"timestamp":"2026-03-01T00:00:01Z"}"#,
            ),
        )
        .unwrap();
    let codex_dir = tmp.path().join("codex_sessions");
    let opts = IndexOptions {
        force: false,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
        codex_archived_dir: None,
    };

    index_from_dirs(&mut conn, &opts, true).unwrap();
    index_chunks(&mut conn).unwrap();
    let counts = |conn: &Connection| -> (i64, i64, i64, i64) {
        let m = conn
            .query_row("SELECT COUNT(*) FROM messages", [], |r| r.get(0))
            .unwrap();
        let c = conn
            .query_row("SELECT COUNT(*) FROM qa_chunks", [], |r| r.get(0))
            .unwrap();
        let s = conn
            .query_row("SELECT COUNT(*) FROM sessions", [], |r| r.get(0))
            .unwrap();
        let v = conn
            .query_row("SELECT COUNT(*) FROM vec_chunks", [], |r| r.get(0))
            .unwrap();
        (m, c, s, v)
    };
    let before = counts(&conn);

    let stats = index_from_dirs(&mut conn, &opts, true).unwrap();
    index_chunks(&mut conn).unwrap();

    assert_eq!(stats.indexed, 0, "an unchanged tree re-parses nothing");
    assert_eq!(
        counts(&conn),
        before,
        "re-index of an unchanged tree must not grow or rewrite the DB"
    );
}

#[test]
fn test_index_from_dirs_reindexes_appended_file() {
    let (_dir, mut conn) = setup_test_db();
    let tmp = TempDir::new().unwrap();
    let claude_dir = tmp.path().join("claude_projects");
    fs::create_dir_all(&claude_dir).unwrap();
    let file = claude_dir.join("s.jsonl");
    fs::write(
            &file,
            concat!(
                r#"{"type":"user","cwd":"/proj","message":{"role":"user","content":"first question"},"timestamp":"2026-03-01T00:00:00Z"}"#,
                "\n",
            ),
        )
        .unwrap();
    let codex_dir = tmp.path().join("codex_sessions");
    let opts = IndexOptions {
        force: false,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
        codex_archived_dir: None,
    };
    index_from_dirs(&mut conn, &opts, true).unwrap();

    let mut all = fs::read_to_string(&file).unwrap();
    all.push_str(
            r#"{"type":"assistant","cwd":"/proj","message":{"role":"assistant","content":"appended answer"},"timestamp":"2026-03-01T00:00:01Z"}"#,
        );
    fs::write(&file, all).unwrap();
    let f = fs::OpenOptions::new().append(true).open(&file).unwrap();
    f.set_modified(SystemTime::now() + Duration::from_secs(10))
        .unwrap();
    drop(f);

    let stats = index_from_dirs(&mut conn, &opts, true).unwrap();

    assert_eq!(stats.indexed, 1, "the appended file must be re-parsed");
    let sessions: i64 = conn
        .query_row("SELECT COUNT(*) FROM sessions", [], |r| r.get(0))
        .unwrap();
    assert_eq!(
        sessions, 1,
        "re-index upserts the session, not duplicates it"
    );
    let appended: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM messages WHERE messages MATCH 'appended answer'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(appended, 1, "the appended message must be searchable");
    let original: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM messages WHERE messages MATCH 'first question'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        original, 1,
        "re-parsing the appended file must keep the original message searchable"
    );
}

const FRESHNESS_QUESTION: &str = concat!(
    r#"{"type":"user","cwd":"/proj","message":{"role":"user","content":"original question"}}"#,
    "\n",
);
const FRESHNESS_ANSWER: &str = concat!(
    r#"{"type":"assistant","message":{"role":"assistant","content":"appended answer"}}"#,
    "\n",
);

fn rewrite_with_mtime(path: &Path, body: &str, mtime: SystemTime) {
    fs::write(path, body).unwrap();
    fs::OpenOptions::new()
        .write(true)
        .open(path)
        .unwrap()
        .set_modified(mtime)
        .unwrap();
}

/// A Claude root and a Codex root under one temp dir. Only the root named `root` exists,
/// and it holds the log `name`.
struct OneLog {
    tmp: TempDir,
    claude: PathBuf,
    codex: PathBuf,
    file: PathBuf,
}

impl OneLog {
    fn new(root: &str, name: &str) -> Self {
        let tmp = TempDir::new().unwrap();
        fs::create_dir(tmp.path().join(root)).unwrap();
        Self {
            claude: tmp.path().join("claude"),
            codex: tmp.path().join("codex"),
            file: tmp.path().join(root).join(name),
            tmp,
        }
    }

    fn opts(&self) -> IndexOptions<'_> {
        IndexOptions {
            force: false,
            claude_dir: &self.claude,
            codex_dir: &self.codex,
            codex_archived_dir: None,
        }
    }

    fn rebuild_opts(&self) -> IndexOptions<'_> {
        IndexOptions {
            force: true,
            ..self.opts()
        }
    }
}

const CLAUDE_METADATA_ONLY: &str = r#"{"type":"system","cwd":"/proj"}"#;
const CODEX_QUESTION: &str = concat!(
    r#"{"type":"response_item","payload":{"role":"user","content":"original question"}}"#,
    "\n",
);
const CODEX_ANSWER: &str = concat!(
    r#"{"type":"response_item","payload":{"role":"assistant","content":"appended answer"}}"#,
    "\n",
);
const CODEX_METADATA_ONLY: &str = r#"{"type":"session_meta","payload":{"cwd":"/proj"}}"#;

/// Runs one more index and returns the observed counts that `--json` reports.
fn counts_of_a_rerun(conn: &mut Connection, opts: &IndexOptions) -> serde_json::Value {
    let observer = Observer::default();
    index_from_dirs_observed(conn, opts, true, &observer).unwrap();
    observer.snapshot()["counts"].clone()
}

/// Indexes `question` in the log `s.jsonl` under the root named `root`, then changes the
/// log's size at an equal mtime and at an mtime 500 microseconds later. An unfinished line
/// must keep the stored messages, and each complete body must be re-indexed.
fn assert_size_changes_are_reindexed(root: &str, question: &str, answer: &str) {
    let (_db_dir, mut conn) = setup_test_db();
    let log = OneLog::new(root, "s.jsonl");
    let opts = log.opts();
    let t0 = UNIX_EPOCH + Duration::from_secs(1_700_000_000);
    rewrite_with_mtime(&log.file, question, t0);
    index_from_dirs(&mut conn, &opts, true).unwrap();
    index_chunks(&mut conn).unwrap();

    rewrite_with_mtime(&log.file, "{", t0);
    assert_eq!(index_from_dirs(&mut conn, &opts, true).unwrap().indexed, 0);
    index_chunks(&mut conn).unwrap();
    assert_eq!(
        collect_strings(&conn, "SELECT text FROM messages"),
        ["original question"],
        "an unfinished line keeps the stored messages"
    );

    rewrite_with_mtime(&log.file, &format!("{question}{answer}"), t0);
    assert_eq!(index_from_dirs(&mut conn, &opts, true).unwrap().indexed, 1);
    index_chunks(&mut conn).unwrap();
    assert_eq!(
        collect_strings(&conn, "SELECT text FROM messages ORDER BY rowid"),
        ["original question", "appended answer"]
    );

    rewrite_with_mtime(&log.file, question, t0 + Duration::from_micros(500));
    assert_eq!(
        fs::metadata(&log.file).unwrap().modified().unwrap(),
        t0 + Duration::from_micros(500),
        "precondition: the file system keeps a submillisecond mtime"
    );
    assert_eq!(index_from_dirs(&mut conn, &opts, true).unwrap().indexed, 1);
    index_chunks(&mut conn).unwrap();
    assert_eq!(
        collect_strings(&conn, "SELECT text FROM messages ORDER BY rowid"),
        ["original question"]
    );

    let counts = counts_of_a_rerun(&mut conn, &opts);
    assert_eq!(counts["files_updated"], 0);
    assert_eq!(counts["files_unchanged"], 1);
}

#[test]
fn size_changes_reindex_with_equal_or_submillisecond_mtime() {
    assert_size_changes_are_reindexed("claude", FRESHNESS_QUESTION, FRESHNESS_ANSWER);
}

#[test]
fn codex_log_whose_size_changes_with_an_equal_or_submillisecond_mtime_is_reindexed() {
    assert_size_changes_are_reindexed("codex", CODEX_QUESTION, CODEX_ANSWER);
}

/// Indexes `question` in the log `s.jsonl` under the root named `root` and reduces the log
/// to `metadata_only` at the same mtime. It then restores `question` 10 seconds later and
/// empties the log 500 microseconds after that. Each step that removes the question must
/// remove its stored message.
fn assert_emptied_log_loses_its_stored_messages(root: &str, question: &str, metadata_only: &str) {
    let (_db_dir, mut conn) = setup_test_db();
    let log = OneLog::new(root, "s.jsonl");
    let opts = log.opts();
    let t0 = UNIX_EPOCH + Duration::from_secs(1_700_000_000);
    let t1 = t0 + Duration::from_secs(10);
    rewrite_with_mtime(&log.file, question, t0);
    index_from_dirs(&mut conn, &opts, true).unwrap();
    index_chunks(&mut conn).unwrap();

    rewrite_with_mtime(&log.file, metadata_only, t0);
    assert_eq!(index_from_dirs(&mut conn, &opts, true).unwrap().indexed, 1);
    index_chunks(&mut conn).unwrap();
    assert!(collect_strings(&conn, "SELECT text FROM messages").is_empty());

    rewrite_with_mtime(&log.file, question, t1);
    assert_eq!(index_from_dirs(&mut conn, &opts, true).unwrap().indexed, 1);
    index_chunks(&mut conn).unwrap();
    assert_eq!(
        collect_strings(&conn, "SELECT text FROM messages"),
        ["original question"],
        "precondition: the log holds a message again"
    );

    rewrite_with_mtime(&log.file, "", t1 + Duration::from_micros(500));
    assert_eq!(
        fs::metadata(&log.file).unwrap().modified().unwrap(),
        t1 + Duration::from_micros(500),
        "precondition: the file system keeps a submillisecond mtime"
    );
    assert_eq!(index_from_dirs(&mut conn, &opts, true).unwrap().indexed, 1);
    index_chunks(&mut conn).unwrap();
    assert!(collect_strings(&conn, "SELECT text FROM messages").is_empty());
    assert!(collect_strings(&conn, "SELECT content FROM qa_chunks").is_empty());

    let counts = counts_of_a_rerun(&mut conn, &opts);
    assert_eq!(counts["files_updated"], 0);
    assert_eq!(counts["files_unchanged"], 1);
}

#[test]
fn claude_log_emptied_or_reduced_to_metadata_with_an_unchanged_mtime_loses_its_stored_messages() {
    assert_emptied_log_loses_its_stored_messages(
        "claude",
        FRESHNESS_QUESTION,
        CLAUDE_METADATA_ONLY,
    );
}

#[test]
fn codex_log_emptied_or_reduced_to_session_meta_with_an_unchanged_mtime_loses_its_stored_messages()
{
    assert_emptied_log_loses_its_stored_messages("codex", CODEX_QUESTION, CODEX_METADATA_ONLY);
}

const SESSION_A_LOG: &str = concat!(
    r#"{"type":"session_meta","payload":{"id":"session-a","source":"exec"}}"#,
    "\n",
    r#"{"type":"response_item","payload":{"role":"user","content":"original question"}}"#,
);
const SESSION_A_SUBAGENT_METADATA: &str =
    r#"{"type":"session_meta","payload":{"id":"session-a","source":{"subagent":"review"}}}"#;
const SESSION_B_SUBAGENT_METADATA: &str =
    r#"{"type":"session_meta","payload":{"id":"session-b","source":{"subagent":"review"}}}"#;

/// The session ids, session types and message texts that search and show read.
fn ids_types_and_messages(conn: &Connection) -> [Vec<String>; 3] {
    [
        collect_strings(conn, "SELECT session_id FROM sessions"),
        collect_strings(conn, "SELECT session_type FROM sessions"),
        collect_strings(conn, "SELECT text FROM messages ORDER BY rowid"),
    ]
}

/// Indexes SESSION_A_LOG written at T0 under a Codex root, labels the session `label`, and
/// rewrites the log with `replacement` at T0 plus 10 seconds, so the next run reads it
/// whatever the two bodies' sizes are.
fn session_a_labelled_then_rewritten(
    conn: &mut Connection,
    label: &str,
    replacement: &str,
) -> OneLog {
    let log = OneLog::new("codex", "rollout-test.jsonl");
    let t0 = UNIX_EPOCH + Duration::from_secs(1_700_000_000);
    rewrite_with_mtime(&log.file, SESSION_A_LOG, t0);
    index_from_dirs(conn, &log.opts(), true).unwrap();
    index_chunks(conn).unwrap();
    conn.execute("UPDATE sessions SET session_type = ?", [label])
        .unwrap();
    rewrite_with_mtime(&log.file, replacement, t0 + Duration::from_secs(10));
    log
}

/// Runs the index and chunk passes, then reclassifies every session. Returns how many logs
/// the index re-parsed, the stored ids, types and messages after it, and the session types
/// after the reclassification.
fn reindex_then_reclassify(
    conn: &mut Connection,
    opts: &IndexOptions,
) -> (usize, [Vec<String>; 3], Vec<String>) {
    let indexed = index_from_dirs(conn, opts, true).unwrap().indexed;
    index_chunks(conn).unwrap();
    let stored = ids_types_and_messages(conn);
    reclassify_sessions(conn, true, false).unwrap();
    let reclassified = collect_strings(conn, "SELECT session_type FROM sessions");
    (indexed, stored, reclassified)
}

#[test]
fn metadata_only_reindex_applies_provenance_only_to_the_matching_session() {
    // The same session's metadata replaces an interactive session's body.
    let (_db_dir, mut conn) = setup_test_db();
    let log =
        session_a_labelled_then_rewritten(&mut conn, "interactive", SESSION_A_SUBAGENT_METADATA);
    let (indexed, stored, reclassified) = reindex_then_reclassify(&mut conn, &log.opts());
    assert_eq!(indexed, 1);
    assert_eq!(stored, [vec!["session-a"], vec!["automated"], vec![]]);
    assert_eq!(
        reclassified,
        ["automated"],
        "explicit reclassification must agree with indexing"
    );
    let (indexed, stored, reclassified) = reindex_then_reclassify(&mut conn, &log.opts());
    assert_eq!(indexed, 0);
    assert_eq!(stored, [vec!["session-a"], vec!["automated"], vec![]]);
    assert_eq!(
        reclassified,
        ["automated"],
        "explicit reclassification must agree with indexing"
    );

    // Another session's metadata replaces an interactive session's body.
    let (_db_dir, mut conn) = setup_test_db();
    let log =
        session_a_labelled_then_rewritten(&mut conn, "interactive", SESSION_B_SUBAGENT_METADATA);
    let (indexed, stored, reclassified) = reindex_then_reclassify(&mut conn, &log.opts());
    assert_eq!(indexed, 0);
    assert_eq!(
        stored,
        [
            vec!["session-a"],
            vec!["interactive"],
            vec!["original question"]
        ],
        "another session's metadata cannot establish that this body is empty"
    );
    assert_eq!(
        reclassified,
        ["interactive"],
        "explicit reclassification must agree with indexing"
    );
    let (indexed, stored, reclassified) = reindex_then_reclassify(&mut conn, &log.opts());
    assert_eq!(indexed, 0);
    assert_eq!(
        stored,
        [
            vec!["session-a"],
            vec!["interactive"],
            vec!["original question"]
        ]
    );
    assert_eq!(
        reclassified,
        ["interactive"],
        "explicit reclassification must agree with indexing"
    );

    // The same session's metadata replaces an automated session's body.
    let (_db_dir, mut conn) = setup_test_db();
    let log =
        session_a_labelled_then_rewritten(&mut conn, "automated", SESSION_A_SUBAGENT_METADATA);
    let (indexed, stored, reclassified) = reindex_then_reclassify(&mut conn, &log.opts());
    assert_eq!(indexed, 1);
    assert_eq!(stored, [vec!["session-a"], vec!["automated"], vec![]]);
    assert_eq!(
        reclassified,
        ["automated"],
        "explicit reclassification must agree with indexing"
    );
    let (indexed, stored, reclassified) = reindex_then_reclassify(&mut conn, &log.opts());
    assert_eq!(indexed, 0);
    assert_eq!(stored, [vec!["session-a"], vec!["automated"], vec![]]);
    assert_eq!(
        reclassified,
        ["automated"],
        "explicit reclassification must agree with indexing"
    );

    // Another session's metadata replaces an automated session's body.
    let (_db_dir, mut conn) = setup_test_db();
    let log =
        session_a_labelled_then_rewritten(&mut conn, "automated", SESSION_B_SUBAGENT_METADATA);
    let (indexed, stored, reclassified) = reindex_then_reclassify(&mut conn, &log.opts());
    assert_eq!(indexed, 0);
    assert_eq!(
        stored,
        [
            vec!["session-a"],
            vec!["automated"],
            vec!["original question"]
        ],
        "another session's metadata cannot establish that this body is empty"
    );
    assert_eq!(
        reclassified,
        ["automated"],
        "a log naming another session cannot reset the stored label"
    );
    let (indexed, stored, reclassified) = reindex_then_reclassify(&mut conn, &log.opts());
    assert_eq!(indexed, 0);
    assert_eq!(
        stored,
        [
            vec!["session-a"],
            vec!["automated"],
            vec!["original question"]
        ]
    );
    assert_eq!(
        reclassified,
        ["automated"],
        "a log naming another session cannot reset the stored label"
    );
}

/// Runs the per-file step of an index run on the Claude log `file`, reading it with `parse`,
/// and returns the open transaction so the caller decides what happens before the commit.
fn index_claude_log_with_parser<'c>(
    conn: &'c mut Connection,
    file: &Path,
    parse: impl FnOnce(&Path, Source) -> Result<Option<ParseResult>>,
) -> Transaction<'c> {
    let existing = load_existing_sessions(conn).unwrap();
    let tx = conn.transaction().unwrap();
    let ctx = IndexContext {
        tx: &tx,
        existing: &existing,
        force: false,
        embedded_sessions: None,
    };
    index_file_with_parser(&ctx, file, &Source::Claude, parse).unwrap();
    tx
}

#[test]
fn append_around_parsing_is_retried_and_unchanged_body_is_not_parsed() {
    let appended = format!("{FRESHNESS_QUESTION}{FRESHNESS_ANSWER}");

    // The append lands before the log is read.
    let (_db_dir, mut conn) = setup_test_db();
    let log = OneLog::new("claude", "s.jsonl");
    fs::write(&log.file, FRESHNESS_QUESTION).unwrap();
    let mtime = fs::metadata(&log.file).unwrap().modified().unwrap();
    index_claude_log_with_parser(&mut conn, &log.file, |path, source| {
        rewrite_with_mtime(path, &appended, mtime);
        parse_session_including_empty(path, source)
    })
    .commit()
    .unwrap();
    index_chunks(&mut conn).unwrap();
    assert_eq!(
        collect_strings(&conn, "SELECT text FROM messages ORDER BY rowid"),
        ["original question", "appended answer"],
        "precondition: the read saw the append"
    );
    assert_eq!(
        index_from_dirs(&mut conn, &log.opts(), true)
            .unwrap()
            .indexed,
        1
    );
    index_chunks(&mut conn).unwrap();
    assert_eq!(
        collect_strings(&conn, "SELECT text FROM messages ORDER BY rowid"),
        ["original question", "appended answer"]
    );
    let counts = counts_of_a_rerun(&mut conn, &log.opts());
    assert_eq!(counts["files_updated"], 0);
    assert_eq!(counts["files_unchanged"], 1);

    // The append lands while the log is read.
    let (_db_dir, mut conn) = setup_test_db();
    let log = OneLog::new("claude", "s.jsonl");
    fs::write(&log.file, FRESHNESS_QUESTION).unwrap();
    let mtime = fs::metadata(&log.file).unwrap().modified().unwrap();
    index_claude_log_with_parser(&mut conn, &log.file, |path, source| {
        let parsed = parse_session_including_empty(path, source);
        rewrite_with_mtime(path, &appended, mtime);
        parsed
    })
    .commit()
    .unwrap();
    index_chunks(&mut conn).unwrap();
    assert_eq!(
        collect_strings(&conn, "SELECT text FROM messages ORDER BY rowid"),
        ["original question"],
        "precondition: the read finished before the append"
    );
    assert_eq!(
        index_from_dirs(&mut conn, &log.opts(), true)
            .unwrap()
            .indexed,
        1
    );
    index_chunks(&mut conn).unwrap();
    assert_eq!(
        collect_strings(&conn, "SELECT text FROM messages ORDER BY rowid"),
        ["original question", "appended answer"]
    );
    let counts = counts_of_a_rerun(&mut conn, &log.opts());
    assert_eq!(counts["files_updated"], 0);
    assert_eq!(counts["files_unchanged"], 1);

    // The append lands after the log is read, before the index commits.
    let (_db_dir, mut conn) = setup_test_db();
    let log = OneLog::new("claude", "s.jsonl");
    fs::write(&log.file, FRESHNESS_QUESTION).unwrap();
    let mtime = fs::metadata(&log.file).unwrap().modified().unwrap();
    let tx = index_claude_log_with_parser(&mut conn, &log.file, parse_session_including_empty);
    rewrite_with_mtime(&log.file, &appended, mtime);
    tx.commit().unwrap();
    index_chunks(&mut conn).unwrap();
    assert_eq!(
        collect_strings(&conn, "SELECT text FROM messages ORDER BY rowid"),
        ["original question"],
        "precondition: the read finished before the append"
    );
    assert_eq!(
        index_from_dirs(&mut conn, &log.opts(), true)
            .unwrap()
            .indexed,
        1
    );
    index_chunks(&mut conn).unwrap();
    assert_eq!(
        collect_strings(&conn, "SELECT text FROM messages ORDER BY rowid"),
        ["original question", "appended answer"]
    );
    let counts = counts_of_a_rerun(&mut conn, &log.opts());
    assert_eq!(counts["files_updated"], 0);
    assert_eq!(counts["files_unchanged"], 1);
}

/// FRESHNESS_QUESTION with another word of the same byte length.
const FRESHNESS_SAME_SIZE_QUESTION: &str = concat!(
    r#"{"type":"user","cwd":"/proj","message":{"role":"user","content":"modified question"}}"#,
    "\n",
);

#[test]
fn same_size_rewrite_while_a_log_is_read_is_indexed_by_the_next_run() {
    assert_eq!(
        FRESHNESS_SAME_SIZE_QUESTION.len(),
        FRESHNESS_QUESTION.len(),
        "precondition: the rewrite keeps the byte length"
    );
    let (_db_dir, mut conn) = setup_test_db();
    let log = OneLog::new("claude", "s.jsonl");
    let t0 = UNIX_EPOCH + Duration::from_secs(1_700_000_000);
    rewrite_with_mtime(&log.file, FRESHNESS_QUESTION, t0);
    index_claude_log_with_parser(&mut conn, &log.file, |path, source| {
        let parsed = parse_session_including_empty(path, source);
        rewrite_with_mtime(
            path,
            FRESHNESS_SAME_SIZE_QUESTION,
            t0 + Duration::from_micros(500),
        );
        parsed
    })
    .commit()
    .unwrap();
    index_chunks(&mut conn).unwrap();
    assert_eq!(
        fs::metadata(&log.file).unwrap().modified().unwrap(),
        t0 + Duration::from_micros(500),
        "precondition: the file system keeps a submillisecond mtime"
    );
    assert_eq!(
        collect_strings(&conn, "SELECT text FROM messages"),
        ["original question"],
        "precondition: the read finished before the rewrite"
    );

    assert_eq!(
        index_from_dirs(&mut conn, &log.opts(), true)
            .unwrap()
            .indexed,
        1
    );
    index_chunks(&mut conn).unwrap();
    assert_eq!(
        collect_strings(&conn, "SELECT text FROM messages"),
        ["modified question"]
    );
    assert_eq!(
        collect_strings(&conn, "SELECT content FROM qa_chunks"),
        ["modified question"]
    );
    let counts = counts_of_a_rerun(&mut conn, &log.opts());
    assert_eq!(counts["files_updated"], 0);
    assert_eq!(counts["files_unchanged"], 1);
}

#[test]
fn legacy_size_upgrade_preserves_embeddings_and_retries_until_model_returns() {
    let db_dir = TempDir::new().unwrap();
    let db_path = db_dir.path().join("index.db");
    let mut conn = open_db(&db_path).unwrap();
    let root = TempDir::new().unwrap();
    let file = root.path().join("s.jsonl");
    fs::write(&file, FRESHNESS_QUESTION).unwrap();
    let mtime = fs::metadata(&file).unwrap().modified().unwrap();
    let absent = root.path().join("absent");
    let opts = IndexOptions {
        force: false,
        claude_dir: root.path(),
        codex_dir: &absent,
        codex_archived_dir: None,
    };
    index_from_dirs(&mut conn, &opts, true).unwrap();
    index_chunks(&mut conn).unwrap();
    assert_eq!(
        embed_recent_chunks(&mut conn, &MockEmbedder::new(), 100, None)
            .unwrap()
            .embedded,
        1
    );
    let snapshot = |conn: &Connection| {
        [
            collect_strings(
                conn,
                "SELECT json_array(session_id, mtime, chunks_indexed, files_scanned) FROM sessions",
            ),
            collect_strings(conn, "SELECT json_array(rowid, text) FROM messages"),
            collect_strings(
                conn,
                "SELECT json_array(id, content, src_rowid_lo, src_rowid_hi) FROM qa_chunks",
            ),
            collect_strings(
                conn,
                "SELECT json_array(rowid, chunk_id, hex(embedding)) FROM vec_chunks",
            ),
        ]
    };
    let before = snapshot(&conn);
    conn.execute_batch("ALTER TABLE sessions DROP COLUMN file_size;")
        .unwrap();
    drop(conn);
    rewrite_with_mtime(
        &file,
        &format!("{FRESHNESS_QUESTION}{FRESHNESS_ANSWER}"),
        mtime,
    );

    let mut conn = open_db(&db_path).unwrap();
    let first = index_from_dirs(&mut conn, &opts, false).unwrap();
    index_chunks(&mut conn).unwrap();
    assert_eq!((first.indexed, first.preserved_embedded), (0, 1));
    assert_eq!(snapshot(&conn), before);
    drop(conn);

    let mut conn = open_db(&db_path).unwrap();
    let second = index_from_dirs(&mut conn, &opts, false).unwrap();
    index_chunks(&mut conn).unwrap();
    assert_eq!((second.indexed, second.preserved_embedded), (0, 1));
    assert_eq!(snapshot(&conn), before);
    drop(conn);

    let mut conn = open_db(&db_path).unwrap();
    assert_eq!(index_from_dirs(&mut conn, &opts, true).unwrap().indexed, 1);
    index_chunks(&mut conn).unwrap();
    assert_eq!(
        collect_strings(&conn, "SELECT text FROM messages ORDER BY rowid"),
        ["original question", "appended answer"]
    );
    drop(conn);

    let mut conn = open_db(&db_path).unwrap();
    assert_eq!(index_from_dirs(&mut conn, &opts, true).unwrap().indexed, 0);
}

fn index_one_claude_session(content: &str) -> Option<String> {
    let (_dir, mut conn) = setup_test_db();
    let tmp = TempDir::new().unwrap();
    let claude_dir = tmp.path().join("claude_projects");
    fs::create_dir_all(&claude_dir).unwrap();
    let line = format!(
        r#"{{"type":"user","cwd":"/proj","message":{{"role":"user","content":{content:?}}},"timestamp":"2026-03-01T00:00:00Z"}}"#
    );
    fs::write(claude_dir.join("s.jsonl"), line).unwrap();
    let codex_dir = tmp.path().join("codex_sessions");
    index_from_dirs(
        &mut conn,
        &IndexOptions {
            force: false,
            claude_dir: &claude_dir,
            codex_dir: &codex_dir,
            codex_archived_dir: None,
        },
        true,
    )
    .unwrap();
    conn.query_row("SELECT session_type FROM sessions", [], |r| r.get(0))
        .unwrap()
}

#[test]
fn slash_command_first_turn_is_stored_as_automated_and_human_first_turn_as_interactive() {
    assert_eq!(
        index_one_claude_session("<command-message>clear</command-message>").as_deref(),
        Some("automated"),
        "a slash-command first turn must be tagged automated"
    );
    assert_eq!(
        index_one_claude_session("how do I implement authentication").as_deref(),
        Some("interactive"),
        "a human first turn must be tagged interactive"
    );
}

#[test]
fn test_incremental_scan_picks_up_new_file_in_existing_codex_day_dir() {
    let (_dir, mut conn) = setup_test_db();
    let tmp = TempDir::new().unwrap();
    let claude_dir = tmp.path().join("claude_projects");
    fs::create_dir_all(&claude_dir).unwrap();
    let codex_dir = tmp.path().join("codex_sessions");
    let day_dir = codex_dir.join("2026/04/27");
    fs::create_dir_all(&day_dir).unwrap();

    let s1 = r#"{"timestamp":"2026-04-27T00:00:00Z","type":"session_meta","payload":{"id":"codex-s1","cwd":"/proj"}}
{"timestamp":"2026-04-27T00:00:01Z","type":"response_item","payload":{"role":"user","content":[{"type":"input_text","text":"first codex session"}]}}"#;
    fs::write(day_dir.join("s1.jsonl"), s1).unwrap();

    let stats1 = index_from_dirs(
        &mut conn,
        &IndexOptions {
            force: false,
            claude_dir: &claude_dir,
            codex_dir: &codex_dir,
            codex_archived_dir: None,
        },
        true,
    )
    .unwrap();
    assert_eq!(stats1.indexed, 1);
    index_chunks(&mut conn).unwrap();

    let s2 = r#"{"timestamp":"2026-04-27T00:00:00Z","type":"session_meta","payload":{"id":"codex-s2","cwd":"/proj"}}
{"timestamp":"2026-04-27T00:00:01Z","type":"response_item","payload":{"role":"user","content":[{"type":"input_text","text":"second codex session"}]}}"#;
    fs::write(day_dir.join("s2.jsonl"), s2).unwrap();

    let stats2 = index_from_dirs(
        &mut conn,
        &IndexOptions {
            force: false,
            claude_dir: &claude_dir,
            codex_dir: &codex_dir,
            codex_archived_dir: None,
        },
        true,
    )
    .unwrap();
    assert_eq!(
        stats2.indexed, 1,
        "a new session in an existing deep dir must be indexed (parent mtime unchanged)"
    );
    assert_eq!(stats2.total_sessions, 2);
}

fn seed_claude_and_codex(conn: &mut Connection) -> (TempDir, PathBuf, PathBuf) {
    let tmp = TempDir::new().unwrap();
    let claude_dir = tmp.path().join("claude_projects");
    fs::create_dir_all(&claude_dir).unwrap();
    fs::write(
            claude_dir.join("c.jsonl"),
            r#"{"type":"user","cwd":"/proj","message":{"role":"user","content":"claude turn"},"timestamp":"2026-03-01T00:00:00Z"}"#,
        )
        .unwrap();
    let codex_dir = tmp.path().join("codex_sessions");
    let day_dir = codex_dir.join("2026/04/27");
    fs::create_dir_all(&day_dir).unwrap();
    fs::write(
            day_dir.join("s1.jsonl"),
            "{\"timestamp\":\"2026-04-27T00:00:00Z\",\"type\":\"session_meta\",\"payload\":{\"id\":\"codex-s1\",\"cwd\":\"/proj\"}}\n{\"timestamp\":\"2026-04-27T00:00:01Z\",\"type\":\"response_item\",\"payload\":{\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"codex turn\"}]}}",
        )
        .unwrap();
    let stats = index_from_dirs(
        conn,
        &IndexOptions {
            force: false,
            claude_dir: &claude_dir,
            codex_dir: &codex_dir,
            codex_archived_dir: None,
        },
        true,
    )
    .unwrap();
    assert_eq!(stats.total_sessions, 2, "seed should index both sessions");
    (tmp, claude_dir, codex_dir)
}

/// The (source, reason, preserved_sessions) of each root an index run reports as skipped.
fn skipped_roots(stats: &IndexStats) -> Vec<(Source, SkippedReason, usize)> {
    stats
        .skipped_roots
        .iter()
        .map(|s| (s.source, s.reason, s.preserved_sessions))
        .collect()
}

#[test]
fn deleted_claude_log_is_removed_while_sessions_under_a_missing_codex_root_are_kept_and_reported() {
    let (_dir, mut conn) = setup_test_db();
    let (tmp, claude_dir, _codex_dir) = seed_claude_and_codex(&mut conn);
    fs::remove_file(claude_dir.join("c.jsonl")).unwrap();
    let missing_codex = tmp.path().join("missing_codex");

    let stats = index_from_dirs(
        &mut conn,
        &IndexOptions {
            force: false,
            claude_dir: &claude_dir,
            codex_dir: &missing_codex,
            codex_archived_dir: None,
        },
        true,
    )
    .unwrap();

    assert_eq!(stats.total_sessions, 1);
    assert_eq!(
        collect_strings(&conn, "SELECT source FROM sessions"),
        ["codex"]
    );
    assert_eq!(
        skipped_roots(&stats),
        [(Source::Codex, SkippedReason::MissingRoot, 1)]
    );
}

#[test]
fn deleting_a_log_under_a_fully_read_root_removes_its_session_messages_chunks_embeddings_and_touched_files()
 {
    let (_dir, mut conn) = setup_test_db();
    let tmp = TempDir::new().unwrap();
    let claude_dir = tmp.path().join("claude_projects");
    fs::create_dir_all(&claude_dir).unwrap();
    let claude_log = claude_dir.join("c.jsonl");
    fs::write(
        &claude_log,
        concat!(
            r#"{"type":"user","cwd":"/proj","message":{"role":"user","content":"claude turn"},"timestamp":"2026-03-01T00:00:00Z"}"#,
            "\n",
            r#"{"type":"assistant","cwd":"/proj","message":{"role":"assistant","content":[{"type":"text","text":"editing"},{"type":"tool_use","id":"t1","name":"Edit","input":{"file_path":"/proj/c.rs","old_string":"x","new_string":"y"}}]},"timestamp":"2026-03-01T00:00:01Z"}"#,
        ),
    )
    .unwrap();
    let codex_dir = tmp.path().join("codex_sessions");
    let day_dir = codex_dir.join("2026/04/27");
    fs::create_dir_all(&day_dir).unwrap();
    fs::write(
        day_dir.join("s1.jsonl"),
        concat!(
            r#"{"timestamp":"2026-04-27T00:00:00Z","type":"session_meta","payload":{"id":"codex-s1","cwd":"/proj"}}"#,
            "\n",
            r#"{"timestamp":"2026-04-27T00:00:01Z","type":"response_item","payload":{"role":"user","content":[{"type":"input_text","text":"codex turn"}]}}"#,
        ),
    )
    .unwrap();
    let opts = IndexOptions {
        force: false,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
        codex_archived_dir: None,
    };
    index_from_dirs(&mut conn, &opts, true).unwrap();
    index_chunks(&mut conn).unwrap();
    assert_eq!(
        embed_recent_chunks(&mut conn, &MockEmbedder::new(), 100, None)
            .unwrap()
            .embedded,
        2
    );
    assert_eq!(
        collect_strings(&conn, "SELECT path FROM session_files"),
        ["/proj/c.rs"]
    );

    fs::remove_file(&claude_log).unwrap();
    let stats = index_from_dirs(&mut conn, &opts, true).unwrap();

    assert_eq!(stats.total_sessions, 1);
    assert!(stats.skipped_roots.is_empty());
    assert_eq!(
        collect_strings(&conn, "SELECT session_id FROM sessions"),
        ["s1"]
    );
    assert_eq!(
        collect_strings(&conn, "SELECT DISTINCT session_id FROM messages"),
        ["s1"]
    );
    assert_eq!(
        collect_strings(&conn, "SELECT DISTINCT session_id FROM qa_chunks"),
        ["s1"]
    );
    let vectors: i64 = conn
        .query_row("SELECT COUNT(*) FROM vec_chunks", [], |r| r.get(0))
        .unwrap();
    assert_eq!(vectors, 1, "the deleted session's embedding is removed");
    assert!(collect_strings(&conn, "SELECT path FROM session_files").is_empty());
}

#[cfg(unix)]
#[test]
fn unreadable_codex_subdirectory_keeps_every_session_and_reports_only_unseen_ones_as_preserved() {
    use std::os::unix::fs::PermissionsExt;

    let (_dir, mut conn) = setup_test_db();
    let (_tmp, claude_dir, codex_dir) = seed_claude_and_codex(&mut conn);
    fs::write(
        codex_dir.join("top.jsonl"),
        concat!(
            r#"{"timestamp":"2026-05-01T00:00:00Z","type":"session_meta","payload":{"id":"codex-top","cwd":"/proj"}}"#,
            "\n",
            r#"{"timestamp":"2026-05-01T00:00:01Z","type":"response_item","payload":{"role":"user","content":[{"type":"input_text","text":"codex top turn"}]}}"#,
        ),
    )
    .unwrap();
    let opts = IndexOptions {
        force: false,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
        codex_archived_dir: None,
    };
    index_from_dirs(&mut conn, &opts, true).unwrap();

    let locked_subdir = codex_dir.join("2026");
    let permissions = fs::metadata(&locked_subdir).unwrap().permissions();
    fs::set_permissions(&locked_subdir, fs::Permissions::from_mode(0o000)).unwrap();
    let stats = index_from_dirs(&mut conn, &opts, true);
    fs::set_permissions(&locked_subdir, permissions).unwrap();
    let stats = stats.unwrap();

    assert_eq!(stats.total_sessions, 3, "the unseen s1 session is kept");
    assert_eq!(
        skipped_roots(&stats),
        [(Source::Codex, SkippedReason::IncompleteEnumeration, 1)],
        "only the unseen s1 session counts as preserved; the re-read top session does not"
    );
}

#[cfg(unix)]
#[test]
fn codex_root_that_cannot_be_fully_read_is_reported_even_with_no_stored_sessions() {
    use std::os::unix::fs::PermissionsExt;

    let (_dir, mut conn) = setup_test_db();
    let tmp = TempDir::new().unwrap();
    let claude_dir = tmp.path().join("claude_projects");
    fs::create_dir_all(&claude_dir).unwrap();
    let codex_dir = tmp.path().join("codex_sessions");
    let locked_subdir = codex_dir.join("2026");
    fs::create_dir_all(&locked_subdir).unwrap();
    fs::write(
            locked_subdir.join("s1.jsonl"),
            "{\"timestamp\":\"2026-04-27T00:00:00Z\",\"type\":\"session_meta\",\"payload\":{\"id\":\"codex-s1\",\"cwd\":\"/proj\"}}",
        )
        .unwrap();
    let orig = fs::metadata(&locked_subdir).unwrap().permissions();
    let mut zero = orig.clone();
    zero.set_mode(0o000);
    fs::set_permissions(&locked_subdir, zero).unwrap();

    let stats = index_from_dirs(
        &mut conn,
        &IndexOptions {
            force: false,
            claude_dir: &claude_dir,
            codex_dir: &codex_dir,
            codex_archived_dir: None,
        },
        true,
    );

    fs::set_permissions(&locked_subdir, orig).unwrap();
    let stats = stats.unwrap();

    assert_eq!(
        skipped_roots(&stats),
        [(Source::Codex, SkippedReason::IncompleteEnumeration, 0)],
        "an incompletely enumerated root must surface even with nothing preserved"
    );
}

#[test]
fn missing_roots_without_stored_sessions_are_not_reported() {
    let (_dir, mut conn) = setup_test_db();
    let tmp = TempDir::new().unwrap();
    let gone_claude = tmp.path().join("missing_claude");
    let gone_codex = tmp.path().join("missing_codex");

    let stats = index_from_dirs(
        &mut conn,
        &IndexOptions {
            force: false,
            claude_dir: &gone_claude,
            codex_dir: &gone_codex,
            codex_archived_dir: None,
        },
        true,
    )
    .unwrap();

    assert!(
        stats.skipped_roots.is_empty(),
        "missing roots with no preserved sessions must not be reported as noise"
    );
}

#[test]
fn missing_root_note_says_nothing_was_deleted_and_to_rerun_once_the_root_is_back() {
    let skipped = SkippedRoot {
        source: Source::Codex,
        preserved_sessions: 3,
        reason: SkippedReason::MissingRoot,
        other_roots_reconciled: false,
    };

    assert_eq!(
        skipped.note(),
        "codex root unavailable — preserved 3 existing session(s), none deleted this run; re-run `recall index` once the root is back so a full scan can reconcile any genuinely deleted sessions"
    );
}

#[test]
fn skipped_root_reasons_are_reported_as_missing_root_and_incomplete_enumeration() {
    assert_eq!(SkippedReason::MissingRoot.as_str(), "missing_root");
    assert_eq!(
        SkippedReason::IncompleteEnumeration.as_str(),
        "incomplete_enumeration"
    );
}

#[test]
fn incomplete_root_note_without_preserved_sessions_warns_that_sessions_may_be_missing() {
    let skipped = SkippedRoot {
        source: Source::Claude,
        preserved_sessions: 0,
        reason: SkippedReason::IncompleteEnumeration,
        other_roots_reconciled: false,
    };

    assert_eq!(
        skipped.note(),
        "claude root could not be fully read; some sessions may be missing from the index. check that the root is an absolute path and check the directory's permissions and access, then re-run `recall index`"
    );
}

#[test]
fn incomplete_root_note_with_preserved_sessions_says_nothing_was_deleted_and_to_fix_access() {
    let skipped = SkippedRoot {
        source: Source::Claude,
        preserved_sessions: 2,
        reason: SkippedReason::IncompleteEnumeration,
        other_roots_reconciled: false,
    };

    assert_eq!(
        skipped.note(),
        "claude root could not be fully read — preserved 2 existing session(s), none deleted this run; check that the root is an absolute path and check the directory's permissions and access, then re-run `recall index` so the next full scan can reconcile any genuinely deleted sessions"
    );
}

/// Runs seed_claude_and_codex, chunks and embeds the Claude session `c` and the Codex
/// session `s1`, and records one touched file for each.
fn embedded_claude_and_codex_sessions(conn: &mut Connection) -> (TempDir, PathBuf, PathBuf) {
    let seeded = seed_claude_and_codex(conn);
    index_chunks(conn).unwrap();
    assert_eq!(
        embed_recent_chunks(conn, &MockEmbedder::new(), 100, None)
            .unwrap()
            .embedded,
        2
    );
    seed_session_file(conn, "c", "/proj/c.rs");
    seed_session_file(conn, "s1", "/proj/s1.rs");
    seeded
}

/// Every stored row of the Codex session `s1`: session, messages, chunks, vectors and
/// touched files.
fn codex_session_rows(conn: &Connection) -> [Vec<String>; 5] {
    [
        "SELECT json_array(session_id, source, file_path, project, slug, timestamp, mtime, file_size, session_type, files_scanned, chunks_indexed) FROM sessions WHERE session_id = 's1'",
        "SELECT json_array(rowid, role, text) FROM messages WHERE session_id = 's1' ORDER BY rowid",
        "SELECT json_array(id, content, timestamp, src_rowid_lo, src_rowid_hi, generation) FROM qa_chunks WHERE session_id = 's1' ORDER BY id",
        "SELECT json_array(rowid, chunk_id, sub_idx, hex(embedding)) FROM vec_chunks WHERE chunk_id IN (SELECT id FROM qa_chunks WHERE session_id = 's1') ORDER BY rowid",
        "SELECT path FROM session_files WHERE session_id = 's1' ORDER BY path",
    ]
    .map(|sql| collect_strings(conn, sql))
}

fn assert_only_the_unchanged_codex_session_remains(conn: &Connection, before: &[Vec<String>; 5]) {
    assert_eq!(codex_session_rows(conn), *before);
    assert_eq!(
        collect_strings(conn, "SELECT DISTINCT session_id FROM messages"),
        ["s1"]
    );
    assert_eq!(
        collect_strings(conn, "SELECT DISTINCT session_id FROM qa_chunks"),
        ["s1"]
    );
    assert_eq!(
        collect_strings(conn, "SELECT DISTINCT session_id FROM session_files"),
        ["s1"]
    );
    let vectors: i64 = conn
        .query_row("SELECT COUNT(*) FROM vec_chunks", [], |r| r.get(0))
        .unwrap();
    assert_eq!(vectors, 1);
}

#[test]
fn deleted_claude_log_replaced_by_an_empty_log_is_removed_without_touching_the_codex_session() {
    let (_dir, mut conn) = setup_test_db();
    let (_tmp, claude_dir, codex_dir) = embedded_claude_and_codex_sessions(&mut conn);
    let before = codex_session_rows(&conn);
    fs::remove_file(claude_dir.join("c.jsonl")).unwrap();
    fs::write(claude_dir.join("fresh.jsonl"), "").unwrap();
    let opts = IndexOptions {
        force: false,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
        codex_archived_dir: None,
    };

    let first = index_from_dirs(&mut conn, &opts, false).unwrap();
    assert_eq!((first.indexed, first.total_sessions), (0, 1));
    assert_only_the_unchanged_codex_session_remains(&conn, &before);
    assert!(first.skipped_roots.is_empty());
    assert!(first.parse_diagnostics.is_empty());

    let second = index_from_dirs(&mut conn, &opts, false).unwrap();
    assert_eq!((second.indexed, second.total_sessions), (0, 1));
    assert_only_the_unchanged_codex_session_remains(&conn, &before);
    assert!(second.skipped_roots.is_empty());
    assert!(second.parse_diagnostics.is_empty());
}

#[test]
fn deleted_claude_log_replaced_by_an_invalid_log_is_removed_without_touching_the_codex_session() {
    let (_dir, mut conn) = setup_test_db();
    let (_tmp, claude_dir, codex_dir) = embedded_claude_and_codex_sessions(&mut conn);
    let before = codex_session_rows(&conn);
    fs::remove_file(claude_dir.join("c.jsonl")).unwrap();
    let fresh = claude_dir.join("fresh.jsonl");
    fs::write(&fresh, "invalid JSON\n").unwrap();
    let opts = IndexOptions {
        force: false,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
        codex_archived_dir: None,
    };

    let first = index_from_dirs(&mut conn, &opts, false).unwrap();
    assert_eq!((first.indexed, first.total_sessions), (0, 1));
    assert_only_the_unchanged_codex_session_remains(&conn, &before);
    assert!(first.skipped_roots.is_empty());
    assert_eq!(invalid_json_lines_by_file(&first), [(fresh.clone(), 1)]);

    let second = index_from_dirs(&mut conn, &opts, false).unwrap();
    assert_eq!((second.indexed, second.total_sessions), (0, 1));
    assert_only_the_unchanged_codex_session_remains(&conn, &before);
    assert!(second.skipped_roots.is_empty());
    assert_eq!(invalid_json_lines_by_file(&second), [(fresh, 1)]);
}

#[cfg(unix)]
#[test]
fn deleted_claude_log_replaced_by_an_unreadable_log_is_removed_and_the_read_failure_is_reported() {
    use std::os::unix::fs::PermissionsExt;

    let (_dir, mut conn) = setup_test_db();
    let (_tmp, claude_dir, codex_dir) = embedded_claude_and_codex_sessions(&mut conn);
    let before = codex_session_rows(&conn);
    fs::remove_file(claude_dir.join("c.jsonl")).unwrap();
    let fresh = claude_dir.join("fresh.jsonl");
    fs::write(&fresh, "").unwrap();
    fs::set_permissions(&fresh, fs::Permissions::from_mode(0o000)).unwrap();
    let opts = IndexOptions {
        force: false,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
        codex_archived_dir: None,
    };

    let first = index_from_dirs(&mut conn, &opts, false).unwrap();
    assert_eq!((first.indexed, first.total_sessions), (0, 1));
    assert_only_the_unchanged_codex_session_remains(&conn, &before);
    assert!(first.skipped_roots.is_empty());
    assert_eq!(read_errors_by_file(&first), [(fresh.as_path(), true)]);

    let second = index_from_dirs(&mut conn, &opts, false).unwrap();
    assert_eq!((second.indexed, second.total_sessions), (0, 1));
    assert_only_the_unchanged_codex_session_remains(&conn, &before);
    assert!(second.skipped_roots.is_empty());
    assert_eq!(read_errors_by_file(&second), [(fresh.as_path(), true)]);
}

#[test]
fn codex_session_nested_past_the_depth_limit_is_kept_and_reported_while_a_deleted_claude_log_is_removed()
 {
    let (_dir, mut conn) = setup_test_db();
    let (_tmp, claude_dir, codex_dir) = embedded_claude_and_codex_sessions(&mut conn);
    let before = codex_session_rows(&conn);
    let too_deep =
        codex_dir.join("nested/nested/nested/nested/nested/nested/nested/nested/nested/nested");
    fs::create_dir_all(&too_deep).unwrap();
    fs::rename(
        codex_dir.join("2026/04/27/s1.jsonl"),
        too_deep.join("s1.jsonl"),
    )
    .unwrap();
    fs::write(codex_dir.join("empty.jsonl"), "").unwrap();
    fs::remove_file(claude_dir.join("c.jsonl")).unwrap();
    fs::write(claude_dir.join("fresh.jsonl"), "").unwrap();
    let opts = IndexOptions {
        force: false,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
        codex_archived_dir: None,
    };

    let first = index_from_dirs(&mut conn, &opts, false).unwrap();
    assert_eq!((first.indexed, first.total_sessions), (0, 1));
    assert_only_the_unchanged_codex_session_remains(&conn, &before);
    assert_eq!(
        skipped_roots(&first),
        [(Source::Codex, SkippedReason::IncompleteEnumeration, 1)]
    );
    assert!(first.parse_diagnostics.is_empty());

    let second = index_from_dirs(&mut conn, &opts, false).unwrap();
    assert_eq!((second.indexed, second.total_sessions), (0, 1));
    assert_only_the_unchanged_codex_session_remains(&conn, &before);
    assert_eq!(
        skipped_roots(&second),
        [(Source::Codex, SkippedReason::IncompleteEnumeration, 1)]
    );
    assert!(second.parse_diagnostics.is_empty());
}

#[test]
fn complete_scan_preserves_sessions_with_unknown_source() {
    let (_dir, mut conn) = setup_test_db();
    seed_session(&conn, "unknown");
    conn.execute("UPDATE sessions SET source = 'unknown'", [])
        .unwrap();
    let root = TempDir::new().unwrap();
    let stats = index_from_dirs(
        &mut conn,
        &IndexOptions {
            force: false,
            claude_dir: root.path(),
            codex_dir: root.path(),
            codex_archived_dir: None,
        },
        true,
    )
    .unwrap();
    assert_eq!(stats.total_sessions, 1);
    assert_eq!(
        collect_strings(&conn, "SELECT session_id FROM sessions"),
        ["unknown"]
    );
}

#[test]
fn rebuild_with_a_missing_codex_root_keeps_its_session_and_embeddings_and_reports_it() {
    let (_dir, mut conn) = setup_test_db();
    let (tmp, claude_dir, _codex_dir) = seed_claude_and_codex(&mut conn);

    index_chunks(&mut conn).unwrap();
    let embedder = MockEmbedder::new();
    embed_recent_chunks(&mut conn, &embedder, 100, None).unwrap();
    let codex_vecs_before: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM vec_chunks WHERE chunk_id IN \
             (SELECT id FROM qa_chunks WHERE session_id = 's1')",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert!(
        codex_vecs_before > 0,
        "the codex session must have embeddings before the rebuild"
    );

    let gone_codex = tmp.path().join("missing_codex");
    let stats = index_from_dirs(
        &mut conn,
        &IndexOptions {
            force: true,
            claude_dir: &claude_dir,
            codex_dir: &gone_codex,
            codex_archived_dir: None,
        },
        true,
    )
    .unwrap();

    assert_eq!(
        stats.total_sessions, 2,
        "force rebuild with a missing root must preserve that root's session"
    );
    let codex_vecs_after: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM vec_chunks WHERE chunk_id IN \
             (SELECT id FROM qa_chunks WHERE session_id = 's1')",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        codex_vecs_after, codex_vecs_before,
        "the missing root's embeddings must survive the rebuild unchanged"
    );
    let claude_chunks_after: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM qa_chunks WHERE session_id = 'c'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        claude_chunks_after, 0,
        "the present root is rebuilt: its chunks are cleared for re-embed"
    );
    assert_eq!(
        skipped_roots(&stats),
        [(Source::Codex, SkippedReason::MissingRoot, 1)],
        "the missing codex root must be reported as preserved"
    );
}

#[test]
fn rebuild_with_every_root_missing_keeps_all_sessions_and_embeddings() {
    let (_dir, mut conn) = setup_test_db();
    let (tmp, _claude_dir, _codex_dir) = seed_claude_and_codex(&mut conn);

    index_chunks(&mut conn).unwrap();
    let embedder = MockEmbedder::new();
    embed_recent_chunks(&mut conn, &embedder, 100, None).unwrap();
    let vecs_before: i64 = conn
        .query_row("SELECT COUNT(*) FROM vec_chunks", [], |r| r.get(0))
        .unwrap();
    assert!(vecs_before > 0, "both sessions must have embeddings first");

    let gone_claude = tmp.path().join("missing_claude");
    let gone_codex = tmp.path().join("missing_codex");
    let stats = index_from_dirs(
        &mut conn,
        &IndexOptions {
            force: true,
            claude_dir: &gone_claude,
            codex_dir: &gone_codex,
            codex_archived_dir: None,
        },
        true,
    )
    .unwrap();

    assert_eq!(
        stats.total_sessions, 2,
        "both sessions must survive when neither root is scannable"
    );
    let vecs_after: i64 = conn
        .query_row("SELECT COUNT(*) FROM vec_chunks", [], |r| r.get(0))
        .unwrap();
    assert_eq!(
        vecs_after, vecs_before,
        "every embedding must survive a force rebuild with both roots missing"
    );
    assert_eq!(
        skipped_roots(&stats),
        [
            (Source::Claude, SkippedReason::MissingRoot, 1),
            (Source::Codex, SkippedReason::MissingRoot, 1)
        ],
        "both missing roots must be reported as preserved"
    );
}

#[test]
fn rebuild_removes_the_session_of_a_log_deleted_under_a_fully_read_root() {
    let (_dir, mut conn) = setup_test_db();
    let (_tmp, claude_dir, codex_dir) = seed_claude_and_codex(&mut conn);

    let claude_sid: String = conn
        .query_row(
            "SELECT session_id FROM sessions WHERE source = 'claude'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    let msgs_before: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM messages WHERE session_id = ?",
            [&claude_sid],
            |r| r.get(0),
        )
        .unwrap();
    assert!(
        msgs_before > 0,
        "seed should leave messages for the session"
    );

    fs::remove_file(claude_dir.join("c.jsonl")).unwrap();
    let stats = index_from_dirs(
        &mut conn,
        &IndexOptions {
            force: true,
            claude_dir: &claude_dir,
            codex_dir: &codex_dir,
            codex_archived_dir: None,
        },
        true,
    )
    .unwrap();

    assert_eq!(
        stats.total_sessions, 1,
        "force must clean the deleted file's session when its root was scanned"
    );
    let surviving_source: String = conn
        .query_row("SELECT source FROM sessions", [], |r| r.get(0))
        .unwrap();
    assert_eq!(surviving_source, "codex", "only the codex session survives");

    let msgs_after: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM messages WHERE session_id = ?",
            [&claude_sid],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        msgs_after, 0,
        "force cleanup must cascade to the session's messages"
    );
    let chunks_after: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM qa_chunks WHERE session_id = ?",
            [&claude_sid],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        chunks_after, 0,
        "force cleanup must cascade to the session's qa_chunks"
    );
    assert!(
        stats.skipped_roots.is_empty(),
        "both roots were scanned, so nothing should be reported as preserved"
    );
}

struct CollidingLogs {
    _tmp: TempDir,
    claude_dir: PathBuf,
    codex_dir: PathBuf,
    original: PathBuf,
    replacement: PathBuf,
}

impl CollidingLogs {
    fn opts(&self) -> IndexOptions<'_> {
        IndexOptions {
            force: false,
            claude_dir: &self.claude_dir,
            codex_dir: &self.codex_dir,
            codex_archived_dir: None,
        }
    }
}

/// Indexes and embeds `a/collision.jsonl` (one question plus one invalid line), records a
/// touched file for it, then writes `b/collision.jsonl`, which carries the same session id.
fn embedded_session_and_a_log_reusing_its_id(conn: &mut Connection) -> CollidingLogs {
    let tmp = TempDir::new().unwrap();
    let claude_dir = tmp.path().join("claude");
    fs::create_dir_all(claude_dir.join("a")).unwrap();
    fs::create_dir_all(claude_dir.join("b")).unwrap();
    let logs = CollidingLogs {
        claude_dir: claude_dir.clone(),
        codex_dir: tmp.path().join("codex"),
        original: claude_dir.join("a/collision.jsonl"),
        replacement: claude_dir.join("b/collision.jsonl"),
        _tmp: tmp,
    };
    fs::write(
        &logs.original,
        concat!(
            r#"{"type":"user","message":{"role":"user","content":"original question"},"timestamp":"2026-03-01T00:00:00Z"}"#,
            "\ninvalid JSON\n",
        ),
    )
    .unwrap();
    assert_eq!(
        index_from_dirs(conn, &logs.opts(), true).unwrap().indexed,
        1
    );
    assert_eq!(index_chunks(conn).unwrap().chunks_created, 1);
    assert_eq!(
        embed_recent_chunks(conn, &MockEmbedder::new(), 100, None)
            .unwrap()
            .embedded,
        1
    );
    seed_session_file(conn, "collision", "/proj/original.rs");
    fs::write(
        &logs.replacement,
        r#"{"type":"user","message":{"role":"user","content":"replacement question"},"timestamp":"2026-03-02T00:00:00Z"}"#,
    )
    .unwrap();
    logs
}

fn collision_snapshot(conn: &Connection) -> [Vec<String>; 5] {
    [
        "SELECT json_array(session_id, file_path, mtime, file_size, chunks_indexed, files_scanned) FROM sessions ORDER BY session_id",
        "SELECT json_array(rowid, session_id, role, text) FROM messages ORDER BY rowid",
        "SELECT json_array(id, session_id, content, timestamp, src_rowid_lo, src_rowid_hi) FROM qa_chunks ORDER BY id",
        "SELECT json_array(rowid, chunk_id, sub_idx, hex(embedding)) FROM vec_chunks ORDER BY rowid",
        "SELECT json_array(session_id, path) FROM session_files ORDER BY session_id, path",
    ]
    .map(|sql| collect_strings(conn, sql))
}

fn invalid_json_lines_by_file(stats: &IndexStats) -> Vec<(PathBuf, i64)> {
    stats
        .parse_diagnostics
        .iter()
        .map(|d| (d.file_path.clone(), d.lines.invalid_json_lines))
        .collect()
}

fn read_errors_by_file(stats: &IndexStats) -> Vec<(&Path, bool)> {
    stats
        .parse_diagnostics
        .iter()
        .map(|d| (d.file_path.as_path(), d.read_error))
        .collect()
}

fn assert_the_session_holds_only_the_replacement(conn: &mut Connection, replacement: &Path) {
    assert_eq!(
        collect_strings(conn, "SELECT file_path FROM sessions"),
        [replacement.to_str().unwrap()]
    );
    assert_eq!(
        collect_strings(conn, "SELECT text FROM messages ORDER BY rowid"),
        ["replacement question"]
    );
    let vectors: i64 = conn
        .query_row("SELECT COUNT(*) FROM vec_chunks", [], |r| r.get(0))
        .unwrap();
    assert_eq!(vectors, 0, "the old embedding is removed");
    assert!(
        collect_strings(conn, "SELECT path FROM session_files").is_empty(),
        "the old touched file is removed"
    );
    assert_eq!(
        index_chunks(conn).unwrap().chunks_created,
        0,
        "indexing the replacement already reconciled its chunks"
    );
    assert_eq!(
        collect_strings(conn, "SELECT content FROM qa_chunks"),
        ["replacement question"]
    );
}

#[test]
fn log_reusing_an_embedded_session_id_waits_for_the_model_and_then_replaces_the_session() {
    let (_dir, mut conn) = setup_test_db();
    let logs = embedded_session_and_a_log_reusing_its_id(&mut conn);
    let before = collision_snapshot(&conn);

    let first = index_from_dirs(&mut conn, &logs.opts(), false).unwrap();
    assert_eq!(
        (
            first.indexed,
            first.preserved_embedded,
            first.total_sessions
        ),
        (0, 1, 1)
    );
    assert_eq!(
        invalid_json_lines_by_file(&first),
        [(logs.original.clone(), 1)]
    );
    assert_eq!(collision_snapshot(&conn), before);
    let second = index_from_dirs(&mut conn, &logs.opts(), false).unwrap();
    assert_eq!(
        (
            second.indexed,
            second.preserved_embedded,
            second.total_sessions
        ),
        (0, 1, 1)
    );
    assert_eq!(
        invalid_json_lines_by_file(&second),
        [(logs.original.clone(), 1)]
    );
    assert_eq!(collision_snapshot(&conn), before);

    let with_model = index_from_dirs(&mut conn, &logs.opts(), true).unwrap();
    assert_eq!(
        (
            with_model.indexed,
            with_model.preserved_embedded,
            with_model.total_sessions
        ),
        (1, 0, 1)
    );
    assert_eq!(
        invalid_json_lines_by_file(&with_model),
        [(logs.original.clone(), 1)]
    );
    assert_the_session_holds_only_the_replacement(&mut conn, &logs.replacement);

    fs::remove_file(&logs.original).unwrap();
    fs::remove_file(&logs.replacement).unwrap();
    let emptied = index_from_dirs(&mut conn, &logs.opts(), false).unwrap();
    assert_eq!(emptied.total_sessions, 0);
    assert!(emptied.parse_diagnostics.is_empty());
}

#[test]
fn log_reusing_the_id_of_a_deleted_embedded_session_waits_for_the_model_and_then_replaces_it() {
    let (_dir, mut conn) = setup_test_db();
    let logs = embedded_session_and_a_log_reusing_its_id(&mut conn);
    let before = collision_snapshot(&conn);
    fs::remove_file(&logs.original).unwrap();

    let first = index_from_dirs(&mut conn, &logs.opts(), false).unwrap();
    assert_eq!(
        (
            first.indexed,
            first.preserved_embedded,
            first.total_sessions
        ),
        (0, 1, 1)
    );
    assert_eq!(
        invalid_json_lines_by_file(&first),
        [(logs.original.clone(), 1)]
    );
    assert_eq!(collision_snapshot(&conn), before);
    let second = index_from_dirs(&mut conn, &logs.opts(), false).unwrap();
    assert_eq!(
        (
            second.indexed,
            second.preserved_embedded,
            second.total_sessions
        ),
        (0, 1, 1)
    );
    assert_eq!(
        invalid_json_lines_by_file(&second),
        [(logs.original.clone(), 1)]
    );
    assert_eq!(collision_snapshot(&conn), before);

    let with_model = index_from_dirs(&mut conn, &logs.opts(), true).unwrap();
    assert_eq!(
        (
            with_model.indexed,
            with_model.preserved_embedded,
            with_model.total_sessions
        ),
        (1, 0, 1)
    );
    assert!(with_model.parse_diagnostics.is_empty());
    assert_the_session_holds_only_the_replacement(&mut conn, &logs.replacement);

    fs::remove_file(&logs.replacement).unwrap();
    let emptied = index_from_dirs(&mut conn, &logs.opts(), false).unwrap();
    assert_eq!(emptied.total_sessions, 0);
    assert!(emptied.parse_diagnostics.is_empty());
}

#[test]
fn session_log_nested_ten_directories_deep_is_not_indexed_and_its_root_is_reported_as_incompletely_read()
 {
    let (_dir, mut conn) = setup_test_db();
    let tmp = TempDir::new().unwrap();
    let claude_dir = tmp.path().join("claude_projects");
    let edge_dir = claude_dir.join("d0/d1/d2/d3/d4/d5/d6/d7/d8");
    let deep_dir = claude_dir.join("d0/d1/d2/d3/d4/d5/d6/d7/d8/d9");
    fs::create_dir_all(&deep_dir).unwrap();
    fs::write(claude_dir.join("shallow.jsonl"), FRESHNESS_QUESTION).unwrap();
    fs::write(edge_dir.join("edge.jsonl"), FRESHNESS_QUESTION).unwrap();
    fs::write(deep_dir.join("deep.jsonl"), FRESHNESS_QUESTION).unwrap();
    let absent_codex = tmp.path().join("codex_sessions");

    let stats = index_from_dirs(
        &mut conn,
        &IndexOptions {
            force: false,
            claude_dir: &claude_dir,
            codex_dir: &absent_codex,
            codex_archived_dir: None,
        },
        true,
    )
    .unwrap();

    assert_eq!(
        collect_strings(&conn, "SELECT session_id FROM sessions ORDER BY session_id"),
        ["edge", "shallow"],
        "a log nested nine directories deep is still indexed"
    );
    assert_eq!(
        skipped_roots(&stats),
        [(Source::Claude, SkippedReason::IncompleteEnumeration, 0)]
    );
}

#[cfg(unix)]
#[test]
fn symlinked_session_log_is_not_indexed_as_a_second_session() {
    use std::os::unix::fs::symlink;

    let (_dir, mut conn) = setup_test_db();
    let tmp = TempDir::new().unwrap();
    let claude_dir = tmp.path().join("claude_projects");
    fs::create_dir_all(&claude_dir).unwrap();
    let real = claude_dir.join("real.jsonl");
    fs::write(&real, FRESHNESS_QUESTION).unwrap();
    symlink(&real, claude_dir.join("link.jsonl")).unwrap();
    let absent_codex = tmp.path().join("codex_sessions");

    let stats = index_from_dirs(
        &mut conn,
        &IndexOptions {
            force: false,
            claude_dir: &claude_dir,
            codex_dir: &absent_codex,
            codex_archived_dir: None,
        },
        true,
    )
    .unwrap();

    assert_eq!((stats.indexed, stats.total_sessions), (1, 1));
    assert_eq!(
        collect_strings(&conn, "SELECT session_id FROM sessions"),
        ["real"]
    );
}

#[test]
fn unreadable_directory_entry_marks_the_root_incompletely_read_and_keeps_the_readable_logs() {
    use std::{io, iter};

    let tmp = TempDir::new().unwrap();
    fs::write(tmp.path().join("real.jsonl"), "{}").unwrap();

    let entries =
        iter::once(Err(io::Error::other("injected"))).chain(fs::read_dir(tmp.path()).unwrap());

    let mut files = Vec::new();
    let fully_read = collect_from_entries(entries, tmp.path(), &mut files, Source::Claude, 0);

    assert!(
        !fully_read,
        "a per-entry error must mark the tree as not fully enumerated"
    );
    assert_eq!(
        files.len(),
        1,
        "the readable jsonl is still collected despite the failed entry"
    );
    assert!(files[0].0.ends_with("real.jsonl"));
}

#[test]
fn a_session_chunked_to_nothing_is_not_chunked_again_after_a_reopen_until_its_log_gains_a_turn() {
    let (dir, mut conn) = setup_test_db();
    let tmp = TempDir::new().unwrap();
    let claude_dir = tmp.path().join("claude_projects");
    fs::create_dir_all(&claude_dir).unwrap();
    let path = claude_dir.join("s1.jsonl");
    let answer = r#"{"type":"assistant","cwd":"/proj","message":{"role":"assistant","content":"hi there"},"timestamp":"2026-03-01T00:00:00Z"}"#;
    let question = r#"{"type":"user","message":{"role":"user","content":"hello"}}"#;
    fs::write(&path, answer).unwrap();
    let codex_dir = tmp.path().join("codex_sessions");
    let opts = IndexOptions {
        force: false,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
        codex_archived_dir: None,
    };
    assert_eq!(index_from_dirs(&mut conn, &opts, true).unwrap().indexed, 1);
    let first = Observer::default();
    index_chunks_observed(&mut conn, &first).unwrap();
    assert_eq!(
        first.snapshot()["counts"],
        serde_json::json!({
            "chunks_created_committed": 0,
            "sessions_chunk_pending": 1,
            "sessions_chunk_remaining": 0,
            "sessions_chunked_committed": 1,
            "sessions_chunked_empty": 1,
        })
    );
    assert!(collect_strings(&conn, "SELECT content FROM qa_chunks").is_empty());

    drop(conn);
    let mut conn = open_db(&dir.path().join("test.db")).unwrap();
    assert_eq!(index_from_dirs(&mut conn, &opts, true).unwrap().indexed, 0);
    let (unchanged, lines) = recording();
    let stats = index_chunks_observed(&mut conn, &unchanged).unwrap();
    // Performance guard (docs/audit/2026-09-20-chunk-index.md): a session already chunked to
    // nothing costs no FTS scan, no body batch and no message read.
    assert_eq!(
        (
            stats.message_scans,
            stats.message_batches,
            stats.messages_read
        ),
        (0, 0, 0)
    );
    assert_eq!(
        stable_lines(&lines.borrow()),
        [
            "index: chunking: started (not yet complete)",
            "index: chunking: no pending sessions"
        ]
    );
    assert_eq!(
        unchanged.snapshot()["counts"],
        serde_json::json!({
            "chunks_created_committed": 0,
            "sessions_chunk_pending": 0,
            "sessions_chunk_remaining": 0,
            "sessions_chunked_committed": 0,
            "sessions_chunked_empty": 0,
        })
    );

    fs::write(&path, format!("{answer}\n{question}")).unwrap();
    assert_eq!(index_from_dirs(&mut conn, &opts, true).unwrap().indexed, 1);
    let appended = Observer::default();
    index_chunks_observed(&mut conn, &appended).unwrap();
    assert_eq!(
        appended.snapshot()["counts"],
        serde_json::json!({
            "chunks_created_committed": 1,
            "sessions_chunk_pending": 1,
            "sessions_chunk_remaining": 0,
            "sessions_chunked_committed": 1,
            "sessions_chunked_empty": 0,
        })
    );
    assert_eq!(
        collect_strings(&conn, "SELECT content FROM qa_chunks"),
        ["hello"]
    );
}

#[test]
fn recall_claude_dir_takes_priority_over_claude_config_dir() {
    let home = Path::new("/fake/home");
    let result = resolve_claude_dir_with(home, |key| match key {
        "RECALL_CLAUDE_DIR" => Some(OsString::from("/custom/recall/dir")),
        "CLAUDE_CONFIG_DIR" => Some(OsString::from("/custom/config/dir")),
        _ => None,
    });
    assert_eq!(
        result,
        PathBuf::from("/custom/recall/dir"),
        "RECALL_CLAUDE_DIR should take priority over CLAUDE_CONFIG_DIR"
    );
}

#[test]
fn claude_config_dir_projects_is_used_when_recall_claude_dir_is_unset() {
    let home = Path::new("/fake/home");
    let result = resolve_claude_dir_with(home, |key| match key {
        "CLAUDE_CONFIG_DIR" => Some(OsString::from("/custom/config")),
        _ => None,
    });
    assert_eq!(
        result,
        PathBuf::from("/custom/config/projects"),
        "CLAUDE_CONFIG_DIR/projects should be used when RECALL_CLAUDE_DIR is unset"
    );
}

#[test]
fn session_root_that_is_not_an_absolute_path_is_not_read_and_keeps_its_sessions() {
    let (_dir, mut conn) = setup_test_db();
    let (_tmp, _claude_dir, codex_dir) = seed_claude_and_codex(&mut conn);

    // `src` exists relative to the package root, where the test runner starts, and holds
    // no session logs: read as a root, it would look like every Claude session was deleted.
    let stats = index_from_dirs(
        &mut conn,
        &IndexOptions {
            force: false,
            claude_dir: Path::new("src"),
            codex_dir: &codex_dir,
            codex_archived_dir: None,
        },
        true,
    )
    .unwrap();

    assert_eq!(
        collect_strings(&conn, "SELECT source FROM sessions ORDER BY source"),
        ["claude", "codex"]
    );
    assert_eq!(
        skipped_roots(&stats),
        [(Source::Claude, SkippedReason::IncompleteEnumeration, 1)]
    );
}

#[test]
fn empty_session_root_path_is_not_read_and_keeps_its_sessions() {
    let (_dir, mut conn) = setup_test_db();
    let (_tmp, _claude_dir, codex_dir) = seed_claude_and_codex(&mut conn);

    let stats = index_from_dirs(
        &mut conn,
        &IndexOptions {
            force: false,
            claude_dir: Path::new(""),
            codex_dir: &codex_dir,
            codex_archived_dir: None,
        },
        true,
    )
    .unwrap();

    assert_eq!(
        collect_strings(&conn, "SELECT source FROM sessions ORDER BY source"),
        ["claude", "codex"]
    );
    assert_eq!(
        skipped_roots(&stats),
        [(Source::Claude, SkippedReason::IncompleteEnumeration, 1)]
    );
}

#[test]
fn recall_codex_dir_replaces_the_default_codex_sessions_dir() {
    let home = Path::new("/fake/home");
    let result = resolve_codex_dir_with(home, |key| match key {
        "RECALL_CODEX_DIR" => Some(OsString::from("/custom/codex/sessions")),
        _ => None,
    });
    assert_eq!(result, PathBuf::from("/custom/codex/sessions"));
}

#[test]
fn codex_sessions_dir_defaults_to_dot_codex_sessions_under_home() {
    let home = Path::new("/fake/home");
    let result = resolve_codex_dir_with(home, |_| None);
    assert_eq!(result, PathBuf::from("/fake/home/.codex/sessions"));
}

fn seed_message(conn: &Connection, role: &str, text: &str) -> i64 {
    conn.execute(
        "INSERT INTO messages (session_id, role, text) VALUES ('s1', ?1, ?2)",
        rusqlite::params![role, text],
    )
    .unwrap();
    conn.query_row(
        "SELECT rowid FROM messages WHERE session_id = 's1' AND text = ?1",
        [text],
        |r| r.get(0),
    )
    .unwrap()
}

fn seed_legacy_chunk(conn: &Connection, id: i64, content: &str) {
    conn.execute(
        "INSERT INTO qa_chunks (id, session_id, content, timestamp) VALUES (?1, 's1', ?2, 0)",
        rusqlite::params![id, content],
    )
    .unwrap();
    let embedding = MockEmbedder::deterministic_vector(content);
    conn.execute(
        "INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?1, ?2)",
        rusqlite::params![id, f32_as_bytes(&embedding)],
    )
    .unwrap();
}

#[test]
fn rowid_backfill_links_a_legacy_chunk_to_its_messages_and_keeps_its_embedding() {
    let (_dir, mut conn) = setup_test_db();
    seed_session(&conn, "s1");
    let user_rowid = seed_message(&conn, "user", "how to rotate tokens");
    let assistant_rowid = seed_message(&conn, "assistant", "use a refresh grant");
    seed_legacy_chunk(&conn, 1, "how to rotate tokens\nuse a refresh grant");

    let vec_before: i64 = conn
        .query_row("SELECT COUNT(*) FROM vec_chunks", [], |r| r.get(0))
        .unwrap();

    let updated = backfill_rowid_ranges(&mut conn).unwrap();
    assert_eq!(updated, 1, "the count-matching session is backfilled");

    let (lo, hi): (Option<i64>, Option<i64>) = conn
        .query_row(
            "SELECT src_rowid_lo, src_rowid_hi FROM qa_chunks WHERE id = 1",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    assert_eq!(
        lo,
        Some(user_rowid),
        "backfill sets lo to the user message rowid"
    );
    assert_eq!(
        hi,
        Some(assistant_rowid),
        "backfill sets hi to the adjacent assistant message rowid"
    );

    let vec_after: i64 = conn
        .query_row("SELECT COUNT(*) FROM vec_chunks", [], |r| r.get(0))
        .unwrap();
    assert_eq!(
        vec_after, vec_before,
        "backfill must not delete or re-create embeddings"
    );
    let embedding_kept: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM vec_chunks WHERE chunk_id = 1",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        embedding_kept, 1,
        "the chunk's embedding stays linked by its unchanged id"
    );
}

#[test]
fn rowid_backfill_leaves_legacy_chunks_unlinked_when_rechunking_yields_a_different_count() {
    let (_dir, mut conn) = setup_test_db();
    seed_session(&conn, "s1");
    seed_message(&conn, "user", "single turn");
    seed_legacy_chunk(&conn, 1, "single turn");
    seed_legacy_chunk(&conn, 2, "orphaned extra chunk");

    let updated = backfill_rowid_ranges(&mut conn).unwrap();
    assert_eq!(
        updated, 0,
        "a count-mismatched session is skipped, not backfilled"
    );

    let null_count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM qa_chunks WHERE session_id = 's1' AND src_rowid_lo IS NULL",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        null_count, 2,
        "both rows keep NULL lo/hi when the chunk count cannot be reconciled"
    );
}

#[test]
fn rowid_backfill_leaves_a_legacy_chunk_unlinked_when_rechunking_yields_the_same_count_but_different_content()
 {
    let (_dir, mut conn) = setup_test_db();
    seed_session(&conn, "s1");
    seed_message(&conn, "user", "how do i tune the cache");
    seed_message(&conn, "assistant", "raise the ttl first");
    seed_message(&conn, "assistant", "then widen the shard count");
    seed_legacy_chunk(&conn, 1, "how do i tune the cache\nraise the ttl first");

    let updated = backfill_rowid_ranges(&mut conn).unwrap();
    assert_eq!(
        updated, 0,
        "a count-matching but content-drifted session is skipped, not backfilled"
    );

    let (lo, hi): (Option<i64>, Option<i64>) = conn
        .query_row(
            "SELECT src_rowid_lo, src_rowid_hi FROM qa_chunks WHERE id = 1",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    assert_eq!(
        (lo, hi),
        (None, None),
        "the drifted chunk keeps NULL lo/hi so it resolves via the instr fallback, \
         never inheriting the widened [Q..A2] range over its stale content"
    );
}

#[test]
fn edit_and_write_file_paths_are_stored_as_touched_files_and_kept_out_of_messages_and_chunks() {
    let (_dir, mut conn) = setup_test_db();
    let tmp = TempDir::new().unwrap();
    let claude_dir = tmp.path().join("claude_projects");
    fs::create_dir_all(&claude_dir).unwrap();
    fs::write(
        claude_dir.join("edit_session.jsonl"),
        concat!(
            r#"{"type":"user","cwd":"/proj","message":{"role":"user","content":"please make the edits"},"timestamp":"2026-03-01T00:00:00Z"}"#,
            "\n",
            r#"{"type":"assistant","cwd":"/proj","message":{"role":"assistant","content":[{"type":"text","text":"doing the edits"},{"type":"tool_use","id":"t1","name":"Edit","input":{"file_path":"/proj/alpha_marker.rs","old_string":"x","new_string":"y"}},{"type":"tool_use","id":"t2","name":"Write","input":{"file_path":"/proj/beta_marker.rs","content":"filecontents"}}]},"timestamp":"2026-03-01T00:00:01Z"}"#,
        ),
    )
    .unwrap();
    let codex_dir = tmp.path().join("codex_sessions");

    index_from_dirs(
        &mut conn,
        &IndexOptions {
            force: false,
            claude_dir: &claude_dir,
            codex_dir: &codex_dir,
            codex_archived_dir: None,
        },
        true,
    )
    .unwrap();
    index_chunks(&mut conn).unwrap();

    assert_eq!(
        collect_strings(
            &conn,
            "SELECT session_id || ' ' || path FROM session_files ORDER BY path"
        ),
        [
            "edit_session /proj/alpha_marker.rs",
            "edit_session /proj/beta_marker.rs"
        ],
        "Edit and Write file_path values are stored as the session's touched files"
    );
    assert_eq!(
        collect_strings(&conn, "SELECT text FROM messages ORDER BY rowid"),
        ["please make the edits", "doing the edits"],
        "messages hold the text blocks only, not the tool_use inputs"
    );
    assert_eq!(
        collect_strings(&conn, "SELECT content FROM qa_chunks"),
        ["please make the edits\ndoing the edits"],
        "chunks hold the text blocks only, not the tool_use inputs"
    );
}

#[test]
fn touched_file_paths_that_are_empty_too_long_or_contain_a_newline_are_dropped() {
    let (_dir, mut conn) = setup_test_db();
    let tmp = TempDir::new().unwrap();
    let claude_dir = tmp.path().join("claude_projects");
    fs::create_dir_all(&claude_dir).unwrap();

    let long_path = format!("/proj/{}.rs", "a".repeat(100_000));
    let assistant = serde_json::json!({
        "type": "assistant",
        "cwd": "/proj",
        "message": {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "editing"},
                {"type": "tool_use", "id": "t1", "name": "Edit",
                 "input": {"file_path": "/proj/valid_control.rs", "old_string": "x", "new_string": "y"}},
                {"type": "tool_use", "id": "t2", "name": "Edit",
                 "input": {"file_path": "/proj/bad\nname.rs", "old_string": "x", "new_string": "y"}},
                {"type": "tool_use", "id": "t3", "name": "Write",
                 "input": {"file_path": long_path, "content": "z"}},
                {"type": "tool_use", "id": "t4", "name": "Edit",
                 "input": {"file_path": "", "old_string": "x", "new_string": "y"}}
            ]
        },
        "timestamp": "2026-03-01T00:00:01Z"
    });
    let user_line = r#"{"type":"user","cwd":"/proj","message":{"role":"user","content":"edit these files"},"timestamp":"2026-03-01T00:00:00Z"}"#;
    let contents = format!(
        "{}\n{}\n",
        user_line,
        serde_json::to_string(&assistant).unwrap()
    );
    fs::write(claude_dir.join("filter_session.jsonl"), contents).unwrap();
    let codex_dir = tmp.path().join("codex_sessions");

    index_from_dirs(
        &mut conn,
        &IndexOptions {
            force: false,
            claude_dir: &claude_dir,
            codex_dir: &codex_dir,
            codex_archived_dir: None,
        },
        true,
    )
    .unwrap();

    let paths = collect_strings(&conn, "SELECT path FROM session_files ORDER BY path");
    assert_eq!(
        paths,
        vec!["/proj/valid_control.rs".to_owned()],
        "only the valid file_path is kept; newline / over-length / empty are filtered"
    );
}

/// A Claude log whose answer edits `/proj/edited.rs`.
const LOG_THAT_EDITS_A_FILE: &str = concat!(
    r#"{"type":"user","cwd":"/proj","message":{"role":"user","content":"edit it"}}"#,
    "\n",
    r#"{"type":"assistant","cwd":"/proj","message":{"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"Edit","input":{"file_path":"/proj/edited.rs","old_string":"x","new_string":"y"}}]}}"#,
    "\n",
);

#[test]
fn file_backfill_leaves_a_session_whose_log_cannot_be_read_pending_and_records_its_touched_files_once_the_log_is_readable()
 {
    let (_dir, mut conn) = setup_test_db();
    let tmp = TempDir::new().unwrap();
    let log = tmp.path().join("legacy.jsonl");
    conn.execute(
        "INSERT INTO sessions (session_id, source, file_path, project, slug, timestamp, mtime, files_scanned) VALUES ('legacy', 'claude', ?1, '/p', 'slug', 0, 0.0, NULL)",
        [log.to_str().unwrap()],
    )
    .unwrap();

    assert_eq!(
        backfill_session_files(&mut conn, None).unwrap(),
        0,
        "a log that cannot be read marks no session"
    );
    assert!(collect_strings(&conn, "SELECT path FROM session_files").is_empty());
    let diagnostics = load_parse_diagnostics(&conn).unwrap();
    assert_eq!(diagnostics.len(), 1);
    assert_eq!(
        diagnostics[0].notes(),
        [format!(
            "{}: 1 file could not be read; check file access and UTF-8 path encoding, then rerun `recall index`; existing indexed data was retained",
            log.display()
        )]
    );

    fs::write(&log, LOG_THAT_EDITS_A_FILE).unwrap();
    assert_eq!(
        backfill_session_files(&mut conn, None).unwrap(),
        1,
        "the next backfill reads the session again"
    );
    assert_eq!(
        collect_strings(&conn, "SELECT session_id || ' ' || path FROM session_files"),
        ["legacy /proj/edited.rs"]
    );
}

#[test]
fn file_backfill_marks_sessions_with_an_unknown_source_or_an_empty_log_as_done_and_does_not_read_them_again()
 {
    let (_dir, mut conn) = setup_test_db();
    let tmp = TempDir::new().unwrap();
    let unknown = tmp.path().join("unknown.jsonl");
    let empty = tmp.path().join("empty.jsonl");
    fs::write(&empty, "").unwrap();
    conn.execute(
        "INSERT INTO sessions (session_id, source, file_path, project, slug, timestamp, mtime, files_scanned) VALUES ('mystery', 'perplexity', ?1, '/p', 'slug', 0, 0.0, NULL)",
        [unknown.to_str().unwrap()],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO sessions (session_id, source, file_path, project, slug, timestamp, mtime, files_scanned) VALUES ('zero', 'claude', ?1, '/p', 'slug', 0, 0.0, NULL)",
        [empty.to_str().unwrap()],
    )
    .unwrap();

    assert_eq!(
        backfill_session_files(&mut conn, None).unwrap(),
        2,
        "both sessions are marked done"
    );
    assert!(collect_strings(&conn, "SELECT path FROM session_files").is_empty());

    fs::write(&unknown, LOG_THAT_EDITS_A_FILE).unwrap();
    fs::write(&empty, LOG_THAT_EDITS_A_FILE).unwrap();
    assert_eq!(
        backfill_session_files(&mut conn, None).unwrap(),
        0,
        "a marked session is not read again"
    );
    assert!(collect_strings(&conn, "SELECT path FROM session_files").is_empty());
}

#[test]
fn a_failed_chunk_completion_rolls_back_the_whole_pass_and_the_next_pass_chunks_every_session() {
    let (_dir, mut conn) = setup_test_db();
    seed_session(&conn, "empty");
    seed_session(&conn, "s1");
    seed_message(&conn, "user", "question");
    conn.execute_batch(
        "CREATE TEMP TRIGGER fail_completion BEFORE UPDATE OF chunks_indexed ON sessions
         WHEN NEW.session_id = 's1'
         BEGIN SELECT RAISE(ABORT, 'injected completion failure'); END;",
    )
    .unwrap();
    let (observer, lines) = recording();

    let error = index_chunks_observed(&mut conn, &observer).err().unwrap();

    assert_eq!(error.to_string(), "injected completion failure");
    assert!(collect_strings(&conn, "SELECT content FROM qa_chunks").is_empty());
    assert_eq!(
        stable_lines(&lines.borrow()),
        [
            "index: chunking: started (not yet complete)",
            "index: chunking: unfinished; only earlier commit notifications confirm saves"
        ]
    );
    assert_eq!(
        observer.snapshot()["counts"],
        serde_json::json!({
            "chunks_created_committed": 0,
            "sessions_chunk_pending": 2,
            "sessions_chunk_remaining": 2,
            "sessions_chunked_committed": 0,
        })
    );

    conn.execute_batch("DROP TRIGGER fail_completion;").unwrap();
    let retry = Observer::default();
    index_chunks_observed(&mut conn, &retry).unwrap();
    assert_eq!(
        retry.snapshot()["counts"],
        serde_json::json!({
            "chunks_created_committed": 1,
            "sessions_chunk_pending": 2,
            "sessions_chunk_remaining": 0,
            "sessions_chunked_committed": 2,
            "sessions_chunked_empty": 1,
        }),
        "the empty session chunked before the failure is pending again"
    );
    assert_eq!(
        collect_strings(&conn, "SELECT content FROM qa_chunks"),
        ["question"]
    );
}

#[test]
fn chunk_pass_with_nothing_pending_reports_no_pending_sessions_and_zero_counts() {
    let (_dir, mut conn) = setup_test_db();
    let (observer, lines) = recording();

    let stats = index_chunks_observed(&mut conn, &observer).unwrap();

    assert_eq!(stats.chunks_created, 0);
    assert_eq!(
        stable_lines(&lines.borrow()),
        [
            "index: chunking: started (not yet complete)",
            "index: chunking: no pending sessions"
        ]
    );
    assert_eq!(
        observer.snapshot()["counts"],
        serde_json::json!({
            "chunks_created_committed": 0,
            "sessions_chunk_pending": 0,
            "sessions_chunk_remaining": 0,
            "sessions_chunked_committed": 0,
            "sessions_chunked_empty": 0,
        })
    );
}

#[test]
fn chunk_pass_with_pending_sessions_reports_its_stage_lines_and_counts_and_stores_each_chunk_with_its_message_range()
 {
    let (_dir, mut conn) = setup_test_db();
    seed_session(&conn, "empty");
    seed_session(&conn, "empty2");
    seed_session(&conn, "s1");
    conn.execute(
        "UPDATE sessions SET timestamp = 1234 WHERE session_id = 's1'",
        [],
    )
    .unwrap();
    seed_message(&conn, "user", "question");
    seed_message(&conn, "unknown-role", "dropped text");
    let (observer, lines) = recording();

    let stats = index_chunks_observed(&mut conn, &observer).unwrap();

    assert_eq!(stats.chunks_created, 1);
    assert_eq!(
        stable_lines(&lines.borrow()),
        [
            "index: chunking: started (not yet complete)",
            "index: chunking: processed=3/3 unprocessed=0 committed=0",
            "index: chunking: committed"
        ]
    );
    assert_eq!(
        observer.snapshot()["counts"],
        serde_json::json!({
            "chunks_created_committed": 1,
            "sessions_chunk_pending": 3,
            "sessions_chunk_remaining": 0,
            "sessions_chunked_committed": 3,
            "sessions_chunked_empty": 2,
        })
    );
    assert_eq!(
        collect_strings(
            &conn,
            "SELECT json_array(session_id, content, timestamp, src_rowid_lo, src_rowid_hi) FROM qa_chunks ORDER BY id"
        ),
        [r#"["s1","question",1234,1,1]"#]
    );

    seed_session(&conn, "later");
    let next = Observer::default();
    index_chunks_observed(&mut conn, &next).unwrap();
    assert_eq!(
        next.snapshot()["counts"],
        serde_json::json!({
            "chunks_created_committed": 0,
            "sessions_chunk_pending": 1,
            "sessions_chunk_remaining": 0,
            "sessions_chunked_committed": 1,
            "sessions_chunked_empty": 1,
        }),
        "a second pass on the same connection chunks only the new session"
    );
}

#[test]
fn chunk_batches_limit_session_count_and_source_bytes() {
    // Performance guard (docs/audit/2026-09-20-chunk-index.md): a body batch holds at most 64
    // sessions and 8 MiB of message text, and a session over 8 MiB is read alone. The guard
    // calls the private chunk_batch_len with session sizes, because reaching 8 MiB through
    // index_chunks would need megabytes of message text.
    let one_byte = ("s".to_owned(), None, 1);
    let four_mib = ("s".to_owned(), None, 4_194_304);
    let over_eight_mib = ("s".to_owned(), None, 8_388_609);
    let no_bytes = ("s".to_owned(), None, 0);

    assert_eq!(chunk_batch_len(&vec![one_byte.clone(); 65]), 64);
    assert_eq!(
        chunk_batch_len(&[four_mib.clone(), four_mib, one_byte.clone()]),
        2,
        "exactly 8 MiB fits in one batch"
    );
    assert_eq!(
        chunk_batch_len(&[one_byte, over_eight_mib.clone()]),
        1,
        "an oversized session is not added to a batch"
    );
    assert_eq!(
        chunk_batch_len(&[over_eight_mib, no_bytes.clone()]),
        1,
        "an oversized session is read alone"
    );
    assert_eq!(
        chunk_batch_len(&[no_bytes]),
        1,
        "a session without text still forms a batch"
    );
}

/// Returns the single integer `sql` selects, such as a `COUNT(*)`.
fn count_rows(conn: &Connection, sql: &str) -> i64 {
    conn.query_row(sql, [], |r| r.get(0)).unwrap()
}

/// Returns the detail column of `EXPLAIN QUERY PLAN` for `sql` bound to `params`.
fn query_plan(conn: &Connection, sql: &str, params: &[&str]) -> Vec<String> {
    conn.prepare(&format!("EXPLAIN QUERY PLAN {sql}"))
        .unwrap()
        .query_map(rusqlite::params_from_iter(params), |r| r.get(3))
        .unwrap()
        .map(Result::unwrap)
        .collect()
}

#[test]
fn chunk_batches_preserve_interleaved_sessions_order_and_search_ranges() {
    let (_dir, mut conn) = setup_test_db();
    // 65 sessions, s000 to s064, cross the 64-session batch limit. Their messages are
    // inserted turn by turn, s064 first, so the rowids of every session interleave.
    conn.execute_batch(
        "WITH RECURSIVE n(i) AS (SELECT 0 UNION ALL SELECT i + 1 FROM n WHERE i < 64)
         INSERT INTO sessions (session_id, source, file_path, project, slug, timestamp, mtime)
         SELECT printf('s%03d', i), 'claude', '/f', '/p', 'slug', 0, 0.0 FROM n;
         WITH RECURSIVE n(i) AS (SELECT 0 UNION ALL SELECT i + 1 FROM n WHERE i < 64),
         turns(turn, role) AS
           (VALUES (0, 'user'), (1, 'system'), (2, 'assistant'), (3, 'assistant'))
         INSERT INTO messages (session_id, role, text)
         SELECT printf('s%03d', i), role, printf('session %d turn %d', i, turn)
         FROM turns, n ORDER BY turn, i DESC;",
    )
    .unwrap();
    let observer = Observer::default();

    let stats = index_chunks_observed(&mut conn, &observer).unwrap();

    assert_eq!(
        observer.snapshot()["counts"],
        serde_json::json!({
            "chunks_created_committed": 65,
            "sessions_chunk_pending": 65,
            "sessions_chunk_remaining": 0,
            "sessions_chunked_committed": 65,
            "sessions_chunked_empty": 0,
        })
    );
    assert_eq!(
        collect_strings(
            &conn,
            "SELECT json_array(session_id, content, src_rowid_lo, src_rowid_hi) FROM qa_chunks WHERE session_id IN ('s000', 's063', 's064') ORDER BY session_id"
        ),
        [
            r#"["s000","session 0 turn 0\nsession 0 turn 2\nsession 0 turn 3",65,260]"#,
            r#"["s063","session 63 turn 0\nsession 63 turn 2\nsession 63 turn 3",2,197]"#,
            r#"["s064","session 64 turn 0\nsession 64 turn 2\nsession 64 turn 3",1,196]"#,
        ],
        "the first and last session of the first batch and the session of the second batch \
         each keep only their own turns, the two answers merged, and their own rowid range"
    );
    assert_eq!(
        count_rows(
            &conn,
            "SELECT COUNT(*) FROM qa_chunks c WHERE c.content IS NOT (
               SELECT group_concat(m.text, char(10) ORDER BY m.rowid) FROM messages m
               WHERE m.session_id = c.session_id AND m.role IN ('user', 'assistant'))"
        ),
        0,
        "every chunk holds its own session's question and answers, newline-joined in rowid order"
    );
    assert_eq!(
        count_rows(
            &conn,
            "SELECT COUNT(*) FROM qa_chunks c WHERE (c.src_rowid_lo, c.src_rowid_hi) IS NOT (
               SELECT min(m.rowid), max(m.rowid) FROM messages m WHERE m.session_id = c.session_id)"
        ),
        0,
        "every chunk's rowid range runs from its own session's first message to its last"
    );
    // Performance guard (docs/audit/2026-09-20-chunk-index.md): one FTS scan stages every
    // pending message, the 65 sessions are read in two body batches, and each of the 260
    // messages is fetched once.
    assert_eq!(
        (
            stats.message_scans,
            stats.message_batches,
            stats.messages_read
        ),
        (1, 2, 260)
    );
    // Performance guard (same audit): staging keeps FTS as the single outer scan and probes
    // sessions by key. Each batch reads message bodies by rowid equality. No other row scans
    // messages or sessions. Index names and the FTS5 index string change with the SQLite
    // version and the schema, so the asserts read only the access and the table. A virtual
    // table row always reads SCAN, and FTS5 marks a rowid equality with `=` in its index string.
    conn.execute_batch(CREATE_CHUNK_MESSAGE_ROWS).unwrap();
    let stage = query_plan(&conn, STAGE_CHUNK_MESSAGES, &[]);
    assert_eq!(stage.len(), 2, "{stage:?}");
    assert!(
        stage[0].starts_with("SCAN m VIRTUAL TABLE "),
        "FTS is the single outer scan: {stage:?}"
    );
    assert!(
        stage[1].starts_with("SEARCH s ")
            && stage[1].ends_with("(session_id=?)")
            && !stage[1].contains("AUTOMATIC"),
        "each staged message probes sessions by their key index, not an automatic one: {stage:?}"
    );
    let read = query_plan(&conn, READ_CHUNK_BATCH, &["s000", "s064"]);
    assert_eq!(read.len(), 2, "{read:?}");
    assert!(
        read[0].starts_with("SEARCH r USING PRIMARY KEY "),
        "a batch reads its sessions' staged rows by key range: {read:?}"
    );
    assert!(
        read[1].starts_with("SCAN m VIRTUAL TABLE ") && read[1].contains('='),
        "each staged row reads its message body by rowid equality: {read:?}"
    );
}

#[test]
fn parse_diagnostics_survive_freshness_preservation_and_root_outages_until_repaired() {
    let (db_dir, mut conn) = setup_test_db();
    let tmp = TempDir::new().unwrap();
    let claude_dir = tmp.path().join("claude");
    let codex_dir = tmp.path().join("codex");
    fs::create_dir(&claude_dir).unwrap();
    let path = claude_dir.join("partial.jsonl");
    let valid = "{\"type\":\"user\",\"message\":\"retained question\"}\n";
    fs::write(&path, format!("{valid}{{broken\n")).unwrap();
    let opts = IndexOptions {
        force: false,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
        codex_archived_dir: None,
    };
    let stats = index_from_dirs(&mut conn, &opts, true).unwrap();
    assert_eq!(stats.indexed, 1);
    assert_eq!(stats.parse_diagnostics[0].lines.invalid_json_lines, 1);
    index_chunks(&mut conn).unwrap();
    embed_recent_chunks(&mut conn, &MockEmbedder::new(), 100, None).unwrap();
    let chunks = collect_strings(
        &conn,
        "SELECT json_array(id, content, generation) FROM qa_chunks",
    );
    let vectors = collect_strings(&conn, "SELECT hex(embedding) FROM vec_chunks");
    drop(conn);
    let mut conn = open_db(&db_dir.path().join("test.db")).unwrap();
    let stats = index_from_dirs(&mut conn, &opts, true).unwrap();
    assert_eq!(stats.indexed, 0);
    assert_eq!(stats.parse_diagnostics[0].lines.invalid_json_lines, 1);
    assert_eq!(
        collect_strings(
            &conn,
            "SELECT json_array(id, content, generation) FROM qa_chunks"
        ),
        chunks
    );

    fs::write(&path, valid).unwrap();
    let stats = index_from_dirs(&mut conn, &opts, false).unwrap();
    assert_eq!(stats.preserved_embedded, 1);
    assert_eq!(stats.parse_diagnostics[0].lines.invalid_json_lines, 1);
    assert_eq!(
        collect_strings(&conn, "SELECT hex(embedding) FROM vec_chunks"),
        vectors
    );

    let hidden = tmp.path().join("hidden");
    fs::rename(&claude_dir, &hidden).unwrap();
    let stats = index_from_dirs(&mut conn, &opts, true).unwrap();
    assert_eq!(stats.skipped_roots.len(), 1);
    assert_eq!(stats.parse_diagnostics.len(), 1);
    fs::rename(&hidden, &claude_dir).unwrap();
    let stats = index_from_dirs(&mut conn, &opts, true).unwrap();
    assert_eq!(stats.indexed, 1);
    assert!(stats.parse_diagnostics.is_empty());
    assert_eq!(
        collect_strings(&conn, "SELECT text FROM messages"),
        ["retained question"]
    );

    let tail = claude_dir.join("tail.jsonl");
    fs::write(
        &tail,
        "{\"type\":\"user\",\"message\":\"completed by append",
    )
    .unwrap();
    let stats = index_from_dirs(&mut conn, &opts, true).unwrap();
    assert_eq!(stats.parse_diagnostics[0].lines.incomplete_tail_lines, 1);
    assert_eq!(stats.total_sessions, 1);
    {
        use std::io::Write;
        let mut file = fs::OpenOptions::new().append(true).open(&tail).unwrap();
        file.write_all(b"\"}\n").unwrap();
    }
    let stats = index_from_dirs(&mut conn, &opts, true).unwrap();
    assert!(stats.parse_diagnostics.is_empty());
    assert_eq!(stats.total_sessions, 2);
    assert!(
        collect_strings(&conn, "SELECT text FROM messages")
            .contains(&"completed by append".to_owned())
    );
    fs::remove_file(tail).unwrap();

    let empty = claude_dir.join("no-messages.jsonl");
    fs::write(&empty, "invalid\n").unwrap();
    let stats = index_from_dirs(&mut conn, &opts, true).unwrap();
    assert_eq!(stats.parse_diagnostics.len(), 1);
    assert_eq!(stats.total_sessions, 1);
    fs::remove_file(empty).unwrap();
    assert!(
        index_from_dirs(&mut conn, &opts, true)
            .unwrap()
            .parse_diagnostics
            .is_empty()
    );
}

#[test]
fn read_failure_retains_body_and_line_diagnostics_then_retries_unchanged_metadata() {
    let (_dir, mut conn) = setup_test_db();
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("s.jsonl");
    fs::write(&path, "{\"type\":\"user\",\"message\":\"kept\"}\ninvalid\n").unwrap();
    let missing = tmp.path().join("no-codex");
    let opts = IndexOptions {
        force: false,
        claude_dir: tmp.path(),
        codex_dir: &missing,
        codex_archived_dir: None,
    };
    index_from_dirs(&mut conn, &opts, true).unwrap();
    let existing = load_existing_sessions(&conn).unwrap();
    let tx = conn.transaction().unwrap();
    let ctx = IndexContext {
        tx: &tx,
        existing: &existing,
        force: true,
        embedded_sessions: None,
    };
    index_file_with_parser(&ctx, &path, &Source::Claude, |_, _| {
        Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "private content must not escape",
        )
        .into())
    })
    .unwrap();
    tx.commit().unwrap();
    let d = load_parse_diagnostics(&conn).unwrap();
    assert!(d[0].read_error);
    assert_eq!(d[0].lines.invalid_json_lines, 1);
    assert_eq!(
        collect_strings(&conn, "SELECT text FROM messages"),
        ["kept"]
    );
    let stats = index_from_dirs(&mut conn, &opts, true).unwrap();
    assert_eq!(stats.indexed, 1);
    assert!(!stats.parse_diagnostics[0].read_error);
    assert_eq!(stats.parse_diagnostics[0].lines.invalid_json_lines, 1);
}

#[cfg(unix)]
#[test]
fn colliding_display_paths_keep_independent_diagnostics_until_each_file_is_resolved() {
    use std::os::unix::ffi::OsStringExt;

    fn observe(conn: &mut Connection, paths: &[PathBuf]) -> Vec<FileParseDiagnostics> {
        let existing = load_existing_sessions(conn).unwrap();
        let tx = conn.transaction().unwrap();
        let ctx = IndexContext {
            tx: &tx,
            existing: &existing,
            force: false,
            embedded_sessions: None,
        };
        let sources: Vec<_> = paths.iter().cloned().map(|p| (p, Source::Claude)).collect();
        index_all(&ctx, &sources, &Observer::default()).unwrap();
        let source_paths = paths.iter().map(PathBuf::as_path).collect();
        let diagnostics = cleanup_parse_diagnostics(
            &tx,
            &source_paths,
            &RemovalGate::fully_read([Source::Claude]),
        )
        .unwrap();
        tx.commit().unwrap();
        diagnostics
    }

    let (db_dir, mut conn) = setup_test_db();
    let tmp = TempDir::new().unwrap();
    let paths = [
        tmp.path()
            .join(OsString::from_vec(b"bad-\xff.jsonl".to_vec())),
        tmp.path()
            .join(OsString::from_vec(b"bad-\xfe.jsonl".to_vec())),
        tmp.path().join("bad-\u{fffd}.jsonl"),
    ];
    fs::write(&paths[2], "invalid JSON\n").unwrap();
    let diagnostics = observe(&mut conn, &paths);
    assert_eq!(diagnostics.len(), 3);
    assert_eq!(diagnostics.iter().filter(|d| d.read_error).count(), 2);
    assert!(
        diagnostics
            .iter()
            .all(|d| { d.display_path() == paths[2].to_str().unwrap() })
    );
    assert_eq!(
        diagnostics
            .iter()
            .map(|d| d.lines.invalid_json_lines)
            .sum::<i64>(),
        1
    );

    drop(conn);
    let mut conn = open_db(&db_dir.path().join("test.db")).unwrap();
    fs::write(&paths[2], "{\"type\":\"user\",\"message\":\"readable\"}\n").unwrap();
    for _ in 0..2 {
        let diagnostics = observe(&mut conn, &paths);

        assert_eq!(diagnostics.len(), 2);
        assert!(diagnostics.iter().all(|d| d.read_error));
        let reported: HashSet<_> = diagnostics.iter().map(|d| &d.file_path).collect();
        assert_eq!(reported, HashSet::from([&paths[0], &paths[1]]));
    }
    let diagnostics = observe(&mut conn, &paths[1..]);
    assert_eq!(diagnostics.len(), 1);
    assert_eq!(diagnostics[0].file_path, paths[1]);
    assert!(diagnostics[0].read_error);
    let diagnostics = observe(&mut conn, &paths[2..]);
    assert!(diagnostics.is_empty());
    assert_eq!(
        collect_strings(&conn, "SELECT text FROM messages"),
        ["readable"]
    );
}

#[test]
fn failed_fts_transaction_never_reports_saved_files() {
    let (_dir, mut conn) = setup_test_db();
    let source = TempDir::new().unwrap();
    fs::write(
        source.path().join("s.jsonl"),
        r#"{"type":"user","message":{"role":"user","content":"question"}}"#,
    )
    .unwrap();
    conn.execute_batch("CREATE TRIGGER fail_session BEFORE INSERT ON sessions BEGIN SELECT RAISE(ABORT, 'injected FTS failure'); END;").unwrap();
    let (observer, lines) = recording();
    let absent = source.path().join("absent");
    let opts = IndexOptions {
        force: false,
        claude_dir: source.path(),
        codex_dir: &absent,
        codex_archived_dir: None,
    };
    let error = index_from_dirs_observed(&mut conn, &opts, true, &observer)
        .err()
        .unwrap();
    assert!(error.to_string().contains("injected FTS failure"));
    assert_eq!(observer.snapshot()["counts"]["files_updated_committed"], 0);
    assert!(collect_strings(&conn, "SELECT session_id FROM sessions").is_empty());
    assert!(
        !lines
            .borrow()
            .iter()
            .any(|line| line.contains("fts_transaction: committed"))
    );
    assert!(
        lines
            .borrow()
            .iter()
            .any(|line| line.contains("fts_transaction: unfinished"))
    );
    conn.execute_batch("DROP TRIGGER fail_session;").unwrap();
    assert_eq!(index_from_dirs(&mut conn, &opts, true).unwrap().indexed, 1);
}

#[test]
fn file_outcome_counts_partition_empty_failed_and_deferred_work() {
    let (_dir, mut conn) = setup_test_db();
    let source = TempDir::new().unwrap();
    let body = r#"{"type":"user","message":{"role":"user","content":"question"}}"#;
    let absent = source.path().join("absent");
    let opts = IndexOptions {
        force: false,
        claude_dir: source.path(),
        codex_dir: &absent,
        codex_archived_dir: None,
    };
    let observer = Observer::default();
    index_from_dirs_observed(&mut conn, &opts, true, &observer).unwrap();
    let counts = observer.snapshot()["counts"].clone();
    for key in [
        "files_discovered",
        "files_updated",
        "files_unchanged",
        "files_empty",
        "files_failed",
        "files_deferred",
        "files_remaining",
        "files_updated_committed",
    ] {
        assert_eq!(counts[key], 0, "empty source: {key}");
    }

    fs::write(source.path().join("good.jsonl"), body).unwrap();
    fs::write(source.path().join("empty.jsonl"), "\n").unwrap();
    fs::write(source.path().join("broken.jsonl"), "{broken}\n").unwrap();
    for pass in 0..3 {
        let observer = Observer::default();
        index_from_dirs_observed(&mut conn, &opts, pass != 2, &observer).unwrap();
        let counts = observer.snapshot()["counts"].clone();
        assert_eq!(counts["files_discovered"], 3);
        assert_eq!(counts["files_empty"], 1);
        assert_eq!(counts["files_failed"], 1);
        assert_eq!(counts["files_updated"], usize::from(pass == 0));
        assert_eq!(counts["files_unchanged"], usize::from(pass == 1));
        assert_eq!(counts["files_deferred"], usize::from(pass == 2));
        assert_eq!(counts["files_remaining"], 0);
        index_chunks(&mut conn).unwrap();
        if pass == 1 {
            embed_recent_chunks(&mut conn, &MockEmbedder::new(), usize::MAX, None).unwrap();
            fs::write(
                source.path().join("good.jsonl"),
                format!("{body}\n{body}\n"),
            )
            .unwrap();
        }
    }

    let protected = chunk_vectors(&conn);
    assert_eq!(protected.len(), 1);
    let unembedded = source.path().join("unembedded.jsonl");
    write_reuse_conversation(&unembedded, &[("user", "pending question")]);
    index_from_dirs(&mut conn, &opts, false).unwrap();
    index_chunks(&mut conn).unwrap();
    let before = collect_strings(
        &conn,
        "SELECT json_array(id, generation, content) FROM qa_chunks WHERE session_id = 'unembedded'",
    );
    write_reuse_conversation(
        &unembedded,
        &[("user", "pending question"), ("user", "appended question")],
    );
    let observer = Observer::default();
    let stats = index_from_dirs_observed(&mut conn, &opts, false, &observer).unwrap();
    assert_eq!(stats.indexed, 1);
    assert_eq!(stats.preserved_embedded, 1);
    let counts = observer.snapshot()["counts"].clone();
    assert_eq!(counts["chunks_matched_committed"], 1);
    assert_eq!(counts["embeddings_reused_committed"], 0);
    assert_eq!(counts["chunks_reconciled_created_committed"], 1);
    assert_eq!(index_chunks(&mut conn).unwrap().sessions_chunked, 0);
    assert_eq!(chunk_vectors(&conn), protected);
    let after = collect_strings(
        &conn,
        "SELECT json_array(id, generation, content) FROM qa_chunks WHERE session_id = 'unembedded' ORDER BY id",
    );
    assert_eq!(after.len(), 2);
    assert_eq!(after[..1], before);
    assert_eq!(
        collect_strings(
            &conn,
            "SELECT text FROM messages WHERE session_id = 'unembedded' ORDER BY rowid"
        ),
        ["pending question", "appended question"]
    );
}

/// One Claude log line per `(role, text)` message.
fn reuse_conversation(messages: &[(&str, &str)]) -> String {
    messages
        .iter()
        .map(|(role, text)| {
            serde_json::json!({"type": role, "message": {"role": role, "content": text}})
                .to_string()
                + "\n"
        })
        .collect()
}

fn write_reuse_conversation(path: &Path, messages: &[(&str, &str)]) {
    fs::write(path, reuse_conversation(messages)).unwrap();
}

/// Each embedded chunk's content, sub-index and vector bytes, without ids, so two stores
/// that gave the same chunks different ids, such as before and after a rebuild, compare equal.
fn chunk_vectors(conn: &Connection) -> Vec<String> {
    collect_strings(
        conn,
        "SELECT json_array(c.content, v.sub_idx, hex(v.embedding)) FROM qa_chunks c JOIN vec_chunks v ON c.id = v.chunk_id ORDER BY c.content, v.sub_idx, hex(v.embedding)",
    )
}

/// Each embedded chunk with its id, generation, vector rowid and vector bytes, in id order.
/// A kept vector compares equal only if neither its chunk nor its vector row was written again.
fn vector_rows(conn: &Connection) -> Vec<String> {
    collect_strings(
        conn,
        "SELECT json_array(c.id, c.generation, c.content, v.rowid, hex(v.embedding)) FROM qa_chunks c JOIN vec_chunks v ON c.id = v.chunk_id ORDER BY c.id",
    )
}

/// The content of each embedded chunk, in id order.
fn embedded_contents(conn: &Connection) -> Vec<String> {
    collect_strings(
        conn,
        "SELECT c.content FROM qa_chunks c JOIN vec_chunks v ON c.id = v.chunk_id ORDER BY c.id",
    )
}

#[test]
fn a_session_indexed_after_enumeration_survives_orphan_cleanup() {
    use rusqlite::{Error as SqliteError, ErrorCode};
    use std::cell::{Cell, RefCell};
    use std::rc::Rc;

    for embed_capable in [true, false] {
        let (db_dir, mut conn) = setup_test_db();
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().to_path_buf();
        let absent = root.join("absent");
        let opts = IndexOptions {
            force: false,
            claude_dir: &root,
            codex_dir: &absent,
            codex_archived_dir: None,
        };
        let writer = open_db(&db_dir.path().join("test.db")).unwrap();
        writer.busy_timeout(Duration::ZERO).unwrap();
        let writer = RefCell::new(writer);
        let writer_root = root.clone();
        let saved = Rc::new(RefCell::new(None));
        let writer_saved = Rc::clone(&saved);
        let index_new = Rc::new(move || -> Result<()> {
            let mut writer = writer.borrow_mut();
            let absent = writer_root.join("absent");
            let opts = IndexOptions {
                force: false,
                claude_dir: &writer_root,
                codex_dir: &absent,
                codex_archived_dir: None,
            };
            index_from_dirs(&mut writer, &opts, true)?;
            index_chunks(&mut writer)?;
            assert_eq!(
                embed_recent_chunks(&mut writer, &MockEmbedder::new(), 100, None)?.embedded,
                1
            );
            *writer_saved.borrow_mut() = Some(chunk_vectors(&writer));
            Ok(())
        });
        let attempted = Rc::new(Cell::new(false));
        let observer_attempted = Rc::clone(&attempted);
        let observer_index = Rc::clone(&index_new);
        let path = root.join("new.jsonl");
        let observer = Observer::with_reporter(move |line| {
            if line.starts_with("index: enumeration: complete") {
                assert!(!observer_attempted.replace(true));
                write_reuse_conversation(
                    &path,
                    &[("user", "new question"), ("assistant", "new answer")],
                );
                if let Err(error) = observer_index() {
                    assert_eq!(
                        error
                            .downcast_ref::<SqliteError>()
                            .and_then(SqliteError::sqlite_error_code),
                        Some(ErrorCode::DatabaseBusy),
                        "only writer contention permits a retry: {error:#}"
                    );
                }
            }
        });

        index_from_dirs_observed(&mut conn, &opts, embed_capable, &observer).unwrap();
        assert!(attempted.get());
        if saved.borrow().is_none() {
            index_new().unwrap();
        }
        assert_eq!(
            collect_strings(&conn, "SELECT session_id FROM sessions"),
            ["new"]
        );
        assert_eq!(
            collect_strings(&conn, "SELECT text FROM messages ORDER BY rowid"),
            ["new question", "new answer"]
        );
        assert_eq!(chunk_vectors(&conn).len(), 1);
        assert_eq!(chunk_vectors(&conn), *saved.borrow().as_ref().unwrap());
    }
}

const REUSE_CONVERSATION: [(&str, &str); 6] = [
    ("user", "duplicate"),
    ("assistant", "same answer"),
    ("user", "duplicate"),
    ("assistant", "same answer"),
    ("user", "tail question"),
    ("assistant", "tail answer"),
];

/// A Claude root holding `reuse.jsonl` with `REUSE_CONVERSATION`, and an absent Codex root.
fn reuse_log() -> OneLog {
    let log = OneLog::new("claude", "reuse.jsonl");
    write_reuse_conversation(&log.file, &REUSE_CONVERSATION);
    log
}

/// Indexes, chunks and embeds `REUSE_CONVERSATION`: three chunks, two of them with the same
/// text. Then stores a message of another session, so a re-read log gets rowids its old
/// messages never had.
fn embedded_reuse_conversation(conn: &mut Connection) -> OneLog {
    let log = reuse_log();
    assert_eq!(index_from_dirs(conn, &log.opts(), true).unwrap().indexed, 1);
    assert_eq!(index_chunks(conn).unwrap().chunks_created, 3);
    assert_eq!(
        embed_recent_chunks(conn, &MockEmbedder::new(), 100, None)
            .unwrap()
            .embedded,
        3
    );
    store_an_unrelated_message(conn);
    log
}

fn store_an_unrelated_message(conn: &Connection) {
    conn.execute(
        "INSERT INTO messages (session_id, role, text) VALUES ('other', 'user', 'unrelated')",
        [],
    )
    .unwrap();
}

/// Rewrites the log with `messages` and an mtime 10 seconds later, so the next index reads it.
fn rewrite_reuse_conversation(path: &Path, messages: &[(&str, &str)]) {
    rewrite_with_mtime(
        path,
        &reuse_conversation(messages),
        SystemTime::now() + Duration::from_secs(10),
    );
}

/// Reindexes and returns the --json counts that report chunk reuse.
fn reuse_counts_of_a_rerun(conn: &mut Connection, opts: &IndexOptions) -> serde_json::Value {
    let counts = counts_of_a_rerun(conn, opts);
    serde_json::json!({
        "files_updated_committed": counts["files_updated_committed"],
        "chunks_matched_committed": counts["chunks_matched_committed"],
        "embeddings_reused_committed": counts["embeddings_reused_committed"],
        "chunks_reconciled_created_committed": counts["chunks_reconciled_created_committed"],
        "chunks_removed_committed": counts["chunks_removed_committed"],
    })
}

/// Each stored chunk with the rowid range of the messages it covers, in conversation order.
fn chunk_ranges(conn: &Connection) -> Vec<String> {
    collect_strings(
        conn,
        "SELECT json_array(content, src_rowid_lo, src_rowid_hi) FROM qa_chunks ORDER BY src_rowid_lo",
    )
}

/// Embeds every pending chunk and returns the --json counts of the chunks the model was given
/// and of the vectors the embed saved.
fn inference_counts_of_an_embed(conn: &mut Connection) -> serde_json::Value {
    let observer = Observer::default();
    embed_pending_observed(
        conn,
        &MockEmbedder::new(),
        None,
        &EmbedOptions::default(),
        &observer,
    )
    .unwrap();
    let counts = observer.snapshot()["counts"].clone();
    serde_json::json!({
        "inference_chunks": counts["inference_chunks"],
        "chunks_saved": counts["chunks_saved"],
    })
}

/// Rebuilds the index from the log and asserts it stores the same chunks and vectors.
fn assert_vectors_match_a_rebuild(conn: &mut Connection, log: &OneLog) {
    let kept = chunk_vectors(conn);
    index_from_dirs(conn, &log.rebuild_opts(), true).unwrap();
    index_chunks(conn).unwrap();
    embed_recent_chunks(conn, &MockEmbedder::new(), 100, None).unwrap();
    assert_eq!(
        chunk_vectors(conn),
        kept,
        "a fresh rebuild stores the same chunks and vectors"
    );
}

#[test]
fn appending_a_turn_reuses_every_stored_vector_and_infers_only_the_new_chunk() {
    let (_db_dir, mut conn) = setup_test_db();
    let log = embedded_reuse_conversation(&mut conn);
    let before = vector_rows(&conn);
    rewrite_reuse_conversation(
        &log.file,
        &[
            ("user", "duplicate"),
            ("assistant", "same answer"),
            ("user", "duplicate"),
            ("assistant", "same answer"),
            ("user", "tail question"),
            ("assistant", "tail answer"),
            ("user", "appended question"),
            ("assistant", "appended answer"),
        ],
    );

    assert_eq!(
        reuse_counts_of_a_rerun(&mut conn, &log.opts()),
        serde_json::json!({
            "files_updated_committed": 1,
            "chunks_matched_committed": 3,
            "embeddings_reused_committed": 3,
            "chunks_reconciled_created_committed": 1,
            "chunks_removed_committed": 0,
        })
    );
    index_chunks(&mut conn).unwrap();

    assert_eq!(vector_rows(&conn), before);
    assert_eq!(
        chunk_ranges(&conn),
        [
            r#"["duplicate\nsame answer",8,9]"#,
            r#"["duplicate\nsame answer",10,11]"#,
            r#"["tail question\ntail answer",12,13]"#,
            r#"["appended question\nappended answer",14,15]"#,
        ]
    );
    assert_eq!(
        inference_counts_of_an_embed(&mut conn),
        serde_json::json!({"inference_chunks": 1, "chunks_saved": 1})
    );
    assert_vectors_match_a_rebuild(&mut conn, &log);
}

#[test]
fn continuing_the_last_answer_reuses_the_unchanged_chunks_and_infers_only_the_extended_one() {
    let (_db_dir, mut conn) = setup_test_db();
    let log = embedded_reuse_conversation(&mut conn);
    let before = vector_rows(&conn);
    rewrite_reuse_conversation(
        &log.file,
        &[
            ("user", "duplicate"),
            ("assistant", "same answer"),
            ("user", "duplicate"),
            ("assistant", "same answer"),
            ("user", "tail question"),
            ("assistant", "tail answer"),
            ("assistant", "continuation needle"),
        ],
    );

    assert_eq!(
        reuse_counts_of_a_rerun(&mut conn, &log.opts()),
        serde_json::json!({
            "files_updated_committed": 1,
            "chunks_matched_committed": 2,
            "embeddings_reused_committed": 2,
            "chunks_reconciled_created_committed": 1,
            "chunks_removed_committed": 1,
        })
    );
    index_chunks(&mut conn).unwrap();

    assert_eq!(vector_rows(&conn), before[..2]);
    assert_eq!(
        chunk_ranges(&conn),
        [
            r#"["duplicate\nsame answer",8,9]"#,
            r#"["duplicate\nsame answer",10,11]"#,
            r#"["tail question\ntail answer\ncontinuation needle",12,14]"#,
        ]
    );
    assert_eq!(
        inference_counts_of_an_embed(&mut conn),
        serde_json::json!({"inference_chunks": 1, "chunks_saved": 1})
    );
    let results = search(&conn, "continuation", &SearchOptions::default()).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(
        results[0].excerpt,
        "tail question\ntail answer\ncontinuation needle"
    );
    let extended: i64 = conn
        .query_row(
            "SELECT id FROM qa_chunks WHERE content = 'tail question\ntail answer\ncontinuation needle'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(results[0].chunk_id, Some(extended));
    assert_vectors_match_a_rebuild(&mut conn, &log);
}

#[test]
fn touching_a_log_without_changing_it_reuses_every_vector_and_infers_nothing() {
    let (_db_dir, mut conn) = setup_test_db();
    let log = embedded_reuse_conversation(&mut conn);
    let before = vector_rows(&conn);
    rewrite_reuse_conversation(&log.file, &REUSE_CONVERSATION);

    assert_eq!(
        reuse_counts_of_a_rerun(&mut conn, &log.opts()),
        serde_json::json!({
            "files_updated_committed": 1,
            "chunks_matched_committed": 3,
            "embeddings_reused_committed": 3,
            "chunks_reconciled_created_committed": 0,
            "chunks_removed_committed": 0,
        })
    );
    index_chunks(&mut conn).unwrap();

    assert_eq!(vector_rows(&conn), before);
    assert_eq!(
        chunk_ranges(&conn),
        [
            r#"["duplicate\nsame answer",8,9]"#,
            r#"["duplicate\nsame answer",10,11]"#,
            r#"["tail question\ntail answer",12,13]"#,
        ]
    );
    assert_eq!(
        inference_counts_of_an_embed(&mut conn),
        serde_json::json!({"inference_chunks": 0, "chunks_saved": 0})
    );
    assert_vectors_match_a_rebuild(&mut conn, &log);
}

#[test]
fn truncating_a_log_keeps_the_vector_of_the_remaining_chunk_and_infers_nothing() {
    let (_db_dir, mut conn) = setup_test_db();
    let log = embedded_reuse_conversation(&mut conn);
    let before = vector_rows(&conn);
    rewrite_reuse_conversation(
        &log.file,
        &[("user", "duplicate"), ("assistant", "same answer")],
    );

    assert_eq!(
        reuse_counts_of_a_rerun(&mut conn, &log.opts()),
        serde_json::json!({
            "files_updated_committed": 1,
            "chunks_matched_committed": 1,
            "embeddings_reused_committed": 1,
            "chunks_reconciled_created_committed": 0,
            "chunks_removed_committed": 2,
        })
    );
    index_chunks(&mut conn).unwrap();

    assert_eq!(vector_rows(&conn), before[..1]);
    assert_eq!(chunk_ranges(&conn), [r#"["duplicate\nsame answer",8,9]"#]);
    assert_eq!(
        inference_counts_of_an_embed(&mut conn),
        serde_json::json!({"inference_chunks": 0, "chunks_saved": 0})
    );
    assert_vectors_match_a_rebuild(&mut conn, &log);
}

#[test]
fn truncating_to_a_duplicated_turn_keeps_the_copy_that_has_a_vector() {
    let (_db_dir, mut conn) = setup_test_db();
    let log = embedded_reuse_conversation(&mut conn);
    conn.execute(
        "DELETE FROM vec_chunks WHERE chunk_id = (SELECT min(id) FROM qa_chunks)",
        [],
    )
    .unwrap();
    let before = vector_rows(&conn);
    rewrite_reuse_conversation(
        &log.file,
        &[("user", "duplicate"), ("assistant", "same answer")],
    );

    assert_eq!(
        reuse_counts_of_a_rerun(&mut conn, &log.opts()),
        serde_json::json!({
            "files_updated_committed": 1,
            "chunks_matched_committed": 1,
            "embeddings_reused_committed": 1,
            "chunks_reconciled_created_committed": 0,
            "chunks_removed_committed": 2,
        })
    );
    index_chunks(&mut conn).unwrap();

    assert_eq!(vector_rows(&conn), before[..1]);
    assert_eq!(chunk_ranges(&conn), [r#"["duplicate\nsame answer",8,9]"#]);
    assert_eq!(
        inference_counts_of_an_embed(&mut conn),
        serde_json::json!({"inference_chunks": 0, "chunks_saved": 0})
    );
    assert_vectors_match_a_rebuild(&mut conn, &log);
}

#[test]
fn replacing_the_opening_turns_keeps_the_vector_of_the_surviving_turn_and_infers_only_the_new_chunk()
 {
    let (_db_dir, mut conn) = setup_test_db();
    let log = embedded_reuse_conversation(&mut conn);
    let before = vector_rows(&conn);
    rewrite_reuse_conversation(
        &log.file,
        &[
            ("user", "tail question"),
            ("assistant", "tail answer"),
            ("user", "replacement needle"),
        ],
    );

    assert_eq!(
        reuse_counts_of_a_rerun(&mut conn, &log.opts()),
        serde_json::json!({
            "files_updated_committed": 1,
            "chunks_matched_committed": 1,
            "embeddings_reused_committed": 1,
            "chunks_reconciled_created_committed": 1,
            "chunks_removed_committed": 2,
        })
    );
    index_chunks(&mut conn).unwrap();

    assert_eq!(vector_rows(&conn), before[2..]);
    assert_eq!(
        chunk_ranges(&conn),
        [
            r#"["tail question\ntail answer",8,9]"#,
            r#"["replacement needle",10,10]"#,
        ]
    );
    assert_eq!(
        inference_counts_of_an_embed(&mut conn),
        serde_json::json!({"inference_chunks": 1, "chunks_saved": 1})
    );
    assert_vectors_match_a_rebuild(&mut conn, &log);
}

#[test]
fn a_session_chunked_under_other_rules_reuses_no_vector_and_infers_every_chunk() {
    let (_db_dir, mut conn) = setup_test_db();
    let log = embedded_reuse_conversation(&mut conn);
    assert_eq!(vector_rows(&conn).len(), 3);
    conn.execute("UPDATE sessions SET chunks_indexed = 99", [])
        .unwrap();

    assert_eq!(
        reuse_counts_of_a_rerun(&mut conn, &log.opts()),
        serde_json::json!({
            "files_updated_committed": 1,
            "chunks_matched_committed": 0,
            "embeddings_reused_committed": 0,
            "chunks_reconciled_created_committed": 0,
            "chunks_removed_committed": 0,
        })
    );
    index_chunks(&mut conn).unwrap();

    assert!(vector_rows(&conn).is_empty());
    assert_eq!(
        chunk_ranges(&conn),
        [
            r#"["duplicate\nsame answer",8,9]"#,
            r#"["duplicate\nsame answer",10,11]"#,
            r#"["tail question\ntail answer",12,13]"#,
        ]
    );
    assert_eq!(
        inference_counts_of_an_embed(&mut conn),
        serde_json::json!({"inference_chunks": 3, "chunks_saved": 3})
    );
    assert_vectors_match_a_rebuild(&mut conn, &log);
}

/// Replaces the stored session with the state an older parser left: the chunks and vectors
/// are gone, every message reads `obsolete parser output`, and `chunks_indexed` is `marker`.
/// With the root missing, the next index keeps the session and the chunk pass chunks and
/// embeds the stored text. Once the root is back, the log must be read again and every
/// chunk inferred again.
fn assert_session_chunked_from_stored_text_is_reread_once_its_root_returns(marker: Option<i64>) {
    let (_db_dir, mut conn) = setup_test_db();
    let log = embedded_reuse_conversation(&mut conn);
    conn.execute_batch(
        "DELETE FROM vec_chunks; DELETE FROM qa_chunks; UPDATE messages SET text = 'obsolete parser output' WHERE session_id = 'reuse';",
    )
    .unwrap();
    conn.execute("UPDATE sessions SET chunks_indexed = ?1", [marker])
        .unwrap();
    let mtime = fs::metadata(&log.file).unwrap().modified().unwrap();
    let offline = log.tmp.path().join("offline");
    fs::rename(&log.claude, &offline).unwrap();
    let missing = index_from_dirs(&mut conn, &log.opts(), true).unwrap();
    assert_eq!((missing.indexed, missing.total_sessions), (0, 1));
    assert_eq!(
        skipped_roots(&missing),
        [(Source::Claude, SkippedReason::MissingRoot, 1)]
    );
    assert_eq!(index_chunks(&mut conn).unwrap().chunks_created, 3);
    assert_eq!(
        index_chunks(&mut conn).unwrap().chunks_created,
        0,
        "chunking the stored text completes the session"
    );
    assert_eq!(
        embed_recent_chunks(&mut conn, &MockEmbedder::new(), 100, None)
            .unwrap()
            .embedded,
        3
    );
    fs::rename(&offline, &log.claude).unwrap();
    assert_eq!(
        fs::metadata(&log.file).unwrap().modified().unwrap(),
        mtime,
        "moving the root back leaves the log's mtime unchanged"
    );
    assert_eq!(
        embedded_contents(&conn),
        ["obsolete parser output\nobsolete parser output"; 3]
    );

    assert_eq!(
        reuse_counts_of_a_rerun(&mut conn, &log.opts()),
        serde_json::json!({
            "files_updated_committed": 1,
            "chunks_matched_committed": 0,
            "embeddings_reused_committed": 0,
            "chunks_reconciled_created_committed": 0,
            "chunks_removed_committed": 0,
        })
    );
    index_chunks(&mut conn).unwrap();

    assert!(vector_rows(&conn).is_empty());
    assert_eq!(
        chunk_ranges(&conn),
        [
            r#"["duplicate\nsame answer",8,9]"#,
            r#"["duplicate\nsame answer",10,11]"#,
            r#"["tail question\ntail answer",12,13]"#,
        ]
    );
    assert_eq!(
        inference_counts_of_an_embed(&mut conn),
        serde_json::json!({"inference_chunks": 3, "chunks_saved": 3})
    );
    assert_vectors_match_a_rebuild(&mut conn, &log);
}

#[test]
fn a_session_of_unknown_rules_chunked_while_its_root_was_missing_is_reread_and_reinferred_once_the_root_returns()
 {
    assert_session_chunked_from_stored_text_is_reread_once_its_root_returns(None);
}

#[test]
fn a_session_parsed_under_old_rules_and_chunked_while_its_root_was_missing_is_reread_and_reinferred_once_the_root_returns()
 {
    assert_session_chunked_from_stored_text_is_reread_once_its_root_returns(Some(-99));
}

#[test]
fn a_session_parsed_under_current_rules_and_chunked_while_its_root_was_missing_reuses_every_vector()
{
    let (db_dir, mut conn) = setup_test_db();
    let log = reuse_log();
    assert_eq!(
        index_from_dirs(&mut conn, &log.opts(), true)
            .unwrap()
            .indexed,
        1
    );
    let offline = log.tmp.path().join("offline");
    fs::rename(&log.claude, &offline).unwrap();
    let mut resumed = open_db(&db_dir.path().join("test.db")).unwrap();
    assert_eq!(
        index_from_dirs(&mut resumed, &log.opts(), true)
            .unwrap()
            .indexed,
        0
    );
    assert_eq!(index_chunks(&mut resumed).unwrap().chunks_created, 3);
    fs::rename(&offline, &log.claude).unwrap();
    assert_eq!(
        index_from_dirs(&mut conn, &log.opts(), true)
            .unwrap()
            .indexed,
        0,
        "the log was already parsed under the current rules"
    );
    assert_eq!(index_chunks(&mut conn).unwrap().chunks_created, 0);
    assert_eq!(
        embed_recent_chunks(&mut conn, &MockEmbedder::new(), 100, None)
            .unwrap()
            .embedded,
        3
    );
    store_an_unrelated_message(&conn);
    let before = vector_rows(&conn);
    rewrite_reuse_conversation(&log.file, &REUSE_CONVERSATION);

    assert_eq!(
        reuse_counts_of_a_rerun(&mut conn, &log.opts()),
        serde_json::json!({
            "files_updated_committed": 1,
            "chunks_matched_committed": 3,
            "embeddings_reused_committed": 3,
            "chunks_reconciled_created_committed": 0,
            "chunks_removed_committed": 0,
        })
    );
    index_chunks(&mut conn).unwrap();

    assert_eq!(vector_rows(&conn), before);
    assert_eq!(
        chunk_ranges(&conn),
        [
            r#"["duplicate\nsame answer",8,9]"#,
            r#"["duplicate\nsame answer",10,11]"#,
            r#"["tail question\ntail answer",12,13]"#,
        ]
    );
    assert_eq!(
        inference_counts_of_an_embed(&mut conn),
        serde_json::json!({"inference_chunks": 0, "chunks_saved": 0})
    );
    assert_vectors_match_a_rebuild(&mut conn, &log);
}

#[test]
fn rebuilding_reuses_no_vector_and_infers_every_chunk_again() {
    let (_db_dir, mut conn) = setup_test_db();
    let log = embedded_reuse_conversation(&mut conn);
    assert_eq!(vector_rows(&conn).len(), 3);

    assert_eq!(
        reuse_counts_of_a_rerun(&mut conn, &log.rebuild_opts()),
        serde_json::json!({
            "files_updated_committed": 1,
            "chunks_matched_committed": 0,
            "embeddings_reused_committed": 0,
            "chunks_reconciled_created_committed": 0,
            "chunks_removed_committed": 0,
        })
    );
    index_chunks(&mut conn).unwrap();

    assert!(vector_rows(&conn).is_empty());
    assert_eq!(
        chunk_ranges(&conn),
        [
            r#"["duplicate\nsame answer",8,9]"#,
            r#"["duplicate\nsame answer",10,11]"#,
            r#"["tail question\ntail answer",12,13]"#,
        ]
    );
    assert_eq!(
        inference_counts_of_an_embed(&mut conn),
        serde_json::json!({"inference_chunks": 3, "chunks_saved": 3})
    );
    assert_vectors_match_a_rebuild(&mut conn, &log);
}

#[test]
fn a_second_chunk_pass_on_the_same_connection_chunks_a_rebuilt_session_whose_messages_keep_their_rowids()
 {
    let (_db_dir, mut conn) = setup_test_db();
    let log = reuse_log();
    assert_eq!(
        index_from_dirs(&mut conn, &log.opts(), true)
            .unwrap()
            .indexed,
        1
    );
    assert_eq!(index_chunks(&mut conn).unwrap().chunks_created, 3);
    assert_eq!(
        index_from_dirs(&mut conn, &log.rebuild_opts(), true)
            .unwrap()
            .indexed,
        1
    );
    let observer = Observer::default();

    index_chunks_observed(&mut conn, &observer).unwrap();

    assert_eq!(
        observer.snapshot()["counts"],
        serde_json::json!({
            "chunks_created_committed": 3,
            "sessions_chunk_pending": 1,
            "sessions_chunk_remaining": 0,
            "sessions_chunked_committed": 1,
            "sessions_chunked_empty": 0,
        })
    );
    assert_eq!(
        chunk_ranges(&conn),
        [
            r#"["duplicate\nsame answer",1,2]"#,
            r#"["duplicate\nsame answer",3,4]"#,
            r#"["tail question\ntail answer",5,6]"#,
        ],
        "the rebuild stores the messages under rowids 1 to 6 again, the rows the first pass staged"
    );
}

#[test]
fn reconciliation_rollback_and_failed_inference_preserve_reusable_work_for_resume() {
    // embed_chunks and pending_chunks are test-only embedder wrappers. No entry point lets a
    // test keep a pending list read before a reindex and save its results after it.
    use crate::embedder::{embed_chunks, pending_chunks};
    let (db_dir, mut conn) = setup_test_db();
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("reuse.jsonl");
    let absent = tmp.path().join("absent");
    let opts = IndexOptions {
        force: false,
        claude_dir: tmp.path(),
        codex_dir: &absent,
        codex_archived_dir: None,
    };
    write_reuse_conversation(
        &path,
        &[
            ("user", "stable"),
            ("assistant", "answer"),
            ("user", "tail"),
        ],
    );
    index_from_dirs(&mut conn, &opts, true).unwrap();
    index_chunks(&mut conn).unwrap();
    let old_pending = pending_chunks(&conn, 100).unwrap();
    let stable = conn
        .query_row(
            "SELECT id FROM qa_chunks WHERE content = 'stable\nanswer'",
            [],
            |r| r.get::<_, i64>(0),
        )
        .unwrap();
    let bytes = f32_as_bytes(&MockEmbedder::deterministic_vector("stable\nanswer")).to_vec();
    conn.execute(
        "INSERT INTO vec_chunks (chunk_id, sub_idx, embedding) VALUES (?1, 0, ?2)",
        rusqlite::params![stable, bytes],
    )
    .unwrap();
    let before = chunk_vectors(&conn);
    write_reuse_conversation(
        &path,
        &[
            ("user", "stable"),
            ("assistant", "answer"),
            ("user", "tail"),
            ("assistant", "new tail"),
        ],
    );
    conn.execute_batch("CREATE TEMP TRIGGER fail_reconcile BEFORE UPDATE OF chunks_indexed ON sessions BEGIN SELECT RAISE(ABORT, 'reconciliation failed'); END;").unwrap();
    assert_eq!(
        index_from_dirs(&mut conn, &opts, true)
            .err()
            .unwrap()
            .to_string(),
        "reconciliation failed"
    );
    assert_eq!(chunk_vectors(&conn), before);
    assert_eq!(
        collect_strings(&conn, "SELECT text FROM messages ORDER BY rowid"),
        ["stable", "answer", "tail"]
    );
    conn.execute_batch("DROP TRIGGER fail_reconcile;").unwrap();
    let mut writer = open_db(&db_dir.path().join("test.db")).unwrap();
    assert_eq!(
        index_from_dirs(&mut writer, &opts, true).unwrap().indexed,
        1
    );
    assert_eq!(
        index_chunks(&mut writer).unwrap().chunks_created,
        0,
        "the reconciliation already chunked the rewritten log"
    );
    assert_eq!(chunk_vectors(&conn), before);
    let stale = embed_chunks(
        &mut conn,
        &MockEmbedder::new(),
        &old_pending,
        None,
        &EmbedOptions::default(),
    )
    .unwrap();
    assert_eq!(stale.embedded, 1, "only the unchanged input can save");
    assert_eq!(chunk_vectors(&conn), before);
    assert_eq!(
        embed_recent_chunks(&mut conn, &MockEmbedder::failing_after(0), 100, None)
            .unwrap()
            .failed_count,
        1
    );
    assert_eq!(chunk_vectors(&conn), before);
    drop(conn);
    drop(writer);
    let mut conn = open_db(&db_dir.path().join("test.db")).unwrap();
    assert_eq!(index_from_dirs(&mut conn, &opts, true).unwrap().indexed, 0);
    assert_eq!(index_chunks(&mut conn).unwrap().chunks_created, 0);
    assert_eq!(
        embed_recent_chunks(&mut conn, &MockEmbedder::new(), 100, None)
            .unwrap()
            .embedded,
        1
    );
    assert_eq!(
        embedded_contents(&conn),
        ["stable\nanswer", "tail\nnew tail"]
    );
}

const ROLLOUT_NAME: &str = "rollout-2026-04-27T00-00-00-01900000-0000-7000-8000-000000000001.jsonl";
const ROLLOUT_TURNS: &str = concat!(
    r#"{"timestamp":"2026-04-27T00:00:01Z","type":"response_item","payload":{"role":"user","content":[{"type":"input_text","text":"which flag disables the cache"}]}}"#,
    "\n",
    r#"{"timestamp":"2026-04-27T00:00:02Z","type":"response_item","payload":{"role":"assistant","content":[{"type":"output_text","text":"pass --no-cache"}]}}"#,
    "\n",
);

fn compress_log(plain: &Path) -> PathBuf {
    let compressed = plain.with_file_name(format!(
        "{}.zst",
        plain.file_name().unwrap().to_str().unwrap()
    ));
    let body = fs::read(plain).unwrap();
    fs::write(&compressed, zstd::encode_all(body.as_slice(), 0).unwrap()).unwrap();
    compressed
}

fn session_row(conn: &Connection) -> Vec<String> {
    collect_strings(
        conn,
        "SELECT json_array(session_id, file_path, slug, project, timestamp) FROM sessions",
    )
}

const ROLLOUT_META: &str = r#"{"timestamp":"2026-04-27T00:00:00Z","type":"session_meta","payload":{"id":"01900000-0000-7000-8000-000000000001","cwd":"/proj"}}"#;

/// The stored session's id, slug, project and start time. The file path is left out
/// because compressing or archiving the log changes it.
fn session_identity(conn: &Connection) -> Vec<String> {
    collect_strings(
        conn,
        "SELECT json_array(session_id, slug, project, timestamp) FROM sessions",
    )
}

/// Indexes and embeds `body` as a Codex rollout, then compresses the log and removes the
/// plain copy. The session must keep its identity and embeddings under the compressed path.
fn assert_compressing_keeps_the_session_and_its_embeddings(body: &str) {
    let (_db, mut conn) = setup_test_db();
    let tmp = TempDir::new().unwrap();
    let claude_dir = tmp.path().join("claude_projects");
    let codex_dir = tmp.path().join("codex_sessions");
    let day_dir = codex_dir.join("2026/04/27");
    fs::create_dir_all(&day_dir).unwrap();
    let plain = day_dir.join(ROLLOUT_NAME);
    fs::write(&plain, body).unwrap();
    let opts = IndexOptions {
        force: false,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
        codex_archived_dir: None,
    };
    index_from_dirs(&mut conn, &opts, true).unwrap();
    index_chunks(&mut conn).unwrap();
    assert_eq!(
        embed_recent_chunks(&mut conn, &MockEmbedder::new(), 100, None)
            .unwrap()
            .embedded,
        1
    );
    let identity_before = session_identity(&conn);
    let vectors_before = chunk_vectors(&conn);

    let compressed = compress_log(&plain);
    fs::remove_file(&plain).unwrap();
    let stats = index_from_dirs(&mut conn, &opts, true).unwrap();

    assert_eq!(stats.total_sessions, 1);
    assert_eq!(
        collect_strings(&conn, "SELECT file_path FROM sessions"),
        [compressed.to_str().unwrap()]
    );
    assert_eq!(session_identity(&conn), identity_before);
    assert_eq!(chunk_vectors(&conn), vectors_before);
    assert!(stats.parse_diagnostics.is_empty());
}

#[test]
fn compressing_a_codex_log_keeps_the_indexed_session_and_its_embeddings() {
    assert_compressing_keeps_the_session_and_its_embeddings(&format!(
        "{ROLLOUT_META}\n{ROLLOUT_TURNS}"
    ));
}

#[test]
fn compressing_a_codex_log_that_has_no_session_meta_keeps_the_indexed_session_and_its_embeddings() {
    assert_compressing_keeps_the_session_and_its_embeddings(ROLLOUT_TURNS);
}

#[test]
fn plain_codex_log_is_indexed_once_when_its_compressed_copy_sits_beside_it() {
    let (_db, mut conn) = setup_test_db();
    let tmp = TempDir::new().unwrap();
    let claude_dir = tmp.path().join("claude_projects");
    let codex_dir = tmp.path().join("codex_sessions");
    let day_dir = codex_dir.join("2026/04/27");
    fs::create_dir_all(&day_dir).unwrap();
    let plain = day_dir.join(ROLLOUT_NAME);
    fs::write(&plain, ROLLOUT_TURNS).unwrap();
    compress_log(&plain);
    let opts = IndexOptions {
        force: false,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
        codex_archived_dir: None,
    };

    let first = index_from_dirs(&mut conn, &opts, true).unwrap();
    index_chunks(&mut conn).unwrap();
    let second = index_from_dirs(&mut conn, &opts, true).unwrap();

    assert_eq!((first.indexed, first.total_sessions), (1, 1));
    assert_eq!(
        second.indexed, 0,
        "the two copies must not replace each other"
    );
    assert_eq!(
        collect_strings(&conn, "SELECT file_path FROM sessions"),
        [plain.to_str().unwrap()]
    );
}

#[test]
fn compressed_codex_log_keeps_its_session_while_an_empty_plain_copy_sits_beside_it() {
    let (_db, mut conn) = setup_test_db();
    let tmp = TempDir::new().unwrap();
    let claude_dir = tmp.path().join("claude_projects");
    let codex_dir = tmp.path().join("codex_sessions");
    let day_dir = codex_dir.join("2026/04/27");
    fs::create_dir_all(&day_dir).unwrap();
    let plain = day_dir.join(ROLLOUT_NAME);
    fs::write(&plain, ROLLOUT_TURNS).unwrap();
    compress_log(&plain);
    fs::remove_file(&plain).unwrap();
    let opts = IndexOptions {
        force: false,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
        codex_archived_dir: None,
    };
    index_from_dirs(&mut conn, &opts, true).unwrap();
    index_chunks(&mut conn).unwrap();
    embed_recent_chunks(&mut conn, &MockEmbedder::new(), 100, None).unwrap();
    let row_before = session_row(&conn);
    let vectors_before = chunk_vectors(&conn);
    assert!(!vectors_before.is_empty());

    fs::write(&plain, "").unwrap();
    index_from_dirs(&mut conn, &opts, true).unwrap();

    assert_eq!(session_row(&conn), row_before);
    assert_eq!(chunk_vectors(&conn), vectors_before);
}

#[test]
fn compressed_logs_are_indexed_only_from_codex_roots() {
    let (_db, mut conn) = setup_test_db();
    let tmp = TempDir::new().unwrap();
    let claude_dir = tmp.path().join("claude_projects");
    let codex_dir = tmp.path().join("codex_sessions");
    fs::create_dir_all(&claude_dir).unwrap();
    fs::create_dir_all(&codex_dir).unwrap();
    // If collected, the Claude `a.jsonl.zst` and both `c.jsonl.gz` logs would be indexed as
    // sessions. Both `b.zst` files would be read as plain text without decompression and
    // would show up in parse_diagnostics.
    fs::write(
        claude_dir.join("a.jsonl.zst"),
        zstd::encode_all(FRESHNESS_QUESTION.as_bytes(), 0).unwrap(),
    )
    .unwrap();
    fs::write(
        claude_dir.join("b.zst"),
        zstd::encode_all(FRESHNESS_QUESTION.as_bytes(), 0).unwrap(),
    )
    .unwrap();
    fs::write(claude_dir.join("c.jsonl.gz"), FRESHNESS_QUESTION).unwrap();
    let codex_log = codex_dir
        .join("rollout-2026-04-27T00-00-00-01900000-0000-7000-8000-000000000002.jsonl.zst");
    fs::write(
        &codex_log,
        zstd::encode_all(rollout_body(2, "compressed").as_bytes(), 0).unwrap(),
    )
    .unwrap();
    fs::write(
        codex_dir.join("b.zst"),
        zstd::encode_all(rollout_body(3, "bare zst").as_bytes(), 0).unwrap(),
    )
    .unwrap();
    fs::write(codex_dir.join("c.jsonl.gz"), rollout_body(4, "gz")).unwrap();

    let stats = index_from_dirs(
        &mut conn,
        &IndexOptions {
            force: false,
            claude_dir: &claude_dir,
            codex_dir: &codex_dir,
            codex_archived_dir: None,
        },
        true,
    )
    .unwrap();

    assert_eq!(
        collect_strings(&conn, "SELECT session_id FROM sessions"),
        ["01900000-0000-7000-8000-000000000002"]
    );
    assert_eq!(
        collect_strings(&conn, "SELECT file_path FROM sessions"),
        [codex_log.to_str().unwrap()]
    );
    assert!(stats.parse_diagnostics.is_empty());
}

#[test]
fn undecodable_compressed_codex_log_is_reported_as_a_read_error_and_other_sessions_index() {
    let (_db, mut conn) = setup_test_db();
    let tmp = TempDir::new().unwrap();
    let claude_dir = tmp.path().join("claude_projects");
    let codex_dir = tmp.path().join("codex_sessions");
    let day_dir = codex_dir.join("2026/04/27");
    fs::create_dir_all(&day_dir).unwrap();
    fs::write(day_dir.join(ROLLOUT_NAME), ROLLOUT_TURNS).unwrap();
    let corrupt =
        day_dir.join("rollout-2026-04-27T00-00-00-01900000-0000-7000-8000-000000000002.jsonl.zst");
    let mut truncated = zstd::encode_all(ROLLOUT_TURNS.as_bytes(), 0).unwrap();
    truncated.truncate(truncated.len() / 2);
    fs::write(&corrupt, truncated).unwrap();

    let stats = index_from_dirs(
        &mut conn,
        &IndexOptions {
            force: false,
            claude_dir: &claude_dir,
            codex_dir: &codex_dir,
            codex_archived_dir: None,
        },
        true,
    )
    .unwrap();

    assert_eq!((stats.indexed, stats.total_sessions), (1, 1));
    assert_eq!(
        stats
            .parse_diagnostics
            .iter()
            .map(|d| (d.file_path.as_path(), d.source, d.read_error))
            .collect::<Vec<_>>(),
        [(corrupt.as_path(), Source::Codex, true)]
    );
}

fn rollout_name(n: u8) -> String {
    format!("rollout-2026-04-27T00-00-00-01900000-0000-7000-8000-0000000000{n:02}.jsonl")
}

fn rollout_body(n: u8, topic: &str) -> String {
    format!(
        r#"{{"timestamp":"2026-04-27T00:00:00Z","type":"session_meta","payload":{{"id":"01900000-0000-7000-8000-0000000000{n:02}","cwd":"/proj"}}}}
{{"timestamp":"2026-04-27T00:00:01Z","type":"response_item","payload":{{"role":"user","content":[{{"type":"input_text","text":"{topic} question"}}]}}}}
"#
    )
}

struct CodexRoots {
    tmp: TempDir,
    claude: PathBuf,
    sessions: PathBuf,
    day: PathBuf,
    archived: PathBuf,
}

impl CodexRoots {
    fn new() -> Self {
        let tmp = TempDir::new().unwrap();
        let claude = tmp.path().join("claude_projects");
        let sessions = tmp.path().join(".codex/sessions");
        let day = sessions.join("2026/04/27");
        let archived = tmp.path().join(".codex/archived_sessions");
        fs::create_dir_all(&day).unwrap();
        fs::create_dir_all(&archived).unwrap();
        Self {
            tmp,
            claude,
            sessions,
            day,
            archived,
        }
    }

    fn opts(&self) -> IndexOptions<'_> {
        IndexOptions {
            force: false,
            claude_dir: &self.claude,
            codex_dir: &self.sessions,
            codex_archived_dir: Some(&self.archived),
        }
    }
}

fn session_ids(conn: &Connection) -> Vec<String> {
    collect_strings(conn, "SELECT session_id FROM sessions ORDER BY session_id")
}

/// Indexes and embeds `body` as a live Codex rollout, then moves its compressed copy into
/// the archived root. The session must keep its identity and embeddings under the archived path.
fn assert_archiving_keeps_the_session_and_its_embeddings(body: &str) {
    let (_db, mut conn) = setup_test_db();
    let roots = CodexRoots::new();
    let live = roots.day.join(ROLLOUT_NAME);
    fs::write(&live, body).unwrap();
    index_from_dirs(&mut conn, &roots.opts(), true).unwrap();
    index_chunks(&mut conn).unwrap();
    assert_eq!(
        embed_recent_chunks(&mut conn, &MockEmbedder::new(), 100, None)
            .unwrap()
            .embedded,
        1
    );
    let identity_before = session_identity(&conn);
    let vectors_before = chunk_vectors(&conn);

    let compressed = compress_log(&live);
    let archived = roots.archived.join(compressed.file_name().unwrap());
    fs::rename(&compressed, &archived).unwrap();
    fs::remove_file(&live).unwrap();
    let stats = index_from_dirs(&mut conn, &roots.opts(), true).unwrap();

    assert_eq!(stats.total_sessions, 1);
    assert_eq!(
        collect_strings(&conn, "SELECT file_path FROM sessions"),
        [archived.to_str().unwrap()]
    );
    assert_eq!(
        session_identity(&conn),
        identity_before,
        "the slug keeps the session's date"
    );
    assert_eq!(chunk_vectors(&conn), vectors_before);
    assert!(stats.skipped_roots.is_empty());
}

#[test]
fn archiving_a_codex_log_keeps_the_indexed_session_and_its_embeddings() {
    assert_archiving_keeps_the_session_and_its_embeddings(&format!(
        "{ROLLOUT_META}\n{ROLLOUT_TURNS}"
    ));
}

#[test]
fn archiving_a_codex_log_that_has_no_session_meta_keeps_the_indexed_session_and_its_embeddings() {
    assert_archiving_keeps_the_session_and_its_embeddings(ROLLOUT_TURNS);
}

#[test]
fn absent_archived_root_neither_reports_nor_blocks_codex_cleanup() {
    let (_db, mut conn) = setup_test_db();
    let roots = CodexRoots::new();
    fs::remove_dir(&roots.archived).unwrap();
    fs::write(roots.day.join(rollout_name(1)), rollout_body(1, "kept")).unwrap();
    let deleted = roots.day.join(rollout_name(3));
    fs::write(&deleted, rollout_body(3, "deleted")).unwrap();
    index_from_dirs(&mut conn, &roots.opts(), true).unwrap();

    fs::remove_file(&deleted).unwrap();
    let stats = index_from_dirs(&mut conn, &roots.opts(), true).unwrap();

    assert_eq!(session_ids(&conn), ["01900000-0000-7000-8000-000000000001"]);
    assert!(stats.skipped_roots.is_empty());
}

#[test]
fn missing_archived_root_keeps_its_sessions_and_still_removes_sessions_deleted_elsewhere() {
    let (_db, mut conn) = setup_test_db();
    let roots = CodexRoots::new();
    fs::write(roots.day.join(rollout_name(1)), rollout_body(1, "live")).unwrap();
    fs::write(
        roots.archived.join(rollout_name(2)),
        rollout_body(2, "archived"),
    )
    .unwrap();
    let deleted = roots.day.join(rollout_name(3));
    fs::write(&deleted, rollout_body(3, "deleted")).unwrap();
    assert_eq!(
        index_from_dirs(&mut conn, &roots.opts(), true)
            .unwrap()
            .total_sessions,
        3
    );

    fs::remove_dir_all(&roots.archived).unwrap();
    fs::remove_file(&deleted).unwrap();
    let stats = index_from_dirs(&mut conn, &roots.opts(), true).unwrap();

    assert_eq!(
        session_ids(&conn),
        [
            "01900000-0000-7000-8000-000000000001",
            "01900000-0000-7000-8000-000000000002"
        ]
    );
    assert_eq!(
        skipped_roots(&stats),
        [(Source::Codex, SkippedReason::MissingRoot, 1)]
    );
    assert_eq!(
        stats.skipped_roots[0].note(),
        "a codex root is unavailable — preserved 1 existing session(s) under it and reconciled the rest; re-run `recall index` once the root is back so a full scan can reconcile them",
        "the other Codex root was cleaned, so the note must not claim nothing was deleted"
    );
}

#[cfg(unix)]
#[test]
fn missing_archived_root_keeps_its_parse_diagnostics_and_still_clears_those_deleted_elsewhere() {
    let (_db, mut conn) = setup_test_db();
    let roots = CodexRoots::new();
    let mut undecodable = zstd::encode_all(rollout_body(2, "broken").as_bytes(), 0).unwrap();
    undecodable.truncate(undecodable.len() / 2);
    let archived = roots.archived.join(format!("{}.zst", rollout_name(2)));
    let deleted = roots.day.join(format!("{}.zst", rollout_name(3)));
    fs::write(&archived, &undecodable).unwrap();
    fs::write(&deleted, &undecodable).unwrap();
    let diagnosed = |stats: &IndexStats| -> Vec<PathBuf> {
        let mut paths: Vec<_> = stats
            .parse_diagnostics
            .iter()
            .map(|d| d.file_path.clone())
            .collect();
        paths.sort();
        paths
    };
    let first = index_from_dirs(&mut conn, &roots.opts(), true).unwrap();
    let mut both = vec![archived.clone(), deleted.clone()];
    both.sort();
    assert_eq!(diagnosed(&first), both);

    fs::remove_dir_all(&roots.archived).unwrap();
    fs::remove_file(&deleted).unwrap();
    let second = index_from_dirs(&mut conn, &roots.opts(), true).unwrap();

    assert_eq!(diagnosed(&second), [archived]);
}

#[test]
fn unreadable_archived_root_keeps_every_codex_session() {
    use std::os::unix::fs::PermissionsExt;

    let (_db, mut conn) = setup_test_db();
    let roots = CodexRoots::new();
    fs::write(roots.day.join(rollout_name(1)), rollout_body(1, "live")).unwrap();
    fs::write(
        roots.archived.join(rollout_name(2)),
        rollout_body(2, "archived"),
    )
    .unwrap();
    let moved = roots.day.join(rollout_name(3));
    fs::write(&moved, rollout_body(3, "moved")).unwrap();
    assert_eq!(
        index_from_dirs(&mut conn, &roots.opts(), true)
            .unwrap()
            .total_sessions,
        3
    );

    fs::rename(&moved, roots.archived.join(rollout_name(3))).unwrap();
    let permissions = fs::metadata(&roots.archived).unwrap().permissions();
    fs::set_permissions(&roots.archived, fs::Permissions::from_mode(0o000)).unwrap();
    let stats = index_from_dirs(&mut conn, &roots.opts(), true);
    fs::set_permissions(&roots.archived, permissions).unwrap();
    let stats = stats.unwrap();

    assert_eq!(
        session_ids(&conn).len(),
        3,
        "a session moved into an unreadable root survives"
    );
    assert_eq!(
        skipped_roots(&stats),
        [(Source::Codex, SkippedReason::IncompleteEnumeration, 2)]
    );
    drop(roots.tmp);
}

#[test]
fn archived_root_whose_parent_cannot_be_searched_keeps_every_codex_session() {
    use std::os::unix::fs::PermissionsExt;

    let (_db, mut conn) = setup_test_db();
    let roots = CodexRoots::new();
    let locked = roots.tmp.path().join("locked");
    let archived = locked.join("archived_sessions");
    fs::create_dir_all(&archived).unwrap();
    let opts = IndexOptions {
        codex_archived_dir: Some(&archived),
        ..roots.opts()
    };
    fs::write(roots.day.join(rollout_name(1)), rollout_body(1, "live")).unwrap();
    let moved = roots.day.join(rollout_name(3));
    fs::write(&moved, rollout_body(3, "moved")).unwrap();
    index_from_dirs(&mut conn, &opts, true).unwrap();

    fs::rename(&moved, archived.join(rollout_name(3))).unwrap();
    let permissions = fs::metadata(&locked).unwrap().permissions();
    fs::set_permissions(&locked, fs::Permissions::from_mode(0o000)).unwrap();
    let stats = index_from_dirs(&mut conn, &opts, true);
    fs::set_permissions(&locked, permissions).unwrap();
    let stats = stats.unwrap();

    assert_eq!(
        session_ids(&conn).len(),
        2,
        "a session moved into a root that cannot be inspected survives"
    );
    assert_eq!(
        skipped_roots(&stats),
        [(Source::Codex, SkippedReason::IncompleteEnumeration, 1)]
    );
}

fn archived_dir_for(vars: &[(&str, &str)]) -> Option<PathBuf> {
    resolve_codex_archived_dir_with(Path::new("/fake/home"), |key| {
        vars.iter()
            .find(|(k, _)| *k == key)
            .map(|(_, v)| OsString::from(v))
    })
}

#[test]
fn empty_codex_archived_dir_turns_the_archive_off() {
    assert_eq!(archived_dir_for(&[("RECALL_CODEX_ARCHIVED_DIR", "")]), None);
}

#[test]
fn codex_archived_dir_defaults_under_home_unless_only_the_sessions_dir_is_overridden() {
    let resolve = archived_dir_for;

    assert_eq!(
        resolve(&[]),
        Some(PathBuf::from("/fake/home/.codex/archived_sessions"))
    );
    assert_eq!(resolve(&[("RECALL_CODEX_DIR", "/tmp/codex")]), None);
    assert_eq!(
        resolve(&[
            ("RECALL_CODEX_DIR", "/tmp/codex"),
            ("RECALL_CODEX_ARCHIVED_DIR", "/tmp/archived")
        ]),
        Some(PathBuf::from("/tmp/archived"))
    );
    assert_eq!(
        resolve(&[("RECALL_CODEX_ARCHIVED_DIR", "/tmp/archived")]),
        Some(PathBuf::from("/tmp/archived"))
    );
}
