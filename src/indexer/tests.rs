use std::cell::RefCell;
use std::ffi::OsString;
use std::io;
use std::sync::Mutex;
use std::time::{Duration, SystemTime};

use super::*;
use crate::commands::classify::reclassify_sessions;
use crate::db::{open_db, seed_session, seed_session_file, setup_test_db};
use crate::embedder::{EMBED_BATCH_SIZE, MockEmbedder, embed_recent_chunks, f32_as_bytes};
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
    index_chunks(&mut conn, None).unwrap();
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
    index_chunks(&mut conn, None).unwrap();

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

#[test]
fn size_changes_reindex_with_equal_or_submillisecond_mtime() {
    let (_db_dir, mut conn) = setup_test_db();
    let root = TempDir::new().unwrap();
    let file = root.path().join("s.jsonl");
    let absent = root.path().join("absent");
    let opts = IndexOptions {
        force: false,
        claude_dir: root.path(),
        codex_dir: &absent,
        codex_archived_dir: None,
    };
    let t0 = UNIX_EPOCH + Duration::from_secs(1_700_000_000);
    rewrite_with_mtime(&file, FRESHNESS_QUESTION, t0);
    index_from_dirs(&mut conn, &opts, true).unwrap();
    index_chunks(&mut conn, None).unwrap();

    rewrite_with_mtime(&file, "{", t0);
    assert_eq!(index_from_dirs(&mut conn, &opts, true).unwrap().indexed, 0);
    index_chunks(&mut conn, None).unwrap();
    assert_eq!(
        collect_strings(&conn, "SELECT text FROM messages"),
        ["original question"],
        "an unfinished line keeps the stored messages"
    );

    rewrite_with_mtime(
        &file,
        &format!("{FRESHNESS_QUESTION}{FRESHNESS_ANSWER}"),
        t0,
    );
    assert_eq!(index_from_dirs(&mut conn, &opts, true).unwrap().indexed, 1);
    index_chunks(&mut conn, None).unwrap();
    assert_eq!(
        collect_strings(&conn, "SELECT text FROM messages ORDER BY rowid"),
        ["original question", "appended answer"]
    );

    rewrite_with_mtime(&file, FRESHNESS_QUESTION, t0 + Duration::from_micros(500));
    assert_eq!(
        fs::metadata(&file).unwrap().modified().unwrap(),
        t0 + Duration::from_micros(500),
        "precondition: the file system keeps a submillisecond mtime"
    );
    assert_eq!(index_from_dirs(&mut conn, &opts, true).unwrap().indexed, 1);
    index_chunks(&mut conn, None).unwrap();
    assert_eq!(
        collect_strings(&conn, "SELECT text FROM messages ORDER BY rowid"),
        ["original question"]
    );

    let counts = counts_of_a_rerun(&mut conn, &opts);
    assert_eq!(counts["files_updated"], 0);
    assert_eq!(counts["files_unchanged"], 1);
}

#[test]
fn codex_log_whose_size_changes_with_an_equal_or_submillisecond_mtime_is_reindexed() {
    let (_db_dir, mut conn) = setup_test_db();
    let root = TempDir::new().unwrap();
    let file = root.path().join("s.jsonl");
    let absent = root.path().join("absent");
    let opts = IndexOptions {
        force: false,
        claude_dir: &absent,
        codex_dir: root.path(),
        codex_archived_dir: None,
    };
    let t0 = UNIX_EPOCH + Duration::from_secs(1_700_000_000);
    rewrite_with_mtime(&file, CODEX_QUESTION, t0);
    index_from_dirs(&mut conn, &opts, true).unwrap();
    index_chunks(&mut conn, None).unwrap();

    rewrite_with_mtime(&file, "{", t0);
    assert_eq!(index_from_dirs(&mut conn, &opts, true).unwrap().indexed, 0);
    index_chunks(&mut conn, None).unwrap();
    assert_eq!(
        collect_strings(&conn, "SELECT text FROM messages"),
        ["original question"],
        "an unfinished line keeps the stored messages"
    );

    rewrite_with_mtime(&file, &format!("{CODEX_QUESTION}{CODEX_ANSWER}"), t0);
    assert_eq!(index_from_dirs(&mut conn, &opts, true).unwrap().indexed, 1);
    index_chunks(&mut conn, None).unwrap();
    assert_eq!(
        collect_strings(&conn, "SELECT text FROM messages ORDER BY rowid"),
        ["original question", "appended answer"]
    );

    rewrite_with_mtime(&file, CODEX_QUESTION, t0 + Duration::from_micros(500));
    assert_eq!(
        fs::metadata(&file).unwrap().modified().unwrap(),
        t0 + Duration::from_micros(500),
        "precondition: the file system keeps a submillisecond mtime"
    );
    assert_eq!(index_from_dirs(&mut conn, &opts, true).unwrap().indexed, 1);
    index_chunks(&mut conn, None).unwrap();
    assert_eq!(
        collect_strings(&conn, "SELECT text FROM messages ORDER BY rowid"),
        ["original question"]
    );

    let counts = counts_of_a_rerun(&mut conn, &opts);
    assert_eq!(counts["files_updated"], 0);
    assert_eq!(counts["files_unchanged"], 1);
}

#[test]
fn claude_log_emptied_or_reduced_to_metadata_with_an_unchanged_mtime_loses_its_stored_messages() {
    let (_db_dir, mut conn) = setup_test_db();
    let root = TempDir::new().unwrap();
    let file = root.path().join("s.jsonl");
    let absent = root.path().join("absent");
    let opts = IndexOptions {
        force: false,
        claude_dir: root.path(),
        codex_dir: &absent,
        codex_archived_dir: None,
    };
    let t0 = UNIX_EPOCH + Duration::from_secs(1_700_000_000);
    rewrite_with_mtime(&file, FRESHNESS_QUESTION, t0);
    index_from_dirs(&mut conn, &opts, true).unwrap();
    index_chunks(&mut conn, None).unwrap();

    rewrite_with_mtime(&file, CLAUDE_METADATA_ONLY, t0);
    assert_eq!(index_from_dirs(&mut conn, &opts, true).unwrap().indexed, 1);
    index_chunks(&mut conn, None).unwrap();
    assert!(collect_strings(&conn, "SELECT text FROM messages").is_empty());

    rewrite_with_mtime(&file, "", t0 + Duration::from_micros(500));
    assert_eq!(
        fs::metadata(&file).unwrap().modified().unwrap(),
        t0 + Duration::from_micros(500),
        "precondition: the file system keeps a submillisecond mtime"
    );
    assert_eq!(index_from_dirs(&mut conn, &opts, true).unwrap().indexed, 1);
    index_chunks(&mut conn, None).unwrap();
    assert!(collect_strings(&conn, "SELECT text FROM messages").is_empty());
    assert!(collect_strings(&conn, "SELECT content FROM qa_chunks").is_empty());

    let counts = counts_of_a_rerun(&mut conn, &opts);
    assert_eq!(counts["files_updated"], 0);
    assert_eq!(counts["files_unchanged"], 1);
}

#[test]
fn codex_log_emptied_or_reduced_to_session_meta_with_an_unchanged_mtime_loses_its_stored_messages()
{
    let (_db_dir, mut conn) = setup_test_db();
    let root = TempDir::new().unwrap();
    let file = root.path().join("s.jsonl");
    let absent = root.path().join("absent");
    let opts = IndexOptions {
        force: false,
        claude_dir: &absent,
        codex_dir: root.path(),
        codex_archived_dir: None,
    };
    let t0 = UNIX_EPOCH + Duration::from_secs(1_700_000_000);
    rewrite_with_mtime(&file, CODEX_QUESTION, t0);
    index_from_dirs(&mut conn, &opts, true).unwrap();
    index_chunks(&mut conn, None).unwrap();

    rewrite_with_mtime(&file, CODEX_METADATA_ONLY, t0);
    assert_eq!(index_from_dirs(&mut conn, &opts, true).unwrap().indexed, 1);
    index_chunks(&mut conn, None).unwrap();
    assert!(collect_strings(&conn, "SELECT text FROM messages").is_empty());

    rewrite_with_mtime(&file, "", t0 + Duration::from_micros(500));
    assert_eq!(
        fs::metadata(&file).unwrap().modified().unwrap(),
        t0 + Duration::from_micros(500),
        "precondition: the file system keeps a submillisecond mtime"
    );
    assert_eq!(index_from_dirs(&mut conn, &opts, true).unwrap().indexed, 1);
    index_chunks(&mut conn, None).unwrap();
    assert!(collect_strings(&conn, "SELECT text FROM messages").is_empty());
    assert!(collect_strings(&conn, "SELECT content FROM qa_chunks").is_empty());

    let counts = counts_of_a_rerun(&mut conn, &opts);
    assert_eq!(counts["files_updated"], 0);
    assert_eq!(counts["files_unchanged"], 1);
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
    index_chunks(conn, None).unwrap();
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
    index_chunks(conn, None).unwrap();
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

/// A Claude root and a Codex root under one temp dir. Only the root named `root` exists,
/// and it holds the log `name`.
struct OneLog {
    _tmp: TempDir,
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
            _tmp: tmp,
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
    index_chunks(&mut conn, None).unwrap();
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
    index_chunks(&mut conn, None).unwrap();
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
    index_chunks(&mut conn, None).unwrap();
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
    index_chunks(&mut conn, None).unwrap();
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
    index_chunks(&mut conn, None).unwrap();
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
    index_chunks(&mut conn, None).unwrap();
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
    index_chunks(&mut conn, None).unwrap();
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
    index_chunks(&mut conn, None).unwrap();
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
    index_chunks(&mut conn, None).unwrap();
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
    index_chunks(&mut conn, None).unwrap();
    assert_eq!((first.indexed, first.preserved_embedded), (0, 1));
    assert_eq!(snapshot(&conn), before);
    drop(conn);

    let mut conn = open_db(&db_path).unwrap();
    let second = index_from_dirs(&mut conn, &opts, false).unwrap();
    index_chunks(&mut conn, None).unwrap();
    assert_eq!((second.indexed, second.preserved_embedded), (0, 1));
    assert_eq!(snapshot(&conn), before);
    drop(conn);

    let mut conn = open_db(&db_path).unwrap();
    assert_eq!(index_from_dirs(&mut conn, &opts, true).unwrap().indexed, 1);
    index_chunks(&mut conn, None).unwrap();
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
    index_chunks(&mut conn, None).unwrap();

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

#[test]
fn test_index_from_dirs_force_reindexes() {
    let (_dir, mut conn) = setup_test_db();
    let tmp = TempDir::new().unwrap();

    let claude_dir = tmp.path().join("claude_projects");
    fs::create_dir_all(&claude_dir).unwrap();
    fs::write(
            claude_dir.join("session1.jsonl"),
            r#"{"type":"user","message":{"role":"user","content":"hello"},"timestamp":"2026-03-01T00:00:00Z"}"#,
        ).unwrap();

    let codex_dir = tmp.path().join("codex_sessions");

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
    assert_eq!(stats.indexed, 1);

    index_chunks(&mut conn, None).unwrap();
    let embedder = MockEmbedder::new();
    embed_recent_chunks(&mut conn, &embedder, 100, None).unwrap();
    let qa_before: i64 = conn
        .query_row("SELECT COUNT(*) FROM qa_chunks", [], |r| r.get(0))
        .unwrap();
    let vec_before: i64 = conn
        .query_row("SELECT COUNT(*) FROM vec_chunks", [], |r| r.get(0))
        .unwrap();
    assert!(qa_before > 0, "should have qa_chunks before force reindex");
    assert!(
        vec_before > 0,
        "should have vec_chunks before force reindex"
    );

    let stats2 = index_from_dirs(
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
    assert_eq!(stats2.indexed, 1);

    let qa_after: i64 = conn
        .query_row("SELECT COUNT(*) FROM qa_chunks", [], |r| r.get(0))
        .unwrap();
    let vec_after: i64 = conn
        .query_row("SELECT COUNT(*) FROM vec_chunks", [], |r| r.get(0))
        .unwrap();
    assert_eq!(qa_after, 0, "force reindex should clear qa_chunks");
    assert_eq!(vec_after, 0, "force reindex should clear vec_chunks");
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
        stats
            .skipped_roots
            .iter()
            .map(|s| (s.source, s.reason, s.preserved_sessions))
            .collect::<Vec<_>>(),
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
    index_chunks(&mut conn, None).unwrap();
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
        stats
            .skipped_roots
            .iter()
            .map(|s| (s.source, s.reason, s.preserved_sessions))
            .collect::<Vec<_>>(),
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

    let codex = stats
        .skipped_roots
        .iter()
        .find(|s| s.source == Source::Codex)
        .expect("an incompletely enumerated root must surface even with nothing preserved");
    assert_eq!(
        codex.preserved_sessions, 0,
        "an empty DB has no rows to preserve"
    );
    assert_eq!(codex.reason, SkippedReason::IncompleteEnumeration);
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
    index_chunks(conn, None).unwrap();
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
    assert_eq!(
        first
            .parse_diagnostics
            .iter()
            .map(|d| (d.file_path.as_path(), d.read_error))
            .collect::<Vec<_>>(),
        [(fresh.as_path(), true)]
    );

    let second = index_from_dirs(&mut conn, &opts, false).unwrap();
    assert_eq!((second.indexed, second.total_sessions), (0, 1));
    assert_only_the_unchanged_codex_session_remains(&conn, &before);
    assert!(second.skipped_roots.is_empty());
    assert_eq!(
        second
            .parse_diagnostics
            .iter()
            .map(|d| (d.file_path.as_path(), d.read_error))
            .collect::<Vec<_>>(),
        [(fresh.as_path(), true)]
    );
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
        first
            .skipped_roots
            .iter()
            .map(|s| (s.source, s.reason, s.preserved_sessions))
            .collect::<Vec<_>>(),
        [(Source::Codex, SkippedReason::IncompleteEnumeration, 1)]
    );
    assert!(first.parse_diagnostics.is_empty());

    let second = index_from_dirs(&mut conn, &opts, false).unwrap();
    assert_eq!((second.indexed, second.total_sessions), (0, 1));
    assert_only_the_unchanged_codex_session_remains(&conn, &before);
    assert_eq!(
        second
            .skipped_roots
            .iter()
            .map(|s| (s.source, s.reason, s.preserved_sessions))
            .collect::<Vec<_>>(),
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

    index_chunks(&mut conn, None).unwrap();
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
        stats.skipped_roots.len(),
        1,
        "the missing codex root must be reported as preserved"
    );
    assert_eq!(stats.skipped_roots[0].source, Source::Codex);
}

#[test]
fn rebuild_with_every_root_missing_keeps_all_sessions_and_embeddings() {
    let (_dir, mut conn) = setup_test_db();
    let (tmp, _claude_dir, _codex_dir) = seed_claude_and_codex(&mut conn);

    index_chunks(&mut conn, None).unwrap();
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
        stats.skipped_roots.len(),
        2,
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
    index_from_dirs(conn, &logs.opts(), true).unwrap();
    index_chunks(conn, None).unwrap();
    embed_recent_chunks(conn, &MockEmbedder::new(), 100, None).unwrap();
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
    index_chunks(conn, None).unwrap();
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
    let deep_dir = claude_dir.join("d0/d1/d2/d3/d4/d5/d6/d7/d8/d9");
    fs::create_dir_all(&deep_dir).unwrap();
    fs::write(claude_dir.join("shallow.jsonl"), FRESHNESS_QUESTION).unwrap();
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
        collect_strings(&conn, "SELECT session_id FROM sessions"),
        ["shallow"]
    );
    assert_eq!(
        stats
            .skipped_roots
            .iter()
            .map(|s| (s.source, s.reason, s.preserved_sessions))
            .collect::<Vec<_>>(),
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
fn test_011_index_chunks_incremental() {
    let (dir, mut conn) = setup_test_db();
    let tmp = TempDir::new().unwrap();
    let claude_dir = tmp.path().join("claude_projects");
    fs::create_dir_all(&claude_dir).unwrap();
    let path = claude_dir.join("s1.jsonl");
    let assistant = r#"{"type":"assistant","cwd":"/proj","message":{"role":"assistant","content":"hi there"},"timestamp":"2026-03-01T00:00:00Z"}"#;
    fs::write(&path, assistant).unwrap();
    let codex_dir = tmp.path().join("codex_sessions");
    let opts = IndexOptions {
        force: false,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
        codex_archived_dir: None,
    };
    index_from_dirs(&mut conn, &opts, true).unwrap();
    let first = index_chunks(&mut conn, None).unwrap();
    assert_eq!(
        (
            first.chunks_created,
            first.sessions_chunked,
            first.messages_read
        ),
        (0, 1, 1)
    );

    drop(conn);
    let mut conn = open_db(&dir.path().join("test.db")).unwrap();
    assert_eq!(index_from_dirs(&mut conn, &opts, true).unwrap().indexed, 0);
    let unchanged = index_chunks(
        &mut conn,
        Some(&|_, _| panic!("unchanged session processed")),
    )
    .unwrap();
    assert_eq!(
        (
            unchanged.chunks_created,
            unchanged.sessions_chunked,
            unchanged.messages_read,
            unchanged.message_scans,
            unchanged.message_batches
        ),
        (0, 0, 0, 0, 0)
    );

    fs::write(
        &path,
        format!(
            "{assistant}\n{}",
            r#"{"type":"user","message":{"role":"user","content":"hello"}}"#
        ),
    )
    .unwrap();
    conn.execute("UPDATE sessions SET mtime = 0", []).unwrap();
    assert_eq!(index_from_dirs(&mut conn, &opts, true).unwrap().indexed, 1);
    let appended = index_chunks(&mut conn, None).unwrap();
    assert_eq!(
        (
            appended.chunks_created,
            appended.sessions_chunked,
            appended.messages_read
        ),
        (1, 1, 2)
    );
    assert_eq!(
        collect_strings(&conn, "SELECT content FROM qa_chunks"),
        ["hello"]
    );
    assert_eq!(index_chunks(&mut conn, None).unwrap().sessions_chunked, 0);
}

#[test]
fn test_index_chunks_progress_callback_counts_sessions() {
    let (_dir, mut conn) = setup_test_db();
    seed_session(&conn, "s1");
    seed_session(&conn, "s2");
    for (role, text) in [("user", "hello"), ("assistant", "hi there")] {
        conn.execute(
            "INSERT INTO messages (session_id, role, text) VALUES ('s1', ?1, ?2)",
            rusqlite::params![role, text],
        )
        .unwrap();
    }

    let calls = Mutex::new(Vec::new());
    let stats = index_chunks(
        &mut conn,
        Some(&|done, total| calls.lock().unwrap().push((done, total))),
    )
    .unwrap();

    assert_eq!(calls.into_inner().unwrap(), vec![(1, 2), (2, 2)]);
    assert_eq!(stats.chunks_created, 1);
}

#[test]
fn test_index_chunks_progress_callback_silent_when_all_chunked() {
    let (_dir, mut conn) = setup_test_db();
    seed_session(&conn, "s1");
    conn.execute(
        "INSERT INTO messages (session_id, role, text) VALUES ('s1', 'user', 'hello')",
        [],
    )
    .unwrap();
    index_chunks(&mut conn, None).unwrap();

    let calls = Mutex::new(Vec::new());
    let stats = index_chunks(
        &mut conn,
        Some(&|done, total| calls.lock().unwrap().push((done, total))),
    )
    .unwrap();

    assert_eq!(stats.chunks_created, 0);
    assert_eq!(calls.into_inner().unwrap(), Vec::<(usize, usize)>::new());
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
        stats
            .skipped_roots
            .iter()
            .map(|s| (s.source, s.reason, s.preserved_sessions))
            .collect::<Vec<_>>(),
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
        stats
            .skipped_roots
            .iter()
            .map(|s| (s.source, s.reason, s.preserved_sessions))
            .collect::<Vec<_>>(),
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

#[test]
fn test_embed_recent_chunks_populates_vec_chunks() {
    let (_dir, mut conn) = setup_test_db();
    let tmp = TempDir::new().unwrap();
    let claude_dir = tmp.path().join("claude_projects");
    fs::create_dir_all(&claude_dir).unwrap();
    fs::write(
            claude_dir.join("s1.jsonl"),
            concat!(
                r#"{"type":"user","cwd":"/proj","message":{"role":"user","content":"hello world"},"timestamp":"2026-03-01T00:00:00Z"}"#,
                "\n",
                r#"{"type":"assistant","message":{"role":"assistant","content":"hi there"}}"#,
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
    index_chunks(&mut conn, None).unwrap();

    let chunk_count: i64 = conn
        .query_row("SELECT COUNT(*) FROM qa_chunks", [], |r| r.get(0))
        .unwrap();
    assert!(chunk_count > 0, "should have chunks to embed");

    let embedder = MockEmbedder::new();
    let result = embed_recent_chunks(&mut conn, &embedder, 100, None).unwrap();

    assert_eq!(result.embedded, usize::try_from(chunk_count).unwrap());
    assert_eq!(result.failed_count, 0);

    let vec_count: i64 = conn
        .query_row("SELECT COUNT(*) FROM vec_chunks", [], |r| r.get(0))
        .unwrap();
    assert_eq!(vec_count, chunk_count, "all chunks should be embedded");

    let result2 = embed_recent_chunks(&mut conn, &embedder, 100, None).unwrap();
    assert_eq!(result2.embedded, 0);
}

#[test]
fn test_embed_chunks_continues_past_failed_batch_and_counts_remainder() {
    let (_dir, mut conn) = setup_test_db();

    seed_session(&conn, "s1");
    let message_count = EMBED_BATCH_SIZE + 10;
    for i in 0..message_count {
        conn.execute(
            "INSERT INTO messages (session_id, role, text) VALUES ('s1', 'user', ?1)",
            [format!("question number {i}")],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO messages (session_id, role, text) VALUES ('s1', 'assistant', ?1)",
            [format!("answer number {i}")],
        )
        .unwrap();
    }
    index_chunks(&mut conn, None).unwrap();

    let chunk_count: i64 = conn
        .query_row("SELECT COUNT(*) FROM qa_chunks", [], |r| r.get(0))
        .unwrap();
    assert!(
        chunk_count > EMBED_BATCH_SIZE as i64,
        "need enough chunks for multiple batches, got {chunk_count}"
    );

    let embedder = MockEmbedder::failing_after(EMBED_BATCH_SIZE);
    let result = embed_recent_chunks(
        &mut conn,
        &embedder,
        usize::try_from(chunk_count).unwrap(),
        None,
    )
    .unwrap();

    assert_eq!(
        result.embedded, EMBED_BATCH_SIZE,
        "the first batch embeds and commits before the failures start"
    );
    let remaining = usize::try_from(chunk_count).unwrap() - EMBED_BATCH_SIZE;
    assert_eq!(
        result.failed_count, remaining,
        "every batch after the first fails (counter not reset), so all remaining chunks are failed"
    );

    let vec_count: i64 = conn
        .query_row("SELECT COUNT(*) FROM vec_chunks", [], |r| r.get(0))
        .unwrap();
    assert_eq!(
        vec_count, EMBED_BATCH_SIZE as i64,
        "the committed first batch survives the later failures"
    );
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
fn test_192e_backfill_populates_rowid_range_count_match() {
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
fn test_192f_backfill_leaves_null_on_count_mismatch() {
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
fn test_229b_backfill_skips_count_match_with_content_drift() {
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
fn test_229_trailing_assistant_in_multi_turn_is_embedded() {
    let (_dir, mut conn) = setup_test_db();
    seed_session(&conn, "s1");
    seed_message(&conn, "user", "how do i configure the retry backoff");
    seed_message(&conn, "assistant", "first set the base interval");
    seed_message(
        &conn,
        "assistant",
        "then cap it with zylophonemax jitter ceiling",
    );

    index_chunks(&mut conn, None).unwrap();
    let embedder = MockEmbedder::new();
    embed_recent_chunks(&mut conn, &embedder, 100, None).unwrap();

    let embedded: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM qa_chunks c JOIN vec_chunks v ON v.chunk_id = c.id \
             WHERE c.content LIKE '%zylophonemax%'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert!(
        embedded > 0,
        "the trailing assistant's text must reach an embedded qa_chunk so semantic \
         search can find it; orphaning it leaves it FTS-only and vector-invisible"
    );
}

#[test]
fn edit_write_の_file_path_が_session_files_に入る() {
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
    index_chunks(&mut conn, None).unwrap();

    let paths = collect_strings(&conn, "SELECT path FROM session_files ORDER BY path");
    assert_eq!(
        paths,
        vec![
            "/proj/alpha_marker.rs".to_owned(),
            "/proj/beta_marker.rs".to_owned(),
        ],
        "Edit and Write file_path values must land in session_files"
    );

    let message_texts = collect_strings(&conn, "SELECT text FROM messages");
    let chunk_contents = collect_strings(&conn, "SELECT content FROM qa_chunks");
    for marker in ["alpha_marker", "beta_marker", "filecontents"] {
        assert!(
            !message_texts.iter().any(|t| t.contains(marker)),
            "tool_use text {marker:?} must not appear in messages"
        );
        assert!(
            !chunk_contents.iter().any(|c| c.contains(marker)),
            "tool_use text {marker:?} must not appear in qa_chunks"
        );
    }
}

#[test]
fn 妥当でない値は捨てられる() {
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

#[test]
fn 読めない_jsonl_のセッションはマーカーを立てず次回に再試行する() {
    let (_dir, mut conn) = setup_test_db();
    seed_session(&conn, "gone-session");
    let unmarked_before: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM sessions WHERE files_scanned IS NULL",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(unmarked_before, 1, "precondition: one unmarked session");

    let marked = backfill_session_files(&mut conn, None).unwrap();

    assert_eq!(
        marked, 0,
        "an unreadable session must not be counted as marked"
    );
    let unmarked_after: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM sessions WHERE files_scanned IS NULL",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        unmarked_after, 1,
        "an unreadable JSONL must leave files_scanned NULL so the next index retries"
    );
    let file_rows: i64 = conn
        .query_row("SELECT COUNT(*) FROM session_files", [], |r| r.get(0))
        .unwrap();
    assert_eq!(
        file_rows, 0,
        "an unreadable JSONL yields no session_files rows"
    );

    let diagnostics = load_parse_diagnostics(&conn).unwrap();
    assert_eq!(diagnostics.len(), 1);
    assert!(diagnostics[0].read_error);

    let retried = backfill_session_files(&mut conn, None).unwrap();
    assert_eq!(
        retried, 0,
        "the retry still fails on the vanished file but keeps the session eligible"
    );
}

#[test]
fn 未知_source_と_空_jsonl_のセッションはマーカーを立てて再読しない() {
    let (_dir, mut conn) = setup_test_db();
    conn.execute(
        "INSERT INTO sessions (session_id, source, file_path, project, slug, timestamp, mtime, files_scanned) VALUES ('mystery', 'perplexity', '/f', '/p', 'slug', 0, 0.0, NULL)",
        [],
    )
    .unwrap();
    let tmp = TempDir::new().unwrap();
    let empty = tmp.path().join("empty-session.jsonl");
    fs::write(&empty, "").unwrap();
    conn.execute(
        "INSERT INTO sessions (session_id, source, file_path, project, slug, timestamp, mtime, files_scanned) VALUES ('zero', 'claude', ?1, '/p', 'slug', 0, 0.0, NULL)",
        [empty.to_string_lossy().to_string()],
    )
    .unwrap();

    let marked = backfill_session_files(&mut conn, None).unwrap();

    assert_eq!(marked, 2, "both terminal cases must be counted as marked");
    let unmarked: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM sessions WHERE files_scanned IS NULL",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        unmarked, 0,
        "terminal cases must not stay eligible for re-read"
    );
    let file_rows: i64 = conn
        .query_row("SELECT COUNT(*) FROM session_files", [], |r| r.get(0))
        .unwrap();
    assert_eq!(
        file_rows, 0,
        "neither terminal case yields session_files rows"
    );
}

#[test]
fn chunk_completion_rolls_back_with_chunks_and_retries() {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    for interrupt in [false, true] {
        let (_dir, mut conn) = setup_test_db();
        seed_session(&conn, "empty");
        seed_session(&conn, "s1");
        seed_message(&conn, "user", "question");
        let (observer, lines) = recording();
        if !interrupt {
            conn.execute_batch(
                "CREATE TEMP TRIGGER fail_completion BEFORE UPDATE OF chunks_indexed ON sessions
                 WHEN NEW.session_id = 's1'
                 BEGIN SELECT RAISE(ABORT, 'injected completion failure'); END;",
            )
            .unwrap();
            let error = index_chunks_observed(&mut conn, None, &observer)
                .err()
                .unwrap();
            assert!(error.to_string().contains("injected completion failure"));
        } else {
            assert!(
                catch_unwind(AssertUnwindSafe(|| {
                    index_chunks_observed(
                        &mut conn,
                        Some(&|done, _| {
                            if done == 2 {
                                panic!("injected interruption");
                            }
                        }),
                        &observer,
                    )
                    .unwrap();
                }))
                .is_err()
            );
        }
        assert_eq!(
            conn.query_row("SELECT count(*) FROM qa_chunks", [], |r| r.get::<_, i64>(0))
                .unwrap(),
            0
        );
        assert_eq!(
            conn.query_row(
                "SELECT count(*) FROM sessions WHERE chunks_indexed IS NOT NULL",
                [],
                |r| r.get::<_, i64>(0)
            )
            .unwrap(),
            0
        );
        assert_eq!(
            observer.snapshot()["counts"]["sessions_chunked_committed"],
            0
        );
        assert!(
            lines
                .borrow()
                .iter()
                .any(|line| line.contains("chunking: unfinished"))
        );
        assert!(
            !lines
                .borrow()
                .iter()
                .any(|line| line.contains("chunking: committed"))
        );
        conn.execute_batch("DROP TRIGGER IF EXISTS fail_completion;")
            .unwrap();
        let retried = index_chunks(&mut conn, None).unwrap();
        assert_eq!((retried.chunks_created, retried.sessions_chunked), (1, 2));
        assert_eq!(index_chunks(&mut conn, None).unwrap().sessions_chunked, 0);
    }
}

#[test]
fn chunk_pass_reports_stage_lines_counts_and_rows_with_and_without_pending_sessions() {
    let (_dir, mut conn) = setup_test_db();

    let (observer, lines) = recording();
    let stats = index_chunks_observed(&mut conn, None, &observer).unwrap();
    assert_eq!(
        (
            stats.chunks_created,
            stats.sessions_chunked,
            stats.message_batches,
            stats.message_scans,
            stats.messages_read
        ),
        (0, 0, 0, 0, 0)
    );
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

    seed_session(&conn, "empty");
    seed_session(&conn, "empty2");
    seed_session(&conn, "s1");
    conn.execute(
        "UPDATE sessions SET timestamp = 1234 WHERE session_id = 's1'",
        [],
    )
    .unwrap();
    let question = seed_message(&conn, "user", "question");
    seed_message(&conn, "unknown-role", "dropped text");
    let (observer, lines) = recording();
    let progress = RefCell::new(Vec::new());
    let on_progress = |done, total| progress.borrow_mut().push((done, total));

    let stats = index_chunks_observed(&mut conn, Some(&on_progress), &observer).unwrap();

    assert_eq!(
        (
            stats.chunks_created,
            stats.sessions_chunked,
            stats.message_batches,
            stats.message_scans,
            stats.messages_read
        ),
        (1, 3, 1, 1, 2)
    );
    assert_eq!(*progress.borrow(), [(1, 3), (2, 3), (3, 3)]);
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
    let chunk: (String, String, i64, i64, i64) = conn
        .query_row(
            "SELECT session_id, content, timestamp, src_rowid_lo, src_rowid_hi FROM qa_chunks",
            [],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)),
        )
        .unwrap();
    assert_eq!(
        chunk,
        (
            "s1".to_owned(),
            "question".to_owned(),
            1234,
            question,
            question
        )
    );
    assert_eq!(
        collect_strings(
            &conn,
            "SELECT session_id || '=' || chunks_indexed FROM sessions ORDER BY session_id"
        ),
        ["empty=0", "empty2=0", "s1=0"]
    );
    assert!(
        !conn
            .prepare("SELECT 1 FROM temp.sqlite_master WHERE name = 'chunk_message_rows'")
            .unwrap()
            .exists([])
            .unwrap(),
        "the staging table must not outlive the pass"
    );
}

#[test]
fn chunk_batches_limit_session_count_and_source_bytes() {
    let small = vec![("s".to_owned(), None, 1); CHUNK_BATCH_SESSIONS + 1];
    assert_eq!(chunk_batch_len(&small), 64);
    let sizes = [
        CHUNK_BATCH_BYTES / 2,
        CHUNK_BATCH_BYTES / 2,
        1,
        CHUNK_BATCH_BYTES + 1,
        0,
    ];
    let sessions: Vec<_> = sizes
        .iter()
        .map(|size| ("s".to_owned(), None, *size))
        .collect();
    assert_eq!(
        chunk_batch_len(&sessions),
        2,
        "exact byte budget is allowed"
    );
    assert_eq!(
        chunk_batch_len(&sessions[2..]),
        1,
        "do not add an oversized session"
    );
    assert_eq!(
        chunk_batch_len(&sessions[3..]),
        1,
        "oversized conversation runs alone"
    );
    assert_eq!(
        chunk_batch_len(&sessions[4..]),
        1,
        "empty conversation still finishes"
    );
}

#[test]
fn chunk_batches_preserve_interleaved_sessions_order_and_search_ranges() {
    let (_dir, mut conn) = setup_test_db();
    for i in 0..65 {
        seed_session(&conn, &format!("s{i:03}"));
    }
    for (turn, role) in [
        (0, "user"),
        (1, "system"),
        (2, "assistant"),
        (3, "assistant"),
    ] {
        for i in (0..65).rev() {
            conn.execute(
                "INSERT INTO messages (session_id, role, text) VALUES (?1, ?2, ?3)",
                rusqlite::params![format!("s{i:03}"), role, format!("session {i} turn {turn}")],
            )
            .unwrap();
        }
    }
    let stats = index_chunks(&mut conn, None).unwrap();
    assert_eq!(
        (
            stats.chunks_created,
            stats.sessions_chunked,
            stats.message_batches,
            stats.messages_read,
            stats.message_scans
        ),
        (65, 65, 2, 260, 1)
    );
    let mut stmt = conn.prepare("SELECT session_id, content, src_rowid_lo, src_rowid_hi FROM qa_chunks ORDER BY session_id").unwrap();
    let actual: Vec<(String, String, i64, i64)> = stmt
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))
        .unwrap()
        .map(Result::unwrap)
        .collect();
    for (i, (sid, content, lo, hi)) in (0_i64..65).zip(actual) {
        assert_eq!(sid, format!("s{i:03}"));
        assert_eq!(
            content,
            format!("session {i} turn 0\nsession {i} turn 2\nsession {i} turn 3")
        );
        assert_eq!((lo, hi), (65 - i, 260 - i));
    }
    conn.execute_batch("CREATE TEMP TABLE chunk_message_rows (session_id TEXT, message_rowid INTEGER, text_bytes INTEGER, PRIMARY KEY(session_id, message_rowid)) WITHOUT ROWID;").unwrap();
    let stage_plan: Vec<String> = conn
        .prepare(&format!("EXPLAIN QUERY PLAN {STAGE_CHUNK_MESSAGES}"))
        .unwrap()
        .query_map([], |r| r.get(3))
        .unwrap()
        .map(Result::unwrap)
        .collect();
    assert!(
        stage_plan[0].contains("SCAN m VIRTUAL TABLE"),
        "FTS must be the single outer scan: {stage_plan:?}"
    );
    assert!(
        stage_plan[1].contains("SEARCH s USING INDEX"),
        "each message probes the session primary key: {stage_plan:?}"
    );
    let plan: Vec<String> = conn
        .prepare(&format!("EXPLAIN QUERY PLAN {READ_CHUNK_BATCH}"))
        .unwrap()
        .query_map(["s000", "s064"], |r| r.get(3))
        .unwrap()
        .map(Result::unwrap)
        .collect();
    assert!(
        plan.iter()
            .any(|line| line.contains("VIRTUAL TABLE INDEX") && line.contains('=')),
        "expected FTS rowid equality lookup: {plan:?}"
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
    index_chunks(&mut conn, None).unwrap();
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
        index_chunks(&mut conn, None).unwrap();
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
    index_chunks(&mut conn, None).unwrap();
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
    assert_eq!(index_chunks(&mut conn, None).unwrap().sessions_chunked, 0);
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

fn write_reuse_conversation(path: &Path, messages: &[(&str, &str)]) {
    let text = messages
        .iter()
        .map(|(role, text)| {
            serde_json::json!({"type": role, "message": {"role": role, "content": text}})
                .to_string()
                + "\n"
        })
        .collect::<String>();
    fs::write(path, text).unwrap();
}

fn chunk_vectors(conn: &Connection) -> Vec<String> {
    collect_strings(
        conn,
        "SELECT json_array(c.content, v.sub_idx, hex(v.embedding)) FROM qa_chunks c JOIN vec_chunks v ON c.id = v.chunk_id ORDER BY c.content, v.sub_idx, hex(v.embedding)",
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
            index_chunks(&mut writer, None)?;
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

#[test]
fn unchanged_chunks_reuse_vectors_and_changed_groups_match_a_fresh_rebuild() {
    let original = [
        ("user", "duplicate"),
        ("assistant", "same answer"),
        ("user", "duplicate"),
        ("assistant", "same answer"),
        ("user", "tail question"),
        ("assistant", "tail answer"),
    ];
    for scenario in [
        "append",
        "assistant",
        "mtime",
        "truncate",
        "partial-duplicate",
        "replace",
        "rules",
        "pending-rules",
        "pending-old-rules",
        "pending-current-rules",
        "rebuild",
    ] {
        let (_db_dir, mut conn) = setup_test_db();
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("claude");
        fs::create_dir(&root).unwrap();
        let path = root.join("reuse.jsonl");
        let absent = tmp.path().join("absent");
        let mut opts = IndexOptions {
            force: false,
            claude_dir: &root,
            codex_dir: &absent,
            codex_archived_dir: None,
        };
        write_reuse_conversation(&path, &original);
        index_from_dirs(&mut conn, &opts, true).unwrap();
        if scenario == "pending-current-rules" {
            let offline = tmp.path().join("offline");
            fs::rename(&root, &offline).unwrap();
            let mut resumed = open_db(&_db_dir.path().join("test.db")).unwrap();
            assert_eq!(
                index_from_dirs(&mut resumed, &opts, true).unwrap().indexed,
                0
            );
            assert_eq!(
                index_chunks(&mut resumed, None).unwrap().sessions_chunked,
                1
            );
            fs::rename(&offline, &root).unwrap();
            assert_eq!(index_from_dirs(&mut conn, &opts, true).unwrap().indexed, 0);
        }
        index_chunks(&mut conn, None).unwrap();
        let model = MockEmbedder::new();
        assert_eq!(
            embed_recent_chunks(&mut conn, &model, 100, None)
                .unwrap()
                .embedded,
            3
        );
        if scenario == "partial-duplicate" {
            conn.execute(
                "DELETE FROM vec_chunks WHERE chunk_id = (SELECT min(id) FROM qa_chunks)",
                [],
            )
            .unwrap();
        }
        let before = collect_strings(
            &conn,
            "SELECT json_array(c.id, c.generation, c.content, v.rowid, hex(v.embedding)) FROM qa_chunks c JOIN vec_chunks v ON c.id = v.chunk_id ORDER BY c.id",
        );
        conn.execute(
            "INSERT INTO messages (session_id, role, text) VALUES ('other', 'user', 'unrelated')",
            [],
        )
        .unwrap();
        let mut changed = original.to_vec();
        let (reused, inferred) = match scenario {
            "append" => {
                changed.extend([
                    ("user", "appended question"),
                    ("assistant", "appended answer"),
                ]);
                (3, 1)
            }
            "assistant" => {
                changed.push(("assistant", "continuation needle"));
                (2, 1)
            }
            "mtime" | "pending-current-rules" => (3, 0),
            "truncate" | "partial-duplicate" => {
                changed.truncate(2);
                (1, 0)
            }
            "replace" => {
                changed = vec![
                    ("user", "tail question"),
                    ("assistant", "tail answer"),
                    ("user", "replacement needle"),
                ];
                (1, 1)
            }
            "rules" => {
                conn.execute("UPDATE sessions SET chunks_indexed = 99", [])
                    .unwrap();
                (0, 3)
            }
            "pending-rules" | "pending-old-rules" => {
                conn.execute_batch("DELETE FROM vec_chunks; DELETE FROM qa_chunks; UPDATE sessions SET chunks_indexed = NULL; UPDATE messages SET text = 'obsolete parser output' WHERE session_id = 'reuse';").unwrap();
                if scenario == "pending-old-rules" {
                    conn.execute("UPDATE sessions SET chunks_indexed = -99", [])
                        .unwrap();
                }
                let stamp = resolve_file_stamp(&path);
                let offline = tmp.path().join("offline");
                fs::rename(&root, &offline).unwrap();
                let stats = index_from_dirs(&mut conn, &opts, true).unwrap();
                assert_eq!(stats.indexed, 0);
                assert_eq!(stats.total_sessions, 1);
                assert_eq!(stats.skipped_roots[0].preserved_sessions, 1);
                assert_eq!(index_chunks(&mut conn, None).unwrap().sessions_chunked, 1);
                assert_eq!(index_chunks(&mut conn, None).unwrap().sessions_chunked, 0);
                embed_recent_chunks(&mut conn, &MockEmbedder::new(), 100, None).unwrap();
                fs::rename(&offline, &root).unwrap();
                assert!(resolve_file_stamp(&path) == stamp);
                (0, 3)
            }
            "rebuild" => {
                opts.force = true;
                (0, 3)
            }
            _ => unreachable!(),
        };
        if !matches!(scenario, "rules" | "pending-rules" | "pending-old-rules") {
            write_reuse_conversation(&path, &changed);
            conn.execute("UPDATE sessions SET mtime = 0", []).unwrap();
        }
        let observer = Observer::default();
        assert_eq!(
            index_from_dirs_observed(&mut conn, &opts, true, &observer)
                .unwrap()
                .indexed,
            1,
            "{scenario}"
        );
        assert_eq!(
            observer.snapshot()["counts"]["embeddings_reused_committed"],
            reused,
            "{scenario}"
        );
        index_chunks(&mut conn, None).unwrap();
        let retained = collect_strings(
            &conn,
            "SELECT json_array(c.id, c.generation, c.content, v.rowid, hex(v.embedding)) FROM qa_chunks c JOIN vec_chunks v ON c.id = v.chunk_id ORDER BY c.id",
        );
        assert_eq!(retained.len(), reused as usize, "{scenario}");
        assert!(
            retained.iter().all(|row| before.contains(row)),
            "{scenario}: reused vectors changed"
        );
        assert_eq!(
            embed_recent_chunks(&mut conn, &model, 100, None)
                .unwrap()
                .embedded,
            inferred as usize
        );
        assert_eq!(
            model.calls(),
            3 + inferred as usize,
            "{scenario}: redundant inference"
        );
        let tx = conn.transaction().unwrap();
        let messages = read_session_messages(&tx, "reuse").unwrap();
        let mut expected: Vec<_> = chunker::chunk_messages("reuse", &messages, None)
            .into_iter()
            .map(|c| (c.content, c.src_rowid_lo, c.src_rowid_hi))
            .collect();
        let mut actual: Vec<(String, i64, i64)> = tx
            .prepare("SELECT content, src_rowid_lo, src_rowid_hi FROM qa_chunks")
            .unwrap()
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
            .unwrap()
            .map(Result::unwrap)
            .collect();
        expected.sort();
        actual.sort();
        assert_eq!(
            actual, expected,
            "{scenario}: incorrect source rowid mapping"
        );
        tx.commit().unwrap();
        if scenario == "assistant" {
            let results = search(&conn, "continuation", &SearchOptions::default()).unwrap();
            assert_eq!(results.len(), 1);
            assert!(results[0].excerpt.contains("continuation needle"));
            assert!(results[0].chunk_id.is_some());
        }
        let expected_vectors = chunk_vectors(&conn);
        opts.force = true;
        index_from_dirs(&mut conn, &opts, true).unwrap();
        index_chunks(&mut conn, None).unwrap();
        embed_recent_chunks(&mut conn, &MockEmbedder::new(), 100, None).unwrap();
        assert_eq!(
            chunk_vectors(&conn),
            expected_vectors,
            "{scenario}: differs from rebuild"
        );
    }
}

#[test]
fn reconciliation_rollback_and_failed_inference_preserve_reusable_work_for_resume() {
    use crate::embedder::{EmbedOptions, embed_chunks, pending_chunks};
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
    let mut messages = vec![
        ("user", "stable"),
        ("assistant", "answer"),
        ("user", "tail"),
    ];
    write_reuse_conversation(&path, &messages);
    index_from_dirs(&mut conn, &opts, true).unwrap();
    index_chunks(&mut conn, None).unwrap();
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
    messages.push(("assistant", "new tail"));
    write_reuse_conversation(&path, &messages);
    conn.execute_batch("CREATE TEMP TRIGGER fail_reconcile BEFORE UPDATE OF chunks_indexed ON sessions BEGIN SELECT RAISE(ABORT, 'reconciliation failed'); END;").unwrap();
    assert!(index_from_dirs(&mut conn, &opts, true).is_err());
    assert_eq!(chunk_vectors(&conn), before);
    assert_eq!(
        collect_strings(&conn, "SELECT text FROM messages ORDER BY rowid"),
        ["stable", "answer", "tail"]
    );
    conn.execute_batch("DROP TRIGGER fail_reconcile;").unwrap();
    let mut writer = open_db(&db_dir.path().join("test.db")).unwrap();
    index_from_dirs(&mut writer, &opts, true).unwrap();
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
    let model = MockEmbedder::new();
    assert_eq!(
        embed_recent_chunks(&mut conn, &model, 100, None)
            .unwrap()
            .embedded,
        1
    );
    assert_eq!(model.calls(), 1);
    assert_eq!(chunk_vectors(&conn).len(), 2);
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

fn stored_vectors(conn: &Connection) -> Vec<String> {
    collect_strings(
        conn,
        "SELECT json_array(c.content, hex(v.embedding)) FROM vec_chunks v JOIN qa_chunks c ON c.id = v.chunk_id ORDER BY c.content, v.sub_idx",
    )
}

const ROLLOUT_META: &str = r#"{"timestamp":"2026-04-27T00:00:00Z","type":"session_meta","payload":{"id":"01900000-0000-7000-8000-000000000001","cwd":"/proj"}}"#;

/// The stored session's id, slug, project and start time: every column but the file path.
fn session_identity(conn: &Connection) -> Vec<String> {
    collect_strings(
        conn,
        "SELECT json_array(session_id, slug, project, timestamp) FROM sessions",
    )
}

#[test]
fn compressing_a_codex_log_keeps_the_indexed_session_and_its_embeddings() {
    let (_db, mut conn) = setup_test_db();
    let tmp = TempDir::new().unwrap();
    let claude_dir = tmp.path().join("claude_projects");
    let codex_dir = tmp.path().join("codex_sessions");
    let day_dir = codex_dir.join("2026/04/27");
    fs::create_dir_all(&day_dir).unwrap();
    let plain = day_dir.join(ROLLOUT_NAME);
    fs::write(&plain, format!("{ROLLOUT_META}\n{ROLLOUT_TURNS}")).unwrap();
    let opts = IndexOptions {
        force: false,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
        codex_archived_dir: None,
    };
    index_from_dirs(&mut conn, &opts, true).unwrap();
    index_chunks(&mut conn, None).unwrap();
    assert_eq!(
        embed_recent_chunks(&mut conn, &MockEmbedder::new(), 100, None)
            .unwrap()
            .embedded,
        1
    );
    let identity_before = session_identity(&conn);
    let vectors_before = stored_vectors(&conn);

    let compressed = compress_log(&plain);
    fs::remove_file(&plain).unwrap();
    let stats = index_from_dirs(&mut conn, &opts, true).unwrap();

    assert_eq!(stats.total_sessions, 1);
    assert_eq!(
        collect_strings(&conn, "SELECT file_path FROM sessions"),
        [compressed.to_str().unwrap()]
    );
    assert_eq!(session_identity(&conn), identity_before);
    assert_eq!(stored_vectors(&conn), vectors_before);
    assert!(stats.parse_diagnostics.is_empty());
}

#[test]
fn compressing_a_codex_log_that_has_no_session_meta_keeps_the_indexed_session_and_its_embeddings() {
    let (_db, mut conn) = setup_test_db();
    let tmp = TempDir::new().unwrap();
    let claude_dir = tmp.path().join("claude_projects");
    let codex_dir = tmp.path().join("codex_sessions");
    let day_dir = codex_dir.join("2026/04/27");
    fs::create_dir_all(&day_dir).unwrap();
    let plain = day_dir.join(ROLLOUT_NAME);
    fs::write(&plain, ROLLOUT_TURNS).unwrap();
    let opts = IndexOptions {
        force: false,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
        codex_archived_dir: None,
    };
    index_from_dirs(&mut conn, &opts, true).unwrap();
    index_chunks(&mut conn, None).unwrap();
    assert_eq!(
        embed_recent_chunks(&mut conn, &MockEmbedder::new(), 100, None)
            .unwrap()
            .embedded,
        1
    );
    let identity_before = session_identity(&conn);
    let vectors_before = stored_vectors(&conn);

    let compressed = compress_log(&plain);
    fs::remove_file(&plain).unwrap();
    let stats = index_from_dirs(&mut conn, &opts, true).unwrap();

    assert_eq!(stats.total_sessions, 1);
    assert_eq!(
        collect_strings(&conn, "SELECT file_path FROM sessions"),
        [compressed.to_str().unwrap()]
    );
    assert_eq!(session_identity(&conn), identity_before);
    assert_eq!(stored_vectors(&conn), vectors_before);
    assert!(stats.parse_diagnostics.is_empty());
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
    index_chunks(&mut conn, None).unwrap();
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
    index_chunks(&mut conn, None).unwrap();
    embed_recent_chunks(&mut conn, &MockEmbedder::new(), 100, None).unwrap();
    let row_before = session_row(&conn);
    let vectors_before = stored_vectors(&conn);
    assert!(!vectors_before.is_empty());

    fs::write(&plain, "").unwrap();
    index_from_dirs(&mut conn, &opts, true).unwrap();

    assert_eq!(session_row(&conn), row_before);
    assert_eq!(stored_vectors(&conn), vectors_before);
}

#[test]
fn compressed_logs_are_indexed_only_from_codex_roots() {
    let (_db, mut conn) = setup_test_db();
    let tmp = TempDir::new().unwrap();
    let claude_dir = tmp.path().join("claude_projects");
    let codex_dir = tmp.path().join("codex_sessions");
    fs::create_dir_all(&claude_dir).unwrap();
    fs::create_dir_all(&codex_dir).unwrap();
    // Every file below holds a session that would be indexed if its name were collected.
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

#[test]
fn archiving_a_codex_log_keeps_the_indexed_session_and_its_embeddings() {
    let (_db, mut conn) = setup_test_db();
    let roots = CodexRoots::new();
    let live = roots.day.join(ROLLOUT_NAME);
    fs::write(&live, format!("{ROLLOUT_META}\n{ROLLOUT_TURNS}")).unwrap();
    index_from_dirs(&mut conn, &roots.opts(), true).unwrap();
    index_chunks(&mut conn, None).unwrap();
    assert_eq!(
        embed_recent_chunks(&mut conn, &MockEmbedder::new(), 100, None)
            .unwrap()
            .embedded,
        1
    );
    let identity_before = session_identity(&conn);
    let vectors_before = stored_vectors(&conn);

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
    assert_eq!(stored_vectors(&conn), vectors_before);
    assert!(stats.skipped_roots.is_empty());
}

#[test]
fn archiving_a_codex_log_that_has_no_session_meta_keeps_the_indexed_session_and_its_embeddings() {
    let (_db, mut conn) = setup_test_db();
    let roots = CodexRoots::new();
    let live = roots.day.join(ROLLOUT_NAME);
    fs::write(&live, ROLLOUT_TURNS).unwrap();
    index_from_dirs(&mut conn, &roots.opts(), true).unwrap();
    index_chunks(&mut conn, None).unwrap();
    assert_eq!(
        embed_recent_chunks(&mut conn, &MockEmbedder::new(), 100, None)
            .unwrap()
            .embedded,
        1
    );
    let identity_before = session_identity(&conn);
    let vectors_before = stored_vectors(&conn);

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
    assert_eq!(stored_vectors(&conn), vectors_before);
    assert!(stats.skipped_roots.is_empty());
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
        stats
            .skipped_roots
            .iter()
            .map(|s| (s.source, s.reason, s.preserved_sessions))
            .collect::<Vec<_>>(),
        [(Source::Codex, SkippedReason::MissingRoot, 1)]
    );
    let note = stats.skipped_roots[0].note();
    assert!(
        note.contains("reconciled the rest") && !note.contains("none deleted this run"),
        "the other Codex root was cleaned, so the note must not claim nothing was deleted: {note}"
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
        stats
            .skipped_roots
            .iter()
            .map(|s| (s.source, s.reason, s.preserved_sessions))
            .collect::<Vec<_>>(),
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
        stats
            .skipped_roots
            .iter()
            .map(|s| (s.source, s.reason, s.preserved_sessions))
            .collect::<Vec<_>>(),
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
