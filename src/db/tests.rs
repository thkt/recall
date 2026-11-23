use super::*;
use crate::embedder::{f32_as_bytes, pending_count};
use tempfile::NamedTempFile;

#[path = "../../tests/support/stale_snapshot_note.rs"]
mod stale_snapshot_note;
use stale_snapshot_note::READ_ONLY_DIR_NOTE;

#[test]
fn a_new_index_opens_with_empty_tables_its_two_indexes_and_the_current_schema() {
    let tmp = NamedTempFile::new().unwrap();
    let conn = open_db(tmp.path()).unwrap();

    assert_eq!(
        collect_strings(
            &conn,
            "SELECT json_array(
                 (SELECT COUNT(*) FROM sessions), (SELECT COUNT(*) FROM messages),
                 (SELECT COUNT(*) FROM qa_chunks), (SELECT COUNT(*) FROM vec_chunks),
                 (SELECT COUNT(*) FROM messages_vocab))"
        ),
        ["[0,0,0,0,0]"]
    );
    assert_eq!(
        collect_strings(
            &conn,
            "SELECT name FROM sqlite_master WHERE type = 'index' AND sql IS NOT NULL ORDER BY name"
        ),
        ["idx_qa_chunks_session", "idx_sessions_timestamp"]
    );
    assert_eq!(schema_state(&conn).unwrap(), SchemaState::Current);
}

#[test]
fn opening_an_index_that_still_has_the_unused_project_index_drops_it() {
    let tmp = NamedTempFile::new().unwrap();
    drop(open_db(tmp.path()).unwrap());
    let earlier_release = Connection::open(tmp.path()).unwrap();
    earlier_release
        .execute_batch("CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project)")
        .unwrap();
    drop(earlier_release);

    let conn = open_db(tmp.path()).unwrap();

    assert_eq!(
        collect_strings(
            &conn,
            "SELECT name FROM sqlite_master WHERE type = 'index' AND sql IS NOT NULL ORDER BY name"
        ),
        ["idx_qa_chunks_session", "idx_sessions_timestamp"]
    );
}

#[test]
fn an_opened_database_runs_the_pinned_sqlite_vec() {
    let tmp = NamedTempFile::new().unwrap();
    let conn = open_db(tmp.path()).unwrap();

    let version: String = conn
        .query_row("SELECT vec_version()", [], |r| r.get(0))
        .unwrap();
    assert_eq!(
        version, "v0.1.9",
        "sqlite-vec version changed — re-verify unsafe transmute ABI in ensure_sqlite_vec()"
    );
}

#[test]
fn opening_an_index_a_second_time_leaves_its_schema_unchanged() {
    let tmp = NamedTempFile::new().unwrap();
    let first = schema_snapshot(&open_db(tmp.path()).unwrap());

    let second = schema_snapshot(&open_db(tmp.path()).unwrap());

    assert_eq!(second, first);
}

#[test]
fn upgrading_an_index_without_message_ranges_stores_each_range_and_drops_messages_without_a_session()
 {
    let tmp = NamedTempFile::new().unwrap();
    let conn = open_db(tmp.path()).unwrap();
    conn.execute_batch(
        "INSERT INTO sessions (session_id) VALUES ('a'), ('b'), ('quiet');
         INSERT INTO messages (rowid, session_id, role, text) VALUES
             (1, 'a', 'user', 'a first'), (2, 'b', 'user', 'b only'),
             (3, 'gone', 'user', 'no session'), (4, 'a', 'assistant', 'a last');
         ALTER TABLE sessions DROP COLUMN message_rowid_lo;
         ALTER TABLE sessions DROP COLUMN message_rowid_hi;",
    )
    .unwrap();
    drop(conn);

    let conn = open_db(tmp.path()).unwrap();

    assert_eq!(
        collect_strings(
            &conn,
            "SELECT json_array(session_id, message_rowid_lo, message_rowid_hi) FROM sessions \
             ORDER BY session_id"
        ),
        [r#"["a",1,4]"#, r#"["b",2,2]"#, r#"["quiet",1,0]"#]
    );
    assert_eq!(
        collect_strings(&conn, "SELECT text FROM messages ORDER BY rowid"),
        ["a first", "b only", "a last"]
    );
    let upgraded = schema_snapshot(&conn);
    drop(conn);
    assert_eq!(schema_snapshot(&open_db(tmp.path()).unwrap()), upgraded);
}

fn schema_snapshot(conn: &Connection) -> (i64, Vec<String>) {
    (
        conn.query_row("PRAGMA schema_version", [], |r| r.get(0))
            .unwrap(),
        collect_strings(
            conn,
            "SELECT json_array(type, name, tbl_name, sql) FROM sqlite_master ORDER BY type, name",
        ),
    )
}

fn embeddings(conn: &Connection) -> Vec<(i64, i64, i64, Vec<u8>)> {
    conn.prepare("SELECT rowid, chunk_id, sub_idx, embedding FROM vec_chunks ORDER BY rowid")
        .unwrap()
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))
        .unwrap()
        .map(Result::unwrap)
        .collect()
}

const SESSIONS_V0_14: &str = "session_id TEXT PRIMARY KEY, source TEXT, file_path TEXT, \
     project TEXT, slug TEXT, timestamp INTEGER, mtime REAL, file_size INTEGER, \
     session_type TEXT, files_scanned INTEGER, chunks_indexed INTEGER";
const CHUNKS_V0_14: &str = "id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, \
     content TEXT NOT NULL, timestamp INTEGER, src_rowid_lo INTEGER, src_rowid_hi INTEGER, \
     generation INTEGER NOT NULL DEFAULT 0";
const VECTORS_V0_14: &str = "+chunk_id INTEGER, +sub_idx INTEGER";
const SUPPORT_TABLES_V0_14: &str = "CREATE TABLE parse_diagnostics (
         file_path TEXT PRIMARY KEY, source TEXT NOT NULL,
         invalid_json_lines INTEGER NOT NULL DEFAULT 0,
         invalid_utf8_lines INTEGER NOT NULL DEFAULT 0,
         incomplete_tail_lines INTEGER NOT NULL DEFAULT 0,
         read_error INTEGER NOT NULL DEFAULT 0
     );
     CREATE TABLE qa_chunk_generation (
         singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
         value INTEGER NOT NULL CHECK (typeof(value) = 'integer' AND value >= 0)
     );
     INSERT INTO qa_chunk_generation VALUES (1, 0);
     CREATE TRIGGER qa_chunks_insert_generation AFTER INSERT ON qa_chunks BEGIN
         UPDATE qa_chunk_generation SET value = value + 1 WHERE singleton = 1;
         UPDATE qa_chunks SET generation = (SELECT value FROM qa_chunk_generation)
         WHERE id = NEW.id;
     END;
     CREATE TRIGGER qa_chunks_update_generation
     AFTER UPDATE OF id, session_id, content ON qa_chunks BEGIN
         UPDATE qa_chunk_generation SET value = value + 1 WHERE singleton = 1;
         UPDATE qa_chunks SET generation = (SELECT value FROM qa_chunk_generation)
         WHERE id = NEW.id;
     END;";

struct Shape<'a> {
    sessions: &'a str,
    tokenizer: &'a str,
    chunks: &'a str,
    vectors: &'a str,
    support_tables: &'a str,
}

const V0_14: Shape<'static> = Shape {
    sessions: SESSIONS_V0_14,
    tokenizer: FTS_TOKENIZER,
    chunks: CHUNKS_V0_14,
    vectors: VECTORS_V0_14,
    support_tables: SUPPORT_TABLES_V0_14,
};

fn build_index(path: &Path, shape: &Shape<'_>) {
    ensure_sqlite_vec().map_err(|e| anyhow::anyhow!(e)).unwrap();
    let conn = Connection::open(path).unwrap();
    conn.execute_batch(&format!(
        "CREATE TABLE sessions ({});
         CREATE VIRTUAL TABLE messages USING fts5(
             session_id UNINDEXED, role, text, tokenize='{}'
         );
         CREATE TABLE qa_chunks ({});
         CREATE VIRTUAL TABLE vec_chunks USING vec0(
             embedding FLOAT[{EMBEDDING_DIMS}], {}
         );
         CREATE VIRTUAL TABLE messages_vocab USING fts5vocab(messages, row);
         {}",
        shape.sessions, shape.tokenizer, shape.chunks, shape.vectors, shape.support_tables
    ))
    .unwrap();
}

fn state_of(shape: &Shape<'_>) -> SchemaState {
    let tmp = NamedTempFile::new().unwrap();
    build_index(tmp.path(), shape);
    schema_state(&Connection::open(tmp.path()).unwrap()).unwrap()
}

fn without(columns: &str, column: &str) -> String {
    assert!(
        columns.contains(column),
        "the fixture has no `{column}` to remove"
    );
    columns.replace(column, "")
}

fn state_after(statement: &str) -> SchemaState {
    let tmp = NamedTempFile::new().unwrap();
    build_index(tmp.path(), &V0_14);
    let conn = Connection::open(tmp.path()).unwrap();
    conn.execute_batch(statement).unwrap();
    schema_state(&conn).unwrap()
}

#[test]
fn an_index_shaped_like_v0_14_0_has_a_current_schema() {
    assert_eq!(state_of(&V0_14), SchemaState::Current);
}

#[test]
fn an_index_whose_sessions_lack_a_type_is_unsupported() {
    let sessions = without(SESSIONS_V0_14, ", session_type TEXT");
    assert_eq!(
        state_of(&Shape {
            sessions: &sessions,
            ..V0_14
        }),
        SchemaState::Unsupported
    );
}

#[test]
fn an_index_whose_sessions_lack_a_scanned_files_marker_is_unsupported() {
    let sessions = without(SESSIONS_V0_14, ", files_scanned INTEGER");
    assert_eq!(
        state_of(&Shape {
            sessions: &sessions,
            ..V0_14
        }),
        SchemaState::Unsupported
    );
}

#[test]
fn an_index_whose_sessions_lack_a_file_size_is_unsupported() {
    let sessions = without(SESSIONS_V0_14, ", file_size INTEGER");
    assert_eq!(
        state_of(&Shape {
            sessions: &sessions,
            ..V0_14
        }),
        SchemaState::Unsupported
    );
}

#[test]
fn an_index_whose_sessions_lack_a_chunk_marker_is_unsupported() {
    let sessions = without(SESSIONS_V0_14, ", chunks_indexed INTEGER");
    assert_eq!(
        state_of(&Shape {
            sessions: &sessions,
            ..V0_14
        }),
        SchemaState::Unsupported
    );
}

#[test]
fn an_index_tokenized_without_trigrams_is_unsupported() {
    assert_eq!(
        state_of(&Shape {
            tokenizer: "unicode61",
            ..V0_14
        }),
        SchemaState::Unsupported
    );
}

#[test]
fn an_index_whose_chunks_still_carry_a_hash_is_unsupported() {
    let chunks = format!("{CHUNKS_V0_14}, chunk_hash TEXT");
    assert_eq!(
        state_of(&Shape {
            chunks: &chunks,
            ..V0_14
        }),
        SchemaState::Unsupported
    );
}

#[test]
fn an_index_whose_chunks_lack_a_message_range_is_unsupported() {
    let chunks = without(CHUNKS_V0_14, ", src_rowid_lo INTEGER, src_rowid_hi INTEGER");
    assert_eq!(
        state_of(&Shape {
            chunks: &chunks,
            ..V0_14
        }),
        SchemaState::Unsupported
    );
}

#[test]
fn an_index_whose_chunks_lack_a_generation_is_unsupported() {
    let chunks = without(CHUNKS_V0_14, ", generation INTEGER NOT NULL DEFAULT 0");
    assert_eq!(
        state_of(&Shape {
            chunks: &chunks,
            support_tables: "",
            ..V0_14
        }),
        SchemaState::Unsupported
    );
}

#[test]
fn an_index_whose_embeddings_lack_a_sub_index_is_unsupported() {
    assert_eq!(
        state_of(&Shape {
            vectors: "+chunk_id INTEGER",
            ..V0_14
        }),
        SchemaState::Unsupported
    );
}

#[test]
fn an_index_without_its_vocabulary_table_has_an_incomplete_schema() {
    assert_eq!(
        state_after("DROP TABLE messages_vocab"),
        SchemaState::Incomplete
    );
}

#[test]
fn an_index_without_its_parse_diagnostics_table_has_an_incomplete_schema() {
    assert_eq!(
        state_after("DROP TABLE parse_diagnostics"),
        SchemaState::Incomplete
    );
}

#[test]
fn an_index_without_its_generation_counter_has_an_incomplete_schema() {
    assert_eq!(
        state_after("DROP TABLE qa_chunk_generation"),
        SchemaState::Incomplete
    );
}

#[test]
fn a_current_sessions_table_alone_has_an_incomplete_schema() {
    let tmp = NamedTempFile::new().unwrap();
    let conn = Connection::open(tmp.path()).unwrap();
    conn.execute_batch(&format!("CREATE TABLE sessions ({SESSIONS_V0_14})"))
        .unwrap();
    assert_eq!(schema_state(&conn).unwrap(), SchemaState::Incomplete);
}

#[test]
fn an_older_sessions_table_alone_is_unsupported() {
    let tmp = NamedTempFile::new().unwrap();
    let conn = Connection::open(tmp.path()).unwrap();
    conn.execute_batch("CREATE TABLE sessions (session_id TEXT PRIMARY KEY)")
        .unwrap();
    assert_eq!(schema_state(&conn).unwrap(), SchemaState::Unsupported);
}

#[test]
fn an_older_embedding_table_without_a_sessions_table_is_unsupported() {
    let tmp = NamedTempFile::new().unwrap();
    let conn = Connection::open(tmp.path()).unwrap();
    conn.execute_batch("CREATE TABLE vec_chunks (chunk_id INTEGER, embedding BLOB)")
        .unwrap();
    assert_eq!(schema_state(&conn).unwrap(), SchemaState::Unsupported);
}

const SESSIONS_V0_13: &str = "session_id TEXT PRIMARY KEY, source TEXT, file_path TEXT, \
     project TEXT, slug TEXT, timestamp INTEGER, mtime REAL, session_type TEXT, \
     files_scanned INTEGER";
const CHUNKS_V0_13: &str = "id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, \
     content TEXT NOT NULL, timestamp INTEGER, src_rowid_lo INTEGER, src_rowid_hi INTEGER";

fn build_v0_13_index(path: &Path) {
    build_index(
        path,
        &Shape {
            sessions: SESSIONS_V0_13,
            chunks: CHUNKS_V0_13,
            support_tables: "INSERT INTO sessions (session_id) VALUES ('s1');",
            ..V0_14
        },
    );
}

#[test]
fn opening_a_v0_13_index_for_writing_is_refused_and_leaves_its_schema_rows_and_journal_mode_as_they_were()
 {
    let tmp = NamedTempFile::new().unwrap();
    build_v0_13_index(tmp.path());
    let before = schema_snapshot(&Connection::open(tmp.path()).unwrap());

    let error = open_db(tmp.path()).err().unwrap();

    let conn = Connection::open(tmp.path()).unwrap();
    assert_eq!(
        (
            error.to_string(),
            schema_snapshot(&conn),
            collect_strings(&conn, "SELECT session_id FROM sessions"),
            conn.query_row("PRAGMA journal_mode", [], |r| r.get::<_, String>(0))
                .unwrap()
        ),
        (
            "this index's schema predates recall 0.14.0 and cannot be opened; build a new index \
             with `recall index` at another `--db-path`, or upgrade this one by running \
             `recall classify` from recall 0.16.1 on it"
                .to_owned(),
            before,
            vec!["s1".to_owned()],
            "delete".to_owned()
        )
    );
}

fn schema_object_names(conn: &Connection) -> Vec<String> {
    collect_strings(
        conn,
        "SELECT type || ' ' || name FROM sqlite_master ORDER BY type, name",
    )
}

#[test]
fn opening_an_index_whose_create_stopped_after_the_sessions_table_finishes_the_create() {
    let fresh = NamedTempFile::new().unwrap();
    let complete = schema_object_names(&open_db(fresh.path()).unwrap());
    let tmp = NamedTempFile::new().unwrap();
    Connection::open(tmp.path())
        .unwrap()
        .execute_batch(&format!(
            "CREATE TABLE sessions ({SESSIONS_V0_14}, message_rowid_lo INTEGER, \
             message_rowid_hi INTEGER)"
        ))
        .unwrap();

    let conn = open_db(tmp.path()).unwrap();

    assert_eq!(
        (schema_state(&conn).unwrap(), schema_object_names(&conn)),
        (SchemaState::Current, complete)
    );
}

#[test]
fn opening_a_v0_14_index_for_writing_adds_both_rowid_ranges_and_keeps_its_rows() {
    let tmp = NamedTempFile::new().unwrap();
    build_index(tmp.path(), &V0_14);
    let conn = Connection::open(tmp.path()).unwrap();
    seed_session(&conn, "s1");
    seed_chunk(&conn, 1, "kept chunk");
    drop(conn);

    let conn = open_db(tmp.path()).unwrap();

    assert_eq!(
        (
            schema_state(&conn).unwrap(),
            collect_strings(
                &conn,
                "SELECT name FROM pragma_table_info('sessions') WHERE name LIKE '%rowid%' \
                 UNION ALL \
                 SELECT name FROM pragma_table_info('qa_chunks') WHERE name LIKE 'vec_rowid%'"
            ),
            collect_strings(&conn, "SELECT session_id FROM sessions"),
            collect_strings(&conn, "SELECT content FROM qa_chunks"),
        ),
        (
            SchemaState::Current,
            vec![
                "message_rowid_lo".to_owned(),
                "message_rowid_hi".to_owned(),
                "vec_rowid_lo".to_owned(),
                "vec_rowid_hi".to_owned()
            ],
            vec!["s1".to_owned()],
            vec!["kept chunk".to_owned()]
        )
    );
}

#[test]
fn opening_an_index_that_still_has_the_update_generation_trigger_drops_it() {
    let tmp = NamedTempFile::new().unwrap();
    build_index(tmp.path(), &V0_14);

    let conn = open_db(tmp.path()).unwrap();

    assert_eq!(
        collect_strings(
            &conn,
            "SELECT name FROM sqlite_master WHERE type = 'trigger' ORDER BY name"
        ),
        ["qa_chunks_insert_generation"]
    );
}

#[test]
fn every_inserted_chunk_gets_the_next_generation_and_a_message_range_update_keeps_it() {
    let tmp = NamedTempFile::new().unwrap();
    let conn = open_db(tmp.path()).unwrap();
    seed_chunk(&conn, 1, "first");
    seed_chunk(&conn, 2, "second");

    conn.execute(
        "UPDATE qa_chunks SET src_rowid_lo = 1, src_rowid_hi = 2",
        [],
    )
    .unwrap();

    assert_eq!(
        collect_strings(&conn, CHUNK_ROWS),
        [
            r#"[1,"s1","first",0,1,2,1]"#,
            r#"[2,"s1","second",0,1,2,2]"#
        ]
    );
    conn.execute("DELETE FROM qa_chunks WHERE id = 2", [])
        .unwrap();
    seed_chunk(&conn, 2, "second");
    assert_eq!(
        collect_strings(&conn, CHUNK_ROWS),
        [
            r#"[1,"s1","first",0,1,2,1]"#,
            r#"[2,"s1","second",0,null,null,3]"#
        ]
    );
}

#[test]
fn a_database_without_a_sessions_table_has_an_empty_schema() {
    let tmp = NamedTempFile::new().unwrap();
    let conn = Connection::open(tmp.path()).unwrap();
    assert_eq!(schema_state(&conn).unwrap(), SchemaState::Empty);
}

const WRITABLE_DIR_NOTE: &str = "read-only open cannot see uncheckpointed changes in the \
     write-ahead log; run `recall index` to checkpoint them.";

fn file_names(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .collect();
    names.sort();
    names
}

#[test]
fn stale_snapshot_note_in_a_writable_directory_asks_to_run_recall_index_and_leaves_no_probe_file() {
    let dir = tempfile::TempDir::new().unwrap();
    let db = dir.path().join("recall.db");
    fs::write(wal_path(&db), b"uncommitted").unwrap();

    let note = stale_wal_note(&db, OpenTier::Immutable);

    assert_eq!(
        (note.as_deref(), file_names(dir.path())),
        (Some(WRITABLE_DIR_NOTE), vec!["recall.db-wal".to_owned()])
    );
}

#[test]
fn a_read_that_sees_the_write_ahead_log_gets_no_stale_snapshot_note() {
    let tmp = NamedTempFile::new().unwrap();
    let wal = wal_path(tmp.path());
    fs::write(&wal, b"uncommitted").unwrap();
    let note = stale_wal_note(tmp.path(), OpenTier::Direct);
    fs::remove_file(&wal).ok();
    assert!(
        note.is_none(),
        "direct tier reads -wal fresh; no false positive"
    );
}

#[test]
fn a_read_from_the_last_checkpoint_gets_no_note_when_the_write_ahead_log_is_missing_or_empty() {
    let tmp = NamedTempFile::new().unwrap();
    assert!(stale_wal_note(tmp.path(), OpenTier::Immutable).is_none());
    let wal = wal_path(tmp.path());
    fs::write(&wal, b"").unwrap();
    let note = stale_wal_note(tmp.path(), OpenTier::Immutable);
    fs::remove_file(&wal).ok();
    assert!(note.is_none(), "empty -wal carries no uncommitted changes");
}

#[cfg(unix)]
#[test]
fn stale_snapshot_note_in_a_read_only_directory_asks_to_index_a_copy_with_its_sidecar_files() {
    use std::os::unix::fs::PermissionsExt;

    let dir = tempfile::TempDir::new().unwrap();
    let db = dir.path().join("recall.db");
    fs::write(wal_path(&db), b"uncommitted").unwrap();
    fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o555)).unwrap();

    if root_skip::skip_if_root(dir.path()) {
        return;
    }

    let note = stale_wal_note(&db, OpenTier::Immutable);
    fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o755)).unwrap();

    assert_eq!(note.as_deref(), Some(READ_ONLY_DIR_NOTE));
}

#[cfg(unix)]
#[test]
fn stale_snapshot_note_in_a_directory_that_refuses_the_owner_new_files_asks_to_index_a_copy() {
    use std::os::unix::fs::PermissionsExt;

    let dir = tempfile::TempDir::new().unwrap();
    let db = dir.path().join("recall.db");
    fs::write(wal_path(&db), b"uncommitted").unwrap();
    let owner_unwritable_but_group_writable = 0o575;
    fs::set_permissions(
        dir.path(),
        fs::Permissions::from_mode(owner_unwritable_but_group_writable),
    )
    .unwrap();

    if root_skip::skip_if_root(dir.path()) {
        return;
    }

    let note = stale_wal_note(&db, OpenTier::Immutable);
    fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o755)).unwrap();

    assert_eq!(note.as_deref(), Some(READ_ONLY_DIR_NOTE));
}

#[test]
fn stale_snapshot_note_in_a_writable_directory_with_a_leftover_write_probe_file_asks_to_run_recall_index()
 {
    let dir = tempfile::TempDir::new().unwrap();
    let db = dir.path().join("recall.db");
    fs::write(wal_path(&db), b"uncommitted").unwrap();
    let leftover = write_probe_path(dir.path());
    fs::write(&leftover, b"stale").unwrap();

    let note = stale_wal_note(&db, OpenTier::Immutable);

    assert_eq!(
        (note.as_deref(), fs::read_to_string(&leftover).unwrap()),
        (Some(WRITABLE_DIR_NOTE), "stale".to_owned())
    );
}

#[test]
fn a_read_only_open_in_a_writable_directory_opens_the_live_database_and_rejects_writes() {
    let dir = tempfile::TempDir::new().unwrap();
    let db = dir.path().join("recall.db");
    let uncheckpointed_writer = open_db(&db).unwrap();
    seed_session(&uncheckpointed_writer, "s1");

    let (conn, tier) = open_db_readonly(&db).unwrap();
    let error = conn
        .execute("CREATE TABLE probe_write (x INTEGER)", [])
        .unwrap_err();

    assert_eq!(
        (
            tier,
            collect_strings(&conn, "SELECT session_id FROM sessions"),
            error.to_string()
        ),
        (
            OpenTier::Direct,
            vec!["s1".to_owned()],
            "attempt to write a readonly database".to_owned()
        )
    );
}

#[test]
fn a_read_only_open_of_a_missing_database_fails_without_creating_any_file() {
    let dir = tempfile::TempDir::new().unwrap();
    let missing = dir.path().join("nope.db");

    let error = open_db_readonly(&missing).err().unwrap();

    assert_eq!(
        (error.to_string(), file_names(dir.path())),
        (
            format!("unable to open database file: {}", missing.display()),
            Vec::<String>::new()
        )
    );
}

#[test]
fn opening_an_existing_index_for_writing_leaves_its_schema_and_journal_mode_as_they_were() {
    let tmp = NamedTempFile::new().unwrap();
    build_v0_13_index(tmp.path());
    let before = schema_snapshot(&Connection::open(tmp.path()).unwrap());

    let conn = open_existing_db(tmp.path()).unwrap();

    assert_eq!(
        (
            schema_snapshot(&conn),
            conn.query_row("PRAGMA journal_mode", [], |r| r.get::<_, String>(0))
                .unwrap()
        ),
        (before, "delete".to_owned())
    );
}

#[test]
fn opening_a_missing_index_for_writing_fails_without_creating_any_file() {
    let dir = tempfile::TempDir::new().unwrap();
    let missing = dir.path().join("nope.db");

    open_existing_db(&missing).unwrap_err();

    assert_eq!(file_names(dir.path()), Vec::<String>::new());
}

#[cfg(unix)]
#[test]
fn a_read_only_open_of_a_missing_database_in_a_read_only_directory_fails_with_the_path_it_was_given()
 {
    use std::os::unix::fs::PermissionsExt;

    let dir = tempfile::TempDir::new().unwrap();
    let missing = dir.path().join("nope.db");
    fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o555)).unwrap();

    if root_skip::skip_if_root(dir.path()) {
        return;
    }

    let result = open_db_readonly(&missing);
    fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o755)).unwrap();

    assert_eq!(
        result.err().unwrap().to_string(),
        format!("unable to open database file: {}", missing.display())
    );
}

#[cfg(unix)]
#[test]
fn a_database_in_a_read_only_directory_whose_name_holds_uri_delimiters_opens_from_its_last_checkpoint()
 {
    use std::os::unix::fs::PermissionsExt;

    let root = tempfile::TempDir::new().unwrap();
    let uri_delimiters_and_a_multibyte_character = "a%20b#c?d日";
    let dir = root.path().join(uri_delimiters_and_a_multibyte_character);
    fs::create_dir(&dir).unwrap();
    let db = dir.join("recall.db");
    seed_session(&open_db(&db).unwrap(), "s1");
    fs::set_permissions(&dir, fs::Permissions::from_mode(0o555)).unwrap();

    if root_skip::skip_if_root(&dir) {
        return;
    }

    let opened = open_db_readonly(&db);
    fs::set_permissions(&dir, fs::Permissions::from_mode(0o755)).unwrap();

    let (conn, tier) = opened.unwrap();
    assert_eq!(
        (
            tier,
            collect_strings(&conn, "SELECT session_id FROM sessions")
        ),
        (OpenTier::Immutable, vec!["s1".to_owned()])
    );
}

#[cfg(unix)]
#[test]
fn failing_to_remove_the_write_probe_warns_with_its_path_and_the_error() {
    use std::sync::Mutex;
    use tracing::subscriber::with_default;

    let dir = tempfile::TempDir::new().unwrap();
    let log = dir.path().join("log");
    let subscriber = tracing_subscriber::fmt()
        .with_env_filter(crate::LOG_FILTER_DEFAULT)
        .with_writer(Mutex::new(fs::File::create(&log).unwrap()))
        .with_ansi(false)
        .without_time()
        .finish();

    with_default(subscriber, || {
        remove_write_probe(Path::new("/nonexistent-recall-dir/.recall-write-probe-0"));
    });

    assert_eq!(
        fs::read_to_string(&log).unwrap(),
        " WARN recall::db: failed to remove write probe \
         probe=/nonexistent-recall-dir/.recall-write-probe-0 \
         error=No such file or directory (os error 2)\n"
    );
}

#[cfg(target_os = "macos")]
fn shm_path(path: &Path) -> PathBuf {
    let mut s = path.as_os_str().to_owned();
    s.push("-shm");
    PathBuf::from(s)
}

#[cfg(target_os = "macos")]
struct ReadOnlyMount {
    mountpoint: tempfile::TempDir,
    _dmg_dir: tempfile::TempDir,
}

#[cfg(target_os = "macos")]
impl Drop for ReadOnlyMount {
    fn drop(&mut self) {
        use std::process::Command;

        match Command::new("hdiutil")
            .args(["detach", "-force", "-quiet"])
            .arg(self.mountpoint.path())
            .status()
        {
            Ok(status) if status.success() => {}
            result => eprintln!(
                "hdiutil detach {} failed ({result:?}); the recalltest volume may stay mounted",
                self.mountpoint.path().display()
            ),
        }
    }
}

#[cfg(target_os = "macos")]
impl ReadOnlyMount {
    fn db_path(&self) -> PathBuf {
        self.mountpoint.path().join("recall.db")
    }
}

#[cfg(target_os = "macos")]
fn attach_readonly_wal_only_db() -> ReadOnlyMount {
    use std::process::Command;

    let stage = tempfile::TempDir::new().unwrap();
    let src_dir = stage.path().join("src");
    fs::create_dir(&src_dir).unwrap();
    let db_path = src_dir.join("recall.db");
    {
        let conn = open_db(&db_path).unwrap();
        drop(conn);
    }
    fs::remove_file(shm_path(&db_path)).ok();
    fs::write(wal_path(&db_path), b"synthetic-uncommitted-wal-bytes").unwrap();

    let dmg_path = stage.path().join("recall.dmg");
    let create = Command::new("hdiutil")
        .args(["create", "-volname", "recalltest", "-srcfolder"])
        .arg(&src_dir)
        .args(["-fs", "HFS+", "-format", "UDZO", "-ov"])
        .arg(&dmg_path)
        .output()
        .expect("failed to spawn hdiutil create");
    assert!(
        create.status.success(),
        "hdiutil create failed: {}",
        String::from_utf8_lossy(&create.stderr)
    );

    let mountpoint = tempfile::TempDir::new().unwrap();
    let attach = Command::new("hdiutil")
        .args(["attach", "-readonly", "-nobrowse", "-mountpoint"])
        .arg(mountpoint.path())
        .arg(&dmg_path)
        .output()
        .expect("failed to spawn hdiutil attach");
    assert!(
        attach.status.success(),
        "hdiutil attach failed: {}",
        String::from_utf8_lossy(&attach.stderr)
    );

    ReadOnlyMount {
        mountpoint,
        _dmg_dir: stage,
    }
}

#[cfg(target_os = "macos")]
#[test]
fn a_database_on_a_read_only_mount_without_a_shared_memory_file_opens_from_its_last_checkpoint() {
    let mount = attach_readonly_wal_only_db();

    let (conn, tier) = open_db_readonly(&mount.db_path())
        .expect("open_db_readonly must succeed via the immutable fallback");
    assert_eq!(
        tier,
        OpenTier::Immutable,
        "a read-only mount without -shm must select the immutable tier"
    );

    conn.query_row("SELECT count(*) FROM sqlite_master", [], |_| Ok(()))
        .expect("sqlite_master read must succeed (no SQLITE_CANTOPEN error 14)");
}

#[cfg(target_os = "macos")]
#[test]
fn stale_wal_note_on_a_read_only_mount_steers_to_the_copy_based_recall_index_db_path_remedy_instead_of_a_bare_recall_index()
 {
    let mount = attach_readonly_wal_only_db();

    let note = stale_wal_note(&mount.db_path(), OpenTier::Immutable);

    assert_eq!(note.as_deref(), Some(READ_ONLY_DIR_NOTE));
}

#[test]
fn a_read_only_open_under_a_missing_directory_fails_without_creating_the_directory() {
    let dir = tempfile::TempDir::new().unwrap();
    let db = dir.path().join("nonexistent_subdir").join("recall.db");

    let error = open_db_readonly(&db).err().unwrap();

    assert_eq!(
        (error.to_string(), file_names(dir.path())),
        (
            format!("unable to open database file: {}", db.display()),
            Vec::<String>::new()
        )
    );
}

#[test]
fn a_path_without_a_parent_component_stays_classified_writable() {
    assert!(
        !dir_is_unwritable(Path::new("/")),
        "the filesystem root (no parent) must not be classified unwritable"
    );
}

#[cfg(unix)]
#[path = "../../tests/support/root_skip.rs"]
mod root_skip;

#[cfg(unix)]
#[test]
fn root_check_treats_a_directory_it_can_write_as_root_and_sets_its_mode_to_0o755() {
    use std::os::unix::fs::PermissionsExt;

    let dir = tempfile::TempDir::new().unwrap();

    let got = root_skip::skip_if_root(dir.path());

    assert!(
        got,
        "create succeeding (root, or here a still-writable dir) must report true"
    );
    assert_eq!(
        fs::read_dir(dir.path()).unwrap().count(),
        0,
        "no entry (probe or otherwise) may remain directly under dir"
    );
    let mode = fs::metadata(dir.path()).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode, 0o755, "dir's write bits must be restored to 0o755");
}

#[cfg(unix)]
#[test]
fn the_unit_suite_fails_when_run_as_root() {
    root_skip::assert_root_canary();
}

#[cfg(unix)]
#[test]
fn db_tests_leave_the_root_probe_file_name_to_the_shared_root_check() {
    root_skip::assert_probe_literal_not_inlined(include_str!("tests.rs"), "src/db/tests.rs");
}

#[test]
fn deleting_one_chunk_vector_reads_only_the_row_with_that_rowid() {
    let (_dir, conn) = setup_test_db();

    let plan: Vec<String> = conn
        .prepare(&format!("EXPLAIN QUERY PLAN {DELETE_CHUNK_VECTOR}"))
        .unwrap()
        .query_map([1, 1], |r| r.get("detail"))
        .unwrap()
        .map(Result::unwrap)
        .collect();

    assert!(
        plan.iter()
            .any(|d| d.contains("vec_chunks VIRTUAL TABLE INDEX")),
        "{plan:?}"
    );
    assert!(!plan.iter().any(|d| d.contains("INDEX 0:")), "{plan:?}");
}

const VECTOR_RANGES: &str =
    "SELECT json_array(id, vec_rowid_lo, vec_rowid_hi) FROM qa_chunks ORDER BY id";

#[test]
fn upgrading_an_index_without_vector_ranges_stores_each_range_and_returns_gapped_chunks_to_inference()
 {
    let tmp = NamedTempFile::new().unwrap();
    let conn = open_db(tmp.path()).unwrap();
    seed_session(&conn, "s1");
    for (id, content) in [
        (1, "two vectors"),
        (2, "split vectors"),
        (3, "between"),
        (4, "inside the orphan"),
    ] {
        seed_chunk(&conn, id, content);
    }
    let emb = f32_as_bytes(&[0.1f32; EMBEDDING_DIMS]).to_vec();
    for (rowid, chunk_id, sub_idx) in [
        (1, 1, 0),
        (2, 1, 1),
        (3, 2, 0),
        (4, 3, 0),
        (5, 2, 1),
        (6, 99, 0),
        (7, 4, 0),
        (8, 99, 1),
    ] {
        conn.execute(
            "INSERT INTO vec_chunks (rowid, chunk_id, sub_idx, embedding) VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![rowid, chunk_id, sub_idx, emb],
        )
        .unwrap();
    }
    conn.execute_batch(
        "ALTER TABLE qa_chunks DROP COLUMN vec_rowid_lo;
         ALTER TABLE qa_chunks DROP COLUMN vec_rowid_hi;",
    )
    .unwrap();
    drop(conn);

    let conn = open_db(tmp.path()).unwrap();

    assert_eq!(
        collect_strings(&conn, VECTOR_RANGES),
        ["[1,1,2]", "[2,null,null]", "[3,4,4]", "[4,7,7]"]
    );
    assert_eq!(
        collect_strings(
            &conn,
            "SELECT json_array(rowid, chunk_id) FROM vec_chunks ORDER BY rowid"
        ),
        ["[1,1]", "[2,1]", "[4,3]", "[6,99]", "[7,4]", "[8,99]"]
    );
    assert_eq!(pending_count(&conn).unwrap(), 1);
    let (schema, ranges, vectors) = (
        schema_snapshot(&conn),
        collect_strings(&conn, VECTOR_RANGES),
        embeddings(&conn),
    );
    drop(conn);

    let reopened = open_db(tmp.path()).unwrap();

    assert_eq!(
        (
            schema_snapshot(&reopened),
            collect_strings(&reopened, VECTOR_RANGES),
            embeddings(&reopened)
        ),
        (schema, ranges, vectors)
    );
}
