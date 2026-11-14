use super::*;
use crate::embedder::f32_as_bytes;
use tempfile::NamedTempFile;

#[test]
fn test_open_db_creates_schema() {
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
        [
            "idx_qa_chunks_session",
            "idx_sessions_project",
            "idx_sessions_timestamp"
        ]
    );
    assert_eq!(schema_state(&conn).unwrap(), SchemaState::Current);
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
fn test_open_db_idempotent() {
    let tmp = NamedTempFile::new().unwrap();
    let first = schema_snapshot(&open_db(tmp.path()).unwrap());

    let second = schema_snapshot(&open_db(tmp.path()).unwrap());

    assert_eq!(second, first);
}

#[test]
fn upgrading_an_index_without_chunk_completion_markers_rolls_back_a_failed_upgrade_then_marks_only_chunked_sessions()
 {
    use crate::indexer::index_chunks;

    let tmp = NamedTempFile::new().unwrap();
    let conn = open_db(tmp.path()).unwrap();
    // The marker upgrade is the only one that writes chunks_indexed, so the trigger fails it alone.
    conn.execute_batch(
        "INSERT INTO sessions (session_id) VALUES ('s1'), ('empty'), ('pending');
         INSERT INTO messages (session_id, role, text) VALUES
             ('empty', 'assistant', 'unpaired answer'), ('pending', 'user', 'new question');
         INSERT INTO qa_chunks (id, session_id, content, timestamp) VALUES (1, 's1', 'q\na', 0);
         ALTER TABLE sessions DROP COLUMN chunks_indexed;
         CREATE TRIGGER fail_marker_upgrade BEFORE UPDATE OF chunks_indexed ON sessions
         BEGIN SELECT RAISE(ABORT, 'injected migration failure'); END;",
    )
    .unwrap();
    conn.execute(
        "INSERT INTO vec_chunks (embedding, chunk_id, sub_idx) VALUES (?1, 1, 0)",
        [f32_as_bytes(&[0.1f32; EMBEDDING_DIMS])],
    )
    .unwrap();
    let schema = schema_snapshot(&conn);
    drop(conn);

    let error = open_db(tmp.path()).err().unwrap();

    let conn = Connection::open(tmp.path()).unwrap();
    assert_eq!(
        (error.to_string(), schema_snapshot(&conn)),
        ("injected migration failure".to_owned(), schema)
    );
    conn.execute_batch("DROP TRIGGER fail_marker_upgrade;")
        .unwrap();
    drop(conn);

    let mut conn = open_db(tmp.path()).unwrap();

    assert_eq!(
        collect_strings(
            &conn,
            "SELECT json_array(session_id, chunks_indexed) FROM sessions ORDER BY session_id"
        ),
        [r#"["empty",null]"#, r#"["pending",null]"#, r#"["s1",1]"#]
    );
    let first = index_chunks(&mut conn).unwrap();
    drop(conn);
    let mut conn = open_db(tmp.path()).unwrap();
    let second = index_chunks(&mut conn).unwrap();

    assert_eq!(
        [
            (first.sessions_chunked, first.chunks_created),
            (second.sessions_chunked, second.chunks_created),
        ],
        [(2, 1), (0, 0)]
    );
    assert_eq!(
        collect_strings(
            &conn,
            "SELECT json_array(session_id, chunks_indexed) FROM sessions ORDER BY session_id"
        ),
        [r#"["empty",0]"#, r#"["pending",0]"#, r#"["s1",1]"#]
    );
    assert_eq!(
        collect_strings(&conn, CHUNK_ROWS),
        [
            r#"[1,"s1","q\na",0,null,null,1]"#,
            r#"[2,"pending","new question",null,2,2,2]"#
        ]
    );
    assert_eq!(
        embeddings(&conn),
        [(1, 1, 0, f32_as_bytes(&[0.1f32; EMBEDDING_DIMS]).to_vec())]
    );
}

#[test]
fn a_failed_parse_diagnostics_upgrade_leaves_the_schema_and_the_session_times_as_they_were() {
    let tmp = NamedTempFile::new().unwrap();
    let conn = open_db(tmp.path()).unwrap();
    seed_session(&conn, "s1");
    conn.execute_batch(
        "DROP TABLE parse_diagnostics;
         CREATE TRIGGER fail_upgrade BEFORE UPDATE OF mtime ON sessions
         BEGIN SELECT RAISE(ABORT, 'injected migration failure'); END;",
    )
    .unwrap();
    let schema = schema_snapshot(&conn);
    drop(conn);

    let error = open_db(tmp.path()).err().unwrap();

    let conn = Connection::open(tmp.path()).unwrap();
    assert_eq!(
        (
            error.to_string(),
            schema_snapshot(&conn),
            collect_strings(&conn, "SELECT json_array(session_id, mtime) FROM sessions"),
        ),
        (
            "injected migration failure".to_owned(),
            schema,
            vec![r#"["s1",0.0]"#.to_owned()],
        )
    );
}

#[test]
fn test_session_type_migration_adds_column_preserving_rows() {
    let tmp = NamedTempFile::new().unwrap();
    {
        let conn = Connection::open(tmp.path()).unwrap();
        conn.execute_batch(&format!(
            "CREATE TABLE sessions (
                    session_id TEXT PRIMARY KEY, source TEXT, file_path TEXT,
                    project TEXT, slug TEXT, timestamp INTEGER, mtime REAL
                );
                CREATE VIRTUAL TABLE messages USING fts5(
                    session_id UNINDEXED, role, text, tokenize='{FTS_TOKENIZER}'
                );"
        ))
        .unwrap();
        conn.execute(
            "INSERT INTO sessions VALUES ('s1', 'claude', '/f', '/p', 'slug', 0, 0.0)",
            [],
        )
        .unwrap();
    }

    let conn = open_db(tmp.path()).unwrap();

    assert_eq!(
        collect_strings(&conn, SESSION_ROWS),
        [r#"["s1","claude","/f","/p","slug",0,null,null,null,null,null]"#]
    );
}

const SESSION_ROWS: &str = "SELECT json_array(session_id, source, file_path, project, slug, \
     timestamp, mtime, file_size, session_type, files_scanned, chunks_indexed) FROM sessions \
     ORDER BY session_id";

/// The schema version with every schema object, so a test can show that an open changed nothing.
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

/// Every stored embedding as `(rowid, chunk_id, sub_idx, embedding)`.
fn embeddings(conn: &Connection) -> Vec<(i64, i64, i64, Vec<u8>)> {
    conn.prepare("SELECT rowid, chunk_id, sub_idx, embedding FROM vec_chunks ORDER BY rowid")
        .unwrap()
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))
        .unwrap()
        .map(Result::unwrap)
        .collect()
}

fn create_old_schema_db(path: &Path) {
    let conn = Connection::open(path).unwrap();
    conn.execute_batch(
        "CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY, source TEXT, file_path TEXT,
                project TEXT, slug TEXT, timestamp INTEGER, mtime REAL
            );
            CREATE VIRTUAL TABLE messages USING fts5(
                session_id UNINDEXED, role, text, tokenize='unicode61'
            );",
    )
    .unwrap();
    conn.execute(
        "INSERT INTO sessions VALUES ('s1', 'claude', '/f', '/p', 'slug', 0, 0.0)",
        [],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO messages (session_id, role, text) VALUES ('s1', 'user', 'hello')",
        [],
    )
    .unwrap();
}

#[test]
fn opening_an_index_with_an_older_tokenizer_moves_its_messages_onto_trigram_and_keeps_its_sessions()
{
    let tmp = NamedTempFile::new().unwrap();
    create_old_schema_db(tmp.path());
    // Rowids with gaps, as deleted sessions leave them; chunks find their messages by rowid.
    // Rowid 9 belongs to no session, so the copy leaves it behind.
    Connection::open(tmp.path())
        .unwrap()
        .execute_batch(
            "INSERT INTO messages (rowid, session_id, role, text)
                 VALUES (3, 's1', 'assistant', 'help is here'), (7, 's1', 'user', 'thanks'),
                        (9, 'gone', 'user', 'a message whose session row is gone');",
        )
        .unwrap();

    let conn = open_db(tmp.path()).unwrap();

    assert_eq!(
        collect_strings(&conn, SESSION_ROWS),
        [r#"["s1","claude","/f","/p","slug",0,null,null,null,null,null]"#]
    );
    assert_eq!(
        collect_strings(
            &conn,
            "SELECT json_array(rowid, session_id, role, text) FROM messages ORDER BY rowid"
        ),
        [
            r#"[1,"s1","user","hello"]"#,
            r#"[3,"s1","assistant","help is here"]"#,
            r#"[7,"s1","user","thanks"]"#
        ]
    );
    // A trigram index matches a word's inner three letters; the older tokenizers do not.
    assert_eq!(
        collect_strings(
            &conn,
            "SELECT json_array(rowid, session_id, role, text) FROM messages
             WHERE messages MATCH 'hel' ORDER BY rowid"
        ),
        [
            r#"[1,"s1","user","hello"]"#,
            r#"[3,"s1","assistant","help is here"]"#
        ]
    );
    let schema = schema_snapshot(&conn);
    drop(conn);

    let reopened = open_db(tmp.path()).unwrap();

    // A second open must not migrate the table again.
    assert_eq!(schema_snapshot(&reopened), schema);
}

#[test]
fn opening_the_oldest_index_shape_keeps_its_chunks_and_reaches_the_current_schema() {
    let tmp = NamedTempFile::new().unwrap();
    create_old_schema_db(tmp.path());
    Connection::open(tmp.path())
        .unwrap()
        .execute_batch(
            "CREATE TABLE qa_chunks (
                 id INTEGER PRIMARY KEY, session_id TEXT NOT NULL,
                 user_text TEXT NOT NULL, assistant_text TEXT,
                 content TEXT NOT NULL, timestamp INTEGER, chunk_hash TEXT NOT NULL
             );
             INSERT INTO qa_chunks VALUES (1, 's1', 'q', 'a', 'content', 0, 'hash1');",
        )
        .unwrap();

    let conn = open_db(tmp.path()).unwrap();

    assert!(matches!(schema_state(&conn), Ok(SchemaState::Current)));
    assert_eq!(
        conn.prepare("SELECT id, session_id, content FROM qa_chunks")
            .unwrap()
            .query_map([], |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                ))
            })
            .unwrap()
            .map(Result::unwrap)
            .collect::<Vec<_>>(),
        [(1, "s1".to_owned(), "content".to_owned())]
    );
    assert_eq!(
        conn.query_row("SELECT COUNT(*) FROM sessions", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        1
    );
}

fn create_pre_cleanup_db(path: &Path) {
    ensure_sqlite_vec().map_err(|e| anyhow::anyhow!(e)).unwrap();
    let conn = Connection::open(path).unwrap();
    conn.execute_batch(&format!(
        "CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY, source TEXT, file_path TEXT,
                project TEXT, slug TEXT, timestamp INTEGER, mtime REAL
            );
            CREATE VIRTUAL TABLE messages USING fts5(
                session_id UNINDEXED, role, text, tokenize='{FTS_TOKENIZER}'
            );
            CREATE TABLE qa_chunks (
                id INTEGER PRIMARY KEY, session_id TEXT NOT NULL,
                user_text TEXT NOT NULL, assistant_text TEXT,
                content TEXT NOT NULL, timestamp INTEGER, chunk_hash TEXT NOT NULL
            );
            CREATE INDEX idx_qa_chunks_hash ON qa_chunks(chunk_hash);
            CREATE TABLE embedded_chunk_ids (
                chunk_id INTEGER NOT NULL, sub_idx INTEGER NOT NULL,
                vec_rowid INTEGER NOT NULL, PRIMARY KEY (chunk_id, sub_idx)
            );
            CREATE VIRTUAL TABLE vec_chunks USING vec0(
                embedding FLOAT[{EMBEDDING_DIMS}], +chunk_id INTEGER, +sub_idx INTEGER
            );"
    ))
    .unwrap();
    conn.execute(
            "INSERT INTO qa_chunks (id, session_id, user_text, assistant_text, content, timestamp, chunk_hash) \
             VALUES (1, 's1', 'q', 'a', 'q\na', 0, 'deadbeef')",
            [],
        )
        .unwrap();
    let emb = vec![0.1f32; EMBEDDING_DIMS];
    conn.execute(
        "INSERT INTO vec_chunks (embedding, chunk_id, sub_idx) VALUES (?1, 1, 0)",
        [f32_as_bytes(&emb)],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO embedded_chunk_ids (chunk_id, sub_idx, vec_rowid) VALUES (1, 0, 1)",
        [],
    )
    .unwrap();
}

#[test]
fn test_qa_chunks_migration_drops_write_only_columns_preserving_embeddings() {
    let tmp = NamedTempFile::new().unwrap();
    create_pre_cleanup_db(tmp.path());

    let conn = open_db(tmp.path()).unwrap();
    let (schema, chunks, vectors) = (
        schema_snapshot(&conn),
        collect_strings(&conn, CHUNK_ROWS),
        embeddings(&conn),
    );

    assert_eq!(
        collect_strings(&conn, "SELECT name FROM pragma_table_info('qa_chunks')"),
        [
            "id",
            "session_id",
            "content",
            "timestamp",
            "src_rowid_lo",
            "src_rowid_hi",
            "generation",
        ]
    );
    assert_eq!(chunks, [r#"[1,"s1","q\na",0,null,null,0]"#]);
    assert_eq!(
        vectors,
        [(1, 1, 0, f32_as_bytes(&[0.1f32; EMBEDDING_DIMS]).to_vec())]
    );
    assert_eq!(
        collect_strings(
            &conn,
            "SELECT name FROM sqlite_master WHERE name = 'embedded_chunk_ids'"
        ),
        Vec::<String>::new()
    );
    drop(conn);

    let reopened = open_db(tmp.path()).unwrap();

    assert_eq!(
        (
            schema_snapshot(&reopened),
            collect_strings(&reopened, CHUNK_ROWS),
            embeddings(&reopened)
        ),
        (schema, chunks, vectors)
    );
}

const SESSIONS_FULL: &str = "session_id TEXT PRIMARY KEY, source TEXT, file_path TEXT, \
     project TEXT, slug TEXT, timestamp INTEGER, mtime REAL, session_type TEXT";
const SESSIONS_NO_TYPE: &str = "session_id TEXT PRIMARY KEY, source TEXT, file_path TEXT, \
     project TEXT, slug TEXT, timestamp INTEGER, mtime REAL";
const QA_NO_GENERATION: &str = "id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, content TEXT NOT NULL, \
     timestamp INTEGER, src_rowid_lo INTEGER, src_rowid_hi INTEGER";
const QA_FULL: &str = "id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, content TEXT NOT NULL, \
     timestamp INTEGER, src_rowid_lo INTEGER, src_rowid_hi INTEGER, generation INTEGER NOT NULL DEFAULT 0";
const QA_NO_ROWID: &str = "id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, content TEXT NOT NULL, timestamp INTEGER, generation INTEGER NOT NULL DEFAULT 0";
const QA_CHUNK_HASH: &str = "id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, content TEXT NOT NULL, \
     timestamp INTEGER, src_rowid_lo INTEGER, src_rowid_hi INTEGER, chunk_hash TEXT, generation INTEGER NOT NULL DEFAULT 0";
const VEC_FULL: &str = "+chunk_id INTEGER, +sub_idx INTEGER";
const VEC_NO_SUB: &str = "+chunk_id INTEGER";

fn build_schema(path: &Path, sessions_cols: &str, tokenizer: &str, qa_cols: &str, vec_cols: &str) {
    ensure_sqlite_vec().map_err(|e| anyhow::anyhow!(e)).unwrap();
    let conn = Connection::open(path).unwrap();
    conn.execute_batch(&format!(
        "CREATE TABLE sessions ({sessions_cols});
         CREATE VIRTUAL TABLE messages USING fts5(
             session_id UNINDEXED, role, text, tokenize='{tokenizer}'
         );
         CREATE TABLE qa_chunks ({qa_cols});
         CREATE VIRTUAL TABLE vec_chunks USING vec0(
             embedding FLOAT[{EMBEDDING_DIMS}], {vec_cols}
         );
         CREATE VIRTUAL TABLE messages_vocab USING fts5vocab(messages, row);"
    ))
    .unwrap();
}

#[test]
fn a_database_without_a_sessions_table_has_an_empty_schema() {
    let tmp = NamedTempFile::new().unwrap();
    let conn = Connection::open(tmp.path()).unwrap();
    assert_eq!(schema_state(&conn).unwrap(), SchemaState::Empty);
}

#[test]
fn a_database_with_only_a_sessions_table_has_a_stale_schema() {
    let tmp = NamedTempFile::new().unwrap();
    let conn = Connection::open(tmp.path()).unwrap();
    conn.execute_batch(&format!("CREATE TABLE sessions ({SESSIONS_FULL});"))
        .unwrap();
    assert_eq!(schema_state(&conn).unwrap(), SchemaState::Stale);
}

#[test]
fn an_index_without_its_vocabulary_table_has_a_stale_schema() {
    let tmp = NamedTempFile::new().unwrap();
    build_schema(tmp.path(), SESSIONS_FULL, FTS_TOKENIZER, QA_FULL, VEC_FULL);
    let conn = Connection::open(tmp.path()).unwrap();
    conn.execute_batch("DROP TABLE messages_vocab;").unwrap();
    assert_eq!(schema_state(&conn).unwrap(), SchemaState::Stale);
}

#[test]
fn an_index_whose_sessions_lack_a_type_column_has_a_stale_schema() {
    let tmp = NamedTempFile::new().unwrap();
    build_schema(
        tmp.path(),
        SESSIONS_NO_TYPE,
        FTS_TOKENIZER,
        QA_FULL,
        VEC_FULL,
    );
    let conn = Connection::open(tmp.path()).unwrap();
    assert_eq!(schema_state(&conn).unwrap(), SchemaState::Stale);
}

#[test]
fn an_index_whose_embeddings_lack_a_sub_index_has_a_stale_schema() {
    let tmp = NamedTempFile::new().unwrap();
    build_schema(
        tmp.path(),
        SESSIONS_FULL,
        FTS_TOKENIZER,
        QA_FULL,
        VEC_NO_SUB,
    );
    let conn = Connection::open(tmp.path()).unwrap();
    assert_eq!(schema_state(&conn).unwrap(), SchemaState::Stale);
}

#[test]
fn an_index_whose_chunks_lack_a_message_range_has_a_stale_schema() {
    let tmp = NamedTempFile::new().unwrap();
    build_schema(
        tmp.path(),
        SESSIONS_FULL,
        FTS_TOKENIZER,
        QA_NO_ROWID,
        VEC_FULL,
    );
    let conn = Connection::open(tmp.path()).unwrap();
    assert_eq!(schema_state(&conn).unwrap(), SchemaState::Stale);
}

#[test]
fn an_index_whose_chunks_still_carry_a_hash_has_a_stale_schema() {
    let tmp = NamedTempFile::new().unwrap();
    build_schema(
        tmp.path(),
        SESSIONS_FULL,
        FTS_TOKENIZER,
        QA_CHUNK_HASH,
        VEC_FULL,
    );
    let conn = Connection::open(tmp.path()).unwrap();
    assert_eq!(schema_state(&conn).unwrap(), SchemaState::Stale);
}

#[test]
fn an_index_tokenized_without_trigrams_has_a_stale_schema() {
    let tmp = NamedTempFile::new().unwrap();
    build_schema(tmp.path(), SESSIONS_FULL, "unicode61", QA_FULL, VEC_FULL);
    let conn = Connection::open(tmp.path()).unwrap();
    assert_eq!(schema_state(&conn).unwrap(), SchemaState::Stale);
}

const WRITABLE_DIR_NOTE: &str = "read-only open cannot see uncheckpointed changes in the \
     write-ahead log; run `recall index` to checkpoint them.";
const READ_ONLY_DIR_NOTE: &str = "read-only open cannot see uncheckpointed changes in the \
     write-ahead log; the directory is read-only, so copy the DB together with its `-wal` \
     (and `-shm`, if present) sidecar files elsewhere and run `recall index --db-path <copy>` \
     to checkpoint them there.";

fn file_names(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .collect();
    names.sort();
    names
}

#[test]
fn test_stale_wal_note_immutable_nonempty_wal_is_some() {
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
    // The group may write, so the mode does not read as read-only, but the owner may not.
    fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o575)).unwrap();

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
    // The writer stays open, so its row sits in the write-ahead log without a checkpoint.
    let writer = open_db(&db).unwrap();
    seed_session(&writer, "s1");

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

#[cfg(unix)]
#[test]
fn a_read_only_open_of_a_missing_database_in_a_read_only_directory_fails_after_trying_the_last_checkpoint()
 {
    use std::os::unix::fs::PermissionsExt;

    // A path under /tmp holds only bytes the URI keeps as they are, so the expected URI below can
    // be written from the plain path, whatever TMPDIR is.
    let dir = tempfile::TempDir::new_in("/tmp").unwrap();
    let missing = dir.path().join("nope.db");
    fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o555)).unwrap();

    if root_skip::skip_if_root(dir.path()) {
        return;
    }

    let result = open_db_readonly(&missing);
    fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o755)).unwrap();

    // The URI in the error shows the open fell back to reading from the last checkpoint.
    assert_eq!(
        result.err().unwrap().to_string(),
        format!(
            "unable to open database file: file:{}?immutable=1",
            missing.display()
        )
    );
}

#[cfg(unix)]
#[test]
fn a_database_in_a_read_only_directory_whose_name_holds_uri_delimiters_opens_from_its_last_checkpoint()
 {
    use std::os::unix::fs::PermissionsExt;

    let root = tempfile::TempDir::new().unwrap();
    // Each of `%`, `#` and `?` changes what the URI names unless the open escapes it. The bytes of
    // `日` must reach the URI as the same bytes.
    let dir = root.path().join("a%20b#c?d日");
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

// The expected line holds the POSIX error text.
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
fn upgrading_chunks_without_generations_keeps_their_embeddings_and_gives_every_later_insert_a_new_generation()
 {
    let tmp = NamedTempFile::new().unwrap();
    build_schema(
        tmp.path(),
        SESSIONS_FULL,
        FTS_TOKENIZER,
        QA_NO_GENERATION,
        VEC_FULL,
    );
    let conn = Connection::open(tmp.path()).unwrap();
    seed_chunk(&conn, 1, "existing content");
    let embedding = f32_as_bytes(&[0.25_f32; EMBEDDING_DIMS]);
    conn.execute(
        "INSERT INTO vec_chunks (chunk_id, sub_idx, embedding) VALUES (1, 0, ?1)",
        [&embedding],
    )
    .unwrap();
    assert_eq!(
        collect_strings(&conn, "SELECT name FROM pragma_table_info('qa_chunks')"),
        [
            "id",
            "session_id",
            "content",
            "timestamp",
            "src_rowid_lo",
            "src_rowid_hi"
        ]
    );
    assert_eq!(
        collect_strings(
            &conn,
            "SELECT name FROM sqlite_master WHERE name = 'qa_chunk_generation'"
        ),
        Vec::<String>::new()
    );
    assert_eq!(schema_state(&conn).unwrap(), SchemaState::Current);
    drop(conn);

    let conn = open_db(tmp.path()).unwrap();

    assert_eq!(
        collect_strings(&conn, CHUNK_ROWS),
        [r#"[1,"s1","existing content",0,null,null,0]"#]
    );
    seed_chunk(&conn, 2, "new content");
    assert_eq!(
        collect_strings(&conn, CHUNK_ROWS),
        [
            r#"[1,"s1","existing content",0,null,null,0]"#,
            r#"[2,"s1","new content",0,null,null,1]"#
        ]
    );
    drop(conn);

    let conn = open_db(tmp.path()).unwrap();
    conn.execute(
        "UPDATE qa_chunks SET src_rowid_lo = 1, src_rowid_hi = 2",
        [],
    )
    .unwrap();

    assert_eq!(embeddings(&conn), [(1, 1, 0, embedding.to_vec())]);
    assert_eq!(
        collect_strings(&conn, CHUNK_ROWS),
        [
            r#"[1,"s1","existing content",0,1,2,0]"#,
            r#"[2,"s1","new content",0,1,2,1]"#
        ]
    );
    conn.execute("DELETE FROM qa_chunks WHERE id = 2", [])
        .unwrap();
    seed_chunk(&conn, 2, "new content");
    assert_eq!(
        collect_strings(&conn, CHUNK_ROWS),
        [
            r#"[1,"s1","existing content",0,1,2,0]"#,
            r#"[2,"s1","new content",0,null,null,2]"#
        ]
    );
}

#[test]
fn parse_diagnostics_upgrade_preserves_content_and_schedules_one_read() {
    let tmp = NamedTempFile::new().unwrap();
    let conn = open_db(tmp.path()).unwrap();
    seed_session(&conn, "s");
    conn.execute(
        "INSERT INTO messages (session_id, role, text) VALUES ('s', 'user', 'kept')",
        [],
    )
    .unwrap();
    conn.execute_batch(
        "UPDATE sessions SET mtime = 123, file_size = 42; DROP TABLE parse_diagnostics;",
    )
    .unwrap();
    drop(conn);

    let conn = open_db(tmp.path()).unwrap();

    assert_eq!(
        (
            collect_strings(&conn, SESSION_ROWS),
            collect_strings(
                &conn,
                "SELECT json_array(rowid, session_id, role, text) FROM messages"
            )
        ),
        (
            vec![r#"["s","claude","/f","/p","slug",0,null,42,null,null,null]"#.to_owned()],
            vec![r#"[1,"s","user","kept"]"#.to_owned()]
        )
    );
    conn.execute("UPDATE sessions SET mtime = 456", []).unwrap();
    conn.execute("INSERT INTO parse_diagnostics (file_path, source, invalid_json_lines) VALUES ('/s.jsonl', 'claude', 2)", []).unwrap();
    drop(conn);

    let conn = open_db(tmp.path()).unwrap();

    assert_eq!(
        (
            collect_strings(&conn, SESSION_ROWS),
            collect_strings(
                &conn,
                "SELECT json_array(file_path, source, invalid_json_lines, invalid_utf8_lines,
                                   incomplete_tail_lines, read_error)
                 FROM parse_diagnostics"
            )
        ),
        (
            vec![r#"["s","claude","/f","/p","slug",0,456.0,42,null,null,null]"#.to_owned()],
            vec![r#"["/s.jsonl","claude",2,0,0,0]"#.to_owned()]
        )
    );
}
