use super::*;
use crate::embedder::f32_as_bytes;
use tempfile::NamedTempFile;

use crate::embedder;
use crate::open_or_create_db;
use crate::tests::seed_healthy_index;

#[test]
fn test_open_db_creates_schema() {
    let tmp = NamedTempFile::new().unwrap();
    let conn = open_db(tmp.path()).unwrap();

    for table in ["sessions", "messages", "qa_chunks", "messages_vocab"] {
        let count: i64 = conn
            .query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 0, "table {table} should be empty");
    }
}

#[test]
fn an_opened_database_runs_sqlite_vec_v0_1_9() {
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
    let _conn1 = open_db(tmp.path()).unwrap();
    let _conn2 = open_db(tmp.path()).unwrap();
}

#[test]
fn upgrading_an_index_without_chunk_completion_markers_rolls_back_a_failed_upgrade_then_marks_only_chunked_sessions()
 {
    use crate::indexer::index_chunks;

    let tmp = NamedTempFile::new().unwrap();
    create_pre_cleanup_db(tmp.path());
    {
        let conn = Connection::open(tmp.path()).unwrap();
        conn.execute_batch(
            "INSERT INTO sessions (session_id) VALUES ('s1'), ('empty'), ('pending');
             INSERT INTO messages (session_id, role, text) VALUES
                 ('empty', 'assistant', 'unpaired answer'), ('pending', 'user', 'new question');
             CREATE TRIGGER fail_marker_upgrade BEFORE UPDATE ON sessions
             BEGIN SELECT RAISE(ABORT, 'injected migration failure'); END;",
        )
        .unwrap();
    }
    let error = open_db(tmp.path()).err().unwrap();
    assert!(error.to_string().contains("injected migration failure"));
    {
        let conn = Connection::open(tmp.path()).unwrap();
        assert!(
            !table_def(&conn, "sessions")
                .unwrap()
                .unwrap()
                .contains("chunks_indexed")
        );
        conn.execute_batch("DROP TRIGGER fail_marker_upgrade;")
            .unwrap();
    }
    for pass in 0..2 {
        let mut conn = open_db(tmp.path()).unwrap();
        let complete: i64 = conn
            .query_row(
                "SELECT chunks_indexed FROM sessions WHERE session_id = 's1'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(complete, 1);
        let stats = index_chunks(&mut conn).unwrap();
        assert_eq!(stats.sessions_chunked, if pass == 0 { 2 } else { 0 });
        assert_eq!(stats.chunks_created, if pass == 0 { 1 } else { 0 });
        let original: (String, Option<i64>, Option<i64>) = conn
            .query_row(
                "SELECT content, src_rowid_lo, src_rowid_hi FROM qa_chunks WHERE id = 1",
                [],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .unwrap();
        assert_eq!(original, ("q\na".to_owned(), None, None));
        let (id, embedding): (i64, Vec<u8>) = conn
            .query_row("SELECT chunk_id, embedding FROM vec_chunks", [], |r| {
                Ok((r.get(0)?, r.get(1)?))
            })
            .unwrap();
        assert_eq!(id, 1);
        assert_eq!(embedding, f32_as_bytes(&[0.1f32; EMBEDDING_DIMS]));
        let markers: Vec<(String, i64)> = conn
            .prepare("SELECT session_id, chunks_indexed FROM sessions ORDER BY session_id")
            .unwrap()
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap();
        assert_eq!(
            markers,
            [
                ("empty".to_owned(), 0),
                ("pending".to_owned(), 0),
                ("s1".to_owned(), 1),
            ]
        );
    }
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

    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM sessions", [], |r| r.get(0))
        .unwrap();
    assert_eq!(count, 1, "existing session row must survive the migration");

    let session_type: Option<String> = conn
        .query_row(
            "SELECT session_type FROM sessions WHERE session_id = 's1'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        session_type, None,
        "pre-existing rows get NULL session_type"
    );
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
        conn.prepare("SELECT session_id, slug FROM sessions")
            .unwrap()
            .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))
            .unwrap()
            .map(Result::unwrap)
            .collect::<Vec<_>>(),
        [("s1".to_owned(), "slug".to_owned())]
    );
    // A trigram index matches a word's inner three letters; the older tokenizers do not.
    assert_eq!(
        conn.prepare(
            "SELECT rowid, role, text FROM messages WHERE messages MATCH 'hel' ORDER BY rowid"
        )
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
        [
            (1, "user".to_owned(), "hello".to_owned()),
            (3, "assistant".to_owned(), "help is here".to_owned()),
        ]
    );
    assert_eq!(
        conn.prepare("SELECT rowid FROM messages ORDER BY rowid")
            .unwrap()
            .query_map([], |r| r.get::<_, i64>(0))
            .unwrap()
            .map(Result::unwrap)
            .collect::<Vec<_>>(),
        [1, 3, 7]
    );
    let schema_version: i64 = conn
        .query_row("PRAGMA schema_version", [], |r| r.get(0))
        .unwrap();
    drop(conn);

    let reopened = open_db(tmp.path()).unwrap();

    assert_eq!(
        reopened
            .query_row("PRAGMA schema_version", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        schema_version,
        "a second open must not migrate the table again"
    );
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

    let cols: Vec<String> = {
        let mut stmt = conn.prepare("PRAGMA table_info(qa_chunks)").unwrap();
        stmt.query_map([], |r| r.get::<_, String>(1))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap()
    };
    for gone in ["chunk_hash", "user_text", "assistant_text"] {
        assert!(
            !cols.contains(&gone.to_owned()),
            "column {gone} should be dropped, got {cols:?}"
        );
    }

    let (id, content): (i64, String) = conn
        .query_row("SELECT id, content FROM qa_chunks WHERE id = 1", [], |r| {
            Ok((r.get(0)?, r.get(1)?))
        })
        .unwrap();
    assert_eq!(id, 1);
    assert_eq!(content, "q\na");

    let vec_count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM vec_chunks v JOIN qa_chunks c ON c.id = v.chunk_id",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        vec_count, 1,
        "vec_chunks row must survive and stay linked to qa_chunks"
    );

    let tbl: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='embedded_chunk_ids'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(tbl, 0, "embedded_chunk_ids table should be dropped");

    drop(conn);
    let conn = open_db(tmp.path()).unwrap();
    let cols2: Vec<String> = {
        let mut stmt = conn.prepare("PRAGMA table_info(qa_chunks)").unwrap();
        stmt.query_map([], |r| r.get::<_, String>(1))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap()
    };
    assert!(
        !cols2.contains(&"chunk_hash".to_owned()),
        "re-open must stay migrated, got {cols2:?}"
    );
    let vec_count2: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM vec_chunks v JOIN qa_chunks c ON c.id = v.chunk_id",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(vec_count2, 1, "re-open must keep vec row linked");
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
fn test_schema_state_current_on_freshly_opened_db() {
    let tmp = NamedTempFile::new().unwrap();
    let conn = open_db(tmp.path()).unwrap();
    assert_eq!(schema_state(&conn).unwrap(), SchemaState::Current);
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

#[test]
fn test_stale_wal_note_immutable_nonempty_wal_is_some() {
    let tmp = NamedTempFile::new().unwrap();
    let wal = wal_path(tmp.path());
    fs::write(&wal, b"uncommitted").unwrap();
    let note = stale_wal_note(tmp.path(), OpenTier::Immutable);
    fs::remove_file(&wal).ok();
    assert!(note.is_some(), "immutable tier + non-empty -wal must warn");
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
    let wal = wal_path(&db);
    fs::write(&wal, b"uncommitted").unwrap();
    fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o555)).unwrap();

    if root_skip::skip_if_root(dir.path()) {
        return;
    }

    let note = stale_wal_note(&db, OpenTier::Immutable);
    fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o755)).unwrap();

    let note = note.expect("non-empty -wal on Immutable tier must warn");
    root_skip::assert_copy_based_remedy(&note, "read-only dir immutable tier");
}

#[test]
fn writable_dir_immutable_tier_非空_wal_で_stale_wal_note_は従来文言を含む() {
    let tmp = NamedTempFile::new().unwrap();
    let wal = wal_path(tmp.path());
    fs::write(&wal, b"uncommitted").unwrap();

    let note = stale_wal_note(tmp.path(), OpenTier::Immutable);
    fs::remove_file(&wal).ok();

    let note = note.expect("non-empty -wal on Immutable tier must warn");
    assert!(
        note.contains("write-ahead log"),
        "note must carry the write-ahead-log wording, got: {note}"
    );
    assert!(
        note.contains("run `recall index`"),
        "writable dir must keep the legacy bare `recall index` wording, got: {note}"
    );
}

#[test]
fn a_read_only_open_in_a_writable_directory_opens_the_live_database_and_rejects_writes() {
    let tmp = NamedTempFile::new().unwrap();
    {
        let conn = open_db(tmp.path()).unwrap();
        conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);").ok();
    }

    let (conn, tier) = open_db_readonly(tmp.path()).unwrap();
    assert_eq!(
        tier,
        OpenTier::Direct,
        "writable dir must select Direct tier"
    );

    let err = conn
        .execute("CREATE TABLE probe_write (x INTEGER)", [])
        .expect_err("read-only connection must reject a write");
    let rusqlite::Error::SqliteFailure(inner, _) = &err else {
        panic!("expected a SqliteFailure, got {err:?}");
    };
    assert_eq!(
        inner.code,
        rusqlite::ErrorCode::ReadOnly,
        "write must fail with SQLITE_READONLY, got {err:?}"
    );
}

#[cfg(unix)]
#[test]
fn test_is_readonly_dir_error_writable_dir_rejects_immutable_fallback() {
    use rusqlite::ffi::Error as FfiError;

    let cantopen = rusqlite::Error::SqliteFailure(FfiError::new(14), None);

    let dir = tempfile::TempDir::new().unwrap();
    let db = dir.path().join("recall.db");

    assert!(
        !is_readonly_dir_error(&db, &cantopen),
        "CannotOpen in a writable dir is not read-only media; the immutable fallback must not fire"
    );
}

#[cfg(unix)]
#[test]
fn test_is_readonly_dir_error_readonly_dir_selects_immutable_fallback() {
    use rusqlite::ffi::Error as FfiError;
    use std::os::unix::fs::PermissionsExt;

    let cantopen = rusqlite::Error::SqliteFailure(FfiError::new(14), None);

    let dir = tempfile::TempDir::new().unwrap();
    let db = dir.path().join("recall.db");
    fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o555)).unwrap();

    if root_skip::skip_if_root(dir.path()) {
        return;
    }

    let got = is_readonly_dir_error(&db, &cantopen);
    fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o755)).unwrap();

    assert!(
        got,
        "CannotOpen in a read-only (0o555) dir is read-only media; the immutable fallback must fire"
    );
}

#[test]
fn a_read_only_open_of_a_missing_database_fails_without_creating_any_file() {
    let dir = tempfile::TempDir::new().unwrap();
    let missing = dir.path().join("nope.db");
    assert!(
        open_db_readonly(&missing).is_err(),
        "a missing DB in a writable dir must error, not silently open"
    );
}

#[cfg(unix)]
#[test]
fn a_read_only_open_of_a_missing_database_in_a_read_only_directory_fails() {
    use std::os::unix::fs::PermissionsExt;

    let dir = tempfile::TempDir::new().unwrap();
    let missing = dir.path().join("nope.db");
    fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o555)).unwrap();

    if root_skip::skip_if_root(dir.path()) {
        return;
    }

    let result = open_db_readonly(&missing);
    fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o755)).unwrap();
    assert!(
        result.is_err(),
        "a missing DB in a read-only dir errors even after the immutable fallback"
    );
}

#[test]
fn test_encode_uri_path_escapes_reserved_bytes() {
    let encoded = encode_uri_path(Path::new("/a b/c?d.db"));
    assert_eq!(encoded, "/a%20b/c%3Fd.db");
}

#[test]
fn write_probe_reports_a_writable_directory_as_writable_and_leaves_no_probe_file_behind() {
    let dir = tempfile::TempDir::new().unwrap();

    let result = dir_write_probe_fails(dir.path());

    assert!(
        !result,
        "a writable directory must not be reported as unwritable"
    );
    let entries: Vec<_> = fs::read_dir(dir.path()).unwrap().collect();
    assert!(
        entries.is_empty(),
        "probe file must be removed after a successful write, got entries: {entries:?}"
    );
}

#[cfg(unix)]
#[test]
fn write_probe_reports_a_directory_with_cleared_write_bits_as_unwritable() {
    use std::os::unix::fs::PermissionsExt;

    let dir = tempfile::TempDir::new().unwrap();
    fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o555)).unwrap();

    if root_skip::skip_if_root(dir.path()) {
        return;
    }

    let result = dir_write_probe_fails(dir.path());
    fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o755)).unwrap();

    assert!(
        result,
        "a directory with cleared write bits (0o555) must be reported as unwritable (EACCES -> PermissionDenied)"
    );
}

#[test]
fn a_leftover_probe_file_reads_as_ambiguous_so_the_directory_stays_classified_writable() {
    let dir = tempfile::TempDir::new().unwrap();
    let leftover = dir
        .path()
        .join(format!(".recall-write-probe-{}", process::id()));
    fs::write(&leftover, b"stale").unwrap();

    assert!(
        !dir_write_probe_fails(dir.path()),
        "a pre-existing probe file (AlreadyExists) must classify the directory \
         as writable, not read-only"
    );
}

#[test]
fn probe_cleanup_failure_warns_instead_of_failing_the_read() {
    remove_write_probe(Path::new("/nonexistent-recall-dir/.recall-write-probe-0"));
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

    let note = stale_wal_note(&mount.db_path(), OpenTier::Immutable)
        .expect("non-empty -wal on a read-only mount must warn");
    root_skip::assert_copy_based_remedy(&note, "read-only mount immutable tier");
}

#[test]
fn a_missing_parent_directory_stays_classified_writable_so_the_original_error_propagates() {
    let dir = tempfile::TempDir::new().unwrap();
    let db = dir.path().join("nonexistent_subdir").join("recall.db");

    assert!(
        !dir_is_unwritable(&db),
        "a missing parent directory must not be classified unwritable \
         (the write probe's NotFound failure does not count as unwritable)"
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
fn root_check_on_a_writable_directory_reports_root_and_restores_the_directory() {
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
fn skip_if_root_は_0o555_dir_で_create_失敗なら_probe_を残さず_false_を返す() {
    use std::os::unix::fs::PermissionsExt;

    let dir = tempfile::TempDir::new().unwrap();
    fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o555)).unwrap();

    let got = root_skip::skip_if_root(dir.path());

    fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o755)).unwrap();

    assert!(
        !got,
        "create in a 0o555 dir must fail and report false; true means the \
         runner bypasses permission bits (e.g. running as root)"
    );
    assert_eq!(
        fs::read_dir(dir.path()).unwrap().count(),
        0,
        "no entry may remain directly under dir"
    );
}

#[cfg(unix)]
#[test]
#[should_panic(expected = "-shm sidecar")]
fn copy_remedy_check_rejects_a_note_without_the_shm_sidecar() {
    root_skip::assert_copy_based_remedy(
        "read-only open cannot see uncommitted changes in the write-ahead log; \
         the directory is read-only, so copy the DB together with its `-wal` \
         sidecar file elsewhere and run `recall index --db-path <copy>` to \
         checkpoint it there.",
        "negative self-test",
    );
}

#[cfg(unix)]
#[test]
#[should_panic(expected = "cannot run a bare")]
fn copy_remedy_check_rejects_a_note_that_also_offers_a_bare_recall_index() {
    root_skip::assert_copy_based_remedy(
        "read-only open cannot see uncommitted changes in the write-ahead log; \
         copy the DB together with its `-wal` (and `-shm`, if present) sidecar \
         files elsewhere and run `recall index --db-path <copy>`, or run \
         `recall index` to checkpoint them.",
        "negative self-test",
    );
}

#[cfg(unix)]
#[test]
fn the_unit_suite_fails_when_run_as_root() {
    root_skip::assert_root_canary();
}

#[cfg(unix)]
#[test]
fn db_tests_spell_the_root_probe_file_only_through_skip_if_root() {
    root_skip::assert_probe_literal_not_inlined(include_str!("tests.rs"), "src/db/tests.rs");
}

#[test]
fn upgrading_chunks_without_generations_keeps_their_embeddings_and_gives_every_later_insert_a_new_generation()
 {
    let tmp = NamedTempFile::new().unwrap();
    build_schema(
        tmp.path(),
        SESSIONS_FULL,
        "trigram",
        QA_NO_GENERATION,
        VEC_FULL,
    );
    let conn = Connection::open(tmp.path()).unwrap();
    seed_chunk(&conn, 1, "existing content");
    let vector = vec![0.25_f32; EMBEDDING_DIMS];
    conn.execute(
        "INSERT INTO vec_chunks (chunk_id, sub_idx, embedding) VALUES (1, 0, ?1)",
        [f32_as_bytes(&vector)],
    )
    .unwrap();
    assert!(
        !table_def(&conn, "qa_chunks")
            .unwrap()
            .unwrap()
            .contains("generation")
    );
    assert!(table_def(&conn, "qa_chunk_generation").unwrap().is_none());
    assert_eq!(schema_state(&conn).unwrap(), SchemaState::Current);
    drop(conn);

    let conn = open_db(tmp.path()).unwrap();
    assert_eq!(schema_state(&conn).unwrap(), SchemaState::Current);
    let (content, generation): (String, i64) = conn
        .query_row(
            "SELECT content, generation FROM qa_chunks WHERE id = 1",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    assert_eq!(content, "existing content");
    assert_eq!(generation, 0, "legacy rows receive the migration default");
    let saved: (i64, Vec<u8>, i64, i64) = conn
        .query_row(
            "SELECT rowid, embedding, chunk_id, sub_idx FROM vec_chunks",
            [],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
        )
        .unwrap();
    assert_eq!(saved.1, f32_as_bytes(&vector));
    assert_eq!((saved.2, saved.3), (1, 0));
    seed_chunk(&conn, 2, "new content");
    let new_generation: i64 = conn
        .query_row("SELECT generation FROM qa_chunks WHERE id = 2", [], |r| {
            r.get(0)
        })
        .unwrap();
    assert_ne!(new_generation, generation);
    drop(conn);

    let conn = open_db(tmp.path()).unwrap();
    let reopened: (i64, Vec<u8>, i64, i64) = conn
        .query_row(
            "SELECT rowid, embedding, chunk_id, sub_idx FROM vec_chunks",
            [],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
        )
        .unwrap();
    assert_eq!(reopened, saved);
    conn.execute(
        "UPDATE qa_chunks SET src_rowid_lo = 1, src_rowid_hi = 2",
        [],
    )
    .unwrap();
    let generations: Vec<i64> = conn
        .prepare("SELECT generation FROM qa_chunks ORDER BY id")
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(generations, vec![generation, new_generation]);
    conn.execute("DELETE FROM qa_chunks WHERE id = 2", [])
        .unwrap();
    seed_chunk(&conn, 2, "new content");
    let replacement: i64 = conn
        .query_row("SELECT generation FROM qa_chunks WHERE id = 2", [], |r| {
            r.get(0)
        })
        .unwrap();
    assert!(
        replacement > new_generation,
        "re-open must not reset the counter"
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
    let stamp: (Option<f64>, i64) = conn
        .query_row("SELECT mtime, file_size FROM sessions", [], |r| {
            Ok((r.get(0)?, r.get(1)?))
        })
        .unwrap();
    assert_eq!(stamp, (None, 42));
    let text: String = conn
        .query_row("SELECT text FROM messages", [], |r| r.get(0))
        .unwrap();
    assert_eq!(text, "kept");
    conn.execute("UPDATE sessions SET mtime = 456", []).unwrap();
    conn.execute("INSERT INTO parse_diagnostics (file_path, source, invalid_json_lines) VALUES ('/s.jsonl', 'claude', 2)", []).unwrap();
    drop(conn);
    let conn = open_db(tmp.path()).unwrap();
    assert_eq!(
        conn.query_row("SELECT mtime FROM sessions", [], |r| r.get::<_, f64>(0))
            .unwrap(),
        456.0
    );
    assert_eq!(
        conn.query_row(
            "SELECT invalid_json_lines FROM parse_diagnostics",
            [],
            |r| r.get::<_, i64>(0)
        )
        .unwrap(),
        2
    );
}

#[test]
fn fix_orphan_embeddings_deletes_only_dangling_rows() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let conn = open_or_create_db(&db_dir.path().join("recall.db")).unwrap();
    seed_healthy_index(&conn);
    let orphan = embedder::MockEmbedder::deterministic_vector("orphan");
    conn.execute(
        "INSERT INTO vec_chunks (embedding, chunk_id, sub_idx) VALUES (?1, 999, 0)",
        rusqlite::params![embedder::f32_as_bytes(&orphan)],
    )
    .unwrap();

    let deleted = fix_orphan_embeddings(&conn).unwrap();

    assert_eq!(deleted, 1, "only the dangling chunk_id=999 row is deleted");
    let remaining: i64 = conn
        .query_row("SELECT COUNT(*) FROM vec_chunks", [], |r| r.get(0))
        .unwrap();
    assert_eq!(remaining, 1, "the linked chunk_id=1 row must survive");
}

#[test]
fn fix_orphan_embeddings_is_noop_when_clean() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let conn = open_or_create_db(&db_dir.path().join("recall.db")).unwrap();
    seed_healthy_index(&conn);

    let deleted = fix_orphan_embeddings(&conn).unwrap();

    assert_eq!(deleted, 0, "a clean index has nothing to repair");
    let remaining: i64 = conn
        .query_row("SELECT COUNT(*) FROM vec_chunks", [], |r| r.get(0))
        .unwrap();
    assert_eq!(remaining, 1, "the linked row must be untouched");
}

#[test]
fn fix_orphan_embeddings_errors_when_table_missing() {
    let conn = Connection::open_in_memory().unwrap();
    assert!(fix_orphan_embeddings(&conn).is_err());
}
