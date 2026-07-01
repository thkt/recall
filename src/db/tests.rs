use super::*;
use crate::embedder::f32_as_bytes;
use tempfile::NamedTempFile;

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

/// Verify sqlite-vec ABI version matches the pinned dependency.
/// If this test fails after a version bump, re-verify the transmute
/// in `ensure_sqlite_vec` against the new C ABI before updating.
#[test]
fn test_sqlite_vec_version_matches_pinned() {
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

// T-009 (#24/FR-004): opening a DB whose sessions table predates session_type
// adds the column non-destructively — existing rows survive with NULL (treated
// as interactive by the search filter), not a destructive rebuild.
#[test]
fn test_session_type_migration_adds_column_preserving_rows() {
    let tmp = NamedTempFile::new().unwrap();
    // Pre-session_type schema: 7-column sessions + a trigram messages table, so
    // the FTS migration does not fire and wipe sessions.
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

/// Create a DB with a pre-trigram tokenizer and one session+message.
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
fn test_fts_migration_rebuilds_on_schema_change() {
    let tmp = NamedTempFile::new().unwrap();
    create_old_schema_db(tmp.path());

    let conn = open_db(tmp.path()).unwrap();

    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM sessions", [], |r| r.get(0))
        .unwrap();
    assert_eq!(count, 0);

    let sql: String = conn
        .query_row(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='messages'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert!(sql.contains(FTS_TOKENIZER));
}

#[test]
fn test_fts_migration_cascades_to_embedding_tables() {
    let tmp = NamedTempFile::new().unwrap();
    create_old_schema_db(tmp.path());

    // Add qa_chunks table with data (not created by create_old_schema_db)
    let conn = Connection::open(tmp.path()).unwrap();
    conn.execute_batch(
        "CREATE TABLE qa_chunks (
                id INTEGER PRIMARY KEY, session_id TEXT NOT NULL,
                user_text TEXT NOT NULL, assistant_text TEXT,
                content TEXT NOT NULL, timestamp INTEGER, chunk_hash TEXT NOT NULL
            );",
    )
    .unwrap();
    conn.execute(
        "INSERT INTO qa_chunks VALUES (1, 's1', 'q', 'a', 'content', 0, 'hash1')",
        [],
    )
    .unwrap();
    drop(conn);

    // Re-open triggers migration (tokenizer changed unicode61 → porter unicode61)
    let conn = open_db(tmp.path()).unwrap();

    let sessions: i64 = conn
        .query_row("SELECT COUNT(*) FROM sessions", [], |r| r.get(0))
        .unwrap();
    assert_eq!(sessions, 0, "sessions should be cleared by migration");

    let chunks: i64 = conn
        .query_row("SELECT COUNT(*) FROM qa_chunks", [], |r| r.get(0))
        .unwrap();
    assert_eq!(
        chunks, 0,
        "qa_chunks should be cleared by migration cascade"
    );
}

/// Build a DB with the pre-cleanup qa_chunks schema (chunk_hash / user_text /
/// assistant_text + idx_qa_chunks_hash), an embedded_chunk_ids table, and a
/// vec_chunks row linked by chunk_id. Uses the trigram tokenizer and the
/// sub_idx vec schema so neither migrate_fts_if_needed nor
/// migrate_vec_chunks_if_needed fires (both would wipe qa_chunks/vec_chunks).
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

    // (a) write-only columns are gone
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

    // (b) content + id preserved
    let (id, content): (i64, String) = conn
        .query_row("SELECT id, content FROM qa_chunks WHERE id = 1", [], |r| {
            Ok((r.get(0)?, r.get(1)?))
        })
        .unwrap();
    assert_eq!(id, 1);
    assert_eq!(content, "q\na");

    // (c) vec_chunks row preserved and still linked by chunk_id (embedding not orphaned)
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

    // (d) embedded_chunk_ids table is gone
    let tbl: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='embedded_chunk_ids'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(tbl, 0, "embedded_chunk_ids table should be dropped");

    // Idempotency: re-opening an already-migrated DB must be a no-op, not
    // re-run or error. This is the path every returning user hits on their
    // next `recall`, so the chunk_hash sniff must stay false after migration.
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

// ---- T-002: schema_state classification (FR-002) ----

const SESSIONS_FULL: &str = "session_id TEXT PRIMARY KEY, source TEXT, file_path TEXT, \
     project TEXT, slug TEXT, timestamp INTEGER, mtime REAL, session_type TEXT";
const SESSIONS_NO_TYPE: &str = "session_id TEXT PRIMARY KEY, source TEXT, file_path TEXT, \
     project TEXT, slug TEXT, timestamp INTEGER, mtime REAL";
const QA_FULL: &str = "id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, content TEXT NOT NULL, \
     timestamp INTEGER, src_rowid_lo INTEGER, src_rowid_hi INTEGER";
const QA_NO_ROWID: &str =
    "id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, content TEXT NOT NULL, timestamp INTEGER";
const QA_CHUNK_HASH: &str = "id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, content TEXT NOT NULL, \
     timestamp INTEGER, src_rowid_lo INTEGER, src_rowid_hi INTEGER, chunk_hash TEXT";
const VEC_FULL: &str = "+chunk_id INTEGER, +sub_idx INTEGER";
const VEC_NO_SUB: &str = "+chunk_id INTEGER";

/// Build a DB with the four core tables from explicit column/tokenizer knobs, so
/// each `schema_state` variant is expressed as a single differing fragment.
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
fn test_schema_state_empty_when_no_sessions_table() {
    // A bare file SQLite opens as an empty database: no `sessions` table.
    let tmp = NamedTempFile::new().unwrap();
    let conn = Connection::open(tmp.path()).unwrap();
    assert_eq!(schema_state(&conn).unwrap(), SchemaState::Empty);
}

#[test]
fn test_schema_state_stale_when_sessions_only() {
    // sessions present but every core table absent: crash-artifact guard.
    let tmp = NamedTempFile::new().unwrap();
    let conn = Connection::open(tmp.path()).unwrap();
    conn.execute_batch(&format!("CREATE TABLE sessions ({SESSIONS_FULL});"))
        .unwrap();
    assert_eq!(schema_state(&conn).unwrap(), SchemaState::Stale);
}

#[test]
fn test_schema_state_stale_missing_messages_vocab() {
    // `messages_vocab` is the fts5vocab table `search` hard-requires; a DB missing
    // only it (otherwise current) is a crash artifact, not a migration target.
    let tmp = NamedTempFile::new().unwrap();
    build_schema(tmp.path(), SESSIONS_FULL, FTS_TOKENIZER, QA_FULL, VEC_FULL);
    let conn = Connection::open(tmp.path()).unwrap();
    conn.execute_batch("DROP TABLE messages_vocab;").unwrap();
    assert_eq!(schema_state(&conn).unwrap(), SchemaState::Stale);
}

#[test]
fn test_schema_state_stale_missing_session_type() {
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
fn test_schema_state_stale_missing_sub_idx() {
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
fn test_schema_state_stale_missing_src_rowid() {
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
fn test_schema_state_stale_with_chunk_hash() {
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
fn test_schema_state_stale_non_trigram_tokenizer() {
    let tmp = NamedTempFile::new().unwrap();
    build_schema(tmp.path(), SESSIONS_FULL, "unicode61", QA_FULL, VEC_FULL);
    let conn = Connection::open(tmp.path()).unwrap();
    assert_eq!(schema_state(&conn).unwrap(), SchemaState::Stale);
}

// ---- T-003: stale_wal_note gating (FR-006) ----

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
fn test_stale_wal_note_direct_is_none() {
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
fn test_stale_wal_note_absent_or_empty_wal_is_none() {
    let tmp = NamedTempFile::new().unwrap();
    // Absent -wal.
    assert!(stale_wal_note(tmp.path(), OpenTier::Immutable).is_none());
    // Zero-length -wal.
    let wal = wal_path(tmp.path());
    fs::write(&wal, b"").unwrap();
    let note = stale_wal_note(tmp.path(), OpenTier::Immutable);
    fs::remove_file(&wal).ok();
    assert!(note.is_none(), "empty -wal carries no uncommitted changes");
}

// ---- T-001: open_db_readonly read-only flag + Direct tier (FR-001, AC-4) ----

#[test]
fn test_open_db_readonly_writable_dir_is_direct_and_read_only() {
    // A DB created in a writable temp dir keeps its `-shm` sidecar, so the tier-1
    // `SQLITE_OPEN_READ_ONLY` probe succeeds and selects Direct (no immutable fallback).
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

    // Proves the connection carries SQLITE_OPEN_READ_ONLY: any write is rejected.
    // Returning OpenTier::Direct alone would not prove the flag; a broken impl
    // could open read-write and still report Direct.
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

// ---- T-002: is_readonly_dir_error gates the immutable fallback on dir writability ----
//
// Regression for the SOW HIGH residual: the tier-2 `immutable=1` fallback disables change
// detection, so it must fire ONLY on read-only media, never in a WRITABLE directory where a
// missing `-shm` (restored single-file backup, EMFILE, TOCTOU delete) surfaces as the same
// SQLITE_CANTOPEN. Without the `dir_is_unwritable` guard the predicate freezes a still-mutating
// DB and races a concurrent `recall index` close-checkpoint (UB). Testing the predicate directly
// avoids reproducing the version-fragile missing-`-shm`-in-writable SQLite state.

#[cfg(unix)]
#[test]
fn test_is_readonly_dir_error_writable_dir_rejects_immutable_fallback() {
    use rusqlite::ffi::Error as FfiError;

    // SQLITE_CANTOPEN (14): the exact code a missing `-shm` surfaces on a WAL-mode read.
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

    // Root ignores dir permission bits, so the read-only condition can't be reproduced.
    let is_root = fs::File::create(dir.path().join(".root_probe")).is_ok();
    if is_root {
        let _ = fs::remove_file(dir.path().join(".root_probe"));
        fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o755)).unwrap();
        return;
    }

    let got = is_readonly_dir_error(&db, &cantopen);
    // Restore write bits before the assert so TempDir's drop can clean up.
    fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o755)).unwrap();

    assert!(
        got,
        "CannotOpen in a read-only (0o555) dir is read-only media; the immutable fallback must fire"
    );
}

// A read-only open of a file that does not exist errors at open (READ_ONLY cannot
// create the DB). In a WRITABLE dir that is not read-only media, so the initial-open
// Err arm must propagate the error rather than reach for the immutable fallback.
#[test]
fn test_open_db_readonly_missing_file_in_writable_dir_errors() {
    let dir = tempfile::TempDir::new().unwrap();
    let missing = dir.path().join("nope.db");
    assert!(
        open_db_readonly(&missing).is_err(),
        "a missing DB in a writable dir must error, not silently open"
    );
}

// A missing file in a READ-ONLY dir sends the initial-open Err through the
// immutable (tier-2) fallback branch; the fallback itself still fails (no file to
// open immutably), so the call errors — but the tier-2 arm is exercised.
#[cfg(unix)]
#[test]
fn test_open_db_readonly_missing_file_in_readonly_dir_takes_immutable_branch() {
    use std::os::unix::fs::PermissionsExt;

    let dir = tempfile::TempDir::new().unwrap();
    let missing = dir.path().join("nope.db");
    fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o555)).unwrap();

    // Root ignores dir permission bits, so the read-only condition can't be reproduced.
    let is_root = fs::File::create(dir.path().join(".root_probe")).is_ok();
    if is_root {
        let _ = fs::remove_file(dir.path().join(".root_probe"));
        fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o755)).unwrap();
        return;
    }

    let result = open_db_readonly(&missing);
    fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o755)).unwrap();
    assert!(
        result.is_err(),
        "a missing DB in a read-only dir errors even after the immutable fallback"
    );
}

// encode_uri_path percent-escapes every byte outside the RFC 3986 unreserved set,
// keeping `/` as the separator; a space and a `?` (URI query delimiter) must be
// escaped so the `file:` URI is not truncated at the DB path.
#[test]
fn test_encode_uri_path_escapes_reserved_bytes() {
    let encoded = encode_uri_path(Path::new("/a b/c?d.db"));
    assert_eq!(encoded, "/a%20b/c%3Fd.db");
}
