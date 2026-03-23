use std::sync::Once;

use anyhow::Result;
use rusqlite::Connection;

use crate::embedder::EMBEDDING_DIMS;

const FTS_TOKENIZER: &str = "trigram";

static SQLITE_VEC_INIT: Once = Once::new();

fn ensure_sqlite_vec() {
    SQLITE_VEC_INIT.call_once(|| {
        // SAFETY: sqlite3_vec_init is a C extern fn matching sqlite3_auto_extension's
        // callback signature. Pinned to sqlite-vec 0.1.7; re-verify ABI on version bumps.
        unsafe {
            #[allow(clippy::missing_transmute_annotations)]
            rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute(
                sqlite_vec::sqlite3_vec_init as *const (),
            )));
        }
    });
}

pub fn open_db(path: &std::path::Path) -> Result<Connection> {
    ensure_sqlite_vec();
    let mut conn = Connection::open(path)?;
    conn.busy_timeout(std::time::Duration::from_secs(5))?;
    let _: String = conn.query_row("PRAGMA journal_mode=WAL", [], |r| r.get(0))?;
    conn.execute_batch("PRAGMA synchronous=NORMAL;")?;
    create_schema(&mut conn)?;
    Ok(conn)
}

fn create_schema(conn: &mut Connection) -> Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            source TEXT,
            file_path TEXT,
            project TEXT,
            slug TEXT,
            timestamp INTEGER,
            mtime REAL
        );",
    )?;

    migrate_fts_if_needed(conn)?;

    conn.execute_batch(
        "CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project);
         CREATE INDEX IF NOT EXISTS idx_sessions_timestamp ON sessions(timestamp);",
    )?;

    conn.execute_batch(&format!(
        "CREATE VIRTUAL TABLE IF NOT EXISTS messages USING fts5(
            session_id UNINDEXED,
            role,
            text,
            tokenize='{FTS_TOKENIZER}'
        );"
    ))?;

    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS recall_meta (
            key TEXT PRIMARY KEY,
            value REAL
        );",
    )?;

    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS qa_chunks (
            id INTEGER PRIMARY KEY,
            session_id TEXT NOT NULL,
            user_text TEXT NOT NULL,
            assistant_text TEXT,
            content TEXT NOT NULL,
            timestamp INTEGER,
            chunk_hash TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_qa_chunks_session ON qa_chunks(session_id);
        CREATE INDEX IF NOT EXISTS idx_qa_chunks_hash ON qa_chunks(chunk_hash);",
    )?;

    conn.execute_batch(&format!(
        "CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
            chunk_id INTEGER PRIMARY KEY,
            embedding FLOAT[{EMBEDDING_DIMS}]
        );"
    ))?;

    conn.execute_batch(
        "CREATE VIRTUAL TABLE IF NOT EXISTS messages_vocab USING fts5vocab(messages, row);",
    )?;

    Ok(())
}

fn migrate_fts_if_needed(conn: &mut Connection) -> Result<()> {
    let sql: Option<String> = match conn.query_row(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='messages'",
        [],
        |r| r.get(0),
    ) {
        Ok(sql) => Some(sql),
        Err(rusqlite::Error::QueryReturnedNoRows) => None,
        Err(e) => return Err(e.into()),
    };

    let Some(sql) = sql else {
        return Ok(());
    };

    if !sql.contains(FTS_TOKENIZER) {
        let session_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM sessions", [], |r| r.get(0))?;
        eprintln!(
            "recall: Index schema changed — rebuilding {session_count} sessions (source files are unaffected)"
        );
        let tx = conn.transaction()?;
        tx.execute_batch(
            "DROP TABLE messages; DELETE FROM sessions; \
             DROP TABLE IF EXISTS qa_chunks; DROP TABLE IF EXISTS vec_chunks;",
        )?;
        tx.commit()?;
    }

    Ok(())
}

#[cfg(test)]
pub(crate) fn setup_test_db() -> (tempfile::TempDir, Connection) {
    let dir = tempfile::TempDir::new().unwrap();
    let db_path = dir.path().join("test.db");
    let conn = open_db(&db_path).unwrap();
    (dir, conn)
}

#[cfg(test)]
mod tests {
    use super::*;
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
            version, "v0.1.7",
            "sqlite-vec version changed — re-verify unsafe transmute ABI in ensure_sqlite_vec()"
        );
    }

    #[test]
    fn test_open_db_idempotent() {
        let tmp = NamedTempFile::new().unwrap();
        let _conn1 = open_db(tmp.path()).unwrap();
        let _conn2 = open_db(tmp.path()).unwrap();
    }

    /// Create a DB with a pre-trigram tokenizer and one session+message.
    fn create_old_schema_db(path: &std::path::Path) {
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
}
