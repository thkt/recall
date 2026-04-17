use std::path::Path;
use std::time::Duration;

use anyhow::Result;
use rurico::embed::EMBEDDING_DIMS;
use rurico::storage::ensure_sqlite_vec;
use rusqlite::Connection;

const FTS_TOKENIZER: &str = "trigram";

pub fn open_db(path: &Path) -> Result<Connection> {
    ensure_sqlite_vec().map_err(|e| anyhow::anyhow!(e))?;
    let mut conn = Connection::open(path)?;
    conn.busy_timeout(Duration::from_secs(5))?;
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
    migrate_vec_chunks_if_needed(conn)?;

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
            embedding FLOAT[{EMBEDDING_DIMS}],
            +chunk_id INTEGER,
            +sub_idx INTEGER
        );"
    ))?;

    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS embedded_chunk_ids (
            chunk_id INTEGER NOT NULL,
            sub_idx INTEGER NOT NULL,
            vec_rowid INTEGER NOT NULL,
            PRIMARY KEY (chunk_id, sub_idx)
        );
        CREATE INDEX IF NOT EXISTS idx_embedded_chunk_ids_chunk ON embedded_chunk_ids(chunk_id);",
    )?;

    conn.execute_batch(
        "CREATE VIRTUAL TABLE IF NOT EXISTS messages_vocab USING fts5vocab(messages, row);",
    )?;

    Ok(())
}

fn migrate_vec_chunks_if_needed(conn: &mut Connection) -> Result<()> {
    let sql: Option<String> = match conn.query_row(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='vec_chunks'",
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

    if !sql.contains("sub_idx") {
        eprintln!(
            "recall: Embedding schema changed — clearing embeddings (re-run `recall index --embed` to rebuild)"
        );
        let tx = conn.transaction()?;
        tx.execute_batch(
            "DROP TABLE IF EXISTS vec_chunks; DROP TABLE IF EXISTS embedded_chunk_ids;",
        )?;
        tx.commit()?;
    }

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
        let session_count: i64 =
            conn.query_row("SELECT COUNT(*) FROM sessions", [], |r| r.get(0))?;
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
}
