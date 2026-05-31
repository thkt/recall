use std::path::Path;
use std::time::Duration;

use amici::migration::notify_schema_change;
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
    migrate_qa_chunks_if_needed(conn)?;

    // embedded_chunk_ids (a removed ledger) was dropped inside two separate
    // migrations; collapse those into one unconditional drop so every pre-cleanup
    // DB ends up without it after open_db.
    conn.execute_batch("DROP TABLE IF EXISTS embedded_chunk_ids;")?;

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
        "CREATE TABLE IF NOT EXISTS qa_chunks (
            id INTEGER PRIMARY KEY,
            session_id TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_qa_chunks_session ON qa_chunks(session_id);",
    )?;

    conn.execute_batch(&format!(
        "CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
            embedding FLOAT[{EMBEDDING_DIMS}],
            +chunk_id INTEGER,
            +sub_idx INTEGER
        );"
    ))?;

    conn.execute_batch(
        "CREATE VIRTUAL TABLE IF NOT EXISTS messages_vocab USING fts5vocab(messages, row);",
    )?;

    Ok(())
}

/// Fetch a table's stored `CREATE` SQL from `sqlite_master`, or `None` if it
/// does not exist. Virtual tables (fts5, vec0) are stored with `type='table'`,
/// so this resolves regular and virtual tables alike.
fn table_def(conn: &Connection, name: &str) -> Result<Option<String>> {
    match conn.query_row(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?1",
        [name],
        |r| r.get(0),
    ) {
        Ok(sql) => Ok(Some(sql)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(e.into()),
    }
}

fn migrate_vec_chunks_if_needed(conn: &mut Connection) -> Result<()> {
    let Some(sql) = table_def(conn, "vec_chunks")? else {
        return Ok(());
    };

    if !sql.contains("sub_idx") {
        notify_schema_change("recall", "embeddings", 0, "recall embed");
        let tx = conn.transaction()?;
        tx.execute_batch("DROP TABLE IF EXISTS vec_chunks;")?;
        tx.commit()?;
    }

    Ok(())
}

/// Drop the write-only columns (chunk_hash / user_text / assistant_text) from
/// pre-cleanup databases. ALTER ... DROP COLUMN keeps content / id and the
/// vec_chunks linkage by chunk_id intact, so existing embeddings survive without
/// a re-index. The removed embedded_chunk_ids ledger is dropped in create_schema.
fn migrate_qa_chunks_if_needed(conn: &mut Connection) -> Result<()> {
    let Some(sql) = table_def(conn, "qa_chunks")? else {
        return Ok(());
    };

    if sql.contains("chunk_hash") {
        let tx = conn.transaction()?;
        tx.execute_batch(
            "DROP INDEX IF EXISTS idx_qa_chunks_hash;
             ALTER TABLE qa_chunks DROP COLUMN chunk_hash;
             ALTER TABLE qa_chunks DROP COLUMN user_text;
             ALTER TABLE qa_chunks DROP COLUMN assistant_text;",
        )?;
        tx.commit()?;
    }

    Ok(())
}

fn migrate_fts_if_needed(conn: &mut Connection) -> Result<()> {
    let Some(sql) = table_def(conn, "messages")? else {
        return Ok(());
    };

    if !sql.contains(FTS_TOKENIZER) {
        let session_count: i64 =
            conn.query_row("SELECT COUNT(*) FROM sessions", [], |r| r.get(0))?;
        let count = usize::try_from(session_count).unwrap_or(0);
        notify_schema_change("recall", "cached sessions", count, "recall index");
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
}
