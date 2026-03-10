use anyhow::Result;
use rusqlite::Connection;

const FTS_TOKENIZER: &str = "porter unicode61";

pub fn open_db(path: &std::path::Path) -> Result<Connection> {
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
            .query_row("SELECT COUNT(*) FROM sessions", [], |r| r.get(0))
            .unwrap_or(0);
        eprintln!(
            "recall: Index schema changed — rebuilding {session_count} sessions (source files are unaffected)"
        );
        let tx = conn.transaction()?;
        tx.execute_batch("DROP TABLE messages; DELETE FROM sessions;")?;
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

        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM sessions", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 0);

        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM messages", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_open_db_idempotent() {
        let tmp = NamedTempFile::new().unwrap();
        let _conn1 = open_db(tmp.path()).unwrap();
        let _conn2 = open_db(tmp.path()).unwrap();
    }

    #[test]
    fn test_fts_migration_rebuilds_on_schema_change() {
        let tmp = NamedTempFile::new().unwrap();

        let conn = Connection::open(tmp.path()).unwrap();
        conn.execute_batch(
            "CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY, source TEXT, file_path TEXT,
                project TEXT, slug TEXT, timestamp INTEGER, mtime REAL
            );
            CREATE VIRTUAL TABLE messages USING fts5(
                session_id UNINDEXED, role, text, tokenize='unicode61'
            );",
        ).unwrap();
        conn.execute(
            "INSERT INTO sessions VALUES ('s1', 'claude', '/f', '/p', 'slug', 0, 0.0)",
            [],
        ).unwrap();
        conn.execute(
            "INSERT INTO messages (session_id, role, text) VALUES ('s1', 'user', 'hello')",
            [],
        ).unwrap();
        drop(conn);

        let conn = open_db(tmp.path()).unwrap();

        let count: i64 = conn.query_row("SELECT COUNT(*) FROM sessions", [], |r| r.get(0)).unwrap();
        assert_eq!(count, 0);

        let sql: String = conn.query_row(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='messages'",
            [],
            |r| r.get(0),
        ).unwrap();
        assert!(sql.contains(FTS_TOKENIZER));
    }
}
