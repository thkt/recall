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
    // 30s, not 5s: the FTS index pass holds one write tx for 3-12s on a full
    // tree (#130 D3, measured at 38k chunks), so concurrent SessionEnd hooks
    // lost their run to SQLITE_BUSY at 5s. 30s absorbs the longest observed
    // window; hooks run in the background, so the wait is invisible.
    conn.busy_timeout(Duration::from_secs(30))?;
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
            mtime REAL,
            session_type TEXT
        );",
    )?;

    migrate_fts_if_needed(conn)?;
    migrate_vec_chunks_if_needed(conn)?;
    migrate_qa_chunks_if_needed(conn)?;
    migrate_qa_chunk_rowid_link_if_needed(conn)?;
    migrate_session_type_if_needed(conn)?;

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
            timestamp INTEGER,
            src_rowid_lo INTEGER,
            src_rowid_hi INTEGER
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

/// Add src_rowid_lo/hi to pre-#192 databases. ALTER ... ADD COLUMN keeps every
/// existing row with the two columns NULL, so legacy chunks route through the
/// instr fallback in fetch_chunks. qa_chunks.id is unchanged, so vec_chunks
/// embeddings survive without a re-embed. `backfill_rowid_ranges` (indexer)
/// populates the NULL rows on the next `recall index`.
fn migrate_qa_chunk_rowid_link_if_needed(conn: &mut Connection) -> Result<()> {
    let Some(sql) = table_def(conn, "qa_chunks")? else {
        return Ok(());
    };

    if !sql.contains("src_rowid_lo") {
        let tx = conn.transaction()?;
        tx.execute_batch(
            "ALTER TABLE qa_chunks ADD COLUMN src_rowid_lo INTEGER;
             ALTER TABLE qa_chunks ADD COLUMN src_rowid_hi INTEGER;",
        )?;
        tx.commit()?;
    }

    Ok(())
}

/// Add session_type to pre-classification databases. ALTER ... ADD COLUMN keeps
/// every existing row (the new column is NULL, treated as interactive by the
/// search filter), so no re-index is forced — retroactive tagging is the explicit
/// `recall classify` command's job (#24).
fn migrate_session_type_if_needed(conn: &mut Connection) -> Result<()> {
    let Some(sql) = table_def(conn, "sessions")? else {
        return Ok(());
    };

    if !sql.contains("session_type") {
        let tx = conn.transaction()?;
        tx.execute_batch("ALTER TABLE sessions ADD COLUMN session_type TEXT;")?;
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

/// Seeds the minimal valid session row used across test modules. Centralizes
/// the 8-column positional INSERT so a schema change breaks one helper, not
/// every fixture (#24 broke them all when session_type was added).
#[cfg(test)]
pub(crate) fn seed_session(conn: &Connection, session_id: &str) {
    conn.execute(
        "INSERT INTO sessions VALUES (?1, 'claude', '/f', '/p', 'slug', 0, 0.0, NULL)",
        [session_id],
    )
    .unwrap();
}

/// Seeds one qa_chunk for the default test session `s1` with timestamp 0.
/// Sites that need another session or a real timestamp keep their inline INSERT.
#[cfg(test)]
pub(crate) fn seed_chunk(conn: &Connection, id: i64, content: &str) {
    conn.execute(
        "INSERT INTO qa_chunks (id, session_id, content, timestamp) VALUES (?1, 's1', ?2, 0)",
        rusqlite::params![id, content],
    )
    .unwrap();
}

#[cfg(test)]
mod tests;
