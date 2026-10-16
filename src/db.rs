use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process;
use std::time::Duration;

use crate::embedding::EMBEDDING_DIMS;
use crate::runtime::notify_schema_change;
use crate::storage::ensure_sqlite_vec;
use anyhow::Result;
use rusqlite::{Connection, OpenFlags, TransactionBehavior};
use tracing::warn;

const FTS_TOKENIZER: &str = "trigram";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenTier {
    Direct,
    // Immutable reads ignore WAL sidecars; callers must report stale snapshots.
    Immutable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchemaState {
    Empty,
    Current,
    Stale,
}

pub fn open_db_readonly(path: &Path) -> Result<(Connection, OpenTier)> {
    ensure_sqlite_vec().map_err(|e| anyhow::anyhow!(e))?;
    let flags = OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI;

    match Connection::open_with_flags(path, flags) {
        Ok(conn) => {
            conn.busy_timeout(Duration::from_secs(30))?;
            match probe_readable(&conn) {
                Ok(()) => Ok((conn, OpenTier::Direct)),
                Err(e) if is_readonly_dir_error(path, &e) => {
                    drop(conn);
                    open_immutable(path, flags)
                }
                Err(e) => Err(e.into()),
            }
        }
        Err(e) if is_readonly_dir_error(path, &e) => open_immutable(path, flags),
        Err(e) => Err(e.into()),
    }
}

fn probe_readable(conn: &Connection) -> rusqlite::Result<()> {
    conn.query_row("SELECT count(*) FROM sqlite_master", [], |_| Ok(()))
}

fn is_readonly_dir_error(db_path: &Path, e: &rusqlite::Error) -> bool {
    let sidecar_failure = matches!(
        e.sqlite_error_code(),
        Some(rusqlite::ErrorCode::CannotOpen | rusqlite::ErrorCode::ReadOnly)
    );
    sidecar_failure && dir_is_unwritable(db_path)
}

fn dir_is_unwritable(db_path: &Path) -> bool {
    let Some(dir) = db_path.parent().filter(|dir| !dir.as_os_str().is_empty()) else {
        return false;
    };
    let Ok(meta) = fs::metadata(dir) else {
        return false;
    };
    meta.permissions().readonly() || dir_write_probe_fails(dir)
}

fn dir_write_probe_fails(dir: &Path) -> bool {
    let probe = dir.join(format!(".recall-write-probe-{}", process::id()));
    match fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&probe)
    {
        Ok(_) => {
            remove_write_probe(&probe);
            false
        }
        Err(e) => matches!(
            e.kind(),
            io::ErrorKind::PermissionDenied | io::ErrorKind::ReadOnlyFilesystem
        ),
    }
}

fn remove_write_probe(probe: &Path) {
    if let Err(e) = fs::remove_file(probe) {
        warn!(probe = %probe.display(), error = %e, "failed to remove write probe");
    }
}

fn open_immutable(path: &Path, flags: OpenFlags) -> Result<(Connection, OpenTier)> {
    let uri = format!("file:{}?immutable=1", encode_uri_path(path));
    let conn = Connection::open_with_flags(uri, flags)?;
    conn.busy_timeout(Duration::from_secs(30))?;
    Ok((conn, OpenTier::Immutable))
}

fn encode_uri_path(path: &Path) -> String {
    let mut out = String::new();
    for &b in path.as_os_str().as_encoded_bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' | b'/' => {
                out.push(b as char);
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

fn wal_path(path: &Path) -> PathBuf {
    let mut s = path.as_os_str().to_owned();
    s.push("-wal");
    PathBuf::from(s)
}

pub fn schema_state(conn: &Connection) -> Result<SchemaState> {
    let Some(sessions_sql) = table_def(conn, "sessions")? else {
        return Ok(SchemaState::Empty);
    };

    let (Some(messages_sql), Some(qa_sql), Some(vec_sql), Some(_)) = (
        table_def(conn, "messages")?,
        table_def(conn, "qa_chunks")?,
        table_def(conn, "vec_chunks")?,
        table_def(conn, "messages_vocab")?,
    ) else {
        return Ok(SchemaState::Stale);
    };

    let stale = !sessions_sql.contains("session_type")
        || !messages_sql.contains(FTS_TOKENIZER)
        || !vec_sql.contains("sub_idx")
        || qa_sql.contains("chunk_hash")
        || !qa_sql.contains("src_rowid_lo");

    Ok(if stale {
        SchemaState::Stale
    } else {
        SchemaState::Current
    })
}

pub fn stale_wal_note(path: &Path, tier: OpenTier) -> Option<String> {
    if tier != OpenTier::Immutable {
        return None;
    }
    const PREFIX: &str =
        "read-only open cannot see uncheckpointed changes in the write-ahead log; ";
    match fs::metadata(wal_path(path)) {
        Ok(m) if m.len() > 0 => Some(if dir_is_unwritable(path) {
            format!(
                "{PREFIX}the directory is read-only, so copy the DB together with its \
                 `-wal` (and `-shm`, if present) sidecar files elsewhere and run \
                 `recall index --db-path <copy>` to checkpoint them there."
            )
        } else {
            format!("{PREFIX}run `recall index` to checkpoint them.")
        }),
        _ => None,
    }
}

pub fn open_db(path: &Path) -> Result<Connection> {
    ensure_sqlite_vec().map_err(|e| anyhow::anyhow!(e))?;
    let mut conn = Connection::open(path)?;
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
            file_size INTEGER,
            session_type TEXT,
            files_scanned INTEGER,
            chunks_indexed INTEGER
        );",
    )?;

    migrate_parse_diagnostics_if_needed(conn)?;

    migrate_fts_if_needed(conn)?;
    migrate_vec_chunks_if_needed(conn)?;
    migrate_qa_chunks_if_needed(conn)?;
    migrate_qa_chunk_rowid_link_if_needed(conn)?;
    migrate_session_type_if_needed(conn)?;
    migrate_files_scanned_if_needed(conn)?;
    migrate_file_size_if_needed(conn)?;

    conn.execute_batch("DROP TABLE IF EXISTS embedded_chunk_ids;")?;

    conn.execute_batch(
        "CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project);
         CREATE INDEX IF NOT EXISTS idx_sessions_timestamp ON sessions(timestamp);",
    )?;

    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS session_files (
            session_id TEXT NOT NULL,
            path TEXT NOT NULL,
            PRIMARY KEY (session_id, path)
        );",
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
            src_rowid_hi INTEGER,
            generation INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_qa_chunks_session ON qa_chunks(session_id);",
    )?;

    migrate_qa_chunk_generation_if_needed(conn)?;

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

    migrate_chunks_indexed_if_needed(conn)?;

    Ok(())
}

fn migrate_parse_diagnostics_if_needed(conn: &mut Connection) -> Result<()> {
    let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
    if table_def(&tx, "parse_diagnostics")?.is_some() {
        tx.commit()?;
        return Ok(());
    }
    tx.execute_batch(
        "CREATE TABLE IF NOT EXISTS parse_diagnostics (
            file_path TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            invalid_json_lines INTEGER NOT NULL DEFAULT 0,
            invalid_utf8_lines INTEGER NOT NULL DEFAULT 0,
            incomplete_tail_lines INTEGER NOT NULL DEFAULT 0,
            read_error INTEGER NOT NULL DEFAULT 0
        );",
    )?;

    tx.execute("UPDATE sessions SET mtime = NULL", [])?;
    tx.commit()?;
    Ok(())
}

fn migrate_file_size_if_needed(conn: &mut Connection) -> Result<()> {
    let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
    if !table_def(&tx, "sessions")?.is_some_and(|sql| sql.contains("file_size")) {
        tx.execute_batch("ALTER TABLE sessions ADD COLUMN file_size INTEGER;")?;
    }
    tx.commit()?;
    Ok(())
}

fn migrate_qa_chunk_generation_if_needed(conn: &mut Connection) -> Result<()> {
    // Global generations distinguish deleted/reused chunk IDs from old inference snapshots.
    let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
    if !table_def(&tx, "qa_chunks")?.is_some_and(|sql| sql.contains("generation")) {
        tx.execute_batch(
            "ALTER TABLE qa_chunks ADD COLUMN generation INTEGER NOT NULL DEFAULT 0;",
        )?;
    }
    tx.execute_batch(
        "CREATE TABLE IF NOT EXISTS qa_chunk_generation (
            singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
            value INTEGER NOT NULL CHECK (typeof(value) = 'integer' AND value >= 0)
         );
         INSERT OR IGNORE INTO qa_chunk_generation VALUES (1, 0);
         CREATE TRIGGER IF NOT EXISTS qa_chunks_insert_generation
         AFTER INSERT ON qa_chunks BEGIN
            UPDATE qa_chunk_generation SET value = value + 1 WHERE singleton = 1;
            UPDATE qa_chunks SET generation = (SELECT value FROM qa_chunk_generation)
            WHERE id = NEW.id;
         END;
         CREATE TRIGGER IF NOT EXISTS qa_chunks_update_generation
         AFTER UPDATE OF id, session_id, content ON qa_chunks BEGIN
            UPDATE qa_chunk_generation SET value = value + 1 WHERE singleton = 1;
            UPDATE qa_chunks SET generation = (SELECT value FROM qa_chunk_generation)
            WHERE id = NEW.id;
         END;",
    )?;
    tx.commit()?;
    Ok(())
}

pub(crate) fn table_def(conn: &Connection, name: &str) -> Result<Option<String>> {
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

fn migrate_files_scanned_if_needed(conn: &mut Connection) -> Result<()> {
    let Some(sql) = table_def(conn, "sessions")? else {
        return Ok(());
    };

    if !sql.contains("files_scanned") {
        let tx = conn.transaction()?;
        tx.execute_batch("ALTER TABLE sessions ADD COLUMN files_scanned INTEGER;")?;
        tx.commit()?;
    }

    Ok(())
}

fn migrate_chunks_indexed_if_needed(conn: &mut Connection) -> Result<()> {
    let tx = conn.transaction()?;
    if table_def(&tx, "sessions")?.is_some_and(|sql| !sql.contains("chunks_indexed")) {
        tx.execute_batch(
            "ALTER TABLE sessions ADD COLUMN chunks_indexed INTEGER;
             UPDATE sessions SET chunks_indexed = 1
             WHERE EXISTS (SELECT 1 FROM qa_chunks c WHERE c.session_id = sessions.session_id);",
        )?;
    }
    tx.commit()?;
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
pub(crate) fn seed_session(conn: &Connection, session_id: &str) {
    conn.execute(
        "INSERT INTO sessions (session_id, source, file_path, project, slug, timestamp, mtime) VALUES (?1, 'claude', '/f', '/p', 'slug', 0, 0.0)",
        [session_id],
    )
    .unwrap();
}

#[cfg(test)]
pub(crate) fn seed_session_file(conn: &Connection, session_id: &str, path: &str) {
    conn.execute(
        "INSERT INTO session_files (session_id, path) VALUES (?1, ?2)",
        [session_id, path],
    )
    .unwrap();
}

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
