use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process;
use std::time::Duration;

use amici::migration::notify_schema_change;
use anyhow::Result;
use rurico::embed::EMBEDDING_DIMS;
use rurico::storage::ensure_sqlite_vec;
use rusqlite::{Connection, OpenFlags};
use tracing::warn;

const FTS_TOKENIZER: &str = "trigram";

/// Which open strategy `open_db_readonly` succeeded with. `Direct` reads a live
/// `-wal` fresh via the existing `-shm`; `Immutable` was forced by a read-only
/// directory and cannot see un-checkpointed `-wal` changes (see `stale_wal_note`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenTier {
    Direct,
    Immutable,
}

/// On-disk schema currency, derived by reading `sqlite_master` only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchemaState {
    /// No `sessions` table: recall never indexed this DB.
    Empty,
    /// Current shape: usable by the read commands as-is.
    Current,
    /// Present but an old shape a `migrate_*` pass would rewrite; a read-only
    /// open cannot migrate it, so the read commands hard-fail (see main.rs).
    Stale,
}

/// Open the index read-only, never writing (no WAL/`-shm` sidecars, no schema
/// creation). This is the read-command counterpart to [`open_db`]; the write
/// path is left untouched (SOW NFR-001). The failed-open diagnosis path may
/// create and remove a transient `.recall-write-probe-<pid>` file in the DB's
/// parent directory (see [`dir_write_probe_fails`]); the DB itself stays untouched.
///
/// Tier 1 opens with `SQLITE_OPEN_READ_ONLY`, which in a writable directory reads
/// a live `-wal` fresh via the existing `-shm`. `open_with_flags` is lazy (it does
/// not touch the DB file until the first query), so a read-only directory does not
/// fail at open: a WAL-mode DB needs to create the `-shm`/`-wal` sidecars on the
/// first read and fails there with `SQLITE_READONLY_DIRECTORY`. We therefore probe
/// with a `sqlite_master` read and, only on that read-only-directory error, fall back
/// to tier 2: an `immutable=1` URI that tells SQLite the DB and its `-wal` will not
/// change, so it reads without any sidecar. Any other error (missing/unreadable file,
/// EMFILE, TOCTOU deletion, corruption) propagates unchanged rather than being masked
/// by a doomed second open.
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

/// Force the pager to touch the DB file: reading `sqlite_master` opens page 1,
/// which for a WAL-mode DB requires the `-shm`/`-wal` sidecars and thus surfaces a
/// read-only-directory failure that lazy `open_with_flags` hid.
fn probe_readable(conn: &Connection) -> rusqlite::Result<()> {
    conn.query_row("SELECT count(*) FROM sqlite_master", [], |_| Ok(()))
}

/// True only when a WAL-mode read failed *because its directory is read-only*, the sole
/// cause that justifies the tier-2 `immutable=1` fallback (which disables change
/// detection). A read-only directory blocks sidecar creation, surfacing as either
/// `CannotOpen` (a missing `-shm` cannot be created; extended code 14) or `ReadOnly`
/// (a needed rollback journal / `-wal` cannot be created; includes
/// `SQLITE_READONLY_DIRECTORY` 1544). Those primary codes are ambiguous on their own,
/// so we additionally require the directory to be genuinely unwritable: a missing or
/// unreadable DB file, an fd exhaustion (EMFILE), or a TOCTOU deletion also surface as
/// `CannotOpen`/`ReadOnly` but occur in a *writable* directory, where `immutable=1`
/// cannot help and would either mask a real error or read a still-mutating DB as frozen.
/// Corruption (`SQLITE_CORRUPT` -> `DatabaseCorrupt`) is a different primary code and
/// never matches, so it always propagates.
///
/// Directory writability is judged by [`dir_is_unwritable`]; see its doc for the two
/// signals (mode bits, then an empirical write probe) and the treated-as-writable
/// ambiguous cases.
fn is_readonly_dir_error(db_path: &Path, e: &rusqlite::Error) -> bool {
    let sidecar_failure = matches!(
        e.sqlite_error_code(),
        Some(rusqlite::ErrorCode::CannotOpen | rusqlite::ErrorCode::ReadOnly)
    );
    sidecar_failure && dir_is_unwritable(db_path)
}

/// True when the DB file's parent directory is unwritable, checked via two signals in
/// order. First, permission mode bits (e.g. `chmod 0o555`), which is independent of the
/// effective uid (a root process still sees `0o555` as read-only) and never touches the
/// filesystem's write path. Only when those bits still look writable does it fall back
/// to [`dir_write_probe_fails`], an empirical write attempt that also catches a genuine
/// read-only mount (DMG, network share, most external media) whose directory bits still
/// read `0o755`. A missing parent or unreadable metadata is treated as writable, so an
/// ambiguous error propagates instead of being masked by the `immutable=1` fallback.
fn dir_is_unwritable(db_path: &Path) -> bool {
    let Some(dir) = db_path.parent().filter(|dir| !dir.as_os_str().is_empty()) else {
        return false;
    };
    let Ok(meta) = fs::metadata(dir) else {
        return false;
    };
    meta.permissions().readonly() || dir_write_probe_fails(dir)
}

/// True when `dir` rejects file creation (read-only mount, or write bits
/// cleared as in `chmod 0o555`), determined empirically rather than from
/// permission mode bits (unlike [`dir_is_unwritable`]): a `.recall-write-probe-<pid>`
/// file is created with `create_new` (so it never collides with or clobbers an
/// existing file) and immediately removed on success. Returns `true` only for
/// `PermissionDenied` / `ReadOnlyFilesystem`; any other failure (e.g. `NotFound`
/// from a missing directory, or an uncategorized error such as EMFILE) returns
/// `false` so the caller falls back to propagating the original error instead of
/// misreporting an ambiguous failure as a read-only directory.
///
/// Called from [`dir_is_unwritable`] only when the mode-bits check finds the
/// directory writable, to also catch a read-only mount whose directory bits
/// still read `0o755`.
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

/// Remove the transient write-probe file, warning instead of failing the read
/// when the cleanup cannot complete and the probe file litters the directory.
fn remove_write_probe(probe: &Path) {
    if let Err(e) = fs::remove_file(probe) {
        warn!(probe = %probe.display(), error = %e, "failed to remove write probe");
    }
}

/// Tier 2: reopen via an `immutable=1` URI, which reads without any sidecar.
fn open_immutable(path: &Path, flags: OpenFlags) -> Result<(Connection, OpenTier)> {
    let uri = format!("file:{}?immutable=1", encode_uri_path(path));
    let conn = Connection::open_with_flags(uri, flags)?;
    conn.busy_timeout(Duration::from_secs(30))?;
    Ok((conn, OpenTier::Immutable))
}

/// Percent-encode a filesystem path for use in a `file:` URI. Every byte outside
/// the RFC 3986 unreserved set is escaped; `/` is kept as the path separator.
/// Hand-rolled (no new dependency) since the only caller is the immutable-open
/// fallback and the input is an absolute local path.
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

/// The `-wal` sidecar path for a DB file (`<path>-wal`).
fn wal_path(path: &Path) -> PathBuf {
    let mut s = path.as_os_str().to_owned();
    s.push("-wal");
    PathBuf::from(s)
}

/// Classify the on-disk schema by reading `sqlite_master` only (never writes),
/// reusing [`table_def`] so "current shape" is defined next to the `migrate_*`
/// sniffs it mirrors. Each `Stale` condition is a `migrate_*` check inverted to
/// "a migration would run".
pub fn schema_state(conn: &Connection) -> Result<SchemaState> {
    let Some(sessions_sql) = table_def(conn, "sessions")? else {
        return Ok(SchemaState::Empty);
    };

    // A core table missing while `sessions` exists is a crash artifact: it would
    // pass every "nothing to migrate" sniff yet leak a raw `no such table` error
    // to the caller, so treat it as stale (needs a rebuild).
    let (Some(messages_sql), Some(qa_sql), Some(vec_sql), Some(_)) = (
        table_def(conn, "messages")?,
        table_def(conn, "qa_chunks")?,
        table_def(conn, "vec_chunks")?,
        // Presence-only: `search` hard-requires this fts5vocab table but its DDL
        // has no versioned shape to sniff, so a missing one is a crash artifact.
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

/// A warning when the immutable (tier-2) open cannot see un-checkpointed `-wal`
/// changes. Returns `None` on the `Direct` tier (SQLite reads the `-wal` fresh
/// there, so a note would be a false positive) and when `<path>-wal` is absent or
/// empty. No checkpoint is performed (byte-for-byte constraint, SOW assumption 3).
pub fn stale_wal_note(path: &Path, tier: OpenTier) -> Option<String> {
    if tier != OpenTier::Immutable {
        return None;
    }
    const PREFIX: &str = "read-only open cannot see uncommitted changes in the write-ahead log; ";
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
