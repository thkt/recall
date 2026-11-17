use std::path::Path;

use anyhow::Result;
use rusqlite::Connection;

use crate::db::{self, SchemaState};
use crate::envelope::{CommandOutput, Note};
use crate::error::RecallError;
use crate::require_current_schema;

/// The index opened read-only by a command that reads it.
pub(crate) struct ReadIndex {
    pub(crate) conn: Connection,
    pub(crate) snapshot: Snapshot,
}

/// Whether the index was opened as an immutable snapshot beside a write-ahead log
/// it cannot read. DR 0001 has a read command report the note of such a snapshot:
/// a result is degraded and carries it, and some failures name it.
pub(crate) struct Snapshot {
    stale_note: Option<String>,
}

impl ReadIndex {
    pub(crate) fn open(path: &Path) -> Result<Self> {
        let (conn, tier) = db::open_db_readonly(path)?;
        Ok(Self {
            conn,
            snapshot: Snapshot {
                stale_note: db::stale_wal_note(path, tier),
            },
        })
    }

    /// Whether recall never indexed into the file. An out-of-date schema is an
    /// error, which names the note: the newer schema may sit in the log.
    pub(crate) fn is_never_indexed(&self) -> Result<bool> {
        let state = db::schema_state(&self.conn)?;
        require_current_schema(state).map_err(|error| self.snapshot.fail(error.into()))?;
        Ok(state == SchemaState::Empty)
    }
}

impl Snapshot {
    /// `out` from a stale snapshot is degraded, and the note follows the notes it
    /// already carries.
    #[must_use]
    pub(crate) fn finish(&self, mut out: CommandOutput) -> CommandOutput {
        if let Some(note) = &self.stale_note {
            out.degraded = true;
            out.notes.push(Note::new(note.clone()));
        }
        out
    }

    /// `error` from a stale snapshot becomes a read error that names the note: what
    /// the command did not find may sit in the log.
    #[must_use]
    pub(crate) fn fail(&self, error: anyhow::Error) -> anyhow::Error {
        match &self.stale_note {
            Some(note) => RecallError::DatabaseRead(format!("{error}; {note}")).into(),
            None => error,
        }
    }

    /// Refuses a stale snapshot, for a command whose output would be wrong, not
    /// just short, without the log.
    pub(crate) fn refuse(&self) -> Result<()> {
        match &self.stale_note {
            Some(note) => Err(RecallError::DatabaseRead(note.clone()).into()),
            None => Ok(()),
        }
    }
}
