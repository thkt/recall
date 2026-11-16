use std::path::Path;

use anyhow::Result;
use rusqlite::Connection;

use crate::db::{self, SchemaState};
use crate::envelope::{CommandOutput, Note};
use crate::error::RecallError;
use crate::require_current_schema;

/// The index opened read-only by a command that reads it. Opened as an immutable
/// snapshot beside a write-ahead log it cannot read, it carries a note, and every
/// read command reports that note the same way (DR 0001).
pub(crate) struct ReadIndex {
    pub(crate) conn: Connection,
    stale_note: Option<String>,
}

impl ReadIndex {
    pub(crate) fn open(path: &Path) -> Result<Self> {
        let (conn, tier) = db::open_db_readonly(path)?;
        Ok(Self {
            conn,
            stale_note: db::stale_wal_note(path, tier),
        })
    }

    /// The schema of the index, `Empty` or `Current`. An out-of-date schema is an
    /// error, which names the note: the newer schema may sit in the log.
    pub(crate) fn schema(&self) -> Result<SchemaState> {
        let state = db::schema_state(&self.conn)?;
        require_current_schema(state).map_err(|error| self.fail(error.into()))?;
        Ok(state)
    }

    /// `out` from a stale snapshot is degraded and carries the note last.
    pub(crate) fn finish(&self, mut out: CommandOutput) -> CommandOutput {
        if let Some(note) = &self.stale_note {
            out.degraded = true;
            out.notes.push(Note::new(note.clone()));
        }
        out
    }

    /// `error` from a stale snapshot becomes a read error that names the note: what
    /// the command did not find may sit in the log.
    pub(crate) fn fail(&self, error: anyhow::Error) -> anyhow::Error {
        match &self.stale_note {
            Some(note) => RecallError::DatabaseRead(format!("{error}; {note}")).into(),
            None => error,
        }
    }

    /// Refuses a stale snapshot, for a command whose output would be wrong, not
    /// just short, without the log.
    pub(crate) fn refuse_stale_snapshot(&self) -> Result<()> {
        match &self.stale_note {
            Some(note) => Err(RecallError::DatabaseRead(note.clone()).into()),
            None => Ok(()),
        }
    }
}
