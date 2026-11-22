use std::path::Path;

use anyhow::Result;
use rusqlite::Connection;

use crate::db::{self, SchemaState};
use crate::envelope::{CommandOutput, Note};
use crate::error::RecallError;
use crate::require_current_schema;

pub(crate) struct ReadIndex {
    pub(crate) conn: Connection,
    pub(crate) snapshot: Snapshot,
}

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

    pub(crate) fn is_never_indexed(&self) -> Result<bool> {
        let state = db::schema_state(&self.conn)?;
        require_current_schema(state).map_err(|error| self.snapshot.fail(error.into()))?;
        Ok(state == SchemaState::Empty)
    }
}

impl Snapshot {
    #[must_use]
    pub(crate) fn finish(&self, mut out: CommandOutput) -> CommandOutput {
        if let Some(note) = &self.stale_note {
            out.degraded = true;
            out.notes.push(Note::new(note.clone()));
        }
        out
    }

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
