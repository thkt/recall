//! What `sessions.chunks_indexed` records about a session. The column holds the version of the
//! index rules its messages were parsed under, and its sign tells whether its chunks were made.
//!
//! | Value | State |
//! | --- | --- |
//! | `-INDEX_RULES_VERSION` | parsed under the current rules; chunks pending |
//! | `INDEX_RULES_VERSION` | parsed and chunked under the current rules |
//! | `NULL` | rules version unknown; chunks pending |
//! | another negative value | parsed under other rules; chunks pending |
//! | `0` | chunked from messages whose rules version was unknown or other |
//! | another positive value | parsed and chunked under other rules, such as `1` from the migration that added the column |
//!
//! Only a session chunked under the current rules counts as unchanged. The scan parses every
//! other session again when its log is found, except that it keeps a session with stored vectors
//! unparsed while the model cannot embed.

use anyhow::Result;
use rusqlite::Transaction;

const INDEX_RULES_VERSION: i64 = 2;

pub(super) const PARSED: i64 = -INDEX_RULES_VERSION;

// why-not: a const, because `concat!` accepts only literals and cannot splice a const
macro_rules! chunks_pending {
    () => {
        "(chunks_indexed IS NULL OR chunks_indexed < 0)"
    };
}
pub(super) use chunks_pending;

pub(super) fn is_current(value: Option<i64>) -> bool {
    value == Some(INDEX_RULES_VERSION)
}

pub(super) fn has_current_chunks(tx: &Transaction, session_id: &str) -> Result<bool> {
    Ok(tx.query_row(
        "SELECT EXISTS(SELECT 1 FROM sessions WHERE session_id = ?1 AND chunks_indexed = ?2 \
         AND EXISTS(SELECT 1 FROM qa_chunks WHERE session_id = ?1))",
        rusqlite::params![session_id, INDEX_RULES_VERSION],
        |row| row.get(0),
    )?)
}

pub(super) fn record_chunked(tx: &Transaction, session_id: &str) -> Result<()> {
    tx.execute(
        "UPDATE sessions SET chunks_indexed = CASE WHEN chunks_indexed = ?1 THEN ?2 ELSE 0 END \
         WHERE session_id = ?3",
        rusqlite::params![PARSED, INDEX_RULES_VERSION, session_id],
    )?;
    Ok(())
}
