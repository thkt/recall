use anyhow::Result;
use rusqlite::Transaction;

const INDEX_RULES_VERSION: i64 = 2;

pub(super) const PARSED_WITH_CHUNKS_PENDING: i64 = -INDEX_RULES_VERSION;

const CHUNKED_UNDER_UNKNOWN_OR_OTHER_RULES: i64 = 0;

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
        "UPDATE sessions SET chunks_indexed = CASE WHEN chunks_indexed = ?1 THEN ?2 ELSE ?3 END \
         WHERE session_id = ?4",
        rusqlite::params![
            PARSED_WITH_CHUNKS_PENDING,
            INDEX_RULES_VERSION,
            CHUNKED_UNDER_UNKNOWN_OR_OTHER_RULES,
            session_id
        ],
    )?;
    Ok(())
}
