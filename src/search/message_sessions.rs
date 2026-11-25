use std::collections::{BTreeSet, HashSet};
use std::str;

use anyhow::Result;
use rusqlite::types::{ToSql, ValueRef};
use rusqlite::{Connection, OptionalExtension};

use super::SearchOptions;
use super::filter::append_search_filters;
use crate::db::table_def;

enum StoredSessionId {
    Readable(Option<String>),
    NotText,
}

fn stored_session_id(value: ValueRef<'_>) -> StoredSessionId {
    match value {
        ValueRef::Null => StoredSessionId::Readable(None),
        ValueRef::Text(bytes) => match str::from_utf8(bytes) {
            Ok(id) => StoredSessionId::Readable(Some(id.to_owned())),
            Err(_) => StoredSessionId::NotText,
        },
        ValueRef::Integer(_) | ValueRef::Real(_) | ValueRef::Blob(_) => StoredSessionId::NotText,
    }
}

pub(super) struct MatchedRow {
    pub(super) rowid: i64,
    pub(super) rank: f64,
    pub(super) session_id: Option<String>,
}

#[derive(Default)]
pub(super) struct MatchedRows {
    pub(super) rows: Vec<MatchedRow>,
    pub(super) not_text_ids: usize,
    pub(super) first_error: Option<rusqlite::Error>,
    pub(super) errors: usize,
}

pub(super) enum MessageSessions {
    StoredPerRow,
    RowidRanges(RangeOwners),
}

pub(super) struct RangeOwners {
    starts: Vec<i128>,
    owners: Vec<Option<usize>>,
    session_ids: Vec<String>,
}

impl RangeOwners {
    fn from_ranges(ranges: Vec<(String, i64, i64)>) -> Self {
        let mut events: Vec<(i128, bool, usize)> = Vec::new();
        let mut session_ids = Vec::with_capacity(ranges.len());
        for (i, (session_id, lo, hi)) in ranges.into_iter().enumerate() {
            session_ids.push(session_id);
            if lo <= hi {
                events.push((i128::from(lo), true, i));
                events.push((i128::from(hi) + 1, false, i));
            }
        }
        events.sort_unstable_by_key(|&(at, _, _)| at);
        let mut starts = Vec::new();
        let mut owners = Vec::new();
        let mut open: BTreeSet<usize> = BTreeSet::new();
        let mut events = events.into_iter().peekable();
        while let Some((at, opens, i)) = events.next() {
            if opens {
                open.insert(i);
            } else {
                open.remove(&i);
            }
            if events.peek().is_some_and(|&(next, _, _)| next == at) {
                continue;
            }
            starts.push(at);
            owners.push(match open.len() {
                1 => open.first().copied(),
                _ => None,
            });
        }
        Self {
            starts,
            owners,
            session_ids,
        }
    }

    fn sole_owner(&self, rowid: i64) -> Option<&str> {
        let segment = self
            .starts
            .partition_point(|&start| start <= i128::from(rowid))
            .checked_sub(1)?;
        let owner = self.owners[segment]?;
        Some(&self.session_ids[owner])
    }
}

impl MessageSessions {
    pub(super) fn load(conn: &Connection) -> Result<Self> {
        if !table_def(conn, "sessions")?.is_some_and(|sql| sql.contains("message_rowid_lo")) {
            return Ok(Self::StoredPerRow);
        }
        let mut stmt =
            conn.prepare("SELECT session_id, message_rowid_lo, message_rowid_hi FROM sessions")?;
        let complete: Option<Vec<(String, i64, i64)>> = stmt
            .query_map([], |row| {
                let StoredSessionId::Readable(Some(session_id)) =
                    stored_session_id(row.get_ref(0)?)
                else {
                    return Ok(None);
                };
                let rowid = |i| match row.get_ref(i) {
                    Ok(ValueRef::Integer(n)) => Ok(Some(n)),
                    Ok(_) => Ok(None),
                    Err(e) => Err(e),
                };
                Ok(rowid(1)?
                    .zip(rowid(2)?)
                    .map(|(lo, hi)| (session_id, lo, hi)))
            })?
            .collect::<rusqlite::Result<Option<_>>>()?;
        Ok(match complete {
            Some(ranges) => Self::RowidRanges(RangeOwners::from_ranges(ranges)),
            None => Self::StoredPerRow,
        })
    }

    pub(super) fn matched_rows(&self, conn: &Connection, fts_query: &str) -> MatchedRows {
        let mut matched = MatchedRows::default();
        let read = match self {
            Self::StoredPerRow => read_rows(
                conn,
                "SELECT session_id, rowid, rank FROM messages WHERE messages MATCH ?1",
                fts_query,
                &mut matched,
                |row| {
                    Ok(match stored_session_id(row.get_ref(0)?) {
                        StoredSessionId::Readable(session_id) => Some(MatchedRow {
                            session_id,
                            rowid: row.get(1)?,
                            rank: row.get(2)?,
                        }),
                        StoredSessionId::NotText => None,
                    })
                },
            ),
            Self::RowidRanges(owners) => read_rows(
                conn,
                "SELECT rowid, rank FROM messages WHERE messages MATCH ?1",
                fts_query,
                &mut matched,
                |row| {
                    Ok(Some(MatchedRow {
                        rowid: row.get(0)?,
                        rank: row.get(1)?,
                        session_id: None,
                    }))
                },
            )
            .and_then(|()| resolve_by_range(conn, owners, &mut matched)),
        };
        if let Err(e) = read {
            matched.errors += 1;
            matched.first_error.get_or_insert(e);
        }
        matched
    }
}

fn read_rows(
    conn: &Connection,
    sql: &str,
    fts_query: &str,
    matched: &mut MatchedRows,
    decode: impl FnMut(&rusqlite::Row) -> rusqlite::Result<Option<MatchedRow>>,
) -> rusqlite::Result<()> {
    let mut stmt = conn.prepare(sql)?;
    for row in stmt.query_map([fts_query], decode)? {
        match row {
            Ok(Some(row)) => matched.rows.push(row),
            Ok(None) => matched.not_text_ids += 1,
            Err(e) => {
                matched.errors += 1;
                matched.first_error.get_or_insert(e);
            }
        }
    }
    Ok(())
}

fn resolve_by_range(
    conn: &Connection,
    owners: &RangeOwners,
    matched: &mut MatchedRows,
) -> rusqlite::Result<()> {
    let mut stored = conn.prepare("SELECT session_id FROM messages WHERE rowid = ?1")?;
    let mut unreadable = Vec::new();
    for (i, row) in matched.rows.iter_mut().enumerate() {
        if let Some(owner) = owners.sole_owner(row.rowid) {
            row.session_id = Some(owner.to_owned());
            continue;
        }
        match stored
            .query_row([row.rowid], |r| Ok(stored_session_id(r.get_ref(0)?)))
            .optional()
        {
            Ok(None) => {}
            Ok(Some(StoredSessionId::Readable(session_id))) => row.session_id = session_id,
            Ok(Some(StoredSessionId::NotText)) => {
                matched.not_text_ids += 1;
                unreadable.push(i);
            }
            Err(e) => {
                matched.errors += 1;
                matched.first_error.get_or_insert(e);
                unreadable.push(i);
            }
        }
    }
    for i in unreadable.into_iter().rev() {
        matched.rows.remove(i);
    }
    Ok(())
}

pub(super) fn distinct_session_ids<'a>(
    rows: impl IntoIterator<Item = &'a MatchedRow>,
) -> Vec<Option<&'a str>> {
    let set: HashSet<Option<&str>> = rows.into_iter().map(|r| r.session_id.as_deref()).collect();
    set.into_iter().collect()
}

pub(super) struct FilteredSessions {
    pub(super) sql: String,
    pub(super) params: Vec<Box<dyn ToSql>>,
}

impl FilteredSessions {
    pub(super) fn new(
        select: &str,
        from_extra: &str,
        session_ids: &[Option<&str>],
        opts: &SearchOptions,
        now_ms: i64,
    ) -> Self {
        let ids = serde_json::Value::from(
            session_ids
                .iter()
                .map(|id| id.map_or(serde_json::Value::Null, serde_json::Value::from))
                .collect::<Vec<_>>(),
        );
        let mut sql = format!("SELECT {select} FROM json_each(?) j{from_extra} WHERE 1 = 1");
        let mut params: Vec<Box<dyn ToSql>> = vec![Box::new(ids.to_string())];
        append_search_filters(&mut sql, &mut params, "j.value", opts, now_ms);
        Self { sql, params }
    }
}

#[cfg(test)]
mod tests;
