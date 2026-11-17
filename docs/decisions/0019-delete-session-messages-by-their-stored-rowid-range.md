---
status: "accepted"
date: 2026-11-17
decision-makers: thkt
scope: "How the indexer deletes a session's rows in the messages FTS5 table (src/indexer.rs), and the sessions columns and schema upgrade that support it (src/db.rs)"
---

# Delete Session Messages by Their Stored Rowid Range

## Context and Problem Statement

`messages` is an FTS5 table, and its `session_id` column is UNINDEXED: FTS5 keeps no index on it, so `DELETE FROM messages WHERE session_id = ?` reads every row (`SCAN messages VIRTUAL TABLE INDEX 0:`). The indexer ran that delete for every session it wrote. On a synthetic index of 6,000 sessions and 58,602 messages (73 MB of text, the coverage scale in `.claude/OUTCOME.md`), one delete took about 25 ms with a warm page cache and about 600 ms cold. The first `recall index` spent 81 s of its 83 s in the full-text transaction, and `recall rebuild`, which writes every present session again, spent 205 s of 208 s. How should the indexer find the messages of one session without reading the whole table?

## Decision Drivers

- `recall rebuild` is the command recall names when the schema is out of date, so its cost reaches every user whose index needs it
- DR 0005: chunks point at messages by rowid, so the messages keep their rowids
- DR 0004: no run deletes a row that its scan did not prove gone
- DR 0007: a schema change is a column add found by its shape, with an upgrade test for an old index
- Search, `show` and `status` open the index read-only and cannot upgrade it

## Considered Options

- Store each session's message rowid range in its session row, and delete inside that range
- Skip the delete only for a session that has no session row
- Move the message text to an external-content FTS5 table whose content table indexes `session_id`
- Delete the messages of every changed session with one statement per run

## Decision Outcome

Chosen option: "Store each session's message rowid range", because it is the only option that stops the full read for a new session and for a changed one while it leaves the `messages` table and its rowids as they are.

`sessions` gains `message_rowid_lo` and `message_rowid_hi`. `upsert_session` records the smallest and the largest rowid it inserted for the session, or the empty range `(1, 0)` (`EMPTY_MESSAGE_RANGE`) for a session without messages. `delete_session_messages` (src/indexer.rs) reads the session row first:

- No row: it deletes nothing. Every message belongs to a session row, as the next paragraph explains
- A range: it runs `DELETE FROM messages WHERE rowid BETWEEN lo AND hi AND session_id = ?`. FTS5 reads only the rows of that rowid range, and the `session_id` test keeps the rows of any other session inside it
- A NULL range: it runs the full-read delete. A recall that predates the columns writes such a row, because its `INSERT OR REPLACE` leaves the two columns NULL

Every caller deletes the messages before the session row. `cleanup_orphans` and the branch for a log that now names another session id used to delete the session row first, and now delete in that order.

`migrate_message_rowid_range_if_needed` (src/db.rs) adds the two columns inside an immediate transaction. In the same transaction it deletes the messages whose session row is gone, then fills each range with one `GROUP BY session_id` read; a session without messages gets the empty range. Search never shows those messages: it skips a match that has no session row and counts it as missing. `schema_state` does not check the columns, because only the writer reads them and the writer upgrades the index when it opens it.

### Consequences

- Good, because a session that a run writes for the first time no longer reads the table, and a changed session reads only its own rowid range. On the synthetic index, deleting one session's messages by its range took 0.2 ms against 25 ms by `session_id`
- Good, because an older recall can still write the index: the NULL ranges it leaves fall back to the full-read delete
- Bad, because correctness now rests on two invariants that no constraint enforces: every message has a session row, and the stored range covers every message of its session. A new path that inserts messages without recording the range, or deletes a session row before its messages, breaks them and no query fails
- Bad, because the upgrade reads the whole `messages` table twice (the delete of messages without a session row, then the `GROUP BY`) while it holds the write lock
- Neutral, because a backfilled range spans the rows of other sessions when an older recall wrote the messages of one session in separate runs. The delete then reads more rows than the session holds, and the `session_id` test still keeps the other rows

### Confirmation

- `upgrading_an_index_without_message_ranges_stores_each_range_and_drops_messages_without_a_session` (src/db/tests.rs) checks the upgrade, its empty range, and that a second open changes nothing
- `a_changed_log_loses_only_its_own_messages_inside_the_rowid_range_its_session_stores` (src/indexer/tests.rs) fails when the range delete drops its `session_id` test
- `a_changed_log_whose_session_row_has_no_rowid_range_still_loses_its_old_messages` (src/indexer/tests.rs) fails when a NULL range deletes nothing
- `a_log_rewritten_under_another_session_id_loses_the_messages_of_the_session_it_held` and `deleting_a_log_under_a_fully_read_root_removes_its_session_messages_chunks_embeddings_and_touched_files` (src/indexer/tests.rs) fail when a caller deletes the session row before its messages

Code review checks that a new `INSERT INTO messages` records the range and that a new `DELETE FROM sessions` runs after `delete_session_dependents`.

## Pros and Cons of the Options

### Store each session's message rowid range

Keep the rowid range of a session's messages in its session row and delete through it.

- Good, because it serves the first index, `rebuild` and the incremental run alike
- Bad, because it adds two columns, an upgrade and two invariants that only review guards

### Skip the delete only for a session that has no session row

Run the full-read delete only when the session was stored before.

- Good, because it changes no schema; in a trial build the first index went from 83 s to 8.2 to 8.7 s
- Bad, because `rebuild` and every changed session still read the whole table
- Bad, because it rests on the same invariant that every message has a session row

### Move the message text to an external-content FTS5 table

Store the messages in an ordinary table with an index on `session_id`, and let FTS5 index that table.

- Good, because the index on `session_id` serves every delete and lookup by session
- Bad, because the upgrade rebuilds the whole full-text index, as DR 0017's copy does, and a delete from an external-content table must hand FTS5 the old text of each row
- Bad, because it changes the table that every search reads

### Delete the messages of every changed session with one statement per run

Collect the changed session ids during the run and delete their old messages in one statement at the end.

- Good, because one run reads the table once however many sessions changed, and the schema stays as it is
- Bad, because each run that changes any session still reads the whole table, which is the common case for the session-end hook
- Bad, because the delete has to spare the rows the run just inserted for the same sessions, which needs a rowid bound taken before the run, and it moves every delete out of the per-session write

## More Information

The measurements in Context come from a synthetic index generated in scratch space, with a release build and no model. The pull request on branch `perf/session-message-range` records the before and after runs.

This record does not make an exception to DR 0004. The upgrade deletes only messages that no session row owns, never a session row, and the removal gate of `cleanup_orphans` is unchanged. DR 0017's tokenizer copy already leaves such messages behind.

For DR 0005, the range comes from the rowids that each insert returned. FTS5 gives a new row the largest rowid plus one, so a session whose messages were last in the table can receive the rowids it has just freed. The recorded range follows the rowids the session actually received.

For DR 0007, the upgrade finds the old shape by the substring `message_rowid_lo`, which no other column name of `sessions` contains.

### Trade-offs

Skipping the delete for a session without a row would have been a change of a few lines, and it would have fixed only the first index. Storing the range adds two columns, an upgrade and two invariants. In return, `rebuild` and every changed session stop reading the whole table.

### Reassessment Triggers

- FTS5 can index a column it does not tokenize, or `messages` moves to an external-content table for another reason
- A second writer starts to insert messages, for example an importer; it must record the range or leave it NULL
- On large indexes, the two reads of the upgrade outlast the 30 s `busy_timeout` of a concurrent session-end hook
