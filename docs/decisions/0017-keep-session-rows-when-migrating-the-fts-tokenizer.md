---
status: "accepted"
date: 2026-11-13
decision-makers: thkt
scope: "src/db.rs schema migrations run by create_schema, chiefly migrate_fts_if_needed; deletions by the indexer (cleanup_orphans, DR 0004) are out of scope"
---

# Keep Session Rows When Migrating the FTS Tokenizer

## Context and Problem Statement

When `open_db` met a `messages` table built with an older tokenizer, `migrate_fts_if_needed` (src/db.rs) ran `DROP TABLE messages; DELETE FROM sessions; DROP TABLE IF EXISTS qa_chunks; DROP TABLE IF EXISTS vec_chunks;`. That deletion bypassed the removal gate of DR 0004, which keeps the rows under a missing or unreadable root. With the Claude root missing, `recall index` kept 1 session on a current index and left 0 on an index with the old tokenizer (#437). DR 0007 allowed a worst-case migration to drop and rebuild (its FTS retokenize example), while also stating that it does not relax DR 0004. How should the tokenizer migration change a table whose rows the index may hold no other copy of?

## Decision Drivers

- DR 0004: a run deletes a session row only when a full scan of its root shows the log is gone
- DR 0005: chunks point at message rows by rowid range, so a rebuilt `messages` must keep each rowid
- A migration runs when the index opens, before any scan of the session roots

## Considered Options

- Copy every row, rowid included, into a new table with the current tokenizer, then rename it
- Route the deletion through DR 0004's removal gate
- Keep the session rows and mark them to be parsed again
- Keep the deleting migration and document it
- Refuse the old index, as `check_index` refuses a legacy index that holds vectors

## Decision Outcome

Chosen option: "Copy every row into a new table", because it is the only option that loses no session text: the full-text table stores each message's text, so the new tokenizer can index it again without the session logs.

`migrate_fts_if_needed` takes the write lock with an immediate transaction and checks the tokenizer inside it, as the `parse_diagnostics`, `file_size` and `qa_chunk_generation` migrations do, so an opener that waited for the lock finds the table already migrated. It then creates the table under a temporary name from the same definition `create_schema` uses (`messages_table_sql`), copies `rowid, session_id, role, text` of every message that belongs to a session into it, drops `messages_vocab` and the old `messages`, renames the new table to `messages`, and recreates `messages_vocab`, all in the same transaction, so a later migration that fails leaves a consistent table. `sessions`, `qa_chunks` and `vec_chunks` are not touched. Before the copy it prints a one-line notice.

### Consequences

- Good, because a session whose root is missing or unreadable keeps its text, its chunks and their rowid ranges; the old migration deleted them
- Neutral, because a session whose log is present is parsed again on the next `recall index` either way: an index this old also lacks `parse_diagnostics`, `file_size` and the current `chunks_indexed` rules version, and those migrations make the indexer treat every session as changed. Such an index never stored embeddings, so there is no embedding pass to save
- Bad, because a kept session from an index older than 3e3eab0 (2026-03-14) keeps the text that parser stored, including the `system-reminder` and `command-name` tags the noise filter now drops, and its old chunks keep their content without rowid ranges (search falls back to `instr`). This lasts until the root returns: then the session is parsed again, or deleted if its log is gone
- Bad, because the copy holds the write lock for the whole retokenize: 7.7 s for 100,000 synthetic messages of about 500 characters with the sqlite3 CLI, so a larger index approaches the 30 s `busy_timeout` of a concurrent `SessionEnd` hook
- Bad, because the file grows during the copy (114 MB to 479 MB in that measurement); SQLite reuses the freed pages later and never shrinks the file on its own
- Bad, because the copy can fail where the deletion could not, for example on a full disk; the transaction then rolls back, and every later writer open tries the copy again and fails the same way until space is freed
- Bad, because the migrations after this one now meet the old `qa_chunks` and `vec_chunks` that the deletion used to drop first. Several of them check the table shape outside the write lock, so two writers opening the same old index at once can make one fail its open (its next open succeeds), and `migrate_vec_chunks_if_needed` still prints its "embeddings schema changed" warning for an old `vec_chunks`

### Confirmation

These tests fail against the deleting migration and pass against this one:

- `opening_an_index_with_an_older_tokenizer_moves_its_messages_onto_trigram_and_keeps_its_sessions` (src/db/tests.rs)
- `opening_the_oldest_index_shape_keeps_its_chunks_and_reaches_the_current_schema` (src/db/tests.rs)
- `index_upgrading_a_pre_trigram_index_keeps_the_sessions_of_a_missing_root_searchable` (tests/cli_integration.rs)

Code review checks that no migration runs `DELETE FROM sessions` and that a migration which rebuilds a table keeps its rows.

## Pros and Cons of the Options

### Copy every row into a new table

Rebuild the full-text index from the text it already stores.

- Good, because it needs nothing outside the database
- Bad, because the retokenize happens inside one write transaction

### Route the deletion through DR 0004's removal gate

Delete only the session rows whose root a scan read in full.

- Good, because it reuses the gate the indexer already applies
- Bad, because a migration runs when the index opens, before any scan result exists to decide with

### Keep the session rows and mark them to be parsed again

Drop `messages` but clear each session's stored file state, so the next index run parses its log again.

- Good, because the session list survives
- Bad, because the text of a session under an unreadable root is still gone until its root returns, and gone for good if its log is deleted first

### Keep the deleting migration and document it

Leave `DELETE FROM sessions` in place and say in the README that an upgrade drops the sessions of a missing root.

- Good, because it needs no code
- Bad, because it keeps a deletion DR 0004 forbids, and a note does not bring the sessions back

### Refuse the old index

Have `check_index` reject an index with an older tokenizer, as it rejects a legacy index that holds vectors.

- Good, because nothing is written to the old index
- Bad, because the user must build a new index, which still cannot hold the sessions of a missing root

## More Information

This decision supersedes, in DR 0007, the Decision Driver that lets a worst-case migration drop and rebuild ("FTS retokenize"), and the Confirmation's references to `test_fts_migration_rebuilds_on_schema_change` and `test_fts_migration_cascades_to_embedding_tables`. The three tests in Confirmation replace those two. The rest of DR 0007 (shape detection without a version table) stands, and DR 0007 itself is left unchanged.

Only some indexes reach this migration. The tokenizer became `trigram` in d1d2172 (2026-03-24). Since da5a4c0 (2026-10-16), `open_or_create_db` runs `check_index` before `open_db` and refuses a legacy index that holds vectors. The migration therefore runs only on an index built before 2026-03-24 that never stored an embedding and that no writer has opened since: an earlier `recall index` already ran the deleting migration on it.

Issue #437.

Challenge (2026-11-13): Conditional GO. It holds while the migration checks the tokenizer inside an immediate transaction and this record states that the copy keeps only the sessions whose logs are missing. Both critic-design passes returned "weakened"; their corrections are in the sections above.

### Trade-offs

Both migrations are followed by the same full parse of the sessions whose logs are present. The copy adds the write lock held for the retokenize (7.7 s per 100,000 messages in the measurement) and keeps the sessions of a missing root, which the deletion lost. Openers that wait for the lock re-check the tokenizer inside their own immediate transaction and skip the copy; one whose wait passes the 30 s `busy_timeout` fails that run, and its next run finds the migrated table. In WAL mode the copy is written to the `-wal` file first, and that file is not truncated after a checkpoint, so disk use can peak above the old database plus the new table.

### Reassessment Triggers

- The full-text table stops storing message text (for example a contentless FTS5 table), so the text can no longer be copied
- An upgraded index is large enough that the copy regularly outlasts the `busy_timeout` of concurrent hooks
- The text an old index stored differs from what the current parser stores enough that keeping it does more harm than losing the session (today's case is the noise tags that 3e3eab0 began filtering)
- A future tokenizer change also needs the session logs, for example because the new tokenizer needs text the table does not store
