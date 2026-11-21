---
status: "accepted"
date: "2026-11-22"
decision-makers: "thkt"
scope: "How the indexer and the embedder find and delete a chunk's rows in the vec_chunks vec0 table (src/indexer.rs, src/embedder.rs), and the qa_chunks columns and schema upgrade that support it (src/db.rs)"
---

# Delete Chunk Embeddings by Their Stored Rowid Range

## Context and Problem Statement

`vec_chunks` is a sqlite-vec vec0 table, and its `chunk_id` column is auxiliary (`+chunk_id`). sqlite-vec 0.1.9 keeps no index on an auxiliary column, so every statement that finds a chunk's embeddings by `chunk_id` reads the whole table (`SCAN vec_chunks VIRTUAL TABLE INDEX 0:1`). Four statements did that on main d37ff86:

- `delete_session_dependents` deleted a session's embeddings with `chunk_id IN (SELECT id FROM qa_chunks WHERE session_id = ?)`. `recall rebuild` runs it for every session it reads again, and `recall index` runs it for a changed session without current chunks and for a session whose log is gone
- `reconcile_chunks` read which chunks of a changed session have embeddings with `SELECT DISTINCT chunk_id FROM vec_chunks WHERE chunk_id IN (...)`. It runs for every changed session with current chunks while the model is present, which is the path of a session-end hook
- `reconcile_chunks` deleted the embeddings of the chunks a changed log no longer has with the same kind of `IN` filter
- The embedder's save deleted the old embeddings of a batch with `chunk_id IN (...)` before inserting new ones

DR 0019 left this cost in place and recorded it as a Bad consequence. On a synthetic index of 6,000 sessions with 29,322 embeddings of 768 dimensions, deleting the embeddings of one session took a median of 35.2 ms (33.4 to 39.1 ms over 20 sessions). Deleting the same session's messages by `session_id` took 26.9 ms, and by DR 0019's range 0.29 ms. A rebuild of 6,000 sessions therefore spends about 3.5 minutes on this delete alone, apart from inference. How should recall find the embeddings of one chunk without reading the whole table?

## Decision Drivers

- `recall rebuild` and the session-end hook both run the full read once per session, and the cost grows with the number of embeddings
- DR 0007: a schema change is a column add found by its shape, with an upgrade test for an old index
- DR 0004: no run deletes a row that its scan did not prove gone
- DR 0019 already deletes messages by a stored rowid range, and the same shape keeps the two delete paths alike
- A stale or wrong record must never delete the embedding of another chunk

## Considered Options

- Store each chunk's embedding rowid range in its `qa_chunks` row, and delete one rowid at a time inside it
- Derive the rowid from the chunk id (`chunk_id × K + sub_idx`)
- Filter on `chunk_id` with a newer sqlite-vec feature (metadata columns or a partition key)
- Leave the statements as they are

## Decision Outcome

Chosen option: "Store each chunk's embedding rowid range", because it is the only option that stops the full read on all four paths with the pinned sqlite-vec 0.1.9, keeps every stored embedding, and lets DR 0007 detect the upgrade by the shape of `qa_chunks`.

`qa_chunks` gains `vec_rowid_lo` and `vec_rowid_hi`. A chunk with embeddings holds the smallest and the largest rowid of its embeddings; a chunk without embeddings holds NULL in both.

- Every delete of a `vec_chunks` row goes through one statement, `DELETE FROM vec_chunks WHERE rowid = ?1 AND chunk_id = ?2` (`DELETE_CHUNK_VECTOR` in src/db.rs). The caller passes each rowid of the stored range with the chunk's id. The `chunk_id` test keeps the embedding of any other chunk when a record is stale
- The embedder's save, inside its IMMEDIATE transaction and after the generation check, deletes the stored range of each chunk it saves, inserts the new embeddings, and records the first and the last `last_insert_rowid()`. When the rowids of one chunk are not consecutive (last − first + 1 differs from the number inserted), the save returns an error. The transaction rolls back, and the batch counts in `chunks_save_failed` and `first_save_error` like any other save failure
- `delete_session_dependents` deletes the stored range of each chunk of the session before it deletes the chunks
- `reconcile_chunks` decides which chunks have embeddings from `vec_rowid_lo IS NOT NULL` without reading `vec_chunks`, and deletes the stored range of each chunk it removes
- A chunk whose record is NULL while `vec_chunks` holds rows for it keeps those rows when it is deleted. After the chunk row is gone, `recall doctor` counts them as orphan embeddings and `--fix` deletes them. Neither the upgrade nor the paths above leave such a chunk

`migrate_vector_rowid_range_if_needed` (src/db.rs) prints a one-line notice and adds the two columns inside an IMMEDIATE transaction. In the same transaction it reads `vec_chunks` once (`rowid` and `chunk_id`) and records each chunk's smallest and largest rowid. A chunk whose embeddings do not fill that range (their count differs from max − min + 1) loses them through `DELETE_CHUNK_VECTOR`, keeps NULL, and waits for inference again. The upgrade does not touch an embedding whose chunk row is gone; `recall doctor` counts it as before. When `migrate_vec_chunks_if_needed` drops a `vec_chunks` table without `sub_idx`, it sets both columns of every chunk to NULL in the same transaction, if the columns exist.

### Consequences

- Good, because a deleted or changed session no longer reads every embedding. The plan of `DELETE_CHUNK_VECTOR` is `INDEX 5:2!___`, not a full read (`INDEX 0:`), and the reconciliation reads only `qa_chunks`. The time of that statement was not measured
- Good, because a stale record cannot delete another chunk's embedding: the delete statement also tests `chunk_id`
- Bad, because correctness rests on invariants that no constraint enforces: every path that inserts embeddings records the range, and every path that deletes embeddings goes through the stored range. A new path that writes `vec_chunks` without the record leaves embeddings that later deletes miss, and no query fails
- Bad, because it rests on how sqlite-vec 0.1.9 assigns rowids (see More Information). The save and the upgrade check the property and fall back to a save failure or to inference, but a sqlite-vec upgrade that breaks it turns the affected saves into failures
- Bad, because an older recall that writes the index after the upgrade does not maintain the record. Its save deletes by `chunk_id` and inserts new rows without updating the range, and its new chunks keep NULL. When this recall later deletes such a chunk, the stale or missing range deletes nothing, and the embeddings remain as orphans until `recall doctor --fix`. `qa_chunks.id` has no AUTOINCREMENT, so a new chunk can take the id of a deleted one before that. The left embeddings then belong to the new chunk: it does not count as pending, and `recall doctor` no longer counts them as orphans
- Bad, because the upgrade reads the whole `vec_chunks` table once while it holds the write lock, and the gapped chunks it finds need inference again
- Neutral, because the effect on deletion and reconciliation is measured after the PR is published, not in this change (Issue #475)

### Confirmation

- `deleting_one_chunk_vector_reads_only_the_row_with_that_rowid` (src/db/tests.rs) takes the plan of `DELETE_CHUNK_VECTOR` itself and fails when it contains a full read (`INDEX 0:`)
- `upgrading_an_index_without_vector_ranges_stores_each_range_and_returns_gapped_chunks_to_inference` (src/db/tests.rs) checks the recorded ranges, the deleted embeddings of a gapped chunk, the kept orphan embedding, the pending count, and that a second open changes nothing
- `rebuilding_the_vector_table_without_a_sub_index_clears_every_stored_vector_range` (src/db/tests.rs) fails when the rebuilt table leaves a record behind
- `deleting_a_session_removes_every_vector_of_its_chunks_and_keeps_the_other_sessions_vectors` and `deleting_a_session_whose_chunk_records_another_chunks_vector_range_keeps_that_vector` (src/indexer/tests.rs) fail when the delete misses one rowid of a range or drops its `chunk_id` test
- `saving_a_changed_chunk_again_replaces_all_its_old_vectors_and_records_the_new_rowids` (src/embedder/tests.rs) fails when the save keeps an old embedding or records another range
- `an_opened_database_runs_the_pinned_sqlite_vec` (src/db/tests.rs) fails when the sqlite-vec version changes, which is the trigger to check the rowid assumptions again

The save's failure on non-consecutive rowids has no test, because sqlite-vec 0.1.9 never assigns such rowids inside one IMMEDIATE transaction. Code review checks that a new write to `vec_chunks` records the range and that a new delete uses `DELETE_CHUNK_VECTOR`.

## Pros and Cons of the Options

### Store each chunk's embedding rowid range

Keep the rowid range of a chunk's embeddings in its `qa_chunks` row and delete one rowid at a time.

- Good, because it serves rebuild, the incremental run, the hook and the save alike, and keeps every stored embedding through the upgrade except those of gapped chunks
- Bad, because it adds two columns, an upgrade and invariants that only review guards

### Derive the rowid from the chunk id

Insert each embedding at `chunk_id × K + sub_idx`, so the rowids of a chunk follow from its id.

- Good, because no record is needed
- Bad, because it needs an upper bound K on `sub_idx`, and every stored embedding has to be inserted again under a new rowid
- Bad, because the schema does not change, so DR 0007 cannot tell from the shape whether the upgrade ran

### Filter on `chunk_id` with a newer sqlite-vec feature

Declare `chunk_id` as a metadata column or a partition key so that vec0 filters on it.

- Good, because the table itself would answer a lookup by chunk
- Bad, because whether a newer version filters such a column without a full read was not checked, and upgrading sqlite-vec is separate work

### Leave the statements as they are

- Good, because nothing changes
- Bad, because rebuild and the hook keep about 35 ms per session on the synthetic index, and the cost grows with the number of embeddings

## More Information

This record does not make an exception to DR 0004. The upgrade deletes only the embeddings of a chunk whose embeddings do not fill their range, never a session or chunk row, and that chunk is embedded again by the next `recall index`.

For DR 0007, the upgrade finds the old shape by the substring `vec_rowid_lo`. No other column of `qa_chunks` contains it: `src_rowid_lo` and `sessions.message_rowid_lo` do not.

The requirements and the agreement are in Issue #475 (agreed with thkt on 2026-11-22). An experiment ran on main d37ff86 on 2026-11-21 with a release build and sqlite-vec 0.1.9 (pinned as `=0.1.9` in Cargo.toml), on a synthetic index of 29,322 vec0 rows. No test was kept from it.

The experiment observed the following behavior of vec0 rowids:

- vec0 narrows the rows only for a rowid equality, bound or literal. `IN`, `BETWEEN` and range conditions read every row, which is why the delete runs one rowid at a time instead of a `BETWEEN` like DR 0019
- `rowid = ?1 AND chunk_id = ?2` deleted no row when `chunk_id` differed, and one row when it matched
- Rows inserted one after another received consecutive rowids (1 to 5), and `last_insert_rowid()` returned them
- A deleted rowid was not reused: after deleting rowid 5 the next row got 6, and after deleting rowid 1 the next got 7. After an explicit rowid 1000000, the next automatic rowid was 1000001
- Inserting an existing rowid explicitly failed with "UNIQUE constraint failed on vec_chunks primary key"
- Deleting a missing rowid changed no row

The plans and medians of the measured statements:

| Statement | Plan | Median |
| --- | --- | --- |
| `SELECT ... WHERE chunk_id IN (a, b)` | `INDEX 0:1` (full read) | 36.9 ms |
| `SELECT ... WHERE rowid = ?` | `INDEX 1:2!___` (one row) | 0.38 ms |
| `SELECT ... WHERE rowid IN (a, b)` | `INDEX 0:1` (full read) | 4.4 ms |
| `SELECT ... WHERE rowid BETWEEN a AND b` | `INDEX 0:1` (full read) | 4.8 ms |
| `SELECT ... WHERE rowid >= a AND rowid < b` | `INDEX 0:1` (full read) | 4.1 ms |
| `DELETE ... WHERE rowid = ?` | `INDEX 1:2!___` (one row) | 0.72 ms |
| `DELETE ... WHERE chunk_id IN (a, b)` | `INDEX 0:1` (full read) | 42.6 ms |
| `DELETE ... WHERE rowid = ?1 AND chunk_id = ?2` | `INDEX 5:2!___` | not measured |

### Trade-offs

Rowids derived from the chunk id would have needed no record, but would have rewritten every embedding and left the upgrade invisible to DR 0007. The stored range costs two columns, one read of `vec_chunks` in the upgrade and review-guarded invariants. In return, no path that deletes or reconciles a session's embeddings reads the whole table.

### Reassessment Triggers

- sqlite-vec is upgraded from 0.1.9. Check again that rows inserted in one transaction get consecutive rowids, that deleted rowids are not reused, that `last_insert_rowid()` returns the vec0 rowid, and that `rowid = ?1 AND chunk_id = ?2` deletes no row of another chunk. Also check that the plan of `DELETE_CHUNK_VECTOR` is not a full read (`INDEX 0:`), and whether the new version filters an auxiliary, metadata or partition-key column without a full read
- A second writer starts to insert embeddings, for example an importer; it must record the range
- Saves start to fail with non-consecutive rowids, which means the rowid assumption no longer holds
