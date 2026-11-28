---
status: "accepted"
date: "2026-11-28"
decision-makers: "thkt"
scope: "How chunk embeddings are stored in and queried from the vec_chunks vec0 table (src/db.rs, src/embedder.rs, src/search.rs), and the one-time upgrade of an index that stores them as float32"
---

# Store Chunk Embeddings as int8 at a Scale of 300

## Context and Problem Statement

After the index files drop out of the page cache, a hybrid `recall search` waits on three steps in series: model preparation (about 430 ms), the query embedding (about 45 ms) and the vector search (about 470 ms). The vector search reads all of `vec_chunks`, which held 160,203 rows of 768 float32 components (498 MB) on 2026-11-28. #535 moved model preparation alongside the full-text search and named vector quantization as the next step. How should recall store embeddings so that the vector search reads less from disk, without changing the rows that DR 0020's rowid ranges point to?

## Decision Drivers

- The vector search reads the whole table, so its cold-cache time follows the table size
  - Verified. A Python prototype with sqlite-vec 0.1.9 on a copy of the 160,203 rows (2026-11-28, load average 12 to 20, median of 8 queries): float32 took 472 ms cold and 110 ms cached at 498 MB; int8 took 188 ms cold and 41 ms cached at 127 MB
- The top results must stay close to the float32 ones
  - Verified. Same prototype, overlap with the float32 top 20 using stored vectors as queries: sqlite-vec's `'unit'` quantization gave a median of 0.95 and a minimum of 0.65; a scale of 300 gave 0.975 and 0.8
- Every production embedding fits the scale
  - Verified. The embedder normalizes each embedding to length 1 (src/native.rs). The largest absolute component among the 160,203 rows was 0.3877, below 127/300 (about 0.423)
- DR 0020: the upgrade keeps each row's `rowid`, `chunk_id` and `sub_idx`, so `qa_chunks.vec_rowid_lo`/`vec_rowid_hi` keep pointing at the same rows
  - By policy. DR 0020
- DR 0007: a schema change is found by its shape, without a version table
  - By policy. DR 0007

## Considered Options

- Store int8 at a scale of 300, and upgrade an index once when it is opened for writing
- Store int8 with sqlite-vec's `vec_quantize_int8(..., 'unit')`
- Fetch extra candidates with int8 and re-rank them with float32
- Truncate the embeddings to 256 dimensions (Matryoshka)
- Keep float32

## Decision Outcome

Chosen option: "Store int8 at a scale of 300", because it cuts the table to about a quarter of its size, keeps the highest minimum overlap of the int8 options, and keeps every row's identity.

- One function, `quantize_int8` (src/embedder.rs), converts a float32 embedding. It multiplies each component by 300, rounds with `f32::round` (half away from zero), clamps to -127..127 and casts with `as i8`. A NaN becomes 0. A component beyond 127/300 is clamped, not rejected. The production write, the query, the upgrade and the test seeds all use it
- `vec_chunks` declares `embedding INT8[768]`, and every int8 value is bound through `vec_int8(?)`
- `open_db` runs the upgrade from `create_schema` as the last step. It decides inside a `BEGIN IMMEDIATE` transaction, by reading the `vec_chunks` definition, like `migrate_vector_rowid_range_if_needed`. Deciding before the lock would let a second writer rebuild a table that the first writer has just converted. The upgrade prints one `info` line ending with "this runs once" and then:
  1. Reads every float32 row into memory as int8, with its `rowid`, `chunk_id` and `sub_idx` (about 127 MB for the 160,203 rows)
  2. Drops `vec_chunks`
  3. Creates `vec_chunks` with `INT8[768]`
  4. Inserts each row at its original `rowid` through `vec_int8(?)`, one statement per row
  5. Commits, then runs `PRAGMA wal_checkpoint(TRUNCATE)`
- The upgrade runs only from `open_db`, which `recall index` and `recall classify` without `--dry-run` use. `recall search`, `recall status`, `recall doctor` and `recall classify --dry-run` open the index read-only, and `recall doctor --fix` uses `open_existing_db`, so none of them upgrades
- A search reads the `vec_chunks` definition in the same read transaction as its data probe and its vector query. It queries a float32 table with float32 and an int8 table with `vec_int8(?)`, so an upgrade that commits during a search cannot split one search between the two shapes

### Consequences

- Good, because the table shrinks from 498 MB to 127 MB. That ratio holds whatever the load; the time figures above were taken under load averages of 12 to 41
- Good, because a read-only command keeps working on an index that has not been upgraded yet
- Bad, because distances change, and the order can change where rounding moves a row. Integer distances tie more often. Tied sessions take the mean of their ranks (DR 0018), and the hybrid result uses ranks, not distances
- Bad, because the upgrade holds the write lock for its whole run. The Python prototype took 46 to 48 seconds on a copy of the real index; the Rust upgrade was not timed. Every `open_db` takes `BEGIN IMMEDIATE` unconditionally, so a writer that opens during the upgrade waits, and fails with "database is locked" after the 30-second `busy_timeout`. The next `recall index` picks up what that writer missed
- Bad, because recall 0.17.2 and earlier cannot use an upgraded index fully:
  - Search fails the vector query on the type mismatch and returns the full-text results with `degraded` true. Verified on 2026-11-28 by upgrading a copy of the real index and searching it with v0.17.2
  - Writing an embedding fails, because sqlite-vec refuses a float32 vector for an int8 column. Verified with sqlite-vec 0.1.9 by writing float32 into an upgraded copy
  - `recall index` infers every pending chunk on each run, then fails at the save and rolls the batch back (src/embedder.rs), so the GPU time of that inference is wasted. This follows from reading the code and was not run
- Bad, because the pages freed by the upgrade stay in the file. The file does not shrink until a `VACUUM`, which this decision does not run
- Bad, because an older recall that opens an upgraded index is not stopped. This decision adds no `user_version` or other guard

### Confirmation

- `the_bundled_sqlite_stores_int8_vectors_at_a_given_rowid_and_finds_them_by_nearest_distance` (src/db/tests.rs) checks, with the bundled SQLite of `rusqlite` 0.43, an explicit-rowid insert through `vec_int8(?)` and a KNN query of the same shape as `vec_search`
- `a_new_index_stores_embeddings_as_768_int8_components` (src/db/tests.rs) checks the definition of a new index
- `opening_an_index_with_float32_embeddings_for_writing_stores_each_one_as_int8_in_place` and `reopening_a_migrated_index_for_writing_runs_no_ddl_and_keeps_every_embedding_byte` (src/db/tests.rs) check the upgrade and that it runs once
- `the_recorded_rowid_range_of_a_chunk_deletes_its_migrated_embeddings` (src/db/tests.rs) checks that DR 0020's ranges still delete the upgraded rows
- `int8_levels_round_half_away_from_zero_and_saturate_beyond_127_over_300` and `indexing_stores_each_embedding_component_times_300_rounded_into_int8` (src/embedder/tests.rs) check the conversion and the write path
- `hybrid_search_on_an_int8_index_finds_a_vector_only_match_without_degrading` and `hybrid_search_on_a_read_only_index_with_float32_embeddings_finds_a_vector_only_match` (src/search/tests.rs) check both shapes of search

## Pros and Cons of the Options

### Store int8 at a scale of 300

- Good, because the minimum top-20 overlap was 0.8, the highest of the int8 options measured
- Bad, because a component beyond 127/300 is clamped, which a future model with less even components could hit

### Store int8 with sqlite-vec's `'unit'` quantization

- Good, because sqlite-vec provides it, and it took 188 ms cold in the prototype
- Bad, because its minimum top-20 overlap was 0.65
  - Verified. Python prototype with sqlite-vec 0.1.9 on 2026-11-28, as in Decision Drivers

### Fetch extra candidates with int8 and re-rank them with float32

- Good, because the final order would follow float32
- Bad, because the cold-cache time did not improve: re-ranking 200 candidates reads float32 rows at scattered places, and took 494 ms against 528 ms for float32 alone
  - Verified. Python prototype with sqlite-vec 0.1.9 on 2026-11-28

### Truncate the embeddings to 256 dimensions

- Good, because the time was about the same as int8
- Bad, because the median top-20 overlap fell to 0.8
  - Verified. Python prototype with sqlite-vec 0.1.9 on 2026-11-28

### Keep float32

- Good, because nothing changes
- Bad, because the vector search keeps reading 498 MB after the cache is dropped
  - Verified. Same prototype, 472 ms cold

## More Information

Requirements and agreement: Issue #538, agreed with thkt on 2026-11-28. thkt accepted the behavior of v0.17.2 and earlier on an upgraded index, the write lock held during the upgrade, and the freed pages left in the file.

The upgrade copies every row and deletes none, so it makes no exception to DR 0004. DR 0007 already accepts that an index has no path back to an older release.

For DR 0007, the shape is found by the substring `embedding INT8[` in the `vec_chunks` definition (`INT8_VECTOR_COLUMN` in src/db.rs). The same constant builds the definition, so the check and the DDL cannot drift apart.

Observed behavior of sqlite-vec 0.1.9 that shaped the upgrade (Python prototype, 2026-11-28):

- `ALTER TABLE ... RENAME` on a vec0 table does not rename its shadow tables (`*_rowids`, `*_chunks` and others). After building a new table and renaming it, v0.17.2's `recall status` stopped with `no such table: main.vec_chunks_rowids`. The upgrade therefore drops and recreates `vec_chunks` instead of renaming
- `vec_int8()` and `vec_quantize_int8()` mark their result as int8 through the SQLite value subtype, and the subtype is lost through `ORDER BY`. `INSERT ... SELECT ... ORDER BY` was refused as a float32 vector. The upgrade therefore inserts row by row from Rust

The query side of the cold-cache gain, and whether hybrid search drops below 2 seconds, are not confirmed. The expected saving is 200 to 300 ms, about 10 to 15 percent of the time above 2 seconds; where the rest of that time goes is unknown.

### Migration Strategy

The upgrade runs once on the first `recall index` or `recall classify` after the release. Until then, every read-only command keeps querying the float32 table.

### Rollback Plan

There is none. An upgraded index stays int8. Running an older recall on it degrades search to full text and fails every embedding save, as listed in Consequences.

### Reassessment Triggers

- The embedding model changes, or a stored component beyond 127/300 shows up. Check the maximum component again and choose the scale again
- sqlite-vec is upgraded from 0.1.9. Check again how subtypes pass through, how `RENAME` treats shadow tables, and that `vec_int8(?)` accepts an explicit rowid
- The measurement after release (`scripts/measure-search.py --drop-from-memory`, v0.17.2 against the new version) shows no cold-cache gain, or the top sessions for the same queries change noticeably
