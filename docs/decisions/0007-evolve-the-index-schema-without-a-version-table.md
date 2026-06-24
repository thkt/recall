---
status: "accepted"
date: 2026-06-24
decision-makers: thkt
---

# Evolve the Index Schema Without a Version Table

## Context and Problem Statement

The recall index is a single-file SQLite database opened by `open_db` (db.rs:12-24), which always calls `create_schema` (db.rs:26-90). The schema has changed repeatedly (trigram FTS tokenizer, `vec_chunks.sub_idx`, the `qa_chunks` write-only-column cleanup, `src_rowid_lo` / `src_rowid_hi`, `sessions.session_type`, the dropped `embedded_chunk_ids` ledger), so opening an old DB must reshape it to the current schema before any read or write. The standard way to drive such migrations is a numbered ledger: a `schema_version` table or `PRAGMA user_version` integer, bumped per migration, with a switch that runs the next step. recall stores no such number. Instead each migration reads the live shape from `sqlite_master` and conditionally runs DDL: `table_def` fetches a table's stored `CREATE` SQL (db.rs:95-105), and a substring sniff on that SQL decides whether to migrate, for example `!sql.contains("session_type")` (db.rs:176), `!sql.contains("src_rowid_lo")` (db.rs:155), `sql.contains("chunk_hash")` (db.rs:131), `!sql.contains("sub_idx")` (db.rs:112), and `!sql.contains(FTS_TOKENIZER)` (db.rs:190). Fresh-DB creation uses the same self-describing idea through `CREATE TABLE | INDEX | VIRTUAL TABLE IF NOT EXISTS` (db.rs:27-87). The decision to record is this version-table-free, shape-detecting migration model and the invariants it forces on every new migration.

## Decision Drivers

- The index is single-user and local, with no replica, server, or team that needs an agreed version number to coordinate against
- The DB is disposable: it re-derives from JSONL session roots, so a worst-case migration may drop-and-rebuild rather than carefully transform (FTS retokenize at db.rs:196-200)
- A migration must be safe whether it meets a fresh DB, a current DB, or any older shape, and must be a no-op on re-open since `create_schema` runs on every `recall` invocation
- Concurrent `SessionEnd` hooks open the same DB, so migration must hold under WAL with a 30s `busy_timeout` (db.rs:19-21) and not corrupt a half-written shape

## Considered Options

- Detect the current shape directly from `sqlite_master` and conditionally run DDL, with no stored version
- A numbered ledger (`schema_version` table or `PRAGMA user_version`) bumped per migration, dispatching the next step by integer
- A drop-everything-and-rebuild on every open, treating the DB as a pure cache with no migration

## Decision Outcome

Chosen option: "Detect the current shape directly", because a single-user local index has no second party to coordinate a version number with, and the shape sniff is self-healing: it converges to the current schema from any prior shape without a contributor needing to know which numbered step a given DB last reached. Each migration runs in its own transaction (db.rs:114-117, 132-139, 157-161, 177-179, 195-201), so a crash leaves the DB at a clean prior or next shape, and the next open re-detects and resumes.

### Consequences

- Good, because a new column or index is added by stating its target shape once: extend the `CREATE ... IF NOT EXISTS` for fresh DBs and add one `if !sql.contains("newcol")` ALTER for old DBs (the `session_type` and `src_rowid` pairs are the template, db.rs:150-183), with no version number to allocate or keep in sync
- Good, because the model is forward-only and idempotent by construction: re-opening a current DB matches every sniff as already-satisfied and runs no DDL, so the same code path serves fresh creation, upgrade, and steady-state re-open
- Good, because column-add and column-drop migrations preserve existing rows and embeddings without a forced re-index (ALTER ADD/DROP COLUMN at db.rs:135-138, 158-159, 178), while only a genuinely incompatible shape escalates to a drop-and-rebuild that signals the user via `notify_schema_change` (db.rs:113, 194)
- Bad, because there is no explicit downgrade path: a newer binary reshapes an old DB in place, and an older binary then meeting the new shape has no version gate to refuse it
- Bad, because correctness rests on every sniff staying idempotent and mutually consistent; a substring like `contains("sub_idx")` is a heuristic on stored DDL text, and a future column name that contains an older sniff's substring could make two detections disagree about the same shape
- Bad, because no stored version means no cheap audit of which migrations a given DB has passed; the only source of truth is re-running the detection

This model governs schema shape only. It does not relax ADR-0004 (the rebuild path still never blanket-deletes rows) or ADR-0005 (the `src_rowid_lo` / `src_rowid_hi` columns this model adds carry the rowid-range invariants documented there); both are referenced, not absorbed.

### Confirmation

Fresh-DB creation is covered by `test_open_db_creates_schema` and re-open idempotency by `test_open_db_idempotent` (db/tests.rs:6, 36). Old-DB upgrade is covered per migration: `test_session_type_migration_adds_column_preserving_rows` (ALTER ADD preserving rows, db/tests.rs:46), `test_fts_migration_rebuilds_on_schema_change` and `test_fts_migration_cascades_to_embedding_tables` (destructive retokenize, db/tests.rs:115, 137), and `test_qa_chunks_migration_drops_write_only_columns_preserving_embeddings`, which also asserts re-open stays migrated (db/tests.rs:226, 279-293). No test exercises the `src_rowid_lo` / `src_rowid_hi` add-column migration (db.rs:150-164); that path is verified by review only. A new migration must add both the fresh-create DDL and an old-DB sniff, and must add an old-DB upgrade test asserting rows survive and re-open is a no-op; code review verifies the sniff substring cannot collide with an existing column name.

## More Information

### Reassessment Triggers

- The index gains a second writer or store that another binary version must coordinate against (then a `user_version` gate to refuse forward-incompatible DBs becomes worth its cost)
- A migration needs an explicit downgrade or rollback path (the shape sniff offers none)
- A new column or table name would make one migration's substring sniff match another shape, breaking detection independence (then move that sniff from substring to a `PRAGMA table_info` column-name or index check)
