---
status: "proposed"
date: "2026-11-23"
decision-makers: "thkt"
scope: "Which index shapes recall opens (schema_state and open_db in src/db.rs) and which schema migrations create_schema runs. The two migrations added after v0.16.1 (DR 0019, DR 0020), the model identity check (model::check_index) and row-level fallbacks for rows an older recall wrote are out of scope"
---

# Refuse Indexes Built Before v0.14.0

## Context and Problem Statement

`create_schema` in src/db.rs runs twelve shape-sniffing migrations on every writer open. DR 0007 requires them to bring an index "from any prior shape" to the current schema, and DR 0017 added a row-preserving copy for the oldest shape (a non-trigram full-text table, last written by v0.5.0).

Ten of the twelve serve shapes that only a recall older than v0.14.0 wrote.

| Schema change | First release that wrote it |
| --- | --- |
| trigram tokenizer | v0.6.0 |
| `vec_chunks.sub_idx`, drop of `qa_chunks.chunk_hash`, `sessions.session_type` | v0.8.0 |
| `qa_chunks.src_rowid_lo/hi` | v0.9.0 |
| `sessions.files_scanned` | v0.13.0 |
| `parse_diagnostics`, `sessions.file_size`, `qa_chunks.generation`, `sessions.chunks_indexed` | v0.14.0 |

Two facts make those shapes rare.

- v0.14.0 (da5a4c0, 2026-10-16) moved the default index from `<data dir>/recall/recall.db` to `<data dir>/recall/indexes/<model-identity>/recall.db`. Every index at the default path was therefore created by v0.14.0 or later. An older shape is reached only when `--db-path` or `RECALL_DB` names an old file.
- Since the same commit, `model::check_index` refuses to open for writing an index that holds vectors and no model identity. An old index with embeddings can already not be upgraded.

The remaining case is an old file without embeddings that a user points recall at by hand. For that case each writer open takes the write lock in four `BEGIN IMMEDIATE` transactions that find nothing to do (`parse_diagnostics`, `file_size`, `qa_chunk_generation`, full-text tokenizer), and the test suite builds old shapes by hand to keep ten migrations passing.

The project outcome (.claude/OUTCOME.md) asks for search over the sessions on this machine. It names no requirement to carry an index across releases, and the session logs, not the index, are the source the index is built from. Should recall keep upgrading every shape it ever wrote?

## Decision Drivers

- A code path stays only while a state it handles can occur in an index a current user has
- An index is rebuilt from the session logs. A session lives in the index alone only while its source root is missing or unreadable: once a root is read in full, `recall index` removes the sessions whose logs are gone (DR 0004)
- A refused index must stay byte-for-byte as it was, so an older binary can still open it
- An index whose first create was interrupted must still be finished, not refused
- Every command must give the same answer about one index, and the refusal must name the next step

## Considered Options

- Refuse every shape older than v0.14.0 in every command, and delete the ten migrations
- Keep every migration (the current state under DR 0007 and DR 0017)
- Refuse older shapes for writing only, and keep reading them
- Keep the migrations but run them only from an explicit `recall migrate` command

## Decision Outcome

Chosen option: "Refuse every shape older than v0.14.0 in every command, and delete the ten migrations", because no index at the default path can be in such a shape, and the one remaining case (an old file named by hand, holding no embeddings) is served by building a new index from the logs.

`schema_state` separates two states that it reports as `Stale` today.

| State | Condition | Writers | Readers |
| --- | --- | --- | --- |
| Unsupported | A table that exists carries a pre-v0.14.0 marker in its stored definition | Refuse | Refuse |
| Incomplete | No marker, and a table of the current schema is missing | Finish the create | Refuse and name `recall index` |

The markers are exactly these.

- `sessions` lacks `session_type`, `files_scanned`, `file_size` or `chunks_indexed`
- `messages` is not tokenized with trigram
- `qa_chunks` has `chunk_hash`, or lacks `src_rowid_lo` or `generation`
- `vec_chunks` lacks `sub_idx`

A missing table is never a marker. `create_schema` commits `sessions` before it creates the other tables, so a first run that is killed, or two `SessionEnd` hooks that open a new index together, leave `sessions` alone with its current definition. That file is Incomplete, and the next writer finishes it with `CREATE ... IF NOT EXISTS`.

Every pre-v0.14.0 index is Unsupported by its `sessions` table alone, which lacks `file_size` and `chunks_indexed`. The other markers cover an index whose upgrade by an older binary stopped partway.

`open_db` checks `schema_state` before it changes the journal mode or runs any DDL, and returns a data error (exit 65) when the index is Unsupported. `search`, `status`, `show`, `classify` (with and without `--dry-run`), `index` and `rebuild` print one message, and `doctor` reports the same text as the detail and remedy of its `schema` check. The message no longer names `recall rebuild`, because `rebuild` refuses the same index:

> this index's schema predates recall 0.14.0 and cannot be opened; build a new index with `recall index` at another `--db-path`, or upgrade this one by running `recall classify` from recall 0.16.1 on it

`create_schema` creates `parse_diagnostics`, `qa_chunk_generation` and the two generation triggers directly. These ten are deleted: `migrate_parse_diagnostics_if_needed`, `migrate_file_size_if_needed`, `migrate_qa_chunk_generation_if_needed`, `migrate_vec_chunks_if_needed`, `migrate_qa_chunks_if_needed`, `migrate_qa_chunk_rowid_link_if_needed`, `migrate_session_type_if_needed`, `migrate_files_scanned_if_needed`, `migrate_chunks_indexed_if_needed`, `migrate_fts_if_needed`. The statement `DROP TABLE IF EXISTS embedded_chunk_ids` goes with them; no tagged release created that table.

`migrate_message_rowid_range_if_needed` (DR 0019) and `migrate_vector_rowid_range_if_needed` (DR 0020) stay. Both shapes were added after v0.16.1, so every index written by v0.14.0 through v0.16.1 needs them.

DR 0007's method (detect the shape, no version table) stays. Its driver that a migration must be safe on "any older shape" narrows to any shape v0.14.0 or later wrote.

### Consequences

- Good, because a writer open takes the write lock in two migration transactions, down from six, and runs two shape sniffs in place of twelve
- Good, because ten migration functions in src/db.rs and the tests that build pre-v0.14.0 shapes by hand are deleted
- Good, because the refusal happens before any write, so the old file keeps its schema and journal mode and a v0.16.1 binary can still open it
- Good, because `doctor` stops sending the user to `recall rebuild` for an index that `rebuild` cannot open
- Bad, because an old index named by `--db-path` or `RECALL_DB` stops working after the upgrade. Today `search`, `status` and `show` read a v0.9.0 to v0.13.0 shape without migrating it, and `classify --dry-run` previews any shape that has `session_type`. The README sentences that promise both ("An otherwise compatible index without generation tracking remains readable", "Older schemas can be previewed if") are removed
- Bad, because a session under a missing or unreadable root that exists only in such an index is no longer reachable with the current binary. DR 0017 rejected refusing old indexes for this reason. The user can upgrade the file with a v0.16.1 binary (Migration Plan). An old index that holds embeddings cannot be upgraded that way, because v0.14.0 and later already refuse to write to it
- Bad, because a read-only open that cannot see an uncheckpointed upgrade in the write-ahead log reports the old shape. The refusal then carries the existing stale-snapshot note, which names the checkpoint remedy
- Bad, because the refusal names two release numbers, so the message must be edited when the floor moves
- Bad, because the markers are substring sniffs of stored DDL, the weakness DR 0007 records

### Confirmation

- A new test builds a v0.13.0-shaped index, opens it with `open_db`, and asserts the error, the unchanged `sqlite_master` rows and the unchanged journal mode
- A new test asserts that `search`, `status`, `show`, `classify`, `classify --dry-run`, `index` and `rebuild` print the same refusal for that index and exit 65, and that `doctor` reports it in its `schema` check
- A new test opens a file that holds only a current-definition `sessions` table with `open_db` and asserts the full current schema afterwards
- `schema_state` has one test per marker, each applying that marker alone and expecting Unsupported
- The existing tests for the two kept migrations pass on a v0.14.0-shaped index: `upgrading_an_index_without_message_ranges_stores_each_range_and_drops_messages_without_a_session` and `upgrading_an_index_without_vector_ranges_stores_each_range_and_returns_gapped_chunks_to_inference`
- `ugrep -c 'fn migrate_' src/db.rs` prints 2

## Pros and Cons of the Options

### Refuse every shape older than v0.14.0 in every command, and delete the ten migrations

One gate in `schema_state` decides for every command.

- Good, because readers and writers agree on which indexes are usable
- Bad, because an old index without embeddings, which upgrades today, is refused

### Keep every migration

Leave DR 0007 and DR 0017 as they are.

- Good, because any index without embeddings upgrades in place and keeps the sessions of a missing root
- Bad, because ten migrations and their hand-built fixtures are maintained for shapes no default-path index can have

### Refuse older shapes for writing only, and keep reading them

Delete the migrations but leave `schema_state` as it is, so v0.9.0 to v0.13.0 shapes stay readable.

- Good, because an old index stays searchable with the current binary
- Bad, because every read query must keep avoiding columns added since v0.9.0, a rule that only the test `read_commands_use_pre_generation_index_without_migrating_it` enforces
- Bad, because `recall index` on the same file fails while `recall search` works, and the user gets two answers about one index

### Keep the migrations but run them only from an explicit `recall migrate` command

Move the ten functions behind a subcommand.

- Good, because the writer open path gets as short as in the chosen option
- Bad, because all ten functions and their tests stay, plus a new subcommand

## More Information

Supersedes DR 0017 when this record is accepted: `migrate_fts_if_needed`, its contract, is deleted. DR 0017's status is left unchanged while this record is proposed.

Narrows DR 0007 as stated in Decision Outcome. DR 0007 itself is left unchanged.

DR 0005 keeps its `instr` fallback and `backfill_rowid_ranges` for chunks with a NULL `src_rowid_lo`, and `backfill_session_files` keeps serving rows with a NULL `files_scanned`. Such rows can remain in a v0.14.0-shaped index: an index upgraded in place from before v0.9.0 holds them for sessions under a missing root, and the Migration Plan below produces exactly that index. Under the first Decision Driver those fallbacks therefore stay while the v0.16.1 upgrade path is offered.

Release dates and first tags come from `git tag --contains` on e4bdde9, d8e798e, 92c1125, db6e29b, da5a4c0 (all v0.14.0), 3460eb1 (v0.13.0), 1bce16a and cee91c2 (no tag). The ten deleted functions are the complete migration set of `git show v0.16.1:src/db.rs`. In v0.14.0 through v0.16.1, `migrate_chunks_indexed_if_needed` runs last of the ten.

Agreement: thkt, 2026-11-23, in the session that produced PR #504 ("古すぎる情報は切り捨てていい"). The floor at v0.14.0, the two-state split and the refusal wording are proposed here and not yet confirmed.

Challenge (2026-11-23): weakened. It holds while the refusal is decided only from pre-v0.14.0 markers in stored definitions, an interrupted create is still finished, and `doctor` and `classify --dry-run` refuse through the same check as the other commands. The critic-design corrections are in the sections above.

### Deprecation Target

Every index that is Unsupported as defined in Decision Outcome. In practice: files written last by recall v0.1.0 through v0.13.0.

### Migration Plan

1. Build a new index from the session logs: `recall index` at the default path, or with a new `--db-path`
2. To keep the sessions of a missing root, run `recall classify --db-path <file>` once with a v0.16.1 binary, then use the file with the current version. `classify` opens the index for writing, which runs the ten migrations, and it removes no session. `recall index` is the wrong command for this step: it removes every session whose log is gone from a root it reads in full, and it starts an embedding pass. The step works only for an index that holds no embeddings

### Deprecation Warning Period

None. No release warns before the refusal. An index at the default path is unaffected, and a warning would reach only users who already pass an old file by hand.

### Rollback Plan

Revert the commit that deletes the migrations. They are forward-only and idempotent, and the gate changes nothing in a refused file, so an index refused in the meantime upgrades as before.

### Reassessment Triggers

- A user reports a refused index that holds sessions of a missing root and that recall 0.16.1 cannot upgrade
- The default index path changes again, for example when a new pinned model becomes the default: indexes at the previous default path become reachable only by hand, and the floor can move to that release by the same reasoning
- The v0.16.1 upgrade path is withdrawn: no v0.14.0-shaped index can then gain new legacy rows, and the row-level fallbacks of DR 0005 can be judged on the rows that remain
- A structure added after v0.14.0 gets a name that one of the markers matches by substring (DR 0007's third trigger)
