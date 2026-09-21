---
status: "accepted"
date: 2026-06-24
decision-makers: thkt
---

# Never Skip a Directory by Its mtime in Incremental Indexing

## Context and Problem Statement

recall's incremental index must find every changed and every newly-added session file on each run while still re-parsing as few files as possible. The tempting optimization is a directory-mtime short-circuit: record each source root's last-scan time, and on the next run skip descending into any directory whose mtime has not advanced since then, on the assumption that an unchanged directory holds no new or modified work. recall does not do this. `index_from_dirs` always full-scans the source roots and decides freshness one file at a time in `check_freshness` in [src/indexer.rs](../../src/indexer.rs), keyed on per-file mtime and size against the stored values, with the explicit comment that a directory-mtime skip "used to miss new files added to existing deep dirs (Codex Y/M/D, Claude subagents) — #52 / #70". The reason a dir-mtime skip is unsound is not visible from the current code, which simply has no such branch: on most filesystems a directory's mtime tracks only entry add/remove within that directory, not in-place writes to the files it already contains, and not changes nested below it. A live Claude or Codex session is an append-only JSONL file written across its lifetime; appending a turn changes the file's mtime but leaves the parent directory's mtime untouched, so a dir-mtime gate would mark the directory clean and permanently skip the grown session. The index is the only place those parsed sessions live, so a silently-skipped append is invisible to search until something else forces a re-scan of that file.

## Decision Drivers

- A skipped append is a silent freshness bug: the live tail of an in-progress session never reaches search, and nothing surfaces the omission
- Directory mtime answers "did an entry get added or removed here", not "did any file under here change"; only the second question licenses skipping a scan
- The append case is the steady state, not an edge case: the production hook re-indexes after each Claude/Codex turn (indexer/tests.rs:50-51), and every turn is an in-place append
- This was already tried, shipped, and reverted under two issues (#52, #70); without a recorded decision the same "skip unchanged dirs" optimization reads as a free win to a future contributor

## Considered Options

- Per-file mtime/size freshness check with an unconditional full directory scan (chosen)
- Directory-mtime skip: store a last-scan timestamp and skip directories whose mtime has not advanced (the reverted attempt)
- Content hashing: hash each file's bytes and re-parse only when the hash changes

## Decision Outcome

Chosen option: "per-file mtime/size freshness check with an unconditional full directory scan". Directory traversal always reaches individual files; file metadata keeps parsing incremental without trusting directory mtime to summarize the files beneath it.

[Issue #321](https://github.com/thkt/recall/issues/321) supplements this directory-traversal rule: skipping a body requires a known, equal file size and an mtime difference below 0.001s. Legacy rows without a size remain pending until re-read. A metadata change during parsing leaves the freshness markers unset for a later retry; it must not save a post-read stamp for an earlier body. Embedded sessions still defer updates when embedding is unavailable. Same-size replacements with unchanged mtime (or a difference below the tolerance) remain outside the detection guarantee; no full-file freshness hashing is performed.

### Consequences

- Good, because size changes are eligible for re-parsing even when mtime is preserved or advances by less than the tolerance; embedding availability can still defer the update
- Good, because a new file dropped into an existing deep directory is found regardless of the parent directory's mtime (Codex `Y/M/D`, Claude subagent trees), the exact #52 / #70 failure
- Good, because freshness stays a per-file decision: a file with known, matching size and mtime within 0.001s skips parsing, so an unchanged tree with saved stamps re-parses nothing
- Bad, because every run `stat`s every session file rather than pruning whole subtrees by directory mtime; the cost scales with total file count, not with the changed set
- Bad, because the constraint is a code-review rule on future scan-path changes, not a type or test the compiler enforces against re-adding a dir-mtime gate

### Confirmation

Any new or revised incremental path must key freshness on per-file metadata (mtime and size) rather than directory mtime, verified by code review against this ADR. The constraint is regression-tested: `test_index_from_dirs_reindexes_appended_file` appends a turn, pushes the file mtime past the 0.001s epsilon, and asserts the appended file is re-parsed (`stats.indexed == 1`) without duplicating the session; `test_incremental_scan_picks_up_new_file_in_existing_codex_day_dir` adds a second session into an existing day directory and asserts it is indexed even though the parent mtime is unchanged. A reviewer rejecting a future dir-mtime skip should confirm both tests still bind the skip path. `size_changes_reindex_with_equal_or_submillisecond_mtime` covers append and truncation with preserved or submillisecond mtime. `append_around_parsing_is_retried_and_unchanged_body_is_not_parsed` fixes append order around parsing and checks the subsequent retry and skip. `legacy_size_upgrade_preserves_embeddings_and_retries_until_model_returns` covers migration, repeated opens and deferred updates. These are verification definitions, not a record of a particular host run.

## More Information

The reverting commit is `97f547f` "fix(indexer): scan all files so new sessions in existing dirs are found (#52, #70)", whose body states the directory-mtime skip "missed new files added to existing deep dirs ... appending a file does not change the parent directory mtime, so dirs_changed_since saw no change and skipped the scan, permanently missing the new session", and which removed the `recall_meta` table that had held the last-scan timestamp the skip relied on.

### Reassessment Triggers

- The session roots gain a filesystem-level change feed (an `fsevents`/`inotify` watcher, or a source-provided manifest) that reports per-file changes directly, removing the need to choose between full-scan and a directory-mtime heuristic
- A measured run shows the per-file `stat` sweep dominates index latency on a real corpus, at which point design a pruning signal that is sound for in-place appends (not directory mtime) against this ADR before merging
