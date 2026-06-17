---
status: "accepted"
date: 2026-06-17
decision-makers: thkt
---

# Forbid Blanket Deletion in the Index Rebuild Path

## Context and Problem Statement

`rebuild` (and any future reset/reset-index path) must re-derive the index from the JSONL session roots. The obvious implementation is "blanket `DELETE FROM sessions` then re-scan and re-insert". recall instead loads `existing` even under force and routes every deletion through the source-keyed `cleanup_orphans` (indexer.rs:404-410, 517-548). The reason is not visible from any single call site: the safety rests on a `fully_read` bool that `collect_jsonl_files` propagates from `collect_from_entries` (indexer.rs:557-621, computed at 600-621), which `classify_root` turns into membership in `scanned` (indexer.rs:351-359), which `cleanup_orphans` then requires before it will delete a row. A blanket DELETE bypasses that proof, and the failure mode is silent and unrecoverable: if the Claude or Codex root is missing or unreadable for one run (an unmounted volume, a transient permission error, a depth-truncated tree), a DELETE-then-rescan wipes every session under that root before the scan can prove the files were actually deleted (#177). The index is the only place those parsed sessions live, so the loss does not reverse on the next run.

## Decision Drivers

- A rebuild that destroys the index on a transient root outage is a data-loss bug, and the index is the only durable copy of the parsed sessions
- The safety invariant spans four functions (`index_from_dirs`, `cleanup_orphans`, `collect_jsonl_files`, `classify_root`) and no single file shows why a blanket DELETE is forbidden
- No current test exercises a future reset/export-reset delete path, so the constraint binding those paths cannot be compiler- or test-enforced today
- "Enumeration succeeded" and "the file is absent" are different claims; only the first licenses deletion

## Considered Options

- Forbid blanket deletion: route every removal through `cleanup_orphans`, gated on a fully-scanned root
- Blanket `DELETE FROM sessions` then re-scan and re-insert on every rebuild
- Delete eagerly but restore from a pre-rebuild backup if the scan comes back short

## Decision Outcome

Chosen option: "Forbid blanket deletion", because it is the only option that never destroys a row the current run could not prove deleted, and it keeps `rebuild` a superset of `index` (re-index every present file) rather than a destructive reset.

### Consequences

- Good, because a missing or unreadable root preserves its existing rows instead of wiping them: the source never enters `scanned`, so `cleanup_orphans` skips it (indexer.rs:528-536)
- Good, because any enumeration uncertainty (a `read_dir` error, a per-entry `file_type` failure, or a `MAX_DIR_DEPTH` truncation) flips `fully_read` to false, so the conservative branch is the default rather than an explicit special case (#181, indexer.rs:567-621)
- Good, because a `None`-typed source row (an unrecognized DB value) is also preserved rather than deleted
- Bad, because a genuinely deleted session under a root that fails to enumerate lingers as a stale row until a clean run of that root confirms its absence
- Bad, because the invariant is a code-review constraint on new delete paths, not a type the compiler enforces

### Confirmation

`cleanup_orphans` deletes a row only when `entry.source` is in `scanned` (indexer.rs:533-534), and `scanned` admits a source only when `collect_jsonl_files` returned `fully_read == true`. Tests inject an `Err` directory entry and a depth-limit truncation to assert the source is preserved (the per-entry path that a real APFS `file_type()` never triggers, indexer.rs:587-592). A new reset/export-reset delete path must reuse `cleanup_orphans` or carry an equivalent scanned-root gate; code review verifies this against this ADR, since no test reaches a path that does not exist yet.

## More Information

### Reassessment Triggers

- A reset or reset-index command is added that must clear rows a scan cannot reach (then design its delete gate against this ADR before merging)
- The source enumeration gains a reliable "this root is genuinely empty" signal distinct from "this root failed to enumerate", which would let cleanup delete confirmed-empty roots
