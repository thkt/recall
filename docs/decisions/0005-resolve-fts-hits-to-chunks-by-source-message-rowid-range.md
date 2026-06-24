---
status: "accepted"
date: 2026-06-17
decision-makers: thkt
---

# Resolve FTS Hits to Chunks by Source-Message Rowid Range

## Context and Problem Statement

A search shows the full QA chunk an FTS hit came from, but the FTS index (`messages`) and the chunk store (`qa_chunks`) are separate tables. The pre-#192 link was an `instr` substring guess: find the chunk whose text contains the matched message. A short hit such as "yes" collided with the wrong chunk. #192 replaced this with a stored source-message rowid range: the chunker function `chunk_messages` (chunker.rs:91, range recorded at chunker.rs:113-137) records a `src_rowid_lo` / `src_rowid_hi` range on each `qa_chunks` row, and search resolves an FTS hit by `m.rowid BETWEEN c.src_rowid_lo AND c.src_rowid_hi` (search.rs:374-385). This JOIN is correct only under four invariants that span the chunker, the schema, the FTS writer, and the backfill, none of which the schema enforces and none visible from a single file. `src_rowid_lo` / `src_rowid_hi` are nullable `INTEGER` with no CHECK constraint (db.rs:71-72), so a future contributor editing any one of the four sites can silently break the attribution without a compile or test failure.

## Decision Drivers

- The `BETWEEN` JOIN is silently wrong, not loud, if any of its underlying invariants breaks: it returns a plausible but incorrect chunk
- The four invariants live in four files (chunker, db, indexer FTS writer, indexer backfill); no single file documents the contract they jointly uphold
- No test reaches the chunker-to-backfill coupling, where a count-preserving re-grouping would mis-align ranges
- The legacy NULL rows and the current linked rows must coexist, so the tri-state meaning of the columns is itself part of the contract

## Considered Options

- Document the four invariants in one ADR and keep the nullable columns
- Add a foreign-key / trigger layer tying chunks to message rowids in the schema
- Revert to the `instr` substring guess and accept short-hit collisions

## Decision Outcome

Chosen option: "Document the four invariants in one ADR", because the invariants are cross-table and partly behavioral (insertion order, single-pass determinism) and cannot be expressed as a single schema constraint, while the `instr` fallback is retained only for legacy NULL rows.

### Consequences

The JOIN's correctness rests on these four invariants. Each is unenforced by the schema and must be preserved by code review against this ADR:

1. Per-group ranges are disjoint. The chunker assigns each chunk a `[lo, hi]` covering one user message plus its adjacent assistant reply (chunker.rs:113-137), and groups do not overlap, so a message rowid falls in at most one chunk's range. Split-siblings of one message deliberately share a single range, which `HAVING COUNT(*) = 1` collapses to the snippet rather than picking an arbitrary sibling (#118, search.rs:378-379).
2. `src_rowid_lo` and `src_rowid_hi` are co-NULL: either both NULL (a legacy or count-mismatched chunk that resolves via the `instr` fallback) or both non-NULL (a linked chunk), and a non-NULL `lo` implies a valid `hi >= lo`. The search JOIN scopes the rowid path to `src_rowid_lo IS NOT NULL` (search.rs:380-385) on this assumption. This is the one invariant expressible in the schema, and is a candidate for a CHECK constraint `((src_rowid_lo IS NULL) = (src_rowid_hi IS NULL))` on a future migration.
3. `messages.rowid` is monotonic in conversation order. `read_session_messages` reads `ORDER BY rowid` (indexer.rs:785-806, query at 787) and the chunker pairs a user message with the immediately following assistant reply (chunker.rs:113-137), so the FTS rowid the JOIN matches is the same rowid the chunker ranged over. A writer that inserts messages out of conversation order would break the range-to-conversation correspondence.
4. Backfill trusts id-order == insertion-order. `backfill_rowid_ranges` re-runs the chunker and zips re-derived ranges onto stored chunks by `qa_chunks.id` ascending (indexer.rs:824). It updates a session only when the re-derived chunks match the stored ones in both count and content, position by position (the content check added by #229 / commit 9d9b96e, indexer.rs:812-814, 867-878). Content equality is the load-bearing guard: a count-preserving re-grouping (e.g. #229 merging consecutive assistants into one chunk) keeps the count but rewrites the content, so without the content check the widened range would be stamped onto stale content and misroute an FTS hit on the newly covered text. A drifted session is therefore skipped (rows stay NULL and resolve via the `instr` fallback) rather than mis-aligned, and `rebuild` re-chunks and re-embeds it.

- Good, because a short FTS hit lands on its true owning chunk instead of an `instr` substring collision
- Good, because legacy NULL rows keep working through the `instr` fallback without a forced re-embed
- Bad, because the four invariants are a documented contract, not a tool-enforced one, so a future edit to any of the four sites can silently desync the attribution

### Confirmation

The `instr` fallback is scoped to `src_rowid_lo IS NULL`, so in a fully-linked DB it matches no chunk (search.rs:398-399), making the rowid path the sole resolver and exercising invariants 1-3 on every linked hit. The backfill count-and-content check (invariant 4) skips and logs a session on either a count or a content mismatch rather than writing a mis-aligned range (indexer.rs:857-878). A schema CHECK enforcing invariant 2 is the recommended next migration; until then, code review verifies any edit to the chunker range assignment, the FTS write order, or the backfill zip against this ADR.

## More Information

### Reassessment Triggers

- A count-preserving chunker re-grouping is introduced (then the backfill count check no longer guards range alignment, and the zip needs a stronger key than id-order)
- A schema migration adds the co-NULL CHECK from invariant 2, moving that invariant from documented to enforced
- The FTS message writer changes to insert messages in any order other than conversation order
