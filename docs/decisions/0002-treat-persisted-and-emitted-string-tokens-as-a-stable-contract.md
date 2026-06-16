---
status: "accepted"
date: 2026-06-15
decision-makers: thkt
---

# Treat Persisted and Emitted String Tokens as a Stable Contract

## Context and Problem Statement

recall stores and emits hand-written string tokens that double as contracts. `Source` and `Role` round-trip through the DB via `as_str` / `from_db` (parser/mod.rs:22-65), `session_type` is persisted as `"interactive"` / `"automated"` with NULL coalesced to interactive in the search filter (db.rs:143, search.rs:140), and skip-reason tokens `"missing_root"` / `"incomplete_enumeration"` flow into the `--json` envelope (indexer.rs:88, main.rs:360). These strings are matched, not type-checked across the boundary, so a rename silently breaks existing DB rows or agent consumers. No document records that they are frozen.

## Decision Drivers

- `from_db` matches stored strings exactly; a rename strands every pre-existing row
- Skip-reason tokens reach `--json` consumers as machine values
- The compiler cannot enforce string-equality contracts across the persistence boundary

## Considered Options

- Treat the token strings as a frozen contract (document + central definition)
- Allow free renaming and migrate the DB on each change
- Replace strings with integer enums in the schema

## Decision Outcome

Chosen option: "Treat the token strings as a frozen contract", because the values already live in user DBs and agent output, and a migration-on-rename policy adds cost without removing the agent-facing break.

### Consequences

- Good, because re-indexing is unnecessary across versions; old rows keep round-tripping
- Good, because agents can pattern-match skip reasons and session types reliably
- Bad, because renaming a token now requires a migration plan plus a consumer notice, not a one-line edit

### Confirmation

A round-trip test asserts `from_db(x.as_str()) == Some(x)` for every `Source` / `Role` variant. A test asserts the NULL-`session_type` row is treated as interactive by the search filter. A test asserts the skip-reason `--json` tokens match the documented set. Renames are caught in review against this ADR.

## More Information

### Quality Attributes

The contract spans two surfaces: persistence (`from_db`, `session_type`) where a break corrupts reads of existing rows, and emission (skip-reason in `--json`) where a break misleads agents. Both are tool-unenforceable string matches.

### Reassessment Triggers

- A token must change for a domain reason (then write a migration + version note)
- The schema moves to integer-coded enums, retiring the string contract
