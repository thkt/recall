---
status: "accepted"
date: 2026-06-15
decision-makers: thkt
---

# Enforce Owner-Only 0600 Permission on the Index Database

## Context and Problem Statement

recall indexes Claude Code and Codex session transcripts, which contain private conversation content, into a local SQLite DB. `create_db_file` opens the DB with mode `0o600` on Unix so only the owner can read it (`create_db_file` at main.rs:149, `opts.mode(0o600)` at main.rs:157). The mode is set at the single creation point, but no rule states that this owner-only invariant must hold for every future path that creates or recreates the DB. A later contributor adding a second DB-write path (export, backup, alternate store) could drop the mode and silently widen access to private data.

## Decision Drivers

- Indexed content is private session history; group/world read is a data-exposure risk
- The guarantee currently rests on one call site with no forward-looking rule
- File-mode correctness is not enforceable by the type system or a lint

## Considered Options

- Make owner-only 0600 a cross-path invariant for all DB creation (chosen)
- Rely on the umask of the invoking shell
- Keep the single-site 0600 with no stated invariant

## Decision Outcome

Chosen option: "Make owner-only 0600 a cross-path invariant", because the data is private by nature and the cost of a future path leaking it exceeds the cost of routing all DB creation through one mode-setting helper.

### Consequences

- Good, because every present and future DB-creation path inherits owner-only access by construction
- Good, because the security rationale is recorded where a future contributor will see it
- Bad, because contributors must route new DB-write paths through `create_db_file` rather than opening files ad hoc

### Confirmation

A Unix test asserts the created DB's mode is `0o600` (`metadata().permissions().mode() & 0o777 == 0o600`). Code review verifies any new DB-creating path goes through `create_db_file` or sets the same mode. A grep for `OpenOptions` / `File::create` on the DB path flags ad-hoc creation that bypasses the helper.

## More Information

### Quality Attributes

This is the SSRF-style incomplete-contract case: the current code is correct, but the missing forward-looking rule ("future DB-write paths MUST preserve owner-only") is exactly what an ADR provides and a comment at one call site does not.

### Reassessment Triggers

- A non-Unix (Windows) target needs an equivalent ACL story
- A deliberate multi-user / shared-index feature is introduced
