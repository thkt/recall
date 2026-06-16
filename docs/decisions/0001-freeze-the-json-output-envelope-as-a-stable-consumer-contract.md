---
status: "accepted"
date: 2026-06-15
decision-makers: thkt
---

# Freeze the --json Output Envelope as a Stable Consumer Contract

## Context and Problem Statement

Under `--json`, recall emits `SuccessEnvelope` (`data` / `degraded` / `notes`) to stdout and `ErrorEnvelope` (`error.code` / `message` / `next_step` / `candidates` / `retryable`) to stderr (envelope.rs). Agents parse these keys. The code decouples the wire schema from internal types via a `serde_json::json!` seam and keeps `SearchResult` / `SessionData` non-`Serialize`, but no document states that the envelope keys are a frozen public contract, nor that every string field reaching stdout must be control-char sanitized (main.rs:800-821). A future contributor could break agent consumers by renaming a key or surfacing a raw field.

## Decision Drivers

- Agents consume `--json` programmatically; key renames are breaking changes
- Untrusted session content flows into output and must not inject control sequences

## Considered Options

- Freeze the envelope shape as a documented contract
- Let the envelope co-evolve freely with internal types
- Derive `Serialize` on internal types and emit them directly

## Decision Outcome

Chosen option: "Freeze the envelope shape", because the envelope is the only stable boundary agents can depend on, and the existing non-`Serialize` + `json!` seam already enforces the decoupling that makes a freeze maintainable.

### Consequences

- Good, because agent consumers get a versionable, stable schema independent of internal refactors
- Good, because the sanitize rule is stated once instead of re-derived per field
- Bad, because adding or renaming an envelope key now requires a conscious contract change, not a silent edit

### Confirmation

A golden/snapshot test asserts the `--json` key set for `search` / `index` / `status` (extend envelope.rs tests). A grep check confirms `SearchResult` and `SessionData` carry no `derive(Serialize)`. Code review verifies every new string field in `format_result` and the `json!` payloads passes `ansi::strip_control_chars`.

## More Information

### Trade-offs

ADR-0060 governs the shared envelope pattern across the tool family; this ADR records the recall-specific freeze + strip-every-field commitment that ADR-0060 does not cover.

### Reassessment Triggers

- A v2 JSON schema is introduced (then version the envelope explicitly)
- A consumer needs a field that only exists on an internal type, forcing a seam redesign
