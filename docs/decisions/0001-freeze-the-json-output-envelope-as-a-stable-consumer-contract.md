---
status: "accepted"
date: 2026-06-15
decision-makers: thkt
---

# Freeze the --json Output Envelope as a Stable Consumer Contract

## Context and Problem Statement

Under `--json`, recall emits `SuccessEnvelope` (`data` / `degraded` / `notes`) to stdout and `ErrorEnvelope` (`error.code` / `message` / `next_step` / `candidates` / `retryable`) to stderr (envelope.rs). Agents parse these keys. The code decouples the wire schema from internal types via a `serde_json::json!` seam and keeps `SearchResult` / `SessionData` non-`Serialize`, but no document states that the envelope keys are a frozen public contract, nor that every string field reaching stdout must be control-char sanitized (`format_result` strips each field at main.rs:1096-1117, the error-note path at main.rs:339). A future contributor could break agent consumers by renaming a key or surfacing a raw field.

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

A golden/snapshot test asserts the `--json` key set for `search` / `index` / `status` / `doctor` (search/index/status assertions live in tests/cli_integration.rs; `doctor`'s `data` payload freezes the top keys `healthy` / `checks` and each check object's `name` / `ok` / `detail` / `remedy`, main.rs:1059). A grep check confirms `SearchResult` and `SessionData` carry no `derive(Serialize)`. Code review verifies every new string field in `format_result` and the `json!` payloads passes `ansi::strip_control_chars`.

## More Information

### Trade-offs

ADR-0060 governs the shared envelope pattern across the tool family; this ADR records the recall-specific freeze + strip-every-field commitment that ADR-0060 does not cover.

### Degraded Trigger Semantics (amendment 2026-06-17)

This freeze fixed the envelope KEY `degraded`, not the conditions that set it. The trigger is recorded here so a future contributor does not "reconcile" an intentional cross-command divergence into a bug.

For `search`, `degraded:true` means semantic (vector) coverage was lost relative to a fully-loaded run; the search still returns its FTS-only results rather than failing. Two sources feed it, and they co-vary with the same boolean so the bool and the notes never disagree:

- Load-time: the embedder could not load (model-less or a probe/backend failure), derived by `search_degraded_state` (main.rs:251-258). A caller-disabled model is intentional and stays non-degraded.
- Runtime: a loaded embedder's `embed_query` / `vec_search` fails at query time (e.g. memory pressure) and is swallowed to FTS-only, carried by `SearchOutcome::vec_degraded` (search.rs:39-57, 676-683). It is the runtime counterpart to the load-time path.

The no-model state is reported on two different axes by design, and this asymmetry is intentional, not a contract violation:

- `search` reports it on a coverage axis: no model means semantic search is unavailable, so results are FTS-only and may be thinner, hence `degraded:true`.
- `doctor` and `status` report it on a health axis: FTS-only is a supported mode, not a broken index, so a not-installed model is an info-level note that never flips `healthy` to false (main.rs:834-845). `doctor` and `status` agree with each other on this axis (#166).

A contributor who tries to make all three commands agree on `degraded` for the no-model case would either wrongly mark `doctor`/`status` unhealthy or wrongly hide search's lost-coverage signal. The divergence is the contract.

### Reassessment Triggers

- A v2 JSON schema is introduced (then version the envelope explicitly)
- A consumer needs a field that only exists on an internal type, forcing a seam redesign
