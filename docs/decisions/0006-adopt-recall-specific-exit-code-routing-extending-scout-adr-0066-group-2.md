---
status: "accepted"
date: 2026-06-24
decision-makers: thkt
consulted: scout ADR-0066, scout ADR-0060 (external family-registry refs, not retrievable files; see Context)
informed: AI agent integrators (retry policy consumers)
---

# Adopt Recall-Specific Exit Code Routing Extending Scout ADR-0066 Group 2

## Context and Problem Statement

recall stays anyhow-based internally, yet at the CLI boundary it must hand AI agents a stable signal: agents branch on the process exit code and the JSON `error.code` string to decide whether to retry, fix the invocation, or give up. scout ADR-0066 sorts the CLI family (scout / xr / notch / rurico / amici / sae / yomu / recall / Hook tool) into error-topology groups and places recall in Group 2 (local semantic search), whose baseline is the sysexits-derived set 0 / 64 / 65 / 70 / 73 / 74 / 75 / 104 built on `amici::cli::exit_code::codes`. recall applies that baseline through `classify_exit_code` and `to_error_envelope` (error.rs:135-159), distinguishing six classes (`UsageError` / `DataError` / `Internal` / `IoError` / `TempFailure` / `Unknown`, error.rs:31-44) with `retryable` true only for `TempFailure` (error.rs:117, 156).

scout ADR-0066 and scout ADR-0060 are external family-registry references; they have no record inside recall/docs/decisions/ (only 0001-0005 exist). The recall-specific routing therefore has no local anchor: its history lives only in the module doc and the issues that shaped it (the replaced `USER_ERROR_MARKERS` stderr-string match that broke on reworded `bail!`, #26 / #82 / #83, error.rs:6-8; the `--json` envelope, #67 Phase 2). An agent integrator or a refactoring contributor cannot read, from one place, which fault maps to which code and why retry is gated to `TempFailure`. A refactor of the classifier could silently shift a class and break agent retry logic.

## Decision Drivers

- AI agent retry policy must stay stable: agents branch on the `error.code` string and the exit code, so the class-to-code mapping is a public contract that must survive refactors
- recall must not drift from the shared Group 2 baseline (`amici::cli::exit_code`), which sae / yomu / rurico / amici also consume
- The `error.code` string tokens intersect ADR-0002 (frozen string tokens) and ADR-0001 (frozen envelope shape); the routing record must point at that intersection
- The JSON `error.code` and the process exit code must never disagree, and that invariant spans two emit sites (`to_error_envelope` and the anyhow `error_envelope`, error.rs:117, 156)

## Considered Options

- Cite scout ADR-0066 only, no local ADR (history stays in PR / CHANGELOG)
- Local ADR anchoring recall's routing as a recorded extension of Group 2
- Fork the exit-code policy locally, independent of the family registry

## Decision Outcome

Chosen option: "Local ADR anchoring recall's routing", because scout ADR-0066 and ADR-0060 have no local record here, so citing them alone leaves a refactorer or agent integrator with no readable anchor for recall's actual class table; and forking the policy would drift recall away from `amici::cli::exit_code`, which the rest of the Group 2 family shares. This mirrors sae ADR-0002, which records the same Group 2 extension pattern for sae and notes recall reuses it. The ADR keeps scout upstream and records only recall's local extension as the diff.

### Consequences

- Good, because the six classes recall distinguishes and the `retryable = TempFailure` rule are readable in one place, citable from a classifier refactor PR
- Good, because the boundary between the shared Group 2 baseline and recall's local extension (drop `CANT_CREAT`, add `Unknown` catch-all, the `SanitizeError` split, the `download_error` permanent-vs-transient split) is explicit
- Good, because the agent-facing contract (`error.code` string plus exit code, with retry gated to `TempFailure`) is recorded against the tests that pin it
- Bad, because a revision of scout ADR-0066 Group 2 (a new `amici::cli::exit_code` constant, a changed meaning) requires revising this ADR
- Bad, because adding a `RecallError` variant or a new mapped error type means updating this ADR's table by hand (code review enforces it, as no test reaches an unmapped path)

### Confirmation

The class-to-code table, the serde string tokens, and the retryable rule are pinned by unit tests in src/error/tests.rs: `error_code_numbers_match_sysexits_baseline` (64 / 65 / 70 / 74 / 75 / 104), `exit_code_wrappers_agree_with_code_table` (the `ExitCode` wrappers match the `code()` table), `error_code_serializes_screaming_snake_case` (T-ERR002, the `"USAGE_ERROR"` ... `"UNKNOWN"` tokens), `to_error_envelope_marks_temp_failure_retryable` (T-ERR004) with its `marks_usage_non_retryable` / `marks_internal_and_data_non_retryable` complements, and `classify_falls_back_to_unknown_for_unmapped_anyhow`. At the real process boundary, tests/cli_integration.rs pins only `USAGE_ERROR` (64, `no_subcommand_json_emits_error_envelope_on_stderr`) and success (0); the `DataError` / `Internal` / `IoError` / `TempFailure` / `Unknown` arms are not exercised through the spawned binary, so at the process level they rest on the src/error/tests.rs unit table plus code review. A classifier refactor must keep both emit sites (`to_error_envelope`, `error_envelope`) agreeing with this table.

## More Information

### Routing Table (recall-local extension of scout ADR-0066 Group 2)

| Exit | `error.code` | Group 2 baseline        | recall routing                                                                                        | source                   |
| ---- | ------------ | ----------------------- | ----------------------------------------------------------------------------------------------------- | ------------------------ |
| 0    | (none)       | Ok                      | success                                                                                               | ADR-0066 G2              |
| 64   | USAGE_ERROR  | bad invocation          | `RecallError::Usage` (missing query, unresolved/ambiguous id, no index)                               | error.rs:64-66, 82       |
| 65   | DATA_ERROR   | malformed input         | `RecallError::DataError`; `SanitizeError::EmptyInput` / `NoSearchableTerms`                           | error.rs:67-69, 169      |
| 70   | INTERNAL     | invariant violation     | `RecallError::Internal`; `SanitizeError::InvalidVocabTable`; `ModelDownloadError::BackendUnavailable` | error.rs:70-73, 168, 188 |
| 73   | CANT_CREAT   | output-create failure   | absent: DB / IO failures map to `IoError`, no distinct create path                                    | error.rs:10-12           |
| 74   | IO_ERROR     | sqlite / IO failure     | raw `io::Error` / `rusqlite::Error`; `SanitizeError::VocabLookupFailed`                               | error.rs:167, 172-175    |
| 75   | TEMP_FAILURE | retryable transient     | `RecallError::TempFailure`; `ModelDownloadError::DownloadFailed` / `ProbeFailed` (retry may succeed)  | error.rs:74-76, 185-186  |
| 104  | UNKNOWN      | unclassified (fallback) | unmapped anyhow error; signals a path to promote to a typed variant                                   | error.rs:42-43, 176      |

### Routing Invariants

- retryable is true only for `TempFailure`, asserted identically at both emit sites so the JSON and the exit code never disagree (error.rs:117, 156, 142-144)
- `download_error` keeps the permanent MLX `BackendUnavailable` mismatch off the retry path (Internal, 70) and routes only the re-runnable download / probe failures to `TempFailure` (75), so a hardware mismatch never invites a retry (error.rs:182-190)
- `Unknown` (104) is reserved for anyhow errors with no mapping; reaching it is the signal to add a typed `RecallError` variant rather than to widen a mapping (error.rs:42-43, 176)
- The `error.code` strings are emitted string tokens frozen by ADR-0002; their key and shape in the envelope are frozen by ADR-0001. A renamed token or class breaks both contracts

### Reassessment Triggers

- scout ADR-0066 Group 2 baseline is revised (a code added, a meaning changed) or `amici::cli::exit_code::codes` exposes a new constant
- A new `RecallError` variant or a newly mapped error type appears (Group 2 cannot express the topology, or a `CANT_CREAT` output-create path emerges)
- A `BREAKING` routing change is needed (then supersede this ADR)

### Related ADRs

- scout ADR-0066 (CLI exit code policy by error topology) — upstream Group 2 baseline, external family registry, no local record
- scout ADR-0060 (agent-friendly CLI design) — retry-policy context, external, no local record
- recall ADR-0001 (freeze the --json output envelope) — freezes the envelope shape carrying `error.code`
- recall ADR-0002 (persisted and emitted string tokens as a stable contract) — freezes the `error.code` string tokens
- sae ADR-0002 (sae-specific exit code routing extending scout ADR-0066) — sister ADR applying the same Group 2 extension pattern
