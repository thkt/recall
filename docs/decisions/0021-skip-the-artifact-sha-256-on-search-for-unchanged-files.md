---
status: "accepted"
date: "2026-11-22"
decision-makers: "thkt"
scope: "How recall search, index, rebuild, doctor and status check the cached model artifacts before use (src/model.rs, src/model/artifact_record.rs). model download, reuse from older caches and the probe child are unchanged"
---

# Skip the Artifact SHA-256 on Search for Unchanged Files

## Context and Problem Statement

Before loading the model, every `recall search` computed the SHA-256 of `model.gguf` (558 MB) and `tokenizer.json` (32 MB) and compared them with the manifest. Issue [#476](https://github.com/thkt/recall/issues/476) measured the `RUST_LOG=recall=debug` stage times on a release build of main 9118809 (Apple M3, 1-minute load average 5.5 to 6.1, 7 runs, with a synthetic scratch index):

| Stage                  | Time                         |
| ---------------------- | ---------------------------- |
| `artifact sha256`      | 263 to 347 ms (median 270 ms) |
| `gpu check`            | 47 to 65 ms                  |
| `model load`           | 204 to 275 ms                |
| `self-check embedding` | 28 to 50 ms                  |
| `query embedding`      | 21 to 32 ms                  |
| Whole search (6 other runs) | 0.62 to 1.04 s          |

The hash is about 40% of a search. The OUTCOME indicator (search within 2 s) already holds at the median, so this is a reduction, not a fix. DR 0013 lists a reduction of this hash among its Reassessment Triggers, and Issue #379 left skipping or remembering the hash as a Backlog candidate that needs its own decision, because it weakens the detection of tampering and corruption. How can search stop hashing unchanged files while a change to them is still detected?

## Decision Drivers

- `recall search` is the interactive command and pays the hash on every query
- A changed or replaced artifact must not be trusted by a later search
- The cache layout under `embedding-models/` stays identical with sae (`src/model/cache.rs`)
- Record I/O must never fail or degrade a command, as with the DR 0013 probe record
- `recall status --json` writes nothing to the scratch directory (`status_json_without_index_reports_model_ready_when_cached` in `tests/cli_integration.rs`)

## Considered Options

- Record the identity of files that passed the hash, and let search skip the hash while the identity is unchanged (chosen)
- Record only size and mtime
- Let only search write the record
- Make the hash faster
- Stop verifying the artifacts
- Give the record an expiry

## Decision Outcome

Chosen option: "record the identity of files that passed the hash, and let search skip the hash while the identity is unchanged", because it removes the hash from a search on unchanged files while any rewrite or replacement that changes the inode or ctime is hashed again. The agreed rules come from the Agreement section of Issue #476 (agreed with thkt on 2026-11-22).

| Rule     | Decision                                                                                                                                                 |
| -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Identity | dev, inode, size, mtime (nanoseconds) and ctime (nanoseconds) of each of `model.gguf` and `tokenizer.json`                                               |
| Reading  | Only `recall search`. `index`, `rebuild`, `doctor` and `status` hash every time, with or without a record                                                |
| Writing  | `search`, `index`, `rebuild` and `doctor` write the record after both files match the manifest. `status` never writes                                    |
| Clearing | Any command whose hash fails removes the record. A search that skipped the hash removes it when the GPU check, the model load or the self-check embedding fails, including a failed parent self-check on the probe path that still returns the model as DR 0013 decided |
| Expiry   | None                                                                                                                                                     |
| I/O      | A record that cannot be read or written means the files are hashed, with `tracing::warn`. It never becomes a `DegradedReason`                             |

### Shape

The record store lives in [src/model/artifact_record.rs](../../src/model/artifact_record.rs), and `ModelHost` in [src/model.rs](../../src/model.rs) picks the check per command. DR 0013 is unchanged: a search that skipped the hash still runs its GPU check, model load and one self-check embedding, or the isolated probe when the environment has no probe record.

| Element    | Behavior                                                                                                                                                                                                       |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Store      | `<cache base>/recall/verified-artifacts/`, with the cache base chosen as for the DR 0013 probe records (`HF_HOME`, then an absolute `XDG_CACHE_HOME`, then `~/.cache`). Nothing is added under `embedding-models/` |
| Key        | SHA-256 over the model directory as resolved (including one chosen by `RECALL_MODEL_DIR`), the manifest `model_sha256` and the manifest `tokenizer_sha256`. One file `<key>.json` per key                     |
| Contents   | JSON with the key and both file identities. A partly written file, unknown fields or another shape count as no record                                                                                         |
| Skip       | Search skips the hash only when the record's key and both identities equal the current ones. It logs `artifact sha256 skipped` at debug level instead of the `artifact sha256` stage                          |
| Writing    | The identities are read before the hash and again after it. The record is written to a temporary file and renamed into place, only when the two reads are equal                                              |
| Platforms  | File identities are read on Unix only. Elsewhere every command hashes                                                                                                                                         |

### Consequences

- Good, because a search on unchanged files no longer spends the 263 to 347 ms of the `artifact sha256` stage measured above
- Good, because a rewrite of the content, a `chmod`, or a rename that replaces a file changes the ctime or the inode on APFS, so the next search hashes again. The Issue checked these cases on main 9118809 with `printf` overwrites, mtime restored with `touch -r`, `chmod`, and rename
- Good, because the session-end `recall index` hashes every time, so a content-only corruption is still found without an expiry
- Bad, because a search does not detect a change when someone who can write both the model files and the record rewrites the record and replaces the files (case ① in the Issue Agreement)
- Bad, because a search does not detect a corruption that changes only the content and keeps the identity (case ②). It is found the next time `index`, `rebuild`, `doctor` or `status` hashes, which also removes the record

### Confirmation

- `src/model/artifact_record_tests.rs` uses synthetic artifacts and a matching `ModelSpec`, without a GPU or a model. It counts the full SHA-256 checks. It covers the skip after a passed check, a same-size rewrite with its mtime restored, a changed manifest checksum, an unreadable or unwritable record, and a failed load or parent self-check after a skipped hash. One test runs `run_status_in`, `run_search_in`, `index_embedder` and `run_doctor_in`, the functions the commands use in production, so a command switched to another check fails it.
- The real-model check is not part of the configured check, capture or CI. Claude runs it after the workflow publishes the draft PR and before the PR is switched to ready, as agreed in Issue #476. On Apple Silicon with `RUST_LOG=recall=debug`, a second `recall search` logs no `artifact sha256` stage and returns non-degraded results, and a following `recall index` logs the `artifact sha256` stage. Main and this change are timed alternately at least 5 times.

## Pros and Cons of the Options

### Record only size and mtime

- Good, because the record is slightly smaller
- Bad, because a same-size rewrite with its mtime restored is missed. Adding ctime and inode barely changes the record size

### Let only search write the record

- Good, because index, rebuild and doctor keep no write path
- Bad, because a passing hash in the hook's `recall index` would not refresh the record, so fewer runs could find a content-only corruption

### Make the hash faster

- Good, because no record is needed
- Bad, because SHA-256 cannot be computed in parallel within one file. Hashing the two files in parallel still leaves the 558 MB file

### Stop verifying the artifacts

- Bad, because tampering and corruption are no longer detected

### Give the record an expiry

- Bad, because it adds a periodic hash that the hook's `recall index` already provides by hashing every time

## More Information

- Issue [#476](https://github.com/thkt/recall/issues/476) holds the measurements, the premise about APFS identities and the Agreement section quoted above
- [DR 0013](0013-skip-the-isolated-model-probe-on-search-in-a-verified-environment.md) defines the probe record whose store location and I/O rules this record follows
- Issue #379 left this reduction as a Backlog candidate

### Reassessment Triggers

- A filesystem in use changes file content without changing its ctime or inode, which turns case ② into a common case
- A report shows case ① in practice, for example another process that writes both the cache and the record
- sae adopts the same record, which would move it into the shared cache protocol
