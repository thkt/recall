---
status: "accepted"
date: 2026-11-02
decision-makers: thkt
scope: "recall search model loading; index, rebuild, doctor and model download keep the probe"
---

# Skip the Isolated Model Probe on Search in a Verified Environment

## Context and Problem Statement

Before loading the embedding model, recall ran an isolated probe in a child process on every command that needs the model. The child checks that the GPU backend is usable, loads the model, embeds one input and exits. The parent then loads the same model again. A crash or hang inside llama.cpp/Metal therefore ends in the child, and the parent falls back to FTS-only search with a `DegradedReason` (`try_load_embedder_default_logging` in [src/model.rs](../../src/model.rs)).

After #377 and #378, this probe is one of the two largest parts of `recall search` time. Measurements recorded in [Issue #379](https://github.com/thkt/recall/issues/379) on real data (20,197 sessions):

- The probe child alone takes about 0.7 to 0.8 s when warm at load average about 10, and about 1.1 s under high load.
- Search p50 was 1.34 to 1.67 s at load average about 7, measured before #378 merged.
- At load average 15 to 24, p50 before #377 was 2.6 to 3.2 s, against the OUTCOME indicator of 2 s for `recall search`.
- The parent's SHA-256 verification and model load (about 0.8 to 0.9 s) remain either way, so they bound how far search can drop.

The probe also has false positives. In #352, real inference succeeded on GitHub's virtual GPU, but the probe hit its 30-second timeout and search fell back to FTS.

## Decision Drivers

- `recall search` is the interactive command; the probe cost is paid on every query
- An unknown environment must keep today's crash isolation
- A user who hits a crash once must not keep hitting it on later searches
- Record I/O must never fail or degrade a search

## Considered Options

- Skip the probe on `recall search` only while the environment matches one where the probe already succeeded (chosen)
- Keep the probe and cut time elsewhere, for example by loading in the parent while the child probes
- Always skip the probe on `recall search`

## Decision Outcome

Chosen option: "skip the probe on `recall search` only in a verified environment". The user accepted the crash risk only for an environment where the probe, the parent load and a parent self-check embedding all succeeded before. The parallel-load option was not adopted because that risk was accepted. Always skipping was rejected because it removes the guarantee for unknown environments.

The agreed rules come from the Issue #379 Agreement section (user answers on 2026-11-01 and 2026-11-02). The crash detection window excludes the query embedding, FTS and output because the probe child never protected those stages either. It also keeps one Ctrl-C during a search from disabling skipping for good.

| Rule                          | Decision                                                                                                   |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------- |
| Risk                          | Accepted only while the environment matches a recorded success                                             |
| Commands                      | `search` only. `index`, `rebuild`, `doctor` and `model download` run the probe every time                  |
| Record expiry                 | None                                                                                                       |
| After a crash is confirmed    | That environment never skips again                                                                         |
| Concurrent searches           | All of them may skip, sharing one skip's risk                                                              |
| Crash detection window        | Only the skipped load: GPU check, model load and self-check embedding. Later stages are not covered        |

### Shape

The record store lives in [src/model/probe_record.rs](../../src/model/probe_record.rs), and both loaders live in [src/model.rs](../../src/model.rs).

Degraded reasons and notes on the skip path stay those of [DR 0001](0001-freeze-the-json-output-envelope-as-a-stable-consumer-contract.md). Reading the two sysctl values needs `libc::sysctlbyname`, so one function carries `#[allow(unsafe_code)]` with `SAFETY` comments under the crate's `unsafe_code = "deny"` lint, as the sqlite-vec registration in [src/storage.rs](../../src/storage.rs) does.

| Element         | Behavior                                                                                                                                                                                                                                       |
| --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Environment key | SHA-256 over the binary's size and mtime, the selected manifest identity (`ModelSpec::identity`), `kern.osversion`, `hw.model`, and the `GGML_*` environment variables sorted by name. A change to any of them is a new, unrecorded environment |
| Store           | `<cache base>/recall/probe/`, where the cache base is `HF_HOME`, then an absolute `XDG_CACHE_HOME`, then `~/.cache`. It sits outside `embedding-models/`, whose layout stays identical with sae                                                 |
| Files           | `<key>.ok` holds the key after a success. `<key>.no-skip` marks an environment that must always probe. `<key>.<pid>.<n>.running` exists during each skipped load                                                                                |
| Writing         | The always-probe loader (index, rebuild, doctor) and the search loader's probe path write `<key>.ok` after the probe, the parent load and a parent self-check embedding succeed. In an environment with `<key>.no-skip` they skip the self-check and write nothing, because no skip reads that record                                                                 |
| Self-check miss | A failed parent self-check on the probe path withholds the record but returns the model as before, so an unrecorded environment keeps today's results                                                                                           |
| Clearing        | Any failure of the probe, the load or the self-check, in either loader, removes `<key>.ok`                                                                                                                                                      |
| Crash detection | A skipped load creates its marker under a staging name, locks it with `File::lock`, then renames it into place, and unlinks it before releasing the lock. A marker whose lock is free is a crash; a locked marker older than 30 seconds (the probe timeout) is a hang. Either writes `<key>.no-skip` |
| Skip path       | With a record, no tombstone and no crashed or hung marker, search checks the GPU in the parent, loads and runs one self-check embedding. No GPU returns `BackendUnavailable`; other failures return `ProbeFailed`; both clear the record          |
| I/O failure     | Failure to read or write any of these files runs the probe and logs `tracing::warn`. It never becomes a `DegradedReason`                                                                                                                        |

### Consequences

- Good, because a search in a verified environment no longer starts the probe child, removing up to the 0.7 to 1.1 s the child took in the measurements above
- Good, because a binary, model, OS build, hardware or `GGML_*` change runs the probe again before any skip
- Good, because a crash or hang during a skipped load stops all later skipping in that environment
- Bad, because a skipped load that crashes now ends the user's search process instead of a child; the user sees one failed search before the environment returns to probing
- Bad, because a hang inside a skipped load has no timeout: the probe child was killed after 30 seconds, but the user's search now waits until interrupted. Only later searches detect the hang and stop skipping
- Bad, because concurrent searches that start before the first crash is detected all share that risk
- Bad, because a probe timeout under heavy load (the #352 false positive) clears the record, so the next search pays the probe again

### Confirmation

- Tests T-001 to T-015 named in Issue #379 run without a GPU. The loader tests (T-008 to T-015) replace only the GPU check, the model load and the probe child; the record store under test is real. T-014 and T-015 run search and doctor through the same functions that pick their loader in production (`run_search_in`, `run_doctor_in`), and an index test does the same through `index_embedder`, so a command switched to the other loader fails a test.
- The public-model CI job accepts a probe timeout (`tests/gemma_cli.rs`), so it is not guaranteed to reach the skip path. A real-model check runs on Apple Silicon with `RUST_LOG=recall=debug`: of two searches, the second logs `isolated model probe skipped` and returns non-degraded results, and a following `recall doctor` and `recall index` each log `starting the isolated model probe`.
- Each model stage logs `model stage finished` with `stage` and `elapsed_ms` at debug level (`artifact sha256`, `isolated probe`, `gpu check`, `model load`, `self-check embedding`, `query embedding`), so the per-stage times can be compared against main.

## More Information

- Issue: [#379](https://github.com/thkt/recall/issues/379), including the measurements and the Agreement section quoted above
- Related: #352 (probe timeout on a virtual GPU), #377 and #378 (earlier search time reductions)

### Reassessment Triggers

- A crash in a skipped load is reported from a real environment; reconsider whether the risk acceptance still holds
- The parent's SHA-256 verification or model load is reduced (Backlog candidates in #379); the probe's share of search time changes
- llama.cpp gains a reliable in-process way to detect an unusable GPU backend without loading the model
