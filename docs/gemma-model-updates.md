# Gemma model validation, updates, and rollback

Recall defaults to the independently evaluated EmbeddingGemma 2 BF16 encoder.
The artifact URLs, full revisions, SHA-256 checksums, 768-dimensional projection,
mean pooling and L2 normalization, noncausal attention, query/document prefixes,
backend/binding commit IDs,
8192-token window and 2048-token overlap are pinned in `src/model.rs`.
The initial 512-token context grows to fit the query; queries beyond 8192 tokens
retain EOS when truncated. Long documents are split by tokenizer offsets and
retokenized with the document prefix. Shorter packing budgets reduce batch packing,
while a single attention window remains atomic and can exceed the requested budget.

The previous evaluation found better overall mean quality and lower encoder process
time/memory for Gemma, but did not establish non-inferiority within two quality
points. Gemma adoption is the user's explicit decision with that uncertainty.
The earlier resource improvement compared Python/MLX FP32 with native BF16 encoder
processes; it is not a measured end-to-end product CLI improvement.

## Migration status

The Gemma backend is included from v0.14.0; amici, rurico and MLX are no longer
runtime or build dependencies. Releases up to v0.13.0 use the previous Ruri backend.
The [migration validation record](audit/2026-10-16-gemma-validation.json) is a dated
report, not the current test count or evidence that a user's index has been promoted.

To switch your own sessions, use the Gemma binary, download its model, and index
into its new default namespace. Remove an old `RECALL_DB` override or select a
new explicit path first. Keep the old binary, model and index for rollback.
This is a new index from the session logs, not an in-place vector conversion.

## Independent validation

All migration validation must pass `--db-path` pointing to an owned test database
and set `RECALL_CLAUDE_DIR` / `RECALL_CODEX_DIR` to fixture directories. Do not run
an unqualified `index`, `rebuild`, `classify`, or `doctor --fix` against real data
as a validation step. Model artifacts can be read from `RECALL_MODEL_DIR`; the
two expected files are `model.gguf` and `tokenizer.json`, and both checksums are
verified before loading. No daemon or embedding service is required.

```sh
recall model manifest > gemma-current.json
RECALL_MODEL_DIR=/absolute/verified/model \
RECALL_CLAUDE_DIR=/absolute/fixture/claude \
RECALL_CODEX_DIR=/absolute/fixture/codex \
recall --model-manifest gemma-current.json --db-path /absolute/test/gemma.db index
RECALL_MODEL_DIR=/absolute/verified/model \
recall --model-manifest gemma-current.json --db-path /absolute/test/gemma.db search 'query' --json
```

`recall model download` installs only the selected manifest's pinned HTTPS
artifacts. Both sae and recall use `~/.cache/embedding-models/<artifact-sha256>/`
(or under `HF_HOME`, then absolute `XDG_CACHE_HOME`) for the same model/tokenizer
bytes. Verified existing app caches are imported automatically with hard links
where possible; old paths and app-specific index fingerprints stay intact.
Explicit `RECALL_MODEL_DIR` selects its own directory and bypasses legacy import.
Downloads are serialized with a shared file lock and verified before atomic replacement; a separate child
process probes the model before normal loading. A timeout kills and reaps that child.
There are no automatic model downloads on search/index and no automatic updates.

The [script guide](../scripts/README.md) documents the synthetic real-CLI
update/rollback check and its Python `sqlite-vec` dependency. These checks use
fixture logs and new databases; they do not promote a production index.

## Index identity

Default indexes live at
`$XDG_DATA_HOME/recall/indexes/<manifest-sha256>/recall.db`, falling back to
`~/.local/share/recall/indexes/<manifest-sha256>/recall.db`. Each database records
the same fingerprint and manifest in `embedding_identity`. `RECALL_DB` and
`--db-path` still override the path, but do not override identity checks.

The legacy `~/.local/share/recall/recall.db` is left in place. A nonempty vector
index without identity is treated as legacy, never inferred to be Gemma just
because it also has 768 dimensions. Semantic reads fall back to FTS with a
degraded note; status and doctor also flag the mismatch, and writes refuse before schema migration. A mismatched identified
index also refuses rebinding, including when empty. Existing session/message
deletion rules and generation-checked embedding saves remain in force.

## Evaluate an update

1. Archive the currently selected manifest with `recall model manifest` and keep
   its model, tokenizer, binary/backend version, and matching index.
2. Create a separate manifest with full revisions and verified checksums. The
   schema rejects unknown keys and unsupported pooling, dimensions, and splitting.
   A different supported artifact or prefix produces a different fingerprint.
3. Download or stage the candidate artifacts in a separate directory. Build a
   separate index using explicit fixture roots and `--db-path`. Never copy new
   vectors into the currently adopted model's index.
4. Compare the same fixed queries and new independent intents against the currently
   adopted Gemma. Preserve nDCG@10, useful-session and useful-excerpt Success@5,
   paired intervals and the two-point tolerance, including identifier, filter and
   long-text diagnostics. Repeated paraphrases alone do not add independent intents.
5. Recheck real CLI/probe output, token boundaries, resume/no-change behavior and
   resource use with matched timing boundaries. Encoder time and end-to-end CLI
   time are separate measurements. Review the evidence before promotion.

## Roll back

Select the archived manifest and its original matching database; do not relabel
the candidate database as the old model. A saved old manifest reproduces the old
default namespace. With explicit paths:

```sh
RECALL_MODEL_DIR=/absolute/previous/model \
recall --model-manifest /absolute/previous/manifest.json \
  --db-path /absolute/previous/index.db search 'query'
```

For an encoder/backend code rollback, retain and restore the corresponding binary
as well. The legacy Ruri binary and index are not converted in place; returning
to that implementation requires that older matching binary. This migration does
not execute it or depend on amici. Keep previous artifacts until the successor
has been reviewed. Validation does not promote a production index.

## CI verification

The [CI workflow](../.github/workflows/ci.yml) stages only the manifest's public, SHA-256-verified
model and tokenizer, then combines standard-test coverage with opt-in native
inference and synthetic CLI/probe tests. The changed-line coverage gate remains
95%. Private frozen queries, rankings and full index copies stay local.

The public tests check all 511/512/513/8192/8193 token boundaries and UTF-8 window
coverage. Short inference exercises the initial 512 context and a 513-token
inference exercises growth; full 8192-token inference was measured separately
on Apple M3 with the frozen evaluation. Hosted runners may expose no GPU or a
slow virtual GPU. Native inference is
measured separately; CLI tests accept only a successful probe, explicit
backend-unavailable error, or the production 30-second probe timeout, and verify
honest FTS fallback for each degraded invocation. This is separate from the
successful real GPU probe on Apple M3.
