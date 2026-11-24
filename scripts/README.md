# Verification and measurement scripts

These are maintainer tools, not prerequisites for normal `recall` use. Run them
from the repository root with Python 3. The current binary uses EmbeddingGemma 2
BF16 via the pinned llama.cpp/Metal backend.

| Script | Purpose | Requirements |
| --- | --- | --- |
| `fetch-gemma-test-artifacts.py` | Stage the public model and tokenizer from a manifest, verifying SHA-256 before replacement. Used by CI coverage. | Network access; Python standard library |
| `verify-gemma-independent.py` | Exercise the real CLI against synthetic sessions, separate model identities, checksum failure and update/rollback. | Built Gemma binary, verified artifacts, Python `sqlite-vec` |
| `measure-index.py` | Measure initial index, no-change, append and interrupt/resume on synthetic logs; optionally verify chunk reuse and pending-body bounds. | Built binary and a working model; macOS or another platform with `os.wait4` |
| `measure-search.py` | Time `recall search` on an existing index, with and without the model, and compare the 95th percentile with the 2-second indicator in `.claude/OUTCOME.md`. | Built binary, an index, a file of queries |

## Public-model tests

From a built checkout, export the plain manifest (without `--json`, which would
wrap it in the CLI output envelope), then download to an explicit test directory:

```sh
./target/release/recall model manifest > /absolute/test/gemma-manifest.json
python3 scripts/fetch-gemma-test-artifacts.py \
  /absolute/test/gemma-manifest.json /absolute/test/gemma-model

RECALL_GEMMA_TEST_ARTIFACT_DIR=/absolute/test/gemma-model \
  cargo test --locked --bin recall public_model_contract -- --ignored --test-threads=1
RECALL_GEMMA_TEST_ARTIFACT_DIR=/absolute/test/gemma-model \
  cargo test --locked --test gemma_cli public_model_cli_contract -- --ignored --test-threads=1
```

Use an existing parent directory for the manifest. These tests do not read real
session logs. Hosted runners can lack a usable GPU: CLI tests check honest FTS
fallback on a production probe failure or timeout, separately from native inference.
See [CI verification](../docs/gemma-model-updates.md#ci-verification) for coverage scope.

## Independent model update and rollback

Use a new work directory and a Python environment with `sqlite-vec` installed:

```sh
python3 -m venv /absolute/test/venv
/absolute/test/venv/bin/python -m pip install sqlite-vec
/absolute/test/venv/bin/python scripts/verify-gemma-independent.py \
  --binary /absolute/path/to/recall \
  --model-dir /absolute/test/gemma-model \
  --work-dir /absolute/test/new-validation
```

The script copies the binary, generates fixture logs and owns all test indexes.
It writes `verification.json` and per-invocation logs. The candidate changes only
the document prefix; passing proves that simulated update's identity separation
and rollback behavior, not the quality of a new set of weights. A fully working
GPU probe is required; degraded results fail this check.

## Synthetic measurements

For index cases and comparison conditions, use the
[index measurement procedure](../docs/index-observability.md). Each invocation
requires explicit binary, output, model identity and machine/cache conditions.
Keep before/after encoder and model settings identical for code comparisons;
measure a model change separately.

```sh
python3 scripts/measure-index.py /absolute/path/to/recall /absolute/test/new-measurement \
  --model 'model ID, revision and artifact identity' \
  --conditions 'commit, chip, RAM, OS, Rust, cache state and load'
```

## Search time

Write one query per line to a file and pass the binary and the file:

```sh
python3 scripts/measure-search.py /absolute/path/to/recall /absolute/path/to/queries.txt
```

The script searches the binary's default index; `--db-path` selects another one. It runs each
query three times with the model and three times without it, after one unmeasured search. The
report is JSON with the index size and, per mode, the median, the 95th percentile, the slowest
search, and the number of searches over the limit. It prints no query and no result, so it can
run on a real index. The exit status is 1 when a 95th percentile exceeds `--limit-seconds`
(default 2) or a search fails.

Searches that follow each other read the index and the model from the OS file cache. A search
after the machine did other work reads them from disk. `--drop-from-memory FILE` measures that
case: it adds the modes `hybrid_after_drop` and `keyword_only_after_drop`, which remove the file
from the OS file cache before every search. Repeat the option for each file, for example the
index (`recall status --verbose` prints its path) and the model's `model.gguf`:

```sh
python3 scripts/measure-search.py /absolute/path/to/recall /absolute/path/to/queries.txt \
  --drop-from-memory /absolute/path/to/recall.db \
  --drop-from-memory /absolute/path/to/model.gguf
```

`left_in_memory_fraction` is the largest share of a file that stayed in the cache after a
removal. A value above 0 means those searches read part of the file from memory, so their
times are shorter than a read from disk. The removal uses `msync` with `MS_INVALIDATE`, which
empties the cache on macOS only.

All scripts preserve their results separately from the
[dated migration validation record](../docs/audit/2026-10-16-gemma-validation.json).
