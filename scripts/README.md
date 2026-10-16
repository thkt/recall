# Verification and measurement scripts

These are maintainer tools, not prerequisites for normal `recall` use. Run them
from the repository root with Python 3. The current binary uses EmbeddingGemma 2
BF16 via the pinned llama.cpp/Metal backend; old Ruri/MLX measurements in
`research/` and `docs/research/` are historical evidence, not Gemma benchmarks.

| Script | Purpose | Requirements |
| --- | --- | --- |
| `fetch-gemma-test-artifacts.py` | Stage the public model and tokenizer from a manifest, verifying SHA-256 before replacement. Used by CI coverage. | Network access; Python standard library |
| `verify-gemma-independent.py` | Exercise the real CLI against synthetic sessions, separate model identities, checksum failure, legacy reads and update/rollback. | Built Gemma binary, verified artifacts, Python `sqlite-vec` |
| `measure-index.py` | Measure initial index, no-change, append and interrupt/resume on synthetic logs; optionally verify chunk reuse and pending-body bounds. | Built binary and a working model; macOS or another platform with `os.wait4` |
| `measure-classify.py` | Compare classification parsing and writer contention against the pinned pre-change parser, using synthetic logs. | Full Git history containing its baseline, Rust build tools and initialized native submodules |

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

For the classification benchmark:

```sh
git submodule update --init --recursive
python3 scripts/measure-classify.py
```

This stages the current sources and vendored native dependencies in a temporary
checkout, inserts only the pinned parser/classification baseline, and runs the
36 comparison conditions at least five times each. The current model dependencies
remain in the staged build; the old MLX build settings in the
[original report](../docs/research/issue-342-scope.md) describe historical runs.
The benchmark uses synthetic classification inputs and does not load a model or
measure Gemma inference. All scripts preserve their results separately from the
[dated migration validation record](../docs/audit/2026-10-16-gemma-validation.json).
