The Rust `llama-cpp-2` and `llama-cpp-sys-2` bindings are vendored from
utilityai/llama-cpp-rs commit `849e04446cb7b6a1fe3cee650e635a081ec072be`
(version 0.1.160, MIT OR Apache-2.0; licenses are in this directory).
Cargo workspace inheritance and upstream example targets are adjusted. The one
Rust source compatibility patch initializes the two added performance-counter
fields (`n_reallocs_diff` and `n_reallocs_same`) to zero in `LlamaTimings::new`.
The same patch was present in the evaluated backend; it does not change inference.
Build-only dependency adjustments disable bindgen's unused logging and
Prettyplease formatter features (the builder keeps its default Rustfmt formatter)
and make the CUDA discovery helper conditional on the existing CUDA feature.
The helper's build-script block uses `#[cfg]` rather than `cfg!` so a Metal-only
build does not compile or resolve CUDA discovery. Upstream inference wrappers and
the pinned llama.cpp source remain unchanged by these dependency adjustments.
Upstream public vocabulary/grammar test fixtures are retained. The adopted Gemma
weights, private corpus, and evaluation vectors are not included in this directory.

The llama.cpp submodule is fixed to
`1b70350c92563449576bcbbf552994b1d1ed847d`. This exact pairing was used by the
independent EmbeddingGemma 2 evaluation. Pinning the pairing prevents a binding
update from silently replacing the model backend with another submodule revision.

## Build-time llama.cpp patches

The build applies one upstream llama.cpp change on top of the pinned revision:

| Patch file                                           | Upstream                                                                                                          | Effect                                                                                                                                                                   |
| ---------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `patches/0001-llama-30679-faster-vocab-load.patch`   | [ggml-org/llama.cpp#30679](https://github.com/ggml-org/llama.cpp/pull/30679), commit `824eaecf48` (2026-10-21), 7 files, +73/-37 | `src/llama-model-loader.cpp` no longer converts every metadata value to a string when logging is discarded, and the vocabulary load no longer converts each array element to a string |

The patch is taken whole from the upstream commit (`git format-patch -1 824eaecf48`).
The pinned revision `1b70350` (2026-10-13) predates it.
Decision record: [DR 0014](../docs/decisions/0014-apply-upstream-llama-cpp-fixes-as-build-time-patches-on-the-pinned-submodule.md).

`build.rs` copies the llama.cpp directories it builds from (`ggml`, `src`, `common`,
`include`, `vendor`, `cmake`, `tools`, `pocs`, the top-level `CMakeLists.txt`,
`convert_hf_to_gguf.py`, and `LICENSE`) into `OUT_DIR`, applies every `patches/*.patch`
to that copy in name order with `git apply`, and passes the copy to CMake.
The submodule work tree and its pointer are not modified, so
`git -C vendor/llama-cpp-sys-2/llama.cpp status --short` stays empty after a build.
bindgen and the cc wrappers keep reading the original tree.

Rules the build enforces:

- A patch may change files below `src/` only. A patch that changes any other path fails the build.
- A patch that does not apply fails the build, and the error names the `.patch` file.
- After applying, the build reverses each patch as a dry run against the staged copy and against the
  synced tree CMake builds from. This catches a patch that `git apply` skipped while exiting 0.
- Syncing gives every rewritten file the current modification time, so toggling
  `RECALL_LLAMA_SKIP_PATCHES` rebuilds the reverted files instead of reusing patched objects.
- `git` runs with `GIT_CEILING_DIRECTORIES` set to the parent of the copy and with `GIT_DIR`,
  `GIT_WORK_TREE`, `GIT_INDEX_FILE`, and `GIT_COMMON_DIR` removed. `OUT_DIR` is under the
  recall repository's `target/`, and without this `git apply` prints `Skipped patch` and does nothing.
- `git` must be on `PATH` during the build.

`RECALL_LLAMA_SKIP_PATCHES=1` builds without applying any patch and emits a `cargo:warning`.
Use it to build the unpatched baseline. Cargo reruns the build script when this variable changes.

`llama_cpp_sys_2::APPLIED_PATCHES` lists the file names of the patches in the linked build.
It is empty for a build with `RECALL_LLAMA_SKIP_PATCHES=1`. A recall test asserts that the
vocabulary loading patch is among them.

When the submodule pin moves to or past `824eaecf48`, delete the patch file and its row above in the same change.

### Verifying that a patch does not change the model output

The patched build must produce the same token ids, vocabulary pieces, and embedding bits as the
unpatched build. The ignored test `real_model_bit_dump_for_build_parity` in
`src/native/real_tests.rs` writes them to one JSON file. Its text set contains search queries,
Japanese text, emoji, combining characters, code, whitespace, and queries and documents of
about 511, 512, 513, 8192, and 8193 tokens, plus a multi-window document. It also writes
`token_to_piece` for every vocabulary id, with and without special tokens. Embeddings are written
as `f32::to_bits` in hexadecimal.

`RECALL_GEMMA_TEST_ARTIFACT_DIR` is a directory with the verified `model.gguf` and
`tokenizer.json`. The test checks their SHA-256 against `ModelSpec` before loading.
Use the same machine, the same profile, and the same artifact directory for every dump.

```sh
dump() { # $1 = output file, remaining args = environment assignments
  out=$1; shift
  env "$@" RECALL_GEMMA_TEST_ARTIFACT_DIR=/path/to/artifacts \
    RECALL_GEMMA_TEST_BITS_OUT="$out" \
    cargo test --locked --release real_model_bit_dump_for_build_parity -- --ignored
}

dump /tmp/a1.json RECALL_LLAMA_SKIP_PATCHES=1   # unpatched, first run
dump /tmp/a2.json RECALL_LLAMA_SKIP_PATCHES=1   # unpatched, second run
cmp /tmp/a1.json /tmp/a2.json                   # A/A

dump /tmp/b.json                                # patched
cmp /tmp/a1.json /tmp/b.json                    # A/B
```

1. Compare the two unpatched dumps (A/A) first. If they differ, inference on that platform is not
   reproducible, and an A/B difference would mean nothing.
2. Compare the unpatched dump with the patched dump (A/B) only after A/A matches byte for byte.
3. If A/B differs, do not adopt the patch. Do not change `ModelSpec`'s `implementation` to absorb the difference.

When A/B matches, `ModelSpec`'s `implementation` (`recall-gemma-llama-v1`) stays unchanged.
The frozen independent vectors are not used for this comparison.

Initialize the source before building:

```sh
git submodule update --init --recursive
cargo build --locked --release
```

Changing this pairing requires encoder parity, token-boundary, index identity,
CLI/probe, and independent-index validation. Update the implementation identity
when an encoder behavior changes; do not reuse vectors from another embedding space.
