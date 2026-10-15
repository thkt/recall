The Rust `llama-cpp-2` and `llama-cpp-sys-2` bindings are vendored from
utilityai/llama-cpp-rs commit `849e04446cb7b6a1fe3cee650e635a081ec072be`
(version 0.1.160, MIT OR Apache-2.0; licenses are in this directory).
Cargo workspace inheritance and upstream example targets are adjusted. The one
Rust source compatibility patch initializes the two added performance-counter
fields (`n_reallocs_diff` and `n_reallocs_same`) to zero in `LlamaTimings::new`.
The same patch was present in the evaluated backend; it does not change inference.
Upstream public vocabulary/grammar test fixtures are retained. The adopted Gemma
weights, private corpus, and evaluation vectors are not included in this directory.

The llama.cpp submodule is fixed to
`1b70350c92563449576bcbbf552994b1d1ed847d`. This exact pairing was used by the
independent EmbeddingGemma 2 evaluation. Pinning the pairing prevents a binding
update from silently replacing the model backend with another submodule revision.

Initialize the source before building:

```sh
git submodule update --init --recursive
cargo build --locked --release
```

Changing this pairing requires encoder parity, token-boundary, index identity,
CLI/probe, and independent-index validation. Update the implementation identity
when an encoder behavior changes; do not reuse vectors from another embedding space.
