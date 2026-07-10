**English** | [日本語](README.ja.md)

# recall

Your past Claude Code and Codex sessions, searchable by keyword and meaning. Fully local, no API keys.

## The problem

You solved an authentication problem with Claude last week. Now you need that approach again — but which project was it? Which session?

**Without recall:**

```sh
find ~/.claude -name "*.jsonl" | wc -l
  3,851

grep -r "authentication" ~/.claude/projects/ | head
  ...12,000+ lines of raw JSONL
```

**With recall:**

```sh
recall "authentication"

[1] 2026-02-27 | stateful-sleeping-cosmos | kagami [claude]
    /Users/me/GitHub/kagami
    > Enable API keys Allow users and/or organizations to authenticate
      with your API programmatically...

[2] 2026-02-08 | fluffy-rolling-lampson | kai [claude]
    /Users/me/GitHub/kai/main
    > authenticator / isAuthenticated path: ...NID OAuth2 + PKCE
      authentication with nonce...
```

6,000+ sessions, 27,000+ Q&A pairs — searched in under 2 seconds.

## Quick start

```sh
# Install
brew install thkt/tap/recall
# or: cargo install --path .

# Download the embedding model once (~1.2 GB), then index your sessions
recall model download
recall index

# Search
recall search "authentication"
```

## How search works

recall indexes every session into keyword search (FTS5) and, in the same pass, embeds each chunk with a local AI model. Search blends both — **semantic search** finds sessions by meaning, not just keywords — and is read-only, so it returns instantly.

```
recall index:   Parse + FTS5 + embed every new chunk (needs the model)
recall search:  Hybrid ranking — FTS5 keyword + vector similarity (RRF), instant
Over time:      More sessions indexed → broader semantic coverage
```

No API keys. No data leaves your machine. The embedding model (Ruri v3) runs locally via MLX on Apple Silicon. Without the model, index still builds FTS5 and search falls back to keyword ranking.

## Usage

### Search

```sh
recall search "error handling"                                   # keyword search
recall search "database migration" --project /Users/me/GitHub/app  # filter by project
recall search "React Router" --days 7                            # last 7 days
recall search "async runtime" --source codex                     # Codex sessions only
recall search "auth AND middleware"                               # boolean operators
```

Backward compatible: `recall "query"` works as shorthand for `recall search "query"`.

| Flag                  | Description                                                         |
| --------------------- | ------------------------------------------------------------------- |
| `--project`           | Filter by project path (prefix match)                               |
| `--days`              | Only sessions from the last N days                                  |
| `--source`            | `claude` or `codex`                                                 |
| `--limit`             | Max results, 1-100 (default: 10)                                    |
| `--exclude-current`   | Exclude the invoking session (default inside a Claude Code session) |
| `--include-current`   | Include the invoking session even inside a session                  |
| `--only-current`      | Return only the invoking session                                    |
| `--include-automated` | Include automated (hook/script/agent) sessions; excluded by default |
| `-v`                  | Verbose output                                                      |

Supports [FTS5 query syntax](https://www.sqlite.org/fts5.html#full_text_query_syntax) — bare words, `"quoted phrases"`, and `AND` / `OR` / `NOT`.

### Index

```sh
recall index            # parse, chunk, and embed new session logs (incremental)
recall rebuild          # re-parse and re-embed every present session; missing roots keep their rows
```

Embedding needs the model: run `recall model download` (~1.2 GB) once. Without it, `recall index` builds FTS5 only and prints a note to download it; the next index after the model is present embeds the backlog.

`recall index` and `recall rebuild` accept two flags (also settable via env, useful for the [Hook](#hook)) that tune the embed pass:

```sh
recall index --token-budget 32000 --forward-pause-ms 200
# or
RECALL_TOKEN_BUDGET=32000 RECALL_FORWARD_PAUSE_MS=200 recall index
```

`--token-budget` (`RECALL_TOKEN_BUDGET`) is reserved for a future embedder revision to consume as a forward-pass token budget override; values above 256000 clamp down to 256000 (never up) so an over-large value cannot exceed the embedder's own ceiling once that revision lands. The current embedder ignores this value, so setting it today has no effect on OOM/swap during a large embed run. `--forward-pause-ms` (`RECALL_FORWARD_PAUSE_MS`) is the flag with a real effect today: it inserts a pause before each embed batch, trading embed throughput for host responsiveness during large runs, for example when indexing runs in the background while you keep working. Start from 0 (unset, no pause) and raise it in small steps (for example 100-200ms) only if you notice the machine sluggish during a big backlog embed; each added millisecond directly extends the total embed time. Leaving both flags unset preserves the current default behavior.

The index lives at `~/.local/share/recall/recall.db` by default (override with `--db-path` or the `RECALL_DB` env var; recall creates the parent directory on first run). Upgrading from a build that stored it at `~/.recall.db`? Move the old file before re-indexing, otherwise recall starts a fresh empty index at the new path and your past sessions stay invisible to search:

```sh
mkdir -p ~/.local/share/recall && mv ~/.recall.db ~/.local/share/recall/recall.db
```

### Model

```sh
recall model download   # download the embedding model and verify it loads
```

### Show

```sh
recall show abc-123     # show full conversation of a session (prefix match)
```

### Status

```sh
recall status           # sessions, chunks, embedding coverage, model status
```

### Classify

```sh
recall classify             # classify unclassified sessions interactive/automated
recall classify --all       # re-classify every session
recall classify --dry-run   # report what would change without writing
```

Each session is classified interactive or automated from its first user turn. Automated sessions (hook/script/agent-generated) are excluded from search by default; pass `--include-automated` to include them.

### Doctor

```sh
recall doctor           # diagnose a broken index; read-only by default
recall doctor --fix      # also delete orphan embeddings, then re-run the checks
```

Runs four checks: SQLite `quick_check`, orphaned embeddings, orphaned chunks, and a live model load-and-embed probe. Each failing check prints the remedy command (`recall doctor --fix` for orphan embeddings, `recall rebuild`, `recall model download`, or re-`recall index` after removing a corrupt DB). `--fix` deletes the dangling `vec_chunks` rows the orphan-embeddings check counts and re-runs every check, so the reported verdict reflects the repaired state (a per-check breakdown of rows repaired appears as `repaired` under `--json`, e.g. `{"orphan_embeddings": 3}`, or `null` when `--fix` is absent); without it `doctor` only reports. A not-installed model is reported as info, not a failure — search runs FTS-only without it, so the index stays healthy. Under `--json`, a failure sets `degraded: true` and lists each remedy in `notes`.

### Hook

`recall index` is the primary way to refresh — run it whenever you want search up to date. Optionally register it as a Claude Code SessionEnd hook to re-index the moment a session ends. Each fire re-scans the whole session tree (incremental — only changed files are re-parsed) and embeds new chunks.

Add to `~/.claude/settings.json`:

```json
{
  "hooks": {
    "SessionEnd": [{ "matcher": ".*", "hooks": [{ "type": "command", "command": "recall index" }] }]
  }
}
```

`recall index` reads its sources from the environment and ignores the hook's stdin payload, so no extra wiring is needed. With the model present, the first fire is a one-time cold start — it embeds your whole backlog (~11 min for 28k chunks); later fires only handle new chunks. Without the model, fires stay FTS-only until you run `recall model download`. Codex has no SessionEnd hook; run `recall index` manually for Codex sessions.

## How it works

```text
~/.claude/projects/**/*.jsonl  ─┐
                                ├─→ Parse → FTS5 + Q&A chunks → Index-time embedding
~/.codex/sessions/**/*.jsonl   ─┘
```

**Indexing** — `recall index` scans session directories, parses JSONL, builds a full-text index, generates Q&A chunks, and embeds new chunks. Incremental by default — it walks every session file but re-parses only those modified since they were last indexed.

**Searching** — `recall search` reads the pre-built index; it does not index. Run `recall index` to refresh first, or register the [Hook](#hook) to auto-index when a session ends. Searching an empty index prints `No sessions indexed. Run recall index first.`

**Embedding** — `recall index` embeds every new chunk (those without an embedding). Uses Ruri v3 (310M params) via mlx-rs with MLX acceleration on Apple Silicon. Batch inference (batch=128) with length-sorted padding minimization. Download the model once with `recall model download`; without it, index builds FTS5 only and search falls back to keyword ranking.

**Ranking** — When embeddings are available, search uses Reciprocal Rank Fusion (RRF) to blend FTS5 keyword scores with vector similarity. A recency boost favors newer sessions when scores are close.

## Architecture

```text
src/
├── main.rs       CLI subcommands (index, search, show, status)
├── parser/       JSONL parsers for Claude Code and Codex formats
├── indexer.rs    Incremental indexer with mtime tracking + chunk generation
├── search.rs     FTS5 + hybrid vector search with graceful degradation
├── hybrid.rs     RRF merge + recency boost
├── embedder.rs   Index-time embedding orchestration (batches chunks via rurico)
├── chunker.rs    Q&A pair chunker with size splitting + SHA256 change detection
├── db.rs         SQLite schema (WAL, FTS5, sqlite-vec)
└── date.rs       Civil calendar date utilities
```

Single binary. SQLite, mlx-rs, and sqlite-vec are statically linked.

## Performance

| Operation                       | Time                             |
| ------------------------------- | -------------------------------- |
| `recall index` (incremental)    | ~0.5s + embedding for new chunks |
| `recall index` (first run, 28k) | ~11 min (embedding-dominated)    |
| `recall rebuild` (28k)          | ~11 min (full re-embed)          |
| `recall search`                 | instant (read-only)              |
| Embedding throughput            | ~45 chunks/sec (M3 + MLX)        |
| Initial model download          | ~1.2 GB                          |

## Limitations

| Limitation          | Details                                                                     |
| ------------------- | --------------------------------------------------------------------------- |
| Local sessions only | Searches `~/.claude/projects/` and `~/.codex/sessions/`. No cloud sync      |
| Text only           | Images, tool results, and binary content are not indexed                    |
| Apple Silicon only  | Requires Apple Silicon. The MLX backend has no CPU/Linux fallback           |
| Excerpts in search  | Search results show excerpts. Use `recall show <id>` for full conversations |

## Exit Codes

recall uses sysexits-style exit codes instead of a generic `1`/`2` split.

| Code | Name           | Meaning                                      |
| ---- | -------------- | -------------------------------------------- |
| 0    | success        | Command completed successfully               |
| 64   | `USAGE_ERROR`  | Invalid command usage or missing local index |
| 65   | `DATA_ERROR`   | Malformed user input, such as a bad query    |
| 70   | `INTERNAL`     | Internal invariant or unsupported backend    |
| 74   | `IO_ERROR`     | Filesystem or SQLite I/O failure             |
| 75   | `TEMP_FAILURE` | Retryable transient failure                  |
| 104  | `UNKNOWN`      | Unclassified error path                      |

## Development

### Setup

Run once after cloning:

```sh
git config --local core.hooksPath .githooks
```

This installs a pre-commit hook that runs `cargo fmt --check` and `cargo clippy --all-targets --all-features -- -D warnings` before each commit. Violations abort the commit. To skip for one commit: `git commit --no-verify`.

### Common commands

```sh
cargo nextest run                                         # all tests (install: cargo install cargo-nextest --locked)
cargo clippy --all-targets --all-features -- -D warnings  # lint (matches CI)
cargo fmt -- --check                                      # format check
```

## Acknowledgements

This project was inspired by [arjunkmrm/recall](https://github.com/arjunkmrm/recall). The original idea of making past Claude Code sessions searchable came from there. This is a Rust reimplementation with semantic search — single binary, local embeddings, and CJK support.

## License

MIT
