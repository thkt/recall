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

After fully enumerating a source tree, `index` and `rebuild` remove from the index any sessions whose file paths are absent from that tree, together with their messages, chunks, embeddings, and edited-file records. This also applies when a deleted log is replaced by an empty or unparseable log at a different path and the file count stays the same. A missing root, directory or entry read failure, or depth limit prevents orphan cleanup for that source; sessions with an unknown source are also retained.

Malformed JSON and invalid UTF-8 lines are skipped while readable messages are indexed. Blank lines and valid excluded events (such as progress or tool events) are not errors. An unterminated final line with a JSON end-of-input error or a truncated UTF-8 sequence is reported separately as a possibly incomplete write; a valid final JSON record needs no newline. Other malformed lines, including newline-terminated truncated records, are reported as corruption.

Both `index` and `rebuild` report unresolved parsing loss with per-file counts and remedies in human warnings and in `--json` (`data.parse_diagnostics` and `notes`, with `degraded: true`). Repair corrupt source files, then run `recall index`; for a possible incomplete write, wait for the next append and run it again, or repair the file if the writer has stopped. For file read failures, check access and rerun. Read failures retain existing indexed data and remain retryable. These are successful partial runs (exit 0), and parsing notes coexist with model, embedding, and root-availability notes. Diagnostics contain no conversation excerpts or raw parser errors; displayed paths have control characters removed and are limited to 240 characters.

Diagnostics persist across runs, including mtime/size skips, deferred embedded updates, and unavailable roots. Counts describe the last recorded unresolved observation, not cumulative failures or a guarantee that every stored message is current. A clean ingestion read clears that file's diagnostics; a legacy path-only backfill can report loss but cannot clear it with a clean read of paths alone. Confirmed file deletion during a complete source scan removes its diagnostics once no indexed session retains that path. If a same-ID replacement at another path is deferred because the model is unavailable, the old body and its diagnostics remain until the replacement is ingested or the session itself is removed. An upgrade from an index without parsing diagnostics schedules one reread of existing sessions without deleting their data; embedded sessions still wait for a working model. Same-size repairs that preserve mtime require `recall rebuild` with a working model. A possibly incomplete tail does not prove that a writer is active.

Index and rebuild announce each stage on stderr in both terminals and redirected output, including model load/probe and pending extraction. Progress before a transaction commits is labeled uncommitted; embedding counts describe committed batches and distinguish inference failure, save failure, stale results, and unattempted work. `--json` adds numeric timings and counts under `data.observations`, preserving the outer envelope. Remaining embedding counts refer to the selected snapshot, not a continuously refreshed database total. See [measurement definitions and the host four-case procedure](docs/index-observability.md) for overlapping timers, empty completion, interruption, and comparison conditions.

Chunk generation records completion even for zero Q&A pairs, so unchanged sessions need no further body retrieval or chunking. When a completed session with chunks is re-parsed, index compares the newly derived chunks by exact content within that session. Matching chunks keep their IDs, generations and embeddings; source message rowid ranges and timestamps are updated with the messages in the same transaction. Appending an assistant response re-derives the final Q&A group. Only new or changed chunks need inference; a content-preserving mtime update keeps all embeddings. Truncation and replacement remove unmatched chunks and vectors, while retaining any exact matches, including duplicate content as separate occurrences.

An interrupted or failed message/chunk transaction rolls back together. After it commits, reusable vectors remain searchable and missing embeddings are pending; inference or save failures can be retried with `recall index`. Existing snapshot counts and notes identify unfinished embedding work. New sessions, previously empty sessions, and invalidated sessions use the batched chunk pass, which commits chunks and completion together. `rebuild` intentionally bypasses reuse. Reuse requires the same recorded pipeline version (parser, chunk rules and pinned embedding model); see [invalidation and measurement details](docs/index-observability.md#チャンク再利用の条件と追記計測). Model absence/probe failure still defers updates to embedded sessions.

When embedding is unavailable, updates to embedded sessions wait until it is available again. Their stored content, chunks, embeddings, and edited-file records are preserved, including when a parsed replacement containing messages at a different path uses the same session ID and the old file has been deleted. Such deferred updates are excluded from orphan cleanup for that run; deleted logs without such a replacement remain subject to the cleanup rules above.

Embedding needs the model: run `recall model download` (~1.2 GB) once. Without it, `recall index` builds FTS5 only and prints a note to download it; the next index after the model is present embeds the backlog.

Overlapping `index` or `rebuild` runs save an embedding only if the chunk still has the same content and generation inside the save transaction. Results for updated, deleted, or replaced chunks are discarded without deleting another run's current vectors or increasing the embedded count. Any replacement still missing an embedding is eligible on the next `recall index`. This protects new writes when all overlapping runs use this guard; it does not detect previously stored content/vector mismatches. With a working model and available source logs, `recall rebuild` regenerates those embeddings.

An otherwise compatible index without generation tracking remains readable by `search`, `status`, and `show`, without migration or a rebuild. Opening it for writing adds generation tracking while preserving existing chunks and embeddings.

Pending embedding bodies are loaded in pages of at most 1,024 chunks and 8 MiB, then released before the next page. A single chunk larger than 8 MiB is processed alone without truncation. The ID/generation/length worklist still grows with pending chunk count; model and other indexing memory are separate. Failed inference or save batches remain retryable while later batches continue; save errors are returned after processing the worklist. See [bounds and host comparison](docs/index-observability.md#未処理本文の有限ページと比較計測) for memory accounting and measurements.

`recall index` and `recall rebuild` accept two flags (also settable via env, useful for the [Hook](#hook)) that tune the embed pass:

```sh
recall index --token-budget 2048 --forward-pause-ms 700
# or
RECALL_TOKEN_BUDGET=2048 RECALL_FORWARD_PAUSE_MS=700 recall index
```

`--token-budget` (`RECALL_TOKEN_BUDGET`) overrides the forward-pass token budget: smaller values split the work into shorter GPU forwards, so interactive processes get GPU time between them. Values above 256000 clamp down to 256000 (never up), keeping every forward under the embedder's own OOM ceiling. `--forward-pause-ms` (`RECALL_FORWARD_PAUSE_MS`) sleeps after each GPU forward pass, handing the freed GPU time to the desktop. The two combine: a small budget alone still runs forwards back-to-back, and a pause alone still leaves each forward long (up to minutes at the default budget), so responsiveness during a big backlog embed needs both, for example `--token-budget 2048 --forward-pause-ms 700`. Both flags trade embed throughput for responsiveness; leaving them unset preserves the default full-speed behavior.

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

**Indexing** — `recall index` scans session directories, parses JSONL, builds a full-text index, generates Q&A chunks, and embeds new chunks. Incremental by default — it walks every session file and skips the body only when its stored size matches and its mtime differs by less than 1 ms. Size changes trigger a re-parse even when mtime is preserved. If metadata changes during parsing, the next index run retries the file. Older indexes without stored sizes are re-read once; embedded sessions remain pending while embedding is unavailable. Same-size replacements with mtime unchanged (or differing by less than 1 ms) are not detected; use `recall rebuild` with a working model to refresh them. Files are not fully hashed for freshness.

**Searching** — `recall search` reads the pre-built index; it does not index. Run `recall index` to refresh first, or register the [Hook](#hook) to auto-index when a session ends. Searching an empty index prints `No sessions indexed. Run recall index first.`

**Embedding** — `recall index` embeds every new chunk (those without an embedding). Uses Ruri v3 (310M params) via mlx-rs with MLX acceleration on Apple Silicon. Batch inference (batch=128) with length-sorted padding minimization. Download the model once with `recall model download`; without it, index builds FTS5 only and search falls back to keyword ranking.

**Ranking** — When embeddings are available, search uses Reciprocal Rank Fusion (RRF) to blend FTS5 keyword scores with vector similarity. A recency boost favors newer sessions when scores are close.

## Architecture

```text
src/
├── main.rs       CLI subcommands (index, search, show, status)
├── parser/       JSONL parsers for Claude Code and Codex formats
├── indexer.rs    Incremental indexer with mtime/size tracking + chunk generation
├── search.rs     FTS5 + hybrid vector search with graceful degradation
├── hybrid.rs     RRF merge + recency boost
├── embedder.rs   Index-time embedding orchestration (batches chunks via rurico)
├── chunker.rs    Q&A pair chunker with size splitting (exact-content reuse in indexer)
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
