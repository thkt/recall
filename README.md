**English** | [日本語](README.ja.md)

# recall

Your past Claude Code and Codex sessions, searchable. Gets smarter every time you search.

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

# Index your sessions (downloads embedding model on first run)
recall index

# Search
recall search "authentication"
```

## How search gets smarter

recall starts with keyword search (FTS5). Each time you search, it quietly embeds nearby chunks using a local AI model. Over time, this enables **semantic search** — finding sessions by meaning, not just keywords.

```
First search:   FTS5 keyword match only
After search:   Result sessions get embedded (10 nearby + 10 recent)
Next search:    Hybrid ranking (FTS5 + vector similarity via RRF)
Over time:      More sessions embedded → better semantic results
```

No API keys. No data leaves your machine. The embedding model (Ruri v3) runs locally via MLX on Apple Silicon.

## Usage

### Search

```sh
recall search "error handling"                                   # keyword search
recall search "database migration" --project /Users/me/GitHub/app  # filter by project
recall search "React Router" --days 7                            # last 7 days
recall search "async runtime" --source codex                     # Codex sessions only
recall search "auth AND middleware"                               # boolean operators
recall search "auth" --no-embed                                  # skip post-search embedding
```

Backward compatible: `recall "query"` works as shorthand for `recall search "query"`.

| Flag         | Description                               |
| ------------ | ----------------------------------------- |
| `--project`  | Filter by project path (prefix match)     |
| `--days`     | Only sessions from the last N days        |
| `--source`   | `claude` or `codex`                       |
| `--limit`    | Max results, 1-100 (default: 10)          |
| `--no-embed` | Skip post-search embedding                |
| `-v`         | Verbose output                            |

Supports [FTS5 query syntax](https://www.sqlite.org/fts5.html#full_text_query_syntax) — bare words, `"quoted phrases"`, and `AND` / `OR` / `NOT`.

### Index

```sh
recall index            # parse + chunk session logs (no model calls)
recall index --force    # full rebuild
```

### Embed

```sh
recall embed            # embed pending chunks (requires model)
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

## How it works

```text
~/.claude/projects/**/*.jsonl  ─┐
                                ├─→ Parse → FTS5 + Q&A chunks → Progressive embedding
~/.codex/sessions/**/*.jsonl   ─┘
```

**Indexing** — `recall index` scans session directories, parses JSONL, builds a full-text index, and generates Q&A chunks. Incremental by default — only changed files are re-indexed. Directory mtime checking skips the scan entirely when nothing changed.

**Embedding** — Each search embeds 20 chunks: 10 from search result sessions + 10 most recent. Uses Ruri v3 (310M params) via mlx-rs with MLX acceleration on Apple Silicon. Batch inference (batch=32) with length-sorted padding minimization. Download the model explicitly with `recall model download`; without it, search falls back to FTS5 keyword ranking only.

**Ranking** — When embeddings are available, search uses Reciprocal Rank Fusion (RRF) to blend FTS5 keyword scores with vector similarity. A recency boost favors newer sessions when scores are close.

## Architecture

```text
src/
├── main.rs       CLI subcommands (index, search, show, status)
├── parser/       JSONL parsers for Claude Code and Codex formats
├── indexer.rs    Incremental indexer with mtime tracking + chunk generation
├── search.rs     FTS5 + hybrid vector search with graceful degradation
├── hybrid.rs     RRF merge + recency boost
├── modernbert.rs ModernBERT model implementation (mlx-rs)
├── embedder.rs   Ruri v3 embedder (mlx-rs default / candle fallback, mean pooling)
├── chunker.rs    Q&A pair chunker with size splitting + SHA256 change detection
├── db.rs         SQLite schema (WAL, FTS5, sqlite-vec)
└── date.rs       Civil calendar date utilities
```

Single binary. SQLite, mlx-rs, and sqlite-vec are statically linked.

## Performance

| Operation                           | Time                                       |
| ----------------------------------- | ------------------------------------------ |
| `recall index` (6k files)           | ~0.5s (incremental), ~4min (full rebuild)  |
| `recall search`                     | ~2s (with embedding), instant (--no-embed) |
| `recall embed` (28k chunks)         | ~11 min (M3, batch=32)                     |
| Embedding throughput                | ~45 chunks/sec (M3 + MLX)                  |
| Initial model download              | ~1.2 GB                                    |

## Limitations

| Limitation          | Details                                                                   |
| ------------------- | ------------------------------------------------------------------------- |
| Local sessions only | Searches `~/.claude/projects/` and `~/.codex/sessions/`. No cloud sync   |
| Text only           | Images, tool results, and binary content are not indexed                  |
| Apple Silicon focus | MLX acceleration requires Apple Silicon. candle (CPU) fallback available for Linux |
| Excerpts in search  | Search results show excerpts. Use `recall show <id>` for full conversations  |

## Acknowledgements

This project was inspired by [arjunkmrm/recall](https://github.com/arjunkmrm/recall). The original idea of making past Claude Code sessions searchable came from there. This is a Rust reimplementation with semantic search — single binary, local embeddings, and CJK support.

## License

MIT
