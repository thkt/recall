# recall

Your past Claude Code and Codex sessions, searchable.

## The problem

You solved an authentication problem with Claude last week. Now you need that approach again — but which project was it? Which session?

**Without recall:**

```
find ~/.claude -name "*.jsonl" | wc -l
  3,851

grep -r "authentication" ~/.claude/projects/ | head
  {"type":"user","message":{"role":"user","content":[{"type":"text","text":"I need to
  {"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"H
  {"type":"user","message":{"role":"user","content":[{"type":"text","text":"Now let's a
  ...12,000+ lines of raw JSONL
```

**With recall:**

```sh
recall "authentication"

[1] 2026-02-27 | stateful-sleeping-cosmos | kagami [claude]
    /Users/me/GitHub/kagami
    > Enable API keys Allow users and/or organizations to **authenticate**
      with your API programmatically...

[2] 2026-02-08 | fluffy-rolling-lampson | kai [claude]
    /Users/me/GitHub/kai/main
    > **authenticator** / isAuthenticated path: ...NID OAuth2 + PKCE
      **authentication** with nonce...

[3] 2026-02-08 | frolicking-juggling-storm | kizalas [claude]
    /Users/me/GitHub/kizalas/dev
    > ...全 57 エンドポイント を 15 グループに分類 — **Authentication** (4),
      Administrator (5), Tenant (5)...
```

3,851 sessions, 52,000 messages — searched in under a second.

## When to use recall (and when not to)

**Use recall when:**

- "How did I set up ESLint in that project?" — you remember the topic, not the session
- "Show me everything about database migrations from the last week" — time + topic filtering
- "What did Claude suggest for error handling in myapp?" — project-scoped search

**Use grep when:**

- You know the exact file path
- You need regex matching over raw JSONL

recall doesn't replace grep. It covers the case grep can't: finding sessions by what you talked about.

## Setup

### Install

Build from source (requires Rust 1.85+):

```sh
cargo install --path .
```

No API keys. No configuration. No runtime dependencies. Just a 2.4 MB binary.

### Claude Code integration

Add to your project's `CLAUDE.md`:

```markdown
## Tools

- `recall "query"` — search past sessions
- `recall "query" --project /path/to/project` — filter by project
- `recall "query" --days 7` — only recent sessions
- `recall "query" --source codex` — filter by source
```

## Usage

```sh
recall "error handling"                                    # basic search
recall "database migration" --project /Users/me/GitHub/app # filter by project
recall "React Router" --days 7                             # last 7 days only
recall "async runtime" --source codex                      # Codex sessions only
recall "auth AND middleware"                                # boolean operators
```

Supports [FTS5 query syntax](https://www.sqlite.org/fts5.html#full_text_query_syntax) — bare words, `"quoted phrases"`, and `AND` / `OR` / `NOT`.

| Flag        | Description                                 |
| ----------- | ------------------------------------------- |
| `--project` | Filter by project path (prefix match)       |
| `--days`    | Only sessions from the last N days          |
| `--source`  | `claude` or `codex`                         |
| `--limit`   | Max results, 1–100 (default: 10)            |
| `--reindex` | Force full rebuild of the index             |
| `-v`        | Verbose output (index stats, skipped files) |

## How it works

```
~/.claude/projects/**/*.jsonl  ─┐
                                ├─→ Parse → SQLite FTS5 index → Search
~/.codex/sessions/**/*.jsonl   ─┘
```

**Indexing** — First run scans `~/.claude/projects/` and `~/.codex/sessions/`, parses JSONL files, and builds a full-text index at `~/.recall.db`. Subsequent runs are incremental — only changed files are re-indexed.

**Parsing** — Claude Code and Codex JSONL formats are both handled. Tool-use blocks and system prompts are filtered out — only user/assistant text is indexed.

**Ranking** — BM25 relevance with a recency boost. Recent sessions rank higher when relevance scores are close.

## Architecture

```
src/
├── main.rs       CLI entry point, display formatting
├── parser.rs     JSONL parsers for Claude Code and Codex formats
├── indexer.rs    Incremental session indexer with mtime tracking
├── search.rs     FTS5 search with BM25 + recency boost
├── db.rs         SQLite schema (WAL mode, FTS5 with Porter stemmer)
└── date.rs       Civil calendar date utilities (Howard Hinnant algorithm)
```

Single binary, zero runtime dependencies. SQLite is statically linked.

## Limitations

- **Local sessions only** — Searches `~/.claude/projects/` and `~/.codex/sessions/`. No cloud sync.
- **Text only** — Images, tool results, and binary content are not indexed.
- **Unix DB permissions** — The database file is created with `0600` permissions on Unix. Other platforms use default permissions.
- **No export** — Search results show excerpts. To view full sessions, open the JSONL file directly.

## Acknowledgements

This project was inspired by [arjunkmrm/recall](https://github.com/arjunkmrm/recall). The original idea of making past Claude Code sessions searchable came from there. This is a Rust reimplementation — single binary, no runtime dependencies, fast startup.

## License

MIT
