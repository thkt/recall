# recall

Search past Claude Code and Codex sessions — instantly, from the terminal.

## The problem

You had a great conversation with Claude about authentication middleware last week. Or was it two weeks ago? Which project was it?

**Without recall:**

```
ls ~/.claude/projects/
  # 47 directories, each named like "-Users-me-GitHub-project"

find ~/.claude -name "*.jsonl" | wc -l
  # 3,851 files

grep -r "authentication" ~/.claude/projects/ | head
  # wall of raw JSONL... 12,000+ lines across hundreds of files
```

Your AI coding history is scattered across thousands of JSONL files with no search.

**With recall:**

```sh
recall "authentication middleware"

Found 3 sessions (index: 3,851 sessions, 52,288 messages):

[1] 2026-02-27 | stateful-sleeping-cosmos | kagami [claude]
    /Users/me/GitHub/kagami
    > Enable API keys Allow users and/or organizations to **authenticate**
      with your API programmatically...

[2] 2026-02-08 | fluffy-rolling-lampson | main [claude]
    /Users/me/GitHub/kai/main
    > **authenticator** / isAuthenticated path: ...NID OAuth2 + PKCE
      **authentication** with nonce...
```

One command. Full-text search across every session, ranked by relevance and recency. Your coding history becomes searchable knowledge.

## When to use recall (and when not to)

**Use recall when:**

- You remember discussing something but not when or where — "that database migration approach"
- You want to reuse a solution from a previous project — "how did I set up ESLint last time?"
- You need to find a specific session to continue work — filter by project, time range, or source

**Use grep when:**

- You know the exact file — `cat ~/.claude/projects/.../session.jsonl`
- You need regex matching over raw JSONL

recall doesn't replace grep. It covers the case grep can't: searching by concept across thousands of session files.

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

### Basic search

```sh
recall "error handling"
```

Uses [FTS5 query syntax](https://www.sqlite.org/fts5.html#full_text_query_syntax) — bare words, `"quoted phrases"`, and `AND` / `OR` / `NOT` operators.

### Filter by project

```sh
recall "database migration" --project /Users/me/GitHub/myapp
```

Prefix match — `/Users/me/GitHub` matches all projects under that path.

### Filter by time

```sh
recall "React Router" --days 7
```

### Filter by source

```sh
recall "async runtime" --source codex
```

`claude` or `codex`.

### Options

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
                                ├─→ JSONL parser → SQLite FTS5 index → BM25 + recency search
~/.codex/sessions/**/*.jsonl   ─┘
```

**Indexing** — On first run, recall scans `~/.claude/projects/` and `~/.codex/sessions/` for JSONL session files, parses them, and builds a full-text index at `~/.recall.db`. Subsequent runs are incremental — only files with changed mtime are re-indexed.

**Parsing** — Two JSONL formats are handled:

- **Claude Code**: `type` / `message.content` / `cwd` / `timestamp` entries
- **Codex**: `session_meta` / `response_item` entries + legacy format fallback

Tool-use blocks, system prompts, and boilerplate are filtered out. Only human-readable user/assistant text is indexed.

**Search** — Three-pass query:

1. FTS5 MATCH with BM25 ranking, grouped by session
2. Batch metadata lookup (project, source, timestamp)
3. Per-session snippet extraction with keyword highlighting (`**bold**`)

Results are re-ranked with an exponential recency boost (30-day half-life, 20% weight) — recent sessions surface higher when relevance scores are close.

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

## License

MIT
