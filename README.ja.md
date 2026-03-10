[English](README.md) | **日本語**

# recall

過去のClaude Code / Codexセッションを全文検索できるCLIツールです。

## 概要

Claude CodeやCodexのセッションは `~/.claude/projects/` や `~/.codex/sessions/` にJSONLとして保存される。数が増えるとgrepでは探しきれない。

```sh
find ~/.claude -name "*.jsonl" | wc -l
  3,851

grep -r "authentication" ~/.claude/projects/ | head
  {"type":"user","message":{"role":"user","content":[{"type":"text","text":"I need to
  {"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"H
  {"type":"user","message":{"role":"user","content":[{"type":"text","text":"Now let's a
  ...12,000+ lines of raw JSONL
```

recallはJSONLをSQLite FTS5でインデックス化し、話した内容からセッションを探せるようにします。

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

3,851セッション、52,000メッセージでも1秒以内で検索できます。

## 使い分け

recallは以下のケースに向いています。

- トピックは覚えているがセッションがわからない
- 期間とトピックで絞り込みたい
- 特定プロジェクトでの会話を探したい

grepは以下のケースに向いています。

- 正確なファイルパスがわかっている
- 生のJSONLに対して正規表現マッチしたい

recallはgrepの代替ではなく、grepではカバーできない「話した内容でセッションを探す」用途で使います。

## インストール

Homebrew (macOS):

```sh
brew install thkt/tap/recall
```

ソースからビルドする場合はRust 1.85+ が必要です。

```sh
cargo install --path .
```

APIキーや設定ファイルは不要です。2.4 MBのシングルバイナリで動作します。

## Claude Code 連携

プロジェクトの `CLAUDE.md` に以下を追加してください。

```markdown
## Tools

- `recall "query"` — 過去のセッションを検索
- `recall "query" --project /path/to/project` — プロジェクトで絞り込み
- `recall "query" --days 7` — 直近のセッションのみ
- `recall "query" --source codex` — ソースで絞り込み
```

## 使い方

```sh
recall "error handling"                                    # 基本検索
recall "database migration" --project /Users/me/GitHub/app # プロジェクト絞り込み
recall "React Router" --days 7                             # 直近 7 日間のみ
recall "async runtime" --source codex                      # Codex セッションのみ
recall "auth AND middleware"                                # ブール演算子
```

[FTS5 クエリ構文](https://www.sqlite.org/fts5.html#full_text_query_syntax)に対応しています。単語、`"フレーズ検索"`、`AND` / `OR` / `NOT` が使えます。

| フラグ      | 説明                                       |
| ----------- | ------------------------------------------ |
| `--project` | プロジェクトパスで絞り込み（前方一致）     |
| `--days`    | 直近 N 日間のセッションのみ                |
| `--source`  | `claude` または `codex`                    |
| `--limit`   | 最大件数、1〜100（デフォルト: 10）         |
| `--reindex` | インデックスを強制再構築                   |
| `-v`        | 詳細出力（インデックス統計、スキップ情報） |

## 仕組み

```text
~/.claude/projects/**/*.jsonl  ─┐
                                ├─→ Parse → SQLite FTS5 index → Search
~/.codex/sessions/**/*.jsonl   ─┘
```

初回実行時に `~/.claude/projects/` と `~/.codex/sessions/` をスキャンし、JSONLを解析して `~/.recall.db` に全文検索インデックスを構築します。以降は変更されたファイルのみ再インデックスします。

Claude CodeとCodexの両方のJSONLフォーマットに対応しています。tool_useブロックやシステムプロンプトは除外し、ユーザーとアシスタントのテキストだけをインデックスします。

検索結果はBM25関連度スコアにrecency boostを加算してランキングします。スコアが近い場合は新しいセッションが上位に表示されます。

## アーキテクチャ

```text
src/
├── main.rs       CLI エントリポイント、表示フォーマット
├── parser.rs     Claude Code / Codex の JSONL パーサー
├── indexer.rs    mtime 追跡によるインクリメンタルインデクサー
├── search.rs     FTS5 検索（BM25 + recency boost）
├── db.rs         SQLite スキーマ（WAL モード、FTS5 + Porter stemmer）
└── date.rs       日付ユーティリティ（Howard Hinnant アルゴリズム）
```

シングルバイナリで、ランタイム依存はありません。SQLiteは静的リンクしています。

## 制限事項

- `~/.claude/projects/` と `~/.codex/sessions/` のローカルセッションのみ検索する。クラウド同期には対応していない。
- 画像、ツール結果、バイナリコンテンツはインデックスしない。
- Unixではデータベースファイルを `0600` パーミッションで作成する。他のプラットフォームではデフォルトのパーミッションである。
- 検索結果は抜粋表示である。完全なセッションを見るにはJSONLファイルを直接開く。

## 謝辞

[arjunkmrm/recall](https://github.com/arjunkmrm/recall) のアイデアをもとにRustで書き直しました。シングルバイナリ、依存なし、高速起動、CJK（日本語・中国語・韓国語）検索に対応しています。

## ライセンス

MIT
