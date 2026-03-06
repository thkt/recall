# recall

過去の Claude Code / Codex セッションを全文検索。

[English](./README.md)

## 課題

先週 Claude で認証まわりの問題を解決した。同じアプローチをもう一度使いたい — でもどのプロジェクト？どのセッション？

**recall なし:**

```
find ~/.claude -name "*.jsonl" | wc -l
  3,851

grep -r "authentication" ~/.claude/projects/ | head
  {"type":"user","message":{"role":"user","content":[{"type":"text","text":"I need to
  {"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"H
  {"type":"user","message":{"role":"user","content":[{"type":"text","text":"Now let's a
  ...12,000+ lines of raw JSONL
```

**recall あり:**

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

3,851 セッション、52,000 メッセージ — 1 秒以内で検索完了。

## いつ使う？

**recall が向いているケース:**

- 「あのプロジェクトで ESLint どう設定したっけ？」 — トピックは覚えてるけどセッションがわからない
- 「先週のデータベースマイグレーション関連を全部見せて」 — 期間 + トピックで絞り込み
- 「myapp でエラーハンドリングについて Claude が何て言ってた？」 — プロジェクト指定検索

**grep が向いているケース:**

- 正確なファイルパスがわかっている
- 生の JSONL に対して正規表現マッチしたい

recall は grep の代替ではなく、grep ではカバーできない「話した内容でセッションを探す」ユースケースを埋めるもの。

## セットアップ

### インストール

ソースからビルド（Rust 1.85+ が必要）:

```sh
cargo install --path .
```

API キー不要。設定不要。ランタイム依存なし。2.4 MB のシングルバイナリ。

### Claude Code 連携

プロジェクトの `CLAUDE.md` に追加:

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

[FTS5 クエリ構文](https://www.sqlite.org/fts5.html#full_text_query_syntax)に対応 — 単語、`"フレーズ検索"`、`AND` / `OR` / `NOT`。

| フラグ      | 説明                                       |
| ----------- | ------------------------------------------ |
| `--project` | プロジェクトパスで絞り込み（前方一致）     |
| `--days`    | 直近 N 日間のセッションのみ                |
| `--source`  | `claude` または `codex`                    |
| `--limit`   | 最大件数、1〜100（デフォルト: 10）         |
| `--reindex` | インデックスを強制再構築                   |
| `-v`        | 詳細出力（インデックス統計、スキップ情報） |

## 仕組み

```
~/.claude/projects/**/*.jsonl  ─┐
                                ├─→ Parse → SQLite FTS5 index → Search
~/.codex/sessions/**/*.jsonl   ─┘
```

**インデックス作成** — 初回実行時に `~/.claude/projects/` と `~/.codex/sessions/` をスキャンし、JSONL ファイルを解析して `~/.recall.db` に全文検索インデックスを構築。以降はインクリメンタル更新 — 変更されたファイルのみ再インデックス。

**パース** — Claude Code と Codex の両 JSONL フォーマットに対応。tool_use ブロックやシステムプロンプトは除外し、ユーザー/アシスタントのテキストのみをインデックス。

**ランキング** — BM25 関連度スコアに recency boost を加算。スコアが近い場合、新しいセッションが上位に表示。

## アーキテクチャ

```
src/
├── main.rs       CLI エントリポイント、表示フォーマット
├── parser.rs     Claude Code / Codex の JSONL パーサー
├── indexer.rs    mtime 追跡によるインクリメンタルインデクサー
├── search.rs     FTS5 検索（BM25 + recency boost）
├── db.rs         SQLite スキーマ（WAL モード、FTS5 + Porter stemmer）
└── date.rs       日付ユーティリティ（Howard Hinnant アルゴリズム）
```

シングルバイナリ、ランタイム依存ゼロ。SQLite は静的リンク。

## 制限事項

- **ローカルセッションのみ** — `~/.claude/projects/` と `~/.codex/sessions/` を検索。クラウド同期なし。
- **テキストのみ** — 画像、ツール結果、バイナリコンテンツはインデックス対象外。
- **Unix DB パーミッション** — Unix ではデータベースファイルを `0600` で作成。他のプラットフォームではデフォルトのパーミッション。
- **エクスポートなし** — 検索結果は抜粋表示。完全なセッションを見るには JSONL ファイルを直接開く。

## 謝辞

このプロジェクトは [arjunkmrm/recall](https://github.com/arjunkmrm/recall) にインスパイアされて作られました。過去の Claude Code セッションを検索可能にするというアイデアはそこから生まれたものです。本プロジェクトは Rust による再実装で、シングルバイナリ・依存なし・高速起動を特徴としています。

## ライセンス

MIT
