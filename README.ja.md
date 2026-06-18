[English](README.md) | **日本語**

# recall

過去のClaude Code / Codexセッションをキーワード＋セマンティックで検索するCLIツール。完全ローカル、APIキー不要。

## 概要

Claude CodeやCodexのセッションは `~/.claude/projects/` や `~/.codex/sessions/` にJSONLとして保存されます。数が増えるとgrepでは探しきれません。

```sh
find ~/.claude -name "*.jsonl" | wc -l
  3,851

grep -r "authentication" ~/.claude/projects/ | head
  ...12,000+ lines of raw JSONL
```

recallはキーワード検索＋セマンティック検索で、話した内容からセッションを探します。

```sh
recall search "authentication"

[1] 2026-02-27 | stateful-sleeping-cosmos | kagami [claude]
    /Users/me/GitHub/kagami
    > Enable API keys Allow users and/or organizations to authenticate
      with your API programmatically...

[2] 2026-02-08 | fluffy-rolling-lampson | kai [claude]
    /Users/me/GitHub/kai/main
    > authenticator / isAuthenticated path: ...NID OAuth2 + PKCE
      authentication with nonce...
```

6,000+セッション、27,000+ Q&Aペアを2秒以内で検索できます。

## クイックスタート

```sh
# インストール
brew install thkt/tap/recall
# or: cargo install --path .

# モデルを一度ダウンロード（約1.2GB）してからインデックス作成
recall model download
recall index

# 検索
recall search "authentication"
```

## 検索の仕組み

recallは各セッションをキーワード検索（FTS5）にインデックスし、同じパスで各チャンクをローカルのAIモデルでembeddingします。検索は両者を統合し（**セマンティック検索** — キーワードが一致しなくても意味で見つかる）、読み取り専用なので即座に返ります。

```
recall index:   解析 + FTS5 + 新規チャンクを embedding（モデルがあれば）
recall search:  ハイブリッドランキング（FTS5 + ベクトル類似度, RRF）、即座
継続:           indexを重ねるほどセマンティック検索のカバレッジ向上
```

APIキー不要。データはマシンの外に出ません。embeddingモデル（Ruri v3）はApple Silicon上のMLXでローカル実行します。モデルがなくてもindexはFTS5を構築し、検索はキーワードランキングにフォールバックします。

## 使い方

### 検索

```sh
recall search "error handling"                                     # キーワード検索
recall search "database migration" --project /Users/me/GitHub/app  # プロジェクト絞り込み
recall search "React Router" --days 7                              # 直近7日間
recall search "async runtime" --source codex                       # Codexセッションのみ
recall search "auth AND middleware"                                 # ブール演算子
```

後方互換: `recall "query"` は `recall search "query"` のショートハンドとして動作します。

| フラグ                | 説明                                                                |
| --------------------- | ------------------------------------------------------------------- |
| `--project`           | プロジェクトパスで絞り込み（前方一致）                              |
| `--days`              | 直近N日間のセッションのみ                                           |
| `--source`            | `claude` または `codex`                                             |
| `--limit`             | 最大件数、1-100（デフォルト: 10）                                   |
| `--exclude-current`   | 呼び出し元セッションを除外（Claude Codeセッション内ではデフォルト） |
| `--include-current`   | セッション内でも呼び出し元セッションを含める                        |
| `--only-current`      | 呼び出し元セッションのみを返す                                      |
| `--include-automated` | automated（hook/script/agent）セッションを含める。デフォルトは除外  |
| `-v`                  | 詳細出力                                                            |

[FTS5クエリ構文](https://www.sqlite.org/fts5.html#full_text_query_syntax)に対応。単語、`"フレーズ検索"`、`AND` / `OR` / `NOT` が使えます。

### インデックス

```sh
recall index            # 新規セッションログを解析・チャンク化・embedding（差分）
recall rebuild          # 存在する全セッションを再解析・再embedding（読めないrootは既存rowを保持）
```

embedding にはモデルが必要です。`recall model download`（約1.2GB）で一度取得してください。モデルがない場合 `recall index` はFTS5のみ構築し、ダウンロードを促す note を出します。モデル導入後の次回 index が backlog を embedding します。

インデックスはデフォルトで `~/.local/share/recall/recall.db` に置かれます（`--db-path` または環境変数 `RECALL_DB` で上書き可能。親ディレクトリは初回実行時に作成します）。`~/.recall.db` に保存する旧ビルドからの移行時は、再 index の前に旧ファイルを移動してください。移動しないと recall は新パスに空のインデックスを作り直し、過去セッションが検索から不可視になります。

```sh
mkdir -p ~/.local/share/recall && mv ~/.recall.db ~/.local/share/recall/recall.db
```

### モデル

```sh
recall model download   # embeddingモデルをダウンロードして verify
```

### セッション表示

```sh
recall show abc-123     # セッションの会話全文を表示（ID前方一致）
```

### ステータス

```sh
recall status           # セッション数、チャンク数、embeddingカバレッジ、モデル状態
```

### 分類（Classify）

```sh
recall classify             # 未分類セッションを interactive/automated に分類
recall classify --all       # 全セッションを再分類
recall classify --dry-run   # 変更内容のみ表示し書き込まない
```

各セッションは最初のユーザー発話から interactive または automated に分類されます。automated（hook/script/agent生成）セッションは検索からデフォルトで除外され、`--include-automated` で含められます。

### Doctor

```sh
recall doctor           # 壊れたインデックスを診断する。報告のみで、修復はしない
```

4つの検査を実行します。SQLite `quick_check`、孤立した embedding、孤立したチャンク、モデルの load-and-embed プローブです。失敗した検査ごとに対処コマンド（`recall rebuild`、`recall model download`、または破損したDBを削除してからの再 `recall index`）を表示します。モデル未インストールは失敗ではなく info として報告します — モデルがなくても検索はFTS5のみで動くため、インデックスは healthy のままです。`--json` では、失敗時に `degraded: true` を立て、各対処を `notes` に列挙します。

### Hook

`recall index` が更新の基本手段です — 検索を最新にしたいときに実行してください。任意で Claude Code の SessionEnd hook に登録すると、セッション終了時に自動でインデックスが更新されます。発火のたびにセッションツリー全体を再スキャンし（差分更新 — 変更されたファイルのみ再解析）、新規チャンクをembeddingします。

`~/.claude/settings.json` に追加:

```json
{
  "hooks": {
    "SessionEnd": [{ "matcher": ".*", "hooks": [{ "type": "command", "command": "recall index" }] }]
  }
}
```

`recall index` はソースを環境変数から読み、hook の stdin payload は無視するため、追加の配線は不要です。モデルがあれば、初回の発火は一度きりの cold start で全バックログを embedding（28kチャンクで約11分）します。以降の発火は新規チャンクのみ処理します。モデルがなければ `recall model download` を実行するまで発火はFTS5のみです。Codex には SessionEnd hook がないため、Codex セッションは `recall index` を手動実行してください。

## 仕組み

```text
~/.claude/projects/**/*.jsonl  ─┐
                                ├─→ Parse → FTS5 + Q&Aチャンク → Index-time embedding
~/.codex/sessions/**/*.jsonl   ─┘
```

**インデックス** — `recall index` でセッションディレクトリをスキャン、JSONLを解析し、全文検索インデックスとQ&Aチャンクを構築し、新規チャンクをembeddingします。差分更新で、全セッションファイルを走査しつつ、前回インデックス以降に更新されたファイルのみを再解析します。

**検索** — `recall search` は構築済みインデックスを読むだけで、インデックスは作成しません。事前に `recall index` でリフレッシュするか、[Hook](#hook) を登録してセッション終了時に自動インデックスしてください。空のインデックスを検索すると `No sessions indexed. Run recall index first.` を表示します。

**Embedding** — `recall index` が新規チャンク（embedding未生成のもの）をすべてembeddingします。Ruri v3（310Mパラメータ）をmlx-rs + MLXでApple Silicon上で実行します。バッチ推論（batch=128）と長さソートによるpadding最小化。モデルは `recall model download` で一度ダウンロードします。未ダウンロード時はindexがFTS5のみ構築し、検索はキーワードランキングにフォールバックします。

**ランキング** — embeddingがあればReciprocal Rank Fusion (RRF) でFTS5キーワードスコアとベクトル類似度を統合します。スコアが近い場合は新しいセッションにrecency boostがかかります。

## アーキテクチャ

```text
src/
├── main.rs       CLIサブコマンド（index, search, show, status）
├── parser/       Claude Code / Codex の JSONL パーサー
├── indexer.rs    mtime追跡によるインクリメンタルインデクサー + チャンク生成
├── search.rs     FTS5 + ハイブリッドベクトル検索（graceful degradation）
├── hybrid.rs     RRF 統合 + recency boost
├── embedder.rs   index時embeddingのオーケストレーション（チャンクをruricoでバッチ処理）
├── chunker.rs    Q&Aペアチャンカー（サイズ分割 + SHA256変更検知）
├── db.rs         SQLite スキーマ（WAL, FTS5, sqlite-vec）
└── date.rs       日付ユーティリティ
```

シングルバイナリ。SQLite、mlx-rs、sqlite-vecは静的リンクしています。

## パフォーマンス

| 操作                        | 所要時間                            |
| --------------------------- | ----------------------------------- |
| `recall index`（差分）      | 約0.5秒 + 新規チャンク分のembedding |
| `recall index`（初回, 28k） | 約11分（embedding支配, M3 + MLX）   |
| `recall rebuild`（28k）     | 約11分（全件再embedding）           |
| `recall search`             | 即座（読み取り専用）                |
| embeddingスループット       | 約45 chunks/sec（M3 + MLX）         |
| 初回モデルダウンロード      | 約1.2 GB                            |

## 制限事項

- `~/.claude/projects/` と `~/.codex/sessions/` のローカルセッションのみ対応。クラウド同期なし
- 画像、ツール結果、バイナリコンテンツはインデックスしない
- Apple Siliconが必要。MLXバックエンドにCPU/Linuxフォールバックはなし
- 検索結果は抜粋表示。完全な会話は `recall show <id>` で表示可能

## 終了コード

recall は汎用的な `1` / `2` ではなく、sysexits 系の終了コードを返します。

| コード | 名前           | 意味                                      |
| ------ | -------------- | ----------------------------------------- |
| 0      | success        | コマンド成功                              |
| 64     | `USAGE_ERROR`  | コマンド指定ミス、またはローカルindexなし |
| 65     | `DATA_ERROR`   | 不正な検索クエリなど、ユーザー入力の不備  |
| 70     | `INTERNAL`     | 内部不変条件違反、または未対応backend     |
| 74     | `IO_ERROR`     | ファイルシステムまたはSQLite I/O失敗      |
| 75     | `TEMP_FAILURE` | リトライ可能な一時失敗                    |
| 104    | `UNKNOWN`      | 未分類のエラー経路                        |

## 謝辞

[arjunkmrm/recall](https://github.com/arjunkmrm/recall) のアイデアをもとにRustで書き直しました。セマンティック検索、シングルバイナリ、ローカルembedding、CJK対応。

## ライセンス

MIT
