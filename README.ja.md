[English](README.md) | **日本語**

# recall

過去のClaude Code / Codexセッションを検索するCLIツール。使うほど賢くなります。

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

# インデックス作成（初回はモデルを自動ダウンロード）
recall index

# 検索
recall search "authentication"
```

## 検索が賢くなる仕組み

recallはキーワード検索（FTS5）からスタートします。検索するたびに、ローカルのAIモデルで結果周辺のチャンクをembeddingします。これにより、キーワードが一致しなくても意味的に近いセッションを見つける**セマンティック検索**が徐々に有効になります。

```
初回検索:     FTS5 キーワードマッチのみ
検索後:       結果セッション10件 + 最新10件を embedding
次の検索:     ハイブリッドランキング（FTS5 + ベクトル類似度, RRF統合）
使い続ける:   embedding カバレッジ ↑ → セマンティック検索の精度 ↑
```

APIキー不要。データはマシンの外に出ません。embeddingモデル（Ruri v3）はApple Silicon上のMLXでローカル実行します。

## 使い方

### 検索

```sh
recall search "error handling"                                     # キーワード検索
recall search "database migration" --project /Users/me/GitHub/app  # プロジェクト絞り込み
recall search "React Router" --days 7                              # 直近7日間
recall search "async runtime" --source codex                       # Codexセッションのみ
recall search "auth AND middleware"                                 # ブール演算子
recall search "auth" --no-embed                                    # post-search embedding スキップ
```

後方互換: `recall "query"` は `recall search "query"` のショートハンドとして動作します。

| フラグ       | 説明                                   |
| ------------ | -------------------------------------- |
| `--project`  | プロジェクトパスで絞り込み（前方一致） |
| `--days`     | 直近N日間のセッションのみ              |
| `--source`   | `claude` または `codex`                |
| `--limit`    | 最大件数、1-100（デフォルト: 10）      |
| `--no-embed` | post-search embedding をスキップ       |
| `-v`         | 詳細出力                               |

[FTS5クエリ構文](https://www.sqlite.org/fts5.html#full_text_query_syntax)に対応。単語、`"フレーズ検索"`、`AND` / `OR` / `NOT` が使えます。

### インデックス

```sh
recall index            # 新規セッションログを解析 + チャンク化（差分、モデル呼び出しなし）
recall rebuild          # インデックスを破棄して全件再構築
```

### Embedding

```sh
recall embed            # 未embeddingのチャンクを一括embedding（モデル必須）
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

### Hook

`recall index` を Claude Code の SessionEnd hook に登録すると、セッション終了時にインデックスが更新されます。発火のたびにセッションツリー全体を再スキャンします（差分更新 — 変更されたファイルのみ再解析）。embedding は別途 `recall embed` で実行します。

`~/.claude/settings.json` に追加:

```json
{
  "hooks": {
    "SessionEnd": [{ "matcher": ".*", "hooks": [{ "type": "command", "command": "recall index" }] }]
  }
}
```

`recall index` はソースを環境変数から読み、hook の stdin payload は無視するため、追加の配線は不要です。Codex には SessionEnd hook がないため、Codex セッションは `recall index` を手動実行してください。

## 仕組み

```text
~/.claude/projects/**/*.jsonl  ─┐
                                ├─→ Parse → FTS5 + Q&Aチャンク → Progressive embedding
~/.codex/sessions/**/*.jsonl   ─┘
```

**インデックス** — `recall index` でセッションディレクトリをスキャン、JSONLを解析し、全文検索インデックスとQ&Aチャンクを構築します。差分更新で、全セッションファイルを走査しつつ、前回インデックス以降に更新されたファイルのみを再解析します。

**検索** — `recall search` は構築済みインデックスを読むだけで、インデックスは作成しません。事前に `recall index` でリフレッシュするか、[Hook](#hook) を登録してセッション終了時に自動インデックスしてください。空のインデックスを検索すると `No sessions indexed. Run recall index first.` を表示します。

**Embedding** — 検索のたびに20チャンクをembed: 結果セッション10件 + 最新10件。Ruri v3（310Mパラメータ）をmlx-rs + MLXでApple Silicon上で実行します。バッチ推論（batch=128）と長さソートによるpadding最小化。モデルは `recall model download` で明示的にダウンロードします。未ダウンロード時はFTS5のキーワード検索のみで動作します。

**ランキング** — embeddingがあればReciprocal Rank Fusion (RRF) でFTS5キーワードスコアとベクトル類似度を統合します。スコアが近い場合は新しいセッションにrecency boostがかかります。

## アーキテクチャ

```text
src/
├── main.rs       CLIサブコマンド（index, search, show, status）
├── parser/       Claude Code / Codex の JSONL パーサー
├── indexer.rs    mtime追跡によるインクリメンタルインデクサー + チャンク生成
├── search.rs     FTS5 + ハイブリッドベクトル検索（graceful degradation）
├── hybrid.rs     RRF 統合 + recency boost
├── embedder.rs   検索後embeddingのオーケストレーション（チャンクをruricoでバッチ処理）
├── chunker.rs    Q&Aペアチャンカー（サイズ分割 + SHA256変更検知）
├── db.rs         SQLite スキーマ（WAL, FTS5, sqlite-vec）
└── date.rs       日付ユーティリティ
```

シングルバイナリ。SQLite、mlx-rs、sqlite-vecは静的リンクしています。

## パフォーマンス

| 操作                           | 所要時間                                   |
| ------------------------------ | ------------------------------------------ |
| `recall index`（6kファイル）   | 約0.5秒（差分）                            |
| `recall rebuild`（6kファイル） | 約4分（全件再構築）                        |
| `recall search`                | 約2秒（embedding込み）、即座（--no-embed） |
| `recall embed`（28kチャンク）  | 約11分（M3, batch=128）                    |
| embeddingスループット          | 約45 chunks/sec（M3 + MLX）                |
| 初回モデルダウンロード         | 約1.2 GB                                   |

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
| 75     | `TEMP_FAILURE` | retry可能な一時失敗                       |
| 104    | `UNKNOWN`      | 未分類のエラー経路                        |

## 謝辞

[arjunkmrm/recall](https://github.com/arjunkmrm/recall) のアイデアをもとにRustで書き直しました。セマンティック検索、シングルバイナリ、ローカルembedding、CJK対応。

## ライセンス

MIT
