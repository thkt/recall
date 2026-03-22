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

APIキー不要。データはマシンの外に出ません。embeddingモデル（Ruri v3）はApple Silicon上のCoreMlでローカル実行します。

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

| フラグ       | 説明                                     |
| ------------ | ---------------------------------------- |
| `--project`  | プロジェクトパスで絞り込み（前方一致）   |
| `--days`     | 直近N日間のセッションのみ                |
| `--source`   | `claude` または `codex`                  |
| `--limit`    | 最大件数、1-100（デフォルト: 10）        |
| `--no-embed` | post-search embedding をスキップ         |
| `-v`         | 詳細出力                                 |

[FTS5クエリ構文](https://www.sqlite.org/fts5.html#full_text_query_syntax)に対応。単語、`"フレーズ検索"`、`AND` / `OR` / `NOT` が使えます。

### インデックス

```sh
recall index            # parse + chunk + モデルダウンロード
recall index --force    # 全件再構築
recall index --embed    # 全チャンクを一括 embedding（初回は時間がかかります）
```

### ステータス

```sh
recall status           # セッション数、チャンク数、embeddingカバレッジ、モデル状態
```

## 仕組み

```text
~/.claude/projects/**/*.jsonl  ─┐
                                ├─→ Parse → FTS5 + Q&Aチャンク → Progressive embedding
~/.codex/sessions/**/*.jsonl   ─┘
```

**インデックス** — `recall index` でセッションディレクトリをスキャン、JSONLを解析し、全文検索インデックスとQ&Aチャンクを構築します。差分更新で、変更ファイルのみ再インデックスします。ディレクトリのmtimeチェックにより、変更がなければスキャン自体をスキップします。

**Embedding** — 検索のたびに20チャンクをembed: 結果セッション10件 + 最新10件。Ruri v3（310Mパラメータ）をONNX Runtime + CoreMLで実行します（M3で約100ms/チャンク）。モデルは初回 `recall index` 時に自動ダウンロードします。

**ランキング** — embeddingがあればReciprocal Rank Fusion (RRF) でFTS5キーワードスコアとベクトル類似度を統合します。スコアが近い場合は新しいセッションにrecency boostがかかります。

## アーキテクチャ

```text
src/
├── main.rs       CLIサブコマンド（index, search, status）
├── parser/       Claude Code / Codex の JSONL パーサー
├── indexer.rs    mtime追跡によるインクリメンタルインデクサー + チャンク生成
├── search.rs     FTS5 + ハイブリッドベクトル検索（graceful degradation）
├── hybrid.rs     RRF 統合 + recency boost
├── embedder.rs   ort + Ruri v3 ONNX embedder（CoreML EP, mean pooling）
├── chunker.rs    Q&Aペアチャンカー（SHA256変更検知）
├── db.rs         SQLite スキーマ（WAL, FTS5, sqlite-vec）
└── date.rs       日付ユーティリティ
```

シングルバイナリ。SQLite、ONNX Runtime、sqlite-vecは静的リンクしています。

## パフォーマンス

| 操作                       | 所要時間                                   |
| -------------------------- | ------------------------------------------ |
| `recall index`（6kファイル）| 約0.5秒（差分）、約4分（全件再構築）       |
| `recall search`            | 約2秒（embedding込み）、即座（--no-embed） |
| 1チャンクのembedding       | 約100ms（M3 + CoreML）                     |
| 初回モデルダウンロード     | 約30秒（1.3 GB）                           |

## 制限事項

- `~/.claude/projects/` と `~/.codex/sessions/` のローカルセッションのみ対応。クラウド同期なし
- 画像、ツール結果、バイナリコンテンツはインデックスしない
- CoreMLアクセラレーションはApple Siliconが必要。CPUフォールバックは低速
- 検索結果は抜粋表示。完全なセッションを見るにはJSONLファイルを直接開く

## 謝辞

[arjunkmrm/recall](https://github.com/arjunkmrm/recall) のアイデアをもとにRustで書き直しました。セマンティック検索、シングルバイナリ、ローカルembedding、CJK対応。

## ライセンス

MIT
