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

`index` と `rebuild` は、sourceのディレクトリ全体を列挙できた場合に、見つからなくなったファイルパスの会話と、そのメッセージ・チャンク・embedding・記録済み編集対象パスを索引から削除します。別パスの空ログや解析できないログへの置換でファイル数が変わらない場合も同様です。ルート不在、ディレクトリやエントリの読取り失敗、深さ上限によって列挙が不完全なsourceは、この削除の対象にしません。未知のsourceの会話も保持します。

不正JSON・不正UTF-8の行を読み飛ばしても、正常に読めたメッセージは索引に取り込みます。空行や、progressなどの正常な対象外イベントはエラーにしません。改行で終わらない最終行がJSONの入力終端エラー、またはUTF-8の途中切れになる場合は、書込み途中の可能性として区別します。正常な最終JSON行に末尾改行は不要です。それ以外の不正行や、改行で終わる途中切れの行は破損として扱います。

`index` と `rebuild` は、未解消の解析欠落をファイルごとの件数と対処方法で伝えます。人向けには警告を出し、`--json` では `data.parse_diagnostics` と `notes` に載せて `degraded: true` にします。破損は元ファイルを修復して `recall index` を再実行してください。書込み途中の可能性がある場合は次の追記を待って再実行し、書込みが停止しているなら元ファイルを修復してください。ファイル読取り失敗はアクセス状態を確認して再実行します。読取り失敗では既存の索引データを保持し、再試行できます。これらは一部を取得できた成功（終了コード0）として扱い、モデル不在・埋め込み失敗・ルート不在などの理由も併記します。診断には会話本文や生のパーサーエラーを含めず、表示パスは制御文字を除去して240文字までに制限します。

診断は、mtime・サイズによる省略、埋め込み済み会話の更新延期、ルート不在の際も保持します。件数は最後に記録した未解消の状態であり、失敗の累積や全保存データの鮮度を示すものではありません。索引用の解析で欠落なく読めればそのファイルの診断を解除します。旧索引の編集対象パスを補完する処理でも欠落を記録しますが、パスだけの正常な再読取りでは既存の診断を解除しません。source全体を列挙してファイルの削除を確認し、そのパスの会話が索引にも残っていなければ診断を削除します。別パスの同一IDへの置換をモデル不在で延期した場合は、置換を取り込むか会話自体を削除するまで、旧本文とその診断を保持します。診断を記録しない旧索引からの移行では、既存データを削除せず、一度だけ再読取りを予約します。埋め込み済み会話は動作するモデルが利用可能になるまで待ちます。修復してもサイズとmtimeが同じなら、動作するモデルを用意して `recall rebuild` を実行してください。末尾の途中切れだけでは、書込みが継続中かどうかは断定できません。

チャンク生成はQ&Aが0件でも完了を記録し、変更のない会話の本文再取得・チャンク化を省きます。チャンクを持つ完了済み会話を再解析すると、同じ会話内で新旧チャンクの本文を完全一致で照合します。一致するチャンクのID・世代・embeddingを維持し、メッセージrowidの範囲とtimestampを本文更新と同じtransactionで更新します。assistantの追記では最後のQ&Aグループを作り直し、新規・変更チャンクだけを推論します。本文が同じmtime更新ではembeddingを維持します。切詰め・差し替えでは一致しないチャンクとベクトルを削除し、一致部分は重複する本文も別々の出現として保持します。

本文・チャンク更新の失敗や中断では同じtransactionをrollbackします。commit後は再利用したベクトルが検索可能なまま残り、embeddingがない部分は未処理になります。推論・保存失敗の後は `recall index` で再試行できます。残件は既存のsnapshot件数とnotesで確認します。新規・従来空・無効化された会話は、チャンクと完了状態を一緒にcommitするバッチ処理を使います。`rebuild` は再利用をせず意図的に再生成します。再利用には、パーサー・チャンク規則・固定モデルの処理版が一致する必要があります。[無効化条件と計測方法](docs/index-observability.md#チャンク再利用の条件と追記計測)も参照してください。モデル不在・probe失敗時は引き続き、embeddingを持つ会話の更新を保留します。

embeddingを持つ会話の更新は、embeddingが利用可能になるまで延期します。別パスの置換ファイルを解析してメッセージと同じセッションIDを確認できた場合は、旧ファイルが削除されていても、保存済み本文・チャンク・embedding・記録済み編集対象パスを保持します。こうした更新延期の対象は、その実行では上記の削除対象から除外します。この条件を満たす置換を確認できない削除済みログには、上記の削除条件が適用されます。

embedding にはモデルが必要です。`recall model download`（約1.2GB）で一度取得してください。モデルがない場合 `recall index` はFTS5のみ構築し、ダウンロードを促す note を出します。モデル導入後の次回 index が backlog を embedding します。

`index` や `rebuild` が同時に動く場合、保存 transaction 内でチャンクの本文と世代が推論前と一致した結果だけを保存します。更新・削除・置換されたチャンクの結果は破棄し、別の実行が保存した現在のベクトルを削除せず、埋め込み成功件数にも加算しません。置換後のチャンクが未処理なら、次の `recall index` で処理できます。この保護は、同時に動く全実行がこの照合を行う版である場合の新しい書込みに適用され、既に保存された本文とベクトルの不一致は検出しません。動作するモデルと読める元ログがあれば、`recall rebuild` で埋め込みを再生成できます。

世代管理がないだけで他の読取り条件を満たす索引は、移行や rebuild をせずに `search`・`status`・`show` で参照できます。書込み用に開く際に、既存のチャンクと埋め込みを保持して世代管理を追加します。

インデックスはデフォルトで `~/.local/share/recall/recall.db` に置かれます（`--db-path` または環境変数 `RECALL_DB` で上書き可能。親ディレクトリは初回実行時に作成します）。`~/.recall.db` に保存する旧ビルドからの移行時は、再 index の前に旧ファイルを移動してください。移動しないと recall は新パスに空のインデックスを作り直し、過去セッションが検索から不可視になります。

```sh
mkdir -p ~/.local/share/recall && mv ~/.recall.db ~/.local/share/recall/recall.db
```

indexとrebuildは、モデルロード/probe、未処理抽出を含む段階の開始をTTY・非TTYともstderrへ表示します。transaction途中の処理件数とcommit済み件数を区別し、埋め込みは保存済み・推論失敗・保存失敗・競合による破棄・未試行を確認できます。`--json` は外側のenvelopeを維持し、`data.observations` に時間と件数を追加します。埋め込み残数は抽出時の集合に対する値で、現在のDB全体を再走査した総数ではありません。[計測の定義とホストでの4ケース比較手順](docs/index-observability.md)に、時間の重複、空完了、中断、比較条件を記載しています。

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

**インデックス** — `recall index` でセッションディレクトリをスキャン、JSONLを解析し、全文検索インデックスとQ&Aチャンクを構築し、新規チャンクをembeddingします。差分更新では全セッションファイルを走査し、保存済みサイズが一致し、mtime差が1ms未満の場合だけ本文の再解析を省きます。mtimeを保持した追記・切詰めもサイズ差で検出します。読取り中に更新印が変わったファイルは次回のindexで再読取りします。サイズ未保存の旧インデックスは一度再読取りし、embeddingが利用できない間は既存embeddingを持つ会話の更新を保留します。同じサイズでmtimeも同じ（または差が1ms未満）の内容差し替えは検出できません。その場合は、動作するモデルを用意して `recall rebuild` で更新してください。更新判定のための全件ハッシュ計算は行いません。

**検索** — `recall search` は構築済みインデックスを読むだけで、インデックスは作成しません。事前に `recall index` でリフレッシュするか、[Hook](#hook) を登録してセッション終了時に自動インデックスしてください。空のインデックスを検索すると `No sessions indexed. Run recall index first.` を表示します。

**Embedding** — `recall index` が新規チャンク（embedding未生成のもの）をすべてembeddingします。Ruri v3（310Mパラメータ）をmlx-rs + MLXでApple Silicon上で実行します。バッチ推論（batch=128）と長さソートによるpadding最小化。モデルは `recall model download` で一度ダウンロードします。未ダウンロード時はindexがFTS5のみ構築し、検索はキーワードランキングにフォールバックします。

**ランキング** — embeddingがあればReciprocal Rank Fusion (RRF) でFTS5キーワードスコアとベクトル類似度を統合します。スコアが近い場合は新しいセッションにrecency boostがかかります。

## アーキテクチャ

```text
src/
├── main.rs       CLIサブコマンド（index, search, show, status）
├── parser/       Claude Code / Codex の JSONL パーサー
├── indexer.rs    mtime・サイズ追跡によるインクリメンタルインデクサー + チャンク生成
├── search.rs     FTS5 + ハイブリッドベクトル検索（graceful degradation）
├── hybrid.rs     RRF 統合 + recency boost
├── embedder.rs   index時embeddingのオーケストレーション（チャンクをruricoでバッチ処理）
├── chunker.rs    Q&Aペアチャンカー（サイズ分割。本文照合による再利用はindexer）
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
