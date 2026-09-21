# index の観測とホスト計測

`index` と `rebuild` は、長い処理に入る前に `index: <段階>: started` をstderrへ出す。TTY・非TTYで同じ段階名を使用する。単一のモデルロードや推論の内部進捗は取得しないため、その間は開始表示が現在の段階を示す。解析とFTS書込みはファイルごとに交互に実行する `parse_fts` 段階である。進捗だけの精密な総数取得を繰り返さず、列挙結果・選択結果・処理中のカウンターを利用する。

正常終了時の `--json` はstdoutに1個の既存envelopeを出し、`data.observations` に `seconds` と `counts` を追加する。人向けの段階・件数・最後の観測値はstderrへ出す。通常のエラー終了では完了していない段階を `unfinished` と表示し、得られた観測値をstderrへ残す。SIGKILLや既定のSIGTERMなど、Rustの後処理が走らない中断では最後のcommit通知までを確認済みの保存範囲とする。通知直前にcommitした内容が残る可能性もある。観測値に本文・パス・モデルの生エラー・認証情報を含めない。既存の解析診断のパス表示は別の契約である。

## 時間の定義

単位は単調時計による秒。未実行の段階はキーを省略する。ゼロと未実行を同一視しない。時間はCPU時間ではなく待機を含む実時間である。

| キー | 含む処理・重複 |
| --- | --- |
| `total` | DBオープンから処理結果作成前まで。下記を内包する。最終観測出力とJSON整形は含まない |
| `database_open` | DBオープン・schema移行 |
| `model_load_probe` | モデルロードと最小入力の推論probe。後の `inference` には含まない |
| `index_prepare` | 既存セッション取得、モデル不使用時の保護対象取得 |
| `enumeration` | ソースディレクトリの列挙。本文解析を含まない |
| `fts_transaction` | FTS更新のtransaction開始・解析・更新・削除・診断保存・commit。ロック待ちを含む |
| `parse_fts` | 上記内のファイル処理ループ。未commitの処理である |
| `parse` | 上記ループで実際に呼んだparserの累積時間。ファイル読込みを含む。metadata検査、DB更新、legacy backfillを含まない |
| `fts_excluding_parse` | `fts_transaction - parse`。FTSだけのCPU時間ではなく、metadata検査・診断・削除・待機・commit・計測/表示の費用も含む残余時間 |
| `fts_finalize` | transaction commit後のFTS optimize/automerge設定 |
| `chunking` | 未完了セッション選択、本文取得、チャンク作成、DB書込み、段階全体のcommit |
| `legacy_backfill` | 既存チャンクのrowidリンクと旧セッションの編集先パス再読込み、解析診断の取得/表示 |
| `pending_extraction` | 埋め込み未処理の一回の抽出。モデル利用時は本文も取得。利用不可時はIDだけで件数を取得 |
| `inference` | 全バッチの推論時間の累積。tokenize、分割、指定したforward pauseを含む |
| `embedding_db_save` | 全バッチの保存時間の累積。writer lock待ち、世代/本文照合、書込み、commitを含む |

`total`、`fts_transaction`、`parse_fts`、`parse`、`fts_excluding_parse`を足してはならない。FTS transactionの分解は `parse + fts_excluding_parse`。独立した段階だけを選んでも、その間のソート・集計・表示やrebuildの孤立vector修復などがあるため `total` との完全一致は要求しない。

## 件数と保存範囲

ファイル、セッション、チャンクは異なる単位であり、足し合わせない。同じIDの別ファイル置換もあるため、ファイル更新件数は新規セッション数ではない。

- `files_discovered`：列挙できたファイル数。未読のroot以下を含む全体数ではない。
- `files_updated` / `files_unchanged` / `files_empty` / `files_failed` / `files_deferred`：処理したファイルの相互排他的な分類。順に、DBへ取込み、mtime/size一致、セッションとして保存する本文のない新規ファイル、読込み失敗または取込み可能な本文のない破損ファイル、埋め込み保護のため取込み保留。既存セッションを正常な空本文に置換した場合は `updated`。部分的に読めた破損ファイルも取込みできれば `updated` であり、行単位の損失は既存 `parse_diagnostics` を見る。
- `files_remaining`：列挙済みで処理結果が未確定のファイル数。通常終了では上記5分類の和が `files_discovered`。SQL失敗で途中終了した場合には未処理分が残る。`files_updated_committed` はFTS transactionのcommit後だけ更新する。未commitの更新を保存済みと扱わない。
- `sessions_stored`：FTS処理後の保存済みセッション総数。root保護や以前のデータも含む。
- `sessions_chunk_pending`：今回のチャンク化対象セッション数。`sessions_chunked_committed` は段階全体のcommit済みセッション数、`sessions_chunked_empty` はそのうち0チャンクで正常完了した数、`chunks_created_committed` は作成したチャンク数。空完了は内数であり、足し合わせない。`sessions_chunk_remaining` はこの段階でまだcommitしていない対象数。途中表示の `processed` / `unprocessed` は作業量であり、commit前は `committed=0` のまま。
- `chunks_pending_snapshot`：抽出時点の未処理チャンク数。`chunks_saved` / `chunks_failed` / `chunks_stale` / `chunks_save_failed` / `chunks_unattempted` は、その集合の相互排他的な分類。順に、この実行でcommit、推論失敗、世代等が変わり保存を破棄、保存失敗したバッチ、まだ推論が完了していないもの。保存失敗時はバッチ全体を `save_failed` とし、そのバッチの世代差分は確定しない。モデル不使用では全件が `unattempted` となる。
- `chunks_remaining_snapshot = chunks_pending_snapshot - chunks_saved`。失敗・破棄・未試行を含む。並行するindexによる削除・再保存・新規追加を追跡する現在のDB総数ではない。再開時は新たに一回抽出し直す。バッチごとに累積件数をstderrへ通知する。

観測のためtransactionを分割しない。チャンクの途中失敗は段階全体をrollbackする。埋め込みの推論失敗は後続バッチを継続し、保存失敗はコマンドを失敗させるが、それより前のバッチcommitは残る。root未読や永続する解析診断の件数はこれらの今回処理数とは別に確認する。

## 4ケースの実モデル計測

macOS / Apple Silicon、Rust 1.96以上、Metal Toolchain、利用可能なモデルを用意したホストで実施する。再現手順を以下に示す。2026-09-21の結果は[実モデル・オーバーヘッド計測](../research/index-observability-measurements.md)を参照。[Issue #324](https://github.com/thkt/recall/issues/324) の受入条件と [ADR-0001](decisions/0001-freeze-the-json-output-envelope-as-a-stable-consumer-contract.md) の出力契約に従う。媒体のcaptureは不要。

```sh
python3 scripts/measure-index.py /absolute/path/to/recall /absolute/path/to/new-results \
  --model 'モデルID・revision・artifact識別値' \
  --conditions 'commit、チップ、RAM、OS、Rust/Metal版、電源/GPU負荷、cold/warm条件'
```

スクリプトは私的ログを参照しない合成512会話で初回、無変更、1会話への1往復追記を実行する。別の新規DBを途中のバッチcommit通知後にSIGTERMで止め、同じDBで再開する。入力・DB・stderr・JSONと各ケースの実時間を指定先へ残す。中断が間に合わなかった場合やモデルが使えない場合は失敗し、有効な4ケース測定として扱わない。必要なら会話数を増やし、新しい保存先でやり直す。合成負荷と実際の長い会話の費用は同一視しない。RSSはこのスクリプトでは測定しない。

各プロセスはモデルを再ロードする。初回DB作成をcold model / cold filesystemと呼ばない。キャッシュを意図的に制御した場合だけ条件と操作を記録し、制御できない場合は未制御と記す。token budget / pauseは既定値に固定している。異なる設定を評価する場合は、同じ入力と双方の条件を揃えて別の比較にする。

比較には各ケースのwall、段階時間、保存・失敗・保留・残数を記録する。段階時間と残数は `.json` の `data.observations` を使い、中断時は最後の通知と再開時の抽出を両方示す。本文や私的ログを貼らない。

計測オーバーヘッドは開始版 `ef814f3dbf1cf2897459e7e8ece1d9e413716ed7` と変更版を同じrelease条件でビルドし、同じ合成入力・モデル・出力先種別で比較する。開始版には `--baseline` を付ける（段階観測と中断通知がないため3ケースのwall比較だけを行う）。繰返し回数、実行順、各測定値とばらつきを残し、短い無変更ケースと推論を含むケースを分ける。旧版の全体時間を段階時間として流用しない。計測あり/なしの差だけでなく今回の進捗表示費用も含む比較であり、改善効果や数値SLOは測定前に断定しない。

実測は検証コマンドやCIの代替ではない。fmt / nextest ci / clippyと既存CIも対象版で確認する。調査版・比較基準・測定候補・引継ぎ版の対応と、証拠を再利用できる範囲は[実測記録](../research/index-observability-measurements.md)を参照する。
