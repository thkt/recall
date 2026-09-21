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
| `index_prepare` | writer lock取得後の既存セッション取得、モデル不使用時の保護対象取得。`fts_transaction` に含まれる |
| `enumeration` | writer lock取得後のソースディレクトリの列挙。`fts_transaction` に含まれ、本文解析を含まない |
| `fts_transaction` | FTS更新のtransaction開始・列挙・準備・解析・更新・チャンク照合・削除・診断保存・commit。ロック待ちを含む |
| `parse_fts` | 上記内のファイル処理ループ。未commitの処理である |
| `parse` | 上記ループで実際に呼んだparserの累積時間。ファイル読込みを含む。metadata検査、DB更新、legacy backfillを含まない |
| `fts_excluding_parse` | `fts_transaction - parse`。FTSだけのCPU時間ではなく、列挙・metadata検査・診断・削除・待機・commit・計測/表示の費用も含む残余時間 |
| `fts_finalize` | transaction commit後のFTS optimize/automerge設定 |
| `chunk_reconciliation` | 再利用対象の会話の旧チャンク取得・新チャンク生成・照合・rowid更新・不要vector削除・再利用件数取得。`parse_fts` と `fts_transaction` に含まれる。対象なしは0 |
| `chunking` | 未完了セッション選択、本文取得、チャンク作成、DB書込み、段階全体のcommit |
| `legacy_backfill` | 既存チャンクのrowidリンクと旧セッションの編集先パス再読込み、解析診断の取得/表示 |
| `pending_extraction` | 埋め込み未処理の一回の抽出。モデル利用時は本文も取得。利用不可時はIDだけで件数を取得 |
| `inference` | 全バッチの推論時間の累積。tokenize、分割、指定したforward pauseを含む |
| `embedding_db_save` | 全バッチの保存時間の累積。writer lock待ち、世代/本文照合、書込み、commitを含む |

`total`、`fts_transaction`、`enumeration`、`parse_fts`、`parse`、`fts_excluding_parse`を足してはならない。FTS transactionの分解は `parse + fts_excluding_parse`。独立した段階だけを選んでも、その間のソート・集計・表示やrebuildの孤立vector修復などがあるため `total` との完全一致は要求しない。

## 件数と保存範囲

ファイル、セッション、チャンクは異なる単位であり、足し合わせない。同じIDの別ファイル置換もあるため、ファイル更新件数は新規セッション数ではない。

- `files_discovered`：列挙できたファイル数。未読のroot以下を含む全体数ではない。
- `files_updated` / `files_unchanged` / `files_empty` / `files_failed` / `files_deferred`：処理したファイルの相互排他的な分類。順に、DBへ取込み、mtime/size一致、セッションとして保存する本文のない新規ファイル、読込み失敗または取込み可能な本文のない破損ファイル、埋め込み保護のため取込み保留。既存セッションを正常な空本文に置換した場合は `updated`。部分的に読めた破損ファイルも取込みできれば `updated` であり、行単位の損失は既存 `parse_diagnostics` を見る。
- `files_remaining`：列挙済みで処理結果が未確定のファイル数。通常終了では上記5分類の和が `files_discovered`。SQL失敗で途中終了した場合には未処理分が残る。`files_updated_committed` はFTS transactionのcommit後だけ更新する。未commitの更新を保存済みと扱わない。
- `sessions_stored`：FTS処理後の保存済みセッション総数。root保護や以前のデータも含む。
- `sessions_chunk_pending`：今回のチャンク化対象セッション数。`sessions_chunked_committed` は段階全体のcommit済みセッション数、`sessions_chunked_empty` はそのうち0チャンクで正常完了した数、`chunks_created_committed` は作成したチャンク数。空完了は内数であり、足し合わせない。`sessions_chunk_remaining` はこの段階でまだcommitしていない対象数。途中表示の `processed` / `unprocessed` は作業量であり、commit前は `committed=0` のまま。
- `chunks_pending_snapshot`：抽出時点の未処理チャンク数。`chunks_saved` / `chunks_failed` / `chunks_stale` / `chunks_save_failed` / `chunks_unattempted` は、その集合の相互排他的な分類。順に、この実行でcommit、推論失敗、世代等が変わり保存を破棄、保存失敗したバッチ、まだ推論が完了していないもの。保存失敗時はバッチ全体を `save_failed` とし、そのバッチの世代差分は確定しない。モデル不使用では全件が `unattempted` となる。
- `chunks_remaining_snapshot = chunks_pending_snapshot - chunks_saved`。失敗・破棄・未試行を含む。並行するindexによる削除・再保存・新規追加を追跡する現在のDB総数ではない。再開時は新たに一回抽出し直す。バッチごとに累積件数をstderrへ通知する。

観測のためtransactionを分割しない。新規チャンクの途中失敗はチャンク段階全体をrollbackする。既存会話の照合失敗はFTS transaction全体をrollbackする。埋め込みの推論失敗は後続バッチを継続し、保存失敗はコマンドを失敗させるが、それより前のバッチcommitは残る。root未読や永続する解析診断の件数はこれらの今回処理数とは別に確認する。

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


## チャンク再利用の条件と追記計測

[Issue #325](https://github.com/thkt/recall/issues/325) の合意に従い、全イベントの追記パーサーや新世代への一括切替は導入しない。変更ファイルは全体を解析し、完了済みでチャンクを持つ会話だけ、FTS transaction内で内容照合する。新規・空・無効化された会話は #320 のまとめ読みを使う。保存済みの本文自体を照合に使い、chunk_hash、vectorの複製、別の再開台帳は追加しない。再利用後のID順は会話順と一致するとは限らないため、全チャンクのsource範囲を同じtransaction内で確定し、ID順で照合するlegacy backfillの対象に残さない。追加費用は会話単位の旧本文mapと埋め込み済みID集合、新チャンク生成、rowid更新、vec0の削除・件数取得である。モデル利用可能時の各照合は、埋め込み済みIDの取得でvec0を1回走査する。モデル不在・probe失敗時は、同じwriter lock内で取得した保護対象からベクトル不在が確定した会話だけを更新し、照合時のID取得を省く。不一致チャンクに既存vectorがある場合だけ削除でもvec0を走査する。この判断には同じwriter lock内で取得済みの埋め込み済みID集合を使い、追加のDB読取りは行わない。vectorがない不一致チャンクも削除し、除去件数に含める。大きなDBで多数の会話を変更する費用も比較する。無変更ファイルは照合しない。

完了マーカー `sessions.chunks_indexed` は完了した処理の版を兼ねる。現在の `INDEX_RULES_VERSION=1` は既存の #320 完了値と互換であり、rurico `7b2b256` の既定モデル `cl-nagoya/ruri-v3-310m`、固定revision `18b60fb8c2b9df296fb4212bb7d23ef94e579cd3` を前提とする。パーサーの意味、グループ化・分割、モデル・tokenizer・poolingの変更では版を上げる。チャンク化を後段へ残す本文解析のcommit時に負の版（現行は `-1`）を保存し、チャンク化のcommit時に、現行版で解析済みの会話だけを正の完了版へ進める。`NULL` は解析版不明のチャンク未完了状態である。解析版不明または旧版の未完了本文もroot不在時にチャンク化できるが、その完了値は `0` とし、同じ本文の重複チャンク化を避けつつ再読取りの必要性を残す。保存済み完了版が異なる会話と未完了の会話は、metadataが同じでも再読取りし、既存チャンクを再利用しない。root復帰後も同じ条件を適用し、モデル不在・probe失敗時にベクトルを持つ会話は、モデル復旧まで更新を保留する。現在のCLIにモデル選択機能はない。手動のモデルファイル変更や既存索引の出自が不明な場合は、正常な固定モデルを用意して `recall rebuild` を使う。過去に保存された本文/vector不一致の検出は #319 と同様に範囲外である。

新しい観測値は次の意味を持つ。

- `chunks_matched_committed`：FTS更新で本文一致として維持したチャンク数。embedding未作成の一致チャンクも含む。
- `embeddings_reused_committed`：上記のうち既存vectorを持つチャンク数。vectorのsub-chunk行数ではない。変更会話の照合だけを数え、無変更ファイル全体のembedding数は含めない。
- `chunks_reconciled_created_committed` / `chunks_removed_committed`：照合で作成／除去したチャンク数。新規会話等の `chunks_created_committed` とは別の処理分である。強制再生成やorphan cleanupによる除去は含まない。これら4値はFTS commit後に確定し、失敗時には出さない。
- `inference_batches` / `inference_chunks`：本番バッチ推論の呼出し回数／渡したチャンク数。失敗した呼出しも含み、モデルprobeは除く。内部のGPU forward数やtoken分割数ではない。

FTS commit後に停止しても、一致したベクトルと新しいrowid対応は残る。未処理部分は次のindexのpending抽出で再選択する。推論中の同時更新は #319 の本文・世代照合で保存を拒否する。一致するチャンクは世代を変えないため、その入力に対する進行中の推論結果は保存できる。正常終了の残件は `chunks_remaining_snapshot`、途中停止は最後のcommit通知と再開時の抽出で確認する。モデル不在・probe失敗時は #215 の保護を優先する。同時実行の保証には、全writerがこれらの保護を持つ版であることが必要である。

ソース列挙、既存セッションとモデル不在時の保護対象の取得、孤立セッションの削除は同じwriter lock内で行う。列挙後に別indexが新規会話を保存し、古い列挙結果からその会話を削除する競合を防ぐ。列挙時間もロック保持時間に含まれる。ファイル生成自体はロックしないため、列挙後の新規ファイルは次のindexで取り込む。

長い会話の実モデル比較は同じ既存スクリプトを使用する。開始版 `e73b80499fdadbce524b20c6ffbf3ddd266dd472` と変更版を同じrelease条件でビルドし、各版を別の新規保存先で実行する。変更版だけ `--verify-reuse` を付ける。開始版も段階通知を持つため `--baseline` は不要である。

測定前に両版のRustソース・Cargo.toml・Cargo.lockと使用する計測スクリプトのSHA-256、ビルドコマンドを記録する。スクリプトの `conditions.json` はbinaryのSHA-256を記録するが、そのbinaryが提出ソースから作られたことまでは証明しない。ビルド前後と提出時の入力一致を別途確認し、コード修正後は古い候補の時間を修正版の結果として扱わない。

```sh
python3 scripts/measure-index.py /absolute/path/to/recall /absolute/path/to/new-results \
  --long-session-turns 512 --verify-reuse \
  --model 'モデルID・固定revision・artifact識別値' \
  --conditions 'commit、チップ、RAM、OS、Rust/Metal版、電源/GPU負荷、cache条件'
```

既定512ファイルのうち先頭だけ512往復とし、残る511ファイルは1往復に保つ。初回は1,023チャンク、追記は先頭へ1往復で合計1,024チャンクとなる。本文は短い合成文であり、512往復の私的な長文ログと同じtoken負荷ではない。初回、無変更、mtimeのみ、1往復追記、最終rebuild、中断・再開を測る。変更版の追記は推論入力1・再利用512、無変更とmtimeのみは推論入力0を検査する。各コマンドのモデルprobeは別の費用として残る。

開始版には `inference_chunks` / `inference_batches` / `embeddings_reused_committed` がない。正常終了でdegradedでない試行では、既存の `chunks_pending_snapshot - chunks_unattempted` を推論入力数、stderrの `index: inference: started` の出現数をバッチ呼出し数として読む。これはモデル内部のforward回数ではない。開始版には変更会話のチャンク再利用機構がなく、追記では513入力・5バッチ、mtimeのみでは512入力・4バッチとなることを実測で確認する。変更版の追記は1入力・1バッチ・512再利用が期待値であり、測定結果と区別する。無変更ファイル511件は再利用カウンターの対象外である。

段階の比較では包含関係の変更にも注意する。開始版の列挙と既存データ準備はFTS transaction外、変更版では内側なので、両版の `fts_transaction` や `fts_excluding_parse` をそのまま同じ作業量の値として比較しない。変更版の照合時間、列挙・準備を含めたFTS時間、推論時間、wallを分けて示し、列挙をロック内へ移した費用も判断に含める。`fts_transaction` はロック待ちも含むためロック保持時間そのものではなく、単一writerの計測だけでは同時indexの待ち時間は検証できない。

保存本文とチャンクのsource rowid範囲は、メッセージの会話内位置へ正規化したSHA-256で照合する。mtime更新前後と追記後/rebuild後の一致をスクリプトが検査し、新旧版間の一致もJSONを比較して確認する。SQLiteの割当ページ容量・使用ページ容量も保存する。ページ容量には既存データや空き領域の影響があり、新しい状態だけの容量とは扱わない。Python計測ではvector値を読まない。保持vectorのbyte一致、一部だけ埋め込み済みの重複本文、末尾assistant、切詰め・差し替え、規則変更、失敗・再開と別接続の旧推論結果拒否はカウント付きstubの回帰検証で確認する。

PRの測定記録にはハードウェア・モデル・件数・変更量、推論入力数とバッチ回数、再利用数、段階時間とwall、保存データの一致、ページ容量を示す。双方を同じ入力・設定で交互順に繰り返し、無変更の追加費用と追記の削減を分ける。中断の1試行成功を一般的な競合保証にせず、stubのbyte一致を実モデルのfloat値一致や検索品質の証明にしない。[#324の既存測定](../research/index-observability-measurements.md) は今回の性能結果ではない。実モデル計測・設定済みfull check・CIと独立評価はホストで実施し、この手順の追加だけで費用に見合うことを確認済みとはしない。

引継ぎ時の[#325実測記録](../research/index-reuse-measurements.md)は、上記の512ファイルの手順とは別に、ホスト用scriptで合成1会話256往復へ1往復を追記した結果である。測定版・入力fingerprint・保存内容の照合結果・未測定範囲は同記録を参照し、提出版との差分を確認して再利用する。過去の実測を提出版のfull checkやCIの成功へ読み替えない。
