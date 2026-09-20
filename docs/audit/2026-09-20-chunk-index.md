# Issue #320: チャンク生成の検証記録

## 対象と根拠

合意した範囲は[Issue #320](https://github.com/thkt/recall/issues/320)の空結果の永続化と、新規・更新会話の本文取得である。初回の開始版は `b4e4eb86beacafea66c723f61a371e321af08c4c`。初回実測・回帰検証はその版に対する未commitの実装差分の記録であり、後続の保護修正、公開後の依存修正・再測定、coverage集計の修正は末尾の各節で版を分けて示す。別の調査報告は指定されていない。

Issueが参照する[調査版のindexer](https://github.com/thkt/recall/blob/f4e3c552a0cbf9221996b3fa3af470d79990c6a8/src/indexer.rs#L757)と初回開始版は、`git diff`で `src/` に差分がないことを確認した。`Cargo.lock` は異なるため、Issueの実DB実測や当時のMetalビルド停止を、後続の成功や速度の根拠として流用しない。初回の比較では両経路に初回開始版の同じ依存を使った。

適用した既存の合意は、[ADR-0005](../decisions/0005-resolve-fts-hits-to-chunks-by-source-message-rowid-range.md)の順序・rowid範囲、[ADR-0007](../decisions/0007-evolve-the-index-schema-without-a-version-table.md)の形状検出・反復可能な移行、[ADR-0012](../decisions/0012-adopt-no-prefix-descriptive-test-names-repo-wide.md)の新規テスト命名。いずれも開始版でacceptedの文書を参照した。Issueが要求する#121のセッション単位の進捗と、単一transactionのロールバックを維持した。

## 実装と適用範囲

`sessions.chunks_indexed` は未処理をNULL、成功を1で表し、0チャンクも成功として保存する。本文の再解析では既存の `INSERT OR REPLACE` がこの列をNULLへ戻す。異なるファイルから同じsession_idを置き換える場合も、旧チャンクを削除して新本文と完了状態を揃える。ただしembeddingが利用できず、置換先IDが既存embeddingを持つ場合は置換前に保護し、本文・チャンク・完了状態・embeddingを保持する。チャンクとマーカーは同一transactionでcommitする。

旧DBの移行は列追加と既存チャンクを持つ会話のマーカー設定を一緒にcommitする。チャンクid・内容・埋め込みを保持し、0チャンクの会話は次の生成処理で一度処理する。列は書込み経路専用なので、読取り専用コマンドの旧DB判定は変更しない。

未処理会話があれば、FTSを一度走査し、対象メッセージのsession_id・rowid・本文バイト数だけを一時B-treeへ保存する。本文はFTSのrowidで取得する。1バッチは最大64会話、本文合計8 MiBを基準に分ける。8 MiBを超える単独会話は単独で処理し、既存chunkerへ会話全体を渡すため、**8 MiBはメモリの絶対上限ではない**。保持量はバッチ予算または最大の単独会話に、メッセージ管理情報と生成チャンクを加えた量に依存する。最大メモリ使用量は実測していない。一時B-treeの規模は対象メッセージ数に依存する。選択時のバイト数算出でも本文にアクセスするので、各本文へのアクセスが一度だけになるという主張ではない。

変更対象は新規チャンク生成。旧NULL範囲の `backfill_rowid_ranges`、ファイル情報のbackfill、show、DELETE全体、embedding処理は引き続き別経路である。index全体のFTS走査が常に一度になるとは扱わない。

## 初回実装時の合成データ実測（履歴）

実行環境はmacOS 27.0、arm64、Rust 1.98.1、bundled SQLite 3.53.2、debug testビルド。再現用の[benchmark（`benchmarks::chunk_index_scale`）](../../src/indexer.rs#L1109)は製品のindexer・chunker・SQLiteを使い、旧方式だけを開始版から比較用に保持する。通常のCIには大規模データの生成・反復比較を課さず、明示実行にする。

```sh
cargo test --locked --bin recall chunk_index_scale -- --ignored --nocapture --test-threads=1
```

20,580会話、245,757メッセージ、本文14,898,284 bytes。61会話・589メッセージはassistantのみ。他の20,519会話は最初にuser、その後assistantの1 Q&Aで、初期チャンク数は20,519。本文は短い合成文字列であり、実DBの本文長分布を再現しない。

初期生成後の同じWAL DBで、無変更と、assistantのみだった1会話へのuser追記を各3回測定した。各組の旧・新の実行順を交互にし、追記比較の前には当該チャンクとマーカーを同じ入力状態へ戻した。初期生成・検証用snapshot・状態の復元は計測外。OS/SQLiteのwarm cacheを含む。以下の採用実測中は別のビルド・テストを同時実行していない。ビルドが重なった予備実行は採用していない。

| 条件 | 旧方式の秒数（3回） | 新方式の秒数（3回） | 中央値 旧 → 新 |
| --- | --- | --- | --- |
| 無変更 | 3.916181 / 3.945468 / 3.824863 | 0.001531 / 0.001482 / 0.001748 | 3.916181 → 0.001531 |
| 1会話追記 | 3.853591 / 3.878729 / 3.847250 | 0.311096 / 0.308702 / 0.302816 | 3.853591 → 0.308702 |

| 条件 | 処理会話 旧 → 新 | 無変更の空会話再処理 旧 → 新 | FTS全走査statement 旧 → 新 | 取得メッセージ 旧 → 新 | 新規チャンク数（両方式） |
| --- | --- | --- | --- | --- | --- |
| 無変更 | 61 → 0 | 61 → 0 | 61 → 0 | 589 → 0 | 0 |
| 1会話追記 | 61 → 1 | 60 → 0 | 61 → 1 | 590 → 11 | 1 |

各回、全チャンクのsession_id・内容・timestamp・検索用rowid範囲を順序付きで比較して一致した。無変更後20,519チャンク、追記後20,520チャンク。初期データ全体の旧方式による再生成は測定していない。初期snapshotは新方式で生成し、今回の比較は無変更・追記時の保持と更新結果を対象とする。走査数は実行した全走査statementのカウンターで、SQLite内部のページ読取り数やrow visit数ではない。対象テストの `EXPLAIN QUERY PLAN` で、選択時のFTS外側走査と、バッチ本文取得時のrowid等価検索を確認する。

この時間はチャンク生成passの計測であり、JSONL走査・再解析・埋め込みを含む総index時間の速度比ではない。実DBの再実行、releaseビルドの時間、長大な本文でのピークメモリは未確認。

## 回帰検証と費用

既存の増分テストを空結果→再open→無変更→user追記へ拡張した。従来のチャンク数だけでは見逃した「0件成功が保存されず、本文取得とチャンク化を繰り返す」不具合を、処理・取得・走査が0回になることまで確認する。session_id衝突の既存テストにも、置換後に旧チャンクが混ざらない確認を加えた。

新しいSQLite回帰テストは、保存時のSQL失敗と進捗callbackのpanicで、先に処理した空会話の完了も作成途中のチャンクも残らず、再実行できることを確認する。移行テストはマーカー更新の故障を注入し、列追加との原子性、再open、未処理会話の回収、既存チャンクid・NULL範囲・embedding bytesの保持を確認する。失敗を成功として固定する危険と、移行時の重複生成・埋め込み破壊は、既存の一般的な列移行テストだけでは検出できなかった。

バッチ境界は小さいメタデータのテストで会話数・バイト数・単独超過を確認し、大きな文字列を通常テストへ追加しない。65会話の実SQLiteテストは、交互に挿入した会話の混入、順序の逆転、未知roleの混入、連続assistantの脱落、rowidの取り違え、FTS全走査への退行を検出する。未知roleの旧単独テストはここへ統合した。削除した検出条件はなく、正確な内容と範囲を確認するようになった。既存のchunker・検索・旧NULL範囲backfill・進捗テストは異なる利用条件を守るため維持する。

実装側は完了列・移行・一時rowid取得とバッチ処理が増えた。テスト側はindexerの3テスト追加・1テスト統合、dbの移行テスト追加、既存テストと列指定fixtureの更新、明示実行benchmarkの追加。文書側は日英READMEへ現在の動作を追記し、実測と制限はこの記録へ分けた。行圧縮やファイル移動による改善は主張しない。

整理前後のindexerテストは、開始版の一時checkoutと変更後checkoutで、同じ依存・マシン・debug profile・`--test-threads=1`、他のビルド・テストを同時実行せずに比較した。`cargo test --locked --bin recall indexer::tests -- --test-threads=1` のテスト本体時間は46件0.35秒 → 48件0.38秒（各1回、コンパイル時間を除外）。共有target内の旧実行ファイルを再利用しないよう変更後ソースを再コンパイルし、追加テストの実行を確認した。この単回差から一般的なテスト高速化や安定性の改善は主張しない。

対象を絞ったチャンク関連60テスト、変更後indexer 48テスト、上記benchmarkが成功した。対象のClippy（`cargo clippy --locked --bin recall --tests -- -D warnings`）も成功。構成済みの全nextest・all-features Clippy・CIの `test` / `coverage` / `security` / `zizmor` はホストの担当であり、この記録はその成功を示さない。実プロセス強制終了・電源断は注入しておらず、回復の検証はSQLite transactionとSQL失敗・panicを対象にした。必要媒体はなく、capture定義は変更していない。

## R1-1: 別パス・同一IDの埋め込み保護

[ホストのreview-1.json](/private/tmp/recall-hardening-20260920-320/verification/review-1.json)は、対象 `922bbf603a6a1b811d7f2c19b69051b8c62ce92719bfa3721723c76f23f88f3f` に必須修正R1-1を記録した。修正着手時に対象記録の全101ファイルのhashを現在のファイルと照合し、差分がないことを確認した。初回の[修正結果](/private/tmp/recall-hardening-20260920-320/repair-codex-H4GmmQ/final.json)は必須修正なしとしていたが、既存の衝突テストはモデル利用可能・embeddingなし、モデル不在テストは同一パスまたは別IDだった。この交差条件を見逃しており、初回評価や修正前ホスト検証の成功を保護の証拠にはできない。これらのリンクと下記ログはホストのローカル記録であり、公開先からの参照は保証しない。

原因は、同一IDの置換で削除する対象をチャンク・embeddingまで広げたのに、保護判定が既存ファイルパスだけを参照していたこと。`index_file` は解析後のsession_idも既存の `embedded_sessions` と照合し、破壊的upsertの前に `Preserved` を返すよう修正した。解析前の既存パスの保護も維持する。前者は置換先ID、後者は現在そのパスに対応するIDを守り、両者は異なるIDになり得る。モデル利用可能時は従来どおり置換する。[ADR-0004](../decisions/0004-forbid-blanket-deletion-in-the-index-rebuild-path.md)のroot保護・孤児削除条件、accepted ADR-0005/0007のrowid・移行条件、Issue #320の完了条件は変更していない。

新規の実SQLiteテストは元ファイルを列挙対象に残し、別ディレクトリの同名Claude JSONLを追加する。`embed_capable=false` で保護件数1・索引更新0・チャンク化0となり、保存先パス・mtime・完了状態・本文rowid/role/text・チャンクid/内容/範囲・embedding bytesが保持されることを確認する。同じ入力をモデル利用可能として再実行し、保護解除、旧チャンクとembeddingの除去、新本文だけの生成も確認する。

このテストを削除すると、別パスの置換がパス単位の保護を迂回するデータ消失を見逃す。既存の衝突テストはembeddingなしでの正常な置換、モデル不在テストは既存パスの保護・新規IDの受入れ・embeddingなしの更新・推論失敗時の保護を引き続き検証するため残した。今回は統合・削除したテストや失った検出条件はない。追加は小さなJSONLと実SQLite、既存MockEmbedderを使う1テストで、実モデル取得・時刻待機・大規模データ生成は不要。実装は保存前の判定追加、文書は日英READMEの現在の保護条件とこの証拠追記である。テスト整理や速度改善は主張しない。

修正前コードに新規テストを適用すると、索引更新0の期待に対して1となり失敗した（[修正前ログ](/private/tmp/recall-320-r1-before.log)）。修正後の衝突2テストは成功、テスト本体0.03秒（[修正後ログ](/private/tmp/recall-320-r1-after.log)）。関連するモデル不在・推論失敗のnextest 8テストも成功、0.222秒（[保護検証ログ](/private/tmp/recall-320-r1-protection.log)、新規テストを含み衝突テスト実行と1件重複）。いずれも単回の対象検証であり、以前のindexer全体の時間とは実行範囲・並列条件が異なるため比較しない。変更Rustファイルの `rustfmt --check --edition 2024 --config skip_children=true` と `git diff --check` も成功した。

上記の合成benchmark・初回検証結果は修正前の履歴として保持し、今回再実行していない。新規回帰テストは実モデルの障害や意味検索の品質を実測せず、既存embeddingの保持を確認する。構成済み全検証と最終独立評価は、修正後成果物についてホストで更新する。

## 公開後の依存修正と再測定（2026-09-20）

今回の開始版は、draft PR #329の公開済みhead `db6e29b3d772713085fcea18b26a18d04ede717a`。上の初回実装・保護修正の検証と独立評価は、それぞれの版の履歴であり、今回の依存セットの成功として扱わない。[PR #329自身のsecurityログ](https://github.com/thkt/recall/actions/runs/35507536691/job/106069775153)で、このheadとmain `95cfc62755a265c4b31660bd0264663fb15b0e60`のmerge版 `c77d26cc290334fb6841438c4f155282667988f3`、および以下の2件の失敗を確認した。PR #328の結果から同じ原因と推定したものではない。

- h2 0.4.15 → 0.4.16: RUSTSEC-2026-0258（空DATAフレームの無制限なキュー蓄積）。
- rustls 0.23.42 → 0.23.45: RUSTSEC-2026-0285（TLS 1.3の暗号化レベル境界をまたぐhandshakeメッセージの誤受入れ）。
- 必要な推移依存のみ、rustls-webpki 0.103.13 → 0.103.14、aws-lc-rs 1.17.3 → 1.18.0、aws-lc-sys 0.43.0 → 0.44.0へ更新した。取得済み公式registry packageのmanifestでも要求を確認した。

`cargo update --offline -p h2 --precise 0.4.16`、`cargo update --offline -p rustls --precise 0.23.45`で解決した。Cargoが併せて変更した無関係なWindows/getrandomの依存選択は開始版へ戻し、lockfileの差分が上記5件のversion/checksumだけで、他のentry・依存選択・source pinが一致することを照合した。`cargo metadata --offline --locked --format-version 1`と`cargo fetch --offline --locked`が成功した。最終Cargo.lockのSHA-256は `c5f50a82a39f2c821f0ef4f0f8b20874c3fe0730a9ba7c5f4433ec86a3c0be2f`。amici/rurico、CI、advisory設定、`.dotagents.json`は変更していない。

チャンク生成・モデル不在時の同一ID置換保護の実装とbenchmark定義は公開版から変更していない。日英READMEは文言を変えず、チャンク完了とモデル不在時の保護の2段落をモデル取得段落の前にまとめ、#319 / PR #328との挿入位置の競合を避けた。#319の実装は含めていない。

再測定はmacOS 27.0（26A428）/ arm64、Rust・Cargo 1.98.1、bundled SQLite 3.53.2、debug testビルド。ホストがMetalとCMake deployment target 14.0を準備した既存の `CARGO_TARGET_DIR=/private/tmp/recall-hardening-target-20260920`を使用した。最初の呼出しは共有targetの古いテストバイナリを再利用し、対象0件だったため成功に数えていない。現checkoutのRustソースを再コンパイルし、次を明示実行して **1 passed / 0 ignored**、テスト本体32.28秒を確認した。

```sh
cargo test --offline --locked --bin recall chunk_index_scale -- --ignored --nocapture --test-threads=1
```

旧・新とも上記の最終依存セットを使用。合成20,580会話・245,757メッセージ・14,898,284 bytes、assistantのみ61会話・589メッセージ、WAL DB、実行順の交互化、各3回、計測外の初期生成・snapshot・復元は初回測定と同じ条件である。今回の計測中は別のビルド・テストを起動していない。

| 条件 | 旧方式の秒数（3回） | 新方式の秒数（3回） | 中央値 旧 → 新 |
| --- | --- | --- | --- |
| 無変更 | 4.060929 / 3.894806 / 3.990312 | 0.001490 / 0.001865 / 0.001604 | 3.990312 → 0.001604 |
| 1会話追記 | 3.931716 / 3.887037 / 3.969410 | 0.311015 / 0.300477 / 0.303307 | 3.931716 → 0.303307 |

無変更時は処理会話・空会話再処理・FTS全走査statementが各61 → 0、取得本文589 → 0、新規チャンク0。追記時は処理会話・走査61 → 1、無変更の空会話再処理60 → 0、取得本文590 → 11、新規チャンク1。各回の全チャンクのsession_id・内容・timestamp・rowid範囲が一致し、総チャンク数は無変更20,519、追記20,520だった。これはチャンク生成passの比較であり、依存更新による速度改善や総index時間の速度比を示さない。初期snapshotが新方式由来、短い合成本文、warm cache、statement数と内部ページ読取り数の違いという制限は初回測定と同じ。8 MiBは絶対メモリ上限ではなく、実DB・release時間・ピークメモリ・実モデル障害・電源断・追加の同時実行実験は未確認である。

修正後の `cargo check --offline --locked` と `cargo nextest run --offline --locked --profile ci -E 'test(indexer::tests) | test(db::tests::chunk_completion_upgrade)'` が成功した（対象50件成功、396件は選択対象外、0.871秒）。既存の空結果→再open→追記、SQL失敗・panic→再試行、移行時の原子性・embedding保持、65会話の混入・順序・未知role・検索rowid・実行計画、別パス同一IDのモデル不在保護を再利用した。単なる件数一致では見逃す再処理、失敗の成功扱い、誤った検索範囲、既存embedding消失を検出するため保持し、依存番号だけを固定する新規テストは追加していない。検出条件の削除はなく、従来統合した未知roleの条件も残る。小さいSQLite/MockEmbedder検証と明示実行benchmarkの分担を維持し、異なる実行範囲の過去の時間からテスト高速化は主張しない。

`cargo fmt -- --check` と `git diff --check` も成功した。構成済み完全check、最終差分・文書を含む独立評価、修正後headのCI（test・coverage・security・zizmor）はホストで更新する。この記録はそれらの成功を先取りしない。前PR本文の「445件成功」「accepted」「benchmark未再実行」は今回の成果物の説明へ転用せず、今回の結果と未確認事項を用いる。提示された前PR本文にアップロード媒体へのリンクはなく、captureは不要のままである。

## 公開後のcoverage集計修正（2026-09-20）

今回の開始版は公開済みhead `93383d1f502021d75de11ad24f4a8bb5801b1e19`。[この版のcoverage job](https://github.com/thkt/recall/actions/runs/35508213196/job/106071481804)の保存ログでは、Rustテストと既存filterの26テストは成功し、変更行coverageだけが45%で95%の基準を満たさなかった。内訳は `src/db.rs` 100%、`src/indexer.rs` 98.9%（未到達801行）、`src/indexer/benchmarks.rs` 0%（123行）。製品の未検証範囲が増えたのではなく、外部モジュール宣言の `#[cfg(test)]` が別ファイルへ伝播しないfilterに、手動benchmarkの実装行が含まれたことが原因だった。

既存filterが示す「テストコードを変更行coverageから除外する」方針に合わせ、benchmarkを宣言位置のインライン `#[cfg(test)] mod benchmarks` へ移した。`src/main.rs` にもある形式で、インデント1段を除く本体・fixture・assertion・コメントは開始版と完全一致し、テスト名と手動実行コマンドも変わらない。製品コード、95%の基準、filter、CI、依存pin、日英READMEの段落位置、移動済みのDB移行テストは変更していない。再現元は上の[benchmarkリンク](../../src/indexer.rs#L1109)を参照する。

変更前後のソース各行に人工のDA/BRDAと関数のFN/FNDAを与え、変更していないfilterへ通した。修正後の除外範囲は既存テスト宣言1106–1107行とbenchmark全体1109–1286行だけで、製品側2,272レコードは変更前後で完全一致し、未到達801行のレコードも残った。これは範囲とレコード保持の局所検証であり、実際のllvm-covや修正後coverage率の測定ではない。ローカルの `python3 -m pytest .github/scripts/test_filter_lcov_cfg_test.py -q -p no:cacheprovider` はpytest未導入で実行できず、既存Pythonテスト群の再実行はホストに残る。

共有targetの別ソース由来のバイナリを使わないよう現checkoutのRustソースのmtimeを更新し、`Compiling recall` を確認してから、同じ `CARGO_TARGET_DIR=/private/tmp/recall-hardening-target-20260920` で次を実行した。**1 passed / 0 ignored**、テスト本体31.15秒。macOS 27.0（26A428）/ arm64、Rust・Cargo 1.98.1、SQLite 3.53.2、debug testビルド、依存セットと生成条件は直前の測定と同じで、この計測中に別のビルド・テストは起動していない。

```sh
cargo test --offline --locked --bin recall chunk_index_scale -- --ignored --nocapture --test-threads=1
```

| 条件 | 旧方式の秒数（3回） | 新方式の秒数（3回） | 中央値 旧 → 新 |
| --- | --- | --- | --- |
| 無変更 | 3.732589 / 3.743320 / 3.774617 | 0.001507 / 0.001483 / 0.001641 | 3.743320 → 0.001507 |
| 1会話追記 | 3.949021 / 3.895734 / 3.756030 | 0.292476 / 0.294683 / 0.297680 | 3.895734 → 0.294683 |

20,580会話・245,757メッセージ・14,898,284 bytesで、無変更の処理会話・空会話再処理・FTS全走査statementは61→0、取得本文589→0。追記時の処理会話・走査は61→1、無変更の空会話再処理60→0、取得本文590→11。各回の全チャンク内容・timestamp・rowid範囲は一致し、総数は無変更20,519、追記20,520だった。従来の測定値は各版の履歴として保持する。これは配置修正後のチャンク生成passの再現確認であり、配置変更による速度改善や総index時間の比較ではない。

今回の検証定義の変更は配置だけで、新規・削除・統合したテストや失った検出条件はない。benchmarkを削除すると、小さい回帰検証では確認できない実DB相当件数での無変更・追記時の走査数と結果一致の再現手段を失うため、費用の大きい実行は引き続き手動に限定する。空結果の永続化・無効化、SQL失敗/panicのrollback、反復移行・embedding保持、65会話の順序・未知role・rowidと実行計画、別パス同一IDの保護は既存の小さいテストを維持する。製品実装の増減はなく、テスト本体175行を移し、文書の参照と今回の証拠を更新した。ファイル移動や過去の32.28秒との差を費用・速度改善とは扱わない。

`cargo fmt -- --check` と `git diff --check` は成功。構成済み完全check、変更文書を含む最終独立評価、修正後headのCIはホストで行い、公開本文の確認とready切替はその後の担当AIに残す。短い合成本文・warm cache・新方式由来の初期snapshot・statement数という測定条件、8 MiBが絶対メモリ上限でないこと、実DB・release時間・ピークメモリ・実モデル障害・電源断・追加の同時実行実験の未確認は引き続き有効である。
