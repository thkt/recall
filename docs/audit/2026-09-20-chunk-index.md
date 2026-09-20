# Issue #320: チャンク生成の検証記録

## 対象と実装

[Issue #320](https://github.com/thkt/recall/issues/320)は、0チャンクの成功の永続化と、新規・更新会話の本文取得を対象とする。[PR #329](https://github.com/thkt/recall/pull/329)の整理は公開版 `f8b7426ebff2e63112c5b1adad18e51df737a293` から行った。過去の測定・検証の詳細は[整理前の記録](https://github.com/thkt/recall/blob/f8b7426ebff2e63112c5b1adad18e51df737a293/docs/audit/2026-09-20-chunk-index.md)に残し、以下は今回のソースでの結果を示す。Issueの実DB測定は調査版 `f4e3c552` の結果であり、今回の性能値には流用しない。

`sessions.chunks_indexed` はNULLが未処理、1が完了。0チャンクでも完了を保存し、本文の再解析で解除する。未処理の存在確認から本文取得・チャンクと完了状態の保存まで同じIMMEDIATE transaction内で行う。進捗はcommit前の処理会話数で、SQL失敗・callback panicでは両方をrollbackする。旧DBでは列追加と既存チャンクを持つ会話の完了設定を一緒にcommitし、チャンクID・本文・NULLのrowid範囲・embeddingを保持する。従来0チャンクだった会話は一度処理する。

未処理がなければ一時表作成とFTS走査を省く。未処理があれば、`CROSS JOIN`でFTSを外側の単一走査に固定し、対象のsession_id・rowid・本文バイト数を一時B-treeへ集める。FTSのsession_idはUNINDEXEDなので、会話ごとに検索すると全走査を繰り返すためである。本文はrowid等価検索で取得し、最大64会話・本文合計8 MiBを基準に分ける。8 MiB超の会話は丸ごと単独処理するため、**8 MiBは絶対メモリ上限ではない**。一時B-treeは対象メッセージ数に依存し、バイト数算出でも本文にアクセスする。

モデル不在・probe失敗時は既存embeddingを持つ会話を保護する。解析前の既存パスの確認は現在そのパスに保存されたIDを、解析後のID確認は別パスから置換されるIDを守る。両者は異なるIDになり得るため、どちらも必要である。embeddingが利用可能なら更新する。

[ADR-0004](../decisions/0004-forbid-blanket-deletion-in-the-index-rebuild-path.md)のroot保護、[ADR-0005](../decisions/0005-resolve-fts-hits-to-chunks-by-source-message-rowid-range.md)の順序・検索用rowid範囲、[ADR-0007](../decisions/0007-evolve-the-index-schema-without-a-version-table.md)の形状検出による反復可能な移行を維持する。旧NULL範囲・ファイル情報のbackfill、show、DELETE全体、embedding生成は別経路であり、index全体のFTS走査が一度になるとは扱わない。

PRには既存CIのRUSTSEC-2026-0258・RUSTSEC-2026-0285対応として、h2 0.4.16・rustls 0.23.45と必須推移依存3件（rustls-webpki、aws-lc-rs、aws-lc-sys）の更新、および既存Rust検証とCIを使う `.dotagents.json` が含まれる。今回の整理では依存・amici/ruricoのpin・CI・95% coverage基準・集計処理・設定を変更していない。手動benchmarkは既存coverageのテスト除外規則に合うインライン `#[cfg(test)] mod benchmarks` に保持した。

## 再現条件と測定結果

測定対象は開始版 `f8b7426` に今回の整理差分を適用したソース。`src/indexer.rs` のGit blobは `164e80e88bc7bb5bc1fb1d4b03e0cefd194777f6`、変更していない `Cargo.lock` のblobは `5aa7b683ef66b50fe838a09e13ec9017e014720b`。比較用の旧処理は初回開始版 `b4e4eb8` に基づく。旧・新は同じ依存・chunkerを使う。[benchmark定義](../../src/indexer.rs)を次で明示実行する（`--offline`には依存の取得済みcacheが必要）。通常CIではignoredのままで、時間のassertは設けない。

```sh
cargo test --offline --locked --bin recall chunk_index_scale -- --ignored --nocapture --test-threads=1
```

macOS 27.0（26A428）/ arm64、Rust・Cargo 1.98.1、bundled SQLite 3.53.2、debug testビルド。Apple公式Metal ToolchainとCMake deployment target 14.0を準備した共有targetを使用した。Rustソースのmtimeを更新して当checkoutの再コンパイルを確認し、対象1件成功・0 ignored、テスト本体32.42秒。計測中に他のビルド・テストは実行していない。

合成20,580会話・245,757メッセージ・本文14,898,284 bytes。assistantのみ61会話・589メッセージ、残り20,519会話は1 Q&Aずつ。短い合成本文であり、実ログや実モデルは使わない。同じWAL DB、warm cache、各3回、旧・新の実行順を組ごとに交互化した。初期生成・検証用snapshot・追記条件の復元は計測外。初期snapshotは新方式で生成しており、旧方式による初期全件再生成の比較ではない。

| 条件 | 旧方式の秒数（3回） | 新方式の秒数（3回） | 中央値 旧 → 新 |
| --- | --- | --- | --- |
| 無変更 | 3.835281 / 3.904408 / 3.924122 | 0.001586 / 0.001816 / 0.001493 | 3.904408 → 0.001586 |
| 1会話追記 | 4.218127 / 4.227036 / 3.859888 | 0.296373 / 0.289886 / 0.295522 | 4.218127 → 0.295522 |

| 条件 | 処理会話 旧 → 新 | 無変更の空会話再処理 旧 → 新 | FTS全走査statement 旧 → 新 | 本文バッチ 旧 → 新 | 取得メッセージ 旧 → 新 | 新規チャンク（両方式） |
| --- | --- | --- | --- | --- | --- | --- |
| 無変更 | 61 → 0 | 61 → 0 | 61 → 0 | 61 → 0 | 589 → 0 | 0 |
| 1会話追記 | 61 → 1 | 60 → 0 | 61 → 1 | 61 → 1 | 590 → 11 | 1 |

各回、全チャンクのsession_id・内容・順序・timestamp・rowid範囲が一致した。総数は無変更20,519、追記20,520。走査数は全走査statementの実行カウンターであり、SQLite内部のページ読取り数・row visit数ではない。

## 検証と限界

今回の実装整理はCOUNTと件数変換をEXISTSに替え、進捗totalを既に必要な会話一覧の長さへ一本化した。fixtureは列指定INSERTを維持し、不要なNULLだけを省いた。分類の未判定・ファイル情報の未処理を示すNULL、旧スキーマ、特殊なsource・パス・mtimeは保持した。既存テストの追加・削除・統合やassertion変更はない。変更した2テストの命名は[ADR-0012](../decisions/0012-adopt-no-prefix-descriptive-test-names-repo-wide.md)に合わせ、誤って移行テストへ付いていた別テストのコメントを元の対象へ戻した。日英READMEは現在の動作を短くし、この文書は重複する履歴を固定リンクへまとめた。

空結果→再open→無変更→追記は、件数だけでは見逃す0チャンク会話の再取得・再処理を検出する。進捗とSQL失敗/panic→rollback→retryは、先に処理した空会話を誤って完了にする事故を防ぐ。移行テストは故障注入と再openで重複生成・ID/embedding破壊を検出する。65会話の実SQLite検証は、会話間混入・順序逆転・未知role混入・連続assistant脱落・検索範囲の誤りと、EXPLAINによる全走査への退行を確認する。小さいメタデータの境界テストは大量の文字列を作らず64会話・8 MiB・単独超過を守る。別パス同一IDのテストは既存パスだけの保護では見逃す置換によるデータ消失と、モデル利用可能時の更新を確認する。

これらは異なる故障条件を守るため再利用し、実装をなぞる新テストは加えなかった。PRで従来の未知role単独テストを65会話検証へ統合した条件も残り、今回失った検出条件はない。小さいSQLite/MockEmbedder検証と、大規模件数で走査数・全結果一致を確かめる手動benchmarkの分担を維持する。

整理前後は同じマシン・依存・debug profileで、他のビルド・テストと並列にせず `cargo test --offline --locked --bin recall indexer::tests -- --test-threads=1` を各1回実行した。両方49件成功、テスト本体0.41秒 → 0.51秒（コンパイル除外）。修正後の移行・show・分類・検索結果の不完全表示に関するnextest 12件、`cargo fmt -- --check`、`git diff --check`も成功した。単回の差・行圧縮・コメント移動を速度や安定性の改善とは扱わない。

これはチャンク生成工程の測定であり、総index時間・推論速度の測定ではない。実DBの再実行、release時間、長大本文のピークメモリ、実モデルの障害、プロセス強制終了・電源断、追加の同時実行実験は未確認。既定設定のクリーンなXcode 27ビルドの成功も示さない。今回の局所確認はホストの完全check・変更文書を含む独立評価・公開headのCI（test / coverage / security / zizmor）を代替しない。過去の成功を今回の成果物の成功として扱わない。
