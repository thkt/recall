# 会話の出自による分類の補完

[Issue #327](https://github.com/thkt/recall/issues/327)の実装前調査。確認日は2026-09-27、対象はrecall `5c14e8ba46cbf1441b2cf40b14bf0f233a7052ca`。Issueの範囲と完了条件を維持し、同日の利用者による「scopingでチェックしてimplementで実装」の依頼に基づいて引き継ぐ。以下の観測は実装前の根拠であり、変更後の検証結果ではない。

## 結論と範囲

先頭ユーザー文の判定を、生成元が記録する既知の出自で補う。確実な出自があれば`automated`、なければ既存の先頭文判定を適用する。不明な形式だけを理由に通常検索から隠さない。`interactive` / `automated`の保存・出力トークン、`--include-automated`、本文と埋め込みは維持する。

初回索引と明示的な`recall classify --all`に同じ規則を適用する。既存DBはこの明示操作で更新できればよく、全会話の再埋め込みやsearch/showでの書込みは不要。索引対象の一律除外、親子会話UI、会話の削除、Jevによる本文分類は対象外。

出自の追加は分類精度の変更であり、索引処理量の削減を保証しない。通常検索の混入と見落としを分けて評価し、索引速度の改善とは区別して報告する。

## 確認した形式

### Claude

ローカルのJSONLから本文を共有せず、配置とトップレベルのメタデータを確認した。パス順の末尾300件のsubagentファイルでは、先頭30行に現れる`isSidechain`はすべて真だった。subagent配置以外の末尾150件ではすべて偽だった。300件はいずれも`<親会話のUUID>/subagents/agent-*.jsonl`という配置で、`agentId`を持っていた。

採用候補は、既知のメッセージレコードのトップレベルにある真偽値`isSidechain: true`と、この限定されたsubagent配置である。`agentId`の存在、ファイル名の`agent-`だけ、パス中の任意の`subagents`だけでは判定しない。本文に引用されたJSONや文字列の`"true"`も根拠にしない。

これは手元の生成物の観測であり、Claude全バージョンの公開仕様を確認したものではない。不明な配置や将来の形式は先頭文判定へ戻す。

### Codex

`session_meta`の`payload.source`と`payload.thread_source`を使える。アプリサーバーの別形式にある`subAgent`と、rolloutの`subagent`を混同しない。

2026年9月のローカルJSONLをパス順で末尾1,200件確認した。`source`の内訳は`exec` 708件、`vscode` 4件、`subagent.thread_spawn` 85件、`subagent: review` 4件、`subagent.other: guardian` 399件だった。`thread_source`は`user` 712件、`subagent` 89件、`guardian_review` 398件、`agent_created_thread` 1件だった。未知のfeature名が併記されても、別フィールドに確実な出自があればその根拠を使える。これらは形式の観測数で、分類の正解ラベルや精度の測定ではない。

公開コードはOpenAI Codexの固定commit `8f195c93d7e7acfef95acf273f0e49cce917e291`で照合した。

- [protocol.rs](https://github.com/openai/codex/blob/8f195c93d7e7acfef95acf273f0e49cce917e291/codex-rs/protocol/src/protocol.rs): `SessionSource`のserde名は小文字、`SubAgentSource`はsnake_case。`review`、`compact`、`memory_consolidation`、構造化された`thread_spawn`、`other`が定義される。`ThreadSource`の`subagent`、`guardian_review`と、任意文字列を受け取るFeatureを区別している。`InternalSessionSource`には`guardian`と`memory_consolidation`がある。
- [guardian/mod.rs](https://github.com/openai/codex/blob/8f195c93d7e7acfef95acf273f0e49cce917e291/codex-rs/core/src/guardian/mod.rs): 承認レビューの名前を`guardian`と定義している。
- [guardian/review.rs](https://github.com/openai/codex/blob/8f195c93d7e7acfef95acf273f0e49cce917e291/codex-rs/core/src/guardian/review.rs): 旧形式の`SubAgent(Other("guardian"))`と新形式の`Internal(Guardian)`を承認レビューとして扱う。

この根拠のある既知形式を分類し、任意の`other`名や未知のfeature名まで一般化しない。`exec`、`vscode`、`originator`、エージェント名、親IDの存在、本文中のapproval/reviewという単語だけでは自動会話と判定しない。不完全な`thread_spawn`など、既知の構造を満たさない値の扱いも合成例で確認する。

## 実装で守る境界

`src/classify.rs`の先頭文判定は現在7種類の文字列prefixだけを使う。`src/parser/claude.rs`と`src/parser/codex.rs`は上述の出自を読み捨て、`src/indexer.rs`は保存時に先頭文だけで分類する。出自は既存の解析中に取り出し、索引時の追加の全ファイル再読込みを避ける。

`src/main.rs`の再分類は保存済みの先頭ユーザー文だけを使い、既定では未分類行、`--all`では分類済み行も対象にする。既存DBへ出自を補うには保存されたsourceとファイルパスを利用できる。必要最小限の保持方法は実装時に選ぶが、メタデータ全体の保存や新たな分類台帳は必要としない。元ファイルの欠落・読取り失敗・不正形式・会話IDの不一致を区別し、無関係なファイルから出自を借りない。未知の出自で先頭文判定へ戻る場合と、既知の分類を再確認できない場合の動作を明記する。

再分類は`--all`、未分類のみ、`--dry-run`の操作契約を保つ。ユーザー文のない会話でも確実な出自を持つケースを扱う。再実行して結果が安定し、本文・チャンク・ベクトルを変更しないことを確認する。ファイル読取り中に索引更新が入る場合も、古い情報で新しい行を上書きしない構造にする。search/showへ移行や再分類処理を追加しない。

先頭文のマーカー増加だけでは、自然な依頼文で始まるsubagentを識別できない。全面再索引は既存の明示分類操作より高コストで、要求に不要な埋め込み更新を伴い得る。出自を優先し、不明時に従来の判定へ戻る案が既存設計に収まる。

## 検証と文書

公開可能な合成fixtureを用い、通常の人の相談、既知のClaude/Codex subagent、承認レビューの旧形式・新形式、不明・不正な出自、本文中のマーカー引用、元ファイルを利用できない既存行を区別する。初回索引と再分類で規則が揃い、再分類の反復・dry-run、通常検索と`--include-automated`で期待した会話集合になることを確認する。既存のFTS・ベクトル検索の共通フィルターを維持する。

合成例について、既知の自動会話の通常検索への混入件数と、人の会話の見落とし件数を示す。公開するのは合成データと集計・判定根拠に限り、私的な会話本文・クエリ・認証情報は含めない。手元の出自集計を一般的な精度の保証に使わない。

`.dotagents.json`のfmt、nextestのci profile、all-targets/all-featuresのclippyと、既存CIを維持する。READMEの両言語版で分類規則と既存DBへの適用方法を更新する。

関連判断は[ADR-0008](../decisions/0008-classify-session-type-by-first-turn-heuristic-with-fail-open-default.md)のfail-open、[ADR-0002](../decisions/0002-treat-persisted-and-emitted-string-tokens-as-a-stable-contract.md)の保存トークン、[ADR-0007](../decisions/0007-evolve-the-index-schema-without-a-version-table.md)の既存DB互換、[FTS/vecの対称性](../wiki/fts-vec-two-leg-symmetry.md)。fail-openの判断は維持する。acceptedな判断記録の本文を現行仕様へ上書きせず、新たな重要判断が必要なら後続記録で扱う。

## 引き継ぎ状態

この報告はIssue #327の根拠補足として作成した未公開資料であり、実装差分とともに検証・独立評価を経てPRで共有する。原本の対象版と観測条件を保持し、変更後の検証結果は区別して追記またはPRに記す。Issueの要求や利用者の許可を拡大する未解決事項はない。Claudeの未観測形式やCodexの未知のfeatureは未対応条件として残す。

## Issue #327 実装時の照合と検証定義（2026-09-27）

開始commitは `0e0ff51b49a5f8bef5f94cc780e7e5647490ab90`。引き継ぎ指定の本報告の blob `626c4f061b6e9cdc754eb043dcd8a9e5cdaa9353` と作業開始時の内容は一致した。上記の実装前観測は変更せず、この節に実装側の結果と限界を分ける。報告対象 `5c14e8ba46cbf1441b2cf40b14bf0f233a7052ca` から開始commitまでの `src` 差分はなく、Issueが参照する `f4e3c552a0cbf9221996b3fa3af470d79990c6a8` の `classify.rs` も開始時と同一だった。これはコードへの適用性の確認であり、観測データの再収集や共有完了を意味しない。

### 採用した根拠と実装上の選択

要求の正本は [Issue #327](https://github.com/thkt/recall/issues/327)。先頭文を出自で補う範囲、本文と埋め込みの保持、明示再分類、保守的な未知形式の扱いはその完了条件に基づく。Claude の配置と `isSidechain` は上記の限定されたローカル観測を根拠に採用し、全バージョンへの保証には広げない。Codex の3つの固定commitリンクは実装時にも読み直し、serde名、`thread_spawn` の必須の親ThreadIdと `i32` depth、guardian の旧・新形式を確認した。同版の `ThreadSource` には `memory_consolidation` も定義されるため、任意の feature と区別して既知の根拠に含めた。

出自の判定をパーサーの既存読取りに加え、初回索引と明示再分類で同じ分類関数を使う。メタデータ全体の保存やスキーマ変更は不要と判断した。採用した優先順と通常の操作手順は [README](../../README.ja.md#分類classify) を参照する。ADR-0008 の不明時に通常検索へ残す方向、ADR-0002 の保存トークン、FTS/vecの共通除外条件は維持する。acceptedなADR本文は変更していない。

`classify --all` は保存source・パスを読み直す。Claude は既知メッセージの `sessionId` がファイルIDまたは限定配置の親UUIDと矛盾しないこと、Codex は解析したIDと保存IDが一致し、複数のメタデータIDが矛盾しないことを確認する。欠落・読取り失敗・不正形式・ID不一致などは理由別に警告し、既存ラベルを保持する。NULL行だけは保存済み先頭文へ戻す。正常に読める未知の出自と、既知ラベルを再確認できない状態を区別するための実装上の選択である。

再分類はファイル読取り前から `IMMEDIATE` トランザクションを使い、古い読取り結果で並行索引の新しい行を上書きしない。dry-runは読取りトランザクションだけを使う。長いログ読取り中は他のwriterを待たせるという代償があり、大規模DBでの待ち時間は未測定。ファイルサイズと更新時刻の読取り前後比較も行うが、同一サイズ・同一更新時刻でのファイル置換まで検知する保証はない。

### 検証の価値と残る実行

[分類の結合テスト](../../src/classify/integration_tests.rs)は、公開可能な12会話（既知の自動会話6、人の相談または未知の出自6）を同じ入力・DB・検索語で比較する。自動会話にはClaudeのメタデータと配置、Codexのspawn、guardian旧形式・新形式、ユーザー発話のないレビューを含む。先頭文だけなら自動会話6件が通常検索に残る入力を用い、変更後は混入0件、人の相談・未知の出自の見落とし0件、`--include-automated` では12件すべてが返ることを期待する。これらはホストで実行するアサーションであり、現時点の製品テストの実測結果ではない。

既存の分類・CLI・FTS/vec除外テストは維持した。追加したパーサーの表形式テストは既知形式の見逃し、文字列の `"true"` や未知のfeature、本文引用による誤非表示、不完全なspawnの誤採用を防ぐ。結合テストは既存の手入力ラベルによる検索テストでは守れない、解析から永続分類への接続、dry-run、反復適用、本文・チャンク・実際のmockベクトル行の保持を確認する。別接続の書込み競合テストは固定sleepを使わず、ファイル読取り前のwriter予約を確認する。無関係なテストの削除・統合は行わず、既存の検出条件は失っていない。モデル取得や私的ログを使う品質テストは追加していない。

sandboxで `cargo fmt -- --check` と `git diff --check` は成功した。対象を絞った `cargo nextest run --offline --locked --profile ci -E 'test(classif) | test(provenance)'` は、`mlx-sys` のCMakeが `ml-explore/mlx.git` を取得する際のDNS制約でビルド停止し、製品テストは実行に到達しなかった。調査時のMetal Toolchain不足とは異なる停止理由である。設定済みのsetup、全nextest、clippyとCI（test / coverage / security / zizmor）は変更せずホストへ引き継ぐ。ブラウザー・サーバー・媒体は不要で、captureはnullのまま。

補助確認として、MLXを含めない一時crateへ対象パーサー・分類・日付のソースと既存テスト、再分類関数と今回の再分類テスト2件をコピーし、同じ直接依存版で実行した。39テストが成功し、同じ範囲のclippy（リポジトリの追加lintを含む）も警告なしだった。再分類は実際のSQLiteを使うが、sessions/messagesだけの最小スキーマであり、製品のDB初期化・索引・検索・ベクトル保存を含む結合テストの代替ではない。一時crateは製品の検証定義へ追加していない。

独立レビューでは、Claudeのroleだけを持つ旧形式でID照合が抜ける経路と、clippyの入れ子ifの指摘を修正した。前者は既知出自を無関係な会話へ適用し得るため、同じパーサー表に回帰ケースを追加した。本文に出自を引用しただけのケースとは検出する条件が異なる。

合成例の期待結果を実DB全体の検索品質に一般化しない。未観測のClaude形式、Codexの未知featureや任意のother、メタデータのない承認レビューは未対応のまま。速度・索引処理量・再分類所要時間の改善は測定しておらず主張しない。本文と埋め込みを保持することは、索引対象や処理量を減らしたという意味ではない。

### 初回レビュー後の修復と対象検証（2026-09-27）

レビュー対象 `0e5dd8ccf335ba6b7fa08e9854293a6226d5a116c336df7c35a72fdd563e38cd` の R1-1、R1-2 を現行コードへ照合した。修復開始時の対象コード・テストと本報告はレビュー記録のハッシュに一致し、先行する修復記録はなかった。指定blobからの本報告の変更は実装時の節の追記であり、原観測は維持されている。今回の要求の正本も Issue #327 で、出自形式や完了条件は追加していない。

R1-1 の原因は、空本文の再索引で保存済みIDへ戻す際に、別IDの解析結果から出自だけを引き継いだことだった。別IDの会話レコードは既存会話が空になった根拠にもできないため、その更新を保留し、既存の本文・分類を保持して再試行対象に残した。同一IDのメタデータだけなら分類を更新でき、会話レコードのない空ログには従来の空本文処理を適用する。現行の操作説明は両READMEへ追記した。R1-2 は有効な `payload.id` の分岐内で取得済みIDを使い、出自判定・複数IDの矛盾検査・`rollout-` の置換条件と順序を保って重複取得を除いた。速度への影響は測定していない。

[索引の回帰テスト](../../src/indexer/tests.rs)を1件追加し、同一ID／別IDと既存の2種類のラベルを区別して、再索引直後・明示再分類後・反復時の保存結果を確認する。既存のパーサー単体テストと初回索引の結合テストが通っても、保存IDへの復元をまたぐ誤分類は検出できなかったため追加した。同一IDの対照は、修復によって正当な出自まで無視する退行を防ぐ。実パーサーとSQLiteを使い、モデル取得・固定sleep・私的ログは不要。既存の空本文処理、Codexの解析、再分類と合成検索の検証はそれぞれ異なる境界を守るため維持し、削除・統合による検出条件の喪失はない。

修復前は、別IDへの差替えで期待する `interactive` が `automated` になることを回帰テストで再現した。修復後の `cargo nextest run --offline --locked --profile ci -E 'test(metadata_only_reindex_applies_provenance_only_to_the_matching_session) | test(size_changes_reindex_with_equal_or_submillisecond_mtime) | test(parser::codex::tests) | test(classify::integration_tests)'` は13件成功した（他462件は選択外）。その後、回帰テストの分類確認を再分類の前後に分け、同テストだけを再実行して成功した（0.096秒、1回の観測）。対象Rustファイルの `rustfmt --check --edition 2024` と `git diff --check` も成功した。今回の環境では製品テストを実行できたが、上記の過去のビルド停止記録を置き換える結果ではない。ホスト記録の473件成功・1件スキップも修復前の結果であり、今回の全検証成功とは扱わない。設定済みの全検証とCIはホストに残し、captureは引き続き不要。実ログ全体の精度、大規模DBの待ち時間、速度改善は未検証のままである。


## Issue #342 による再分類方式の変更

上記の Issue #327 実装時の writer 予約と未測定事項は、その時点の記録として残す。後続の [Issue #342](https://github.com/thkt/recall/issues/342) では、ログの読取り・分類を予約の外へ移し、検証した集合を一括適用する。DB 全体の commit とファイル状態の変化で全候補を破棄し、初回＋最大2回の再試行とする。旧形式の出自・ID 検証は全文パーサーと分類専用読取りで共有する。現行の操作と保証の限界は [README](../../README.ja.md#分類classify)、比較条件・結果・未確認事項は [#342 の既存報告](issue-342-scope.md) に記載する。過去の件数や私的ログの観測を、今回の性能・並行制御の証拠へ転用しない。
