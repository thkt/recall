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
