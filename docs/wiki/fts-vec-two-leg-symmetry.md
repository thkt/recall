# 検索変更は FTS / vec 両経路に対称適用する

## 内容

hybrid 検索は FTS leg と vector leg の 2 経路を持ち、RRF が両集合を再 union するため、フィルタ・重み・抜粋・正規化・候補数を片経路にだけ適用すると除外漏れや非対称な結果になる。この「片側だけ実装」が本リポジトリで最も再発したバグ型。新しいフィルタ軸・検索ロジックは共有 helper 経由で両 leg に継承させ、対称性テストで固定する。

## 定型手順

1. フィルタ条件は `append_search_filters` (共有 helper)に追加し、FTS 候補クエリ・vec 検索・file-list の全経路が同じ helper を通ることを確認する
2. 「フィルタ指定なしなら早期 return」等のガード条件も新しい軸に合わせて更新する
3. `crate::compat::assert_filter_symmetric` で vec-only 回帰テストを併設する(FTS 経路のみのテストで済ませない。既存フィルタ project / days / source には全て併設済み)
4. 候補を件数で打ち切るのは、新しさの加点と RRF の融合を両 leg に適用した後にする。FTS 側は一致した全セッションを読み、vec 側の件数は `vec_candidate_count` の 1 か所で決める(DR 0016)
5. クエリ正規化を入れる場合は index 側と query 側の設定を同一インスタンスで揃える(片側だけ正規化すると match が空振る)

## 根拠

- #35 / #44 hybrid の vec パスに project/days/source フィルタが伝播せずフィルタ違反セッションが混入
- #36 / #99 vector-only ヒットの excerpt が常に空(抜粋生成が FTS MATCH 依存だった)
- #27 / #40 recency 重みが FTS 専用パスにのみ適用され hybrid は暗黙 ×1.0 だった
- #54 / #116 oversample 倍率 `limit * 3` が FTS と vec で独立定義され silent desync の危険
- #45 / #47 / #48 フィルタロジック 2 箇所重複の解消として共有 helper へ一本化
- #101 クエリ側だけの正規化は index 側と非対称になり全角・CJK が silent miss するため無効化を選択
- #114 current-session filter を「RRF merge が両集合を再 union するため片経路フィルタだと漏れる」として両経路に無条件適用
- #122 automated filter を FTS・vec 両候補クエリに適用
- #283 新規 `--file` 軸の設計要件として「共有 helper 経由で FTS / vec 両 leg が継承する」を明記
- #293 `--file` テストが FTS leg のみだった gap を audit が検出し `assert_filter_symmetric` を追加
- #339 / #340 / #341 / #344 出自による自動会話の分類で、FTS・vec 両方の除外と `--include-automated` での復帰を完了条件にし、#344 で両方を確かめた。#341 はこのページを参照した
- #357 全文・ベクトル・ファイル一覧のフィルターを共通の入口から適用するよう整理し、ベクトル単独の順位を確かめる回帰テストを足した
- #376 automated 除外件数の絞り込みを `append_search_filters` で検索本体と揃え、ベクトルだけで一致するセッションを数えない非対称を Review focus に挙げた
- #416 両 leg を 3×limit で打ち切ってから加点と融合を適用していたため、小さい `--limit` で新しいセッションが落ちた。FTS 側の打ち切りを外し、vec 側の件数を limit 100 まで同じ 300 件にした
- [research:session-provenance-classification.md](../research/session-provenance-classification.md) 共有フィルターの維持を完了条件にし、このページを参照した
- 現行コード: `src/search.rs` の `append_search_filters` (全フィルタの適用)、`build_keyword_session_query` / `vec_search` / `file_list_outcome` (3 経路からの呼び出し)、`src/search/tests.rs` の `combined_filters_apply_to_text_search` / `combined_filters_apply_to_hybrid_search` / `combined_filters_apply_to_the_file_list` (同じ fixture を 3 経路それぞれで検索) と `assert_filter_symmetric` を用いたテスト (対称性固定)
