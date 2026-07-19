# 検索変更は FTS / vec 両経路に対称適用する

## 内容

hybrid 検索は FTS leg と vector leg の 2 経路を持ち、RRF が両集合を再 union するため、フィルタ・重み・抜粋・正規化・候補数を片経路にだけ適用すると除外漏れや非対称な結果になる。この「片側だけ実装」が本リポジトリで最も再発したバグ型。新しいフィルタ軸・検索ロジックは共有 helper 経由で両 leg に継承させ、対称性テストで固定する。

## 定型手順

1. フィルタ条件は `append_session_filter` 群(共有 helper)に追加し、FTS 候補クエリ・vec 検索・file-list の全経路が同じ helper を通ることを確認する
2. 「フィルタ指定なしなら早期 return」等のガード条件も新しい軸に合わせて更新する
3. `amici::testing::hybrid::assert_filter_symmetric` で vec-only 回帰テストを併設する(FTS 経路のみのテストで済ませない。既存フィルタ project / days / source には全て併設済み)
4. 候補数の倍率などの定数は共有 const に置き、FTS 側と vec 側で独立に再計算しない
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
- 現行コード: `src/search.rs:138`(`append_session_filter`)、`src/search.rs:229-231` / `571-573` / `926-928`(3 経路への適用)、`src/search/tests.rs:2114` 以降(`assert_filter_symmetric` による対称性固定)
