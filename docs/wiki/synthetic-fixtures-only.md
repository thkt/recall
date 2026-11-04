---
globs: ["tests/**", "src/**/tests.rs", "src/**/*_tests.rs"]
scenes: ["issue-create", "implement", "pr-create"]
---

# 検証は合成 fixture と scratch path で行い、本番索引と私的ログに触れない

## 内容

再現・テスト・計測は合成 fixture と明示した scratch path だけで行い、本番の索引、インストール済みのバイナリ、リリースには触れない。私的な会話本文、実 ID、実 DB、生ログ、その集計は Issue・PR・fixture・research に載せない。

## 定型手順

1. fixture は合成データで作り、ID は架空のものにする。私的ログの本文を fixture へ写さない
2. テストと計測の出力先は `tempfile` などの scratch path を明示する
3. 実モデルを使う検証も、artifact の場所を環境変数で明示して本番の索引を使わない
4. PR 本文に、本番索引の移行・インストール済み CLI の切替・リリース・Homebrew 更新を実施していないことを書く
5. ローカルで観測した集計値を、一般的な精度の保証として使わない

## 参照コード

- `src/native/real_tests.rs` の `real_model_bit_dump_for_build_parity`（`RECALL_GEMMA_TEST_ARTIFACT_DIR` と `RECALL_GEMMA_TEST_BITS_OUT` で入出力を明示する）
- `tests/support/root_skip.rs` の `assert_root_canary`（`tempfile::TempDir` 上で権限を作る）

## 根拠

- #340 / #341 / #342 公開するのは合成 fixture だけで、私的本文・実 ID・ログ・DB・集計・実行記録は公開しないと明記した。#340 は例に架空の ID を使った
- #319 / #320 / #321 / #324 / #325 / #326 / #327 私的ログ本文の公開と fixture への複製を禁じた
- #339 分類の検証を合成 fixture で行った
- #352 本番索引の移行、インストール済み CLI の切替、release の公開を実施していないと明記した
- #354 / #355 すべての検証が独立 fixture と明示した scratch path を使い、本番索引とインストール済みバイナリを変えないと書いた
- #358 実データの index 操作とリリース・Homebrew 更新を実施していないと明記した
- [research:session-provenance-classification.md](../research/session-provenance-classification.md) ローカル観測の集計を一般的な精度の保証にしない
- [research:issue-342-scope.md](../research/issue-342-scope.md) 合成データのみで再現した
- [research:bounded-pending-measurements.md](../../research/bounded-pending-measurements.md) 合成データのみで、私的ログや本番 DB を使わない計測
- [research:index-reuse-measurements.md](../../research/index-reuse-measurements.md) 同上
