---
globs: ["src/**/tests.rs", "src/**/*_tests.rs", "tests/**"]
scenes: ["plan", "implement"]
---

# 並行競合のテストは固定 sleep に頼らず順序を固定する

## 内容

writer の競合や同時ダウンロードのような並行の不具合は、固定時間の sleep で順序を作らずにテストする。処理の途中に差し込むフックやバリアで順序を固定するか、実際のプロセスを 2 つ起動して確かめる。順序を固定したテストが実際のスレッド競合を再現しない場合は、そう明記する。

## 定型手順

1. 競合を起こしたい地点に、テストから差し込めるフック（クロージャ引数など）を用意する
2. フックの中で別の接続から書き込むなどして、競合の順序をテスト側で固定する
3. 複数プロセスの競合そのものを確かめたい場合は、実プロセスを二重に起動する
4. 固定した順序が実際の競合を再現しない場合は、テストと PR 本文にその限界を書く

## 参照コード

- `src/classify/integration_tests.rs` の `reclassification_retries_after_a_writer_commits_during_source_read`（読み取りのフックの中で別の接続から書き込み、再試行を確かめる）
- `src/model/cache_tests.rs` の `concurrent_sae_and_recall_download_each_artifact_only_once`（同時ダウンロードの競合）

## 根拠

- #342 固定 sleep を使わず、明示した順序かバリアを使う並行テストを完了条件にした
- #345 writer 予約の旧テストを、読み取りをフックで止めるテストに置き換えた
- #328 OS プロセス 2 つで競合テストを実行した
- #331 テストが実行順序を固定しており、実際のスレッド競合は再現しないと明記した
- [research:issue-342-scope.md](../research/issue-342-scope.md) 固定 sleep を使わない並行テストの要件
- [research:session-provenance-classification.md](../research/session-provenance-classification.md) writer の並行テストで固定 sleep を使わなかった
