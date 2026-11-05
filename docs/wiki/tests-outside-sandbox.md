---
globs: ["src/db/tests.rs", "src/model/cache_tests.rs"]
scenes: ["implement", "pr-create"]
---

# 読み取り専用マウント・ローカル TCP を使うテストはサンドボックス外で実行する

## 内容

一時ディスクの読み取り専用マウントとローカル TCP を使うテストは、サンドボックス外で実行する。PR 本文には、サンドボックス外で実行したことと、未実行のテストを書く。

## 定型手順

1. 読み取り専用マウントまたはローカル TCP を使うテストをサンドボックス外で実行する
2. サンドボックス外で実行したテストを PR 本文に書く
3. 未実行の任意実行テストを PR 本文に書く

## 参照コード

- `src/db/tests.rs` の `ReadOnlyMount` と `attach_readonly_wal_only_db`（`hdiutil` で一時ディスクをマウントする）
- `src/model/cache_tests.rs` の `concurrent_sae_and_recall_download_each_artifact_only_once`（`127.0.0.1` のローカル TCP を使う）

## 根拠

- #357 読み取り専用マウントを使うテストをサンドボックス外で実行し、任意実行テストは検証に含まないと書いた
- #367 一時ディスクのマウントとローカル TCP を使うテストをサンドボックス外で実行し、未実行の 8 件を書いた
