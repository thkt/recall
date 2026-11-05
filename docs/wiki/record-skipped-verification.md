---
globs: []
scenes: ["pr-create"]
---

# 実施しなかった検証は PR の Scope に Not included として書く

## 内容

実施しなかった検証は、PR 本文の Scope に Not included として書く。途中で実行できなくなった検証には、理由と代わりの確認手段を添える。

## 定型手順

1. 実モデルでの比較や性能の再計測など、この PR で実施しなかった検証を挙げる
2. Scope に `Not included:` で 1 件ずつ書く
3. 実行できなかった検証は、理由と代わりの確認手段を書く
4. テストで届かない経路が残る場合は、その経路と理由を書く

## 参照コード

- `.github/workflows/ci.yml` の `coverage`（#389 がローカルで実行できなかった差分カバレッジを判定するジョブ）
- `docs/wiki/performance-measurement.md` の `定型手順`（測っていない条件を本文に書く）
- `docs/wiki/host-only-model-verification.md` の `定型手順`（実モデルの検証を代替ビルドの結果で済ませない）

## 根拠

- #385 実モデルでの埋め込みの比較、性能の再計測、`cc` が C/C++ のコンパイルに渡すフラグの比較を Not included に書いた
- #388 実モデルでの埋め込みの比較と性能の再計測を Not included に書いた
- #389 同じ 2 件を Not included に書いた。差分カバレッジを CI に任せた理由と、CLI の比較が実際の埋め込みを通らないことも書いた
