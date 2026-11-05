---
globs: ["Cargo.lock"]
scenes: ["implement", "pr-create"]
---

# 作業と無関係な Cargo.lock の更新が混ざったら、別コミットに分けるか差し戻す

## 内容

作業の途中で Cargo.lock に無関係な更新が入ったら、作業の差分に混ぜない。別コミットに分けるか、Cargo.lock から戻す。戻した更新を取り込むときは単独の PR にする。

## 定型手順

1. PR を出す前に Cargo.lock の差分を読み、作業と関係のない更新を探す
2. 更新を残す場合は別コミットに分け、そのコミットとテストに使った lockfile を PR 本文に書く
3. 更新が不要な場合は Cargo.lock から戻し、戻したことを PR 本文に書く
4. 戻した更新を取り込むときは単独の PR にし、[依存更新 PR](dependency-update-prs.md)の手順で本文を書く

## 参照コード

- `.github/workflows/ci.yml` の `cargo nextest run --profile ci --locked`（lockfile を固定してテストする。差分が作業と関係するかは判定しない）

## 根拠

- #373 分割と無関係な Cargo.lock の更新を `4ffc9ad` に分け、テストをその lockfile で実行したと書いた
- #383 関係のない `cc` 1.10.0 から 1.11.0 への更新を、同じコミットで Cargo.lock から戻した
- #385 #383 で戻した `cc` の更新を単独の PR にした
