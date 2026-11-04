---
globs: ["README.md", "README.ja.md"]
scenes: ["plan", "pr-create"]
---

# 利用者向けの挙動を変える PR は README を日英両方で更新する

## 内容

利用者から見える挙動や制約を変える PR は、`README.md` と `README.ja.md` を同じ PR で更新する。片方だけ更新すると、もう片方の README が古い挙動を説明し続ける。

## 定型手順

1. 変更が README のどの節に当たるかを確かめる。削除・保持の条件、移行時の挙動、分類規則、dry-run、並行実行の制約などが対象になる
2. `README.md` と `README.ja.md` の対応する節を同じ PR で更新する
3. Issue の完了条件に「両 README の記載が一致する」を入れ、PR 本文に両言語を更新したと書く

## 参照コード

- `README.md` の `## Usage` と `## Performance`（利用者向けの挙動と性能の説明）
- `README.ja.md` の `## パフォーマンス`（`README.md` の対応節）

## 根拠

- #331 / #332 削除・保持の条件と移行時の挙動を両 README に書いた
- #339 / #344 分類規則を両 README に書いた。#344 は「README両言語へ記載しました」と明記した
- #343 dry-run の挙動を両 README に書いた
- #345 並行実行の制約を両 README に書いた
- #340 / #341 / #342 両 README の一致を完了条件にした
- [research:session-provenance-classification.md](../research/session-provenance-classification.md) 両 README の更新を作業範囲に含めた
- [research:issue-342-scope.md](../research/issue-342-scope.md) 同上
