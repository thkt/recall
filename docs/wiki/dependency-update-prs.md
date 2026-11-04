---
globs: ["Cargo.toml", "Cargo.lock", "deny.toml"]
scenes: ["pr-create"]
---

# 依存更新 PR は Cargo ファイルだけを変え、影響と残る警告を書く

## 内容

依存更新の PR は Cargo.lock（必要なら Cargo.toml）だけを変え、Git 依存の固定 rev と各依存のバージョン指定は保つ。本文には、更新した crate が recall に届く経路と影響、cargo deny / cargo audit を回して残った警告を書く。

## 定型手順

1. 変更を Cargo.lock（必要なら Cargo.toml）に限る
2. `cargo deny check` と `cargo audit` を実行し、残った警告が既存の許可（`paste` の unmaintained など）だけであることを書く
3. 更新した crate の依存経路（例: `reqwest → … → schannel`）と changelog・MSRV を確かめ、recall への影響を書く
4. MSRV を上げる場合は、ソースからのビルドへの影響を書く

## 参照コード

- `.github/workflows/ci.yml` の `security` ジョブ（`cargo-deny check` と `cargo-audit audit`）
- `deny.toml`（ライセンス許可一覧と advisory の扱い）

## 根拠

- #330 Cargo.lock だけを更新し、cargo deny・cargo audit の結果と残った paste の警告を記載した
- #347 Cargo ファイルだけを変え、Git 依存の rev を保ち、MSRV を 1.98.1 に上げた影響を書いた
- #354 cargo deny・cargo audit が通り、残る警告は既存許可の paste だけと記載し、sha2 の changelog と MSRV を確かめた
- #360 cargo audit と cargo deny を回し、`reqwest → … → schannel` の依存経路と新旧のソースを比べた
- #362 unicode-normalization の Unicode 18 更新が正規化結果に与える影響と、ライセンス許可一覧を確かめた
