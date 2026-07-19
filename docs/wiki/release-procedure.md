# リリース手順

## 内容

リリースは Cargo.toml / Cargo.lock の version のみを変える単独 bump PR で行い、マージ後に merge commit へ `vX.Y.Z` タグを push すると release.yml が build → GitHub Release → homebrew-tap formula 更新まで自動実行する。release.yml 自体の修正は tag push でしか検証できない。

## 定型手順

1. version bump 単独 PR を出す(変更は Cargo.toml / Cargo.lock の version のみ)。本文に前リリース以降の変更一覧と semver 判断根拠を書く(fix のみ→patch / 機能追加→minor、0.x 系は BREAKING を含む場合も minor)
2. マージ後、merge commit へ `vX.Y.Z` タグを push する
3. release.yml が aarch64-apple-darwin バイナリ(mlx.metallib 同梱)をビルドし、GitHub Release 公開と homebrew-tap formula 更新まで自動実行する
4. `brew upgrade recall` で新バージョンが入ることを確認する
5. release.yml を修正した場合は「tag trigger は再現できないため PR 時点では検証不能」と本文に明記し、次リリースの tag push を実検証と位置づける
6. 「機能が動かない」報告は installed 版(brew tap)と main のズレが原因のことがある。その場合はコードを触らず version bump + リリースで対応する

## 根拠

- #136 v0.8.0 の bump PR(BREAKING 含みのため 0.x 慣例で minor、tag push → release workflow → formula 自動更新の確認まで記載)
- #137 update-homebrew の artifact ハードコードパス想定が崩れて失敗 → find による発見へ撤廃、「検証は次回リリースの tag push 時のみ可能」
- #140 v0.8.1 bump。release workflow 完走が #137 の実検証を兼ねると明記、`brew upgrade recall` 確認
- #206 / #208 / #227 / #245 bump PR の定型(version のみ変更 + 変更一覧 + semver 判断)
- #256 installed 版が古いことが不具合の原因と判明し、コード修正でなく minor リリースで対応した実例
- 現行コード: `.github/workflows/release.yml:5`(tags trigger)、`:20`(aarch64-apple-darwin matrix)、`:54,142`(find による artifact 発見)、`:102`(update-homebrew job)
