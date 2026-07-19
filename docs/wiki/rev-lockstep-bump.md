# rurico/amici の rev lockstep 依存更新

## 内容

recall は rurico を直接 pin しつつ amici 経由でも間接依存する。cargo は同一 git source を依存グラフ全体で 1 rev に解決するため、rurico の rev は amici が pin する rev と常に一致させる。ズレると rev conflict で build が壊れる。bump は amici 先行で、rev 更新と API 追従を同一 PR で原子的に行う(分割不可)。

## 定型手順

1. amici を先に bump し、recall の rurico rev をその amici が pin する rurico rev に合わせる
2. `Cargo.toml` の `[dependencies]` と `[dev-dependencies]` の両方の rev を更新する
3. その rev に含まれる破壊的 API 変更への追従を同一 PR で完結させる(rev だけ上げると build が壊れる)
4. `cargo tree -i rurico` / `cargo tree -i libsqlite3-sys` が単一 rev / 単一バージョンに解決されることを確認する(libsqlite3-sys は `links = "sqlite3"` のため複数バージョン共存不可)
5. feature branch の rev で開発した場合は、recall 側 merge 前に main rev へ re-pin するコミットを積む
6. 機能変更ゼロの上流変更は単独で追従せず、次の破壊的変更追従とまとめる(bump defer)
7. 検証: deps のみの PR は `cargo check --all-targets` exit 0、コード追従を含む PR は build / test / clippy / fmt の通過を本文に記載する

## 根拠

- #30 前コミットが rev だけ上げて API 追従せず `cargo check` が5件のエラーで失敗した実例
- #101 「rev bump はコンパイルを通すために API 変更を一括で追従する必要がある(分割不可、原子的コミット)」
- #132 libsqlite3-sys の `links = "sqlite3"` により amici 先行・同一変更が必須と確定
- #135 rusqlite 0.40 化が rurico/amici rev bump と同一変更でしか成立しなかった実例
- #157 lockstep 制約を Cargo.toml のコメントとして明文化(yomu PR #313 と同型)
- #171 「amici が pin する rev と一致させる必要がある」+ amici マージ待ちで同時更新
- #224 / #244 「rurico の rev は最新 amici が pin する rev と一致」を検証して bump
- #271 feature branch rev で開発 → merge 前に main rev へ re-pin する手順を明記
- #79 機能変更ゼロの上流削除は「次回の他 Breaking Changes と一緒に追従」(bump defer)
- #62 `[dependencies]` と `[dev-dependencies]` の両方に rev があるため両方 bump すると明記
- #55 / #56 / #57 / #64 / #75 / #76 / #78 / #86 上流 API 追従 issue の定型(旧/新対応表 + 影響箇所 + 検証チェックリスト)
- #172 / #176 / #273 / #295 手動 deps PR の `cargo check --all-targets` 検証
- 現行コード: `Cargo.toml:14-16`(lockstep コメント)、`Cargo.toml:17,23`(完全ハッシュ pin)、`Cargo.toml:31,34`(dev-dependencies 側の rev)
