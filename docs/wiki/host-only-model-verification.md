---
globs: ["src/native/real_tests.rs", "tests/gemma_cli.rs"]
scenes: ["plan", "pr-create"]
---

# CI とサンドボックスが届かない実モデル検証は Apple Silicon ホストで行う

## 内容

GitHub の CI や調査用サンドボックスでは、実モデル・GPU・Metal を使う検証まで届かない。probe のタイムアウトや代替ビルドで済んだ結果を成功扱いせず、Apple Silicon のホストで手動検証して、その結果を PR に記載する。

## 定型手順

1. CI が通った経路を確かめる。公開モデルの CI ジョブは、isolated model probe のタイムアウト（終了コード 75）を合格として扱う
2. 実モデルが要る受け入れ条件は Issue の Manual verification に置き、test_command に含めない
3. Apple Silicon ホストで ignored の実モデルテストや計測を実行し、結果と実行条件を PR に貼る
4. サンドボックスでビルドが止まった場合や代替ビルドで済ませた場合は、製品の検証に数えず、ホストでの full check に引き継ぐと書く

## 参照コード

- `tests/gemma_cli.rs` の `public_model_cli_contract`（`isolated model probe timed out` を `timeout` として受け入れる）
- `src/native/real_tests.rs` の `real_model_bit_dump_for_build_parity`（ホストで実行する ignored テスト）

## 根拠

- #352 GitHub の仮想 GPU で子 probe が 30 秒でタイムアウトし、GPU probe の成功とは扱わなかった
- #355 CI は許可された download-probe タイムアウトの経路を通っただけだった
- #379 / #380 公開モデルの CI ジョブがタイムアウトを合格扱いするため skip 経路に届かないとして、Manual verification を手で実行した
- #381 / #383 実モデルの比較と計測を Manual verification として PR に記載した
- #319〜#327 調査時に Metal Toolchain がなく（`cannot execute tool 'metal'`）製品テストをビルドできなかったと記録した（当時の MLX 依存）
- [research:session-provenance-classification.md](../research/session-provenance-classification.md) サンドボックスで CMake の mlx.git 取得が DNS 制約で止まり、MLX を外した一時 crate の確認は製品の検証に数えなかった
- [research:issue-342-scope.md](../research/issue-342-scope.md) 同じ制約と、ホストの full check への引き継ぎ
- [research:index-observability-measurements.md](../../research/index-observability-measurements.md) Metal が使えないサンドボックスの計測を除外した
