---
globs: []
scenes: ["plan", "pr-create"]
---

# 性能改善は変更前後のビルドを交互に複数回測ってから主張する

## 内容

性能改善の PR や Issue は、変更前と変更後のビルドを交互に複数回動かし、中央値と範囲、load average、結果の一致をそろえて記録してから改善を主張する。1 回だけの計測や範囲が重なる差は改善と断定せず、遅くなったケースも報告する。

## 定型手順

1. 比較元を決める。通常は main の release ビルドで、ビルド時パッチのように main と比べられないときは同じブランチを機能を外してビルドする（#383 は `RECALL_LLAMA_SKIP_PATCHES=1`）
2. 2 つのビルドを交互に各 N 回動かす（#342 は 5 回以上を完了条件にした。#376〜#383 は 7〜15 回）
3. 段階時間は `RUST_LOG=recall=debug` の `stage="model load"` などから取り、中央値と最小〜最大を表にする。段階時間が足し算にならない場合はその旨を書く
4. load average、ハードウェア、モデル rev、commit、バイナリの SHA-256 を記録する
5. 全回で `degraded: false` であることと、両ビルドで `results` が一致することを確かめる
6. 検索の p50 は `.claude/OUTCOME.md` の Time Indicator（`recall search` が 2s 以内）と比べる
7. 計測した版と提出した版の差分、測っていない条件を本文に書く

## 参照コード

- `src/model.rs` の `timed`（`model load` などの段階時間を debug ログに出す）
- `docs/index-observability.md` の「時間の定義」「4ケースの実モデル計測」（index 側の計測条件）
- `.claude/OUTCOME.md` の Indicators 表（Time の 2s）

## 根拠

- #342 計測は 5 回以上・中央値と範囲、ノイズと分けられない差は改善としない、を完了条件にした
- #324 ベースラインがないうちは SLO を置かないとした
- #334 / #345 1 回の計測や範囲の重なりから速くなったと断定しなかった。#345 は遅くなった many_short も報告した
- #335 計測版と提出版の製品差分を明記した
- #320 / #325 / #326 / #329 / #333 / #336 計測条件を完了条件や PR 本文に書いた
- #376 / #377 / #378 / #380 実データの索引で main と交互に測った p50 を表にし、load average を併記した。#376 / #378 / #380 は 2s と比べた
- #379 / #381 この計測を Acceptance Criteria と Manual verification に組み込んだ
- #383 パッチなしビルドと交互に各 15 回測り、全回 `degraded: false`、results 一致を確かめた
- [research:issue-342-scope.md](../research/issue-342-scope.md) 計測条件と範囲の扱い
- [research:index-observability-measurements.md](../../research/index-observability-measurements.md) HW・commit・SHA の記録、段階時間が足し算にならないこと
- [research:bounded-pending-measurements.md](../../research/bounded-pending-measurements.md) 候補 commit の計測を提出版の合格に流用しない
- [research:index-reuse-measurements.md](../../research/index-reuse-measurements.md) 計測版と提出版の差分の記録
