# docs/wiki

このプロジェクトで繰り返し現れる定型手順 / 規約と再発指摘を、過去の PR/issue から蒸留して蓄積する場所。issueのPlanやレビューで毎回同じ説明 / 同じ指摘を繰り返す代わりに、ここ1箇所を参照して二重管理と陳腐化を防ぐ。

1共通項1ページで、各ページは 内容 → 定型手順 → 根拠 の順。根拠が1件だけの共通項は `_candidates.md` に置かれ、2件目が現れたらページへ昇格する。

## 育て方

`/scribe` スキルまたは定期巡回の scribe が、closed になった PR/issue から自動で抽出する。初回は全履歴を読んで基礎化し、以降は前回の scribe PR マージ以降に closed になった分から増分更新する。追加 / 修正はPRで提案されるので、レビューしてマージする。マージが承認になる。手動追記も可。

## 現行の手順

- [リリース](release-procedure.md)
- [FTS・vectorのフィルタ対称性](fts-vec-two-leg-symmetry.md)
- [Gemmaの検証・モデル更新・切戻し](../gemma-model-updates.md)

旧amici/ruricoのrev lockstep手順は依存の撤去に伴い削除した。過去の更新理由はGit履歴で参照できる。
