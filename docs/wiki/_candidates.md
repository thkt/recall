# candidates

ページ化前の候補置き場。1 行は「内容 1 行 + 根拠」で、根拠は PR/issue 由来なら #番号、調査原本なら `[research:name.md](../research/name.md)` と書く。旧 `(research)` は原本未特定として残し、推測でリンクしない。同じ原本の別配置や改訂は独立した根拠に数えない。

「単発」は根拠が 1 件の項目で、2 件目が現れたらページへ昇格して行を消す。「昇格待ち」は根拠が 2 件以上ありながら、1 回あたりのページ上限を超えて持ち越した項目。次の run では根拠の多い順にここから先へ進む。「棄却」は Phase 4 が現行コードと突き合わせて落とした項目で、行の次に落とした理由が付く。順位付けの対象から外れる。

## 昇格待ち

## 単発

- cargo machete で未使用 direct dependency を検出し、Cargo.lock 上のトランジティブ残存を確認して削除する — #176
- CI 実行時間削減の定石: cargo install → taiki-e/install-action 化、clippy が strict superset のため cargo check step 削除、Format check を Test より前で fast-fail — #89
- 仕様上発生しえない入力には guard / raise を足さず通す(不可能ケースを防御しない) — #196
- PR 本文が直前 PR のコピーのまま提出される事故(タイトルと本文の不一致) — #275
- /polish は未コミット diff 専用設計で、commit 済み PR には空振りする(pipeline を手動再現し Codex は `codex review --base main` 直接実行) — #293
- /build の Code ステージが docs-only 判定で実装 edit を素通りさせ、成果物が未実装のまま停止する — #287
- 決定論的に書けないテスト(時間制御を要する並行競合など)は書かず、判断根拠をソース側コメントに残す — #142
- ドキュメントや issue タイトルにバージョン番号を固定せず「現行実装」等と書いて陳腐化を防ぐ — #109
- /polish (Codex レビュー) は 2 回走らせる(1 回目の見落としが 2 回目で出ることがある) — #61
- /build の Red ゲートが Rust の snake_case テスト名を計画のテスト名と照合できず、Green を飛ばして未実装のまま進む — #383
- 検証は合成 fixture と scratch path で行い、本番索引と私的ログに触れない #389
- 依存更新 PR は Cargo ファイルだけを変え、影響と残る警告を書く #385
- リリース手順 #386
- 並行競合のテストは固定 sleep に頼らず順序を固定する #389
- コード移動だけの PR は、移動前後のトークン比較と `git diff --color-moved` で移動だけであることを示す #388
- 本番コードを壊す変異を入れて、固定テストが検出できるかを確かめる #389
- テスト専用のコードを別ファイルへ移すときは、先頭の `#![cfg(test)]` で coverage の対象から外す #388

## 棄却

- accepted の DR 本文を現行仕様で上書きせず、新しい判断は後続の DR で扱う #340 [research:session-provenance-classification.md](../research/session-provenance-classification.md) [research:issue-342-scope.md](../research/issue-342-scope.md) #388
  現行の履歴と合わないため落とした。#321 は DR-0011 の Chosen option を書き換え、#341 は DR-0006 の終了コード表に追記し、#383 は DR-0014 の本文を更新した。
