# 候補 (根拠1件)

2件目の根拠が現れたらページへ昇格する。

- cargo machete で未使用 direct dependency を検出し、Cargo.lock 上のトランジティブ残存を確認して削除する — #176
- CI 実行時間削減の定石: cargo install → taiki-e/install-action 化、clippy が strict superset のため cargo check step 削除、Format check を Test より前で fast-fail — #89
- 仕様上発生しえない入力には guard / raise を足さず通す(不可能ケースを防御しない) — #196
- PR 本文が直前 PR のコピーのまま提出される事故(タイトルと本文の不一致) — #275
- /polish は未コミット diff 専用設計で、commit 済み PR には空振りする(pipeline を手動再現し Codex は `codex review --base main` 直接実行) — #293
- /build の Code ステージが docs-only 判定で実装 edit を素通りさせ、成果物が未実装のまま停止する — #287
- 決定論的に書けないテスト(時間制御を要する並行競合など)は書かず、判断根拠をソース側コメントに残す — #142
- ドキュメントや issue タイトルにバージョン番号を固定せず「現行実装」等と書いて陳腐化を防ぐ — #109
- /polish (Codex レビュー) は 2 回走らせる(1 回目の見落としが 2 回目で出ることがある) — #61
