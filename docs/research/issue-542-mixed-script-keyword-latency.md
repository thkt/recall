# Issue #542: 英字と日本語が混ざった問い合わせで全文検索が遅くなる原因

| 項目 | 値 |
| --- | --- |
| 対象 | [thkt/recall #542](https://github.com/thkt/recall/issues/542) |
| 確認日 | 2026-11-28 |
| コードの基点 | `20f2ddc9a8c25950790342e2a5233081200c09fc`（v0.18.0 のリリース後） |

recall は「GitHub Actions のワークフロー」の 1 文字の「の」を、語彙の表で「の」から始まる頻出の 3 文字の語 25 個に広げる。そのうえで固定の 3 語を 25 個の節すべてに写すので、MATCH の文は 25 個の AND 節の OR になり、語句は 100 個になる。FTS5 は `"github"` と `"actions"` の位置リストを節ごとに読み、さらに文書の大半に現れる「の〜」の 25 語の位置リストも読む。`rank`（bm25）は一致した行ごとに 100 個の語句を数える。合成した索引では、この文が手で書いた 3 語の AND より 27 倍遅かった。

## recall が組み立てる MATCH の文

合成した索引（後述）に対して recall が組み立てた文は次のとおりである。`の〜` の 25 語は索引の語彙で決まるので、実インデックスでは別の 25 語になる。

```text
("GitHub" AND "Actions" AND "の引数" AND "ワークフロー")
OR ("GitHub" AND "Actions" AND "の条件" AND "ワークフロー")
OR ... (同じ形の節が続き、全部で 25 節)
OR ("GitHub" AND "Actions" AND "の設定" AND "ワークフロー")
```

展開前の中間の形（`MatchFtsQuery` の `Display`）は次のとおりである。

```text
"GitHub" AND "Actions" AND ("の引数" OR "の条件" OR ... OR "の設定") AND "ワークフロー"
```

### 確かめた方法

1. `src/fts_query/tests.rs` に `#[ignore]` を付けた一時的なテストを足した。
2. そのテストは `crate::db::open_db` で本番と同じ表（`tokenize='trigram'` の `messages` と `messages_vocab`）を作り、合成した 48 万行を入れた。
3. 同じテストで `prepare_match_query(&conn, "GitHub Actions のワークフロー")` を呼び、結果の `Display` と `to_trigram_match()` を標準出力に書いた。
4. 出力を取り出した後、テストは消した。リポジトリには残していない。

実インデックスと会話の記録は読んでいない。

### 経路と、文を作るファイルと行

| 段階 | 場所 | この問い合わせでの動き |
| --- | --- | --- |
| 演算子の判定 | `src/search.rs:608` `keyword_query` | 大文字の `AND` / `OR` / `NOT` を含まないので、`prepare_operator_query` は `None` を返す。`src/search.rs:622` の `prepare_match_query` に進む |
| 文字種の境目で分ける | `src/fts_query.rs:292` `segment_script_boundaries` | ひらがなの「の」とカタカナの「ワ」の間に空白を入れる。`GitHub Actions の ワークフロー` になる。長音「ー」はカタカナに分類されるので、`ワークフロー` は 1 語のまま残る |
| 分けた形を採るかの判定 | `src/fts_query.rs:322` `keeps_a_trigram_anchor`（DR 0010） | 3 文字以上の語があるので、分けた形を採る |
| 短い語を語彙で広げる | `src/fts_query.rs:639`、`src/fts_query.rs:643` `fts_expand_short_terms` | 2 文字以下の「の」だけが `expand_token_via_vocab`（`src/fts_query.rs:598`）に渡る |
| 語彙の表を引く | `src/fts_query.rs:616` `VOCAB_LOOKUP_SQL` | 「の」から始まる語を `cnt` の降順で 25 個返す（`src/fts_query.rs:618` の `LIMIT 25`）。合成した索引では「の〜」の 3 文字の語が 49 種類あり、25 個が返った。実インデックスでも 25 個を超えると見ているが、これは推測である。実インデックスで文を取り出す段階で確かめる |
| 組み合わせの数の上限 | `src/fts_query.rs:123` `MAX_TRIGRAM_COMBINATIONS` = 100 | 組み合わせは 25 なので上限を超えない。候補の OR は捨てられない |
| 節に展開する | `src/fts_query.rs:126` `to_trigram_match` | 固定の語は `src/fts_query.rs:139` の繰り返しで全節に写される。候補の OR は `src/fts_query.rs:154` の `flat_map` で節を 25 倍にする。`src/fts_query.rs:177` で節を `OR` でつなぐ |
| 実行 | `src/search.rs:786` の `keyword search` → `src/search.rs:225` → `src/search/message_sessions.rs:131` `matched_rows` | `SELECT rowid, rank FROM messages WHERE messages MATCH ?1` に上の文を渡す |

## 手で書いた 3 語の AND より遅くなる理由

### 合成した索引での時間

`SELECT rowid, rank FROM messages WHERE messages MATCH ?`（`matched_rows` と同じ形）を、Python の `sqlite3`（SQLite 3.53.1）で 1 回空回しした後に 5 回実行し、中央値をとった。recall が使う bundled SQLite は 3.54.1（`libsqlite3-sys` 0.40.0）である。テストは debug プロファイルでビルドしたので、テストの中では時間を計らなかった。「`rank` なし」は `SELECT rowid FROM messages WHERE messages MATCH ?` で計った。

| MATCH の文 | 一致した行 | `rank` あり | `rank` なし |
| --- | --- | --- | --- |
| (a) recall が組み立てた文（25 節の OR） | 190 | 626.5 ms | 125.4 ms |
| (b) 手で書いた 3 語の AND `"GitHub" AND "Actions" AND "ワークフロー"` | 195 | 22.8 ms | 12.5 ms |
| (c) (b) に最も多い「の〜」の語を 1 つだけ足した文 | 18 | 32.4 ms | 計っていない |
| (d) 固定の語を OR の外に出した文 `(b) AND ("の引数" OR ... OR "の設定")` | 190 | 153.7 ms | 29.7 ms |
| (f) 「の」を後ろの語につないだ文 `"GitHub" AND "Actions" AND "のワークフロー"` | 0 | 4.5 ms | 計っていない |
| (参考) `"GitHub"` だけ | 64,945 | 108.2 ms | 計っていない |

合成した索引の中身は次のとおりである。

| 項目 | 値 |
| --- | --- |
| 行数 | 480,000（実インデックスの最大の rowid 482,615 に合わせた） |
| 本文 | 40 個の日本語の語を 10〜29 個並べ、間に助詞を挟む。助詞の 3 割を「の」にした |
| `GitHub` を含む行 | 64,945（13.5%、実インデックスの 65,221 に合わせた） |
| `Actions` を含む行 | `GitHub` を含む行の 12% |
| `ワークフロー` を含む行 | `Actions` を含む行の 2.5%、それ以外の行の 0.3% |
| 「の〜」の 3 文字の語 | 49 種類。上位 3 語はそれぞれ約 67,000 行に現れる |
| 「の〜」の上位 25 語の OR に一致する行 | 460,967（全体の 96%） |

### 遅くなる仕組み

1. 固定の語の位置リストを節の数だけ読む。(a) は `"GitHub"` と `"Actions"` と `"ワークフロー"` をそれぞれ 25 回含む。根拠は「`rank` なし」の時間である。(a) は 125.4 ms で、固定の語を 1 回だけ含む (d) は 29.7 ms だった。2 つは同じ「の〜」の 25 語を含み、一致する行も同じなので、約 96 ms の差は固定の語を 24 回多く読む時間と見ている。FTS5 の実装は読んで確かめていない。
2. 「の〜」の語は、ほぼすべての行に現れる。上位 25 語の OR は 96% の行に一致し、1 語あたり約 67,000 行の位置リストを持つ。「`rank` なし」で比べると、(d) は 29.7 ms、(b) は 12.5 ms で、約 17 ms の差がある。この差が「の〜」の 25 語の位置リストを読む時間である。`rank` ありの差（153.7 ms と 22.8 ms）の残りは、(d) の bm25 が 28 個の語句を数え、(b) が 3 個を数える差である。
3. `rank` の時間は語句の数に比例して増える。(a) は `rank` を外すと 626 ms から 125 ms になる。(b) は 23 ms から 13 ms になる。(a) の語句は 100 個、(b) は 3 個である。Issue の `sample` で `sqlite3Fts5PoslistNext64` が時間を使っていたことは、bm25 が一致した行ごとに各語句の位置リストを走査することと合う。この対応は時間の差から推した。SQLite の bm25 の実装は読んで確かめていない。

(a) と (b) の差の大部分（約 600 ms のうち約 500 ms）は `rank` の計算である。残りの約 110 ms が一致する行を探す時間である。

### 実インデックスの時間との関係

実インデックスでは手で書いた 3 語の AND が 279 ms、`keyword search` の段階が 1.5〜1.9 秒だった（Issue の計測）。比は約 5〜7 倍で、合成した索引の 27 倍より小さい。合成した本文は短く、語の種類も少ないので、比は一致しないと見ている。これは推測である。実インデックスで (a)〜(d) と (f) の時間と件数を計る段階は、Issue のとおり AI の担当者が行う。

### 検索の結果にも差がある

遅さとは別に、(a) は (b) より一致する行が少ない（190 行と 195 行）。「の〜」の上位 25 語のどれも含まない行は、`GitHub` と `Actions` と `ワークフロー` を含んでいても落ちる。25 語は索引全体の頻度で選ぶので、問い合わせの中の `のワー` が 25 語に入るとは限らない。この合成した索引では、`のワー` は語彙にあり、cnt は 397 である。上位 25 語の cnt はそれぞれ約 67,000 なので、`のワー` は `LIMIT 25` で落ちる。

## 直し方の候補と、それぞれで変わる検索の結果

時間は合成した索引で計った値である。どの候補も、実インデックスでの時間はまだわからない。

候補 2、3、4 は、DR 0010 の「2 文字以下の語を語彙で広げる」という振る舞いを、3 文字以上の語がある問い合わせで変える。選ぶときは DR 0010 の見直しの条件に当たるかを確かめる。候補 1 は一致する行を変えないが、bm25 の値を変えるので、`rank` を使う並び（DR 0009、DR 0016、DR 0018）への影響を確かめる。

| 候補 | 変える場所 | 一致する行 | 順位 | 合成した索引での時間 |
| --- | --- | --- | --- | --- |
| 1. 固定の語を OR の外に出す。展開を節の積ではなく `固定 AND (候補 OR ...)` の形で書く | `to_trigram_match`（`src/fts_query.rs:126`） | (a) と同じ集合（190 行で一致を確かめた） | 変わる。上位 10 件のうち 9 件が同じで、並び順は違った。bm25 が数える語句が 100 個から 28 個に減るためである | 626 ms → 154 ms |
| 2. 3 文字以上の語が 1 つでもあれば、2 文字以下の語を広げずに捨てる | `fts_expand_short_terms`（`src/fts_query.rs:620`）か `to_trigram_match` | (b) と同じ集合になり、増える（190 行 → 195 行）。「の〜」の上位 25 語を含まない行も一致する | 変わる | 626 ms → 23 ms |
| 3. 文字種の境目で分けた後、ひらがなだけの 2 文字以下の語（助詞）を捨てる | `segment_script_boundaries` か `prepare_match_query`（`src/fts_query.rs:292`、`src/fts_query.rs:320`） | この問い合わせでは候補 2 と同じ。漢字やカタカナの 2 文字の語（例: 「設定」）は今までどおり広げる | 変わる | この問い合わせでは候補 2 と同じ |
| 4. 2 文字以下の語を隣の語につないで 1 語にする（`のワークフロー`） | `segment_script_boundaries` | 減る。「の」の直後に `ワークフロー` がある行だけが一致する。`GitHub Actions ワークフロー` のように書いた行は落ちる | 変わる | 4.5 ms で 0 行だった。合成した索引では 397 行が「のワークフロー」を含む。しかし生成のコードは ` GitHub ` と ` Actions ` の後ろに空白を置くので、その 397 行はどれも `GitHub` と `Actions` を含まない。0 行は合成した本文の作り方による |
| 5. `VOCAB_LOOKUP_SQL` の `LIMIT 25` を減らす | `src/fts_query.rs:618` | 減る。候補の語が減るぶん、(a) よりさらに行が落ちる | 変わる | 計っていない。節と語句の数は `LIMIT` に比例して減る |

## 引き継ぐ確認

| 確認 | 担当 |
| --- | --- |
| この文書が、組み立てた文、ファイルと行、遅くなる理由、直し方の候補を書いているか | thkt が PR のレビューで確かめる |
| 実インデックスの写しで (a)〜(d) と (f) を計る。時間と件数だけを PR のコメントに書く | AI の担当者が、PR を公開した後、マージの前に行う |

### 実インデックスで計る手順

実インデックスの「の〜」の 25 語は合成した索引と違うので、(a) の文はその写しから取り出し直す。次の 2 つは文、件数、時間だけを出力し、会話の本文を出力しない。

1 つ目は、`src/fts_query/tests.rs` に一時的に足すテストである。`RECALL_542_DB` に写しのパスを渡して `cargo test --locked tmp_issue_542 -- --ignored --nocapture` で実行し、`MATCH=` の行をファイルに保存する。テストは計測の後に消す。`open_db` は写しの表の定義を確かめて書き換えることがあるので、写しのさらに写しに対して実行する。

```rust
#[test]
#[ignore]
fn tmp_issue_542_print_match() {
    let path = std::path::PathBuf::from(std::env::var("RECALL_542_DB").unwrap());
    let conn = crate::db::open_db(&path).unwrap();
    let built = prepare_match_query(&conn, "GitHub Actions のワークフロー").unwrap();
    println!("MATCH={}", built.to_trigram_match().unwrap());
}
```

2 つ目は、上の表の時間を計った Python のスクリプトである。`python3 -I time542.py <写しのパス> <MATCH の文だけを書いたファイル>` で実行する。写しは読み取り専用で開く。

```python
import sqlite3, sys, time, re, statistics
db, built_path = sys.argv[1], sys.argv[2]
built = open(built_path, encoding="utf-8").read().strip()
nos = re.findall(r'"(の[^"]*)"', built)
nos = list(dict.fromkeys(nos))
fixed = '"GitHub" AND "Actions" AND "ワークフロー"'
variants = {
    "a_built": built,
    "b_hand_and": fixed,
    "c_one_no": f'"GitHub" AND "Actions" AND "{nos[0]}" AND "ワークフロー"',
    "d_factored": fixed + " AND (" + " OR ".join(f'"{t}"' for t in nos) + ")",
    "f_no_phrase": '"GitHub" AND "Actions" AND "のワークフロー"',
    "g_github": '"GitHub"',
}
conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
print("sqlite", sqlite3.sqlite_version, "rows", conn.execute("select max(rowid) from messages").fetchone()[0])
results = {}
for sql_name, sql in [("rank", "SELECT rowid, rank FROM messages WHERE messages MATCH ?"), ("norank", "SELECT rowid FROM messages WHERE messages MATCH ?")]:
    for name, q in variants.items():
        if sql_name == "norank" and name not in ("a_built", "d_factored", "b_hand_and"):
            continue
        conn.execute(sql, (q,)).fetchall()
        ts = []
        for _ in range(5):
            t = time.perf_counter(); rows = conn.execute(sql, (q,)).fetchall(); ts.append(time.perf_counter() - t)
        results[(sql_name, name)] = rows
        print(f"{sql_name:7} {name:12} rows={len(rows):6} median_ms={statistics.median(ts)*1000:8.1f}")
a = results[("rank", "a_built")]; d = results[("rank", "d_factored")]
print("a==d set", {r[0] for r in a} == {r[0] for r in d})
top = lambda rs: [r[0] for r in sorted(rs, key=lambda r: r[1])[:10]]
print("top10 overlap a/d", len(set(top(a)) & set(top(d))), "same order", top(a) == top(d))
```
