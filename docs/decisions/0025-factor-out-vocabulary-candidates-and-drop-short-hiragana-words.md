---
status: "accepted"
date: "2026-11-28"
decision-makers: "thkt"
scope: "How a plain (no uppercase operator) query becomes an FTS5 MATCH string: the hiragana rule in fts_expand_short_terms and the shape MatchFtsQuery::to_trigram_match writes (src/fts_query.rs)"
---

# Factor Out Vocabulary Candidates and Drop Short Hiragana Words

## Context and Problem Statement

A plain query that mixes Latin and Japanese, such as `GitHub Actions のワークフロー`, took 3.5 to 9.6 seconds in the full-text search (#542). DR 0010 splits it at script boundaries into `GitHub Actions の ワークフロー`. `fts_expand_short_terms` then expands the one-character `の` into the 25 most frequent vocabulary terms that start with it. `MatchFtsQuery::to_trigram_match` copied the three fixed terms into each of the 25 clauses, so the MATCH string was an OR of 25 AND clauses holding 100 phrases. `matched_rows` (src/search/message_sessions.rs) selects `rank`, and bm25 scores every phrase for every matched row. The same expansion also loses rows. A row that holds `GitHub`, `Actions` and `ワークフロー` but none of the 25 `の` terms does not match. DR 0010 compared the split form only against the unsplit query (`"GitHub" AND "Actions" AND "のワークフロー"`), so this loss is outside the comparison DR 0010 made. How should the plain path write the MATCH string so that the search time approaches a hand-written AND of the long terms, and the rows lost to the `の` expansion match again?

## Decision Drivers

- The OR of 25 clauses spends most of its time in `rank`
  - Verified. Copy of the real index, Python sqlite3 3.53.1, load average 20 to 25, 2026-11-28 (PR #543 comment). The 25-clause OR matched 185 rows in 3,532.7 to 9,626.3 ms with `rank` and 545.8 ms without it. The factored form matched 185 rows in 237.3 ms with `rank` and 38.4 ms without. The hand-written AND of the three long terms matched 211 rows in 77.7 to 122.8 ms with `rank` and 18.3 ms without
- Expanding a particle next to long words drops rows the long words alone would find
  - Verified. Same measurement. The 25-clause OR matched 26 fewer rows (185) than the hand-written AND (211)
- A hiragana-only word of one or two characters next to a word of three or more characters is treated as a particle
  - By policy. thkt's agreement on 2026-11-28 (#544)
- Short kanji, katakana and Latin words (`設定`) carry meaning and must keep narrowing the result
  - By policy. thkt's agreement on 2026-11-28 (#544)
- The plain path must keep DR 0010's behavior when no word reaches three characters (`校正する`, `うーん`), and keep the cap that searches by the long terms alone on overflow
  - By policy. DR 0010, and #544's out-of-scope list

## Considered Options

- Factor the candidate groups out of the fixed terms, and drop a hiragana-only word of one or two characters when the query has a word of three or more characters
- Factor the candidate groups out only
- Drop every word of one or two characters when the query has a word of three or more characters
- Join a short word to the next word (`のワークフロー`)
- Lower the `LIMIT 25` of `VOCAB_LOOKUP_SQL`
- Keep the product of clauses

## Decision Outcome

Chosen option: "Factor the candidate groups out of the fixed terms, and drop a hiragana-only word of one or two characters when the query has a word of three or more characters", because it is the only option that removes both the repeated fixed terms and the particle's rows-losing expansion while keeping short kanji, katakana and Latin words as filters.

By distributivity, `(A AND x1 AND B) OR (A AND x2 AND B)` equals `A AND (x1 OR x2) AND B`, so factoring out leaves the set of matched rows unchanged. Every term is quoted by `fts_quote` and every group is parenthesized, so operator precedence cannot change the meaning.

The two rules are as follows.

| Rule | Where | Behavior |
| --- | --- | --- |
| Factor out | `MatchFtsQuery::to_trigram_match` | Writes `fixed AND (candidate OR candidate ...) AND fixed ...` in query order. Each fixed term appears once. A group with one candidate is written bare, without parentheses. Fixed terms under three characters are dropped, a group whose candidates are all under three characters is skipped, and on overflow every group is dropped, all as before |
| Drop short hiragana | `fts_expand_short_terms` | Before expanding, checks whether any `Piece::Word` or `Piece::Phrase` of the query has three or more characters (lowercase `and` and `not` count). If one does, a `Piece::Word` of one or two characters whose every character `classify_script` puts in `Script::Hiragana` is dropped without a vocabulary lookup. A quoted `Piece::Phrase` is never dropped. `ー` classifies as Katakana, so `のー` is still expanded |

For `GitHub Actions のワークフロー`, dropping `の` leaves no candidate group, and the MATCH string becomes `"GitHub" AND "Actions" AND "ワークフロー"`. Factoring out therefore shows on queries such as `GitHub Actions 設定`.

`MAX_TRIGRAM_COMBINATIONS` stays at 100 and still compares the product of the group sizes. After factoring out, the string grows with the sum of the group sizes, so the cap no longer bounds the size of the string. It stays only to keep DR 0010's behavior of searching by the long terms alone on overflow.

### Consequences

- Good, because a mixed-script query with a particle is searched as the AND of its long terms. On the real index copy that AND took 77.7 to 122.8 ms with `rank`, against 3,532.7 to 9,626.3 ms for the 25-clause OR
- Good, because `GitHub Actions のワークフロー` is expected to match 211 rows instead of 185. The rows lost because they hold none of the top 25 `の` terms match again
- Good, because a query with a short kanji or katakana word keeps narrowing by that word's candidates while each fixed term is read once
- Bad, because bm25 scores fewer phrases, so `rank` values change and the order of results can change. On the synthetic index with factoring out alone, 9 of the top 10 rows were the same and their order differed (docs/research/issue-542-mixed-script-keyword-latency.md). The order with both rules applied has not been measured. DR 0009, DR 0016 and DR 0018 decide how ranks are used, not their values, so those rules stand
- Bad, because a query whose short hiragana word was meant as a content word (`GitHub すし`) no longer filters by it
- Bad, because `count_excluded_automated` (src/search.rs) also goes through `keyword_query`, so the note counting excluded automated sessions changes in the same way

### Confirmation

Tests in src/fts_query/tests.rs and src/search/tests.rs cover the decision, run by `cargo nextest run --profile ci --locked`.

- `the_match_string_writes_each_fixed_word_once_and_each_candidate_group_as_one_or_group_in_query_order` compares the whole string, covering a one-candidate group without parentheses, a dropped short fixed term and a skipped all-short group.
- `the_match_string_keeps_candidate_groups_at_100_combinations_and_drops_them_at_101` pins the overflow boundary of `MAX_TRIGRAM_COMBINATIONS`.
- `the_factored_match_string_finds_the_same_sessions_as_the_product_of_its_clauses` runs both forms on the bundled SQLite against the production schema and compares the sets of sessions. A test-only helper keeps the old product of clauses.
- `a_query_with_a_long_word_finds_a_session_without_any_word_that_starts_with_its_short_hiragana_word` checks the recovered rows. It failed before the change.
- `a_query_with_a_long_word_narrows_by_the_words_that_start_with_its_short_kanji_word` checks that `設定` is still expanded.
- `a_query_whose_short_parts_match_too_many_words_is_searched_by_its_long_part_alone` checks the cap. Its query changed from `Rust校正する` to `Rust 校正 設定`, because dropping `する` would keep `Rust校正する` under the cap. The test seeds 15 words starting with `校正` and 15 starting with `設定`, and the matching message adds one trigram for each, so the query expands to 16 × 16 = 256 combinations. It passes with the cap at 255 and fails at 256 (checked on 2026-11-28). This keeps DR 0010's link: at a cap of 256 or more, the `校正する` gate test stops detecting a missing gate, and this test fails first.

## Pros and Cons of the Options

### Factor the candidate groups out only

Rewrite `to_trigram_match` and keep expanding `の`.

- Good, because the matched rows stay exactly the same
- Bad, because it still loses the rows that hold none of the top 25 `の` terms, and still scores 25 extra phrases
  - Verified. Real index copy, 2026-11-28: 185 rows in 237.3 ms with `rank`, against 211 rows in 77.7 to 122.8 ms for the AND of the long terms

### Drop every word of one or two characters when the query has a word of three or more characters

Candidate 2 of the research document.

- Good, because it gives the AND of the long terms for every mixed query
- Bad, because a short kanji or katakana word such as `設定` stops filtering, so `GitHub 設定` would match every session that mentions GitHub
  - By policy. thkt chose on 2026-11-28 to keep expanding short kanji, katakana and Latin words (#544)

### Join a short word to the next word

Candidate 4 of the research document: search `のワークフロー` as one phrase.

- Good, because the string stays short
- Bad, because only rows where `の` directly precedes `ワークフロー` match, so rows written as `GitHub Actions ワークフロー` are lost
  - Verified. Synthetic index of 480,000 rows, 2026-11-28: 0 rows matched (docs/research/issue-542-mixed-script-keyword-latency.md)

### Lower the `LIMIT 25` of `VOCAB_LOOKUP_SQL`

Candidate 5 of the research document.

- Good, because the OR shrinks in proportion to the limit
- Bad, because fewer candidates can only lose more rows than the 25-clause OR already does
  - Verified. Read `VOCAB_LOOKUP_SQL` on 2026-11-28: a smaller `LIMIT` returns a prefix of the same `ORDER BY cnt DESC, term` list, so every group shrinks to a subset

### Keep the product of clauses

Leave the MATCH string as an OR of AND clauses.

- Good, because the `rank` values and the order of results stay as they are
- Bad, because the search takes seconds and loses 26 rows on the example query
  - Verified. Real index copy, 2026-11-28: 185 rows in 3,532.7 to 9,626.3 ms with `rank`

## More Information

- thkt chose "外に出す + 助詞を捨てる (Recommended)" on 2026-11-28 (session d27b59e1-c914-436a-a388-128379db2207), and #544 records that agreement
- #542 and PR #543 investigated the cause, and docs/research/issue-542-mixed-script-keyword-latency.md holds the synthetic measurements. The real-index measurements above are from the PR #543 comment
- The bundled SQLite is 3.54.1 (`libsqlite3-sys` 0.40.0). The real-index measurements used Python sqlite3 3.53.1. `the_factored_match_string_finds_the_same_sessions_as_the_product_of_its_clauses` runs the factored form on 3.54.1
- The time and row counts on the real index copy for `GitHub Actions のワークフロー` and `GitHub Actions 設定`, before and after this change, are measured on the PR's release build and on v0.18.0 after the PR is published and before it merges (#544)
- This DR adds to [DR 0010](0010-segment-cjk-queries-at-script-boundaries-with-a-min-anchor-gate.md) without superseding it. DR 0010's gate still decides whether the split form is used, and this DR changes what happens to the short words of an adopted split form

### Reassessment Triggers

- The order of results changes in a way users notice after release: bm25 now scores fewer phrases, and the combined effect on order was not measured
- A content word written in hiragana of one or two characters turns out to be searched often next to a long word: the drop rule would then hide a filter the user meant
- `MAX_TRIGRAM_COMBINATIONS` changes, or the cap test's 16 × 16 = 256 link breaks: re-derive the link with DR 0010's gate tests
- `VOCAB_LOOKUP_SQL`'s `LIMIT 25` changes: the factored string's length and the bm25 phrase count follow it
