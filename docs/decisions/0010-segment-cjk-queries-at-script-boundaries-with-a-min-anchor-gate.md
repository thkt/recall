---
status: "accepted"
date: 2026-06-24
decision-makers: thkt
---

# Segment CJK Queries at Script Boundaries With a Min-Anchor Gate

## Context and Problem Statement

The degraded (FTS-only, no embedder) search path tokenizes with an FTS5 `trigram` tokenizer, which has no word segmenter. A no-space natural-language CJK query is therefore one contiguous phrase that matches only a contiguous substring of indexed text. A query whose 助詞 were dropped (`textlint日本語ドキュメント校正`) never reaches a session whose message keeps them (`textlintで日本語ドキュメントを校正する`), because the trigram phrase is not a substring of the stored text (#167). Query-side normalization is deliberately disabled to stay byte-symmetric with the indexer (search.rs:614-617, #51), so the only lever is segmentation. But naive segmentation has a counter-failure: splitting an all-short query (`校正する` → `校正 する`) vocab-expands each run into an OR-group whose Cartesian product can exceed amici's `MAX_COMBOS`, at which point `clean_for_trigram` drops every OR-group to `fixed_only()` (amici/src/storage/fts.rs:52-53). With no fixed term that yields `None` → zero results, a regression below the raw contiguous-phrase match the unsegmented query already had. The decision is which segmentation rule recovers CJK recall without tipping into that overflow.

## Decision Drivers

- A no-space CJK query in degraded mode silently misses sessions a space-separated query would reach (#167); recall loss is invisible to the agent
- Query-side normalization is off by design (search.rs:614-617), so segmentation is the only available recall lever on the FTS path
- Over-segmentation is not free: each short run becomes an OR-group, and the product across groups is bounded downstream by amici's `MAX_COMBOS` (= 100, amici/src/storage/fts.rs:15)
- The vector path must stay on raw natural text — embedding wants the unsegmented query — so any split applies to FTS input only (search.rs:619-620)
- An ASCII technical token (`utf8`, `sha256`, `v2`) must never split at a letter↔digit transition (precision guard)

## Considered Options

- No segmentation: pass the raw query straight to FTS (pre-#167 behavior)
- Full n-gram / per-run expansion with no anchor gate: split at every script transition unconditionally
- Script-boundary split gated on a ≥3-char anchor run (chosen)

## Decision Outcome

Chosen option: "Script-boundary split gated on a ≥3-char anchor", because it is the only option that recovers the no-space CJK recall (#167) while guaranteeing a `fixed` phrase survives, so the segmented form is a strict recall superset of the raw query rather than a possible regression into `MAX_COMBOS` overflow.

The rule: `classify_script` maps each char to one of {Han, Hiragana, Katakana, Latin, Other}; ASCII letters and digits share the Latin class so `utf8`/`sha256` are never split (search.rs:537-551). `segment_script_boundaries` inserts a space at every transition among {Latin, Han, Hiragana, Katakana}, leaving a pure single-script run intact (splitting a kanji compound needs a dictionary, out of scope) and preserving double-quoted spans verbatim (search.rs:568-594). The gate then adopts the segmented form only when at least one resulting token is ≥3 chars; otherwise it keeps the raw query (search.rs:628-633):

```rust
let segmented = segment_script_boundaries(query);
let fts_input = if segmented.split_whitespace().any(|t| t.chars().count() >= 3) {
    segmented
} else {
    query.to_owned()
};
```

The threshold of 3 is the trigram floor: a ≥3-char run is the shortest token the trigram tokenizer can index as a `fixed` phrase, and that surviving fixed term is what keeps `clean_for_trigram` from collapsing to an empty `fixed_only()` when the OR-groups overflow `MAX_COMBOS` (amici/src/storage/fts.rs:36-54). The gate thus encodes recall's dependency on amici's combination cap: segment only when a fixed anchor will outlive the cap.

### Consequences

- Good, because a no-space CJK query reaches the same sessions as the space-separated form the user could have typed — byte-equivalent per-word phrases under implicit AND (search.rs:553-562)
- Good, because an all-short query (`校正する`) with no ≥3-char run keeps its raw contiguous-phrase recall instead of overflowing `MAX_COMBOS` into zero results (search.rs:621-627)
- Good, because the ASCII letter↔digit non-split and the verbatim quote handling keep precision on technical and explicit-phrase queries
- Bad, because the threshold is a code-review constant coupled to amici's `MAX_COMBOS`: if amici raises or removes the cap, the gate's necessity changes but nothing tool-enforces the link across the two crates
- Bad, because a pure single-script kanji compound (`機械学習日本語処理`) still has no transition to split on, so dictionary-segmentable recall remains out of scope

### Confirmation

Eight tests in src/search/tests.rs cover the rule, the gate and the cap, each through `search`.

- `a_no_space_query_splits_between_kanji_and_hiragana` checks that a han↔hiragana split reaches a message whose two runs are not adjacent.
- `a_letter_digit_token_is_matched_whole` checks the letter↔digit precision guard: `sha256` does not reach a message holding `sha` and `256` apart.
- `a_kanji_compound_in_a_query_is_matched_whole` checks that a single-script kanji run stays whole: `Rust機械学習` does not reach a message holding `機械` and `学習` apart.
- `a_quoted_cjk_span_is_not_split_into_separate_terms` checks that a quoted span stays verbatim: `"日本語ドキュメント"` does not reach a message holding `日本語` and `ドキュメント` apart.
- `a_no_space_query_across_scripts_finds_the_same_session_as_the_spaced_query` checks end-to-end #167 recall across Latin, Han and Katakana runs, plus parity with the spaced form.
- `a_short_kanji_and_hiragana_query_finds_the_session_when_many_words_start_with_its_parts` checks that the gate keeps raw-phrase recall for `校正する`. Its split form expands to 16 × 16 = 256 combinations, so the test detects a missing gate only while `MAX_COMBINATIONS` stays below 256.
- `a_hiragana_word_with_a_prolonged_sound_mark_finds_the_session` checks the gate on `うーん`, whose split form expands to 6 × 6 × 5 = 180 combinations. Without the gate, the search finds nothing while the cap is below 180 and also finds a decoy at 180 or more.
- `a_query_whose_short_parts_match_too_many_words_is_searched_by_its_long_part_alone` checks the cap itself: `Rust校正する` overflows it, so only the anchor `Rust` is searched. The test fails if `MAX_COMBINATIONS` reaches 256 or the overflow check is removed.

## More Information

### Reassessment Triggers

- `MAX_COMBINATIONS` in src/compat.rs changes (raised, lowered, or removed): the ≥3-char anchor gate's framing of "a fixed term survives the cap" no longer holds and must be re-derived against the new value
- A way is found to recover all-short no-space queries without `MAX_COMBINATIONS` overflow: the gate currently declines to segment them (no ≥3-char token), keeping the raw query, so a dropped-particle all-short query like `校正する` against indexed `校正をする` still misses — a residual #167-class miss the gate trades for overflow safety, not a solved case
- A pure single-script CJK segmenter (dictionary or model) is adopted: the script-boundary-only rule becomes a subset, and the no-anchor fallback for compounds like `機械学習日本語処理` can be revisited
- The prolonged-sound-mark stranding class of bug recurs (ー/U+30FC classifies as Katakana, splitting うーん into sub-trigram runs): currently absorbed by the same no-anchor fallback (`a_hiragana_word_with_a_prolonged_sound_mark_finds_the_session`), but a recurrence under a different script edge would signal the gate is masking, not fixing, a classification gap rather than the intended #167 recall win

### Scope Note

The segmentation and the min-anchor gate above apply to plain queries only. A query containing an uppercase `AND`, `OR` or `NOT` outside quotes takes a separate path that skips both, as recorded in [DR 0015](0015-search-uppercase-fts5-operators-on-a-separate-path.md).

The cap moved from amici's `MAX_COMBOS` into this crate as `MAX_COMBINATIONS` in src/compat.rs. In Context, Decision Drivers, Decision Outcome and Consequences, read each `MAX_COMBOS` as `MAX_COMBINATIONS`. Read each amici/src/storage/fts.rs path as src/compat.rs, where `clean_for_trigram` now lives. The fts.rs line numbers do not carry over. amici's `fixed_only()` has no counterpart by name. On overflow, `clean_for_trigram` skips every OR-group and keeps only the fixed terms of 3 or more characters. It returns `None` when none remain. The search.rs line numbers in those sections are from the decision date. The code they cite is now in `classify_script`, `segment_script_boundaries` and `keyword_query` in src/search.rs.

The Bad consequence that nothing tool-enforces the link to the cap no longer holds. `a_query_whose_short_parts_match_too_many_words_is_searched_by_its_long_part_alone` fails if the cap reaches 256, the value at which the `校正する` gate test stops detecting a missing gate.
