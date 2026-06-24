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

Four tests in src/search/tests.rs cover the rule and the gate: `segment_script_boundaries_splits_multi_script_run` (script-transition splits incl. han↔hiragana, tests.rs:1802-1814), `segment_script_boundaries_keeps_ascii_alnum_whole` (the letter↔digit non-split precision guard, tests.rs:1816-1826), `degraded_no_space_cjk_query_reaches_session` (end-to-end #167 recall + parity with the spaced form, tests.rs:1848-1894), and `segment_falls_back_to_raw_when_no_anchor_term` (the anchor gate: an all-short query at corpus scale keeps raw-phrase recall instead of overflowing `MAX_COMBOS`, tests.rs:1895-1941). amici's cap itself is covered by `falls_back_to_fixed_only_when_combos_exceed_max` (amici/src/storage/fts/tests.rs:55). The cross-crate coupling (recall's gate assumes amici's cap value) is a code-review constraint, not a compile-time one.

## More Information

### Reassessment Triggers

- amici changes `MAX_COMBOS` (raises, lowers, or removes it): the ≥3-char anchor gate's framing of "a fixed term survives the cap" no longer holds and must be re-derived against the new value (amici/src/storage/fts.rs:15)
- A way is found to recover all-short no-space queries without `MAX_COMBOS` overflow: the gate currently declines to segment them (no ≥3-char token), keeping the raw query, so a dropped-particle all-short query like `校正する` against indexed `校正をする` still misses — a residual #167-class miss the gate trades for overflow safety, not a solved case
- A pure single-script CJK segmenter (dictionary or model) is adopted: the script-boundary-only rule becomes a subset, and the no-anchor fallback for compounds like `機械学習日本語処理` can be revisited
- The prolonged-sound-mark stranding class of bug recurs (ー/U+30FC classifies as Katakana, splitting うーん into sub-trigram runs): currently absorbed by the same no-anchor fallback (tests.rs:1943-1948), but a recurrence under a different script edge would signal the gate is masking, not fixing, a classification gap rather than the intended #167 recall win
