---
status: "accepted"
date: 2026-11-11
decision-makers: thkt
scope: "src/search.rs candidate selection: the keyword-only path and the hybrid path of search_with_embedder; the file-list path and the filter helpers are out of scope"
---

# Rank Every Match Before Cutting Search Results to the Limit

## Context and Problem Statement

`recall search --limit 1` could return a different session from the first result of the default limit (#416). The search cut its candidates to `3 x limit` first, and only then applied the recency blend (and, in hybrid search, the RRF fusion). The cut did not account for the score transforms that came after it. A session that the blend or the fusion would have lifted to the front never reached those steps.

The two paths failed for different reasons.

- **Keyword-only path.** The candidate SQL ended in `ORDER BY best_rank, session_id LIMIT 3 x limit`. A bm25 value is not bounded by a count, so with many sessions matching within a few percent of each other, a new session whose raw rank sat past the cut was dropped for any multiplier. The blend can raise a score by at most `1 + RECENCY_BOOST_WEIGHT` (1.2, `src/hybrid.rs`), but the raw ranks of the competing sessions can be arbitrarily close.
- **Hybrid path.** Both legs were cut to `3 x limit`, so the RRF score of a session depended on `limit`. A session at keyword rank 91 and vector rank 6 and a session at keyword rank 1 only swap order between `limit 1` and `limit 10`, because the cut changes which legs list each session and at what position (`rrf_merge_sessions`, `src/hybrid.rs`).

The recency blend is the cross-path invariant of DR 0009. This record adds the rule about where the cut sits relative to it. The decision is how deep each leg must read so that a smaller limit returns the head of a larger limit's results.

## Decision Drivers

- The result of `--limit N` must equal the first N results of any larger limit, or an agent that asks for one result gets a different answer than an agent that asks for ten
- The keyword leg has no count that makes the cut safe, because the transform that follows it (a multiplier up to 1.2) acts on a value the count does not bound
- The fused score must not depend on `limit`, or two limits order the same two sessions differently
- `recall search` must stay within 2 seconds (`.claude/OUTCOME.md` Time indicator), so reading more rows needs a measurement, not an assumption
- A session that is skipped (no `sessions` row, unknown source, unreadable row) must not silently shrink the results; the existing `results_incomplete` signal has to keep meaning something the caller can act on

## Considered Options

- Apply the recency blend inside SQL, so the `LIMIT` sees the blended score
- Keep the cut and raise the multiplier
- Make the hybrid depth depend on `limit` (read deeper as `limit` grows), and keep a cut on each leg
- Read every keyword match, fuse and blend the whole set, and cut last; read the same vector depth at every limit (chosen)

## Decision Outcome

Chosen option: "Read every keyword match, fuse and blend the whole set, and cut last", because it is the only option that makes the cut come after every transform for the keyword leg, and it keeps the fused score independent of `limit` for the vector leg.

The rule: **truncate candidates to `limit` only after every later score transform has run.** The transforms are the recency blend and, in hybrid search, the RRF fusion. Any new transform added after the cut must move the cut behind it, or bound its effect and account for that bound the way `results_incomplete` does below.

What follows from the rule:

- **Keyword-only path.** `build_keyword_session_query` has no `LIMIT` and joins `sessions` in the same read, so one pass returns every matching session with its metadata. `rank_sessions` blends all of them, sorts by `(blended rank, session_id)`, and cuts to `limit`. Excerpts are fetched (`fetch_chunks`) only for the sessions that survive the cut.
- **Hybrid path, keyword leg.** `rrf_merge_sessions` receives every keyword session in the original `(best_rank, session_id)` order. A skipped session still takes its position in that order, so the sessions after it keep the RRF rank they had before.
- **Hybrid path, vector leg.** Every search reads `VEC_CANDIDATE_COUNT` nearest chunks, three per result at the largest limit: `3 x MAX_LIMIT = 300`. `MAX_LIMIT` (100) is also the bound that `recall search` puts on `--limit`. Every limit therefore reads the same chunks, and the fused scores do not change with `limit`.
- **Ceiling of 4096.** sqlite-vec rejects a vec0 query with `k > SQLITE_VEC_VEC0_K_MAX` (4096, `sqlite-vec-0.1.9/sqlite-vec.c:7111`). A compile-time assertion keeps `VEC_CANDIDATE_COUNT` at or below it. Raising `MAX_LIMIT` above 1365 then fails the build instead of the vector leg.
- **300 is the smallest count that gives every limit three chunks per result.** A smaller count gives the largest limits fewer. A larger count reads more chunks without changing any result.
- **Hybrid search costs 45 ms more at `--limit 1`.** On a synthetic index of 6,000 sessions and 24,000 chunks embedded with the real model, a query matching about 90% of the sessions took 0.788 s at p50 with this change. `main` took 0.743 s; at that limit it reads 3 keyword candidates and 3 chunks. Each build ran 7 times, alternated. About 18 ms of the difference is the full keyword read, which took 0.134 s against 0.116 s on the keyword-only path. The share of the floor alone was not measured. At `--limit 100`, where both read 300 chunks, the two took 0.792 s and 0.790 s.
- **`results_incomplete` means a skipped session could have ranked.** The flag is set when a row could not be read at all, because its rank cannot be estimated. It is also set when a skipped session could still reach the `limit`-th result. Its raw score times the largest recency factor it could get must tie or beat the cut, or the results must have a free slot. A session with an unknown source still has a readable `sessions` row, so its own date gives that factor. A session with no `sessions` row or unreadable metadata has no known date. It counts with `MAX_RECENCY_FACTOR` (`1 + RECENCY_BOOST_WEIGHT`, next to `recency_decay` in `src/hybrid.rs`), which can only over-flag, never miss. Hybrid search runs the same check, `left_out_session_could_rank`, on the fused scores of sessions pruned for missing metadata.

### Consequences

- Good, because `--limit N` returns the head of a larger limit on both paths, at every limit `recall search` accepts
- Good, because a session lifted by the blend or the fusion is no longer cut beforehand, and a skipped session no longer takes one of the `limit` slots
- Good, because `results_incomplete` (and the `degraded` flag and note that `src/commands/search.rs` derives from it) fires only when the omission could change the visible results, so an agent can trust it when it is set
- Bad, because the keyword leg reads every matching session before cutting. A broad query over a large index reads and sorts all matches where it used to read `3 x limit`. On the index above without a model, the query took 0.134 s at p50 at `--limit 1` against 0.116 s on `main`. The cost grows with the number of matches; see the second Reassessment Trigger
- Bad, because `MAX_RECENCY_FACTOR` couples the flag to DR 0009's non-negative, at-most-1.0 `recency_decay`. If the decay can exceed 1.0, the bound underestimates and a session that could rank is not flagged
- Bad, because a session with no readable date counts with the largest factor. It can raise the flag when its real date would have kept it out of the results

### Confirmation

Tests in `src/search/tests.rs` run each path against a real FTS5 database through `search` or `search_with_embedder`.

Keyword-only path:

- `a_limit_of_one_returns_a_new_session_that_matches_slightly_worse_than_older_ones` fails if the cut returns to before the blend.
- `a_limit_of_one_returns_the_newest_of_sessions_whose_text_matches_equally` fails if tied raw ranks are cut before the blend separates them.
- `results_for_a_smaller_limit_are_the_first_results_of_a_larger_limit` fails if the head of a larger limit differs from a smaller limit's results.
- `an_unknown_source_session_ranked_first_does_not_leave_a_limit_of_one_empty` fails if a skipped session takes a slot.

`results_incomplete`:

- `an_unknown_source_session_that_could_not_reach_the_results_leaves_them_complete` fails if the flag is set for a skipped session below the cut.
- `an_unknown_source_session_too_old_to_reach_the_results_leaves_them_complete` and `an_unknown_source_vector_match_too_old_to_reach_the_results_leaves_them_complete` fail if a session with an unknown source counts with the largest factor instead of its own date.
- `a_session_missing_from_the_sessions_table_marks_the_results_incomplete_when_it_could_rank_first` and `a_vector_match_missing_from_the_sessions_table_marks_the_results_incomplete_when_it_could_rank_first` fail if a session with no `sessions` row is judged without the largest factor.
- `a_session_with_unreadable_metadata_that_matches_best_is_skipped_and_flags_incomplete` fails if a session whose `sessions` row cannot be decoded is dropped without being counted.
- `an_unknown_source_session_that_could_not_reach_the_hybrid_results_leaves_them_complete` fails if hybrid search flags a pruned session whose fused score cannot reach the results.
- `run_search_is_not_degraded_when_a_skipped_session_could_not_reach_the_results` in `src/commands/search/tests.rs` fails if such a session marks the envelope degraded or adds a note.

Hybrid path:

- `hybrid_search_with_a_limit_of_one_returns_a_new_session_ranked_fourth_by_text` and `hybrid_search_with_a_limit_of_one_returns_a_new_session_ranked_fourth_by_vector` fail if either leg is cut by `limit` before fusion.
- `hybrid_search_with_a_limit_of_one_returns_the_first_result_of_a_limit_of_ten` fails if the fused order depends on `limit` for limits 1 through 4.

No test covers the 4096 ceiling; the compile-time assertion on `VEC_CANDIDATE_COUNT` guards it. The DR 0009 `debug_assert!(rank < 0.0)` stays in `rank_sessions`, so the FTS sign guard applies to the new read path with the same limit: it is compiled out of release builds.

## More Information

### Alternatives Not Chosen

- **Apply the blend in SQL.** It fixes the keyword path only. The hybrid fusion happens in Rust after two separate reads, so a SQL-side blend cannot reach it. It also needs rusqlite's `functions` feature and a registered function on every connection that searches, while `Cargo.toml` enables only `bundled`.
- **Raise the multiplier.** It cannot fix the keyword path. A bm25 value is not bounded by a count, so for any multiplier a large enough set of near-tied sessions pushes the new session past the cut. The cost of reading grows with the multiplier and the failure stays.
- **Let the hybrid depth follow `limit`.** The fused score then changes with `limit`. A session at keyword rank 91 and vector rank 6 and a session at keyword rank 1 swap order between `limit 1` and `limit 10`, so no limit's results are a prefix of a larger limit's. The chosen design reads the same depth at every limit.

### Related

- Issue: #416. The cause was found in review of #414, whose tie-order change (b2aa6b1) was reverted because it did not fix near-ties and made recency count twice in hybrid search
- DR 0009: the recency blend, its sign invariant, and the `recency_decay` bound that `MAX_RECENCY_FACTOR` relies on. This record does not change that invariant and adds where the cut sits relative to the blend
- DR 0009 names `score_and_sort`, which is now `rank_sessions`. The keyword-only path now orders ties only in `rank_sessions`. So `sessions_that_match_equally_at_the_same_time_are_listed_in_session_id_order` fails if that tie-break is reversed, but not if the SQL tie-break in `build_keyword_session_query` is. The SQL order still sets each session's keyword position in hybrid search, and `a_text_match_and_a_vector_match_that_fuse_equally_are_listed_in_session_id_order` fails if it is reversed
- `docs/wiki/fts-vec-two-leg-symmetry.md`: a change to candidate selection applies to both legs. Its fourth step now states this record's rule for both legs

### Reassessment Triggers

- `MAX_LIMIT` changes from 100: `VEC_CANDIDATE_COUNT` follows it, so measure the hybrid cost again at the new count
- A broad keyword query exceeds the 2-second Time indicator on a 6,000-session index: add an early stop that does not read rows below the blended `limit`-th rank, which Issue #416 lists as a backlog candidate
- A new score transform is added after the cut, or `recency_decay` can return more than 1.0: move the cut behind the transform or recompute `MAX_RECENCY_FACTOR`, and recheck `results_incomplete` against the new bound
- sqlite-vec changes `SQLITE_VEC_VEC0_K_MAX`: update the compile-time assertion on `VEC_CANDIDATE_COUNT`
