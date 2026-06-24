---
status: "accepted"
date: 2026-06-24
decision-makers: thkt
---

# Preserve the Recency-Blend Sign Invariant Across Ranking Paths

## Context and Problem Statement

recall blends a recency component into ranking scores so that a newer session sorts ahead of an equally relevant older one. The blend is applied in two separate code paths whose base scores have opposite sign conventions, and both call the same decay function. The FTS-only path multiplies a negative bm25 rank: `let blended_rank = rank * (1.0 + RECENCY_BOOST_WEIGHT * recency_boost)` (search.rs:344), then sorts ascending (search.rs:348), so amplifying a negative rank's magnitude moves a newer session earlier. The hybrid/vector path multiplies a positive RRF score: `*score *= 1.0 + weight * boost` (hybrid.rs:33), then sorts descending (hybrid.rs:35), so amplifying a positive score moves a newer session earlier. Both call `hybrid::recency_decay` (search.rs:343, hybrid.rs:32), which returns 1.0 at age 0, decaying toward 0.0 and never negative, with `None` timestamp mapping to 0.0 (hybrid.rs:11-19).

The decay term is always non-negative, so the recency contribution is a `(1.0 + w * decay)` multiplier on the base score in both paths. The direction the recency term pushes a result therefore depends entirely on the sign of the base score paired with the sort direction. The FTS path is correct only while bm25 ranks stay negative; if a base were positive there, the same multiplier would push newer sessions the wrong way under the ascending sort. The two paths must agree that "newer moves toward the front", and that agreement is unenforced by any type: it rests on the documented sign convention of each path's base score.

## Decision Drivers

- The recency blend must move a newer session the same direction (toward the front of the sort) in every ranking path, or a search returns stale-first results in one path and fresh-first in the other for the same query
- The decay function is non-negative, so direction is carried by the base-score sign and the sort order, not by the decay term, making the invariant invisible from `recency_decay` alone
- The two paths use opposite sign conventions (negative bm25 ascending vs positive RRF descending), so a single shared multiplier cannot be copied between them without also matching the sign and sort
- A base score of exactly 0.0 nullifies the multiplicative blend entirely (`0.0 * anything == 0.0`), removing recency from that candidate

## Considered Options

- Keep per-path blend expressions that each pair the shared `recency_decay` with that path's sign convention, and document the cross-path sign invariant in this ADR
- Extract one shared recency-blend function as a single source of truth that both paths call
- Encode the sign in the type system, so a negative-rank path and a positive-score path are distinct types the compiler keeps from mixing

## Decision Outcome

Chosen option: "Keep per-path blend expressions and document the cross-path sign invariant", because the two paths share the decay term but not the sign convention or the sort direction, so a single shared blend function would have to take the sign and sort as parameters and would not remove the real coupling: that newer must move forward in both. The decay computation is already shared (`recency_decay`); only the sign-bearing multiply and sort differ, and each lives next to the path's own sort so the pairing is locally visible. Encoding the sign in the type was rejected as more indirection than a two-site invariant warrants.

### Consequences

- Good, because each blend sits next to its own sort, so a reader checks the sign-and-sort pairing in one place per path rather than threading a shared signed wrapper
- Good, because the shared `recency_decay` keeps the half-life, weight semantics, and the `None`-timestamp = 0.0 behavior identical across paths, so only the sign-bearing multiply can diverge
- Bad, because the cross-path sign agreement is a documented contract, not a compiler- or release-enforced one: an edit that flips a base-score sign in one path, or introduces a third ranking path with a different convention, can silently invert recency there without breaking the build
- Bad, because the only live guard against a sign flip in the FTS path is a `debug_assert!`, which the release profile compiles out (see Confirmation)

### Confirmation

The FTS path carries `debug_assert!(rank < 0.0, "FTS bm25 rank must be negative; rank={rank} would nullify the recency blend or invert ordering")` (search.rs:339-342). This is the only in-code guard tying the FTS base-score sign to the blend's correctness, and it is the incomplete part of the enforcement: `debug_assert!` is compiled out in release builds, so a release binary running on an index where a bm25 rank reached 0.0 or turned non-negative would silently nullify or invert the FTS recency blend with no panic. There is no release-active check and no `debug_assert` on the hybrid path's positive-score assumption at all.

No single test asserts cross-path sign equality. The behavior is covered per path: T-119 asserts the FTS path sorts an equal-rank newer session first (search.rs:846-857), T-119b asserts a `None`-timestamp candidate sorts last there (search.rs:864-872), and `test_hybrid_recency_boost_favors_newer` asserts the hybrid path ranks the newer session first (search.rs:1242-1281). Each test pins one path's direction; none runs both paths on one fixture and asserts they agree. The real confirmation this ADR's invariant is upheld is therefore still missing: a unit test that drives both paths over identical timestamped candidates and asserts newer-first in each, or a release-active assertion on each path's base-score sign.

## More Information

### Reassessment Triggers

- The FTS bm25 ranks can reach 0.0 or turn non-negative on a real index (the `debug_assert` premise breaks, and because it is compiled out of release the blend silently nullifies or inverts there): file a follow-up to replace the debug-only guard with a release-active check or to clamp/normalize the base score
- A third ranking path is added (a different fusion, a re-rank stage, a new base-score convention): its recency blend must be checked to move newer sessions toward the front under its own sort, against this ADR, before merging
- A contributor proposes extracting the shared signed blend or encoding the sign in the type: revisit the chosen option, since a second sign-divergence site would shift the cost balance
- The non-negative contract of `recency_decay` changes (a decay that can return a negative value would let the multiplier reduce, not amplify, magnitude and would break both paths at once)
