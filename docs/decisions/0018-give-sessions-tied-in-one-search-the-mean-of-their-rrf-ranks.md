---
status: "accepted"
date: "2026-11-16"
decision-makers: "thkt"
scope: "src/hybrid.rs rrf_merge_sessions and the two lists src/search.rs passes to it; the keyword-only path and its tie-break in rank_sessions are out of scope"
---

# Give Sessions Tied in One Search the Mean of Their RRF Ranks

## Context and Problem Statement

Hybrid search fuses the keyword search and the vector search by Reciprocal Rank Fusion (RRF). Each session found by a search adds `1 / (60 + rank)` to its fused score, where `rank` is its position from 0 in that search's list. #414 lists exact ties in session id order in both searches. So two sessions that one search scores the same still got different ranks there. For example, a text tie gave the session with the smaller id `1/60` and the other `1/61`. The id order of two sessions that one search cannot tell apart then moved their fused order (#415).

## Decision Drivers

- A: a session outside a tie keeps the fused score it has today
- B: a large tie gains no total weight over the positions it spans
- C: a tie keeps about the weight its positions carry, so it neither gains nor loses against untied sessions
- D: each session keeps one rank in each search, so the fused score stays RRF over ranks
- The order among tied sessions must not move any fused score

## Considered Options

- Positional ranks: each session takes its own position, as before this record
- Competition ranks (1, 2, 2, 4): tied sessions take the position of the first of them
- Dense ranks (1, 2, 2, 3): tied sessions take one rank, and the next session takes the next rank
- Last-position ranks (1, 3, 3, 4): tied sessions take the position of the last of them
- Mean ranks (1, 2.5, 2.5, 4): tied sessions take the mean of the positions they span
- Mean contributions: tied sessions take the mean of `1 / (60 + p)` over the positions `p` they span

## Decision Outcome

Chosen option: "Mean ranks", because it meets A, B and D. It also meets C for the ties a search meets in practice. A tie of ten sessions at the top loses 0.2% of its weight, and a tie of two loses 0.01%.

- `rrf_merge_sessions` receives each search as `(session id, score)` pairs listed best first. A lower score ranks higher in both searches. The keyword score is the best bm25 rank of the session, and the vector score is the distance of its closest chunk.
- `tied_ranks` groups the sessions with the same score and gives each group the mean of its positions.
- Ties are detected with `==`. Only exactly equal scores tie.
- A `debug_assert` checks that each list is sorted best first. The grouping depends on tied scores sitting next to each other.
- The keyword ranks are computed over every matching session, skipped ones included. A skipped session keeps its position, as DR 0016 requires, and the sessions after it keep their ranks.
- `vec_search` returns the distance with each hit (`VecHit`).
- The final tie-break of `rrf_merge_sessions` stays. Sessions whose fused scores are exactly equal are listed in session id order.

### Consequences

- Good, because the order among tied sessions no longer moves a fused score. The other search decides their order.
- Good, because a session outside any tie keeps its rank and its fused score.
- Bad, because fusion collects both lists and groups them. Fusing 20,000 keyword sessions and 300 vector hits took 3.12 ms against 2.83 ms before this record.
- Bad, because a large tie loses weight, since `1 / (60 + rank)` is convex. A tie of a hundred sessions at the top loses 7.4% of the weight its positions carry.
- Bad, because a near-tie still gets distinct ranks. Two bm25 ranks that differ in the last bit are ranked apart.
- Bad, because a tie that the vector search's `LIMIT` cuts gets its mean from the rows that survive the cut. sqlite-vec still picks those rows, since vec0 orders a nearest-neighbor search by distance alone.

### Confirmation

`sessions_tied_in_one_search_share_the_mean_of_the_positions_they_span` in `src/hybrid/tests.rs` compares the fused scores of a tie at positions 1 and 2. It fails with positional, competition, dense or last-position ranks, and when a tie loses its offset.

Tests in `src/search/tests.rs` run hybrid search through `search_with_embedder` with `MockEmbedder`.

- `sessions_that_match_the_text_equally_are_ordered_by_the_vector_search` fails with positional ranks. The two sessions then fuse to the same score, and the session id decides.
- `sessions_found_equally_by_the_vector_search_are_ordered_by_the_text_search` fails with positional ranks for the same reason.
- `three_equal_text_matches_share_the_middle_rank_and_follow_the_first_vector_match` fails with positional, competition or dense ranks.
- `a_text_match_and_a_vector_match_that_fuse_equally_are_listed_in_session_id_order` fails if the final session id tie-break is reversed or removed. Each search's five sessions share a rank there, so all ten fuse to the same score.

## Pros and Cons of the Options

### Positional ranks

- Good, because it needs no score in the fusion.
- Bad, because the id order among tied sessions moves their fused order.

### Competition ranks

- Good, because a session after a tie keeps its rank, so it meets A.
- Bad, because every tied session takes the first position of the tie. Ten sessions with identical text would each add the first-place contribution, so it fails B.

### Dense ranks

- Bad, because every session after a tie moves up, so it fails A.
- Bad, because tied sessions take the first position, so it fails B.

### Last-position ranks

- Good, because a session after a tie keeps its rank, and a tie gains no weight, so it meets A and B.
- Bad, because every tied session takes the last position of the tie. The tie loses weight against the positions it spans, so it fails C.

### Mean ranks

- Good, because it meets A, B and D.
- Bad, because a tie loses weight as it grows: 0.2% for ten sessions, 7.4% for a hundred.

### Mean contributions

- Good, because a tie keeps exactly the weight its positions carry, so it meets A, B and C.
- Bad, because a tied session has no single rank in its search, so it fails D. Its gain over mean ranks is under 0.2% for a tie of up to ten sessions.

## More Information

### Overturned Statements

DRs are not edited after acceptance, so these statements stay in place. They no longer hold after this record.

- DR 0009, Confirmation: `a_text_match_and_a_vector_match_that_fuse_equally_are_listed_in_session_id_order` "also fails if the FTS or vector tie order is reversed". Reversing a search's tie order no longer moves fused ranks. The tied sessions of each search now share a rank, so only the final session id tie-break orders them.
- DR 0009, Confirmation: "The vector leg of the hybrid path lists equally distant sessions in session id order too, so their fused ranks follow that order." The vector search still lists them in that order, but they now share a rank, so the order does not set their fused ranks.
- DR 0009, Confirmation: `sessions_found_equally_by_the_vector_search_are_listed_in_session_id_order` "fails if that order is reversed". The tied sessions now share a rank, so the final session id tie-break alone sets their order.
- DR 0016, Related: `a_text_match_and_a_vector_match_that_fuse_equally_are_listed_in_session_id_order` "fails if it is reversed", for the SQL tie-break in `build_keyword_session_query`. That SQL order still sets the order of the keyword list. Tied sessions in that list now share a rank, so their order does not move the fused score.

### Related

- Issue: #415. Its fifth site is `rrf_merge_sessions`. #461 fixed its four other sites, where SQLite or sqlite-vec decided a tie.
- DR 0009: the recency blend multiplies the fused score after this fusion. This record does not change that blend.
- DR 0016: the cut to `limit` still comes after the fusion and the blend.
- `docs/wiki/fts-vec-two-leg-symmetry.md`: both searches take their ranks from the same `tied_ranks`.

### Reassessment Triggers

- A fusion other than RRF replaces `rrf_merge_sessions`.
- A search is added whose score cannot be compared for equality, or whose list is not sorted best first.
- A test or a user report shows near-ties that should rank as ties. Grouping scores within a tolerance would then replace `==`.
- Ties of many sessions become common in one search, for example when automated sessions are searched by default. Mean contributions would then keep their weight.
