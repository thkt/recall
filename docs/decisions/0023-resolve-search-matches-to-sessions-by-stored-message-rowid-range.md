---
status: "accepted"
date: 2026-11-25
decision-makers: thkt
scope: "How `recall search` finds the session of each full-text match (src/search.rs, src/search/message_sessions.rs), and what that makes of the sessions rowid-range columns from DR 0019"
---

# Resolve Search Matches to Sessions by Stored Message Rowid Range

## Context and Problem Statement

`messages` is an FTS5 table whose `session_id` column is UNINDEXED. FTS5 keeps that value in the same `messages_content` row as the text. Each search read `session_id` for every matched row, for four purposes.

- the best rank per session (`MIN(rank) ... GROUP BY session_id`)
- the matching messages of the top sessions (`session_id IN (...)`)
- the NOT filter on vector-only sessions
- the count of automated sessions left out by default

Every filter (`--project`, `--days`, `--source`, `--file`, the current session, and the default exclusion of automated sessions) was also added to the match as `session_id IN (SELECT ... FROM sessions ...)`.

Matched rows lie all over the file. A query that matches tens of thousands of rows therefore reads most of `messages_content` (934 MB on thkt's index) from disk on the first search after a pause. The `.claude/OUTCOME.md` Time indicator asks for `recall search` within 2 s. How should a search learn the session of a matched row without reading the row?

## Decision Drivers

- `.claude/OUTCOME.md`: `recall search` returns within 2 s
  - Verified. Issue #531 measured on thkt's v0.17.0 index (2026-11-25, Apple M3, Python SQLite 3.53.1, cache dropped with `msync(MS_INVALIDATE)`): query A matched 71,858 rows; reading rowid alone took 13 ms, and reading `session_id` took 1,752 ms. Hybrid search p95 was 3.57 s
- Search results stay the same: sessions, order, excerpts, and the count and wording of `notes`. The one exception is a match whose `session_id` is not UTF-8 text
  - By policy. Issue #531 scope
- Search opens the index read-only and cannot add columns, so an index built before the range columns must still be searchable
  - By policy. DR 0019 and DR 0007
- No schema change: the `messages` table, the place of `session_id`, and the indexes on `sessions` stay
  - By policy. Issue #531 out-of-scope list

## Considered Options

- Resolve each match from the stored rowid ranges, looked up in memory, with a per-row read for doubtful rows
- Keep reading `session_id` for every matched row
- Join each match to `sessions` in SQL by `rowid BETWEEN message_rowid_lo AND message_rowid_hi`
- Take the session whose range starts last at or before the rowid, without checking for overlap
- Move `session_id` out of the FTS5 table

## Decision Outcome

Chosen option: "Resolve each match from the stored rowid ranges", because it reads only rowid and rank from the full-text index for the common row, and it falls back to the stored `session_id` wherever the ranges cannot name exactly one session.

`MessageSessions::load` (src/search/message_sessions.rs) reads every session's `message_rowid_lo` and `message_rowid_hi` once per search. It sweeps them into disjoint segments, each tagged with its sole owner or with none, and finds a rowid's segment by binary search. Empty ranges, such as `EMPTY_MESSAGE_RANGE`, own no rowid.

- A rowid inside exactly one range belongs to that range's session
- A rowid inside no range, or inside two or more, is resolved by `SELECT session_id FROM messages WHERE rowid = ?`
- The whole search reads `session_id` with each match, and maps rows to sessions only that way, in any of these cases. Filtering, ranking, excerpts, and counts still take the same path as with ranges
  - `sessions` has no `message_rowid_lo` column
  - Any session has a NULL `lo` or `hi`. A row of such a session can lie inside another session's range, so the ranges are dropped for the whole search, not per session
  - Any session row has a `session_id` that is not UTF-8 text, or a `lo` or `hi` that is not an integer
- A matched row whose `session_id` is an integer, a real, a blob, or text that is not valid UTF-8 counts as an unreadable row. It sets `unreadable_rows`, which adds the "results may be incomplete" note, whatever filters are active. The search does not fail, and the automated count leaves the row out. Before this change, the same row failed the whole search with `DataError` under `--include-automated` and was dropped silently by the SQL filters otherwise. thkt agreed on 2026-11-25 not to keep that behavior

Both modes produce the same list of `(rowid, rank, session_id)`, and the rest of the search runs on that list.

- the best rank per session, sorted by rank, then by session id
- the best and the first matching message of each top session, for the excerpt
- the NOT filter
- the automated count

The filters now run once over the matched session ids instead of once per matched row. They pass the ids through `json_each` and reuse `append_search_filters` on `j.value`. A NULL session id and a session without a `sessions` row therefore meet the same SQL conditions as before.

`search_with_embedder` and `count_excluded_automated` each run inside one read transaction (`unchecked_transaction`, deferred). Ranges and matches then come from one snapshot. Otherwise, a `recall index` that commits between the two reads could delete a session and give its rowids to a new one; FTS5 does not use AUTOINCREMENT.

### Consequences

- Good, because the first search after a pause no longer reads `messages_content` for each match whose session the ranges name. On thkt's index every one of the 393,586 rows lay in exactly one range, as Issue #531 checked
- Good, because an index from before v0.17.0 and one with NULL ranges keep the old per-row read, so they return the same results without an upgrade
- Good, because the matches are read once per search. The excerpt lookup reuses them instead of running a second MATCH
- Bad, because DR 0019's statement that "only the writer reads" the range columns no longer holds. `schema_state` still does not check the columns; search detects their absence by itself
- Bad, because DR 0019's unenforced invariant now decides search correctness, not only delete speed. If a message lies outside its own session's range and inside exactly one other range, search attributes it to that other session and returns a wrong result without an error. If a message without a session row lies inside exactly one range, search attributes it to that session and stops counting it as missing
- Bad, because each search reads the whole `sessions` table for the ranges, and the filter step passes every matched session id to SQLite as one JSON array
- Neutral, because a backfilled range that spans other sessions' rows (DR 0019, Neutral consequence) makes those rows lie in two ranges, and they fall back to the per-row read

### Confirmation

- `stored_ranges_return_the_same_sessions_order_and_excerpts_for_every_filter_and_not` (src/search/range_tests.rs) runs keyword and hybrid searches with every filter and NOT, first with NULL ranges and then with recorded ranges, and compares the sessions, order, excerpts, and flags
- `a_row_that_lies_only_in_another_sessions_range_is_taken_as_that_sessions_row` fails when search stops using the ranges; it pins the misattribution described under Consequences
- `a_row_in_two_overlapping_ranges_belongs_to_its_stored_session` and `a_rowid_inside_two_ranges_has_no_sole_owner` fail when an overlapped rowid is given to one of its ranges
- `an_index_without_the_range_columns_opened_read_only_returns_the_same_results` drops both columns and searches through `open_db_readonly`
- `a_search_reads_the_ranges_and_every_match_from_the_index_as_it_was_when_it_began` fails when the read transaction is removed: a write committed during the query embedding becomes visible to the later NOT filter
- `a_match_whose_session_id_is_not_utf8_text_marks_the_results_incomplete_with_or_without_the_automated_filter` and `a_search_whose_only_match_has_a_session_id_that_is_not_utf8_text_succeeds_and_marks_the_results_incomplete` fail when such a row fails the search or the automated count
- `a_session_row_that_cannot_hold_a_range_makes_the_search_read_every_session_id` fails when a blob `session_id` or a non-integer range in `sessions` makes the search fail

Code review checks that a new reader of `messages` that needs a session goes through `MessageSessions`, and that a new writer of `messages` records the range, as DR 0019 already requires.

## Pros and Cons of the Options

### Resolve each match from the stored rowid ranges

Read rowid and rank, and name the session from the ranges loaded once per search.

- Good, because it uses columns that the indexer already writes
- Bad, because a broken range invariant yields wrong results silently

### Keep reading `session_id` for every matched row

The behavior of v0.17.0.

- Good, because it rests on no invariant
- Bad, because reading `session_id` costs one `messages_content` read per match, which exceeds the 2 s indicator on the first search after a pause
  - Verified. Issue #531: 1,752 ms for query A against 13 ms for rowid alone, and hybrid p95 3.57 s, on thkt's index on 2026-11-25

### Join each match to `sessions` in SQL by `rowid BETWEEN lo AND hi`

Let SQLite find the range of each match.

- Good, because it needs no code outside SQL
- Bad, because it reads `sessions` once per match. Checking that no other range overlaps a rowid needs every range with `lo <= rowid`, even with an index on `lo`, and adding that index is out of scope
  - By policy. Issue #531 rules out a per-row scan of `sessions` and an index on `sessions`

### Take the session whose range starts last at or before the rowid

A binary search on `lo` alone.

- Good, because it is simpler than the sweep into segments
- Bad, because a rowid in two overlapping ranges goes to one of them instead of to its stored `session_id`
  - Verified. A variant that gives a rowid to the first open range, run on 2026-11-25, failed `a_row_in_two_overlapping_ranges_belongs_to_its_stored_session` and `a_rowid_inside_two_ranges_has_no_sole_owner`

### Move `session_id` out of the FTS5 table

Keep the session of each message in an ordinary indexed table.

- Good, because the session of a match would never depend on ranges
- Bad, because it rewrites the table that every search and every index run reads, with an upgrade of the whole full-text index
  - By policy. Issue #531 lists schema changes to `messages` as out of scope

## More Information

- Issue #531 (thkt/recall) holds the requirement, agreed with thkt on 2026-11-25
- The filter step uses `json_each` (https://sqlite.org/json1.html#jeach) from the SQLite that `rusqlite` 0.43 bundles
- This record adds search as a second reader of `message_rowid_lo` and `message_rowid_hi`. It does not change how the indexer writes or deletes by them, and it does not supersede DR 0019. A search on an index whose ranges are NULL or missing reads as before
- This change does not confirm that search returns within 2 s right after the cache is dropped on a real index. thkt checks it after release with `scripts/measure-search.py --drop-from-memory`
- critic-design (2026-11-25, during scoping of #531): weakened; holds provided the filters are applied after the rowid-to-session mapping, an index without the range columns or with any NULL range resolves every match by its session_id, the sole-owner check runs in Rust over ranges loaded once, and the range load and the MATCH share one read transaction.

### Trade-offs

The per-row read needs no invariant but costs one random read per match. The ranges cost nothing per match but turn a broken invariant into a wrong result. The fallback for overlapping and unowned rows, and for indexes with NULL or missing ranges, limits that exposure to the one case the indexer does not create: a row outside its own range and inside exactly one other range.

### Reassessment Triggers

- A writer other than `upsert_session` inserts into `messages`, or a path deletes a session row before its messages
- `messages` moves to an external-content table, or `session_id` leaves the FTS5 table
- A search on a real index after the cache is dropped still exceeds 2 s, with the full-text step as the main cost
