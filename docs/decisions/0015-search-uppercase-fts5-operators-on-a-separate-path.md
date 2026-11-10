---
status: "accepted"
date: 2026-11-10
decision-makers: thkt
scope: "search query handling: queries containing uppercase AND / OR / NOT; plain-word queries stay on the DR 0010 path"
---

# Search Uppercase FTS5 Operators on a Separate Path

## Context and Problem Statement

Quoted phrases, `AND` / `OR` / `NOT`, and terms containing `-` or `:` returned zero sessions even when a message contained them verbatim (#411). `fts_expand_short_terms` split the sanitized query on whitespace and wrapped every piece in `fts_quote`, so a quote became a literal character (`"token-rotation"` searched as `"""token-rotation"""`), an operator became a search term (`hello OR x` searched as `"hello" AND "OR" AND "x"`), and a phrase split in two at its space.

The README promises this syntax, so the query must reach the trigram index with its operators intact. But the plain path cannot carry them. The CJK segmentation and anchor gate of DR 0010 insert spaces at script transitions and read the sanitized query back through `clean_for_trigram`, and neither step understands an operator tree. The short-term vocab expansion (the prefix match that lets a term under 3 characters find indexed words starting with it) expands a term into an OR-group, which cannot sit under a `NOT`. The decision is how a query with operators reaches FTS5 without disturbing the results of the plain-word queries that work today.

## Decision Drivers

- A plain-word query must return the same sessions as before; the CJK recall of DR 0010 depends on that path staying as it is
- The trigram tokenizer indexes nothing shorter than 3 characters, so a short term has no meaning inside an operator expression
- Hybrid search merges two legs with RRF (`rrf_merge_sessions`), so a NOT applied to one leg only lets the other leg return the excluded sessions again (`docs/wiki/fts-vec-two-leg-symmetry.md`)
- The embedder wants natural text; `AND`, `OR`, `NOT` and the terms after a `NOT` would pull the query vector toward what the user excluded
- Lowercase `and` / `or` / `not` appear in ordinary sentences, so treating them as operators would change the results of queries that work today

## Considered Options

- Repair the plain path: make `fts_expand_short_terms` pass phrases and operators through, and keep segmentation and vocab expansion on every query
- Hand the query to FTS5 as typed, with no sanitizing or tree
- Detect uppercase operators on the raw query and send those queries down a separate path that parses a tree and renders the MATCH string (chosen)

## Decision Outcome

Chosen option: "Detect uppercase operators and take a separate path", because it is the only option that keeps every plain-word query byte-identical to today while giving operators their FTS5 meaning. Repairing the plain path would put segmentation and vocab expansion in front of an expression they cannot read. Typed-as-is hands FTS5 syntax errors and column filters (`role:user`) to the user.

The rules:

- **Operators are uppercase tokens.** `AND`, `OR` and `NOT` count as operators only when they stand alone outside quotes (`tokenize_operator_query`). Lowercase `and` / `or` / `not` and any operator inside `"..."` stay ordinary words. `prepare_operator_query` returns `None` when the query has no operator token, and the caller keeps the plain path.
- **The operator path is separate.** It parses a tree (`OR` below `AND` below `NOT`, parentheses outermost, dangling operators ignored, open parentheses closed at the end) and renders the final MATCH string without CJK segmentation, the min-anchor gate, or vocab expansion. Each term or phrase is quoted once with `fts_quote`. The result skips `clean_for_trigram`. The plain path also quotes a phrase and a `-` / `:` term once, as part of the same fix.
- **Short terms follow their position in the tree.** A term under 3 characters is dropped under `AND`, drops its branch under `OR`, and drops the whole NOT clause on the right of `NOT`. A short term on the left of `NOT`, or a `NOT` with nothing on its left, is `RecallError::QuerySyntax` (exit 65) with its own hint listing the supported syntax. A query whose every term is short renders no MATCH and runs the vector leg only.
- **NOT is applied on both legs.** The keyword leg renders `(left NOT right)`, which FTS5 evaluates per message. The vector leg cannot evaluate an FTS5 expression, so `drop_vec_only_sessions_matching_not` removes each session found only by the vector search when any of its messages matches a recorded right side. Sessions the keyword leg returned already satisfy the whole expression.
- **Only a NOT that covers the whole query records its right side.** `Node::render` records it for a NOT reached from the root only through `AND` or the left side of another `NOT`. A NOT under an `OR`, or inside another NOT's right side, records nothing. Its exclusion holds inside one branch only, so applying it to every vector-only session would drop sessions the query does not exclude. This holds even when short terms leave the `OR` one branch in the MATCH, because the embedder still searches for the dropped branch's terms.
- **The embedder sees the positive terms only.** `OperatorQuery::embed_text` joins the terms outside every NOT right side, with no operators.
- **`search_with_embedder` and `count_excluded_automated` build the keyword query through one function** (`keyword_query`), so the excluded-automated count and the result list agree on which sessions a query matches.

### Consequences

- Good, because phrases, `-` / `:` terms, and `AND` / `OR` / `NOT` with parentheses find the sessions that contain them, and plain-word queries keep their current results
- Good, because a NOT cannot be undone by the vector leg: a session containing the excluded word is dropped from the merge whichever leg found it
- Good, because the embedding input carries no operator and no excluded term, so the vector leg searches for what the user asked to find
- Bad, because an operator query gets no CJK segmentation and no short-term expansion; `東京 AND 日本語ドキュメント校正する` searches each term as one contiguous trigram phrase, the pre-DR 0010 behavior, and a 2-character term is dropped instead of prefix-matched
- Bad, because `AND` and `NOT` are evaluated per message on the keyword leg but `NOT` is evaluated per session on the vector-only filter: `alpha NOT beta` returns a session whose message A has `alpha` and whose message B has `beta` when the keyword leg finds it, yet drops that session when only the vector leg finds it
- Bad, because the uppercase rule is a convention to learn: a user who types lowercase `not` gets an ordinary-word search, not an error
- Bad, because an uppercase `AND`, `OR` or `NOT` inside literal text, such as `404 NOT FOUND` or `IS NOT NULL`, becomes an operator. The user must quote such text to search it as written
- Bad, because a NOT under an `OR`, or inside another NOT's right side, does not filter vector-only sessions. `(alpha NOT beta) OR gamma` can return a vector-only session containing `beta`. The vector leg under-filters here rather than dropping sessions the query allows

### Confirmation

Tests in `src/search/syntax_tests.rs` run the user-visible behavior against a real FTS5 database: `lowercase_and_or_not_in_a_query_are_matched_as_ordinary_words` (the plain path keeps lowercase operators as words), `an_explicit_and_between_two_words_finds_the_message_containing_both`, `an_or_finds_a_message_containing_either_word`, `parentheses_group_an_or_before_the_and_that_follows_it`, `a_not_excludes_messages_containing_the_word_after_it`, `a_short_word_after_not_is_ignored_and_the_rest_of_the_query_still_matches`, `a_short_word_before_not_is_rejected_as_a_query_syntax_error`, `a_not_with_nothing_before_it_is_rejected_as_a_query_syntax_error`, `a_session_found_only_by_the_vector_search_is_dropped_when_it_matches_the_word_after_not`, `a_session_found_only_by_the_vector_search_stays_when_it_does_not_match_the_word_after_not`, `a_session_found_only_by_the_vector_search_stays_when_the_not_sits_under_an_or`, and `a_session_found_only_by_the_vector_search_stays_when_it_matches_a_not_nested_after_another_not` (the NOT on the vector leg).

Tests in `src/storage/search/tests.rs` pin the rendered MATCH strings: `operator_query_is_absent_without_uppercase_operators`, `operator_query_gives_and_precedence_over_or_and_parentheses_over_both`, `operator_query_drops_short_terms_by_tree_position`, `operator_query_rejects_a_not_without_a_usable_left_side`, `operator_query_keeps_not_operands_out_of_the_embedding_text`, `operator_query_records_a_not_operand_only_when_its_exclusion_covers_the_whole_query`, and `operator_query_ignores_dangling_operators_and_closes_open_parentheses`.

## More Information

- Issue: #411
- Related: DR 0010 (the plain path keeps its segmentation and anchor gate), DR 0005 (FTS hits resolve to chunks by rowid range), `docs/wiki/fts-vec-two-leg-symmetry.md`
- User-facing description: the "Query syntax" table in `README.md` and `README.ja.md`

### Reassessment Triggers

- A user reports an operator query that misses a CJK or 2-character-term session the plain path would find: the missing segmentation and vocab expansion in the operator path becomes worth the work of making them read the tree
- A user reports `NOT` returning a session that contains the excluded word in another message: move the keyword leg to a session-level NOT, which also removes the per-message versus per-session difference between the legs
- `AND` is requested across messages of one session: the per-message evaluation of FTS5 no longer matches what users expect, and both operators need a session-level evaluation
- amici or SQLite changes how FTS5 evaluates `NOT` or parenthesized groups on the trigram tokenizer: the rendered `(left NOT right)` form must be rechecked against the rendering tests
