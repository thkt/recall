---
status: "accepted"
date: 2026-06-24
decision-makers: thkt
---

# Classify Session Type by First-Turn Heuristic with a Fail-Open Default

## Context and Problem Statement

`recall search` excludes hook/script/agent-generated sessions from default results so an agent recalling past work is not buried in synthetic noise (#24). To do that, every session must carry a label: interactive (human-driven) or automated. recall derives that label from the session's first user turn with a literal marker heuristic (classify.rs:34-62) and persists it in `sessions.session_type`. Two facts make the heuristic incomplete by construction: it inspects only the first turn, and it matches a fixed set of prefixes with no regex or config (classify.rs:7). So the label can be wrong, and it can be absent entirely for sessions indexed before classification existed (NULL rows). The load-bearing question this ADR records is not the marker spelling (ADR-0002 freezes the token strings) but the direction recall errs when the label is uncertain or missing: which way does the classifier round off, and what does an agent see at search time when `session_type` is NULL? Those two stances are the non-derivable design choices, and they must agree, because a wrong default silently hides real sessions from recall.

## Decision Drivers

- A false positive (labeling a human session automated) hides a real session from default search, which directly defeats recall; a false negative (an automated session shown) is mere noise the agent can skim past (classify.rs:5-7)
- NULL `session_type` is not an error condition: every session indexed before #24 has no label, and re-classifying the whole DB is not a precondition for search to work
- The classifier producer literal (classify.rs:20-25) and the search-side default literal (search.rs:175) are not coupled by the compiler, so the default direction is an explicit, reviewable choice
- Markers are matched literally with no config surface, so the marker set is the entire classification contract and any gap is a silent mislabel

## Considered Options

- First-turn heuristic plus a search-side default that fails open to interactive (NULL and unmatched both stay findable)
- First-turn heuristic plus a default that fails open to automated (NULL coalesces to automated)
- First-turn heuristic with no default: treat NULL as a third state that errors or is excluded from default search

## Decision Outcome

Chosen option: "fail open to interactive", because it is the only direction under which a session whose classification is uncertain or absent stays visible in default recall. The classifier errs the same way at its own boundary: `classify_first_turn` returns `Interactive` whenever no marker matches, including an empty or whitespace-only first turn (classify.rs:44-62), so only prefixes a human is very unlikely to open a session with are admitted as automated markers (classify.rs:5-7,32-33). The search filter mirrors this with `COALESCE(s3.session_type, 'interactive') <> 'automated'` (search.rs:174-175), so a NULL row is read as interactive and survives the default exclusion. The decision is the combination: a classifier that rounds ambiguity toward interactive, and a default that rounds absence toward interactive, so the only sessions ever hidden are those a positive marker match proved automated.

### Consequences

- Good, because a session indexed before classification (NULL `session_type`) and a session with an unrecognized opener both remain in default search results: an agent never loses a real session to an uncertain label
- Good, because the asymmetry is intentional and one-directional: the cost of a miss is a little noise, never a hidden session, which matches the recall outcome
- Good, because no DB-wide re-classification is required for search correctness; old NULL rows behave as interactive immediately
- Bad, because an automated session whose opener is not in the marker set, or whose synthetic content begins after the first turn, is labeled interactive and pollutes default results until the marker set is extended
- Bad, because the producer literal (classify.rs:20-25) and the COALESCE default (search.rs:175) are two literals the compiler cannot keep in sync; their token spelling is governed by ADR-0002, but their agreement on direction is enforced only by review and the test below

### Confirmation

Three classifier tests pin the heuristic: marker-prefixed turns (with injected leading whitespace) classify automated (classify/tests.rs:5-19,21-27), and a human turn plus empty/whitespace turns classify interactive (classify/tests.rs:29-41). The fail-open default is pinned separately at the SQL layer: `test_default_search_excludes_automated_keeps_null` inserts one automated row and one NULL-`session_type` row, then asserts the automated row is excluded and the NULL row stays visible, exercising the `COALESCE(..., 'interactive')` branch (search/tests.rs:1429-1448). No test couples the producer literal and the default literal directly; review verifies they share the token per ADR-0002.

## More Information

### Relation to ADR-0002

ADR-0002 freezes the `"interactive"` / `"automated"` token spelling as a stable persisted/emitted contract. This ADR does not restate that freeze; it records the orthogonal decision of which token is the fail-open default and why (agent findability), and the contract of the heuristic that produces the token.

### Reassessment Triggers

- A new automated wrapper appears that the literal marker set does not cover, so genuine automated sessions classify interactive and pollute default search; extend `AUTOMATED_MARKERS` (classify.rs:34-42) and add a marker test
- An automated session is observed whose first turn is a benign human-like opener with synthetic content only later; first-turn-only classification would need to widen its scan window
- A real human opener is found to collide with a marker prefix (the single false-positive case the asymmetry forbids), which would force loosening or anchoring that marker
