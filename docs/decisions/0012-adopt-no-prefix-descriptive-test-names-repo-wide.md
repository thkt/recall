---
status: "accepted"
date: 2026-07-19
decision-makers: thkt
---

# Adopt No-Prefix Descriptive Test Names Repo-Wide

## Context and Problem Statement

recall's test suite carries two naming styles side by side. The legacy style prefixes every test function with `test_` and packs the spec into a snake_case identifier (`test_stale_wal_note_immutable_nonempty_wal_is_some`, src/db/tests.rs:452). The newer style drops the prefix and names the function as a descriptive sentence that states what the test verifies (`stale_wal_note_on_a_read_only_mount_steers_to_the_copy_based_recall_index_db_path_remedy_instead_of_a_bare_recall_index`, src/db/tests.rs:862; `error_code_numbers_match_sysexits_baseline`, src/error/tests.rs; `write_output_appends_newline_when_missing`, src/output/tests.rs). The two styles are split almost evenly: of 441 `#[test]` functions across the tree, 220 keep the `test_` prefix and 221 do not. A PR #264 audit surfaced the mixed styles, and #267 / PR #287 documented the convention for one module in the src/db/tests.rs:1-9 module doc comment (new tests take no `test_` prefix; existing names are renamed only when the test is next touched, rename-only, asserts unchanged). That doc comment binds only src/db, and PR #287 listed repo-wide application and any bulk rename as explicitly out of scope. So the repo has a documented stance in one file and no recorded stance anywhere else, leaving a reviewer of a new test in src/search or tests/ with no single source to cite for which style a new test must follow. This ADR promotes the src/db module rule to repo scope unchanged, so the naming stance is one decision for the whole tree rather than a per-module folk convention.

The current per-module distribution, measured by counting `#[test]` attributes against `fn test_` definitions:

| Module               | `test_` prefix | no-prefix | total |
| -------------------- | -------------: | --------: | ----: |
| src/ansi             |              2 |         0 |     2 |
| src/chunker          |             18 |         0 |    18 |
| src/classify         |              3 |         0 |     3 |
| src/date             |              6 |         0 |     6 |
| src/db               |             25 |        16 |    41 |
| src/embedder         |              8 |         0 |     8 |
| src/envelope         |              0 |        11 |    11 |
| src/error            |              0 |        17 |    17 |
| src/hybrid           |              6 |         0 |     6 |
| src/indexer          |             41 |         5 |    46 |
| src/main             |             25 |       116 |   141 |
| src/output           |              0 |         7 |     7 |
| src/parser           |              8 |         0 |     8 |
| src/parser/claude    |              9 |         0 |     9 |
| src/parser/codex     |              7 |         0 |     7 |
| src/search           |             62 |        13 |    75 |
| tests/ (integration) |              0 |        36 |    36 |
| Total                |            220 |       221 |   441 |

The distribution is not random: the oldest, rarely-touched modules (chunker, embedder, the parser trees, ansi, classify, date, hybrid) are still 100% `test_`, the newest surfaces (error, output, envelope, and every integration test under tests/) are 100% no-prefix, and the four actively-churned modules (db, indexer, main, search) are mid-migration. That shape is what opportunistic migration looks like partway through, and it is the reason a repo-wide stance is worth recording rather than left to per-file drift.

## Decision Drivers

- A test's name is read far more often in `cargo nextest run` output than in source; a name that states the spec ("`write_output_appends_newline_when_missing`") is legible at the point of failure without opening the file, and the `test_` prefix adds nothing there because nextest already reports the function as a test
- The convention already exists, is documented at module scope, and half the tree already follows it (221 of 441); the open question is only whether it is the repo's stance or one file's habit, and leaving that unanswered means every new-test PR re-litigates style
- A one-shot rename of all 220 prefixed tests would be a 220-function, 16-module diff of pure churn: expensive to review, a magnet for rebase conflicts against every in-flight branch, and it rewrites `git blame` on test bodies that did not change
- PR #287 already chose opportunistic migration for src/db and deferred the repo-wide call; ratifying the same rule at repo scope keeps one direction rather than inventing a second

## Considered Options

- Adopt no-prefix descriptive names as the repo-wide convention and migrate legacy names opportunistically, renaming a `test_` function to a no-prefix descriptive name only when that test is next touched, rename-only with asserts unchanged (chosen, extends the src/db rule from #267 / PR #287 to repo scope)
- Adopt no-prefix descriptive names repo-wide and migrate in one big-bang sweep now, renaming all 220 prefixed tests in a single dedicated PR
- Standardize on the legacy `test_` prefix repo-wide, renaming the 221 already-migrated no-prefix tests backward to carry the prefix

## Decision Outcome

Chosen option: "adopt no-prefix descriptive names repo-wide with opportunistic migration", because it commits the whole tree to the style the newer half already uses and the src/db doc comment already prescribes, while paying the migration cost incrementally on files a change already opens rather than in one churning sweep. The rename-only, asserts-unchanged constraint keeps each migration a pure name change: a reviewer confirms the diff touches only the function signature line and no assertion, so renaming a test can never silently alter what it checks.

### Consequences

- Good, because a new test now has one documented stance for the whole repo; a reviewer cites this ADR instead of arguing style per PR, and the answer is the same in src/search as in tests/
- Good, because there is no big-bang churn: `git blame` on test bodies stays meaningful, no 16-module rename PR has to be reviewed or rebased, and the migration cost lands on files an unrelated change already touches
- Good, because the chosen name states the spec the test verifies, so a failure line in `cargo nextest run` reads as a sentence about the broken behavior rather than a prefixed identifier
- Bad, because both styles persist as the steady state, not a transient: a test that is never touched keeps its `test_` name indefinitely, so the tree can sit at a mix like today's 220 / 221 for a long time and never reach uniformity on its own
- Bad, because the convention and its rename-only constraint are code-review rules, not compiler- or lint-enforced; nothing stops a new `test_`-prefixed test from being added, or an assert change from being folded into a rename, other than a reviewer catching it

### Confirmation

Enforced by code review against this ADR. A new `#[test]` function in a diff must carry no `test_` prefix and must name the spec it verifies; a reviewer rejects a newly-added `fn test_...`. A migration that renames a legacy test must be rename-only: the diff shows only the function signature changed and the assertion body byte-identical, so a rename that also edits an assert is split into two changes. Progress is measurable without judgment: `grep -rc 'fn test_' src tests` against the total `#[test]` count (currently 220 of 441) trends monotonically toward zero prefixed as touched files migrate, and a rising prefixed count signals the new-test rule was breached.

## More Information

The mixed styles were first flagged as a PR #264 audit finding. #267 / PR #287 ("docs(db/tests): document the test-naming convention in a module doc comment", commit 014014e) wrote the rule into the src/db/tests.rs:1-9 module doc comment and, in its own scope note, listed "repo-wide application of the convention" and "bulk rename of existing `test_` names" as out of scope. This ADR is the deferred repo-wide decision: it does not change the rule PR #287 wrote, only the scope it binds, from one module to the tree.

### Reassessment Triggers

- A lint or CI gate that can assert no-prefix on newly-added tests becomes available (a custom clippy lint, or a diff-scoped grep gate on added `fn test_` lines); at that point the new-test half of the constraint can move from a review rule to an enforced gate and this ADR's Confirmation should be updated to reference it
- The team decides the indefinite mixed-style steady state is not worth its ambiguity and schedules the big-bang sweep (the rejected second option) as a dedicated, behavior-frozen, rename-only PR; that PR supersedes the opportunistic-migration half of this decision while keeping the no-prefix target
