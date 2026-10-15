use super::*;

#[test]
fn fts_quote_simple() {
    assert_eq!(fts_quote("hello"), "\"hello\"");
}

#[test]
fn fts_quote_with_quotes() {
    assert_eq!(fts_quote("he\"llo"), "\"he\"\"llo\"");
}

fn setup_fts_db() -> Connection {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(
        "CREATE VIRTUAL TABLE fts_chunks USING fts5(content);
             INSERT INTO fts_chunks(content) VALUES ('authentication login session');
             INSERT INTO fts_chunks(content) VALUES ('authorization permission role');
             INSERT INTO fts_chunks(content) VALUES ('audit logging trace');
             CREATE VIRTUAL TABLE fts_chunks_vocab USING fts5vocab(fts_chunks, row);",
    )
    .unwrap();
    conn
}

#[test]
fn expand_long_term_quoted_as_is() {
    let conn = setup_fts_db();
    let result =
        fts_expand_short_terms(&conn, &sanitized("authentication"), "fts_chunks_vocab").unwrap();
    assert_eq!(result.as_str(), "\"authentication\"");
}

#[test]
fn expand_short_term_with_vocab_matches() {
    let conn = setup_fts_db();
    let result = fts_expand_short_terms(&conn, &sanitized("au"), "fts_chunks_vocab").unwrap();
    assert!(result.as_str().contains("\"audit\""), "{}", result.as_str());
    assert!(
        result.as_str().contains("\"authentication\""),
        "{}",
        result.as_str()
    );
    assert!(result.as_str().contains(" OR "), "{}", result.as_str());
}

#[test]
fn expand_short_term_no_matches() {
    let conn = setup_fts_db();
    let result = fts_expand_short_terms(&conn, &sanitized("zz"), "fts_chunks_vocab").unwrap();
    assert_eq!(result.as_str(), "\"zz\"");
}

#[test]
fn expand_mixed_short_and_long() {
    let conn = setup_fts_db();
    let result = fts_expand_short_terms(&conn, &sanitized("au login"), "fts_chunks_vocab").unwrap();
    assert!(result.as_str().contains("\"login\""), "{}", result.as_str());
    assert!(result.as_str().contains(" OR "), "{}", result.as_str());
}

#[test]
fn expand_operator_like_terms_are_quoted_not_expanded() {
    let conn = setup_fts_db();
    // NOT (3 chars) and OR (2 chars) must both be quoted as-is,
    // not expanded via vocab (e.g. OR must not become "order" OR ...).
    let result =
        fts_expand_short_terms(&conn, &sanitized("NOT secret"), "fts_chunks_vocab").unwrap();
    assert_eq!(result.as_str(), "\"NOT\" AND \"secret\"");

    let result = fts_expand_short_terms(&conn, &sanitized("OR"), "fts_chunks_vocab").unwrap();
    assert_eq!(result.as_str(), "\"OR\"");

    let result =
        fts_expand_short_terms(&conn, &sanitized("foo or bar"), "fts_chunks_vocab").unwrap();
    assert_eq!(result.as_str(), "\"foo\" AND \"or\" AND \"bar\"");
}

#[test]
fn expand_special_chars_escaped() {
    let conn = setup_fts_db();
    let result = fts_expand_short_terms(&conn, &sanitized("a%"), "fts_chunks_vocab").unwrap();
    assert!(
        result.as_str().contains("\"audit\"") || result.as_str() == "\"a%\"",
        "{}",
        result.as_str()
    );
}

#[test]
fn expand_without_vocab_table_degrades() {
    let conn = Connection::open_in_memory().unwrap();
    let result = fts_expand_short_terms(&conn, &sanitized("au login"), "fts_chunks_vocab").unwrap();
    assert_eq!(result.as_str(), "\"au\" AND \"login\"");
}

#[test]
fn expand_single_char_term() {
    let conn = setup_fts_db();
    let result = fts_expand_short_terms(&conn, &sanitized("a"), "fts_chunks_vocab").unwrap();
    assert!(result.as_str().contains("\"audit\"") || result.as_str() == "\"a\"");
}

/// Tests below pin the sanitize / expand contract — pass `disabled()` so
/// changes to the Phase 5 normalization defaults can never alter what
/// these tests measure.
fn no_norm() -> QueryNormalizationConfig {
    QueryNormalizationConfig::disabled()
}

#[test]
fn prepare_match_query_end_to_end() {
    let conn = setup_fts_db();
    let result = prepare_match_query(&conn, "au login", "fts_chunks_vocab", &no_norm()).unwrap();
    assert!(result.as_str().contains("\"login\""), "{}", result.as_str());
    assert!(result.as_str().contains(" OR "), "{}", result.as_str());
}

// #224: a vocab-expanded group adjacent to another token must form a valid
// FTS5 expression. Implicit AND rejects parenthesised groups as operands
// (`(a OR b) c` is an fts5 syntax error), so the query must execute, not
// just look right as a string.
#[test]
fn prepare_match_query_expanded_group_is_executable() {
    let conn = setup_fts_db();
    let query = prepare_match_query(&conn, "au login", "fts_chunks_vocab", &no_norm()).unwrap();
    let count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM fts_chunks WHERE fts_chunks MATCH ?1",
            [query.as_str()],
            |row| row.get(0),
        )
        .unwrap_or_else(|e| panic!("MATCH rejected {:?}: {e}", query.as_str()));
    assert_eq!(
        count,
        1,
        "expected exactly 'authentication login session' to match, query: {:?}",
        query.as_str()
    );
}

#[test]
fn prepare_match_query_empty_input() {
    let conn = setup_fts_db();
    assert_eq!(
        prepare_match_query(&conn, "", "fts_chunks_vocab", &no_norm()),
        Err(SanitizeError::EmptyInput)
    );
}

#[test]
fn prepare_match_query_operators_are_quoted() {
    let conn = setup_fts_db();
    let result = prepare_match_query(&conn, "foo OR bar", "fts_chunks_vocab", &no_norm()).unwrap();
    assert_eq!(result.as_str(), "\"foo\" AND \"OR\" AND \"bar\"");
}

#[test]
fn prepare_match_query_rejects_invalid_vocab_table_name() {
    let conn = setup_fts_db();
    assert_eq!(
        prepare_match_query(&conn, "au", "1vocab", &no_norm()),
        Err(SanitizeError::InvalidVocabTable("1vocab".into()))
    );
}

#[test]
fn prepare_match_query_surfaces_non_missing_vocab_errors() {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute("CREATE TABLE bad_vocab(term TEXT)", [])
        .unwrap();

    let result = prepare_match_query(&conn, "au", "bad_vocab", &no_norm());
    assert!(
        matches!(result, Err(SanitizeError::VocabLookupFailed(_))),
        "expected vocab lookup failure, got {result:?}"
    );
}

#[test]
fn expand_rejects_leading_digit_vocab_name() {
    let conn = setup_fts_db();
    assert_eq!(
        fts_expand_short_terms(&conn, &sanitized("au"), "1vocab"),
        Err(SanitizeError::InvalidVocabTable("1vocab".into()))
    );
}

#[test]
fn expand_rejects_empty_vocab_name() {
    let conn = setup_fts_db();
    assert_eq!(
        fts_expand_short_terms(&conn, &sanitized("au"), ""),
        Err(SanitizeError::InvalidVocabTable("".into()))
    );
}

#[test]
fn expand_surfaces_non_missing_vocab_errors() {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute("CREATE TABLE bad_vocab(term TEXT)", [])
        .unwrap();

    let result = fts_expand_short_terms(&conn, &sanitized("au"), "bad_vocab");
    assert!(
        matches!(result, Err(SanitizeError::VocabLookupFailed(_))),
        "expected vocab lookup failure, got {result:?}"
    );
}

#[test]
fn prepare_match_query_with_missing_vocab_degrades() {
    let conn = Connection::open_in_memory().unwrap();
    let result = prepare_match_query(&conn, "au login", "fts_chunks_vocab", &no_norm()).unwrap();
    assert_eq!(result.as_str(), "\"au\" AND \"login\"");
}

#[test]
fn prepare_match_query_default_normalizes_fullwidth_input() {
    let conn = setup_fts_db();
    // Phase 5 default: NFKC folds `ＬＯＧＩＮ` → `LOGIN` then ASCII
    // lowercase folds to `login`. Without normalization the trigram
    // index would never match the half-width form.
    let result = prepare_match_query(
        &conn,
        "ＬＯＧＩＮ",
        "fts_chunks_vocab",
        &QueryNormalizationConfig::default(),
    )
    .unwrap();
    assert_eq!(result.as_str(), "\"login\"");
}

fn prepare_err(conn: &Connection, sql: &str) -> rusqlite::Error {
    let Err(e) = conn.prepare_cached(sql) else {
        panic!("expected prepare_cached to fail for: {sql}");
    };
    e
}

#[test]
fn is_missing_table_error_detects_sqlite_no_such_table() {
    let conn = Connection::open_in_memory().unwrap();
    let err = prepare_err(&conn, "SELECT * FROM definitely_nonexistent");
    assert!(is_missing_table_error(&err), "got {err:?}");
}

#[test]
fn is_missing_table_error_rejects_syntax_error() {
    let conn = Connection::open_in_memory().unwrap();
    let err = prepare_err(&conn, "SELECT FROM");
    assert!(!is_missing_table_error(&err), "got {err:?}");
}

#[test]
fn is_missing_table_error_rejects_missing_column() {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute("CREATE TABLE t(x INTEGER)", []).unwrap();
    let err = prepare_err(&conn, "SELECT y FROM t");
    assert!(!is_missing_table_error(&err), "got {err:?}");
}

fn sanitized(s: &str) -> SanitizedFtsQuery {
    SanitizedFtsQuery(s.to_owned())
}

fn ok(s: &str) -> Result<SanitizedFtsQuery, SanitizeError> {
    Ok(SanitizedFtsQuery(s.to_owned()))
}

// T-001: near_removal
#[test]
fn near_removal() {
    assert_eq!(sanitize_fts_query("NEAR(a b) hello"), ok("hello"));
}

// T-001b: near_with_distance
#[test]
fn near_with_distance() {
    assert_eq!(sanitize_fts_query("NEAR/3(a b c) hello"), ok("hello"));
}

// T-001c: near_unclosed_paren
#[test]
fn near_unclosed_paren() {
    assert_eq!(
        sanitize_fts_query("NEAR(a b hello"),
        Err(SanitizeError::NoSearchableTerms)
    );
}

// T-002: auto_quote_hyphen
#[test]
fn auto_quote_hyphen() {
    assert_eq!(sanitize_fts_query("rate-limit"), ok("\"rate-limit\""));
}

// T-003: auto_quote_colon
#[test]
fn auto_quote_colon() {
    assert_eq!(sanitize_fts_query("std::io"), ok("\"std::io\""));
}

// T-004: prefix_strip
#[test]
fn prefix_strip() {
    assert_eq!(sanitize_fts_query("^+hello"), ok("hello"));
}

// T-005b: quote_balancing_exact_output
#[test]
fn quote_balancing_exact_output() {
    assert_eq!(sanitize_fts_query("unbalanced\""), ok("unbalanced\"\""));
}

// T-006: empty_input
#[test]
fn empty_input() {
    assert_eq!(sanitize_fts_query(""), Err(SanitizeError::EmptyInput));
}

// T-006b: whitespace_only
#[test]
fn whitespace_only() {
    assert_eq!(sanitize_fts_query("   "), Err(SanitizeError::EmptyInput));
}

// T-011: sandwiched_operator_preserved
#[test]
fn sandwiched_operator_preserved() {
    assert_eq!(sanitize_fts_query("foo AND bar"), ok("foo AND bar"));
    assert_eq!(sanitize_fts_query("foo OR bar"), ok("foo OR bar"));
}

// T-012: dangling_operator_dropped
#[test]
fn dangling_operator_dropped() {
    // Leading/trailing operators without both neighbours are dropped.
    assert_eq!(sanitize_fts_query("NOT secret"), ok("secret"));
    assert_eq!(sanitize_fts_query("foo OR"), ok("foo"));
}

// T-012b: consecutive_operators_between_terms
#[test]
fn consecutive_operators_between_terms() {
    // Neither AND nor OR has a non-operator on both sides → both dropped.
    assert_eq!(sanitize_fts_query("foo AND OR bar"), ok("foo bar"));
    assert_eq!(sanitize_fts_query("NOT foo NOT"), ok("foo"));
}

// T-013: operator_only_returns_error
#[test]
fn operator_only_returns_error() {
    assert_eq!(
        sanitize_fts_query("NOT"),
        Err(SanitizeError::NoSearchableTerms)
    );
    assert_eq!(
        sanitize_fts_query("AND OR NOT"),
        Err(SanitizeError::NoSearchableTerms)
    );
}

// T-014: near_then_dangling_operator
#[test]
fn near_then_dangling_operator() {
    // "foo OR NEAR(bar baz)" → NEAR stripped → "foo OR" → OR dangling → "foo"
    assert_eq!(sanitize_fts_query("foo OR NEAR(bar baz)"), ok("foo"));
}

// T-015: case_insensitive_operators
#[test]
fn case_insensitive_operators() {
    assert_eq!(sanitize_fts_query("foo or bar"), ok("foo or bar"));
    assert_eq!(sanitize_fts_query("Not secret"), ok("secret"));
}

// T-016: prefix_only_returns_error
#[test]
fn prefix_only_returns_error() {
    assert_eq!(
        sanitize_fts_query("^"),
        Err(SanitizeError::NoSearchableTerms)
    );
}

// T-105-013: drop_dangling_operators_drops_operator_at_position_zero
//
// Boundary: a valid FTS5 operator at index 0 cannot have a left neighbour,
// so `has_left` is false and the operator must be dropped even when a
// non-operator follows. Pins the `i > 0` short-circuit against a future
// off-by-one rewrite that would let leading `AND` / `OR` slip through and
// form an invalid FTS5 expression.
#[test]
fn drop_dangling_operators_drops_operator_at_position_zero() {
    let tokens: Vec<String> = vec!["AND".into(), "foo".into(), "bar".into()];
    let result = drop_dangling_operators(&tokens);
    assert_eq!(
        result,
        vec!["foo", "bar"],
        "operator at index 0 has no left neighbour → must drop"
    );
}
