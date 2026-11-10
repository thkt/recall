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

#[test]
fn near_removal() {
    assert_eq!(sanitize_fts_query("NEAR(a b) hello"), ok("hello"));
}

#[test]
fn near_with_distance() {
    assert_eq!(sanitize_fts_query("NEAR/3(a b c) hello"), ok("hello"));
}

#[test]
fn near_group_ignores_parentheses_inside_a_phrase() {
    assert_eq!(sanitize_fts_query("NEAR(\"x ) y\" z) hello"), ok("hello"));
}

#[test]
fn near_unclosed_paren() {
    assert_eq!(
        sanitize_fts_query("NEAR(a b hello"),
        Err(SanitizeError::NoSearchableTerms)
    );
}

#[test]
fn auto_quote_hyphen() {
    assert_eq!(sanitize_fts_query("rate-limit"), ok("\"rate-limit\""));
}

#[test]
fn auto_quote_colon() {
    assert_eq!(sanitize_fts_query("std::io"), ok("\"std::io\""));
}

#[test]
fn prefix_strip() {
    assert_eq!(sanitize_fts_query("^+hello"), ok("hello"));
}

#[test]
fn quote_balancing_exact_output() {
    assert_eq!(sanitize_fts_query("unbalanced\""), ok("unbalanced\"\""));
}

#[test]
fn empty_input() {
    assert_eq!(sanitize_fts_query(""), Err(SanitizeError::EmptyInput));
}

#[test]
fn whitespace_only() {
    assert_eq!(sanitize_fts_query("   "), Err(SanitizeError::EmptyInput));
}

#[test]
fn sandwiched_operator_preserved() {
    assert_eq!(sanitize_fts_query("foo AND bar"), ok("foo AND bar"));
    assert_eq!(sanitize_fts_query("foo OR bar"), ok("foo OR bar"));
}

#[test]
fn dangling_operator_dropped() {
    assert_eq!(sanitize_fts_query("NOT secret"), ok("secret"));
    assert_eq!(sanitize_fts_query("foo OR"), ok("foo"));
}

#[test]
fn consecutive_operators_between_terms() {
    assert_eq!(sanitize_fts_query("foo AND OR bar"), ok("foo bar"));
    assert_eq!(sanitize_fts_query("NOT foo NOT"), ok("foo"));
}

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

#[test]
fn near_then_dangling_operator() {
    assert_eq!(sanitize_fts_query("foo OR NEAR(bar baz)"), ok("foo"));
}

#[test]
fn case_insensitive_operators() {
    assert_eq!(sanitize_fts_query("foo or bar"), ok("foo or bar"));
    assert_eq!(sanitize_fts_query("Not secret"), ok("secret"));
}

#[test]
fn prefix_only_returns_error() {
    assert_eq!(
        sanitize_fts_query("^"),
        Err(SanitizeError::NoSearchableTerms)
    );
}

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

#[test]
fn expand_quotes_a_phrase_once_as_a_whole() {
    let conn = setup_fts_db();
    let result = prepare_match_query(
        &conn,
        "\"login session\" audit",
        "fts_chunks_vocab",
        &no_norm(),
    )
    .unwrap();
    assert_eq!(result.as_str(), "\"login session\" AND \"audit\"");
}

#[test]
fn expand_quotes_a_hyphen_or_colon_term_once() {
    let conn = setup_fts_db();
    let result =
        prepare_match_query(&conn, "rate-limit std::io", "fts_chunks_vocab", &no_norm()).unwrap();
    assert_eq!(result.as_str(), "\"rate-limit\" AND \"std::io\"");
}

#[test]
fn expand_passes_a_phrase_with_and_or_not_words_through_verbatim() {
    let conn = setup_fts_db();
    let result = prepare_match_query(
        &conn,
        "\"to be or not to be\"",
        "fts_chunks_vocab",
        &no_norm(),
    )
    .unwrap();
    assert_eq!(result.as_str(), "\"to be or not to be\"");
}

#[test]
fn expand_drops_the_parentheses_around_a_phrase() {
    let conn = setup_fts_db();
    let result =
        prepare_match_query(&conn, "(\"login session\")", "fts_chunks_vocab", &no_norm()).unwrap();
    assert_eq!(result.as_str(), "\"login session\"");
}

#[test]
fn expand_splits_a_colon_word_from_the_phrase_attached_to_it() {
    let conn = setup_fts_db();
    let result = prepare_match_query(
        &conn,
        "key:\"some value\" (foo)",
        "fts_chunks_vocab",
        &no_norm(),
    )
    .unwrap();
    assert_eq!(result.as_str(), "\"key:\" AND \"some value\" AND \"foo\"");
}

#[test]
fn sanitize_keeps_a_hyphen_inside_a_phrase_unquoted() {
    assert_eq!(
        sanitize_fts_query("\"foo bar-baz\" qux"),
        ok("\"foo bar-baz\" qux")
    );
}

fn operator(query: &str) -> OperatorQuery {
    prepare_operator_query(query).unwrap().unwrap()
}

#[test]
fn operator_query_is_absent_without_uppercase_operators() {
    assert_eq!(prepare_operator_query("tokio and async"), Ok(None));
    assert_eq!(prepare_operator_query("\"cats AND dogs\""), Ok(None));
}

#[test]
fn operator_query_gives_and_precedence_over_or_and_parentheses_over_both() {
    assert_eq!(
        operator("alpha OR beta AND gamma").match_query.as_deref(),
        Some("(\"alpha\" OR (\"beta\" AND \"gamma\"))")
    );
    assert_eq!(
        operator("(alpha OR beta) AND gamma").match_query.as_deref(),
        Some("((\"alpha\" OR \"beta\") AND \"gamma\")")
    );
}

#[test]
fn operator_query_drops_short_terms_by_tree_position() {
    assert_eq!(
        operator("alpha AND zz").match_query.as_deref(),
        Some("\"alpha\"")
    );
    assert_eq!(
        operator("alpha OR zz").match_query.as_deref(),
        Some("\"alpha\"")
    );
    let not_short = operator("alpha NOT zz");
    assert_eq!(not_short.match_query.as_deref(), Some("\"alpha\""));
    assert!(not_short.not_operands.is_empty());
    assert_eq!(operator("zz AND yy").match_query, None);
}

#[test]
fn operator_query_rejects_a_not_without_a_usable_left_side() {
    assert!(prepare_operator_query("zz NOT beta").is_err());
    assert!(prepare_operator_query("NOT beta").is_err());
    assert!(prepare_operator_query("alpha AND NOT beta").is_err());
}

#[test]
fn operator_query_keeps_not_operands_out_of_the_embedding_text() {
    let query = operator("alpha NOT (beta OR gamma)");
    assert_eq!(query.embed_text, "alpha");
    assert_eq!(query.not_operands, ["(\"beta\" OR \"gamma\")".to_owned()]);
}

#[test]
fn operator_query_records_a_not_operand_only_when_its_exclusion_covers_the_whole_query() {
    assert_eq!(
        operator("release AND (alpha NOT beta)").not_operands,
        ["\"beta\""]
    );
    assert_eq!(
        operator("(alpha NOT beta) NOT gamma").not_operands,
        ["\"beta\"", "\"gamma\""]
    );
    assert_eq!(
        operator("zz OR (alpha NOT beta)").not_operands,
        Vec::<String>::new()
    );
    assert_eq!(
        operator("(alpha NOT beta) OR gamma").not_operands,
        Vec::<String>::new()
    );
    assert_eq!(
        operator("alpha NOT (beta NOT gamma)").not_operands,
        ["(\"beta\" NOT \"gamma\")"]
    );
}

#[test]
fn operator_query_renders_a_chain_of_nots_left_nested() {
    let query = operator("alpha NOT beta NOT zz NOT gamma");
    assert_eq!(
        query.match_query.as_deref(),
        Some("((\"alpha\" NOT \"beta\") NOT \"gamma\")")
    );
    assert_eq!(query.not_operands, ["\"beta\"", "\"gamma\""]);
}

#[test]
fn operator_query_skips_empty_parentheses() {
    assert_eq!(
        operator("() ( ) alpha OR beta").match_query.as_deref(),
        Some("(\"alpha\" OR \"beta\")")
    );
}

#[test]
fn operator_query_ignores_dangling_operators_and_closes_open_parentheses() {
    assert_eq!(
        operator("hello OR").match_query.as_deref(),
        Some("\"hello\"")
    );
    assert_eq!(
        operator("(alpha OR beta AND gamma").match_query.as_deref(),
        Some("(\"alpha\" OR (\"beta\" AND \"gamma\"))")
    );
}
