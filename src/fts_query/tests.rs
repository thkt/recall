use super::*;
use crate::db::setup_test_db;

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
        "CREATE VIRTUAL TABLE messages USING fts5(content);
             INSERT INTO messages(content) VALUES ('authentication login session');
             INSERT INTO messages(content) VALUES ('authorization permission role');
             INSERT INTO messages(content) VALUES ('audit logging trace');
             CREATE VIRTUAL TABLE messages_vocab USING fts5vocab(messages, row);",
    )
    .unwrap();
    conn
}

#[test]
fn expand_long_term_quoted_as_is() {
    let conn = setup_fts_db();
    let result = fts_expand_short_terms(&conn, &sanitized("authentication")).unwrap();
    assert_eq!(result.to_string(), "\"authentication\"");
}

#[test]
fn expand_short_term_no_matches() {
    let conn = setup_fts_db();
    let result = fts_expand_short_terms(&conn, &sanitized("zz")).unwrap();
    assert_eq!(result.to_string(), "\"zz\"");
}

#[test]
fn expand_operator_like_terms_are_quoted_not_expanded() {
    let conn = setup_fts_db();
    let result = fts_expand_short_terms(&conn, &sanitized("NOT secret")).unwrap();
    assert_eq!(result.to_string(), "\"NOT\" AND \"secret\"");

    let result = fts_expand_short_terms(&conn, &sanitized("OR")).unwrap();
    assert_eq!(result.to_string(), "\"OR\"");

    let result = fts_expand_short_terms(&conn, &sanitized("foo or bar")).unwrap();
    assert_eq!(result.to_string(), "\"foo\" AND \"or\" AND \"bar\"");
}

fn index_with_messages(texts: &[&str]) -> (tempfile::TempDir, Connection) {
    let (dir, conn) = setup_test_db();
    for text in texts {
        conn.execute(
            "INSERT INTO messages (session_id, role, text) VALUES ('s', 'user', ?1)",
            [text],
        )
        .unwrap();
    }
    (dir, conn)
}

fn expand_in(conn: &Connection, query: &str) -> String {
    fts_expand_short_terms(conn, &sanitized(query))
        .unwrap()
        .to_string()
}

#[test]
fn a_short_word_expands_only_to_the_terms_that_start_with_it() {
    let (_dir, conn) = index_with_messages(&["xaby", "xabz", "xab😀", "xacq", "xaaq"]);
    assert_eq!(expand_in(&conn, "ab"), r#"("aby" OR "abz" OR "ab😀")"#);
}

#[test]
fn a_one_character_word_expands_to_the_terms_that_start_with_it() {
    let (_dir, conn) = index_with_messages(&["abc", "bcd"]);
    assert_eq!(expand_in(&conn, "a"), r#"("abc")"#);
}

#[test]
fn an_upper_case_short_word_expands_to_the_terms_of_its_lower_case_form() {
    let (_dir, conn) = index_with_messages(&["Abc", "abd"]);
    assert_eq!(expand_in(&conn, "AB"), r#"("abc" OR "abd")"#);
}

#[test]
fn a_short_word_with_like_wildcards_expands_only_to_the_terms_that_hold_them_as_written() {
    let (_dir, conn) = index_with_messages(&["5%x", "5_y", "5zz", "5\\w"]);
    assert_eq!(expand_in(&conn, "5%"), r#"("5%x")"#);
    assert_eq!(expand_in(&conn, "5_"), r#"("5_y")"#);
    assert_eq!(expand_in(&conn, "5\\"), r#"("5\w")"#);
}

#[test]
fn a_two_character_cjk_word_expands_to_the_trigrams_that_start_with_it() {
    let (_dir, conn) = index_with_messages(&["設定を変更", "設計書"]);
    assert_eq!(expand_in(&conn, "設定"), r#"("設定を")"#);
}

#[test]
fn a_short_word_expands_to_no_term_whose_next_character_is_the_last_unicode_scalar() {
    let (_dir, conn) = index_with_messages(&["xaby", "xab\u{10FFFF}"]);
    assert_eq!(expand_in(&conn, "ab"), r#"("aby")"#);
}

const FTS5_VOCAB_TERM_GE: u32 = 0x0200;
const FTS5_VOCAB_TERM_LE: u32 = 0x0400;

#[test]
fn the_vocabulary_lookup_hands_both_bounds_of_its_range_to_the_vocabulary_table() {
    let (_dir, conn) = setup_test_db();
    let plan: String = conn
        .query_row(
            &format!("EXPLAIN QUERY PLAN {}", VOCAB_LOOKUP_SQL),
            ["ab", "ac"],
            |row| row.get(3),
        )
        .unwrap();
    let idx_num: u32 = plan
        .split_once("VIRTUAL TABLE INDEX ")
        .and_then(|(_, rest)| rest.split(':').next())
        .and_then(|n| n.parse().ok())
        .unwrap_or_else(|| panic!("no index number in the plan: {plan}"));
    let range_on_term = FTS5_VOCAB_TERM_GE | FTS5_VOCAB_TERM_LE;
    assert_eq!(idx_num & range_on_term, range_on_term, "{plan}");
}

#[test]
fn the_match_string_of_an_expanded_short_word_finds_only_the_message_holding_both_words() {
    let conn = setup_fts_db();
    let query = prepare_match_query(&conn, "au login")
        .unwrap()
        .to_trigram_match()
        .unwrap();
    let count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM messages WHERE messages MATCH ?1",
            [&query],
            |row| row.get(0),
        )
        .unwrap_or_else(|e| panic!("MATCH rejected {query:?}: {e}"));
    assert_eq!(
        count, 1,
        "expected exactly 'authentication login session' to match, query: {query:?}"
    );
}

#[test]
fn prepare_match_query_empty_input() {
    let conn = setup_fts_db();
    assert_eq!(
        prepare_match_query(&conn, ""),
        Err(SanitizeError::EmptyInput)
    );
}

#[test]
fn prepare_match_query_operators_are_quoted() {
    let conn = setup_fts_db();
    let result = prepare_match_query(&conn, "foo OR bar").unwrap();
    assert_eq!(result.to_string(), "\"foo\" AND \"OR\" AND \"bar\"");
}

#[test]
fn prepare_match_query_surfaces_vocab_errors() {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute("CREATE TABLE messages_vocab(term TEXT)", [])
        .unwrap();

    let result = prepare_match_query(&conn, "au");
    assert!(
        matches!(result, Err(SanitizeError::VocabLookupFailed(_))),
        "expected vocab lookup failure, got {result:?}"
    );
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
fn an_operator_that_opens_the_query_is_dropped() {
    assert_eq!(sanitize_fts_query("AND foo bar"), ok("foo bar"));
}

#[test]
fn expand_quotes_a_phrase_once_as_a_whole() {
    let conn = setup_fts_db();
    let result = prepare_match_query(&conn, "\"login session\" audit").unwrap();
    assert_eq!(result.to_string(), "\"login session\" AND \"audit\"");
}

#[test]
fn expand_quotes_a_hyphen_or_colon_term_once() {
    let conn = setup_fts_db();
    let result = prepare_match_query(&conn, "rate-limit std::io").unwrap();
    assert_eq!(result.to_string(), "\"rate-limit\" AND \"std::io\"");
}

#[test]
fn expand_passes_a_phrase_with_and_or_not_words_through_verbatim() {
    let conn = setup_fts_db();
    let result = prepare_match_query(&conn, "\"to be or not to be\"").unwrap();
    assert_eq!(result.to_string(), "\"to be or not to be\"");
}

#[test]
fn expand_drops_the_parentheses_around_a_phrase() {
    let conn = setup_fts_db();
    let result = prepare_match_query(&conn, "(\"login session\")").unwrap();
    assert_eq!(result.to_string(), "\"login session\"");
}

#[test]
fn expand_splits_a_colon_word_from_the_phrase_attached_to_it() {
    let conn = setup_fts_db();
    let result = prepare_match_query(&conn, "key:\"some value\" (foo)").unwrap();
    assert_eq!(
        result.to_string(),
        "\"key:\" AND \"some value\" AND \"foo\""
    );
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
    for (query, message) in [
        (
            "zz NOT beta",
            "NOT needs a left term of at least 3 characters",
        ),
        ("NOT beta", "NOT needs a term on its left"),
        ("alpha AND NOT beta", "NOT needs a term on its left"),
    ] {
        assert_eq!(
            prepare_operator_query(query).unwrap_err().to_string(),
            message,
            "query: {query:?}"
        );
    }
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

#[test]
fn operator_query_renders_a_chain_of_100_000_nots_without_overflowing_the_stack() {
    let query = format!("alpha{}", " NOT beta".repeat(100_000));

    let rendered = operator(&query);

    assert_eq!(rendered.not_operands.len(), 100_000);
}

#[test]
fn the_sanitizer_the_expansion_and_the_operator_query_all_end_an_unclosed_phrase_at_the_query_end()
{
    let conn = setup_fts_db();

    assert_eq!(
        sanitize_fts_query("alpha \"beta gamma"),
        ok("alpha \"beta gamma\"")
    );
    assert_eq!(
        prepare_match_query(&conn, "alpha \"beta gamma")
            .unwrap()
            .to_string(),
        "\"alpha\" AND \"beta gamma\""
    );
    assert_eq!(
        operator("alpha AND \"beta gamma").match_query.as_deref(),
        Some("(\"alpha\" AND \"beta gamma\")")
    );
}
