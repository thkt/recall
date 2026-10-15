use rusqlite::Connection;

use super::query_normalize::{QueryNormalizationConfig, normalize_for_fts};

/// Filter out `NEAR(...)` and `NEAR/N(...)` groups from whitespace-split tokens.
fn strip_near_groups(query: &str) -> Vec<&str> {
    let mut tokens = Vec::new();
    let mut paren_depth: usize = 0;
    for w in query.split_whitespace() {
        let upper = w.to_ascii_uppercase();
        if upper.starts_with("NEAR(") || upper.starts_with("NEAR/") {
            paren_depth += w.chars().filter(|&c| c == '(').count();
            paren_depth = paren_depth.saturating_sub(w.chars().filter(|&c| c == ')').count());
            continue;
        }
        if paren_depth > 0 {
            paren_depth = paren_depth.saturating_sub(w.chars().filter(|&c| c == ')').count());
            continue;
        }
        tokens.push(w);
    }
    tokens
}

/// FTS5 operator-like keywords that must not be vocab-expanded, and are dropped
/// when dangling (no non-operator neighbour on both sides).
const FTS5_OPERATORS: &[&str] = &["AND", "OR", "NOT"];

fn is_fts5_operator(token: &str) -> bool {
    FTS5_OPERATORS
        .iter()
        .any(|op| token.eq_ignore_ascii_case(op))
}

/// Remove operator-like keywords that lack a non-operator neighbour on both
/// sides. Keeps operators sandwiched between real terms.
fn drop_dangling_operators(tokens: &[String]) -> Vec<&str> {
    tokens
        .iter()
        .enumerate()
        .filter(|&(i, t)| {
            if !is_fts5_operator(t) {
                return true;
            }
            let has_left = i > 0 && !is_fts5_operator(&tokens[i - 1]);
            let has_right = i + 1 < tokens.len() && !is_fts5_operator(&tokens[i + 1]);
            has_left && has_right
        })
        .map(|(_, t)| t.as_str())
        .collect()
}

/// A pre-processed FTS5 query string — intermediate representation between
/// raw user input and [`MatchFtsQuery`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SanitizedFtsQuery(String);

impl SanitizedFtsQuery {
    pub(crate) fn as_str(&self) -> &str {
        &self.0
    }
}

/// A fully expanded and quoted FTS5 query string, safe to pass to `MATCH`.
///
/// # Cross-repo wire-format contract (amici ADR-0008)
///
/// The MATCH string emitted here is a wire-format contract consumed by amici's
/// `parse_fts_segments` (`src/storage/fts.rs`): `"..."` quoting for fixed terms,
/// `( ... )` for OR-groups, ` OR ` as the in-group separator, and ` AND ` as the
/// top-level separator (see `fts_expand_short_terms`). amici parses this format
/// rather than a typed API, so a separator or quoting change here is a contract
/// change requiring a coordinated update to `parse_fts_segments`. Neither repo's
/// compiler or unit tests can enforce this; amici's round-trip test
/// (`src/storage/fts/tests.rs`, `parse_fts_segments_recovers_rurico_or_group_wire_format`)
/// detects drift at rurico rev-bump time. Tracking issue: thkt/amici#159.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatchFtsQuery(String);

impl MatchFtsQuery {
    /// Borrow the inner query string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Reasons why [`prepare_match_query`] cannot produce a usable query.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SanitizeError {
    /// The input string was empty or contained only whitespace.
    #[error("search query is empty")]
    EmptyInput,
    /// The input contained tokens, but none survived sanitization
    /// (e.g. only `NEAR()` groups or prefix characters).
    #[error("search query contains no searchable terms")]
    NoSearchableTerms,
    /// The supplied vocab table name is not a safe SQL identifier.
    #[error("invalid vocab table name: {0:?}")]
    InvalidVocabTable(String),
    /// SQLite failed while consulting the vocab table for short-term expansion.
    #[error("fts vocab lookup failed: {0}")]
    VocabLookupFailed(String),
}

/// Neutralize FTS5 special syntax in user queries: `NEAR()` grouping,
/// start-of-column `^`, required `+` / excluded `-` prefixes, column-filter
/// colons, and unbalanced quotes. Operator-like keywords (`AND`, `OR`, `NOT`)
/// sandwiched between non-operator terms are preserved (unquoted); dangling
/// operators (e.g. leading `NOT`, trailing `OR` after NEAR removal) are dropped.
pub(crate) fn sanitize_fts_query(query: &str) -> Result<SanitizedFtsQuery, SanitizeError> {
    if query.trim().is_empty() {
        return Err(SanitizeError::EmptyInput);
    }
    let tokens: Vec<String> = strip_near_groups(query)
        .into_iter()
        .map(|w| {
            let stripped = w.trim_start_matches(['^', '+', '-']);
            let cleaned = stripped.trim_matches(['(', ')']);
            if (cleaned.contains(':') || cleaned.contains('-')) && !cleaned.starts_with('"') {
                let unquoted = cleaned.replace('"', "");
                format!("\"{unquoted}\"")
            } else {
                cleaned.to_owned()
            }
        })
        .filter(|w| !w.is_empty())
        .collect();
    let result = drop_dangling_operators(&tokens).join(" ");
    if result.is_empty() {
        return Err(SanitizeError::NoSearchableTerms);
    }
    let quote_count = result.chars().filter(|&c| c == '"').count();
    if quote_count % 2 != 0 {
        Ok(SanitizedFtsQuery(format!("{result}\"")))
    } else {
        Ok(SanitizedFtsQuery(result))
    }
}

/// Wrap `s` in double quotes for FTS5 MATCH syntax, escaping internal `"` as `""`.
pub fn fts_quote(s: &str) -> String {
    format!("\"{}\"", s.replace('"', "\"\""))
}

/// Normalize, sanitize, and expand short terms into a query safe for FTS5 `MATCH`.
///
/// Phase 5 (#69): `normalization` is applied **before** `sanitize_fts_query`
/// so full-width punctuation (e.g. `（`) folds to ASCII (`(`) prior to NEAR
/// detection. Callers must apply the same `normalization` config to indexed
/// text — applying only to one side leaves the FTS5 token streams disagreed
/// and silently misses matches.
///
/// Combines `normalize_for_fts`, `sanitize_fts_query`, and
/// `fts_expand_short_terms` into a single call.
///
/// `vocab_table` names the `fts5vocab` virtual table to consult for short-term
/// expansion (e.g. `"fts_chunks_vocab"` or `"messages_vocab"`). Must be created
/// with the `row` or `col` vocabulary type — the `instance` type lacks the
/// `cnt` column this query relies on. The value is interpolated into the SQL
/// unescaped.
///
/// # Errors
///
/// Returns:
/// - [`SanitizeError::EmptyInput`] if `query` is empty or whitespace-only
/// - [`SanitizeError::NoSearchableTerms`] if sanitization strips all searchable terms
///
/// SQLite failures while consulting the vocab table surface as `Err`, except
/// for the specific "no such table" case which degrades to "no expansion".
pub fn prepare_match_query(
    conn: &Connection,
    query: &str,
    vocab_table: &str,
    normalization: &QueryNormalizationConfig,
) -> Result<MatchFtsQuery, SanitizeError> {
    let normalized = normalize_for_fts(query, normalization);
    let sanitized = sanitize_fts_query(&normalized)?;
    fts_expand_short_terms(conn, &sanitized, vocab_table)
}

/// Expand a single short token via vocab prefix matching.
///
/// Returns `Some(expanded_group)` when matches are found, `None` when the
/// vocab lookup returns no results or the vocab table disappeared.
fn expand_token_via_vocab(
    stmt: &mut rusqlite::CachedStatement<'_>,
    token: &str,
) -> Result<Option<String>, SanitizeError> {
    let escaped = token
        .replace('\\', "\\\\")
        .replace('%', "\\%")
        .replace('_', "\\_");
    let pattern = format!("{escaped}%");
    match stmt
        .query_map([&pattern], |row| row.get::<_, String>(0))
        .and_then(|rows| rows.collect::<Result<Vec<_>, _>>())
    {
        Ok(terms) if !terms.is_empty() => {
            let quoted: Vec<String> = terms.iter().map(|t| fts_quote(t)).collect();
            Ok(Some(format!("({})", quoted.join(" OR "))))
        }
        Ok(_) => Ok(None),
        Err(e) if is_missing_table_error(&e) => Ok(None),
        Err(e) => Err(SanitizeError::VocabLookupFailed(e.to_string())),
    }
}

// SQLite reports missing tables via `SqliteFailure` with an English `errmsg`
// containing "no such table". The extended code (`SQLITE_ERROR = 1`) covers
// many other causes, so the message substring is the only portable signal.
fn is_missing_table_error(err: &rusqlite::Error) -> bool {
    matches!(
        err,
        rusqlite::Error::SqliteFailure(_, Some(msg)) if msg.contains("no such table")
    )
}

fn is_valid_sql_identifier(value: &str) -> bool {
    let mut chars = value.chars();
    let head_ok = chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_');
    let tail_ok = chars.all(|c| c.is_ascii_alphanumeric() || c == '_');
    head_ok && tail_ok
}

/// Expand short terms (1-2 chars) via `fts5vocab` prefix matching.
/// Falls back to quoting as-is when the vocab table is missing.
///
/// `vocab_table` must be an `fts5vocab` of type `row` or `col` (the SQL
/// references the `term` and `cnt` columns). The name is interpolated into
/// the SQL unescaped.
///
/// # Errors
///
/// Returns:
/// - [`SanitizeError::InvalidVocabTable`] if `vocab_table` is not a valid
///   SQL identifier (non-empty, leading character ASCII alphabetic or `_`,
///   remaining characters ASCII alphanumeric or `_`).
/// - [`SanitizeError::VocabLookupFailed`] if SQLite fails while preparing
///   or executing the vocab query for any reason other than the vocab table
///   being absent.
pub(crate) fn fts_expand_short_terms(
    conn: &Connection,
    query: &SanitizedFtsQuery,
    vocab_table: &str,
) -> Result<MatchFtsQuery, SanitizeError> {
    if !is_valid_sql_identifier(vocab_table) {
        return Err(SanitizeError::InvalidVocabTable(vocab_table.to_owned()));
    }
    let sql = format!(
        "SELECT term FROM {vocab_table} \
         WHERE term LIKE ?1 ESCAPE '\\' \
         ORDER BY cnt DESC LIMIT 25"
    );
    let mut stmt = match conn.prepare_cached(&sql) {
        Ok(s) => Some(s),
        Err(e) if is_missing_table_error(&e) => None,
        Err(e) => return Err(SanitizeError::VocabLookupFailed(e.to_string())),
    };

    let mut parts = Vec::new();
    for token in query.as_str().split_whitespace() {
        // Length check covers AND/NOT (3 chars); operator guard adds OR (2 chars).
        if token.chars().count() >= 3 || is_fts5_operator(token) {
            parts.push(fts_quote(token));
            continue;
        }
        let expanded = match stmt.as_mut() {
            Some(s) => expand_token_via_vocab(s, token)?,
            None => None,
        };
        parts.push(expanded.unwrap_or_else(|| fts_quote(token)));
    }
    // Join with explicit AND: FTS5 implicit adjacency rejects parenthesised
    // groups as operands (`(a OR b) c` is a syntax error); explicit AND
    // accepts them with identical semantics.
    //
    // Wire-format contract (amici ADR-0008): this ` AND ` separator and the
    // `"..."` / `( ... )` / ` OR ` shapes are parsed by amici's
    // `parse_fts_segments`. Changing them is a cross-repo contract change;
    // amici's round-trip test surfaces drift at rev-bump time. See
    // [`MatchFtsQuery`].
    Ok(MatchFtsQuery(parts.join(" AND ")))
}

#[cfg(test)]
mod tests;
