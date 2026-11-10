use std::iter::repeat_n;

use rusqlite::Connection;

use super::query_normalize::{QueryNormalizationConfig, normalize_for_fts};

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

const FTS5_OPERATORS: &[&str] = &["AND", "OR", "NOT"];

fn is_fts5_operator(token: &str) -> bool {
    FTS5_OPERATORS
        .iter()
        .any(|op| token.eq_ignore_ascii_case(op))
}

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SanitizedFtsQuery(String);

impl SanitizedFtsQuery {
    pub(crate) fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatchFtsQuery(String);

impl MatchFtsQuery {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SanitizeError {
    #[error("search query is empty")]
    EmptyInput,
    #[error("search query contains no searchable terms")]
    NoSearchableTerms,
    #[error("invalid vocab table name: {0:?}")]
    InvalidVocabTable(String),
    #[error("fts vocab lookup failed: {0}")]
    VocabLookupFailed(String),
}

pub(crate) fn sanitize_fts_query(query: &str) -> Result<SanitizedFtsQuery, SanitizeError> {
    if query.trim().is_empty() {
        return Err(SanitizeError::EmptyInput);
    }
    let mut in_phrase = false;
    let tokens: Vec<String> = strip_near_groups(query)
        .into_iter()
        .map(|w| {
            let was_in_phrase = in_phrase;
            if w.matches('"').count() % 2 == 1 {
                in_phrase = !in_phrase;
            }
            if was_in_phrase {
                return w.to_owned();
            }
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

pub fn fts_quote(s: &str) -> String {
    format!("\"{}\"", s.replace('"', "\"\""))
}

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

/// A query that uses uppercase AND / OR / NOT is parsed into a tree and
/// rendered as an FTS5 MATCH without CJK segmentation or short-term vocab
/// expansion. FTS5 evaluates the operators per message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperatorQuery {
    /// Final MATCH string, `None` when every term was shorter than 3 characters.
    pub match_query: Option<String>,
    /// MATCH strings for the right side of every NOT.
    pub not_operands: Vec<String>,
    /// Terms outside every NOT right side, for the embedder.
    pub embed_text: String,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct QuerySyntaxError(String);

const MIN_OPERATOR_TERM_CHARS: usize = 3;

#[derive(Debug, Clone, PartialEq, Eq)]
enum Token {
    Open,
    Close,
    And,
    Or,
    Not,
    Term(String),
}

fn tokenize_operator_query(query: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut depth: usize = 0;
    let mut rest = query.trim_start();
    while let Some(c) = rest.chars().next() {
        match c {
            '(' => {
                depth += 1;
                tokens.push(Token::Open);
                rest = &rest[1..];
            }
            ')' => {
                if depth > 0 {
                    depth -= 1;
                    tokens.push(Token::Close);
                }
                rest = &rest[1..];
            }
            '"' => {
                let after_quote = &rest[1..];
                let end = after_quote.find('"').unwrap_or(after_quote.len());
                if end > 0 {
                    tokens.push(Token::Term(after_quote[..end].to_owned()));
                }
                rest = after_quote.get(end + 1..).unwrap_or("");
            }
            _ => {
                let end = rest
                    .find(|c: char| c.is_whitespace() || matches!(c, '(' | ')' | '"'))
                    .unwrap_or(rest.len());
                let word = &rest[..end];
                rest = &rest[end..];
                match word {
                    "AND" => tokens.push(Token::And),
                    "OR" => tokens.push(Token::Or),
                    "NOT" => tokens.push(Token::Not),
                    _ => {
                        let term = word.trim_start_matches(['^', '+', '-']);
                        if !term.is_empty() {
                            tokens.push(Token::Term(term.to_owned()));
                        }
                    }
                }
            }
        }
        rest = rest.trim_start();
    }
    tokens.extend(repeat_n(Token::Close, depth));
    tokens
}

enum Node {
    Term(String),
    And(Vec<Node>),
    Or(Vec<Node>),
    Not(Box<Node>, Box<Node>),
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn eat(&mut self, token: &Token) -> bool {
        let found = self.peek() == Some(token);
        if found {
            self.pos += 1;
        }
        found
    }

    fn list(
        &mut self,
        separator: &Token,
        item: fn(&mut Self) -> Result<Option<Node>, QuerySyntaxError>,
        join: fn(Vec<Node>) -> Node,
    ) -> Result<Option<Node>, QuerySyntaxError> {
        let mut items = Vec::new();
        loop {
            if self.eat(separator) {
                continue;
            }
            match item(self)? {
                Some(node) => items.push(node),
                None => break,
            }
        }
        Ok(match items.len() {
            0 => None,
            1 => items.pop(),
            _ => Some(join(items)),
        })
    }

    fn or(&mut self) -> Result<Option<Node>, QuerySyntaxError> {
        self.list(&Token::Or, Self::and, Node::Or)
    }

    fn and(&mut self) -> Result<Option<Node>, QuerySyntaxError> {
        self.list(&Token::And, Self::not, Node::And)
    }

    fn not(&mut self) -> Result<Option<Node>, QuerySyntaxError> {
        let Some(mut left) = self.primary()? else {
            if self.peek() == Some(&Token::Not) {
                return Err(QuerySyntaxError("NOT needs a term on its left".to_owned()));
            }
            return Ok(None);
        };
        while self.eat(&Token::Not) {
            if let Some(right) = self.primary()? {
                left = Node::Not(Box::new(left), Box::new(right));
            }
        }
        Ok(Some(left))
    }

    fn primary(&mut self) -> Result<Option<Node>, QuerySyntaxError> {
        match self.peek() {
            Some(Token::Term(text)) => {
                let node = Node::Term(text.clone());
                self.pos += 1;
                Ok(Some(node))
            }
            Some(Token::Open) => {
                self.pos += 1;
                let inner = self.or()?;
                self.eat(&Token::Close);
                match inner {
                    Some(node) => Ok(Some(node)),
                    None => self.primary(),
                }
            }
            _ => Ok(None),
        }
    }
}

impl Node {
    fn render(&self, not_operands: &mut Vec<String>) -> Result<Option<String>, QuerySyntaxError> {
        match self {
            Self::Term(text) => {
                Ok((text.chars().count() >= MIN_OPERATOR_TERM_CHARS).then(|| fts_quote(text)))
            }
            Self::And(nodes) => Self::render_group(nodes, " AND ", not_operands),
            Self::Or(nodes) => Self::render_group(nodes, " OR ", not_operands),
            Self::Not(left, right) => {
                let Some(left) = left.render(not_operands)? else {
                    return Err(QuerySyntaxError(
                        "NOT needs a left term of at least 3 characters".to_owned(),
                    ));
                };
                let Some(right) = right.render(not_operands)? else {
                    return Ok(Some(left));
                };
                let rendered = format!("({left} NOT {right})");
                not_operands.push(right);
                Ok(Some(rendered))
            }
        }
    }

    fn render_group(
        nodes: &[Self],
        separator: &str,
        not_operands: &mut Vec<String>,
    ) -> Result<Option<String>, QuerySyntaxError> {
        let mut parts = Vec::new();
        for node in nodes {
            parts.extend(node.render(not_operands)?);
        }
        Ok(match parts.len() {
            0 => None,
            1 => parts.pop(),
            _ => Some(format!("({})", parts.join(separator))),
        })
    }

    fn collect_embed_terms<'a>(&'a self, terms: &mut Vec<&'a str>) {
        match self {
            Self::Term(text) => terms.push(text),
            Self::And(nodes) | Self::Or(nodes) => {
                nodes.iter().for_each(|n| n.collect_embed_terms(terms));
            }
            Self::Not(left, _) => left.collect_embed_terms(terms),
        }
    }
}

/// Returns `Ok(None)` when the query has no uppercase AND / OR / NOT outside
/// quotes, so the caller keeps the plain path.
pub fn prepare_operator_query(query: &str) -> Result<Option<OperatorQuery>, QuerySyntaxError> {
    let tokens = tokenize_operator_query(query);
    if !tokens
        .iter()
        .any(|t| matches!(t, Token::And | Token::Or | Token::Not))
    {
        return Ok(None);
    }
    let mut parser = Parser { tokens, pos: 0 };
    let Some(tree) = parser.or()? else {
        return Ok(Some(OperatorQuery {
            match_query: None,
            not_operands: Vec::new(),
            embed_text: String::new(),
        }));
    };
    let mut not_operands = Vec::new();
    let match_query = tree.render(&mut not_operands)?;
    let mut terms = Vec::new();
    tree.collect_embed_terms(&mut terms);
    Ok(Some(OperatorQuery {
        match_query,
        not_operands,
        embed_text: terms.join(" "),
    }))
}

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

/// Splits a sanitized query into bare words and quoted phrases, so a phrase
/// is quoted once as a whole instead of once per word. The flag is true for a
/// quoted phrase, whose text comes back without its quotes.
fn split_terms(query: &str) -> Vec<(bool, &str)> {
    let mut terms = Vec::new();
    let mut rest = query.trim_start();
    while !rest.is_empty() {
        if let Some(after_quote) = rest.strip_prefix('"') {
            let end = after_quote.find('"').unwrap_or(after_quote.len());
            if end > 0 {
                terms.push((true, &after_quote[..end]));
            }
            rest = after_quote.get(end + 1..).unwrap_or("");
        } else {
            let end = rest
                .find(|c: char| c.is_whitespace() || c == '"')
                .unwrap_or(rest.len());
            terms.push((false, &rest[..end]));
            rest = &rest[end..];
        }
        rest = rest.trim_start();
    }
    terms
}

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
    for (quoted, token) in split_terms(query.as_str()) {
        if quoted || token.chars().count() >= 3 || is_fts5_operator(token) {
            parts.push(fts_quote(token));
            continue;
        }
        let expanded = match stmt.as_mut() {
            Some(s) => expand_token_via_vocab(s, token)?,
            None => None,
        };
        parts.push(expanded.unwrap_or_else(|| fts_quote(token)));
    }
    Ok(MatchFtsQuery(parts.join(" AND ")))
}

#[cfg(test)]
mod tests;
