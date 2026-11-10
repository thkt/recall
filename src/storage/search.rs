use std::iter::repeat_n;

use rusqlite::Connection;

use super::query_normalize::{QueryNormalizationConfig, normalize_for_fts};

/// A quoted phrase or a bare word of a query. A phrase runs from a quote to the
/// next quote, or to the end of the query when the quote is unclosed, and its
/// text excludes the quotes.
#[derive(Debug, Clone, Copy)]
enum Piece<'a> {
    Phrase(&'a str),
    Word(&'a str),
}

/// Splits a query at whitespace outside quotes into tokens, each the phrases
/// and words written with nothing between them. A word ends at whitespace or a
/// quote; a char for which `stands_alone` is true also ends a word and becomes
/// a one-char word of its own. The plain sanitizer, the short-term expansion
/// and the operator tokenizer all read phrases through this function, so they
/// agree on where a phrase starts and ends.
fn scan_tokens(query: &str, stands_alone: fn(char) -> bool) -> Vec<Vec<Piece<'_>>> {
    let mut tokens: Vec<Vec<Piece<'_>>> = Vec::new();
    let mut rest = query;
    loop {
        let trimmed = rest.trim_start();
        let attached = trimmed.len() == rest.len();
        rest = trimmed;
        let Some(c) = rest.chars().next() else {
            break;
        };
        let piece = if let Some(after_quote) = rest.strip_prefix('"') {
            let end = after_quote.find('"').unwrap_or(after_quote.len());
            rest = after_quote.get(end + 1..).unwrap_or("");
            Piece::Phrase(&after_quote[..end])
        } else {
            let end = if stands_alone(c) {
                c.len_utf8()
            } else {
                rest.find(|c: char| c.is_whitespace() || c == '"' || stands_alone(c))
                    .unwrap_or(rest.len())
            };
            let word = &rest[..end];
            rest = &rest[end..];
            Piece::Word(word)
        };
        match tokens.last_mut() {
            Some(token) if attached => token.push(piece),
            _ => tokens.push(vec![piece]),
        }
    }
    tokens
}

fn count_in_words(token: &[Piece<'_>], paren: char) -> usize {
    token
        .iter()
        .map(|piece| match piece {
            Piece::Word(word) => word.matches(paren).count(),
            Piece::Phrase(_) => 0,
        })
        .sum()
}

fn strip_near_groups(query: &str) -> Vec<Vec<Piece<'_>>> {
    let mut tokens = Vec::new();
    let mut paren_depth: usize = 0;
    for token in scan_tokens(query, |_| false) {
        let opens_near = match token.first() {
            Some(Piece::Word(w)) => {
                let upper = w.to_ascii_uppercase();
                upper.starts_with("NEAR(") || upper.starts_with("NEAR/")
            }
            _ => false,
        };
        if opens_near {
            paren_depth += count_in_words(&token, '(');
            paren_depth = paren_depth.saturating_sub(count_in_words(&token, ')'));
            continue;
        }
        if paren_depth > 0 {
            paren_depth = paren_depth.saturating_sub(count_in_words(&token, ')'));
            continue;
        }
        tokens.push(token);
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
    let tokens: Vec<String> = strip_near_groups(query)
        .iter()
        .map(|token| clean_token(token))
        .filter(|w| !w.is_empty())
        .collect();
    let result = drop_dangling_operators(&tokens).join(" ");
    if result.is_empty() {
        return Err(SanitizeError::NoSearchableTerms);
    }
    Ok(SanitizedFtsQuery(result))
}

/// A lone word loses its `^` / `+` / `-` prefix and its outer parentheses. It is
/// quoted when it holds `:` or `-`. A token holding a phrase keeps the phrase
/// verbatim and closed, and only the words at its edges lose the prefix and
/// parentheses. Such a token is never an operator, so `drop_dangling_operators`
/// leaves the and / or / not words inside a phrase in place.
fn clean_token(token: &[Piece<'_>]) -> String {
    if let [Piece::Word(word)] = token {
        let cleaned = word
            .trim_start_matches(['^', '+', '-'])
            .trim_matches(['(', ')']);
        return if cleaned.contains([':', '-']) {
            format!("\"{cleaned}\"")
        } else {
            cleaned.to_owned()
        };
    }
    let last = token.len() - 1;
    token
        .iter()
        .enumerate()
        .map(|(i, piece)| match piece {
            Piece::Phrase(text) => format!("\"{text}\""),
            Piece::Word(word) => {
                let word = if i == 0 {
                    word.trim_start_matches(['^', '+', '-', '(', ')'])
                } else {
                    word
                };
                let word = if i == last {
                    word.trim_end_matches(['(', ')'])
                } else {
                    word
                };
                word.to_owned()
            }
        })
        .collect()
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
    let pieces = scan_tokens(query, |c| matches!(c, '(' | ')'));
    for piece in pieces.into_iter().flatten() {
        match piece {
            Piece::Phrase("") => {}
            Piece::Phrase(text) => tokens.push(Token::Term(text.to_owned())),
            Piece::Word("(") => {
                depth += 1;
                tokens.push(Token::Open);
            }
            Piece::Word(")") => {
                if depth > 0 {
                    depth -= 1;
                    tokens.push(Token::Close);
                }
            }
            Piece::Word("AND") => tokens.push(Token::And),
            Piece::Word("OR") => tokens.push(Token::Or),
            Piece::Word("NOT") => tokens.push(Token::Not),
            Piece::Word(word) => {
                let term = word.trim_start_matches(['^', '+', '-']);
                if !term.is_empty() {
                    tokens.push(Token::Term(term.to_owned()));
                }
            }
        }
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
    // A phrase is quoted once as a whole instead of once per word.
    let pieces = scan_tokens(query.as_str(), |_| false);
    for piece in pieces.into_iter().flatten() {
        let token = match piece {
            Piece::Phrase("") => continue,
            Piece::Phrase(text) => {
                parts.push(fts_quote(text));
                continue;
            }
            Piece::Word(word) => word,
        };
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
    Ok(MatchFtsQuery(parts.join(" AND ")))
}

#[cfg(test)]
mod tests;
