use crate::storage::MatchFtsQuery;

const MAX_COMBINATIONS: usize = 100;
enum Segment {
    Fixed(String),
    Alternatives(Vec<String>),
}
struct Wire<'a> {
    text: &'a [u8],
    pos: usize,
}
impl Wire<'_> {
    fn whitespace(&mut self) {
        while self.text.get(self.pos).is_some_and(u8::is_ascii_whitespace) {
            self.pos += 1;
        }
    }
    fn take(&mut self, word: &[u8]) -> bool {
        if self.text[self.pos..].starts_with(word) {
            self.pos += word.len();
            true
        } else {
            false
        }
    }
    fn quoted(&mut self) -> Option<String> {
        if !self.take(b"\"") {
            return None;
        }
        let mut decoded = Vec::new();
        while let Some(&ch) = self.text.get(self.pos) {
            self.pos += 1;
            if ch == b'"' {
                if self.text.get(self.pos) == Some(&b'"') {
                    decoded.push(b'"');
                    self.pos += 1;
                } else {
                    return String::from_utf8(decoded).ok();
                }
            } else {
                decoded.push(ch);
            }
        }
        None
    }
    fn segments(mut self) -> Option<Vec<Segment>> {
        let mut segments = Vec::new();
        self.whitespace();
        while self.pos < self.text.len() {
            let segment = if self.take(b"(") {
                self.whitespace();
                let mut terms = vec![self.quoted()?];
                loop {
                    self.whitespace();
                    if self.take(b")") {
                        break;
                    }
                    if !self.take(b"OR") {
                        return None;
                    }
                    self.whitespace();
                    terms.push(self.quoted()?);
                }
                Segment::Alternatives(terms)
            } else {
                Segment::Fixed(self.quoted()?)
            };
            segments.push(segment);
            self.whitespace();
            if self.pos == self.text.len() {
                break;
            }
            if !self.take(b"AND") {
                return None;
            }
            self.whitespace();
            if self.pos == self.text.len() {
                return None;
            }
        }
        Some(segments)
    }
}
fn quote(term: &str) -> String {
    format!("\"{}\"", term.replace('"', "\"\""))
}

pub fn clean_for_trigram(query: &MatchFtsQuery) -> Option<String> {
    // The frozen FTS contract caps combinations before discarding short terms.
    let segments = Wire {
        text: query.as_str().as_bytes(),
        pos: 0,
    }
    .segments()?;
    let combinations = segments.iter().fold(1usize, |n, segment| match segment {
        Segment::Fixed(_) => n,
        Segment::Alternatives(terms) => n.saturating_mul(terms.len()),
    });
    let overflow = combinations > MAX_COMBINATIONS;
    let mut clauses: Vec<Vec<String>> = vec![Vec::new()];
    for segment in segments {
        match segment {
            Segment::Fixed(term) if term.chars().count() >= 3 => {
                let term = quote(&term);
                for clause in &mut clauses {
                    clause.push(term.clone());
                }
            }
            Segment::Alternatives(terms) if !overflow => {
                let terms: Vec<_> = terms
                    .into_iter()
                    .filter(|t| t.chars().count() >= 3)
                    .map(|t| quote(&t))
                    .collect();
                if terms.is_empty() {
                    continue;
                }
                clauses = clauses
                    .into_iter()
                    .flat_map(|clause| {
                        terms.iter().map(move |term| {
                            let mut expanded = clause.clone();
                            expanded.push(term.clone());
                            expanded
                        })
                    })
                    .collect();
            }
            _ => {}
        }
    }
    if clauses[0].is_empty() {
        return None;
    }
    if clauses.len() == 1 {
        return Some(clauses.remove(0).join(" AND "));
    }
    Some(
        clauses
            .into_iter()
            .map(|terms| format!("({})", terms.join(" AND ")))
            .collect::<Vec<_>>()
            .join(" OR "),
    )
}

pub fn anon_placeholders(n: usize) -> String {
    vec!["?"; n].join(",")
}
#[cfg(test)]
pub fn assert_filter_symmetric<S, F, U>(seed: S, filtered: F, unfiltered: U)
where
    S: FnOnce() -> (String, String),
    F: FnOnce() -> Vec<String>,
    U: FnOnce() -> Vec<String>,
{
    let (keep, drop) = seed();
    let before = unfiltered();
    assert!(
        before.contains(&keep) && before.contains(&drop),
        "unfiltered paths must reach both seeded sessions"
    );
    let after = filtered();
    assert!(
        after.contains(&keep) && !after.contains(&drop),
        "filter must retain the matching session and remove the nonmatching session from the merged result"
    );
}
pub mod filter {
    use rusqlite::types::ToSql;
    pub fn escape_like(s: &str) -> String {
        s.replace('\\', "\\\\")
            .replace('%', "\\%")
            .replace('_', "\\_")
    }
    pub fn append_eq_filter(
        sql: &mut String,
        p: &mut Vec<Box<dyn ToSql>>,
        col: &str,
        v: Option<&str>,
    ) {
        if let Some(v) = v {
            sql.push_str(&format!(" AND {col} = ?"));
            p.push(Box::new(v.to_owned()));
        }
    }
    pub fn append_timestamp_cutoff_filter(
        sql: &mut String,
        p: &mut Vec<Box<dyn ToSql>>,
        col: &str,
        v: Option<i64>,
    ) {
        if let Some(v) = v {
            sql.push_str(&format!(" AND {col} >= ?"));
            p.push(Box::new(v));
        }
    }
    pub fn append_like_prefix_filter(
        sql: &mut String,
        p: &mut Vec<Box<dyn ToSql>>,
        col: &str,
        vals: &[String],
    ) {
        if !vals.is_empty() {
            sql.push_str(" AND (");
            sql.push_str(
                &vals
                    .iter()
                    .map(|_| format!("{col} LIKE ? ESCAPE '\\'"))
                    .collect::<Vec<_>>()
                    .join(" OR "),
            );
            sql.push(')');
            for v in vals {
                p.push(Box::new(format!("{}%", escape_like(v))));
            }
        }
    }
}
