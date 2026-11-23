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
