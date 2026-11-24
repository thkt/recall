use rusqlite::types::ToSql;

use super::{CurrentSession, SearchOptions};
use crate::date::MS_PER_DAY;
use crate::parser::Source;

pub(super) fn anon_placeholders(n: usize) -> String {
    vec!["?"; n].join(",")
}

fn escape_like(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('%', "\\%")
        .replace('_', "\\_")
}

pub(crate) fn like_prefix(s: &str) -> String {
    format!("{}%", escape_like(s))
}

fn append_eq_filter(sql: &mut String, p: &mut Vec<Box<dyn ToSql>>, col: &str, v: Option<&str>) {
    if let Some(v) = v {
        sql.push_str(&format!(" AND {col} = ?"));
        p.push(Box::new(v.to_owned()));
    }
}

fn append_timestamp_cutoff_filter(
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

fn append_like_prefix_filter(
    sql: &mut String,
    p: &mut Vec<Box<dyn ToSql>>,
    col: &str,
    v: Option<&str>,
) {
    if let Some(v) = v {
        sql.push_str(&format!(" AND ({col} LIKE ? ESCAPE '\\')"));
        p.push(Box::new(like_prefix(v)));
    }
}

pub(super) fn append_search_filters(
    sql: &mut String,
    params: &mut Vec<Box<dyn ToSql>>,
    column: &'static str,
    opts: &SearchOptions,
    now_ms: i64,
) {
    if opts.project.is_some() || opts.days.is_some() || opts.source.is_some() {
        sql.push_str(" AND ");
        sql.push_str(column);
        sql.push_str(" IN (SELECT s2.session_id FROM sessions s2 WHERE 1 = 1");
        append_like_prefix_filter(sql, params, "s2.project", opts.project.as_deref());
        let cutoff = opts.days.map(|d| now_ms - d * MS_PER_DAY);
        append_timestamp_cutoff_filter(sql, params, "s2.timestamp", cutoff);
        let source_str = opts.source.as_ref().map(Source::as_str);
        append_eq_filter(sql, params, "s2.source", source_str);
        sql.push(')');
    }
    if let Some(ref file) = opts.file {
        sql.push_str(" AND ");
        sql.push_str(column);
        sql.push_str(
            " IN (SELECT sf.session_id FROM session_files sf \
             WHERE sf.path = ? OR sf.path LIKE ? ESCAPE '\\')",
        );
        params.push(Box::new(file.clone()));
        params.push(Box::new(format!("%/{}", escape_like(file))));
    }
    let current = match &opts.current {
        CurrentSession::Ignore => None,
        CurrentSession::Exclude(id) => Some((" <> ?", id)),
        CurrentSession::Only(id) => Some((" = ?", id)),
    };
    if let Some((op, id)) = current {
        sql.push_str(" AND ");
        sql.push_str(column);
        sql.push_str(op);
        params.push(Box::new(id.clone()));
    }
    if !opts.include_automated {
        sql.push_str(" AND ");
        sql.push_str(column);
        sql.push_str(
            " IN (SELECT s3.session_id FROM sessions s3 \
             WHERE COALESCE(s3.session_type, 'interactive') <> 'automated')",
        );
    }
}
