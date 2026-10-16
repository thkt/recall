use std::f64::consts::LN_2;

use std::collections::HashMap;

use crate::date::MS_PER_DAY;

pub(crate) const RECENCY_HALF_LIFE_DAYS: f64 = 30.0;
pub(crate) const RECENCY_BOOST_WEIGHT: f64 = 0.2;

pub(crate) fn recency_decay(now_ms: i64, ts: Option<i64>) -> f64 {
    match ts {
        Some(ts) => {
            let age_days = ((now_ms as f64 - ts as f64) / MS_PER_DAY as f64).max(0.0);
            (-LN_2 * age_days / RECENCY_HALF_LIFE_DAYS).exp()
        }
        None => 0.0,
    }
}

pub(crate) fn apply_recency_boost(
    results: &mut [(String, f64)],
    get_timestamp: impl Fn(&str) -> Option<i64>,
    now_ms: i64,
    weight: f64,
) {
    for (session_id, score) in results.iter_mut() {
        let boost = recency_decay(now_ms, get_timestamp(session_id));
        *score *= 1.0 + weight * boost;
    }
    results.sort_by(|a, b| b.1.total_cmp(&a.1));
}

pub(crate) fn rrf_merge_sessions<'a>(
    fts_ids: impl IntoIterator<Item = &'a str>,
    vec_ids: impl IntoIterator<Item = &'a str>,
) -> Vec<(String, f64)> {
    let mut scores: HashMap<String, f64> = HashMap::new();
    // Enumerate each source independently: fusion uses rank, not input scores.
    for (rank, sid) in fts_ids
        .into_iter()
        .enumerate()
        .chain(vec_ids.into_iter().enumerate())
    {
        *scores.entry(sid.to_owned()).or_default() += 1.0 / (60.0 + rank as f64);
    }
    let mut result: Vec<_> = scores.into_iter().collect();
    result.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    result
}

#[cfg(test)]
mod tests;
