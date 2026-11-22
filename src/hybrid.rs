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

pub(crate) fn recency_factor(now_ms: i64, ts: Option<i64>) -> f64 {
    1.0 + RECENCY_BOOST_WEIGHT * recency_decay(now_ms, ts)
}

pub(crate) const MAX_RECENCY_FACTOR: f64 = 1.0 + RECENCY_BOOST_WEIGHT;

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
    fts_hits: impl IntoIterator<Item = (&'a str, f64)>,
    vec_hits: impl IntoIterator<Item = (&'a str, f64)>,
) -> Vec<(String, f64)> {
    let mut scores: HashMap<String, f64> = HashMap::new();
    let fts_hits: Vec<_> = fts_hits.into_iter().collect();
    let vec_hits: Vec<_> = vec_hits.into_iter().collect();
    for (sid, rank) in tied_ranks(&fts_hits).chain(tied_ranks(&vec_hits)) {
        *scores.entry(sid.to_owned()).or_default() += 1.0 / (60.0 + rank);
    }
    let mut result: Vec<_> = scores.into_iter().collect();
    result.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    result
}

fn tied_ranks<'a>(hits: &[(&'a str, f64)]) -> impl Iterator<Item = (&'a str, f64)> {
    debug_assert!(
        hits.windows(2).all(|pair| pair[0].1 <= pair[1].1),
        "hits must be listed best first"
    );
    let mut ranks = Vec::with_capacity(hits.len());
    for tie in hits.chunk_by(|a, b| a.1 == b.1) {
        let rank = ranks.len() as f64 + (tie.len() - 1) as f64 / 2.0;
        ranks.extend(tie.iter().map(|&(sid, _)| (sid, rank)));
    }
    ranks.into_iter()
}

#[cfg(test)]
mod tests;
