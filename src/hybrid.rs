use std::collections::HashMap;

use crate::date::MS_PER_DAY;

/// RRF parameter. Higher k reduces the influence of high rankings.
const RRF_K: f64 = 60.0;
/// Sessions older than 30 days lose half their recency boost.
pub(crate) const RECENCY_HALF_LIFE_DAYS: f64 = 30.0;

/// Exponential recency decay: 1.0 for now, 0.5 at half-life, approaching 0.0.
pub(crate) fn recency_decay(now_ms: i64, ts: Option<i64>) -> f64 {
    match ts {
        Some(ts) => {
            let age_days = ((now_ms as f64 - ts as f64) / MS_PER_DAY as f64).max(0.0);
            (-std::f64::consts::LN_2 * age_days / RECENCY_HALF_LIFE_DAYS).exp()
        }
        None => 0.0,
    }
}

/// A scored candidate from one retrieval source (FTS5 or vector).
pub(crate) struct RankedHit {
    pub session_id: String,
    /// 0-based rank within its source list (0 = best).
    pub rank: usize,
}

/// Merge FTS5 and vector results using Reciprocal Rank Fusion.
///
/// Sessions appearing in both lists receive scores from both.
/// Score = sum(1 / (k + rank)) across lists.
pub(crate) fn rrf_merge(fts_hits: &[RankedHit], vec_hits: &[RankedHit]) -> Vec<(String, f64)> {
    let mut scores: HashMap<String, f64> = HashMap::new();

    for hit in fts_hits {
        *scores.entry(hit.session_id.clone()).or_default() += 1.0 / (RRF_K + hit.rank as f64);
    }
    for hit in vec_hits {
        *scores.entry(hit.session_id.clone()).or_default() += 1.0 / (RRF_K + hit.rank as f64);
    }

    let mut results: Vec<(String, f64)> = scores.into_iter().collect();
    results.sort_by(|a, b| b.1.total_cmp(&a.1)); // descending by score
    results
}

/// Apply exponential recency boost to RRF scores.
///
/// `get_timestamp` resolves a session_id to its epoch-ms timestamp.
pub(crate) fn apply_recency_boost(
    results: &mut [(String, f64)],
    get_timestamp: impl Fn(&str) -> Option<i64>,
    now_ms: i64,
) {
    for (session_id, score) in results.iter_mut() {
        let boost = recency_decay(now_ms, get_timestamp(session_id));
        *score *= 1.0 + boost;
    }
    results.sort_by(|a, b| b.1.total_cmp(&a.1));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hit(session_id: &str, rank: usize) -> RankedHit {
        RankedHit {
            session_id: session_id.to_string(),
            rank,
        }
    }

    // T-008: FTS5 + vec results merged by RRF (FR-005)
    #[test]
    fn test_008_rrf_merge_both_lists() {
        let fts = vec![hit("A", 0), hit("B", 2)];
        let vec = vec![hit("B", 0), hit("C", 1)];

        let merged = rrf_merge(&fts, &vec);

        // B appears in both lists → highest score
        assert_eq!(merged[0].0, "B");
        // B's score = 1/(60+2) + 1/(60+0) = 1/62 + 1/60
        let b_expected = 1.0 / 62.0 + 1.0 / 60.0;
        assert!(
            (merged[0].1 - b_expected).abs() < 1e-10,
            "B score: expected {b_expected}, got {}",
            merged[0].1
        );
    }

    // T-010: single-list hit scoring (FR-005)
    #[test]
    fn test_010_rrf_single_list_hit() {
        let fts = vec![hit("A", 0)];
        let vec: Vec<RankedHit> = vec![];

        let merged = rrf_merge(&fts, &vec);

        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].0, "A");
        let expected = 1.0 / (60.0 + 0.0);
        assert!(
            (merged[0].1 - expected).abs() < 1e-10,
            "single-list score: expected {expected}, got {}",
            merged[0].1
        );
    }

    // T-009: recency boost after RRF (FR-005)
    #[test]
    fn test_009_recency_boost_favors_newer() {
        let now_ms = 1_750_000_000_000_i64;

        // Same RRF score but different timestamps
        let mut results = vec![("old".to_string(), 0.01), ("new".to_string(), 0.01)];

        let mut timestamps = HashMap::new();
        timestamps.insert("old".to_string(), Some(now_ms - 365 * MS_PER_DAY));
        timestamps.insert("new".to_string(), Some(now_ms));

        apply_recency_boost(
            &mut results,
            |sid| timestamps.get(sid).copied().flatten(),
            now_ms,
        );

        // "new" should be first after boost
        assert_eq!(results[0].0, "new");
        assert!(
            results[0].1 > results[1].1,
            "new ({}) should score higher than old ({})",
            results[0].1,
            results[1].1
        );
    }

    #[test]
    fn test_recency_boost_no_timestamp() {
        let mut results = vec![("no-ts".to_string(), 0.5)];
        apply_recency_boost(&mut results, |_| None, 1_000_000);

        // boost = 0 → score * (1 + 0) = score unchanged
        assert!((results[0].1 - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_rrf_empty_inputs() {
        let merged = rrf_merge(&[], &[]);
        assert!(merged.is_empty());
    }
}
