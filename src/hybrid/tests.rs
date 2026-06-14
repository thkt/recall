use std::collections::HashMap;

use super::*;

// T-008: FTS5 + vec results merged by RRF (FR-005)
#[test]
fn test_008_rrf_merge_both_lists() {
    let fts: Vec<(String, f64)> = vec![("A".into(), 0.0), ("B".into(), 0.0)];
    // B at position 0 in vec list (rank=0), C at position 1 (rank=1)
    let vec: Vec<(String, f64)> = vec![("B".into(), 0.0), ("C".into(), 0.0)];

    let merged = rrf_merge_strings(&fts, &vec);

    // B appears in both lists → highest score
    assert_eq!(merged[0].0, "B");
    // B: fts rank=1 (position 1), vec rank=0 → 1/(60+1) + 1/(60+0)
    let b_expected = 1.0 / 61.0 + 1.0 / 60.0;
    assert!(
        (merged[0].1 - b_expected).abs() < 1e-10,
        "B score: expected {b_expected}, got {}",
        merged[0].1
    );
}

// T-010: single-list hit scoring (FR-005)
#[test]
fn test_010_rrf_single_list_hit() {
    let fts: Vec<(String, f64)> = vec![("A".into(), 0.0)];
    let vec: Vec<(String, f64)> = vec![];

    let merged = rrf_merge_strings(&fts, &vec);

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
    let mut results = vec![("old".to_owned(), 0.01), ("new".to_owned(), 0.01)];

    let mut timestamps = HashMap::new();
    timestamps.insert("old".to_owned(), Some(now_ms - 365 * MS_PER_DAY));
    timestamps.insert("new".to_owned(), Some(now_ms));

    apply_recency_boost(
        &mut results,
        |sid| timestamps.get(sid).copied().flatten(),
        now_ms,
        0.2,
    );

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
    let mut results = vec![("no-ts".to_owned(), 0.5)];
    apply_recency_boost(&mut results, |_| None, 1_000_000, 0.2);

    assert!((results[0].1 - 0.5).abs() < 1e-10);
}

#[test]
fn test_apply_recency_boost_scales_with_weight() {
    // At now (age=0), recency_decay returns 1.0, so final score = base * (1.0 + weight).
    let now_ms = 1_000_000;
    let base = 1.0;

    for (weight, expected) in [(0.0, 1.0), (0.2, 1.2), (1.0, 2.0)] {
        let mut results = vec![("s1".to_owned(), base)];
        apply_recency_boost(&mut results, |_| Some(now_ms), now_ms, weight);
        assert!(
            (results[0].1 - expected).abs() < 1e-10,
            "weight={weight} → expected {expected}, got {}",
            results[0].1
        );
    }
}

#[test]
fn test_rrf_empty_inputs() {
    let fts: Vec<(String, f64)> = vec![];
    let vec: Vec<(String, f64)> = vec![];
    let merged = rrf_merge_strings(&fts, &vec);
    assert!(merged.is_empty());
}
