use super::*;

const NOW_MS: i64 = 1_750_000_000_000;

#[test]
fn fusion_ranks_a_session_found_by_both_searches_above_sessions_found_by_one() {
    let merged = rrf_merge_sessions(["A", "B"], ["B", "C"]);

    assert_eq!(
        merged.iter().map(|(id, _)| id.as_str()).collect::<Vec<_>>(),
        ["B", "A", "C"]
    );
}

#[test]
fn fusion_of_vector_hits_alone_keeps_their_rank_order() {
    let merged = rrf_merge_sessions([], ["B", "A"]);

    assert_eq!(
        merged.iter().map(|(id, _)| id.as_str()).collect::<Vec<_>>(),
        ["B", "A"]
    );
}

#[test]
fn fusion_of_no_hits_is_empty() {
    assert!(rrf_merge_sessions([], []).is_empty());
}

#[test]
fn a_fresh_session_overtakes_a_year_old_one_that_scored_ten_percent_higher() {
    let mut results = vec![("old".to_owned(), 1.1), ("new".to_owned(), 1.0)];

    apply_recency_boost(
        &mut results,
        |sid| match sid {
            "new" => Some(NOW_MS),
            _ => Some(NOW_MS - 365 * MS_PER_DAY),
        },
        NOW_MS,
        RECENCY_BOOST_WEIGHT,
    );

    assert_eq!(results[0].0, "new");
}

#[test]
fn a_session_without_a_timestamp_keeps_its_score() {
    let mut results = vec![("undated".to_owned(), 0.5)];

    apply_recency_boost(&mut results, |_| None, NOW_MS, RECENCY_BOOST_WEIGHT);

    assert_eq!(results, [("undated".to_owned(), 0.5)]);
}
