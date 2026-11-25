use super::*;
use crate::db::setup_test_db;

fn owners(ranges: &[(&str, i64, i64)]) -> RangeOwners {
    RangeOwners::from_ranges(
        ranges
            .iter()
            .map(|(sid, lo, hi)| ((*sid).to_owned(), *lo, *hi))
            .collect(),
    )
}

fn owner_of(owners: &RangeOwners, rowids: impl IntoIterator<Item = i64>) -> Vec<Option<&str>> {
    rowids.into_iter().map(|r| owners.sole_owner(r)).collect()
}

#[test]
fn a_rowid_belongs_to_a_range_from_its_first_rowid_through_its_last() {
    let owners = owners(&[("a", 3, 5)]);

    assert_eq!(
        owner_of(&owners, 2..=6),
        [None, Some("a"), Some("a"), Some("a"), None]
    );
}

#[test]
fn adjacent_ranges_each_own_their_rowids() {
    let owners = owners(&[("b", 4, 6), ("a", 1, 3)]);

    assert_eq!(
        owner_of(&owners, 1..=7),
        [
            Some("a"),
            Some("a"),
            Some("a"),
            Some("b"),
            Some("b"),
            Some("b"),
            None
        ]
    );
}

#[test]
fn a_rowid_inside_two_ranges_has_no_sole_owner() {
    let owners = owners(&[("wide", 1, 10), ("inner", 4, 5), ("tail", 9, 12)]);

    assert_eq!(
        owner_of(&owners, [1, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13]),
        [
            Some("wide"),
            Some("wide"),
            None,
            None,
            Some("wide"),
            Some("wide"),
            None,
            None,
            Some("tail"),
            Some("tail"),
            None
        ]
    );
}

#[test]
fn an_empty_range_owns_no_rowid_and_hides_no_other_range() {
    let owners = owners(&[("empty", 1, 0), ("a", 1, 2)]);

    assert_eq!(owner_of(&owners, 0..=3), [None, Some("a"), Some("a"), None]);
}

#[test]
fn ranges_at_the_ends_of_the_rowid_space_are_looked_up_without_overflow() {
    let owners = owners(&[
        ("low", i64::MIN, i64::MIN),
        ("high", i64::MAX - 1, i64::MAX),
    ]);

    assert_eq!(
        owner_of(&owners, [i64::MIN, i64::MIN + 1, i64::MAX - 1, i64::MAX]),
        [Some("low"), None, Some("high"), Some("high")]
    );
}

#[test]
fn sessions_are_resolved_per_row_when_any_session_lacks_a_range() {
    let (_dir, conn) = setup_test_db();
    conn.execute_batch(
        "INSERT INTO sessions (session_id, message_rowid_lo, message_rowid_hi) VALUES ('a', 1, 2);
         INSERT INTO sessions (session_id, message_rowid_lo, message_rowid_hi) VALUES ('b', 3, NULL);",
    )
    .unwrap();

    assert!(matches!(
        MessageSessions::load(&conn).unwrap(),
        MessageSessions::StoredPerRow
    ));

    conn.execute(
        "UPDATE sessions SET message_rowid_hi = 3 WHERE session_id = 'b'",
        [],
    )
    .unwrap();

    assert!(matches!(
        MessageSessions::load(&conn).unwrap(),
        MessageSessions::RowidRanges(_)
    ));
}

#[test]
fn sessions_are_resolved_per_row_when_the_index_has_no_range_columns() {
    let (_dir, conn) = setup_test_db();
    conn.execute_batch(
        "ALTER TABLE sessions DROP COLUMN message_rowid_lo;
         ALTER TABLE sessions DROP COLUMN message_rowid_hi;",
    )
    .unwrap();

    assert!(matches!(
        MessageSessions::load(&conn).unwrap(),
        MessageSessions::StoredPerRow
    ));
}

#[test]
fn sessions_are_resolved_per_row_when_a_session_id_is_not_utf8_text() {
    for session_id in ["NULL", "x'0102'", "CAST(x'ff' AS TEXT)"] {
        let (_dir, conn) = setup_test_db();
        conn.execute_batch(&format!(
            "INSERT INTO sessions (session_id, message_rowid_lo, message_rowid_hi) VALUES ('a', 1, 2);
             INSERT INTO sessions (session_id, message_rowid_lo, message_rowid_hi) VALUES ({session_id}, 3, 4);"
        ))
        .unwrap();

        assert!(
            matches!(
                MessageSessions::load(&conn).unwrap(),
                MessageSessions::StoredPerRow
            ),
            "{session_id}"
        );
    }
}
