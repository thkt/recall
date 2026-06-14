use super::*;

#[test]
fn test_epoch() {
    assert_eq!(civil_from_days(0), (1970, 1, 1));
    assert_eq!(days_from_civil(1970, 1, 1), Some(0));
}

#[test]
fn test_known_date() {
    // 2024-03-01 = day 19783
    assert_eq!(civil_from_days(19783), (2024, 3, 1));
    assert_eq!(days_from_civil(2024, 3, 1), Some(19783));
}

#[test]
fn test_roundtrip() {
    for days in [0, 1, 365, 10000, 19783, 20000, -1, -719468] {
        let (y, m, d) = civil_from_days(days);
        assert_eq!(
            days_from_civil(y, m, d),
            Some(days),
            "roundtrip failed for day {days}"
        );
    }
}

#[test]
fn test_leap_year() {
    let days = days_from_civil(2024, 2, 29).unwrap();
    assert_eq!(civil_from_days(days), (2024, 2, 29));
}

#[test]
fn test_invalid_month_day() {
    assert_eq!(days_from_civil(2024, 0, 1), None);
    assert_eq!(days_from_civil(2024, 13, 1), None);
    assert_eq!(days_from_civil(2024, 1, 0), None);
    assert_eq!(days_from_civil(2024, 1, 32), None);
}

#[test]
fn test_invalid_day_per_month() {
    // Feb 29 on non-leap year
    assert_eq!(days_from_civil(2023, 2, 29), None);
    // Feb 30 even on leap year
    assert_eq!(days_from_civil(2024, 2, 30), None);
    // Apr, Jun, Sep, Nov have 30 days max
    assert_eq!(days_from_civil(2024, 4, 31), None);
    assert_eq!(days_from_civil(2024, 6, 31), None);
    assert_eq!(days_from_civil(2024, 9, 31), None);
    assert_eq!(days_from_civil(2024, 11, 31), None);
}
