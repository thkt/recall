use super::*;

#[test]
fn days_since_the_unix_epoch_convert_to_calendar_dates_and_back() {
    assert_eq!(civil_from_days(0), (1970, 1, 1));
    assert_eq!(civil_from_days(1), (1970, 1, 2));
    assert_eq!(civil_from_days(365), (1971, 1, 1));
    assert_eq!(civil_from_days(10_000), (1997, 5, 19));
    assert_eq!(civil_from_days(19_783), (2024, 3, 1));
    assert_eq!(civil_from_days(20_000), (2024, 10, 4));

    assert_eq!(days_from_civil(1970, 1, 1), Some(0));
    assert_eq!(days_from_civil(1971, 1, 1), Some(365));
    assert_eq!(days_from_civil(2024, 3, 1), Some(19_783));
    assert_eq!(days_from_civil(2024, 10, 4), Some(20_000));
}

#[test]
fn dates_before_the_unix_epoch_have_negative_day_counts() {
    assert_eq!(civil_from_days(-1), (1969, 12, 31));
    assert_eq!(days_from_civil(1969, 12, 31), Some(-1));
    assert_eq!(civil_from_days(-719_468), (0, 3, 1));
    assert_eq!(days_from_civil(0, 3, 1), Some(-719_468));
}

#[test]
fn a_leap_day_exists_only_in_leap_years() {
    assert_eq!(days_from_civil(2024, 2, 29), Some(19_782));
    assert_eq!(civil_from_days(19_782), (2024, 2, 29));
    assert_eq!(days_from_civil(2023, 2, 29), None);
    assert_eq!(days_from_civil(2024, 2, 30), None);
}

#[test]
fn months_and_days_outside_the_calendar_are_rejected() {
    assert_eq!(days_from_civil(2024, 0, 1), None);
    assert_eq!(days_from_civil(2024, 13, 1), None);
    assert_eq!(days_from_civil(2024, 1, 0), None);
    assert_eq!(days_from_civil(2024, 1, 32), None);
    assert_eq!(days_from_civil(2024, 4, 31), None);
    assert_eq!(days_from_civil(2024, 6, 31), None);
    assert_eq!(days_from_civil(2024, 9, 31), None);
    assert_eq!(days_from_civil(2024, 11, 31), None);
}
