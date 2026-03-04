/// Days since 1970-01-01 for a given civil date.
///
/// Returns `None` if month or day are out of range.
///
/// Howard Hinnant's algorithm: <https://howardhinnant.github.io/date_algorithms.html>
pub fn days_from_civil(year: i64, month: i64, day: i64) -> Option<i64> {
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }
    let (y, m) = if month <= 2 {
        (year - 1, month + 9)
    } else {
        (year, month - 3)
    };
    let era = y / 400;
    let yoe = y - era * 400;
    let doy = (153 * m + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    Some(era * 146097 + doe - 719468)
}

/// Civil date (year, month, day) from days since 1970-01-01.
///
/// Howard Hinnant's algorithm: <https://howardhinnant.github.io/date_algorithms.html>
pub fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z.rem_euclid(146097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

#[cfg(test)]
mod tests {
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
            assert_eq!(days_from_civil(y, m, d), Some(days), "roundtrip failed for day {days}");
        }
    }

    #[test]
    fn test_leap_year() {
        // 2024-02-29 (leap year)
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
}
