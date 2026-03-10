mod claude;
mod codex;

pub use claude::parse_claude_session;
pub use codex::parse_codex_session;

use std::path::Path;

use anyhow::{Context, Result};
use serde_json::Value;

use crate::date;

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Source {
    Claude,
    Codex,
}

impl Source {
    pub fn as_str(&self) -> &'static str {
        match self {
            Source::Claude => "claude",
            Source::Codex => "codex",
        }
    }

    pub fn from_db(s: &str) -> Option<Source> {
        match s {
            "claude" => Some(Source::Claude),
            "codex" => Some(Source::Codex),
            _ => None,
        }
    }
}

impl std::fmt::Display for Source {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
}

impl Role {
    pub fn as_str(&self) -> &'static str {
        match self {
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }
}

#[derive(Debug, Clone)]
pub struct SessionData {
    pub session_id: String,
    pub source: Source,
    pub file_path: String,
    pub project: String,
    /// Short display name for the session (slug from Claude, date+uuid from Codex).
    pub slug: String,
    /// Epoch milliseconds (UTC). `None` means unknown.
    pub timestamp: Option<i64>,
}

/// Tool calls and system messages are excluded.
pub struct Message {
    pub role: Role,
    pub text: String,
}

pub struct ParseResult {
    pub metadata: SessionData,
    pub messages: Vec<Message>,
    pub skipped_lines: usize,
}

const TEXT_BLOCK_TYPES: &[&str] = &["text", "input_text", "output_text"];

pub(super) fn extract_text(content: Option<&Value>) -> String {
    let Some(content) = content else {
        return String::new();
    };
    match content {
        Value::String(s) => s.clone(),
        Value::Array(blocks) => {
            let mut result = String::new();
            for block in blocks {
                let Some(block_type) = block.get("type").and_then(|v| v.as_str()) else {
                    continue;
                };
                if TEXT_BLOCK_TYPES.contains(&block_type)
                    && let Some(text) = block.get("text").and_then(|v| v.as_str())
                {
                    if !result.is_empty() {
                        result.push('\n');
                    }
                    result.push_str(text);
                }
            }
            result
        }
        _ => String::new(),
    }
}

/// Byte offset where seconds end in `YYYY-MM-DDTHH:MM:SS` (position of '.' or tz).
const ISO_SECONDS_END: usize = 19;
/// Byte offset where fractional digits begin (after the '.').
const ISO_FRAC_START: usize = 20;

fn parse_fractional_millis(s: &str, b: &[u8]) -> i64 {
    if s.len() <= ISO_FRAC_START || b[ISO_SECONDS_END] != b'.' {
        return 0;
    }
    let end = s[ISO_FRAC_START..]
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(s.len() - ISO_FRAC_START);
    let frac = &b[ISO_FRAC_START..ISO_FRAC_START + end];
    let d = |i: usize| (frac[i] - b'0') as i64;
    match frac.len().min(3) {
        1 => d(0) * 100,
        2 => d(0) * 100 + d(1) * 10,
        3 => d(0) * 100 + d(1) * 10 + d(2),
        _ => 0,
    }
}

/// Parse `HH:MM` or `HHMM` into (hours, minutes). Returns `None` if too short.
fn parse_hhmm(s: &str) -> Option<(i64, i64)> {
    let b = s.as_bytes();
    if s.len() >= 5 && b[2] == b':' {
        Some((s[0..2].parse().ok()?, s[3..5].parse().ok()?))
    } else if s.len() >= 4 {
        Some((s[0..2].parse().ok()?, s[2..4].parse().ok()?))
    } else {
        None
    }
}

/// Parse timezone offset (e.g. `Z`, `+09:00`, `-0530`) and return offset in milliseconds.
/// Positive offset means local time is ahead of UTC (subtract to get UTC).
fn parse_tz_offset_ms(s: &str, b: &[u8]) -> i64 {
    let tz_start = if s.len() > ISO_SECONDS_END && b[ISO_SECONDS_END] == b'.' {
        s[ISO_FRAC_START..]
            .find(|c: char| !c.is_ascii_digit())
            .map(|i| ISO_FRAC_START + i)
            .unwrap_or(s.len())
    } else {
        ISO_SECONDS_END
    };
    if tz_start >= s.len() {
        return 0;
    }
    match b[tz_start] {
        b'Z' => 0,
        b'+' | b'-' => {
            let sign = if b[tz_start] == b'+' { 1_i64 } else { -1_i64 };
            let Some((h, m)) = parse_hhmm(&s[tz_start + 1..]) else {
                return 0;
            };
            sign * (h * 3_600_000 + m * 60_000)
        }
        _ => 0,
    }
}

pub(super) fn parse_iso_timestamp(val: &Value) -> Option<i64> {
    match val {
        Value::String(s) => {
            let s = s.trim();
            if s.len() < 19 || !s.is_ascii() {
                return None;
            }
            let b = s.as_bytes();
            if b[4] != b'-' || b[7] != b'-' || b[13] != b':' || b[16] != b':' {
                return None;
            }
            let year: i64 = s[0..4].parse().ok()?;
            let month: i64 = s[5..7].parse().ok()?;
            let day: i64 = s[8..10].parse().ok()?;
            let hour: i64 = s[11..13].parse().ok()?;
            let min: i64 = s[14..16].parse().ok()?;
            let sec: i64 = s[17..19].parse().ok()?;
            if !(0..24).contains(&hour) || !(0..60).contains(&min) || !(0..60).contains(&sec) {
                return None;
            }
            let millis = parse_fractional_millis(s, b);
            let tz_offset = parse_tz_offset_ms(s, b);
            let days = date::days_from_civil(year, month, day)?;
            Some(days * date::MS_PER_DAY + hour * 3_600_000 + min * 60_000 + sec * 1000 + millis - tz_offset)
        }
        Value::Number(n) => n.as_i64(),
        _ => None,
    }
}

pub(super) fn parse_jsonl_entries(
    path: &Path,
    mut process: impl FnMut(&Value) -> Option<Message>,
) -> Result<(Vec<Message>, usize)> {
    use std::io::{BufRead, BufReader};

    let file =
        std::fs::File::open(path).with_context(|| format!("Failed to open {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut messages = Vec::new();
    let mut skipped_lines = 0;

    for line_result in reader.lines() {
        let raw_line = match line_result {
            Ok(line) => line,
            Err(e) if e.kind() == std::io::ErrorKind::InvalidData => {
                skipped_lines += 1;
                continue;
            }
            Err(e) => {
                return Err(e)
                    .with_context(|| format!("I/O error reading {}", path.display()));
            }
        };
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        let entry: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => {
                skipped_lines += 1;
                continue;
            }
        };
        if let Some(msg) = process(&entry) {
            messages.push(msg);
        }
    }

    Ok((messages, skipped_lines))
}

pub(super) fn session_id_from_path(path: &Path) -> Option<String> {
    let stem = path.file_stem()?.to_str()?;
    if stem.is_empty() {
        return None;
    }
    Some(stem.to_string())
}

pub(super) fn update_earliest(earliest: &mut Option<i64>, ts: i64) {
    *earliest = Some(match *earliest {
        Some(prev) => prev.min(ts),
        None => ts,
    });
}

#[cfg(test)]
pub(super) fn write_test_jsonl(lines: &[&str]) -> tempfile::NamedTempFile {
    use std::io::Write;
    let mut tmp = tempfile::NamedTempFile::new().unwrap();
    for line in lines {
        writeln!(tmp, "{}", line).unwrap();
    }
    tmp.flush().unwrap();
    tmp
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_iso_timestamp_with_millis() {
        let val = Value::String("2026-03-01T12:30:45.123Z".to_string());
        let ts = parse_iso_timestamp(&val).unwrap();
        let expected_days = date::days_from_civil(2026, 3, 1).unwrap();
        let expected = expected_days * date::MS_PER_DAY + 12 * 3_600_000 + 30 * 60_000 + 45 * 1000 + 123;
        assert_eq!(ts, expected);
    }

    #[test]
    fn test_parse_iso_timestamp_numeric() {
        let val = Value::Number(serde_json::Number::from(1709251200000_i64));
        assert_eq!(parse_iso_timestamp(&val), Some(1709251200000));
    }

    #[test]
    fn test_parse_iso_timestamp_too_short() {
        let val = Value::String("2026".to_string());
        assert_eq!(parse_iso_timestamp(&val), None);
    }

    #[test]
    fn test_parse_iso_timestamp_null() {
        assert_eq!(parse_iso_timestamp(&Value::Null), None);
    }

    #[test]
    fn test_parse_iso_timestamp_no_millis() {
        let val = Value::String("2026-03-01T00:00:00Z".to_string());
        let ts = parse_iso_timestamp(&val).unwrap();
        let expected_days = date::days_from_civil(2026, 3, 1).unwrap();
        assert_eq!(ts, expected_days * date::MS_PER_DAY);
    }

    #[test]
    fn test_parse_iso_timestamp_non_ascii_rejected() {
        let val = Value::String("2026-03-01T00:00:00Ü".to_string());
        assert_eq!(parse_iso_timestamp(&val), None);
    }

    #[test]
    fn test_parse_iso_timestamp_invalid_time_rejected() {
        assert_eq!(parse_iso_timestamp(&Value::String("2026-03-01T24:00:00Z".to_string())), None);
        assert_eq!(parse_iso_timestamp(&Value::String("2026-03-01T12:60:00Z".to_string())), None);
        assert_eq!(parse_iso_timestamp(&Value::String("2026-03-01T12:00:60Z".to_string())), None);
    }

    #[test]
    fn test_parse_fractional_millis_1digit() {
        let val = Value::String("2026-03-01T12:30:45.5Z".to_string());
        let ts = parse_iso_timestamp(&val).unwrap();
        let expected_days = date::days_from_civil(2026, 3, 1).unwrap();
        let expected = expected_days * date::MS_PER_DAY + 12 * 3_600_000 + 30 * 60_000 + 45 * 1000 + 500;
        assert_eq!(ts, expected);
    }

    #[test]
    fn test_parse_iso_timestamp_applies_positive_offset() {
        // +09:00 means local time is 9h ahead of UTC → subtract 9h
        let utc = parse_iso_timestamp(&Value::String("2026-03-01T12:00:00Z".to_string())).unwrap();
        let with_offset = parse_iso_timestamp(&Value::String("2026-03-01T12:00:00+09:00".to_string())).unwrap();
        assert_eq!(utc - with_offset, 9 * 3_600_000);
    }

    #[test]
    fn test_parse_iso_timestamp_applies_negative_offset() {
        // -05:00 means local time is 5h behind UTC → add 5h
        let utc = parse_iso_timestamp(&Value::String("2026-03-01T12:00:00Z".to_string())).unwrap();
        let with_offset = parse_iso_timestamp(&Value::String("2026-03-01T12:00:00-05:00".to_string())).unwrap();
        assert_eq!(with_offset - utc, 5 * 3_600_000);
    }

    #[test]
    fn test_parse_iso_timestamp_offset_without_colon() {
        let utc = parse_iso_timestamp(&Value::String("2026-03-01T12:00:00Z".to_string())).unwrap();
        let with_offset = parse_iso_timestamp(&Value::String("2026-03-01T12:00:00+0900".to_string())).unwrap();
        assert_eq!(utc - with_offset, 9 * 3_600_000);
    }

    #[test]
    fn test_parse_iso_timestamp_offset_with_millis() {
        let utc = parse_iso_timestamp(&Value::String("2026-03-01T12:00:00.500Z".to_string())).unwrap();
        let with_offset = parse_iso_timestamp(&Value::String("2026-03-01T12:00:00.500+09:00".to_string())).unwrap();
        assert_eq!(utc - with_offset, 9 * 3_600_000);
    }

    #[test]
    fn test_parse_fractional_millis_2digit() {
        let val = Value::String("2026-03-01T12:30:45.12Z".to_string());
        let ts = parse_iso_timestamp(&val).unwrap();
        let expected_days = date::days_from_civil(2026, 3, 1).unwrap();
        let expected = expected_days * date::MS_PER_DAY + 12 * 3_600_000 + 30 * 60_000 + 45 * 1000 + 120;
        assert_eq!(ts, expected);
    }
}
