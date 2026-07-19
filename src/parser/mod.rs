mod claude;
mod codex;

pub use claude::parse_claude_session;
pub use codex::parse_codex_session;

use std::fmt;
use std::path::Path;

use anyhow::{Context, Result};
use serde_json::Value;

use crate::date;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, clap::ValueEnum)]
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

impl fmt::Display for Source {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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

    pub fn from_db(s: &str) -> Option<Role> {
        match s {
            "user" => Some(Role::User),
            "assistant" => Some(Role::Assistant),
            _ => None,
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
    /// Distinct filesystem paths this session edited, extracted from write-type
    /// tool_use blocks (Edit / Write / MultiEdit / NotebookEdit). Never routed
    /// into `messages` / `qa_chunks`; the indexer persists them to `session_files`.
    pub scanned_files: Vec<String>,
    pub skipped_lines: usize,
}

const TEXT_BLOCK_TYPES: &[&str] = &["text", "input_text", "output_text"];

/// Claude Code tools whose `input` names a file that the assistant wrote to.
/// Read / Grep / Bash carry no such target, so they are excluded (contract U-002).
const PATH_TOOL_NAMES: &[&str] = &["Edit", "Write", "MultiEdit", "NotebookEdit"];

/// Longest path value kept as a scanned file. Beyond this the value is not a real
/// filesystem path (PATH_MAX is 1024 on macOS, 4096 on Linux), so it is dropped by
/// the validity filter rather than stored.
const MAX_SCANNED_PATH_LEN: usize = 4096;

/// Extract the target path from one message content block when it is a write-type
/// tool_use (`Edit` / `Write` / `MultiEdit` use `file_path`, `NotebookEdit` uses
/// `notebook_path`). Returns `None` for any other block, so text / thinking /
/// tool_result blocks and non-write tools (Read / Grep / Bash) contribute nothing.
pub(super) fn extract_tool_use_path(block: &Value) -> Option<&str> {
    if block.get("type").and_then(|v| v.as_str()) != Some("tool_use") {
        return None;
    }
    let name = block.get("name").and_then(|v| v.as_str())?;
    if !PATH_TOOL_NAMES.contains(&name) {
        return None;
    }
    let input = block.get("input")?;
    input
        .get("file_path")
        .or_else(|| input.get("notebook_path"))
        .and_then(|v| v.as_str())
}

/// Whether a tool_use path value is a plausible filesystem path worth recording.
/// Three independent checks mirror the three invalid shapes the contract filters:
/// non-empty (path shape), no embedded newline, and within the length cap.
pub(super) fn is_valid_scanned_path(path: &str) -> bool {
    !path.is_empty() && !path.contains(['\n', '\r']) && path.len() <= MAX_SCANNED_PATH_LEN
}

/// Remove noise tags from text while preserving valuable tags like `<local-command-stdout>`.
///
/// Noise tags are Claude Code system-injected XML markers with no search value.
/// Review when upstream changes tag vocabulary (check Claude Code release notes).
/// See also: `codex.rs` `CODEX_SKIP_MARKERS` for message-level boilerplate filtering.
///
/// Unclosed tags are left intact (safe skip).
pub fn strip_noise_tags(mut result: String) -> String {
    const NOISE_TAGS: &[(&str, &str)] = &[
        ("<command-name>", "</command-name>"),
        ("<local-command-caveat>", "</local-command-caveat>"),
        ("<system-reminder>", "</system-reminder>"),
    ];

    for (open, close) in NOISE_TAGS {
        while let Some(start) = result.find(open) {
            let Some(end_offset) = result[start..].find(close) else {
                break; // unclosed tag — leave intact
            };
            let end = start + end_offset + close.len();
            result.replace_range(start..end, "");
        }
    }
    result
}

pub(super) fn extract_text(content: Option<&Value>) -> String {
    let Some(content) = content else {
        return String::new();
    };
    let raw = match content {
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
        _ => return String::new(),
    };
    strip_noise_tags(raw)
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
            Some(
                days * date::MS_PER_DAY + hour * 3_600_000 + min * 60_000 + sec * 1000 + millis
                    - tz_offset,
            )
        }
        Value::Number(n) => n.as_i64(),
        _ => None,
    }
}

pub(super) fn parse_jsonl_entries(
    path: &Path,
    mut process: impl FnMut(&Value) -> Option<Message>,
) -> Result<(Vec<Message>, usize)> {
    use std::fs::File;
    use std::io::{BufRead, BufReader, ErrorKind};

    let file = File::open(path).with_context(|| format!("Failed to open {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut messages = Vec::new();
    let mut skipped_lines = 0;

    for line_result in reader.lines() {
        let raw_line = match line_result {
            Ok(line) => line,
            Err(e) if e.kind() == ErrorKind::InvalidData => {
                skipped_lines += 1;
                continue;
            }
            Err(e) => {
                return Err(e).with_context(|| format!("I/O error reading {}", path.display()));
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
    Some(stem.to_owned())
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
mod tests;
