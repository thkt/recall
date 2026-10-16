mod claude;
mod codex;
mod provenance;

pub use claude::parse_claude_session;
pub use codex::parse_codex_session;

use std::fmt;
use std::fs::File;
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
    pub slug: String,
    pub timestamp: Option<i64>,
}

pub struct Message {
    pub role: Role,
    pub text: String,
}

pub struct ParseResult {
    pub metadata: SessionData,
    pub messages: Vec<Message>,
    pub scanned_files: Vec<String>,
    pub diagnostics: LineDiagnostics,
    pub origin: provenance::SessionOrigin,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct LineDiagnostics {
    pub invalid_json_lines: i64,
    pub invalid_utf8_lines: i64,
    pub incomplete_tail_lines: i64,
}

impl LineDiagnostics {
    pub fn is_empty(self) -> bool {
        self == Self::default()
    }
}

pub(crate) fn parse_session_including_empty(
    path: &Path,
    source: Source,
) -> Result<Option<ParseResult>> {
    match source {
        Source::Claude => parse_claude_session(path),
        Source::Codex => parse_codex_session(path),
    }
}

pub(crate) struct ClassificationRead {
    pub session_id: String,
    pub origin: provenance::SessionOrigin,
    pub diagnostics: LineDiagnostics,
}

pub(crate) fn parse_classification(
    file: File,
    path: &Path,
    source: Source,
) -> Result<Option<ClassificationRead>> {
    use provenance::{ClaudeOrigin, CodexOrigin};
    let Some(session_id) = session_id_from_path(path) else {
        return Ok(None);
    };
    let (session_id, origin, diagnostics) = match source {
        Source::Claude => {
            let mut origin = ClaudeOrigin::new(path, &session_id);
            let diagnostics = scan_jsonl_reader(file, |entry| origin.observe(entry))?;
            let origin = origin.finish();
            (session_id, origin, diagnostics)
        }
        Source::Codex => {
            let mut origin = CodexOrigin::new(session_id);
            let diagnostics = scan_jsonl_reader(file, |entry| origin.observe(entry))?;
            let (session_id, origin) = origin.finish();
            (session_id, origin, diagnostics)
        }
    };
    Ok(Some(ClassificationRead {
        session_id,
        origin,
        diagnostics,
    }))
}

const TEXT_BLOCK_TYPES: &[&str] = &["text", "input_text", "output_text"];

const PATH_TOOL_NAMES: &[&str] = &["Edit", "Write", "MultiEdit", "NotebookEdit"];

const MAX_SCANNED_PATH_LEN: usize = 4096;

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

pub(super) fn is_valid_scanned_path(path: &str) -> bool {
    !path.is_empty() && !path.contains(['\n', '\r']) && path.len() <= MAX_SCANNED_PATH_LEN
}

pub fn strip_noise_tags(mut result: String) -> String {
    const NOISE_TAGS: &[(&str, &str)] = &[
        ("<command-name>", "</command-name>"),
        ("<local-command-caveat>", "</local-command-caveat>"),
        ("<system-reminder>", "</system-reminder>"),
    ];

    for (open, close) in NOISE_TAGS {
        while let Some(start) = result.find(open) {
            let Some(end_offset) = result[start..].find(close) else {
                break;
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

const ISO_SECONDS_END: usize = 19;
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
) -> Result<(Vec<Message>, LineDiagnostics)> {
    let mut messages = Vec::new();
    let diagnostics = scan_jsonl_entries(path, |entry| {
        if let Some(message) = process(entry) {
            messages.push(message);
        }
    })?;
    Ok((messages, diagnostics))
}

fn scan_jsonl_entries(path: &Path, process: impl FnMut(&Value)) -> Result<LineDiagnostics> {
    let file = File::open(path).context("Failed to open session file")?;
    scan_jsonl_reader(file, process)
}

fn scan_jsonl_reader(file: File, mut process: impl FnMut(&Value)) -> Result<LineDiagnostics> {
    use std::io::{BufRead, BufReader};
    use std::str;

    let mut reader = BufReader::new(file);
    let mut diagnostics = LineDiagnostics::default();
    let mut bytes = Vec::new();

    loop {
        bytes.clear();
        if reader
            .read_until(b'\n', &mut bytes)
            .context("Failed to read session file")?
            == 0
        {
            break;
        }
        let terminated = bytes.last() == Some(&b'\n');
        let raw_line = match str::from_utf8(&bytes) {
            Ok(line) => line,
            Err(e) => {
                if !terminated && e.error_len().is_none() {
                    diagnostics.incomplete_tail_lines += 1;
                } else {
                    diagnostics.invalid_utf8_lines += 1;
                }
                continue;
            }
        };
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        let entry: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(e) => {
                if !terminated && e.is_eof() {
                    diagnostics.incomplete_tail_lines += 1;
                } else {
                    diagnostics.invalid_json_lines += 1;
                }
                continue;
            }
        };
        process(&entry);
    }

    Ok(diagnostics)
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

#[cfg(test)]
fn assert_classification_matches(path: &Path, source: Source, full: &ParseResult) {
    let classification = parse_classification(File::open(path).unwrap(), path, source)
        .unwrap()
        .unwrap();
    assert_eq!(classification.session_id, full.metadata.session_id);
    assert_eq!(classification.origin, full.origin);
    assert_eq!(classification.diagnostics, full.diagnostics);
}
