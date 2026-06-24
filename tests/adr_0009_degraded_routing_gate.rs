//! amici ADR-0009 mechanical gate, ported to recall: deny bare cause-erasing
//! `map_err` into a degraded reason. A typed error must route through
//! `degrade_with_warn` / `record_degraded` so the cause stays in the structured
//! warn event. recall already conforms (achieved in #230); this source scan is
//! the continuous check the ADR's Confirmation specifies, kept hermetic (no
//! `git` dependency) so it runs in the normal test suite. It stands in for the
//! aspirational dylint custom lint: adding dylint to enforce ADR-0009 would
//! itself violate ADR-0011 (dependency minimization).

use std::fs;
use std::path::{Path, PathBuf};

fn collect_rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            collect_rs_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// Return the 1-based line numbers of every cause-discarding `map_err(|_…|)`
/// whose closure body reaches a degraded reason.
///
/// Matches the `map_err(|_` prefix so an underscore-named-but-unused binding
/// (`map_err(|_err| DegradedReason::X)`) is caught alongside the plain
/// `map_err(|_|` form — both discard the cause. A binding that actually threads
/// the cause (`map_err(|e| …e…)`) has no leading underscore and is not flagged.
///
/// `//` comments are stripped first so the anti-pattern quoted verbatim in
/// main.rs docs cannot trip the scan; the surviving code is rejoined into one
/// string and scanned as a whole, so a multi-line `map_err(|_| {
/// DegradedReason::X })` closure — the form a real regression takes — is caught
/// where a line-by-line scan would miss it. A 160-char window spans the closure
/// body; the sanctioned routes (`degrade_with_warn` / `record_degraded`) never
/// pair a `map_err` closure with a `Degraded` reason, so no false positive arises.
///
/// Known limit: `//` inside a string literal is treated as a comment start, so a
/// line mixing such a literal with a bare `map_err` could slip past. No source
/// line does this today; avoid the combination rather than parse Rust here.
fn scan_bare_map_err_into_degraded(content: &str) -> Vec<usize> {
    const NEEDLE: &str = "map_err(|_";
    let code = content
        .lines()
        .map(|line| match line.find("//") {
            Some(i) => &line[..i],
            None => line,
        })
        .collect::<Vec<_>>()
        .join("\n");

    let mut hits = Vec::new();
    for (start, _) in code.match_indices(NEEDLE) {
        let window: String = code[start..].chars().take(160).collect();
        if window.contains("Degraded") {
            let line_no = code[..start].bytes().filter(|&b| b == b'\n').count() + 1;
            hits.push(line_no);
        }
    }
    hits
}

#[test]
fn no_bare_cause_erasing_map_err_into_degraded() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    collect_rs_files(&src, &mut files);
    assert!(!files.is_empty(), "no source files scanned under {src:?}");

    let mut offenders = Vec::new();
    for file in &files {
        let content = fs::read_to_string(file).unwrap();
        for line_no in scan_bare_map_err_into_degraded(&content) {
            offenders.push(format!("{}:{}", file.display(), line_no));
        }
    }

    assert!(
        offenders.is_empty(),
        "bare cause-erasing map_err into a degraded reason (ADR-0009 forbids \
         this; route through degrade_with_warn / record_degraded):\n{}",
        offenders.join("\n")
    );
}

#[test]
fn scan_flags_single_and_multi_line_closures_but_not_comments_or_other_errors() {
    // Positive: single-line and multi-line closures both reach a degraded reason.
    // recall's DegradedReason variants are unit (no payload), unlike amici's.
    let single = "let e = x.map_err(|_| DegradedReason::ProbeFailed)?;";
    assert_eq!(scan_bare_map_err_into_degraded(single), vec![1]);

    let multi = "fn f() {\n    x.map_err(|_| {\n        DegradedReason::Disabled\n    })\n}";
    assert_eq!(
        scan_bare_map_err_into_degraded(multi),
        vec![2],
        "multi-line closure into a degraded reason must be flagged"
    );

    // Positive: an underscore-named-but-unused binding still discards the cause.
    let named = "x.map_err(|_err| DegradedReason::ProbeFailed)?;";
    assert_eq!(
        scan_bare_map_err_into_degraded(named),
        vec![1],
        "named-but-ignored binding into a degraded reason must be flagged"
    );

    // Negative: a doc line quoting the pattern is stripped at `//`.
    let doc = "/// never `.map_err(|_| DegradedReason::X)`, which erases the cause";
    assert!(scan_bare_map_err_into_degraded(doc).is_empty());

    // Negative: a bare closure into a non-degraded error is allowed (mirrors
    // recall's real RecallError sites, which carry no degraded reason).
    let other = "x.map_err(|_| RecallError::DataError(\"corrupt index\".into()))?;";
    assert!(scan_bare_map_err_into_degraded(other).is_empty());
}
