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
    let single = "let e = x.map_err(|_| DegradedReason::ProbeFailed)?;";
    assert_eq!(scan_bare_map_err_into_degraded(single), vec![1]);

    let multi = "fn f() {\n    x.map_err(|_| {\n        DegradedReason::Disabled\n    })\n}";
    assert_eq!(
        scan_bare_map_err_into_degraded(multi),
        vec![2],
        "multi-line closure into a degraded reason must be flagged"
    );

    let named = "x.map_err(|_err| DegradedReason::ProbeFailed)?;";
    assert_eq!(
        scan_bare_map_err_into_degraded(named),
        vec![1],
        "named-but-ignored binding into a degraded reason must be flagged"
    );

    let doc = "/// never `.map_err(|_| DegradedReason::X)`, which erases the cause";
    assert!(scan_bare_map_err_into_degraded(doc).is_empty());

    let other = "x.map_err(|_| RecallError::DataError(\"corrupt index\".into()))?;";
    assert!(scan_bare_map_err_into_degraded(other).is_empty());
}
