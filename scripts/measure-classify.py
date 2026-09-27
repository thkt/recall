#!/usr/bin/env python3
"""Measure Issue #342 with the pinned pre-change reader and transaction code.

Build a temporary copy, never modify the working sources or verification config.
The normal product build/MLX prerequisites must already be available on the host.
"""
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile

BASELINE = "5a9fab0c9e69aee084ba4e273a9f0c80cb65ff9a"


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f"measurement source changed; expected one {old!r}")
    return text.replace(old, new, 1)


def install_baseline(root, dest):
    """Inject the exact handoff parser and reclassification function for control."""
    paths = git(root, "ls-tree", "-r", "--name-only", BASELINE, "src/parser").decode().splitlines()
    for path in paths:
        target = dest / "src/classify/baseline_parser" / Path(path).relative_to("src/parser")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(git(root, "show", f"{BASELINE}:{path}"))

    main = git(root, "show", f"{BASELINE}:src/main.rs").decode()
    begin = main.index("fn reclassify_sessions_with(")
    end = main.index("/// `recall classify`:", begin)
    control = replace_once(main[begin:end], "fn reclassify_sessions_with(", "pub(super) fn reclassify_sessions_with(")
    control = replace_once(control, "    mut read_origin:", "    metrics: &RefCell<Metrics>,\n    mut read_origin:")
    control = replace_once(control, "    let tx =", "    let wait = Instant::now();\n    let tx =")
    control = replace_once(control, "    let outcomes = {", "    metrics.borrow_mut().wait += wait.elapsed();\n    let hold = Instant::now();\n    let outcomes = {")
    control = replace_once(control, "    tx.commit()?;", "    tx.commit()?;\n    metrics.borrow_mut().hold += hold.elapsed();")
    prefix = """use std::{cell::RefCell, time::Instant};
use anyhow::Result;
use rusqlite::Connection;
use crate::{ClassifyOutcome, classify, error::RecallError};
use super::Metrics;
"""
    (dest / "src/classify/baseline.rs").write_text(prefix + control)
    bench = dest / "src/classify/benchmark.rs"
    code = bench.read_text()
    code = replace_once(code, "use crate::parser::{self, Source};", "use baseline_parser::{self as parser, Source};")
    start = code.index("// Baseline is installed")
    end = code.index("struct Corpus", start)
    code = code[:start] + """#[path = "baseline_parser/mod.rs"]
mod baseline_parser;
#[path = "baseline.rs"]
mod baseline;

fn legacy(conn: &mut Connection, mut read: impl FnMut(&str, &str, &str) -> Result<bool, &'static str>, metrics: &RefCell<Metrics>) {
    baseline::reclassify_sessions_with(conn, true, false, metrics, |path, source, id| {
        let stat = || { let m = fs::metadata(path).unwrap(); (m.len(), m.modified().unwrap()) };
        let before = stat();
        let origin = read(path, source, id);
        assert_eq!(before, stat());
        origin
    }).unwrap();
}

""" + code[end:]
    bench.write_text(code)
    return {"start_commit": BASELINE, "baseline_main_sha256": hashlib.sha256(main.encode()).hexdigest()}


def main():
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="recall-classify-measure-") as directory:
        dest = Path(directory)
        for name in ["src", "tests", ".config"]:
            shutil.copytree(root / name, dest / name)
        for name in ["Cargo.toml", "Cargo.lock", "README.md", "README.ja.md"]:
            shutil.copy(root / name, dest / name)
        current_sources = ["src/main.rs", "src/classify/benchmark.rs", "src/parser/mod.rs",
                           "src/parser/claude.rs", "src/parser/codex.rs", "src/parser/provenance.rs"]
        current_hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                          for name in current_sources}
        provenance = install_baseline(root, dest)
        command = ["cargo", "test", "--release", "--locked", "--bin", "recall",
                   "classify::benchmark::synthetic_classification_measurement", "--", "--ignored", "--nocapture"]
        print(json.dumps({"environment": {"platform": platform.platform(),
              "rustc": subprocess.check_output(["rustc", "--version"]).decode().strip(),
              "os_cache": "warm, no cache eviction", "source": "deterministic synthetic enumeration only",
              "baseline": provenance, "current_source_sha256": current_hashes}, "command": command}), flush=True)
        env = os.environ.copy()
        env["CARGO_TARGET_DIR"] = str(root / "target/classify-measurement")
        summaries = []
        with subprocess.Popen(command, cwd=dest, env=env, stdout=subprocess.PIPE, text=True) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                if line.startswith("CLASSIFY_MEASUREMENT "):
                    summaries.append(json.loads(line.split(" ", 1)[1]))
            if process.wait() != 0:
                raise subprocess.CalledProcessError(process.returncode, command)
        expected = {(dataset, mode, repeat, concurrent)
                    for dataset in ["few_short", "few_long", "many_short"]
                    for mode in ["Legacy", "SplitFull", "SplitOrigin"]
                    for repeat in [False, True] for concurrent in [False, True]}
        actual = {(row["dataset"], row["mode"], row["repeat"], row["concurrent"])
                  for row in summaries}
        if actual != expected or len(summaries) != len(expected) or any(row["runs"] < 5 for row in summaries):
            raise RuntimeError("incomplete measurement: require all 36 comparisons with at least 5 runs each")


if __name__ == "__main__":
    main()
