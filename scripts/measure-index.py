#!/usr/bin/env python3
"""Host-only synthetic index measurements; never reads private session roots.

Usage: python3 scripts/measure-index.py /path/to/recall /path/to/new-results-dir
See docs/index-observability.md for comparison conditions and interpretation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time


def record(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sessions", type=int, default=512)
    parser.add_argument("--model", required=True, help="model ID, revision and artifact identity")
    parser.add_argument("--conditions", required=True, help="hardware/RAM, OS, toolchain, cache state, settings, commit")
    parser.add_argument("--baseline", action="store_true", help="older binary without commit notifications: omit interrupt/resume")
    args = parser.parse_args()
    if args.sessions < 256:
        parser.error("use at least 256 sessions so interruption leaves multiple batches")
    binary = args.binary.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = output / "synthetic-claude"
    source.mkdir()
    env = dict(os.environ)
    for key in ("RECALL_TOKEN_BUDGET", "RECALL_FORWARD_PAUSE_MS", "CLAUDE_CODE_SESSION_ID"):
        env.pop(key, None)
    env.update(RECALL_CLAUDE_DIR=str(source), RECALL_CODEX_DIR=str(output / "absent-codex"))

    def turn(number):
        return "".join(json.dumps({"type": role, "message": {"role": role, "content": f"Synthetic {role} turn {number}: indexing measurement only."}}) + "\n" for role in ("user", "assistant"))

    for i in range(args.sessions):
        (source / f"session-{i:06}.jsonl").write_text(turn(i))
    record(output / "conditions.json", {
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "platform": platform.platform(), "machine": platform.machine(),
        "model": args.model, "conditions": args.conditions, "sessions": args.sessions,
        "data": "synthetic, one user/assistant pair per file; one pair appended to one file",
        "cache": "each invocation is a new process; filesystem/model-cache state is operator-supplied",
        "options": "default token budget and forward pause; inherited overrides removed",
    })

    def run(name, db, interrupt=False):
        env["RECALL_DB"] = str(db)
        start = time.monotonic()
        interrupted = False
        last_counts = None
        with (output / f"{name}.stdout").open("w") as stdout, (output / f"{name}.stderr").open("w") as stderr:
            process = subprocess.Popen([str(binary), "index", "--json"], env=env, stdout=stdout, stderr=subprocess.PIPE, text=True)
            try:
                for line in process.stderr:
                    stderr.write(line)
                    if line.startswith("index: embedding counts: "):
                        last_counts = json.loads(line.split(": ", 2)[2])
                        if interrupt and not interrupted and last_counts["chunks_saved"] > 0 and last_counts["chunks_remaining_snapshot"] > 0:
                            process.terminate()
                            interrupted = True
                code = process.wait()
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait()
                process.stderr.close()
        summary = {"case": name, "wall_seconds": time.monotonic() - start, "exit_code": code,
                   "interruption_requested": interrupted, "last_reported_counts": last_counts}
        if code == 0:
            envelope = json.loads((output / f"{name}.stdout").read_text())
            summary["data"] = envelope["data"]
            summary["degraded"] = envelope["degraded"]
            if envelope["degraded"]:
                record(output / f"{name}.json", summary)
                raise RuntimeError("model or indexing degraded; this is not a valid real-model timing sample")
        record(output / f"{name}.json", summary)
        if interrupt:
            if not interrupted or code == 0:
                raise RuntimeError("interruption did not stop a partial run; increase session count and repeat in a new output directory")
        elif code != 0:
            raise RuntimeError(f"{name} failed; inspect retained synthetic stderr")
        return summary

    db = output / "index.db"
    results = [run("initial", db), run("unchanged", db)]
    with (source / "session-000000.jsonl").open("a") as file:
        file.write(turn(args.sessions))
    results.append(run("append-one", db))
    if not args.baseline:
        # Fresh DB plus the same corpus; stop after a confirmed partial save.
        resume_db = output / "resume.db"
        results.append(run("interrupted", resume_db, interrupt=True))
        results.append(run("resume", resume_db))
        interrupted_counts = results[-2]["last_reported_counts"]
        resumed_data = results[-1]["data"]
        if not (0 < resumed_data["embedded"] <= interrupted_counts["chunks_remaining_snapshot"]):
            raise RuntimeError("resume did not demonstrate a preserved partial save; repeat with a larger corpus")
    record(output / "comparison.json", results)


if __name__ == "__main__":
    main()
