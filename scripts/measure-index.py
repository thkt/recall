#!/usr/bin/env python3
"""Host-only synthetic index measurements; never reads private session roots.

Usage: python3 scripts/measure-index.py /path/to/recall /path/to/new-results-dir
See docs/index-observability.md for comparison conditions and interpretation.
"""
import argparse
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sqlite3
import time


def record(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sessions", type=int, default=512)
    parser.add_argument("--long-session-turns", type=int, default=1,
                        help="Q&A pairs in the first session; use 512 for the #325 append comparison")
    parser.add_argument("--verify-reuse", action="store_true",
                        help="assert #325 inference/reuse counts; omit for the old binary")
    parser.add_argument("--model", required=True, help="model ID, revision and artifact identity")
    parser.add_argument("--conditions", required=True, help="hardware/RAM, OS, toolchain, cache state, settings, commit")
    parser.add_argument("--baseline", action="store_true", help="older binary without commit notifications: omit interrupt/resume")
    args = parser.parse_args()
    if args.long_session_turns < 1:
        parser.error("long-session-turns must be positive")
    if args.verify_reuse and args.baseline:
        parser.error("verify-reuse applies to the changed binary, not the baseline")
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
        text = turn(i)
        if i == 0:
            text += "".join(turn(args.sessions + j) for j in range(args.long_session_turns - 1))
        (source / f"session-{i:06}.jsonl").write_text(text)
    record(output / "conditions.json", {
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "platform": platform.platform(), "machine": platform.machine(),
        "model": args.model, "conditions": args.conditions, "sessions": args.sessions,
        "data": "synthetic Q&A pairs; one pair appended to the first file",
        "long_session_turns": args.long_session_turns,
        "append_pairs": 1,
        "cache": "each invocation is a new process; filesystem/model-cache state is operator-supplied",
        "options": "default token budget and forward pause; inherited overrides removed",
    })

    def stored_data(db):
        # Native tables + built-in FTS5 only: no Python sqlite-vec dependency.
        # Normalize rowids to message positions so rebuild can allocate new IDs.
        with closing(sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
            messages = list(conn.execute("SELECT rowid, session_id, role, text FROM messages ORDER BY session_id, rowid"))
            positions = {}
            by_session = {}
            for rowid, session_id, role, text in messages:
                body = by_session.setdefault(session_id, [])
                positions[(session_id, rowid)] = len(body)
                body.append([role, text])
            chunks = sorted([session_id, content, timestamp,
                             positions[(session_id, lo)], positions[(session_id, hi)]]
                            for session_id, content, timestamp, lo, hi in conn.execute(
                                "SELECT session_id, content, timestamp, src_rowid_lo, src_rowid_hi FROM qa_chunks"))
            bodies = {"messages": by_session, "chunks": chunks}
            page_count = conn.execute("PRAGMA page_count").fetchone()[0]
            free_pages = conn.execute("PRAGMA freelist_count").fetchone()[0]
            page_size = conn.execute("PRAGMA page_size").fetchone()[0]
        return {"body_sha256": hashlib.sha256(json.dumps(bodies, sort_keys=True).encode()).hexdigest(),
                "sessions": len(by_session), "messages": len(messages), "chunks": len(chunks),
                "database_allocated_bytes": page_count * page_size,
                "database_used_page_bytes": (page_count - free_pages) * page_size,
                "vector_values": "not inspected by this Python measurement; regression tests check exact retained blobs"}

    def run(name, db, interrupt=False, rebuild=False):
        env["RECALL_DB"] = str(db)
        start = time.monotonic()
        interrupted = False
        last_counts = None
        with (output / f"{name}.stdout").open("w") as stdout, (output / f"{name}.stderr").open("w") as stderr:
            process = subprocess.Popen([str(binary), "rebuild" if rebuild else "index", "--json"], env=env, stdout=stdout, stderr=subprocess.PIPE, text=True)
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
            summary["stored"] = stored_data(db)
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
    first = source / "session-000000.jsonl"
    if args.long_session_turns > 1:
        stamp = first.stat()
        os.utime(first, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 2_000_000_000))
        results.append(run("mtime-only", db))
        if results[-1]["stored"]["body_sha256"] != results[0]["stored"]["body_sha256"]:
            raise RuntimeError("mtime-only run changed stored bodies or source links")
    with first.open("a") as file:
        file.write(turn(args.sessions + args.long_session_turns - 1))
    appended = run("append-one", db)
    results.append(appended)
    if args.verify_reuse:
        counts = appended["data"]["observations"]["counts"]
        if (counts["inference_chunks"], counts["embeddings_reused_committed"]) != (1, args.long_session_turns):
            raise RuntimeError("append inferred more than the new Q&A pair or failed to reuse the prefix")
        for result in results:
            if result["case"] in ("unchanged", "mtime-only"):
                counts = result["data"]["observations"]["counts"]
                if counts["inference_chunks"] != 0:
                    raise RuntimeError("unchanged content was inferred again")
    if args.long_session_turns > 1:
        rebuilt = run("rebuild-final", db, rebuild=True)
        results.append(rebuilt)
        if rebuilt["stored"]["body_sha256"] != appended["stored"]["body_sha256"]:
            raise RuntimeError("incremental stored bodies or source links differ from full rebuild")
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
