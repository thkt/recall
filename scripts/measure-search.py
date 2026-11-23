#!/usr/bin/env python3
"""Time `recall search` against an existing index and compare it with the 2-second indicator.

Usage: python3 scripts/measure-search.py /path/to/recall /path/to/queries.txt
The queries file holds one query per line; blank lines and lines starting with `#` are skipped.
The report carries timings and counts only. It never prints a query or a search result.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time


def read_queries(path):
    lines = (line.strip() for line in Path(path).read_text().splitlines())
    return [line for line in lines if line and not line.startswith("#")]


def percentile(sorted_values, fraction):
    index = max(0, min(len(sorted_values) - 1, round(fraction * len(sorted_values)) - 1))
    return sorted_values[index]


def summarize(seconds, failed, degraded, limit_seconds):
    ordered = sorted(seconds)
    return {
        "runs": len(ordered),
        "p50_seconds": round(statistics.median(ordered), 3),
        "p95_seconds": round(percentile(ordered, 0.95), 3),
        "max_seconds": round(ordered[-1], 3),
        "over_limit": sum(1 for value in ordered if value > limit_seconds),
        "failed": failed,
        "degraded": degraded,
    }


def search_once(binary, query, env):
    started = time.perf_counter()
    done = subprocess.run(
        [str(binary), "search", query, "--json"], capture_output=True, text=True, env=env
    )
    elapsed = time.perf_counter() - started
    degraded = False
    if done.returncode == 0:
        degraded = json.loads(done.stdout).get("degraded") is True
    return elapsed, done.returncode != 0, degraded


def measure(binary, queries, runs, env, limit_seconds):
    search_once(binary, queries[0], env)
    seconds, failed, degraded = [], 0, 0
    for query in queries:
        for _ in range(runs):
            elapsed, did_fail, was_degraded = search_once(binary, query, env)
            seconds.append(elapsed)
            failed += did_fail
            degraded += was_degraded
    return summarize(seconds, failed, degraded, limit_seconds)


def index_size(binary, env):
    done = subprocess.run(
        [str(binary), "status", "--json"], capture_output=True, text=True, env=env
    )
    if done.returncode != 0:
        sys.exit(f"`recall status` failed: {done.stderr.strip()}")
    data = json.loads(done.stdout)["data"]
    return {key: data[key] for key in ("sessions", "qa_chunks", "embedded", "model_ready")}


def passes(report):
    return all(
        mode["p95_seconds"] <= report["limit_seconds"] and mode["failed"] == 0
        for mode in report["modes"].values()
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("queries", type=Path)
    parser.add_argument("--db-path", type=Path, help="index to search; default: the binary's default index")
    parser.add_argument("--runs", type=int, default=3, help="searches per query and mode")
    parser.add_argument("--limit-seconds", type=float, default=2.0)
    args = parser.parse_args()

    queries = read_queries(args.queries)
    if not queries:
        sys.exit("the queries file holds no query")
    base_env = dict(os.environ)
    if args.db_path:
        base_env["RECALL_DB"] = str(args.db_path)

    with tempfile.TemporaryDirectory() as empty_model_dir:
        keyword_env = dict(base_env, RECALL_MODEL_DIR=empty_model_dir)
        report = {
            "index": index_size(args.binary, base_env),
            "queries": len(queries),
            "limit_seconds": args.limit_seconds,
            "modes": {
                "hybrid": measure(args.binary, queries, args.runs, base_env, args.limit_seconds),
                "keyword_only": measure(
                    args.binary, queries, args.runs, keyword_env, args.limit_seconds
                ),
            },
        }
    report["within_limit"] = passes(report)
    print(json.dumps(report, indent=2))
    return 0 if report["within_limit"] else 1


if __name__ == "__main__":
    sys.exit(main())
