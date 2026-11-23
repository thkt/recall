import json
import os
from pathlib import Path
import subprocess
import sys

SCRIPT = Path(__file__).with_name("measure-search.py")

STUB = """#!/usr/bin/env python3
import json, os, sys, time
args = sys.argv[1:]
if args[0] == "status":
    print(json.dumps({"data": {"sessions": 3, "qa_chunks": 5, "embedded": 5, "model_ready": True}}))
    sys.exit(0)
query = args[1]
if query == "slow":
    time.sleep(0.4)
if query == "bad":
    sys.stderr.write("error: boom\\n")
    sys.exit(65)
model_dir = os.environ.get("RECALL_MODEL_DIR")
degraded = bool(model_dir) and not os.listdir(model_dir)
print(json.dumps({"data": {"results": [{"session_id": "private-session"}]}, "degraded": degraded}))
"""


def measure(tmp_path, queries, *options):
    binary = tmp_path / "recall"
    binary.write_text(STUB)
    binary.chmod(0o755)
    queries_file = tmp_path / "queries.txt"
    queries_file.write_text(queries)
    env = {key: value for key, value in os.environ.items() if key != "RECALL_MODEL_DIR"}
    done = subprocess.run(
        [sys.executable, str(SCRIPT), str(binary), str(queries_file), *options],
        capture_output=True,
        text=True,
        env=env,
    )
    return done


def test_searches_under_the_limit_pass_and_report_the_index_size(tmp_path):
    done = measure(tmp_path, "# a comment\n\nfirst\nsecond\n", "--runs", "2")

    report = json.loads(done.stdout)
    assert (done.returncode, report["within_limit"], report["queries"]) == (0, True, 2)
    assert report["index"] == {"sessions": 3, "qa_chunks": 5, "embedded": 5, "model_ready": True}
    assert {name: mode["runs"] for name, mode in report["modes"].items()} == {
        "hybrid": 4,
        "keyword_only": 4,
    }


def test_a_search_slower_than_the_limit_fails_the_report(tmp_path):
    done = measure(tmp_path, "first\nslow\n", "--runs", "2", "--limit-seconds", "0.3")

    report = json.loads(done.stdout)
    assert (done.returncode, report["within_limit"]) == (1, False)
    assert {name: mode["over_limit"] for name, mode in report["modes"].items()} == {
        "hybrid": 2,
        "keyword_only": 2,
    }


def test_a_failed_search_fails_the_report_even_when_it_is_fast(tmp_path):
    done = measure(tmp_path, "first\nbad\n", "--runs", "1")

    report = json.loads(done.stdout)
    assert (done.returncode, report["within_limit"]) == (1, False)
    assert {name: mode["failed"] for name, mode in report["modes"].items()} == {
        "hybrid": 1,
        "keyword_only": 1,
    }


def test_only_the_keyword_mode_runs_without_a_model(tmp_path):
    done = measure(tmp_path, "first\n", "--runs", "2")

    report = json.loads(done.stdout)
    assert {name: mode["degraded"] for name, mode in report["modes"].items()} == {
        "hybrid": 0,
        "keyword_only": 2,
    }


def test_the_report_carries_no_query_and_no_result(tmp_path):
    done = measure(tmp_path, "a private question\n", "--runs", "1")

    assert done.returncode == 0
    assert "a private question" not in done.stdout + done.stderr
    assert "private-session" not in done.stdout + done.stderr


def test_a_queries_file_without_a_query_is_refused(tmp_path):
    done = measure(tmp_path, "# only a comment\n\n")

    assert (done.returncode, done.stdout) == (1, "")
    assert "holds no query" in done.stderr
