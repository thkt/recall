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
watched = os.environ.get("STUB_WATCHED_FILE")
if watched:
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("measure_search", os.environ["STUB_SCRIPT"])
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    with open(os.environ["STUB_LOG"], "a") as log:
        log.write(f"{script.fraction_in_memory(Path(watched))}\\n")
    Path(watched).read_bytes()
model_dir = os.environ.get("RECALL_MODEL_DIR")
degraded = bool(model_dir) and not os.listdir(model_dir)
print(json.dumps({"data": {"results": [{"session_id": "private-session"}]}, "degraded": degraded}))
"""


def measure(tmp_path, queries, *options, watched_file=None):
    binary = tmp_path / "recall"
    binary.write_text(STUB)
    binary.chmod(0o755)
    queries_file = tmp_path / "queries.txt"
    queries_file.write_text(queries)
    env = {key: value for key, value in os.environ.items() if key != "RECALL_MODEL_DIR"}
    if watched_file:
        env.update(
            STUB_WATCHED_FILE=str(watched_file),
            STUB_SCRIPT=str(SCRIPT),
            STUB_LOG=str(tmp_path / "in-memory.log"),
        )
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


def test_every_search_after_a_drop_starts_with_the_file_out_of_memory(tmp_path):
    index = tmp_path / "index.db"
    index.write_bytes(os.urandom(1 << 20))

    done = measure(
        tmp_path, "first\nsecond\n", "--runs", "2", "--drop-from-memory", str(index),
        watched_file=index,
    )

    report = json.loads(done.stdout)
    assert {name: mode["runs"] for name, mode in report["modes"].items()} == {
        "hybrid": 4,
        "keyword_only": 4,
        "hybrid_after_drop": 4,
        "keyword_only_after_drop": 4,
    }
    assert report["modes"]["hybrid_after_drop"]["left_in_memory_fraction"] == 0
    in_memory = [float(line) for line in (tmp_path / "in-memory.log").read_text().split()]
    kept, hybrid_dropped, keyword_dropped = in_memory[:10], in_memory[11:15], in_memory[16:]
    assert (set(kept), set(hybrid_dropped), set(keyword_dropped)) == ({1.0}, {0.0}, {0.0})


def test_a_search_slower_than_the_limit_after_a_drop_fails_the_report(tmp_path):
    index = tmp_path / "index.db"
    index.write_bytes(b"index")

    done = measure(
        tmp_path, "slow\n", "--runs", "1", "--limit-seconds", "0.3", "--drop-from-memory", str(index)
    )

    report = json.loads(done.stdout)
    assert (done.returncode, report["within_limit"]) == (1, False)
    assert report["modes"]["hybrid_after_drop"]["over_limit"] == 1


def test_a_missing_file_to_drop_is_refused_before_any_search(tmp_path):
    done = measure(tmp_path, "first\n", "--drop-from-memory", str(tmp_path / "absent.db"))

    assert (done.returncode, done.stdout) == (1, "")
    assert "names no file: absent.db" in done.stderr


def test_an_empty_file_to_drop_is_refused(tmp_path):
    empty = tmp_path / "empty.db"
    empty.touch()

    done = measure(tmp_path, "first\n", "--drop-from-memory", str(empty))

    assert done.returncode == 1
    assert "cannot map empty.db" in done.stderr
