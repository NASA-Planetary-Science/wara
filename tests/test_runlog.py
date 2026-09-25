"""Tests for wara.runlog: entry formatting, round-trip parsing and the
50-entries-per-file rollover."""
from datetime import datetime

import pytest

from wara import runlog


def test_log_run_writes_parseable_entry(tmp_path):
    path = runlog.log_run(
        "Cf-252 source\n10 cm, no shielding", source="Neutrons",
        metadata={"File": "cf252.npz", "Note": "two\nlines"},
        stats={"Traces": "12,000", "FOM": "1.31"},
        log_dir=tmp_path, timestamp=datetime(2026, 9, 24, 10, 30))  # noqa: DTZ001
    assert path == tmp_path / "runlog_001.txt"
    (e,) = runlog.read_entries(path)
    assert e["number"] == 1
    assert e["timestamp"] == "2026-09-24 10:30:00"
    assert e["source"] == "Neutrons"
    assert e["description"] == "Cf-252 source\n10 cm, no shielding"
    # Newlines inside a value are folded so each field stays on one line.
    assert e["metadata"] == {"File": "cf252.npz", "Note": "two / lines"}
    assert e["stats"] == {"Traces": "12,000", "FOM": "1.31"}


def test_empty_description_is_marked(tmp_path):
    path = runlog.log_run("   ", source="API", log_dir=tmp_path)
    assert runlog.read_entries(path)[0]["description"] == "(no description)"


def test_rollover_after_max_entries(tmp_path):
    for i in range(7):
        runlog.log_run(f"run {i}", "API", log_dir=tmp_path, max_entries=3)
    files = runlog.log_files(tmp_path)
    assert [p.name for p in files] == [
        "runlog_001.txt", "runlog_002.txt", "runlog_003.txt"]
    assert [runlog.count_entries(p) for p in files] == [3, 3, 1]
    # Numbering restarts in each file; descriptions stay in order.
    assert [e["number"] for e in runlog.read_entries(files[1])] == [1, 2, 3]
    assert runlog.read_entries(files[2])[0]["description"] == "run 6"


def test_default_limit_is_50(tmp_path):
    assert runlog.MAX_ENTRIES == 50
    for i in range(51):
        runlog.log_run(f"run {i}", "API", log_dir=tmp_path)
    assert [runlog.count_entries(p) for p in runlog.log_files(tmp_path)] == [50, 1]


def test_env_var_sets_default_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("WARA_RUNLOG_DIR", str(tmp_path / "logs"))
    path = runlog.log_run("x", "API")
    assert path.parent == tmp_path / "logs"


def test_bad_max_entries(tmp_path):
    with pytest.raises(ValueError):
        runlog.log_run("x", "API", log_dir=tmp_path, max_entries=0)
