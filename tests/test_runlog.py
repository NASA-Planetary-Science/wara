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


def _write_stats(path, output_counts):
    import json
    path.write_text(json.dumps([{"module": 0, "output_counts": output_counts}]),
                    encoding="utf-8")


def test_run_folder_fields(tmp_path):
    import json
    run = tmp_path / "RUN-2026-09-24-00001"
    for sub in ("settings", "trace-data", "binary-data", "parquet-data", "MCA-data"):
        (run / sub).mkdir(parents=True)
    (run / "metadata.json").write_text(json.dumps(
        {"setup": "ATLAS_NEUTRONS", "detectors": {"EJ250": {"channel": 1}}}),
        encoding="utf-8")
    counts = [0] * 16
    counts[1], counts[9] = 100, 250
    # "-initial" snapshots are taken before the acquisition and not counted.
    _write_stats(run / "settings" / "R-stats-a-initial.json", [999] * 16)
    _write_stats(run / "settings" / "R-stats-a.json", counts)
    _write_stats(run / "settings" / "R-stats-b.json", counts)
    traces = [0] * 16
    traces[9] = 7
    _write_stats(run / "trace-data" / "R-stats-t.json", traces)
    (run / "binary-data" / "x.bin").write_bytes(b"\0" * 2500)
    (run / "trace-data" / "t.bin").write_bytes(b"")   # empty: no data
    (run / "parquet-data" / "p.parquet").write_bytes(b"\0" * 10)

    meta, stats = runlog.run_folder_fields(run)
    assert meta["Setup"] == "ATLAS_NEUTRONS"
    assert meta["Channels with data"] == "1, 9"
    assert meta["Binary data"] == "yes, 1 file, 2.5 kB"
    assert meta["Parquet data"] == "yes, 1 file, 10 B"
    assert meta["Trace data"] == "no"
    assert meta["MCA data"] == "no"
    assert stats == {"Recorded events ch 1": "200", "Recorded events ch 9": "500",
                     "Traces ch 9": "7"}


def test_run_folder_fields_parquet(tmp_path):
    import pandas as pd
    pq_dir = tmp_path / "parquet-data"
    pq_dir.mkdir()
    # Two chunks: per-channel reconstructed events and the reduction chain
    # are summed over them.
    pd.DataFrame({"channel": [1, 1, 4]}).to_parquet(pq_dir / "R-00001-pandas.parquet")
    pd.DataFrame({"channel": [1, 4, 4]}).to_parquet(pq_dir / "R-00002-pandas.parquet")
    err = {"input-total": [100], "verified-total": [10], "verified-no-error": [6],
           "verified-error": [4], "group-incomplete-group": [3],
           "group-discarded-input": [20], "time-window-incomplete-group": [2],
           "time-window-discarded-input": [30], "verified-pileup-1": [1],
           "verified-trace-flag-1": [2], "verified-CFD-error-4": [0]}
    pd.DataFrame(err).to_parquet(pq_dir / "R-00001-errors.parquet")
    pd.DataFrame(err).to_parquet(pq_dir / "R-00002-errors.parquet")

    _, stats = runlog.run_folder_fields(tmp_path)
    assert stats["Reconstructed events ch 1"] == "3"
    assert stats["Reconstructed events ch 4"] == "3"
    assert stats["Reco input events"] == "200"
    assert stats["Reco incomplete groups"] == "6 (40 events discarded)"
    assert stats["Reco time-window rejects"] == "4 (60 events discarded)"
    assert stats["Reco verified, no error"] == "12"
    assert stats["Reco errors ch 1"] == "pileup 2, trace flag 4"
    assert "Reco errors ch 4" not in stats          # all-zero counts are skipped


def test_run_folder_fields_parquet_legacy_labr(tmp_path):
    # 2022-2024 runs have no channel column: LaBr = channel 4, otherwise 5.
    import pandas as pd
    pq_dir = tmp_path / "parquet-data"
    pq_dir.mkdir()
    pd.DataFrame({"LaBr[y/n]": [True, False, False]}).to_parquet(
        pq_dir / "R-00001-pandas.parquet")
    _, stats = runlog.run_folder_fields(tmp_path)
    assert stats == {"Reconstructed events ch 4": "1",
                     "Reconstructed events ch 5": "2"}


def test_run_folder_fields_missing_folder(tmp_path):
    # An older / non-API run without metadata.json or data folders.
    meta, stats = runlog.run_folder_fields(tmp_path)
    assert "Setup" not in meta and "Channels with data" not in meta
    assert meta["Parquet data"] == "no"
    assert stats == {}


def test_find_run_and_replace_entry(tmp_path):
    md = {"Date": "2026-09-24", "Run": "1"}
    runlog.log_run("other run", "API", {"Date": "2026-09-24", "Run": "2"},
                   log_dir=tmp_path)
    assert runlog.find_run("API", md, tmp_path) is None
    path = runlog.log_run("first", "API", md, {"Recorded events ch 1": "10"},
                          log_dir=tmp_path)
    runlog.log_run("after", "Neutrons", {"File": "x.npz", "Path": "/d/x.npz"},
                   log_dir=tmp_path)
    found_path, e = runlog.find_run("API", md, tmp_path)
    assert (found_path, e["number"], e["description"]) == (path, 2, "first")
    # Same date/run from another tab is a different entry.
    assert runlog.find_run("Neutrons", md, tmp_path) is None
    assert runlog.find_run("Neutrons", {"Path": "/d/x.npz"}, tmp_path)[1]["number"] == 3
    assert runlog.find_run("API", {}, tmp_path) is None

    runlog.replace_entry(path, 2, "second\nline", "API", md, {"Recorded events ch 1": "20"})
    entries = runlog.read_entries(path)
    assert [x["number"] for x in entries] == [1, 2, 3]
    assert entries[1]["description"] == "second\nline"
    assert entries[1]["stats"] == {"Recorded events ch 1": "20"}
    assert entries[2]["description"] == "after"
    with pytest.raises(ValueError):
        runlog.replace_entry(path, 9, "x", "API")


def test_delete_entry_renumbers(tmp_path):
    for i in range(3):
        path = runlog.log_run(f"run {i}", "API", {"Date": "d", "Run": str(i)},
                              log_dir=tmp_path)
    assert runlog.delete_entry(path, 2) == 2
    entries = runlog.read_entries(path)
    assert [(e["number"], e["description"]) for e in entries] == [
        (1, "run 0"), (2, "run 2")]
    # The next entry continues the numbering.
    runlog.log_run("run 3", "API", log_dir=tmp_path)
    assert [e["number"] for e in runlog.read_entries(path)] == [1, 2, 3]
    with pytest.raises(ValueError):
        runlog.delete_entry(path, 7)
    runlog.delete_entry(path, 1); runlog.delete_entry(path, 1)
    runlog.delete_entry(path, 1)
    assert runlog.read_entries(path) == [] and runlog.count_entries(path) == 0
