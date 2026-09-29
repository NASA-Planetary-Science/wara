"""Tests for wara.read_parquet_api.read_parquet_file on real parquet files in
a temporary run folder: column projection, channel pushdown, file order."""
import numpy as np
import pandas as pd
import pytest

from wara import read_parquet_api as rpa

pytest.importorskip("pyarrow")

FNAME = "RUN-2024-01-02-00007"


def _write_run(tmp_path, legacy=False):
    """Two parquet chunks of a run under tmp_path; returns the combined frame."""
    pq_dir = tmp_path / "2024-01-02" / FNAME / "parquet-data"
    pq_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    frames = []
    for i in range(2):
        n = 50
        df = pd.DataFrame({
            "dt": rng.normal(0, 1e-8, n),
            "energy": rng.uniform(0, 1000, n),
            "File time": rng.uniform(0, 1, n),
            "Trace": [np.arange(4, dtype=np.uint16) + k for k in range(n)],
        })
        if legacy:
            df["LaBr[y/n]"] = rng.integers(0, 2, n).astype(bool)
        else:
            df["channel"] = rng.choice([1, 4, 5], n)
        df.to_parquet(pq_dir / f"{FNAME}-{i:05d}-pandas.parquet")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def test_columns_and_channel_pushdown(tmp_path):
    full = _write_run(tmp_path)
    df = rpa.read_parquet_file("2024-01-02", 7, ch=4, data_path_txt=tmp_path,
                               columns=("dt", "energy", "missing"))
    ref = full[full["channel"] == 4].reset_index(drop=True)
    assert list(df.columns) == ["dt", "energy"]       # absent names skipped
    assert list(df.index) == list(range(len(ref)))
    np.testing.assert_array_equal(df["energy"], ref["energy"])  # file order kept


def test_default_read_keeps_every_column(tmp_path):
    full = _write_run(tmp_path)
    df = rpa.read_parquet_file("2024-01-02", 7, data_path_txt=tmp_path)
    assert set(df.columns) == set(full.columns) and len(df) == len(full)
    assert isinstance(df["Trace"].iloc[0], np.ndarray)   # plain object traces


def test_lists_as_arrow_keeps_traces_arrow_backed(tmp_path):
    _write_run(tmp_path)
    df = rpa.read_parquet_file("2024-01-02", 7, ch=1, data_path_txt=tmp_path,
                               columns=("Trace",), lists_as_arrow=True)
    assert isinstance(df["Trace"].dtype, pd.ArrowDtype)


def test_legacy_labr_files_still_split_by_channel(tmp_path):
    full = _write_run(tmp_path, legacy=True)
    df = rpa.read_parquet_file("2024-01-02", 7, ch=5, data_path_txt=tmp_path,
                               columns=("energy",))
    ref = full[~full["LaBr[y/n]"]]
    np.testing.assert_array_equal(df["energy"], ref["energy"])
