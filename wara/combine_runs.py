"""Combine several API runs into one.

The runs (any dates) are simply concatenated into a single combined dataframe
that loads straight back through :func:`wara.read_parquet_api.read_parquet_file`.
No drift correction is applied here: if the combined run needs gain/time
alignment, do it afterwards with the single-run Shifts window (combine first,
then shift). This module also builds the per-run overlay spectra the Combine
window uses to *visualize* the runs before stitching them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import read_parquet_api
from . import spectrum as sp


def energy_base_col(df: pd.DataFrame) -> str:
    """Raw-channel column to plot: ``energy_orig`` if present, else ``energy``."""
    return "energy_orig" if "energy_orig" in df.columns else "energy"


def read_runs(runs, data_path=None):
    """Read every run in *runs* (a list of ``(date, runnr)``) at full channel
    width. Returns a list of ``(date, runnr, df)`` in the given order; raises
    ``FileNotFoundError`` for the first run with no parquet data."""
    out = []
    for date, runnr in runs:
        df = read_parquet_api.read_parquet_file(
            date=date, runnr=runnr, ch=None, data_path_txt=data_path)
        if df is None:
            raise FileNotFoundError(f"No parquet data for run {date}-{runnr}")
        out.append((date, runnr, df.reset_index(drop=True)))
    return out


def channels_in_common(runs_data):
    """Sorted list of channels present in *every* run (used to populate the
    preview channel selector). Falls back to the legacy LaBr split (channels
    4 & 5) when the runs carry a ``LaBr[y/n]`` flag instead of a ``channel``
    column."""
    common = None
    for (_, _, df) in runs_data:
        if "channel" in df.columns:
            chans = set(int(c) for c in pd.unique(df["channel"]))
        elif "LaBr[y/n]" in df.columns:
            chans = {4, 5}
        else:
            chans = set()
        common = chans if common is None else (common & chans)
    return sorted(common or [])


def run_spectra(runs_data, ch, axis, bins=None):
    """Per-run overlay spectra for one *channel* on one *axis*, for visualization.

    ``axis`` is ``"energy"`` (raw channels) or ``"time"`` (``dt`` in ns). All runs
    are histogrammed over a shared range so the overlay is directly comparable:
    energy uses ``[0, p99.5]`` and time ``[p0.2, p99.5]`` of the pooled values, so
    stray edge events don't squash the real structure. Returns a list of
    :class:`wara.spectrum.Spectrum`, one per run (same order as *runs_data*).
    """
    if axis == "energy":
        base = energy_base_col(runs_data[0][2])
        bins = bins or 4096
        get = lambda df, m: df.loc[m, base].astype(float).to_numpy()  # noqa: E731
    elif axis == "time":
        bins = bins or 512
        get = lambda df, m: df.loc[m, "dt"].astype(float).to_numpy() * 1e9  # noqa: E731
    else:
        raise ValueError("axis must be 'energy' or 'time'")

    per_run = []
    for (_, _, df) in runs_data:
        m = read_parquet_api.channel_mask(df, ch).to_numpy()
        per_run.append(get(df, m))

    pooled = np.concatenate(per_run) if per_run else np.array([], dtype=float)
    if axis == "energy":
        hi = float(np.percentile(pooled, 99.5)) if len(pooled) else 1.0
        erange = (0.0, hi if hi > 0 else 1.0)
    elif len(pooled):
        lo, hi = np.percentile(pooled, [0.2, 99.5])
        erange = (float(lo), float(hi)) if hi > lo else (float(pooled.min()),
                                                         float(pooled.max()))
    else:
        erange = (0.0, 1.0)

    spectra = []
    for vals in per_run:
        cts, edges = np.histogram(vals, bins=bins, range=erange)
        x = (edges[:-1] + edges[1:]) / 2
        spectra.append(sp.Spectrum(counts=cts, energies=x))
    return spectra


# Calibration columns that must NOT survive a combine or a shift: they were
# produced for the individual source run and can't be assumed valid afterwards.
CAL_COLS = ("energy_cal", "dt_cal")


def combine_runs(runs_data):
    """Concatenate the runs (all channels) in order into one dataframe.

    Any per-run calibration columns (``energy_cal`` / ``dt_cal``) are **dropped**:
    combining runs invalidates them, so the combined run is left uncalibrated and
    the user must re-calibrate (energy) / re-align (time) afterwards. No drift
    correction is applied here. Returns ``(combined_df, info)`` where ``info``
    records the source runs, the total event count, and which calibration columns
    were removed (for the warning + README).
    """
    dropped = set()
    frames = []
    for (_, _, df) in runs_data:
        present = [c for c in CAL_COLS if c in df.columns]
        dropped.update(present)
        frames.append(df.drop(columns=present) if present else df)
    combined = pd.concat(frames, ignore_index=True)
    info = {"sources": [(d, r, int(len(df))) for d, r, df in runs_data],
            "n_events": int(len(combined)),
            "dropped_cal": sorted(dropped)}
    return combined, info
