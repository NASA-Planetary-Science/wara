"""
Read API parquet files
"""

import dateparser
import numpy as np
import pandas as pd
#import pkg_resources
from importlib.resources import files
from pathlib import Path


def get_data_path(data_path=None):
    if data_path is not None:
        return [Path(data_path)]
    #data_path_file = Path(pkg_resources.resource_filename("wara", "")).parent / "data-path.txt"
    data_path_file = Path(files("wara")).parent / "data-path.txt"
    with Path(data_path_file).open() as f:
        paths = [Path(line.strip()) for line in f if line.strip()]
    return paths


def get_files_in_path(date, runnr, folder="binary-data", data_path_txt=None):
    # TODO: merge with read_parquet_file
    DATE = dateparser.parse(date)
    if DATE is None:
        raise ValueError(f"Cannot parse date '{date}'")
    date_dir = f"{DATE.year}-{DATE.month:02d}-{DATE.day:02d}"
    fname = f"RUN-{DATE.year}-{DATE.month:02d}-{DATE.day:02d}-{runnr:05d}"
    for DATA_PATH in get_data_path(data_path_txt):
        DATA_DIR = DATA_PATH / date_dir
        FILE = DATA_DIR / fname
        if FILE.is_dir():
            return list(FILE.glob(f"{folder}/*"))
    print(f"ERROR: cannot find run {fname} in any path listed in data-path.txt")
    return []


def load_parquet_data_files(date, runnr, data_path_txt=None):
    # only channels 4 (LaBr==True) and 5 (LaBr==False)
    DATE = dateparser.parse(date)
    if DATE is None:
        raise ValueError(f"Cannot parse date '{date}'")
    date_dir = f"{DATE.year}-{DATE.month:02d}-{DATE.day:02d}"
    fname = f"RUN-{DATE.year}-{DATE.month:02d}-{DATE.day:02d}-{runnr:05d}"
    for DATA_PATH in get_data_path(data_path_txt):
        DATA_DIR = DATA_PATH / date_dir
        FILE = DATA_DIR / fname
        if FILE.is_dir():
            return list(FILE.glob(f"parquet-data/{fname}-*-pandas.parquet"))
    print(f"ERROR: cannot find run {fname} in any path listed in data-path.txt")
    return []


def channel_mask(df, ch):
    """Boolean mask selecting the events that belong to channel *ch*.

    Mirrors the per-channel filter used when loading a run: a ``channel`` column
    is matched directly; otherwise the legacy LaBr flag splits channel 4 (LaBr)
    from channel 5 (non-LaBr). Returns an all-False mask for any other channel
    when no ``channel`` column is present."""
    if "channel" in df.columns:
        return df["channel"] == ch
    if "LaBr[y/n]" in df.columns:
        if ch == 4:
            return df["LaBr[y/n]"].astype(bool)
        if ch == 5:
            return ~df["LaBr[y/n]"].astype(bool)
    return pd.Series(False, index=df.index)


def run_parquet_path(date, runnr, data_path=None, make=False):
    """Return ``(run_dir, parquet_dir, fname)`` for run (date, runnr) under the
    first configured data path. With ``make=True`` the ``parquet-data`` folder is
    created. ``fname`` is the ``RUN-YYYY-MM-DD-NNNNN`` stem the loader globs on."""
    DATE = dateparser.parse(date)
    if DATE is None:
        raise ValueError(f"cannot parse date '{date}'")
    date_dir = f"{DATE.year}-{DATE.month:02d}-{DATE.day:02d}"
    fname = f"RUN-{DATE.year}-{DATE.month:02d}-{DATE.day:02d}-{runnr:05d}"
    DATA_PATH = get_data_path(data_path)[0]
    run_dir = DATA_PATH / date_dir / fname
    parquet_dir = run_dir / "parquet-data"
    if make:
        parquet_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, parquet_dir, fname


def _parquet_num_rows(path):
    """Row count of the parquet at *path*, read from footer metadata when
    possible so a large file isn't loaded into memory just to be counted.

    Tries pyarrow's metadata, then fastparquet's, and only falls back to a full
    read if neither exposes the count. Raises if the file cannot be read at all
    (used by :func:`save_combined_run` to detect a truncated write)."""
    try:
        import pyarrow.parquet as pq
        return pq.ParquetFile(path).metadata.num_rows
    except Exception:
        pass
    try:
        from fastparquet import ParquetFile
        return ParquetFile(str(path)).count()
    except Exception:
        pass
    return len(pd.read_parquet(path))


def save_combined_run(df, date, runnr, data_path=None, overwrite=False):
    """Write *df* as a single combined parquet for run (date, runnr) so it loads
    straight back through :func:`read_parquet_file`.

    The file is named ``{fname}-00000-pandas.parquet`` so the loader's
    ``{fname}-*-pandas.parquet`` glob picks it up. When *overwrite* is set, any
    pre-existing ``{fname}-*-pandas.parquet`` chunks in the target folder are
    removed first so a reload doesn't concatenate stale data on top of the new
    combined file.

    After writing, the file is re-read and its row count is checked against the
    in-memory frame: a partial/truncated write (e.g. the app was closed or ran
    out of space mid-write of a large single parquet) otherwise passes silently
    and the combined run loads with missing events. On mismatch the bad file is
    removed and ``IOError`` is raised so the caller can report the failure
    instead of leaving a corrupt run on disk. Returns the written file path."""
    run_dir, parquet_dir, fname = run_parquet_path(date, runnr, data_path, make=True)
    if overwrite:
        for old in parquet_dir.glob(f"{fname}-*-pandas.parquet"):
            old.unlink()
    out = parquet_dir / f"{fname}-00000-pandas.parquet"
    try:
        df.to_parquet(out, engine="pyarrow")
    except Exception:
        df.to_parquet(out, engine="fastparquet")

    # Verify the write landed intact: re-read and compare row counts. Catches a
    # truncated write that would otherwise load as a silently-incomplete run.
    expected = len(df)
    try:
        written = _parquet_num_rows(out)
    except Exception as exc:  # unreadable file => the write is unusable
        out.unlink(missing_ok=True)
        raise IOError(
            f"Combined run {date}-{runnr} could not be verified after writing "
            f"(file unreadable: {exc}). The partial file was removed; retry the "
            f"combine.") from exc
    if written != expected:
        out.unlink(missing_ok=True)
        raise IOError(
            f"Combined run {date}-{runnr} was written incompletely: wrote "
            f"{written:,} of {expected:,} rows (likely a truncated/interrupted "
            f"write). The partial file was removed; retry the combine.")
    return out


def read_parquet_file(date, runnr, ch=None, flat_field=False, data_path_txt=None):
    files = load_parquet_data_files(date, runnr, data_path_txt)
    if not files:
        print(f"ERROR: No parquet file available for run {date}-{runnr}")
        return None
    try:
        df = pd.concat([pd.read_parquet(f, engine="pyarrow") for f in files])
    except Exception as exc:
        # pyarrow rejecting a file usually means it is corrupt or truncated.
        # fastparquet can often salvage a *partial* read from such a file, which
        # then loads as a silently-incomplete run (missing events). Warn loudly
        # so this is never mistaken for good data.
        print(f"WARNING: pyarrow could not read run {date}-{runnr} "
              f"({exc}); falling back to fastparquet. The file(s) may be "
              f"corrupt or truncated and the loaded data may be INCOMPLETE.")
        df = pd.concat([pd.read_parquet(f, engine="fastparquet") for f in files])

    if flat_field or ch is None:
        df.reset_index(drop=True, inplace=True)
        return df

    # df["dt"] *= dt_multiplier  # to ns
    df = df[channel_mask(df, ch)]
    df.reset_index(drop=True, inplace=True)
    return df


def read_parquet_file_time_aligned(date, runnr, ch=None, flat_field=False, data_path_txt=None,
                                   dt_bins=512, min_snr=5, ref_fwhm=3):
    from .peaksearch import PeakSearch
    from .spectrum import Spectrum

    files = load_parquet_data_files(date, runnr, data_path_txt)
    if not files:
        print(f"ERROR: No parquet file available for run {date}-{runnr}")
        return None

    dfs = []
    for f in files:
        df = pd.read_parquet(f)

        if not flat_field and ch is not None:
            if "channel" in df.columns:
                df = df[df["channel"] == ch]
            else:
                if ch == 4:
                    df = df[df["LaBr[y/n]"]]
                elif ch == 5:
                    df = df[~df["LaBr[y/n]"]]

        counts, bin_edges = np.histogram(df["dt"], bins=dt_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        spect = Spectrum(counts=counts, energies=bin_centers)
        ps = PeakSearch(spect, ref_x=dt_bins // 2, ref_fwhm=ref_fwhm, fwhm_at_0=1.0,
                        min_snr=min_snr, method="fast")

        if len(ps.peaks_idx) > 0:
            first_peak_dt = spect.energies[ps.peaks_idx[0]]
            df = df.copy()
            df["dt"] = df["dt"] - first_peak_dt
        else:
            print(f"WARNING: No prominent peak found in {f.name}, skipping time alignment")

        dfs.append(df)

    result = pd.concat(dfs)
    result.reset_index(drop=True, inplace=True)
    return result


def read_parquet_file_from_path(filepath, ch):
    path = Path(filepath)
    files = list(path.glob("parquet-data/*-pandas.parquet"))
    if not files:
        print(f"ERROR: No parquet files found in {path}")
        return None
    df = pd.concat([pd.read_parquet(f) for f in files])

    # df["dt"] *= 1e9  # to ns
    if "channel" in df.columns:
        df = df[df["channel"] == ch]
    else:
        if ch == 4:
            df = df[df["LaBr[y/n]"]]
        elif ch == 5:
            df = df[~df["LaBr[y/n]"]]
    df.reset_index(drop=True, inplace=True)
    return df


def initialize_plots(df, ax_g, ax_t, ax_xy):
    tr = [-40, 80]  # dt range
    ebins = 2048  # energy bins
    hexbins = 80  # x-y bins
    tbins = 512  # time bins
    xyplane = (-0.9, 0.9, -0.9, 0.9)  # x and y limits
    # xyplane = (-0.1,0.1,-0.1,0.1) # for X_alpha, y_alpha
    # magma, plt.cm.BuGn_r, plt.cm.Greens, plasma, jet, viridis, cvidis
    colormap = "plasma"
    plot_energy_hist(df=df, ebins=ebins, ax=ax_g)
    plot_time_hist(df=df, tbins=tbins, trange=tr, ax=ax_t)
    plot_xy(df, hexbins, colormap, xyplane, ax=ax_xy, cbar=True)


def plot_energy_hist(df, ebins, ax):
    gam, e = np.histogram(df["energy"], bins=ebins)
    bin_centers = (e[:-1] + e[1:]) / 2
    ax.plot(bin_centers, gam, color="green")
    return gam, e


def plot_time_hist(df, tbins, trange, ax):
    df["dt"].plot.hist(bins=tbins, ax=ax, range=trange, alpha=0.7, edgecolor="black")


def plot_xy(df, hexbins, colormap, xyplane, ax, cbar=True):
    df.plot.hexbin(
        x="X2",
        y="Y2",
        gridsize=hexbins,
        cmap=colormap,
        ax=ax,
        colorbar=cbar,
        extent=xyplane,
    )  # , bins="log")
