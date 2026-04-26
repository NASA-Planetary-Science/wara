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
        print(f"ERROR: cannot parse date '{date}'")
        return []
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
        print(f"ERROR: cannot parse date '{date}'")
        return []
    date_dir = f"{DATE.year}-{DATE.month:02d}-{DATE.day:02d}"
    fname = f"RUN-{DATE.year}-{DATE.month:02d}-{DATE.day:02d}-{runnr:05d}"
    for DATA_PATH in get_data_path(data_path_txt):
        DATA_DIR = DATA_PATH / date_dir
        FILE = DATA_DIR / fname
        if FILE.is_dir():
            return list(FILE.glob(f"parquet-data/{fname}-*-pandas.parquet"))
    print(f"ERROR: cannot find run {fname} in any path listed in data-path.txt")
    return []


def read_parquet_file(date, runnr, ch=None, flood_field=False, data_path_txt=None):
    files = load_parquet_data_files(date, runnr, data_path_txt)
    if not files:
        print(f"ERROR: No parquet file available for run {date}-{runnr}")
        return None
    df = pd.concat([pd.read_parquet(f) for f in files])

    if flood_field or ch is None:
        df.reset_index(drop=True, inplace=True)
        return df

    # df["dt"] *= dt_multiplier  # to ns
    if "channel" in df.columns:
        # df["channel"] = df["channel"].str.extract(r"(\d+)$").astype(int)
        df = df[df["channel"] == ch]
    else:
        if ch == 4:
            df = df[df["LaBr[y/n]"]]
        elif ch == 5:
            df = df[~df["LaBr[y/n]"]]
    df.reset_index(drop=True, inplace=True)
    return df


def read_parquet_file_time_aligned(date, runnr, ch=None, flood_field=False, data_path_txt=None,
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

        if not flood_field and ch is not None:
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
                        min_snr=min_snr, method="scipy")

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
