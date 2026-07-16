"""
Functions used to analyze API data

"""

import numpy as np
import matplotlib.pyplot as plt

# from mayavi import mlab
import pandas as pd
import os

# import mpl_scatter_density  # adds projection='scatter_density'
from matplotlib.colors import LinearSegmentedColormap
import dateparser
from wara.read_parquet_api import get_data_path
from wara import read_parquet_api
import matplotlib.cm as cm
from scipy.signal import correlate
import scipy.constants as SC
from wara import spectrum as sp


# ---------------------------------------------------------------------------
# Physics constants for the full position reconstruction (see
# docs/API_position_reconstruction.pdf for the derivation).
# ---------------------------------------------------------------------------
_E_ALPHA = 3.5      # MeV, alpha kinetic energy from D-T
_E_NEUTRON = 14.1   # MeV, neutron kinetic energy from D-T
_M_ALPHA = SC.physical_constants["alpha particle mass energy equivalent in MeV"][0]
_M_NEUTRON = SC.physical_constants["neutron mass energy equivalent in MeV"][0]
_M_DEUTERON = SC.physical_constants["deuteron mass energy equivalent in MeV"][0]
_M_TRITON = SC.physical_constants["triton mass energy equivalent in MeV"][0]


def _speed(E, m, relativistic=True):
    """Particle speed [m/s] from kinetic energy E and rest mass m (both MeV)."""
    if relativistic:
        g = 1.0 + E / m
        beta = np.sqrt(1.0 - 1.0 / g / g)
    else:
        beta = np.sqrt(2.0 * E / m)
    return beta * SC.c


def v_com_from_beam(ion_energy_kev):
    """Center-of-mass speed [m/s] of the D-T reaction for a deuteron beam of
    energy ``ion_energy_kev`` (keV) on a stationary triton. Reproduces the
    ``center-of-mass.npy`` lookup used by ROOTS ``calcXYZ`` from kinematics:
    v_com = sqrt(2 m_d E_d) / (m_d + m_t) * c."""
    Ed = ion_energy_kev / 1000.0  # MeV
    p_d = np.sqrt(2.0 * _M_DEUTERON * Ed)  # MeV/c
    return p_d / (_M_DEUTERON + _M_TRITON) * SC.c


def test_limits(df, elog=True, Vmax=None, ekey="energy", tkey="dt", xkey="X", ykey="Y"):
    """Plots all data to define energy, time, and xy limits. df is a pandas dataframe"""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas dataframe")
    plt.figure()
    plt.hist(df[ekey], 256)
    plt.yscale("log" if elog else "linear")
    plt.title("Energy")

    res, xed, yed = np.histogram2d(df[ykey], df[xkey], bins=100)
    plt.figure()
    plt.imshow(
        res,
        extent=[df[xkey].min(), df[xkey].max(), df[ykey].min(), df[ykey].max()],
        cmap="inferno",
        interpolation="bilinear",
        vmax=Vmax,
        origin="lower",
    )
    plt.title("All events")

    plt.figure()
    plt.hist(df[tkey], 256)
    plt.xlabel("ns")
    plt.title("dt - all events")


def calc_own_pos(df, xkey_new="X2", ykey_new="Y2"):
    """Return a pandas data frame with two extra columns X2, Y2
    with calculated positions based on the equations below"""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas dataframe")
    if xkey_new in df.columns and ykey_new in df.columns:
        return df
    else:
        tot = df["A"] + df["B"] + df["C"] + df["D"]
        X2 = (df["B"] + df["C"] - df["D"] - df["A"]) / tot
        Y2 = (df["B"] + df["A"] - df["D"] - df["C"]) / tot
        df2 = df.copy(deep=True)
        df2[xkey_new] = X2
        df2[ykey_new] = Y2
        df2.fillna(0, inplace=True)
        return df2


def dfe(df, erange, ekey="energy"):
    """Return a pandas data frame filtered in energy"""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas dataframe")
    dfenergy = df[(df[ekey] > erange[0]) & (df[ekey] < erange[1])]
    return dfenergy


def dfxy(df, xrange, yrange, xkey="X", ykey="Y"):
    """Return a pandas data frame filtered in x-y"""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas dataframe")
    df_xy = df[
        (df[xkey] > xrange[0])
        & (df[xkey] < xrange[1])
        & (df[ykey] > yrange[0])
        & (df[ykey] < yrange[1])
    ]
    return df_xy

def dfxy_exclude(df, xrange, yrange, xkey="X", ykey="Y"):
    """Return a pandas DataFrame with rows inside the x-y range excluded (only edges remain)."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    df_xy = df[
        (df[xkey] <= xrange[0])
        | (df[xkey] >= xrange[1])
        | (df[ykey] <= yrange[0])
        | (df[ykey] >= yrange[1])
    ]
    return df_xy

def dfxye(df, xrange, yrange, erange, xkey="X", ykey="Y", ekey="energy"):
    """Return a pandas data frame filtered in x-y and energy"""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas dataframe")
    dfen = dfe(df, erange, ekey)
    df_xy_e = dfen[
        (dfen[xkey] > xrange[0])
        & (dfen[xkey] < xrange[1])
        & (dfen[ykey] > yrange[0])
        & (dfen[ykey] < yrange[1])
    ]
    return df_xy_e


def dftxye(
    df, xrange, yrange, erange, trange, xkey="X", ykey="Y", ekey="energy", tkey="dt"
):
    """Return a pandas data frame filtered in x-y, energy, and time"""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas dataframe")
    dfxyen = dfxye(df, xrange, yrange, erange, xkey, ykey, ekey)
    dftxy_e = dfxyen[(dfxyen[tkey] > trange[0]) & (dfxyen[tkey] < trange[1])]

    return dftxy_e


def dftxy(df, xrange, yrange, trange, xkey="X", ykey="Y", tkey="dt"):
    """Return a pandas data frame filtered in x-y, and time"""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas dataframe")
    df_xy = dfxy(df, xrange, yrange, xkey, ykey)
    dft_xy = df_xy[(df_xy[tkey] > trange[0]) & (df_xy[tkey] < trange[1])]

    return dft_xy


def dft(df, trange, tkey="dt"):
    """Return a pandas data frame filtered in time"""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas dataframe")
    df_t = df[(df[tkey] > trange[0]) & (df[tkey] < trange[1])]
    return df_t


def dfet(df, erange, trange, ekey="energy", tkey="dt"):
    """Return a pandas data frame filtered in energy and time"""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas dataframe")
    dfen = dfe(df, erange, ekey)
    df_e_t = dfen[(dfen[tkey] > trange[0]) & (dfen[tkey] < trange[1])]
    return df_e_t


def time_cut(df, t_start, t_stop, step, xkey="X", ykey="Y", tkey="dt"):
    """2D plots of time cuts in ns. Use a dataframe filtered in x-y and energy"""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas dataframe")
    tdiff = df[tkey]
    X = df[xkey]
    Y = df[ykey]
    for k in np.arange(t_start, t_stop, step):
        mask = (tdiff > k) & (tdiff < k + step)
        X2 = X[mask]
        Y2 = Y[mask]
        # Z2 = [x for i,x in enumerate(Z) if i not in idx2]

        result, xedges, yedges = np.histogram2d(X2, Y2, 80)
        plt.figure()
        plt.imshow(result, interpolation="bilinear", cmap="inferno")
        plt.title(f"time cut {k,k+step} ns")
        plt.show()

def compute_fft(signal, sampling_rate):
    """
    Compute the FFT of a signal.

    Parameters:
        signal (numpy.ndarray): The input signal.
        sampling_rate (float): Sampling rate in Hz.

    Returns:
        freqs (numpy.ndarray): Frequencies corresponding to FFT components.
        fft_magnitude (numpy.ndarray): Magnitude of the FFT.
    """
    n = len(signal)
    fft_values = np.fft.fft(signal)
    fft_magnitude = np.abs(fft_values) / n  # Normalize
    freqs = np.fft.fftfreq(n, d=1/sampling_rate)

    # Only keep the positive frequencies
    pos_mask = freqs >= 0
    return freqs[pos_mask], fft_magnitude[pos_mask]

## helper functions
def find_data_path(date, runnr, data_path=None):
    RUNNR = runnr
    DATE = dateparser.parse(date)
    if DATE is None:
        raise ValueError(f"Could not parse date string: '{date}'")

    date_dir = f"{DATE.year}-{DATE.month:02d}-{DATE.day:02d}"
    fname = f"RUN-{DATE.year}-{DATE.month:02d}-{DATE.day:02d}-{RUNNR:05d}"
    for dp in get_data_path(data_path):
        file_path = dp / date_dir / fname
        if file_path.is_dir():
            return file_path
    raise FileNotFoundError(f"Cannot find run {fname} in any path listed in data-path.txt")


# Read .npy MCA data, join if more than 1 file
def read_mca(date, runnr):
    # only channels 4 (LaBr==True) and 5 (LaBr==False)
    file_path = find_data_path(date, runnr)
    # load data
    files = list(file_path.glob("MCA-data/*.npy"))
    if len(files) == 0:
        raise FileNotFoundError(f"No .npy files found in {file_path / 'MCA-data'}")
    elif len(files) > 1:
        data = 0
        for f in files:
            data0 = np.load(f)
            data = data + data0
    else:
        data = np.load(files[0])

    return data


def read_time_from_settings(settings_file, ch):
    with open(settings_file, mode="r") as myfile:
        lines = myfile.readlines()

    idx = None
    for i, line in enumerate(lines):
        tmp = line.replace('"', "").split()
        if "live_time:" in tmp:
            idx = i
    if idx is None:
        raise ValueError(f"'live_time:' key not found in {settings_file}")
    time = float(lines[idx + ch + 1].split(",")[0])
    return time


def read_input_CR_from_settings(settings_file, ch=9):
    with open(settings_file, mode="r") as myfile:
        lines = myfile.readlines()

    idx = None
    for i, line in enumerate(lines):
        tmp = line.replace('"', "").split()
        if "input_count_rate:" in tmp:
            idx = i
    if idx is None:
        raise ValueError(f"'input_count_rate:' key not found in {settings_file}")
    CR = float(lines[idx + ch + 1].split(",")[0])
    return CR


def read_input_counts_from_settings(settings_file, ch=9):
    with open(settings_file, mode="r") as myfile:
        lines = myfile.readlines()

    idx = None
    for i, line in enumerate(lines):
        tmp = line.replace('"', "").split()
        if "input_counts:" in tmp:
            idx = i
    if idx is None:
        raise ValueError(f"'input_counts:' key not found in {settings_file}")
    CR = float(lines[idx + ch + 1].split(",")[0])
    return CR


def get_total_time(date, runnr, ch, data_path=None, mca=False):
    file_path = find_data_path(date, runnr, data_path)
    # load data
    if mca:
        files = sorted(list(file_path.glob("MCA-data/*-stats-*")))[1:]
    else:
        files = sorted(list(file_path.glob("settings/*-stats-*")))[1:]
    t_tot = 0
    for f in files:
        t0 = read_time_from_settings(f, ch=ch)
        t_tot += t0
    return t_tot


def get_settings_files(date, runnr, data_path=None):
    file_path = find_data_path(date, runnr, data_path)
    files = list(file_path.glob("settings/*-stats-*"))
    return files


def get_total_counts(date, runnr, ch, data_path=None):
    files = get_settings_files(date, runnr, data_path)
    cts_tot = 0
    for f in files:
        cts0 = read_input_counts_from_settings(f, ch=ch)
        cts_tot += cts0
    return cts_tot


def combine_settings_files(dates, runnrs, data_path=None):
    all_files = []
    for d, r in zip(dates, runnrs):
        files = get_settings_files(date=d, runnr=r, data_path=data_path)
        all_files.append(files)
    all_files_flatten = [x for xs in all_files for x in xs]
    return all_files_flatten


def calculate_neutron_yield(date, runnr, ch=9, data_path=None):
    alpha_counts = get_total_counts(date, runnr, ch, data_path)
    time_total = get_total_time(date, runnr, ch, data_path)
    alpha_cr = alpha_counts / time_total
    d = 6.7  # cm alpha detector-neutron source distance
    alpha_area = 4.8 * 4.8  # cm2
    phi_a = alpha_cr / alpha_area  # flux at alpha detector
    Y0 = 4 * np.pi * d**2 * phi_a  # neutron yield (n/s)
    # alpha_frac = 0.91  # correction factor for true alphas
    alpha_frac = 1
    return alpha_frac * Y0


def calculate_neutron_flux(date, runnr, ch, L=30):
    Y0 = calculate_neutron_yield(date=date, runnr=runnr, ch=ch)
    phi_s = Y0 / (4 * np.pi * L**2)  # neutron flux on sample
    return phi_s


def approximate_fa(L=30, S=10):
    """
    Approximate fraction of alpha particles that intersect a square sample
    of length S located at a distance L from the neutron source.

    Parameters
    ----------
    L : float, optional
        Distance in cm from the neutron source to the sample. The default is 30.
    S : float, optional
        Side lenght of square sample in cm. The default is 10.

    Returns
    -------
    fa : float
        fraction of counts in the alpha detector covered by the sample.

    """
    d = 6.7  # cm alpha detector-neutron source distance
    xa = 4.8  # cm alpha detector active area
    alpha_area = xa * xa  # cm2
    L_alpha = (d / L) * (S / 2) * 2
    sample_area = L_alpha * L_alpha
    fa = sample_area / alpha_area
    return min(fa, 1.0)

class GainShift:
    """
    Splits a list-mode DataFrame into N time segments and corrects drift in one
    axis by aligning each segment onto the first, writing the corrected values
    back as a new column.

    The axis is configurable so the same machinery serves both the energy axis
    (the default, ``energy_orig`` → ``energy_cal``) and the time axis
    (``col="dt"``, ``out_col="dt_cal"``).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the ``col`` column. Rows are assumed to be in time order
        (list mode).
    n_segments : int
        Number of equal-length time segments (split by row count).
    bins : int
        Number of histogram bins used when building spectra.
    erange : (float, float)
        (min, max) range of the ``col`` axis for histogramming.
    col : str
        Source column to align (the drifting axis). Default ``"energy_orig"``.
    out_col : str
        Column the corrected values are written to. Default ``"energy_cal"``.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        n_segments: int = 10,
        bins: int = 2**12,
        erange: tuple[float, float] = (0, 2**16),
        col: str = "energy_orig",
        out_col: str = "energy_cal",
    ) -> None:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain a {col!r} column.")

        self.df         = df.copy()
        self.n_segments = n_segments
        self.bins       = bins
        self.erange     = erange
        self.col        = col
        self.out_col    = out_col

        self._seg_idx: list[np.ndarray] = np.array_split(np.arange(len(self.df)), n_segments)
        self._segments: list[pd.DataFrame] = [self.df.iloc[i] for i in self._seg_idx]
        self._spectra:  list[sp.Spectrum]  = self._build_spectra()

        # One calibration dict (or None) per segment.
        # dict keys: segment, channels_fit, energies_kev, slope, intercept
        self._cal_params: list[dict | None] = [None] * n_segments
        # Aligned (gain-corrected) spectra, filled by align().
        self._aligned_spectra: list[sp.Spectrum] | None = None

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_spectra(self) -> list[sp.Spectrum]:
        spectra = []
        for seg in self._segments:
            cts, edges = np.histogram(
                seg[self.col], bins=self.bins, range=self.erange
            )
            x = (edges[1:] + edges[:-1]) / 2
            spectra.append(sp.Spectrum(counts=cts, energies=x))
        return spectra

    @staticmethod
    def _apply_cal(energy_orig: pd.Series, cal: dict) -> pd.Series:
        """
        Apply one calibration dict to a raw-channel Series.

        Handles both linear and quadratic calibrations transparently.
        """
        if cal.get("quadratic"):
            ch = energy_orig.values.astype(float)
            result = cal["intercept"] + cal["slope"] * ch + cal["curvature"] * ch ** 2
            return pd.Series(result, index=energy_orig.index)
        return cal["intercept"] + cal["slope"] * energy_orig

    # ── Alignment ─────────────────────────────────────────────────────────────

    def align(
        self,
        reference: int = 0,
        method: str = "shift",
        xrange: tuple[float, float] | None = None,
        n_windows: int = 8,
    ):
        """Gain-correct every segment onto the reference segment.

        Each segment is mapped onto the reference (default segment 0 — the
        earliest in time, i.e. the left-most band in :meth:`plot_overlay`) with
        an affine correction ``energy_cal = intercept + slope * energy_orig``,
        written back to ``self.df["energy_cal"]``. The per-segment coefficients
        land in ``self._cal_params`` and aligned spectra are cached for
        :meth:`plot_aligned`.

        Two models are available:

        ``"shift"`` (default)
            Pure additive slide (``slope = 1``); the shift is the
            cross-correlation lag between the segment and reference histograms,
            refined to sub-bin precision. Matches wara's gain-shift convention
            (:meth:`wara.spectrum.Spectrum.gain_shift`). Robust even with a
            single peak.

        ``"linear"``
            Slope **and** offset. The local cross-correlation shift is measured
            in ``n_windows`` channel windows across the axis; under an affine
            gain drift that shift is linear in channel, so a (count-weighted)
            straight-line fit through the per-window shifts gives the slope and
            offset. Needs structure spread across the axis (≥2 windows with
            signal); with too little it falls back to a pure shift for that
            segment.

        Parameters
        ----------
        reference : int
            Index of the segment every other segment is aligned to. Default 0.
        method : {"shift", "linear"}
            Correction model (see above). Default "shift".
        xrange : (xmin, xmax) or None
            Restrict the correlation to this raw-channel window. None uses the
            full axis.
        n_windows : int
            Number of channel windows used to estimate the slope when
            ``method="linear"``. Ignored for ``method="shift"``.

        Returns
        -------
        list of dict
            Per-segment calibration dicts with ``slope`` and ``intercept`` (the
            coefficients of ``energy_cal = intercept + slope * energy_orig``).
        """
        method = method.lower()
        if method not in ("shift", "linear"):
            raise ValueError("method must be 'shift' or 'linear'")

        x = self._spectra[reference].energies
        binwidth = float(x[1] - x[0])
        mask = (np.ones(len(x), dtype=bool) if xrange is None
                else (x >= xrange[0]) & (x <= xrange[1]))
        ref = self._spectra[reference].counts.astype(float)

        raw = self.df[self.col].to_numpy(dtype=float)
        corrected = raw.copy()
        self._cal_params = [None] * self.n_segments
        for i, (pos, spe) in enumerate(zip(self._seg_idx, self._spectra)):
            seg = spe.counts.astype(float)
            if method == "shift":
                slope, intercept = self._fit_shift(ref, seg, mask, binwidth)
            else:
                slope, intercept = self._fit_linear(
                    ref, seg, x, mask, binwidth, n_windows)
            corrected[pos] = intercept + slope * raw[pos]
            self._cal_params[i] = {"segment": i, "slope": slope,
                                   "intercept": intercept}

        self.df[self.out_col] = corrected
        self._aligned_spectra = self._build_aligned_spectra()
        return self._cal_params

    def _fit_shift(self, ref, seg, mask, binwidth):
        """Pure additive shift: (slope=1, intercept=shift) from the global lag."""
        r, s = ref[mask], seg[mask]
        if r.std() == 0 or s.std() == 0:
            return 1.0, 0.0
        lag = self._xcorr_lag(r - r.mean(), s - s.mean())
        return 1.0, lag * binwidth

    def _fit_linear(self, ref, seg, x, mask, binwidth, n_windows):
        """Slope + offset from per-window local shifts.

        Under an affine drift energy_cal = a + b·ch, the shift at channel ch is
        a + (b-1)·ch — linear in ch. Measure it in several windows and fit a
        line; b = 1 + p1, a = p0. Falls back to a pure shift when fewer than two
        windows carry usable signal.
        """
        idx = np.flatnonzero(mask)
        centers, shifts, weights = [], [], []
        for chunk in np.array_split(idx, n_windows):
            if chunk.size < 4:
                continue
            r, s = ref[chunk], seg[chunk]
            if r.std() == 0 or s.std() == 0:
                continue
            lag = self._xcorr_lag(r - r.mean(), s - s.mean())
            centers.append(float(x[chunk].mean()))
            shifts.append(lag * binwidth)
            weights.append(float(min(r.sum(), s.sum())))   # signal in the window

        if len(centers) < 2 or np.ptp(centers) == 0:
            return self._fit_shift(ref, seg, mask, binwidth)   # can't get a slope

        w = np.sqrt(np.clip(weights, 0, None))
        p1, p0 = np.polyfit(centers, shifts, 1, w=w)   # shift = p1·center + p0
        return 1.0 + p1, p0

    @staticmethod
    def _xcorr_lag(ref: np.ndarray, seg: np.ndarray) -> float:
        """Channel shift (in bins) to add to *seg* so it lines up with *ref*.

        Positive means seg's features sit at lower channels than ref and must be
        moved up; the value is refined to sub-bin precision with a 3-point
        parabolic fit around the cross-correlation peak.
        """
        corr = correlate(ref, seg, mode="full")
        lags = np.arange(-len(seg) + 1, len(ref))
        j = int(np.argmax(corr))
        lag = float(lags[j])
        # Sub-bin refinement: vertex of the parabola through the 3 peak samples.
        if 0 < j < len(corr) - 1:
            y0, y1, y2 = corr[j - 1], corr[j], corr[j + 1]
            denom = y0 - 2.0 * y1 + y2
            if denom != 0:
                lag += 0.5 * (y0 - y2) / denom
        return lag

    def _build_aligned_spectra(self) -> list[sp.Spectrum]:
        spectra = []
        corrected = self.df[self.out_col].to_numpy(dtype=float)
        for pos in self._seg_idx:
            cts, edges = np.histogram(corrected[pos], bins=self.bins, range=self.erange)
            xc = (edges[1:] + edges[:-1]) / 2
            spectra.append(sp.Spectrum(counts=cts, energies=xc))
        return spectra

    # ── Plotting — raw ────────────────────────────────────────────────────────

    def _overlay(
        self,
        spectra: list[sp.Spectrum],
        title: str,
        xlabel: str,
        xrange: list[float] | None = None,
        yscale: str = "log",
        ax: plt.Axes | None = None,
        figsize: tuple[float, float] = (14, 6),
    ) -> plt.Figure:
        """Overlay a list of segment spectra, coloured by time order (viridis)."""
        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        colors = cm.viridis(np.linspace(0, 1, self.n_segments))
        for i, spe in enumerate(spectra):
            label = f"seg {i}" if self.n_segments <= 12 else None
            ax.step(spe.energies, spe.counts, where="mid",
                    color=colors[i], alpha=0.75, lw=1.2, label=label)

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Counts")
        ax.set_yscale(yscale)
        ax.set_title(title)
        if xrange is not None:
            ax.set_xlim(xrange)
        if self.n_segments <= 12:
            ax.legend(fontsize=8, ncol=2)

        fig = fig or ax.figure
        sm = plt.cm.ScalarMappable(
            cmap="viridis", norm=plt.Normalize(0, self.n_segments - 1)
        )
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="Segment index  (time →)")
        plt.tight_layout()
        return fig

    def plot_overlay(
        self,
        xrange: list[float] | None = None,
        yscale: str = "log",
        ax: plt.Axes | None = None,
        figsize: tuple[float, float] = (14, 6),
    ) -> plt.Figure:
        """
        Overlay all N raw spectra, coloured by time order (viridis).

        Parameters
        ----------
        xrange : [xmin, xmax] or None
            X-axis limits in raw channel units.
        yscale : "log" or "linear"
        ax : existing Axes to draw on, or None to create a new figure.
        figsize : figure size when a new figure is created.
        """
        return self._overlay(
            self._spectra,
            title=f"Raw energy overlay  —  {self.n_segments} segments",
            xlabel="Raw channel (energy_orig)",
            xrange=xrange, yscale=yscale, ax=ax, figsize=figsize,
        )

    def plot_aligned(
        self,
        xrange: list[float] | None = None,
        yscale: str = "log",
        ax: plt.Axes | None = None,
        figsize: tuple[float, float] = (14, 6),
    ) -> plt.Figure:
        """
        Overlay the gain-corrected spectra after :meth:`align`. Each segment has
        been slid onto the reference, so the bands should now sit on top of one
        another instead of drifting with time.

        Raises
        ------
        RuntimeError
            If :meth:`align` has not been called yet.
        """
        if self._aligned_spectra is None:
            raise RuntimeError("Call align() before plot_aligned().")
        return self._overlay(
            self._aligned_spectra,
            title=f"Gain-aligned overlay  —  {self.n_segments} segments",
            xlabel="Aligned channel (energy_cal)",
            xrange=xrange, yscale=yscale, ax=ax, figsize=figsize,
        )


def create_directory(directory):
    """
    Ensure that the directory exists; if it does not, create it.

    Parameters
    ----------
    directory : str
        Directory path to check/create.

    Returns
    -------
    None.

    """
    os.makedirs(directory, exist_ok=True)


def data_cleanup(runs_dict):
    # Initial filters
    xrange_labr = [-0.5, 0.5]
    yrange_labr = [-0.62, 0.655]
    trange_labr = [-20, 60]

    xrange_cebr = [-0.518, 0.526]
    yrange_cebr = [-0.685, 0.655]
    trange_cebr = [-20, 60]

    date = runs_dict["date"]
    runnr = runs_dict["run"]
    ch = runs_dict["channel"]
    dfs = read_parquet_api.read_parquet_file(date=date, runnr=runnr, ch=ch)

    dfs["dt"] = dfs["dt"] + runs_dict["dt"]
    if ch == 5:
        dfs["energy_orig"] = dfs["energy_orig"] + 5435.24 - runs_dict["erg846"]
        dfxy = dftxy(
            df=dfs,
            xrange=xrange_labr,
            yrange=yrange_labr,
            trange=trange_labr,
            xkey="X2",
            ykey="Y2",
            tkey="dt",
        )
    elif ch == 4:
        dfs["energy_orig"] = dfs["energy_orig"] + 6565.2 - runs_dict["erg846"]
        dfxy = dftxy(
            df=dfs,
            xrange=xrange_cebr,
            yrange=yrange_cebr,
            trange=trange_cebr,
            xkey="X2",
            ykey="Y2",
            tkey="dt",
        )
    else:
        raise ValueError(f"Channel {ch} not supported; expected 4 or 5.")

    df_final = dfxy[["dt", "energy", "energy_orig", "LaBr[y/n]", "X2", "Y2"]]
    df_final["dt"] *= 1e-9
    return df_final


## plotting functions
def plot_dt(dt, tbins, trange, ax=None, **kwargs):
    """plot histogram of time differences"""
    if ax is None:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot()
    ax.hist(
        dt,
        bins=tbins,
        range=trange,
        edgecolor="black",
        **kwargs,
    )
    ax.set_xlabel("Time [ns]")


def plot_dz(z, zbins, zrange):
    """plot histogram of Z"""
    plt.figure()
    plt.hist(
        z, bins=zbins, range=zrange, alpha=0.7, edgecolor="black", label="Z histogram"
    )
    plt.xlabel("Z [cm]")
    plt.grid()
    plt.legend()


def plot_2D_alphas(
    df,
    xkey="X",
    ykey="Y",
    colormap="plasma",
    hexbins=80,
    xyplane=(-0.9, 0.9, -0.9, 0.9),
    ax=None,
    **kwargs,
):
    if ax is None:
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot()
    df.plot.hexbin(
        x=xkey,
        y=ykey,
        gridsize=hexbins,
        cmap=colormap,
        ax=ax,
        colorbar=False,
        extent=xyplane,
        **kwargs,
    )  # , bins="log")


def plot_2Dposition(X, Y, pbins, Vmax=None):
    """Plot 2D intensity map"""
    result, xedges, yedges = np.histogram2d(X, Y, pbins)
    plt.figure()
    if Vmax is not None:
        plt.imshow(
            result,
            extent=[-1, 1, -1, 1],
            interpolation="bilinear",
            cmap="inferno",
            vmax=Vmax,
            origin="lower",
        )
    else:
        plt.imshow(result, interpolation="bilinear", cmap="inferno", origin="lower")
    plt.title("Intensity Map")
    plt.show()

def plot_intensity_map2D(H, xrange, yrange, interp="bilinear",
                         title="Intensity Map", colormap="plasma", vmax=None):
    """Plot 2D intensity map"""
    if vmax is None:
        vmax = H.max()
    xmin = xrange[0]
    xmax = xrange[1]
    ymin = yrange[0]
    ymax = yrange[1]
    plt.figure(figsize=(12, 12))
    plt.imshow(
        H,
        extent=[xmin, xmax, ymin, ymax],
        interpolation=interp,
        cmap=colormap,
        vmax=vmax,
        origin="lower",
    )
    
    plt.title(title)
    plt.show()


def plot_2Dposition_hexbins(df, xkey, ykey, ax, colormap="Grays"):
    if ax is None:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot()
    df.plot.hexbin(x=xkey, y=ykey, gridsize=80, cmap=colormap, colorbar=False, ax=ax)


def plot_energy(en, ebins, erange=None, ax=None, log=True, **kwargs):
    """Plot energy histogram"""
    if ax is None:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot()
    if erange is None:
        erange = [en.min(), en.max()]
    ax.hist(en, bins=ebins, range=erange, **kwargs)
    if log:
        ax.set_yscale("log")
    else:
        ax.set_yscale("linear")
    ax.set_title("Energy")


def plot_3D(X, Y, Z, Vmax=None):
    """Plot 3D cloud using Mayavi"""
    from mayavi import mlab

    mlab.figure(1, fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
    r, ed = np.histogramdd((X, Y, Z), bins=15)
    mlab.figure(1, fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
    if Vmax is not None:
        mlab.pipeline.volume(mlab.pipeline.scalar_field(r), vmin=0, vmax=Vmax)
    else:
        mlab.pipeline.volume(mlab.pipeline.scalar_field(r))
    # mlab.axes()
    mlab.outline()
    # mlab.orientation_axes()


def plot_scatter_density(df):
    """Viridis-like" colormap with white background"""
    white_viridis = LinearSegmentedColormap.from_list(
        "white_viridis",
        [
            (0, "#ffffff"),
            (1e-20, "#440053"),
            (0.2, "#404388"),
            (0.4, "#2a788e"),
            (0.6, "#21a784"),
            (0.8, "#78d151"),
            (1, "#fde624"),
        ],
        N=256,
    )

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="scatter_density")
    density = ax.scatter_density(df.X2, df.Y2, cmap=white_viridis)
    fig.colorbar(density, label="Number of points per pixel")
    plt.show()


def _alpha_xy_cm(df, xkey="X2", ykey="Y2", a_side=4.8):
    """Map the dimensionless corner ratios in ``xkey``/``ykey`` to centimetres on
    the YAP face. The populated edges of the alpha image (histogram region above
    ~1/3 of the peak) are mapped linearly onto [-a/2, +a/2] cm, assuming the
    image fills the ``a_side`` cm active area. Returns (xa, ya) arrays in cm."""
    X = df[xkey].to_numpy()
    Y = df[ykey].to_numpy()
    res, xed, yed = np.histogram2d(Y, X, bins=1000)
    resx = res.sum(axis=0)
    maskx = resx > resx.max() / 3
    minx, maxx = xed[0:-1][maskx][0], xed[0:-1][maskx][-1]
    resy = res.sum(axis=1)
    masky = resy > resy.max() / 3
    miny, maxy = yed[0:-1][masky][0], yed[0:-1][masky][-1]

    mx = a_side / (maxx - minx)
    my = a_side / (maxy - miny)
    xa = mx * (X - maxx) + a_side / 2.0
    ya = my * (Y - maxy) + a_side / 2.0
    return xa, ya


def api_xyz(
    df,
    det_pos=(-18.0, 0.0, -25.0),
    z_t=6.7,
    beam_axis=(0.0, 0.0, 1.0),
    ion_energy_kev=50.0,
    toffset=None,
    dt_col="dt",
    dt_unit="s",
    use_det=True,
    relativistic=True,
    use_com=True,
    xkey="X2",
    ykey="Y2",
):
    r"""Full-model reconstruction of the neutron-interaction point (X, Y, Z).

    This is the authoritative reconstruction documented in
    ``docs/API_position_reconstruction.pdf`` (the ``calcXYZ`` level of rigour):
    relativistic alpha/neutron speeds, the center-of-mass correction (the alpha
    and neutron are back-to-back only in the reaction CM frame), and the exact
    gamma-detector vector. For the legacy non-relativistic, point-source model
    use :func:`api_xyz_simple`.

    Geometry / convention
    ---------------------
    The origin is the neutron production point (source). ``+z`` points up, so
    **-z points down**, toward the sample. The YAP alpha detector sits above the
    source at ``z = +z_t``; the alpha hit is ``(x0, y0, +z_t)``. The neutron
    travels downward and the interaction lands at negative ``z``. ``det_pos`` is
    the gamma-detector centre in this frame.

    Only ``xkey``/``ykey`` (the dimensionless corner ratios, default ``X2``/
    ``Y2`` -- built with :func:`calc_own_pos` if missing) and ``dt_col`` are
    used; any precomputed ``X``/``Y``/``Z`` columns are ignored.

    Parameters
    ----------
    df : pandas.DataFrame
        Must provide ``xkey``/``ykey`` (or ``A,B,C,D`` to build them) and
        ``dt_col``.
    det_pos : (x, y, z)
        Gamma-detector centre in **cm**, measured from the source. Default
        ``(-18, 0, -25)``.
    z_t : float
        Source-to-YAP-face distance in **cm** (default 6.7).
    beam_axis : (x, y, z)
        Ion-beam unit vector (direction of the CM drift). Magnitude is
        normalised; the correction is small (~0.3%).
    ion_energy_kev : float
        Deuteron beam energy in keV, used for the CM speed.
    toffset : float or None
        Electronic/z-align timing offset subtracted from ``dt`` (same unit as
        ``dt_unit``) before reconstruction. Calibrates the absolute depth.
    dt_col : str
        Column holding the measured alpha-gamma time difference (the alpha
        flight time is added internally).
    dt_unit : {"s", "ns"}
        Unit of ``dt_col`` (and of ``toffset``).
    use_det : bool
        If False, ignore the detector and use the straight-ray estimate
        ``|n| = v_n * dt_a``.
    relativistic, use_com : bool
        Toggle the relativistic speeds and the CM correction (for comparison).
    xkey, ykey : str
        Alpha-position columns.

    Returns
    -------
    (X, Y, Z) : tuple of numpy.ndarray
        Interaction coordinates in **cm**.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas dataframe")
    if xkey not in df.columns or ykey not in df.columns:
        df = calc_own_pos(df, xkey_new=xkey, ykey_new=ykey)

    c = SC.c
    va = _speed(_E_ALPHA, _M_ALPHA, relativistic)
    vn = _speed(_E_NEUTRON, _M_NEUTRON, relativistic)
    vcom = v_com_from_beam(ion_energy_kev) if use_com else 0.0
    chat = np.asarray(beam_axis, dtype=float)
    chat = chat / np.linalg.norm(chat)
    vcv = vcom * chat  # CM velocity vector [m/s]

    # alpha hit on the YAP face, in metres (z = +z_t, above the source)
    xa_cm, ya_cm = _alpha_xy_cm(df, xkey, ykey)
    ax = xa_cm / 100.0
    ay = ya_cm / 100.0
    az = z_t / 100.0  # scalar

    # alpha flight time: |A - v_com t| = v_alpha t  -> quadratic, positive root
    qa = va * va - vcom * vcom
    qb = 2.0 * (ax * vcv[0] + ay * vcv[1] + az * vcv[2])
    qc = -(ax * ax + ay * ay + az * az)
    t_alpha = (-qb + np.sqrt(qb * qb - 4.0 * qa * qc)) / (2.0 * qa)

    # neutron lab velocity vector: -(alpha CM velocity)/v_a * v_n + v_com
    wx = (ax - vcv[0] * t_alpha) / t_alpha
    wy = (ay - vcv[1] * t_alpha) / t_alpha
    wz = (az - vcv[2] * t_alpha) / t_alpha
    vnx = -wx / va * vn + vcv[0]
    vny = -wy / va * vn + vcv[1]
    vnz = -wz / va * vn + vcv[2]

    # measured time difference -> dt_a = dt + t_alpha
    dt = df[dt_col].to_numpy(dtype=float)
    if dt_unit == "ns":
        dt = dt * 1e-9
        toff = (toffset or 0.0) * 1e-9
    elif dt_unit == "s":
        toff = (toffset or 0.0)
    else:
        raise ValueError("dt_unit must be 's' or 'ns'")
    dta = dt - toff + t_alpha

    if use_det:
        dx, dy, dz = np.asarray(det_pos, dtype=float) / 100.0  # m
        A2 = c * c - (vnx * vnx + vny * vny + vnz * vnz)
        B2 = -(2.0 * c * c * dta - 2.0 * (dx * vnx + dy * vny + dz * vnz))
        C2 = c * c * dta * dta - (dx * dx + dy * dy + dz * dz)
        t_n = (-B2 - np.sqrt(B2 * B2 - 4.0 * A2 * C2)) / (2.0 * A2)
    else:
        # straight ray: |n| = v_n * dt_a  ->  |v_n_vec| * t_n = v_n * dt_a
        vmag = np.sqrt(vnx * vnx + vny * vny + vnz * vnz)
        t_n = dta * vn / vmag

    X = vnx * t_n * 100.0
    Y = vny * t_n * 100.0
    Z = vnz * t_n * 100.0
    return X, Y, Z


def api_xyz_simple(df, det_pos=[0, 22.2, -25.515], toffset=None, use_det=True):
    """Returns X, Y, Z reconstructed positions. Use the raw dataframe.
    if use_det=False, the location of the gamma detector is ignored.

    Legacy, non-relativistic, point-source model (origin at the YAP centre,
    sample at +z). Kept for reference; prefer :func:`api_xyz` for the full
    center-of-mass-corrected, relativistic reconstruction."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas dataframe")

    if toffset is not None:
        df = df.copy()
        df["dt"] = df["dt"] - toffset
    # convert alpha detector hits to cm
    res, xed, yed = np.histogram2d(df.Y2, df.X2, bins=1000)
    resx = res.sum(axis=0)
    maskx = resx > resx.max() / 3
    minx = xed[0:-1][maskx][0]
    maxx = xed[0:-1][maskx][-1]

    resy = res.sum(axis=1)
    masky = resy > resy.max() / 3
    miny = yed[0:-1][masky][0]
    maxy = yed[0:-1][masky][-1]

    mx = (2.4 + 2.4) / (maxx - minx)  # assume 4.8 cm active detector area
    xa = mx * (df["X2"] - maxx) + 2.4
    my = (2.4 + 2.4) / (maxy - miny)  # assume 4.8 cm active detector area
    ya = my * (df["Y2"] - maxy) + 2.4
    za = 6.7 * np.ones(len(xa))  # distance YAP-target (cm)

    # gamma detector location [cm]
    dx = det_pos[0]
    dy = det_pos[1]
    dz = det_pos[2]
    dg = np.sqrt(dx**2 + dy**2 + dz**2)  # cm
    c = 3e10  # speed of light [cm/s]
    # normal directional vectors
    alength = np.sqrt(xa**2 + ya**2 + za**2)  # a
    ux = -xa / alength
    uy = -ya / alength
    uz = za / alength
    # nTOF
    Ea = 3.5  # MeV
    ma = 3727.379  # MeV/c2
    va = np.sqrt(2 * Ea / ma) * 3e10  # cm/s
    ta = alength / va  # alpha travel time [s]
    En = 14.1  # MeV
    mn = 939.56563  # MeV/c2
    vn = np.sqrt(2 * En / mn) * 3e10  # cm/s
    ntof = df.dt * 1e-9 + ta  # neutron time of flight [s], dta

    if use_det:
        theta = np.arccos((ux * dx + uy * dy + uz * dz) / dg)  # radians
        aa = 1 - c**2 / vn**2
        bb = 2 * ntof * c**2 / vn - 2 * dg * np.cos(theta)
        cc = dg**2 - c**2 * ntof**2

        Ln = (-bb - np.sqrt(bb**2 - 4 * aa * cc)) / (2 * aa)  # cm

        X2 = -xa / alength * Ln
        Y2 = -ya / alength * Ln
        Z2 = za / alength * Ln + za
    else:
        # don't take into account the position of the gamma detector
        Ln = vn * ntof  # neutron travel path (cm)
        # coordinates of interaction
        X2 = Ln * ux  # x-coordinate of interaction (cm)
        Y2 = Ln * uy  # y-coordinate of interaction (cm)
        Z2 = Ln * uz  # z-coordinate of interaction (cm)

    return X2, Y2, Z2
