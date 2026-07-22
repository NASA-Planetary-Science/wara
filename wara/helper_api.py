"""
Helper functions
"""

import numpy as np
import dateparser
from wara.read_parquet_api import get_data_path
from wara import read_parquet_api
from wara import apicalc as api
from wara import list_mode_data_reader
import os
import re
import warnings
import json


def find_data_path(date, runnr):
    RUNNR = runnr
    DATE = dateparser.parse(date)

    date_dir = f"{DATE.year}-{DATE.month:02d}-{DATE.day:02d}"
    fname = f"RUN-{DATE.year}-{DATE.month:02d}-{DATE.day:02d}-{RUNNR:05d}"
    for data_path in get_data_path():
        file_path = data_path / date_dir / fname
        if file_path.is_dir():
            return file_path
    raise FileNotFoundError(f"Cannot find run {fname} in any path listed in data-path.txt")


def read_json(file):
    with open(file, "r") as f:
        data = json.load(f)  # Parse the JSON file, store as a list
    return data[0]  # a dictionary


# Read .npy MCA data, join if more than 1 file
def read_mca(date, runnr):
    file_path = find_data_path(date, runnr)
    # load data
    files = list(file_path.glob("MCA-data/*.npy"))
    if len(files) > 1:
        data = np.load(files[0])
        # data = 0
        # for f in files:
        #     data0 = np.load(f)
        #     data = data + data0
    else:
        data = np.load(files[0])

    return data


def read_mca_stats(date, runnr):
    """Per-channel MCA statistics for a run.

    Reads the run's MCA stats JSON (``MCA-data/*-stats-*``). When a run dumped
    several snapshots, the latest (last sorted by name, i.e. timestamp) is used.

    Returns
    -------
    dict
        The stats dictionary with per-channel lists (``real_time``,
        ``live_time``, ``input_counts``, ``input_count_rate``,
        ``output_counts``, ``output_count_rate``) plus the ``module`` index.
    str
        The name of the stats file that was read.
    """
    file_path = find_data_path(date, runnr)
    files = sorted(file_path.glob("MCA-data/*-stats-*"))
    if len(files) == 0:
        raise FileNotFoundError(
            f"No MCA stats file (*-stats-*) found in {file_path / 'MCA-data'}")
    latest = files[-1]
    return read_json(latest), latest.name


# CFD state options accepted by ``read_trace_data``. Each acquisition writes a
# companion ``*-traces-CFD_<state>-<timestamp>.pdf`` file recording whether CFD
# was turned "off" or left "unchanged" (i.e. enabled). The binary trace files
# share that timestamp, which lets us tell CFD and non-CFD traces apart.
_CFD_ALIASES = {
    "off": "off",
    False: "off",
    "unchanged": "unchanged",
    "on": "unchanged",
    True: "unchanged",
}

# Timestamp embedded in trace/binary/pdf file names, e.g. ``2022-06-06-17-12-20``
_TRACE_TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})")
_TRACE_CFD_PDF_RE = re.compile(
    r"traces-CFD_([A-Za-z]+)-(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})")


def _cfd_state_by_timestamp(file_path):
    """Map acquisition timestamp -> CFD state from the ``traces-CFD_*.pdf`` files.

    Returns a dict such as ``{"2022-06-06-17-12-20": "unchanged", ...}``.
    """
    mapping = {}
    for pdf in file_path.glob("trace-data/*traces-CFD_*.pdf"):
        m = _TRACE_CFD_PDF_RE.search(pdf.name)
        if m:
            mapping[m.group(2)] = m.group(1).lower()
    return mapping


def _cfd_state_of_file(name, ts_map):
    """CFD state ("off"/"unchanged") of a trace ``.bin`` file, or ``None``.

    Two on-disk conventions are supported:

    * Newer runs encode the state in the binary file name itself, e.g.
      ``...-binary-<ts>-with-cfd-mod0.bin`` / ``...-without-cfd-mod0.bin``.
    * Older runs write a companion ``*-traces-CFD_<state>-<ts>.pdf`` sharing the
      binary file's timestamp; *ts_map* maps that timestamp to the state.
    """
    low = name.lower()
    if "without-cfd" in low or "without_cfd" in low:
        return "off"
    if "with-cfd" in low or "with_cfd" in low:
        return "unchanged"
    m = _TRACE_TS_RE.search(name)
    if m:
        return ts_map.get(m.group(1))
    return None


def _filter_trace_files_by_cfd(file_path, files, cfd):
    """Keep only the trace ``.bin`` files whose acquisition matches *cfd*.

    *cfd* may be ``"off"``/``False`` (CFD disabled), ``"unchanged"``/``"on"``/
    ``True`` (CFD enabled). The acquisition's CFD state is read from the binary
    file name when it encodes it (``with-cfd`` / ``without-cfd``), otherwise
    from the companion ``*-traces-CFD_<state>-<timestamp>.pdf`` file that shares
    the binary file's timestamp.
    """
    key = cfd.lower() if isinstance(cfd, str) else cfd
    want = _CFD_ALIASES.get(key)
    if want is None:
        raise ValueError(
            f"Unknown cfd option {cfd!r}; expected one of "
            "'off', 'unchanged'/'on', True, False, or None")

    ts_map = _cfd_state_by_timestamp(file_path)
    return [f for f in files if _cfd_state_of_file(f.name, ts_map) == want]


# ── trace time alignment (for pulse-shape discrimination) ────────────────────
#
# For PSD every trace must sit on a common time axis, but the recorded pulses
# jitter in time: the 48-bit timestamp is latched on the (coarse) clock tick, so
# the fast-trigger point lands anywhere within a tick -- on a 500 MHz module that
# is up to 10 ns = 5 samples of jitter. The hardware CFD *fractional* time only
# corrects the sub-sample part and, on its own, does not remove this multi-sample
# jitter (verified on 2026-07-21 RUN19). What works, with or without CFD, is to
# derive a per-trace reference time from the trace shape itself and shift every
# trace so that reference lands on the same sample index. See
# ``examples/pixie/example_trace_alignment_pixie.py``.


def _trace_matrix(df):
    """Stack the trace column into a 2-D float array.

    Returns ``(T, idx)`` where ``T`` has shape ``(n, L)`` for the ``L`` = modal
    trace length and ``idx`` are the DataFrame positional indices of those rows.
    Rows whose trace has a different length (or none) are left out so callers can
    return them unchanged.
    """
    lengths = df["trace"].map(lambda t: len(t) if hasattr(t, "__len__") else 0)
    if len(lengths) == 0 or lengths.max() == 0:
        return np.empty((0, 0)), np.empty(0, dtype=int)
    modal = int(lengths.mode().iloc[0])
    idx = np.flatnonzero((lengths == modal).to_numpy())
    T = np.stack([np.asarray(t, dtype=float) for t in df["trace"].iloc[idx]])
    return T, idx


def _rising_edge_ref(T, fraction):
    """Sub-sample index of the ``fraction``-of-amplitude crossing on the rising
    edge of each (baseline-subtracted) trace. NaN where no crossing is found."""
    thr = fraction * T.max(axis=1)
    above = T > thr[:, None]
    k = np.argmax(above, axis=1)  # first sample above threshold
    n = T.shape[0]
    rows = np.arange(n)
    y1 = T[rows, k]
    y0 = T[rows, np.clip(k - 1, 0, None)]
    denom = y1 - y0
    frac = np.where(denom != 0, (thr - y0) / denom, 0.0)
    ref = (k - 1) + frac
    ref[k == 0] = np.nan  # threshold already exceeded at sample 0 -> unusable
    return ref


def _read_run_settings(date, runnr):
    """Load a run's settings JSON as a dict (the state the run actually used)."""
    file_path = find_data_path(date, runnr)
    files = (sorted(file_path.glob("trace-data/*-settings-*.json"))
             or sorted(file_path.glob("settings/*-settings-*.json")))
    if not files:
        raise FileNotFoundError(
            f"No settings JSON found for run in {file_path}")
    # prefer a non-"initial" settings file (the state the run actually used)
    chosen = next((f for f in files if "initial" not in f.name), files[0])
    with open(chosen) as f:
        data = json.load(f)
    return data[0] if isinstance(data, list) else data


def read_sample_interval_ns(date, runnr, ch=0):
    """Trace sample interval in nanoseconds (``1000 / adc_msps``).

    For a 500 MHz module this is 2 ns, for a 250 MHz module 4 ns. Read from the
    run's settings metadata so it stays correct across module types.
    """
    s = _read_run_settings(date, runnr)
    config = s["metadata"]["config"]
    cfg = config[ch] if ch < len(config) else config[0]
    return 1000.0 / cfg["adc_msps"]


def read_slow_filter_geometry(date, runnr, ch):
    """Energy (slow) filter window lengths for channel *ch*, in ADC samples.

    Reads the run's settings JSON and converts the filter settings to trace
    (ADC-sample) units::

        L = SlowLength * 2**SlowFilterRange * adc_clk_div   # leading/trailing
        G = SlowGap    * 2**SlowFilterRange * adc_clk_div   # gap (flat top)

    where ``adc_clk_div = adc_msps / fpga_clk_mhz`` is the number of ADC samples
    per filter clock tick (5 for a 500 MHz module, 2 for a 250 MHz one). The
    energy sums stored per event (``Esum_trailing``, ``Esum_gap``,
    ``Esum_leading``) are the raw sums of the trace over three consecutive
    windows of lengths ``L``, ``G`` and ``L``.

    Returns
    -------
    (int, int)
        ``(L, G)`` in ADC samples.
    """
    s = _read_run_settings(date, runnr)
    R = s["module"]["input"]["SlowFilterRange"]
    slow_length = s["channel"]["input"]["SlowLength"][ch]
    slow_gap = s["channel"]["input"]["SlowGap"][ch]
    div = s["metadata"]["config"][ch]["adc_clk_div"]
    L = int(slow_length * 2 ** R * div)
    G = int(slow_gap * 2 ** R * div)
    return L, G


def find_LGL_sums(trace, esum_trailing, esum_gap, esum_leading, L, G,
                  relative=True, baseline_samples=30):
    """Locate the energy-filter sum windows on a trace by matching stored sums.

    The Pixie energy (slow) filter latches three raw running sums over three
    consecutive windows placed in time order ``[trailing: L][gap: G][leading: L]``
    (see Fig. 4-1 of the Pixie-16 manual). This function sweeps that window
    layout across *trace* and returns the position where the three window sums
    best reproduce the stored ``(esum_trailing, esum_gap, esum_leading)`` values.

    This is the energy-filter counterpart of :func:`_fast_filter_ref` (which
    reconstructs the *fast* trigger): the two should sit at a fixed offset from
    each other, so matching the stored sums is a cross-check of both.

    Parameters
    ----------
    trace : array_like
        A single recorded trace (ADC samples).
    esum_trailing, esum_gap, esum_leading : float
        The stored energy sums for that event.
    L, G : int
        Leading/trailing window length and gap (flat-top) length, in ADC
        samples. Get them from :func:`read_slow_filter_geometry`.
    relative : bool, optional
        If ``True`` (default) match the *shape* of the sums (baseline removed and
        scale-normalised), which is robust even if the absolute scale differs. If
        ``False`` match absolute values (they coincide exactly when *L* and *G*
        are correct).
    baseline_samples : int, optional
        Number of leading samples averaged for the baseline (used by
        ``relative``).

    Returns
    -------
    dict
        ``start`` (trailing-window start index), ``bounds`` = the four window
        boundaries ``(start, start+L, start+L+G, start+2L+G)`` delimiting the
        trailing/gap/leading (L, G, L) regions, ``sums`` = the matched
        ``(trailing, gap, leading)`` window sums, and ``error`` of the match.
    """
    tr = np.asarray(trace, dtype=float)
    n = tr.shape[0]
    width = 2 * L + G
    if n < width:
        raise ValueError(
            f"Trace of length {n} is shorter than the sum window 2L+G={width}")

    stored = np.array([esum_trailing, esum_gap, esum_leading], dtype=float)
    if relative:
        bl = tr[:baseline_samples].mean()
        tr_ref = tr - bl
        stored = stored - np.array([L, G, L]) * bl  # drop baseline contribution
        stored = stored / (np.linalg.norm(stored) or 1.0)
    else:
        tr_ref = tr

    cs = np.concatenate([[0.0], np.cumsum(tr_ref)])
    p = np.arange(0, n - width + 1)  # all candidate trailing-window starts
    t = cs[p + L] - cs[p]
    g = cs[p + L + G] - cs[p + L]
    lead = cs[p + 2 * L + G] - cs[p + L + G]
    v = np.stack([t, g, lead], axis=1)  # (n_pos, 3)
    if relative:
        norm = np.linalg.norm(v, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        err = np.linalg.norm(v / norm - stored, axis=1)
    else:
        err = np.abs(v - stored).sum(axis=1)
    best = int(np.argmin(err))
    p0 = int(p[best])
    return {
        "start": p0,
        "bounds": (p0, p0 + L, p0 + L + G, p0 + 2 * L + G),
        "sums": (float(t[best]), float(g[best]), float(lead[best])),
        "error": float(err[best]),
    }


def _fast_filter_ref(T, rise, gap, threshold):
    """Trigger position from a trapezoidal *fast filter* (leading - trailing
    running sums), i.e. the offline analogue of the hardware fast trigger that
    latches the leading/gap/trailing energy sums. Threshold crossing, sub-sample
    interpolated. NaN where the filter never crosses."""
    cs = np.cumsum(T, axis=1)
    cs = np.concatenate([np.zeros((T.shape[0], 1)), cs], axis=1)  # prepend 0
    n = T.shape[1]
    ff = np.full_like(T, np.nan)
    for i in range(2 * rise + gap, n):
        lead = cs[:, i + 1] - cs[:, i + 1 - rise]
        trail = cs[:, i + 1 - rise - gap] - cs[:, i + 1 - 2 * rise - gap]
        ff[:, i] = lead - trail
    peak = np.nanmax(ff, axis=1)
    thr = threshold * peak
    above = ff > thr[:, None]
    ref = np.full(T.shape[0], np.nan)
    for r in range(T.shape[0]):
        k = np.argmax(above[r])
        if k > 0:
            y0, y1 = ff[r, k - 1], ff[r, k]
            ref[r] = (k - 1) + ((thr[r] - y0) / (y1 - y0) if y1 != y0 else 0.0)
    return ref


def _peak_ref(T):
    """Sub-sample peak position of each trace via parabolic interpolation."""
    k = T.argmax(axis=1)
    n = T.shape[1]
    kc = np.clip(k, 1, n - 2)
    rows = np.arange(T.shape[0])
    ym1, y0, yp1 = T[rows, kc - 1], T[rows, kc], T[rows, kc + 1]
    denom = ym1 - 2 * y0 + yp1
    delta = np.where(denom != 0, 0.5 * (ym1 - yp1) / denom, 0.0)
    return kc + delta


def _shift_rows(T, shift):
    """Shift each row of *T* by ``shift[i]`` samples (positive = later in time)
    using linear interpolation; exposed ends are filled with the edge value."""
    n = T.shape[1]
    x = np.arange(n)
    src = x[None, :] - shift[:, None]  # source coordinate for each output sample
    i0 = np.floor(src).astype(int)
    frac = src - i0
    i0c = np.clip(i0, 0, n - 1)
    i1c = np.clip(i0 + 1, 0, n - 1)
    rows = np.arange(T.shape[0])[:, None]
    out = T[rows, i0c] * (1.0 - frac) + T[rows, i1c] * frac
    out = np.where(src < 0, T[:, :1], out)
    out = np.where(src > n - 1, T[:, -1:], out)
    return out


def align_traces(df, method="edge", fraction=0.5, ref=None,
                 rise=5, gap=2, threshold=0.2, inplace=False):
    """Time-align the traces in *df* onto a common sample grid (for PSD).

    A per-trace reference time is derived from the pulse shape and every trace is
    shifted (linear interpolation, baseline preserved) so that reference lands on
    the same sample index.

    Parameters
    ----------
    df : pandas.DataFrame
        Trace data as returned by :func:`read_trace_data`.
    method : {"edge", "fast", "peak"}, optional
        How to locate each pulse. ``"edge"`` (default) uses the
        ``fraction``-of-amplitude crossing on the rising edge -- simple and
        robust, works with and without CFD. ``"fast"`` uses a trapezoidal fast
        filter (the offline analogue of the hardware fast trigger that latches
        the leading/gap/trailing energy sums). ``"peak"`` aligns on the
        (parabolically interpolated) trace maximum.
    fraction : float, optional
        Amplitude fraction for the ``"edge"`` crossing (default 0.5).
    ref : int or None, optional
        Target sample index the reference is shifted to. ``None`` (default) uses
        the rounded median reference so the net shift is minimal; pass an int for
        a fixed grid that is reproducible across runs.
    rise, gap : int, optional
        Fast-filter rise time and flat-top (in samples) for ``method="fast"``.
    threshold : float, optional
        Fast-filter trigger threshold as a fraction of the filter peak.
    inplace : bool, optional
        Modify *df* in place instead of returning a copy.

    Returns
    -------
    pandas.DataFrame
        With the ``trace`` column replaced by aligned float traces and an added
        ``align_shift`` column (samples each trace was moved; NaN if the
        reference could not be found and the trace was left unshifted).
    """
    if not inplace:
        df = df.copy()
    df["align_shift"] = np.nan
    T, idx = _trace_matrix(df)
    if len(idx) == 0:
        warnings.warn("align_traces: no traces to align.", stacklevel=2)
        return df

    base = T - T[:, :30].mean(axis=1, keepdims=True)  # baseline-subtracted view
    if method == "edge":
        pos = _rising_edge_ref(base, fraction)
    elif method == "fast":
        pos = _fast_filter_ref(base, rise, gap, threshold)
    elif method == "peak":
        pos = _peak_ref(base)
    else:
        raise ValueError(
            f"Unknown align method {method!r}; expected 'edge', 'fast' or 'peak'")

    target = float(np.round(np.nanmedian(pos))) if ref is None else float(ref)
    shift = target - pos
    shift_safe = np.where(np.isfinite(shift), shift, 0.0)
    aligned = _shift_rows(T, shift_safe)

    trace_col = df["trace"].to_numpy(dtype=object)
    for j, row in enumerate(idx):
        trace_col[row] = aligned[j]
    df["trace"] = trace_col
    shift_col = df["align_shift"].to_numpy()
    shift_col[idx] = shift
    df["align_shift"] = shift_col
    return df


def read_trace_data(date, runnr, cfd=None, align=None, align_fraction=0.5,
                    align_ref=None):
    """Read trace (list-mode) binary data for a run.

    Parameters
    ----------
    date, runnr :
        Run identifiers (see :func:`find_data_path`).
    cfd : {None, "off", "unchanged", "on", True, False}, optional
        Select trace acquisitions by their CFD state. Each acquisition writes a
        companion ``*-traces-CFD_<state>-<timestamp>.pdf`` whose name records
        whether CFD was turned ``off`` or left ``unchanged`` (enabled); the
        binary trace files share that timestamp. When ``None`` (default) every
        trace file is read regardless of CFD state. ``"off"``/``False`` reads
        only the CFD-off (no-CFD) acquisitions; ``"unchanged"``/``"on"``/``True``
        reads only the CFD-enabled acquisitions.
    align : {None, "edge", "fast", "peak"}, optional
        Time-align the traces onto a common sample grid so they can be used for
        pulse-shape discrimination. ``None`` (default) leaves the raw traces
        untouched. See :func:`align_traces` for the methods. When set, an
        ``align_shift`` column is added and the traces become float arrays.
    align_fraction : float, optional
        Amplitude fraction for ``align="edge"`` (default 0.5).
    align_ref : int or None, optional
        Target sample index for the alignment reference (see
        :func:`align_traces`). ``None`` uses the median reference.
    """
    file_path = find_data_path(date, runnr)
    # load data
    files = list(file_path.glob("trace-data/*.bin"))
    if len(files) == 0:
        raise FileNotFoundError(
            f"No trace binary files (*.bin) found in {file_path / 'trace-data'}")
    if cfd is not None:
        files = _filter_trace_files_by_cfd(file_path, files, cfd)
        if len(files) == 0:
            raise FileNotFoundError(
                f"No trace binary files with CFD state '{cfd}' found in "
                f"{file_path / 'trace-data'}")
    df = list_mode_data_reader.read_list_mode_data(files)
    if align is not None:
        df = align_traces(df, method=align, fraction=align_fraction,
                          ref=align_ref, inplace=True)
    return df


def read_binary_data(date, runnr):
    file_path = find_data_path(date, runnr)
    # load data
    files = list(file_path.glob("binary-data/*.bin"))
    if len(files) == 0:
        raise FileNotFoundError(
            f"No binary files (*.bin) found in {file_path / 'binary-data'}")
    try:
        files_sorted = sorted(files, key=lambda x: int(x.name[-9:-4]))
    except Exception:
        files_sorted = files
        warnings.warn("Could not sort binary files by run index; "
                      "processing them in glob order.", stacklevel=2)
    df = list_mode_data_reader.read_list_mode_data(files_sorted)
    return df


def read_mca_time(date, runnr, ch, key="real"):  # total combined real or live time
    if key == "real":
        k = "real_time"
    elif key == "live":
        k = "live_time"
    else:
        k = "live_time"
    file_path = find_data_path(date, runnr)
    files = sorted(list(file_path.glob("MCA-data/*-stats-*")))
    time = 0
    for f in files:
        dic = read_json(f)
        time += dic[k][ch]
    return time


def read_mca_live_time(date, runnr, ch):  # total combined time
    file_path = find_data_path(date, runnr)
    files = sorted(list(file_path.glob("MCA-data/*-stats-*")))
    real_time = 0
    for f in files:
        dic = read_json(f)
        real_time += dic["live_time"]
    return real_time


def read_time_from_settings(settings_file, ch):
    with open(settings_file, mode="r") as myfile:
        lines = myfile.readlines()

    for i, line in enumerate(lines):
        tmp = line.replace('"', "").split()
        if "live_time:" in tmp:
            idx = i
    time = float(lines[idx + ch + 1].split(",")[0])
    return time


def read_input_CR_from_settings(settings_file, ch=9):
    with open(settings_file, mode="r") as myfile:
        lines = myfile.readlines()

    for i, line in enumerate(lines):
        tmp = line.replace('"', "").split()
        if "input_count_rate:" in tmp:
            idx = i
    CR = float(lines[idx + ch + 1].split(",")[0])
    return CR


def read_input_counts_from_settings(settings_file, ch=9):
    with open(settings_file, mode="r") as myfile:
        lines = myfile.readlines()

    for i, line in enumerate(lines):
        tmp = line.replace('"', "").split()
        if "input_counts:" in tmp:
            idx = i
    CR = float(lines[idx + ch + 1].split(",")[0])
    return CR


def get_total_time(date, runnr, ch, mca=False):
    file_path = find_data_path(date, runnr)
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


def get_total_counts(date, runnr, ch):
    file_path = find_data_path(date, runnr)
    # load data
    files = list(file_path.glob("settings/*-stats-*"))[1:]
    cts_tot = 0
    for f in files:
        cts0 = read_input_counts_from_settings(f, ch=ch)
        cts_tot += cts0
    return cts_tot


def calculate_neutron_flux(date, runnr, ch, L=30):
    # L = neutron source to sample distance in cm
    alpha_counts = get_total_counts(date, runnr, ch)
    time_total = get_total_time(date, runnr, ch)
    alpha_cr = alpha_counts / time_total
    d = 6.7  # cm alpha detector-neutron source distance
    alpha_area = 4.8 * 4.8  # cm2
    phi_a = alpha_cr / alpha_area  # flux at alpha detector
    Y0 = 4 * np.pi * d**2 * phi_a  # neutron yield (n/s)
    phi_s = Y0 / (4 * np.pi * L**2)  # neutron flux on sample
    alpha_frac = 0.91  # correction factor for true alphas
    return Y0 * alpha_frac, phi_s * alpha_frac


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
    return fa


def create_directory(directory):
    """
    Ensure that the directory exists; if it does not, create it.

    Parameters:
    directory (str): Directory path to check/create.
    """
    os.makedirs(directory, exist_ok=True)


def data_reduction(dates, run_numbers, new_date, new_run_number, ch):
    pass


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
        dfxy = api.dftxy(
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
        dfxy = api.dftxy(
            df=dfs,
            xrange=xrange_cebr,
            yrange=yrange_cebr,
            trange=trange_cebr,
            xkey="X2",
            ykey="Y2",
            tkey="dt",
        )

    df_final = dfxy[["dt", "energy", "energy_orig", "LaBr[y/n]", "X2", "Y2"]]
    df_final["dt"] *= 1e-9
    return df_final
