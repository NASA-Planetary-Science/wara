"""Pulse-shape-discrimination (PSD) analysis of digitized neutron/gamma traces.

Ported from the ``neutron_trace_analysis_pico.py`` scratch script into a small,
testable module the Neutrons GUI tab builds on. The physics follows Olcek et
al. (2026): each baseline-corrected pulse is integrated into a total charge
(the MCA / energy channel) and split into a prompt and a tail charge, whose
ratio gives the discrimination parameter ``PSD = 1 - Q_prompt / Q_tail``.

All functions assume *positive-going* pulses (call :func:`orient_positive`
first). Time is in nanoseconds and amplitude in volts, matching the PicoScope
``.npz`` acquisitions (keys ``time_ns`` and ``A_traces``).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "NeutronTraces",
    "orient_positive",
    "baseline_correct",
    "mca_integral",
    "psd_ratio",
    "auto_pre_trigger",
    "load_trace_file",
    "double_gaussian",
    "figure_of_merit",
    "FOMResult",
    "SIGMA_TO_FWHM",
]

#: ``FWHM = 2*sqrt(2*ln2) * sigma`` — the Gaussian width conversion factor.
SIGMA_TO_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))


def _synth_time(traces):
    """Synthesise a sample-index time axis (ns) for formats carrying none."""
    return np.arange(np.asarray(traces).shape[1], dtype=float)


def load_trace_file(path):
    """Load traces from *path* into ``(time_ns, traces, mean)``.

    Supported formats:

    * ``.npz`` with the PicoScope layout (``time_ns`` + ``A_traces``, optional
      ``A_mean``) — those arrays are used directly. Any other ``.npz`` falls
      back to its first 2-D array as the traces.
    * ``.npy`` / ``.txt`` / ``.csv`` / ``.dat`` — one trace per row. These carry
      no time axis, so it is synthesised as the sample index (in ns).

    ``mean`` is the acquisition's mean trace when present, else ``None``. Rows
    are individual pulses; a single-row file is treated as one trace.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        data = np.load(path)
        keys = list(data.keys())
        if "time_ns" in keys and "A_traces" in keys:
            mean = data["A_mean"] if "A_mean" in keys else None
            return np.asarray(data["time_ns"], dtype=float), data["A_traces"], mean
        # No PicoScope layout: use the first 2-D array in the archive.
        for k in keys:
            arr = np.asarray(data[k])
            if arr.ndim == 2:
                return _synth_time(arr), arr, None
        raise ValueError(
            f"No 2-D trace array found in '{os.path.basename(path)}' "
            f"(keys: {keys}); expected 'time_ns' + 'A_traces' or a 2-D array")
    if ext == ".npy":
        traces = np.load(path)
    elif ext in (".txt", ".csv", ".dat"):
        # CSV is comma-delimited; txt/dat default to whitespace. '#' comments
        # and blank lines are skipped by loadtxt.
        traces = np.loadtxt(path, delimiter="," if ext == ".csv" else None)
    else:
        raise ValueError(
            f"Unsupported trace file type '{ext}'; use .npz, .npy, .txt or .csv")
    traces = np.atleast_2d(np.asarray(traces))
    return _synth_time(traces), traces, None


def orient_positive(traces, reference=None):
    """Return *traces* flipped so pulses are positive-going, plus the sign used.

    Polarity is decided from the larger absolute excursion of a reference wave
    (the acquisition's mean trace when available, otherwise the traces' own
    column mean): if the dominant excursion is negative the traces are
    sign-flipped. Returns ``(oriented_traces, sign)`` where ``sign`` is +1 or
    -1 (``oriented = sign * traces``).
    """
    traces = np.asarray(traces)
    ref = reference if reference is not None else traces.mean(axis=0)
    ref = np.asarray(ref, dtype=float)
    sign = -1 if abs(ref.min()) > abs(ref.max()) else 1
    # Keep the (large) trace matrix in its native dtype to bound memory; the
    # sign flip is exact in float32.
    return (traces if sign == 1 else -traces), float(sign)


def auto_pre_trigger(time_ns, reference, frac=0.2, margin_ns=10.0):
    """Estimate the pre-trigger / gate-start time (ns) from a positive reference
    pulse. Finds where the reference first rises above *frac* of its peak and
    steps back *margin_ns* so the baseline region sits fully before the pulse.
    """
    time_ns = np.asarray(time_ns, dtype=float)
    ref = np.asarray(reference, dtype=float)
    peak = ref.max()
    if not np.isfinite(peak) or peak <= 0:
        return float(time_ns[0] + 0.1 * (time_ns[-1] - time_ns[0]))
    above = np.flatnonzero(ref >= frac * peak)
    rise_ns = time_ns[above[0]] if above.size else time_ns[0]
    start = rise_ns - margin_ns
    return float(np.clip(start, time_ns[0], time_ns[-1]))


def baseline_correct(traces, time_ns, pre_trigger_ns):
    """Subtract each trace's pre-trigger-region mean from itself.

    The baseline is the mean of every sample before *pre_trigger_ns*. If no
    sample falls in that region the traces are returned unchanged.
    """
    traces = np.asarray(traces)
    time_ns = np.asarray(time_ns, dtype=float)
    mask = time_ns < pre_trigger_ns
    if not mask.any():
        return traces
    # Accumulate the baseline in float64 for accuracy, then subtract in the
    # traces' native dtype so the (large) corrected matrix stays compact.
    baseline = traces[:, mask].mean(axis=1, keepdims=True, dtype=np.float64)
    return (traces - baseline.astype(traces.dtype))


def mca_integral(traces_corrected, time_ns, gate=None):
    """Integrate each baseline-corrected trace into a pulse area (MCA value).

    Assumes positive-going traces. When *gate* is ``(start_ns, end_ns)`` only
    that window is integrated; otherwise the full record is used. Returns an
    array of areas in V·ns.
    """
    traces_corrected = np.asarray(traces_corrected)
    time_ns = np.asarray(time_ns, dtype=float)
    dt = float(time_ns[1] - time_ns[0])
    if gate is not None:
        start, end = gate
        mask = (time_ns >= start) & (time_ns <= end)
        return traces_corrected[:, mask].sum(axis=1, dtype=np.float64) * dt
    return traces_corrected.sum(axis=1, dtype=np.float64) * dt


def psd_ratio(traces_corrected, time_ns, gate_start_ns, prompt_end_ns,
              tail_end_ns):
    """Pulse-shape-discrimination ratio ``PSD = 1 - Q_prompt / Q_tail``.

    ``Q_prompt`` integrates the fast part of the pulse from *gate_start_ns* to
    *prompt_end_ns*; ``Q_total`` runs from *gate_start_ns* to *tail_end_ns*, and
    ``Q_tail = Q_total - Q_prompt``. Traces must already be positive-going and
    baseline-corrected.

    Returns ``(psd, q_total)`` as arrays; ``psd`` is NaN where ``Q_tail <= 0``.
    """
    traces_corrected = np.asarray(traces_corrected)
    time_ns = np.asarray(time_ns, dtype=float)
    dt = float(time_ns[1] - time_ns[0])
    prompt_mask = (time_ns >= gate_start_ns) & (time_ns < prompt_end_ns)
    total_mask = (time_ns >= gate_start_ns) & (time_ns <= tail_end_ns)

    q_prompt = traces_corrected[:, prompt_mask].sum(axis=1, dtype=np.float64) * dt
    q_total = traces_corrected[:, total_mask].sum(axis=1, dtype=np.float64) * dt
    q_tail = q_total - q_prompt

    with np.errstate(divide="ignore", invalid="ignore"):
        psd = 1.0 - q_prompt / q_tail
    psd[q_tail <= 0] = np.nan
    return psd, q_total


# ── Figure of merit ────────────────────────────────────────────────────────────
def double_gaussian(x, a_g, mu_g, sigma_g, a_n, mu_n, sigma_n):
    """Sum of two Gaussians (gamma + neutron) evaluated at *x*.

    ``y = a_g*exp(-(x-mu_g)^2 / 2*sigma_g^2) + a_n*exp(-(x-mu_n)^2 / 2*sigma_n^2)``
    """
    x = np.asarray(x, dtype=float)
    return (a_g * np.exp(-((x - mu_g) ** 2) / (2.0 * sigma_g ** 2))
            + a_n * np.exp(-((x - mu_n) ** 2) / (2.0 * sigma_n ** 2)))


@dataclass
class FOMResult:
    """Double-Gaussian fit of a 1-D PSD projection and the resulting FOM.

    ``FOM = |mu_n - mu_gamma| / (FWHM_gamma + FWHM_n)``, the standard
    neutron/gamma separation quality metric; ``FOM >= 1.27`` is the usual
    threshold for clean discrimination. The two Gaussians are ordered by their
    means: the lower one is labelled gamma, the higher one neutron.
    """

    fom: float
    separation: float
    mu_gamma: float
    sigma_gamma: float
    amp_gamma: float
    mu_n: float
    sigma_n: float
    amp_n: float
    n_events: int
    r_squared: float
    counts: np.ndarray = field(repr=False)
    edges: np.ndarray = field(repr=False)

    @property
    def centers(self):
        """Histogram bin centres of the PSD projection."""
        return 0.5 * (self.edges[1:] + self.edges[:-1])

    @property
    def fwhm_gamma(self):
        return SIGMA_TO_FWHM * self.sigma_gamma

    @property
    def fwhm_n(self):
        return SIGMA_TO_FWHM * self.sigma_n

    @property
    def popt(self):
        """Fit parameters as ``(a_g, mu_g, sigma_g, a_n, mu_n, sigma_n)``."""
        return (self.amp_gamma, self.mu_gamma, self.sigma_gamma,
                self.amp_n, self.mu_n, self.sigma_n)

    def curve(self, x):
        """The fitted double Gaussian evaluated at *x*."""
        return double_gaussian(x, *self.popt)

    def component(self, x, which):
        """One fitted Gaussian, *which* being ``"gamma"`` or ``"neutron"``."""
        if which not in ("gamma", "neutron"):
            raise ValueError("which must be 'gamma' or 'neutron'")
        if which == "gamma":
            a, mu, s = self.amp_gamma, self.mu_gamma, self.sigma_gamma
        else:
            a, mu, s = self.amp_n, self.mu_n, self.sigma_n
        x = np.asarray(x, dtype=float)
        return a * np.exp(-((x - mu) ** 2) / (2.0 * s ** 2))


def _two_peak_guess(centers, counts):
    """Initial ``(a, mu, sigma)`` pairs for the two PSD peaks in a histogram.

    Uses the two most prominent maxima of a lightly smoothed histogram; falls
    back to splitting the distribution at its highest bin when only one peak is
    resolved (a poorly separated detector still has to produce a fit).
    """
    span = float(centers[-1] - centers[0])
    smooth = np.convolve(counts, np.ones(3) / 3.0, mode="same")
    idx = []
    try:
        from scipy.signal import find_peaks

        found, props = find_peaks(smooth, distance=max(2, len(centers) // 20),
                                  prominence=0.02 * smooth.max())
        order = np.argsort(props["prominences"])[::-1]
        idx = [int(found[i]) for i in order[:2]]
    except Exception:  # noqa: BLE001 — fall through to the split heuristic
        idx = []
    if len(idx) < 2:
        # One peak only: seed the second on the heaviest side lobe.
        top = int(np.argmax(smooth))
        left, right = smooth[:top], smooth[top + 1:]
        if right.sum() >= left.sum() and right.size:
            other = top + 1 + int(np.argmax(right))
        elif left.size:
            other = int(np.argmax(left))
        else:
            other = top
        idx = [top, other]
    idx = sorted(idx)
    guesses = []
    for i in idx:
        # Half-maximum width around the peak, floored so curve_fit can move.
        half = 0.5 * smooth[i]
        lo = i
        while lo > 0 and smooth[lo] > half:
            lo -= 1
        hi = i
        while hi < len(smooth) - 1 and smooth[hi] > half:
            hi += 1
        fwhm = max(centers[hi] - centers[lo], span / 50.0)
        guesses.append((float(smooth[i]), float(centers[i]),
                        float(fwhm / SIGMA_TO_FWHM)))
    return guesses


def figure_of_merit(psd_values, bins=80, psd_range=None):
    """Fit a double Gaussian to a 1-D PSD projection and return its FOM.

    *psd_values* are the PSD parameters of the pulses in one energy slice (the
    projection of a horizontal PSD-plot slice onto the PSD axis), which for a
    discriminating detector shows a gamma peak and a neutron peak. They are
    histogrammed into *bins* over *psd_range* (``(lo, hi)``, default the data
    range), fitted with :func:`double_gaussian`, and reduced to

    ``FOM = |mu_n - mu_gamma| / (FWHM_gamma + FWHM_n)``.

    Returns a :class:`FOMResult`. Raises ``ValueError`` when fewer than 10
    finite values are given and ``RuntimeError`` when the fit does not converge.
    """
    from scipy.optimize import curve_fit

    vals = np.asarray(psd_values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size < 10:
        raise ValueError(
            f"Need at least 10 finite PSD values to fit, got {vals.size}")
    if psd_range is None:
        psd_range = (float(vals.min()), float(vals.max()))
    lo, hi = float(psd_range[0]), float(psd_range[1])
    if hi <= lo:
        raise ValueError(f"Empty PSD range ({lo}, {hi})")

    counts, edges = np.histogram(vals, bins=int(bins), range=(lo, hi))
    centers = 0.5 * (edges[1:] + edges[:-1])
    counts = counts.astype(float)
    if counts.max() <= 0:
        raise ValueError("PSD projection is empty over the requested range")

    # Nothing narrower than a bin can be resolved: flooring sigma there keeps a
    # single-bin spike from collapsing the fit's gradient into a delta function.
    bin_w = float(edges[1] - edges[0])
    span = hi - lo
    (a1, m1, s1), (a2, m2, s2) = _two_peak_guess(centers, counts)
    s1, s2 = max(s1, bin_w), max(s2, bin_w)
    p0 = [a1, m1, s1, a2, m2, s2]
    bounds = ([0.0, lo, 0.5 * bin_w] * 2, [10.0 * counts.max(), hi, span] * 2)
    try:
        popt, _pcov = curve_fit(double_gaussian, centers, counts, p0=p0,
                                bounds=bounds, maxfev=20000)
    except Exception as exc:  # noqa: BLE001 — re-raise with a usable message
        raise RuntimeError(f"Double-Gaussian fit did not converge: {exc}") from exc

    # Order the two components by mean: lower = gamma, higher = neutron.
    comp = [tuple(popt[:3]), tuple(popt[3:])]
    comp.sort(key=lambda c: c[1])
    (a_g, mu_g, s_g), (a_n, mu_n, s_n) = comp
    s_g, s_n = abs(s_g), abs(s_n)

    fwhm_sum = SIGMA_TO_FWHM * (s_g + s_n)
    separation = abs(mu_n - mu_g)
    fom = separation / fwhm_sum if fwhm_sum > 0 else float("nan")

    resid = counts - double_gaussian(centers, a_g, mu_g, s_g, a_n, mu_n, s_n)
    ss_tot = float(((counts - counts.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")

    return FOMResult(
        fom=float(fom), separation=float(separation),
        mu_gamma=float(mu_g), sigma_gamma=float(s_g), amp_gamma=float(a_g),
        mu_n=float(mu_n), sigma_n=float(s_n), amp_n=float(a_n),
        n_events=int(vals.size), r_squared=float(r2),
        counts=counts, edges=edges)


class NeutronTraces:
    """Loaded trace dataset with a full PSD pipeline behind cached results.

    Construct from a PicoScope ``.npz`` (:meth:`from_npz`) or arrays. Set the
    analysis parameters — voltage threshold, gate start, prompt/tail
    boundaries — and call :meth:`compute` to (re)derive per-trace energy, PSD
    and the validity mask. Results are exposed as attributes so the GUI can
    filter and plot without re-integrating on every interaction.
    """

    def __init__(self, time_ns, traces, mean=None):
        self.time_ns = np.asarray(time_ns, dtype=float)
        # Keep the trace matrix in its native (typically float32) dtype: for a
        # 120k×750 acquisition that is ~360 MB vs. ~720 MB in float64.
        self.traces, self.sign = orient_positive(np.asarray(traces), reference=mean)
        self.mean = (self.sign * np.asarray(mean, dtype=float)
                     if mean is not None else self.traces.mean(axis=0))
        self.n_traces = self.traces.shape[0]

        # Analysis parameters (sensible defaults; the GUI overrides these).
        self.threshold_v = 0.0
        self.gate_start_ns = auto_pre_trigger(self.time_ns, self.mean)
        span = self.time_ns[-1] - self.gate_start_ns
        self.prompt_end_ns = self.gate_start_ns + min(19.0, 0.15 * span)
        self.tail_end_ns = self.gate_start_ns + min(150.0, 0.95 * span)

        # Cached results (filled by compute()).
        self.corrected = None
        self.energy = None      # q_total per trace (V·ns)
        self.psd = None
        self.valid = None       # bool mask: above threshold, finite psd, +ve area

    @classmethod
    def from_npz(cls, path):
        """Load ``time_ns`` / ``A_traces`` (and optional ``A_mean``) from *path*."""
        data = np.load(path)
        if "time_ns" not in data or "A_traces" not in data:
            raise ValueError(
                "Expected an .npz with 'time_ns' and 'A_traces' arrays; got keys "
                f"{list(data.keys())}")
        mean = data["A_mean"] if "A_mean" in data else None
        return cls(data["time_ns"], data["A_traces"], mean=mean)

    @classmethod
    def from_file(cls, path):
        """Load traces from any supported file type (see :func:`load_trace_file`).

        Handles the PicoScope ``.npz`` layout as well as ``.npy`` / ``.txt`` /
        ``.csv`` files holding one trace per row (with a synthesised time axis).
        """
        time_ns, traces, mean = load_trace_file(path)
        return cls(time_ns, traces, mean=mean)

    @classmethod
    def from_pixie(cls, date=None, runnr=None, channel=None, cfd="on",
                   align="fast", df=None, dt_ns=None):
        """Build a PSD dataset from PIXIE-16 list-mode traces.

        Reads a run with :func:`wara.helper_api.read_trace_data` (time-aligning
        the traces via *align*, one of ``"fast"``/``"edge"``/``"peak"``/``None``,
        and selecting the *cfd* acquisition), keeps clean single pulses of the
        selected *channel*, and stacks them into the trace matrix. PIXIE pulses
        are positive-going ADC on a large pedestal, so a baseline-subtracted mean
        is passed as the polarity / gate-start reference.

        Pass a pre-read *df* (and *dt_ns*) to rebuild for a different *channel*
        without re-reading the run from disk.
        """
        from wara import helper_api

        if df is None:
            df = helper_api.read_trace_data(date=date, runnr=runnr, cfd=cfd,
                                            align=align)
        if dt_ns is None:
            dt_ns = helper_api.read_sample_interval_ns(date, runnr)

        if channel is not None and "channel" in df.columns:
            df = df[df.channel == channel]
        if "pileup" in df.columns:
            df = df[df.pileup == 0]              # pile-up wrecks the PSD integrals
        if align is not None and "align_shift" in df.columns:
            df = df[df.align_shift.notna()]      # drop traces that failed to align
        if df.shape[0] == 0:
            raise ValueError(
                f"No usable traces for run {runnr}, channel {channel} "
                f"(cfd={cfd}, align={align})")

        lengths = df["trace"].map(lambda t: len(t) if hasattr(t, "__len__") else 0)
        modal = int(lengths.mode().iloc[0])
        df = df[lengths == modal]
        traces = np.stack([np.asarray(t, dtype=np.float32) for t in df["trace"]])
        time_ns = np.arange(modal, dtype=float) * float(dt_ns)

        # Baseline-subtracted mean: the raw pedestal (~8000 ADC) would otherwise
        # sit above the auto-pre-trigger's 20%-of-peak rise test and break it.
        npre = max(int(round(0.1 * modal)), 5)
        mean_ref = (traces - traces[:, :npre].mean(axis=1, keepdims=True)).mean(axis=0)
        return cls(time_ns, traces, mean=mean_ref)

    def set_params(self, *, threshold_v=None, gate_start_ns=None,
                   prompt_end_ns=None, tail_end_ns=None):
        """Update one or more analysis parameters (unset ones are unchanged)."""
        if threshold_v is not None:
            self.threshold_v = float(threshold_v)
        if gate_start_ns is not None:
            self.gate_start_ns = float(gate_start_ns)
        if prompt_end_ns is not None:
            self.prompt_end_ns = float(prompt_end_ns)
        if tail_end_ns is not None:
            self.tail_end_ns = float(tail_end_ns)

    def compute(self):
        """Recompute baseline-corrected traces, energy, PSD and the valid mask."""
        self.corrected = baseline_correct(self.traces, self.time_ns,
                                           self.gate_start_ns)
        self.psd, self.energy = psd_ratio(
            self.corrected, self.time_ns, self.gate_start_ns,
            self.prompt_end_ns, self.tail_end_ns)
        above = self.traces.max(axis=1) >= self.threshold_v
        positive = self.energy > 0
        self.valid = above & positive & np.isfinite(self.psd)
        return self

    def fom(self, energy_range=None, psd_range=None, bins=80):
        """Figure of merit of the valid pulses inside an energy/PSD window.

        *energy_range* is the ``(lo, hi)`` slice of the energy axis to project
        (the whole spectrum when ``None``) and *psd_range* limits the projected
        PSD axis, which must span both the gamma and the neutron band. See
        :func:`figure_of_merit`; :meth:`compute` must have been called first.
        """
        if self.valid is None:
            raise RuntimeError("Call compute() before fom()")
        mask = self.valid.copy()
        if energy_range is not None:
            elo, ehi = energy_range
            mask &= (self.energy >= elo) & (self.energy <= ehi)
        if psd_range is not None:
            plo, phi = psd_range
            mask &= (self.psd >= plo) & (self.psd <= phi)
        return figure_of_merit(self.psd[mask], bins=bins, psd_range=psd_range)
