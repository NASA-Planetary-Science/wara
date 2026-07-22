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

import numpy as np

__all__ = [
    "NeutronTraces",
    "orient_positive",
    "baseline_correct",
    "mca_integral",
    "psd_ratio",
    "auto_pre_trigger",
    "load_trace_file",
]


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
