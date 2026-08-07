"""Offline reconstruction of the Pixie-16 fast trigger and CFD from raw traces.

This reimplements, from the DSP register values and the recorded ADC trace
alone, the four quantities the 500 MHz Pixie-16 firmware computes in hardware
to time-stamp an event (Pixie-16 User Manual v3.06, secs 3.3.7/3.3.8):

1. the fast filter (Eq 3-1) -- a trapezoidal leading-minus-trailing running
   sum used to detect that a pulse has arrived,
2. the fast filter threshold (``FastThresh``) -- the raw fast filter's fast
   trigger point is the first sample where the filter response crosses this
   value,
3. the CFD trace (Eq 3-5, the 500 MHz variant) -- a bipolar trace whose
   zero crossing marks the pulse's sub-sample arrival time, and
4. the CFD trigger threshold (``CFDThresh``) -- suppresses zero crossings
   caused by noise before the real pulse.

Everything here operates directly on raw (2 ns) ADC samples; the 500 MHz
firmware's ``w``, ``B``, ``D`` and ``L`` CFD parameters are fixed (Table 3-4)
at ``w=1, B=5, D=5, L=1`` and are *not* the same thing as the ``CFDScale`` /
``CFDDelay`` DSP registers (those apply to the 100/250 MHz CFD variant, Eq
3-2, not the 500 MHz one used here).

See :func:`read_fast_trigger_geometry` for converting the ``FastLength`` /
``FastGap`` DSP registers (stored in ``fast_filter_cycles`` units) to ADC
samples, and :func:`compute_cfd_timing` for the end-to-end reconstruction.
"""
import numpy as np

from wara import helper_api

# 500 MHz Pixie-16 CFD parameters are fixed in hardware (manual Table 3-4).
CFD_W = 1
# Some custom Pixie-16 firmware builds run the CFD with a fractional weight
# instead of the stock ``w=1``. This is the value used by our local custom
# firmware; expose it so callers/GUI can reconstruct that variant's CFD.
CFD_W_CUSTOM = 0.3125
CFD_B = 5
CFD_D = 5
CFD_L = 1
CFD_SCALE_N = 8192  # Eq 3-6 scaling factor for 500 MHz modules


def read_fast_trigger_geometry(date, runnr, ch):
    """Fast filter geometry and threshold for channel *ch*, in ADC-sample units.

    Reads the run's settings JSON and converts the fast filter's DSP
    registers to ADC-sample (2 ns) units the same way
    :func:`wara.helper_api.read_slow_filter_geometry` does for the energy
    filter::

        FL = FastLength * 2**FastFilterRange * adc_clk_div
        FG = FastGap    * 2**FastFilterRange * adc_clk_div

    ``FastThresh`` is a raw fast-filter amplitude and needs no conversion --
    :func:`fast_filter` returns its response in the same (unnormalized) raw
    ADC-sum units.

    Returns
    -------
    (int, int, int)
        ``(FL, FG, FastThresh)``.
    """
    s = helper_api._read_run_settings(date, runnr)
    R = s["module"]["input"]["FastFilterRange"]
    fast_length = s["channel"]["input"]["FastLength"][ch]
    fast_gap = s["channel"]["input"]["FastGap"][ch]
    fast_thresh = s["channel"]["input"]["FastThresh"][ch]
    div = s["metadata"]["config"][ch]["adc_clk_div"]
    FL = int(fast_length * 2 ** R * div)
    FG = int(fast_gap * 2 ** R * div)
    return FL, FG, int(fast_thresh)


def read_cfd_threshold(date, runnr, ch):
    """``CFDThresh`` DSP register for channel *ch* (raw CFD-amplitude units)."""
    s = helper_api._read_run_settings(date, runnr)
    return int(s["channel"]["input"]["CFDThresh"][ch])


def fast_filter(traces, FL, FG):
    """Trapezoidal fast filter response (manual Eq 3-1), raw ADC-sum units.

    ``FF[i] = sum(Trace[i-FL+1 .. i]) - sum(Trace[i-2*FL-FG+1 .. i-FL-FG])``

    Unlike :func:`wara.helper_api._fast_filter_ref` (an alignment helper that
    normalizes by peak amplitude) this returns the response in the same raw
    units as the hardware ``FastThresh`` register, unnormalized -- it must be
    compared against ``FastThresh`` directly, not a fraction of the peak.

    Parameters
    ----------
    traces : array_like, shape (N, M)
        Raw ADC traces (baseline need not be removed).
    FL, FG : int
        Fast filter length and gap, in ADC samples (see
        :func:`read_fast_trigger_geometry`).

    Returns
    -------
    numpy.ndarray, shape (N, M)
        The fast filter response. Samples before ``2*FL+FG-1`` (where the
        trailing window is not yet fully inside the trace) are NaN.
    """
    T = np.atleast_2d(np.asarray(traces, dtype=float))
    n = T.shape[1]
    cs = np.concatenate([np.zeros((T.shape[0], 1)), np.cumsum(T, axis=1)], axis=1)
    ff = np.full_like(T, np.nan)
    for i in range(2 * FL + FG - 1, n):
        lead = cs[:, i + 1] - cs[:, i + 1 - FL]
        trail = cs[:, i + 1 - FL - FG] - cs[:, i + 1 - 2 * FL - FG]
        ff[:, i] = lead - trail
    return ff


def find_fast_trigger(ff, threshold):
    """First sample where the fast filter crosses above *threshold*.

    Parameters
    ----------
    ff : array_like, shape (N, M)
        Fast filter response, as returned by :func:`fast_filter`.
    threshold : float
        ``FastThresh`` (raw units, see :func:`read_fast_trigger_geometry`).

    Returns
    -------
    numpy.ndarray of int, shape (N,)
        Index of the first sample with ``ff[i-1] < threshold <= ff[i]``, or
        -1 where the filter never crosses.
    """
    ff = np.atleast_2d(np.asarray(ff, dtype=float))
    above = np.nan_to_num(ff, nan=-np.inf) >= threshold
    idx = np.full(ff.shape[0], -1, dtype=int)
    for r in range(ff.shape[0]):
        crossings = np.where(above[r, 1:] & ~above[r, :-1])[0]
        if crossings.size:
            idx[r] = crossings[0] + 1
    return idx


def cfd_trace(traces, w=CFD_W, B=CFD_B, D=CFD_D, L=CFD_L):
    """500 MHz Pixie-16 CFD response (manual Eq 3-5), raw ADC-sample domain.

    ``CFD[k] = w*(sum(a,k..k+L) - sum(a,k-B..k-B+L))``
    ``       - (sum(a,k-D..k-D+L) - sum(a,k-D-B..k-D-B+L))``

    *w*, *B*, *D*, *L* default to the fixed 500 MHz values (Table 3-4:
    ``w=1, B=5, D=5, L=1``) -- pass others only to explore the general form.

    Parameters
    ----------
    traces : array_like, shape (N, M)
        Raw ADC traces.

    Returns
    -------
    numpy.ndarray, shape (N, M)
        The CFD response, indexed the same as *traces*. Samples before
        ``B+D`` or after ``M-L-1`` (where the sliding sums fall outside the
        trace) are NaN.
    """
    T = np.atleast_2d(np.asarray(traces, dtype=float))
    n = T.shape[1]
    cs = np.concatenate([np.zeros((T.shape[0], 1)), np.cumsum(T, axis=1)], axis=1)

    def window_sum(k):
        # sum(a, k .. k+L) inclusive, via the cumulative-sum array `cs`
        return cs[:, k + L + 1] - cs[:, k]

    cfd = np.full_like(T, np.nan)
    for k in range(B + D, n - L):
        cfd[:, k] = w * (window_sum(k) - window_sum(k - B)) - (
            window_sum(k - D) - window_sum(k - D - B))
    return cfd


def find_cfd_crossing(cfd, cfd_threshold, start_index):
    """Sub-sample CFD zero-crossing position, gated by *cfd_threshold*.

    The zero-crossing point (ZCP) is where ``CFD[k] >= 0`` and
    ``CFD[k+1] < 0`` (manual sec 3.3.8.1). To suppress zero crossings caused
    by noise ahead of the real pulse, the search starts only once the CFD
    trace has first risen above *cfd_threshold* at or after *start_index*
    (the fast trigger point) -- mirroring the manual's description of
    ``CFDThresh`` (sec 3.3.8.1, page 45).

    Parameters
    ----------
    cfd : array_like, shape (N, M)
        CFD response, as returned by :func:`cfd_trace`.
    cfd_threshold : float
        ``CFDThresh`` (raw units, see :func:`read_cfd_threshold`).
    start_index : array_like of int, shape (N,)
        Per-trace sample to start searching from (the fast trigger index,
        see :func:`find_fast_trigger`). Rows with ``start_index < 0`` are
        marked invalid.

    Returns
    -------
    position : numpy.ndarray of float, shape (N,)
        Sub-sample zero-crossing position (``k + f``, in samples), NaN where
        not found -- the CFD-forced-trigger case (manual: CFD trigger source
        ``111``, "CFD time is invalid").
    valid : numpy.ndarray of bool, shape (N,)
        ``False`` wherever *position* is NaN.
    """
    cfd = np.atleast_2d(np.asarray(cfd, dtype=float))
    start_index = np.asarray(start_index, dtype=int)
    n_rows = cfd.shape[0]
    position = np.full(n_rows, np.nan)

    for r in range(n_rows):
        s0 = start_index[r]
        if s0 < 0:
            continue
        row = cfd[r]
        finite = np.isfinite(row)
        # gate: first sample at/after the fast trigger where CFD rises above
        # cfd_threshold (suppresses noise-caused zero crossings ahead of it)
        above = finite & (row >= cfd_threshold)
        above_idx = np.where(above[s0:])[0]
        if above_idx.size == 0:
            continue
        gate = s0 + above_idx[0]

        seg = row[gate:]
        seg_finite = np.isfinite(seg)
        zc = np.where((seg[:-1] >= 0) & (seg[1:] < 0)
                      & seg_finite[:-1] & seg_finite[1:])[0]
        if zc.size == 0:
            continue
        k = gate + zc[0]
        out1, out2 = row[k], row[k + 1]
        f = out1 / (out1 - out2) if out1 != out2 else 0.0
        position[r] = k + f

    return position, np.isfinite(position)


def compute_cfd_timing(traces, FL, FG, fast_threshold, cfd_threshold,
                       w=CFD_W, B=CFD_B, D=CFD_D, L=CFD_L):
    """End-to-end fast trigger + CFD reconstruction for a batch of traces.

    Combines :func:`fast_filter`, :func:`find_fast_trigger`, :func:`cfd_trace`
    and :func:`find_cfd_crossing` to reproduce, from raw traces and DSP
    settings alone, the four quantities the 500 MHz Pixie-16 firmware
    computes for triggering and timing (manual secs 3.3.7-3.3.8).

    Parameters
    ----------
    traces : array_like, shape (N, M)
        Raw ADC traces.
    FL, FG, fast_threshold : int
        Fast filter geometry and threshold, see
        :func:`read_fast_trigger_geometry`.
    cfd_threshold : int
        ``CFDThresh``, see :func:`read_cfd_threshold`.
    w, B, D, L : int, optional
        CFD parameters; default to the fixed 500 MHz values (Table 3-4).

    Returns
    -------
    dict
        ``fast_filter`` (N, M), ``fast_trigger_index`` (N,), ``cfd_trace``
        (N, M), ``cfd_position`` (N,, sub-sample zero-crossing in samples)
        and ``cfd_valid`` (N,, bool).
    """
    ff = fast_filter(traces, FL, FG)
    trigger_index = find_fast_trigger(ff, fast_threshold)
    cfd = cfd_trace(traces, w=w, B=B, D=D, L=L)
    position, valid = find_cfd_crossing(cfd, cfd_threshold, trigger_index)
    return {
        "fast_filter": ff,
        "fast_trigger_index": trigger_index,
        "cfd_trace": cfd,
        "cfd_position": position,
        "cfd_valid": valid,
    }
