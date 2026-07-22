"""Trace time-alignment for pulse-shape discrimination (PSD).

For PSD every recorded trace must sit on a common time axis, but the pulses
jitter in time: the 48-bit event timestamp is latched on the coarse clock tick,
so the fast-trigger point lands anywhere within a tick -- on a 500 MHz module
that is up to 10 ns = 5 samples of jitter. This example compares ways of
removing that jitter on 2026-07-21 RUN 19, acquired both WITH and WITHOUT CFD.

Key finding (see the printed "sharpness" numbers): the hardware CFD *fractional*
time only corrects the sub-sample part and, on its own, does NOT remove the
multi-sample jitter. Deriving a per-trace reference from the trace shape itself
(rising edge / fast filter / peak) and shifting every trace so it lands on a
common sample index works -- with or without CFD -- and roughly doubles the
sharpness of the averaged pulse.

``helper_api.read_trace_data(..., align=...)`` exposes this directly; here we
also call the lower-level ``align_traces`` to compare methods side by side.
"""
import numpy as np
import matplotlib.pyplot as plt

from wara import helper_api

DATE = "2026-07-21"
RUNNR = 19


def sharpness(df):
    """Max slope of the averaged pulse -- higher means better time-aligned."""
    d = df[(df.energy > 5000) & (df.pileup == 0)]
    T = np.stack([np.asarray(t, float) for t in d.trace])
    T -= T[:, :30].mean(axis=1, keepdims=True)
    return np.max(np.diff(T.mean(axis=0)))


def mean_pulse(df):
    d = df[(df.energy > 5000) & (df.pileup == 0)]
    T = np.stack([np.asarray(t, float) for t in d.trace])
    T -= T[:, :30].mean(axis=1, keepdims=True)
    return T.mean(axis=0)


fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for ax, cfd in zip(axes, ["on", "off"]):
    raw = helper_api.read_trace_data(DATE, RUNNR, cfd=cfd)
    print(f"\n=== CFD {cfd}  (n={len(raw)}) ===")
    print(f"  raw (no align) sharpness = {sharpness(raw):8.1f}")
    ax.plot(mean_pulse(raw), label="raw", lw=2, color="0.6")

    for method in ["edge", "fast", "peak"]:
        df = helper_api.read_trace_data(DATE, RUNNR, cfd=cfd, align=method)
        print(f"  align={method:5s}    sharpness = {sharpness(df):8.1f}   "
              f"(median shift {np.nanmedian(df.align_shift):+.2f} samples)")
        ax.plot(mean_pulse(df), label=f"align={method}", lw=1.2)

    ax.set_title(f"CFD {cfd}: averaged pulse")
    ax.set_xlabel("sample (2 ns)")
    ax.set_xlim(40, 120)
    ax.legend()

axes[0].set_ylabel("amplitude (baseline-subtracted)")
fig.suptitle(f"Trace alignment comparison -- {DATE} RUN {RUNNR}")
fig.tight_layout()
plt.show()
