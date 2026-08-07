"""Reconstruct the Pixie-16 fast trigger and CFD directly from raw traces.

The Pixie-16 firmware never stores the fast filter or CFD trace it computes
internally to time-stamp an event -- it only stores the final result (the CFD
fractional time). :mod:`wara.pixie_trace_analysis` reimplements that pipeline
offline from the raw ADC trace and the run's DSP settings (``FastLength``,
``FastGap``, ``FastThresh``, ``CFDThresh``), so it can be inspected and
plotted like any other quantity.

Two runs are shown: one read with :func:`wara.helper_api.read_trace_data`
(the usual ``trace-data`` acquisition) and one read with
:func:`wara.helper_api.read_binary_data` (a run where traces ended up in the
``binary-data`` folder instead).
"""
import numpy as np
import matplotlib.pyplot as plt

from wara import helper_api, pixie_trace_analysis as pta

RUNS = [
    {"date": "2025-04-18", "runnr": 8, "ch": 7, "reader": "trace_data"},
    {"date": "2026-08-05", "runnr": 0, "ch": 4, "reader": "binary_data"},
]

fig, axes = plt.subplots(len(RUNS), 2, figsize=(12, 4.5 * len(RUNS)))

for row, run in enumerate(RUNS):
    date, runnr, ch = run["date"], run["runnr"], run["ch"]
    FL, FG, fast_thresh = pta.read_fast_trigger_geometry(date, runnr, ch)
    cfd_thresh = pta.read_cfd_threshold(date, runnr, ch)
    print(f"\n{date} RUN{runnr} ch{ch}: FL={FL} FG={FG} "
          f"FastThresh={fast_thresh} CFDThresh={cfd_thresh}")

    if run["reader"] == "trace_data":
        df = helper_api.read_trace_data(date, runnr)
    else:
        df = helper_api.read_binary_data(date, runnr)
    df = df[(df.channel == ch) & (df.pileup == 0) & (~df.CFD_error)]
    df = df.sort_values("energy", ascending=False).head(1).reset_index(drop=True)

    trace = np.asarray(df.trace[0], dtype=float)
    res = pta.compute_cfd_timing(trace, FL, FG, fast_thresh, cfd_thresh)
    ff = res["fast_filter"][0]
    trig = res["fast_trigger_index"][0]
    cfd = res["cfd_trace"][0]
    pos = res["cfd_position"][0]
    print(f"  energy={df.energy[0]}  fast_trigger_index={trig}  "
          f"cfd_position={pos:.2f} samples  recorded CFD_fraction={df.CFD_fraction[0]}")

    ax_trace, ax_ff = axes[row]
    ax_trace.plot(trace, color="0.35", lw=1.0)
    ax_trace.axvline(trig, color="tab:blue", ls="--", label="fast trigger")
    if np.isfinite(pos):
        ax_trace.axvline(pos, color="tab:red", ls=":", label="CFD zero crossing")
    ax_trace.set_title(f"{date} RUN{runnr} ch{ch} (E={df.energy[0]})")
    ax_trace.set_xlabel("sample (2 ns)")
    ax_trace.set_ylabel("ADC")
    ax_trace.legend(loc="upper right", fontsize=8)

    ax_ff.plot(ff, color="tab:blue", lw=1.0, label="fast filter")
    ax_ff.axhline(fast_thresh, color="tab:blue", ls="--", lw=0.8, label="FastThresh")
    ax_ff2 = ax_ff.twinx()
    ax_ff2.plot(cfd, color="tab:red", lw=1.0, label="CFD trace")
    ax_ff2.axhline(cfd_thresh, color="tab:red", ls="--", lw=0.8, label="CFDThresh")
    ax_ff2.axhline(0, color="0.7", lw=0.8)
    ax_ff.set_xlabel("sample (2 ns)")
    ax_ff.set_ylabel("fast filter", color="tab:blue")
    ax_ff2.set_ylabel("CFD", color="tab:red")
    lines1, labels1 = ax_ff.get_legend_handles_labels()
    lines2, labels2 = ax_ff2.get_legend_handles_labels()
    ax_ff.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

fig.suptitle("Offline fast filter / CFD reconstruction (wara.pixie_trace_analysis)")
fig.tight_layout()
plt.show()
