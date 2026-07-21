"""
Example for the 'Neutrons' tab / ``wara.neutron_psd`` module.

Loads a small subset of EJ-309 PicoScope neutron-run traces and reproduces the
three linked panels of the Neutrons GUI tab as static matplotlib plots:

  * Traces  — example baseline-corrected pulses coloured by peak amplitude,
              with the voltage threshold and the Q_prompt / Q_tail gate windows.
  * MCA     — histogram of the per-pulse energy (total charge integral).
  * PSD     — 2-D histogram of PSD = 1 - Q_prompt/Q_tail vs. energy, where the
              neutron (long-tail) and gamma (short-tail) populations separate.

The dataset ``examples/data/EJ309_neutrons_traces_subset.npz`` is a 2000-pulse
random subset of a full 120k-pulse acquisition, kept small enough to ship.
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from wara.neutron_psd import NeutronTraces

npz = Path("../data/EJ309_neutrons_traces_subset.npz")

# Load, orient positive, and run the full PSD pipeline with default gates.
nt = NeutronTraces.from_npz(npz).compute()

valid = nt.valid
energy = nt.energy[valid]
psd = nt.psd[valid]
print(f"Loaded {nt.n_traces} pulses ({valid.sum()} valid after threshold/quality cuts)")

fig = plt.figure(figsize=(13, 7))
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.15])
ax_traces = fig.add_subplot(gs[0, 0])
ax_mca = fig.add_subplot(gs[1, 0])
ax_psd = fig.add_subplot(gs[:, 1])

# ── Traces: a spread of example pulses coloured by peak amplitude ─────────────
valid_idx = np.flatnonzero(valid)
peak = nt.corrected[valid_idx].max(axis=1)
order = valid_idx[np.argsort(peak)]
pick = order[np.linspace(0, len(order) - 1, 80).astype(int)]

norm = plt.Normalize(vmin=peak.min(), vmax=peak.max())
cmap = matplotlib.colormaps["viridis"]
for i in pick:
    amp = nt.corrected[i].max()
    ax_traces.plot(nt.time_ns, nt.corrected[i], lw=0.6, alpha=0.6,
                   color=cmap(norm(amp)))
sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
fig.colorbar(sm, ax=ax_traces, label="Peak amplitude (V)")

ax_traces.axhline(nt.threshold_v, color="r", ls="--", lw=0.8, label="threshold")
ax_traces.axvspan(nt.gate_start_ns, nt.prompt_end_ns, color="tab:red",
                  alpha=0.12, label="Q_prompt")
ax_traces.axvspan(nt.prompt_end_ns, nt.tail_end_ns, color="tab:purple",
                  alpha=0.12, label="Q_tail")
ax_traces.set_xlabel("Time (ns)")
ax_traces.set_ylabel("Amplitude (V)")
ax_traces.set_title(f"EJ-309 — {len(pick)} example traces")
ax_traces.legend(fontsize=8)

# ── MCA: per-pulse energy (total charge integral) ────────────────────────────
ax_mca.hist(energy, bins=512, histtype="step", color="tab:cyan")
ax_mca.set_yscale("log")
ax_mca.set_xlabel("Energy / pulse integral (V·ns)")
ax_mca.set_ylabel("Counts")
ax_mca.set_title("MCA spectrum")

# ── PSD: discrimination parameter vs. energy ─────────────────────────────────
psd_lo, psd_hi = np.percentile(psd, [1.0, 99.5])
pad = 0.05 * (psd_hi - psd_lo)
ax_psd.hist2d(energy, psd, bins=[256, 256],
              range=[[energy.min(), energy.max()], [psd_lo - pad, psd_hi + pad]],
              norm=matplotlib.colors.LogNorm(), cmap="jet")
ax_psd.set_xlabel("Energy / pulse integral (V·ns)")
ax_psd.set_ylabel("PSD = 1 - Q_prompt/Q_tail")
ax_psd.set_title("PSD vs. energy")

fig.tight_layout()
plt.show()
