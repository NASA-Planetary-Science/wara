"""
Example for the Neutrons tab's "Average trace only" option.

The GUI checkbox collapses the traces panel to a single averaged pulse. This
script reproduces both of its modes on the shipped EJ-309 subset:

  * left  — the mean of a 150-pulse random sample (what the checkbox shows when
            no PSD selection is armed), against the individual pulses;
  * right — one averaged pulse per PSD region, using *all* pulses inside each
            region (what the checkbox shows with "PSD selections" ON). The
            long-tail (neutron-like) region averages to a visibly slower decay
            than the short-tail (gamma-like) one.
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from wara.neutron_psd import NeutronTraces

npz = Path("../data/EJ309_neutrons_traces_subset.npz")

nt = NeutronTraces.from_npz(npz).compute()
valid = nt.valid
energy, psd = nt.energy, nt.psd
print(f"Loaded {nt.n_traces} pulses ({valid.sum()} valid)")

fig, (ax_s, ax_r) = plt.subplots(1, 2, figsize=(12, 4.5))

# ── Mode 1: average of the random sample drawn on the Traces panel ────────────
rng = np.random.default_rng(0)
idx = np.flatnonzero(valid)
sample = rng.choice(idx, size=min(150, idx.size), replace=False)
for i in sample:
    ax_s.plot(nt.time_ns, nt.corrected[i], lw=0.5, alpha=0.25, color="0.6")
ax_s.plot(nt.time_ns, nt.corrected[sample].mean(axis=0), lw=2.0, color="tab:cyan",
          label=f"average of {sample.size} sampled pulses")
ax_s.set_title("Average of the displayed random sample")
ax_s.legend(fontsize=8)

# ── Mode 2: one average per PSD selection, over every pulse in the box ────────
e_lo, e_hi = np.percentile(energy[valid], [20, 95])
p_lo, p_mid, p_hi = np.percentile(psd[valid], [5, 50, 98])
regions = [((e_lo, e_hi, p_mid, p_hi), "tab:orange", "high-PSD (neutron-like)"),
           ((e_lo, e_hi, p_lo, p_mid), "tab:green", "low-PSD (gamma-like)")]
for (elo, ehi, plo, phi), color, label in regions:
    mask = (valid & (energy >= elo) & (energy <= ehi)
            & (psd >= plo) & (psd <= phi))
    if not mask.any():
        continue
    mean = np.mean(nt.corrected[mask], axis=0, dtype=np.float64)
    ax_r.plot(nt.time_ns, mean, lw=2.0, color=color,
              label=f"{label} — {int(mask.sum())} pulses")
ax_r.set_title("One average per PSD selection (all selected pulses)")
ax_r.legend(fontsize=8)

for ax in (ax_s, ax_r):
    ax.axvspan(nt.gate_start_ns, nt.prompt_end_ns, color="tab:red", alpha=0.10)
    ax.axvspan(nt.prompt_end_ns, nt.tail_end_ns, color="tab:purple", alpha=0.10)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Amplitude (V)")

fig.tight_layout()
plt.show()
