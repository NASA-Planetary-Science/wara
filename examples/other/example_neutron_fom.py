"""
Example for the Neutrons tab's "Figure of merit" button.

Arming the GUI button lets you drag a box across both PSD bands, which pops the
FOM up in its own window. This script does the same thing headlessly on the
shipped EJ-309 subset:

  * left  — the 2-D PSD-vs-energy histogram with the selected slice outlined;
  * middle — the pulses in that slice projected onto the PSD axis, fitted with a
            double Gaussian, and reduced to

                FOM = |mu_n - mu_gamma| / (FWHM_gamma + FWHM_n)

            with FOM >= 1.27 the usual threshold for clean separation;
  * right  — the same fit on a synthetic, cleanly separated detector, for
            comparison.

Note: the shipped 2000-pulse subset is not a well-separated acquisition — its
bands overlap, so the fit converges on a broad gamma component plus a narrow
neutron one and the FOM lands far below 1.27. The workflow (and the API) is
identical on a full, well-gated run.
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from wara.neutron_psd import NeutronTraces, figure_of_merit

npz = Path("../data/EJ309_neutrons_traces_subset.npz")

nt = NeutronTraces.from_npz(npz).compute()
valid = nt.valid
energy, psd = nt.energy[valid], nt.psd[valid]
print(f"Loaded {nt.n_traces} pulses ({valid.sum()} valid)")

# The slice to project: an energy window (discrimination varies strongly with
# light output) spanning both the gamma and the neutron band in PSD. The PSD
# limits come from the pulses inside the energy window, trimming the outliers
# that would otherwise stretch the projection axis.
e_lo, e_hi = 3.0, 8.0
in_slice = (energy >= e_lo) & (energy <= e_hi)
p_lo, p_hi = np.percentile(psd[in_slice], [1, 99])

res = nt.fom(energy_range=(e_lo, e_hi), psd_range=(p_lo, p_hi), bins=80)
print(f"gamma : mu = {res.mu_gamma:.4f}  FWHM = {res.fwhm_gamma:.4f}")
print(f"neutron: mu = {res.mu_n:.4f}  FWHM = {res.fwhm_n:.4f}")
print(f"S = {res.separation:.4f}   FOM = {res.fom:.3f}   R2 = {res.r_squared:.3f}")

fig, (ax_psd, ax_fom, ax_ref) = plt.subplots(1, 3, figsize=(16, 4.5))

pad = 0.15 * (p_hi - p_lo)
ax_psd.hist2d(energy, psd, bins=[200, 200],
              range=[[energy.min(), energy.max()], [p_lo - pad, p_hi + pad]],
              norm="log", cmap="jet")
ax_psd.add_patch(plt.Rectangle((e_lo, p_lo), e_hi - e_lo, p_hi - p_lo,
                               fill=False, edgecolor="lime", ls="--", lw=1.8))
ax_psd.set_xlabel("Energy / pulse integral (V·ns)")
ax_psd.set_ylabel("PSD = 1 - Q_prompt / Q_tail")
ax_psd.set_title("PSD vs. energy — selected slice")

x = np.linspace(res.centers[0], res.centers[-1], 600)
ax_fom.bar(res.centers, res.counts, width=np.diff(res.edges), color="0.7",
           align="center", label="PSD projection")
ax_fom.plot(x, res.component(x, "gamma"), "--", color="tab:cyan",
            label=f"γ: FWHM = {res.fwhm_gamma:.4f}")
ax_fom.plot(x, res.component(x, "neutron"), "--", color="tab:orange",
            label=f"n: FWHM = {res.fwhm_n:.4f}")
ax_fom.plot(x, res.curve(x), color="tab:red", lw=2, label="double Gaussian")
ax_fom.set_xlabel("PSD = 1 - Q_prompt / Q_tail")
ax_fom.set_ylabel("Counts")
ax_fom.set_title(f"FOM = {res.fom:.3f}  ({res.n_events} pulses)")
ax_fom.legend(fontsize=8)

# ── Reference: what a cleanly discriminating detector looks like ─────────────
rng = np.random.default_rng(0)
clean = np.concatenate([rng.normal(0.10, 0.020, 8000),    # gamma band
                        rng.normal(0.30, 0.025, 4000)])   # neutron band
ref = figure_of_merit(clean, bins=100)
xr = np.linspace(ref.centers[0], ref.centers[-1], 600)
ax_ref.bar(ref.centers, ref.counts, width=np.diff(ref.edges), color="0.7",
           align="center")
ax_ref.plot(xr, ref.component(xr, "gamma"), "--", color="tab:cyan")
ax_ref.plot(xr, ref.component(xr, "neutron"), "--", color="tab:orange")
ax_ref.plot(xr, ref.curve(xr), color="tab:red", lw=2)
ax_ref.set_xlabel("PSD = 1 - Q_prompt / Q_tail")
ax_ref.set_ylabel("Counts")
ax_ref.set_title(f"Reference two-band case — FOM = {ref.fom:.3f} (≥ 1.27)")
print(f"reference well-separated case: FOM = {ref.fom:.3f}")

fig.tight_layout()
plt.show()
