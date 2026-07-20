"""Detector solid-angle (1/r^2) correction -- example on 2023-07-02 RUN93 CH5.

The gamma-detection probability falls with the distance r from the interaction
point to the detector, so the target side closer to the detector shows more
counts. Weighting each event by (r/r_ref)^p undoes it (p = 2 for a point-like
1/r^2 response; r_ref = mean r keeps the total roughly unchanged).

Outputs:
  * raw vs corrected X-Y maps and X/Y profiles of the iron target
  * the fitted exponent p* that flattens the profile for a detector at 18 cm
  * a scan of detector distances: where pure 1/r^2 flattens this run
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wara import read_parquet_api as rp
from wara import apicalc

OUT = os.path.dirname(os.path.abspath(__file__))
Z_T = 6.7
Z0 = -30.0

df = rp.read_parquet_file(date="2023-07-02", runnr=93, ch=5)
df = df.copy()
df["dt"] *= 1e9
df = apicalc.calc_own_pos(df)
dt = df["dt"].to_numpy(float)
dfm = df[(dt > 1.8) & (dt < 9.8)].reset_index(drop=True)   # middle (iron) peak
print(f"{len(dfm):,} events in the iron window")


def reconstruct(det):
    toff = apicalc.fit_toffset(dfm["dt"], z0=Z0, z_t=Z_T, det_pos=det,
                               dt_unit="ns")
    X, Y, Z = apicalc.api_xyz(dfm, det_pos=det, z_t=Z_T, toffset=toff,
                              dt_col="dt", dt_unit="ns", use_det=True)
    m = (np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z)
         & (Z > -45) & (Z < -20) & (np.abs(X) < 12) & (np.abs(Y) < 12))
    return X[m], Y[m], Z[m]


def weights(X, Y, Z, det, p=2.0):
    r = np.sqrt((X - det[0]) ** 2 + (Y - det[1]) ** 2 + (Z - det[2]) ** 2)
    return (r / r.mean()) ** p


def halves(v, w=None):
    w = np.ones_like(v) if w is None else w
    return w[v < 0].sum() / max(w[v > 0].sum(), 1e-30)


DET = (-18.0, 0.0, -25.0)
X, Y, Z = reconstruct(DET)
raw_x, raw_y = halves(X), halves(Y)
print(f"raw asymmetry: X left/right {raw_x:.3f}, Y low/high {raw_y:.3f}")

# fit the exponent p* that flattens the X profile at the entered det position
ps = np.linspace(0.0, 3.0, 61)
ratios = np.array([halves(X, weights(X, Y, Z, DET, p)) for p in ps])
p_star = float(np.interp(1.0, ratios[::-1], ps[::-1]))
print(f"exponent that flattens at det {DET}: p* = {p_star:.2f}")

# scan: at what detector distance does PURE 1/r^2 flatten the profile?
print("\ndistance scan (det on -x side, z=-25, pure 1/r^2):")
best = None
for d in (18, 30, 50, 75, 100, 150, 200, 300):
    det_d = (-float(d), 0.0, -25.0)
    Xd, Yd, Zd = reconstruct(det_d)
    ratio = halves(Xd, weights(Xd, Yd, Zd, det_d, 2.0))
    print(f"  d = {d:4d} cm  ->  corrected X left/right = {ratio:.3f}")
    if best is None or abs(ratio - 1) < abs(best[1] - 1):
        best = (d, ratio)
print(f"pure 1/r^2 flattens best near d = {best[0]} cm")

# ---- figure --------------------------------------------------------------
w2 = weights(X, Y, Z, DET, 2.0)
wp = weights(X, Y, Z, DET, p_star)
fig, ax = plt.subplots(2, 3, figsize=(16, 10))
maps = [(None, "raw"), (w2, "1/r$^2$ (p=2)"), (wp, f"fitted p={p_star:.2f}")]
for i, (w, lab) in enumerate(maps):
    h, xe, ye = np.histogram2d(X, Y, bins=80, range=[[-12, 12], [-12, 12]],
                               weights=w)
    ax[0, i].imshow(h.T, origin="lower", extent=[-12, 12, -12, 12],
                    cmap="plasma")
    ax[0, i].set(title=f"X-Y {lab}, det ({DET[0]:.0f},{DET[1]:.0f},{DET[2]:.0f})",
                 xlabel="X [cm]", ylabel="Y [cm]")

my = np.abs(Y) < 8
ctr = None
for w, lab, color in [(None, "raw", "gray"), (w2, "1/r$^2$ (p=2)", "tab:blue"),
                      (wp, f"fitted p={p_star:.2f}", "tab:red")]:
    h, ed = np.histogram(X[my], bins=48, range=(-12, 12),
                         weights=None if w is None else w[my])
    ctr = 0.5 * (ed[:-1] + ed[1:])
    ax[1, 0].step(ctr, h, where="mid", color=color, label=lab)
ax[1, 0].set(title=f"X profile (|Y|<8)  raw asym {raw_x:.3f}",
             xlabel="X [cm]", ylabel="counts")
ax[1, 0].legend(fontsize=8)

mx = np.abs(X) < 8
for w, lab, color in [(None, "raw", "gray"), (wp, f"p={p_star:.2f}", "tab:red")]:
    h, ed = np.histogram(Y[mx], bins=48, range=(-12, 12),
                         weights=None if w is None else w[mx])
    ax[1, 1].step(ctr, h, where="mid", color=color, label=lab)
ax[1, 1].set(title="Y profile (|X|<8)", xlabel="Y [cm]", ylabel="counts")
ax[1, 1].legend(fontsize=8)

ax[1, 2].plot(ps, ratios, "k-")
ax[1, 2].axhline(1.0, color="tab:red", ls=":")
ax[1, 2].axvline(p_star, color="tab:red", ls=":")
ax[1, 2].set(title=f"X asymmetry vs exponent p (det at 18 cm)\np* = {p_star:.2f}",
             xlabel="p  in  w = (r/r$_{ref}$)$^p$", ylabel="X left/right ratio")

fig.suptitle("RUN93 CH5 iron target -- detector solid-angle correction")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "example_r2_correction.png"), dpi=120)
print("wrote example_r2_correction.png")
