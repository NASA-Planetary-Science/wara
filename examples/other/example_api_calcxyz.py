"""
Test XYZ API
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wara import apicalc as api
from wara import read_parquet_api as read

dfs = read.read_parquet_file(date="2023-07-02", runnr=92)
dfs = api.calc_own_pos(dfs)
dfs["X2"] = dfs["X2"]*1.27
dfs["dt"] = dfs["dt"]*1e9 # to ns

#api.plot_2D_alphas(df=dfs, xkey="X2", ykey="Y2")

## X-Y
xcts, xedg = np.histogram(dfs["X2"], bins=1000)
xl = (xedg[1:] + xedg[:-1])/2

plt.figure()
plt.plot(xl, xcts)

## Calibration: map X2, Y2 to physical cm using detector active area
ACTIVE_CM = 4.8   # detector active area in each axis
TAIL_FRAC = 0.03  # fraction to clip on each tail

def alpha_extremes(values, tail_frac=TAIL_FRAC, method="percentile", bins=500, edge_frac=0.25):
    vals = values.dropna().to_numpy()
    if method == "percentile":
        lo = np.percentile(vals, tail_frac * 100)
        hi = np.percentile(vals, (1 - tail_frac) * 100)
    elif method == "peak":
        counts, edges = np.histogram(vals, bins=bins)
        mids = (edges[1:] + edges[:-1]) / 2
        n = len(counts)
        edge_n = max(1, int(n * edge_frac))
        lo = mids[np.argmax(counts[:edge_n])]
        hi = mids[(n - edge_n) + np.argmax(counts[n - edge_n:])]
    return lo, hi

def to_cm(values, lo, hi):
    center = (lo + hi) / 2.0
    scale = ACTIVE_CM / (hi - lo)
    return (values - center) * scale

x2_lo, x2_hi = alpha_extremes(dfs["X2"], method="peak")
y2_lo, y2_hi = alpha_extremes(dfs["Y2"], method="peak")

dfs["X_alpha"] = to_cm(dfs["X2"], x2_lo, x2_hi)
dfs["Y_alpha"] = to_cm(dfs["Y2"], y2_lo, y2_hi)

## Plot calibration results
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Alpha position calibration")

ycts, yedg = np.histogram(dfs["Y2"], bins=1000)
yl = (yedg[1:] + yedg[:-1]) / 2

for ax, vals, cts, edges, lo, hi, raw_label in [
    (axes[0, 0], dfs["X2"], xcts, xedg, x2_lo, x2_hi, "X2 (arb.)"),
    (axes[1, 0], dfs["Y2"], ycts, yedg, y2_lo, y2_hi, "Y2 (arb.)"),
]:
    mids = (edges[1:] + edges[:-1]) / 2
    ax.plot(mids, cts)
    ax.axvline(lo, color="r", ls="--", label=f"lo = {lo:.4f}")
    ax.axvline(hi, color="r", ls="--", label=f"hi = {hi:.4f}")
    ax.set_xlabel(raw_label)
    ax.legend(fontsize=8)

xcts_cal, xedg_cal = np.histogram(dfs["X_alpha"], bins=500, range=[-ACTIVE_CM/2, ACTIVE_CM/2])
ycts_cal, yedg_cal = np.histogram(dfs["Y_alpha"], bins=500, range=[-ACTIVE_CM/2, ACTIVE_CM/2])

for ax, cts_cal, edg_cal, label in [
    (axes[0, 1], xcts_cal, xedg_cal, "X_alpha (cm)"),
    (axes[1, 1], ycts_cal, yedg_cal, "Y_alpha (cm)"),
]:
    mids = (edg_cal[1:] + edg_cal[:-1]) / 2
    ax.plot(mids, cts_cal)
    ax.set_xlabel(label)
    ax.axvline(-ACTIVE_CM/2, color="g", ls="--", alpha=0.6, label="detector edge")
    ax.axvline( ACTIVE_CM/2, color="g", ls="--", alpha=0.6)
    ax.legend(fontsize=8)

plt.tight_layout()

# filter in XY and energy
dfxye = api.dfxye(df=dfs, xrange=[-0.6, 0.6], yrange=[-0.6, 0.6], erange=[670,1100],
                      xkey="X2", ykey="Y2", ekey="energy")

# plot time histogram
tbins = 2048
cts_ts, edg_ts = np.histogram(dfxye["dt"], bins=tbins, range=[-20,50])
ets = (edg_ts[1:]+edg_ts[:-1])/2

plt.figure(figsize=(12,8))
plt.plot(ets, cts_ts)
plt.xlabel("Time (ns)")


## Perform XYZ reconstruction
C_CM_PER_NS = 29.9792458
BETA_N = 0.17131                  # 14.1 MeV neutron
BETA_ALPHA = 0.04318              # 3.5 MeV alpha
V_N = BETA_N * C_CM_PER_NS        # ~5.1356 cm/ns (neutron)
V_ALPHA = BETA_ALPHA * C_CM_PER_NS  # ~1.2947 cm/ns (alpha)


def reconstruct_xyz(df, det_pos, da=6.7, t0="auto", physical_only=False):
    """
    Reconstruct (Xr, Yr, Zr) of the gamma emission point.
 
    Parameters
    ----------
    df : pandas.DataFrame with columns
            "dt" [ns], "X" = xa [cm], "Y" = ya [cm]
    det_pos : [xd, yd, zd]
            Gamma-detector pixel position [cm], from the neutron beam spot.
    da : float, optional
            Distance from alpha-detector center to beam spot [cm]. Default 6.7.
    t0 : float or "auto", optional
            Constant time offset [ns] added to dt before reconstruction.
            "auto" (default) assumes dt has been calibrated so the
            neutron-generator peak (Dn ~ 0, alpha-at-center) sits at dt = 0,
            and sets t0 = Rd/c - da/v_alpha.
            Set t0 = 0.0 if your dt is the raw t_gamma_det - t_alpha_det.
    physical_only : bool, optional
            If True, reject Dn < 0 (unphysical -- gamma "behind" the beam spot).
            If False (default), keep negative Dn so you can see the full
            distribution; this is the natural mirror of the physical region
            across dt = 0 and is useful for background/resolution diagnostics.
            The branch-selection (square-root sign) check is always applied.
 
    Returns
    -------
    DataFrame copy with added Dn, Xr, Yr, Zr [cm] (NaN where no real root or
    where the branch check fails; additionally NaN for Dn<0 if physical_only).
    """
    xd, yd, zd = det_pos
    Rd = np.sqrt(xd * xd + yd * yd + zd * zd)
 
    if t0 == "auto":
        t0 = Rd / C_CM_PER_NS - da / V_ALPHA
 
    xa = df["X"].to_numpy(dtype=float)
    ya = df["Y"].to_numpy(dtype=float)
    dt = df["dt"].to_numpy(dtype=float) + float(t0)   # convert to raw dt
 
    # Alpha-hit position from beam spot: r_a = (xa, ya, -da)
    L_alpha = np.sqrt(xa * xa + ya * ya + da * da)
    nx = -xa / L_alpha
    ny = -ya / L_alpha
    nz = da / L_alpha
 
    # Emission -> gamma-detection time
    tau = dt + L_alpha / V_ALPHA
 
    # kappa = r_d . n_hat
    kappa = xd * nx + yd * ny + zd * nz
 
    # Quadratic: A Dn^2 + B Dn + C = 0
    A = 1.0 / (BETA_N ** 2) - 1.0
    ctau = C_CM_PER_NS * tau
    B = 2.0 * (kappa - ctau / BETA_N)
    C = ctau ** 2 - Rd ** 2
 
    disc = B * B - 4.0 * A * C
    sqrt_disc = np.full_like(disc, np.nan)
    ok = disc >= 0
    sqrt_disc[ok] = np.sqrt(disc[ok])
 
    D1 = (-B - sqrt_disc) / (2.0 * A)
    D2 = (-B + sqrt_disc) / (2.0 * A)
 
    TOL = 1e-9
 
    def valid(D):
        # Branch check is mandatory (correct square-root sign).
        # Dn >= 0 is the physicality cut, applied only if physical_only=True.
        branch = (ctau - D / BETA_N >= -TOL)
        if physical_only:
            return branch & (D >= -TOL)
        return branch
 
    v1, v2 = valid(D1), valid(D2)
    Dn = np.full_like(D1, np.nan)
    only1 = v1 & ~v2; Dn[only1] = D1[only1]
    only2 = v2 & ~v1; Dn[only2] = D2[only2]
    both  = v1 & v2;  Dn[both]  = np.minimum(D1[both], D2[both])
 
    out = df.copy()
    out["Dn"] = Dn
    out["Xr"] = Dn * nx
    out["Yr"] = Dn * ny
    out["Zr"] = Dn * nz
    return out


dfc = reconstruct_xyz(df=dfxye, det_pos=[18.4, 0.25, -27])

# plot Z histogram
zbins = 2048
cts_z, edg_z = np.histogram(dfc["Zr"], bins=zbins, range=[-100,300])
ez = (edg_z[1:]+edg_z[:-1])/2

plt.figure(figsize=(12,8))
plt.plot(ez, cts_z)
plt.xlabel("Z (cm)")