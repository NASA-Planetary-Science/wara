r"""
Example / test of the full-model api_xyz on 2023-07-02 RUN 93 CH 5.

Reconstructs the neutron-interaction cloud from X2/Y2 + dt only (the precomputed
X/Y/Z columns are ignored), using:
  * apicalc.api_xyz          -- full model, gamma detector at (-18,0,-25) cm
  * apicalc.api_xyz(use_det=False) -- full model, straight ray
  * apicalc.api_xyz_simple   -- legacy non-relativistic point-source model

Prints summary statistics and writes scratch/api_xyz_example.png.

Coordinate convention (full model): origin at the neutron source, +z up so
**-z points down** toward the sample; the YAP sits at +z_t and the interaction
lands at negative z.
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wara import read_parquet_api as rp
from wara import apicalc

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "api_xyz_example.png")

DATE, RUN, CH = "2023-07-02", 93, 5
DET_POS = (-18.0, 0.0, -25.0)   # gamma detector [cm] from source, -z down
Z_T = 6.7                       # source -> YAP [cm]
Z0_TARGET = -100.0              # nominal sample depth [cm] for timing calibration


def fit_toffset(raw_dt_s, z0_cm=Z0_TARGET, z_t=Z_T):
    """Crude per-detector timing calibration: choose the offset that places the
    dominant raw-dt peak at depth z0 along a central ray (this is the role of the
    ROOTS z-align constant). Returns toffset in seconds."""
    h, ed = np.histogram(raw_dt_s * 1e9, bins=400, range=(-60, 100))
    ctr = 0.5 * (ed[:-1] + ed[1:])
    dt_peak_s = ctr[np.argmax(h)] * 1e-9
    va = apicalc._speed(apicalc._E_ALPHA, apicalc._M_ALPHA)
    vn = apicalc._speed(apicalc._E_NEUTRON, apicalc._M_NEUTRON)
    t_alpha_central = (z_t / 100.0) / va
    dta_target = (abs(z0_cm) / 100.0) / vn
    return dt_peak_s + t_alpha_central - dta_target


def main():
    print(f"Loading {DATE} RUN {RUN} CH {CH} ...")
    df = rp.read_parquet_file(DATE, RUN, ch=CH)
    df = apicalc.calc_own_pos(df)                    # ensure X2/Y2
    print(f"  {len(df):,} events; columns include X2/Y2: "
          f"{'X2' in df and 'Y2' in df}")

    # Use the genuinely-raw measured time. In this run calcXYZ overwrote `dt`
    # (it folded in t_alpha and a z-align), so the raw measurement is `dt_orig`;
    # for an unprocessed run there is only `dt`, which is already raw.
    dt_col = "dt_orig" if "dt_orig" in df.columns else "dt"
    print(f"  raw time column: {dt_col!r} (seconds)")

    # Coincidence window around the alpha-gamma peak to drop random background.
    raw = df[dt_col].to_numpy()
    h, ed = np.histogram(raw * 1e9, bins=400, range=(-60, 100))
    peak_ns = 0.5 * (ed[:-1] + ed[1:])[np.argmax(h)]
    win = (raw * 1e9 > peak_ns - 25) & (raw * 1e9 < peak_ns + 70)
    dfc = df[win].reset_index(drop=True)
    print(f"  coincidence peak at {peak_ns:.1f} ns; "
          f"{len(dfc):,} events in window")

    toff = fit_toffset(dfc[dt_col].to_numpy())
    print(f"  fitted timing offset (z-align): {toff*1e9:.2f} ns "
          f"(places dominant feature at z={Z0_TARGET:.0f} cm)")

    # --- reconstructions (X2/Y2 + dt only) ---------------------------------
    Xf, Yf, Zf = apicalc.api_xyz(
        dfc, det_pos=DET_POS, z_t=Z_T, dt_col=dt_col, dt_unit="s",
        toffset=toff, use_det=True)
    Xs_, Ys_, Zs_ = apicalc.api_xyz(
        dfc, det_pos=DET_POS, z_t=Z_T, dt_col=dt_col, dt_unit="s",
        toffset=toff, use_det=False)

    # legacy simple model wants dt in ns in a column called 'dt'
    dleg = dfc[["X2", "Y2"]].copy()
    dleg["dt"] = dfc[dt_col].to_numpy() * 1e9
    Xl, Yl, Zl = apicalc.api_xyz_simple(
        dleg, det_pos=DET_POS, toffset=toff * 1e9, use_det=True)

    def summ(name, Z):
        p = np.nanpercentile(Z, [5, 50, 95])
        print(f"  {name:30s} Z[cm] p5/50/95 = "
              f"{p[0]:8.1f} {p[1]:8.1f} {p[2]:8.1f}")

    print("\nReconstructed depth summaries:")
    summ("full  (use_det=True)", Zf)
    summ("full  (use_det=False)", Zs_)
    summ("simple (legacy)", Zl)

    dz = np.nanmedian(Zf - Zs_)
    print(f"\n  median depth shift, detector term: {dz:+.2f} cm")
    print(f"  median |full - simple| in z:        "
          f"{np.nanmedian(np.abs(Zf - Zl)):.2f} cm")

    # --- figure ------------------------------------------------------------
    from matplotlib.colors import LogNorm
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))

    xa, ya = apicalc._alpha_xy_cm(dfc)
    ax[0, 0].hist2d(xa, ya, bins=120, range=[[-2.6, 2.6], [-2.6, 2.6]],
                    cmap="plasma", norm=LogNorm())
    ax[0, 0].set(title="Alpha image on YAP", xlabel="x0 [cm]", ylabel="y0 [cm]")
    ax[0, 0].set_aspect("equal")

    m = np.isfinite(Xf) & np.isfinite(Yf)
    ax[0, 1].hist2d(Xf[m], Yf[m], bins=140,
                    range=[[-70, 70], [-70, 70]], cmap="plasma", norm=LogNorm())
    ax[0, 1].set(title="Reconstructed X-Y at the sample (full)",
                 xlabel="X [cm]", ylabel="Y [cm]")
    ax[0, 1].set_aspect("equal")

    zr = (-220, 20)
    # legacy model uses the opposite +z convention -> sign-flip to compare shape
    for Z, lab in [(Zf, "full (use_det)"), (Zs_, "full (no det)"),
                   (-Zl, "simple (legacy, -Z)")]:
        ax[1, 0].hist(Z[np.isfinite(Z)], bins=220, range=zr,
                      histtype="step", label=lab, lw=1.5)
    ax[1, 0].axvline(Z0_TARGET, color="k", ls=":", lw=1,
                     label=f"calib. target {Z0_TARGET:.0f} cm")
    ax[1, 0].set(title="Depth (Z) histogram", xlabel="Z [cm] (down = negative)",
                 ylabel="counts")
    ax[1, 0].legend(fontsize=8)

    m2 = np.isfinite(Xf) & np.isfinite(Zf)
    ax[1, 1].hist2d(Xf[m2], Zf[m2], bins=160,
                    range=[[-70, 70], [-200, 10]], cmap="plasma", norm=LogNorm())
    ax[1, 1].set(title="X-Z projection (full) — neutron cone",
                 xlabel="X [cm]", ylabel="Z [cm]")

    fig.suptitle(f"api_xyz full-model reconstruction — {DATE} RUN {RUN} CH {CH}",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT, dpi=130)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
