"""
Anorthosite (top), Olivine (middle), Ilmenite (bottom)
Data: 2024-08-21 RUN 95 (calibrated in energy)

Split the data into time slices and, for each line of interest (Si, Mg, ...),
fit the peak two ways per slice:
  - Gaussian + linear background (peakfit.PeakFit)
  - linear-background ROI integration (advanced_fit.PeakAreaLinearBkg)
Then show the per-slice fits and the area / normalized area / reduced-chi2 as
functions of time.
"""

from wara import read_parquet_api
from wara import apicalc
import matplotlib.pyplot as plt
import numpy as np
from wara import spectrum as sp
from wara import peaksearch as ps
from wara import peakfit as pf
from wara import advanced_fit as af

date = "2024-08-21"
run_no = 95
ch = 4
df = read_parquet_api.read_parquet_file(date=date, runnr=run_no, ch=ch)
df = apicalc.calc_own_pos(df=df, xkey_new="X2", ykey_new="Y2")

# xyt cuts
xrange = [-0.5, 0.5]
yrange = [-0.5, 0.5]
trange = [9, 25]
df["dt"] = df["dt"] * 1e9
dfsxyt = apicalc.dftxy(
    df=df, xrange=xrange, yrange=yrange, trange=trange, xkey="X2", ykey="Y2"
)

# time slices
dt = 2  # ns
tmin = dfsxyt["dt"].min()
tmax = dfsxyt["dt"].max()
no_slices = round((tmax - tmin) / dt)
e_bins = 2**10

# PeakSearch resolution reference is in CHANNELS, so it must scale with the
# number of energy bins. These were tuned at 2**11 bins; rescale to e_bins so
# the same physical reference energy/resolution is used at any binning.
_bin_ref = 2**11
ref_x = 420 * e_bins / _bin_ref
ref_fwhm = 12 * e_bins / _bin_ref

# Lines of interest. Windows are generous (wide) on purpose: the Gaussian fit
# needs enough background on each side to constrain the linear continuum, and
# the linear-ROI peak region runs between the inner edges of bkg_l / bkg_r.
lines = {
    "Si": dict(e=1778.97, win=[1700.0, 1850.0],
               bkg_l=[1680.0, 1715.0], bkg_r=[1840.0, 1880.0]),
    "Mg": dict(e=1368.67, win=[1290.0, 1450.0],
               bkg_l=[1270.0, 1305.0], bkg_r=[1430.0, 1470.0]),
    # Fe right side kept below the Ti line (983 keV) to avoid contamination.
    "Fe": dict(e=846.78, win=[775.0, 920.0],
               bkg_l=[755.0, 790.0], bkg_r=[905.0, 935.0]),
}


def fit_gauss(search, line_e, win, tol=20):
    """Fit a line with a Gaussian; return (PeakFit, peak_index) or None."""
    try:
        fit = pf.PeakFit(search=search, xrange=list(win))
    except Exception:
        return None
    best, best_d = None, np.inf
    for k, info in enumerate(fit.peak_info):
        d = abs(info["mean"] - line_e)
        if d < best_d:
            best_d, best = d, k
    if best is None or best_d > tol:
        return None
    return fit, best


def fit_linear(spe, bkg_l, bkg_r):
    """Net area via linear-background ROI integration (PeakAreaLinearBkg)."""
    pa = af.PeakAreaLinearBkg(spe, bkg_l[0], bkg_r[1])
    try:
        pa.calculate_peak_area(bkg_l, bkg_r)
    except Exception:
        return None
    return pa


def line_snr(search, line_e, tol=20):
    """Peak SNR near a line: max of search.snr within +/- tol of the centroid.
    search.snr is a per-energy-bin array (method 'km'), aligned to the spectrum."""
    snr = getattr(search, "snr", None)
    if snr is None:
        return np.nan
    e = search.spectrum.energies
    band = np.abs(e - line_e) <= tol
    if not np.any(band):
        return np.nan
    return float(np.nanmax(snr[band]))


def _normalize(a, a_err):
    """Normalize an array (and its error) to the array's max over time."""
    amax = np.nanmax(a) if np.any(np.isfinite(a)) else np.nan
    return a / amax, a_err / amax


def ratio_to_ref(num, num_err, den, den_err):
    """Element-area ratio num/den per time slice, with uncorrelated Poisson
    error propagation. Slices with a non-positive reference area are NaN."""
    num = np.asarray(num, float)
    den = np.asarray(den, float)
    num_err = np.asarray(num_err, float)
    den_err = np.asarray(den_err, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = num / den
        rerr = np.abs(r) * np.sqrt((num_err / num) ** 2 + (den_err / den) ** 2)
    bad = (den <= 0) | ~np.isfinite(r)
    r = np.where(bad, np.nan, r)
    rerr = np.where(np.isfinite(rerr), rerr, np.nan)
    rerr = np.where(bad, np.nan, rerr)
    return r, rerr


# --- Build the per-slice spectra once (shared across all lines) ---
slices = []  # one dict per time slice
for j in range(no_slices):
    t0 = tmin + dt * j
    t1 = tmin + dt * (j + 1)
    df0 = apicalc.dft(df=dfsxyt, trange=[t0, t1])
    cts, edg = np.histogram(a=df0["energy_cal"], bins=e_bins)
    ergs = (edg[1:] + edg[:-1]) / 2
    spe = sp.Spectrum(counts=cts, energies=ergs, label=f"t=[{t0:.1f},{t1:.1f}]")
    search = ps.PeakSearch(spectrum=spe, ref_x=ref_x, ref_fwhm=ref_fwhm, min_snr=3)
    slices.append(dict(t0=t0, t1=t1, tc=(t0 + t1) / 2,
                       ergs=ergs, cts=cts, spe=spe, search=search))


def process_line(cfg):
    """Run both area estimators for one line across all slices."""
    R = {k: [] for k in ("tc", "area", "area_err", "ratio", "ratio_err", "chi2",
                         "area_lin", "area_lin_err", "snr", "slice_fits")}
    for s in slices:
        R["tc"].append(s["tc"])
        R["snr"].append(line_snr(s["search"], cfg["e"]))
        result = fit_gauss(s["search"], cfg["e"], cfg["win"])
        pa = fit_linear(s["spe"], cfg["bkg_l"], cfg["bkg_r"])

        win = (s["ergs"] >= cfg["win"][0]) & (s["ergs"] <= cfg["win"][1])
        R["slice_fits"].append((s["t0"], s["t1"], s["ergs"][win], s["cts"][win],
                                result, pa))

        # linear-background ROI area (independent of the Gaussian fit)
        if pa is None:
            R["area_lin"].append(np.nan)
            R["area_lin_err"].append(np.nan)
        else:
            R["area_lin"].append(pa.A)
            R["area_lin_err"].append(pa.sigA)

        # Gaussian fit area: if the peak was not found, the area is zero
        if result is None:
            R["area"].append(0.0)
            R["area_err"].append(0.0)
            R["ratio"].append(0.0)
            R["ratio_err"].append(0.0)
            R["chi2"].append(np.nan)  # no fit -> chi2 undefined
            continue
        fit, k = result
        a = fit.peak_info[k]["area"]
        a_err = fit.peak_err[k]["area_err"]
        cont = fit.continuum  # summed background-continuum counts over the window
        R["area"].append(a)
        R["area_err"].append(a_err)
        R["chi2"].append(fit.fit_result.redchi)
        R["ratio"].append(a / cont if cont > 0 else np.nan)
        R["ratio_err"].append(a_err / cont if cont > 0 else np.nan)

    for key in ("tc", "area", "area_err", "ratio", "ratio_err", "chi2",
                "area_lin", "area_lin_err", "snr"):
        R[key] = np.array(R[key])
    R["area_norm"], R["area_norm_err"] = _normalize(R["area"], R["area_err"])
    R["area_lin_norm"], R["area_lin_norm_err"] = _normalize(R["area_lin"], R["area_lin_err"])
    return R


def plot_grid(name, cfg, R):
    """Per-slice fit grid for one line."""
    ncols = 3
    nrows = int(np.ceil(no_slices / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 4 * nrows),
                             num=f"{name} fit per time slice")
    axes = np.atleast_1d(axes).ravel()
    for ax, (t0, t1, ew, cw, result, pa) in zip(axes, R["slice_fits"]):
        ax.plot(ew, cw, "o", ms=4, alpha=0.5, color="C0", label="data")
        if result is not None:
            fit, k = result
            xf = np.linspace(ew[0], ew[-1], 400)
            yf = fit.fit_result.eval(x=xf)
            bkg = fit.fit_result.eval_components(x=xf)[fit.bkg_key]
            ax.plot(xf, yf, color="C3", lw=2, label="Gauss fit")
            ax.plot(xf, bkg, "--", color="C2", lw=2, label="Gauss bkg")
        if pa is not None:
            ax.plot(pa._x_full_range, pa.y_eqn, "-.", color="C4", lw=2, label="linear bkg")
            ax.axvspan(pa.prange[0], pa.prange[1], color="C4", alpha=0.06)
        ax.set_title(f"t = [{t0:.1f}, {t1:.1f}] ns", fontsize=11)
        ax.axvline(cfg["e"], color="grey", ls=":", lw=1)
    for ax in axes[no_slices:]:
        ax.axis("off")
    axes[0].legend(fontsize=9)
    fig.supxlabel("Energy (keV)")
    fig.supylabel("Counts")
    fig.tight_layout()


def plot_metrics(name, R):
    """Area, normalized area, and reduced chi2 vs time for one line."""
    fig, (axa, axn, axc) = plt.subplots(3, 1, sharex=True, figsize=(16, 16),
                                        num=f"{name} area, normalized area, chi2 vs time")
    axa.errorbar(R["tc"], R["area"], yerr=R["area_err"], marker="s", lw=2, capsize=4,
                 color="C0", label="Gaussian fit")
    axa.errorbar(R["tc"], R["area_lin"], yerr=R["area_lin_err"], marker="o", lw=2, capsize=4,
                 color="C4", label="linear ROI")
    axa.set_ylabel(f"{name} peak area (counts)")
    axa.legend()

    axn.errorbar(R["tc"], R["area_norm"], yerr=R["area_norm_err"], marker="s", lw=2, capsize=4,
                 color="C0", label="Gaussian fit")
    axn.errorbar(R["tc"], R["area_lin_norm"], yerr=R["area_lin_norm_err"], marker="o", lw=2, capsize=4,
                 color="C4", label="linear ROI")
    axn.set_ylabel("Normalized area")
    axn.legend()

    axc.plot(R["tc"], R["chi2"], marker="s", lw=2, color="C1")
    axc.axhline(1.0, color="grey", ls="--", lw=1)
    axc.set_ylabel(r"Reduced $\chi^2$")
    axc.set_xlabel("Time (ns)")
    fig.tight_layout()


def plot_ratios_to_ref(results, ref="Si"):
    """Area of every other line normalized to the reference line vs time,
    for both the Gaussian-fit and linear-ROI estimators."""
    tc = results[ref]["tc"]
    others = [n for n in results if n != ref]
    fig, (axg, axl) = plt.subplots(
        2, 1, sharex=True, figsize=(16, 12),
        num=f"Area ratios to {ref} vs time")
    for name, color in zip(others, ("C0", "C3", "C5", "C6")):
        rg, rge = ratio_to_ref(results[name]["area"], results[name]["area_err"],
                               results[ref]["area"], results[ref]["area_err"])
        axg.errorbar(tc, rg, yerr=rge, marker="s", lw=2, capsize=4,
                     color=color, label=f"{name}/{ref}")
        rl, rle = ratio_to_ref(results[name]["area_lin"], results[name]["area_lin_err"],
                               results[ref]["area_lin"], results[ref]["area_lin_err"])
        axl.errorbar(tc, rl, yerr=rle, marker="o", lw=2, capsize=4,
                     color=color, label=f"{name}/{ref}")
    axg.set_ylabel(f"Area ratio to {ref}\n(Gaussian fit)")
    axg.legend()
    axl.set_ylabel(f"Area ratio to {ref}\n(linear ROI)")
    axl.set_xlabel("Time (ns)")
    axl.legend()
    fig.tight_layout()


def plot_snr(results):
    """Peak SNR vs time for every line, overlaid on one axis."""
    plt.figure("SNR vs time", figsize=(16, 8))
    for name, color in zip(results, ("C0", "C1", "C2", "C3", "C4")):
        R = results[name]
        plt.plot(R["tc"], R["snr"], marker="s", lw=2, color=color, label=name)
    plt.axhline(3, color="grey", ls="--", lw=1, label="min_snr = 3")
    plt.xlabel("Time (ns)")
    plt.ylabel("Peak SNR")
    plt.legend()
    plt.tight_layout()


# --- Overlaid spectra per time slice ---
cmap = plt.get_cmap("plasma")
colors = cmap(np.linspace(0, 1, no_slices))
plt.figure("Energy spectra per time slice", figsize=(22, 14))
for j, s in enumerate(slices):
    plt.plot(s["ergs"], s["cts"], color=colors[j], lw=1.5, alpha=0.7,
             label=f"t = [{s['t0']:.1f}, {s['t1']:.1f}] ns")
plt.xlabel("Energy (keV)")
plt.ylabel("Counts")
plt.legend(fontsize=10, ncol=2)
plt.tight_layout()

# --- Process and plot each line ---
results = {}
for name, cfg in lines.items():
    R = process_line(cfg)
    results[name] = R
    plot_grid(name, cfg, R)
    plot_metrics(name, R)

# Mg and Fe areas normalized to Si, vs time
plot_ratios_to_ref(results, ref="Si")

# Peak SNR for Si, Mg, Fe vs time
plot_snr(results)

plt.show()
