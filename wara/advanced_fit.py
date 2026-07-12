"""
Advanced fitting classes and functions
"""
import lmfit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mplcursors
from scipy.special import erfc, erfcx
from wara import spectrum as sp
from wara.matplotlib_theme import apply_theme
from wara.peakfit import PeakFit


class BkgFitResult:
    """Result of a background-only polynomial fit over an ROI."""
    __slots__ = ("coeffs", "poly", "x", "y", "degree",
                 "area_fit", "area_fit_err", "area_raw", "area_raw_err")

    def __init__(self, x, y, degree=1):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.degree = int(degree)

        self.coeffs = np.polyfit(self.x, self.y, self.degree)
        self.poly = np.poly1d(self.coeffs)

        # Trapezoidal area under the fit, in *channel* space (dx=1 bin) rather
        # than against the calibrated energy axis: each channel is one bin of
        # width 1 regardless of its keV/channel dispersion, so integrating
        # against energy (the original np.trapz(y_fit, self.x)) scaled the
        # result by that dispersion and made it incomparable to area_raw.
        # Note this deliberately is NOT np.sum(y_fit): for any least-squares
        # polynomial fit with a constant term, sum(y_fit) is *exactly*
        # sum(self.y) by construction (OLS residuals always sum to zero),
        # which would make area_fit a trivial echo of area_raw regardless of
        # fit quality. The trapezoidal rule under-weights the two edge points
        # by half their value, so area_fit is close to area_raw but not
        # identical — a real (if small) distinction between "area under the
        # fitted trapezoid" and "sum of the raw bins".
        y_fit = self.poly(self.x)
        self.area_fit = float(np.trapz(y_fit, dx=1.0))
        self.area_fit_err = float(np.sqrt(np.abs(self.area_fit)))

        self.area_raw = float(np.sum(self.y))
        self.area_raw_err = float(np.sqrt(self.area_raw))


def fit_bkg(spectrum, xrange, degree=1):
    """Fit a pure polynomial background (no peaks) over *xrange*.

    Parameters
    ----------
    spectrum : Spectrum
        A wara Spectrum object.
    xrange : list
        ``[lo, hi]`` bounds in the spectrum's x-axis units.
    degree : int
        Polynomial degree (1 = linear).

    Returns
    -------
    BkgFitResult
    """
    xs = spectrum.energies if spectrum.energies is not None else spectrum.channels
    ys = spectrum.counts
    mask = (xs >= xrange[0]) & (xs <= xrange[1])
    x, y = xs[mask], ys[mask]
    if len(x) < 2:
        raise ValueError("Not enough data points in range for background fit")
    return BkgFitResult(x, y, degree=degree)


class PeakAreaLinearBkg:
    def __init__(self, spectrum, x1, x2):
        if not isinstance(spectrum, sp.Spectrum):
            raise Exception("spectrum must be a Spectrum object")
        self.spect = spectrum
        # results — populated by calculate_peak_area
        self.A = 0
        self.B = 0
        self.sigA = 0
        self.sigB = 0
        self.y_eqn = None
        self.y_eqn_peak = None
        self.prange = None
        self.pchrange = None
        self.xr = None
        self.yr = None
        self._slope = None
        self._intercept = None

        # ROI bounds — used for initial plot before calculate_peak_area is called
        self._ch_roi_l = self._x_to_ch(x1, self.spect)
        self._ch_roi_r = self._x_to_ch(x2, self.spect)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _x_to_ch(x, spect):
        """Convert a single x value (channel or energy) to a channel index."""
        if spect.energies is None:
            return int(x)
        else:
            return int(spect.channels[spect.energies >= x][0])

    def _collect_bkg_points(self, x_input):
        """
        Given a scalar or 2-element list, return all x/y points for background fitting.

        Parameters
        ----------
        x_input : scalar or 2-element list
            Single point or range [a, b].

        Returns
        -------
        x_bkg : numpy array
            x-axis values of background points (channels or energies).
        y_bkg : numpy array
            Corresponding counts.
        """
        if np.isscalar(x_input):
            ch = self._x_to_ch(x_input, self.spect)
            x_val = x_input if self.spect.energies is not None else float(ch)
            x_bkg = np.array([x_val])
            y_bkg = np.array([self.spect.counts[ch]])
        else:
            ch_a = self._x_to_ch(x_input[0], self.spect)
            ch_b = self._x_to_ch(x_input[1], self.spect)
            if self.spect.energies is None:
                x_bkg = self.spect.channels[ch_a : ch_b + 1].astype(float)
            else:
                x_bkg = self.spect.energies[ch_a : ch_b + 1]
            y_bkg = self.spect.counts[ch_a : ch_b + 1]
        return x_bkg, y_bkg

    # ------------------------------------------------------------------
    # Main calculation
    # ------------------------------------------------------------------

    def calculate_peak_area(self, x1, x2):
        """
        Calculate the net peak area above a linear background.

        Parameters
        ----------
        x1 : scalar or 2-element list
            Left background region. Scalar = single point; list = range [a, b].
        x2 : scalar or 2-element list
            Right background region. Scalar = single point; list = range [a, b].
        """
        # --- collect background points from both sides ---
        x_bkg_l, y_bkg_l = self._collect_bkg_points(x1)
        x_bkg_r, y_bkg_r = self._collect_bkg_points(x2)

        # derive the four edges explicitly:
        #   outer_l  = leftmost of x1  (full plot left edge)
        #   inner_l  = rightmost of x1 (left boundary of peak region)
        #   inner_r  = leftmost of x2  (right boundary of peak region)
        #   outer_r  = rightmost of x2 (full plot right edge)
        ch_outer_l = self._x_to_ch(x1[0] if not np.isscalar(x1) else x1, self.spect)
        ch_inner_l = self._x_to_ch(x1[1] if not np.isscalar(x1) else x1, self.spect)
        ch_inner_r = self._x_to_ch(x2[0] if not np.isscalar(x2) else x2, self.spect)
        ch_outer_r = self._x_to_ch(x2[1] if not np.isscalar(x2) else x2, self.spect)

        # combine all background points and fit a line
        x_bkg_all = np.concatenate([x_bkg_l, x_bkg_r])
        y_bkg_all = np.concatenate([y_bkg_l, y_bkg_r])
        slope, intercept = np.polyfit(x_bkg_all, y_bkg_all, 1)
        self._slope = slope
        self._intercept = intercept

        # --- peak region: between the two inner edges ---
        self.pchrange = [ch_inner_l, ch_inner_r]
        if self.spect.energies is None:
            self.prange = [float(ch_inner_l), float(ch_inner_r)]
            self.xr = self.spect.channels[ch_inner_l : ch_inner_r + 1].astype(float)
            x_full_range = self.spect.channels[ch_outer_l : ch_outer_r + 1].astype(float)
        else:
            self.prange = [
                self.spect.energies[ch_inner_l],
                self.spect.energies[ch_inner_r],
            ]
            self.xr = self.spect.energies[ch_inner_l : ch_inner_r + 1]
            x_full_range = self.spect.energies[ch_outer_l : ch_outer_r + 1]
        self.yr = self.spect.counts[ch_inner_l : ch_inner_r + 1]

        # evaluate the fitted line over the full outer range (for plotting the line)
        self.y_eqn = slope * x_full_range + intercept

        # evaluate over the peak region (for area calculation and fill_between)
        self.y_eqn_peak = slope * self.xr + intercept

        # --- areas ---
        self.A = self.yr.sum() - self.y_eqn_peak.sum()
        self.B = self.y_eqn_peak.sum()

        # --- errors (Poisson) ---
        sigAB = np.sqrt(self.yr.sum())
        self.sigB = np.sqrt(np.abs(self.y_eqn_peak.sum()))
        self.sigA = np.sqrt(sigAB**2 + self.sigB**2)

        # store outer edges, full x range, and x-axis edge values for plotting
        self._ch_outer_l = ch_outer_l
        self._ch_outer_r = ch_outer_r
        self._x_full_range = x_full_range
        if self.spect.energies is None:
            self._x_outer_l = float(ch_outer_l)
            self._x_inner_l = float(ch_inner_l)
            self._x_inner_r = float(ch_inner_r)
            self._x_outer_r = float(ch_outer_r)
        else:
            self._x_outer_l = self.spect.energies[ch_outer_l]
            self._x_inner_l = self.spect.energies[ch_inner_l]
            self._x_inner_r = self.spect.energies[ch_inner_r]
            self._x_outer_r = self.spect.energies[ch_outer_r]

    # ------------------------------------------------------------------
    # Average background method
    # ------------------------------------------------------------------

    def calculate_peak_area_avg(self, x1, x2, gap=0):
        """
        Calculate the net peak area using per-side average backgrounds.

        The background line is anchored at the mean count level of each
        background range, pinned at the inner edges (plus any gap).

        Parameters
        ----------
        x1 : 2-element list
            Left background range [a, b]. Must be a range, not a scalar.
        x2 : 2-element list
            Right background range [c, d]. Must be a range, not a scalar.
        gap : float, optional
            Gap in x-units between the background ranges and the peak region.
            Defaults to 0 (background ranges touch the peak region directly).
        """
        if np.isscalar(x1) or np.isscalar(x2):
            raise ValueError(
                "x1 and x2 must be 2-element lists for calculate_peak_area_avg. "
                "A range is required to compute a meaningful average."
            )

        # --- outer edges (for plot range) ---
        ch_outer_l = self._x_to_ch(x1[0], self.spect)
        ch_outer_r = self._x_to_ch(x2[1], self.spect)

        # --- background regions ---
        ch_bkg_l0 = self._x_to_ch(x1[0], self.spect)
        ch_bkg_l1 = self._x_to_ch(x1[1], self.spect)
        ch_bkg_r0 = self._x_to_ch(x2[0], self.spect)
        ch_bkg_r1 = self._x_to_ch(x2[1], self.spect)

        y_left  = self.spect.counts[ch_bkg_l0 : ch_bkg_l1 + 1].mean()
        y_right = self.spect.counts[ch_bkg_r0 : ch_bkg_r1 + 1].mean()

        # --- peak region inner edges, shifted inward by gap ---
        ch_inner_l = self._x_to_ch(x1[1] + gap, self.spect)
        ch_inner_r = self._x_to_ch(x2[0] - gap, self.spect)

        if ch_inner_l >= ch_inner_r:
            raise ValueError(
                "Gap is too large — peak region has collapsed. Reduce the gap value."
            )

        # --- x-axis arrays ---
        if self.spect.energies is None:
            x_inner_l = float(ch_inner_l)
            x_inner_r = float(ch_inner_r)
            self.xr  = self.spect.channels[ch_inner_l : ch_inner_r + 1].astype(float)
            x_full_range = self.spect.channels[ch_outer_l : ch_outer_r + 1].astype(float)
        else:
            x_inner_l = self.spect.energies[ch_inner_l]
            x_inner_r = self.spect.energies[ch_inner_r]
            self.xr  = self.spect.energies[ch_inner_l : ch_inner_r + 1]
            x_full_range = self.spect.energies[ch_outer_l : ch_outer_r + 1]

        self.yr = self.spect.counts[ch_inner_l : ch_inner_r + 1]
        self.pchrange = [ch_inner_l, ch_inner_r]
        self.prange   = [x_inner_l, x_inner_r]

        # --- background line anchored at (x_inner_l, y_left) and (x_inner_r, y_right) ---
        slope     = (y_right - y_left) / (x_inner_r - x_inner_l)
        intercept = y_left - slope * x_inner_l
        self._slope     = slope
        self._intercept = intercept

        # evaluate over full outer range (plotting) and peak region (area)
        self.y_eqn      = slope * x_full_range + intercept
        self.y_eqn_peak = slope * self.xr      + intercept

        # --- areas ---
        self.A = self.yr.sum() - self.y_eqn_peak.sum()
        self.B = self.y_eqn_peak.sum()

        # --- errors (Poisson) ---
        sigAB       = np.sqrt(self.yr.sum())
        self.sigB   = np.sqrt(np.abs(self.y_eqn_peak.sum()))
        self.sigA   = np.sqrt(sigAB**2 + self.sigB**2)

        # --- store edges for plotting ---
        self._ch_outer_l  = ch_outer_l
        self._ch_outer_r  = ch_outer_r
        self._x_full_range = x_full_range
        if self.spect.energies is None:
            self._x_outer_l = float(ch_outer_l)
            self._x_inner_l = float(ch_inner_l)
            self._x_inner_r = float(ch_inner_r)
            self._x_outer_r = float(ch_outer_r)
            self._x_bkg_l1  = float(ch_bkg_l1)   # inner edge of left bkg range (pre-gap)
            self._x_bkg_r0  = float(ch_bkg_r0)   # inner edge of right bkg range (pre-gap)
        else:
            self._x_outer_l = self.spect.energies[ch_outer_l]
            self._x_inner_l = self.spect.energies[ch_inner_l]
            self._x_inner_r = self.spect.energies[ch_inner_r]
            self._x_outer_r = self.spect.energies[ch_outer_r]
            self._x_bkg_l1  = self.spect.energies[ch_bkg_l1]
            self._x_bkg_r0  = self.spect.energies[ch_bkg_r0]

    
    def plot(self, ax=None, areas=False):
        plt.rc("font", size=14)
        apply_theme()
        if ax is None:
            fig = plt.figure(figsize=(10, 6))
            fig.patch.set_alpha(0.3)
            ax = fig.add_subplot()

        # use background fit outer edges if available, otherwise fall back to ROI bounds
        ch_l = self._ch_outer_l if self.prange is not None else self._ch_roi_l
        ch_r = self._ch_outer_r if self.prange is not None else self._ch_roi_r

        if self.spect.energies is None:
            x_full = self.spect.channels[ch_l : ch_r + 1].astype(float)
        else:
            x_full = self.spect.energies[ch_l : ch_r + 1]
        y_full = self.spect.counts[ch_l : ch_r + 1]

        line = ax.plot(x_full, y_full, drawstyle="steps")

        if areas:
            if self.prange is None:
                raise RuntimeError("Call calculate_peak_area() before plot(areas=True).")
            ax.plot(self._x_full_range, self.y_eqn, color="C1", label="Linear background fit")
            ax.fill_between(
                x=self._x_full_range, y1=0, y2=self.y_eqn,
                step="pre", alpha=0.2, color="r",
                label=f"B = {round(self.B, 3)}",
            )
            ax.fill_between(
                x=self.xr, y1=self.y_eqn_peak, y2=self.yr,
                step="pre", alpha=0.2, color="g",
                label=f"A = {round(self.A, 3)}",
            )
            # draw boundary lines — stop at background line height at each x
            y_at_outer_l = self._slope * self._x_outer_l + self._intercept
            y_at_inner_l = self._slope * self._x_inner_l + self._intercept
            y_at_inner_r = self._slope * self._x_inner_r + self._intercept
            y_at_outer_r = self._slope * self._x_outer_r + self._intercept

            if self._x_outer_l != self._x_inner_l:
                ax.vlines(self._x_outer_l, 0, y_at_outer_l, linestyle="dotted",
                          color="gray", lw=2,
                          label=f"x1 range: [{round(self._x_outer_l,3)}, {round(self._x_inner_l,3)}]")
                ax.vlines(self._x_inner_l, 0, y_at_inner_l, linestyle="dotted",
                          color="gray", lw=2)
            else:
                ax.vlines(self._x_inner_l, 0, y_at_inner_l, linestyle="dotted",
                          color="gray", lw=2,
                          label=f"x1 = {round(self._x_inner_l, 3)}")
            if self._x_inner_r != self._x_outer_r:
                ax.vlines(self._x_inner_r, 0, y_at_inner_r, linestyle="dotted",
                          color="C1", lw=2,
                          label=f"x2 range: [{round(self._x_inner_r,3)}, {round(self._x_outer_r,3)}]")
                ax.vlines(self._x_outer_r, 0, y_at_outer_r, linestyle="dotted",
                          color="C1", lw=2)
            else:
                ax.vlines(self._x_inner_r, 0, y_at_inner_r, linestyle="dotted",
                          color="C1", lw=2,
                          label=f"x2 = {round(self._x_inner_r, 3)}")

            # if gap was used, also draw the pre-gap bkg inner edges
            if hasattr(self, "_x_bkg_l1") and self._x_bkg_l1 != self._x_inner_l:
                y_at_bkg_l1 = self._slope * self._x_bkg_l1 + self._intercept
                y_at_bkg_r0 = self._slope * self._x_bkg_r0 + self._intercept
                ax.vlines(self._x_bkg_l1, 0, y_at_bkg_l1, linestyle="dotted",
                          color="gray", lw=2, alpha=0.5,
                          label=f"gap: {round(self._x_inner_l - self._x_bkg_l1, 3)}")
                ax.vlines(self._x_bkg_r0, 0, y_at_bkg_r0, linestyle="dotted",
                          color="C1", lw=2, alpha=0.5)
            ax.legend(loc="upper right")

        mplcursors.cursor(line, hover=True)
        ax.set_yscale("linear")
        ax.set_xlabel(self.spect.x_units)
        ax.set_ylabel(self.spect.y_label)
        plt.show()


# ---------------------------------------------------------------------------
# Extra peak-profile families on top of the basic PeakFit Gaussian model.
# ---------------------------------------------------------------------------

# Each value is a factory ``f(prefix) -> lmfit model``. lmfit's built-in
# model classes already satisfy that (``Cls(prefix=...)``); the "emg" entry
# wraps a custom *left*-tailed EMG (``emg_left_peak``, defined below) because
# lmfit's ExponentialGaussianModel tails toward high energy — the wrong way
# for the HPGe low-energy tail.
_PROFILE_MODELS = {
    "gauss":       lmfit.models.GaussianModel,
    "voigt":       lmfit.models.VoigtModel,
    "pvoigt":      lmfit.models.PseudoVoigtModel,
    "skewed":      lmfit.models.SkewedGaussianModel,
    "skewedvoigt": lmfit.models.SkewedVoigtModel,
    "emg":         lambda prefix: lmfit.Model(emg_left_peak, prefix=prefix),
    "doniach":     lmfit.models.DoniachModel,
}


_DOPPLER_KEYS = {"energy", "fwhm", "min_fwhm", "max_fwhm",
                 "replace_narrow", "suppress_fwhm", "drift"}


def _normalize_doppler(doppler):
    """
    Normalize the ``doppler`` kwarg to a list of spec dicts.

    Accepts ``None``, a single energy (number), a single spec dict, or a
    sequence mixing energies and dicts. Each returned dict has at least an
    ``"energy"`` key. Raises ValueError on unknown dict keys or a missing
    energy.
    """
    if doppler is None:
        return []
    if isinstance(doppler, dict) or not np.iterable(doppler):
        doppler = [doppler]

    specs = []
    for entry in doppler:
        if isinstance(entry, dict):
            unknown = set(entry) - _DOPPLER_KEYS
            if unknown:
                raise ValueError(
                    f"Unknown doppler key(s) {sorted(unknown)}; "
                    f"allowed: {sorted(_DOPPLER_KEYS)}."
                )
            if "energy" not in entry:
                raise ValueError("Each doppler dict needs an 'energy' key.")
            specs.append(dict(entry))
        else:
            specs.append({"energy": float(entry)})
    return specs


class MultiProfilePeakFit(PeakFit):
    """
    PeakFit variant supporting additional line shapes:

    * ``"gauss"``       — plain Gaussian (same as :class:`PeakFit`).
    * ``"voigt"``       — Voigt (Gaussian * Lorentzian).
    * ``"pvoigt"``      — Pseudo-Voigt (weighted G+L sum).
    * ``"skewed"``      — SkewedGaussian (low-energy tailing).
    * ``"skewedvoigt"`` — SkewedVoigt (asymmetric Voigt; tailing + Lorentzian
      wings).
    * ``"emg"``         — Exponentially-Modified Gaussian: a Gaussian
      convolved with a one-sided exponential. A principled, area-normalised
      model for the low-energy tail of HPGe full-energy peaks — usually the
      best single-component choice when a plain Gaussian leaves structured
      residuals on the low-energy flank.
    * ``"doniach"``     — Doniach-Sunjic (photoemission asymmetry).

    Parameters
    ----------
    profile : str
        Line-shape family. Default ``"voigt"``.
    doppler : float | dict | sequence, optional
        Add one or more *broad Doppler-broadened companion* peaks on top of
        the narrow peaks found by the search — the case of a wide,
        recoil-broadened gamma line underlying sharp lines (Evans et al.,
        2006, e.g. the 2211 keV Al and 2230 keV S lines around the 2223 keV
        hydrogen peak). Each companion is built as an ordinary Gaussian-
        family component, but with a broad width seed, a width *floor* so it
        cannot collapse onto a narrow peak, and a centroid that is free to
        drift within the fit window (Doppler peaks are often shifted slightly
        in energy). Companions are exempt from ``shared_sigma``.

        Accepts a single energy, a sequence of energies, or a sequence of
        dicts for finer control. Dict keys (all but ``energy`` optional)::

            {"energy": 2211,      # companion centroid seed (required)
             "fwhm": 8.0,         # width seed in x-units (default ~3x narrow)
             "min_fwhm": 4.0,     # width floor    (default ~1.5x narrow)
             "max_fwhm": 25.0,    # width ceiling  (default: unbounded)
             "replace_narrow": True,  # drop auto-detected narrow peaks the
                                  #   companion swallows (for broad-ONLY lines;
                                  #   leave False for a true narrow-on-broad
                                  #   overlap such as the 6129 keV O line)
             "suppress_fwhm": 4}  # replace_narrow radius in x-units
                                  #   (default: one narrow FWHM)

    peaks : sequence of float, optional
        Explicit peak centroids (in x-units). When given, these *replace* the
        automatic peak search — the user declares exactly how many peaks the
        region holds and roughly where, which is essential for overlapping or
        weak lines the finder merges or misses. Width and amplitude are seeded
        automatically and refined by the fit (constrain them with ``fix_fwhm``
        / ``fix_center`` / ``ratios`` if needed)::

            peaks=[1173.2, 1332.5]   # fit exactly these two lines

    fix_center : dict, optional
        Constrain peak centroids, keyed by approximate energy (the peak
        nearest each key is selected). The value is either an exact centroid
        to pin to, or ``True`` to freeze the peak at its fitted/seeded
        position::

            fix_center={511.0: 511.0,   # pin this peak to exactly 511.0
                        1173.2: True}   # freeze at the detected position

    fix_fwhm : dict, optional
        Constrain peak widths, keyed by approximate energy. The value is a
        single FWHM (in x-units) to fix the width to, or a ``(min, max)``
        tuple to bound it::

            fix_fwhm={511.0: 3.0,          # pin FWHM to 3.0 keV
                      1332.5: (2.0, 4.0)}  # keep FWHM within 2-4 keV

        The width is the *Gaussian-sigma* FWHM (exact for ``profile="gauss"``;
        the Gaussian component for Voigt/EMG/etc.).
    ratios : sequence, optional
        Tie peak amplitudes (net areas) to a fixed ratio. Each entry is
        ``(numerator_energy, denominator_energy, r)`` and imposes
        ``area(numerator) = r * area(denominator)`` — e.g. a known branching
        ratio or escape-peak fraction::

            ratios=[(1173.2, 1332.5, 1.0)]   # equal-area doublet

        For equal-width peaks the area ratio equals the height ratio.
    *args, **kwargs
        Forwarded to :class:`wara.peakfit.PeakFit`. Per-parameter ``hints``
        still work and take precedence over the constraint kwargs above. The
        legacy ``skew=True`` kwarg is honoured for backwards compatibility but
        ``profile="skewed"`` is the preferred spelling.

    Notes
    -----
    ``shared_sigma=True`` constrains every peak's *Gaussian* sigma via
    ``fwhm = a + b*sqrt(E)``. For Voigt / PseudoVoigt / SkewedVoigt / EMG the
    *total* FWHM also depends on gamma (or the tail rate), so the shared
    constraint is approximate (it only links the Gaussian component).
    Override the extra parameters via ``hints`` if you need finer control.
    The reported ``peak_info["fwhm"]`` is the Gaussian-sigma FWHM; use
    :func:`shape_summary` / :func:`shape_metrics` for the true measured
    width and tailing of these asymmetric shapes.
    """

    def __init__(self, search, xrange, *args, profile="voigt", doppler=None,
                 peaks=None, fix_center=None, fix_fwhm=None, ratios=None,
                 **kwargs):
        profile = profile.lower()
        if profile not in _PROFILE_MODELS:
            raise ValueError(
                f"Unknown profile {profile!r}. "
                f"Choose from {sorted(_PROFILE_MODELS)}."
            )
        self.profile = profile
        # Explicit narrow-peak centroids (override the automatic peak search).
        self._peaks = [float(e) for e in peaks] if peaks is not None else None
        self._doppler_specs = _normalize_doppler(doppler)
        # Energy-referenced peak constraints, translated to parameter hints in
        # _inject_constraint_peaks (once the peak centroids are known).
        self._fix_center = dict(fix_center) if fix_center else {}
        self._fix_fwhm = dict(fix_fwhm) if fix_fwhm else {}
        self._ratios = list(ratios) if ratios else []
        # Filled in _inject_constraint_peaks: prefix -> width/centroid bounds
        # to apply in _set_peak_initial_values.
        self._doppler_bounds = {}
        # Filled in _inject_constraint_peaks when peaks= is given: prefix ->
        # (lo, hi) centroid window keeping each declared peak near its seed so
        # the optimizer can't collapse a doublet onto a single line.
        self._user_center_bounds = {}
        # The "skewed" profile is also reachable via skew=True on the
        # parent class; pick consistent behaviour without surprising the
        # parent's gamma seeding.
        if profile == "skewed":
            kwargs.setdefault("skew", True)
        super().__init__(search, xrange, *args, **kwargs)

    def find_peaks_range(self):
        # A region may legitimately contain no auto-detected narrow peaks when
        # the user supplies the peaks explicitly (``peaks=``) or declares only
        # broad Doppler companions (a wide, low line the finder skips). Fall
        # back to an empty narrow set so the fit can still be built.
        try:
            return super().find_peaks_range()
        except ValueError:
            if self._peaks is not None or self._doppler_specs:
                none = np.zeros(len(self.search.peaks_idx), dtype=bool)
                return none, self.search.peaks_idx[none]
            raise

    def _make_peak_component(self, prefix):
        # Doppler broadening reflects a velocity distribution, so the broad
        # companion is intrinsically a symmetric Gaussian — never the (possibly
        # tailed) narrow-peak profile, which is pathological at large width.
        if prefix in self._doppler_bounds:
            return lmfit.models.GaussianModel(prefix=prefix)
        return _PROFILE_MODELS[self.profile](prefix=prefix)

    def _set_peak_initial_values(self, pars, prefix, center, sigma, amplitude):
        bounds = self._doppler_bounds.get(prefix)
        if bounds is not None:
            # Broad Gaussian companion: floor the width (so it can't masquerade
            # as a narrow peak), cap it if asked, and let the centroid drift
            # only within its bounded window.
            sig_min, sig_max, lo, hi = bounds
            pars[f"{prefix}center"].set(value=float(center), min=lo, max=hi)
            pars[f"{prefix}sigma"].set(value=float(sigma), min=sig_min,
                                       max=sig_max)
            pars[f"{prefix}amplitude"].set(value=float(amplitude), min=0.0)
            return

        super()._set_peak_initial_values(pars, prefix, center, sigma, amplitude)
        cbounds = self._user_center_bounds.get(prefix)
        if cbounds is not None:
            lo, hi = cbounds
            pars[f"{prefix}center"].set(value=float(center), min=lo, max=hi)
        if self.profile == "emg":
            tname = f"{prefix}tau"
            if tname in pars:
                pars[tname].set(value=float(max(sigma, 1e-3)), min=1e-3)
        if self.profile == "doniach":
            pars[f"{prefix}sigma"].set(min=1e-6)
            pars[f"{prefix}gamma"].set(value=0.1, min=0.0, max=1.0)

    def _inject_constraint_peaks(self, erg, sig, amp):
        """Append broad Doppler companions, then translate energy-referenced
        ``fix_center`` / ``fix_fwhm`` / ``ratios`` constraints into hints."""
        erg = np.asarray(erg, dtype=float)
        sig = np.asarray(sig, dtype=float)
        amp = np.asarray(amp, dtype=float)

        # Reference narrow width: median of the search seeds, or a small
        # fraction of the window if the search found nothing. Computed before
        # any replace_narrow culling (and before user peaks replace the seeds)
        # so the width seeds stay stable.
        if sig.size:
            narrow_sig = float(np.median(sig))
        else:
            narrow_sig = (self.xrange[1] - self.xrange[0]) / 40.0

        # Explicit ``peaks=`` replace the automatic detections entirely — the
        # user is declaring exactly how many peaks the region has and where.
        if self._peaks is not None:
            erg, sig, amp = self._seed_user_peaks(narrow_sig)
            self._bound_user_centers(erg, narrow_sig)

        if not self._doppler_specs:
            self._apply_peak_constraints(erg, narrow_sig)
            return erg, sig, amp

        # ``replace_narrow`` companions absorb any auto-detected narrow peak
        # within ``suppress_fwhm`` of their centroid — for a broad-only line
        # (e.g. the Al/S Doppler bumps) the peak finder otherwise latches onto
        # the broad apex as if it were a narrow peak and the two collide.
        keep = np.ones(len(erg), dtype=bool)
        for spec_d in self._doppler_specs:
            if not spec_d.get("replace_narrow"):
                continue
            e0 = float(spec_d["energy"])
            # Default radius is ONE narrow FWHM: wide enough to drop the false
            # narrow detection sitting on the broad apex, narrow enough to
            # spare genuine narrow neighbours (e.g. a line a few keV away).
            radius = spec_d.get("suppress_fwhm")
            if radius is None:
                radius = 2.355 * narrow_sig
            keep &= np.abs(erg - e0) > radius
        erg, sig, amp = erg[keep], sig[keep], amp[keep]
        n0 = len(erg)

        spec = self.search.spectrum
        xs = self.x
        ys = spec.counts
        win = (xs >= self.xrange[0]) & (xs <= self.xrange[1])
        base = float(ys[win].min()) if np.any(win) else 0.0

        new_e, new_s, new_a = [], [], []
        for k, spec_d in enumerate(self._doppler_specs):
            energy = float(spec_d["energy"])
            # Width seed and bounds (stored in x-units as sigma).
            sig_seed = (spec_d["fwhm"] / 2.355 if spec_d.get("fwhm")
                        else 3.0 * narrow_sig)
            sig_min = (spec_d["min_fwhm"] / 2.355 if spec_d.get("min_fwhm")
                       else 1.5 * narrow_sig)
            sig_max = (spec_d["max_fwhm"] / 2.355 if spec_d.get("max_fwhm")
                       else None)
            # Amplitude seed from the local peak height above the window floor.
            idx = int(np.argmin(np.abs(xs - energy)))
            height = max(float(ys[idx]) - base, 1.0)
            amp_seed = 0.5 * height * sig_seed * np.sqrt(2.0 * np.pi)

            # Centroid is free to drift (Doppler lines are often shifted a
            # little in energy) but only within +/- ``drift`` of the seed, so
            # a weak companion cannot wander onto a strong neighbour. Default
            # drift is one seed sigma; clamp to the fit window.
            drift = spec_d.get("drift", sig_seed)
            lo = max(float(self.xrange[0]), energy - drift)
            hi = min(float(self.xrange[1]), energy + drift)

            prefix = f"g{n0 + k + 1}_"
            self._doppler_bounds[prefix] = (sig_min, sig_max, lo, hi)
            new_e.append(energy)
            new_s.append(sig_seed)
            new_a.append(amp_seed)
            # This companion's sigma must stay free of the resolution curve.
            self._free_sigma_idx.add(n0 + k)

        erg = np.concatenate([erg, new_e])
        sig = np.concatenate([sig, new_s])
        amp = np.concatenate([amp, new_a])
        self._apply_peak_constraints(erg, narrow_sig)
        return erg, sig, amp

    def _seed_user_peaks(self, narrow_sig):
        """Build (center, sigma, amplitude) seeds from explicit ``peaks=``."""
        spec = self.search.spectrum
        xs = self.x
        ys = spec.counts
        win = (xs >= self.xrange[0]) & (xs <= self.xrange[1])
        base = float(ys[win].min()) if np.any(win) else 0.0
        erg, sig, amp = [], [], []
        for energy in sorted(self._peaks):
            idx = int(np.argmin(np.abs(xs - energy)))
            height = max(float(ys[idx]) - base, 1.0)
            erg.append(energy)
            sig.append(narrow_sig)
            amp.append(height * narrow_sig * np.sqrt(2.0 * np.pi))
        return np.asarray(erg), np.asarray(sig), np.asarray(amp)

    def _bound_user_centers(self, erg, narrow_sig):
        """Pin each explicitly declared peak (``peaks=``) to a window around
        its seed centroid.

        With centres left fully free the optimizer can land in a degenerate
        minimum where two declared lines pile onto the same strong peak and the
        weak companion is never fitted. The user asserted a peak sits at each
        energy, so we let the centroid drift only within the smaller of one
        narrow FWHM or half the spacing to the nearest declared neighbour --
        wide enough to absorb a small mis-call, tight enough that two windows
        never cross. ``fix_center`` hints (applied later) still override this.
        """
        erg = np.asarray(erg, dtype=float)
        narrow_fwhm = 2.355 * narrow_sig
        # Peaks the user pins outright (fix_center, or a g{i}_center hint) must
        # be left for that pin alone -- a window here would clamp the pinned
        # value back to the boundary.
        pinned = set()
        for fe in self._fix_center:
            pinned.add(int(np.argmin(np.abs(erg - float(fe)))))
        for i in range(len(erg)):
            if f"g{i + 1}_center" in self.hints:
                pinned.add(i)
        for i, energy in enumerate(erg):
            if i in pinned:
                continue
            gaps = []
            if i > 0:
                gaps.append(energy - erg[i - 1])
            if i < len(erg) - 1:
                gaps.append(erg[i + 1] - energy)
            drift = narrow_fwhm if not gaps else min(narrow_fwhm, 0.5 * min(gaps))
            lo = max(float(self.xrange[0]), energy - drift)
            hi = min(float(self.xrange[1]), energy + drift)
            self._user_center_bounds[f"g{i + 1}_"] = (lo, hi)

    def _apply_peak_constraints(self, centers, narrow_sig):
        """
        Translate the energy-referenced ``fix_center`` / ``fix_fwhm`` /
        ``ratios`` kwargs into per-parameter ``hints``.

        ``centers`` are the seeded peak centroids (narrow peaks first, then
        Doppler companions), so peak ``i`` carries the ``g{i+1}_`` prefix. A
        constraint energy is matched to the nearest centroid; explicit
        user-supplied ``hints`` always win (we only fill keys not already set).
        """
        if not (self._fix_center or self._fix_fwhm or self._ratios):
            return
        centers = np.asarray(centers, dtype=float)
        if centers.size == 0:
            raise ValueError(
                "Peak constraints were given but the fit has no peaks to "
                "constrain (the search found none and no doppler= was set)."
            )
        # Match within ~3 narrow FWHM so a constraint can't silently attach to
        # a far-away peak.
        tol = 3.0 * 2.355 * narrow_sig

        def prefix_for(energy):
            i = int(np.argmin(np.abs(centers - float(energy))))
            if abs(centers[i] - float(energy)) > tol:
                raise ValueError(
                    f"No fitted peak within {tol:.2f} of {energy}; the nearest "
                    f"peak is at {centers[i]:.2f}. Check the energy or widen "
                    "the search."
                )
            return f"g{i + 1}_", i

        def fill(name, attrs):
            self.hints.setdefault(name, attrs)

        for energy, target in self._fix_center.items():
            prefix, i = prefix_for(energy)
            value = centers[i] if target is True else float(target)
            # expr="" clears any shared_sigma centroid coupling (harmless else).
            fill(f"{prefix}center", {"value": value, "vary": False, "expr": ""})

        for energy, spec in self._fix_fwhm.items():
            prefix, _ = prefix_for(energy)
            if isinstance(spec, (tuple, list)):
                lo, hi = spec
                fill(f"{prefix}sigma",
                     {"min": lo / 2.355, "max": hi / 2.355, "expr": ""})
            else:
                fill(f"{prefix}sigma",
                     {"value": float(spec) / 2.355, "vary": False, "expr": ""})

        for numerator, denominator, r in self._ratios:
            pnum, inum = prefix_for(numerator)
            pden, iden = prefix_for(denominator)
            if inum == iden:
                raise ValueError(
                    f"ratios: numerator ({numerator}) and denominator "
                    f"({denominator}) resolve to the same peak at "
                    f"{centers[inum]:.2f}. They must be two distinct peaks."
                )
            fill(f"{pnum}amplitude", {"expr": f"{float(r)}*{pden}amplitude"})


# ---------------------------------------------------------------------------
# Gaussian peak on a smoothed-step background.
# ---------------------------------------------------------------------------


def sharp_step_background(x, step_amplitude, step_center, offset):
    """
    Sharp (Heaviside) step plus a constant offset.

    Models the classic gamma-ray peak background as a hard discontinuity at
    the peak centroid: the continuum sits at a higher level on the
    low-energy side of a peak (from incomplete charge collection /
    small-angle Compton scattering) and drops abruptly to a lower level on
    the high-energy side::

        bkg(x) = offset + step_amplitude * H(center - x)

    where ``H`` is the Heaviside step (1 below the centroid, 0 above, 0.5
    at it). ``step_amplitude`` is the height of the step (high side minus
    low side) and ``offset`` is the high-energy plateau. This is the
    default background used by :class:`GaussStepFit`; for a continuous
    transition tied to the peak width use :func:`step_background`.

    Parameters
    ----------
    x : array_like
        Independent variable (channels or energies).
    step_amplitude : float
        Height of the step (non-negative for a normal Compton step).
    step_center : float
        Location of the discontinuity. Tied to the Gaussian peak by
        :class:`GaussStepFit`.
    offset : float
        Constant level on the high-energy side of the step.
    """
    return offset + step_amplitude * np.heaviside(
        step_center - np.asarray(x, dtype=float), 0.5
    )


def step_background(x, step_amplitude, step_center, step_sigma, offset):
    """
    Smoothed (error-function) step plus a constant offset.

    Like :func:`sharp_step_background`, but the discontinuity is replaced by
    a continuous transition that follows the integral of the Gaussian peak,
    so it is centred on the peak and has the same width::

        bkg(x) = offset + step_amplitude * 0.5 * erfc((x - center) / (sqrt(2) * sigma))

    For ``x << center`` the ``erfc`` term -> 2 so the level is
    ``offset + step_amplitude``; for ``x >> center`` it -> 0 so the level
    is ``offset``. ``step_amplitude`` is the height of the step (high side
    minus low side) and ``offset`` is the high-energy plateau. Selected via
    ``step="smooth"`` on :class:`GaussStepFit`.

    Parameters
    ----------
    x : array_like
        Independent variable (channels or energies).
    step_amplitude : float
        Height of the step (non-negative for a normal Compton step).
    step_center, step_sigma : float
        Centre and width of the transition. Tied to the Gaussian peak by
        :class:`GaussStepFit`.
    offset : float
        Constant level on the high-energy side of the step.
    """
    return offset + step_amplitude * 0.5 * erfc(
        (x - step_center) / (np.sqrt(2.0) * step_sigma)
    )


# Map the friendly ``step=`` keyword to the canonical bkg string stored on
# the instance (so save_json/load_json round-trip the choice through the
# parent's ``bkg`` field) and the model function used to build it.
_STEP_MODELS = {
    "sharp": ("step", sharp_step_background),
    "smooth": ("step-smooth", step_background),
}
# Reverse lookup for reconstructing ``step`` from a serialized ``bkg``.
_BKG_TO_STEP = {bkg: name for name, (bkg, _) in _STEP_MODELS.items()}


class GaussStepFit(PeakFit):
    """
    Fit Gaussian peak(s) on a step background.

    A drop-in alternative to :class:`wara.peakfit.PeakFit` for the common
    case where the continuum under a peak is not a straight line but a
    *step*: higher on the low-energy side, lower on the high-energy side,
    with the transition happening at the peak. Two step shapes are
    available:

    * ``step="sharp"`` (default) — a hard Heaviside discontinuity at the
      peak centroid (:func:`sharp_step_background`). Free background
      parameters: step height and high-energy offset.
    * ``step="smooth"`` — a continuous ``erfc`` transition tied to the
      peak's centre *and* width (:func:`step_background`).

    The public interface mirrors :class:`PeakFit` — ``peak_info``,
    ``peak_err``, ``summary()``, ``fit_quality()``, ``plot()`` and
    ``save_json()``/``load_json()`` all work unchanged. When more than one
    peak falls inside ``xrange`` every peak is fit with its own Gaussian
    and the single step is anchored to the most prominent one.

    Parameters
    ----------
    search : PeakSearch
        Provides the spectrum and the peak locations/widths (same as
        :class:`PeakFit`).
    xrange : list
        ``[lo, hi]`` window for the fit, in the spectrum's x-axis units
        (energies if calibrated, otherwise channels).
    step : {"sharp", "smooth"}, optional
        Step shape. Default ``"sharp"``.
    bkg : optional
        Accepted only so ``save_json``/``load_json`` round-trip (the parent
        serializes ``bkg``). When given a canonical step string it selects
        the matching ``step`` shape; otherwise ignored.
    skew : optional
        Accepted for interface compatibility with :class:`PeakFit` but
        ignored — peaks are always plain Gaussians.
    hints : dict, optional
        Per-parameter overrides, applied last. See :class:`PeakFit`.
    """

    def __init__(self, search, xrange, step="sharp", bkg=None, skew=False, *,
                 shared_sigma=False, hints=None):
        if shared_sigma:
            raise NotImplementedError(
                "shared_sigma is not supported for the step-background fit"
            )
        # When reconstructed by load_json the choice arrives via ``bkg``
        # (the field the parent serializes); translate it back to ``step``.
        if bkg is not None and bkg in _BKG_TO_STEP:
            step = _BKG_TO_STEP[bkg]
        if step not in _STEP_MODELS:
            raise ValueError(
                f"Unknown step {step!r}. Choose from {sorted(_STEP_MODELS)}."
            )
        self.step = step
        canonical_bkg, self._step_func = _STEP_MODELS[step]
        # Force the step background and a plain Gaussian regardless of skew.
        super().__init__(
            search, xrange, bkg=canonical_bkg, skew=False,
            shared_sigma=False, hints=hints,
        )

    def gaussians_bkg(self, skeweness):
        """
        Build and fit (one or more Gaussians) + step background.

        Overrides :meth:`PeakFit.gaussians_bkg`; the data preparation,
        weighting (``1/counts_err``, ``scale_covar=False``) and the
        ``peak_info``/``peak_err`` bookkeeping match the parent so the rest
        of the API behaves identically.
        """
        maskx = (self.x > self.xrange[0]) & (self.x < self.xrange[1])
        _, _, amp, erg, sig = self.init_values()
        npeaks = len(erg)

        # .copy() so we never mutate spectrum.counts (see PeakFit notes).
        y0 = self.search.spectrum.counts[maskx].copy()
        x0 = self.x[maskx]

        err0 = self.search.spectrum.counts_err[maskx].copy()
        err0[err0 <= 0] = 1.0

        y0_min = np.amin(y0[y0 != 0.0]) if np.any(y0 != 0.0) else 1.0
        y0[y0 == 0.0] = y0_min * 1e-1

        self.y_data = y0
        self.x_data = x0

        smooth = self.step == "smooth"

        # --- step + constant background -----------------------------------
        bkg_mod = lmfit.Model(self._step_func, prefix="bkg_")
        model = bkg_mod
        self.bkg_key = "bkg_"

        # Seed the step from the edge levels of the window: the high side
        # is the low-energy edge, the low side is the high-energy edge.
        edge = max(1, len(y0) // 10)
        left_level = float(np.mean(y0[:edge]))
        right_level = float(np.mean(y0[-edge:]))
        offset0 = right_level
        step0 = max(left_level - right_level, 0.0)

        # Anchor the step to the most prominent peak in the window.
        anchor = int(np.argmax(amp))
        anchor_prefix = f"g{anchor + 1}_"

        pars = bkg_mod.make_params()
        pars["bkg_offset"].set(value=offset0)
        pars["bkg_step_amplitude"].set(value=step0, min=0.0)
        # center (and, when smooth, sigma) get values now and are pinned to
        # the peak below.
        pars["bkg_step_center"].set(value=float(erg[anchor]))
        if smooth:
            pars["bkg_step_sigma"].set(
                value=float(max(sig[anchor], 1e-6)), min=1e-9
            )

        # --- Gaussian peak component(s) -----------------------------------
        for i in range(npeaks):
            prefix = f"g{i+1}_"
            comp = self._make_peak_component(prefix)
            pars.update(comp.make_params())
            self._set_peak_initial_values(pars, prefix, erg[i], sig[i], amp[i])
            model += comp

        # Tie the step's location (and width, when smooth) to the anchoring
        # Gaussian so the transition sits exactly under the peak.
        pars["bkg_step_center"].set(expr=f"{anchor_prefix}center")
        if smooth:
            pars["bkg_step_sigma"].set(expr=f"{anchor_prefix}sigma")

        # --- Apply user hints last so they always win ---------------------
        for pname, attrs in self.hints.items():
            if pname not in pars:
                raise KeyError(
                    f"hints contains {pname!r}, but no such parameter exists. "
                    f"Available: {sorted(pars.keys())}"
                )
            pars[pname].set(**attrs)

        fit0 = model.fit(y0, pars, x=x0, weights=1.0 / err0, scale_covar=False)
        components = fit0.eval_components()
        self.fit_result = fit0
        self.continuum = components[self.bkg_key].sum()
        epar = fit0.params

        # Area unit corrections — identical to PeakFit.gaussians_bkg. "area"
        # is the net counts (native y-units); we do not divide by livetime.
        spect = self.search.spectrum
        if spect.energies is None:
            correct_f = 1.0
        else:
            che = self.chan[maskx]
            correct_f = float(np.diff(spect.energies[che]).mean())

        for i in range(npeaks):
            mean0 = fit0.best_values[f"g{i+1}_center"]
            area0 = fit0.best_values[f"g{i+1}_amplitude"] / correct_f
            fwhm0 = fit0.best_values[f"g{i+1}_sigma"] * 2.355
            self.peak_info.append({"mean": mean0, "area": area0, "fwhm": fwhm0})

            mean_err = epar[f"g{i+1}_center"].stderr
            mean_err = mean_err if mean_err is not None else np.nan

            amp_err = epar[f"g{i+1}_amplitude"].stderr
            amp_err = amp_err if amp_err is not None else np.nan
            area_err = amp_err / correct_f

            sig_err = epar[f"g{i+1}_sigma"].stderr
            sig_err = sig_err if sig_err is not None else np.nan
            fwhm_err = sig_err * 2.355

            self.peak_err.append({
                "mean_err": mean_err,
                "area_err": area_err,
                "fwhm_err": fwhm_err,
            })

    # -- Plotting hooks ----------------------------------------------------
    def _bkg_drawstyle(self):
        # Render a sharp Heaviside step as an actual step, not a sloped ramp
        # between the two channels that straddle the discontinuity.
        return "steps-mid" if self.step == "sharp" else "default"

    def _component_curve(self, comps, cp):
        """
        Draw each peak as a *smooth* Gaussian on a single background level.

        The background is a step, so drawing ``bkg + gaussian`` (the parent
        default) would carve the step's discontinuity into the peak. Here
        the Gaussian is instead lifted onto the background level under its
        own centroid, so it reads as the smooth peak it actually is while
        still sitting on the continuum. The total ``Fit`` line still shows
        the genuine step in the background.
        """
        prefix = f"g{cp+1}_"
        center = self.fit_result.best_values[f"{prefix}center"]
        base = float(
            self.fit_result.eval_components(x=np.array([center]))[self.bkg_key][0]
        )
        return base + comps[prefix]


# ---------------------------------------------------------------------------
# Hypermet peak: Gaussian core + low-energy exponential tail (HPGe).
# ---------------------------------------------------------------------------

_SQRT2 = np.sqrt(2.0)


def _gaussian_unit(x, center, sigma):
    """Unit-area Gaussian."""
    return np.exp(-0.5 * ((x - center) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))


def _emg_left_unit(x, center, sigma, tau):
    """
    Unit-area Gaussian with a low-energy (left) exponential tail.

    This is a standard exponentially-modified Gaussian reflected about
    ``center`` so the tail points toward *lower* x — the HPGe charge-
    collection tail. ``tau`` is the tail length (same units as x). Written
    with ``erfcx`` (scaled complementary error function) for numerical
    stability: the naive ``exp(...) * erfc(...)`` overflows for small
    ``tau``. The identity used is

        exp(A) * erfc(B) = exp(-d^2 / 2 sigma^2) * erfcx(B),
        d = center - x,  B = sigma/(sqrt2 tau) - d/(sqrt2 sigma),

    so the only large factor left is ``erfcx``, which stays finite. The few
    deep-tail points where ``erfcx`` still overflows correspond to an
    amplitude below ~1e-27 and are clamped to zero.
    """
    x = np.asarray(x, dtype=float)
    d = center - x
    B = sigma / (_SQRT2 * tau) - d / (_SQRT2 * sigma)
    with np.errstate(over="ignore", invalid="ignore"):
        val = (1.0 / (2.0 * tau)) * np.exp(-0.5 * (d / sigma) ** 2) * erfcx(B)
    return np.where(np.isfinite(val), val, 0.0)


def emg_left_peak(x, amplitude, center, sigma, tau):
    """
    Area-normalised Gaussian with a low-energy exponential tail.

    A plain exponentially-modified Gaussian whose tail points toward *lower*
    x (unlike :class:`lmfit.models.ExponentialGaussianModel`, whose tail
    points high). ``amplitude`` is the total area; ``tau`` is the tail
    length. Backs ``MultiProfilePeakFit(profile="emg")``.
    """
    return amplitude * _emg_left_unit(x, center, sigma, tau)


def hypermet(x, amplitude, center, sigma, tail_fraction, tail_tau):
    """
    Hypermet / GF3-style peak: Gaussian core + smoothly joined left tail.

    The shape is the area-weighted sum of a unit-area Gaussian and a
    unit-area left-tailed EMG::

        peak(x) = amplitude * [ (1 - tail_fraction) * Gaussian(x)
                                + tail_fraction      * EMG_left(x) ]

    Because both components are unit-area, ``amplitude`` is the **total peak
    area** exactly and ``tail_fraction`` is the fraction of that area in the
    low-energy tail. The tail joins the core smoothly (the EMG is a
    convolution, hence C-infinity), so there is no junction discontinuity.

    Parameters
    ----------
    x : array_like
        Independent variable (channels or energies).
    amplitude : float
        Total peak area (Gaussian core + tail).
    center, sigma : float
        Gaussian core centroid and width.
    tail_fraction : float
        Fraction of the area in the low-energy tail, in [0, 1).
    tail_tau : float
        Tail decay length (same units as x).
    """
    core = _gaussian_unit(x, center, sigma)
    tail = _emg_left_unit(x, center, sigma, tail_tau)
    return amplitude * ((1.0 - tail_fraction) * core + tail_fraction * tail)


class _HypermetPeakMixin:
    """
    Mixin supplying a :func:`hypermet` peak component and its seeding.

    Pulls the peak-shape pieces out of :class:`HypermetFit` so they can be
    reused on top of a different background (see :class:`FullHPGePeakFit`).
    Hosts must set ``self._tail_fraction0`` and ``self._tail_tau0`` before
    the fit runs (i.e. before ``PeakFit.__init__``).
    """

    def _make_peak_component(self, prefix):
        return lmfit.Model(hypermet, prefix=prefix)

    def _set_peak_initial_values(self, pars, prefix, center, sigma, amplitude):
        pars[f"{prefix}center"].set(value=center)
        pars[f"{prefix}sigma"].set(value=sigma, min=1e-6)
        pars[f"{prefix}amplitude"].set(value=amplitude, min=0.0)
        pars[f"{prefix}tail_fraction"].set(
            value=self._tail_fraction0, min=0.0, max=0.95
        )
        tau0 = self._tail_tau0 if self._tail_tau0 is not None else max(sigma, 1e-3)
        pars[f"{prefix}tail_tau"].set(value=tau0, min=1e-3)

    def _sub_component_curves(self, cp, x):
        prefix = f"g{cp+1}_"
        bv = self.fit_result.best_values
        amp = bv[f"{prefix}amplitude"]
        cen = bv[f"{prefix}center"]
        sig = bv[f"{prefix}sigma"]
        tf  = bv[f"{prefix}tail_fraction"]
        tau = bv[f"{prefix}tail_tau"]
        core = amp * (1.0 - tf) * _gaussian_unit(x, cen, sig)
        tail = amp * tf * _emg_left_unit(x, cen, sig, tau)
        return [(core, "Core"), (tail, "Tail")]


class HypermetFit(_HypermetPeakMixin, PeakFit):
    """
    Fit HPGe peaks with a Hypermet shape (Gaussian core + low-energy tail).

    A :class:`PeakFit` whose peak component is :func:`hypermet` instead of a
    plain Gaussian. This is the canonical model for high-resolution HPGe
    full-energy peaks, which carry a low-energy tail from incomplete charge
    collection that a symmetric Gaussian cannot reproduce (it leaves
    structured residuals on the low side and biases the area).

    Because the Hypermet is built from unit-area components,
    ``peak_info["area"]`` is the correct *total* area (core + tail) with no
    extra bookkeeping. Each peak adds two parameters over a Gaussian:
    ``g{i}_tail_fraction`` (area fraction in the tail) and ``g{i}_tail_tau``
    (tail length). As with the other asymmetric profiles, the reported
    ``fwhm`` is the Gaussian-core value — use :func:`shape_summary` for the
    true measured FWHM/FWTM and tailing.

    For peaks on a stepped continuum, :class:`FullHPGePeakFit` pairs this
    tail with the smoothed-step background.

    Parameters
    ----------
    search : PeakSearch
        Spectrum and peak locations/widths (same as :class:`PeakFit`).
    xrange : list
        ``[lo, hi]`` fit window in the spectrum's x-axis units.
    bkg : str, optional
        Background model, as in :class:`PeakFit`. Default ``"poly1"``.
    tail_fraction : float, optional
        Initial tail area fraction. Default 0.1.
    tail_tau : float or None, optional
        Initial tail length. Default None → seeded to each peak's sigma.
    shared_sigma : bool, optional
        Link peak Gaussian-core sigmas through the resolution curve, as in
        :class:`PeakFit`.
    hints : dict, optional
        Per-parameter overrides, applied last.
    """

    def __init__(self, search, xrange, bkg="poly1", skew=False, *,
                 tail_fraction=0.1, tail_tau=None, shared_sigma=False,
                 hints=None):
        # ``skew`` is accepted only so PeakFit.save_json's kwargs round-trip
        # through load_json; the Hypermet supplies its own tail, so a skewed
        # Gaussian core is never used.
        # Stored before super().__init__ because the fit (and thus the
        # per-peak seeding) runs inside the parent constructor.
        self._tail_fraction0 = float(tail_fraction)
        self._tail_tau0 = None if tail_tau is None else float(tail_tau)
        super().__init__(
            search, xrange, bkg=bkg, skew=False,
            shared_sigma=shared_sigma, hints=hints,
        )


class FullHPGePeakFit(_HypermetPeakMixin, GaussStepFit):
    """
    Full HPGe full-energy-peak model: Hypermet tail on a smoothed step.

    Combines the two HPGe-specific pieces in this module — a :func:`hypermet`
    peak (Gaussian core + low-energy tail) and the smoothed
    :func:`step_background` continuum (higher on the low-energy side, lower
    on the high-energy side, tied to the peak). This is the shape dedicated
    gamma codes (GF3, FitzPeaks, GammaVision) fit to clean HPGe singlets and
    is the recommended default for careful HPGe peak-area work.

    Inherits the step background and smooth-peak plotting from
    :class:`GaussStepFit` and the Hypermet peak component from
    :class:`HypermetFit`; ``summary()``, ``shape_summary()``,
    ``fit_quality()``, ``plot()`` and ``save_json()``/``load_json()`` all
    work as usual. ``peak_info["area"]`` is the total peak area (core +
    tail).

    Parameters
    ----------
    search : PeakSearch
        Spectrum and peak locations/widths.
    xrange : list
        ``[lo, hi]`` fit window in the spectrum's x-axis units.
    step : {"sharp", "smooth"}, optional
        Step shape. Default ``"sharp"`` — a Heaviside shelf at the centroid.
        This is the right companion to the Hypermet tail: the *tail* already
        models the smooth low-energy excess just below the peak, while the
        *step* captures the discrete Compton shelf (the continuum being
        higher on the low-energy side). A ``"smooth"`` step shares the peak's
        centre *and* width, so it overlaps the tail's own erfc and the two
        become degenerate — fine when a strong real step is present, but the
        fit can fail to converge when the true step is near zero. Prefer
        ``"sharp"`` unless you have a specific reason.
    tail_fraction : float, optional
        Initial tail area fraction. Default 0.1.
    tail_tau : float or None, optional
        Initial tail length. Default None → seeded to each peak's sigma.
    hints : dict, optional
        Per-parameter overrides, applied last.

    Notes
    -----
    ``shared_sigma`` is not supported on the step background (inherited from
    :class:`GaussStepFit`); passing ``shared_sigma=True`` raises.
    """

    def __init__(self, search, xrange, step="sharp", bkg=None, skew=False, *,
                 tail_fraction=0.1, tail_tau=None, shared_sigma=False,
                 hints=None):
        self._tail_fraction0 = float(tail_fraction)
        self._tail_tau0 = None if tail_tau is None else float(tail_tau)
        super().__init__(
            search, xrange, step=step, bkg=bkg, skew=skew,
            shared_sigma=shared_sigma, hints=hints,
        )


# ---------------------------------------------------------------------------
# Peak-shape diagnostics (FWHM / FWTM / tailing) for any fitted profile.
# ---------------------------------------------------------------------------

# Reference shape ratio for a pure Gaussian (no tailing). A measured
# FWTM/FWHM noticeably above this signals low-energy tailing.
GAUSS_FWTM_FWHM = float(np.sqrt(np.log(10.0) / np.log(2.0)))  # 1.82262


def _side_crossing(x_side, y_side, thr, rising):
    """
    Interpolate the x where a monotonic peak flank crosses level ``thr``.

    ``rising`` is True for the left flank (y increasing towards the apex)
    and False for the right flank (y decreasing away from it). Returns NaN
    if the flank never reaches down to ``thr`` within the given window.
    """
    if rising:
        if y_side[0] > thr:
            return np.nan  # flank starts above the level — no crossing in view
        return float(np.interp(thr, y_side, x_side))
    if y_side[-1] > thr:
        return np.nan
    # right flank is decreasing; reverse so the xp passed to interp ascends
    return float(np.interp(thr, y_side[::-1], x_side[::-1]))


def peak_shape_metrics(x, y):
    """
    Width and tailing metrics measured directly from a peak's profile.

    Operates on the sampled peak component (background already removed), so
    it is valid for *any* line shape — Gaussian, Voigt, skewed, EMG,
    Hypermet — unlike the ``sigma * 2.3548`` FWHM, which only holds for a
    pure Gaussian. Widths are found by linear interpolation of the flanks at
    50% and 10% of the peak maximum.

    Parameters
    ----------
    x : array_like
        Sample positions (channels or energies), sorted ascending. Use a
        fine grid for accurate crossings.
    y : array_like
        Peak amplitude at each ``x`` (continuum subtracted).

    Returns
    -------
    dict
        ``fwhm``      — full width at half maximum.
        ``fwtm``      — full width at tenth maximum.
        ``fwtm_fwhm`` — their ratio; ~1.823 for a pure Gaussian (see
                        :data:`GAUSS_FWTM_FWHM`), larger when tailed.
        ``hwhm_left`` / ``hwhm_right`` — apex-to-flank half widths at half
                        maximum (a longer side reveals asymmetry).
        ``asymmetry`` — ``(left - right) / fwtm`` half widths at *tenth*
                        maximum, where tails dominate; positive means a
                        low-energy (left) tail, the usual HPGe signature.
        Any value is NaN if the flank does not descend to the level inside
        the supplied window.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nan = {
        "fwhm": np.nan, "fwtm": np.nan, "fwtm_fwhm": np.nan,
        "hwhm_left": np.nan, "hwhm_right": np.nan, "asymmetry": np.nan,
    }
    if x.size < 3 or not np.any(y > 0):
        return nan

    imax = int(np.argmax(y))
    ymax = float(y[imax])
    xmax = float(x[imax])
    if imax == 0 or imax == x.size - 1:
        return nan  # apex on an edge — widths would be cut off

    xl, yl = x[: imax + 1], y[: imax + 1]
    xr, yr = x[imax:], y[imax:]

    def width(frac):
        thr = frac * ymax
        left = _side_crossing(xl, yl, thr, rising=True)
        right = _side_crossing(xr, yr, thr, rising=False)
        return left, right

    l_half, r_half = width(0.5)
    l_tenth, r_tenth = width(0.1)

    fwhm = r_half - l_half
    fwtm = r_tenth - l_tenth
    return {
        "fwhm": fwhm,
        "fwtm": fwtm,
        "fwtm_fwhm": fwtm / fwhm if fwhm else np.nan,
        "hwhm_left": xmax - l_half,
        "hwhm_right": r_half - xmax,
        "asymmetry": ((xmax - l_tenth) - (r_tenth - xmax)) / fwtm if fwtm else np.nan,
    }


def shape_metrics(fit, npoints=2000):
    """
    Measured shape metrics for every peak of a fitted ``PeakFit``.

    Evaluates each peak component on a fine grid spanning the fit window and
    runs :func:`peak_shape_metrics` on it. Works for any
    :class:`wara.peakfit.PeakFit` or subclass (Gaussian, Voigt, skewed, EMG,
    Hypermet, step-background, ...).

    Parameters
    ----------
    fit : PeakFit
        A fitted object exposing ``x_data``, ``fit_result`` and ``peak_info``.
    npoints : int, optional
        Number of grid points for the crossing search. Default 2000.

    Returns
    -------
    list of dict
        One :func:`peak_shape_metrics` dict per peak, in peak order.
    """
    x = np.asarray(fit.x_data, dtype=float)
    x_fine = np.linspace(x[0], x[-1], int(npoints))
    comps = fit.fit_result.eval_components(x=x_fine)
    return [
        peak_shape_metrics(x_fine, comps[f"g{i+1}_"])
        for i in range(len(fit.peak_info))
    ]


def shape_summary(fit, npoints=2000):
    """
    ``fit.summary()`` augmented with measured shape/tailing columns.

    Appends ``fwhm_meas`` (numerically measured FWHM, valid for non-Gaussian
    profiles where the ``fwhm`` column — derived from sigma — is only
    approximate), ``fwtm``, ``fwtm_fwhm`` and ``asymmetry``. See
    :func:`peak_shape_metrics` for definitions.

    Parameters
    ----------
    fit : PeakFit
        A fitted object.
    npoints : int, optional
        Grid density passed to :func:`shape_metrics`. Default 2000.

    Returns
    -------
    pandas.DataFrame
        One row per peak: the base summary columns plus the shape columns.
    """
    base = fit.summary()
    mets = pd.DataFrame(shape_metrics(fit, npoints=npoints))
    extra = pd.DataFrame({
        "fwhm_meas": mets["fwhm"],
        "fwtm": mets["fwtm"],
        "fwtm_fwhm": mets["fwtm_fwhm"],
        "asymmetry": mets["asymmetry"],
    })
    return pd.concat([base.reset_index(drop=True), extra], axis=1)


# ---------------------------------------------------------------------------
# Continuum-only background fit (no peaks).
# ---------------------------------------------------------------------------


class ContinuumFit:
    """
    Fit a polynomial continuum to a spectrum, masking out peak regions.

    Useful for estimating the smooth baseline under one or many peaks
    without modelling the peaks themselves. The peak locations and widths
    come from a PeakSearch; every channel within +/-``mask_fwhm`` FWHMs
    of a detected peak is excluded from the fit. The polynomial is fit to
    what remains, weighted by the spectrum's per-bin uncertainties.

    Parameters
    ----------
    search : PeakSearch
        Provides both the spectrum and the list of peaks to mask out.
    xrange : 2-element list or None, optional
        Window over which to fit the continuum, in the same units as
        PeakFit's xrange (energies if calibrated, otherwise channels).
        Defaults to the full spectrum.
    degree : int, optional
        Polynomial degree (0-7, lmfit's limit). Default 3.
    mask_fwhm : float, optional
        Number of FWHMs around each peak to exclude from the fit.
        Default 3.0 (covers ~99.7% of a Gaussian peak's area).

    Attributes
    ----------
    fit_result : lmfit.model.ModelResult
        The underlying lmfit result. Carries ``redchi``, ``aic``, ``bic``,
        covariance, etc.
    continuum_mask : ndarray of bool
        True for channels used in the fit (in-range and not under a peak).
    x_cont, y_cont, err_cont : ndarray
        The points actually fed to the polynomial fit.
    """

    def __init__(self, search, xrange=None, degree=3, mask_fwhm=3.0):
        from . import peaksearch as ps

        if not isinstance(search, ps.PeakSearch):
            raise TypeError(
                f"search must be a PeakSearch object, got {type(search)}"
            )
        if not 0 <= degree <= 7:
            raise ValueError(
                f"degree must be in [0, 7] (lmfit limit), got {degree}"
            )
        self.search = search
        self.degree = int(degree)
        self.mask_fwhm = float(mask_fwhm)

        spect = search.spectrum
        if spect.energies is None:
            self.x = spect.channels
        else:
            self.x = spect.energies
        if xrange is None:
            xrange = [float(self.x[0]), float(self.x[-1])]
        self.xrange = list(xrange)

        self._fit()

    def _build_mask(self):
        """Return boolean mask of channels kept for the continuum fit."""
        x = self.x
        keep = (x >= self.xrange[0]) & (x <= self.xrange[1])
        # PeakSearch.fwhm_guess is parametrized in channels, so we mask
        # in channel-index space — this avoids the channel<->energy
        # FWHM rescaling that bites elsewhere in the package.
        n = len(x)
        for pidx, fwhm in zip(
            self.search.peaks_idx, self.search.fwhm_guess
        ):
            half = int(np.ceil(self.mask_fwhm * fwhm))
            lo = max(0, int(pidx) - half)
            hi = min(n, int(pidx) + half + 1)
            keep[lo:hi] = False
        return keep

    def _fit(self):
        spect = self.search.spectrum
        mask = self._build_mask()
        if mask.sum() < self.degree + 1:
            raise ValueError(
                f"Only {int(mask.sum())} continuum points available after "
                f"masking, need at least {self.degree + 1} for a "
                f"degree-{self.degree} polynomial. Try a wider xrange, "
                "a smaller mask_fwhm, or a lower degree."
            )
        x_cont = self.x[mask]
        y_cont = spect.counts[mask].copy()
        err_cont = spect.counts_err[mask].copy()
        err_cont[err_cont <= 0] = 1.0

        model = lmfit.models.PolynomialModel(degree=self.degree)
        pars = model.guess(y_cont, x=x_cont)
        # scale_covar=False matches PeakFit's convention: treat the
        # provided weights as absolute (known) uncertainties.
        self.fit_result = model.fit(
            y_cont,
            pars,
            x=x_cont,
            weights=1.0 / err_cont,
            scale_covar=False,
        )
        self.continuum_mask = mask
        self.x_cont = x_cont
        self.y_cont = y_cont
        self.err_cont = err_cont

    def evaluate(self, x=None):
        """
        Evaluate the fitted continuum at the given x values.

        Defaults to evaluating across the full xrange of the fit.
        """
        if x is None:
            in_range = (self.x >= self.xrange[0]) & (self.x <= self.xrange[1])
            x = self.x[in_range]
        return self.fit_result.eval(x=np.asarray(x))

    def subtract(self):
        """
        Return (x, residual) over the fit window — counts minus continuum.

        Useful as a quick way to look at "what's left" after the smooth
        baseline is removed (peaks plus noise).
        """
        in_range = (self.x >= self.xrange[0]) & (self.x <= self.xrange[1])
        x_full = self.x[in_range]
        y_full = self.search.spectrum.counts[in_range]
        return x_full, y_full - self.evaluate(x_full)

    def plot(self, ax=None):
        """
        Plot the spectrum, the masked-out peak regions, and the fitted
        continuum.
        """
        from .matplotlib_theme import apply_theme

        apply_theme()
        with plt.rc_context({"font.size": 12}):
            if ax is None:
                fig, ax = plt.subplots(figsize=(10, 6))
                fig.patch.set_alpha(0.3)

            in_range = (self.x >= self.xrange[0]) & (self.x <= self.xrange[1])
            x_full = self.x[in_range]
            y_full = self.search.spectrum.counts[in_range]
            ax.plot(x_full, y_full, drawstyle="steps-mid", lw=1, label="data")

            # Highlight the channels under peaks (excluded from the fit).
            masked_out = in_range & ~self.continuum_mask
            ax.plot(
                self.x[masked_out],
                self.search.spectrum.counts[masked_out],
                ".",
                color="red",
                ms=4,
                alpha=0.5,
                label=f"masked ({self.mask_fwhm:g}xFWHM around peaks)",
            )

            y_cont_full = self.evaluate(x_full)
            ax.plot(
                x_full,
                y_cont_full,
                "g-",
                lw=2,
                label=f"poly{self.degree} continuum",
            )

            ax.set_xlabel(self.search.spectrum.x_units)
            ax.set_ylabel("Counts")
            ax.set_title(
                rf"Continuum fit: $\chi^2_\nu$ = {self.fit_result.redchi:.3f}"
            )
            ax.legend(loc="best")
