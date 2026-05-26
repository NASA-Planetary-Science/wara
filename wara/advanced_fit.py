"""
Advanced fitting classes and functions
"""
import lmfit
import matplotlib.pyplot as plt
import numpy as np
import mplcursors
from wara import spectrum as sp
from wara.matplotlib_theme import apply_theme
from wara.peakfit import PeakFit


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

_PROFILE_MODELS = {
    "gauss":   lmfit.models.GaussianModel,
    "voigt":   lmfit.models.VoigtModel,
    "pvoigt":  lmfit.models.PseudoVoigtModel,
    "skewed":  lmfit.models.SkewedGaussianModel,
    "doniach": lmfit.models.DoniachModel,
}


class MultiProfilePeakFit(PeakFit):
    """
    PeakFit variant supporting additional line shapes:

    * ``"gauss"``   — plain Gaussian (same as :class:`PeakFit`).
    * ``"voigt"``   — Voigt (Gaussian * Lorentzian).
    * ``"pvoigt"``  — Pseudo-Voigt (weighted G+L sum).
    * ``"skewed"``  — SkewedGaussian (low-energy tailing).
    * ``"doniach"`` — Doniach-Sunjic (photoemission asymmetry).

    Parameters
    ----------
    profile : str
        Line-shape family. Default ``"voigt"``.
    *args, **kwargs
        Forwarded to :class:`wara.peakfit.PeakFit`. The legacy
        ``skew=True`` kwarg is honoured for backwards compatibility but
        ``profile="skewed"`` is the preferred spelling.

    Notes
    -----
    ``shared_sigma=True`` constrains every peak's *Gaussian* sigma via
    ``fwhm = a + b*sqrt(E)``. For Voigt / PseudoVoigt the *total* FWHM is
    a function of both sigma and gamma, so the shared constraint is
    approximate (it only links the Gaussian component). Override gamma
    or fraction via ``hints`` if you need finer control.
    """

    def __init__(self, search, xrange, *args, profile="voigt", **kwargs):
        profile = profile.lower()
        if profile not in _PROFILE_MODELS:
            raise ValueError(
                f"Unknown profile {profile!r}. "
                f"Choose from {sorted(_PROFILE_MODELS)}."
            )
        self.profile = profile
        # The "skewed" profile is also reachable via skew=True on the
        # parent class; pick consistent behaviour without surprising the
        # parent's gamma seeding.
        if profile == "skewed":
            kwargs.setdefault("skew", True)
        super().__init__(search, xrange, *args, **kwargs)

    def _make_peak_component(self, prefix):
        return _PROFILE_MODELS[self.profile](prefix=prefix)


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
