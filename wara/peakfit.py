"""
Peak fit class: Gaussian + Linear fit
"""

import json
import re
import warnings
from pathlib import Path

import lmfit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import peaksearch as ps
from .matplotlib_theme import apply_theme, get_theme, DARK
from .fit_utils import safe_eval_uncertainty


class PeakFit:
    def __init__(
        self,
        search,
        xrange,
        bkg="poly1",
        skew=False,
        *,
        shared_sigma=False,
        hints=None,
    ):
        """
        Use the package LMfit to fit and plot peaks with Gaussians with a
        given background. It must be initialized with a PeakSearch object.

        Parameters
        ----------
        search : PeakSearch object.
            from peaksearch.py.
        xrange : list
            range of 2 x-values where the fit is attempted. Can be either in
            energy values or channels depending on how the search object
            was initialized.
        bkg : string, optional
            Can be 'linear', 'quadratic', 'exponential' or 'polyn', where n
            is any degree polynomial. The default is 'poly1'.
        skew : bool, optional
            fit a skewed Gaussian instead. The default is 'False'.
        shared_sigma : bool, optional
            Link every peak's sigma through the same resolution curve

                fwhm(E) = |a + b * sqrt(E)|,  sigma = fwhm / 2.355

            (the same parametrization used by PeakSearch, wrapped in abs()
            so the derived sigma can never go negative during optimization).
            Two free shared parameters ``_fwhm_a`` and ``_fwhm_b`` replace
            the individual sigmas. Initial values come from the PeakSearch.
            When the window contains only one peak, ``_fwhm_b`` is fixed at
            the search's value (otherwise the pair is underdetermined).
            Either parameter can be pinned through ``hints``. Note: in a
            narrow window where all peaks have similar sqrt(E), the (a, b)
            pair is degenerate — the fit will pick *some* (a, b) that
            reproduces the right sigmas, possibly with unphysical signs;
            only the derived FWHMs are meaningful in that regime. The
            default is False.
        hints : dict, optional
            Per-parameter overrides applied after defaults (and after the
            shared-sigma expressions) but before the fit. Maps parameter
            name to a dict of attributes accepted by
            ``lmfit.Parameter.set`` (e.g. ``value``, ``min``, ``max``,
            ``vary``, ``expr``). Example::

                hints={"g1_center": {"value": 661.7, "vary": False}}

        Raises
        ------
        TypeError
            'search' must be a PeakSearch object.

        """

        if not isinstance(search, ps.PeakSearch):
            raise TypeError(f"search must be a PeakSearch object, got {type(search)} instead")
        self.search = search
        self.xrange = xrange
        self.continuum = None
        self.bkg_key = None
        # Normalize the user-supplied bkg name once, here, so the rest of
        # the class can rely on canonical values ("exponential", not "exp").
        self.bkg = "exponential" if bkg == "exp" else bkg
        self.x_data = None
        self.y_data = None
        self.peak_info = []
        self.peak_err = []
        self.fit_result = None
        self.x_units = search.spectrum.x_units
        self.chan = search.spectrum.channels
        self.skew = bool(skew)
        self.shared_sigma = bool(shared_sigma)
        self.hints = dict(hints) if hints else {}
        if search.spectrum.energies is None:
            self.x = search.spectrum.channels
        else:
            self.x = search.spectrum.energies
        self.gaussians_bkg(skew)

    # -- Hooks for subclasses ----------------------------------------------
    def _make_peak_component(self, prefix):
        """
        Build a single peak component model. Override in subclasses to use
        Voigt, PseudoVoigt, etc. The returned model MUST expose parameters
        ``{prefix}center``, ``{prefix}sigma``, and ``{prefix}amplitude``.
        """
        if self.skew:
            return lmfit.models.SkewedGaussianModel(prefix=prefix)
        return lmfit.models.GaussianModel(prefix=prefix)

    def _set_peak_initial_values(self, pars, prefix, center, sigma, amplitude):
        """Seed per-peak parameter values. Override to seed any extras."""
        pars[f"{prefix}center"].set(value=center)
        pars[f"{prefix}sigma"].set(value=sigma)
        pars[f"{prefix}amplitude"].set(value=amplitude)
        if self.skew:
            pars[f"{prefix}gamma"].set(value=-1)

    def _inject_constraint_peaks(self, erg, sig, amp):
        """
        Hook: append seeds for user-constrained extra peaks.

        Called after the automatic peak search and before the model is
        built. Default is a no-op. Overrides may append extra peaks to the
        ``(center, sigma, amplitude)`` seed arrays — they are built as
        ordinary ``g{i}_`` components (so they are fitted, reported and
        plotted like any other peak). Record the appended indices in
        ``self._free_sigma_idx`` to exempt them from the shared-sigma
        resolution curve (used for broad Doppler companions, whose width is
        unrelated to the narrow-peak detector resolution).

        Returns the (possibly extended) ``(erg, sig, amp)`` arrays.
        """
        return erg, sig, amp

    # -- Hooks for plotting ------------------------------------------------
    def _component_curve(self, comps, cp):
        """
        y-values drawn for peak component ``cp`` in :meth:`plot`.

        Default: the peak sitting on its background (``bkg + gaussian``),
        which reads naturally for smooth continua. Subclasses with a
        discontinuous background (e.g. a step) override this so the step
        doesn't carve a notch into the otherwise-smooth peak.
        """
        return comps[self.bkg_key] + comps[f"g{cp+1}_"]

    def _sub_component_curves(self, cp, x):
        """
        Sub-component decomposition for peak ``cp`` in :meth:`plot`.

        Returns a list of ``(y_values, label)`` tuples — peak-only values
        (no background offset; the caller adds the same base used by
        :meth:`_component_curve`).  Default: empty (no sub-components).
        """
        return []

    def _bkg_drawstyle(self):
        """matplotlib ``drawstyle`` for the background line in :meth:`plot`."""
        return "default"

    def find_peaks_range(self):
        """
        Find data points within xrange where peaks were found
        with peaksearch.

        Returns
        -------
        mask : numpy array
            array of booleans where fit is performed.
        pidx : numpy array
            peak indices as found with peaksearch.

        """

        mask = (self.x[self.search.peaks_idx] > self.xrange[0]) & (
            self.x[self.search.peaks_idx] < self.xrange[1]
        )

        if mask.sum() == 0:
            raise ValueError(
                f"No peaks found within range {self.xrange}. "
                "Try lowering min_snr or adjusting the xrange."
            )
        pidx = self.search.peaks_idx[mask]
        return mask, pidx

    def init_values(self):
        """
        Try to find suitable initial value guesses for the fit. This works
        for a linear background only, and it should be attempted when the
        automatic guess from LMfit fails.

        Returns
        -------
        m0 : float
            slope guess for linear background.
        b0 : float
            y-intercept guess for linear background.
        amp0 : float
            Gaussian amplitude guess.
        erg0 : float
            Gaussian mean guess.
        sig0 : float
            Gaussian sigma guess.

        """
        cts = self.search.spectrum.counts
        maskx = (self.x > self.xrange[0]) & (self.x < self.xrange[1])
        chan_range = self.chan[maskx]
        x_range = self.x[maskx]
        if len(chan_range) > 1 and self.search.spectrum.energies is not None:
            m = np.polyfit(chan_range, x_range, 1)[0]  # energy/channel
        else:
            m = 1.0
        left_idx = np.where(self.x > self.xrange[0])[0]
        right_idx = np.where(self.x > self.xrange[1])[0]
        left = cts[left_idx[0]] if len(left_idx) > 0 else cts[0]
        right = cts[right_idx[0]] if len(right_idx) > 0 else cts[-1]

        mask, pks_idx = self.find_peaks_range()
        erg0 = self.x[pks_idx]
        sig0 = self.search.fwhm_guess[mask] * m / 2.355
        height0 = abs(cts[pks_idx] - (right + left) / 2)
        amp0 = height0 * sig0 / 0.4
        m0 = (right - left) / (self.xrange[1] - self.xrange[0])
        b0 = left - m0 * self.xrange[0]
        return m0, b0, amp0, erg0, sig0

    def gaussians_bkg(self, skeweness):
        """
        Fit one or multiple Gaussians with a given background.

        Weights are derived from the spectrum's counts_err attribute, which
        carries fully propagated uncertainties (e.g. after background
        subtraction with livetime scaling). For a plain spectrum this
        defaults to Poisson sqrt(N), which is identical to the previous
        sqrt(1/y) formulation.

        Returns
        -------
        None.

        """
        maskx = (self.x > self.xrange[0]) & (self.x < self.xrange[1])
        m, b, amp, erg, sig = self.init_values()
        # Subclasses may append extra, user-constrained peaks here (e.g. a
        # broad Doppler companion). Indices recorded in self._free_sigma_idx
        # are exempt from the shared-sigma resolution curve below.
        self._free_sigma_idx = set()
        erg, sig, amp = self._inject_constraint_peaks(erg, sig, amp)
        npeaks = len(erg)

        # .copy() is important: y0 gets mutated below (zero-replacement) and
        # boolean indexing usually copies, but we don't want to depend on that
        # — we must never alter spectrum.counts.
        y0 = self.search.spectrum.counts[maskx].copy()
        x0 = self.x[maskx]

        # Use propagated per-bin uncertainties from the Spectrum object.
        # Guard against any zero errors to avoid infinite weights.
        err0 = self.search.spectrum.counts_err[maskx].copy()
        err0[err0 <= 0] = 1.0

        # zeroes in the data can cause fitting problems with some backgrounds;
        # replace them with a small positive value.
        y0_min = np.amin(y0[y0 != 0.0]) if np.any(y0 != 0.0) else 1.0
        y0[y0 == 0.0] = y0_min * 1e-1

        self.y_data = y0
        self.x_data = x0

        # Build background model
        if self.bkg == "linear":
            lin_mod = lmfit.models.LinearModel(prefix="linear")
            pars = lin_mod.make_params(slope=m, intercept=b)
            model = lin_mod
            self.bkg_key = "linear"
        elif self.bkg == "quadratic":
            quad_mod = lmfit.models.QuadraticModel(prefix="quadratic")
            pars = quad_mod.guess(y0, x=x0)
            model = quad_mod
            self.bkg_key = "quadratic"
        elif self.bkg == "exponential":
            exp_mod = lmfit.models.ExponentialModel()
            pars = exp_mod.guess(y0, x=x0)
            model = exp_mod
            self.bkg_key = "exponential"
        else:
            # polynomial of degree n, specified as "polyN" (e.g. "poly3").
            # lmfit's PolynomialModel supports degree 0–7.
            match = re.fullmatch(r"poly(\d+)", self.bkg)
            if match is None:
                raise ValueError(
                    f"Unknown bkg {self.bkg!r}. Expected 'linear', "
                    "'quadratic', 'exponential', or 'polyN' with N=0..7."
                )
            n = int(match.group(1))
            if not 0 <= n <= 7:
                raise ValueError(
                    f"Polynomial degree must be 0..7, got {n} (bkg={self.bkg!r})."
                )
            poly_mod = lmfit.models.PolynomialModel(degree=n)
            pars = poly_mod.guess(y0, x=x0)
            model = poly_mod
            self.bkg_key = "polynomial"

        for i in range(npeaks):
            prefix = f"g{i+1}_"
            comp = self._make_peak_component(prefix)
            pars.update(comp.make_params())
            self._set_peak_initial_values(pars, prefix, erg[i], sig[i], amp[i])
            model += comp

        # --- Shared sigma: link every peak's sigma through fwhm = a + b*sqrt(E)
        # Only the resolution-curve (non-companion) peaks participate; a region
        # made up solely of broad Doppler companions has nothing to share.
        n_curve = npeaks - len(self._free_sigma_idx)
        if self.shared_sigma and n_curve > 0:
            # Seed a, b from PeakSearch's resolution curve. The search
            # works in channels; rescale to whatever units self.x is in
            # by the local energy-per-channel slope. NOTE: the `m`
            # returned by init_values() is the BACKGROUND slope (can be
            # negative) — not what we want here, so compute the
            # energy/channel slope explicitly.
            chan_in = self.chan[maskx]
            x_in = self.x[maskx]
            if len(chan_in) > 1 and self.search.spectrum.energies is not None:
                eperch = float(np.polyfit(chan_in, x_in, 1)[0])
            else:
                eperch = 1.0
            a_init = eperch * self.search.fwhm_at_0
            b_init = (
                np.sqrt(eperch)
                * self.search.ref_fwhm
                / np.sqrt(self.search.ref_x)
            )
            # Leave _fwhm_a, _fwhm_b unbounded so the optimizer can compute
            # a proper covariance (hard bounds at zero block the Jacobian
            # estimate when peaks at similar energies drive b to ~0).
            # The abs() in the sigma expression below keeps the derived
            # sigma strictly positive regardless of sign.
            pars.add("_fwhm_a", value=float(a_init), vary=True)
            # With only one peak, a and b are degenerate — keep b at the
            # search's value and let a absorb any offset.
            b_vary = n_curve > 1
            pars.add("_fwhm_b", value=float(b_init), vary=b_vary)
            # Bound each peak's center to the fit window so the
            # sqrt(center) term in the expression below can never see a
            # negative number during optimization.
            for i in range(npeaks):
                if i in self._free_sigma_idx:
                    # Broad/constrained companion peaks do not follow the
                    # narrow-peak resolution curve; leave their sigma free.
                    continue
                prefix = f"g{i+1}_"
                # A declared peak (peaks=) already carries a tighter centroid
                # window from _bound_user_centers; keep it instead of widening
                # back to the full fit window. Still ensure a non-negative min.
                user_bounds = getattr(self, "_user_center_bounds", {}).get(prefix)
                if user_bounds is not None:
                    lo, hi = user_bounds
                    pars[f"{prefix}center"].set(min=max(0.0, lo), max=hi)
                else:
                    pars[f"{prefix}center"].set(
                        min=max(0.0, float(self.xrange[0])),
                        max=float(self.xrange[1]),
                    )
                pars[f"{prefix}sigma"].set(
                    expr=(
                        f"abs(_fwhm_a + _fwhm_b * sqrt({prefix}center))"
                        " / 2.355"
                    )
                )

        # --- Apply user hints last so they always win
        for pname, attrs in self.hints.items():
            if pname not in pars:
                raise KeyError(
                    f"hints contains {pname!r}, but no such parameter exists. "
                    f"Available: {sorted(pars.keys())}"
                )
            pars[pname].set(**attrs)

        # weights = 1/σ so that χ² = Σ [(y - f) / σ]²
        # Using counts_err gives correct weights even for background-subtracted
        # spectra where the per-bin uncertainty is no longer simply sqrt(N).
        # scale_covar=False: treat the provided weights as absolute (known)
        # uncertainties — do NOT rescale the covariance matrix by chi²_reduced.
        # With scale_covar=True (lmfit default), over-fitted spectra (e.g. after
        # background subtraction) get chi²_red < 1 and their errors are
        # artificially deflated, breaking the expected propagation relationship
        # err_net ≈ sqrt(err1² + err2²).
        fit0 = model.fit(y0, pars, x=x0, weights=1.0 / err0, scale_covar=False)
        components = fit0.eval_components()
        self.fit_result = fit0

        # Save peak info: mean, net area, fwhm, and corresponding errors
        self.continuum = components[self.bkg_key].sum()
        epar = fit0.params

        # Correction factor: lmfit's amplitude is the integral in
        # (counts * x-units); divide by the mean bin width (x-units per bin)
        # to recover the net peak area in the spectrum's native y-units —
        # raw counts, or counts/sec for a spectrum already stored as cps.
        # NOTE: "area" is the net counts, NOT a rate; we do not divide by
        # livetime. Divide by spectrum.livetime yourself if you want cps.
        spect = self.search.spectrum
        if spect.energies is None:
            correct_f = 1.0
        else:
            # Mean bin width in x-units (energy/bin) across the ROI. self.x is
            # the energy axis, so slice it positionally with the boolean maskx.
            correct_f = float(np.diff(self.x[maskx]).mean())

        for i in range(npeaks):
            mean0 = fit0.best_values[f"g{i+1}_center"]
            area0 = fit0.best_values[f"g{i+1}_amplitude"] / correct_f
            fwhm0 = fit0.best_values[f"g{i+1}_sigma"] * 2.355
            dict_peak_info = {
                "mean": mean0,
                "area": area0,
                "fwhm": fwhm0,
            }
            self.peak_info.append(dict_peak_info)

            # Errors
            mean_err = epar[f"g{i+1}_center"].stderr
            mean_err = mean_err if mean_err is not None else np.nan

            amp_err = epar[f"g{i+1}_amplitude"].stderr
            amp_err = amp_err if amp_err is not None else np.nan
            area_err = amp_err / correct_f

            sig_err = epar[f"g{i+1}_sigma"].stderr
            sig_err = sig_err if sig_err is not None else np.nan
            fwhm_err = sig_err * 2.355

            dict_peak_err = {
                "mean_err": mean_err,
                "area_err": area_err,
                "fwhm_err": fwhm_err,
            }
            self.peak_err.append(dict_peak_err)

    # -- Result introspection ----------------------------------------------
    def summary(self):
        """
        Return a DataFrame with one row per fitted peak.

        Columns: mean, mean_err, area, area_err, fwhm, fwhm_err.
        """
        rows = []
        for info, err in zip(self.peak_info, self.peak_err):
            rows.append({
                "mean": info["mean"],
                "mean_err": err["mean_err"],
                "area": info["area"],
                "area_err": err["area_err"],
                "fwhm": info["fwhm"],
                "fwhm_err": err["fwhm_err"],
            })
        return pd.DataFrame(rows)

    def fit_quality(self):
        """
        Goodness-of-fit metrics beyond reduced chi-squared.

        Returns
        -------
        dict with:
            redchi : reduced chi-squared from lmfit
            aic    : Akaike information criterion
            bic    : Bayesian information criterion
            nfev   : number of model evaluations
            success : lmfit's success flag
            normaltest_pvalue : p-value of D'Agostino-Pearson normality test
                on standardized residuals (y - model)/sigma. p > ~0.05 is
                consistent with normally distributed residuals; very small
                values suggest the model is missing structure. NaN if there
                are too few points to compute the test.
        """
        from scipy import stats

        res = self.fit_result
        err = self.search.spectrum.counts_err[
            (self.x > self.xrange[0]) & (self.x < self.xrange[1])
        ]
        err = np.where(err > 0, err, 1.0)
        std_resid = res.residual  # lmfit residual = (data - model) * weights
        if len(std_resid) >= 8:
            _, pval = stats.normaltest(std_resid)
        else:
            pval = float("nan")
        return {
            "redchi": float(res.redchi),
            "aic": float(res.aic),
            "bic": float(res.bic),
            "nfev": int(res.nfev),
            "success": bool(res.success),
            "normaltest_pvalue": float(pval),
        }

    # -- Serialization -----------------------------------------------------
    def save_json(self, path):
        """
        Save the underlying lmfit result and minimal metadata to a JSON
        file. Restore with ``PeakFit.load_json(path, search)``.

        Note: the parent ``PeakSearch`` and ``Spectrum`` are NOT serialized
        — load_json takes the search separately so the same spectrum can
        be re-attached.
        """
        path = Path(path)
        payload = {
            "fit_result": self.fit_result.dumps(),
            "xrange": list(self.xrange),
            "bkg": self.bkg,
            "skew": self.skew,
            "shared_sigma": self.shared_sigma,
            "hints": self.hints,
        }
        path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load_json(cls, path, search):
        """Restore a PeakFit previously written with ``save_json``."""
        path = Path(path)
        payload = json.loads(path.read_text())
        obj = cls(
            search=search,
            xrange=payload["xrange"],
            bkg=payload["bkg"],
            skew=payload["skew"],
            shared_sigma=payload.get("shared_sigma", False),
            hints=payload.get("hints", {}),
        )
        # Replace the freshly-fit result with the saved one so the round
        # trip preserves the exact reported parameters/covariance.
        # lmfit's ModelResult.loads() reconstructs into self; we use the
        # freshly-built result as the host (its .model matches the saved one).
        obj.fit_result.loads(payload["fit_result"])
        return obj

    def plot(
        self,
        legend="on",
        fig=None,
        ax_res=None,
        ax_fit=None,
        context_data=None,
    ):
        """
        Plot the data points, best fit, fit components, residual,
        and n-sigma band.

        Parameters
        ----------
        legend : string, optional
            Either "on" or "off". The default is "on".
        fig : matplotlib figure object, optional
            plotting figure. The default is None.
        ax_res : axis object, optional
            axis for residual plot. The default is None.
        ax_fit : axis object, optional
            axis for the best-fit plot. The default is None.
        context_data : tuple of (x_array, y_array), optional
            Extra data points to show dimmed behind the fit (e.g. the
            full ROI when the fit range has been trimmed).

        Returns
        -------
        None.

        """
        x = self.x_data
        y = self.y_data
        res = self.fit_result
        x_pred = np.linspace(x[0], x[-1], num=500)
        y_pred = res.eval(x=x_pred)
        comps = res.eval_components()
        if self.search.spectrum.cps and self.search.spectrum.livetime is not None:
            redchi_display = res.redchi * self.search.spectrum.livetime
        else:
            redchi_display = res.redchi

        apply_theme()

        if get_theme() == DARK:
            c_data = "#00e5ff"
            c_fit  = "#ff5577"
            c_bkg  = "#39ff14"
            c_comp = "#b388ff"
            c_band = "#ffffff"
            c_res  = "#ffb300"
            c_subs = ["#ffa726", "#ffee58"]
            a_band = 0.12
        else:
            c_data = "b"
            c_fit  = "r"
            c_bkg  = "g"
            c_comp = "k"
            c_band = "#ABABAB"
            c_res  = None
            c_subs = ["#e65100", "#00838f"]
            a_band = 0.4

        with plt.rc_context({"font.size": 12}):
            if fig is None:
                fig = plt.figure(constrained_layout=True, figsize=(8, 6))
            if ax_res is None or ax_fit is None:
                gs = fig.add_gridspec(2, 1, height_ratios=[1, 4])
            if ax_res is None:
                ax_res = fig.add_subplot(gs[0, 0])
            if ax_fit is None:
                ax_fit = fig.add_subplot(gs[1, 0])
            fig.patch.set_alpha(0.3)

            ax_res.plot(x, res.residual, ".", ms=10, alpha=0.5, color=c_res)
            ax_res.hlines(y=0, xmin=x.min(), xmax=x.max(), lw=3)
            ax_res.set_ylabel("Residual")
            ax_res.set_xlim([x.min(), x.max()])
            ax_res.set_xticks([])

            ax_fit.set_title(rf"Reduced $\chi^2$ = {redchi_display:.4f}")
            if context_data is not None:
                cx, cy = context_data
                ax_fit.plot(cx, cy, "o", color=c_data, alpha=0.15, ms=5)
            ax_fit.plot(x, y, "o", color=c_data, alpha=0.5)
            ax_fit.plot(x_pred, y_pred, color=c_fit, lw=3, alpha=0.5, label="Fit")
            ax_fit.plot(
                x, comps[self.bkg_key], "--", color=c_bkg, lw=3,
                label=f"Bkg: {self.bkg}", drawstyle=self._bkg_drawstyle(),
            )
            n_gauss = len(comps) - 1
            sub_label_used = set()
            for cp in range(n_gauss):
                label = "Comp." if cp == 0 else None
                comp_curve = self._component_curve(comps, cp)
                ax_fit.plot(
                    x, comp_curve, "--", color=c_comp, lw=2, label=label,
                )
                base = comp_curve - comps[f"g{cp+1}_"]
                for si, (y_sub, sub_label) in enumerate(
                    self._sub_component_curves(cp, x)
                ):
                    show_label = sub_label if sub_label not in sub_label_used else None
                    sub_label_used.add(sub_label)
                    ax_fit.plot(
                        x, base + y_sub, ":",
                        color=c_subs[si % len(c_subs)],
                        lw=2, label=show_label,
                    )

            dely = safe_eval_uncertainty(res, x=x_pred, sigma=3)
            ax_fit.fill_between(
                x_pred,
                y_pred - dely,
                y_pred + dely,
                color=c_band,
                alpha=a_band,
                label=r"3-$\sigma$",
            )
            ax_fit.set_xlabel(self.x_units)
            if legend == "on":
                ax_fit.legend(
                    loc="upper right",
                    ncol=2, frameon=False, fontsize=10,
                )

    def optimize_xrange(
        self, max_extend=2.0, n_steps=10, symmetric=False, verbose=False
    ):
        """
        Optimize the fit range by widening the left and right edges and
        selecting the range whose reduced chi-squared is closest to 1.

        By default the two edges are searched independently (asymmetric
        grid). The candidate range is

            [x0 - δ_left, x1 + δ_right]

        with each δ ∈ [0, max_extend * avg_fwhm]. Set ``symmetric=True`` to
        force δ_left == δ_right (the old behaviour), which is cheaper
        (n_steps trials vs n_steps² trials).

        The objective is min |redchi - 1|, which penalises both under-fitting
        (redchi >> 1, range too narrow to constrain the background) and
        over-fitting / noise inclusion (redchi << 1).

        After the search the instance is updated in-place with the best fit.

        Parameters
        ----------
        max_extend : float, optional
            Maximum number of FWHMs by which each edge may be moved outward
            beyond the original xrange. The default is 2.0.
        n_steps : int, optional
            Number of candidate offsets per edge. With ``symmetric=False``
            the total number of trial fits is ``n_steps**2``. The default
            is 10.
        symmetric : bool, optional
            If True, search only a single δ applied to both edges. The
            default is False.
        verbose : bool, optional
            If True, print a summary table of all trials. The default is False.

        Returns
        -------
        best_xrange : list
            The [left, right] range that produced the best fit.
        best_redchi : float
            The reduced chi-squared of the best fit.
        """
        x0_orig, x1_orig = self.xrange

        # --- Estimate average FWHM in the fit region as the natural step unit ---
        _, pidx = self.find_peaks_range()
        mask_peaks = (self.search.peaks_idx >= pidx[0]) & (self.search.peaks_idx <= pidx[-1])
        avg_fwhm = float(np.mean(self.search.fwhm_guess[mask_peaks]))
        delta_max = max_extend * avg_fwhm

        deltas = np.linspace(0.0, delta_max, n_steps)
        if symmetric:
            candidates = [(d, d) for d in deltas]
        else:
            candidates = [(dl, dr) for dl in deltas for dr in deltas]

        # Carry constructor kwargs that aren't reconstructible from the
        # required positional args (skew, shared_sigma, hints would
        # otherwise silently revert to defaults).
        cls = type(self)
        trial_kwargs = dict(
            bkg=self.bkg,
            skew=self.skew,
            shared_sigma=self.shared_sigma,
            hints=self.hints,
        )

        best_pair    = (0.0, 0.0)
        best_redchi  = self.fit_result.redchi
        best_obj     = abs(best_redchi - 1.0)
        best_fit_obj = None
        results = []

        for dl, dr in candidates:
            trial_range = [x0_orig - dl, x1_orig + dr]
            try:
                trial = cls(self.search, trial_range, **trial_kwargs)
                rc = trial.fit_result.redchi
                obj = abs(rc - 1.0)
                results.append((dl, dr, trial_range, rc, obj))
                if obj < best_obj:
                    best_obj     = obj
                    best_pair    = (dl, dr)
                    best_redchi  = rc
                    best_fit_obj = trial
            except Exception:
                results.append((dl, dr, trial_range, None, None))

        if verbose:
            print(
                f"\n{'δL':>8}  {'δR':>8}  {'left':>10}  {'right':>10}  "
                f"{'redchi':>10}  {'|χ²-1|':>10}"
            )
            print("-" * 64)
            for dl, dr, rng, rc, obj in results:
                rc_str  = f"{rc:.4f}"  if rc  is not None else "   failed"
                obj_str = f"{obj:.4f}" if obj is not None else "        "
                print(
                    f"{dl:8.4f}  {dr:8.4f}  {rng[0]:10.4f}  {rng[1]:10.4f}  "
                    f"{rc_str:>10}  {obj_str:>10}"
                )
            dl_b, dr_b = best_pair
            print(
                f"\nBest (δL, δR) = ({dl_b:.4f}, {dr_b:.4f})  →  xrange = "
                f"[{x0_orig - dl_b:.4f}, {x1_orig + dr_b:.4f}]  "
                f"redchi = {best_redchi:.4f}"
            )

        # --- Update instance with best fit ---
        if best_fit_obj is not None:
            self.xrange     = best_fit_obj.xrange
            self.x_data     = best_fit_obj.x_data
            self.y_data     = best_fit_obj.y_data
            self.fit_result = best_fit_obj.fit_result
            self.peak_info  = best_fit_obj.peak_info
            self.peak_err   = best_fit_obj.peak_err
            self.continuum  = best_fit_obj.continuum
            self.bkg_key    = best_fit_obj.bkg_key

        return self.xrange, best_redchi


class GaussianComponents:
    def __init__(self, fit_obj_lst=None, df_peak=None):
        """
        Extract background subtracted Gaussian components and plot them
        together with a table of relevant values and uncertainties.

        Parameters
        ----------
        fit_obj_lst : list, optional
            list of PeakFit objects. The default is None.
        df_peak : pandas dataframe, optional
            dataframe of peak info as saved by using the class AddPeaks.
            The default is None.

        Returns
        -------
        None.

        """
        self.fit_obj_lst = fit_obj_lst
        self.df_peak = df_peak
        self.npeaks = 0
        self.mean = []
        self.area = []
        self.fwhm = []
        self.gauss = []
        self.x_data = []
        self.peak_err = []
        if fit_obj_lst is not None:
            self.x_units = fit_obj_lst[0].x_units
            self.gauss_peakfit()
        elif df_peak is not None:
            self.x_units = df_peak.loc[0, "x_units"]
            self.gauss_df()

    def gauss_peakfit(self):
        for fit_obj in self.fit_obj_lst:
            npeaks = len(fit_obj.peak_info)
            self.npeaks += npeaks
            comps = fit_obj.fit_result.eval_components()
            for i in range(npeaks):
                self.peak_err.append(fit_obj.peak_err[i])
                x_data = fit_obj.x_data
                self.x_data.append(x_data)
                mean = fit_obj.peak_info[i]["mean"]
                area = fit_obj.peak_info[i]["area"]
                fwhm = fit_obj.peak_info[i]["fwhm"]
                # Address gaussian components by their known prefix instead
                # of relying on dict-insertion order in eval_components().
                gauss = comps[f"g{i+1}_"]
                self.mean.append(mean)
                self.area.append(area)
                self.fwhm.append(fwhm)
                self.gauss.append(gauss)

    def gauss_df(self):
        self.npeaks = self.df_peak.index.shape[0]
        for i in self.df_peak.index:
            self.x_data.append(self.df_peak.loc[i, "x_data"])
            self.gauss.append(self.df_peak.loc[i, "gauss"])
            self.mean.append(self.df_peak.loc[i, "mean"])
            self.area.append(self.df_peak.loc[i, "area"])
            self.fwhm.append(self.df_peak.loc[i, "fwhm"])
            dict_err = {}
            dict_err["mean_err"] = self.df_peak.loc[i, "mean_err"]
            dict_err["area_err"] = self.df_peak.loc[i, "area_err"]
            dict_err["fwhm_err"] = self.df_peak.loc[i, "fwhm_err"]
            self.peak_err.append(dict_err)

    def plot_gauss(self, plot_type="simple", fig=None, ax=None):
        """
        Plot background subtracted Gaussian components.

        Parameters
        ----------
        plot_type : string, optional
            Either "simple" or "fwhm". The default is "simple".
        fig : matplotlib figure object, optional
            plotting figure. The default is None.
        ax : matplotlib Axes object, optional
            axis for plotting. The default is None.

        Returns
        -------
        None.
        """
        apply_theme()
        with plt.rc_context({"font.size": 12}):
            if fig is None:
                fig = plt.figure(figsize=(12, 8))
            if ax is None:
                ax = fig.add_subplot()

            if plot_type == "simple":
                for i in range(self.npeaks):
                    x = self.x_data[i]
                    y = self.gauss[i]
                    ax.fill_between(x, 0, y, alpha=0.2)
                    x0 = round(self.mean[i], 2)
                    y0 = y.max()
                    a = round(self.area[i], 2)
                    str0 = f"mean={x0} \narea={a}"
                    ax.text(x0, y0, str0)
                ax.set_xlabel(self.x_units)
                ax.set_ylabel("Cts")

            elif plot_type == "fwhm":
                for i in range(self.npeaks):
                    x = self.x_data[i]
                    y = self.gauss[i]
                    ax.fill_between(x, 0, y, alpha=0.2)
                    x0 = self.mean[i]
                    y0 = y.max()
                    ax.text(
                        x0,
                        y0,
                        int(i + 1),
                        bbox=dict(facecolor="red", alpha=0.1),
                        weight="bold",
                    )
                ax.set_xlabel(self.x_units)
                ax.set_ylabel("Cts")
                ax.set_title("Gaussian Components")


class AddPeaks:
    def __init__(self, filename, n=0, overwrite=False):
        """
        Save peak fit objects to a pandas dataframe for further analysis.

        Parameters
        ----------
        filename : str or pathlib.Path
            Path to the hdf file. The ``.hdf`` suffix is appended if missing
            (so both ``"out"`` and ``"out.hdf"`` resolve to ``out.hdf``).
        n : integer, optional
            Add peaks to an existing hdf file with n being the row number
            to append peaks. The default is 0.
        overwrite : bool, optional
            When n == 0 and the target hdf file already exists, the previous
            behaviour was to silently overwrite it. Set ``overwrite=True`` to
            opt in. The default is False, which raises FileExistsError.

        Returns
        -------
        None.

        """
        # Accept Path or str; ensure exactly one ".hdf" suffix.
        hdf_path = Path(filename)
        if hdf_path.suffix != ".hdf":
            hdf_path = hdf_path.with_suffix(hdf_path.suffix + ".hdf")
        self.hdf_path = hdf_path
        # Keep .filename as the suffix-less stem for backward compatibility
        # with downstream code that did f"{self.filename}.hdf".
        self.filename = str(hdf_path.with_suffix(""))
        self.all_peaks = []
        self.n = n
        if n == 0:
            if hdf_path.exists() and not overwrite:
                raise FileExistsError(
                    f"{hdf_path} already exists. Pass overwrite=True to "
                    "replace it, or pass n > 0 to append to existing rows."
                )
            cols = [
                "x_data",
                "y_data",
                "mean",
                "area",
                "fwhm",
                "best_fit",
                "redchi",
                "gauss",
                "uncertainty",
                "bkg",
                "bkg_type",
                "mean_err",
                "area_err",
                "fwhm_err",
                "x_units",
            ]
            self.df = pd.DataFrame(columns=cols)
            self.df.to_hdf(str(hdf_path), key="data")
        else:
            print(f"Appending to existing file: {hdf_path}")
            self.df = pd.read_hdf(str(hdf_path), key="data")

    def add_peak(self, fit_obj):
        self.all_peaks.append(fit_obj)
        npeaks = len(fit_obj.peak_info)

        x_data = fit_obj.x_data
        y_data = fit_obj.y_data
        best_fit = fit_obj.fit_result.best_fit
        redchi = fit_obj.fit_result.redchi
        comps = fit_obj.fit_result.eval_components()
        uncertainty = safe_eval_uncertainty(fit_obj.fit_result)
        bkg_type = fit_obj.bkg
        for i in range(npeaks):
            self.df.loc[self.n, "x_data"] = x_data
            self.df.loc[self.n, "y_data"] = y_data
            self.df.loc[self.n, "mean"] = fit_obj.peak_info[i]["mean"]
            self.df.loc[self.n, "area"] = fit_obj.peak_info[i]["area"]
            self.df.loc[self.n, "fwhm"] = fit_obj.peak_info[i]["fwhm"]
            self.df.loc[self.n, "best_fit"] = best_fit
            self.df.loc[self.n, "redchi"] = redchi
            self.df.loc[self.n, "bkg"] = comps[fit_obj.bkg_key]
            self.df.loc[self.n, "gauss"] = comps[f"g{i+1}_"]
            self.df.loc[self.n, "uncertainty"] = uncertainty
            self.df.loc[self.n, "bkg_type"] = bkg_type
            self.df.loc[self.n, "mean_err"] = fit_obj.peak_err[i]["mean_err"]
            self.df.loc[self.n, "area_err"] = fit_obj.peak_err[i]["area_err"]
            self.df.loc[self.n, "fwhm_err"] = fit_obj.peak_err[i]["fwhm_err"]
            self.df.loc[self.n, "x_units"] = fit_obj.x_units
            self.n += 1
        self.df.to_hdf(str(self.hdf_path), key="data")

    def reset(self):
        self.all_peaks = []
        self.n = 0

    def del_peak(self, pos):
        self.all_peaks.pop(pos)


def consecutive(data, stepsize=0):
    idx = np.where(np.diff(data[:, 1]) != stepsize)[0] + 1
    return np.split(data, idx)


def auto_range(search, fwhm_factor):
    """
    Group nearby peaks into fit windows.

    Returned ranges are expressed in the same units PeakFit's xrange expects:
    energies if the spectrum is calibrated, otherwise channels.
    """
    f = fwhm_factor
    pidx = search.peaks_idx
    chan = search.spectrum.channels[:-1]
    n_chan = len(search.spectrum.channels)
    energies = search.spectrum.energies

    fwhm_guess = search.fwhm_guess
    density = sum((abs(xi - chan) < f * fw) for xi, fw in zip(pidx, fwhm_guess))

    dens2 = np.vstack((chan, density)).T
    rs = consecutive(dens2)
    ranges = []
    for arr in rs:
        if arr[:, 1].sum() != 0 and np.isin(arr[:, 0], pidx).sum() > 0:
            mi = arr[:, 0].min()
            ma = arr[:, 0].max()
            left = int(round(mi - 2 * search.fwhm(mi)))
            right = int(round(ma + 2 * search.fwhm(ma)))
            # Clamp to the valid channel range — previously this clamped to
            # pidx.max(), which is the last peak channel, not the last bin.
            left = max(left, 0)
            right = min(right, n_chan - 1)
            if energies is not None:
                ranges.append([float(energies[left]), float(energies[right])])
            else:
                ranges.append([left, right])

    return ranges


def auto_scan(search, xlst=None, bkglst=None, plot=False):
    """
    scan a PeakSearch object either automatically or with given xrange and bkg
    lists

    Parameters
    ----------
    search : PeakSearch object
        from peaksearch.py.
    xlst : list, optional
        list of x ranges either in channels or energies
        (as defined by the search object). The default is None.
    bkglst : list, optional
        list of strings specifying the background type for each xrange in
        xlst (e.g. poly3). The default is None.
    plot : bool, optional
        plot each PeakFit object. The default is False.

    Returns
    -------
    fits : list
        PeakFit objects.

    """
    fits = []
    if xlst is None and bkglst is None:
        ranges = auto_range(search, 2)
        bkgs = ["poly1", "poly2"]
        for rg in ranges:
            redchi = 1e10
            fitx = None
            for bk in bkgs:
                try:
                    fit0 = PeakFit(search, rg, bkg=bk)
                except ValueError:
                    continue
                if "Fit succeeded." != fit0.fit_result.message:
                    continue
                elif fit0.fit_result.redchi < redchi:
                    fitx = fit0
                    redchi = fitx.fit_result.redchi
            if fitx is None:
                warnings.warn(
                    f"No successful fit found for range {rg}. Skipped.",
                    UserWarning,
                )
                continue
            if plot:
                fitx.plot()
            fits.append(fitx)
    elif xlst is not None and bkglst is not None:
        for rg, bk in zip(xlst, bkglst):
            fit0 = PeakFit(search, rg, bkg=bk)
            if plot:
                fit0.plot()
            fits.append(fit0)
    return fits
