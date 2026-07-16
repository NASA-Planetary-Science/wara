"""
Energy calibration functions
"""
import lmfit
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
from math import comb
from scipy.stats import linregress
from wara.matplotlib_theme import apply_theme
from wara.fit_utils import safe_eval_uncertainty


class EnergyCalibration:
    def __init__(self, mean_vals, erg, channels, n=1, e_units="keV"):
        """
        Perform energy calibration with LMfit. Polynomial degree = n.

        Parameters
        ----------
        mean_vals : list or numpy array
            mean values of fitted peaks (in channel numbers).
        erg : list or numpy array
            energy values corresponding to mean_vals.
        channels : list or numpy array
            channel values.
        n : integer, optional
            polynomial degree. The default is 1.
        e_units : string, optional
            energy units. The default is "keV".

        Returns
        -------
        None.
        """
        if len(mean_vals) != len(erg):
            raise ValueError(
                f"mean_vals and erg must have the same length, "
                f"got {len(mean_vals)} and {len(erg)}"
            )
        if len(mean_vals) < n + 1:
            raise ValueError(
                f"Need at least {n + 1} points to fit a degree-{n} polynomial, "
                f"got {len(mean_vals)}"
            )
        self.mean_vals = np.array(mean_vals, dtype=float)
        self.erg = np.array(erg, dtype=float)
        self.channels = np.array(channels, dtype=float)
        self.n = n
        self.e_units = e_units
        self.predicted = None
        self.fit = None
        self._calibrate()

    def _calibrate(self):
        """
        Run the polynomial calibration fit.

        Returns
        -------
        None.
        """
        poly_mod = lmfit.models.PolynomialModel(degree=self.n)
        pars = poly_mod.guess(self.erg, x=self.mean_vals)
        self.fit = poly_mod.fit(self.erg, params=pars, x=self.mean_vals)
        self.predicted = self.fit.eval(x=self.channels)
        ss_res = np.sum(self.fit.residual ** 2)
        ss_tot = np.sum((self.erg - self.erg.mean()) ** 2)
        self.rsquared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    def _build_equation(self):
        """
        Build a string representation of the calibration equation.

        Returns
        -------
        string
            equation string for plot labels.
        """
        coeffs = list(self.fit.best_values.values())
        terms = [f"${coeffs[0]:.3E}$", f"${coeffs[1]:.3E}x$"]
        for i, c in enumerate(coeffs):
            if i >= 2:
                terms.append(rf"${c:.3E}x^{i}$")
        return " + ".join(terms)

    def metadata(self):
        """
        Return calibration metadata as a dictionary.

        Returns
        -------
        dict
            dictionary containing calibration parameters and results.
        """
        return {
            "n": self.n,
            "e_units": self.e_units,
            "mean_vals": self.mean_vals,
            "erg": self.erg,
            "rsquared": self.rsquared if self.fit is not None else None,
            "coefficients": list(self.fit.best_values.values()) if self.fit is not None else None,
        }

    def plot(
        self,
        residual=True,
        ax_fit=None,
        ax_res=None,
    ):
        ye = safe_eval_uncertainty(self.fit, x=self.mean_vals)
        equation = self._build_equation()
    
        plt.rc("font", size=14)
        apply_theme()
        x_offset = 100
    
        if residual:
            if ax_res is None or ax_fit is None:
                fig = plt.figure(constrained_layout=False, figsize=(12, 8))
                gs = fig.add_gridspec(2, 1, height_ratios=[1, 4])
                ax_res = fig.add_subplot(gs[0, 0])
                ax_fit = fig.add_subplot(gs[1, 0])
            ax_res.plot(
                self.mean_vals, self.fit.residual, ".", ms=15, alpha=0.5, color="red"
            )
            ax_res.hlines(
                y=0,
                xmin=min(self.channels) - x_offset,
                xmax=max(self.channels),
                lw=3,
            )
            ax_res.set_ylabel("Residual")
            ax_res.set_xlim([min(self.channels) - x_offset, max(self.channels)])
            ax_res.set_xticks([])
        else:
            if ax_fit is None:
                fig = plt.figure(constrained_layout=False, figsize=(12, 8))
                ax_fit = fig.add_subplot()
    
        ax_fit.set_title(rf"$R^2$ = {self.rsquared:.6f}")
        ax_fit.errorbar(
            self.mean_vals,
            self.erg,
            yerr=ye,
            ecolor="red",
            elinewidth=5,
            capsize=12,
            capthick=3,
            marker="s",
            mfc="black",
            mec="black",
            markersize=7,
            ls=" ",
            lw=3,
            label="Data",
        )
        ax_fit.plot(
            self.channels,
            self.predicted,
            ls="--",
            lw=3,
            color="green",
            label=f"Predicted: {equation}",
        )
        ax_fit.set_xlim([min(self.channels) - x_offset, max(self.channels)])
        ax_fit.set_xlabel("Channels")
        ax_fit.set_ylabel(f"Energy [{self.e_units}]")
        ax_fit.legend()
    
        return ax_fit


class PiecewiseLinearCalibration:
    """
    Two-segment piecewise linear energy calibration:  E = f(channel).

    The lower segment (E < e_break) is fitted freely via linear regression.
    The upper segment (E >= e_break) is constrained to pass through the
    breakpoint (ch_break, e_break) derived from the lower fit, so the two
    segments are always continuous.

    Parameters
    ----------
    mean_vals : list or numpy array
        Mean channel positions of fitted calibration peaks.
    erg : list or numpy array
        Known energy values corresponding to mean_vals (same units as e_units).
    channels : list or numpy array
        Full channel array of the spectrum (used to build the predicted
        energy array and for plotting).
    e_break : float, optional
        Energy at which the slope is allowed to change.  Calibration points
        are split into E < e_break (lower) and E >= e_break (upper).
        Default is 3000.0.
    e_units : str, optional
        Label for energy axis.  Default is "keV".

    Attributes
    ----------
    predicted : numpy array
        Energy predicted for every element of *channels*.
    slope1, intercept1 : float
        Coefficients of the lower segment:  E = slope1·ch + intercept1.
    slope2 : float
        Slope of the upper segment:  E = e_break + slope2·(ch − ch_break).
    ch_break : float
        Channel that maps to e_break under the lower-segment equation.
    r2_lower, r2_upper : float
        Coefficient of determination for each segment.
    """

    def __init__(self, mean_vals, erg, channels, e_break=3000.0, e_units="keV"):
        if len(mean_vals) != len(erg):
            raise ValueError(
                f"mean_vals and erg must have the same length, "
                f"got {len(mean_vals)} and {len(erg)}"
            )
        self.mean_vals = np.array(mean_vals, dtype=float)
        self.erg = np.array(erg, dtype=float)
        self.channels = np.array(channels, dtype=float)
        self.e_break = float(e_break)
        self.e_units = e_units

        # Populated by _calibrate()
        self.predicted = None
        self.slope1 = None
        self.intercept1 = None
        self.slope2 = None
        self.ch_break = None
        self.r2_lower = None
        self.r2_upper = None

        self._calibrate()

    # ------------------------------------------------------------------
    # Core fitting
    # ------------------------------------------------------------------

    def _calibrate(self):
        """Fit the two continuous linear segments."""
        mask_low = self.erg < self.e_break
        mask_high = ~mask_low

        n_low = int(mask_low.sum())
        n_high = int(mask_high.sum())

        if n_low < 2:
            raise ValueError(
                f"Need at least 2 calibration points below e_break={self.e_break} "
                f"{self.e_units}, got {n_low}. "
                "Lower e_break or add more low-energy calibration peaks."
            )
        if n_high < 2:
            raise ValueError(
                f"Need at least 2 calibration points at or above "
                f"e_break={self.e_break} {self.e_units}, got {n_high}. "
                "Raise e_break or add more high-energy calibration peaks."
            )

        # ---- lower segment: free linear fit --------------------------------
        res1 = linregress(self.mean_vals[mask_low], self.erg[mask_low])
        self.slope1 = float(res1.slope)
        self.intercept1 = float(res1.intercept)
        self.r2_lower = float(res1.rvalue ** 2)

        # Channel at which the lower segment reaches e_break
        self.ch_break = (self.e_break - self.intercept1) / self.slope1

        # ---- upper segment: forced through (ch_break, e_break) -------------
        # Model:  E - e_break = slope2 · (ch - ch_break)
        # Ordinary least squares with no intercept term.
        x_u = self.mean_vals[mask_high] - self.ch_break
        y_u = self.erg[mask_high] - self.e_break
        self.slope2 = float(np.dot(x_u, y_u) / np.dot(x_u, x_u))

        # R² for upper segment (relative to its own mean)
        y_pred_u = self.slope2 * x_u
        ss_res = float(np.sum((y_u - y_pred_u) ** 2))
        ss_tot = float(np.sum((y_u - y_u.mean()) ** 2))
        self.r2_upper = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

        # ---- predicted energies for every channel --------------------------
        below = self.channels <= self.ch_break
        self.predicted = np.where(
            below,
            self.slope1 * self.channels + self.intercept1,
            self.e_break + self.slope2 * (self.channels - self.ch_break),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_equations(self):
        """Return (eq_lower, eq_upper) as LaTeX strings."""
        eq1 = (
            rf"$E = {self.slope1:.4E}\,ch + {self.intercept1:.4E}$"
        )
        eq2 = (
            rf"$E = {self.e_break:.1f} + {self.slope2:.4E}"
            rf"\,(ch - {self.ch_break:.1f})$"
        )
        return eq1, eq2

    def channel_to_energy(self, ch):
        """
        Convert a single channel value (or array) to energy.

        Parameters
        ----------
        ch : float or numpy array

        Returns
        -------
        float or numpy array
        """
        ch = np.asarray(ch, dtype=float)
        return np.where(
            ch <= self.ch_break,
            self.slope1 * ch + self.intercept1,
            self.e_break + self.slope2 * (ch - self.ch_break),
        )

    def metadata(self):
        """
        Return a dictionary of calibration parameters.

        Returns
        -------
        dict
        """
        return {
            "e_break": self.e_break,
            "e_units": self.e_units,
            "ch_break": self.ch_break,
            "mean_vals": self.mean_vals.tolist(),
            "erg": self.erg.tolist(),
            "slope1": self.slope1,
            "intercept1": self.intercept1,
            "r2_lower": self.r2_lower,
            "slope2": self.slope2,
            "r2_upper": self.r2_upper,
        }

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(self, ax=None):
        """
        Plot the two-segment calibration curve alongside the data points.

        Parameters
        ----------
        ax : matplotlib Axes, optional
            Existing axes to draw on.  A new figure is created if None.

        Returns
        -------
        ax : matplotlib Axes
        """
        plt.rc("font", size=14)
        apply_theme()

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)

        mask_low = self.erg < self.e_break
        mask_high = ~mask_low
        eq1, eq2 = self._build_equations()

        # --- calibration data points ----------------------------------------
        ax.plot(
            self.mean_vals[mask_low],
            self.erg[mask_low],
            "s",
            color="steelblue",
            ms=10,
            zorder=5,
            label=f"Lower data (n={mask_low.sum()})",
        )
        ax.plot(
            self.mean_vals[mask_high],
            self.erg[mask_high],
            "^",
            color="darkorange",
            ms=10,
            zorder=5,
            label=f"Upper data (n={mask_high.sum()})",
        )

        # --- fitted lines ----------------------------------------------------
        ch_all = self.channels
        ch_lo = ch_all[ch_all <= self.ch_break]
        ch_hi = ch_all[ch_all > self.ch_break]

        ax.plot(
            ch_lo,
            self.slope1 * ch_lo + self.intercept1,
            "--",
            lw=2.5,
            color="steelblue",
            label=f"Seg 1: {eq1}",
        )
        ax.plot(
            ch_hi,
            self.e_break + self.slope2 * (ch_hi - self.ch_break),
            "--",
            lw=2.5,
            color="darkorange",
            label=f"Seg 2: {eq2}",
        )

        # --- continuity breakpoint marker -----------------------------------
        ax.axvline(
            self.ch_break,
            ls=":",
            lw=2,
            color="gray",
            label=(
                f"Breakpoint: ch={self.ch_break:.1f}, "
                f"E={self.e_break:.1f} {self.e_units}"
            ),
        )
        ax.plot(self.ch_break, self.e_break, "o", color="gray", ms=10, zorder=6)

        # --- labels ---------------------------------------------------------
        x_offset = (self.channels.max() - self.channels.min()) * 0.02
        ax.set_xlim([self.channels.min() - x_offset, self.channels.max() + x_offset])
        ax.set_xlabel("Channels")
        ax.set_ylabel(f"Energy [{self.e_units}]")
        ax.set_title(
            rf"Piecewise linear calibration — "
            rf"$R^2_{{low}}$ = {self.r2_lower:.4f},  "
            rf"$R^2_{{high}}$ = {self.r2_upper:.4f}"
        )
        ax.legend(fontsize=10, loc="upper left")

        return ax

    
def smart_calibration(
    channels: list,
    energies: list,
    n: int = 1,
    min_points: int = 3,
    require_monotonic: bool = True,
    max_combinations: int = 100_000):
    """
    Automatically find the best energy calibration E = f(channel) by
    exhaustively searching over all monotone-preserving subsets of the larger
    input list paired against the full smaller list.

    Supports linear (n=1) and quadratic (n=2) polynomial fits.

    The two input lists can have different lengths. The algorithm fixes the
    shorter list and iterates over all C(N, k) size-k subsets of the longer
    list (where k = min(len(channels), len(energies))), scoring each candidate
    pairing by its coefficient of determination R².

    Parameters
    ----------
    channels : list
        Detected peak positions in channel number. Need not be sorted.
    energies : list
        Known reference energies in keV. Need not be sorted.
    n : int, optional
        Polynomial degree of the calibration fit.  Must be 1 (linear,
        default) or 2 (quadratic).
    min_points : int, optional
        Minimum number of channel–energy pairs required (default 3).
        Must be at least n + 1.
    require_monotonic : bool, optional
        Reject fits that are not strictly increasing over the fitted channel
        range, which are physically meaningless for standard detector
        geometries (default True).  For n=1 this is equivalent to requiring
        a positive slope.
    max_combinations : int, optional
        Maximum number of subset combinations to search before raising
        an error. The default is 100_000.

    Returns
    -------
    best : dict
        Dictionary of best fitting values:

        Always present
            ``c0`` intercept, ``c1`` linear coefficient, ``r2``,
            ``channels``, ``energies``, ``n``.
        n=2 only
            ``c2`` quadratic coefficient.

    Raises
    ------
    ValueError
        If n is not 1 or 2, if fewer than ``max(min_points, n+1)`` channels
        or energies are supplied, or if no valid monotonic fit is found.

    Examples
    --------
    >>> channels = [237, 1764, 351, 609, 1120, 1460]
    >>> energies = [121.8, 344.3, 778.9, 964.1, 1112.1, 1408.0]
    >>> result = smart_calibration(channels, energies)
    >>> print(result)

    >>> result_q = smart_calibration(channels, energies, n=2)
    >>> print(result_q)
    """
    if n not in (1, 2):
        raise ValueError(f"n must be 1 or 2; got {n}.")

    channels = np.sort(np.asarray(channels, dtype=float))
    energies = np.sort(np.asarray(energies, dtype=float))

    n_ch = len(channels)
    n_en = len(energies)
    k = min(n_ch, n_en)

    # Need at least n+1 points to uniquely determine an n-degree polynomial,
    # and at least min_points for a statistically meaningful fit.
    required = max(min_points, n + 1)
    if n_ch < required:
        raise ValueError(f"Need at least {required} channels for n={n}; got {n_ch}.")
    if n_en < required:
        raise ValueError(f"Need at least {required} energies for n={n}; got {n_en}.")
    if k < n + 1:
        raise ValueError(
            f"The smaller of (channels, energies) has {k} elements, "
            f"but a degree-{n} fit needs at least {n + 1}."
        )

    n_combos = comb(max(n_ch, n_en), k)
    if n_combos > max_combinations:
        raise ValueError(
            f"Too many combinations to search ({n_combos:,}). "
            f"Reduce the number of input points or increase max_combinations."
        )

    best_r2 = -np.inf
    best: dict = {}

    def _is_monotonic_increasing(coeffs_high_first, x_arr):
        """
        Return True if the polynomial with the given coefficients (highest
        degree first, as returned by np.polyfit) is strictly increasing at
        every point in x_arr.

        For n=1: derivative is constant (c1 > 0).
        For n=2: derivative is 2*c2*x + c1; check all x in x_arr.
        """
        poly_deriv = np.polyder(np.poly1d(coeffs_high_first))
        return np.all(poly_deriv(x_arr) > 0)

    def _r2(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    def _evaluate(x_arr, y_arr):
        """Fit and score a single channel–energy pairing."""
        nonlocal best_r2, best

        if n == 1:
            slope, intercept, r, *_ = linregress(x_arr, y_arr)
            coeffs = np.array([slope, intercept])  # high→low: [c1, c0]
            if require_monotonic and slope <= 0:
                return
            r2 = r ** 2
            if r2 > best_r2:
                best_r2 = r2
                best = {
                    "n": 1,
                    "c0": intercept,
                    "c1": slope,
                    "r2": r2,
                    "channels": x_arr.copy(),
                    "energies": y_arr.copy(),
                }

        else:  # n == 2
            # np.polyfit returns [c2, c1, c0] (highest degree first)
            coeffs = np.polyfit(x_arr, y_arr, 2)
            if require_monotonic and not _is_monotonic_increasing(coeffs, x_arr):
                return
            y_pred = np.polyval(coeffs, x_arr)
            r2 = _r2(y_arr, y_pred)
            if r2 > best_r2:
                best_r2 = r2
                best = {
                    "n": 2,
                    "c0": coeffs[2],
                    "c1": coeffs[1],
                    "c2": coeffs[0],
                    "r2": r2,
                    "channels": x_arr.copy(),
                    "energies": y_arr.copy(),
                }

    if n_ch <= n_en:
        # Channels are the fixed axis; choose k energies from the pool.
        for idx in combinations(range(n_en), k):
            _evaluate(channels, energies[list(idx)])
    else:
        # Energies are the fixed axis; choose k channels from the pool.
        for idx in combinations(range(n_ch), k):
            _evaluate(channels[list(idx)], energies)

    if not best:
        raise ValueError(
            "No valid calibration found. All candidate fits were non-monotonic. "
            "Check that channels and energies are in ascending order or set "
            "require_monotonic=False."
        )
    return best


# ---------------------------------------------------------------------------
# Outlier-tolerant automatic calibration (RANSAC-style partial matching)
# ---------------------------------------------------------------------------
def _poly_is_increasing(coeffs_high_first, lo, hi, samples=256):
    """True if the polynomial (np.polyfit order, highest degree first) is
    strictly increasing across [lo, hi]. The derivative is checked on a dense
    grid so curved (quadratic/cubic) fits cannot turn over *between* the fitted
    points — unlike a check at the fitted x-values alone."""
    deriv = np.polyder(np.poly1d(coeffs_high_first))
    xs = np.linspace(lo, hi, samples)
    return bool(np.all(deriv(xs) > 0))


def _align_channels_to_energies(pred, ergs, tol):
    """Order-preserving optimal matching between predicted energies ``pred``
    (one per sorted channel) and candidate ``ergs`` (sorted).

    Returns ``(count, cost, pairs)`` where *pairs* is a list of
    ``(channel_index, energy_index)``. Maximises the number of matches, then
    minimises the total absolute residual. A channel matches an energy only
    when ``|pred_i - erg_j| <= tol``. This is the classic sequence-alignment
    DP, which (because both sequences are sorted) yields a globally consistent,
    one-to-one, monotone pairing — outliers on either side are simply skipped.
    """
    N, M = len(pred), len(ergs)
    # dp[i][j] = (count, cost) for pred[:i] vs ergs[:j]; choice[i][j] records
    # the move taken so we can backtrack the actual pairs.
    dp = [[(0, 0.0)] * (M + 1) for _ in range(N + 1)]
    choice = [[0] * (M + 1) for _ in range(N + 1)]   # 0 skip-ch, 1 skip-e, 2 match

    def better(a, b):
        return a[0] > b[0] or (a[0] == b[0] and a[1] < b[1])

    for i in range(1, N + 1):
        for j in range(1, M + 1):
            best, mv = dp[i - 1][j], 0                 # skip channel i
            if better(dp[i][j - 1], best):
                best, mv = dp[i][j - 1], 1             # skip energy j
            d = abs(pred[i - 1] - ergs[j - 1])
            if d <= tol:
                cand = (dp[i - 1][j - 1][0] + 1, dp[i - 1][j - 1][1] + d)
                if better(cand, best):
                    best, mv = cand, 2                 # match i-1 ↔ j-1
            dp[i][j], choice[i][j] = best, mv

    pairs = []
    i, j = N, M
    while i > 0 and j > 0:
        mv = choice[i][j]
        if mv == 2:
            pairs.append((i - 1, j - 1)); i -= 1; j -= 1
        elif mv == 1:
            j -= 1
        else:
            i -= 1
    pairs.reverse()
    count, cost = dp[N][M]
    return count, cost, pairs


def smart_calibration_auto(
    channels,
    energies,
    n: int = 1,
    tol: float = None,
    min_matches: int = None,
    require_monotonic: bool = True,
    channel_range=None,
    max_exhaustive: int = 200_000,
    max_iters: int = 5000,
    random_state: int = 0):
    """Automatically pair detected peaks with reference energies and calibrate,
    tolerating spurious peaks and absent lines on *both* inputs.

    Unlike :func:`smart_calibration`, this does **not** force every element of
    the shorter list into the fit. It searches for the calibration that maps the
    *largest consistent subset* of channels onto energies (RANSAC-style), so a
    noise peak in ``channels`` or an energy line that is not actually present can
    be left unmatched.

    Algorithm
    ---------
    1. Draw a minimal sample of ``n + 1`` channel↔energy correspondences
       (order-preserving). Small inputs are searched exhaustively (deterministic);
       large inputs fall back to random sampling capped at ``max_iters``.
    2. Fit the exact degree-``n`` polynomial through the sample and reject it if
       it is not strictly increasing across ``channel_range``.
    3. Score the model by how many of *all* channels it maps to within ``tol`` of
       an available energy (optimal order-preserving assignment), breaking ties
       by smallest total residual.
    4. Refit a least-squares degree-``n`` polynomial on the best model's matched
       (inlier) pairs — that refit is the returned calibration.

    Parameters
    ----------
    channels, energies : array-like
        Detected peak positions (channel) and candidate reference energies
        (keV). Need not be sorted or the same length.
    n : int
        Polynomial degree, 1–3.
    tol : float, optional
        Energy match tolerance (keV). Defaults to ``0.3 * (smallest gap between
        candidate energies)``, which protects against matching a peak to the
        wrong (adjacent) line while staying permissive on calibration residuals.
    min_matches : int, optional
        Minimum number of matched pairs to accept a calibration. Defaults to
        ``max(n + 1, 3)``.
    require_monotonic : bool
        Reject calibrations that are not strictly increasing over
        ``channel_range`` (default True).
    channel_range : (lo, hi), optional
        Channel span over which monotonicity is enforced. Defaults to the span
        of the input ``channels``. Pass the spectrum's full ``(0, n_channels)``
        so the calibration is monotonic everywhere it will be applied.
    max_exhaustive : int
        If the number of minimal samples ``C(Nc, n+1) * C(Ne, n+1)`` does not
        exceed this, every sample is tried (deterministic result). Otherwise a
        random subset of ``max_iters`` samples is drawn.
    max_iters : int
        Number of random minimal samples when not searching exhaustively.
    random_state : int
        Seed for the random sampler (reproducible results).

    Returns
    -------
    dict
        ``c0, c1[, c2, c3]`` polynomial coefficients (ascending power),
        ``coeffs`` (same, as a list), ``n``, ``channels`` and ``energies`` (the
        matched, sorted pairs — feed straight into :class:`EnergyCalibration`),
        ``pairs`` (list of ``(channel, energy)``), ``residuals`` (keV, per pair),
        ``r2``, ``rmse``, ``n_matched``, ``unmatched_channels`` and
        ``unmatched_energies``.

    Raises
    ------
    ValueError
        If ``n`` is not 1–3, too few inputs are supplied, or no calibration
        reaches ``min_matches`` valid, monotonic pairs.
    """
    if n not in (1, 2, 3):
        raise ValueError(f"n must be 1, 2 or 3; got {n}.")

    ch = np.sort(np.asarray(channels, dtype=float))
    en = np.sort(np.asarray(energies, dtype=float))
    Nc, Ne = len(ch), len(en)
    m = n + 1
    if min_matches is None:
        min_matches = max(m, 3)
    if Nc < m or Ne < m:
        raise ValueError(
            f"Need at least {m} channels and {m} energies for a degree-{n} fit; "
            f"got {Nc} channels and {Ne} energies.")

    if tol is None:
        gaps = np.diff(en)
        gaps = gaps[gaps > 0]
        tol = 0.3 * float(gaps.min()) if len(gaps) else 1.0
        tol = max(tol, 1e-9)

    if channel_range is None:
        lo, hi = float(ch.min()), float(ch.max())
    else:
        lo, hi = float(channel_range[0]), float(channel_range[1])

    # Enumerate (or sample) minimal channel/energy index subsets.
    n_samples = comb(Nc, m) * comb(Ne, m)
    if n_samples <= max_exhaustive:
        sample_iter = (
            (ci, ei)
            for ci in combinations(range(Nc), m)
            for ei in combinations(range(Ne), m))
    else:
        rng = np.random.default_rng(random_state)
        def _random_samples():
            for _ in range(max_iters):
                ci = tuple(sorted(rng.choice(Nc, size=m, replace=False)))
                ei = tuple(sorted(rng.choice(Ne, size=m, replace=False)))
                yield ci, ei
        sample_iter = _random_samples()

    best_key = (0, np.inf)        # (match count ↑, total cost ↓)
    best_pairs = None
    for ci, ei in sample_iter:
        x = ch[list(ci)]
        y = en[list(ei)]
        if len(np.unique(x)) < m:                 # degenerate sample
            continue
        coeffs = np.polyfit(x, y, n)              # exact through m points
        if require_monotonic and not _poly_is_increasing(coeffs, lo, hi):
            continue
        pred = np.polyval(coeffs, ch)
        count, cost, pairs = _align_channels_to_energies(pred, en, tol)
        if count < min_matches:
            continue
        key = (count, -cost)                      # more matches, then less cost
        if key[0] > best_key[0] or (key[0] == best_key[0] and cost < best_key[1]):
            best_key = (count, cost)
            best_pairs = pairs

    if not best_pairs:
        raise ValueError(
            f"No calibration matched at least {min_matches} peaks to energies "
            f"within {tol:.3g} keV. Loosen `tol`, add points, or check the data.")

    ch_idx = [p[0] for p in best_pairs]
    e_idx = [p[1] for p in best_pairs]
    mx, my = ch[ch_idx], en[e_idx]

    # Final least-squares refit on the matched inliers.
    coeffs = np.polyfit(mx, my, n)                # high → low
    y_pred = np.polyval(coeffs, mx)
    residuals = my - y_pred
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((my - my.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    rmse = float(np.sqrt(ss_res / len(mx)))

    asc = coeffs[::-1]                            # low → high: c0, c1, ...
    result = {
        "n": n,
        "coeffs": [float(c) for c in asc],
        "channels": mx.copy(),
        "energies": my.copy(),
        "pairs": list(zip(mx.tolist(), my.tolist())),
        "residuals": residuals.tolist(),
        "r2": float(r2),
        "rmse": rmse,
        "n_matched": len(mx),
        "unmatched_channels": [float(c) for k, c in enumerate(ch) if k not in ch_idx],
        "unmatched_energies": [float(e) for k, e in enumerate(en) if k not in e_idx],
    }
    for i, name in enumerate(("c0", "c1", "c2", "c3")[:n + 1]):
        result[name] = float(asc[i])
    return result

    