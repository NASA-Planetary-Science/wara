"""Small shared helpers around lmfit fit results."""
import numpy as np


def safe_eval_uncertainty(fit_result, **kwargs):
    """``ModelResult.eval_uncertainty`` wrapped to silence a benign NumPy warning.

    lmfit builds a finite-difference Jacobian by stepping each varying parameter
    by its standard error (``dval = stderr * dscale``) and dividing the model
    difference by ``2 * dval``. A perfectly-determined coefficient has
    ``stderr == 0`` -- e.g. an exact polynomial calibration through the minimum
    number of points, or a coefficient pinned by collinear data -- which makes
    that step a ``0 / 0`` and emits "invalid value encountered in divide". The
    numerical result is identical with or without the warning, so we suppress
    just that floating-point category here rather than at every call site.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        return fit_result.eval_uncertainty(**kwargs)
