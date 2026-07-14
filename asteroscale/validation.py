"""Validation for Solver.solve() inputs -- catches typos and physically
nonsensical combinations early, before running an expensive sampler, rather
than either crashing cryptically mid-run or silently returning a result
that looks fine but isn't.
"""
import warnings

import numpy as np

from .relations import DERIVED, FUNDAMENTAL

_ALL_NAMES = set(FUNDAMENTAL) | set(DERIVED)

# Quantities that are physically meaningless at or below zero.
_POSITIVE = {"M", "R", "Teff", "plx", "numax", "dnu", "L", "d", "rho"}

# Quantities that can be zero but not negative.
_NONNEGATIVE = {"A_G", "A_BP", "A_RP"}

# Generous sanity ranges -- well outside "wrong units" or "typo" territory,
# not tight physical limits. Violating one is a warning, not an error: it
# might be exactly what you mean to explore, but it's also the classic
# "typed nm instead of muHz" mistake.
_SANITY_RANGES = {
    "Teff": (3000.0, 10000.0),
    "FeH": (-2.5, 0.75),
    "M": (0.1, 10.0),
    "R": (0.1, 100.0),
    "A_G": (0.0, 5.0),
    "numax": (0.1, 5000.0),
    "dnu": (0.1, 300.0),
    "plx": (0.001, 800.0),  # Proxima Centauri, the closest star, is ~768 mas
}


def validate_names(names, kind):
    """kind: a short label for the error message, e.g. 'given' or 'want'."""
    unknown = [n for n in names if n not in _ALL_NAMES]
    if unknown:
        raise KeyError(
            f"Unknown quantity name(s) in {kind}: {unknown}. "
            f"Valid names are: {sorted(_ALL_NAMES)}"
        )


def normalize_want(want):
    """Validate and expand a requested output list.

    Parameters
    ----------
    want : str or sequence of str
        Requested quantity names. ``'all'`` and ``['all']`` expand to every
        fundamental and derived quantity.

    Returns
    -------
    list of str
        Validated, expanded quantity names.
    """
    if want == "all" or want == ["all"] or want == ("all",):
        return list(FUNDAMENTAL) + list(DERIVED)
    if isinstance(want, str):
        want = [want]
    if not want:
        raise ValueError("want must contain at least one quantity name.")
    if "all" in want:
        raise ValueError("'all' cannot be combined with individual quantity names.")
    validate_names(want, "want")
    return list(want)


def validate_want(want):
    """Validate requested outputs without returning the normalized list."""
    normalize_want(want)


def validate_given(given):
    """Validates given's structure and, for plain-scalar or (mean, error)
    entries, checks physical plausibility. Custom distribution objects
    (anything with .logpdf/.ppf) are trusted as-is -- there's no generic
    way to sanity-check an arbitrary distribution's shape.
    """
    if not given:
        raise ValueError(
            "given is empty -- solve() needs at least one constraint. "
            "(For a pure prior-predictive draw with no data, sample the "
            "priors directly instead: Solver().priors[name].ppf(...).)"
        )
    validate_names(given.keys(), "given")

    for name, value in given.items():
        if isinstance(value, tuple):
            if len(value) != 2:
                raise ValueError(
                    f"given['{name}'] = {value!r} should be a (mean, error) "
                    f"pair, got a tuple of length {len(value)}."
                )
            mean, err = value
            _check_error(name, err)
            _check_value(name, mean)
        elif isinstance(value, (int, float, np.floating, np.integer)):
            _check_value(name, value)
        elif not (hasattr(value, "logpdf") or hasattr(value, "ppf")):
            raise TypeError(
                f"given['{name}'] = {value!r} isn't a recognized type -- "
                "use a plain number, a (mean, error) tuple, or an object "
                "with .logpdf and/or .ppf (e.g. a frozen scipy.stats "
                "distribution)."
            )


def _check_error(name, err):
    if not np.isfinite(err) or err <= 0:
        raise ValueError(
            f"given['{name}'] error must be a positive finite number, "
            f"got {err}."
        )


def _check_value(name, value):
    if not np.isfinite(value):
        raise ValueError(f"given['{name}'] = {value} is not finite.")
    if name in _POSITIVE and value <= 0:
        raise ValueError(
            f"given['{name}'] = {value} is not physically valid -- "
            f"{name} must be strictly positive."
        )
    if name in _NONNEGATIVE and value < 0:
        raise ValueError(
            f"given['{name}'] = {value} is not physically valid -- "
            f"{name} cannot be negative."
        )
    lo, hi = _SANITY_RANGES.get(name, (-np.inf, np.inf))
    if not (lo <= value <= hi):
        warnings.warn(
            f"given['{name}'] = {value} is well outside the range this "
            f"package is typically used/calibrated for ({lo} to {hi}). "
            "Double check units and value -- if this is intentional, "
            "the result may still be informative but treat it cautiously.",
            stacklevel=4,
        )


def check_point_estimate_residuals(result, targets, tol=1e-3):
    """Warn if the point-estimate least-squares solve didn't actually
    satisfy the given constraints -- most likely because they're mutually
    inconsistent (no combination of the free parameters can match all of
    them at once), rather than a fixable optimizer failure.
    """
    if len(result.fun) == 0:
        return
    max_target = max(1.0, max(abs(v) for v in targets.values()))
    max_resid = np.max(np.abs(result.fun))
    if max_resid > tol * max_target:
        detail = ", ".join(
            f"{name}: target={target:.6g}"
            for name, target in targets.items()
        )
        warnings.warn(
            f"Point estimate did not fully satisfy the given constraints "
            f"(largest residual {max_resid:.4g}). The given values may be "
            f"physically inconsistent with each other, or the free "
            f"parameters couldn't reach a solution within their prior "
            f"bounds. Targets were: {detail}.",
            stacklevel=3,
        )
