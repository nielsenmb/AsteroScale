"""Calibration uncertainty for empirical scaling relations."""

import numpy as np

from .relations import DERIVED


# Fractional one-sigma scatter. The amplitude values are empirical scatter
# reported by Ball et al. (2018) and Kallinger et al. (2014). The seismic
# values are conservative model-discrepancy floors for the lightweight
# scaling relations, not measurement uncertainties.
DEFAULT_RELATION_SCATTER = {
    "numax": 0.02,
    "dnu": 0.015,
    "A_env": 0.25,
    "A_gran": 0.144,
    "b_gran_low": 0.102,
    "b_gran_high": 0.087,
}


def normalize_relation_scatter(scatter, base=None):
    """Validate and merge relation-scatter settings.

    Parameters
    ----------
    scatter : float, dict or None
        Fractional one-sigma scatter. A scalar applies to every calibrated
        relation, a dictionary overrides named relations, and ``None`` keeps
        ``base`` or the package defaults.
    base : dict, optional
        Starting configuration. The package defaults are used when omitted.

    Returns
    -------
    dict
        Validated fractional scatter for all configured relations.

    Raises
    ------
    KeyError
        If a relation name is unknown.
    ValueError
        If any scatter is negative or non-finite.
    TypeError
        If ``scatter`` is not a scalar, mapping, or ``None``.
    """
    result = dict(DEFAULT_RELATION_SCATTER if base is None else base)
    if scatter is None:
        return result
    if np.isscalar(scatter):
        updates = {name: float(scatter) for name in result}
    elif isinstance(scatter, dict):
        updates = scatter
    else:
        raise TypeError("relation_scatter must be a number, dictionary, or None.")

    unknown = set(updates) - set(DERIVED)
    if unknown:
        raise KeyError(f"Unknown relation-scatter name(s): {sorted(unknown)}")
    for name, value in updates.items():
        if not np.isfinite(value) or value < 0:
            raise ValueError(
                f"relation_scatter[{name!r}] must be finite and non-negative, "
                f"got {value!r}."
            )
        result[name] = float(value)
    return result


def fractional_to_log_scatter(fraction):
    """Convert fractional standard deviation to log-normal scatter.

    Parameters
    ----------
    fraction : float or array-like
        Coefficient of variation, ``standard deviation / mean``.

    Returns
    -------
    float or ndarray
        Standard deviation in natural-log space.
    """
    return np.sqrt(np.log1p(np.asarray(fraction) ** 2))
