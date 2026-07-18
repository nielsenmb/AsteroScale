"""Calibration-domain checks for scaling-relation predictions."""

import warnings

import numpy as np

from .relations import amplitude_red_edge


def _fraction_true(condition):
    """Return the fraction of scalar or array conditions that are true.

    Parameters
    ----------
    condition : bool or array-like
        Boolean validity mask.

    Returns
    -------
    float
        Fraction of valid entries.
    """
    return float(np.mean(np.asarray(condition, dtype=bool)))


def _entry(condition, domain):
    """Build a serializable validity entry.

    Parameters
    ----------
    condition : bool or array-like
        Boolean validity mask.
    domain : str
        Human-readable calibration domain.

    Returns
    -------
    dict
        Status, valid fraction, and domain description.
    """
    fraction = _fraction_true(condition)
    if fraction == 1.0:
        status = "within_calibration"
    elif fraction == 0.0:
        status = "outside_calibration"
    else:
        status = "partly_outside_calibration"
    return {"status": status, "fraction_within": fraction, "domain": domain}


def assess_validity(values, active_names):
    """Assess calibration domains for active empirical relations.

    Parameters
    ----------
    values : dict
        Fundamental and derived scalar values or posterior arrays.
    active_names : iterable of str
        Relations used by or requested from the current problem.

    Returns
    -------
    dict
        Per-relation status dictionaries. Evolutionary state is not inferred,
        so the large-separation check cannot distinguish RGB and core-helium
        burning stars.
    """
    active = set(active_names)
    report = {}
    if "dnu" in active and all(
        name in values for name in ("M", "Teff", "FeH", "numax")
    ):
        valid = (
            (np.asarray(values["M"]) >= 0.8)
            & (np.asarray(values["M"]) <= 2.0)
            & (np.asarray(values["FeH"]) >= -1.0)
            & (np.asarray(values["FeH"]) <= 0.5)
            & (np.asarray(values["Teff"]) >= 3800.0)
            & (np.asarray(values["Teff"]) <= 7000.0)
            & (np.asarray(values["numax"]) >= 6.0)
        )
        report["dnu"] = _entry(
            valid,
            "0.8 <= M/Msun <= 2.0, -1.0 <= [Fe/H] <= 0.5, "
            "3800 <= Teff/K <= 7000, numax >= 6 microhertz; main sequence "
            "to slightly beyond the RGB bump",
        )

    if "numax" in active and "FeH" in values:
        valid = (
            (np.asarray(values["FeH"]) >= -1.0)
            & (np.asarray(values["FeH"]) <= 0.5)
        )
        report["numax"] = _entry(
            valid,
            "-1.0 <= [Fe/H] <= 0.5 for the adopted molecular-weight "
            "approximation; Gamma_1 is fixed to its solar value",
        )

    if "A_env" in active and all(name in values for name in ("L", "Teff")):
        valid = np.asarray(values["Teff"]) < amplitude_red_edge(values["L"])
        report["A_env"] = _entry(
            valid,
            "Teff below the adopted red edge of the delta-Scuti instability strip",
        )
    return report


def warn_outside_calibration(report):
    """Warn once for every relation outside its calibration domain.

    Parameters
    ----------
    report : dict
        Output from :func:`assess_validity`.
    """
    for name, entry in report.items():
        if entry["status"] == "within_calibration":
            continue
        warnings.warn(
            f"{name} is {entry['status'].replace('_', ' ')}: "
            f"{entry['fraction_within']:.1%} of evaluated samples are within "
            f"the adopted domain ({entry['domain']}).",
            UserWarning,
            stacklevel=3,
        )
