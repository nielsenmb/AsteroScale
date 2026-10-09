"""Calibration-domain checks for scaling-relation predictions."""

import warnings

import numpy as np

from .photometry import MARCS_DOMAIN
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
        status = "within_checked_bounds"
    elif fraction == 0.0:
        status = "outside_checked_bounds"
    else:
        status = "partly_outside_checked_bounds"
    return {"status": status, "fraction_within": fraction, "domain": domain,
            "calibration_certified": False}


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

    amplitude_names = active & {"amplitude_bolometric", "A_env"}
    if amplitude_names and all(name in values for name in ("L", "Teff")):
        teff = np.asarray(values["Teff"])
        valid = ((teff < amplitude_red_edge(values["L"]))
                 & (teff >= 4000) & (teff <= 7500))
        for name in amplitude_names:
            report[name] = _entry(
                valid,
                "Teff below the adopted red edge of the delta-Scuti "
                "instability strip and 4000 <= Teff/K <= 7500 for the Kepler correction",
            )

    photometric = {
        "BC_G", "BC_BP", "BC_RP", "A_BP", "A_RP",
        "M_G", "M_BP", "M_RP", "G_mag", "BP_mag", "RP_mag", "BP_RP",
    }
    if active & photometric and all(
        name in values for name in ("Teff", "logg", "FeH")
    ):
        valid = (
            (np.asarray(values["Teff"]) >= MARCS_DOMAIN["Teff"][0])
            & (np.asarray(values["Teff"]) <= MARCS_DOMAIN["Teff"][1])
            & (np.asarray(values["logg"]) >= MARCS_DOMAIN["logg"][0])
            & (np.asarray(values["logg"]) <= MARCS_DOMAIN["logg"][1])
            & (np.asarray(values["FeH"]) >= MARCS_DOMAIN["FeH"][0])
            & (np.asarray(values["FeH"]) <= MARCS_DOMAIN["FeH"][1])
        )
        extinction_active = bool(active & {"A_BP", "A_RP", "BP_mag", "RP_mag", "BP_RP"})
        if extinction_active and "A_G" in values:
            valid = (
                valid
                & (np.asarray(values["A_G"]) >= MARCS_DOMAIN["A_G"][0])
                & (np.asarray(values["A_G"]) <= MARCS_DOMAIN["A_G"][1])
            )
        domain = (
            f"{MARCS_DOMAIN['Teff'][0]:.0f} <= Teff/K <= "
            f"{MARCS_DOMAIN['Teff'][1]:.0f}, "
            f"{MARCS_DOMAIN['logg'][0]:.1f} <= logg <= "
            f"{MARCS_DOMAIN['logg'][1]:.1f}, "
            f"{MARCS_DOMAIN['FeH'][0]:.1f} <= [Fe/H] <= "
            f"{MARCS_DOMAIN['FeH'][1]:.1f}"
        )
        if extinction_active:
            domain += (
                f", {MARCS_DOMAIN['A_G'][0]:.1f} <= A_G/mag <= "
                f"{MARCS_DOMAIN['A_G'][1]:.1f}"
            )
        report["Gaia_photometry"] = _entry(valid, domain)
    if "dnu" in report:
        report["dnu"]["unverified"] = [
            "evolutionary state: early main sequence and core-helium burning are excluded",
            "solar renormalization and systematic accuracy",
        ]
    if "numax" in report:
        report["numax"]["unverified"] = ["ionization, helium abundance and Gamma_1"]
    for name in active & {"A_gran", "b_gran_low", "b_gran_high", "FWHM_env"}:
        report[name] = {"status": "applicability_unverified", "calibration_certified": False,
                        "domain": "Empirical solar-like-oscillator relation; no complete domain check."}
    return report


def warn_outside_calibration(report):
    """Warn once for every relation outside its calibration domain.

    Parameters
    ----------
    report : dict
        Output from :func:`assess_validity`.
    """
    for name, entry in report.items():
        if entry["status"] in {"within_checked_bounds", "applicability_unverified"}:
            continue
        warnings.warn(
            f"{name} is {entry['status'].replace('_', ' ')}: "
            f"{entry['fraction_within']:.1%} of evaluated samples are within "
            f"the adopted domain ({entry['domain']}).",
            UserWarning,
            stacklevel=3,
        )
