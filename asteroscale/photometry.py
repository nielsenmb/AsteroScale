"""Fast interpolation of packaged Gaia DR3 synthetic photometry.

The compact table is derived from the MARCS-based bolometric-correction grids
of Casagrande & VandenBerg (2014, 2018a, 2018b).  It is regularized offline so
the runtime calculation needs only multilinear interpolation.
"""

from itertools import product
from pathlib import Path

import numpy as np


_DATA_PATH = Path(__file__).with_name("data") / "marcs_gaia_dr3.npz"
with np.load(_DATA_PATH) as _grid:
    TEFF_AXIS = _grid["teff"]
    LOGG_AXIS = _grid["logg"]
    FEH_AXIS = _grid["feh"]
    AG_AXIS = _grid["ag"]
    _BC = _grid["bc"]
    _EXTINCTION = _grid["extinction"]

MARCS_DOMAIN = {
    "Teff": (float(TEFF_AXIS[0]), float(TEFF_AXIS[-1])),
    "logg": (float(LOGG_AXIS[0]), float(LOGG_AXIS[-1])),
    "FeH": (float(FEH_AXIS[0]), float(FEH_AXIS[-1])),
    "A_G": (float(AG_AXIS[0]), float(AG_AXIS[-1])),
}

_BAND_INDEX = {"G": 0, "BP": 1, "RP": 2}
_EXTINCTION_INDEX = {"BP": 0, "RP": 1}

# Ballot et al. (2011) power-law correction from the Kepler response to
# bolometric radial-mode amplitudes.  Ball et al. (2018) adopt a TESS/Kepler
# amplitude ratio of 2.1 / 2.5 for their solar normalization.
KEPLER_BOLOMETRIC_REFERENCE_TEFF = 5934.0
KEPLER_BOLOMETRIC_EXPONENT = 0.8
_AMPLITUDE_RESPONSE = {"KEPLER": 1.0, "TESS": 2.1 / 2.5}


def _normalize_amplitude_mission(mission):
    """Return the canonical name of a supported amplitude response."""
    if not isinstance(mission, str):
        raise ValueError("mission must be a string.")
    canonical = mission.strip().upper()
    if canonical not in _AMPLITUDE_RESPONSE:
        raise ValueError(
            f"Unsupported mission {mission!r}; choose 'TESS' or 'Kepler'."
        )
    return canonical


def kepler_bolometric_correction(Teff, backend=np):
    """Return the Kepler-to-bolometric radial-mode amplitude correction.

    This is the power-law approximation from Ballot et al. (2011).  If
    ``A_Kepler`` is the amplitude measured in the Kepler response, then
    ``A_bolometric = A_Kepler * correction``.

    Parameters
    ----------
    Teff : float or array-like
        Effective temperature in kelvin.
    backend : module, default=numpy
        NumPy-compatible array backend.

    Returns
    -------
    float or ndarray
        Multiplicative correction from Kepler to bolometric amplitude.
    """
    return (
        backend.asarray(Teff) / KEPLER_BOLOMETRIC_REFERENCE_TEFF
    ) ** KEPLER_BOLOMETRIC_EXPONENT


def convert_bolometric_amplitude(amplitude_bolometric, Teff, mission):
    """Convert a bolometric radial-mode RMS amplitude to a mission response.

    Parameters
    ----------
    amplitude_bolometric : float or array-like
        Bolometric radial-mode RMS amplitude in ppm.
    Teff : float or array-like
        Effective temperature in kelvin.
    mission : {'TESS', 'Kepler'}
        Photometric response to predict (case-insensitive).

    Returns
    -------
    float or ndarray
        Radial-mode RMS amplitude in the selected mission response, in ppm.

    Notes
    -----
    The Kepler correction uses the Ballot et al. (2011) power law.  The TESS
    conversion additionally applies the 2.1/2.5 response ratio adopted by
    Ball et al. (2018).  These are approximate empirical conversions.
    """
    mission = _normalize_amplitude_mission(mission)
    correction = kepler_bolometric_correction(Teff)
    return (
        np.asarray(amplitude_bolometric)
        * _AMPLITUDE_RESPONSE[mission]
        / correction
    )


def _bracket(axis, value, backend):
    """Return lower grid indices, fractional positions, and validity."""
    axis = backend.asarray(axis)
    value = backend.asarray(value)
    lower = backend.searchsorted(axis, value, side="right") - 1
    lower = backend.clip(lower, 0, len(axis) - 2)
    left = axis[lower]
    right = axis[lower + 1]
    fraction = (value - left) / (right - left)
    valid = (value >= axis[0]) & (value <= axis[-1])
    return lower, fraction, valid


def _multilinear(table, axes, coordinates, backend):
    """Interpolate an N-dimensional regular table without extrapolation."""
    brackets = [
        _bracket(axis, coordinate, backend)
        for axis, coordinate in zip(axes, coordinates)
    ]
    indices = [item[0] for item in brackets]
    fractions = [item[1] for item in brackets]
    valid = brackets[0][2]
    for _, _, current in brackets[1:]:
        valid = valid & current

    table = backend.asarray(table)
    # Broadcasting all coordinates first gives scalar and array inputs the
    # same advanced-indexing behavior in both NumPy and JAX.
    broadcast = backend.broadcast_arrays(*indices, *fractions)
    ndim = len(indices)
    indices = broadcast[:ndim]
    fractions = broadcast[ndim:]
    trailing_shape = table.shape[ndim:]
    result = backend.zeros(
        backend.shape(indices[0]) + trailing_shape, dtype=float
    )

    for corner in product((0, 1), repeat=ndim):
        corner_indices = tuple(
            index + offset for index, offset in zip(indices, corner)
        )
        weight = backend.ones_like(result, dtype=float)
        for fraction, offset in zip(fractions, corner):
            factor = fraction if offset else 1.0 - fraction
            for _ in trailing_shape:
                factor = factor[..., None]
            weight = weight * factor
        result = result + weight * table[corner_indices]
    for _ in trailing_shape:
        valid = valid[..., None]
    return backend.where(valid, result, backend.nan)


def bolometric_corrections(Teff, logg, FeH, backend=np):
    """Interpolate G, BP, and RP bolometric corrections together.

    Parameters
    ----------
    Teff : float or array-like
        Effective temperature in kelvin.
    logg : float or array-like
        Base-10 surface gravity with gravity in cgs units.
    FeH : float or array-like
        Metallicity in dex.
    backend : module, default=numpy
        NumPy-compatible array backend.

    Returns
    -------
    ndarray
        Corrections in G, BP, RP order along the final axis.
    """
    return _multilinear(
        np.moveaxis(_BC, 0, -1),
        (TEFF_AXIS, LOGG_AXIS, FEH_AXIS),
        (Teff, logg, FeH),
        backend,
    )


def bolometric_correction(Teff, logg, FeH, band, backend=np):
    """Interpolate an unreddened Gaia DR3 bolometric correction.

    Parameters
    ----------
    Teff : float or array-like
        Effective temperature in kelvin.
    logg : float or array-like
        Base-10 surface gravity with gravity in cgs units.
    FeH : float or array-like
        Metallicity in dex.
    band : {'G', 'BP', 'RP'}
        Gaia DR3 passband.
    backend : module, default=numpy
        NumPy-compatible array backend.

    Returns
    -------
    float or ndarray
        Bolometric correction in magnitudes. Values outside the packaged
        MARCS domain are returned as ``nan`` rather than extrapolated.
    """
    try:
        table = _BC[_BAND_INDEX[band.upper()]]
    except (AttributeError, KeyError) as error:
        raise ValueError("band must be one of 'G', 'BP', or 'RP'.") from error
    return _multilinear(
        table,
        (TEFF_AXIS, LOGG_AXIS, FEH_AXIS),
        (Teff, logg, FeH),
        backend,
    )


def extinction_from_ag(A_G, Teff, logg, FeH, band, backend=np):
    """Interpolate Gaia BP or RP extinction from G-band extinction.

    Parameters
    ----------
    A_G : float or array-like
        Gaia G-band extinction in magnitudes.
    Teff : float or array-like
        Effective temperature in kelvin.
    logg : float or array-like
        Base-10 surface gravity with gravity in cgs units.
    FeH : float or array-like
        Metallicity in dex.
    band : {'BP', 'RP'}
        Gaia DR3 passband whose extinction is requested.
    backend : module, default=numpy
        NumPy-compatible array backend.

    Returns
    -------
    float or ndarray
        Passband extinction in magnitudes. Values outside the packaged MARCS
        domain are returned as ``nan`` rather than extrapolated.
    """
    try:
        table = _EXTINCTION[_EXTINCTION_INDEX[band.upper()]]
    except (AttributeError, KeyError) as error:
        raise ValueError("band must be either 'BP' or 'RP'.") from error
    return _multilinear(
        table,
        (TEFF_AXIS, LOGG_AXIS, FEH_AXIS, AG_AXIS),
        (Teff, logg, FeH, A_G),
        backend,
    )


def extinctions_from_ag(A_G, Teff, logg, FeH, backend=np):
    """Interpolate BP and RP extinction together.

    Parameters
    ----------
    A_G : float or array-like
        Gaia G-band extinction in magnitudes.
    Teff : float or array-like
        Effective temperature in kelvin.
    logg : float or array-like
        Base-10 surface gravity with gravity in cgs units.
    FeH : float or array-like
        Metallicity in dex.
    backend : module, default=numpy
        NumPy-compatible array backend.

    Returns
    -------
    ndarray
        Extinctions in BP, RP order along the final axis.
    """
    return _multilinear(
        np.moveaxis(_EXTINCTION, 0, -1),
        (TEFF_AXIS, LOGG_AXIS, FEH_AXIS, AG_AXIS),
        (Teff, logg, FeH, A_G),
        backend,
    )
