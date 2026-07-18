"""Scaling relations linking fundamental stellar params to observables.

All masses/radii in solar units, Teff in K, numax/dnu in muHz.
"""
import numpy as np

# Array backend, swappable for a jax.jit-able forward pass:
#   import jax.numpy as jnp
#   from asteroscale import relations
#   relations.xp = jnp
# Every function below resolves xp.* at call time (not at import time), so
# flipping this one line is enough to make the whole forward pass
# traceable/jittable -- no separate jax version of this file needed.
xp = np

NUMAX_SUN = 3090.0
TEFF_SUN = 5772.0
GSUN_CGS = 27400.0
MBOL_SUN = 4.74  # IAU absolute bolometric magnitude of the Sun

FUNDAMENTAL = ("M", "R", "Teff", "plx", "A_G", "FeH")


def _metal_mass_fraction(FeH):
    """Z from [Fe/H], assuming solar-scaled composition ([alpha/Fe]=0).
    Calibration from Bertelli et al. 1994 / Salaris et al. 1993, as used in
    e.g. Zinn et al. 2019 to estimate mean molecular weight from [Fe/H].

    Parameters
    ----------
    FeH : float or array-like
        Metallicity in dex relative to solar.

    Returns
    -------
    float or ndarray
        Approximate metal mass fraction.
    """
    return 10 ** (0.977 * FeH - 1.699)


def _mean_molecular_weight(FeH):
    """Fully-ionized H+He mean molecular weight, mu = 4/(3X+1).

    Assumes primordial Y_p=0.248 and helium enrichment dY/dZ=1.0. This
    ignores metals' contribution to free electrons -- fine near solar
    metallicity, increasingly approximate for very metal-poor stars.

    Parameters
    ----------
    FeH : float or array-like
        Metallicity in dex relative to solar.

    Returns
    -------
    float or ndarray
        Fully ionized mean molecular weight.
    """
    Z = _metal_mass_fraction(FeH)
    Y = 0.248 + 1.0 * Z
    X = 1.0 - Y - Z
    return 4.0 / (3 * X + 1)


_MU_SUN = _mean_molecular_weight(0.0)


def f_numax(FeH):
    """Viani et al. 2017 (ApJ 843, 11) mu-term correction for numax.

    Their full relation also includes a Gamma_1 (first adiabatic exponent)
    term, sqrt(Gamma_1/Gamma_1_sun), which the paper itself notes requires
    stellar model grids to evaluate and isn't practical from observables
    alone. That term is fixed to 1 here -- this is the mu-only
    approximation used e.g. in Zinn et al. 2019.

    Parameters
    ----------
    FeH : float or array-like
        Metallicity in dex relative to solar.

    Returns
    -------
    float or ndarray
        Dimensionless molecular-weight correction.
    """
    mu = _mean_molecular_weight(FeH)
    return xp.sqrt(mu / _MU_SUN)


def numax(M, R, Teff, FeH):
    """Predict the frequency of maximum oscillation power.

    Parameters
    ----------
    M, R : float or array-like
        Stellar mass and radius in solar units.
    Teff : float or array-like
        Effective temperature in kelvin.
    FeH : float or array-like
        Metallicity in dex relative to solar.

    Returns
    -------
    float or ndarray
        Frequency of maximum power in microhertz.
    """
    return NUMAX_SUN * M / R**2 / xp.sqrt(Teff / TEFF_SUN) * f_numax(FeH)


DNU_SUN = 135.1  # Huber et al. 2011, used to anchor the reference function below


def _dnu_ref_raw(Teff, FeH):
    """Evaluate the unnormalized Guggenberger reference function.

    Parameters
    ----------
    Teff : float or array-like
        Effective temperature in kelvin.
    FeH : float or array-like
        Metallicity in dex relative to solar.

    Returns
    -------
    float or ndarray
        Unnormalized reference large separation in microhertz.
    """
    x = Teff / 1.0e4
    A = 0.64 * FeH + 1.78
    lam = -0.55 * FeH + 1.23
    omega = 22.21
    phi = 0.48 * FeH + 0.12
    B = 0.66 * FeH + 134.92
    return A * xp.exp(lam * x) * xp.cos(omega * x + phi) + B


# Guggenberger+2016's own coefficients don't reproduce DNU_SUN exactly at
# solar Teff/[Fe/H] (their most solar-like model gave 136.1 muHz -- they
# attribute the ~1-3 muHz offset to surface effects not in their grid, see
# their Sec. 2.1). We renormalize so the reference function is anchored to
# the precisely-known solar value while preserving its Teff/[Fe/H] shape.
_DNU_REF_NORM = DNU_SUN / _dnu_ref_raw(TEFF_SUN, 0.0)


def dnu_ref(Teff, FeH):
    """Guggenberger et al. 2016 (MNRAS 460, 4277) Teff-[Fe/H] reference
    function, replacing the fixed solar Delta-nu in the classic relation.
    Calibrated for -1.0 < [Fe/H] < 0.5, 0.8-2.0 Msun, main sequence to cool
    red giants (down to numax ~ 6 muHz). Renormalized to equal DNU_SUN
    exactly at solar Teff/[Fe/H] -- see _DNU_REF_NORM above.

    Parameters
    ----------
    Teff : float or array-like
        Effective temperature in kelvin.
    FeH : float or array-like
        Metallicity in dex relative to solar.

    Returns
    -------
    float or ndarray
        Reference large separation in microhertz.
    """
    return _dnu_ref_raw(Teff, FeH) * _DNU_REF_NORM


def dnu(M, R, Teff, FeH):
    """Predict the large frequency separation.

    Parameters
    ----------
    M, R : float or array-like
        Stellar mass and radius in solar units.
    Teff : float or array-like
        Effective temperature in kelvin.
    FeH : float or array-like
        Metallicity in dex relative to solar.

    Returns
    -------
    float or ndarray
        Large frequency separation in microhertz.
    """
    return dnu_ref(Teff, FeH) * xp.sqrt(M / R**3)


def luminosity(R, Teff):
    """Compute luminosity from radius and effective temperature.

    Parameters
    ----------
    R : float or array-like
        Radius in solar units.
    Teff : float or array-like
        Effective temperature in kelvin.

    Returns
    -------
    float or ndarray
        Luminosity in solar units.
    """
    return R**2 * (Teff / TEFF_SUN) ** 4


def logg(M, R):
    """Compute base-10 surface gravity in cgs units.

    Parameters
    ----------
    M, R : float or array-like
        Stellar mass and radius in solar units.

    Returns
    -------
    float or ndarray
        Logarithmic surface gravity in dex.
    """
    return xp.log10(GSUN_CGS * M / R**2)


def mean_density(M, R):
    """Compute mean stellar density.

    Parameters
    ----------
    M, R : float or array-like
        Stellar mass and radius in solar units.

    Returns
    -------
    float or ndarray
        Mean density in solar-density units.
    """
    return M / R**3


def envelope_fwhm(numax, Teff):
    """Return the FWHM of the oscillation power envelope.

    This is equation 19 of Ball et al. (2018): the Mosser et al. (2012)
    relation with their stated temperature correction above solar
    effective temperature.

    Parameters
    ----------
    numax : float or array-like
        Frequency of maximum oscillation power in microhertz.
    Teff : float or array-like
        Effective temperature in kelvin.

    Returns
    -------
    float or ndarray
        Full width at half maximum in microhertz.
    """
    base = 0.66 * numax**0.88
    correction = xp.where(Teff > TEFF_SUN, 1.0 + 6.0e-4 * (Teff - TEFF_SUN), 1.0)
    return base * correction


A_ENV_SUN_TESS = 2.1  # ppm, maximum radial-mode rms amplitude in TESS
A_ENV_SUN_KEPLER = 2.5  # ppm, Kepler calibration before TESS response correction
_A_ENV_SUN = {"TESS": A_ENV_SUN_TESS, "KEPLER": A_ENV_SUN_KEPLER}


def normalize_bandpass(bandpass):
    """Return the canonical name of a supported photometric bandpass.

    Parameters
    ----------
    bandpass : str
        ``'TESS'`` or ``'Kepler'`` (case-insensitive).

    Returns
    -------
    str
        Canonical upper-case bandpass name.

    Raises
    ------
    ValueError
        If the bandpass is not supported.
    """
    if not isinstance(bandpass, str):
        raise ValueError("bandpass must be a string.")
    
    canonical = bandpass.strip().upper()
    
    if canonical not in _A_ENV_SUN:
        raise ValueError(
            f"Unsupported bandpass {bandpass!r}; choose 'TESS' or 'Kepler'."
        )
    return canonical


def amplitude_red_edge(L):
    """Return the adopted red-edge temperature of the instability strip.

    Parameters
    ----------
    L : float or array-like
        Luminosity in solar units.

    Returns
    -------
    float or ndarray
        Red-edge effective temperature in kelvin.
    """
    return 8907.0 * L**-0.093


def envelope_amplitude(M, L, Teff, bandpass="TESS"):
    """Return the maximum radial-mode rms amplitude in a photometric band.

    Implements equations 16--18 of Ball et al. (2018), including the
    suppression factor near the red edge of the delta-Scuti instability
    strip. The 2.1 ppm TESS zero-point is from Ball et al.; the 2.5 ppm
    Kepler zero-point underlying the TESS response correction is given by
    Campante et al. (2016).

    Parameters
    ----------
    M, L : float or array-like
        Mass and luminosity in solar units.
    Teff : float or array-like
        Effective temperature in kelvin.
    bandpass : {'TESS', 'Kepler'}, default='TESS'
        Photometric response used for the amplitude calibration. Kepler
        amplitudes are slightly larger because its response is bluer.

    Returns
    -------
    float or ndarray
        Maximum radial-mode rms amplitude in parts per million.
    """
    bandpass = normalize_bandpass(bandpass)
    red_edge = amplitude_red_edge(L)
    beta = 1.0 - xp.exp((Teff - red_edge) / 1250.0)
    beta = xp.clip(beta, 0.0, 1.0)
    return _A_ENV_SUN[bandpass] * beta * (L / M) * (Teff / 5777.0) ** -2.0


def granulation_amplitude(numax, M):
    """Return the Kallinger et al. (2014) granulation rms amplitude.

    Parameters
    ----------
    numax : float or array-like
        Frequency of maximum oscillation power in microhertz.
    M : float or array-like
        Mass in solar units.

    Returns
    -------
    float or ndarray
        RMS granulation amplitude in parts per million.
    """
    return 3710.0 * numax**-0.613 * M**-0.26


def granulation_frequency_low(numax):
    """Return the lower Kallinger characteristic frequency.

    Parameters
    ----------
    numax : float or array-like
        Frequency of maximum oscillation power in microhertz.

    Returns
    -------
    float or ndarray
        Lower granulation frequency in microhertz.
    """
    return 0.317 * numax**0.970


def granulation_frequency_high(numax):
    """Return the higher Kallinger characteristic frequency.

    Parameters
    ----------
    numax : float or array-like
        Frequency of maximum oscillation power in microhertz.

    Returns
    -------
    float or ndarray
        Higher granulation frequency in microhertz.
    """
    return 0.948 * numax**0.992


# def harvey_super_lorentzian(frequency, amplitude, characteristic_frequency):
#     """Evaluate a normalized Kallinger super-Lorentzian component.

#     Parameters
#     ----------
#     frequency : float or array-like
#         Frequencies in microhertz.
#     amplitude : float
#         RMS intensity amplitude in parts per million.
#     characteristic_frequency : float
#         Characteristic frequency in microhertz.

#     Returns
#     -------
#     float or ndarray
#         Power density in ppm squared per microhertz. Its integral from zero
#         to infinity equals ``amplitude**2``.
#     """
#     normalization = 2.0 * xp.sqrt(2.0) / xp.pi
#     ratio = frequency / characteristic_frequency
#     return normalization * amplitude**2 / characteristic_frequency / (1.0 + ratio**4)


def distance(plx):
    """Convert parallax to distance.

    Parameters
    ----------
    plx : float or array-like
        Parallax in milliarcseconds.

    Returns
    -------
    float or ndarray
        Distance in parsecs.
    """
    return 1000.0 / plx


def mbol(L):
    """Convert luminosity to absolute bolometric magnitude.

    Parameters
    ----------
    L : float or array-like
        Luminosity in solar units.

    Returns
    -------
    float or ndarray
        Absolute bolometric magnitude.
    """
    return MBOL_SUN - 2.5 * xp.log10(L)


def bc_g(Teff):
    """Rough bolometric correction for Gaia G band, linear near solar Teff.

    Placeholder only -- swap in a real BC grid (e.g. MIST/Casagrande &
    VandenBerg 2018) for anything beyond order-of-magnitude checks.

    Parameters
    ----------
    Teff : float or array-like
        Effective temperature in kelvin.

    Returns
    -------
    float or ndarray
        Approximate Gaia G bolometric correction in magnitudes.
    """
    x = (Teff - TEFF_SUN) / 1000.0
    return -0.068 - 0.008 * x


def bc_bp(Teff):
    """Rough bolometric correction for Gaia BP band, linear near solar Teff.

    Placeholder only, calibrated to give M_BP,sun ~ 5.03 (Mbol_sun - BC_BP =
    4.74 - (-0.29) = 5.03), matching approximate published Gaia solar colors
    (BP-RP_sun ~ 0.82). Swap in a real BC grid for anything more serious.

    Parameters
    ----------
    Teff : float or array-like
        Effective temperature in kelvin.

    Returns
    -------
    float or ndarray
        Approximate Gaia BP bolometric correction in magnitudes.
    """
    x = (Teff - TEFF_SUN) / 1000.0
    return -0.29 - 0.30 * x


def bc_rp(Teff):
    """Rough bolometric correction for Gaia RP band, linear near solar Teff.

    Placeholder only, calibrated to give M_RP,sun ~ 4.21 (Mbol_sun - BC_RP =
    4.74 - 0.53 = 4.21), matching approximate published Gaia solar colors.
    Swap in a real BC grid for anything more serious.

    Parameters
    ----------
    Teff : float or array-like
        Effective temperature in kelvin.

    Returns
    -------
    float or ndarray
        Approximate Gaia RP bolometric correction in magnitudes.
    """
    x = (Teff - TEFF_SUN) / 1000.0
    return 0.53 + 0.15 * x


# Gaia DR2 extinction coefficients (Danielski et al. 2018), roughly constant
# for G2V-like SEDs; A_0 here stands in for a monochromatic ~550nm
# extinction, estimated back out from A_G. Fine for back-of-envelope
# reddening; the true coefficients have some Teff/extinction dependence
# this ignores.
_A_G_OVER_A0 = 0.789
_A_BP_OVER_A0 = 1.002
_A_RP_OVER_A0 = 0.589


def a_bp(A_G):
    """Convert Gaia G extinction to approximate BP extinction.

    Parameters
    ----------
    A_G : float or array-like
        Gaia G-band extinction in magnitudes.

    Returns
    -------
    float or ndarray
        Approximate Gaia BP extinction in magnitudes.
    """
    return (_A_BP_OVER_A0 / _A_G_OVER_A0) * A_G


def a_rp(A_G):
    """Convert Gaia G extinction to approximate RP extinction.

    Parameters
    ----------
    A_G : float or array-like
        Gaia G-band extinction in magnitudes.

    Returns
    -------
    float or ndarray
        Approximate Gaia RP extinction in magnitudes.
    """
    return (_A_RP_OVER_A0 / _A_G_OVER_A0) * A_G


def bp_rp_color(BP_mag, RP_mag):
    """Gaia BP-RP color. Note this comes out independent of distance --
    the distance modulus cancels between BP_mag and RP_mag -- so it
    constrains Teff/extinction without needing a parallax at all.

    Parameters
    ----------
    BP_mag, RP_mag : float or array-like
        Apparent or absolute Gaia BP and RP magnitudes.

    Returns
    -------
    float or ndarray
        Gaia BP-RP color in magnitudes.
    """
    return BP_mag - RP_mag


def abs_g_mag(Mbol, BC_G):
    """Compute absolute magnitude from bolometric magnitude and correction.

    Parameters
    ----------
    Mbol : float or array-like
        Absolute bolometric magnitude.
    BC_G : float or array-like
        Bolometric correction in magnitudes.

    Returns
    -------
    float or ndarray
        Absolute magnitude in the selected band.
    """
    return Mbol - BC_G


def app_g_mag(M_G, d, A_G):
    """Compute apparent magnitude using the distance modulus.

    Parameters
    ----------
    M_G : float or array-like
        Absolute magnitude.
    d : float or array-like
        Distance in parsecs.
    A_G : float or array-like
        Extinction in magnitudes.

    Returns
    -------
    float or ndarray
        Apparent magnitude.
    """
    return M_G + 5 * xp.log10(d) - 5 + A_G


# name -> (function, argument names). Order matters: args must already
# be available (fundamental params or earlier derived quantities computed
# earlier in this dict).
DERIVED = {
    "numax": (numax, ("M", "R", "Teff", "FeH")),
    "dnu": (dnu, ("M", "R", "Teff", "FeH")),
    "L": (luminosity, ("R", "Teff")),
    "logg": (logg, ("M", "R")),
    "rho": (mean_density, ("M", "R")),
    "FWHM_env": (envelope_fwhm, ("numax", "Teff")),
    "A_env": (envelope_amplitude, ("M", "L", "Teff")),
    "A_gran": (granulation_amplitude, ("numax", "M")),
    "b_gran_low": (granulation_frequency_low, ("numax",)),
    "b_gran_high": (granulation_frequency_high, ("numax",)),
    "d": (distance, ("plx",)),
    "Mbol": (mbol, ("L",)),
    "BC_G": (bc_g, ("Teff",)),
    "BC_BP": (bc_bp, ("Teff",)),
    "BC_RP": (bc_rp, ("Teff",)),
    "A_BP": (a_bp, ("A_G",)),
    "A_RP": (a_rp, ("A_G",)),
    "M_G": (abs_g_mag, ("Mbol", "BC_G")),
    "M_BP": (abs_g_mag, ("Mbol", "BC_BP")),
    "M_RP": (abs_g_mag, ("Mbol", "BC_RP")),
    "G_mag": (app_g_mag, ("M_G", "d", "A_G")),
    "BP_mag": (app_g_mag, ("M_BP", "d", "A_BP")),
    "RP_mag": (app_g_mag, ("M_RP", "d", "A_RP")),
    "BP_RP": (bp_rp_color, ("BP_mag", "RP_mag")),
}
