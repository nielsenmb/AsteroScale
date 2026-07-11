"""Scaling relations linking fundamental stellar params to observables.

All masses/radii in solar units, Teff in K, numax/dnu in muHz.
"""
import numpy as np

NUMAX_SUN = 3090.0
TEFF_SUN = 5777.0
GSUN_CGS = 27400.0
MBOL_SUN = 4.74  # IAU absolute bolometric magnitude of the Sun

FUNDAMENTAL = ("M", "R", "Teff", "plx", "A_G", "FeH")


def _metal_mass_fraction(FeH):
    """Z from [Fe/H], assuming solar-scaled composition ([alpha/Fe]=0).
    Calibration from Bertelli et al. 1994 / Salaris et al. 1993, as used in
    e.g. Zinn et al. 2019 to estimate mean molecular weight from [Fe/H].
    """
    return 10 ** (0.977 * FeH - 1.699)


def _mean_molecular_weight(FeH):
    """Fully-ionized H+He mean molecular weight, mu = 4/(3X+1).

    Assumes primordial Y_p=0.248 and helium enrichment dY/dZ=1.0. This
    ignores metals' contribution to free electrons -- fine near solar
    metallicity, increasingly approximate for very metal-poor stars.
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
    """
    mu = _mean_molecular_weight(FeH)
    return np.sqrt(mu / _MU_SUN)


def numax(M, R, Teff, FeH):
    return NUMAX_SUN * M / R**2 / np.sqrt(Teff / TEFF_SUN) * f_numax(FeH)


def dnu_ref(Teff, FeH):
    """Guggenberger et al. 2016 (MNRAS 460, 4277) Teff-[Fe/H] reference
    function, replacing the fixed solar Delta-nu in the classic relation.
    Calibrated for -1.0 < [Fe/H] < 0.5, 0.8-2.0 Msun, main sequence to cool
    red giants (down to numax ~ 6 muHz).
    """
    x = Teff / 1.0e4
    A = 0.64 * FeH + 1.78
    lam = -0.55 * FeH + 1.23
    omega = 22.12
    phi = 0.48 * FeH + 0.12
    B = 0.66 * FeH + 134.92
    return A * np.exp(lam * x) * np.cos(omega * x + phi) + B


def dnu(M, R, Teff, FeH):
    return dnu_ref(Teff, FeH) * np.sqrt(M / R**3)


def luminosity(R, Teff):
    return R**2 * (Teff / TEFF_SUN) ** 4


def logg(M, R):
    return np.log10(GSUN_CGS * M / R**2)


def mean_density(M, R):
    """Mean density in solar units (rho/rho_sun = (M/Msun)/(R/Rsun)^3)."""
    return M / R**3


def distance(plx):
    """Distance in pc from parallax in mas."""
    return 1000.0 / plx


def mbol(L):
    """Absolute bolometric magnitude from luminosity (solar units)."""
    return MBOL_SUN - 2.5 * np.log10(L)


def bc_g(Teff):
    """Rough bolometric correction for Gaia G band, linear near solar Teff.

    Placeholder only -- swap in a real BC grid (e.g. MIST/Casagrande &
    VandenBerg 2018) for anything beyond order-of-magnitude checks.
    """
    x = (Teff - TEFF_SUN) / 1000.0
    return -0.068 - 0.008 * x


def abs_g_mag(Mbol, BC_G):
    return Mbol - BC_G


def app_g_mag(M_G, d, A_G):
    """Apparent Gaia G magnitude given absolute mag, distance (pc), extinction."""
    return M_G + 5 * np.log10(d) - 5 + A_G


# name -> (function, argument names). Order matters: args must already
# be available (fundamental params or earlier derived quantities computed
# earlier in this dict).
DERIVED = {
    "numax": (numax, ("M", "R", "Teff", "FeH")),
    "dnu": (dnu, ("M", "R", "Teff", "FeH")),
    "L": (luminosity, ("R", "Teff")),
    "logg": (logg, ("M", "R")),
    "rho": (mean_density, ("M", "R")),
    "d": (distance, ("plx",)),
    "Mbol": (mbol, ("L",)),
    "BC_G": (bc_g, ("Teff",)),
    "M_G": (abs_g_mag, ("Mbol", "BC_G")),
    "G_mag": (app_g_mag, ("M_G", "d", "A_G")),
}
