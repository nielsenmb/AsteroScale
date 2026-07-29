"""Runtime loading and coordinate handling for population GMM priors."""

from __future__ import annotations

from importlib import resources
from os import PathLike
from pathlib import Path

import numpy as np

from .training.gmm import PopulationGMM


POPULATION_FUNDAMENTALS = ("M", "R", "Teff", "FeH")
POPULATION_COORDINATES = {
    "M": "log10_mass",
    "R": "log10_radius",
    "Teff": "log10_teff",
    "FeH": "feh",
}
BUILTIN_POPULATION_PRIORS = {
    "trilegal_solar_neighbourhood":
        "trilegal_solar_neighbourhood_gmm.npz",
}


def load_population_prior(population_prior):
    """Load a configured population GMM.

    Parameters
    ----------
    population_prior : str, path-like, PopulationGMM or None
        A built-in model name, path to a saved model, already loaded model,
        or ``None`` to disable the correlated population prior.

    Returns
    -------
    PopulationGMM or None
        Loaded model, or ``None`` when disabled.

    Raises
    ------
    TypeError
        If the specification has an unsupported type.
    ValueError
        If a string is neither a built-in name nor an existing model path.
    """
    if population_prior is None:
        return None
    if isinstance(population_prior, PopulationGMM):
        return population_prior
    if not isinstance(population_prior, (str, PathLike)):
        raise TypeError(
            "population_prior must be a built-in name, path, "
            "PopulationGMM, or None."
        )

    if isinstance(population_prior, str) and (
        population_prior in BUILTIN_POPULATION_PRIORS
    ):
        filename = BUILTIN_POPULATION_PRIORS[population_prior]
        resource = (
            resources.files("asteroscale")
            .joinpath("data")
            .joinpath("population_models")
            .joinpath(filename)
        )
        with resources.as_file(resource) as model_path:
            return PopulationGMM.load(model_path)

    model_path = Path(population_prior).expanduser()
    if not model_path.is_file():
        available = ", ".join(sorted(BUILTIN_POPULATION_PRIORS))
        raise ValueError(
            f"Unknown population prior {str(population_prior)!r}. "
            f"Choose a built-in model ({available}) or an existing NPZ path."
        )
    return PopulationGMM.load(model_path)


def to_population_coordinate(name, value):
    """Convert a public fundamental value to its GMM training coordinate.

    Parameters
    ----------
    name : {'M', 'R', 'Teff', 'FeH'}
        Fundamental-parameter name.
    value : float
        Value in AsteroScale's public physical units.

    Returns
    -------
    float
        Base-10 logarithm for mass, radius, and temperature, or unchanged
        metallicity.

    Raises
    ------
    ValueError
        If a logarithmic coordinate is not strictly positive.
    """
    value = float(value)
    if name == "FeH":
        return value
    if name not in POPULATION_COORDINATES:
        raise ValueError(f"{name!r} is not represented by the population GMM.")
    if value <= 0.0:
        raise ValueError(
            f"Exact {name!r} must be positive to condition the population GMM."
        )
    return float(np.log10(value))


def population_to_sampler_coordinate(name, value):
    """Convert a GMM coordinate into AsteroScale's sampler coordinate.

    Parameters
    ----------
    name : {'M', 'R', 'Teff', 'FeH'}
        Fundamental-parameter name.
    value : float
        Value in the model's training coordinate.

    Returns
    -------
    float
        Solver coordinate: logarithmic mass/radius, temperature in kelvin,
        or metallicity in dex.
    """
    if name in ("M", "R", "FeH"):
        return float(value)
    if name == "Teff":
        return float(np.power(10.0, value))
    raise ValueError(f"{name!r} is not represented by the population GMM.")
