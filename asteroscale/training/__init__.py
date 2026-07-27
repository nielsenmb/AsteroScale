"""Offline tools for building portable stellar-population priors.

This subpackage is deliberately separate from AsteroScale's inference path.
Importing or using AsteroScale does not train or load a population model.
"""

from .catalogue import (
    PopulationCatalogue,
    read_standard_catalogue,
    read_trilegal,
    write_standard_catalogue,
)
from .gmm import PopulationGMM, fit_candidate_models, fit_weighted_gmm

__all__ = [
    "PopulationCatalogue",
    "PopulationGMM",
    "fit_candidate_models",
    "fit_weighted_gmm",
    "read_standard_catalogue",
    "read_trilegal",
    "write_standard_catalogue",
]
