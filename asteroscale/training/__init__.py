"""Tools for building and representing portable stellar-population priors.

Fitting remains an explicitly offline operation. The lightweight
``PopulationGMM`` container and transform are also reused when a saved model
is selected for inference.
"""

from .catalogue import (
    PopulationCatalogue,
    read_standard_catalogue,
    read_trilegal,
    write_standard_catalogue,
)
from .gmm import (
    PopulationGMM,
    PopulationPriorTransform,
    fit_candidate_models,
    fit_weighted_gmm,
)

__all__ = [
    "PopulationCatalogue",
    "PopulationGMM",
    "PopulationPriorTransform",
    "fit_candidate_models",
    "fit_weighted_gmm",
    "read_standard_catalogue",
    "read_trilegal",
    "write_standard_catalogue",
]
