from .solver import INPUT_MODES, Solver
from . import relations
from .utils import summarize
from .viz import plot_posterior
from .batch import solve_many
from .sampling import PRESETS, SamplerSettings
from .calibration import DEFAULT_RELATION_SCATTER

_default_solver = None


def solve(
    given, want, nlive=None, priors=None, seed=None, preset=None,
    input_mode=None, **solve_kwargs,
):
    """Solve a single stellar inference problem.

    Parameters
    ----------
    given : dict
        Exact values or uncertain constraints supplied to the solver.
    want : str or sequence of str
        Quantities to return, or ``"all"``.
    nlive : int, optional
        Number of Dynesty live points.
    priors : dict, optional
        Fundamental-parameter priors overriding the package defaults.
    seed : int, optional
        Seed for the random-number generator.
    preset : {'fast', 'standard', 'precise'}, optional
        Named sampling configuration.
    input_mode : {'propagate', 'likelihood'}, optional
        Statistical interpretation of uncertain fundamental inputs.
    **solve_kwargs
        Additional arguments passed to :meth:`Solver.solve`.

    Returns
    -------
    dict
        Requested point estimates or posterior samples.
    """
    global _default_solver
    if any(value is not None for value in
           (nlive, priors, seed, preset, input_mode)):
        kwargs = {}
        if nlive is not None:
            kwargs["nlive"] = nlive
        if priors is not None:
            kwargs["priors"] = priors
        if seed is not None:
            kwargs["seed"] = seed
        if preset is not None:
            kwargs["preset"] = preset
        if input_mode is not None:
            kwargs["input_mode"] = input_mode
        return Solver(**kwargs).solve(given, want, **solve_kwargs)
    if _default_solver is None:
        _default_solver = Solver()
    return _default_solver.solve(given, want, **solve_kwargs)


__all__ = [
    "Solver", "relations", "solve", "solve_many", "summarize",
    "plot_posterior", "PRESETS", "SamplerSettings", "INPUT_MODES",
    "DEFAULT_RELATION_SCATTER",
]
