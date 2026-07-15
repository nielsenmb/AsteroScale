from .solver import INPUT_MODES, Solver
from . import relations
from .utils import summarize
from .viz import plot_posterior
from .batch import solve_many
from .sampling import PRESETS, SamplerSettings

_default_solver = None


def solve(
    given, want, nlive=None, priors=None, seed=None, preset=None,
    input_mode=None, **solve_kwargs,
):
    """One-off convenience wrapper: solve(given, want) without instantiating
    Solver yourself. Reuses a shared default Solver unless nlive/priors/seed
    or ``input_mode`` are given, in which case a fresh Solver is created with
    those settings. Remaining kwargs pass through to :meth:`Solver.solve`.
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
]
