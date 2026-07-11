from .solver import Solver
from . import relations
from .utils import summarize
from .viz import plot_posterior

_default_solver = None


def solve(given, want, nlive=None, priors=None, seed=None, **solve_kwargs):
    """One-off convenience wrapper: solve(given, want) without instantiating
    Solver yourself. Reuses a shared default Solver unless nlive/priors/seed
    are given, in which case a fresh Solver is created with those settings.
    Remaining kwargs (dlogz, print_progress, return_results) pass through
    to Solver.solve().
    """
    global _default_solver
    if nlive is not None or priors is not None or seed is not None:
        kwargs = {}
        if nlive is not None:
            kwargs["nlive"] = nlive
        if priors is not None:
            kwargs["priors"] = priors
        if seed is not None:
            kwargs["seed"] = seed
        return Solver(**kwargs).solve(given, want, **solve_kwargs)
    if _default_solver is None:
        _default_solver = Solver()
    return _default_solver.solve(given, want, **solve_kwargs)


__all__ = ["Solver", "relations", "solve", "summarize", "plot_posterior"]
