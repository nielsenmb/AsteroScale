"""Generic any-subset-in -> any-subset-out solver via nested sampling (dynesty)."""
import numpy as np
import dynesty
from dynesty.utils import resample_equal

from .relations import DERIVED, FUNDAMENTAL

DEFAULT_PRIORS = {
    "M": (0.5, 3.0),      # solar masses, uniform bounds
    "R": (0.5, 20.0),     # solar radii
    "Teff": (4000.0, 7000.0),  # K
    "plx": (0.01, 50.0),  # mas -- widen/narrow based on expected distance
    "A_G": (0.0, 1.0),    # mag -- tighten if you have a dust-map estimate
    "FeH": (-2.5, 0.5),   # dex -- Guggenberger 2016 calibrated over -1.0 to 0.5
}


class Solver:
    def __init__(self, priors=None, nlive=500, seed=None):
        self.priors = {**DEFAULT_PRIORS, **(priors or {})}
        self.nlive = nlive
        self.rng = np.random.default_rng(seed)

    def _prior_transform(self, u):
        """Map unit cube -> fundamental params, uniform bounds from self.priors."""
        theta = np.empty_like(u)
        for i, p in enumerate(FUNDAMENTAL):
            lo, hi = self.priors[p]
            theta[i] = lo + u[i] * (hi - lo)
        return theta

    def _forward(self, fundamentals):
        """fundamentals: dict {name: value or array}. Returns dict incl. derived."""
        out = dict(fundamentals)
        for name, (func, args) in DERIVED.items():
            out[name] = func(*[out[a] for a in args])
        return out

    def _loglike(self, theta, given):
        fund = dict(zip(FUNDAMENTAL, theta))
        full = self._forward(fund)
        logl = 0.0
        for name, (val, err) in given.items():
            if name not in full:
                raise KeyError(f"Unknown quantity '{name}'")
            logl += -0.5 * ((full[name] - val) / err) ** 2
        return logl

    def solve(self, given, want, dlogz=0.5, print_progress=False, return_results=False):
        """
        given: dict {name: (value, error)}, e.g. {"Teff": (5777, 50)}
        want:  list of quantity names to return equal-weighted posterior samples for
        """
        ndim = len(FUNDAMENTAL)
        loglike = lambda theta: self._loglike(theta, given)

        sampler = dynesty.NestedSampler(
            loglike, self._prior_transform, ndim, nlive=self.nlive, rstate=self.rng
        )
        sampler.run_nested(dlogz=dlogz, print_progress=print_progress)
        results = sampler.results

        weights = np.exp(results.logwt - results.logz[-1])
        eq_samples = resample_equal(results.samples, weights)

        fund = {p: eq_samples[:, i] for i, p in enumerate(FUNDAMENTAL)}
        full = self._forward(fund)

        out = {name: full[name] for name in want}
        if return_results:
            out["_results"] = results
        return out
