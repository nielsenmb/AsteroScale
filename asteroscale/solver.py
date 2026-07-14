"""Generic any-subset-in -> any-subset-out solver via nested sampling (dynesty).

Plain numpy by default -- dynesty calls the likelihood one point at a time,
so jax.jit doesn't pay off here (per-call dispatch overhead dominates for a
problem this small/serial; see relations.py for how to switch the forward
pass to JAX if you later move to a vectorized sampler like jaxns).
"""
import numpy as np
from scipy import optimize
import dynesty
from dynesty.utils import resample_equal

from .relations import DERIVED, FUNDAMENTAL
from .forward import evaluate_relations
from .sampling import get_sampler_settings
from .priors import ParallaxPrior 
 
from .distributions import normal, uniform, TruncatedPowerLaw, Exponential, TruncatedNormal
from . import validation

# Any object with a .ppf(u) method works here: frozen scipy.stats
# distributions, the classes in priors.py, or your own custom class (e.g.
# the JAX-jittable ones in distributions.py, if you go that route later).
# These defaults aim for "plausible for a random field star", not
# "uninformative" -- a flat prior over mass, say, implies high-mass stars
# are as common as low-mass ones, which is badly wrong.
#
# priors.py's classes are used here rather than scipy.stats, not because
# scipy.stats can't do this, but because its frozen-distribution objects
# are much slower for the scalar, one-value-at-a-time .ppf/.logpdf calls
# this solver actually makes -- see priors.py's module docstring.
DEFAULT_PRIORS = {
    "M": TruncatedPowerLaw(alpha=2.35, low=0.5, high=3.0),  # Salpeter IMF slope
    "R": TruncatedPowerLaw(alpha=1.0, low=0.5, high=20.0),  # log-uniform: R spans >1 decade
    "Teff": uniform(loc=4000.0, scale=3000.0),   # flat 4000-7000 K, no strong prior
    "plx": ParallaxPrior(length_scale_pc=1350.0),  # Bailer-Jones distance prior
    "A_G": Exponential(scale=0.2),           # most stars nearby have low extinction
    "FeH": TruncatedNormal(                  # solar-neighborhood metallicity spread,
        loc=-0.1, scale=0.25, low=-1.0, high=0.5,  # truncated to the range the dnu
    ),                                        # metallicity correction is calibrated over
}


def _as_distribution(p):
    """Allow priors={'Teff': (mean, error)} as shorthand for a Gaussian --
    the same convention (mean, error) tuples have in `given`, for
    consistency. For a uniform prior specifically, pass
    priors.uniform(loc, scale) (or scipy.stats.uniform, or any other
    distribution) directly."""
    if isinstance(p, tuple):
        mean, err = p
        return normal(loc=mean, scale=err)
    return p


def _parse_given(given):
    """Split given into exact scalar values ('fixed') and probabilistic
    constraints ('constraints': objects with .logpdf and/or .ppf).

    Each given[name] can be:
      - a plain number -- treated as exactly known.
      - a (mean, error) tuple -- wrapped as a Gaussian (priors.normal).
      - any object with .logpdf and/or .ppf -- used directly, e.g. for
        asymmetric or otherwise non-Gaussian uncertainties.
    """
    fixed, constraints = {}, {}
    for name, value in given.items():
        if isinstance(value, tuple):
            mean, err = value
            constraints[name] = normal(loc=mean, scale=err)
        elif hasattr(value, "logpdf") or hasattr(value, "ppf"):
            constraints[name] = value
        else:
            fixed[name] = value
    return fixed, constraints


class Solver:
    """Infer stellar properties from a flexible set of constraints.

    Parameters
    ----------
    priors : dict, optional
        Distributions replacing entries in :data:`DEFAULT_PRIORS`.
    preset : {'standard', 'fast', 'precise'}, default='standard'
        Named Dynesty accuracy/runtime configuration.
    nlive : int, optional
        Override the preset's number of live points.
    seed : int, optional
        Seed for the NumPy random generator.
    sample, bound : str, optional
        Dynesty sampling and bounding methods.
    bootstrap, walks, update_interval : int, optional
        Additional Dynesty settings. The standard defaults are zero, five,
        and ``10 * nlive``, respectively.
    """

    def __init__(
        self,
        priors=None,
        preset="standard",
        nlive=None,
        seed=None,
        sample=None,
        bound=None,
        bootstrap=None,
        walks=None,
        update_interval=None,
    ):
        priors = {**DEFAULT_PRIORS, **(priors or {})}
        self.priors = {k: _as_distribution(v) for k, v in priors.items()}
        self.settings = get_sampler_settings(
            preset,
            nlive=nlive,
            sample=sample,
            bound=bound,
            bootstrap=bootstrap,
            walks=walks,
            update_interval=update_interval,
        )
        self.nlive = self.settings.nlive
        self.sample = self.settings.sample
        self.bound = self.settings.bound
        self.rng = np.random.default_rng(seed)
        self._last_fund = None  # fundamentals from the last solve() call,
        # used by predict() to derive further quantities without resampling

    def _forward(self, fundamentals):
        """Evaluate all relations for scalar or array fundamentals."""
        return evaluate_relations(fundamentals)

    def _bounds_for(self, name):
        """Best-effort bounds for a fundamental, from its prior's support --
        used to keep the point-estimate least-squares solve out of
        unphysical territory (e.g. negative mass) mid-iteration."""
        prior = self.priors[name]
        if hasattr(prior, "support"):
            try:
                return prior.support()
            except Exception:
                pass
        if hasattr(prior, "low") and hasattr(prior, "high"):
            return (prior.low, prior.high)
        return (-np.inf, np.inf)

    def _point_estimate(self, fixed, want):
        """Fast path: every given value was a plain scalar, so there's
        nothing to marginalize over. If the fixed values cover every
        fundamental, this is just a direct forward evaluation. Otherwise,
        least-squares solves for the remaining fundamentals against any
        given derived-quantity targets (e.g. given exact Teff, numax, dnu
        -> solve for M, R). No sampler involved either way.
        """
        fixed_fund = {k: v for k, v in fixed.items() if k in FUNDAMENTAL}
        derived_targets = {k: v for k, v in fixed.items() if k in DERIVED}
        free = [p for p in FUNDAMENTAL if p not in fixed_fund]

        if not free:
            full = self._forward(fixed_fund)
            self._last_fund = fixed_fund
            return {name: full[name] for name in want}

        if not derived_targets:
            raise ValueError(
                f"Not enough information for a point estimate: {free} are "
                "unconstrained and no derived-quantity targets were given. "
                "Either fix all of M/R/Teff/plx/A_G/FeH, or give at least "
                "as many derived-quantity targets as free parameters."
            )

        x0 = np.array([self.priors[p].ppf(0.5) for p in free])
        bounds = [self._bounds_for(p) for p in free]
        lo, hi = zip(*bounds)

        def residuals(x):
            theta = dict(zip(free, x))
            theta.update(fixed_fund)
            full = self._forward(theta)
            return [full[name] - target for name, target in derived_targets.items()]

        result = optimize.least_squares(residuals, x0, bounds=(lo, hi))
        validation.check_point_estimate_residuals(result, derived_targets)
        theta = dict(zip(free, result.x))
        theta.update(fixed_fund)
        full = self._forward(theta)
        self._last_fund = theta
        return {name: full[name] for name in want}

    def solve(self, given, want, dlogz=None, print_progress=False, return_results=False):
        """Infer requested quantities from exact or uncertain constraints.

        Parameters
        ----------
        given : dict
            Values may be exact numbers, ``(mean, error)`` Gaussian pairs,
            or distribution-like objects with ``logpdf`` or ``ppf``.
        want : list of str or {'all'}
            Quantities to return. ``'all'`` returns every available
            fundamental and derived quantity.
        dlogz : float, optional
            Override the preset's evidence stopping tolerance.
        print_progress : bool, default=False
            Display Dynesty progress.
        return_results : bool, default=False
            Include raw Dynesty results under ``'_results'``.

        Returns
        -------
        dict
            Requested point estimates or posterior arrays.
        """
        validation.validate_given(given)
        want = validation.normalize_want(want)
        dlogz = self.settings.dlogz if dlogz is None else dlogz

        fixed, constraints = _parse_given(given)

        if not constraints:
            return self._point_estimate(fixed, want)

        # Fundamentals given a distribution use it as their prior directly
        # (replacing the default), rather than sampling broadly and
        # penalizing via the likelihood -- equivalent when it's the only
        # constraint on that parameter, and more efficient. Everything else
        # (derived-quantity constraints, and any *scalar* given for a
        # derived quantity -- approximated here as a tight Gaussian, since
        # there's no sampled dimension to pin exactly) goes into the
        # likelihood instead.
        priors = dict(self.priors)
        likelihood_terms = {}
        for name, dist_obj in constraints.items():
            if name in FUNDAMENTAL and hasattr(dist_obj, "ppf"):
                priors[name] = dist_obj
            else:
                likelihood_terms[name] = dist_obj

        fixed_fund = {k: v for k, v in fixed.items() if k in FUNDAMENTAL}
        for name, target in fixed.items():
            if name in FUNDAMENTAL:
                continue
            eps = max(abs(target) * 1e-3, 1e-6)
            likelihood_terms[name] = normal(loc=target, scale=eps)

        free_fundamentals = [p for p in FUNDAMENTAL if p not in fixed_fund]

        if not free_fundamentals:
            # Every fundamental was pinned exactly, even though some other
            # given value was probabilistic (e.g. a redundant/consistency
            # check) -- nothing left to sample.
            full = self._forward(fixed_fund)
            self._last_fund = fixed_fund
            return {name: full[name] for name in want}

        if not likelihood_terms:
            # Every given constraint landed on a fundamental and became a
            # prior directly (see above), leaving nothing for the
            # likelihood -- it's flat everywhere. That's not actually a
            # sampling problem, it's a prior-predictive draw: sample the
            # (possibly-overridden) priors directly instead of handing
            # dynesty a constant log-likelihood, which it can technically
            # handle but warns about ("likelihood plateau") and gains
            # nothing from.
            n = max(self.nlive, 2000)
            u = self.rng.uniform(size=(n, len(free_fundamentals)))
            fund = {p: priors[p].ppf(u[:, j]) for j, p in enumerate(free_fundamentals)}
            for k, v in fixed_fund.items():
                fund[k] = np.full(n, v)
            full = self._forward(fund)
            self._last_fund = fund
            out = {name: full[name] for name in want}
            if return_results:
                out["_results"] = None
            return out

        ndim = len(free_fundamentals)

        def prior_transform(u):
            return np.array(
                [priors[p].ppf(u[i]) for i, p in enumerate(free_fundamentals)]
            )

        def loglike(theta):
            fund = dict(zip(free_fundamentals, theta))
            fund.update(fixed_fund)
            full = self._forward(fund)
            return float(sum(dist_obj.logpdf(full[name])
                              for name, dist_obj in likelihood_terms.items()))

        sampler = dynesty.NestedSampler(
            loglike, prior_transform, ndim, nlive=self.nlive, rstate=self.rng,
            sample=self.sample, bound=self.bound,
            bootstrap=self.settings.bootstrap, walks=self.settings.walks,
            update_interval=self.settings.update_interval,
        )
        sampler.run_nested(dlogz=dlogz, print_progress=print_progress, save_bounds=False)
        results = sampler.results

        weights = np.exp(results.logwt - results.logz[-1])
        eq_samples = resample_equal(results.samples, weights)

        fund = {p: eq_samples[:, i] for i, p in enumerate(free_fundamentals)}
        for k, v in fixed_fund.items():
            fund[k] = np.full(eq_samples.shape[0], v)
        full = self._forward(fund)
        self._last_fund = fund

        out = {name: full[name] for name in want}
        if return_results:
            out["_results"] = results
        return out

    def predict(self, want):
        """Compute additional quantities from the posterior/point estimate
        of the last solve() call, without re-running the sampler -- e.g.

            solver.solve({"Teff": (5777, 50), "numax": (3090, 30),
                          "dnu": (135.1, 1.0)}, want=["M", "R"])
            extra = solver.predict(["L", "rho", "logg", "A_env"])

        Works for both the nested-sampling path (returns arrays, same
        length as the posterior from the last solve() call) and the
        point-estimate path (returns plain floats).
        """
        if self._last_fund is None:
            raise RuntimeError(
                "predict() needs a previous solve() call to derive "
                "quantities from -- call solve() first."
            )
        want = validation.normalize_want(want)
        full = self._forward(self._last_fund)
        return {name: full[name] for name in want}
