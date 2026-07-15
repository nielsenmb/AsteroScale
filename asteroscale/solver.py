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

from .relations import DERIVED, FUNDAMENTAL, normalize_bandpass
from .forward import evaluate_relations, required_fundamentals
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

INPUT_MODES = ("propagate", "likelihood")


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


def _normalize_input_mode(input_mode):
    """Validate and normalize the interpretation of uncertain inputs."""
    if not isinstance(input_mode, str):
        raise ValueError(
            f"input_mode must be one of {INPUT_MODES}, got {input_mode!r}."
        )
    normalized = input_mode.lower()
    if normalized not in INPUT_MODES:
        raise ValueError(
            f"Unknown input_mode {input_mode!r}; choose from {INPUT_MODES}."
        )
    return normalized


def _partition_constraints(base_priors, constraints, input_mode):
    """Return effective priors and likelihood terms for an input mode.

    In ``propagate`` mode, uncertain fundamental parameters replace their
    corresponding population priors. In ``likelihood`` mode, every uncertain
    input is treated as a measurement distribution and the population priors
    remain in force.
    """
    priors = dict(base_priors)
    likelihood_terms = {}
    for name, dist_obj in constraints.items():
        if (
            input_mode == "propagate"
            and name in FUNDAMENTAL
            and hasattr(dist_obj, "ppf")
        ):
            priors[name] = dist_obj
        else:
            if not hasattr(dist_obj, "logpdf"):
                raise TypeError(
                    f"The uncertain input given[{name!r}] needs a logpdf() "
                    f"method when used as a likelihood term in "
                    f"input_mode={input_mode!r}."
                )
            likelihood_terms[name] = dist_obj
    return priors, likelihood_terms


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
    bandpass : {'TESS', 'Kepler'}, default='TESS'
        Photometric response used for ``A_env``. May be overridden by an
        individual :meth:`solve` call.
    input_mode : {'propagate', 'likelihood'}, default='propagate'
        Interpretation of uncertain fundamental inputs. ``'propagate'``
        treats them as the current distributions to propagate, replacing the
        corresponding population priors. ``'likelihood'`` treats them as
        measurement likelihoods and retains the population priors. This is
        independent of the Dynesty accuracy ``preset``.
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
        bandpass="TESS",
        input_mode="propagate",
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
        self.bandpass = normalize_bandpass(bandpass)
        self.input_mode = _normalize_input_mode(input_mode)
        self.rng = np.random.default_rng(seed)
        self._last_fund = None  # fundamentals from the last solve() call,
        self._last_bandpass = None
        # used by predict() to derive further quantities without resampling

    def _forward(self, fundamentals, bandpass=None):
        """Evaluate all relations for scalar or array fundamentals."""
        bandpass = self.bandpass if bandpass is None else normalize_bandpass(bandpass)
        return evaluate_relations(fundamentals, bandpass=bandpass)

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

    def _point_estimate(self, fixed, want, bandpass):
        """Fast path: every given value was a plain scalar, so there's
        nothing to marginalize over. If the fixed values cover every
        fundamental, this is just a direct forward evaluation. Otherwise,
        least-squares solves for the remaining fundamentals against any
        given derived-quantity targets (e.g. given exact Teff, numax, dnu
        -> solve for M, R). No sampler involved either way.
        """
        fixed_fund = {k: v for k, v in fixed.items() if k in FUNDAMENTAL}
        derived_targets = {k: v for k, v in fixed.items() if k in DERIVED}
        needed = required_fundamentals(list(want) + list(derived_targets))
        free = [p for p in needed if p not in fixed_fund]

        if not free:
            full = self._forward(fixed_fund, bandpass=bandpass)
            self._last_fund = fixed_fund
            self._last_bandpass = bandpass
            return {name: full[name] for name in want}

        if not derived_targets:
            raise ValueError(
                f"Not enough information for a point estimate: {free} are "
                "unconstrained and no derived-quantity targets were given. "
                "Provide the missing quantities directly, or give enough "
                "derived-quantity constraints to solve for them."
            )

        x0 = np.array([self.priors[p].ppf(0.5) for p in free])
        bounds = [self._bounds_for(p) for p in free]
        lo, hi = zip(*bounds)

        def residuals(x):
            theta = dict(zip(free, x))
            theta.update(fixed_fund)
            full = self._forward(theta, bandpass=bandpass)
            return [full[name] - target for name, target in derived_targets.items()]

        result = optimize.least_squares(residuals, x0, bounds=(lo, hi))
        validation.check_point_estimate_residuals(result, derived_targets)
        theta = dict(zip(free, result.x))
        theta.update(fixed_fund)
        full = self._forward(theta, bandpass=bandpass)
        self._last_fund = theta
        self._last_bandpass = bandpass
        return {name: full[name] for name in want}

    def solve(
        self, given, want, dlogz=None, print_progress=False,
        return_results=False, bandpass=None, input_mode=None,
    ):
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
        bandpass : {'TESS', 'Kepler'}, optional
            Photometric response used for ``A_env``. Overrides the value
            supplied to :class:`Solver` for this call.
        input_mode : {'propagate', 'likelihood'}, optional
            Override how uncertain fundamental inputs are interpreted. See
            :class:`Solver`. Exact scalar inputs are fixed in either mode.

        Returns
        -------
        dict
            Requested point estimates or posterior arrays.
        """
        validation.validate_given(given)
        want = validation.normalize_want(want)
        dlogz = self.settings.dlogz if dlogz is None else dlogz
        bandpass = self.bandpass if bandpass is None else normalize_bandpass(bandpass)
        input_mode = self.input_mode if input_mode is None else _normalize_input_mode(input_mode)

        fixed, constraints = _parse_given(given)

        if not constraints:
            return self._point_estimate(fixed, want, bandpass)

        # ``propagate`` is the calculator-style, backwards-compatible path;
        # ``likelihood`` performs Bayesian conditioning on the configured
        # population priors. Derived constraints are likelihood terms in
        # both modes.
        priors, likelihood_terms = _partition_constraints(
            self.priors, constraints, input_mode
        )

        fixed_fund = {k: v for k, v in fixed.items() if k in FUNDAMENTAL}
        for name, target in fixed.items():
            if name in FUNDAMENTAL:
                continue
            eps = max(abs(target) * 1e-3, 1e-6)
            likelihood_terms[name] = normal(loc=target, scale=eps)

        active_names = list(want) + list(given)
        needed = required_fundamentals(active_names)
        free_fundamentals = [p for p in needed if p not in fixed_fund]

        if not free_fundamentals:
            # Every fundamental was pinned exactly, even though some other
            # given value was probabilistic (e.g. a redundant/consistency
            # check) -- nothing left to sample.
            full = self._forward(fixed_fund, bandpass=bandpass)
            self._last_fund = fixed_fund
            self._last_bandpass = bandpass
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
            full = self._forward(fund, bandpass=bandpass)
            self._last_fund = fund
            self._last_bandpass = bandpass
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
            full = self._forward(fund, bandpass=bandpass)
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
        full = self._forward(fund, bandpass=bandpass)
        self._last_fund = fund
        self._last_bandpass = bandpass

        out = {name: full[name] for name in want}
        if return_results:
            out["_results"] = results
        return out

    def predict(self, want, bandpass=None):
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
        bandpass = self._last_bandpass if bandpass is None else normalize_bandpass(bandpass)
        missing = [
            name for name in required_fundamentals(want)
            if name not in self._last_fund
        ]
        if missing:
            raise ValueError(
                f"Cannot predict {want} from the previous solve: required "
                f"fundamentals {missing} were not part of that problem."
            )
        full = self._forward(self._last_fund, bandpass=bandpass)
        return {name: full[name] for name in want}
