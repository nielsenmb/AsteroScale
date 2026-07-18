"""Generic any-subset-in -> any-subset-out solver via nested sampling (dynesty).

"""
import numpy as np
from scipy import optimize
import dynesty
from dynesty.utils import resample_equal

from .relations import DERIVED, FUNDAMENTAL, normalize_bandpass
from .forward import evaluate_relations, required_fundamentals, required_relations
from .sampling import get_sampler_settings
from .priors import ParallaxPrior 
from .calibration import fractional_to_log_scatter, normalize_relation_scatter
from .distributions import normal, uniform, TruncatedPowerLaw, Exponential, TruncatedNormal
from . import validation, validity

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
    """Convert tuple shorthand to a normal distribution.

    Parameters
    ----------
    p : tuple or object
        ``(mean, error)`` pair or distribution-like object.

    Returns
    -------
    object
        Normal distribution for tuple input, otherwise the original object.
    """
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

    Parameters
    ----------
    given : dict
        Mapping from quantity names to exact or uncertain inputs.

    Returns
    -------
    fixed : dict
        Exact scalar inputs.
    constraints : dict
        Distribution-like uncertain inputs.
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
    """Validate and normalize the interpretation of uncertain inputs.

    Parameters
    ----------
    input_mode : str
        Requested input interpretation.

    Returns
    -------
    str
        Lower-case canonical mode.

    Raises
    ------
    ValueError
        If the mode is not supported.
    """
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

    Parameters
    ----------
    base_priors : dict
        Configured fundamental-parameter priors.
    constraints : dict
        Uncertain inputs parsed from ``given``.
    input_mode : {'propagate', 'likelihood'}
        Statistical interpretation of uncertain fundamental inputs.

    Returns
    -------
    priors : dict
        Effective priors for the current solve.
    likelihood_terms : dict
        Distributions evaluated as likelihood terms.

    Raises
    ------
    TypeError
        If a likelihood term does not provide ``logpdf``.
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
        Named Dynesty accuracy/runtime configuration, precise is lower but
        provides more samples and smoother posteriors.
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
    relation_scatter : float or dict, optional
        Fractional intrinsic scatter for empirical relations. A scalar applies
        to every calibrated relation; a dictionary overrides named defaults.
        Zero disables intrinsic scatter for a relation.
    warn_validity : bool, default=True
        Warn when evaluated samples leave an adopted calibration domain.
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
        relation_scatter=None,
        warn_validity=True,
    ):
        """Initialize a stellar-property solver.

        Parameters
        ----------
        priors : dict, optional
            Fundamental-parameter prior overrides.
        preset : {'standard', 'fast', 'precise'}, default='standard'
            Named Dynesty configuration.
        nlive : int, optional
            Number of live points overriding the preset.
        seed : int, optional
            Random-number generator seed.
        sample, bound : str, optional
            Dynesty sampling and bounding methods.
        bootstrap, walks, update_interval : int, optional
            Additional Dynesty settings.
        bandpass : {'TESS', 'Kepler'}, default='TESS'
            Photometric response used for envelope amplitudes.
        input_mode : {'propagate', 'likelihood'}, default='propagate'
            Statistical interpretation of uncertain fundamental inputs.
        relation_scatter : float or dict, optional
            Fractional intrinsic scatter for empirical relations.
        warn_validity : bool, default=True
            Warn about samples outside adopted calibration domains.
        """
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
        self.relation_scatter = normalize_relation_scatter(relation_scatter)
        self.warn_validity = bool(warn_validity)
        self.rng = np.random.default_rng(seed)
        self._last_fund = None  # fundamentals from the last solve() call,
        self._last_bandpass = None
        self._last_relation_offsets = {}
        self._last_relation_scatter = dict(self.relation_scatter)
        self._last_was_sampled = False
        self.last_validity = {}

    def _forward(self, fundamentals, bandpass=None, relation_offsets=None):
        """Evaluate available relations for fundamental parameters.

        Parameters
        ----------
        fundamentals : dict
            Scalar or array-valued fundamental parameters.
        bandpass : {'TESS', 'Kepler'}, optional
            Photometric response, defaulting to the solver setting.
        relation_offsets : dict, optional
            Natural-log calibration offsets for empirical relations.

        Returns
        -------
        dict
            Supplied fundamentals and available derived quantities.
        """
        bandpass = self.bandpass if bandpass is None else normalize_bandpass(bandpass)
        return evaluate_relations(
            fundamentals,
            bandpass=bandpass,
            relation_offsets=relation_offsets,
        )

    def _active_scatter(self, names, scatter):
        """Return active relations with nonzero intrinsic scatter.

        Parameters
        ----------
        names : iterable of str
            Requested or constrained quantity names.
        scatter : dict
            Fractional relation-scatter configuration.

        Returns
        -------
        list of str
            Active relation names in dependency order.
        """
        return [
            name for name in required_relations(names)
            if scatter.get(name, 0.0) > 0.0
        ]

    def _draw_relation_offsets(self, names, scatter, size=None):
        """Draw independent natural-log relation offsets.

        Parameters
        ----------
        names : iterable of str
            Relation names to perturb.
        scatter : dict
            Fractional relation-scatter configuration.
        size : int, optional
            Number of offsets per relation. The default draws scalars.

        Returns
        -------
        dict
            Natural-log offsets keyed by relation name.
        """
        return {
            name: self.rng.normal(
                scale=fractional_to_log_scatter(scatter[name]), size=size
            )
            for name in names
        }

    def _store_solution(
        self, fundamentals, full, bandpass, offsets, active_relations,
        relation_scatter, was_sampled, warn_validity,
    ):
        """Store reusable state and assess calibration validity.

        Parameters
        ----------
        fundamentals, full : dict
            Fundamental-only and complete evaluated solution mappings.
        bandpass : {'TESS', 'Kepler'}
            Photometric response used by the solution.
        offsets : dict
            Natural-log relation offsets represented in the solution.
        active_relations : iterable of str
            Relations used in the current problem.
        relation_scatter : dict
            Scatter configuration used by the current problem.
        was_sampled : bool
            Whether the solution consists of posterior or predictive samples.
        warn_validity : bool
            Emit calibration-domain warnings when true.

        Returns
        -------
        dict
            Calibration-domain report.
        """
        self._last_fund = fundamentals
        self._last_bandpass = bandpass
        self._last_relation_offsets = offsets
        self._last_relation_scatter = dict(relation_scatter)
        self._last_was_sampled = was_sampled
        self.last_validity = validity.assess_validity(full, active_relations)
        if warn_validity:
            validity.warn_outside_calibration(self.last_validity)
        return self.last_validity

    def _point_output(
        self, fundamentals, want, derived_targets, bandpass,
        relation_scatter, sample_relation_scatter, warn_validity,
        return_validity,
    ):
        """Build output for an exact-input solution.

        Parameters
        ----------
        fundamentals : dict
            Solved exact fundamental parameters.
        want : sequence of str
            Requested output names.
        derived_targets : dict
            Exact derived constraints used in an inversion.
        bandpass : {'TESS', 'Kepler'}
            Photometric response used for envelope amplitudes.
        relation_scatter : dict
            Fractional relation-scatter configuration.
        sample_relation_scatter : bool
            Draw predictive relation offsets for direct forward calculations.
        warn_validity : bool
            Emit calibration-domain warnings when true.
        return_validity : bool
            Include validity flags in the output when true.

        Returns
        -------
        dict
            Central values for an exact inversion, or predictive arrays when
            exact fundamentals are propagated through uncertain relations.
        """
        active_names = list(want) + list(derived_targets)
        active_relations = required_relations(active_names)
        scatter_relations = self._active_scatter(want, relation_scatter)

        # Exact derived targets define a deterministic inversion. Supporting
        # relation scatter there would require constrained sampling and a
        # Jacobian, so only direct forward predictions receive scatter.
        if sample_relation_scatter and scatter_relations and not derived_targets:
            n = max(self.nlive, 2000)
            fund = {
                name: np.full(n, value) for name, value in fundamentals.items()
            }
            offsets = self._draw_relation_offsets(
                scatter_relations, relation_scatter, size=n
            )
            full = self._forward(
                fund, bandpass=bandpass, relation_offsets=offsets
            )
            was_sampled = True
        else:
            fund = fundamentals
            offsets = {}
            full = self._forward(fund, bandpass=bandpass)
            was_sampled = False

        report = self._store_solution(
            fund, full, bandpass, offsets, active_relations,
            relation_scatter, was_sampled, warn_validity,
        )
        out = {name: full[name] for name in want}
        if return_validity:
            out["_validity"] = report
        return out

    def _bounds_for(self, name):
        """Best-effort bounds for a fundamental, from its prior's support --
        used to keep the point-estimate least-squares solve out of
        unphysical territory (e.g. negative mass) mid-iteration.

        Parameters
        ----------
        name : str
            Fundamental-parameter name.

        Returns
        -------
        tuple of float
            Lower and upper bounds, possibly infinite.
        """
        prior = self.priors[name]
        if hasattr(prior, "support"):
            try:
                return prior.support()
            except Exception:
                pass
        if hasattr(prior, "low") and hasattr(prior, "high"):
            return (prior.low, prior.high)
        return (-np.inf, np.inf)

    def _point_estimate(
        self, fixed, want, bandpass, relation_scatter, warn_validity,
        return_validity, sample_relation_scatter,
    ):
        """Fast path: every given value was a plain scalar, so there's
        nothing to marginalize over. If the fixed values cover every
        fundamental, this is just a direct forward evaluation. Otherwise,
        least-squares solves for the remaining fundamentals against any
        given derived-quantity targets (e.g. given exact Teff, numax, dnu
        -> solve for M, R). No sampler involved either way.

        Parameters
        ----------
        fixed : dict
            Exact input quantities.
        want : sequence of str
            Requested outputs.
        bandpass : {'TESS', 'Kepler'}
            Photometric response used for envelope amplitudes.
        relation_scatter : dict
            Relation-scatter configuration retained for later predictions.
        warn_validity : bool
            Emit calibration-domain warnings when true.
        return_validity : bool
            Include the validity report in the returned dictionary.
        sample_relation_scatter : bool
            Draw relation scatter for a direct forward prediction.

        Returns
        -------
        dict
            Requested point estimates.

        Raises
        ------
        ValueError
            If the exact inputs do not determine the requested outputs.
        """
        fixed_fund = {k: v for k, v in fixed.items() if k in FUNDAMENTAL}
        derived_targets = {k: v for k, v in fixed.items() if k in DERIVED}
        needed = required_fundamentals(list(want) + list(derived_targets))
        free = [p for p in needed if p not in fixed_fund]

        if not free:
            return self._point_output(
                fixed_fund, want, derived_targets, bandpass,
                relation_scatter, sample_relation_scatter, warn_validity,
                return_validity,
            )

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
            """Evaluate least-squares residuals for free fundamentals.

            Parameters
            ----------
            x : ndarray
                Trial values of the free fundamental parameters.

            Returns
            -------
            list of float
                Predicted minus target values for derived constraints.
            """
            theta = dict(zip(free, x))
            theta.update(fixed_fund)
            full = self._forward(theta, bandpass=bandpass)
            return [full[name] - target for name, target in derived_targets.items()]

        result = optimize.least_squares(residuals, x0, bounds=(lo, hi))
        validation.check_point_estimate_residuals(result, derived_targets)
        theta = dict(zip(free, result.x))
        theta.update(fixed_fund)
        return self._point_output(
            theta, want, derived_targets, bandpass,
            relation_scatter, sample_relation_scatter, warn_validity,
            return_validity,
        )

    def solve(
        self, given, want, dlogz=None, print_progress=False,
        return_results=False, return_validity=False, bandpass=None,
        input_mode=None, relation_scatter=None, sample_relation_scatter=False,
        warn_validity=None,
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
        return_validity : bool, default=False
            Include calibration-domain flags under ``'_validity'``. The same
            report is always available as :attr:`last_validity`.
        bandpass : {'TESS', 'Kepler'}, optional
            Photometric response used for ``A_env``. Overrides the value
            supplied to :class:`Solver` for this call.
        input_mode : {'propagate', 'likelihood'}, optional
            Override how uncertain fundamental inputs are interpreted. See
            :class:`Solver`. Exact scalar inputs are fixed in either mode.
        relation_scatter : float or dict, optional
            Override fractional intrinsic scatter for this call. A scalar
            applies to every configured relation and zero disables scatter.
        sample_relation_scatter : bool, default=False
            For an all-exact direct forward calculation, return predictive
            arrays including relation scatter rather than central scalars.
            Exact inversions remain deterministic.
        warn_validity : bool, optional
            Override calibration-domain warnings for this call.

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
        relation_scatter = normalize_relation_scatter(
            relation_scatter, base=self.relation_scatter
        )
        warn_validity = self.warn_validity if warn_validity is None else bool(warn_validity)

        fixed, constraints = _parse_given(given)

        if not constraints:
            return self._point_estimate(
                fixed, want, bandpass, relation_scatter, warn_validity,
                return_validity, sample_relation_scatter,
            )

        exact_derived = [name for name in fixed if name in DERIVED]
        if exact_derived:
            raise ValueError(
                "Exact derived constraints cannot be combined with uncertain "
                f"inputs during sampling: {exact_derived}. Supply a positive "
                "measurement uncertainty as a (value, uncertainty) pair, or "
                "make every input exact for a point-estimate solve."
            )

        # ``propagate`` is the calculator-style, backwards-compatible path;
        # ``likelihood`` performs Bayesian conditioning on the configured
        # population priors. Derived constraints are likelihood terms in
        # both modes.
        priors, likelihood_terms = _partition_constraints(
            self.priors, constraints, input_mode
        )

        fixed_fund = {k: v for k, v in fixed.items() if k in FUNDAMENTAL}

        active_names = list(want) + list(given)
        active_relations = required_relations(active_names)
        scatter_relations = self._active_scatter(active_names, relation_scatter)
        scatter_sigmas = {
            name: fractional_to_log_scatter(relation_scatter[name])
            for name in scatter_relations
        }
        needed = required_fundamentals(active_names)
        free_fundamentals = [p for p in needed if p not in fixed_fund]

        if not free_fundamentals and not scatter_relations:
            # Every fundamental was pinned exactly, even though some other
            # given value was probabilistic (e.g. a redundant/consistency
            # check) -- nothing left to sample.
            full = self._forward(fixed_fund, bandpass=bandpass)
            report = self._store_solution(
                fixed_fund, full, bandpass, {}, active_relations,
                relation_scatter, False, warn_validity,
            )
            out = {name: full[name] for name in want}
            if return_results:
                out["_results"] = None
            if return_validity:
                out["_validity"] = report
            return out

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
            offsets = self._draw_relation_offsets(
                scatter_relations, relation_scatter, size=n
            )
            full = self._forward(
                fund, bandpass=bandpass, relation_offsets=offsets
            )
            report = self._store_solution(
                fund, full, bandpass, offsets, active_relations,
                relation_scatter, True, warn_validity,
            )
            out = {name: full[name] for name in want}
            if return_results:
                out["_results"] = None
            if return_validity:
                out["_validity"] = report
            return out

        n_fundamentals = len(free_fundamentals)
        ndim = n_fundamentals + len(scatter_relations)

        def prior_transform(u):
            """Transform a unit-cube point to fundamental parameters.

            Parameters
            ----------
            u : ndarray
                Unit-cube coordinates.

            Returns
            -------
            ndarray
                Fundamental parameters drawn from their priors.
            """
            fundamentals = [
                priors[p].ppf(u[i]) for i, p in enumerate(free_fundamentals)
            ]
            scatter_z = [
                normal().ppf(u[n_fundamentals + i])
                for i, _ in enumerate(scatter_relations)
            ]
            return np.asarray(fundamentals + scatter_z, dtype=float)

        def loglike(theta):
            """Evaluate the joint log-likelihood.

            Parameters
            ----------
            theta : ndarray
                Free fundamental parameters followed by standard-normal
                relation-scatter variables.

            Returns
            -------
            float
                Sum of log-density terms for all uncertain constraints.
            """
            fund = dict(zip(free_fundamentals, theta[:n_fundamentals]))
            fund.update(fixed_fund)
            offsets = {
                name: theta[n_fundamentals + i] * scatter_sigmas[name]
                for i, name in enumerate(scatter_relations)
            }
            full = self._forward(
                fund, bandpass=bandpass, relation_offsets=offsets
            )
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
        offsets = {
            name: eq_samples[:, n_fundamentals + i] * scatter_sigmas[name]
            for i, name in enumerate(scatter_relations)
        }
        full = self._forward(
            fund, bandpass=bandpass, relation_offsets=offsets
        )
        report = self._store_solution(
            fund, full, bandpass, offsets, active_relations,
            relation_scatter, True, warn_validity,
        )

        out = {name: full[name] for name in want}
        if return_results:
            out["_results"] = results
        if return_validity:
            out["_validity"] = report
        return out

    def predict(self, want, bandpass=None, return_validity=False):
        """Compute quantities from the most recent solution.

        This does not rerun the sampler. A sampled solution returns arrays
        with the same length as the previous posterior, while a point-estimate
        solution returns scalar values.

        Parameters
        ----------
        want : str or sequence of str
            Additional quantities to derive.
        bandpass : {'TESS', 'Kepler'}, optional
            Photometric response. The default reuses the previous solve.
        return_validity : bool, default=False
            Include calibration-domain flags under ``'_validity'``.

        Returns
        -------
        dict
            Requested quantities evaluated for the previous result.

        Raises
        ------
        RuntimeError
            If :meth:`solve` has not been called.
        ValueError
            If required fundamentals were absent from the previous problem.

        Examples
        --------
        Solve for mass and radius, then derive additional quantities from the
        same posterior samples:

        .. code-block:: python

           solver.solve(
               {
                   "Teff": (5777, 50),
                   "numax": (3090, 30),
                   "dnu": (135.1, 1.0),
               },
               want=["M", "R"],
           )
           extra = solver.predict(["L", "rho", "logg", "A_env"])
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
        active_relations = required_relations(want)
        offsets = dict(self._last_relation_offsets)
        if self._last_was_sampled:
            missing_scatter = [
                name for name in active_relations
                if self._last_relation_scatter.get(name, 0.0) > 0.0
                and name not in offsets
            ]
            if missing_scatter:
                first = np.asarray(next(iter(self._last_fund.values())))
                size = first.shape[0] if first.ndim else None
                offsets.update(self._draw_relation_offsets(
                    missing_scatter, self._last_relation_scatter, size=size
                ))
        full = self._forward(
            self._last_fund, bandpass=bandpass, relation_offsets=offsets
        )
        report = self._store_solution(
            self._last_fund, full, bandpass, offsets, active_relations,
            self._last_relation_scatter, self._last_was_sampled,
            self.warn_validity,
        )
        out = {name: full[name] for name in want}
        if return_validity:
            out["_validity"] = report
        return out
