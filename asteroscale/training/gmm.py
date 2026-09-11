"""Weighted Gaussian-mixture compression of a stellar population."""

from __future__ import annotations

from dataclasses import dataclass
import json
import warnings
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import logsumexp, ndtri


COORDINATE_NAMES = ("log10_mass", "log10_radius", "log10_teff", "feh")


@dataclass(frozen=True)
class PopulationPriorTransform:
    """Unit-cube transform for a marginal or conditional population GMM.

    Parameters
    ----------
    weights : ndarray
        Normalized mixture weights.
    means : ndarray
        Component means in standardized selected coordinates.
    cholesky_factors : ndarray
        Lower-triangular component covariance factors.
    coordinate_centre, coordinate_scale : ndarray
        Standardization constants for the selected physical coordinates.
    coordinate_names : tuple of str
        Selected coordinate names in output order.
    """

    weights: np.ndarray
    means: np.ndarray
    cholesky_factors: np.ndarray
    coordinate_centre: np.ndarray
    coordinate_scale: np.ndarray
    coordinate_names: tuple

    def ppf(self, unit):
        """Transform a unit-cube point into population coordinates.

        The first unit coordinate jointly selects the mixture component and
        supplies the first within-component Gaussian quantile. This preserves
        the dimensionality of the continuous parameter vector.

        Parameters
        ----------
        unit : array-like
            Unit-cube point with one entry per selected coordinate.

        Returns
        -------
        ndarray
            Draw in the physical training coordinates and requested order.

        Raises
        ------
        ValueError
            If the unit-cube point has the wrong shape or lies outside
            ``[0, 1]``.
        """
        unit = np.asarray(unit, dtype=float)
        dimension = len(self.coordinate_names)
        if unit.shape != (dimension,):
            raise ValueError(
                f"Expected a unit-cube point with shape ({dimension},), "
                f"got {unit.shape}."
            )
        if np.any(~np.isfinite(unit)) or np.any((unit < 0.0) | (unit > 1.0)):
            raise ValueError("Unit-cube coordinates must be finite and in [0, 1].")

        cumulative = np.cumsum(self.weights)
        component = min(
            int(np.searchsorted(cumulative, unit[0], side="right")),
            len(self.weights) - 1,
        )
        lower = 0.0 if component == 0 else cumulative[component - 1]
        local_first = (unit[0] - lower) / self.weights[component]

        lower_open = np.nextafter(0.0, 1.0)
        upper_open = np.nextafter(1.0, 0.0)
        gaussian_unit = np.clip(unit, lower_open, upper_open)
        gaussian_unit[0] = np.clip(local_first, lower_open, upper_open)
        standard_normal = ndtri(gaussian_unit)
        standardized = (
            self.means[component]
            + self.cholesky_factors[component] @ standard_normal
        )
        return (
            standardized * self.coordinate_scale + self.coordinate_centre
        )


def _effective_sample_size(weight):
    """Return the Kish effective sample size of positive weights."""
    weight = np.asarray(weight, dtype=float)
    return float(weight.sum() ** 2 / np.square(weight).sum())


def _log_gaussian_density(values, means, covariances):
    """Evaluate every full-covariance Gaussian at every row."""
    count, dimension = values.shape
    output = np.empty((count, len(means)))
    constant = dimension * np.log(2.0 * np.pi)
    for component, (mean, covariance) in enumerate(zip(means, covariances)):
        factor = np.linalg.cholesky(covariance)
        whitened = solve_triangular(
            factor, (values - mean).T, lower=True, check_finite=False
        )
        output[:, component] = -0.5 * (
            constant
            + 2.0 * np.log(np.diag(factor)).sum()
            + np.square(whitened).sum(axis=0)
        )
    return output


@dataclass
class PopulationGMM:
    """A standardized, full-covariance Gaussian mixture.

    The Gaussian parameters live in standardized versions of
    ``(log10 M, log10 R, log10 Teff, [Fe/H])``.
    """

    weights: np.ndarray
    means: np.ndarray
    covariances: np.ndarray
    coordinate_centre: np.ndarray
    coordinate_scale: np.ndarray
    support_bounds: np.ndarray
    metadata: dict

    def __post_init__(self):
        """Validate dimensions, finite parameters, weights and covariance matrices."""
        for name in ("weights", "means", "covariances", "coordinate_centre",
                     "coordinate_scale", "support_bounds"):
            value = np.asarray(getattr(self, name), dtype=float)
            if not np.all(np.isfinite(value)):
                raise ValueError(f"Population model {name} contains non-finite values.")
            setattr(self, name, value)
        count = self.weights.size
        shapes = {"weights": (count,), "means": (count, 4),
                  "covariances": (count, 4, 4), "coordinate_centre": (4,),
                  "coordinate_scale": (4,), "support_bounds": (4, 2)}
        for name, shape in shapes.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"Population model {name} must have shape {shape}.")
        if (count == 0 or np.any(self.weights <= 0)
                or not np.isclose(self.weights.sum(), 1, rtol=1e-8, atol=1e-12)):
            raise ValueError("Population weights must be positive and sum to one.")
        self.weights = self.weights / self.weights.sum()
        if np.any(self.coordinate_scale <= 0):
            raise ValueError("Population coordinate scales must be positive.")
        if np.any(self.support_bounds[:, 0] >= self.support_bounds[:, 1]):
            raise ValueError("Population support bounds must be increasing.")
        if not np.allclose(self.covariances, self.covariances.swapaxes(-1, -2)):
            raise ValueError("Population covariance matrices must be symmetric.")
        try:
            np.linalg.cholesky(self.covariances)
        except np.linalg.LinAlgError as error:
            raise ValueError("Population covariance matrices must be positive definite.") from error
        if not isinstance(self.metadata, dict):
            raise ValueError("Population metadata must be a dictionary.")

    def marginal_condition(
        self,
        coordinate_names,
        conditioned=None,
    ):
        """Build a cached transform for selected and exactly fixed coordinates.

        Unselected, unconditioned coordinates are analytically marginalized.
        Exactly fixed coordinates update both the component weights and the
        Gaussian parameters through the standard conditional-normal formula.

        Parameters
        ----------
        coordinate_names : sequence of str
            Coordinates to draw, in the desired output order.
        conditioned : dict, optional
            Exact values keyed by names in :data:`COORDINATE_NAMES`. Values
            use the physical training coordinates, for example
            ``log10_teff`` rather than temperature in kelvin.

        Returns
        -------
        PopulationPriorTransform
            Cached unit-cube transform for the requested marginal or
            conditional mixture.

        Raises
        ------
        ValueError
            If names are unknown, duplicated, overlapping, or no coordinates
            are requested.
        """
        coordinate_names = tuple(coordinate_names)
        conditioned = dict(conditioned or {})
        known = set(COORDINATE_NAMES)
        unknown = (set(coordinate_names) | set(conditioned)) - known
        if unknown:
            raise ValueError(
                f"Unknown population coordinates: {sorted(unknown)}."
            )
        if not coordinate_names:
            raise ValueError("At least one population coordinate is required.")
        if len(set(coordinate_names)) != len(coordinate_names):
            raise ValueError("Population coordinate names must be unique.")
        overlap = set(coordinate_names) & set(conditioned)
        if overlap:
            raise ValueError(
                "Coordinates cannot be both sampled and conditioned: "
                f"{sorted(overlap)}."
            )

        selected = np.asarray(
            [COORDINATE_NAMES.index(name) for name in coordinate_names],
            dtype=int,
        )
        means = self.means[:, selected]
        covariances = self.covariances[
            :, selected[:, None], selected[None, :]
        ]
        weights = np.asarray(self.weights, dtype=float)

        if conditioned:
            fixed_names = tuple(conditioned)
            fixed = np.asarray(
                [COORDINATE_NAMES.index(name) for name in fixed_names],
                dtype=int,
            )
            fixed_values = np.asarray(
                [conditioned[name] for name in fixed_names],
                dtype=float,
            )
            if not np.all(np.isfinite(fixed_values)):
                raise ValueError("Conditioned population coordinates must be finite.")
            fixed_standardized = (
                fixed_values - self.coordinate_centre[fixed]
            ) / self.coordinate_scale[fixed]
            fixed_means = self.means[:, fixed]
            fixed_covariances = self.covariances[
                :, fixed[:, None], fixed[None, :]
            ]
            cross_covariances = self.covariances[
                :, selected[:, None], fixed[None, :]
            ]
            conditional_means = np.empty_like(means)
            conditional_covariances = np.empty_like(covariances)
            for component in range(len(weights)):
                gain = np.linalg.solve(
                    fixed_covariances[component],
                    cross_covariances[component].T,
                ).T
                conditional_means[component] = (
                    means[component]
                    + gain @ (
                        fixed_standardized - fixed_means[component]
                    )
                )
                conditional_covariances[component] = (
                    covariances[component]
                    - gain @ cross_covariances[component].T
                )
            means = conditional_means
            covariances = conditional_covariances
            log_weights = (
                np.log(weights)
                + _log_gaussian_density(
                    fixed_standardized[None, :],
                    fixed_means,
                    fixed_covariances,
                )[0]
            )
            log_weights -= logsumexp(log_weights)
            weights = np.exp(log_weights)

        # Round-off in the Schur complement can make a theoretically
        # symmetric conditional covariance differ at machine precision.
        covariances = 0.5 * (
            covariances + np.swapaxes(covariances, -1, -2)
        )
        positive = weights > 0.0
        weights = weights[positive]
        weights /= weights.sum()
        means = means[positive]
        covariances = covariances[positive]
        return PopulationPriorTransform(
            weights=weights,
            means=means,
            cholesky_factors=np.linalg.cholesky(covariances),
            coordinate_centre=self.coordinate_centre[selected],
            coordinate_scale=self.coordinate_scale[selected],
            coordinate_names=coordinate_names,
        )

    def logpdf(self, coordinates, batch_size=100_000):
        """Evaluate log density in the physical training coordinates.

        Parameters
        ----------
        coordinates : array-like
            One coordinate vector or an array with shape ``(n_samples, 4)``.
        batch_size : int, default=100000
            Maximum rows evaluated together. Batching bounds the temporary
            ``n_samples * n_components`` allocation for large catalogues.
        """
        coordinates = np.asarray(coordinates, dtype=float)
        scalar = coordinates.ndim == 1
        coordinates = np.atleast_2d(coordinates)
        if coordinates.ndim != 2 or coordinates.shape[1] != 4:
            raise ValueError("coordinates must have shape (4,) or (n_samples, 4).")
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("coordinates must be finite.")
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        standardized = (
            coordinates - self.coordinate_centre
        ) / self.coordinate_scale
        density = np.empty(len(standardized))
        for start in range(0, len(standardized), batch_size):
            stop = min(start + batch_size, len(standardized))
            component = _log_gaussian_density(
                standardized[start:stop], self.means, self.covariances
            )
            density[start:stop] = logsumexp(
                component + np.log(self.weights), axis=1
            )
        density -= np.log(self.coordinate_scale).sum()
        return density[0] if scalar else density

    def sample(self, size, random_state=None):
        """Draw samples in the physical training coordinates.

        Parameters
        ----------
        size : int
            Number of rows to draw.
        random_state : int or numpy.random.Generator, optional
            Random seed or generator.

        Returns
        -------
        ndarray
            Array of shape (size, 4), in logarithmic mass, radius, temperature,
            and linear metallicity coordinates.
        """
        if size < 1:
            raise ValueError("size must be at least one.")
        generator = np.random.default_rng(random_state)
        component = generator.choice(len(self.weights), size=size, p=self.weights)
        standardized = np.empty((size, self.means.shape[1]))
        for index in range(len(self.weights)):
            selected = component == index
            if np.any(selected):
                standardized[selected] = generator.multivariate_normal(
                    self.means[index],
                    self.covariances[index],
                    size=np.count_nonzero(selected),
                )
        return standardized * self.coordinate_scale + self.coordinate_centre

    def save(self, path):
        """Save the model and its provenance to a compressed NPZ file.

        Parameters
        ----------
        path : path-like
            Destination NPZ filename; parent directories are created as needed.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            format_version=np.asarray(1),
            coordinate_names=np.asarray(COORDINATE_NAMES),
            weights=self.weights,
            means=self.means,
            covariances=self.covariances,
            cholesky_factors=np.linalg.cholesky(self.covariances),
            coordinate_centre=self.coordinate_centre,
            coordinate_scale=self.coordinate_scale,
            support_bounds=self.support_bounds,
            metadata_json=np.asarray(json.dumps(self.metadata, sort_keys=True)),
        )

    @classmethod
    def load(cls, path):
        """Load a model saved by :meth:`save`.

        Parameters
        ----------
        path : path-like
            Source NPZ filename.

        Returns
        -------
        PopulationGMM
            Validated model with numerical arrays and metadata.

        Raises
        ------
        ValueError
            If the format, coordinates or numerical parameters are unsupported.
        """
        with np.load(path, allow_pickle=False) as data:
            if ("format_version" not in data or data["format_version"].shape != ()
                    or data["format_version"].item() != 1):
                raise ValueError("Unsupported population model format_version.")
            names = tuple(data["coordinate_names"].tolist())
            if names != COORDINATE_NAMES:
                raise ValueError(f"Unsupported population coordinates: {names}")
            return cls(
                weights=data["weights"],
                means=data["means"],
                covariances=data["covariances"],
                coordinate_centre=data["coordinate_centre"],
                coordinate_scale=data["coordinate_scale"],
                support_bounds=data["support_bounds"],
                metadata=json.loads(str(data["metadata_json"])),
            )


def _fit_once(
    values,
    sample_weight,
    n_components,
    *,
    generator,
    max_iter,
    tolerance,
    reg_covar,
    batch_size,
):
    """Run one weighted expectation-maximization fit."""
    count, dimension = values.shape
    probability = sample_weight / sample_weight.sum()
    indices = generator.choice(
        count, size=n_components, replace=False, p=probability
    )
    means = values[indices].copy()
    centred = values - np.average(values, axis=0, weights=sample_weight)
    covariance = (
        (centred * sample_weight[:, None]).T @ centred / sample_weight.sum()
    )
    covariance.flat[:: dimension + 1] += reg_covar
    covariances = np.repeat(covariance[None, :, :], n_components, axis=0)
    mixture_weight = np.full(n_components, 1.0 / n_components)
    previous = -np.inf
    converged = False

    for iteration in range(1, max_iter + 1):
        component_mass = np.zeros(n_components)
        first_moment = np.zeros((n_components, dimension))
        second_moment = np.zeros((n_components, dimension, dimension))
        objective_sum = 0.0
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            batch = values[start:stop]
            batch_weight = sample_weight[start:stop]
            log_joint = _log_gaussian_density(batch, means, covariances)
            log_joint += np.log(mixture_weight)
            normalizer = logsumexp(log_joint, axis=1)
            responsibility = np.exp(log_joint - normalizer[:, None])
            weighted = responsibility * batch_weight[:, None]
            component_mass += weighted.sum(axis=0)
            first_moment += weighted.T @ batch
            for component in range(n_components):
                second_moment[component] += (
                    (batch * weighted[:, component, None]).T @ batch
                )
            objective_sum += np.sum(batch_weight * normalizer)

        weak = component_mass <= np.finfo(float).eps * sample_weight.sum()
        if np.any(weak):
            replacements = generator.choice(
                count, size=np.count_nonzero(weak), p=probability
            )
            component_mass[weak] = np.finfo(float).eps * sample_weight.sum()

        mixture_weight = component_mass / component_mass.sum()
        updated_means = first_moment / component_mass[:, None]
        if np.any(weak):
            updated_means[weak] = values[replacements]
        means = updated_means
        for component in range(n_components):
            if weak[component]:
                covariances[component] = covariance
                continue
            covariances[component] = (
                second_moment[component] / component_mass[component]
                - np.outer(means[component], means[component])
            )
            covariances[component].flat[:: dimension + 1] += reg_covar

        objective = objective_sum / sample_weight.sum()
        if objective - previous < tolerance and objective >= previous:
            converged = True
            break
        previous = objective
    # Score the parameters actually returned, after the final M-step.
    objective_sum = 0.0
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        joint = _log_gaussian_density(values[start:stop], means, covariances)
        scores = logsumexp(joint + np.log(mixture_weight), axis=1)
        objective_sum += np.sum(sample_weight[start:stop] * scores)
    objective = objective_sum / sample_weight.sum()
    return mixture_weight, means, covariances, objective, iteration, converged


def fit_weighted_gmm(
    coordinates,
    sample_weight=None,
    n_components=32,
    *,
    n_init=5,
    max_iter=500,
    tolerance=1e-5,
    reg_covar=1e-6,
    batch_size=100_000,
    random_state=None,
    metadata=None,
):
    """Fit a standardized full-covariance GMM using weighted EM.

    Parameters
    ----------
    coordinates : array-like, shape (n_samples, 4)
        ``log10(M)``, ``log10(R)``, ``log10(Teff)``, and ``[Fe/H]``.
    sample_weight : array-like, optional
        Relative population weights. Equal weights are used when omitted.
    n_components : int, default=32
        Number of Gaussian components.
    n_init : int, default=5
        Independent initializations; the best weighted likelihood is retained.
    max_iter : int, default=500
        Maximum EM iterations per initialization.
    tolerance : float, default=1e-5
        Convergence threshold in mean weighted log density.
    reg_covar : float, default=1e-6
        Positive value added to every standardized covariance diagonal.
    batch_size : int, default=100000
        Maximum rows used in one expectation step, bounding memory use for
        large synthesis catalogues.
    random_state : int, optional
        Reproducible random seed.
    metadata : dict, optional
        Additional provenance saved with the model.

    Returns
    -------
    PopulationGMM
        Fitted portable mixture.
    """
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 4:
        raise ValueError("coordinates must have shape (n_samples, 4).")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("coordinates contain non-finite values.")
    count = len(coordinates)
    if not 1 <= n_components <= count:
        raise ValueError("n_components must be between one and n_samples.")
    if n_init < 1 or max_iter < 1 or batch_size < 1:
        raise ValueError("n_init, max_iter, and batch_size must be positive.")
    if not np.isfinite(tolerance + reg_covar) or tolerance <= 0.0 or reg_covar <= 0.0:
        raise ValueError("tolerance and reg_covar must be positive.")

    if sample_weight is None:
        sample_weight = np.ones(count)
    sample_weight = np.asarray(sample_weight, dtype=float)
    if sample_weight.shape != (count,) or not np.all(np.isfinite(sample_weight)):
        raise ValueError("sample_weight must be a finite one-dimensional column.")
    if np.any(sample_weight <= 0.0):
        raise ValueError("sample_weight must be strictly positive.")

    centre = np.average(coordinates, axis=0, weights=sample_weight)
    variance = np.average(
        np.square(coordinates - centre), axis=0, weights=sample_weight
    )
    scale = np.sqrt(variance)
    if np.any(scale <= 0.0):
        raise ValueError("Every training coordinate must have non-zero variance.")
    standardized = (coordinates - centre) / scale

    generator = np.random.default_rng(random_state)
    best = None
    for _ in range(n_init):
        result = _fit_once(
            standardized,
            sample_weight,
            n_components,
            generator=generator,
            max_iter=max_iter,
            tolerance=tolerance,
            reg_covar=reg_covar,
            batch_size=batch_size,
        )
        if best is None or result[3] > best[3]:
            best = result
    mixture_weight, means, covariances, objective, iterations, converged = best
    if not converged:
        warnings.warn("Selected GMM initialization reached max_iter without convergence.")
    model_metadata = dict(metadata or {})
    model_metadata.update(
        {
            "n_components": int(n_components),
            "n_training_rows": int(count),
            "effective_sample_size": _effective_sample_size(sample_weight),
            "training_mean_log_density_standardized": float(objective),
            "em_iterations": int(iterations),
            "em_converged": bool(converged),
            "fit_settings": {"n_init": n_init, "max_iter": max_iter,
                             "tolerance": tolerance, "reg_covar": reg_covar,
                             "batch_size": batch_size},
            "random_state": random_state,
        }
    )
    return PopulationGMM(
        weights=mixture_weight,
        means=means,
        covariances=covariances,
        coordinate_centre=centre,
        coordinate_scale=scale,
        support_bounds=np.column_stack(
            (coordinates.min(axis=0), coordinates.max(axis=0))
        ),
        metadata=model_metadata,
    )


def fit_candidate_models(
    coordinates,
    sample_weight,
    component_counts,
    *,
    validation_fraction=0.2,
    selection_tolerance=0.01,
    random_state=0,
    **fit_kwargs,
):
    """Compare candidate mixtures and refit the smallest near-best model.

    The selected component count is the smallest whose held-out mean log
    density is within ``selection_tolerance`` of the best candidate.

    Parameters
    ----------
    coordinates : array-like, shape (n_samples, 4)
        Population training coordinates.
    sample_weight : array-like
        Positive relative row weights.
    component_counts : sequence of int
        Candidate component counts.
    validation_fraction : float, default=0.2
        Fraction of rows reserved for held-out scoring.
    selection_tolerance : float, default=0.01
        Allowed difference from the best mean validation log density.
    random_state : int, default=0
        Seed controlling split, initializations and refit.
    **fit_kwargs
        Additional options passed to :func:`fit_weighted_gmm`.

    Returns
    -------
    model : PopulationGMM
        Selected model refitted to the complete catalogue.
    report : list of dict
        Validation score and BIC-like diagnostic for each component count.
    """
    coordinates = np.asarray(coordinates, dtype=float)
    sample_weight = np.asarray(sample_weight, dtype=float)
    counts = sorted(set(int(value) for value in component_counts))
    if not counts or counts[0] < 1:
        raise ValueError("component_counts must contain positive integers.")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie between zero and one.")
    if selection_tolerance < 0.0:
        raise ValueError("selection_tolerance cannot be negative.")

    generator = np.random.default_rng(random_state)
    order = generator.permutation(len(coordinates))
    split = int(round((1.0 - validation_fraction) * len(order)))
    split = min(max(split, max(counts)), len(order) - 1)
    train, validation = order[:split], order[split:]
    if len(train) < max(counts) or len(validation) == 0:
        raise ValueError("The catalogue is too small for the requested candidates.")

    report = []
    dimension = coordinates.shape[1]
    effective_n = _effective_sample_size(sample_weight[train])
    for offset, count in enumerate(counts):
        model = fit_weighted_gmm(
            coordinates[train],
            sample_weight[train],
            count,
            random_state=random_state + offset,
            **fit_kwargs,
        )
        train_logpdf = model.logpdf(coordinates[train])
        validation_logpdf = model.logpdf(coordinates[validation])
        train_mean = float(
            np.average(train_logpdf, weights=sample_weight[train])
        )
        train_total = effective_n * train_mean
        validation_mean = float(
            np.average(validation_logpdf, weights=sample_weight[validation])
        )
        parameters = (
            count - 1
            + count * dimension
            + count * dimension * (dimension + 1) // 2
        )
        report.append(
            {
                "n_components": count,
                "em_converged": model.metadata["em_converged"],
                "em_iterations": model.metadata["em_iterations"],
                "validation_mean_log_density": validation_mean,
                "bic": float(parameters * np.log(effective_n) - 2.0 * train_total),
            }
        )

    best_score = max(item["validation_mean_log_density"] for item in report)
    selected = min(
        item["n_components"]
        for item in report
        if item["validation_mean_log_density"]
        >= best_score - selection_tolerance
    )
    model = fit_weighted_gmm(
        coordinates,
        sample_weight,
        selected,
        random_state=random_state,
        metadata={
            "selection": {
                "candidate_results": report,
                "validation_fraction": validation_fraction,
                "selection_tolerance": selection_tolerance,
            }
        },
        **fit_kwargs,
    )
    return model, report
