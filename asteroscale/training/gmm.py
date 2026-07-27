"""Weighted Gaussian-mixture compression of a stellar population."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import logsumexp


COORDINATE_NAMES = ("log10_mass", "log10_radius", "log10_teff", "feh")


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
        """Draw samples in the physical training coordinates."""
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
        """Save the model and its provenance to a compressed NPZ file."""
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
        """Load a model saved by :meth:`save`."""
        with np.load(path, allow_pickle=False) as data:
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
            break
        previous = objective
    return mixture_weight, means, covariances, objective, iteration


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
    if tolerance <= 0.0 or reg_covar <= 0.0:
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
    mixture_weight, means, covariances, objective, iterations = best
    model_metadata = dict(metadata or {})
    model_metadata.update(
        {
            "n_components": int(n_components),
            "n_training_rows": int(count),
            "effective_sample_size": _effective_sample_size(sample_weight),
            "training_mean_log_density_standardized": float(objective),
            "em_iterations": int(iterations),
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
