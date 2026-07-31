from types import SimpleNamespace

import numpy as np
import pytest
from baldr import Normal

from asteroscale import Solver
from asteroscale.population import load_population_prior
from asteroscale.training.gmm import (
    COORDINATE_NAMES,
    PopulationGMM,
)


def single_component_model(covariance=None):
    """Return a simple standardized four-dimensional test mixture."""
    if covariance is None:
        covariance = np.eye(4)
    return PopulationGMM(
        weights=np.asarray([1.0]),
        means=np.zeros((1, 4)),
        covariances=np.asarray([covariance], dtype=float),
        coordinate_centre=np.zeros(4),
        coordinate_scale=np.ones(4),
        support_bounds=np.column_stack((-5.0 * np.ones(4), 5.0 * np.ones(4))),
        metadata={"source": "test"},
    )


def test_population_transform_marginalizes_unused_coordinates():
    model = single_component_model()
    transform = model.marginal_condition(
        ("log10_radius", "feh")
    )
    assert transform.coordinate_names == ("log10_radius", "feh")
    np.testing.assert_allclose(transform.ppf([0.5, 0.5]), [0.0, 0.0])


def test_population_transform_conditions_correlated_gaussian():
    covariance = np.eye(4)
    covariance[0, 1] = covariance[1, 0] = 0.8
    model = single_component_model(covariance)
    transform = model.marginal_condition(
        ("log10_mass",),
        conditioned={"log10_radius": 1.0},
    )
    # For unit variances and rho=0.8, x|y=1 has mean 0.8 and variance 0.36.
    assert transform.ppf([0.5])[0] == pytest.approx(0.8)
    assert transform.cholesky_factors[0, 0, 0] == pytest.approx(0.6)


def test_population_transform_preserves_mixture_moments():
    model = PopulationGMM(
        weights=np.asarray([0.25, 0.75]),
        means=np.asarray([
            [-2.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ]),
        covariances=np.repeat((0.2 * np.eye(4))[None, :, :], 2, axis=0),
        coordinate_centre=np.zeros(4),
        coordinate_scale=np.ones(4),
        support_bounds=np.column_stack((-5.0 * np.ones(4), 5.0 * np.ones(4))),
        metadata={"source": "test"},
    )
    transform = model.marginal_condition(("log10_mass",))
    unit = (np.arange(20_000, dtype=float) + 0.5) / 20_000
    draws = np.asarray([transform.ppf([value])[0] for value in unit])
    expected_mean = 0.25 * -2.0 + 0.75 * 1.0
    assert draws.mean() == pytest.approx(expected_mean, abs=0.01)


def test_population_transform_rejects_invalid_coordinate_requests():
    model = single_component_model()
    with pytest.raises(ValueError, match="Unknown population coordinates"):
        model.marginal_condition(("age",))
    with pytest.raises(ValueError, match="both sampled and conditioned"):
        model.marginal_condition(
            ("feh",), conditioned={"feh": 0.0}
        )


def test_bundled_population_prior_loads():
    model = load_population_prior("trilegal_solar_neighbourhood")
    assert model.means.shape[1] == len(COORDINATE_NAMES)
    assert model.means.shape[0] == 256
    assert model.metadata["generation"]["synthesis_code"] == "TRILEGAL"


def test_unknown_population_prior_has_helpful_error(tmp_path):
    missing = tmp_path / "missing.npz"
    with pytest.raises(ValueError, match="Unknown population prior"):
        load_population_prior(missing)


def test_propagate_mode_keeps_independent_priors():
    given = {
        "M": (1.0, 0.05),
        "R": 1.0,
        "Teff": (5772.0, 20.0),
        "FeH": 0.0,
    }
    baseline = Solver(seed=24, nlive=10).solve(given, ["L"])["L"]
    configured = Solver(
        seed=24,
        nlive=10,
        population_prior="trilegal_solar_neighbourhood",
    ).solve(given, ["L"])["L"]
    np.testing.assert_array_equal(configured, baseline)


def test_correlated_prior_rejects_overlapping_custom_priors():
    solver = Solver(
        input_mode="likelihood",
        population_prior=single_component_model(),
        priors={"M": Normal(loc=1.0, scale=0.1, backend="numpy")},
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        solver.solve({"M": (1.0, 0.1)}, ["R"])


def test_solver_uses_population_transform_in_likelihood_mode(monkeypatch):
    captured = {}

    class FakeSampler:
        """Minimal Dynesty stand-in that records one transformed point."""

        def __init__(self, loglike, prior_transform, ndim, **kwargs):
            sample = prior_transform(np.full(ndim, 0.5))
            captured["sample"] = sample
            captured["loglike"] = loglike(sample)
            self.results = SimpleNamespace(
                samples=sample[None, :],
                logwt=np.asarray([0.0]),
                logz=np.asarray([0.0]),
            )

        def run_nested(self, **kwargs):
            """Match Dynesty's execution interface."""

    covariance = np.eye(4)
    covariance[0, 1] = covariance[1, 0] = 0.8
    model = single_component_model(covariance)
    monkeypatch.setattr("asteroscale.solver.dynesty.NestedSampler", FakeSampler)
    result = Solver(
        input_mode="likelihood",
        population_prior=model,
        nlive=10,
        warn_validity=False,
    ).solve(
        {
            "M": (1.0, 0.1),
            "Teff": 5772.0,
            "FeH": 0.0,
        },
        ["R"],
    )
    assert np.isfinite(captured["loglike"])
    assert result["R"].shape == (1,)
    assert result["R"][0] == pytest.approx(1.0)
