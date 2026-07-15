import numpy as np
import pytest

from asteroscale import Solver
from asteroscale.distributions import normal
from asteroscale.solver import _partition_constraints
from asteroscale.relations import DERIVED, FUNDAMENTAL
from asteroscale.validation import normalize_want


SOLAR = {"M": 1.0, "R": 1.0, "Teff": 5772.0, "plx": 100.0, "A_G": 0.0, "FeH": 0.0}


def test_want_all_expands_to_every_public_quantity():
    expected = list(FUNDAMENTAL) + list(DERIVED)
    assert normalize_want("all") == expected
    assert normalize_want(["all"]) == expected


def test_all_cannot_be_mixed_with_names():
    with pytest.raises(ValueError, match="cannot be combined"):
        normalize_want(["all", "M"])


def test_exact_forward_solve_returns_all_values():
    result = Solver(seed=4).solve(SOLAR, want="all")
    assert set(result) == set(FUNDAMENTAL) | set(DERIVED)
    assert result["M"] == 1.0
    assert result["numax"] == pytest.approx(3090.0)
    assert result["dnu"] == pytest.approx(135.1)


def test_exact_seismic_prediction_does_not_require_gaia_parameters():
    result = Solver().solve(
        {"M": 1.0, "R": 1.0, "Teff": 5772.0, "FeH": 0.0},
        want=["numax", "dnu", "A_env"],
    )
    assert result["numax"] == pytest.approx(3090.0)
    assert result["dnu"] == pytest.approx(135.1)


def test_solver_and_per_call_bandpass_control_envelope_amplitude():
    star = {"M": 1.0, "R": 1.0, "Teff": 5772.0, "FeH": 0.0}
    solver = Solver(bandpass="Kepler")
    kepler = solver.solve(star, want=["A_env"])["A_env"]
    tess = solver.solve(star, want=["A_env"], bandpass="tess")["A_env"]
    assert kepler / tess == pytest.approx(2.5 / 2.1)


def test_predict_requires_a_previous_solve():
    with pytest.raises(RuntimeError, match="previous solve"):
        Solver().predict(["L"])


def test_point_estimate_recovers_mass_and_radius():
    solver = Solver()
    result = solver.solve(
        {"Teff": 5772.0, "FeH": 0.0, "numax": 3090.0, "dnu": 135.1},
        want=["M", "R"],
    )
    assert result["M"] == pytest.approx(1.0, rel=1e-5)
    assert result["R"] == pytest.approx(1.0, rel=1e-5)


def test_prior_predictive_is_reproducible():
    given = {"Teff": (5772.0, 20.0), "M": (1.0, 0.05), "R": 1.0,
             "plx": 100.0, "A_G": 0.0, "FeH": 0.0}
    first = Solver(seed=12, nlive=10).solve(given, ["L"])["L"]
    second = Solver(seed=12, nlive=10).solve(given, ["L"])["L"]
    np.testing.assert_array_equal(first, second)


def test_propagate_mode_replaces_fundamental_prior():
    population_prior = normal(loc=0.8, scale=0.1)
    measurement = normal(loc=1.2, scale=0.05)
    priors, likelihood = _partition_constraints(
        {"M": population_prior}, {"M": measurement}, "propagate"
    )
    assert priors["M"] is measurement
    assert likelihood == {}


def test_likelihood_mode_retains_prior_and_conditions_on_measurement():
    population_prior = normal(loc=0.8, scale=0.1)
    measurement = normal(loc=1.2, scale=0.05)
    priors, likelihood = _partition_constraints(
        {"M": population_prior}, {"M": measurement}, "likelihood"
    )
    assert priors["M"] is population_prior
    assert likelihood["M"] is measurement


def test_input_mode_can_be_set_on_solver_or_overridden_per_call():
    solver = Solver(input_mode="likelihood")
    assert solver.input_mode == "likelihood"
    with pytest.raises(ValueError, match="Unknown input_mode"):
        solver.solve({"M": (1.0, 0.1)}, ["M"], input_mode="invalid")


def test_likelihood_mode_requires_logpdf():
    class QuantileOnly:
        def ppf(self, u):
            return u

    with pytest.raises(TypeError, match=r"needs a logpdf\(\) method"):
        Solver(input_mode="likelihood").solve({"M": QuantileOnly()}, ["M"])
