import numpy as np
import pytest

from asteroscale import Solver
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
