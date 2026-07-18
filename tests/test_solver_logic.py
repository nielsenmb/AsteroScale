import numpy as np
import pytest

from asteroscale import Solver
from asteroscale.calibration import DEFAULT_RELATION_SCATTER
from asteroscale.distributions import normal
from asteroscale.solver import _partition_constraints
from asteroscale.forward import evaluate_relations
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


def test_exact_derived_constraint_is_rejected_in_sampling_problem():
    with pytest.raises(ValueError, match="Exact derived constraints"):
        Solver().solve(
            {"Teff": (5772.0, 20.0), "numax": 3090.0},
            ["M"],
        )


def test_relation_offset_is_applied_multiplicatively():
    fundamentals = {"M": 1.0, "R": 1.0, "Teff": 5772.0, "FeH": 0.0}
    central = evaluate_relations(fundamentals)["numax"]
    shifted = evaluate_relations(
        fundamentals, relation_offsets={"numax": np.log(1.1)}
    )["numax"]
    assert shifted == pytest.approx(1.1 * central)


@pytest.mark.parametrize("preset", ["fast", "standard"])
def test_fast_and_standard_presets_disable_relation_scatter(preset):
    solver = Solver(preset=preset)
    assert set(solver.relation_scatter) == set(DEFAULT_RELATION_SCATTER)
    assert all(value == 0.0 for value in solver.relation_scatter.values())


def test_precise_preset_uses_default_relation_scatter():
    solver = Solver(preset="precise")
    assert solver.relation_scatter == DEFAULT_RELATION_SCATTER


def test_explicit_relation_scatter_overrides_selected_preset():
    solver = Solver(
        preset="standard", relation_scatter={"numax": 0.03}
    )
    assert solver.relation_scatter["numax"] == pytest.approx(0.03)
    assert solver.relation_scatter["dnu"] == 0.0


def test_zero_relation_scatter_disables_predictive_broadening():
    given = {
        "M": (1.0, 1e-6), "R": 1.0, "Teff": 5772.0, "FeH": 0.0,
    }
    without = Solver(seed=8, relation_scatter={"numax": 0.0}).solve(
        given, ["numax"]
    )["numax"]
    with_default = Solver(seed=8, preset="precise").solve(
        given, ["numax"]
    )["numax"]
    assert np.std(with_default) > 1000.0 * np.std(without)


def test_exact_forward_prediction_can_sample_relation_scatter():
    result = Solver(seed=9, preset="precise").solve(
        {"M": 1.0, "R": 1.0, "Teff": 5772.0, "FeH": 0.0},
        ["numax"],
        sample_relation_scatter=True,
    )["numax"]
    assert result.shape == (2000,)
    assert np.std(result) > 0.0


@pytest.mark.parametrize("quantity", ["dnu", "A_env", "A_gran"])
def test_predictive_scatter_operates_for_each_relation_group(quantity):
    result = Solver(seed=10, relation_scatter=0.0).solve(
        {"M": 1.0, "R": 1.0, "Teff": 5772.0, "FeH": 0.0},
        [quantity],
        relation_scatter={quantity: 0.1},
        sample_relation_scatter=True,
    )[quantity]
    assert result.shape == (2000,)
    assert np.std(result) > 0.0


def test_validity_report_is_returned_and_stored():
    solver = Solver(warn_validity=False)
    result = solver.solve(
        {"M": 2.5, "R": 12.0, "Teff": 4900.0, "FeH": 0.1},
        ["dnu"],
        return_validity=True,
    )
    assert result["_validity"]["dnu"]["status"] == "outside_calibration"
    assert solver.last_validity == result["_validity"]
