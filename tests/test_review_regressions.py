"""Scientific and inference failures identified in the referee review."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm

from asteroscale import Solver, relations
from asteroscale.calibration import normalize_relation_scatter
from asteroscale.solver import _apply_photometric_error_floor
from asteroscale.training import PopulationGMM, fit_weighted_gmm, read_trilegal

SOLAR = {"M": 1., "R": 1., "Teff": 5772., "FeH": 0.}


def test_exact_inversion_rejects_missing_and_dependent_constraints():
    """One seismic constraint, or two equivalent ones, cannot identify M and R."""
    with pytest.raises(ValueError, match="Not enough information"):
        Solver().solve({"Teff": 5772., "FeH": 0., "numax": 3090.}, ["M", "R"])
    with pytest.raises(ValueError, match="rank deficient"):
        Solver().solve(
            {"Teff": 5772., "FeH": 0., "numax": 3090.,
             "logg": relations.logg(1., 1.)}, ["M", "R"]
        )


def test_exact_inversion_recovers_independent_constraints():
    """Dimensionless residuals retain a well-determined solar inversion."""
    answer = Solver().solve(
        {"Teff": 5772., "FeH": 0., "numax": 3090., "dnu": 135.1}, ["M", "R"]
    )
    assert answer["M"] == pytest.approx(1., abs=1e-7)
    assert answer["R"] == pytest.approx(1., abs=1e-7)


def test_fixed_model_checks_exact_and_noisy_redundant_data():
    """A fixed star still needs consistency checks against extra observations."""
    with pytest.raises(ValueError, match="Inconsistent exact"):
        Solver().solve({**SOLAR, "numax": 100.}, ["numax"])
    with pytest.warns(UserWarning, match="conflicts"):
        answer = Solver().solve({**SOLAR, "numax": (100., 1.)}, ["M"], return_metadata=True)
    assert answer["_metadata"]["fixed_model_diagnostics"]["numax"]["standardized_residual"] == 2990.


def test_calibration_sensitivity_reproduces_analytic_mass_radius_shift():
    """Changing only the reference separation shifts inferred R by k² and M by k⁴."""
    given = {"Teff": 5772., "FeH": 0., "numax": 3090., "dnu": 135.1}
    anchored = Solver().solve(given, ["M", "R"])
    original = Solver(dnu_calibration="guggenberger2016").solve(given, ["M", "R"])
    factor = relations._DNU_REF_NORM
    assert anchored["R"] / original["R"] == pytest.approx(factor**2)
    assert anchored["M"] / original["M"] == pytest.approx(factor**4)
    assert relations.numax(1., 1., 5772., -.5, correction="none") == 3090.


def test_amplitude_alias_preserves_scatter_and_predict_state():
    """Mission conversion must preserve the canonical latent amplitude exactly."""
    solver = Solver(preset="precise", seed=9)
    bol = solver.solve(SOLAR, ["amplitude_bolometric"], sample_relation_scatter=True)
    with pytest.warns(FutureWarning):
        legacy = solver.predict(["A_env"])
    ratio = relations._legacy_amplitude(1., 5772.)
    np.testing.assert_allclose(legacy["A_env"], bol["amplitude_bolometric"] * ratio)
    with pytest.raises(ValueError, match="share one scatter"):
        normalize_relation_scatter({"A_env": .1, "amplitude_bolometric": .2})
    assert normalize_relation_scatter({"A_env": .1})["amplitude_bolometric"] == .1


@pytest.mark.parametrize("name", ["BC_G", "A_BP", "BP_RP", "logg", "L"])
def test_unsupported_scatter_is_rejected(name):
    """Settings must not be accepted for ignored or unsuitable offsets."""
    with pytest.raises(KeyError):
        Solver(relation_scatter={name: .1})


def test_negative_parallax_likelihood_and_seeded_resampling():
    """Real Dynesty runs accept negative observations and reproduce final arrays."""
    results = []
    for measurement in [(-.1, .5), norm(-.1, .5)]:
        solver = Solver(input_mode="likelihood", seed=42, nlive=30, warn_validity=False)
        results.append(solver.solve({"plx": measurement}, ["plx"], dlogz=.5,
                                    return_metadata=True, return_results=True))
    assert np.all(results[0]["plx"] > 0)
    np.testing.assert_array_equal(results[0]["plx"], results[1]["plx"])
    assert results[0]["_metadata"]["raw_sample_columns"] == ["plx"]


def test_propagation_rejects_unphysical_samples():
    """Broad measurement Gaussians cannot become negative physical parameters."""
    with pytest.raises(ValueError, match="invalid physical draws"):
        Solver(seed=3).solve({**SOLAR, "M": (1., 2.)}, ["L", "M"])
    with pytest.raises(ValueError):
        Solver().solve({"plx": -.1}, ["d"])


def test_gaussian_photometry_representations_have_identical_floors():
    """Tuple, SciPy and Baldr Gaussians describe the same effective measurement."""
    from baldr import Normal
    measurements = [(9., .01), norm(9., .01), Normal(loc=9., scale=.01, backend="numpy")]
    for measurement in measurements:
        result = _apply_photometric_error_floor({"G_mag": measurement}, .02)
        assert result["G_mag"] == pytest.approx((9., np.hypot(.01, .02)))


def test_exact_gmm_conditioning_uses_analytic_conditional():
    """Exact fundamentals alone can update a population prior without Dynesty."""
    covariance = np.eye(4) * .01
    covariance[0, 1] = covariance[1, 0] = .008
    model = PopulationGMM(np.ones(1), np.zeros((1, 4)), covariance[None],
                          np.array([0., 0., np.log10(5772.), 0.]), np.ones(4),
                          np.array([[-1., 1.], [-1., 1.], [3., 4.], [-1., 1.]]), {})
    solver = Solver(population_prior=model, input_mode="likelihood", seed=3, warn_validity=False)
    result = solver.solve({"R": 10**.1}, ["M"], return_metadata=True)
    logmass = np.log10(result["M"])
    assert np.mean(logmass) == pytest.approx(.08, abs=.004)
    assert np.std(logmass) == pytest.approx(.06, abs=.004)
    assert result["_metadata"]["population_prior_active"]


def test_all_requests_only_available_dependencies():
    """Exploring stellar outputs must not add parallax or extinction priors."""
    solver = Solver()
    answer = solver.solve(SOLAR, "all")
    assert "numax" in answer and "plx" not in answer and "G_mag" not in answer
    assert solver.predict("all").keys() == answer.keys()
    with pytest.raises(KeyError, match="Unknown fundamental prior"):
        Solver(priors={"mass": (1., .1)})


def test_redundant_photometric_likelihood_is_rejected():
    """Three independent terms cannot represent two magnitudes and their colour."""
    with pytest.raises(ValueError, match="cannot be independent"):
        Solver().solve({**SOLAR, "plx": 10., "A_G": .1,
                        "BP_mag": (10., .1), "RP_mag": (9., .1), "BP_RP": (1., .1)}, ["L"])


def test_streaming_import_preserves_rows_weights_and_provenance(tmp_path):
    """Chunk boundaries do not alter selection, and unequal areas get correct weights."""
    source = "# Mact logL logTe [M/H] m-M0\n1 0 3.76 0 5\n2 1 3.8 -.2 15\n.9 0 3.75 .1 6\n"
    paths = [tmp_path / "a.dat", tmp_path / "b.dat"]
    for path in paths:
        path.write_text(source)
    result = read_trilegal(paths, max_distance_pc=1000., R_range=(.09, 1000.),
                           chunk_size=1, field_areas_deg2=[.5, 1.])
    comparison = read_trilegal(paths, max_distance_pc=1000., R_range=(.09, 1000.),
                               chunk_size=100, field_areas_deg2=[.5, 1.])
    np.testing.assert_array_equal(result.coordinates, comparison.coordinates)
    np.testing.assert_array_equal(result.weight, [2., 2., 1., 1.])
    manifest = result.metadata["source_manifest"][0]
    assert manifest["raw_rows"] == 3 and manifest["retained_rows"] == 2
    assert manifest["sha256"] == hashlib.sha256(source.encode()).hexdigest()
    assert result.metadata["R_range"] == (.09, 1000.)


def test_em_objective_describes_saved_parameters_and_exposes_nonconvergence(tmp_path):
    """The reported final likelihood must agree with evaluating the returned model."""
    coordinates = np.random.default_rng(11).normal(size=(70, 4))
    with pytest.warns(UserWarning, match="without convergence"):
        model = fit_weighted_gmm(coordinates, n_components=2, max_iter=1, n_init=1, random_state=2)
    expected = np.mean(model.logpdf(coordinates)) + np.log(model.coordinate_scale).sum()
    assert model.metadata["training_mean_log_density_standardized"] == pytest.approx(expected)
    assert not model.metadata["em_converged"]
    path = tmp_path / "model.npz"
    model.save(path)
    with np.load(path) as data:
        arrays = dict(data)
    arrays["format_version"] = 999
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="format_version"):
        PopulationGMM.load(path)
    arrays["format_version"] = 1
    arrays["coordinate_scale"][0] = 0
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="scales must be positive"):
        PopulationGMM.load(path)


def test_grid_zero_point_does_not_modify_extinction_inputs(monkeypatch, tmp_path):
    """Changing the bolometric zero point must leave the reddening table untouched."""
    path = Path(__file__).parents[1] / "tools" / "generate_marcs_grid.py"
    spec = importlib.util.spec_from_file_location("grid_generator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    regular = np.arange(6., dtype=float).reshape(1, 1, 1, 2, 3)
    original = regular.copy()
    monkeypatch.setattr(module, "_load_source", lambda _: (None, np.array([0., .1]), None))
    monkeypatch.setattr(module, "_regularize_spatial", lambda *_: regular)
    def regrid(values, ebv):
        """Assert that the source BC array is unchanged at the extinction calculation."""
        np.testing.assert_array_equal(values, original)
        return np.array([0., 1.]), np.zeros((1, 1, 1, 2, 2))
    monkeypatch.setattr(module, "_regrid_extinction", regrid)
    module.generate(tmp_path, tmp_path / "grid.npz")


def test_exact_inversion_cannot_silently_ignore_configured_population():
    """An active GMM requires probabilistic data for derived inverse constraints."""
    with pytest.raises(ValueError, match="does not apply the population prior"):
        Solver(input_mode="likelihood", population_prior="trilegal_solar_neighbourhood").solve(
            {"Teff": 5772., "FeH": 0., "numax": 3090., "dnu": 135.1}, ["M", "R"])


def test_worker_reports_constructor_errors_and_limits_loaded_blas():
    """Batch error handling includes construction; loaded BLAS respects the limit."""
    from threadpoolctl import threadpool_info, threadpool_limits
    from asteroscale.batch import _init_worker, _solve_one
    result = _solve_one(("bad", SOLAR, ["L"], {"priors": {"typo": (1., .1)}}, 4))
    assert result[0] == "bad" and result[1] is None and "KeyError" in result[2]
    with threadpool_limits(limits=2):
        _init_worker()
        assert all(pool["num_threads"] == 1 for pool in threadpool_info())
