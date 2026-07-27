import numpy as np
import pytest

from asteroscale import relations as rel
from asteroscale.forward import evaluate_relations
from asteroscale.photometry import MARCS_DOMAIN
from asteroscale.solver import (
    DEFAULT_PHOTOMETRIC_ERROR_FLOOR,
    _apply_photometric_error_floor,
    _normalize_photometric_error_floor,
)
from asteroscale.validity import assess_validity


def test_marcs_grid_matches_source_node():
    """Check an exact MARCS node, including the Mbol zero-point shift."""
    assert rel.bc_g(5750.0, 4.5, 0.0) == pytest.approx(0.0795, abs=1e-6)
    assert rel.bc_bp(5750.0, 4.5, 0.0) == pytest.approx(-0.2399, abs=1e-6)
    assert rel.bc_rp(5750.0, 4.5, 0.0) == pytest.approx(0.5616, abs=1e-6)


def test_extinction_matches_source_interpolation():
    # Direct interpolation through the original E(B-V) tables gives
    # A_BP=1.2443 and A_RP=0.7613 at this atmosphere and A_G=1.0 mag.
    # The compact four-dimensional table remains within 0.005 mag.
    assert rel.a_bp(1.0, 5750.0, 4.5, 0.0) == pytest.approx(
        1.2443125, abs=0.005
    )
    assert rel.a_rp(1.0, 5750.0, 4.5, 0.0) == pytest.approx(
        0.7613306, abs=0.005
    )


def test_zero_ag_produces_zero_bp_and_rp_extinction():
    assert rel.a_bp(0.0, 5000.0, 3.0, -0.5) == pytest.approx(0.0)
    assert rel.a_rp(0.0, 6500.0, 4.0, 0.25) == pytest.approx(0.0)


def test_photometry_vectorizes_and_varies_with_stellar_parameters():
    teff = np.array([4500.0, 5500.0, 6500.0])
    correction = rel.bc_g(teff, np.array([2.5, 4.0, 4.5]), -0.25)
    assert correction.shape == (3,)
    assert np.all(np.isfinite(correction))
    assert len(np.unique(correction)) == 3


def test_grid_does_not_extrapolate():
    assert np.isnan(rel.bc_g(MARCS_DOMAIN["Teff"][0] - 1.0, 4.0, 0.0))
    assert np.isnan(rel.a_bp(MARCS_DOMAIN["A_G"][1] + 0.01, 5500.0, 4.0, 0.0))


def test_solar_gaia_colour_is_plausible():
    full = evaluate_relations(
        {
            "M": 1.0,
            "R": 1.0,
            "Teff": rel.TEFF_SUN,
            "plx": 100.0,
            "A_G": 0.0,
            "FeH": 0.0,
        }
    )
    assert full["BP_RP"] == pytest.approx(0.794, abs=0.02)
    assert full["M_G"] == pytest.approx(4.66, abs=0.02)


def test_photometric_validity_report_uses_marcs_domain():
    inside = evaluate_relations(
        {
            "M": 1.0,
            "R": 1.0,
            "Teff": 5772.0,
            "plx": 10.0,
            "A_G": 0.1,
            "FeH": 0.0,
        }
    )
    report = assess_validity(inside, ["BP_mag"])
    assert report["Gaia_photometry"]["status"] == "within_calibration"

    outside = dict(inside)
    outside["A_G"] = MARCS_DOMAIN["A_G"][1] + 0.1
    report = assess_validity(outside, ["BP_mag"])
    assert report["Gaia_photometry"]["status"] == "outside_calibration"


def test_numpy_and_jax_photometry_agree():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    original_backend = rel.xp
    try:
        expected = np.array(
            [
                rel.bc_bp(5300.0, 3.89, -0.23),
                rel.a_rp(0.2, 5300.0, 3.89, -0.23),
            ]
        )
        rel.xp = jnp
        compiled = jax.jit(
            lambda: jnp.stack(
                (
                    rel.bc_bp(5300.0, 3.89, -0.23),
                    rel.a_rp(0.2, 5300.0, 3.89, -0.23),
                )
            )
        )()
        np.testing.assert_allclose(np.asarray(compiled), expected, rtol=1e-6)
    finally:
        rel.xp = original_backend


def test_photometric_error_floor_is_added_in_quadrature():
    given = {
        "G_mag": (9.9, 0.01),
        "BP_RP": (0.8, 0.03),
        "numax": (3090.0, 30.0),
    }
    adjusted = _apply_photometric_error_floor(
        given, DEFAULT_PHOTOMETRIC_ERROR_FLOOR
    )
    assert adjusted["G_mag"][1] == pytest.approx(np.hypot(0.01, 0.02))
    assert adjusted["BP_RP"][1] == pytest.approx(np.hypot(0.03, 0.02))
    assert adjusted["numax"] == given["numax"]
    assert given["G_mag"] == (9.9, 0.01)


def test_zero_floor_and_custom_distribution_are_unchanged():
    class CustomDistribution:
        def logpdf(self, value):
            return -value**2

    custom = CustomDistribution()
    given = {"G_mag": (9.9, 0.01), "BP_RP": custom}
    assert _apply_photometric_error_floor(given, 0.0) is given
    assert _apply_photometric_error_floor(given, 0.02)["BP_RP"] is custom


@pytest.mark.parametrize("value", [-0.01, np.nan, np.inf])
def test_invalid_photometric_error_floor_is_rejected(value):
    with pytest.raises(ValueError, match="photometric_error_floor"):
        _normalize_photometric_error_floor(value)
