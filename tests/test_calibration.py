"""Tests for intrinsic-scatter configuration and calibration flags."""

import numpy as np
import pytest

from asteroscale.calibration import (
    DEFAULT_RELATION_SCATTER,
    fractional_to_log_scatter,
    normalize_relation_scatter,
)
from asteroscale.validity import assess_validity, warn_outside_calibration


def test_default_scatter_covers_seismic_amplitude_and_granulation_relations():
    """Keep uncertainty floors for every requested empirical relation group."""
    assert set(DEFAULT_RELATION_SCATTER) == {
        "numax",
        "dnu",
        "A_env",
        "A_gran",
        "b_gran_low",
        "b_gran_high",
    }
    assert all(value > 0.0 for value in DEFAULT_RELATION_SCATTER.values())


def test_relation_scatter_can_be_overridden_or_disabled():
    """Allow per-relation overrides, including deterministic zero scatter."""
    configured = normalize_relation_scatter({"numax": 0.03, "dnu": 0.0})
    assert configured["numax"] == 0.03
    assert configured["dnu"] == 0.0
    assert configured["A_env"] == DEFAULT_RELATION_SCATTER["A_env"]


@pytest.mark.parametrize("value", [-0.1, np.inf, np.nan])
def test_invalid_relation_scatter_is_rejected(value):
    """Reject values that cannot define a finite dispersion."""
    with pytest.raises(ValueError, match="finite and non-negative"):
        normalize_relation_scatter({"numax": value})


def test_fractional_scatter_conversion_has_requested_coefficient_of_variation():
    """Verify the analytic conversion used for log-normal offsets."""
    fraction = 0.25
    sigma_log = fractional_to_log_scatter(fraction)
    recovered = np.sqrt(np.exp(sigma_log**2) - 1.0)
    assert recovered == pytest.approx(fraction)


def test_validity_checks_report_each_supported_domain():
    """Return explicit flags for dnu, numax, and envelope amplitude."""
    report = assess_validity(
        {
            "M": np.array([1.0, 2.5]),
            "Teff": np.array([5772.0, 9000.0]),
            "FeH": np.array([0.0, 0.8]),
            "numax": np.array([3090.0, 5.0]),
            "L": np.array([1.0, 1.0]),
        },
        ["numax", "dnu", "A_env"],
    )
    assert set(report) == {"numax", "dnu", "A_env"}
    assert report["numax"]["status"] == "partly_outside_calibration"
    assert report["dnu"]["fraction_within"] == pytest.approx(0.5)
    assert report["A_env"]["status"] == "partly_outside_calibration"


def test_outside_calibration_emits_a_warning():
    """Make extrapolation visible unless users explicitly suppress warnings."""
    report = {
        "dnu": {
            "status": "outside_calibration",
            "fraction_within": 0.0,
            "domain": "test domain",
        }
    }
    with pytest.warns(UserWarning, match="dnu is outside calibration"):
        warn_outside_calibration(report)
