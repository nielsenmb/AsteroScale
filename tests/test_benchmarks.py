"""Numerical checks against independent benchmark-star measurements."""

import pytest

from asteroscale import Solver


# Literature properties and seismic measurements are deliberately kept in
# this test rather than inferred by AsteroScale. They check that future
# relation changes remain reasonably close to independent stars.
BENCHMARKS = {
    # Solar anchors adopted by AsteroScale. This explicit case makes changes
    # to the normalization visible in the benchmark suite as well as in the
    # lower-level relation tests.
    "Sun": {
        "regime": "solar anchor",
        "fundamental": {"M": 1.0, "R": 1.0, "Teff": 5772.0, "FeH": 0.0},
        "seismic": {"numax": 3090.0, "dnu": 135.1},
        "tolerance": {"numax": 1e-10, "dnu": 1e-10},
    },
    # White et al. (2013), MNRAS 433, 1262: interferometric radius,
    # near-model-independent mass and Teff, plus global seismic parameters.
    # This is also a Kepler LEGACY dwarf (Lund et al. 2017).
    "16 Cyg A": {
        "regime": "Kepler LEGACY dwarf",
        "fundamental": {
            "M": 1.07,
            "R": 1.218,
            "Teff": 5839.0,
            "FeH": 0.10,
        },
        "seismic": {"numax": 2201.0, "dnu": 103.5},
        "tolerance": {"numax": 0.10, "dnu": 0.07},
    },
    # Kervella et al. (2017), A&A 598, L7; seismic inputs summarized in the
    # real-star tutorial from Kjeldsen et al. (2005).
    "alpha Cen A": {
        "regime": "interferometric dwarf",
        "fundamental": {
            "M": 1.1055,
            "R": 1.2234,
            "Teff": 5795.0,
            "FeH": 0.23,
        },
        "seismic": {"numax": 2410.0, "dnu": 106.0},
        "tolerance": {"numax": 0.10, "dnu": 0.07},
    },
    # North et al. (2007), MNRAS 380, L80: interferometric R and Teff and
    # seismic-density mass. Bedding et al. (2007) measured dnu=57.24 uHz;
    # Tian et al. (2014) report numax about 1000 uHz.
    "beta Hyi": {
        "regime": "subgiant",
        "fundamental": {
            "M": 1.07,
            "R": 1.814,
            "Teff": 5872.0,
            "FeH": -0.04,
        },
        "seismic": {"numax": 1000.0, "dnu": 57.24},
        "tolerance": {"numax": 0.10, "dnu": 0.07},
    },
    # Arentoft et al. (2019), A&A 622, A190.
    "epsilon Tau": {
        "regime": "red giant",
        "fundamental": {
            "M": 2.458,
            "R": 12.46,
            "Teff": 4950.0,
            "FeH": 0.15,
        },
        "seismic": {"numax": 56.4, "dnu": 5.00},
        "tolerance": {"numax": 0.10, "dnu": 0.07},
    },
    # Themeßl et al. (2018), MNRAS 478, 4669: dynamical M/R and
    # spectroscopic Teff/[M/H] from the eclipsing binary, independently of
    # their measured numax=46.4 uHz and dnu=4.564 uHz.
    "KIC 8410637": {
        "regime": "eclipsing-binary red giant",
        "fundamental": {
            "M": 1.47,
            "R": 10.60,
            "Teff": 4605.0,
            "FeH": 0.02,
        },
        "seismic": {"numax": 46.4, "dnu": 4.564},
        "tolerance": {"numax": 0.10, "dnu": 0.07},
    },
}


@pytest.mark.parametrize("star", BENCHMARKS)
def test_forward_relations_remain_close_to_benchmark_stars(star):
    """Guard against large numerical or calibration regressions."""
    data = BENCHMARKS[star]
    prediction = Solver(warn_validity=False).solve(
        data["fundamental"], ["numax", "dnu"]
    )
    assert prediction["numax"] == pytest.approx(
        data["seismic"]["numax"], rel=data["tolerance"]["numax"]
    )
    assert prediction["dnu"] == pytest.approx(
        data["seismic"]["dnu"], rel=data["tolerance"]["dnu"]
    )
