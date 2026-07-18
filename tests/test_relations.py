import numpy as np
import pytest

from asteroscale import relations as rel


def test_solar_reference_values():
    assert rel.numax(1.0, 1.0, rel.TEFF_SUN, 0.0) == pytest.approx(rel.NUMAX_SUN)
    assert rel.dnu(1.0, 1.0, rel.TEFF_SUN, 0.0) == pytest.approx(rel.DNU_SUN)
    assert rel.luminosity(1.0, rel.TEFF_SUN) == pytest.approx(1.0)
    assert rel.mean_density(1.0, 1.0) == pytest.approx(1.0)


def test_ball_amplitude_and_fwhm_numerically():
    beta_sun = 1.0 - np.exp((rel.TEFF_SUN - 8907.0) / 1250.0)
    expected_amplitude = 2.1 * beta_sun * (rel.TEFF_SUN / 5777.0) ** -2
    assert rel.envelope_amplitude(1.0, 1.0, rel.TEFF_SUN) == pytest.approx(expected_amplitude)
    expected_kepler = 2.5 * beta_sun * (rel.TEFF_SUN / 5777.0) ** -2
    assert rel.envelope_amplitude(
        1.0, 1.0, rel.TEFF_SUN, bandpass="Kepler"
    ) == pytest.approx(expected_kepler)
    assert rel.envelope_fwhm(1000.0, 5000.0) == pytest.approx(0.66 * 1000.0**0.88)
    hot = 0.66 * 1000.0**0.88 * (1.0 + 6e-4 * (6272.0 - rel.TEFF_SUN))
    assert rel.envelope_fwhm(1000.0, 6272.0) == pytest.approx(hot)


def test_unknown_amplitude_bandpass_is_rejected():
    with pytest.raises(ValueError, match="TESS.*Kepler"):
        rel.envelope_amplitude(1.0, 1.0, rel.TEFF_SUN, bandpass="Gaia")


def test_amplitude_is_zero_beyond_instability_strip_red_edge():
    luminosity = 10.0
    hotter_than_red_edge = rel.amplitude_red_edge(luminosity) + 100.0
    assert rel.envelope_amplitude(
        1.5, luminosity, hotter_than_red_edge
    ) == pytest.approx(0.0)


def test_kallinger_background_scalings():
    numax, mass = 100.0, 1.2
    assert rel.granulation_amplitude(numax, mass) == pytest.approx(
        3710.0 * numax**-0.613 * mass**-0.26
    )
    assert rel.granulation_frequency_low(numax) == pytest.approx(0.317 * numax**0.970)
    assert rel.granulation_frequency_high(numax) == pytest.approx(0.948 * numax**0.992)


# def test_super_lorentzian_integrates_to_variance():
#     frequency = np.geomspace(1e-5, 1e5, 200_000)
#     power = rel.harvey_super_lorentzian(frequency, 20.0, 100.0)
#     assert np.trapezoid(power, frequency) == pytest.approx(20.0**2, rel=2e-4)


def test_relations_vectorize():
    values = rel.envelope_fwhm(np.array([100.0, 1000.0]), np.array([5000.0, 6200.0]))
    assert values.shape == (2,)
    assert np.all(values > 0)
