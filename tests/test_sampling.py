import pytest

from asteroscale.sampling import get_sampler_settings


def test_standard_dynesty_defaults():
    settings = get_sampler_settings()
    assert settings.sample == "rwalk"
    assert settings.bound == "single"
    assert settings.bootstrap == 0
    assert settings.walks == 5
    assert settings.update_interval == 10 * settings.nlive


def test_presets_and_overrides():
    assert get_sampler_settings("fast").nlive == 200
    assert get_sampler_settings("precise").dlogz == pytest.approx(0.1)
    settings = get_sampler_settings("fast", nlive=123, walks=8)
    assert settings.nlive == 123
    assert settings.walks == 8
    assert settings.update_interval == 1230


def test_unknown_preset_is_rejected():
    with pytest.raises(ValueError, match="Unknown preset"):
        get_sampler_settings("quickish")
