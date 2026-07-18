"""Configuration helpers for Dynesty sampling."""

from dataclasses import dataclass, replace
from typing import Optional


@dataclass(frozen=True)
class SamplerSettings:
    """Settings controlling a static Dynesty run.

    Parameters
    ----------
    nlive : int
        Number of live points.
    dlogz : float
        Evidence tolerance used to stop nested sampling.
    sample, bound : str
        Dynesty sampling and bounding methods.
    bootstrap : int
        Number of bootstrap realizations used to expand bounds.
    walks : int
        Number of random-walk steps per proposal.
    update_interval : int or None
        Number of likelihood calls between bound updates. If ``None``, it
        is set to ten times ``nlive``.
    """

    nlive: int = 500
    dlogz: float = 0.5
    sample: str = "rwalk"
    bound: str = "single"
    bootstrap: int = 0
    walks: int = 5
    update_interval: Optional[int] = None

    def resolved(self):
        """Return settings with a concrete bound-update interval.

        Returns
        -------
        SamplerSettings
            Copy with ``update_interval`` resolved to an integer.
        """
        interval = 10 * self.nlive if self.update_interval is None else self.update_interval
        return replace(self, update_interval=interval)


PRESETS = {
    "standard": SamplerSettings(),
    "fast": SamplerSettings(nlive=200, dlogz=1.0),
    "precise": SamplerSettings(nlive=1000, dlogz=0.1, walks=10),
}


def get_sampler_settings(preset="standard", **overrides):
    """Build sampler settings from a named preset and explicit overrides.

    Parameters
    ----------
    preset : {'standard', 'fast', 'precise'}
        Starting configuration.
    **overrides
        Non-``None`` values replacing fields in the preset.

    Returns
    -------
    SamplerSettings
        Fully resolved immutable configuration.

    Raises
    ------
    ValueError
        If ``preset`` is unknown.
    TypeError
        If an override is not a sampler setting.
    """
    try:
        settings = PRESETS[preset]
    except KeyError as exc:
        raise ValueError(f"Unknown preset {preset!r}; choose from {sorted(PRESETS)}") from exc
    clean = {key: value for key, value in overrides.items() if value is not None}
    unknown = set(clean) - set(SamplerSettings.__dataclass_fields__)
    if unknown:
        raise TypeError(f"Unknown sampler setting(s): {sorted(unknown)}")
    return replace(settings, **clean).resolved()
