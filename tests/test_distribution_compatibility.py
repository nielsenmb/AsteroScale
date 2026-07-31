"""Tests for AsteroScale's transitional Baldr distribution aliases."""

import numpy as np

from asteroscale.distributions import TruncatedNormal, normal


def test_normal_alias_uses_broadcasting_baldr_backend():
    """The compatibility alias should retain array-valued evaluations."""

    distribution = normal(loc=1.0, scale=0.2)
    quantiles = distribution.ppf(np.array([0.25, 0.5, 0.75]))

    assert quantiles.shape == (3,)
    assert quantiles[1] == 1.0
    assert type(distribution).__module__ == "baldr.numpy"


def test_truncated_normal_alias_retains_support_metadata():
    """Point-estimate bounds still rely on low and high attributes."""

    distribution = TruncatedNormal(
        loc=1.0, scale=0.2, low=0.5, high=1.5
    )

    assert distribution.low == 0.5
    assert distribution.high == 1.5
    np.testing.assert_allclose(distribution.ppf([0.0, 1.0]), [0.5, 1.5])
