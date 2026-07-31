"""Backward-compatible access to distributions provided by Baldr.

New code should import the public constructors directly from :mod:`baldr` and
select ``backend="numpy"``.  AsteroScale retains these aliases so existing
custom-prior code continues to work while the distribution implementations
live in their dedicated package.
"""

from baldr import (
    Beta,
    CallableDistribution,
    Exponential as _Exponential,
    Normal,
    TruncatedNormal as _TruncatedNormal,
    TruncatedPowerLaw as _TruncatedPowerLaw,
    Uniform,
)


def beta(a=1.0, b=1.0, loc=0.0, scale=1.0):
    """Construct a broadcasting Baldr beta distribution.

    Parameters
    ----------
    a, b : float, default=1.0
        Positive shape parameters.
    loc : float, default=0.0
        Lower support boundary.
    scale : float, default=1.0
        Positive support width.

    Returns
    -------
    baldr.numpy.Beta
        NumPy-backed beta distribution.
    """

    return Beta(a=a, b=b, loc=loc, scale=scale, backend="numpy")


def normal(loc=0.0, scale=1.0):
    """Construct a broadcasting Baldr normal distribution.

    Parameters
    ----------
    loc : float, default=0.0
        Distribution mean.
    scale : float, default=1.0
        Positive standard deviation.

    Returns
    -------
    baldr.numpy.Normal
        NumPy-backed normal distribution.
    """

    return Normal(loc=loc, scale=scale, backend="numpy")


def uniform(loc=0.0, scale=1.0):
    """Construct a broadcasting Baldr uniform distribution.

    Parameters
    ----------
    loc : float, default=0.0
        Lower support boundary.
    scale : float, default=1.0
        Positive support width.

    Returns
    -------
    baldr.numpy.Uniform
        NumPy-backed uniform distribution.
    """

    return Uniform(loc=loc, scale=scale, backend="numpy")


def TruncatedPowerLaw(alpha, low, high):
    """Construct a broadcasting Baldr truncated power-law distribution.

    Parameters
    ----------
    alpha : float
        Exponent in the density proportional to ``x**(-alpha)``.
    low, high : float
        Positive ordered support boundaries.

    Returns
    -------
    baldr.numpy.TruncatedPowerLaw
        NumPy-backed truncated power-law distribution.
    """

    return _TruncatedPowerLaw(
        alpha=alpha, low=low, high=high, backend="numpy"
    )


def TruncatedNormal(loc, scale, low, high):
    """Construct a broadcasting Baldr truncated normal distribution.

    Parameters
    ----------
    loc : float
        Mean of the untruncated normal distribution.
    scale : float
        Positive standard deviation.
    low, high : float
        Ordered truncation boundaries.

    Returns
    -------
    baldr.numpy.TruncatedNormal
        NumPy-backed truncated normal distribution.
    """

    return _TruncatedNormal(
        loc=loc, scale=scale, low=low, high=high, backend="numpy"
    )


def Exponential(scale=1.0):
    """Construct a broadcasting Baldr exponential distribution.

    Parameters
    ----------
    scale : float, default=1.0
        Positive inverse-rate scale.

    Returns
    -------
    baldr.numpy.Exponential
        NumPy-backed exponential distribution.
    """

    return _Exponential(scale=scale, backend="numpy")


distribution = CallableDistribution

__all__ = [
    "Exponential",
    "TruncatedNormal",
    "TruncatedPowerLaw",
    "beta",
    "distribution",
    "normal",
    "uniform",
]
