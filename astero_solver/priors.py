"""Small custom prior distributions not already in scipy.stats.

Anything with a .ppf(u) method (mapping the unit interval to the parameter's
support) works as a prior here -- frozen scipy.stats distributions, these
classes, or your own.
"""
import numpy as np
from scipy import stats


class TruncatedPowerLaw:
    """pdf(x) ~ x^-alpha on [low, high], via the analytic inverse CDF.

    alpha=2.35 is the Salpeter (1955) IMF slope -- a reasonable default for
    a stellar mass prior, reflecting that lower-mass stars are genuinely
    far more common than higher-mass ones (unlike a flat/uniform prior,
    which overweights high mass relative to any real stellar population).
    alpha=1 is the log-uniform special case (handled separately below).

    scipy.stats.powerlaw doesn't cover this: its shape parameter must be
    positive, so it can't represent a *decreasing* power law like this one.
    """

    def __init__(self, alpha, low, high):
        self.alpha = alpha
        self.low = low
        self.high = high
        self._p = 1.0 - alpha
        self._is_loguniform = abs(self._p) < 1e-12  # alpha == 1 special case
        if self._is_loguniform:
            self._log_low = np.log(low)
            self._log_high = np.log(high)
        else:
            self._low_p = low**self._p
            self._high_p = high**self._p

    def ppf(self, u):
        u = np.asarray(u)
        if self._is_loguniform:
            return np.exp(self._log_low + u * (self._log_high - self._log_low))
        return (self._low_p + u * (self._high_p - self._low_p)) ** (1.0 / self._p)

    def pdf(self, x):
        x = np.asarray(x)
        if self._is_loguniform:
            norm = 1.0 / (x * (self._log_high - self._log_low))
        else:
            norm = self._p * x ** (-self.alpha) / (self._high_p - self._low_p)
        return np.where((x >= self.low) & (x <= self.high), norm, 0.0)


class ParallaxPrior:
    """Bailer-Jones (2015) exponentially-decreasing space density prior,
    p(r) ~ r^2 exp(-r/L) -- the standard distance prior used for Gaia
    parallax inference, favoring the volume-weighted distances of a
    roughly uniform stellar density out to a few times the length scale L,
    falling off beyond it. Returns parallax in mas.

    r ~ Gamma(shape=3, scale=L) has this density. Since parallax = 1000/r
    is a decreasing function of r, ppf_plx(u) = 1000 / ppf_r(1-u).
    """

    def __init__(self, length_scale_pc=1350.0):
        self.length_scale_pc = length_scale_pc
        self._gamma = stats.gamma(a=3, scale=length_scale_pc)
        # Sensible physical bounds for callers that want them (e.g. the
        # point-estimate least-squares solve) -- not used by .ppf itself.
        self.low = 1e-3   # mas, ~1 Mpc: astrophysically irrelevant beyond this
        self.high = 1000.0  # mas, ~1 pc: nearer than any known star

    def ppf(self, u):
        r_pc = self._gamma.ppf(1.0 - np.asarray(u))
        return 1000.0 / r_pc
