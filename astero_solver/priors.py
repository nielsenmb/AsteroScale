"""Small hand-written distributions used throughout Solver, as priors and
(for Normal/TruncatedNormal) as likelihood terms.

Anything with a .ppf(u) method (mapping the unit interval to the
parameter's support) works as a prior; anything with .logpdf(x) works as a
likelihood term. Frozen scipy.stats distributions satisfy both and remain
completely fine to pass in yourself (Solver(priors={...}) and given={...}
both accept them via duck typing) -- these classes exist because
scipy.stats' generic distribution-object machinery is surprisingly slow
for the scalar, one-value-at-a-time calls this package's sampler actually
makes (see the README's JAX/performance section for the benchmark:
scipy.stats.norm.ppf managed ~23,500 calls/s in that regime, vs. ~5.4M/s
for the equivalent scipy.special call used directly, as below). Nothing
here reflects scipy.stats being unpolished -- its overhead is the price of
a very general, batch-oriented interface that isn't the shape of dynesty's
workload.
"""
import numpy as np
import scipy.special as sp


def _std_normal_cdf(z):
    return 0.5 * (1.0 + sp.erf(z / np.sqrt(2.0)))


class Normal:
    """Gaussian, via scipy.special.erfinv/erf directly rather than
    scipy.stats.norm."""

    def __init__(self, loc=0.0, scale=1.0):
        self.loc = loc
        self.scale = scale
        self.low = -np.inf
        self.high = np.inf

    def ppf(self, u):
        u = np.asarray(u)
        return self.loc + self.scale * np.sqrt(2.0) * sp.erfinv(2.0 * u - 1.0)

    def logpdf(self, x):
        x = np.asarray(x)
        z = (x - self.loc) / self.scale
        return -0.5 * z**2 - np.log(self.scale) - 0.5 * np.log(2.0 * np.pi)

    def pdf(self, x):
        return np.exp(self.logpdf(x))


class Uniform:
    """Uniform(loc, loc + scale) -- matches scipy.stats' loc/scale
    convention (not (lo, hi))."""

    def __init__(self, loc=0.0, scale=1.0):
        self.loc = loc
        self.scale = scale
        self.low = loc
        self.high = loc + scale

    def ppf(self, u):
        u = np.asarray(u)
        return self.loc + u * self.scale

    def logpdf(self, x):
        x = np.asarray(x)
        inside = (x >= self.loc) & (x <= self.loc + self.scale)
        return np.where(inside, -np.log(self.scale), -np.inf)

    def pdf(self, x):
        return np.exp(self.logpdf(x))


class Exponential:
    """Exponential with the given scale (1/rate) -- scipy.stats convention."""

    def __init__(self, scale=1.0):
        self.scale = scale
        self.low = 0.0
        self.high = np.inf

    def ppf(self, u):
        u = np.asarray(u)
        return -self.scale * np.log1p(-u)

    def logpdf(self, x):
        x = np.asarray(x)
        return np.where(x >= 0, -x / self.scale - np.log(self.scale), -np.inf)

    def pdf(self, x):
        return np.exp(self.logpdf(x))


class TruncatedNormal:
    """Normal(loc, scale) truncated to [low, high] -- the standard
    truncated-Gaussian inverse-CDF trick (invert the CDF restricted to the
    truncation window), using scipy.special's erf/erfinv directly rather
    than scipy.stats.truncnorm.
    """

    def __init__(self, loc, scale, low, high):
        self.loc, self.scale, self.low, self.high = loc, scale, low, high
        self._cdf_low = _std_normal_cdf((low - loc) / scale)
        self._cdf_high = _std_normal_cdf((high - loc) / scale)

    def ppf(self, u):
        u = np.asarray(u)
        p = self._cdf_low + u * (self._cdf_high - self._cdf_low)
        z = np.sqrt(2.0) * sp.erfinv(2.0 * p - 1.0)
        return self.loc + self.scale * z

    def logpdf(self, x):
        x = np.asarray(x)
        z = (x - self.loc) / self.scale
        norm_const = self._cdf_high - self._cdf_low
        base = (-0.5 * z**2 - np.log(self.scale) - 0.5 * np.log(2.0 * np.pi)
                - np.log(norm_const))
        inside = (x >= self.low) & (x <= self.high)
        return np.where(inside, base, -np.inf)

    def pdf(self, x):
        return np.exp(self.logpdf(x))


class TruncatedPowerLaw:
    """pdf(x) ~ x^-alpha on [low, high], via the analytic inverse CDF.

    alpha=2.35 is the Salpeter (1955) IMF slope -- a reasonable default for
    a stellar mass prior, reflecting that lower-mass stars are genuinely
    far more common than higher-mass ones (unlike a flat/uniform prior,
    which overweights high mass relative to any real stellar population).
    alpha=1 is the log-uniform special case, handled separately below --
    this is also what backs the R (radius) default prior.

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

    r ~ Gamma(shape=3, scale=L) has this density. Gamma's ppf isn't
    elementary, but scipy.special.gammaincinv (the regularized incomplete
    gamma function's inverse) gives it directly and much faster than
    going through scipy.stats.gamma's frozen-distribution wrapper. Since
    parallax = 1000/r is a decreasing function of r,
    ppf_plx(u) = 1000 / ppf_r(1-u).
    """

    def __init__(self, length_scale_pc=1350.0):
        self.length_scale_pc = length_scale_pc
        # Sensible physical bounds for callers that want them (e.g. the
        # point-estimate least-squares solve) -- not used by .ppf itself.
        self.low = 1e-3   # mas, ~1 Mpc: astrophysically irrelevant beyond this
        self.high = 1000.0  # mas, ~1 pc: nearer than any known star

    def ppf(self, u):
        u = np.asarray(u)
        r_pc = self.length_scale_pc * sp.gammaincinv(3.0, 1.0 - u)
        return 1000.0 / r_pc
