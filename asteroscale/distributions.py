from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import scipy.special as sp

import numpy as np
from numpy.random import PCG64


class beta:
    def __init__(self, a=1, b=1, loc=0, scale=1, rng=None):
        """beta distribution class
        Create instances a probability density which follows the beta
        distribution.

        Parameters
        ----------
        a : float
            The first shape parameter of the beta distribution.
        b : float
            The second shape parameter of the beta distribution.
        loc : float
            The lower limit of the beta distribution. The probability at this
            limit and below is 0.
        scale : float
            The width of the beta distribution. Effectively sets the upper
            bound for the distribution, which is loc+scale.
        rng : numpy.random.Generator, optional
            Random-number generator used by :meth:`rv`.
        """

        # Turn init args into attributes
        self.__dict__.update(
            (k, v) for k, v in locals().items() if k not in ["self"]
        )

        if self.rng is None:
            self.rng = np.random.Generator(PCG64(42))

        self.logfac = (
            jsp.gammaln(self.a + self.b)
            - jsp.gammaln(self.a)
            - jsp.gammaln(self.b)
            - jnp.log(self.scale)
        )

        self.fac = jnp.exp(self.logfac)

        self.am1 = self.a - 1

        self.bm1 = self.b - 1

        self._set_stdatt()

    def rv(self):
        """Draw random variable from distribution

        Returns
        -------
        x : float
            Random variable drawn from the distribution
        """

        u = self.rng.uniform(0, 1)

        x = self.ppf(u)

        return x

    def _set_stdatt(self):
        """Set numerical mean and median attributes."""
        x = jnp.linspace(self.ppf(1e-6), self.ppf(1 - 1e-6), 1000)

        self.mean = jnp.trapezoid(x * jnp.array([self.pdf(_x) for _x in x]), x)

        self.median = self.ppf(0.5)

    @partial(jax.jit, static_argnums=(0,))
    def _transformx(self, x):
        """Transform values to the unit interval.

        Parameters
        ----------
        x : float or array-like
            Values in the configured location-scale coordinates.

        Returns
        -------
        float or ndarray
            Values transformed to the standard beta coordinates.
        """
        return (x - self.loc) / self.scale

    @partial(jax.jit, static_argnums=(0,))
    def _inverse_transform(self, x):
        """Transform standard beta values to configured coordinates.

        Parameters
        ----------
        x : float or array-like
            Values in standard beta coordinates.

        Returns
        -------
        float or ndarray
            Values in the configured location-scale coordinates.
        """
        return x * self.scale + self.loc

    @partial(jax.jit, static_argnums=(0,))
    def pdf(self, x, norm=True):
        """Evaluate the beta probability density.

        Parameters
        ----------
        x : float or array-like
            Evaluation points.
        norm : bool, default=True
            Apply the normalization constant.

        Returns
        -------
        float or ndarray
            Probability density.
        """
        _x = self._transformx(x)

        T = jax.lax.lt(_x, 0.0) | jax.lax.lt(1.0, _x)

        y = jax.lax.cond(
            T, lambda: 0.0, lambda: _x**self.am1 * (1 - _x) ** self.bm1
        )

        y = jax.lax.cond(norm, lambda: y * self.fac, lambda: y)

        return y

    @partial(jax.jit, static_argnums=(0,))
    def logpdf(self, x, norm=True):
        """Evaluate the beta log-probability density.

        Parameters
        ----------
        x : float or array-like
            Evaluation points.
        norm : bool, default=True
            Apply the normalization constant.

        Returns
        -------
        float or ndarray
            Log-probability density.
        """
        x = jnp.array(x)

        _x = self._transformx(x)

        T = jax.lax.lt(_x, 0.0) | jax.lax.lt(1.0, _x)

        y = jax.lax.cond(
            T,
            lambda: -jnp.inf,
            lambda: self.am1 * jnp.log(_x) + self.bm1 * jnp.log(1 - _x),
        )

        y = jax.lax.cond(norm, lambda: y + self.logfac, lambda: y)

        return y

    def cdf(self, x):
        """Evaluate the beta cumulative distribution function.

        Parameters
        ----------
        x : float or array-like
            Evaluation points.

        Returns
        -------
        float or ndarray
            Cumulative probability.
        """
        _x = self._transformx(x)

        y = jsp.betainc(self.a, self.b, _x)

        y = y.at[_x <= 0].set(0)

        y = y.at[_x >= 1].set(1)

        return y

    @partial(jax.jit, static_argnums=(0,))
    def ppf(self, y):
        """Evaluate the beta quantile function.

        Parameters
        ----------
        y : float or array-like
            Cumulative probabilities in [0, 1].

        Returns
        -------
        float or ndarray
            Distribution quantiles.
        """
        _x = self.betaincinv(self.a, self.b, y)

        x = self._inverse_transform(_x)

        return x

    @partial(jax.jit, static_argnums=(0,))
    def update_x(self, x, a, b, p, a1, b1, afac):
        """Apply one Newton-Halley update for the inverse beta CDF.

        Parameters
        ----------
        x : float or array-like
            Current quantile estimate.
        a, b : float
            Beta shape parameters.
        p : float or array-like
            Target cumulative probability.
        a1, b1 : float
            Shape parameters minus one.
        afac : float
            Negative log beta normalization.

        Returns
        -------
        tuple
            Updated estimate and step size.
        """
        err = jsp.betainc(a, b, x) - p
        t = jnp.exp(a1 * jnp.log(x) + b1 * jnp.log(1.0 - x) + afac)
        u = err / t
        tmp = u * (a1 / x - b1 / (1.0 - x))
        t = u / (1.0 - 0.5 * jnp.clip(tmp, max=1.0))
        x -= t
        x = jnp.where(x <= 0.0, 0.5 * (x + t), x)
        x = jnp.where(x >= 1.0, 0.5 * (x + t + 1.0), x)

        return x, t

    @partial(jax.jit, static_argnums=(0,))
    def func_1(self, a, b, p):
        """Initialize inverse-CDF iteration when both shapes exceed one.

        Parameters
        ----------
        a, b : float
            Beta shape parameters.
        p : float or array-like
            Target cumulative probability.

        Returns
        -------
        float or ndarray
            Initial quantile estimate.
        """
        pp = jnp.where(p < 0.5, p, 1.0 - p)
        t = jnp.sqrt(-2.0 * jnp.log(pp))
        x = (2.30753 + t * 0.27061) / (1.0 + t * (0.99229 + t * 0.04481)) - t
        x = jnp.where(p < 0.5, -x, x)
        al = (jnp.power(x, 2) - 3.0) / 6.0
        h = 2.0 / (1.0 / (2.0 * a - 1.0) + 1.0 / (2.0 * b - 1.0))
        w = (x * jnp.sqrt(al + h) / h) - (
            1.0 / (2.0 * b - 1) - 1.0 / (2.0 * a - 1.0)
        ) * (al + 5.0 / 6.0 - 2.0 / (3.0 * h))
        return a / (a + b * jnp.exp(2.0 * w))

    @partial(jax.jit, static_argnums=(0,))
    def func_2(self, a, b, p):
        """Initialize inverse-CDF iteration for small shape parameters.

        Parameters
        ----------
        a, b : float
            Beta shape parameters.
        p : float or array-like
            Target cumulative probability.

        Returns
        -------
        float or ndarray
            Initial quantile estimate.
        """
        lna = jnp.log(a / (a + b))
        lnb = jnp.log(b / (a + b))
        t = jnp.exp(a * lna) / a
        u = jnp.exp(b * lnb) / b
        w = t + u

        return jnp.where(
            p < t / w,
            jnp.power(a * w * p, 1.0 / a),
            1.0 - jnp.power(b * w * (1.0 - p), 1.0 / b),
        )

    @partial(jax.jit, static_argnums=(0,))
    def compute_x(self, p, a, b):
        """Select an initial inverse-CDF estimate.

        Parameters
        ----------
        p : float or array-like
            Target cumulative probability.
        a, b : float
            Beta shape parameters.

        Returns
        -------
        float or ndarray
            Initial quantile estimate.
        """
        return jnp.where(
            jnp.logical_and(a >= 1.0, b >= 1.0),
            self.func_1(a, b, p),
            self.func_2(a, b, p),
        )

    @partial(jax.jit, static_argnums=(0,))
    def betaincinv(self, a, b, p):
        """Invert the regularized incomplete beta function.

        Parameters
        ----------
        a, b : float
            Beta shape parameters.
        p : float or array-like
            Target cumulative probability.

        Returns
        -------
        float or ndarray
            Standard beta quantile.
        """
        a1 = a - 1.0
        b1 = b - 1.0

        ERROR = 1e-8

        p = jnp.clip(p, min=0.0, max=1.0)

        x = jnp.where(
            jnp.logical_or(p <= 0.0, p >= 1.0), p, self.compute_x(p, a, b)
        )

        afac = -jsp.betaln(a, b)
        stop = jnp.logical_or(
            jnp.isclose(x, 0.0, atol=1e-8), jnp.isclose(x, 1.0, atol=1e-8)
        )
        for _ in range(10):
            x_new, t = self.update_x(x, a, b, p, a1, b1, afac)
            x = jnp.where(stop, x, x_new)
            stop = jnp.where(
                jnp.logical_or(jnp.abs(t) < ERROR * x, stop), True, False
            )

        return x


class distribution:
    def __init__(self, ppf, pdf, logpdf, cdf, rng=None):
        """Generic distribution object

        Wraps a set of ppf/pdf/logpdf/cdf callables (e.g. from a KDE fit or
        any other source) with the same interface as the other classes here.

        Parameters
        ----------
        ppf, pdf, logpdf, cdf : callable
            Quantile, density, log-density, and cumulative functions.
        rng : numpy.random.Generator, optional
            Random-number generator used by :meth:`rv`.
        """

        if rng is None:
            self.rng = np.random.Generator(PCG64(42))
        else:
            self.rng = rng

        self.pdf = pdf
        self.ppf = ppf
        self.logpdf = logpdf
        self.cdf = cdf

        self._set_stdatt()

    def rv(self):
        """Draw one random variate.

        Returns
        -------
        float
            Random draw from the wrapped distribution.
        """
        u = self.rng.uniform(0, 1)
        return self.ppf(u)

    def _set_stdatt(self):
        """Set numerical mean and median attributes."""
        x = jnp.linspace(self.ppf(1e-6), self.ppf(1 - 1e-6), 1000)
        self.mean = jnp.trapezoid(x * jnp.array([self.pdf(_x) for _x in x]), x)
        self.median = self.ppf(0.5)


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
        """Initialize a normalized truncated power law.

        Parameters
        ----------
        alpha : float
            Positive exponent in ``p(x) proportional to x**(-alpha)``.
        low, high : float
            Inclusive lower and upper truncation limits.
        """
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
        """Evaluate the quantile function.

        Parameters
        ----------
        u : float or array-like
            Cumulative probability in [0, 1].

        Returns
        -------
        float or ndarray
            Power-law quantile.
        """
        u = np.asarray(u)
        if self._is_loguniform:
            return np.exp(self._log_low + u * (self._log_high - self._log_low))
        return (self._low_p + u * (self._high_p - self._low_p)) ** (1.0 / self._p)

    def pdf(self, x):
        """Evaluate the normalized probability density.

        Parameters
        ----------
        x : float or array-like
            Evaluation points.

        Returns
        -------
        float or ndarray
            Probability density, zero outside the truncation interval.
        """
        x = np.asarray(x)
        if self._is_loguniform:
            norm = 1.0 / (x * (self._log_high - self._log_low))
        else:
            norm = self._p * x ** (-self.alpha) / (self._high_p - self._low_p)
        return np.where((x >= self.low) & (x <= self.high), norm, 0.0)
    
def _std_normal_cdf(z):
    """Evaluate the standard-normal cumulative distribution function.

    Parameters
    ----------
    z : float or array-like
        Standardized evaluation points.

    Returns
    -------
    float or ndarray
        Cumulative probability.
    """
    return 0.5 * (1.0 + sp.erf(z / np.sqrt(2.0)))


class TruncatedNormal:
    """Normal(loc, scale) truncated to [low, high] -- the standard
    truncated-Gaussian inverse-CDF trick (invert the CDF restricted to the
    truncation window), using scipy.special's erf/erfinv directly rather
    than scipy.stats.truncnorm.
    """

    def __init__(self, loc, scale, low, high):
        """Initialize a truncated normal distribution.

        Parameters
        ----------
        loc : float
            Mean of the untruncated normal distribution.
        scale : float
            Standard deviation of the untruncated distribution.
        low, high : float
            Inclusive truncation limits.
        """
        self.loc, self.scale, self.low, self.high = loc, scale, low, high
        self._cdf_low = _std_normal_cdf((low - loc) / scale)
        self._cdf_high = _std_normal_cdf((high - loc) / scale)

    def ppf(self, u):
        """Evaluate the truncated-normal quantile function.

        Parameters
        ----------
        u : float or array-like
            Cumulative probability in [0, 1].

        Returns
        -------
        float or ndarray
            Distribution quantile.
        """
        u = np.asarray(u)
        p = self._cdf_low + u * (self._cdf_high - self._cdf_low)
        z = np.sqrt(2.0) * sp.erfinv(2.0 * p - 1.0)
        return self.loc + self.scale * z

    def logpdf(self, x):
        """Evaluate the truncated-normal log-density.

        Parameters
        ----------
        x : float or array-like
            Evaluation points.

        Returns
        -------
        float or ndarray
            Log-density, or negative infinity outside the limits.
        """
        x = np.asarray(x)
        z = (x - self.loc) / self.scale
        norm_const = self._cdf_high - self._cdf_low
        base = (-0.5 * z**2 - np.log(self.scale) - 0.5 * np.log(2.0 * np.pi)
                - np.log(norm_const))
        inside = (x >= self.low) & (x <= self.high)
        return np.where(inside, base, -np.inf)

    def pdf(self, x):
        """Evaluate the truncated-normal probability density.

        Parameters
        ----------
        x : float or array-like
            Evaluation points.

        Returns
        -------
        float or ndarray
            Probability density.
        """
        return np.exp(self.logpdf(x))
    
class Exponential:
    """Exponential with the given scale (1/rate) -- scipy.stats convention."""

    def __init__(self, scale=1.0):
        """Initialize an exponential distribution.

        Parameters
        ----------
        scale : float, default=1.0
            Distribution scale, equal to the inverse rate.
        """
        self.scale = scale
        self.low = 0.0
        self.high = np.inf

    def ppf(self, u):
        """Evaluate the exponential quantile function.

        Parameters
        ----------
        u : float or array-like
            Cumulative probability in [0, 1].

        Returns
        -------
        float or ndarray
            Distribution quantile.
        """
        u = np.asarray(u)
        return -self.scale * np.log1p(-u)

    def logpdf(self, x):
        """Evaluate the exponential log-density.

        Parameters
        ----------
        x : float or array-like
            Evaluation points.

        Returns
        -------
        float or ndarray
            Log-density, or negative infinity for negative values.
        """
        x = np.asarray(x)
        return np.where(x >= 0, -x / self.scale - np.log(self.scale), -np.inf)

    def pdf(self, x):
        """Evaluate the exponential probability density.

        Parameters
        ----------
        x : float or array-like
            Evaluation points.

        Returns
        -------
        float or ndarray
            Probability density.
        """
        return np.exp(self.logpdf(x))
    
class uniform:
    """Uniform(loc, loc + scale) -- matches scipy.stats' loc/scale
    convention (not (lo, hi))."""

    def __init__(self, loc=0.0, scale=1.0):
        """Initialize a uniform distribution.

        Parameters
        ----------
        loc : float, default=0.0
            Inclusive lower limit.
        scale : float, default=1.0
            Width of the distribution.
        """
        self.loc = loc
        self.scale = scale
        self.low = loc
        self.high = loc + scale

    def ppf(self, u):
        """Evaluate the uniform quantile function.

        Parameters
        ----------
        u : float or array-like
            Cumulative probability in [0, 1].

        Returns
        -------
        float or ndarray
            Distribution quantile.
        """
        u = np.asarray(u)
        return self.loc + u * self.scale

    def logpdf(self, x):
        """Evaluate the uniform log-density.

        Parameters
        ----------
        x : float or array-like
            Evaluation points.

        Returns
        -------
        float or ndarray
            Log-density, or negative infinity outside the support.
        """
        x = np.asarray(x)
        inside = (x >= self.loc) & (x <= self.loc + self.scale)
        return np.where(inside, -np.log(self.scale), -np.inf)

    def pdf(self, x):
        """Evaluate the uniform probability density.

        Parameters
        ----------
        x : float or array-like
            Evaluation points.

        Returns
        -------
        float or ndarray
            Probability density.
        """
        return np.exp(self.logpdf(x))

class normal:
    """Gaussian, via scipy.special.erfinv/erf directly rather than
    scipy.stats.norm."""

    def __init__(self, loc=0.0, scale=1.0):
        """Initialize a normal distribution.

        Parameters
        ----------
        loc : float, default=0.0
            Distribution mean.
        scale : float, default=1.0
            Distribution standard deviation.
        """
        self.loc = loc
        self.scale = scale
        self.low = -np.inf
        self.high = np.inf

    def ppf(self, u):
        """Evaluate the normal quantile function.

        Parameters
        ----------
        u : float or array-like
            Cumulative probability in [0, 1].

        Returns
        -------
        float or ndarray
            Distribution quantile.
        """
        u = np.asarray(u)
        return self.loc + self.scale * np.sqrt(2.0) * sp.erfinv(2.0 * u - 1.0)

    def logpdf(self, x):
        """Evaluate the normal log-density.

        Parameters
        ----------
        x : float or array-like
            Evaluation points.

        Returns
        -------
        float or ndarray
            Log-probability density.
        """
        x = np.asarray(x)
        z = (x - self.loc) / self.scale
        return -0.5 * z**2 - np.log(self.scale) - 0.5 * np.log(2.0 * np.pi)

    def pdf(self, x):
        """Evaluate the normal probability density.

        Parameters
        ----------
        x : float or array-like
            Evaluation points.

        Returns
        -------
        float or ndarray
            Probability density.
        """
        return np.exp(self.logpdf(x))

 
