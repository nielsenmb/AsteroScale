from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
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
        """Set mean and median for the distribution"""
        x = jnp.linspace(self.ppf(1e-6), self.ppf(1 - 1e-6), 1000)

        self.mean = jnp.trapezoid(x * jnp.array([self.pdf(_x) for _x in x]), x)

        self.median = self.ppf(0.5)

    @partial(jax.jit, static_argnums=(0,))
    def _transformx(self, x):
        """Translates and scales the input x to the unit interval according
        to the loc and scale parameters."""
        return (x - self.loc) / self.scale

    @partial(jax.jit, static_argnums=(0,))
    def _inverse_transform(self, x):
        """Invert scaling on input."""
        return x * self.scale + self.loc

    @partial(jax.jit, static_argnums=(0,))
    def pdf(self, x, norm=True):
        """Return PDF. Normalized to unit integral by default."""
        _x = self._transformx(x)

        T = jax.lax.lt(_x, 0.0) | jax.lax.lt(1.0, _x)

        y = jax.lax.cond(
            T, lambda: 0.0, lambda: _x**self.am1 * (1 - _x) ** self.bm1
        )

        y = jax.lax.cond(norm, lambda: y * self.fac, lambda: y)

        return y

    @partial(jax.jit, static_argnums=(0,))
    def logpdf(self, x, norm=True):
        """Return log-PDF. Normalized to unit integral by default."""
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
        _x = self._transformx(x)

        y = jsp.betainc(self.a, self.b, _x)

        y = y.at[_x <= 0].set(0)

        y = y.at[_x >= 1].set(1)

        return y

    @partial(jax.jit, static_argnums=(0,))
    def ppf(self, y):
        _x = self.betaincinv(self.a, self.b, y)

        x = self._inverse_transform(_x)

        return x

    @partial(jax.jit, static_argnums=(0,))
    def update_x(self, x, a, b, p, a1, b1, afac):
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
        return jnp.where(
            jnp.logical_and(a >= 1.0, b >= 1.0),
            self.func_1(a, b, p),
            self.func_2(a, b, p),
        )

    @partial(jax.jit, static_argnums=(0,))
    def betaincinv(self, a, b, p):
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
        u = self.rng.uniform(0, 1)
        return self.ppf(u)

    def _set_stdatt(self):
        x = jnp.linspace(self.ppf(1e-6), self.ppf(1 - 1e-6), 1000)
        self.mean = jnp.trapezoid(x * jnp.array([self.pdf(_x) for _x in x]), x)
        self.median = self.ppf(0.5)


class uniform:
    def __init__(self, loc=0, scale=1, rng=None):
        """Uniform distribution. Emulates scipy.stats, but jaxed.

        Parameters
        ----------
        loc : float
            Left side of the uniform distribution
        scale : float
            Width of the uniform distribution, right side is loc+scale
        """

        self.__dict__.update(
            (k, v) for k, v in locals().items() if k not in ["self"]
        )

        if self.rng is None:
            self.rng = np.random.Generator(PCG64(42))

        self.a = self.loc
        self.b = self.loc + self.scale
        self.mean = 0.5 * (self.a + self.b)

        self._set_stdatt()

    def rv(self):
        u = self.rng.uniform(0, 1)
        return self.ppf(u)

    def _set_stdatt(self):
        x = jnp.linspace(self.ppf(1e-6), self.ppf(1 - 1e-6), 1000)
        self.mean = jnp.trapezoid(x * jnp.array([self.pdf(_x) for _x in x]), x)
        self.median = self.ppf(0.5)

    @partial(jax.jit, static_argnums=(0,))
    def pdf(self, x):
        T = jax.lax.lt(x, self.a) | jax.lax.lt(self.b, x)
        y = jax.lax.cond(T, lambda: 0.0, lambda: 1.0 / self.scale)
        return y

    @partial(jax.jit, static_argnums=(0,))
    def logpdf(self, x):
        T = jax.lax.lt(x, self.a) | jax.lax.lt(self.b, x)
        y = jax.lax.cond(T, lambda: -jnp.inf, lambda: -jnp.log(self.scale))
        return y

    @partial(jax.jit, static_argnums=(0,))
    def cdf(self, x):
        y = (x - self.a) / (self.b - self.a)
        return jnp.clip(y, 0.0, 1.0)

    @partial(jax.jit, static_argnums=(0,))
    def ppf(self, y):
        y = jnp.array(y)
        x = y * (self.b - self.a) + self.a
        return x


class normal:
    def __init__(self, loc=0, scale=1, rng=None):
        """Normal distribution class.

        Parameters
        ----------
        loc : float
            The mean of the normal distribution.
        scale : float
            The standard deviation of the normal distribution.
        """
        self.__dict__.update(
            (k, v) for k, v in locals().items() if k not in ["self"]
        )

        if self.rng is None:
            self.rng = np.random.Generator(PCG64(42))

        self.fac = -0.5 / self.scale**2
        self.norm = 1 / (jnp.sqrt(2 * jnp.pi) * self.scale)
        self.lognorm = jnp.log(self.norm)

        self._set_stdatt()

    def rv(self):
        u = self.rng.uniform(0, 1)
        return self.ppf(u)

    def _set_stdatt(self):
        x = jnp.linspace(self.ppf(1e-6), self.ppf(1 - 1e-6), 1000)
        self.mean = jnp.trapezoid(x * jnp.array([self.pdf(_x) for _x in x]), x)
        self.median = self.ppf(0.5)

    @partial(jax.jit, static_argnums=(0,))
    def pdf(self, x, norm=True):
        y = jnp.exp(self.fac * (x - self.loc) ** 2)
        Y = jax.lax.cond(norm, lambda y: y * self.norm, lambda y: y, y)
        return Y

    @partial(jax.jit, static_argnums=(0,))
    def logpdf(self, x, norm=True):
        y = self.fac * (x - self.loc) ** 2
        Y = jax.lax.cond(norm, lambda y: y + self.lognorm, lambda y: y, y)
        return Y

    @partial(jax.jit, static_argnums=(0,))
    def cdf(self, x):
        y = 0.5 * (1 + jsp.erf((x - self.loc) / (jnp.sqrt(2) * self.scale)))
        return y

    @partial(jax.jit, static_argnums=(0,))
    def ppf(self, y):
        x = self.loc + self.scale * jnp.sqrt(2) * jsp.erfinv(2 * y - 1)
        return x


class truncsine:
    def __init__(self, rng=None):
        """Sine truncated between 0 and pi/2 (e.g. for isotropic inclination
        angle priors)."""
        if rng is None:
            self.rng = np.random.Generator(PCG64(42))
        else:
            self.rng = rng

        self._set_stdatt()

    def rv(self):
        u = self.rng.uniform(0, 1)
        return self.ppf(u)

    def _set_stdatt(self):
        x = jnp.linspace(self.ppf(1e-6), self.ppf(1 - 1e-6), 1000)
        self.mean = jnp.trapezoid(x * jnp.array([self.pdf(_x) for _x in x]), x)
        self.median = self.ppf(0.5)

    @partial(jax.jit, static_argnums=(0,))
    def pdf(self, x):
        T = jax.lax.lt(x, 0.0) | jax.lax.lt(jnp.pi / 2, x)
        y = jax.lax.cond(T, lambda: 0.0, lambda: jnp.sin(x))
        return y

    @partial(jax.jit, static_argnums=(0,))
    def logpdf(self, x):
        T = jax.lax.lt(x, 0.0) | jax.lax.lt(jnp.pi / 2, x)
        y = jax.lax.cond(T, lambda: -jnp.inf, lambda: jnp.log(jnp.sin(x)))
        return y

    @partial(jax.jit, static_argnums=(0,))
    def cdf(self, x):
        x_clamped = jnp.clip(x, 0.0, jnp.pi / 2)
        y = 1 + jnp.cos(x_clamped - jnp.pi)
        return jnp.clip(y, 0.0, 1)

    @partial(jax.jit, static_argnums=(0,))
    def ppf(self, y):
        x = jnp.arccos(1 - y)
        return x
