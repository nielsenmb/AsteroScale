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
from .distributions import uniform, TruncatedPowerLaw, Exponential, TruncatedNormal
 
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

DEFAULT_PRIORS = {
    "M": TruncatedPowerLaw(alpha=2.35, low=0.5, high=3.0),  # Salpeter IMF slope
    "R": TruncatedPowerLaw(alpha=1.0, low=0.5, high=20.0),  # log-uniform: R spans >1 decade
    "Teff": uniform(loc=4000.0, scale=3000.0),   # flat 4000-7000 K, no strong prior
    "plx": ParallaxPrior(length_scale_pc=1350.0),  # Bailer-Jones distance prior
    "A_G": Exponential(scale=0.2),           # most stars nearby have low extinction
    "FeH": TruncatedNormal(                  # solar-neighborhood metallicity spread,
        loc=-0.1, scale=0.25, low=-1.0, high=0.5,  # truncated to the range the dnu
    ),                                        # metallicity correction is calibrated over
}