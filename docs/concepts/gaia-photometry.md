# Gaia photometry and bolometric corrections

A stellar model or scaling relation usually predicts **bolometric**
luminosity: the energy emitted over all wavelengths. Gaia measures only the
light transmitted by a particular passband. A bolometric correction connects
the two:

$$
M_X = M_{\rm bol} - BC_X,
$$

where $X$ is the Gaia DR3 G, BP, or RP passband. The correction is not a single
constant because the fraction of a star's light falling in a passband depends
on its spectrum. AsteroScale therefore evaluates

$$
BC_X = BC_X(T_{\rm eff}, \log g, [\mathrm{Fe/H}]).
$$

The apparent magnitude also includes distance and extinction:

$$
m_X = M_X + 5\log_{10}(d/{\rm pc}) - 5 + A_X.
$$

Here `A_G` is a fundamental AsteroScale input. The BP and RP extinctions are
calculated from the same synthetic spectra rather than from fixed ratios, so
they also depend on the stellar parameters and on the amount of extinction.

## MARCS-based Gaia DR3 table

AsteroScale packages a compact table derived from the MARCS-based synthetic
photometry of
[Casagrande & VandenBerg (2014)](https://ui.adsabs.harvard.edu/abs/2014MNRAS.444..392C/abstract),
[Casagrande & VandenBerg (2018a)](https://ui.adsabs.harvard.edu/abs/2018MNRAS.475.5023C/abstract),
and
[Casagrande & VandenBerg (2018b)](https://ui.adsabs.harvard.edu/abs/2018MNRAS.479L.102C/abstract).
The full atmosphere tables are regularized once during package development;
runtime evaluation is then a small multilinear interpolation, so using Gaia
photometry adds little cost to a sampler.

The packaged table:

- uses the Gaia DR3 G, BP, and RP passbands and Vega zero-points;
- assumes $R_V=3.1$;
- uses the source grid's standard alpha-enhancement prescription;
- covers $4000\le T_{\rm eff}/{\rm K}\le7000$,
  $1.5\le\log g\le5.0$, $-1.0\le[\mathrm{Fe/H}]\le0.5$, and
  $0\le A_G/{\rm mag}\le1.5$;
- returns `nan` outside that domain rather than extrapolating; and
- agrees with the source interpolation to within 0.005 mag in the package
  validation tests.

The source tables adopt $M_{\rm bol,\odot}=4.75$. AsteroScale shifts the
corrections by $-0.01$ mag so that they are consistent with its
$M_{\rm bol,\odot}=4.74$ convention. Without this adjustment, every predicted
Gaia magnitude would inherit a 0.01 mag zero-point offset.

## What accuracy should I assume?

The interpolation error is only one part of the uncertainty. Atmosphere
models, passbands, extinction laws, abundance patterns, and measured stellar
parameters can introduce larger systematic errors. AsteroScale therefore adds
a default 0.02 mag model-error floor in quadrature to Gaussian magnitude and
colour constraints:

```python
import asteroscale as ast

solver = ast.Solver(photometric_error_floor=0.02)
```

Set the value to zero to disable it. The floor applies to `(value,
uncertainty)` inputs; a custom distribution is left unchanged because its
scale cannot be modified generically.

This simple floor does not capture correlated atmosphere errors, and
AsteroScale does not vary $R_V$. A result based on millimagnitude Gaia
uncertainties should therefore still not be interpreted as
millimagnitude-accurate synthetic photometry.

Use an external extinction estimate where possible and inspect the
`Gaia_photometry` entry in the validity report. For full multi-band SED fitting,
a dedicated SED package that marginalizes over extinction-law and atmosphere
choices is more appropriate.
