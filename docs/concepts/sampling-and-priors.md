# Sampling, uncertainties, and priors

An exact calculation and an uncertain calculation answer slightly different
questions. When every supplied value is a plain number, AsteroScale evaluates
or inverts the relations and returns scalar point estimates. When any supplied
value has an uncertainty, AsteroScale uses Dynesty nested sampling and returns
arrays of samples. A histogram of an output array approximates its probability
distribution.

```python
import asteroscale as ast

given = {
    "Teff": (5777, 50),
    "FeH": (0.0, 0.05),
    "numax": (3090, 30),
    "dnu": (135.1, 1.0),
}
solver = ast.Solver(seed=42)
samples = solver.solve(given=given, want=["M", "R"])
ast.summarize(samples)
```

Each tuple is interpreted as `(mean, one-sigma uncertainty)` for an independent
Gaussian distribution. Plain numbers are treated as exactly known. For
asymmetric or non-Gaussian information, supply a distribution object with the
appropriate `ppf` or `logpdf` method.

AsteroScale uses [Baldr](https://github.com/nielsenmb/Baldr) for its default
one-dimensional distributions. Custom priors can use the same constructors;
select the NumPy backend because AsteroScale may evaluate a prior for either
one value or an array:

```python
from baldr import TruncatedNormal

mass_prior = TruncatedNormal(
    loc=1.05,
    scale=0.10,
    low=0.7,
    high=1.4,
    backend="numpy",
)
solver = ast.Solver(priors={"M": mass_prior})
```

Frozen `scipy.stats` distributions and other objects exposing the required
methods remain supported.

## What a prior does

A prior describes plausible values before the current measurements are
applied. AsteroScale's default priors are broad distributions for field stars,
not a stellar-evolution model. They are independent, so weak data can admit
combinations of mass, radius, and temperature that are uncommon on a real
Hertzsprung--Russell diagram. A narrow posterior is therefore only convincing
when the supplied measurements genuinely constrain it.

The likelihood describes how probable the measurements are for a trial star.
Combining prior and likelihood gives a posterior. This distinction matters for
uncertain fundamental inputs.

## Two meanings of an uncertain fundamental input

Set `input_mode` according to what an input distribution represents:

- `input_mode="propagate"` (the default) treats an uncertain fundamental such
  as `Teff=(5772, 50)` as your current knowledge of that parameter. It replaces
  the default temperature prior. This calculator-style mode is useful when the
  input is already a catalogue posterior or when you simply want to propagate
  quoted errors through the relations.
- `input_mode="likelihood"` treats the tuple as a new measurement likelihood.
  The field-star prior remains in the calculation and is updated by that
  measurement. Use this for a Bayesian analysis when the tuple describes the
  measurement process rather than an already inferred posterior.

```python
posterior = ast.Solver(input_mode="likelihood", seed=42).solve(
    given,
    want=["M", "R"],
)
```

The sampling preset and `input_mode` are independent: `precise` does **not**
automatically select `likelihood`. Plain scalar inputs remain fixed in both
modes. Derived observables such as `numax` and `dnu` are always likelihood
constraints because the forward model predicts them from the fundamentals.

Avoid applying a population prior twice. If a catalogue value is already a
posterior obtained using a population prior, use `propagate` or reconstruct its
measurement likelihood before using `likelihood`.

## Correlated stellar-population prior

For likelihood-based inference, AsteroScale can replace the independent
one-dimensional priors on mass, radius, effective temperature, and metallicity
with a full-covariance Gaussian mixture trained on a synthetic stellar
population:

```python
solver = ast.Solver(
    preset="precise",
    input_mode="likelihood",
    population_prior="trilegal_solar_neighbourhood",
    seed=42,
)
posterior = solver.solve(given, want=["M", "R"])
```

The bundled model represents a TRILEGAL 1.6 solar-neighbourhood simulation in

```{math}
(\log_{10}M,\log_{10}R,\log_{10}T_{\rm eff},[\mathrm{Fe/H}]).
```

It therefore prefers combinations that occur together in the simulated
population rather than combining four independent marginal draws. Exact
values supplied for any of these quantities condition the mixture
analytically. Population coordinates not required by a calculation are
marginalized analytically. Parallax and extinction retain their independent
priors.

A user-trained model can be selected by path:

```python
solver = ast.Solver(
    input_mode="likelihood",
    population_prior="/path/to/population_gmm.npz",
)
```

The correlated model is deliberately inactive in `propagate` mode. That mode
continues to use the original independent priors, replacing each one with an
uncertain fundamental input where supplied. This preserves its calculator-like
meaning: the interface only receives separate one-dimensional input
distributions and therefore cannot reconstruct their unknown joint
correlations.

Do not pass an independent custom prior for `M`, `R`, `Teff`, or `FeH` while
using the correlated model in likelihood mode. Combining the two would
double-count prior information, so AsteroScale raises an error. Independent
custom priors for `plx` and `A_G` remain supported.

## Presets and relation scatter

- `fast` gives a rough result with fewer live points;
- `standard` is the default for ordinary calculations; and
- `precise` uses more live points, a tighter stopping criterion, and the
  package's default empirical relation-scatter terms.

`fast` and `standard` use zero relation scatter unless you explicitly enable
it. This means `precise` can return wider distributions and slightly shifted
posterior summaries even for the same measurements. That change reflects a
different uncertainty model, not merely a longer run. See
{doc}`calibration-and-scatter` for details.

Mass, radius, and parallax are sampled internally in base-10 logarithmic
coordinates to make positive, correlated scales easier for Dynesty to explore.
Inputs, priors, likelihoods, and returned arrays remain in the physical units
listed in {doc}`../reference/quantities`.

Different random seeds should give statistically compatible summaries, not
identical samples. If conclusions change materially with the seed, increase
the sampling accuracy and inspect the posterior shape rather than choosing the
most convenient run.

## Custom priors and raw results

Advanced users can replace individual priors with `Solver(priors={...})` and
request the raw Dynesty result with `return_results=True`. The
{doc}`../tutorials/advanced-workflows` notebook gives worked examples.
