# Training a stellar-population prior

AsteroScale's current default priors treat mass, radius, effective temperature,
and metallicity independently. That is intentionally broad, but it permits
combinations that are rare in a real stellar population. The offline training
tools introduced here compress a population-synthesis catalogue into a
multivariate Gaussian mixture model (GMM).

This is preparatory infrastructure. The saved GMM is **not yet used by
`Solver`**, so training one does not change an AsteroScale result. Runtime
loading and conditional sampling will be added after a real synthesis
catalogue has been validated.

## What the mixture represents

The fitted coordinates are

```{math}
\boldsymbol{x} =
\left(
\log_{10} M,\,
\log_{10} R,\,
\log_{10} T_{\rm eff},\,
[\mathrm{Fe/H}]
\right).
```

They are standardized before fitting. The trainer uses weighted
expectation--maximization with full covariance matrices, so each Gaussian can
follow correlations such as the mass--radius relation. A GMM is best thought
of as a compressed density map: unlike a kernel-density estimate, it does not
place one kernel on every catalogue star.

The source catalogue still determines what the density means. Its initial
mass function, star-formation history, metallicity distribution, evolutionary
dwell times, Galactic volume, magnitude limit, and other selection effects do
not disappear during fitting. Record them whenever a model is trained.

## Portable input catalogue

The generic input is a comma-separated table. Column names and units are:

| Column | Required | Meaning |
|---|---:|---|
| `mass` | Yes | Current mass in $M_\odot$ |
| `radius` | Yes | Radius in $R_\odot$ |
| `teff` | Yes | Effective temperature in K |
| `feh` | Yes | $[\mathrm{Fe/H}]$ in dex |
| `weight` | No | Relative number of stars represented by the row |
| `age` | No | Age in years, retained for diagnostics |
| `state` | No | Evolutionary-state label retained for later conditioned fits |
| `population` | No | Population label, such as thin disc or halo |

Missing weights default to one. All fitted columns and weights must be finite;
mass, radius, temperature, and weights must be positive.

For a Monte Carlo synthesis catalogue, equal row weights are normally correct
because the sampling already encodes the population density. For a grid of
stellar tracks, equal weights are generally wrong: each row must be weighted
by the initial mass function, population history, and time represented by the
grid point.

## Adapting TRILEGAL output

The TRILEGAL adapter accepts one or more whitespace tables whose first
non-empty line contains column names. It expects current mass, luminosity,
effective temperature, and metallicity columns. Radius is calculated from

```{math}
\frac{R}{R_\odot}
=
\sqrt{\frac{L}{L_\odot}}
\left(\frac{T_{\rm eff,\odot}}{T_{\rm eff}}\right)^2.
```

Current mass (`Mact`) is used rather than initial mass because the seismic
scaling relations describe the star at its present evolutionary stage.

Standard TRILEGAL output provides $[\mathrm{M/H}]$, whereas AsteroScale names
the fitted coordinate `feh`. Equating the two is reasonable for a solar-scaled
mixture, but it should be recorded as an approximation for alpha-enhanced
populations.

To normalize files before fitting:

```bash
asteroscale-train-prior trilegal/*.dat \
    --adapter trilegal \
    --max-distance-pc 1000 \
    --teff-min 3500 \
    --teff-max 8000 \
    --metadata trilegal-settings.json \
    --normalized-catalogue solar_neighbourhood.csv \
    --output solar_neighbourhood_gmm.npz
```

The distance cut uses the true distance-modulus column (`m-M0`). Interstellar
extinction is not used to derive any fitted coordinate.

The optional metadata file must contain one JSON object. Use it to preserve
the TRILEGAL version, web-form settings, pointing list, random seeds,
generation date, and citations. These choices are scientifically part of the
prior even though they are not numerical GMM parameters.

## Fitting and model selection

By default the command fits 8, 16, 32, and 64 component models, using five
initializations for each. For a larger real catalogue, expand the search:

```bash
asteroscale-train-prior solar_neighbourhood.csv \
    --components 8,16,32,64,128,256 \
    --validation-fraction 0.2 \
    --selection-tolerance 0.01 \
    --diagnostic-plot population_gmm.png \
    --report population_gmm.json \
    --output population_gmm.npz
```

The trainer reserves a random validation subset and selects the smallest model
whose held-out mean log density is within `selection-tolerance` of the best
candidate. BIC is written to the report as a secondary diagnostic, not used as
the sole decision rule. A final model with the selected component count is
then fitted to all rows.

Expectation steps and density evaluation are processed in batches of 100,000
rows by default, so the largest temporary array scales with the batch rather
than the complete synthesis catalogue. Change this with `--batch-size` if
memory is limited; doing so should not change the fitted objective apart from
minor floating-point summation differences.

The diagnostic plot compares the input catalogue and GMM samples in the
temperature--radius, mass--radius, and temperature--metallicity planes. These
global projections are necessary but not sufficient. Before adopting a
default prior, also inspect:

- narrow mass and metallicity slices;
- the main sequence, subgiant branch, RGB, and core-helium-burning sequences;
- density assigned to physically sparse regions;
- distributions in predicted $\nu_{\max}$--$\Delta\nu$ space; and
- recovery and coverage for simulated observations.

Sharp boundaries require particular care because full-covariance Gaussians can
leak probability across them. Survey selection and the red edge of the
solar-like oscillation regime should remain explicit selection or
applicability models rather than being disguised as boundaries of the
underlying stellar population.

## Saved model

The compressed NPZ contains no Python objects and can be loaded with
`allow_pickle=False`. It stores:

- mixture weights, means, covariances, and Cholesky factors;
- coordinate names, centring, and scaling;
- the rectangular range covered by the training catalogue;
- component-selection scores; and
- JSON provenance metadata.

The model file is usually small enough to package with AsteroScale. The full
population catalogue and fitting dependencies will not be needed at runtime.

## Proposed first TRILEGAL population

For the first solar-neighbourhood prototype, use 12 equal-area Galactic
pointings:

```text
l =  45, 135, 225, 315; b = +41.81
l =   0,  90, 180, 270; b =   0.00
l =  45, 135, 225, 315; b = -41.81
```

Suggested settings for each pointing are:

- field area: 10 square degrees;
- limiting band and magnitude: Gaia $G=20$;
- extinction at infinity: $A_V=0$;
- binary fraction: zero for the first single-star model;
- all Galactic components;
- default TRILEGAL IMF, star-formation, age--metallicity, and structural
  parameters; and
- a different random seed for every field.

Combining equal-area pointings with equal row weights approximates an all-sky
directional average. After combining them, apply the 1 kpc distance cut. Check
completeness by repeating representative fields to $G=22$ and verifying that
the retained 1 kpc population is unchanged.

TRILEGAL should be cited via
[Girardi et al. (2005)](https://ui.adsabs.harvard.edu/abs/2005A%26A...436..895G/abstract).
The PARSEC/COLIBRI ingredients used by a particular service version should
also be recorded and cited from that run's documentation.
