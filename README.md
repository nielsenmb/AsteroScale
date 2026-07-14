# AsteroScale

AsteroScale gives quick estimates of where solar-like stellar oscillations and
granulation should appear, and how large they might be. It can also run the
usual asteroseismic scaling relations in reverse to estimate stellar
properties from measured oscillations.

The package is intended for exploratory work. You do not need prior
asteroseismology experience to use it.

## Why might I need this?

Stars are not perfectly quiet light sources. Convection near the surface
drives a background called **granulation**, and many cool stars also pulsate
in a collection of **solar-like oscillation modes**. In a power spectrum the
oscillations form a broad hump rather than one isolated peak.

For exoplanet observations, this variability can:

- add correlated noise to a transit or phase curve;
- complicate radial-velocity measurements;
- overlap with an astrophysical signal of interest; or
- provide an independent estimate of the host star's mass, radius, density,
  or surface gravity.

AsteroScale is a first-look calculator for questions such as:

- At approximately what frequency should the oscillation power peak?
- How broad is the oscillation envelope?
- Is my observing cadence fast enough to sample it?
- What rough oscillation and granulation amplitudes should I expect?
- If I measured the oscillations, what do they imply for the star?

## Quick start: where should I look for oscillations?

Suppose an exoplanet host has a mass of 1.1 solar masses, a radius of 1.3
solar radii, an effective temperature of 6000 K, and `[Fe/H] = 0.1`.

```python
import asteroscale as ast

star = {
    "M": 1.1,       # solar masses
    "R": 1.3,       # solar radii
    "Teff": 6000,   # K
    "FeH": 0.1,     # dex
    "plx": 10.0,    # mas; does not affect the oscillation quantities below
    "A_G": 0.0,     # mag; does not affect the oscillation quantities below
}

prediction = ast.solve(
    star,
    want=[
        "numax", "dnu", "FWHM_env", "A_env",
        "A_gran", "b_gran_low", "b_gran_high",
    ],
)

for name, value in prediction.items():
    print(f"{name:12s} {value:.2f}")
```

This gives approximately:

```text
numax        1982.25
dnu            94.86
FWHM_env      598.04
A_env           2.96
A_gran         34.47
b_gran_low    500.38
b_gran_high  1768.44
```

The most useful interpretation for a first inspection is:

- Oscillation power should peak near **1982 microhertz** (`numax`).
- Much of the power excess should lie within a band roughly one
  `FWHM_env` wide, here approximately **1683--2281 microhertz**. This is a
  search region, not a hard boundary.
- Adjacent radial orders are separated by roughly **95 microhertz**
  (`dnu`). Individual modes have a more complicated pattern within each
  order.
- `A_env` is the predicted maximum **radial-mode rms amplitude** in the
  TESS band, in parts per million. It is not the total rms of the complete
  oscillation envelope.
- `A_gran` is a granulation rms scale. The two `b_gran` values locate the
  characteristic frequencies of the adopted two-component granulation
  background.

Always compare the predicted frequency range with the Nyquist frequency and
frequency resolution of your data. Cadence, bandpass, dilution from nearby
stars, activity, data gaps, and the length of the observations all affect
whether the signal is detectable.

## Including measurement uncertainties

A plain number is treated as exact. Use `(value, uncertainty)` to represent
a Gaussian measurement. AsteroScale then propagates the uncertainties and
returns posterior samples:

```python
given = {
    "M": (1.10, 0.08),
    "R": (1.30, 0.04),
    "Teff": (6000, 80),
    "FeH": (0.10, 0.05),
}

samples = ast.solve(
    given,
    want=["numax", "dnu", "FWHM_env", "A_env", "A_gran"],
    preset="fast",
    seed=42,
)

ast.summarize(samples)
```

`preset="fast"` is useful for an initial look. Use the standard preset for
normal work and `preset="precise"` as a more expensive starting point when
the numerical sampling itself needs to be checked carefully.

## Going the other way: what do measured oscillations tell me?

Two commonly measured summary quantities are:

- **`numax`**: the frequency where the oscillation envelope has maximum
  power. It is most strongly related to surface gravity.
- **`dnu`** (Delta-nu): the average large frequency separation. It is most
  strongly related to mean stellar density.

Together with effective temperature, they can constrain mass and radius:

```python
given = {
    "Teff": (5777, 50),
    "FeH": (0.0, 0.05),
    "numax": (3090, 30),       # microhertz
    "dnu": (135.1, 1.0),      # microhertz
}

samples = ast.solve(given, want=["M", "R", "rho", "logg"])
ast.summarize(samples)
```

AsteroScale can work with incomplete information. Quantities that were not
measured are handled using priors, so results may become broad or
prior-sensitive when few constraints are supplied. A numerical answer does
not necessarily mean that the data strongly determined that answer.

## Installation

AsteroScale currently installs from the repository:

```bash
git clone https://github.com/nielsenmb/AsteroScale.git
cd AsteroScale
python -m pip install -e .
```

For development and tests:

```bash
python -m pip install -e ".[test]"
pytest
```

## Input and output names

All masses, radii, luminosities, and mean densities are in solar units.
Frequencies are in microhertz and photometric amplitudes are in parts per
million.

| Name | Meaning | Unit |
|---|---|---|
| `M` | Stellar mass | solar masses |
| `R` | Stellar radius | solar radii |
| `Teff` | Effective temperature | K |
| `FeH` | Metallicity `[Fe/H]` | dex |
| `plx` | Parallax | mas |
| `A_G` | Gaia G-band extinction | mag |
| `numax` | Frequency of maximum oscillation power | microhertz |
| `dnu` | Average large frequency separation | microhertz |
| `FWHM_env` | Width of the oscillation power envelope | microhertz |
| `A_env` | Maximum radial-mode rms amplitude in the TESS band | ppm |
| `A_gran` | Granulation rms amplitude | ppm |
| `b_gran_low`, `b_gran_high` | Granulation characteristic frequencies | microhertz |
| `L` | Luminosity | solar luminosities |
| `rho` | Mean density | solar density |
| `logg` | Base-10 surface gravity in cgs units | dex |

Use `want="all"` to calculate every available quantity. Additional Gaia
photometry and distance outputs are demonstrated in
[`example.ipynb`](example.ipynb).

## Common patterns

### Reuse a solver

```python
solver = ast.Solver(preset="fast", seed=42)
samples = solver.solve(given, want=["M", "R"])

# Calculate more quantities from the same result without sampling again.
more = solver.predict(["L", "rho", "logg", "A_env", "FWHM_env"])
```

### Use non-Gaussian measurements

Values in `given` may also be distributions with `.logpdf()` and `.ppf()`
methods, including frozen `scipy.stats` distributions:

```python
from scipy import stats

given = {
    "Teff": stats.skewnorm(a=3, loc=5700, scale=100),
    "numax": (2500, 100),
    "dnu": (110, 2),
}
```

### Process several independent stars

```python
targets = {
    "star_a": {"M": (1.0, 0.1), "R": (1.0, 0.05), "Teff": (5770, 80)},
    "star_b": {"M": (1.4, 0.1), "R": (2.0, 0.1), "Teff": (6200, 100)},
}

results = ast.solve_many(
    targets,
    want=["numax", "dnu", "FWHM_env", "A_env"],
    preset="fast",
)
```

Run multiprocessing code from a Python script protected by
`if __name__ == "__main__":` rather than directly from an interactive
notebook.

## What the model does

AsteroScale treats mass, radius, effective temperature, parallax,
extinction, and metallicity as fundamental quantities. Scaling relations
then predict the oscillation, granulation, luminosity, surface-gravity,
density, distance, and approximate Gaia-photometry quantities.

When inputs have uncertainties, the `dynesty` nested sampler explores
combinations of the fundamental quantities and keeps combinations that are
consistent with the supplied measurements. Missing quantities are averaged
over their priors. This allows the same forward model to be used both to
predict oscillations and to infer stellar properties.

The default priors describe plausible field stars; they are not
uninformative. Advanced users can replace them through
`Solver(priors={...})`. See the NumPy-style docstrings in `solver.py`,
`relations.py`, and `priors.py` for the detailed API and coefficients.

## Important limitations

- These are empirical scaling relations, not individual-frequency stellar
  modelling. They are best used for approximate predictions and sanity
  checks.
- `A_env` is a radial-mode amplitude, not the total time-domain stellar
  noise. Do not add it directly to a transit-depth uncertainty budget.
- The granulation quantities describe an idealized background. Instrumental
  noise, stellar activity, filtering, cadence, dilution, and the observing
  window are not included.
- Predictions do not establish detectability. Always inspect the actual
  power spectrum and account for the instrument and observing strategy.
- The metallicity corrections are calibrated over approximately
  `-1.0 < [Fe/H] < 0.5`; extrapolations require care.
- Gaia bolometric corrections are rough, near-solar placeholders. They
  should not be used for precision photometry.
- A single Gaia BP-RP color cannot independently determine temperature and
  extinction.
- Very tight or numerous constraints may require more live points and
  convergence checks. `return_results=True` exposes the raw Dynesty result.

Warnings about unusual values are intended to catch unit mistakes and
extrapolation. Do not silence them without checking the input.

## Scaling relations and references

- `numax`: Kjeldsen & Bedding (1995), with the mean-molecular-weight
  correction from Viani et al. (2017).
- `dnu`: density scaling with the temperature/metallicity reference function
  from Guggenberger et al. (2016).
- Oscillation-envelope amplitude and width: Ball et al. (2018).
- Granulation amplitude, characteristic frequencies, and super-Lorentzian
  background: Kallinger et al. (2014).

Exact formulae and citations are given in [`asteroscale/relations.py`](asteroscale/relations.py).
The notebook contains additional examples, including real stars with
literature values for comparison: [`example.ipynb`](example.ipynb).

## Scope

AsteroScale is deliberately a lightweight scaling-relation tool. For
detailed mode fitting, stellar-model grids, hierarchical population
inference, or a complete instrument-specific noise model, use a dedicated
analysis package.
