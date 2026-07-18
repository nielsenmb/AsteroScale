# AsteroScale

AsteroScale gives quick estimates of where solar-like stellar oscillations
and granulation should appear and how large they might be. It can also use
measured oscillations to estimate stellar mass, radius, density and surface
gravity.

The package is intended for exploratory work and does not require prior
asteroseismology experience. A particular target audience is exoplanet
observers who want to understand where stellar variability may appear in a
light curve or power spectrum.

**[Read the full AsteroScale documentation](https://asteroscale.readthedocs.io/en/latest/)**

## Quick start

Predict the oscillation properties of a star from its mass, radius,
effective temperature and metallicity:

```python
import asteroscale as ast

star = {
    "M": 1.1,       # solar masses
    "R": 1.3,       # solar radii
    "Teff": 6000,   # K
    "FeH": 0.1,     # dex
}

prediction = ast.solve(
    star,
    want=["numax", "dnu", "FWHM_env", "A_env", "A_gran"],
    bandpass="TESS",
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
```

`numax` and `FWHM_env` describe approximately where to search for the
oscillation power excess. `A_env` is the maximum radial-mode RMS amplitude
in the selected TESS or Kepler band, not the total light-curve RMS.

Measurements with uncertainties can be supplied as `(value, uncertainty)`:

```python
given = {
    "Teff": (5777, 50),
    "FeH": (0.0, 0.05),
    "numax": (3090, 30),
    "dnu": (135.1, 1.0),
}

samples = ast.solve(given, want=["M", "R", "rho", "logg"])
ast.summarize(samples)
```

## Installation

```bash
git clone https://github.com/nielsenmb/AsteroScale.git
cd AsteroScale
python -m pip install -e .
```

## Documentation

The [hosted documentation](https://asteroscale.readthedocs.io/en/latest/)
includes:

- an introduction for exoplanet users;
- amplitude and bandpass conventions;
- uncertainty propagation, priors and sampler settings;
- quantities and units;
- an API reference generated from the package docstrings; and
- worked examples, including real stars with literature comparisons.

The complete example notebook is available as
[`example.ipynb`](example.ipynb). The documentation is configured for
Sphinx and Read the Docs.

Build it locally with:

```bash
python -m pip install -e ".[docs]"
sphinx-build -W -b html docs docs/_build/html
```

## Important scope and limitations

- AsteroScale uses empirical scaling relations rather than detailed stellar
  models or individual oscillation frequencies.
- Predicted amplitudes do not establish detectability. Instrumental noise,
  cadence attenuation, dilution, activity, gaps and the observing window
  are not included.
- `A_env` is a radial-mode amplitude and should not be added directly to a
  transit-depth uncertainty budget.
- Weakly constrained results may be sensitive to the default field-star
  priors.
- Gaia photometric relations are approximate and unsuitable for precision
  photometry.

See the [full limitations page](docs/limitations.md) before using results in
a precision analysis.

## References

The implemented relations draw primarily on Kjeldsen & Bedding (1995),
Huber et al. (2011), Viani et al. (2017), Guggenberger et al. (2016), Ball
et al. (2018), Campante et al. (2016), and Kallinger et al. (2014). Exact
formulae and citations are given in
[`asteroscale/relations.py`](asteroscale/relations.py).

## Development

```bash
python -m pip install -e ".[test,docs]"
pytest
sphinx-build -W -b html docs docs/_build/html
```
