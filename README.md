# AsteroScale

AsteroScale gives quick estimates of where solar-like stellar oscillations and
granulation should appear and how large they might be. It can also use measured
global oscillation properties to estimate stellar mass, radius, mean density,
and surface gravity.

The package is intended for exploratory work and assumes no prior
asteroseismology experience. It is particularly useful when you want a first
answer to questions such as:

- Where should I look for an oscillation-power excess in a TESS or Kepler power
  spectrum?
- Could stellar oscillations or granulation be relevant variability for an
  exoplanet observation?
- Given measured `numax`, `dnu`, and effective temperature, what approximate
  mass and radius follow from global scaling relations?

**[Read the full AsteroScale documentation](https://asteroscale.readthedocs.io/en/latest/)**,
starting with [Asteroseismology in a nutshell](https://asteroscale.readthedocs.io/en/latest/concepts/asteroseismology-in-a-nutshell.html)
if `numax` and `dnu` are unfamiliar.

## Quick start

Predict the oscillation properties of a star from its mass, radius, effective
temperature, and metallicity:

```python
import asteroscale as ast

star = {
    "M": 1.1,       # solar masses
    "R": 1.3,       # solar radii
    "Teff": 6000,   # K
    "FeH": 0.1,     # dex; 0 is solar metallicity
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

`numax` is the approximate centre of the oscillation-power excess, `dnu` is
the average spacing of modes with consecutive radial order, and `FWHM_env`
describes the approximate width of the excess. Frequencies are in microhertz.
`A_env` is the maximum radial-mode RMS amplitude in the selected TESS or Kepler
band, not the total light-curve RMS or a detection probability.

Measurements with independent Gaussian uncertainties can be supplied as
`(value, one-sigma uncertainty)`:

```python
given = {
    "Teff": (5777, 50),
    "FeH": (0.0, 0.05),
    "numax": (3090, 30),
    "dnu": (135.1, 1.0),
}
samples = ast.solve(
    given=given,
    want=["M", "R", "rho", "logg"],
    seed=42,
)
ast.summarize(samples)
```

A plain numeric input is treated as exact and returns scalar point predictions
when all inputs are exact. Uncertain inputs launch the sampler and return arrays
representing output distributions.

## Installation

```bash
git clone https://github.com/nielsenmb/AsteroScale.git
cd AsteroScale
python -m pip install -e .
```

## Documentation

The [hosted documentation](https://asteroscale.readthedocs.io/en/latest/)
includes:

- a beginner-oriented introduction to solar-like oscillations and power
  spectra;
- complete tables of fundamental and derived quantities, dependencies, and
  units;
- amplitude and bandpass conventions;
- uncertainty propagation, priors, relation scatter, and sampler settings;
- an API reference generated from NumPy-style docstrings; and
- worked notebooks, including real stars with literature comparisons.

The documentation is configured for Sphinx and Read the Docs. Build it locally
with:

```bash
python -m pip install -e ".[docs]"
sphinx-build -W -b html docs docs/_build/html
```

## Important scope and limitations

- AsteroScale uses empirical global scaling relations rather than detailed
  stellar models, evolutionary tracks, or individual oscillation frequencies.
- Predicted amplitudes do not establish detectability. Instrumental noise,
  cadence attenuation, dilution, activity, gaps, and the observing window are
  not included.
- `A_env` is a radial-mode RMS amplitude and should not be added directly to a
  transit-depth uncertainty budget.
- Weakly constrained results may be sensitive to the independent default
  field-star priors.
- The `precise` preset includes provisional independent relation-scatter terms;
  `fast` and `standard` disable them by default.
- Gaia photometric relations are approximate and unsuitable for precision
  photometry.

See the [full limitations page](https://asteroscale.readthedocs.io/en/latest/limitations.html)
before using results in a precision analysis.

## References and further reading

The implemented relations draw primarily on
[Kjeldsen & Bedding (1995)](https://ui.adsabs.harvard.edu/abs/1995A%26A...293...87K/abstract),
[Huber et al. (2011)](https://ui.adsabs.harvard.edu/abs/2011ApJ...743..143H/abstract),
[Kallinger et al. (2014)](https://ui.adsabs.harvard.edu/abs/2014A%26A...570A..41K/abstract),
[Campante et al. (2016)](https://ui.adsabs.harvard.edu/abs/2016ApJ...830..138C/abstract),
[Guggenberger et al. (2016)](https://ui.adsabs.harvard.edu/abs/2016MNRAS.460.4277G/abstract),
[Viani et al. (2017)](https://ui.adsabs.harvard.edu/abs/2017ApJ...843...11V/abstract),
and [Ball et al. (2018)](https://ui.adsabs.harvard.edu/abs/2018ApJS..239...34B/abstract).
The documentation contains the
[full reference list](https://asteroscale.readthedocs.io/en/latest/reference/references.html).

For a broader introduction, see
[Chaplin & Miglio (2013)](https://ui.adsabs.harvard.edu/abs/2013ARA%26A..51..353C/abstract)
or [García & Ballot (2019)](https://ui.adsabs.harvard.edu/abs/2019LRSP...16....4G/abstract).

## Development

```bash
python -m pip install -e ".[test,docs]"
pytest
sphinx-build -W -b html docs docs/_build/html
```
