# Getting started

AsteroScale can be used in two directions:

1. **Forward prediction:** given basic stellar properties, estimate where
   solar-like oscillations and granulation should appear.
2. **Inverse calculation:** given measured global seismic quantities, estimate
   mass, radius, density, and surface gravity.

If `numax`, `dnu`, or a power spectrum is unfamiliar, read
{doc}`concepts/asteroseismology-in-a-nutshell` first. All accepted names and
units are listed in {doc}`reference/quantities`.

## Predict where oscillations should appear

For a star with known mass, radius, effective temperature, and metallicity:

```python
import asteroscale as ast

given = {
    "M": 1.1,       # solar masses
    "R": 1.3,       # solar radii
    "Teff": 6000,   # K
    "FeH": 0.1,     # dex; 0 is solar metallicity
}
want = ["numax", "dnu", "FWHM_env", "A_env", "A_gran"]

prediction = ast.solve(given=given, want=want, bandpass="TESS")

for name, value in prediction.items():
    print(f"{name:12s} {value:.2f}")
```

Because every input is a plain number, this returns central point predictions
as scalars. The most useful outputs for an initial power-spectrum inspection
are:

- `numax`: approximately where the oscillation-power excess peaks;
- `FWHM_env`: the approximate width of that excess;
- `dnu`: the average spacing between consecutive modes of the same angular
  degree;
- `A_env`: the maximum radial-mode RMS amplitude in the selected band; and
- `A_gran`: an empirical granulation RMS scale.

These are not a detection probability or a complete time-domain noise model.

## Include measurement uncertainties

A plain number is treated as exact. A `(value, uncertainty)` pair represents an
independent Gaussian distribution with the quoted one-sigma uncertainty:

```python
given = {
    "M": (1.10, 0.08),
    "R": (1.30, 0.04),
    "Teff": (6000, 80),
    "FeH": (0.10, 0.05),
}

samples = ast.solve(
    given=given,
    want=["numax", "dnu", "FWHM_env", "A_env"],
    seed=42,
)
ast.summarize(samples)
```

The outputs are now arrays of samples rather than scalars. Use their median and
16th--84th percentile interval instead of selecting one arbitrary element.

## Infer stellar properties from oscillations

Measured $\nu_{\max}$ and $\Delta\nu$ are normally combined with an effective
temperature. A metallicity constraint is also recommended because the adopted
relations include metallicity-dependent corrections:

```python
given = {
    "Teff": (5777, 50),
    "FeH": (0.0, 0.05),
    "numax": (3090, 30),
    "dnu": (135.1, 1.0),
}

samples = ast.solve(
    given,
    want=["M", "R", "rho", "logg"],
    seed=42,
)
ast.summarize(samples)
```

This is a global scaling-relation estimate, not a fit to individual modes or a
stellar evolutionary track. It does not return an age.

## Which preset should I use?

Start with `preset="fast"` while checking units and inputs, then use
`preset="standard"` for ordinary calculations. The `precise` preset uses more
live points and also enables empirical relation scatter, so it changes both
runtime and the uncertainty model. Read {doc}`concepts/calibration-and-scatter`
before comparing precise results with a deterministic inversion.

Continue with the {doc}`tutorials/index`, and consult {doc}`limitations` before
using AsteroScale in a precision analysis.
