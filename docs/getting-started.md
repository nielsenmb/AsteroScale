# Getting started

## Predict where oscillations should appear

For a star with known mass, radius, effective temperature and metallicity:

```python
import asteroscale as ast

given = {
    "M": 1.1,       # solar masses
    "R": 1.3,       # solar radii
    "Teff": 6000,   # K
    "FeH": 0.1,     # dex
}
want = ["numax", "dnu"],

prediction = ast.solve(given=given,
                       want=want,
                       )

for name, value in prediction.items():
    print(f"{name:12s} {value:.2f}")
```

The most important outputs for an initial power-spectrum inspection are:

- `numax`: approximately where the oscillation power excess peaks;
- `FWHM_env`: the approximate width of that excess;
- `dnu`: the average spacing between consecutive modes of the same angular degree;
- `A_env`: the maximum radial-mode RMS amplitude in the selected band;
- `A_gran`: the characteristic granulation RMS amplitude.

See {doc}`concepts/asteroseismology-in-a-nutshell` before interpreting these as a detectability prediction or time-domain noise estimate.

## Include measurement uncertainties

A plain number is treated as exact. A `(value, uncertainty)` pair represents a Gaussian measurement:

```python
given = {"M": (1.10, 0.08),
         "R": (1.30, 0.04),
         "Teff": (6000, 80),
         "FeH": (0.10, 0.05),
         }

want = ["numax", "dnu", "FWHM_env", "A_env"]
samples = ast.solve(given=given,
                    want=want,
                   )
ast.summarize(samples)
```

## Infer stellar properties from oscillations

```python
given = {"Teff": (5777, 50),
         "FeH": (0.0, 0.05),
         "numax": (3090, 30),
         "dnu": (135.1, 1.0),
         }
want =
samples = ast.solve(given, want=["M", "R", "rho", "logg"])
ast.summarize(samples)
```

Continue with the {doc}`tutorials/index` or consult the {doc}`reference/quantities` table for names and units.
