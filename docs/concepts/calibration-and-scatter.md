# Scaling-relation error, calibration, and validation

Measurement uncertainty and scaling-relation uncertainty are not the same.
For example, an extremely precise $\Delta\nu$ measurement does not make the
approximate density scaling exact. AsteroScale's `precise` preset therefore
adds independent, multiplicative log-normal offsets to calibrated empirical
relations when a problem is sampled. The `fast` and `standard` presets use zero
relation scatter by default.

The calibrated fractional one-sigma values used by `precise` are:

| Relation | Default scatter | Motivation |
|---|---:|---|
| `numax` | 2% | provisional model-discrepancy floor for the adopted scaling relation |
| `dnu` | 1.5% | provisional residual uncertainty in the corrected scaling relation |
| `A_env` | 25% | empirical spread discussed by [Ball et al. (2018)](https://ui.adsabs.harvard.edu/abs/2018ApJS..239...34B/abstract) |
| `A_gran` | 14.4% | fit scatter from [Kallinger et al. (2014)](https://ui.adsabs.harvard.edu/abs/2014A%26A...570A..41K/abstract) |
| `b_gran_low` | 10.2% | fit scatter from Kallinger et al. (2014) |
| `b_gran_high` | 8.7% | fit scatter from Kallinger et al. (2014) |

The seismic values are uncertainty floors adopted by AsteroScale rather than
published universal constants. Scatter is currently independent between
relations; correlations between real scaling residuals are not represented.

Enable or override selected values when constructing a solver:

```python
solver = ast.Solver(
    preset="standard",
    relation_scatter={"numax": 0.03, "dnu": 0.02},
)
```

Disable all relation scatter, including in `precise`, with a scalar zero:

```python
solver = ast.Solver(preset="precise", relation_scatter=0.0)
```

Relation scatter broadens the combinations of mass and radius compatible with
the observations. The field-star priors then have more influence, so posterior
means or medians need not equal the deterministic inversion. This is an
expected consequence of the model, although the adopted scatter values remain
provisional.

An all-exact forward calculation returns central values as scalars. To draw a
predictive distribution containing relation scatter, request it explicitly:

```python
prediction = ast.Solver(preset="precise", seed=42).solve(
    {"M": 1.0, "R": 1.0, "Teff": 5772.0, "FeH": 0.0},
    want=["numax", "dnu"],
    sample_relation_scatter=True,
)
```

## Calibration-domain reports

The solver checks whether the main empirical relations are being evaluated in
their adopted calibration ranges. The latest report is stored on the solver:

```python
samples = solver.solve(given, want=["M", "R"])
print(solver.last_validity)
```

It can also be included in the returned dictionary:

```python
samples = solver.solve(
    given,
    want=["M", "R"],
    return_validity=True,
)
print(samples["_validity"])
```

For sampled calculations, the report records the fraction of posterior samples
inside each adopted domain. A warning is a request to inspect extrapolation,
not proof that every returned sample is unusable. Warnings can be disabled with
`warn_validity=False` after checking the cause. AsteroScale does not sample
evolutionary state, so a validity check cannot by itself distinguish an RGB
star from a cool main-sequence star.

## Numerical benchmark stars

The test suite checks forward predictions across several regimes:

| Regime | Benchmark |
|---|---|
| Solar anchor | Sun |
| Kepler LEGACY dwarf | 16 Cyg A |
| Interferometric dwarf | $\alpha$ Cen A |
| Subgiant | $\beta$ Hyi |
| Red giant | $\epsilon$ Tau |
| Eclipsing-binary red giant | KIC 8410637 |

These tests verify rough agreement, not precision accuracy for every star. The
non-solar checks currently allow 10% differences in `numax` and 7% in `dnu`.
The eclipsing-binary case is valuable because its mass and radius are dynamical
rather than inferred from the same seismic scaling relations. Sources for the
documented comparison stars are listed on the {doc}`../reference/references`
page.
