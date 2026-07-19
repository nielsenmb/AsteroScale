# Scaling relation error, calibration, and validation

AsteroScale separates measurement uncertainty from uncertainty in the empirical scaling relations. In any calculation using the `preset=precise` containing uncertain inputs, scaling relations are by default assigned independent log-normal scatter which captures their inability to reflect true stellar physics. 

The default fractional one-sigma scatter is:

| Relation | Default scatter | Motivation |
|---|---:|---|
| `numax` | 2% | model-discrepancy floor for the adopted scaling relation |
| `dnu` | 1.5% | residual uncertainty in the corrected scaling relation |
| `A_env` | 25% | empirical spread discussed by Ball et al. (2018) |
| `A_gran` | 14.4% | Kallinger et al. (2014) fit scatter |
| `b_gran_low` | 10.2% | Kallinger et al. (2014) fit scatter |
| `b_gran_high` | 8.7% | Kallinger et al. (2014) fit scatter |

The scatter is currently independent between relations; a future multivariate form could represent their covariance.

Override one or more values when constructing a solver:

```python
solver = ast.Solver(relation_scatter={"numax": 0.03, "dnu": 0.02})
```

A value of zero makes that empirical relation more deterministic:

```python
solver = ast.Solver(relation_scatter={"numax": 0.0})
```

Note that due to sampling taking place some randomness is always present unless you fix the sampling seed.

For backward compatibility, a calculation in which every input is a point estimate will return the central scaling-relation values as scalars. To obtain a predictive distribution containing only relation scatter, request it explicitly:

```python
prediction = solver.solve({"M": 1.0, "R": 1.0, "Teff": 5772.0, "FeH": 0.0},
                          want=["numax", "dnu"],
                          sample_relation_scatter=True,
                          )
```

## Calibration-domain reports

The solver checks the ranges over which the main empirical relations were calibrated. The latest report is stored on the solver:

```python
samples = solver.solve(given, want=["M", "R"])
print(solver.last_validity)
```

It can also be included in the returned dictionary:

```python
samples = solver.solve(given, want=["M", "R"], return_validity=True)
print(samples["_validity"])
```

The report records the fraction of posterior samples within each adopted domain. Warnings can be disabled with `warn_validity=False`. Evolutionary state is not a
fundamental parameter in AsteroScale, so validity checks cannot distinguish an RGB star from a cool main-sequence star.

## Numerical benchmark stars

The test suite compares forward predictions with independently measured properties across several evolutionary and observational regimes:

| Regime | Benchmark |
|---|---|
| Solar anchor | Sun |
| Kepler LEGACY dwarf | 16 Cyg A |
| Interferometric dwarf | alpha Cen A |
| Subgiant | beta Hyi |
| Red giant | epsilon Tau |
| Eclipsing-binary red giant | KIC 8410637 |

These checks are not a claim that AsteroScale is accurate to the observational uncertainty of every star, but are used to test. The non-solar tests
currently require agreement within 10% for `numax` and 7% for `dnu`. The eclipsing-binary test is especially useful because its mass and radius are
dynamical measurements rather than quantities inferred from the same seismic scaling relations.
