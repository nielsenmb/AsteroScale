# Calibration, intrinsic scatter and validity

AsteroScale separates measurement uncertainty from uncertainty in the
empirical scaling relations. In any calculation containing uncertain inputs,
positive relations are assigned independent log-normal calibration offsets.
The default fractional one-sigma scatter is:

| Relation | Default scatter | Motivation |
|---|---:|---|
| `numax` | 2% | model-discrepancy floor for the adopted scaling relation |
| `dnu` | 1.5% | residual uncertainty in the corrected scaling relation |
| `A_env` | 25% | empirical spread discussed by Ball et al. (2018) |
| `A_gran` | 14.4% | Kallinger et al. (2014) fit scatter |
| `b_gran_low` | 10.2% | Kallinger et al. (2014) fit scatter |
| `b_gran_high` | 8.7% | Kallinger et al. (2014) fit scatter |

The seismic values are deliberately transparent provisional floors, rather
than claims that the underlying systematics are known exactly. They should be
recalibrated as the benchmark suite grows. Scatter is currently independent
between relations; a future multivariate calibration could represent their
covariance.

Override one or more values when constructing a solver:

```python
solver = ast.Solver(
    relation_scatter={"numax": 0.03, "dnu": 0.02},
    seed=42,
)
```

A value of zero makes that empirical relation deterministic:

```python
solver = ast.Solver(relation_scatter={"numax": 0.0})
```

For backward compatibility, a calculation in which every input is exact
returns the central scaling-relation values as scalars. To obtain a predictive
distribution containing only relation scatter, request it explicitly:

```python
prediction = solver.solve(
    {"M": 1.0, "R": 1.0, "Teff": 5772.0, "FeH": 0.0},
    want=["numax", "dnu"],
    sample_relation_scatter=True,
)
```

## Exact derived constraints

Exact derived values can be used when every input is exact and AsteroScale is
performing a deterministic inversion. During a sampling calculation, derived
measurements must have positive uncertainty:

```python
given = {"numax": (3090.0, 30.0)}  # accepted
```

An exact derived constraint occupies a zero-volume surface and cannot be
represented by a zero-width Gaussian in ordinary nested sampling. AsteroScale
therefore raises an error rather than silently assigning an arbitrary small
uncertainty.

## Calibration-domain reports

The solver checks the ranges over which the main empirical relations were
calibrated. The latest report is stored on the solver:

```python
samples = solver.solve(given, want=["M", "R"])
print(solver.last_validity)
```

It can also be included in the returned dictionary:

```python
samples = solver.solve(given, want=["M", "R"], return_validity=True)
print(samples["_validity"])
```

The report records the fraction of posterior samples within each adopted
domain. Warnings can be disabled with `warn_validity=False`, but doing so does
not make an extrapolated relation more reliable. Evolutionary state is not a
fundamental parameter in AsteroScale, so validity checks cannot distinguish an
RGB star from a core-helium-burning star.

## Numerical benchmark stars

The test suite compares forward predictions with independently measured
properties across several evolutionary and observational regimes:

| Regime | Benchmark |
|---|---|
| Solar anchor | Sun |
| Kepler LEGACY dwarf | 16 Cyg A |
| Interferometric dwarf | alpha Cen A |
| Subgiant | beta Hyi |
| Red giant | epsilon Tau |
| Eclipsing-binary red giant | KIC 8410637 |

These checks are regression guards, not a claim that every relation is
accurate to the observational uncertainty of every star. The non-solar tests
currently require agreement within 10% for `numax` and 7% for `dnu`. The
eclipsing-binary test is especially useful because its mass and radius are
dynamical measurements rather than quantities inferred from the same seismic
scaling relations.
