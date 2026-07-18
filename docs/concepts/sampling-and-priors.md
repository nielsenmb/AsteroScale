# Sampling, uncertainties and priors

When every supplied input is exact, AsteroScale evaluates or inverts the
relations without running a sampler. With uncertain inputs, it uses Dynesty
nested sampling to propagate those uncertainties and marginalise over
quantities that were not supplied.

```python
solver = ast.Solver(preset="fast", seed=42)
samples = solver.solve(given, want=["M", "R"])
```

The available presets are:

- `fast`: an initial, approximate result;
- `standard`: the default for normal use;
- `precise`: a more expensive starting point for careful calculations.

The default priors describe plausible field stars and are not
uninformative. Results can therefore be prior-sensitive when few or weak
measurements are supplied. Supplying Gaia parallax and photometry usually
provides much stronger constraints, but users should still examine broad or
unexpected posteriors carefully.

Advanced users can replace individual priors with `Solver(priors={...})` and
can request the raw Dynesty result using `return_results=True`.

## Two meanings of an uncertain input

AsteroScale keeps the statistical interpretation of an input separate from
the `fast`, `standard`, and `precise` numerical presets. Choose it with
`input_mode`:

- `input_mode="propagate"` (the default) treats an uncertain fundamental
  parameter such as `Teff=(5772, 50)` as your current knowledge of that
  parameter. Its distribution replaces AsteroScale's default prior and is
  propagated through the scaling relations. This is the calculator-style
  mode and is appropriate when the input is already a posterior from an
  external analysis, such as a Gaia catalogue product.
- `input_mode="likelihood"` treats every uncertain input as a measurement
  likelihood. The configured population priors remain in the model and the
  returned samples represent the Bayesian posterior. Use this mode when the
  supplied uncertainties describe the measurement process rather than an
  already-inferred posterior.

```python
solver = ast.Solver(input_mode="likelihood", preset="standard", seed=42)
posterior = solver.solve(
    {"Teff": (5772, 50), "numax": (3090, 30), "dnu": (135.1, 1.0)},
    want=["M", "R"],
)
```

The mode can also be overridden for one calculation:

```python
samples = solver.solve(given, want=["M", "R"], input_mode="propagate")
```

Plain scalar inputs remain fixed exactly in both modes. Derived quantities
such as `numax` and `dnu` are always likelihood constraints because they are
predictions of the fundamental stellar parameters. Avoid applying a
population prior twice: if a catalogue value is already a posterior obtained
with a population prior, either use `propagate` or reconstruct its measurement
likelihood before using `likelihood`.

When any input is uncertain, empirical relation scatter is also marginalized
over. See {doc}`calibration-and-scatter` for its interpretation and how to
override or disable it. Exact derived scalars are accepted for an all-exact
point estimate, but a sampling problem requires a positive measurement
uncertainty.
