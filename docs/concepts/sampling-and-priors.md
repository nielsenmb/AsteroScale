# Sampling, uncertainties and priors

When every supplied input is exact, AsteroScale evaluates or inverts the relations without running a sampler. However, with uncertain inputs it uses Dynesty nested sampling to propagate those uncertainties and marginalise over quantities that were not supplied.

```python
given = {"Teff": (5777, 50),
         "numax": (3090, 30),
         "dnu": (135.1, 0.1),}
want = ["M", "R"]
solver = ast.Solver()
samples = solver.solve(given=given, want=want)
```

The default priors describe plausible field stars. Results can therefore be prior-sensitive when few or weak measurements are supplied. Supplying Gaia parallax and photometry usually provides much stronger constraints, but users should still examine broad or unexpected posteriors carefully.

## Two meanings of an uncertain input

AsteroScale uses two statistical interpretations for the input, set by the `input_mode` argument:

- `input_mode="propagate"` (the default) treats an uncertain fundamental parameter such as `Teff=(5772, 50)` as your current knowledge of that parameter. Its distribution replaces AsteroScale's default prior and is propagated through the scaling relations. This is the calculator-style mode and is appropriate when the input is already a posterior from an external analysis, such as a Gaia catalogue product.
- `input_mode="likelihood"` treats every uncertain input as a measurement as a new data point that updates the current knowledge of the parameter. The population priors remain in the model and the returned samples represent the true Bayesian posterior. Use this mode when the supplied uncertainties describe the measurement process rather than an already-inferred posterior. Note that this is the default setting when using the `precise` mode.

```python
solver = ast.Solver(input_mode="likelihood")
given = {"Teff": (5777, 50),
         "numax": (3090, 30),
         "dnu": (135.1, 0.1),}
want = ["M", "R"]
posterior = solver.solve(given=given, want=want)
```

The mode can also be overridden for one calculation:

```python
samples = solver.solve(given, want=["M", "R"], input_mode="propagate")
```

Plain scalar inputs remain fixed exactly in both modes. Derived quantities such as `numax` and `dnu` are always likelihood constraints because they are
predictions of the fundamental stellar parameters. Avoid applying a population prior twice: if a catalogue value is already a posterior obtained with a population prior, either use `propagate` or reconstruct its measurement likelihood before using `likelihood`.

When any input is uncertain, empirical relation scatter is also marginalized over. See {doc}`calibration-and-scatter` for its interpretation and how to
override or disable it. Exact derived scalars are accepted for an all-exact point estimate, but a sampling problem requires a positive measurement
uncertainty.

## Priors
Advanced users can replace individual priors with `Solver(priors={...})` and can request the raw Dynesty result using `return_results=True`.
