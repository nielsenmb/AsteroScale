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
