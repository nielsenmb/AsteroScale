# astero_solver

A back-of-envelope calculator for asteroseismic scaling relations. Give it
any subset of a star's observed parameters (with uncertainties), and it
returns any other parameter(s) you ask for -- marginalizing over whatever
wasn't given, using nested sampling.

```python
import astero_solver as ast

given = {
    "Teff": (5777, 50),
    "FeH": (0.0, 0.05),
    "numax": (3090, 30),
    "dnu": (135.1, 1.0),
}
out = ast.solve(given, want=["M", "R"])
ast.summarize(out)
```
```
param             mean         std         p16         p50         p84
----------------------------------------------------------------------
M                1.008     0.041       0.968       1.006       1.047
R                1.003     0.017       0.986       1.003       1.020
```

## Install

```bash
pip install -e .
```

Requires `numpy`, `dynesty`, `matplotlib`.

## How it works

Three parameters -- mass, radius, and Teff -- are treated as fundamental and
sampled from priors. Everything else (numax, dnu, luminosity, log g, mean
density, distance, Gaia G magnitude, ...) is a deterministic function of
those three plus parallax, extinction, and [Fe/H] (also fundamental). Nested
sampling (via `dynesty`) explores the fundamental parameters, weighting by
how well the forward-computed quantities match whatever you supplied in
`given`. Anything you didn't supply is only constrained by its prior --
i.e. marginalized over -- automatically.

This means there's no separate "inverse" relation for e.g. going from
(numax, dnu, Teff) to (M, R): the same forward model handles every
direction, and any given quantity, fundamental or derived, becomes a
constraint.

## Fundamental parameters

| name | meaning | units |
|---|---|---|
| `M` | mass | Msun |
| `R` | radius | Rsun |
| `Teff` | effective temperature | K |
| `plx` | parallax | mas |
| `A_G` | extinction (Gaia G band) | mag |
| `FeH` | metallicity [Fe/H] | dex |

Default priors are broad uniform bounds (see `solver.DEFAULT_PRIORS`).
Override per-parameter via `Solver(priors={"Teff": (5000, 6500)})`.

## Derived quantities

| name | meaning | relation |
|---|---|---|
| `numax` | frequency of max oscillation power | Kjeldsen & Bedding 1995, with Viani et al. 2017 mu-term metallicity correction |
| `dnu` | large frequency separation | Ulrich 1986, with Guggenberger et al. 2016 Teff/[Fe/H] reference function |
| `L` | luminosity | Stefan-Boltzmann |
| `logg` | surface gravity | classic scaling |
| `rho` | mean density | M/R^3 |
| `d` | distance | 1000/plx |
| `Mbol` | absolute bolometric magnitude | from L |
| `BC_G`, `BC_BP`, `BC_RP` | bolometric corrections, Gaia G/BP/RP | linear placeholders near solar Teff -- see caveat below |
| `A_BP`, `A_RP` | extinction, Gaia BP/RP bands | fixed ratio to `A_G` (Danielski et al. 2018 coefficients) |
| `M_G`, `M_BP`, `M_RP` | absolute Gaia magnitudes | Mbol - BC |
| `G_mag`, `BP_mag`, `RP_mag` | apparent Gaia magnitudes | distance modulus + extinction |
| `BP_RP` | Gaia BP-RP color | BP_mag - RP_mag; distance cancels out, so this constrains Teff/extinction without needing a parallax |

See docstrings in `relations.py` for exact formulae, coefficients, and
citations.

## API

- `astero_solver.solve(given, want, **kwargs)` -- one-off convenience
  wrapper. Reuses a shared default `Solver` unless `nlive`/`priors`/`seed`
  are passed, in which case a fresh one is created.
- `astero_solver.Solver(priors=None, nlive=500, seed=None)` -- for reuse
  across multiple calls, or custom priors.
  - `.solve(given, want, dlogz=0.5, print_progress=False, return_results=False)`
    -- `given` is `{name: (value, error)}`; `want` is a list of names.
    Returns `{name: array_of_posterior_samples}`. Pass `return_results=True`
    to also get dynesty's raw `Results` object (evidence, run diagnostics).
- `astero_solver.summarize(samples, params=None)` -- prints mean/std and
  16/50/84th percentiles for each quantity.
- `astero_solver.plot_posterior(samples, params=None)` -- quick pairwise
  scatter/histogram grid for a fast visual sanity check on degeneracies.
- `astero_solver.relations` -- the scaling relations module, if you want to
  call individual functions directly (e.g. `relations.f_numax(-1.5)`).

## Known caveats

- **`BC_G`/`BC_BP`/`BC_RP` (bolometric corrections) are crude linear
  placeholders**, not a real grid. Solar zero-points are calibrated to
  match published values (M_G,sun~4.81, M_BP,sun~5.03, M_RP,sun~4.21,
  giving BP-RP_sun~0.82), but the Teff slopes away from solar are rough
  guesses. Fine for order-of-magnitude checks; swap in a proper
  MIST/Casagrande & VandenBerg bolometric correction table for anything
  more serious.
- **BP-RP alone is degenerate with extinction.** A single color can't
  independently pin down both Teff and A_G -- with the default broad `A_G`
  prior, a `BP_RP` constraint mostly just constrains a combination of Teff
  and reddening, not Teff on its own. This is the real reddening-Teff
  degeneracy you'd hit with actual Gaia photometry, not a bug. Narrow the
  `A_G` prior (e.g. from a dust map estimate) if you want `BP_RP` to
  usefully tighten Teff.
- **`f_numax`'s Gamma_1 term is fixed to 1.** Viani et al. 2017's full
  relation includes a `sqrt(Gamma_1/Gamma_1_sun)` term that needs stellar
  model grids to evaluate; only the mu (mean molecular weight) term is
  implemented here.
- **`dnu_ref` is renormalized to equal the solar Delta-nu exactly at solar
  Teff/[Fe/H]**, correcting a ~1-3 muHz offset in Guggenberger et al.
  2016's own published coefficients (they note this themselves -- their
  most solar-like model gives 136.1 muHz, attributed to surface effects not
  in their model grid). The renormalization preserves the function's
  Teff/[Fe/H] shape.
- Both metallicity corrections are calibrated over roughly
  `-1.0 < [Fe/H] < 0.5`; the default `FeH` prior is wider than that, so
  double check results for very metal-poor stars.
- Nested sampling with many simultaneous tight constraints can occasionally
  need more live points than the default (`nlive=500`) to get a well-mixed
  posterior -- pass `return_results=True` and check `results.eff` /
  effective sample size if a result looks suspiciously narrow.

## Performance note (JAX)

`relations.py` uses a module-level `xp = np` that every function resolves
at call time, so the whole forward pass can be made JAX-traceable by
setting `relations.xp = jax.numpy` -- no forked codebase needed. This
isn't wired up by default: `dynesty`'s default sampler calls the likelihood
one point at a time, so `jax.jit`'s per-call dispatch overhead actually
made things ~1.7x *slower* in testing, not faster. It would be worth
revisiting if this ever moves to a genuinely vectorized/batched sampler
(e.g. [jaxns](https://github.com/Joshuaalbert/jaxns), which vectorizes the
entire nested sampling loop in JAX/XLA) where JIT has something to
amortize against.

`distributions.py` contains JAX-jittable prior distribution classes
(`uniform`, `normal`, `beta`, `truncsine`, plus a generic wrapper for
arbitrary ppf/pdf/logpdf/cdf callables) for that future use -- not
currently wired into `Solver`.

## Extending

- New derived quantity: write a function in `relations.py`, add it to the
  `DERIVED` dict as `"name": (func, ("arg1", "arg2", ...))`, in dependency
  order (args must already be computable from fundamentals or earlier
  entries).
- New fundamental parameter: add to `FUNDAMENTAL` in `relations.py` and
  `DEFAULT_PRIORS` in `solver.py`.
