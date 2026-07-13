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

Default priors aim for "plausible for a random field star", not
"uninformative": mass follows a Salpeter-slope power law (fewer high-mass
stars), radius is log-uniform, parallax follows the Bailer-Jones (2015)
exponentially-decreasing space density prior used for Gaia distance
inference, extinction is exponential (favoring low values), and [Fe/H] is a
normal distribution truncated to the range the metallicity corrections
above are actually calibrated over. See `priors.py` and
`solver.DEFAULT_PRIORS` for exact parameters. Override per-parameter via
`Solver(priors={...})`: a `(mean, error)` tuple is shorthand for a
Gaussian (same convention as `given`), or pass any object with a `.ppf(u)`
method (a frozen `scipy.stats` distribution, one of the classes in
`priors.py`, or your own) directly for something else -- including
`scipy.stats.uniform(loc, scale)` if you specifically want uniform.

`Solver(..., sample="auto", bound="multi")` are passed straight through to
`dynesty.NestedSampler`. The default `sample="auto"` heuristic doesn't
always pick well for narrow or strongly correlated posteriors (e.g. many
simultaneous tight constraints); `sample="rwalk"` or `"rslice"` can help,
and dynesty's own runtime warnings will suggest this when it detects the
problem itself. See dynesty's docs for the full set of options.

## Derived quantities

| name | meaning | relation |
|---|---|---|
| `numax` | frequency of max oscillation power | Kjeldsen & Bedding 1995, with Viani et al. 2017 mu-term metallicity correction |
| `dnu` | large frequency separation | Ulrich 1986, with Guggenberger et al. 2016 Teff/[Fe/H] reference function |
| `L` | luminosity | Stefan-Boltzmann |
| `logg` | surface gravity | classic scaling |
| `rho` | mean density | M/R^3 |
| `FWHM_env` | FWHM of the p-mode power excess envelope | Mosser et al. 2012a: 0.66 * numax^0.88 |
| `A_env` | peak bolometric oscillation amplitude (ppm) | Kjeldsen & Bedding 1995 L/M scaling, normalized to A_env,sun = 3.6 ppm (Huber et al. 2011) |
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
- `astero_solver.Solver(priors=None, nlive=500, seed=None, sample="auto", bound="multi")`
  -- for reuse across multiple calls, custom priors, or sampler tuning.
  - `.solve(given, want, dlogz=0.5, print_progress=False, return_results=False)`
    -- `want` is a list of names. Each value in `given` can be:
    - **a plain number** -- treated as exactly known. If *every* given
      value is like this, `solve()` skips the sampler entirely and returns
      a fast point estimate (a single `scipy.optimize.least_squares` call
      if there are more free fundamentals than exact fundamentals given,
      or just a direct forward evaluation if every fundamental is pinned).
      Returns plain floats, not posterior samples.
    - **a `(mean, error)` tuple** -- wrapped as a Gaussian (`priors.Normal`).
    - **any object with `.logpdf`/`.ppf`** -- used directly, e.g. for
      asymmetric or otherwise non-Gaussian uncertainty on a measurement
      (`scipy.stats.skewnorm(...)`, or your own).

    A distribution given for a *fundamental* parameter specifically
    replaces its prior directly, rather than adding a separate likelihood
    term (more efficient, and equivalent when it's the only constraint on
    that dimension). If this ends up being true for every given value --
    e.g. `{"M": (1.0, 0.05), "R": (1.0, 0.02), "Teff": (5777, 50)}` -- the
    likelihood is flat everywhere, so `solve()` skips dynesty and draws
    directly from the (overridden) priors instead. This is the same
    result dynesty would give (a flat likelihood makes nested sampling
    mathematically equivalent to prior-predictive sampling), just without
    the "likelihood plateau" warning and wasted run time.

    Mixing types is fine -- e.g. `{"Teff": 5777.0, "FeH": (0.0, 0.05), ...}`
    pins Teff exactly (reducing the sampled dimensionality by one) while
    the rest still go through nested sampling as usual. If a distribution
    is given for a *fundamental* parameter specifically, it replaces that
    parameter's prior directly (more efficient than adding a separate
    likelihood term, and equivalent when it's the only constraint on that
    dimension).

    Returns `{name: array_of_posterior_samples}` for the sampled path, or
    `{name: float}` for the point-estimate path. Pass `return_results=True`
    to also get dynesty's raw `Results` object (or the `scipy.optimize`
    result, for the point-estimate path).
  - `.predict(want)` -- compute additional quantities from the
    posterior/point estimate of the *last* `solve()` call, without
    re-running the sampler:
    ```python
    solver.solve({"Teff": (5777, 50), "numax": (3090, 30), "dnu": (135.1, 1.0)}, want=["M", "R"])
    extra = solver.predict(["L", "rho", "logg", "A_env", "FWHM_env"])
    ```
    Works for both the nested-sampling path (returns arrays over the same
    posterior) and the point-estimate path (returns floats). Raises if
    called before any `solve()`.
- `astero_solver.summarize(samples, params=None)` -- prints mean/std and
  16/50/84th percentiles for each quantity.
- `astero_solver.plot_posterior(samples, params=None)` -- quick pairwise
  scatter/histogram grid for a fast visual sanity check on degeneracies.
- `astero_solver.relations` -- the scaling relations module, if you want to
  call individual functions directly (e.g. `relations.f_numax(-1.5)`).
- `astero_solver.solve_many(targets, want, priors=None, nlive=500, sample="auto", bound="multi", n_jobs=None, base_seed=0, show_progress=False)`
  -- solve many independent targets in parallel (one process per target, up
  to `n_jobs` at a time). `targets` is `{target_id: given_dict}`; `want` is
  either one list applied to every target or `{target_id: want_list}` for
  per-target requests. Returns `{target_id: solve()_output}` in the
  original target order. A target whose `solve()` call raises gets
  `{"_error": "..."}` instead of aborting the whole batch.

  Each target is a fully independent problem -- this is deliberately just a
  parallel loop, not joint/hierarchical inference (there's no shared state
  or population-level parameters across targets; if you want that, you
  need a different tool, e.g. a proper hierarchical Bayesian model).

  Uses multiprocessing's `spawn` start method (not the Linux default
  `fork`) and caps each worker to a single BLAS thread
  (`OMP_NUM_THREADS=1` etc.) -- otherwise numpy/scipy's linear algebra
  backend can spawn its own threads *per worker process*, and N worker
  processes each doing that oversubscribes the machine's cores, eating
  into (or reversing) the speedup from parallelizing at all. Custom
  `priors` must be picklable (frozen `scipy.stats` distributions and the
  classes in `priors.py` are; anything JAX-jitted likely isn't).

## Input validation

`solve()` checks its inputs before running anything expensive:

- Unknown quantity names in `given` or `want` raise `KeyError` immediately
  (previously this could fail deep inside a sampler run, or silently on a
  `want` typo since it wasn't checked at all).
- Malformed `given` entries -- wrong tuple length, a non-positive or
  non-finite error, a non-finite value, an unrecognized type -- raise
  `ValueError`/`TypeError` with the offending name and value.
- Values that are physically impossible (negative mass/radius/Teff/parallax,
  negative extinction) raise `ValueError`.
- Values that are merely unusual (Teff outside ~3000-10000 K, [Fe/H]
  outside -2.5 to 0.75, etc.) raise a `UserWarning` rather than an error --
  might be exactly what you intend, but it's also the classic
  wrong-units/typo mistake, so it's flagged either way.
- For the point-estimate path (see below), if the given constraints turn
  out to be mutually inconsistent -- no combination of the free parameters
  can satisfy all of them -- a `UserWarning` reports the largest residual
  rather than silently returning a poor fit that looks like a real answer.

See `validation.py` for the exact rules (`_POSITIVE`, `_NONNEGATIVE`,
`_SANITY_RANGES`) if you want to adjust them.

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

## Performance note (JAX, and scipy.stats vs. hand-written priors)

`priors.py`'s distribution classes (`Normal`, `Uniform`, `Exponential`,
`TruncatedNormal`, `TruncatedPowerLaw`, `ParallaxPrior`) are hand-written
using `scipy.special` primitives directly (`erfinv`, `gammaincinv`, ...)
rather than `scipy.stats`'s frozen-distribution objects, which `Solver`
used originally. Benchmarked (scalar calls, i.e. dynesty's actual calling
pattern, one point at a time):

| distribution | scipy.stats | hand-written | speedup |
|---|---|---|---|
| Normal | 13,300 calls/s | 592,000 calls/s | ~44x |
| Uniform | 13,700 calls/s | 895,000 calls/s | ~65x |
| Exponential | 11,600 calls/s | 724,000 calls/s | ~62x |
| TruncatedNormal | 2,900 calls/s | 517,000 calls/s | ~180x |
| Parallax (Gamma-based) | 1,500 calls/s | 352,000 calls/s | ~230x |

This isn't scipy.stats being unpolished -- its overhead is the cost of a
very general, mostly batch-oriented interface, and it's *fast* when called
on whole arrays at once (see the JAX comparison below). It's just a poor
fit for dynesty's one-value-at-a-time calls specifically. In practice
dynesty's own per-iteration bookkeeping still dominates total `solve()`
time (this doesn't turn a 10s run into a 0.1s run), but it's a real,
non-negligible contributor, not a rounding error.

Custom priors/`given` distributions you supply yourself (a `scipy.stats`
object, or anything else with `.ppf`/`.logpdf`) work exactly as before --
this only changes what the *defaults* are built from.

`relations.py` uses a module-level `xp = np` that every function resolves
at call time, so the whole forward pass can be made JAX-traceable by
setting `relations.xp = jax.numpy` -- no forked codebase needed. This
isn't wired up by default: `dynesty`'s default sampler calls the likelihood
one point at a time, so `jax.jit`'s per-call dispatch overhead actually
made things ~1.7x *slower* in testing, not faster -- and the same
scalar-vs-batched pattern shows up for the distributions above: a
benchmarked jitted JAX `.ppf()` call (115,927 calls/s) was still ~9x
*slower* than the hand-written scipy.special version here (5,370,428
calls/s) at the scalar-call granularity dynesty actually uses, because
per-call dispatch overhead dominates at that scale regardless of whether
the function itself is jitted. JAX only wins once you can batch many
calls into one dispatch (412,034,383 values/s for JAX `vmap`+jit vs.
36,732,531 for scipy.stats and 80,736,547 for scipy.special, all batched)
-- which is the shape of a genuinely vectorized/batched sampler (e.g.
[jaxns](https://github.com/Joshuaalbert/jaxns), which vectorizes the
entire nested sampling loop in JAX/XLA), not a per-point speedup on the
current dynesty-based architecture.

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
