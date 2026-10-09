# Scientific assumptions and review fixes

AsteroScale is a scaling-relation calculator and approximate inference tool.
Passing its numerical tests establishes implementation consistency; it does not
establish calibrated masses, radii or interval coverage for all stellar populations.

## Seismic calibration choices

The default `dnu_calibration="solar_anchored"` multiplies the
[Guggenberger et al. (2016)](https://arxiv.org/abs/1606.01917) reference function
by 135.1/138.2910855 = 0.976924865. This is an AsteroScale modification of the
published coefficients, motivated by solar agreement. It is not a validated
universal surface correction. At fixed seismic observations, this changes
inferred radius by a factor 0.95438 and mass by 0.91085 relative to using the
original reference function.

Compare the alternatives explicitly:

```python
from asteroscale import Solver

adopted = Solver(input_mode="likelihood", dnu_calibration="solar_anchored")
published = Solver(input_mode="likelihood", dnu_calibration="guggenberger2016")
```

The `numax_correction="mu"` default uses an approximate **neutral** H/He
particle-count prescription, with an assumed helium-enrichment law and fixed
solar Gamma_1. It is not a fully ionized molecular weight or the complete
[Viani et al. (2017)](https://arxiv.org/abs/1705.03472) calculation. Helium,
ionization and atmospheric structure cannot be inferred from metallicity alone.
Use `numax_correction="none"` to test sensitivity to this approximation. These
options are also supported by `ast.solve` and `ast.solve_many`.

`within_checked_bounds` means that the implemented numerical cuts passed.
It does not certify calibration. In particular, evolutionary state is absent;
early main-sequence and core-helium-burning stars are outside the published
Guggenberger calibration even if their numerical values pass the checks.
Envelope-width and granulation applicability is explicitly reported as
unverified. The amplitude check includes the 4000–7500 K range of its Kepler
bolometric response approximation.

## Meaning of the amplitudes and scatter

`amplitude_bolometric` is the maximum radial-mode RMS oscillation amplitude.
The deprecated `A_env` is a deterministic mission conversion of the same
latent amplitude, including exactly the same scatter draw.

`A_gran` retains the [Kallinger et al. (2014)](https://arxiv.org/abs/1408.0817)
**per-component Kepler RMS** convention. For the two equal-amplitude components,
the total Kepler RMS is `sqrt(2) * A_gran`; a bolometric total additionally
requires the Kepler bolometric correction. The characteristic frequencies are
those of their super-Lorentzian components. Do not substitute these into a
Harvey profile with a different frequency or amplitude normalization.

The empirical scatters are not universal intrinsic noise parameters. The 2%
numax and 1.5% dnu floors remain provisional. Multiplying a central relation by
`exp(Normal(0, sigma_log))` makes the central relation the **median**, not the
mean; the mean is larger by `exp(sigma_log**2 / 2)`. Offsets are independent.
Only supported positive empirical relations accept `relation_scatter`; identity
relations, magnitudes, colours and logarithms reject it.

`precise` changes both sampler settings and the uncertainty model by default.
For a numerical convergence comparison, hold `relation_scatter` fixed across
presets. Reliable scientific coverage requires a separate calibration sample
and repeated injection/recovery experiments; this review did not supply those.

## What the bundled population represents

The bundled 256-component model was fitted to 20,913 retained TRILEGAL 1.6
rows from **11 directions**. The intended 12-direction grid omitted
**l=0, b=0 deliberately**: that sightline produced gigabytes of raw output even
at 0.5 deg², while reducing the common field size further left too few stars
in other fields. This is a practical sampling decision, not a symmetry argument.
The omitted direction is not reconstructed or reweighted in the existing model.
Consequently, this is an incomplete directional approximation to a nearby
population, not an exactly integrated all-sky prior.

The original metadata incorrectly asserted 10 deg² for the downloaded fields.
That assertion has been removed. The 0.5 deg² size discussed for the omitted
field is recorded separately; actual per-file areas remain unverified without
the submitted forms or raw catalogue records. Binary output representation is
also unresolved. These unknowns are explicit in both the JSON report and NPZ
metadata; no numerical GMM parameters were changed during this provenance fix.
The radius cut `(0.09, 1000)` is recorded from the training notebook.

Zero `A_V(infinity)` is intentional and removes dust throughout the simulation.
Dust-free does not mean volume-complete: the G<20 limit can remove intrinsically
faint stars even within 1 kpc. The training temperature and radius cuts also
select the population. TRILEGAL [M/H] is stored in the `feh` coordinate;
identifying it with spectroscopic [Fe/H] is approximate for alpha-enhanced stars.
Unresolved binaries can also make a system row inconsistent with a single-star
mass/radius/temperature interpretation.

GMMs have unbounded Gaussian tails. The training box is diagnostic, not an
implicit truncation. Returned metadata reports the fraction inside the box
projected onto available coordinates; it cannot detect departures from a thin
stellar-evolution sequence *inside* that box. Marginal and narrow-slice checks,
prior sensitivity, evolutionary-state tests and posterior coverage remain
necessary before adopting this prior for a scientific sample. A likelihood
improvement through 256 components establishes the selection within this search,
not optimal complexity or scientific validity.

For future files, `read_trilegal(..., chunk_size=100_000)` filters each raw block
before retaining it. This bounds raw-input memory use; the retained catalogue
still occupies memory. It records SHA-256, raw/retained counts and all cuts.
Supply `field_areas_deg2` in the same order as the paths when areas differ;
weights become inverse area, representing equal solid angle per direction.
This weighting cannot compensate for an absent direction. If later downsampling
within a field, multiply weights by retained-count/sample-count. Avoid equalizing
star counts between directions unless that is the intended target distribution.

## Statistical and operational safeguards

- Exact inversions require enough independent constraints, successful optimization,
  a numerically full-rank scaled Jacobian and agreement with every exact target.
  This local rank check does not prove global uniqueness of nonlinear inversions.
- Conflicting exact observations raise an error. Noisy observations against a
  fully fixed model are still evaluated; Gaussian conflicts above five sigma warn
  and standardized residuals are included in metadata.
- Likelihood means can be negative even for positive physical quantities, such
  as noisy Gaia parallaxes. Latent and propagated physical samples must remain
  valid. Use bounded propagation distributions when a Gaussian assigns appreciable
  probability to unphysical values; invalid draws raise rather than being clipped.
- Fundamental uncertainties in `propagate` replace priors. Derived uncertain
  observations remain likelihood terms, so that mode can be a hybrid calculation.
  Input covariances and relation-residual covariance are not implemented. Do not
  multiply terms derived from the same measurements as if independent. The full
  BP/RP/colour triple and the two amplitude aliases are explicitly rejected as
  independent likelihood terms.
- Tuple, SciPy Normal and Baldr Normal photometric inputs receive the same model
  floor. Other distributions warn if the floor cannot be applied; incorporate
  model uncertainty explicitly before supplying them.
- Exact fundamental observations alone can condition a GMM in likelihood mode
  and draw its remaining coordinates directly. Exact derived inversions with
  missing fundamentals reject an enabled GMM rather than silently ignoring it.
  Propagate mode leaves the GMM
  inactive. `want="all"` selects outputs available from the problem's fundamental
  dependencies without adding unrelated parallax/extinction parameters.
- `return_metadata=True` adds `_metadata`; `solver.last_metadata` is always
  available. It records input mode, active population prior, calibration choices,
  scatter, photometric floor and raw-sample column names. Raw fundamentals use
  physical units; columns prefixed `scatter_z:` are standard-normal latent offsets.
- The solver seed now controls equal-weight posterior resampling as well as
  Dynesty. Reproducibility assumes the same software versions, inputs and call order.
- Model loading validates format, array dimensions, finite values, weights,
  positive scales, support bounds and positive-definite covariance matrices.
  EM reports convergence, fitting settings and a likelihood evaluated at the
  parameters actually returned. Historical model convergence is not inferred
  retroactively from its iteration count.
- Batch workers limit already-loaded numerical thread pools at runtime. Per-target
  construction failures are returned through the same `_error` path as solve failures.

The bundled MARCS table was regenerated after fixing a NumPy view mutation:
changing the bolometric zero point no longer changes the extinction calculation.
The original atmosphere source and interpolation choices are retained. The fix
changes extinction corrections by at most about 0.0042 mag across the bundled
grid; it is a numerical correction, not a new extinction calibration.
