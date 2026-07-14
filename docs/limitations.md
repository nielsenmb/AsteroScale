# Limitations

AsteroScale is a lightweight scaling-relation tool rather than a replacement
for detailed stellar modelling or an instrument-specific noise model.

- Scaling relations are approximate and may show systematic offsets for
  particular evolutionary states.
- The amplitude relations do not account for activity-related suppression,
  dilution, cadence attenuation or the observing window.
- The metallicity corrections are calibrated over approximately
  `-1.0 < [Fe/H] < 0.5`.
- Gaia bolometric corrections in the package are rough, near-solar
  placeholders and are unsuitable for precision photometry.
- BP-RP alone cannot independently constrain effective temperature and
  extinction.
- Weakly constrained results may be sensitive to the default field-star
  priors.
- Tight combinations of constraints may require more live points and
  explicit convergence checks.

Warnings about unusual values are intended to catch unit mistakes and
extrapolation. Check the input and calibration range before suppressing them.
