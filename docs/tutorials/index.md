# Tutorials

In AsteroScale, **fundamental quantities** are the six parameters used as the
basis of the forward model, including mass, radius, and effective temperature.
They are not necessarily fundamental in a philosophical or physical sense.
**Derived quantities** are calculated from them through the registered scaling
relations. Either kind can be supplied as a constraint or requested as an
output; see {doc}`../reference/quantities` for the complete tables.

You do not need to know every fundamental. AsteroScale marginalises over its
prior for a missing parameter when the remaining data constrain the requested
calculation. The current default priors are independent rather than a
population model of the Hertzsprung--Russell diagram. With weak constraints,
inspect the posterior and its prior sensitivity instead of assuming that every
sample represents a common type of real star.

The tutorials are organised around common questions:

- {doc}`predicting-oscillations`: Given basic stellar properties, where should
  oscillations and granulation appear?
- {doc}`inferring-stellar-properties`: What do measured global oscillations
  imply for mass, radius, density, and surface gravity?
- {doc}`gaia-constraints`: How can Gaia-like information complement seismic
  measurements, and where are the current photometric approximations unsafe?
- {doc}`real-star-comparisons`: How closely do scaling-relation results agree
  with published values?
- {doc}`priors-and-relations`: What shapes do the default priors have, and how
  do the implemented relations vary with stellar properties?
- {doc}`advanced-workflows`: How do I choose `input_mode`, control relation
  scatter, inspect validity, request all quantities, customize a prior, or run
  a batch?

```{toctree}
:maxdepth: 1
:hidden:

predicting-oscillations
inferring-stellar-properties
gaia-constraints
real-star-comparisons
priors-and-relations
advanced-workflows
```
