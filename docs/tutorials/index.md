# Tutorials

AsteroScale is a toolbox for estimating different stellar fundamental properties and observable quantities that are derived from them. The fundamental properties are for example mass, radius, and effective temperature. Note that these are not necessarily fundamental in a physical sense, but rather they form the basis upon which all the other observable quantities of interest are calculated. 

This doesn't mean, however, that these must always be known for you to use AsteroScale. You can supply any combination of parameters and AsteroScale will try to provide an answer by marginalizing over a set of priors for the other parameters. These priors are currently 1D and not correlated, so they may produce unphysical values if you don't provide enough constraints for the quantities of interest. In future updates we aim to incorporate more realistic priors based on populations synthesis models which capture correlations between parameters across the HR-diagram.

The following tutorials provide basic usage examples and are organised around common questions:

- {doc}`inferring-stellar-properties`: How do I use AsteroScale and what do measured oscillations imply for mass, radius, density and surface gravity?
- {doc}`predicting-oscillations`: If I know some basic stuff about my stars, where should oscillations and granulation appear?
- {doc}`gaia-constraints`: How can asteroseismology and Gaia data constrain the result?
- {doc}`real-star-comparisons`: How closely do scaling-relation results agree with published values?
- {doc}`priors-and-relations`: What shapes do the default priors have, and how do the scaling relations vary with stellar properties?

```{toctree}
:maxdepth: 1
:hidden:

predicting-oscillations
inferring-stellar-properties
gaia-constraints
real-star-comparisons
priors-and-relations
```
