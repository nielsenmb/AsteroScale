# AsteroScale

AsteroScale predicts where solar-like stellar oscillations and granulation
should appear and how large they may be. It can also use measured
asteroseismic quantities to estimate stellar mass, radius, density and
surface gravity.

The documentation is written for users who may have little or no prior
experience with asteroseismology, particularly people studying exoplanet
systems who need to understand stellar variability in their data.

AsteroScale is developed openly in the
[GitHub repository](https://github.com/nielsenmb/AsteroScale), where you can
inspect the source, report problems, and contribute improvements.

::::{grid} 1 2 2 2
:::{grid-item-card} Start here
:link: getting-started
:link-type: doc

Predict an oscillation frequency range for an exoplanet host or infer basic
stellar properties from measured oscillations.
:::
:::{grid-item-card} Asteroseismology for exoplanet users
:link: concepts/asteroseismology-for-exoplanets
:link-type: doc

Learn what the predicted frequencies and amplitudes mean in an observed
light curve or power spectrum.
:::
:::{grid-item-card} Tutorials
:link: tutorials/index
:link-type: doc

Work through common use cases and comparisons with real stars.
:::
:::{grid-item-card} Priors and relations
:link: tutorials/priors-and-relations
:link-type: doc

Inspect the default prior distributions and visualize how the implemented
scaling relations vary across their inputs.
:::
:::{grid-item-card} API reference
:link: reference/api
:link-type: doc

Detailed documentation generated from the package's NumPy-style docstrings.
:::
::::

```{toctree}
:maxdepth: 2
:hidden:

installation
getting-started
concepts/index
tutorials/index
reference/index
limitations
```
