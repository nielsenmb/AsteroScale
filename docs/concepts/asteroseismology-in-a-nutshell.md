# Asteroseismology in a nutshell

Asteroseismology uses a star's natural oscillations to learn about the star,
much as seismology uses earthquakes to study the Earth. AsteroScale focuses on
**solar-like oscillators**: cool main-sequence stars, subgiants, and red giants
with outer convection zones. Turbulent convection continually excites and
damps acoustic pressure modes, so the signal is a changing pattern of many
peaks rather than one perfectly coherent sinusoid.

The same convection also produces **granulation**. In a light curve,
granulation and oscillations are correlated stellar variability. In a power
spectrum, granulation forms a sloping background and the oscillation modes sit
in a broad hump above it. For exoplanet observations these signals can add
noise to transits, phase curves, and radial-velocity measurements; when they
are detected, they can also improve the host-star mass and radius and therefore
the inferred planet properties.

## From a light curve to a power spectrum

A light curve records brightness against time. A power spectrum rearranges its
variance by frequency: a peak shows a timescale on which the star varies. The
frequencies used here are in microhertz ($\mathrm{\mu Hz}$):

$$
1\,\mathrm{\mu Hz}=10^{-6}\,\mathrm{Hz},
\qquad
P=\frac{1}{\nu}.
$$

For orientation, $1\,\mathrm{\mu Hz}$ corresponds to a period of about
11.6 days, while the solar $\nu_{\max}\simeq3090\,\mathrm{\mu Hz}$ corresponds
to about 5.4 minutes. AsteroScale predicts frequency locations and empirical
amplitudes; it does not calculate a power spectrum from a light curve.

## Reading the main predictions

`numax` ($\nu_{\max}$) is the frequency at which the broad oscillation-power
envelope is expected to peak. It scales approximately with surface gravity:

$$
\nu_{\max}\propto\frac{M}{R^2\sqrt{T_{\mathrm{eff}}}}.
$$

`FWHM_env` is the full width of the envelope measured where the approximate
envelope falls to half its maximum. A useful first search interval is

$$
\nu_{\max}-\frac{\mathrm{FWHM}_{\mathrm{env}}}{2}
\quad\text{to}\quad
\nu_{\max}+\frac{\mathrm{FWHM}_{\mathrm{env}}}{2}.
$$

This is a guide, not a hard boundary. Individual modes form a repeated pattern
within the envelope. `dnu` ($\Delta\nu$), the **large frequency separation**,
is the average spacing between consecutive radial orders of the same angular
degree. It mainly traces mean density:

$$
\Delta\nu\propto\sqrt{\frac{M}{R^3}}
=\sqrt{\frac{\bar\rho}{\bar\rho_\odot}}\,\Delta\nu_\odot.
$$

This is why measuring both $\nu_{\max}$ and $\Delta\nu$, together with an
effective temperature, can constrain mass and radius. AsteroScale uses global
quantities only; it does not fit individual mode frequencies or infer stellar
age.

`A_env` is the predicted RMS brightness amplitude of the strongest radial mode
near $\nu_{\max}$, not the height or integrated area of the entire oscillation
envelope. `A_gran` describes the empirical granulation RMS scale. See
{doc}`amplitudes` before comparing these values with a time-domain RMS or a
power-spectrum ordinate.

## Before looking for the signal

Two properties of the observations matter immediately:

- the Nyquist frequency is approximately $1/(2\,\mathrm{cadence})$ for evenly
  sampled data, so signals above it may be aliased;
- the frequency resolution is approximately $1/T$, where $T$ is the observing
  baseline, so a short light curve cannot resolve closely spaced modes.

Detectability also depends on target brightness, instrumental noise, dilution,
stellar activity, data gaps, and the observing window. Consequently, a
prediction inside the observable frequency range still does not guarantee a
detection.

## Where AsteroScale stops

AsteroScale is a scaling-relation calculator and lightweight inference tool.
It does not currently model:

- an instrument-specific noise spectrum or detection probability;
- cadence attenuation, dilution, data gaps, or the observing window;
- individual oscillation modes, rotational splittings, or mixed modes;
- stellar evolutionary tracks or ages; or
- activity-dependent suppression beyond the empirical amplitude relation.

Continue with {doc}`../getting-started`, consult the complete
{doc}`../reference/quantities` table, or see the review papers on the
{doc}`../reference/references` page for a deeper introduction.
