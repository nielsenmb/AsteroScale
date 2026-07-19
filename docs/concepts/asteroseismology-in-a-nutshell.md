# Asteroseismology in a nutshell

Cool stars are not perfectly quiet light sources. Near-surface convection produces granulation and excites many acoustic oscillation modes. In a power spectrum, the modes form a broad power excess rather than a single isolated frequency.

For exoplanet observations for example this variability can add correlated noise to transits, phase curves and radial-velocity measurements. The same signal can also constrain the host star and therefore improve planetary properties.

## Reading the main predictions

`numax` is the frequency at which the oscillation envelope is expected to peak. A useful first search interval is approximately

\[
\nu_{\max} - \frac{\mathrm{FWHM}_{\rm env}}{2}
\quad\text{to}\quad
\nu_{\max} + \frac{\mathrm{FWHM}_{\rm env}}{2}.
\]

This is a guide rather than a hard boundary. The individual modes form a structured pattern within the envelope, with consecutive radial orders
separated by approximately `dnu`.

## What AsteroScale does not predict

The package does not include instrumental noise, dilution, stellar activity, data gaps, cadence attenuation or the observing window. Consequently:

- a predicted amplitude does not guarantee a detection;
- `A_env` is not the total light-curve RMS;
- `A_gran` should not be added directly to a transit-depth uncertainty; and
- the predicted frequencies must be compared with the data's Nyquist
  frequency and frequency resolution.

See {doc}`amplitudes` for the distinction between radial-mode, bolometric and mission-specific amplitudes.
