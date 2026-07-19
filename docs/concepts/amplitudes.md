# Oscillation amplitudes and bandpasses

AsteroScale reports `A_env` as the maximum **radial-mode RMS amplitude** in the
selected photometric band. For example, `A_env = 3 ppm` means that the
root-mean-square brightness variation of the strongest radial mode is about
three parts per million. It is not the integrated amplitude of every mode in
the envelope, a power-spectral-density height, or the total light-curve RMS.

Use either a solver-wide default or a per-call override:

```python
import asteroscale as ast

solver = ast.Solver(bandpass="Kepler")
kepler = solver.solve(given, want=["A_env"])
tess = solver.solve(given, want=["A_env"], bandpass="TESS")
```

TESS is the default. The adopted solar normalisations are 2.1 ppm for TESS and
2.5 ppm for Kepler before applying the empirical hot-star suppression in
[Ball et al. (2018)](https://ui.adsabs.harvard.edu/abs/2018ApJS..239...34B/abstract).
The Kepler amplitude is slightly larger because its response is bluer.

## Approximate bolometric conversion

A bolometric amplitude describes the brightness variation integrated over all
wavelengths. A detector measures a bandpass-specific amplitude instead. The
Kepler response correction adopted by
[Huber et al. (2011)](https://ui.adsabs.harvard.edu/abs/2011ApJ...743..143H/abstract)
is

$$
c_K(T_{\mathrm{eff}})=
\left(\frac{T_{\mathrm{eff}}}{5934\,\mathrm{K}}\right)^{0.8}.
$$

An approximate conversion from Kepler RMS amplitude is

$$
A_{\mathrm{bol,rms}}\simeq A_{\mathrm{Kp}}\,c_K(T_{\mathrm{eff}}).
$$

Using the approximate response ratio from
[Campante et al. (2016)](https://ui.adsabs.harvard.edu/abs/2016ApJ...830..138C/abstract),

$$
A_{\mathrm{TESS}}\simeq0.85A_{\mathrm{Kp}},
$$

and therefore

$$
A_{\mathrm{bol,rms}}\simeq
A_{\mathrm{TESS}}\frac{c_K(T_{\mathrm{eff}})}{0.85}.
$$

Peak sinusoidal amplitudes are larger than RMS amplitudes by a factor of
$\sqrt{2}$. These are empirical conversions, not exact changes of unit:
published solar normalisations differ by a few percent, and real amplitudes
also vary with activity and observing conditions.

## Amplitude is not detectability

To assess whether a signal is observable, compare an instrument-appropriate
oscillation model with the local background and noise in the power spectrum.
Do not add `A_env` or `A_gran` directly to a transit-depth error budget. The
{doc}`../limitations` page lists the missing instrumental effects.
