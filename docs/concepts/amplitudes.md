# Oscillation amplitudes and bandpasses

AsteroScale reports `A_env` as the maximum **radial-mode RMS amplitude** in the selected instrumental band. It is not the integrated amplitude of all modes in the oscillation envelope.

Use either a solver-wide default or a per-call override:

```python
solver = ast.Solver(bandpass="Kepler")
kepler = solver.solve(given, want=["A_env"])
tess = solver.solve(given, want=["A_env"], bandpass="TESS")
```

TESS is the default. The adopted solar normalisations are 2.1 ppm for TESS and 2.5 ppm for Kepler before applying the empirical hot-star suppression
factor in the Ball et al. relation.

## Approximate bolometric conversion

The Kepler response correction adopted by Huber et al. (2011) is

\[
c_K(T_{\rm eff}) =
\left(\frac{T_{\rm eff}}{5934\,{\rm K}}\right)^{0.8}.
\]

An approximate conversion from Kepler RMS amplitude is

\[
A_{\rm bol,rms} \simeq A_{\rm Kp}\,c_K(T_{\rm eff}).
\]

Using the approximate response ratio from Campante et al. (2016),

\[
A_{\rm TESS}\simeq0.85A_{\rm Kp},
\]

and therefore

\[
A_{\rm bol,rms}\simeq
A_{\rm TESS}\frac{c_K(T_{\rm eff})}{0.85}.
\]

Peak amplitudes are larger than RMS amplitudes by a factor of \(\sqrt{2}\). These are empirical conversions: solar normalisations differ by a few percent between publications, so they should not be treated as exact changes of unit.
