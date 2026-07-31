# Oscillation amplitudes and mission responses

AsteroScale reports `amplitude_bolometric` as the maximum **bolometric
radial-mode RMS amplitude**. For example, a value of 3 ppm means that the
root-mean-square brightness variation of the strongest radial mode would be
three parts per million after correcting for the wavelength response of a
particular detector. It is not the integrated amplitude of every mode in the
envelope, a power-spectral-density height, or the total light-curve RMS.

Keeping this quantity bolometric separates the stellar prediction from the
observing instrument. It is analogous to reporting intrinsic luminosity and
applying a filter response only when an observation is being modelled.

## Convert to TESS or Kepler

Use {func}`asteroscale.convert_bolometric_amplitude` after solving:

```python
import asteroscale as ast

prediction = ast.solve(
    given,
    want=["amplitude_bolometric"],
)

amplitude_bolometric = prediction["amplitude_bolometric"]
amplitude_kepler = ast.convert_bolometric_amplitude(
    amplitude_bolometric,
    given["Teff"],
    mission="Kepler",
)
amplitude_tess = ast.convert_bolometric_amplitude(
    amplitude_bolometric,
    given["Teff"],
    mission="TESS",
)
```

The [Ballot et al. (2011)](https://ui.adsabs.harvard.edu/abs/2011A%26A...531A.124B/abstract)
power-law correction is

$$
c_{K\rightarrow\mathrm{bol}}(T_{\mathrm{eff}})=
\left(\frac{T_{\mathrm{eff}}}{5934\,\mathrm{K}}\right)^{0.8},
$$

with

$$
A_{\mathrm{bol}}=A_{\mathrm{Kepler}}
c_{K\rightarrow\mathrm{bol}},
\qquad
A_{\mathrm{Kepler}}=\frac{A_{\mathrm{bol}}}
{c_{K\rightarrow\mathrm{bol}}}.
$$

For TESS, AsteroScale retains the solar-response ratio used by
[Ball et al. (2018)](https://ui.adsabs.harvard.edu/abs/2018ApJS..239...34B/abstract):

$$
A_{\mathrm{TESS}}=\frac{2.1}{2.5}A_{\mathrm{Kepler}}
=0.84A_{\mathrm{Kepler}}.
$$

The Ballot correction was calibrated over approximately
$4000\leq T_{\mathrm{eff}}\leq7500$ K. These conversions are empirical
approximations: atmosphere spectra, metallicity, surface gravity, and the
precise instrument throughput can matter at higher precision.

## Migrating older code

The solver's `bandpass` argument and the `A_env` output are deprecated. They
temporarily retain the previous mission-specific behaviour, but emit a
`FutureWarning`:

```python
# Deprecated compatibility path
legacy = ast.solve(given, want=["A_env"], bandpass="TESS")
```

New code should request `amplitude_bolometric` and convert it explicitly. This
makes the chosen detector visible at the point where it actually matters.

Peak sinusoidal amplitudes are larger than RMS amplitudes by a factor of
$\sqrt{2}$. Published solar normalisations differ by a few percent, and real
amplitudes also vary with activity and observing conditions.

## Amplitude is not detectability

To assess whether a signal is observable, compare an instrument-appropriate
oscillation model with the local background and noise in the power spectrum.
Do not add `amplitude_bolometric` or `A_gran` directly to a transit-depth error
budget. The {doc}`../limitations` page lists the missing instrumental effects.
