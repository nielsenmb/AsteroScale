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

## Convert to a mission response

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
amplitude_plato = ast.convert_bolometric_amplitude(
    amplitude_bolometric,
    given["Teff"],
    mission="PLATO",
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

For PLATO, AsteroScale uses the polynomial corrections from
[Lund, Ballot & Chaplin (2026)](https://arxiv.org/abs/2603.12750):

$$
c_{P\rightarrow\mathrm{bol}}(T_{\mathrm{eff}})=
\sum_{i=0}^{2}a_i(T_{\mathrm{eff}}-T_0)^i,
\qquad
A_{\mathrm{PLATO}}=\frac{A_{\mathrm{bol}}}
{c_{P\rightarrow\mathrm{bol}}}.
$$

Use `mission="PLATO"` for the normal cameras, which is the paper's default
meaning of PLATO. The blue and red fast-camera responses are available as
`mission="PLATO-FCB"` and `mission="PLATO-FCR"`, respectively.

| Response | $T_0$ (K) | $a_0$ | $a_1$ (K$^{-1}$) | $a_2$ (K$^{-2}$) |
|---|---:|---:|---:|---:|
| Normal cameras | 5446 | 1 | $1.512\times10^{-4}$ | $-4.229\times10^{-9}$ |
| Blue fast camera | 6137 | 1 | $1.451\times10^{-4}$ | $-3.530\times10^{-9}$ |
| Red fast camera | 4728 | 1 | $1.874\times10^{-4}$ | $-6.856\times10^{-9}$ |

The Kepler and PLATO relations cover approximately
$4000\leq T_{\mathrm{eff}}\leq7500$ K. The PLATO relations use Planck spectra
and the mission response estimates available before launch. Atmosphere
spectra, metallicity, surface gravity, and the final instrument throughput can
matter at higher precision.

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
