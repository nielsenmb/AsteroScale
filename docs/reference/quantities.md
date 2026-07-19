# Quantities and units

AsteroScale uses **fundamental** to mean a parameter from which its forward
relations are evaluated. This is a software distinction, not a claim that the
quantity is fundamental physics. A quantity in either table can be supplied in
`given` or requested in `want`; the solver determines which underlying
fundamentals are needed.

Mass, radius, luminosity, and mean density are expressed relative to the Sun.
Frequencies are in microhertz ($\mathrm{\mu Hz}$), where
$1\,\mathrm{\mu Hz}=10^{-6}\,\mathrm{Hz}$.

## Fundamental quantities

| Name | Meaning | Unit | Practical interpretation |
|---|---|---|---|
| `M` | Stellar mass | $M_\odot$ | One means one solar mass. |
| `R` | Stellar radius | $R_\odot$ | One means one solar radius. |
| `Teff` | Effective temperature | K | Photospheric temperature from spectroscopy or photometry. |
| `plx` | Parallax | mas | Used to calculate distance; it must be positive in the current model. |
| `A_G` | Gaia G-band extinction | mag | Dimming in the Gaia G band. Use an external constraint when available. |
| `FeH` | Metallicity $[\mathrm{Fe/H}]$ | dex | Zero is solar iron abundance; negative values are metal poor. |

## Derived quantities

| Name | Meaning | Unit | Calculated from |
|---|---|---|---|
| `numax` | Frequency at the peak of the oscillation-power envelope | $\mathrm{\mu Hz}$ | `M`, `R`, `Teff`, `FeH` |
| `dnu` | Average large frequency separation | $\mathrm{\mu Hz}$ | `M`, `R`, `Teff`, `FeH` |
| `L` | Bolometric luminosity | $L_\odot$ | `R`, `Teff` |
| `logg` | Base-10 surface gravity | dex, with $g$ in $\mathrm{cm\,s^{-2}}$ | `M`, `R` |
| `rho` | Mean density | $\rho_\odot$ | `M`, `R` |
| `FWHM_env` | Full width at half maximum of the oscillation envelope | $\mathrm{\mu Hz}$ | `numax`, `Teff` |
| `A_env` | Maximum radial-mode RMS amplitude in the selected bandpass | ppm | `M`, `L`, `Teff` |
| `A_gran` | Granulation RMS amplitude | ppm | `numax`, `M` |
| `b_gran_low` | Lower granulation characteristic frequency | $\mathrm{\mu Hz}$ | `numax` |
| `b_gran_high` | Higher granulation characteristic frequency | $\mathrm{\mu Hz}$ | `numax` |
| `d` | Distance | pc | `plx` |
| `Mbol` | Absolute bolometric magnitude | mag | `L` |
| `BC_G` | Approximate Gaia G bolometric correction | mag | `Teff` |
| `BC_BP` | Approximate Gaia BP bolometric correction | mag | `Teff` |
| `BC_RP` | Approximate Gaia RP bolometric correction | mag | `Teff` |
| `A_BP` | Approximate Gaia BP extinction | mag | `A_G` |
| `A_RP` | Approximate Gaia RP extinction | mag | `A_G` |
| `M_G` | Absolute Gaia G magnitude | mag | `Mbol`, `BC_G` |
| `M_BP` | Absolute Gaia BP magnitude | mag | `Mbol`, `BC_BP` |
| `M_RP` | Absolute Gaia RP magnitude | mag | `Mbol`, `BC_RP` |
| `G_mag` | Apparent Gaia G magnitude | mag | `M_G`, `d`, `A_G` |
| `BP_mag` | Apparent Gaia BP magnitude | mag | `M_BP`, `d`, `A_BP` |
| `RP_mag` | Apparent Gaia RP magnitude | mag | `M_RP`, `d`, `A_RP` |
| `BP_RP` | Gaia BP-RP colour | mag | `BP_mag`, `RP_mag` |

The Gaia bolometric corrections and extinction conversions are deliberately
rough, near-solar placeholders. They demonstrate how Gaia-like constraints can
enter a calculation but should not be used for precision photometric inference.

Use `want="all"` to return every available quantity whose dependencies are
part of the current problem. See {mod}`asteroscale.relations` for the relation
registry and {doc}`../limitations` before interpreting precision results.
