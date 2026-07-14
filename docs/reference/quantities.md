# Quantities and units

Mass, radius, luminosity and mean density are expressed in solar units.

| Name | Meaning | Unit |
|---|---|---|
| `M` | Stellar mass | solar masses |
| `R` | Stellar radius | solar radii |
| `Teff` | Effective temperature | K |
| `FeH` | Metallicity `[Fe/H]` | dex |
| `plx` | Parallax | mas |
| `A_G` | Gaia G-band extinction | mag |
| `numax` | Frequency of maximum oscillation power | microhertz |
| `dnu` | Average large frequency separation | microhertz |
| `FWHM_env` | Width of the oscillation power envelope | microhertz |
| `A_env` | Maximum radial-mode RMS amplitude | ppm |
| `A_gran` | Granulation RMS amplitude | ppm |
| `b_gran_low`, `b_gran_high` | Granulation characteristic frequencies | microhertz |
| `L` | Luminosity | solar luminosities |
| `rho` | Mean density | solar density |
| `logg` | Base-10 surface gravity in cgs units | dex |
| `d` | Distance | pc |

Use `want="all"` to return every available fundamental and derived
quantity. See {mod}`asteroscale.relations` for the complete relation registry.
