# Gaia DR3 bolometric-correction table

`marcs_gaia_dr3.npz` is a compact, regularized derivative of the MARCS-based
bolometric-correction tables distributed by Luca Casagrande at
https://github.com/casaluca/bolometric-corrections.
The checked-in file was generated from source commit
`5f8a2f8e214806303350179a1095583af061fac2`.

The table contains unreddened Gaia DR3 G, BP, and RP bolometric corrections,
plus BP and RP extinction as a function of G-band extinction. It assumes
`R_V = 3.1` and the source grid's standard alpha-enhancement prescription.
The source convention `Mbol_sun = 4.75` has been shifted by -0.01 mag to match
AsteroScale's `Mbol_sun = 4.74`.

Please cite:

- Casagrande & VandenBerg (2014), MNRAS, 444, 392
- Casagrande & VandenBerg (2018a), MNRAS, 475, 5023
- Casagrande & VandenBerg (2018b), MNRAS, 479, L102

The checked-in table can be reproduced with:

```bash
python tools/generate_marcs_grid.py /path/to/bolometric-corrections/grid
```
