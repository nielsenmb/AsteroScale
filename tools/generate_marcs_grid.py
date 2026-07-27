"""Generate AsteroScale's compact Gaia DR3 bolometric-correction grid.

This is a developer utility, not part of the runtime package.  It reads the
``grid`` directory distributed by Casagrande & VandenBerg's
``bolometric-corrections`` repository and regularizes the MARCS tables onto
the domain used by AsteroScale's default priors.

The source tables can be obtained from
https://github.com/casaluca/bolometric-corrections by unpacking
``grid.tar.gz``.  Run this script from the AsteroScale repository root:

    python tools/generate_marcs_grid.py /path/to/bolometric-corrections/grid

References
----------
Casagrande, L. & VandenBerg, D. A. 2014, MNRAS, 444, 392.
Casagrande, L. & VandenBerg, D. A. 2018a, MNRAS, 475, 5023.
Casagrande, L. & VandenBerg, D. A. 2018b, MNRAS, 479, L102.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.interpolate import Akima1DInterpolator, interp1d


TEFF_AXIS = np.arange(4000.0, 7000.1, 125.0)
LOGG_AXIS = np.arange(1.5, 5.0 + 0.01, 0.25)
FEH_AXIS = np.arange(-1.0, 0.5 + 0.001, 0.125)
MBOL_SOURCE = 4.75
MBOL_ASTERSCALE = 4.74


def _interpolate(x, y, target):
    """Interpolate vector-valued data without extrapolation."""
    if len(x) > 2:
        return Akima1DInterpolator(x, y, axis=0, extrapolate=False)(target)
    if len(x) == 2:
        return interp1d(
            x, y, axis=0, bounds_error=False, fill_value=np.nan
        )(target)
    return np.full(np.shape(target) + np.shape(y)[1:], np.nan)


def _load_source(grid_dir):
    """Load the R_V=3.1 Gaia DR3 source tables."""
    files = sorted(grid_dir.glob("STcolors*Rv3.1_EBV_*.dat"))
    files = [path for path in files if float(path.stem[-4:]) < 0.72]
    if not files:
        raise FileNotFoundError(
            f"No R_V=3.1 MARCS tables were found under {grid_dir}."
        )

    ebv = np.array([float(path.stem[-4:]) for path in files])
    order = np.argsort(ebv)
    files = [files[index] for index in order]
    ebv = ebv[order]

    coordinates = None
    values = []
    for path in files:
        table = np.genfromtxt(path, names=True)
        current = np.column_stack((table["Teff"], table["logg"], table["feh"]))
        if coordinates is None:
            coordinates = current
        elif not np.array_equal(current, coordinates):
            raise ValueError(f"Grid coordinates differ in {path}.")
        values.append(
            np.column_stack(
                (
                    table["mbol"] - table["G3"],
                    table["mbol"] - table["BP3"],
                    table["mbol"] - table["RP3"],
                )
            )
        )
    return coordinates, ebv, np.asarray(values)


def _regularize_spatial(coordinates, values):
    """Apply MARCS-style sequential interpolation onto regular spatial axes."""
    source_teff = np.unique(coordinates[:, 0])
    source_logg = np.unique(coordinates[:, 1])
    source_feh = np.unique(coordinates[:, 2])
    n_ebv, _, n_filter = values.shape

    # First interpolate Teff at every available ([Fe/H], logg) combination.
    teff_stage = np.full(
        (
            len(source_feh),
            len(source_logg),
            len(TEFF_AXIS),
            n_ebv,
            n_filter,
        ),
        np.nan,
    )
    for i, feh in enumerate(source_feh):
        for j, logg in enumerate(source_logg):
            mask = (coordinates[:, 2] == feh) & (coordinates[:, 1] == logg)
            if np.count_nonzero(mask) < 2:
                continue
            x = coordinates[mask, 0]
            order = np.argsort(x)
            y = np.moveaxis(values[:, mask, :][:, order, :], 1, 0)
            teff_stage[i, j] = _interpolate(x[order], y, TEFF_AXIS)

    # Then interpolate logg, retaining only finite Teff interpolants.
    logg_stage = np.full(
        (
            len(source_feh),
            len(TEFF_AXIS),
            len(LOGG_AXIS),
            n_ebv,
            n_filter,
        ),
        np.nan,
    )
    for i in range(len(source_feh)):
        for j in range(len(TEFF_AXIS)):
            y = teff_stage[i, :, j]
            mask = np.all(np.isfinite(y), axis=(1, 2))
            if np.count_nonzero(mask) < 2:
                continue
            logg_stage[i, j] = _interpolate(
                source_logg[mask], y[mask], LOGG_AXIS
            )

    # Finally interpolate metallicity.
    regular = np.full(
        (
            len(TEFF_AXIS),
            len(LOGG_AXIS),
            len(FEH_AXIS),
            n_ebv,
            n_filter,
        ),
        np.nan,
    )
    for i in range(len(TEFF_AXIS)):
        for j in range(len(LOGG_AXIS)):
            y = logg_stage[:, i, j]
            mask = np.all(np.isfinite(y), axis=(1, 2))
            if np.count_nonzero(mask) < 2:
                continue
            regular[i, j] = np.moveaxis(
                _interpolate(source_feh[mask], y[mask], FEH_AXIS), 0, 0
            )

    if not np.all(np.isfinite(regular)):
        bad = np.argwhere(~np.all(np.isfinite(regular), axis=(3, 4)))
        bad_coordinates = [
            (
                float(TEFF_AXIS[i]),
                float(LOGG_AXIS[j]),
                float(FEH_AXIS[k]),
            )
            for i, j, k in bad[:20]
        ]
        raise ValueError(
            "The requested compact domain is not fully covered by the MARCS "
            f"tables. First invalid coordinates: {bad_coordinates}"
        )
    return regular


def _regrid_extinction(regular, ebv):
    """Express BP/RP extinction as a function of the public A_G parameter."""
    zero_index = np.flatnonzero(ebv == 0.0)
    if len(zero_index) != 1:
        raise ValueError("The source grid must contain exactly one E(B-V)=0 table.")
    unreddened = regular[..., zero_index[0], :]
    extinction = unreddened[..., None, :] - regular

    # Every atmosphere has a slightly different A_G(E(B-V)) curve.  Use only
    # the common A_G range so runtime interpolation never extrapolates.
    common_max = float(np.floor(np.min(extinction[..., -1, 0]) * 10.0) / 10.0)
    ag_axis = np.arange(0.0, common_max + 0.001, 0.1)
    bp_rp = np.empty(regular.shape[:3] + (len(ag_axis), 2))
    for index in np.ndindex(regular.shape[:3]):
        local_ag = extinction[index + (slice(None), 0)]
        if np.any(np.diff(local_ag) <= 0.0):
            raise ValueError(f"Non-monotonic A_G curve at grid index {index}.")
        for output_index, source_index in enumerate((1, 2)):
            local_ax = extinction[index + (slice(None), source_index)]
            bp_rp[index + (slice(None), output_index)] = np.interp(
                ag_axis, local_ag, local_ax
            )
    return ag_axis, bp_rp


def generate(grid_dir, output):
    """Generate and save the compact runtime table."""
    coordinates, ebv, values = _load_source(grid_dir)
    regular = _regularize_spatial(coordinates, values)
    zero_index = int(np.flatnonzero(ebv == 0.0)[0])
    bc = regular[..., zero_index, :]
    bc += MBOL_ASTERSCALE - MBOL_SOURCE
    ag_axis, extinction = _regrid_extinction(regular, ebv)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        teff=TEFF_AXIS.astype(np.float32),
        logg=LOGG_AXIS.astype(np.float32),
        feh=FEH_AXIS.astype(np.float32),
        ag=ag_axis.astype(np.float32),
        bc=np.moveaxis(bc, -1, 0).astype(np.float32),
        extinction=np.moveaxis(extinction, -1, 0).astype(np.float32),
    )
    print(f"Wrote {output} ({output.stat().st_size / 1024:.1f} KiB)")
    print(f"A_G domain: {ag_axis[0]:.1f} to {ag_axis[-1]:.1f} mag")


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("grid_dir", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("asteroscale/data/marcs_gaia_dr3.npz"),
    )
    args = parser.parse_args()
    generate(args.grid_dir, args.output)


if __name__ == "__main__":
    main()
