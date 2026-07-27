"""Command-line interface for offline population-prior training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .catalogue import (
    read_standard_catalogue,
    read_trilegal,
    write_standard_catalogue,
)
from .gmm import fit_candidate_models


def _component_counts(value):
    """Parse a comma-separated list of component counts."""
    try:
        counts = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "component counts must be comma-separated integers"
        ) from error
    if not counts or any(count < 1 for count in counts):
        raise argparse.ArgumentTypeError("component counts must be positive")
    return counts


def _write_diagnostic_plot(catalogue, model, path, random_state):
    """Compare catalogue and fitted-model projections."""
    import matplotlib.pyplot as plt

    coordinates = catalogue.coordinates
    generated = model.sample(min(len(catalogue), 100_000), random_state)
    physical = {
        "catalogue": (
            10.0 ** coordinates[:, 0],
            10.0 ** coordinates[:, 1],
            10.0 ** coordinates[:, 2],
            coordinates[:, 3],
        ),
        "GMM": (
            10.0 ** generated[:, 0],
            10.0 ** generated[:, 1],
            10.0 ** generated[:, 2],
            generated[:, 3],
        ),
    }
    figure, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    pairs = (
        (2, 1, r"$T_{\rm eff}$ [K]", r"$R/R_\odot$"),
        (0, 1, r"$M/M_\odot$", r"$R/R_\odot$"),
        (2, 3, r"$T_{\rm eff}$ [K]", r"$[\mathrm{Fe/H}]$"),
    )
    for row, (label, values) in enumerate(physical.items()):
        for axis, (x_index, y_index, x_label, y_label) in zip(axes[row], pairs):
            axis.hexbin(
                values[x_index],
                values[y_index],
                gridsize=55,
                mincnt=1,
                bins="log",
            )
            axis.set(xlabel=x_label, ylabel=y_label, title=label)
        axes[row, 0].invert_xaxis()
        axes[row, 2].invert_xaxis()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def build_parser():
    """Build the training command parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Normalize a population-synthesis catalogue and fit a portable "
            "full-covariance Gaussian-mixture prior."
        )
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument(
        "--adapter", choices=("standard", "trilegal"), default="standard"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--components", type=_component_counts, default=[8, 16, 32, 64]
    )
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--selection-tolerance", type=float, default=0.01)
    parser.add_argument("--n-init", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--reg-covar", type=float, default=1e-6)
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-distance-pc", type=float)
    parser.add_argument("--teff-min", type=float)
    parser.add_argument("--teff-max", type=float)
    parser.add_argument(
        "--normalized-catalogue",
        type=Path,
        help="Optionally save the adapted catalogue as portable CSV.",
    )
    parser.add_argument(
        "--diagnostic-plot",
        type=Path,
        help="Optionally save catalogue-versus-GMM projection plots.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="JSON report path; defaults to OUTPUT with a .json suffix.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help=(
            "JSON object containing synthesis version, generation settings, "
            "selection assumptions, and citations to preserve in the model."
        ),
    )
    return parser


def main(argv=None):
    """Train and save a portable population GMM."""
    args = build_parser().parse_args(argv)
    if args.adapter == "standard":
        if len(args.inputs) != 1:
            raise SystemExit("The standard adapter accepts exactly one CSV file.")
        catalogue = read_standard_catalogue(args.inputs[0])
    else:
        teff_range = None
        if args.teff_min is not None or args.teff_max is not None:
            if args.teff_min is None or args.teff_max is None:
                raise SystemExit("--teff-min and --teff-max must be used together.")
            teff_range = (args.teff_min, args.teff_max)
        catalogue = read_trilegal(
            args.inputs,
            max_distance_pc=args.max_distance_pc,
            teff_range=teff_range,
        )

    if args.metadata is not None:
        user_metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
        if not isinstance(user_metadata, dict):
            raise SystemExit("--metadata must contain a JSON object.")
        catalogue.metadata["generation"] = user_metadata
    if args.normalized_catalogue is not None:
        write_standard_catalogue(catalogue, args.normalized_catalogue)
    model, candidates = fit_candidate_models(
        catalogue.coordinates,
        catalogue.weight,
        args.components,
        validation_fraction=args.validation_fraction,
        selection_tolerance=args.selection_tolerance,
        random_state=args.seed,
        n_init=args.n_init,
        max_iter=args.max_iter,
        reg_covar=args.reg_covar,
        batch_size=args.batch_size,
    )
    model.metadata["catalogue"] = catalogue.metadata
    model.save(args.output)

    report = {
        "output": str(args.output),
        "selected_components": model.metadata["n_components"],
        "n_catalogue_rows": len(catalogue),
        "candidates": candidates,
        "coordinate_bounds": np.asarray(model.support_bounds).tolist(),
        "catalogue_metadata": catalogue.metadata,
    }
    report_path = args.report or args.output.with_suffix(".json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    if args.diagnostic_plot is not None:
        _write_diagnostic_plot(
            catalogue, model, args.diagnostic_plot, args.seed
        )
    print(
        f"Wrote {args.output} with {model.metadata['n_components']} components "
        f"from {len(catalogue)} catalogue rows."
    )


if __name__ == "__main__":
    main()
