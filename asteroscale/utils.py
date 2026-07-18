"""Small convenience helpers for reporting solver output."""
import numpy as np


def summarize(samples, params=None):
    """Print summary statistics for solver samples.

    Parameters
    ----------
    samples : dict
        Mapping from quantity names to sample arrays, as returned by
        :meth:`asteroscale.Solver.solve`.
    params : sequence of str, optional
        Quantities to include. By default, all keys not starting with an
        underscore are summarized.
    """
    if params is None:
        params = [k for k in samples if not k.startswith("_")]

    header = f"{'param':<10}{'mean':>12}{'std':>12}{'p16':>12}{'p50':>12}{'p84':>12}"
    print(header)
    print("-" * len(header))
    for name in params:
        x = np.asarray(samples[name])
        mean, std = np.mean(x), np.std(x)
        p16, p50, p84 = np.percentile(x, [16, 50, 84])
        print(f"{name:<10}{mean:>12.4g}{std:>12.4g}{p16:>12.4g}{p50:>12.4g}{p84:>12.4g}")
