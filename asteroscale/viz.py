"""Quick-look posterior visualization -- not a replacement for a real corner
plot, just a fast sanity check on marginal distributions and correlations."""
import numpy as np
import matplotlib.pyplot as plt


def plot_posterior(samples, params=None, bins=40):
    """samples: dict {name: array}, as returned by Solver.solve().
    params: optional list of names to plot (default: all non-'_' keys).
    Diagonal panels show histograms, off-diagonal panels show scatter.
    """
    if params is None:
        params = [k for k in samples if not k.startswith("_")]
    n = len(params)

    fig, axes = plt.subplots(n, n, figsize=(2.5 * n, 2.5 * n))
    if n == 1:
        axes = np.array([[axes]])

    for i, pi in enumerate(params):
        for j, pj in enumerate(params):
            ax = axes[i, j]
            if i == j:
                ax.hist(samples[pi], bins=bins, color="steelblue")
            elif i > j:
                ax.scatter(samples[pj], samples[pi], s=2, alpha=0.2, color="steelblue")
            else:
                ax.axis("off")
            if i == n - 1:
                ax.set_xlabel(pj)
            if j == 0 and i != j:
                ax.set_ylabel(pi)
            if j == 0 and i == 0:
                ax.set_ylabel(pi)

    fig.tight_layout()
    return fig
