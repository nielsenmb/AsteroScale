"""Forward-model utilities shared by optimization and sampling."""

from .relations import DERIVED


def evaluate_relations(fundamentals):
    """Evaluate all scaling relations in dependency order.

    Parameters
    ----------
    fundamentals : dict
        Fundamental quantities, containing scalar or array values.

    Returns
    -------
    dict
        Input fundamentals plus every derived quantity.
    """
    output = dict(fundamentals)
    for name, (function, arguments) in DERIVED.items():
        output[name] = function(*(output[arg] for arg in arguments))
    return output
