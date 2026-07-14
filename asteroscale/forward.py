"""Forward-model utilities shared by optimization and sampling."""

from .relations import DERIVED, FUNDAMENTAL, normalize_bandpass


def required_fundamentals(names):
    """Return the fundamentals needed to evaluate a collection of names.

    Dependencies are followed recursively through :data:`DERIVED`. The
    original order in :data:`FUNDAMENTAL` is retained for deterministic
    sampler dimensions.
    """
    required = set()

    def visit(name):
        if name in FUNDAMENTAL:
            required.add(name)
        elif name in DERIVED:
            for argument in DERIVED[name][1]:
                visit(argument)

    for name in names:
        visit(name)
    return [name for name in FUNDAMENTAL if name in required]


def evaluate_relations(fundamentals, bandpass="TESS"):
    """Evaluate all scaling relations in dependency order.

    Parameters
    ----------
    fundamentals : dict
        Fundamental quantities, containing scalar or array values.
    bandpass : {'TESS', 'Kepler'}, default='TESS'
        Photometric response used for the oscillation-envelope amplitude.

    Returns
    -------
    dict
        Input fundamentals plus every derived quantity whose dependencies
        are available.
    """
    bandpass = normalize_bandpass(bandpass)
    output = dict(fundamentals)
    for name, (function, arguments) in DERIVED.items():
        if not all(argument in output for argument in arguments):
            continue
        kwargs = {"bandpass": bandpass} if name == "A_env" else {}
        output[name] = function(*(output[arg] for arg in arguments), **kwargs)
    return output
