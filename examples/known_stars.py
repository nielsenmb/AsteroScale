"""Small AsteroScale examples spanning three evolutionary stages."""

import asteroscale as ast


# Approximate literature observables. These examples demonstrate the API;
# their quoted values are not intended to replace a homogeneous catalogue.
STARS = {
    "Sun": {
        "Teff": (5772.0, 5.0),
        "FeH": (0.0, 0.01),
        "numax": (3090.0, 30.0),
        "dnu": (135.1, 0.1),
    },
    "alpha Cen A": {
        "Teff": (5790.0, 30.0),
        "FeH": (0.22, 0.03),
        "numax": (2400.0, 50.0),
        "dnu": (105.5, 0.1),
    },
    "epsilon Tau": {
        "Teff": (4950.0, 70.0),
        "FeH": (0.15, 0.05),
        "numax": (56.4, 1.1),
        "dnu": (5.00, 0.01),
    },
}


if __name__ == "__main__":
    results = ast.solve_many(STARS, want=["M", "R", "logg"], preset="fast")
    for name, posterior in results.items():
        print(f"\n{name}")
        ast.summarize(posterior)
