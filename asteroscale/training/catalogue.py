"""Catalogue normalization for offline population-prior training."""

from __future__ import annotations

from dataclasses import dataclass, field
import csv
from pathlib import Path
import re

import numpy as np

from ..relations import TEFF_SUN


REQUIRED_COLUMNS = ("mass", "radius", "teff", "feh")
OPTIONAL_COLUMNS = ("weight", "age", "state", "population")


def _as_1d(values, name, dtype=float):
    """Convert a catalogue column to a one-dimensional array."""
    array = np.asarray(values, dtype=dtype)
    if array.ndim != 1:
        raise ValueError(f"{name!r} must be a one-dimensional column.")
    return array


@dataclass
class PopulationCatalogue:
    """Normalized stellar population used to train a density model.

    Parameters
    ----------
    mass : array-like
        Current stellar mass in solar masses.
    radius : array-like
        Stellar radius in solar radii.
    teff : array-like
        Effective temperature in kelvin.
    feh : array-like
        Metallicity in dex.
    weight : array-like, optional
        Relative number of stars represented by each row. Equal weights are
        used when omitted.
    age : array-like, optional
        Stellar age in years, retained for diagnostics but not fitted.
    state : array-like, optional
        Evolutionary-state label, retained for future state-conditioned fits.
    population : array-like, optional
        Population label such as ``thin_disc`` or ``halo``.
    metadata : dict, optional
        Provenance and selection information.
    """

    mass: np.ndarray
    radius: np.ndarray
    teff: np.ndarray
    feh: np.ndarray
    weight: np.ndarray | None = None
    age: np.ndarray | None = None
    state: np.ndarray | None = None
    population: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        """Normalize columns and reject invalid training rows."""
        for name in REQUIRED_COLUMNS:
            setattr(self, name, _as_1d(getattr(self, name), name))
        size = len(self.mass)
        if size == 0:
            raise ValueError("The population catalogue is empty.")
        if any(len(getattr(self, name)) != size for name in REQUIRED_COLUMNS):
            raise ValueError("All population columns must have the same length.")

        if self.weight is None:
            self.weight = np.ones(size, dtype=float)
        else:
            self.weight = _as_1d(self.weight, "weight")

        for name in OPTIONAL_COLUMNS[1:]:
            value = getattr(self, name)
            if value is not None:
                dtype = float if name == "age" else str
                value = _as_1d(value, name, dtype=dtype)
                setattr(self, name, value)

        columns = [self.weight]
        columns.extend(
            getattr(self, name)
            for name in OPTIONAL_COLUMNS[1:]
            if getattr(self, name) is not None
        )
        if any(len(column) != size for column in columns):
            raise ValueError("All population columns must have the same length.")

        finite = np.column_stack(
            (self.mass, self.radius, self.teff, self.feh, self.weight)
        )
        if not np.all(np.isfinite(finite)):
            raise ValueError("Training columns contain non-finite values.")
        if np.any(self.mass <= 0.0) or np.any(self.radius <= 0.0):
            raise ValueError("Mass and radius must be strictly positive.")
        if np.any(self.teff <= 0.0):
            raise ValueError("Effective temperature must be strictly positive.")
        if np.any(self.weight <= 0.0):
            raise ValueError("Population weights must be strictly positive.")

    def __len__(self):
        """Return the number of catalogue rows."""
        return len(self.mass)

    @property
    def coordinates(self):
        """Return the four training coordinates.

        Returns
        -------
        ndarray
            Columns are ``log10(mass)``, ``log10(radius)``,
            ``log10(teff)``, and ``feh``.
        """
        return np.column_stack(
            (
                np.log10(self.mass),
                np.log10(self.radius),
                np.log10(self.teff),
                self.feh,
            )
        )

    def subset(self, selection):
        """Return a row subset while preserving metadata."""
        selection = np.asarray(selection)
        kwargs = {
            name: getattr(self, name)[selection]
            for name in REQUIRED_COLUMNS + OPTIONAL_COLUMNS
            if getattr(self, name) is not None
        }
        return PopulationCatalogue(**kwargs, metadata=dict(self.metadata))


def read_standard_catalogue(path):
    """Read AsteroScale's portable comma-separated training format.

    Parameters
    ----------
    path : path-like
        CSV file containing the four required columns and any optional columns.

    Returns
    -------
    PopulationCatalogue
        Validated normalized catalogue.
    """
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No catalogue rows were found in {path}.")
    missing = set(REQUIRED_COLUMNS) - set(rows[0])
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    kwargs = {
        name: np.asarray([row[name] for row in rows], dtype=float)
        for name in REQUIRED_COLUMNS
    }
    for name in OPTIONAL_COLUMNS:
        if name not in rows[0]:
            continue
        dtype = float if name in {"weight", "age"} else str
        kwargs[name] = np.asarray([row[name] for row in rows], dtype=dtype)
    return PopulationCatalogue(
        **kwargs,
        metadata={"source": str(path), "adapter": "standard_csv"},
    )


def write_standard_catalogue(catalogue, path):
    """Write a normalized population catalogue as portable CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(REQUIRED_COLUMNS) + [
        name
        for name in OPTIONAL_COLUMNS
        if getattr(catalogue, name) is not None
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(names)
        for index in range(len(catalogue)):
            writer.writerow(
                [getattr(catalogue, name)[index] for name in names]
            )


def _normalized_name(name):
    """Normalize source column names for tolerant alias matching."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _read_whitespace_table(path):
    """Read a whitespace table whose first non-empty line is the header."""
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        lines = [line for line in stream if line.strip()]
    if not lines:
        raise ValueError(f"No data were found in {path}.")
    header = lines[0].lstrip("#").split()
    data_lines = [line for line in lines[1:] if not line.lstrip().startswith("#")]
    if not data_lines:
        raise ValueError(f"No data rows were found in {path}.")
    data = np.loadtxt(data_lines, ndmin=2)
    if data.shape[1] != len(header):
        raise ValueError(
            f"{path} has {len(header)} column names but {data.shape[1]} values "
            "per row."
        )
    return {_normalized_name(name): data[:, index] for index, name in enumerate(header)}


def _column(table, aliases, required=True):
    """Return the first matching source column."""
    for alias in aliases:
        key = _normalized_name(alias)
        if key in table:
            return table[key]
    if required:
        raise ValueError(f"None of the required columns {aliases} were found.")
    return None


def read_trilegal(
    paths,
    *,
    max_distance_pc=None,
    teff_range=None,
    feh_kind="mh",
):
    """Convert one or more TRILEGAL outputs to the generic catalogue.

    Parameters
    ----------
    paths : path-like or sequence of path-like
        TRILEGAL whitespace tables. The first non-empty line must contain the
        output column names and may start with ``#``.
    max_distance_pc : float, optional
        Retain only stars no farther away than this distance. This requires a
        true distance-modulus column such as ``m-M0``.
    teff_range : pair of float, optional
        Inclusive effective-temperature limits in kelvin.
    feh_kind : {'mh', 'feh'}, default='mh'
        Meaning assigned to TRILEGAL's metallicity column. Standard TRILEGAL
        output is ``[M/H]``. Treating it as ``[Fe/H]`` is an approximation
        unless the simulated mixture is solar-scaled.

    Returns
    -------
    PopulationCatalogue
        Combined catalogue with equal row weights.

    Notes
    -----
    Radius is derived from luminosity and temperature using
    ``R/Rsun = sqrt(L/Lsun) * (Teff_sun/Teff)**2``. Current mass is preferred
    over initial mass because AsteroScale's scaling relations use present-day
    mass.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    paths = [Path(path) for path in paths]
    if not paths:
        raise ValueError("At least one TRILEGAL file is required.")
    if feh_kind not in {"mh", "feh"}:
        raise ValueError("feh_kind must be either 'mh' or 'feh'.")

    chunks = []
    for path in paths:
        table = _read_whitespace_table(path)
        mass = _column(
            table, ("Mact", "m_act", "current_mass", "mass", "mcur")
        )
        log_l = _column(table, ("logL", "log_l", "loglum"))
        log_teff = _column(table, ("logTe", "logTeff", "log_teff"))
        metallicity = _column(table, ("[M/H]", "MH", "FeH", "metallicity"))
        log_age = _column(table, ("logAge", "log_age"), required=False)
        distance_modulus = _column(
            table, ("m-M0", "mM0", "distance_modulus"), required=False
        )

        teff = 10.0**log_teff
        radius = np.sqrt(10.0**log_l) * (TEFF_SUN / teff) ** 2
        keep = np.ones(len(mass), dtype=bool)
        if max_distance_pc is not None:
            if max_distance_pc <= 0.0:
                raise ValueError("max_distance_pc must be positive.")
            if distance_modulus is None:
                raise ValueError(
                    "A distance-modulus column is required for a distance cut."
                )
            distance_pc = 10.0 ** (distance_modulus / 5.0 + 1.0)
            keep &= distance_pc <= max_distance_pc
        if teff_range is not None:
            low, high = teff_range
            if not 0.0 < low < high:
                raise ValueError("teff_range must contain increasing positive limits.")
            keep &= (teff >= low) & (teff <= high)

        chunk = {
            "mass": mass[keep],
            "radius": radius[keep],
            "teff": teff[keep],
            "feh": metallicity[keep],
            "weight": np.ones(np.count_nonzero(keep)),
        }
        if log_age is not None:
            chunk["age"] = 10.0**log_age[keep]
        chunks.append(chunk)

    if not sum(len(chunk["mass"]) for chunk in chunks):
        raise ValueError("No TRILEGAL rows remain after the requested cuts.")
    kwargs = {
        name: np.concatenate([chunk[name] for chunk in chunks])
        for name in REQUIRED_COLUMNS + ("weight",)
    }
    if all("age" in chunk for chunk in chunks):
        kwargs["age"] = np.concatenate([chunk["age"] for chunk in chunks])
    return PopulationCatalogue(
        **kwargs,
        metadata={
            "adapter": "trilegal",
            "sources": [str(path) for path in paths],
            "max_distance_pc": max_distance_pc,
            "teff_range": teff_range,
            "metallicity_interpretation": feh_kind,
        },
    )
