import json

import numpy as np
import pytest

from asteroscale.training import (
    PopulationCatalogue,
    PopulationGMM,
    fit_candidate_models,
    fit_weighted_gmm,
    read_standard_catalogue,
    read_trilegal,
    write_standard_catalogue,
)
from asteroscale.training.cli import main


def synthetic_catalogue(seed=10, size=400):
    """Return a small two-sequence population for fast fitting tests."""
    generator = np.random.default_rng(seed)
    state = generator.choice(2, size=size, p=(0.7, 0.3))
    mass = np.where(
        state == 0,
        generator.normal(1.0, 0.08, size),
        generator.normal(1.35, 0.10, size),
    )
    radius = np.where(
        state == 0,
        mass**0.8 * np.exp(generator.normal(0.0, 0.025, size)),
        3.5 * mass**0.3 * np.exp(generator.normal(0.0, 0.04, size)),
    )
    teff = np.where(
        state == 0,
        5772.0 * mass**0.45 + generator.normal(0.0, 35.0, size),
        5050.0 * mass**0.10 + generator.normal(0.0, 45.0, size),
    )
    feh = generator.normal(-0.1, 0.18, size)
    return PopulationCatalogue(
        mass=mass,
        radius=radius,
        teff=teff,
        feh=feh,
        weight=np.where(state == 0, 1.0, 2.0),
        state=np.where(state == 0, "main_sequence", "giant"),
    )


def test_catalogue_coordinates_and_validation():
    catalogue = synthetic_catalogue(size=20)
    assert catalogue.coordinates.shape == (20, 4)
    np.testing.assert_allclose(
        catalogue.coordinates[:, 0], np.log10(catalogue.mass)
    )
    with pytest.raises(ValueError, match="strictly positive"):
        PopulationCatalogue(
            mass=[1.0, -1.0],
            radius=[1.0, 1.0],
            teff=[5700.0, 5700.0],
            feh=[0.0, 0.0],
        )


def test_standard_catalogue_round_trip(tmp_path):
    original = synthetic_catalogue(size=20)
    path = tmp_path / "population.csv"
    write_standard_catalogue(original, path)
    restored = read_standard_catalogue(path)
    for name in ("mass", "radius", "teff", "feh", "weight", "state"):
        np.testing.assert_array_equal(
            getattr(restored, name), getattr(original, name)
        )


def test_trilegal_adapter_derives_radius_and_applies_cuts(tmp_path):
    path = tmp_path / "trilegal.dat"
    path.write_text(
        "#Gc logAge [M/H] Mact logL logTe logg m-M0\n"
        "1 9.0 0.0 1.0 0.0 3.761326 4.44 5.0\n"
        "1 9.5 -0.2 1.2 1.0 3.698970 3.00 11.0\n",
        encoding="utf-8",
    )
    catalogue = read_trilegal(
        path, max_distance_pc=150.0, teff_range=(5000.0, 6500.0)
    )
    assert len(catalogue) == 1
    assert catalogue.mass[0] == pytest.approx(1.0)
    assert catalogue.radius[0] == pytest.approx(1.0, rel=2e-6)
    assert catalogue.teff[0] == pytest.approx(5772.0, rel=2e-6)
    assert catalogue.age[0] == pytest.approx(1e9)


def test_trilegal_distance_cut_requires_distance_modulus(tmp_path):
    path = tmp_path / "trilegal.dat"
    path.write_text(
        "# Mact logL logTe [M/H]\n"
        "1.0 0.0 3.761326 0.0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="distance-modulus"):
        read_trilegal(path, max_distance_pc=1000.0)


def test_weighted_gmm_recovers_weighted_component_fraction():
    generator = np.random.default_rng(4)
    first = generator.normal(-1.0, 0.12, size=(150, 4))
    second = generator.normal(1.0, 0.12, size=(150, 4))
    coordinates = np.vstack((first, second))
    weight = np.r_[np.ones(150), np.full(150, 3.0)]
    model = fit_weighted_gmm(
        coordinates,
        weight,
        n_components=2,
        n_init=3,
        max_iter=100,
        batch_size=31,
        random_state=4,
    )
    np.testing.assert_allclose(
        np.sort(model.weights), (0.25, 0.75), atol=0.03
    )
    assert np.all(np.isfinite(model.logpdf(coordinates)))
    np.testing.assert_allclose(
        model.logpdf(coordinates, batch_size=17),
        model.logpdf(coordinates, batch_size=len(coordinates)),
    )


def test_gmm_save_load_and_sampling(tmp_path):
    catalogue = synthetic_catalogue(size=200)
    model = fit_weighted_gmm(
        catalogue.coordinates,
        catalogue.weight,
        n_components=2,
        n_init=2,
        max_iter=100,
        random_state=2,
        metadata={"source": "synthetic"},
    )
    path = tmp_path / "model.npz"
    model.save(path)
    restored = PopulationGMM.load(path)
    np.testing.assert_allclose(
        restored.logpdf(catalogue.coordinates[:10]),
        model.logpdf(catalogue.coordinates[:10]),
    )
    assert restored.metadata["source"] == "synthetic"
    assert restored.sample(12, random_state=1).shape == (12, 4)
    with np.load(path, allow_pickle=False) as data:
        assert data["cholesky_factors"].shape == (2, 4, 4)


def test_candidate_selection_and_report():
    catalogue = synthetic_catalogue(size=300)
    model, report = fit_candidate_models(
        catalogue.coordinates,
        catalogue.weight,
        (1, 2),
        validation_fraction=0.2,
        selection_tolerance=0.0,
        random_state=3,
        n_init=2,
        max_iter=100,
    )
    assert len(report) == 2
    assert model.metadata["n_components"] in {1, 2}
    assert {
        "n_components",
        "validation_mean_log_density",
        "bic",
    } <= report[0].keys()


def test_training_cli_writes_model_and_report(tmp_path):
    catalogue = synthetic_catalogue(size=120)
    source = tmp_path / "catalogue.csv"
    model_path = tmp_path / "model.npz"
    report_path = tmp_path / "report.json"
    write_standard_catalogue(catalogue, source)
    main(
        [
            str(source),
            "--output",
            str(model_path),
            "--components",
            "1,2",
            "--n-init",
            "1",
            "--max-iter",
            "50",
            "--report",
            str(report_path),
        ]
    )
    assert model_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["n_catalogue_rows"] == 120
    assert report["selected_components"] in {1, 2}
