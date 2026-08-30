"""Focused checks for Exp66 population splitting and common metrics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from exp66_search import (  # noqa: E402
    broad_screen_indices,
    load_references,
    population_split,
    profile_metrics,
)


def test_full_population_split_has_exact_sizes_and_no_leakage() -> None:
    reference = load_references("full")
    split = population_split(reference)
    expected_galaxies = {"discovery": 1000, "selection": 1000, "validation": 1380}
    all_rows: list[np.ndarray] = []
    for name, expected in expected_galaxies.items():
        indices = split[name]
        rows = np.unique(reference["rows"][indices])
        assert len(indices) == 5 * expected
        assert len(rows) == expected
        assert np.array_equal(
            np.bincount(reference["epochs"][indices], minlength=5),
            np.full(5, expected),
        )
        all_rows.append(rows)
    assert len(np.unique(np.concatenate(all_rows))) == 3380


def test_population_split_is_deterministic_and_mass_stratified() -> None:
    reference = load_references("full")
    first = population_split(reference)
    second = population_split(reference)
    for name in first:
        assert np.array_equal(first[name], second[name])
    z04 = reference["epochs"] == 0
    full_median = np.nanmedian(reference["halo_mass"][z04])
    for indices in first.values():
        subset = indices[reference["epochs"][indices] == 0]
        assert abs(np.nanmedian(reference["halo_mass"][subset]) - full_median) < 0.03


def test_broad_screen_is_250_discovery_galaxies_at_all_epochs() -> None:
    reference = load_references("full")
    discovery = population_split(reference)["discovery"]
    broad = broad_screen_indices(reference, discovery)
    assert len(broad) == 1250
    assert len(np.unique(reference["rows"][broad])) == 250
    assert np.all(np.isin(broad, discovery))
    assert np.array_equal(
        np.bincount(reference["epochs"][broad], minlength=5), np.full(5, 250)
    )


def test_common_metrics_are_zero_for_exact_prediction() -> None:
    reference = load_references("demo")
    metrics = profile_metrics(reference["truth_log"], reference["truth_log"])
    assert np.max(metrics["full_cog_rms_dex"]) == 0.0
    assert np.max(metrics["middle_cog_rms_dex"]) == 0.0
    assert np.max(metrics["shell_density_rms_dex"]) == 0.0
    assert np.max(metrics["absolute_size_error_dex"]) == 0.0
