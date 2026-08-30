"""Focused checks for Exp67 population accounting and metrics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from exp67_search import (  # noqa: E402
    load_references,
    operational_gate_indices,
    population_split,
    profile_metrics,
)


def test_split_has_exact_population_sizes_and_explicit_missing_stratum() -> None:
    reference = load_references("full")
    split = population_split(reference)
    expected = {"discovery": 1200, "selection": 1000, "validation": 1180}
    row_sets = []
    for name, n_galaxy in expected.items():
        indices = split[name]
        rows = np.unique(reference["rows"][indices])
        assert len(rows) == n_galaxy
        assert len(indices) == 5 * n_galaxy
        assert np.array_equal(
            np.bincount(reference["epochs"][indices], minlength=5),
            np.full(5, n_galaxy),
        )
        row_sets.append(set(rows.tolist()))
    assert not row_sets[0] & row_sets[1]
    assert not row_sets[0] & row_sets[2]
    assert not row_sets[1] & row_sets[2]
    assert len(set.union(*row_sets)) == 3380
    assert split["stratum_counts"] == {
        "finite_halo_mass": {"discovery": 850, "selection": 708, "validation": 837},
        "missing_halo_mass": {"discovery": 350, "selection": 292, "validation": 343},
    }


def test_split_and_operational_gate_are_deterministic() -> None:
    reference = load_references("full")
    first = population_split(reference)
    second = population_split(reference)
    for name in ("discovery", "selection", "validation"):
        assert np.array_equal(first[name], second[name])
    first_gate = operational_gate_indices(reference, first["discovery"])
    second_gate = operational_gate_indices(reference, second["discovery"])
    assert np.array_equal(first_gate, second_gate)
    assert len(first_gate) == 125
    assert len(np.unique(reference["rows"][first_gate])) == 25


def test_common_metrics_are_zero_for_exact_prediction() -> None:
    reference = load_references("demo")
    metrics = profile_metrics(reference["truth_log"], reference["truth_log"])
    assert np.max(metrics["full_cog_rms_dex"]) == 0.0
    assert np.max(metrics["middle_cog_rms_dex"]) == 0.0
    assert np.max(metrics["shell_density_rms_dex"]) == 0.0
    assert np.max(metrics["absolute_size_error_dex"]) == 0.0
