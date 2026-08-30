"""Focused population, gate, and ranking checks for Exp70."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from exp70_run import (  # noqa: E402
    candidate_is_eligible,
    choose_global_damping,
    load_reference_data,
    operational_gate_indices,
    stage_indices,
)


def test_reference_loader_requires_the_preserved_exp67_root() -> None:
    configured = os.environ.get("HONGSHAO_EXP67_ROOT")
    assert configured is not None
    assert Path(configured).name == "hongshao_exp67_squared_slope_symbolic_density"


def test_stage_indices_preserve_the_complete_blind_partition() -> None:
    reference = load_reference_data()
    expected = {"discovery": 1200, "selection": 1000, "validation": 1180}
    row_sets = []
    for stage, n_galaxy in expected.items():
        indices = stage_indices(reference, stage)
        rows = np.unique(reference["rows"][indices])
        assert len(indices) == 5 * n_galaxy
        assert len(rows) == n_galaxy
        assert np.array_equal(
            np.bincount(reference["epochs"][indices], minlength=5),
            np.full(5, n_galaxy),
        )
        row_sets.append(set(rows.tolist()))
    assert not row_sets[0] & row_sets[1]
    assert not row_sets[0] & row_sets[2]
    assert not row_sets[1] & row_sets[2]


def test_operational_gate_is_the_frozen_discovery_only_sample() -> None:
    reference = load_reference_data()
    discovery = stage_indices(reference, "discovery")
    gate = operational_gate_indices(reference, discovery)
    assert len(gate) == 125
    assert len(np.unique(reference["rows"][gate])) == 25
    assert set(gate.tolist()) <= set(discovery.tolist())
    assert np.array_equal(
        np.bincount(reference["epochs"][gate], minlength=5), np.full(5, 25)
    )


def _passing_record(
    name: str, damping: float, worst_accuracy: float
) -> dict[str, object]:
    return {
        "name": name,
        "fixed_damping": damping,
        "success_fraction": 1.0,
        "boundary_fraction": 0.01,
        "maximum_near_profile_spread_dex": 0.0005,
        "jacobian_ratio_to_cubic_by_epoch": [0.3] * 5,
        "jacobian_factor_to_free_by_epoch": [5.0] * 5,
        "full_cog_ratio_to_free_by_epoch": [worst_accuracy] * 5,
        "middle_cog_ratio_to_free_by_epoch": [1.02] * 5,
        "shell_ratio_to_free_by_epoch": [1.03] * 5,
        "nearest_continuity_ratio_to_free": 0.6,
        "epoch_continuity_ratio_to_free": 0.7,
        "nearest_continuity_median_ratio_to_free": 0.9,
        "epoch_continuity_median_ratio_to_free": 0.9,
    }


def test_discovery_gate_and_global_choice_are_deterministic() -> None:
    records = [
        _passing_record("fixed_lambda_0p75", 0.75, 1.08),
        _passing_record("fixed_lambda_1", 1.0, 1.04),
        _passing_record("fixed_lambda_1p25", 1.25, 1.04),
        _passing_record("fixed_lambda_1p5", 1.5, 1.06),
    ]
    assert all(candidate_is_eligible(record, stage="discovery") for record in records)
    assert choose_global_damping(records)["name"] == "fixed_lambda_1"

    failed = dict(records[1])
    failed["boundary_fraction"] = 0.051
    assert not candidate_is_eligible(failed, stage="discovery")
    failed = dict(records[1])
    failed["full_cog_ratio_to_free_by_epoch"] = [1.11] * 5
    assert not candidate_is_eligible(failed, stage="discovery")
