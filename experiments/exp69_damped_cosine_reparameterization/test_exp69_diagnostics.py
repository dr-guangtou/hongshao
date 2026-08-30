"""Focused solution-selection and continuity checks for Exp69."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from exp69_diagnostics import (  # noqa: E402
    canonical_solution_index,
    continuity_metrics,
    nearest_profile_pairs,
    weakest_direction_path,
)


def test_canonical_rule_prefers_conditioning_then_loss_then_coordinates() -> None:
    objectives = np.asarray([1.0, 1.005, 1.004, 1.2])
    singular_minimum = np.asarray([0.02, 0.05, 0.05, 0.8])
    descriptors = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.4, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.3, 0.9, 1.1],
            [0.0, 0.0, 0.0, 0.2, 0.8, 1.0],
            [0.0, 0.0, 0.0, 0.1, 0.1, 0.1],
        ]
    )
    assert canonical_solution_index(objectives, singular_minimum, descriptors) == 2


def test_weakest_direction_path_respects_the_parameter_box() -> None:
    parameter = np.asarray([0.25, 0.5, 0.75])
    direction = np.asarray([1.0, -2.0, 0.5])
    lower = np.zeros(3)
    upper = np.ones(3)
    path = weakest_direction_path(parameter, direction, lower, upper, n_point=21)
    assert path.shape == (21, 3)
    assert np.all(path >= lower)
    assert np.all(path <= upper)
    assert np.allclose(path[10], parameter)


def test_profile_neighbors_and_continuity_are_deterministic() -> None:
    truth = np.asarray(
        [
            [10.0, 10.3, 10.5],
            [11.2, 11.3, 11.5],
            [10.0, 10.2, 10.5],
            [10.0, 10.1, 10.5],
        ]
    )
    epochs = np.asarray([0, 0, 0, 1])
    first = nearest_profile_pairs(truth, epochs)
    second = nearest_profile_pairs(truth, epochs)
    assert np.array_equal(first, second)
    assert tuple(first[0]) == (0, 2)

    descriptors = np.asarray(
        [
            [0.0, 0.0],
            [0.5, 0.5],
            [0.1, 0.0],
            [0.2, 0.0],
        ]
    )
    rows = np.asarray([1, 2, 3, 3])
    result = continuity_metrics(descriptors, truth, rows, epochs)
    assert result["nearest_pair_count"] == 3
    assert result["adjacent_epoch_pair_count"] == 1
    assert result["nearest_median"] <= result["nearest_p95"]
    assert result["epoch_median"] == result["epoch_p95"]
