"""Focused checks for Exp65 search bookkeeping and common metrics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from search import (  # noqa: E402
    development_masks,
    load_references,
    profile_metrics,
    teacher_indices,
)


def test_demo_reference_alignment_and_masks() -> None:
    reference = load_references("demo")
    discovery, holdout = development_masks(reference["folds"])
    assert len(reference["truth_log"]) == 90
    assert np.all(discovery ^ holdout)
    assert not np.any(discovery & holdout)
    assert set(np.unique(reference["epochs"][discovery])) == set(range(5))
    assert set(np.unique(reference["epochs"][holdout])) == set(range(5))


def test_teacher_selection_is_three_discovery_profiles_per_epoch() -> None:
    reference = load_references("demo")
    discovery, _ = development_masks(reference["folds"])
    selected = teacher_indices(reference, discovery)
    assert len(selected) == 15
    assert np.all(discovery[selected])
    assert np.array_equal(
        np.bincount(reference["epochs"][selected], minlength=5), np.full(5, 3)
    )


def test_common_metrics_are_zero_for_exact_prediction() -> None:
    reference = load_references("demo")
    metrics = profile_metrics(reference["truth_log"], reference["truth_log"])
    assert np.max(metrics["full_cog_rms_dex"]) == 0.0
    assert np.max(metrics["inner_cog_rms_dex"]) == 0.0
    assert np.max(metrics["shell_density_rms_dex"]) == 0.0
    assert np.max(metrics["absolute_size_error_dex"]) == 0.0
