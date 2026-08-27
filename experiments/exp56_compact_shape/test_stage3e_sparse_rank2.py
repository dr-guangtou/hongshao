"""Mechanical checks for the Exp56 Stage 3e sparse rank-2 relation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from stage3d_low_rank_relation import SPARSE_TERMS, fit_reduced_rank_model  # noqa: E402
from stage3e_sparse_rank2 import (  # noqa: E402
    candidate_supports,
    effective_count_for_support,
    fit_supported_rank2_model,
)


def test_candidate_grid_contains_every_unique_nonlinear_support() -> None:
    supports = candidate_supports()
    encoded = {tuple(support.tolist()) for support in supports}
    assert len(supports) == 128
    assert len(encoded) == 128
    assert tuple([False] * 7) in encoded
    assert tuple([True] * 7) in encoded


def test_effective_count_adds_two_coefficients_per_nonlinear_term() -> None:
    for support in candidate_supports():
        assert effective_count_for_support(support) == 18 + 2 * np.sum(support)
    assert effective_count_for_support(np.zeros(7, dtype=bool)) == 18
    assert effective_count_for_support(np.ones(7, dtype=bool)) == 32


def test_complete_support_exactly_reproduces_complete_rank_two() -> None:
    rng = np.random.default_rng(5691)
    features = rng.normal(size=(500, 5))
    targets = rng.normal(size=(500, 4))
    reference = fit_reduced_rank_model(features, targets, SPARSE_TERMS, rank=2)
    candidate = fit_supported_rank2_model(
        features,
        targets,
        np.ones(len(SPARSE_TERMS), dtype=bool),
    )
    difference = candidate.predict_mean(features) - reference.predict_mean(features)
    assert np.max(np.abs(difference)) < 1e-12
