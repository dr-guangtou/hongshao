"""Mechanical checks for the Exp56 Stage 3d low-rank mean."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from stage3d_low_rank_relation import (  # noqa: E402
    SPARSE_TERMS,
    effective_coefficient_count,
    fit_reduced_rank_model,
)
from stage3c_sparse_relation import standardized_design  # noqa: E402
from refine_halo_map import fit_conditional_model  # noqa: E402


def test_effective_coefficient_counts_match_predeclaration() -> None:
    assert [effective_coefficient_count(rank, 12, 4) for rank in range(1, 5)] == [
        19,
        32,
        43,
        52,
    ]


def test_rank_four_reproduces_the_stage3_coordinate_mean() -> None:
    rng = np.random.default_rng(5671)
    features = rng.normal(size=(320, 5))
    targets = rng.normal(size=(320, 4))
    reference = fit_conditional_model(features, targets, SPARSE_TERMS)
    candidate = fit_reduced_rank_model(features, targets, SPARSE_TERMS, rank=4)
    difference = candidate.predict_mean(features) - reference.predict_mean(features)
    assert np.max(np.abs(difference)) < 1e-10


def test_rank_two_exactly_recovers_a_noiseless_rank_two_relation() -> None:
    rng = np.random.default_rng(5672)
    features = rng.normal(size=(600, 5))
    design, _, _ = standardized_design(features)
    centered_basis = design[:, 1:] - np.mean(design[:, 1:], axis=0)
    halo_loadings = rng.normal(size=(centered_basis.shape[1], 2))
    output_loadings = rng.normal(size=(2, 4))
    targets = 0.2 * rng.normal(size=4) + centered_basis @ (
        halo_loadings @ output_loadings
    )
    model = fit_reduced_rank_model(features, targets, SPARSE_TERMS, rank=2)
    assert np.linalg.matrix_rank(model.standardized_coefficients, tol=1e-10) == 2
    assert np.max(np.abs(model.predict_mean(features) - targets)) < 1e-10
