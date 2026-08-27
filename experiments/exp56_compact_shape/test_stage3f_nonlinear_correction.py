"""Mechanical checks for the Exp56 Stage 3f block-rank relation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from stage3d_low_rank_relation import fit_reduced_rank_model  # noqa: E402
from stage3f_nonlinear_correction import (  # noqa: E402
    candidate_supports,
    effective_count_for_support,
    fit_block_rank_model,
    nonlinear_block,
)


def test_candidate_grid_contains_all_one_to_three_term_supports() -> None:
    supports = candidate_supports()
    encoded = {tuple(support.tolist()) for support in supports}
    assert len(supports) == 63
    assert len(encoded) == 63
    assert {int(np.sum(support)) for support in supports} == {1, 2, 3}


def test_effective_count_is_22_to_24() -> None:
    assert effective_count_for_support(np.zeros(7, dtype=bool)) == 18
    for support in candidate_supports():
        assert effective_count_for_support(support) == 21 + np.sum(support)
    assert sorted(
        {effective_count_for_support(value) for value in candidate_supports()}
    ) == [
        22,
        23,
        24,
    ]


def test_linear_core_closes_to_established_rank_two() -> None:
    rng = np.random.default_rng(5711)
    features = rng.normal(size=(500, 5))
    targets = rng.normal(size=(500, 4))
    reference = fit_reduced_rank_model(features, targets, (), rank=2)
    candidate = fit_block_rank_model(
        features,
        targets,
        np.zeros(7, dtype=bool),
    )
    difference = candidate.predict_mean(features) - reference.predict_mean(features)
    assert np.max(np.abs(difference)) < 1e-12


def test_residualized_nonlinear_block_is_training_orthogonal() -> None:
    rng = np.random.default_rng(5712)
    features = rng.normal(size=(600, 5))
    targets = rng.normal(size=(600, 4))
    support = np.asarray([True, True, False, False, True, False, False])
    model = fit_block_rank_model(features, targets, support)
    linear = np.column_stack([np.ones(len(features)), model.standardized(features)])
    nonlinear = nonlinear_block(model, features)
    assert np.max(np.abs(linear.T @ nonlinear / len(features))) < 1e-12


def test_exact_synthetic_rank_two_plus_rank_one_recovery() -> None:
    rng = np.random.default_rng(5713)
    features = rng.normal(size=(900, 5))
    seed_targets = rng.normal(size=(900, 4))
    support = np.asarray([True, True, False, False, True, False, False])
    seed_model = fit_block_rank_model(features, seed_targets, support)
    linear = seed_model.standardized(features)
    nonlinear = nonlinear_block(seed_model, features)
    linear_coefficients = rng.normal(size=(5, 2)) @ rng.normal(size=(2, 4))
    nonlinear_coefficients = rng.normal(size=(3, 1)) @ rng.normal(size=(1, 4))
    targets = (
        rng.normal(size=4)
        + linear @ linear_coefficients
        + nonlinear @ nonlinear_coefficients
    )
    recovered = fit_block_rank_model(features, targets, support)
    assert np.max(np.abs(recovered.predict_mean(features) - targets)) < 1e-10
