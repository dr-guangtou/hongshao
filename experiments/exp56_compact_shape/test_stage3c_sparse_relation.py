"""Mechanical checks for the Exp56 Stage 3c sparse mean."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from stage3c_sparse_relation import (  # noqa: E402
    CORE_COEFFICIENT_COUNT,
    SPARSE_TERMS,
    fit_sparse_group_model,
    fit_conditional_model,
    sparse_group_proximal,
)


def test_sparse_group_proximal_matches_element_then_group_shrinkage() -> None:
    coefficients = np.asarray([[3.0, 4.0], [0.30, -0.20]])
    actual = sparse_group_proximal(coefficients, penalty=1.0, alpha=0.5)
    soft = np.sign(coefficients) * np.maximum(np.abs(coefficients) - 0.5, 0.0)
    norm = np.linalg.norm(soft, axis=1, keepdims=True)
    expected = soft * np.maximum(1.0 - 0.5 / np.maximum(norm, 1e-30), 0.0)
    assert np.allclose(actual, expected)
    assert np.all(actual[1] == 0.0)


def test_zero_penalty_reproduces_unpenalized_sparse_degree_two_mean() -> None:
    rng = np.random.default_rng(5632)
    features = rng.normal(size=(240, 5))
    targets = rng.normal(size=(240, 4))
    reference = fit_conditional_model(features, targets, SPARSE_TERMS)
    candidate, metadata = fit_sparse_group_model(
        features,
        targets,
        alpha=0.5,
        penalty_fraction=0.0,
    )
    assert (
        np.max(
            np.abs(candidate.predict_mean(features) - reference.predict_mean(features))
        )
        < 1e-10
    )
    assert metadata["active_mean_coefficient_count"] == 52


def test_active_coefficient_count_includes_the_unpenalized_linear_core() -> None:
    rng = np.random.default_rng(5633)
    features = rng.normal(size=(300, 5))
    targets = np.column_stack(
        [
            0.8 * features[:, 0] + 0.5 * features[:, 3] ** 2,
            -0.4 * features[:, 1],
            0.2 * features[:, 2],
            0.1 * features[:, 4],
        ]
    )
    model, metadata = fit_sparse_group_model(
        features,
        targets,
        alpha=0.75,
        penalty_fraction=0.5,
    )
    assert metadata["active_mean_coefficient_count"] == (
        CORE_COEFFICIENT_COUNT + np.count_nonzero(model.nonlinear_support)
    )
    assert model.nonlinear_support.shape == (len(SPARSE_TERMS), 4)
    assert np.isfinite(model.predict_mean(features)).all()
