"""Focused numerical checks for the Exp56 Stage 3b direct CoG fit."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import stage3b_direct_cog as direct  # noqa: E402


def test_vectorized_decoder_matches_established_decoder():
    rng = np.random.default_rng(56031)
    parameters = np.column_stack(
        [
            rng.uniform(10.8, 12.2, 12),
            rng.uniform(-1.5, 0.2, 12),
            rng.uniform(0.25, 0.65, 12),
            rng.uniform(1.2, 3.8, 12),
        ]
    )
    gamma = rng.uniform(1.25, 1.4, len(parameters))
    expected = direct.stage3.decode_profiles(parameters, gamma)
    actual = direct.vectorized_decode(parameters, gamma)
    assert np.max(np.abs(actual - expected)) < 1e-10


def test_chained_jacobian_matches_independent_finite_difference():
    rng = np.random.default_rng(56032)
    features = rng.normal(size=(9, 3))
    design = np.column_stack([np.ones(len(features)), features])
    coefficients = np.asarray(
        [
            [11.5, 0.08, -0.03, 0.02],
            [-0.6, 0.05, 0.03, -0.02],
            [0.42, 0.02, -0.01, 0.01],
            [2.3, -0.04, 0.03, 0.02],
        ]
    )
    gamma = np.linspace(1.4, 1.25, len(features))
    truth = direct.vectorized_decode(design @ coefficients.T, gamma)
    fitter = direct.DecodedCogLeastSquares(design, truth, gamma)
    chained = fitter.jacobian(coefficients.ravel())
    step = 1e-6
    independent = np.empty_like(chained)
    for column in range(coefficients.size):
        offset = np.zeros(coefficients.size)
        offset[column] = step
        independent[:, column] = (
            fitter.residual(coefficients.ravel() + offset)
            - fitter.residual(coefficients.ravel() - offset)
        ) / (2.0 * step)
    relative_error = np.linalg.norm(chained - independent) / np.linalg.norm(independent)
    assert relative_error < 1e-5


def test_synthetic_shared_relation_recovers_profiles():
    rng = np.random.default_rng(56033)
    features = rng.normal(size=(36, 3))
    design = np.column_stack([np.ones(len(features)), features])
    true_coefficients = np.asarray(
        [
            [11.5, 0.10, -0.04, 0.03],
            [-0.5, 0.08, 0.04, -0.03],
            [0.42, 0.03, -0.02, 0.01],
            [2.4, -0.05, 0.04, 0.02],
        ]
    )
    gamma = np.linspace(1.4, 1.25, len(features))
    truth = direct.vectorized_decode(design @ true_coefficients.T, gamma)
    start = true_coefficients + rng.normal(scale=0.01, size=true_coefficients.shape)
    result = direct.fit_decoded_coefficients(design, truth, gamma, start)
    prediction = direct.vectorized_decode(design @ result.coefficients.T, gamma)
    assert np.max(np.abs(prediction - truth)) < 1e-6
