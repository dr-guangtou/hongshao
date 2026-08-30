"""Focused checks for the read-only Exp67 comparative QA."""

from __future__ import annotations

import numpy as np
import pytest

import exp67_qa


def test_formula_records_cover_reference_and_three_candidates() -> None:
    assert tuple(exp67_qa.MODELS) == (
        "double_sersic_free",
        "free_damped_cosine",
        "wave_packet_sine_w0p5_w0p5",
        "log_harmonic_sine_w4",
    )
    assert all(record.formula for record in exp67_qa.MODELS.values())
    assert exp67_qa.MODELS["double_sersic_free"].n_parameter == 6
    assert exp67_qa.MODELS["wave_packet_sine_w0p5_w0p5"].n_parameter == 5


def test_mass_quantities_match_direct_aperture_and_positive_shells() -> None:
    profile = 10.0 + np.log10(np.arange(1.0, len(exp67_qa.RADII_KPC) + 1.0))
    quantities = exp67_qa.mass_quantities(profile[None, :])
    expected_m30 = np.interp(np.log10(30.0), np.log10(exp67_qa.RADII_KPC), profile)
    assert np.isclose(quantities["m30"][0], expected_m30)
    assert all(np.isfinite(value[0]) for value in quantities.values())


@pytest.mark.skipif(
    not exp67_qa.discovery_artifacts_available(),
    reason="gitignored Exp67 discovery archives are not present in this worktree",
)
def test_discovery_loader_keeps_stored_predictions_aligned() -> None:
    data = exp67_qa.load_discovery_data(limit_galaxies=2)
    assert data["truth_log"].shape == (10, len(exp67_qa.RADII_KPC))
    assert np.array_equal(np.unique(data["epochs"]), np.arange(5))
    for name in exp67_qa.MODELS:
        assert data["predictions"][name].shape == data["truth_log"].shape
        assert np.all(np.isfinite(data["predictions"][name]))
