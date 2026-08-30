"""Focused checks for Exp64's monotone-density-envelope follow-up."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from envelope import (  # noqa: E402
    CUBIC_LOWER,
    CUBIC_UPPER,
    base_cubic_density,
    envelope_cog_log,
    envelope_diagnostics,
    envelope_surface_density,
    fit_envelope,
    optimizer_to_physical,
    stationary_envelope_density,
    synthetic_envelope_parameters,
)
from families import RADII_KPC, shell_log_density  # noqa: E402


def test_optimizer_transform_preserves_cubic_constraints() -> None:
    physical = optimizer_to_physical(synthetic_envelope_parameters())
    assert physical.aperture_mass > 0.0
    assert physical.half_mass_radius_kpc > 0.0
    assert physical.derivative_margin > 0.0
    assert physical.cubic_curvature > 0.0


@pytest.mark.parametrize("fraction", [0.01, 0.5, 0.99])
def test_envelope_hard_constraints_across_allowed_coordinates(fraction: float) -> None:
    parameters = CUBIC_LOWER + fraction * (CUBIC_UPPER - CUBIC_LOWER)
    radius = np.concatenate(([0.0], np.geomspace(1.0e-5, 1.0e4, 1200)))
    density = envelope_surface_density(parameters, radius, grid_size=4096)
    profile = envelope_cog_log(parameters, RADII_KPC, grid_size=4096)

    assert np.isfinite(density).all()
    assert density[0] > 0.0
    assert np.all(density > 0.0)
    assert np.all(np.diff(density) <= 1.0e-10 * density[0])
    assert np.all(np.diff(10.0**profile) >= 0.0)
    assert profile[-1] == pytest.approx(parameters[0], abs=3.0e-7)


def test_reverse_maximum_agrees_with_stationary_point_construction() -> None:
    parameters = synthetic_envelope_parameters()
    physical = optimizer_to_physical(parameters)
    radius = np.geomspace(0.01, 1000.0, 300)
    numerical = envelope_surface_density(
        parameters, radius, grid_size=8192, aperture_normalized=False
    )
    stationary = stationary_envelope_density(
        radius,
        physical.half_mass_radius_kpc,
        physical.derivative_margin,
        physical.asymmetry,
        physical.cubic_curvature,
    )
    assert np.max(np.abs(numerical / stationary - 1.0)) < 2.0e-3


def test_envelope_grid_convergence() -> None:
    parameters = synthetic_envelope_parameters()
    low_resolution = envelope_cog_log(parameters, RADII_KPC, grid_size=2048)
    high_resolution = envelope_cog_log(parameters, RADII_KPC, grid_size=8192)
    assert np.max(np.abs(low_resolution - high_resolution)) < 2.0e-5


def test_envelope_changes_only_density_below_the_original_peak() -> None:
    parameters = synthetic_envelope_parameters()
    diagnostics = envelope_diagnostics(parameters, RADII_KPC, grid_size=4096)
    assert diagnostics["central_density"] > 0.0
    assert diagnostics["affected_mass_fraction_148"] > 0.0
    assert 0 <= diagnostics["n_changed_measured_density"] <= len(RADII_KPC)
    unchanged = np.isclose(
        diagnostics["measured_density_ratio"],
        np.ones(len(RADII_KPC)),
        rtol=2.0e-3,
    )
    assert np.count_nonzero(unchanged) >= len(RADII_KPC) // 2


def test_base_density_has_the_known_central_hole_but_envelope_does_not() -> None:
    parameters = synthetic_envelope_parameters()
    radius = np.asarray([0.0, 1.0e-8, 1.0e-5])
    base = base_cubic_density(parameters, radius)
    envelope = envelope_surface_density(parameters, radius)
    assert base[0] == 0.0
    assert base[1] < base[2]
    assert np.all(envelope > 0.0)
    assert envelope[0] == pytest.approx(envelope[1], rel=1.0e-6)


@pytest.mark.parametrize("objective", ["log_cog", "absolute_cog", "log_shell"])
def test_displaced_start_synthetic_recovery(objective: str) -> None:
    parameters = synthetic_envelope_parameters()
    truth = envelope_cog_log(parameters, RADII_KPC)
    fitted = fit_envelope(
        RADII_KPC,
        truth,
        objective=objective,
        initial_parameters=parameters,
        synthetic_recovery=True,
    )
    assert fitted["success"]
    assert fitted["cog_rms_dex"] < 3.0e-5
    assert np.max(np.abs(fitted["parameters"] - parameters)) < 5.0e-3
    assert np.isfinite(shell_log_density(fitted["prediction"], RADII_KPC)).all()
