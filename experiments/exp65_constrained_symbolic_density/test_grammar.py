"""Focused checks for the Exp65 constrained symbolic grammar."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from grammar import (  # noqa: E402
    APERTURE_RADIUS_KPC,
    RADII_KPC,
    build_round_b_specs,
    cog_log,
    density_log_slope,
    fit_profile,
    round_a_specs,
    surface_density,
    synthetic_parameters,
)


def test_round_a_contains_both_trigonometric_phases() -> None:
    specs = round_a_specs()
    trig_names = [name for name in specs if "harmonic" in name]
    assert any("sine" in name for name in trig_names)
    assert any("cosine" in name for name in trig_names)
    assert all(spec.n_parameter <= 6 for spec in specs.values())


def test_round_b_is_deterministic_and_respects_coordinate_cap() -> None:
    specs = round_a_specs()
    first = build_round_b_specs(
        specs["sigmoid_w1"],
        specs["harmonic_sine_l1_w1"],
        specs["harmonic_cosine_l1_w2"],
    )
    second = build_round_b_specs(
        specs["sigmoid_w1"],
        specs["harmonic_sine_l1_w1"],
        specs["harmonic_cosine_l1_w2"],
    )
    assert tuple(first) == tuple(second)
    assert any(spec.has_trigonometric_atom for spec in first.values())
    assert all(spec.n_parameter <= 6 for spec in first.values())


def test_every_round_a_expression_has_a_physical_density() -> None:
    radius = np.concatenate(([0.0], np.geomspace(1.0e-6, 1.0e5, 1200)))
    for name in round_a_specs():
        parameters = synthetic_parameters(name)
        density = surface_density(name, parameters, radius, grid_size=2048)
        slope = density_log_slope(name, parameters, radius)
        assert np.all(np.isfinite(density)), name
        assert np.all(density > 0.0), name
        assert np.all(np.diff(density) <= 2.0e-10 * density[0]), name
        assert np.all(slope <= 1.0e-12), name
        assert abs(density[1] / density[0] - 1.0) < 1.0e-8, name


def test_cog_is_aperture_normalized_and_quadrature_converged() -> None:
    for name in round_a_specs():
        parameters = synthetic_parameters(name)
        coarse = cog_log(name, parameters, RADII_KPC, grid_size=512)
        fine = cog_log(name, parameters, RADII_KPC, grid_size=2048)
        assert abs(fine[-1] - parameters[0]) < 1.0e-12, name
        assert np.max(np.abs(coarse - fine)) < 2.0e-4, name
        assert np.all(np.diff(fine) > 0.0), name


def test_synthetic_recovery_from_displaced_starts() -> None:
    representatives = (
        "constant",
        "sigmoid_w1",
        "rational",
        "harmonic_sine_l1_w1",
        "harmonic_cosine_l1_w2",
    )
    for name in representatives:
        parameters = synthetic_parameters(name)
        truth = cog_log(name, parameters, RADII_KPC, grid_size=1024)
        result = fit_profile(
            name,
            RADII_KPC,
            truth,
            grid_size=1024,
            synthetic_truth_parameters=parameters,
        )
        assert result["success"], name
        assert result["cog_rms_dex"] < 2.0e-5, name


def test_aperture_radius_is_last_measured_radius() -> None:
    assert APERTURE_RADIUS_KPC == RADII_KPC[-1]
