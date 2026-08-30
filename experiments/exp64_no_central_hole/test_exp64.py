"""Focused mechanics checks for Exp64's centrally finite density families."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from families import (  # noqa: E402
    APERTURE_RADIUS_KPC,
    RADII_KPC,
    abel_density,
    cog_log,
    cumulative_mass,
    density_log_slope,
    fit_profile,
    physical_parameters,
    projected_density,
    shell_log_density,
    synthetic_parameters,
)


@pytest.mark.parametrize(
    "family",
    [
        "one_break_k1",
        "one_break_k2",
        "one_break_k4",
        "two_break_k1_k1",
        "two_break_k1_k2",
        "two_break_k2_k1",
        "two_break_k2_k2",
        "two_break_k2_k4",
        "two_break_k4_k2",
        "two_break_k4_k4",
        "two_break_free_sharpness",
    ],
)
def test_hard_mathematical_constraints(family: str) -> None:
    parameters = synthetic_parameters(family)
    radius = np.concatenate(([0.0], np.geomspace(1.0e-6, 1.0e5, 1500)))

    density = projected_density(family, parameters, radius)
    mass = cumulative_mass(family, parameters, radius)
    slope = density_log_slope(family, parameters, radius)

    assert np.isfinite(density).all()
    assert density[0] > 0.0
    assert np.all(density > 0.0)
    assert np.all(np.diff(density) <= 1.0e-12 * density[0])
    assert mass[0] == 0.0
    assert np.all(np.diff(mass) > 0.0)
    assert np.all(slope <= 0.0)
    assert slope[0] == 0.0
    assert physical_parameters(family, parameters).outer_slope > 2.0


@pytest.mark.parametrize(
    "family",
    ["one_break_k2", "two_break_k1_k2", "two_break_k2_k4", "two_break_free_sharpness"],
)
def test_aperture_normalization_and_quadrature_convergence(family: str) -> None:
    parameters = synthetic_parameters(family)
    expected_mass = 10.0 ** parameters[0]
    mass_default = cumulative_mass(
        family, parameters, np.asarray([APERTURE_RADIUS_KPC])
    )[0]
    mass_high_order = cumulative_mass(
        family,
        parameters,
        RADII_KPC,
        quadrature_order=192,
    )
    mass_default_grid = cumulative_mass(family, parameters, RADII_KPC)

    assert mass_default == pytest.approx(expected_mass, rel=2.0e-10)
    assert np.max(np.abs(np.log10(mass_default_grid / mass_high_order))) < 2.0e-6
    assert cog_log(family, parameters, RADII_KPC)[-1] == pytest.approx(
        parameters[0], abs=2.0e-10
    )


def test_two_break_parameter_constraints_are_structural() -> None:
    for family in ("two_break_k2_k2", "two_break_free_sharpness"):
        physical = physical_parameters(family, synthetic_parameters(family))
        assert 0.0 < physical.inner_radius_kpc < physical.outer_radius_kpc
        assert 0.0 < physical.intermediate_slope < physical.outer_slope
        assert physical.outer_slope > 2.0
        assert physical.inner_sharpness > 0.0
        assert physical.outer_sharpness > 0.0


def test_nonnegative_spherical_abel_deprojection() -> None:
    family = "two_break_k2_k4"
    parameters = synthetic_parameters(family)
    radius = np.geomspace(0.01, 1000.0, 80)
    density_3d = abel_density(family, parameters, radius)
    assert np.isfinite(density_3d).all()
    assert np.all(density_3d >= 0.0)
    assert np.any(density_3d > 0.0)


def test_shell_density_includes_central_aperture() -> None:
    family = "two_break_k2_k2"
    parameters = synthetic_parameters(family)
    profile = cog_log(family, parameters, RADII_KPC)
    shell_density = shell_log_density(profile, RADII_KPC)
    direct_central_average = np.log10(
        cumulative_mass(family, parameters, np.asarray([RADII_KPC[0]]))[0]
        / (np.pi * RADII_KPC[0] ** 2)
    )

    assert shell_density.shape == RADII_KPC.shape
    assert shell_density[0] == pytest.approx(direct_central_average, abs=2.0e-8)


@pytest.mark.parametrize("family", ["two_break_k1_k1", "two_break_k4_k4"])
def test_allowed_extremes_preserve_positive_numerical_shells(family: str) -> None:
    from families import FAMILY_SPECS

    spec = FAMILY_SPECS[family]
    lower = np.asarray(spec.lower)
    upper = np.asarray(spec.upper)
    for fraction in (0.01, 0.5, 0.99):
        parameters = lower + fraction * (upper - lower)
        profile = cog_log(family, parameters, RADII_KPC)
        assert np.all(np.diff(10.0**profile) > 0.0)
        assert np.isfinite(shell_log_density(profile, RADII_KPC)).all()


def test_measured_zero_mass_shell_uses_declared_logarithmic_floor() -> None:
    profile = np.linspace(10.0, 11.0, len(RADII_KPC))
    profile[7] = profile[6]
    shell_density = shell_log_density(profile, RADII_KPC)
    shell_area = np.pi * (RADII_KPC[7] ** 2 - RADII_KPC[6] ** 2)
    assert shell_density[7] == pytest.approx(np.log10(1.0 / shell_area))


def test_nonmonotonic_reference_is_floored_only_when_explicit() -> None:
    profile = np.linspace(10.0, 11.0, len(RADII_KPC))
    profile[7] = profile[6] - 0.1
    with pytest.raises(ValueError):
        shell_log_density(profile, RADII_KPC)
    diagnostic = shell_log_density(profile, RADII_KPC, allow_nonpositive_floor=True)
    assert np.isfinite(diagnostic).all()


@pytest.mark.parametrize(
    "family",
    ["one_break_k2", "two_break_k2_k2", "two_break_free_sharpness"],
)
def test_displaced_start_synthetic_recovery(family: str) -> None:
    truth_parameters = synthetic_parameters(family)
    truth = cog_log(family, truth_parameters, RADII_KPC)
    fitted = fit_profile(
        family,
        RADII_KPC,
        truth,
        objective="log_cog",
        synthetic_truth_parameters=truth_parameters,
    )

    assert fitted["success"]
    assert fitted["cog_rms_dex"] < 2.0e-6
    assert np.max(np.abs(fitted["parameters"] - truth_parameters)) < 2.0e-3
    assert fitted["boundary_distance"] > 0.01


def test_invalid_family_and_shapes_raise() -> None:
    with pytest.raises(ValueError):
        synthetic_parameters("not_a_family")
    with pytest.raises(ValueError):
        cumulative_mass("two_break_k2_k2", np.zeros(3), RADII_KPC)
    with pytest.raises(ValueError):
        projected_density(
            "two_break_k2_k2", synthetic_parameters("two_break_k2_k2"), [-1.0]
        )
