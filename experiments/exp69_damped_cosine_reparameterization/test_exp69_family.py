"""Focused mechanics and recovery checks for Exp69."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from exp69_family import (  # noqa: E402
    RADII_KPC,
    candidate_specs,
    cog_log,
    descriptor_bounds,
    fit_profile,
    integrated_density_rate,
    logarithmic_density_rate,
    normalized_parameters,
    original_to_pivot,
    orthogonal_modulation,
    physical_descriptors,
    pivot_to_original,
    surface_density,
    synthetic_parameters,
)


def test_candidate_set_is_exactly_the_predeclared_narrow_family() -> None:
    specs = candidate_specs()
    assert tuple(specs) == (
        "original_free",
        "fixed_lambda_0",
        "fixed_lambda_0p25",
        "fixed_lambda_0p75",
        "fixed_lambda_1p5",
        "window_pivot",
        "window_orthogonal",
    )
    assert specs["original_free"].n_parameter == 6
    assert specs["window_pivot"].n_parameter == 6
    assert specs["window_orthogonal"].n_parameter == 6
    assert all(specs[name].n_parameter == 5 for name in tuple(specs)[1:5])
    assert [specs[name].fixed_damping for name in tuple(specs)[1:5]] == [
        0.0,
        0.25,
        0.75,
        1.5,
    ]


def test_pivot_coordinates_round_trip_the_original_family() -> None:
    original = synthetic_parameters(candidate_specs()["original_free"], case_index=2)
    pivot = original_to_pivot(original)
    recovered = pivot_to_original(pivot)
    assert np.allclose(recovered, original, rtol=2.0e-13, atol=2.0e-13)
    original_cog = cog_log(
        candidate_specs()["original_free"], original, RADII_KPC, grid_size=1024
    )
    pivot_cog = cog_log(
        candidate_specs()["window_pivot"], pivot, RADII_KPC, grid_size=1024
    )
    assert np.max(np.abs(original_cog - pivot_cog)) < 2.0e-12


def test_orthogonal_modulation_removes_constant_and_core_directions() -> None:
    spec = candidate_specs()["window_orthogonal"]
    parameters = synthetic_parameters(spec, case_index=3)
    core_radius = 10.0 ** parameters[1]
    coordinate = np.log1p(RADII_KPC / core_radius)
    modulation = orthogonal_modulation(
        parameters, coordinate, fit_coordinate=coordinate
    )
    constant = np.ones_like(coordinate)
    core_direction = 1.0 - np.exp(-coordinate)
    design = np.column_stack((constant, core_direction))
    residual = modulation - design @ np.linalg.lstsq(design, modulation, rcond=None)[0]
    assert abs(np.dot(residual, constant)) < 1.0e-12
    assert abs(np.dot(residual, core_direction)) < 1.0e-12
    assert abs(modulation[-1]) < 0.2


def test_every_candidate_keeps_the_squared_slope_physical_guarantees() -> None:
    coordinate = np.linspace(0.0, 15.0, 20001)
    radius = np.concatenate(([0.0], np.geomspace(1.0e-7, 1.0e5, 2000)))
    for case_index, spec in enumerate(candidate_specs().values()):
        parameters = synthetic_parameters(spec, case_index=case_index % 6)
        rate = logarithmic_density_rate(spec, parameters, coordinate)
        density = surface_density(spec, parameters, radius, grid_size=2048)
        assert rate[0] == 0.0, spec.name
        assert np.all(rate >= 0.0), spec.name
        assert np.all(np.isfinite(density)), spec.name
        assert np.all(density > 0.0), spec.name
        assert np.all(np.diff(density) <= 5.0e-11 * density[0]), spec.name


def test_elementary_integral_and_doubled_quadrature_agree() -> None:
    coordinate = np.linspace(0.0, 9.0, 60001)
    for case_index, spec in enumerate(candidate_specs().values()):
        parameters = synthetic_parameters(spec, case_index=case_index % 6)
        exact = integrated_density_rate(spec, parameters, coordinate)
        numeric = cumulative_trapezoid(
            logarithmic_density_rate(spec, parameters, coordinate),
            coordinate,
            initial=0.0,
        )
        assert np.max(np.abs(exact - numeric)) < 3.0e-7, spec.name
        coarse = cog_log(spec, parameters, RADII_KPC, grid_size=512)
        fine = cog_log(spec, parameters, RADII_KPC, grid_size=2048)
        assert abs(fine[-1] - physical_descriptors(spec, parameters)[0]) < 1.0e-12
        assert np.all(np.diff(fine) > 0.0), spec.name
        assert np.max(np.abs(coarse - fine)) < 4.0e-4, spec.name


def test_noiseless_synthetic_profiles_recover_in_common_descriptors() -> None:
    for case_index in range(6):
        for name, spec in candidate_specs().items():
            parameters = synthetic_parameters(spec, case_index=case_index)
            truth = cog_log(spec, parameters, RADII_KPC, grid_size=1024)
            result = fit_profile(
                spec,
                RADII_KPC,
                truth,
                n_starts=4,
                max_nfev=600,
                grid_size=512,
                synthetic_truth_parameters=parameters,
            )
            descriptor_lower, descriptor_upper = descriptor_bounds(spec)
            descriptor_error = np.max(
                np.abs(
                    physical_descriptors(spec, result["parameters"])
                    - physical_descriptors(spec, parameters)
                )
                / np.maximum(
                    descriptor_upper - descriptor_lower,
                    1.0e-12,
                )
            )
            assert result["cog_rms_dex"] < 3.0e-5, (name, case_index)
            assert descriptor_error < 0.03, (name, case_index, descriptor_error)
            unit = normalized_parameters(spec, result["parameters"])
            assert np.all((unit >= 0.0) & (unit <= 1.0))
