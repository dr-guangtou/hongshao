"""Least non-increasing density envelope of the cubic-logit profile."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import least_squares
from scipy.special import logit

from families import APERTURE_RADIUS_KPC, shell_log_density
from hongshao.cubic_logit import (
    cubic_logit_cumulative_fraction,
    cubic_logit_density_stationary_radii,
    cubic_logit_surface_density,
    cubic_logit_warp,
    cubic_logit_warp_derivative,
)

CUBIC_LOWER = np.asarray([8.0, np.log10(0.15), np.log(0.05), -8.0, np.log(1.0e-4)])
CUBIC_UPPER = np.asarray([14.0, np.log10(500.0), np.log(20.0), 8.0, np.log(5.0)])
DEFAULT_GRID_SIZE = 2048
TAIL_PROBABILITY = 1.0e-13


@dataclass(frozen=True)
class CubicPhysicalParameters:
    aperture_mass: float
    half_mass_radius_kpc: float
    derivative_margin: float
    asymmetry: float
    cubic_curvature: float


def _validated_parameters(parameters: np.ndarray) -> np.ndarray:
    values = np.asarray(parameters, float)
    if values.shape != (5,):
        raise ValueError("the envelope needs five cubic-logit coordinates")
    if not np.all(np.isfinite(values)):
        raise ValueError("parameters must be finite")
    if np.any(values < CUBIC_LOWER) or np.any(values > CUBIC_UPPER):
        raise ValueError("parameters lie outside the declared cubic-logit bounds")
    return values


def optimizer_to_physical(parameters: np.ndarray) -> CubicPhysicalParameters:
    values = _validated_parameters(parameters)
    return CubicPhysicalParameters(
        aperture_mass=10.0 ** values[0],
        half_mass_radius_kpc=10.0 ** values[1],
        derivative_margin=np.exp(values[2]),
        asymmetry=values[3],
        cubic_curvature=np.exp(values[4]),
    )


def synthetic_envelope_parameters() -> np.ndarray:
    return np.asarray([11.5, np.log10(12.0), np.log(0.8), 0.4, np.log(0.08)])


def _log_radius_for_fraction(
    fraction: np.ndarray,
    derivative_margin: float,
    asymmetry: float,
    cubic_curvature: float,
) -> np.ndarray:
    target_logit = logit(np.asarray(fraction, float))
    depressed_linear = derivative_margin / cubic_curvature
    depressed_constant = (
        -(asymmetry**3) / (27.0 * cubic_curvature**3)
        - asymmetry * derivative_margin / (3.0 * cubic_curvature**2)
        - target_logit / cubic_curvature
    )
    argument = (
        3.0
        * depressed_constant
        / (2.0 * depressed_linear)
        * np.sqrt(3.0 / depressed_linear)
    )
    depressed_root = (
        -2.0 * np.sqrt(depressed_linear / 3.0) * np.sinh(np.arcsinh(argument) / 3.0)
    )
    return depressed_root - asymmetry / (3.0 * cubic_curvature)


def _coordinate_grid(
    physical: CubicPhysicalParameters,
    requested_radius: np.ndarray,
    grid_size: int,
) -> np.ndarray:
    if grid_size < 256:
        raise ValueError("grid_size must be at least 256")
    tail_coordinate = _log_radius_for_fraction(
        np.asarray([TAIL_PROBABILITY, 1.0 - TAIL_PROBABILITY]),
        physical.derivative_margin,
        physical.asymmetry,
        physical.cubic_curvature,
    )
    positive = requested_radius[requested_radius > 0.0]
    if len(positive):
        requested_coordinate = np.log(positive / physical.half_mass_radius_kpc)
        lower_requested = float(np.min(requested_coordinate))
        upper_requested = float(np.max(requested_coordinate))
    else:
        lower_requested = 0.0
        upper_requested = 0.0
    lower = min(float(tail_coordinate[0]) - 6.0, lower_requested - 6.0, -12.0)
    upper = max(float(tail_coordinate[1]) + 6.0, upper_requested + 6.0, 12.0)
    return np.linspace(lower, upper, grid_size)


def _envelope_components(
    parameters: np.ndarray,
    requested_radius: np.ndarray,
    grid_size: int,
) -> dict[str, np.ndarray | float | CubicPhysicalParameters]:
    physical = optimizer_to_physical(parameters)
    coordinate = _coordinate_grid(physical, requested_radius, grid_size)
    warp = cubic_logit_warp(
        coordinate,
        physical.derivative_margin,
        physical.asymmetry,
        physical.cubic_curvature,
    )
    derivative = cubic_logit_warp_derivative(
        coordinate,
        physical.derivative_margin,
        physical.asymmetry,
        physical.cubic_curvature,
    )
    log_density = (
        -np.logaddexp(0.0, -warp)
        - np.logaddexp(0.0, warp)
        + np.log(derivative)
        - np.log(2.0 * np.pi)
        - 2.0 * coordinate
    )
    log_density_scale = float(np.max(log_density))
    base_scaled = np.exp(log_density - log_density_scale)
    envelope_scaled = np.maximum.accumulate(base_scaled[::-1])[::-1]
    mass_integrand = 2.0 * np.pi * np.exp(2.0 * coordinate) * envelope_scaled
    initial_mass = np.pi * np.exp(2.0 * coordinate[0]) * envelope_scaled[0]
    cumulative_mass_scaled = initial_mass + cumulative_trapezoid(
        mass_integrand, coordinate, initial=0.0
    )
    return {
        "physical": physical,
        "coordinate": coordinate,
        "base_scaled": base_scaled,
        "envelope_scaled": envelope_scaled,
        "log_density_scale": log_density_scale,
        "cumulative_mass_scaled": cumulative_mass_scaled,
    }


def _validated_radius(radius: np.ndarray | float) -> np.ndarray:
    values = np.asarray(radius, float)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("radius must be finite and non-negative")
    return values


def _raw_envelope_mass(
    parameters: np.ndarray,
    radius: np.ndarray,
    grid_size: int,
) -> tuple[np.ndarray, dict[str, object]]:
    combined = np.concatenate((radius.reshape(-1), np.asarray([APERTURE_RADIUS_KPC])))
    components = _envelope_components(parameters, combined, grid_size)
    physical = components["physical"]
    coordinate = components["coordinate"]
    cumulative = components["cumulative_mass_scaled"]
    output = np.zeros_like(combined)
    positive = combined > 0.0
    requested_coordinate = np.log(combined[positive] / physical.half_mass_radius_kpc)
    output[positive] = np.interp(requested_coordinate, coordinate, cumulative)
    return output, components


def envelope_cog_log(
    parameters: np.ndarray,
    radius: np.ndarray | float,
    grid_size: int = DEFAULT_GRID_SIZE,
) -> np.ndarray:
    radius_values = _validated_radius(radius)
    if np.any(radius_values <= 0.0):
        raise ValueError("logarithmic CoG requires strictly positive radii")
    raw_mass, components = _raw_envelope_mass(parameters, radius_values, grid_size)
    physical = components["physical"]
    mass = physical.aperture_mass * raw_mass[:-1] / raw_mass[-1]
    return np.log10(mass.reshape(radius_values.shape))


def envelope_surface_density(
    parameters: np.ndarray,
    radius: np.ndarray | float,
    grid_size: int = DEFAULT_GRID_SIZE,
    *,
    aperture_normalized: bool = True,
) -> np.ndarray:
    radius_values = _validated_radius(radius)
    combined = np.concatenate(
        (radius_values.reshape(-1), np.asarray([APERTURE_RADIUS_KPC]))
    )
    components = _envelope_components(parameters, combined, grid_size)
    physical = components["physical"]
    coordinate = components["coordinate"]
    envelope_scaled = components["envelope_scaled"]
    log_density_scale = float(components["log_density_scale"])
    cumulative = components["cumulative_mass_scaled"]
    density_scaled = np.empty_like(radius_values.reshape(-1))
    positive = radius_values.reshape(-1) > 0.0
    requested_coordinate = np.log(
        radius_values.reshape(-1)[positive] / physical.half_mass_radius_kpc
    )
    density_scaled[positive] = np.interp(
        requested_coordinate, coordinate, envelope_scaled
    )
    density_scaled[~positive] = envelope_scaled[0]
    if aperture_normalized:
        aperture_coordinate = np.log(
            APERTURE_RADIUS_KPC / physical.half_mass_radius_kpc
        )
        aperture_mass_scaled = float(
            np.interp(aperture_coordinate, coordinate, cumulative)
        )
        normalization = physical.aperture_mass / aperture_mass_scaled
        density = normalization * density_scaled / physical.half_mass_radius_kpc**2
    else:
        density = (
            np.exp(log_density_scale)
            * density_scaled
            / physical.half_mass_radius_kpc**2
        )
    return np.maximum(density, np.nextafter(0.0, 1.0)).reshape(radius_values.shape)


def base_cubic_density(
    parameters: np.ndarray,
    radius: np.ndarray | float,
) -> np.ndarray:
    radius_values = _validated_radius(radius)
    physical = optimizer_to_physical(parameters)
    return cubic_logit_surface_density(
        radius_values,
        1.0,
        physical.half_mass_radius_kpc,
        physical.derivative_margin,
        physical.asymmetry,
        physical.cubic_curvature,
    )


def base_cubic_aperture_density(
    parameters: np.ndarray,
    radius: np.ndarray | float,
) -> np.ndarray:
    """Original cubic-logit density normalized by measured aperture mass."""
    radius_values = _validated_radius(radius)
    physical = optimizer_to_physical(parameters)
    aperture_fraction = float(
        cubic_logit_cumulative_fraction(
            APERTURE_RADIUS_KPC,
            physical.half_mass_radius_kpc,
            physical.derivative_margin,
            physical.asymmetry,
            physical.cubic_curvature,
        )
    )
    return cubic_logit_surface_density(
        radius_values,
        physical.aperture_mass / aperture_fraction,
        physical.half_mass_radius_kpc,
        physical.derivative_margin,
        physical.asymmetry,
        physical.cubic_curvature,
    )


def stationary_envelope_density(
    radius: np.ndarray | float,
    half_mass_radius_kpc: float,
    derivative_margin: float,
    asymmetry: float,
    cubic_curvature: float,
) -> np.ndarray:
    radius_values = _validated_radius(radius)
    if np.any(radius_values <= 0.0):
        raise ValueError("stationary envelope is evaluated at positive radii")
    stationary = cubic_logit_density_stationary_radii(
        half_mass_radius_kpc,
        derivative_margin,
        asymmetry,
        cubic_curvature,
        grid_size=8192,
    )
    current = cubic_logit_surface_density(
        radius_values,
        1.0,
        half_mass_radius_kpc,
        derivative_margin,
        asymmetry,
        cubic_curvature,
    )
    stationary_density = cubic_logit_surface_density(
        stationary,
        1.0,
        half_mass_radius_kpc,
        derivative_margin,
        asymmetry,
        cubic_curvature,
    )
    output = np.asarray(current, float)
    for stationary_radius, density in zip(stationary, stationary_density, strict=True):
        output = np.where(
            radius_values <= stationary_radius,
            np.maximum(output, density),
            output,
        )
    return output


def envelope_diagnostics(
    parameters: np.ndarray,
    radius: np.ndarray,
    grid_size: int = DEFAULT_GRID_SIZE,
) -> dict[str, object]:
    radius_values = _validated_radius(radius)
    physical = optimizer_to_physical(parameters)
    envelope_density = envelope_surface_density(
        parameters, radius_values, grid_size, aperture_normalized=False
    )
    base_density = base_cubic_density(parameters, radius_values)
    ratio = envelope_density / np.maximum(base_density, np.finfo(float).tiny)
    raw_mass, components = _raw_envelope_mass(parameters, radius_values, grid_size)
    envelope_aperture_mass = raw_mass[-1] * np.exp(
        float(components["log_density_scale"])
    )
    base_aperture_mass = float(
        cubic_logit_cumulative_fraction(
            APERTURE_RADIUS_KPC,
            physical.half_mass_radius_kpc,
            physical.derivative_margin,
            physical.asymmetry,
            physical.cubic_curvature,
        )
    )
    envelope_scaled = np.asarray(components["envelope_scaled"])
    base_scaled = np.asarray(components["base_scaled"])
    changed = envelope_scaled > base_scaled * (1.0 + 1.0e-10)
    transitions = np.count_nonzero(np.diff(changed.astype(int)) != 0)
    return {
        "central_density": float(envelope_surface_density(parameters, 0.0, grid_size)),
        "affected_mass_fraction_148": float(
            max(
                0.0,
                (envelope_aperture_mass - base_aperture_mass) / envelope_aperture_mass,
            )
        ),
        "n_changed_measured_density": int(np.count_nonzero(ratio > 1.0 + 2.0e-3)),
        "measured_density_ratio": ratio,
        "n_envelope_transitions": int(transitions),
    }


def _initial_parameters(
    initial_parameters: np.ndarray,
    synthetic_recovery: bool,
) -> list[np.ndarray]:
    initial = _validated_parameters(initial_parameters)
    if synthetic_recovery:
        offsets = np.asarray(
            [
                [0.02, -0.08, 0.12, -0.20, 0.10],
                [-0.02, 0.10, -0.10, 0.25, -0.12],
                [0.01, -0.14, -0.18, 0.30, 0.16],
                [-0.01, 0.16, 0.16, -0.35, -0.15],
            ]
        )
    else:
        offsets = np.asarray(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, -0.08, 0.15, -0.25, 0.12],
                [0.0, 0.10, -0.12, 0.30, -0.10],
                [0.0, -0.15, -0.18, 0.40, 0.18],
            ]
        )
    return [
        np.clip(initial + offset, CUBIC_LOWER + 1.0e-8, CUBIC_UPPER - 1.0e-8)
        for offset in offsets
    ]


def _residual_vector(
    parameters: np.ndarray,
    radius: np.ndarray,
    truth_log: np.ndarray,
    objective: str,
    grid_size: int,
) -> np.ndarray:
    prediction = envelope_cog_log(parameters, radius, grid_size)
    if objective == "log_cog":
        return prediction - truth_log
    if objective == "absolute_cog":
        return (10.0**prediction - 10.0**truth_log) / 10.0 ** truth_log[-1]
    if objective == "log_shell":
        return shell_log_density(prediction, radius) - shell_log_density(
            truth_log, radius
        )
    raise ValueError(f"unknown fitting objective {objective!r}")


def fit_envelope(
    radius: np.ndarray,
    truth_log: np.ndarray,
    *,
    objective: str,
    initial_parameters: np.ndarray,
    synthetic_recovery: bool = False,
    grid_size: int = DEFAULT_GRID_SIZE,
    max_nfev: int = 500,
) -> dict[str, object]:
    solutions = []
    for start in _initial_parameters(initial_parameters, synthetic_recovery):
        solution = least_squares(
            _residual_vector,
            start,
            args=(radius, truth_log, objective, grid_size),
            bounds=(CUBIC_LOWER, CUBIC_UPPER),
            method="trf",
            x_scale="jac",
            max_nfev=max_nfev,
        )
        prediction = envelope_cog_log(solution.x, radius, grid_size)
        score = float(np.mean((prediction - truth_log) ** 2))
        solutions.append((score, solution, prediction))
    solutions.sort(key=lambda item: item[0])
    score, best, prediction = solutions[0]
    near = [item for item in solutions if item[0] <= 1.01 * score + 1.0e-14]
    near_parameters = np.asarray([item[1].x for item in near])
    near_predictions = np.asarray([item[2] for item in near])
    parameter_range = CUBIC_UPPER - CUBIC_LOWER
    scaled_jacobian = best.jac * parameter_range[None, :]
    singular_values = np.linalg.svd(scaled_jacobian, compute_uv=False)
    position = (best.x - CUBIC_LOWER) / parameter_range
    return {
        "parameters": best.x,
        "prediction": prediction,
        "cog_rms_dex": float(np.sqrt(score)),
        "success": bool(best.success),
        "nfev": int(best.nfev),
        "jacobian_min_ratio": (
            float(singular_values[-1] / singular_values[0])
            if singular_values[0] > 0.0
            else 0.0
        ),
        "boundary_distance": float(np.min(np.minimum(position, 1.0 - position))),
        "near_solution_count": len(near),
        "near_parameter_spread": (
            float(np.max(np.ptp(near_parameters, axis=0) / parameter_range))
            if len(near) > 1
            else 0.0
        ),
        "near_profile_spread_dex": (
            float(
                np.max(np.sqrt(np.mean((near_predictions - prediction) ** 2, axis=1)))
            )
            if len(near) > 1
            else 0.0
        ),
    }
