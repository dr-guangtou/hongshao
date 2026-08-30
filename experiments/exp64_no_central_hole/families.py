"""Centrally finite projected-density families for Exp64."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit, logit

from hongshao.tng_data import COG_RAD_KPC

RADII_KPC = np.asarray(COG_RAD_KPC, float)
APERTURE_RADIUS_KPC = float(RADII_KPC[-1])

LOG_MASS_BOUNDS = (8.0, 14.0)
LOG_RADIUS_BOUNDS = (np.log10(0.05), np.log10(200.0))
LOG_RADIUS_RATIO_BOUNDS = (np.log10(1.02), np.log10(2000.0))
MID_SLOPE_LOGIT_BOUNDS = (-5.0, 5.0)
LOG_OUTER_EXCESS_BOUNDS = (np.log(0.05), np.log(20.0))
LOG_SHARPNESS_BOUNDS = (np.log(0.4), np.log(8.0))
DEFAULT_QUADRATURE_ORDER = 32


@dataclass(frozen=True)
class FamilySpec:
    name: str
    label: str
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    inner_sharpness: float | None
    outer_sharpness: float | None
    two_break: bool
    free_sharpness: bool = False

    @property
    def n_parameter(self) -> int:
        return len(self.lower)


@dataclass(frozen=True)
class PhysicalParameters:
    aperture_mass: float
    inner_radius_kpc: float
    outer_radius_kpc: float
    intermediate_slope: float
    outer_slope: float
    inner_sharpness: float
    outer_sharpness: float


def _one_break_spec(sharpness: float) -> FamilySpec:
    label_value = f"{sharpness:g}".replace(".", "p")
    return FamilySpec(
        f"one_break_k{label_value}",
        f"One-break cored power law, k={sharpness:g}",
        (LOG_MASS_BOUNDS[0], LOG_RADIUS_BOUNDS[0], LOG_OUTER_EXCESS_BOUNDS[0]),
        (LOG_MASS_BOUNDS[1], LOG_RADIUS_BOUNDS[1], LOG_OUTER_EXCESS_BOUNDS[1]),
        sharpness,
        sharpness,
        False,
    )


def _two_break_spec(inner_sharpness: float, outer_sharpness: float) -> FamilySpec:
    inner_label = f"{inner_sharpness:g}".replace(".", "p")
    outer_label = f"{outer_sharpness:g}".replace(".", "p")
    return FamilySpec(
        f"two_break_k{inner_label}_k{outer_label}",
        (
            "Two-break cored power law, "
            f"k_inner={inner_sharpness:g}, k_outer={outer_sharpness:g}"
        ),
        (
            LOG_MASS_BOUNDS[0],
            LOG_RADIUS_BOUNDS[0],
            LOG_RADIUS_RATIO_BOUNDS[0],
            MID_SLOPE_LOGIT_BOUNDS[0],
            LOG_OUTER_EXCESS_BOUNDS[0],
        ),
        (
            LOG_MASS_BOUNDS[1],
            LOG_RADIUS_BOUNDS[1],
            LOG_RADIUS_RATIO_BOUNDS[1],
            MID_SLOPE_LOGIT_BOUNDS[1],
            LOG_OUTER_EXCESS_BOUNDS[1],
        ),
        inner_sharpness,
        outer_sharpness,
        True,
    )


FAMILY_SPECS = {
    **{f"one_break_k{value:g}": _one_break_spec(value) for value in (1.0, 2.0, 4.0)},
    **{
        f"two_break_k{inner:g}_k{outer:g}": _two_break_spec(inner, outer)
        for inner, outer in (
            (1.0, 1.0),
            (1.0, 2.0),
            (2.0, 1.0),
            (2.0, 2.0),
            (2.0, 4.0),
            (4.0, 2.0),
            (4.0, 4.0),
        )
    },
}
FAMILY_SPECS["two_break_free_sharpness"] = FamilySpec(
    "two_break_free_sharpness",
    "Two-break cored power law with one free shared sharpness",
    (
        LOG_MASS_BOUNDS[0],
        LOG_RADIUS_BOUNDS[0],
        LOG_RADIUS_RATIO_BOUNDS[0],
        MID_SLOPE_LOGIT_BOUNDS[0],
        LOG_OUTER_EXCESS_BOUNDS[0],
        LOG_SHARPNESS_BOUNDS[0],
    ),
    (
        LOG_MASS_BOUNDS[1],
        LOG_RADIUS_BOUNDS[1],
        LOG_RADIUS_RATIO_BOUNDS[1],
        MID_SLOPE_LOGIT_BOUNDS[1],
        LOG_OUTER_EXCESS_BOUNDS[1],
        LOG_SHARPNESS_BOUNDS[1],
    ),
    None,
    None,
    True,
    True,
)

ONE_BREAK_FAMILIES = tuple(
    name for name in FAMILY_SPECS if name.startswith("one_break")
)
TWO_BREAK_GRID_FAMILIES = tuple(
    name
    for name in FAMILY_SPECS
    if name.startswith("two_break_k") and name != "two_break_free_sharpness"
)


def _validated_parameters(
    family: str, parameters: np.ndarray
) -> tuple[FamilySpec, np.ndarray]:
    if family not in FAMILY_SPECS:
        raise ValueError(f"unknown Exp64 family {family!r}")
    spec = FAMILY_SPECS[family]
    values = np.asarray(parameters, float)
    if values.shape != (spec.n_parameter,):
        raise ValueError(f"{family} needs {spec.n_parameter} parameters")
    if not np.all(np.isfinite(values)):
        raise ValueError("parameters must be finite")
    lower = np.asarray(spec.lower)
    upper = np.asarray(spec.upper)
    if np.any(values < lower) or np.any(values > upper):
        raise ValueError("parameters lie outside the declared bounds")
    return spec, values


def physical_parameters(family: str, parameters: np.ndarray) -> PhysicalParameters:
    spec, values = _validated_parameters(family, parameters)
    aperture_mass = 10.0 ** values[0]
    inner_radius = 10.0 ** values[1]
    outer_slope = 2.0 + np.exp(values[-2] if spec.free_sharpness else values[-1])
    if not spec.two_break:
        return PhysicalParameters(
            aperture_mass,
            inner_radius,
            inner_radius,
            0.0,
            outer_slope,
            float(spec.inner_sharpness),
            float(spec.outer_sharpness),
        )

    outer_radius = inner_radius * 10.0 ** values[2]
    intermediate_slope = outer_slope * expit(values[3])
    if spec.free_sharpness:
        sharpness = np.exp(values[5])
        inner_sharpness = sharpness
        outer_sharpness = sharpness
    else:
        inner_sharpness = float(spec.inner_sharpness)
        outer_sharpness = float(spec.outer_sharpness)
    return PhysicalParameters(
        aperture_mass,
        inner_radius,
        outer_radius,
        intermediate_slope,
        outer_slope,
        inner_sharpness,
        outer_sharpness,
    )


def _shape_density(radius: np.ndarray, physical: PhysicalParameters) -> np.ndarray:
    radius = np.asarray(radius, float)
    if physical.intermediate_slope == 0.0:
        return np.exp(
            -physical.outer_slope
            / physical.inner_sharpness
            * np.log1p((radius / physical.inner_radius_kpc) ** physical.inner_sharpness)
        )
    inner_term = (
        -physical.intermediate_slope
        / physical.inner_sharpness
        * np.log1p((radius / physical.inner_radius_kpc) ** physical.inner_sharpness)
    )
    outer_term = (
        -(physical.outer_slope - physical.intermediate_slope)
        / physical.outer_sharpness
        * np.log1p((radius / physical.outer_radius_kpc) ** physical.outer_sharpness)
    )
    return np.exp(inner_term + outer_term)


@lru_cache(maxsize=None)
def _quadrature_nodes(order: int) -> tuple[np.ndarray, np.ndarray]:
    if order < 8:
        raise ValueError("quadrature_order must be at least 8")
    return np.polynomial.legendre.leggauss(order)


def _shape_cumulative_integral(
    radius: np.ndarray,
    physical: PhysicalParameters,
    quadrature_order: int,
) -> np.ndarray:
    requested = np.asarray(radius, float)
    positive = requested > 0.0
    output = np.zeros_like(requested)
    if not np.any(positive):
        return output

    positive_radius = requested[positive]
    if np.max(positive_radius) <= 2.0 * APERTURE_RADIUS_KPC:
        boundaries = np.unique(positive_radius)
        lower = np.concatenate(([0.0], boundaries[:-1]))
        upper = boundaries
        nodes, weights = _quadrature_nodes(quadrature_order)
        sampled_radius = (
            0.5 * (upper - lower)[:, None] * nodes[None, :]
            + 0.5 * (upper + lower)[:, None]
        )
        integrand = sampled_radius * _shape_density(sampled_radius, physical)
        interval_integral = (
            np.pi * (upper - lower) * np.sum(integrand * weights[None, :], axis=1)
        )
        cumulative = np.cumsum(interval_integral)
        output[positive] = cumulative[np.searchsorted(boundaries, positive_radius)]
        return output

    boundaries = np.unique(
        np.concatenate(
            (
                positive_radius,
                np.asarray(
                    [physical.inner_radius_kpc, physical.outer_radius_kpc], float
                ),
            )
        )
    )
    boundaries = boundaries[boundaries > 0.0]
    smallest_scale = min(boundaries[0], physical.inner_radius_kpc)
    lower_radius = smallest_scale * 1.0e-12
    log_boundaries = np.log(np.concatenate(([lower_radius], boundaries)))
    lower = log_boundaries[:-1]
    upper = log_boundaries[1:]
    nodes, weights = _quadrature_nodes(quadrature_order)
    log_radius = 0.5 * (upper - lower)[:, None] * nodes + 0.5 * (upper + lower)[:, None]
    sampled_radius = np.exp(log_radius)
    integrand = (
        2.0 * np.pi * sampled_radius**2 * _shape_density(sampled_radius, physical)
    )
    interval_integrals = (
        0.5 * (upper - lower) * np.sum(integrand * weights[None, :], axis=1)
    )
    initial_integral = np.pi * lower_radius**2
    cumulative = initial_integral + np.cumsum(interval_integrals)
    positions = np.searchsorted(boundaries, positive_radius)
    output[positive] = cumulative[positions]
    return output


def _validated_radius(radius: np.ndarray | float) -> np.ndarray:
    values = np.asarray(radius, float)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("radius must be finite and non-negative")
    return values


def cumulative_mass(
    family: str,
    parameters: np.ndarray,
    radius: np.ndarray | float,
    quadrature_order: int = DEFAULT_QUADRATURE_ORDER,
) -> np.ndarray:
    """Projected enclosed mass normalized inside the measured aperture."""
    radius_values = _validated_radius(radius)
    physical = physical_parameters(family, parameters)
    combined_radius = np.concatenate(
        (radius_values.reshape(-1), np.asarray([APERTURE_RADIUS_KPC]))
    )
    raw_mass = _shape_cumulative_integral(combined_radius, physical, quadrature_order)
    aperture_integral = raw_mass[-1]
    mass = physical.aperture_mass * raw_mass[:-1] / aperture_integral
    return mass.reshape(radius_values.shape)


def cog_log(
    family: str,
    parameters: np.ndarray,
    radius: np.ndarray | float,
    quadrature_order: int = DEFAULT_QUADRATURE_ORDER,
) -> np.ndarray:
    mass = cumulative_mass(family, parameters, radius, quadrature_order)
    if np.any(mass <= 0.0):
        raise ValueError("logarithmic CoG requires strictly positive radii")
    return np.log10(mass)


def projected_density(
    family: str,
    parameters: np.ndarray,
    radius: np.ndarray | float,
    quadrature_order: int = DEFAULT_QUADRATURE_ORDER,
) -> np.ndarray:
    """Projected surface density with mass per squared-kpc units."""
    radius_values = _validated_radius(radius)
    physical = physical_parameters(family, parameters)
    aperture_integral = _shape_cumulative_integral(
        np.asarray([APERTURE_RADIUS_KPC]), physical, quadrature_order
    )[0]
    normalization = physical.aperture_mass / aperture_integral
    return normalization * _shape_density(radius_values, physical)


def density_log_slope(
    family: str,
    parameters: np.ndarray,
    radius: np.ndarray | float,
) -> np.ndarray:
    radius_values = _validated_radius(radius)
    physical = physical_parameters(family, parameters)
    inner_power = (
        radius_values / physical.inner_radius_kpc
    ) ** physical.inner_sharpness
    inner_transition = inner_power / (1.0 + inner_power)
    if physical.intermediate_slope == 0.0:
        return -physical.outer_slope * inner_transition
    outer_power = (
        radius_values / physical.outer_radius_kpc
    ) ** physical.outer_sharpness
    outer_transition = outer_power / (1.0 + outer_power)
    return (
        -physical.intermediate_slope * inner_transition
        - (physical.outer_slope - physical.intermediate_slope) * outer_transition
    )


def abel_density(
    family: str,
    parameters: np.ndarray,
    radius: np.ndarray | float,
    quadrature_order: int = 96,
) -> np.ndarray:
    """Numerical spherical Abel deprojection at strictly positive radii."""
    radius_values = _validated_radius(radius)
    if np.any(radius_values <= 0.0):
        raise ValueError("Abel density is evaluated only at positive radii")
    nodes, weights = _quadrature_nodes(quadrature_order)
    theta = 0.25 * np.pi * (nodes + 1.0)
    theta_weights = 0.25 * np.pi * weights
    secant = 1.0 / np.cos(theta)
    projected_radius = radius_values.reshape(-1, 1) * secant[None, :]
    surface_density = projected_density(family, parameters, projected_radius)
    slope = density_log_slope(family, parameters, projected_radius)
    radial_derivative = surface_density * slope / projected_radius
    integrand = -radial_derivative * secant[None, :]
    result = np.sum(integrand * theta_weights[None, :], axis=1) / np.pi
    return result.reshape(radius_values.shape)


def shell_log_density(
    cog_log_values: np.ndarray,
    radius: np.ndarray,
    allow_nonpositive_floor: bool = False,
) -> np.ndarray:
    radius_values = _validated_radius(radius)
    cog_log_values = np.asarray(cog_log_values, float)
    if cog_log_values.shape != radius_values.shape:
        raise ValueError("CoG and radius must have the same shape")
    enclosed_mass = 10.0**cog_log_values
    shell_mass = np.diff(np.concatenate(([0.0], enclosed_mass)))
    shell_area = np.pi * np.diff(np.concatenate(([0.0], radius_values**2)))
    if np.any(shell_mass < 0.0) and not allow_nonpositive_floor:
        raise ValueError("shell masses must be non-negative")
    return np.log10(np.clip(shell_mass, 1.0, None) / shell_area)


def _observed_half_mass_radius(truth_log: np.ndarray, radius: np.ndarray) -> float:
    fraction_log = truth_log - truth_log[-1]
    return float(
        10.0
        ** np.interp(
            np.log10(0.5),
            fraction_log,
            np.log10(radius),
        )
    )


def synthetic_parameters(family: str) -> np.ndarray:
    if family not in FAMILY_SPECS:
        raise ValueError(f"unknown Exp64 family {family!r}")
    spec = FAMILY_SPECS[family]
    if not spec.two_break:
        return np.asarray([11.5, np.log10(8.0), np.log(2.2)])
    base = [11.5, np.log10(3.0), np.log10(8.0), logit(0.42), np.log(2.2)]
    if spec.free_sharpness:
        base.append(np.log(1.7))
    return np.asarray(base)


def initial_parameters(
    family: str,
    truth_log: np.ndarray,
    radius: np.ndarray,
    synthetic_truth_parameters: np.ndarray | None = None,
) -> list[np.ndarray]:
    spec = FAMILY_SPECS[family]
    lower = np.asarray(spec.lower)
    upper = np.asarray(spec.upper)
    if synthetic_truth_parameters is not None:
        truth = np.asarray(synthetic_truth_parameters, float)
        offsets = np.asarray(
            [
                [0.02, -0.12, 0.10, -0.35, 0.18, -0.12],
                [-0.03, 0.16, -0.14, 0.30, -0.15, 0.16],
                [0.01, -0.20, 0.18, 0.45, 0.10, 0.22],
                [-0.02, 0.22, -0.18, -0.50, -0.10, -0.20],
            ]
        )[:, : spec.n_parameter]
        return [
            np.clip(truth + offset, lower + 1.0e-8, upper - 1.0e-8)
            for offset in offsets
        ]

    amplitude = float(truth_log[-1])
    half_mass_radius = _observed_half_mass_radius(truth_log, radius)
    if not spec.two_break:
        configurations = (
            (0.45, 1.0),
            (0.8, 2.0),
            (1.3, 4.0),
            (2.0, 8.0),
        )
        starts = [
            np.asarray([amplitude, np.log10(scale * half_mass_radius), np.log(excess)])
            for scale, excess in configurations
        ]
    else:
        configurations = (
            (0.12, 8.0, 0.30, 1.0, 1.0),
            (0.25, 6.0, 0.45, 2.0, 2.0),
            (0.45, 4.0, 0.60, 4.0, 3.0),
            (0.70, 3.0, 0.75, 7.0, 5.0),
        )
        starts = []
        for scale, radius_ratio, mid_fraction, excess, sharpness in configurations:
            values = [
                amplitude,
                np.log10(scale * half_mass_radius),
                np.log10(radius_ratio),
                logit(mid_fraction),
                np.log(excess),
            ]
            if spec.free_sharpness:
                values.append(np.log(sharpness))
            starts.append(np.asarray(values))
    return [np.clip(start, lower + 1.0e-8, upper - 1.0e-8) for start in starts]


def _residual_vector(
    parameters: np.ndarray,
    family: str,
    radius: np.ndarray,
    truth_log: np.ndarray,
    objective: str,
) -> np.ndarray:
    prediction = cog_log(family, parameters, radius)
    if objective == "log_cog":
        return prediction - truth_log
    if objective == "absolute_cog":
        return (10.0**prediction - 10.0**truth_log) / 10.0 ** truth_log[-1]
    if objective == "log_shell":
        return shell_log_density(prediction, radius) - shell_log_density(
            truth_log, radius
        )
    raise ValueError(f"unknown fitting objective {objective!r}")


def fit_profile(
    family: str,
    radius: np.ndarray,
    truth_log: np.ndarray,
    objective: str = "log_cog",
    max_nfev: int = 600,
    synthetic_truth_parameters: np.ndarray | None = None,
) -> dict[str, object]:
    """Fit one profile from separated starts and report local diagnostics."""
    spec = FAMILY_SPECS[family]
    lower = np.asarray(spec.lower)
    upper = np.asarray(spec.upper)
    solutions = []
    for start in initial_parameters(
        family, truth_log, radius, synthetic_truth_parameters
    ):
        solution = least_squares(
            _residual_vector,
            start,
            args=(family, radius, truth_log, objective),
            bounds=(lower, upper),
            method="trf",
            x_scale="jac",
            max_nfev=max_nfev,
        )
        prediction = cog_log(family, solution.x, radius)
        score = float(np.mean((prediction - truth_log) ** 2))
        solutions.append((score, solution, prediction))
    solutions.sort(key=lambda item: item[0])
    score, best, prediction = solutions[0]
    near = [item for item in solutions if item[0] <= 1.01 * score + 1.0e-14]
    near_parameters = np.asarray([item[1].x for item in near])
    near_predictions = np.asarray([item[2] for item in near])
    parameter_range = upper - lower
    scaled_jacobian = best.jac * parameter_range[None, :]
    singular_values = np.linalg.svd(scaled_jacobian, compute_uv=False)
    position = (best.x - lower) / parameter_range
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
