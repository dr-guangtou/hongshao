"""Narrow damped-cosine coordinate systems for Exp69."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import least_squares

from hongshao.tng_data import COG_RAD_KPC

RADII_KPC = np.asarray(COG_RAD_KPC, float)
APERTURE_RADIUS_KPC = float(RADII_KPC[-1])
DEFAULT_GRID_SIZE = 512
TAIL_SLOPE_MARGIN = 0.02

LOG_MASS_BOUNDS = (8.0, 14.0)
LOG_RADIUS_BOUNDS = (np.log10(0.3), np.log10(300.0))
LOG_A_BOUNDS = (-4.0, 4.0)
SIGNED_CONTRAST_BOUNDS = (-20.0, 20.0)
LOG_FREQUENCY_BOUNDS = (np.log(0.25), np.log(8.0))
LOG_DAMPING_BOUNDS = (np.log(0.05), np.log(3.0))
FIXED_DAMPING_VALUES = (0.0, 0.25, 0.75, 1.5)


@dataclass(frozen=True)
class CandidateSpec:
    """One predeclared optimizer coordinate system."""

    name: str
    label: str
    parameter_names: tuple[str, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    coordinate_system: str
    fixed_damping: float | None = None
    orthogonalized: bool = False

    @property
    def n_parameter(self) -> int:
        return len(self.parameter_names)


def _window_bounds() -> tuple[tuple[float, ...], tuple[float, ...]]:
    core_radii = 10.0 ** np.asarray(LOG_RADIUS_BOUNDS)
    delta_values = []
    for core_radius in core_radii:
        u_min = np.log1p(RADII_KPC[0] / core_radius)
        u_max = np.log1p(RADII_KPC[-1] / core_radius)
        delta_values.append(u_max - u_min)
    minimum_delta = min(delta_values)
    maximum_delta = max(delta_values)
    damping_span = (
        np.exp(LOG_DAMPING_BOUNDS[0]) * minimum_delta,
        np.exp(LOG_DAMPING_BOUNDS[1]) * maximum_delta,
    )
    phase_span = (
        np.exp(LOG_FREQUENCY_BOUNDS[0]) * minimum_delta,
        np.exp(LOG_FREQUENCY_BOUNDS[1]) * maximum_delta,
    )
    lower = (
        LOG_MASS_BOUNDS[0],
        LOG_RADIUS_BOUNDS[0],
        2.0 * LOG_A_BOUNDS[0],
        SIGNED_CONTRAST_BOUNDS[0],
        np.log(damping_span[0]),
        np.log(phase_span[0]),
    )
    upper = (
        LOG_MASS_BOUNDS[1],
        LOG_RADIUS_BOUNDS[1],
        2.0 * LOG_A_BOUNDS[1],
        SIGNED_CONTRAST_BOUNDS[1],
        np.log(damping_span[1]),
        np.log(phase_span[1]),
    )
    return lower, upper


def candidate_specs() -> dict[str, CandidateSpec]:
    """Return the exact ordered candidate set frozen in the predeclaration."""
    original_names = (
        "log_mstar_148",
        "log_core_radius",
        "log_a",
        "signed_contrast",
        "log_frequency",
        "log_damping",
    )
    original_lower = (
        LOG_MASS_BOUNDS[0],
        LOG_RADIUS_BOUNDS[0],
        LOG_A_BOUNDS[0],
        SIGNED_CONTRAST_BOUNDS[0],
        LOG_FREQUENCY_BOUNDS[0],
        LOG_DAMPING_BOUNDS[0],
    )
    original_upper = (
        LOG_MASS_BOUNDS[1],
        LOG_RADIUS_BOUNDS[1],
        LOG_A_BOUNDS[1],
        SIGNED_CONTRAST_BOUNDS[1],
        LOG_FREQUENCY_BOUNDS[1],
        LOG_DAMPING_BOUNDS[1],
    )
    specs: dict[str, CandidateSpec] = {
        "original_free": CandidateSpec(
            "original_free",
            "Original free damping",
            original_names,
            original_lower,
            original_upper,
            "original",
        )
    }
    for damping, suffix in zip(
        FIXED_DAMPING_VALUES, ("0", "0p25", "0p75", "1p5"), strict=True
    ):
        specs[f"fixed_lambda_{suffix}"] = CandidateSpec(
            f"fixed_lambda_{suffix}",
            rf"Fixed $\lambda={damping:g}$",
            original_names[:-1],
            original_lower[:-1],
            original_upper[:-1],
            "fixed",
            fixed_damping=damping,
        )
    window_lower, window_upper = _window_bounds()
    specs["window_pivot"] = CandidateSpec(
        "window_pivot",
        "Finite-window pivot coordinates",
        (
            "log_mstar_148",
            "log_core_radius",
            "log_a_squared",
            "pivot_contrast",
            "log_damping_span",
            "log_phase_advance",
        ),
        window_lower,
        window_upper,
        "pivot",
    )
    specs["window_orthogonal"] = CandidateSpec(
        "window_orthogonal",
        "Finite-window orthogonal modulation",
        original_names,
        original_lower,
        original_upper,
        "original",
        orthogonalized=True,
    )
    return specs


def _validate_parameters(spec: CandidateSpec, parameters: np.ndarray) -> np.ndarray:
    values = np.asarray(parameters, float)
    if values.shape != (spec.n_parameter,):
        raise ValueError(f"{spec.name} needs {spec.n_parameter} parameters")
    lower = np.asarray(spec.lower)
    upper = np.asarray(spec.upper)
    if not np.all(np.isfinite(values)):
        raise ValueError("parameters must be finite")
    if np.any(values < lower - 1.0e-12) or np.any(values > upper + 1.0e-12):
        raise ValueError("parameters lie outside the declared bounds")
    return values


def _window_coordinates(log_core_radius: float) -> tuple[float, float, float]:
    core_radius = 10.0**log_core_radius
    u_min = np.log1p(RADII_KPC[0] / core_radius)
    u_max = np.log1p(RADII_KPC[-1] / core_radius)
    return u_min, u_max, u_max - u_min


def original_to_pivot(parameters: np.ndarray) -> np.ndarray:
    """Map original free coordinates to the finite-window coordinates."""
    values = np.asarray(parameters, float)
    if values.shape != (6,):
        raise ValueError("original coordinates must have length six")
    log_mass, log_core, log_a, contrast, log_frequency, log_damping = values
    u_min, u_max, delta_u = _window_coordinates(log_core)
    pivot_u = 0.5 * (u_min + u_max)
    damping = np.exp(log_damping)
    frequency = np.exp(log_frequency)
    return np.asarray(
        (
            log_mass,
            log_core,
            2.0 * log_a,
            contrast * np.exp(-damping * pivot_u),
            np.log(damping * delta_u),
            np.log(frequency * delta_u),
        )
    )


def pivot_to_original(parameters: np.ndarray) -> np.ndarray:
    """Decode finite-window coordinates into the original physical form."""
    values = np.asarray(parameters, float)
    if values.shape != (6,):
        raise ValueError("pivot coordinates must have length six")
    log_mass, log_core, log_a_squared, pivot_contrast, log_decay, log_phase = values
    u_min, u_max, delta_u = _window_coordinates(log_core)
    pivot_u = 0.5 * (u_min + u_max)
    damping = np.exp(log_decay) / delta_u
    frequency = np.exp(log_phase) / delta_u
    contrast = pivot_contrast * np.exp(damping * pivot_u)
    return np.asarray(
        (
            log_mass,
            log_core,
            0.5 * log_a_squared,
            contrast,
            np.log(frequency),
            np.log(damping),
        )
    )


def _decoded_original(spec: CandidateSpec, parameters: np.ndarray) -> np.ndarray:
    values = _validate_parameters(spec, parameters)
    if spec.coordinate_system == "pivot":
        return pivot_to_original(values)
    if spec.fixed_damping is not None:
        return np.concatenate((values, [np.log(max(spec.fixed_damping, 1.0e-300))]))
    return values.copy()


def physical_descriptors(spec: CandidateSpec, parameters: np.ndarray) -> np.ndarray:
    """Return common mass, scale, slope, pivot, decay, and phase descriptors."""
    original = _decoded_original(spec, parameters)
    log_mass, log_core, log_a, contrast, log_frequency, log_damping = original
    u_min, u_max, delta_u = _window_coordinates(log_core)
    pivot_u = 0.5 * (u_min + u_max)
    damping = 0.0 if spec.fixed_damping == 0.0 else np.exp(log_damping)
    frequency = np.exp(log_frequency)
    return np.asarray(
        (
            log_mass,
            log_core,
            np.exp(2.0 * log_a),
            contrast * np.exp(-damping * pivot_u),
            damping * delta_u,
            frequency * delta_u,
        )
    )


def descriptor_bounds(spec: CandidateSpec) -> tuple[np.ndarray, np.ndarray]:
    """Return the Cartesian image bounds of the declared native box."""
    corners = np.asarray(
        [
            physical_descriptors(spec, np.asarray(values))
            for values in product(*zip(spec.lower, spec.upper, strict=True))
        ]
    )
    return np.min(corners, axis=0), np.max(corners, axis=0)


def normalized_parameters(spec: CandidateSpec, parameters: np.ndarray) -> np.ndarray:
    """Map native optimizer coordinates onto their declared unit box."""
    values = _validate_parameters(spec, parameters)
    lower = np.asarray(spec.lower)
    upper = np.asarray(spec.upper)
    return (values - lower) / (upper - lower)


def orthogonal_modulation(
    parameters: np.ndarray,
    coordinate: np.ndarray,
    *,
    fit_coordinate: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate the modulation after finite-window core orthogonalization."""
    values = np.asarray(parameters, float)
    log_core = values[1]
    frequency = np.exp(values[4])
    damping = np.exp(values[5])
    if fit_coordinate is None:
        core_radius = 10.0**log_core
        fit_coordinate = np.log1p(RADII_KPC / core_radius)
    fit_coordinate = np.asarray(fit_coordinate, float)
    harmonic = np.exp(-damping * fit_coordinate) * np.cos(frequency * fit_coordinate)
    core_direction = 1.0 - np.exp(-fit_coordinate)
    design = np.column_stack((np.ones_like(fit_coordinate), core_direction))
    _, core_coefficient = np.linalg.lstsq(design, harmonic, rcond=None)[0]
    requested = np.asarray(coordinate, float)
    return np.exp(-damping * requested) * np.cos(
        frequency * requested
    ) + core_coefficient * np.exp(-requested)


def _modulation_components(
    spec: CandidateSpec, parameters: np.ndarray
) -> list[tuple[float, float, float]]:
    original = _decoded_original(spec, parameters)
    damping = 0.0 if spec.fixed_damping == 0.0 else np.exp(original[5])
    frequency = np.exp(original[4])
    components = [(1.0, damping, frequency)]
    if spec.orthogonalized:
        core_radius = 10.0 ** original[1]
        fit_coordinate = np.log1p(RADII_KPC / core_radius)
        harmonic = np.exp(-damping * fit_coordinate) * np.cos(
            frequency * fit_coordinate
        )
        core_direction = 1.0 - np.exp(-fit_coordinate)
        design = np.column_stack((np.ones_like(fit_coordinate), core_direction))
        _, core_coefficient = np.linalg.lstsq(design, harmonic, rcond=None)[0]
        components.append((float(core_coefficient), 1.0, 0.0))
    return components


def logarithmic_density_rate(
    spec: CandidateSpec, parameters: np.ndarray, coordinate: np.ndarray
) -> np.ndarray:
    """Return the positive squared logarithmic-density rate dQ/du."""
    original = _decoded_original(spec, parameters)
    coordinate_values = np.asarray(coordinate, float)
    a_value = np.exp(original[2])
    contrast = original[3]
    modulation = np.zeros_like(coordinate_values)
    for coefficient, decay, frequency in _modulation_components(spec, parameters):
        modulation += (
            coefficient
            * np.exp(-decay * coordinate_values)
            * np.cos(frequency * coordinate_values)
        )
    bracket = 2.0 + TAIL_SLOPE_MARGIN + (a_value + contrast * modulation) ** 2
    return -np.expm1(-coordinate_values) * bracket


def _exponential_cosine_integral(
    decay: float, frequency: float, coordinate: np.ndarray
) -> np.ndarray:
    if frequency == 0.0 and decay == 0.0:
        return coordinate.copy()
    denominator = decay**2 + frequency**2
    angle = frequency * coordinate
    endpoint = np.exp(-decay * coordinate) * (
        -decay * np.cos(angle) + frequency * np.sin(angle)
    )
    return (endpoint + decay) / denominator


def _weighted_cosine_integral(
    decay: float, frequency: float, coordinate: np.ndarray
) -> np.ndarray:
    return _exponential_cosine_integral(
        decay, frequency, coordinate
    ) - _exponential_cosine_integral(decay + 1.0, frequency, coordinate)


def integrated_density_rate(
    spec: CandidateSpec, parameters: np.ndarray, coordinate: np.ndarray
) -> np.ndarray:
    """Return the elementary finite-sum density exponent Q(u)."""
    original = _decoded_original(spec, parameters)
    coordinate_values = np.asarray(coordinate, float)
    a_value = np.exp(original[2])
    contrast = original[3]
    base = (2.0 + TAIL_SLOPE_MARGIN + a_value**2) * (
        coordinate_values + np.exp(-coordinate_values) - 1.0
    )
    components = _modulation_components(spec, parameters)
    first = np.zeros_like(coordinate_values)
    for coefficient, decay, frequency in components:
        first += coefficient * _weighted_cosine_integral(
            decay, frequency, coordinate_values
        )
    squared = np.zeros_like(coordinate_values)
    for first_component in components:
        for second_component in components:
            coefficient = first_component[0] * second_component[0]
            decay = first_component[1] + second_component[1]
            difference = first_component[2] - second_component[2]
            total = first_component[2] + second_component[2]
            squared += (
                0.5
                * coefficient
                * (
                    _weighted_cosine_integral(decay, difference, coordinate_values)
                    + _weighted_cosine_integral(decay, total, coordinate_values)
                )
            )
    return base + 2.0 * a_value * contrast * first + contrast**2 * squared


def _components(
    spec: CandidateSpec,
    parameters: np.ndarray,
    requested_radius: np.ndarray,
    grid_size: int,
) -> dict[str, np.ndarray | float]:
    if grid_size < 128:
        raise ValueError("grid_size must be at least 128")
    original = _decoded_original(spec, parameters)
    radius = np.asarray(requested_radius, float)
    if not np.all(np.isfinite(radius)) or np.any(radius < 0.0):
        raise ValueError("radius must be finite and non-negative")
    core_radius = 10.0 ** original[1]
    maximum_radius = max(float(np.max(radius, initial=0.0)), APERTURE_RADIUS_KPC)
    maximum_coordinate = np.log1p(maximum_radius / core_radius)
    coordinate = np.linspace(0.0, maximum_coordinate, grid_size)
    integrated_rate = integrated_density_rate(spec, parameters, coordinate)
    density_shape = np.exp(-integrated_rate)
    radial_grid = core_radius * np.expm1(coordinate)
    radial_derivative = core_radius * np.exp(coordinate)
    annular_integrand = 2.0 * np.pi * density_shape * radial_grid * radial_derivative
    cumulative_mass_shape = cumulative_trapezoid(
        annular_integrand, coordinate, initial=0.0
    )
    aperture_coordinate = np.log1p(APERTURE_RADIUS_KPC / core_radius)
    aperture_mass_shape = float(
        np.interp(aperture_coordinate, coordinate, cumulative_mass_shape)
    )
    if not np.isfinite(aperture_mass_shape) or aperture_mass_shape <= 0.0:
        raise FloatingPointError("nonpositive aperture normalization")
    return {
        "core_radius": core_radius,
        "coordinate": coordinate,
        "integrated_rate": integrated_rate,
        "cumulative_mass_shape": cumulative_mass_shape,
        "aperture_mass_shape": aperture_mass_shape,
    }


def cog_log(
    spec: CandidateSpec,
    parameters: np.ndarray,
    radius: np.ndarray | float,
    grid_size: int = DEFAULT_GRID_SIZE,
) -> np.ndarray:
    """Evaluate aperture-normalized logarithmic cumulative projected mass."""
    radius_values = np.asarray(radius, float)
    if np.any(radius_values <= 0.0):
        raise ValueError("logarithmic CoG needs positive radii")
    components = _components(spec, parameters, radius_values, grid_size)
    requested_coordinate = np.log1p(
        radius_values.reshape(-1) / float(components["core_radius"])
    )
    raw_mass = np.interp(
        requested_coordinate,
        np.asarray(components["coordinate"]),
        np.asarray(components["cumulative_mass_shape"]),
    )
    aperture_mass = 10.0 ** physical_descriptors(spec, parameters)[0]
    mass = aperture_mass * raw_mass / float(components["aperture_mass_shape"])
    return np.log10(mass.reshape(radius_values.shape))


def surface_density(
    spec: CandidateSpec,
    parameters: np.ndarray,
    radius: np.ndarray | float,
    grid_size: int = DEFAULT_GRID_SIZE,
) -> np.ndarray:
    """Evaluate the aperture-normalized projected surface density."""
    radius_values = np.asarray(radius, float)
    components = _components(spec, parameters, radius_values, grid_size)
    requested_coordinate = np.log1p(
        radius_values.reshape(-1) / float(components["core_radius"])
    )
    integrated_rate = np.interp(
        requested_coordinate,
        np.asarray(components["coordinate"]),
        np.asarray(components["integrated_rate"]),
    )
    scale = 10.0 ** physical_descriptors(spec, parameters)[0] / float(
        components["aperture_mass_shape"]
    )
    return (scale * np.exp(-integrated_rate)).reshape(radius_values.shape)


def shell_log_density(log_cog: np.ndarray, radius: np.ndarray) -> np.ndarray:
    """Return log10 mean surface density in the measured annuli."""
    values = 10.0 ** np.asarray(log_cog, float)
    shell_mass = np.concatenate(([values[0]], np.diff(values)))
    if np.any(shell_mass < -1.0e-8 * values[-1]):
        raise ValueError("candidate CoG produced a negative shell")
    shell_mass = np.maximum(shell_mass, np.finfo(float).tiny)
    radius_values = np.asarray(radius, float)
    inner_squared = np.concatenate(([0.0], radius_values[:-1] ** 2))
    return np.log10(shell_mass / (np.pi * (radius_values**2 - inner_squared)))


def _objective_residual(
    parameters: np.ndarray,
    spec: CandidateSpec,
    radius: np.ndarray,
    truth_log: np.ndarray,
    objective: str,
    grid_size: int,
) -> np.ndarray:
    try:
        prediction = cog_log(spec, parameters, radius, grid_size)
        if objective == "log_cog":
            return prediction - truth_log
        if objective == "absolute_cog":
            return (10.0**prediction - 10.0**truth_log) / 10.0 ** truth_log[-1]
        if objective == "log_shell":
            return shell_log_density(prediction, radius) - shell_log_density(
                truth_log, radius
            )
    except (ValueError, FloatingPointError, OverflowError):
        return np.full(len(radius), 100.0)
    raise ValueError(f"unknown objective {objective!r}")


def _half_aperture_radius(truth_log: np.ndarray, radius: np.ndarray) -> float:
    target = float(truth_log[-1] + np.log10(0.5))
    if target <= truth_log[0]:
        return float(radius[0] * 10.0 ** (0.5 * (target - truth_log[0])))
    return float(10.0 ** np.interp(target, truth_log, np.log10(radius)))


def synthetic_parameters(spec: CandidateSpec, case_index: int = 0) -> np.ndarray:
    """Return one of six frozen synthetic cases spanning damping and contrast."""
    cases = (
        (0.05, 1.0, -1.3),
        (0.25, 1.8, 1.3),
        (0.75, 0.7, -5.0),
        (1.5, 2.5, 5.0),
        (0.25, 4.0, -1.3),
        (1.5, 0.4, 1.3),
    )
    damping, frequency, contrast = cases[case_index % len(cases)]
    if spec.fixed_damping is not None:
        damping = spec.fixed_damping
    original = np.asarray(
        (
            11.5,
            np.log10(12.0),
            np.log(1.6),
            contrast,
            np.log(frequency),
            np.log(max(damping, 0.05)),
        )
    )
    if spec.coordinate_system == "pivot":
        values = original_to_pivot(original)
    elif spec.fixed_damping is not None:
        values = original[:-1]
    else:
        values = original
    return np.clip(
        values, np.asarray(spec.lower) + 1.0e-7, np.asarray(spec.upper) - 1.0e-7
    )


def _initial_parameters(
    spec: CandidateSpec,
    truth_log: np.ndarray,
    radius: np.ndarray,
    n_starts: int,
    synthetic_truth_parameters: np.ndarray | None,
) -> list[np.ndarray]:
    lower = np.asarray(spec.lower)
    upper = np.asarray(spec.upper)
    if synthetic_truth_parameters is not None:
        truth = np.asarray(synthetic_truth_parameters, float)
        fractions = (0.015, -0.025, 0.04, -0.06, 0.08, -0.1, 0.12, -0.14)
        direction = np.linspace(-1.0, 1.0, spec.n_parameter)
        return [
            np.clip(
                truth + fraction * (upper - lower) * direction,
                lower + 1.0e-7,
                upper - 1.0e-7,
            )
            for fraction in fractions[:n_starts]
        ]
    half_radius = _half_aperture_radius(truth_log, radius)
    templates = (
        (0.6, 0.7, -2.0, 0.5, 0.25),
        (1.0, 1.5, 2.0, 1.5, 0.75),
        (1.7, 3.0, -5.0, 3.0, 1.5),
        (0.35, 0.25, 5.0, 0.5, 0.05),
        (2.2, 0.5, -9.0, 5.0, 0.25),
        (0.8, 2.5, 9.0, 0.3, 1.5),
        (1.3, 1.0, -1.0, 7.0, 0.75),
        (0.45, 4.0, 1.0, 1.0, 0.05),
    )
    starts = []
    for radius_factor, a_value, contrast, frequency, damping in templates[:n_starts]:
        original = np.asarray(
            (
                truth_log[-1],
                np.log10(np.clip(radius_factor * half_radius, 0.3, 300.0)),
                np.log(a_value),
                contrast,
                np.log(frequency),
                np.log(damping),
            )
        )
        if spec.fixed_damping is not None:
            start = original[:-1]
        elif spec.coordinate_system == "pivot":
            start = original_to_pivot(original)
        else:
            start = original
        starts.append(np.clip(start, lower + 1.0e-7, upper - 1.0e-7))
    return starts


def fit_profile(
    spec: CandidateSpec,
    radius: np.ndarray,
    truth_log: np.ndarray,
    *,
    objective: str = "log_cog",
    grid_size: int = DEFAULT_GRID_SIZE,
    n_starts: int = 4,
    max_nfev: int = 400,
    synthetic_truth_parameters: np.ndarray | None = None,
) -> dict[str, object]:
    """Fit one profile and retain every start plus local SVD diagnostics."""
    radius_values = np.asarray(radius, float)
    truth_values = np.asarray(truth_log, float)
    lower = np.asarray(spec.lower)
    upper = np.asarray(spec.upper)
    parameter_range = upper - lower
    solutions = []
    for start in _initial_parameters(
        spec,
        truth_values,
        radius_values,
        n_starts,
        synthetic_truth_parameters,
    ):
        solution = least_squares(
            _objective_residual,
            start,
            args=(spec, radius_values, truth_values, objective, grid_size),
            bounds=(lower, upper),
            method="trf",
            x_scale="jac",
            max_nfev=max_nfev,
        )
        prediction = cog_log(spec, solution.x, radius_values, grid_size)
        residual = _objective_residual(
            solution.x, spec, radius_values, truth_values, objective, grid_size
        )
        scaled_jacobian = solution.jac * parameter_range[None, :]
        _, singular_values, right_vectors = np.linalg.svd(
            scaled_jacobian, full_matrices=False
        )
        solutions.append(
            {
                "score": float(np.mean(residual**2)),
                "parameters": solution.x,
                "prediction": prediction,
                "success": bool(solution.success and np.all(np.isfinite(prediction))),
                "nfev": int(solution.nfev),
                "singular_values": singular_values,
                "weakest_direction": right_vectors[-1] / parameter_range,
                "descriptors": physical_descriptors(spec, solution.x),
            }
        )
    solutions.sort(key=lambda item: item["score"])
    scores = np.asarray([item["score"] for item in solutions])
    near = np.flatnonzero(scores <= 1.01 * scores[0] + 1.0e-14)
    singular_minimum = np.asarray([item["singular_values"][-1] for item in solutions])
    descriptors = np.asarray([item["descriptors"] for item in solutions])
    from exp69_diagnostics import canonical_solution_index

    canonical = canonical_solution_index(scores, singular_minimum, descriptors)
    best = solutions[0]
    position = (best["parameters"] - lower) / parameter_range
    near_predictions = np.asarray([solutions[index]["prediction"] for index in near])
    near_parameters = np.asarray([solutions[index]["parameters"] for index in near])
    first_singular = best["singular_values"][0]
    return {
        "parameters": best["parameters"],
        "prediction": best["prediction"],
        "objective_rms": float(np.sqrt(best["score"])),
        "cog_rms_dex": float(
            np.sqrt(np.mean((best["prediction"] - truth_values) ** 2))
        ),
        "success": best["success"],
        "nfev": int(sum(item["nfev"] for item in solutions)),
        "jacobian_min_ratio": float(best["singular_values"][-1] / first_singular)
        if first_singular > 0.0
        else 0.0,
        "singular_values": best["singular_values"],
        "weakest_direction": best["weakest_direction"],
        "boundary_distance": float(np.min(np.minimum(position, 1.0 - position))),
        "near_solution_count": int(len(near)),
        "near_parameter_spread": float(
            np.max(np.ptp(near_parameters, axis=0) / parameter_range)
        )
        if len(near) > 1
        else 0.0,
        "near_profile_spread_dex": float(
            np.max(
                np.sqrt(
                    np.mean(
                        (near_predictions[:, None] - near_predictions[None, :]) ** 2,
                        axis=2,
                    )
                )
            )
        )
        if len(near) > 1
        else 0.0,
        "canonical_index": int(canonical),
        "canonical_parameters": solutions[canonical]["parameters"],
        "canonical_prediction": solutions[canonical]["prediction"],
        "all_scores": scores,
        "all_parameters": np.asarray([item["parameters"] for item in solutions]),
        "all_predictions": np.asarray([item["prediction"] for item in solutions]),
        "all_success": np.asarray([item["success"] for item in solutions], bool),
        "all_singular_minimum": singular_minimum,
        "all_singular_values": np.asarray(
            [item["singular_values"] for item in solutions]
        ),
        "all_descriptors": descriptors,
        "near_indices": near,
    }
