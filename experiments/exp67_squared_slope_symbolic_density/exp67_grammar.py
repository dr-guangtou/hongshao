"""Squared-slope projected-density grammar for Exp67."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
EXP66_DIR = HERE.parent / "exp66_expanded_symbolic_density"
sys.path.insert(0, str(EXP66_DIR))

from exp66_grammar import (  # noqa: E402
    APERTURE_RADIUS_KPC,
    DEFAULT_GRID_SIZE,
    LOG_MASS_BOUNDS,
    LOG_RADIUS_BOUNDS,
    RADII_KPC as _RADII_KPC,
    TAIL_SLOPE_MARGIN,
    ExpressionSpec,
    _atom_values,
    build_round_b_specs as _build_exp66_round_b_specs,
    round_a_specs as _exp66_round_a_specs,
)

RADII_KPC = _RADII_KPC
LOG_A_BOUNDS = (-4.0, 4.0)
SIGNED_CONTRAST_BOUNDS = (-20.0, 20.0)


def _convert_spec(spec: ExpressionSpec) -> ExpressionSpec:
    names = ["log_mstar_148", "log_core_radius", "log_a"]
    lower = [LOG_MASS_BOUNDS[0], LOG_RADIUS_BOUNDS[0], LOG_A_BOUNDS[0]]
    upper = [LOG_MASS_BOUNDS[1], LOG_RADIUS_BOUNDS[1], LOG_A_BOUNDS[1]]
    if spec.atoms:
        names.append("signed_contrast")
        lower.append(SIGNED_CONTRAST_BOUNDS[0])
        upper.append(SIGNED_CONTRAST_BOUNDS[1])
        names.extend(spec.parameter_names[4:])
        lower.extend(spec.lower[4:])
        upper.extend(spec.upper[4:])
    if not spec.atoms:
        integration_class = "elementary_squared_constant"
    elif (
        spec.operation == "atom"
        and len(spec.atoms) == 1
        and spec.atoms[0].kind in {"harmonic", "damped_harmonic"}
    ):
        integration_class = "elementary_squared_harmonic"
    else:
        integration_class = "numerical_squared"
    return ExpressionSpec(
        name=spec.name,
        label=spec.label,
        atoms=spec.atoms,
        operation=spec.operation,
        parameter_names=tuple(names),
        lower=tuple(lower),
        upper=tuple(upper),
        integration_class=integration_class,
    )


def round_a_specs() -> dict[str, ExpressionSpec]:
    """Return all 88 Exp66 atoms under the Exp67 squared-slope wrapper."""
    return {name: _convert_spec(spec) for name, spec in _exp66_round_a_specs().items()}


def build_round_b_specs(
    transitions: list[ExpressionSpec], trigonometric: list[ExpressionSpec]
) -> dict[str, ExpressionSpec]:
    """Generate and convert the deterministic bounded Exp66 expression trees."""
    generated = _build_exp66_round_b_specs(transitions, trigonometric)
    return {name: _convert_spec(spec) for name, spec in generated.items()}


def _validate_parameters(spec: ExpressionSpec, parameters: np.ndarray) -> np.ndarray:
    values = np.asarray(parameters, float)
    if values.shape != (spec.n_parameter,):
        raise ValueError(f"{spec.name} needs {spec.n_parameter} parameters")
    lower = np.asarray(spec.lower)
    upper = np.asarray(spec.upper)
    if not np.all(np.isfinite(values)):
        raise ValueError("parameters must be finite")
    if np.any(values < lower) or np.any(values > upper):
        raise ValueError("parameters lie outside the declared bounds")
    return values


def normalized_expression_value(
    expression: ExpressionSpec,
    parameters: np.ndarray,
    coordinate: np.ndarray,
) -> np.ndarray:
    """Evaluate the bounded atom tree T(u)."""
    spec = expression
    values = _validate_parameters(spec, parameters)
    coordinate_values = np.asarray(coordinate, float)
    if not spec.atoms:
        return np.zeros_like(coordinate_values)
    atoms = [
        _atom_values(atom, index, values, spec, coordinate_values)
        for index, atom in enumerate(spec.atoms, start=1)
    ]
    if spec.operation == "atom":
        output = atoms[0]
    elif spec.operation == "mean":
        output = 0.5 * (atoms[0] + atoms[1])
    elif spec.operation == "product":
        output = atoms[0] * atoms[1]
    elif spec.operation == "cube":
        output = atoms[0] ** 3
    elif spec.operation == "centered_square":
        output = 2.0 * atoms[0] ** 2 - 1.0
    else:
        raise ValueError(f"unknown operation {spec.operation!r}")
    return np.clip(output, -1.0, 1.0)


def logarithmic_density_rate(
    expression: ExpressionSpec,
    parameters: np.ndarray,
    coordinate: np.ndarray,
) -> np.ndarray:
    """Return q=dQ/du with algebraically positive squared modulation."""
    spec = expression
    values = _validate_parameters(spec, parameters)
    coordinate_values = np.asarray(coordinate, float)
    a_value = np.exp(values[2])
    contrast = values[3] if spec.atoms else 0.0
    transition = normalized_expression_value(spec, values, coordinate_values)
    bracket = TAIL_SLOPE_MARGIN + 2.0 + (a_value + contrast * transition) ** 2
    return -np.expm1(-coordinate_values) * bracket


def _exponential_integral(decay: float, coordinate: np.ndarray) -> np.ndarray:
    if decay == 0.0:
        return coordinate.copy()
    return -np.expm1(-decay * coordinate) / decay


def _exponential_sine_integral(
    decay: float,
    frequency: float,
    phase: float,
    coordinate: np.ndarray,
) -> np.ndarray:
    denominator = decay**2 + frequency**2
    angle = frequency * coordinate + phase
    endpoint = np.exp(-decay * coordinate) * (
        -decay * np.sin(angle) - frequency * np.cos(angle)
    )
    origin = decay * np.sin(phase) + frequency * np.cos(phase)
    return (endpoint + origin) / denominator


def _exponential_cosine_integral(
    decay: float,
    frequency: float,
    phase: float,
    coordinate: np.ndarray,
) -> np.ndarray:
    denominator = decay**2 + frequency**2
    angle = frequency * coordinate + phase
    endpoint = np.exp(-decay * coordinate) * (
        -decay * np.cos(angle) + frequency * np.sin(angle)
    )
    origin = decay * np.cos(phase) - frequency * np.sin(phase)
    return (endpoint + origin) / denominator


def integrated_density_rate(
    expression: ExpressionSpec,
    parameters: np.ndarray,
    coordinate: np.ndarray,
) -> np.ndarray:
    """Return Q(u), exactly for constant and single harmonic atom trees."""
    spec = expression
    values = _validate_parameters(spec, parameters)
    coordinate_values = np.asarray(coordinate, float)
    a_value = np.exp(values[2])
    base = (2.0 + TAIL_SLOPE_MARGIN + a_value**2) * (
        coordinate_values + np.exp(-coordinate_values) - 1.0
    )
    if spec.integration_class == "elementary_squared_constant":
        return base
    if spec.integration_class == "elementary_squared_harmonic":
        atom = spec.atoms[0]
        lookup = dict(zip(spec.parameter_names, values, strict=True))
        frequency = (
            np.exp(lookup["log_frequency_1"])
            if atom.free_frequency
            else float(atom.frequency)
        )
        phase = lookup["phase_1"] if atom.free_phase else atom.phase
        damping = (
            np.exp(lookup["log_damping_1"])
            if atom.free_damping
            else float(atom.damping or 0.0)
        )
        contrast = values[3]
        first_harmonic = _exponential_sine_integral(
            damping, frequency, phase, coordinate_values
        ) - _exponential_sine_integral(
            damping + 1.0, frequency, phase, coordinate_values
        )
        squared_constant = _exponential_integral(
            2.0 * damping, coordinate_values
        ) - _exponential_integral(2.0 * damping + 1.0, coordinate_values)
        second_harmonic = _exponential_cosine_integral(
            2.0 * damping, 2.0 * frequency, 2.0 * phase, coordinate_values
        ) - _exponential_cosine_integral(
            2.0 * damping + 1.0,
            2.0 * frequency,
            2.0 * phase,
            coordinate_values,
        )
        return (
            base
            + 2.0 * a_value * contrast * first_harmonic
            + 0.5 * contrast**2 * (squared_constant - second_harmonic)
        )
    if coordinate_values.ndim != 1 or coordinate_values[0] != 0.0:
        raise ValueError(
            "numerical integration requires a one-dimensional grid from zero"
        )
    rate = logarithmic_density_rate(spec, values, coordinate_values)
    return cumulative_trapezoid(rate, coordinate_values, initial=0.0)


def _components(
    spec: ExpressionSpec,
    parameters: np.ndarray,
    requested_radius: np.ndarray,
    grid_size: int,
) -> dict[str, np.ndarray | float]:
    if grid_size < 128:
        raise ValueError("grid_size must be at least 128")
    values = _validate_parameters(spec, parameters)
    radius = np.asarray(requested_radius, float)
    if not np.all(np.isfinite(radius)) or np.any(radius < 0.0):
        raise ValueError("radius must be finite and non-negative")
    core_radius = 10.0 ** values[1]
    maximum_radius = max(float(np.max(radius, initial=0.0)), APERTURE_RADIUS_KPC)
    maximum_coordinate = np.log1p(maximum_radius / core_radius)
    coordinate = np.linspace(0.0, maximum_coordinate, grid_size)
    integrated_rate = integrated_density_rate(spec, values, coordinate)
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
    expression: ExpressionSpec,
    parameters: np.ndarray,
    radius: np.ndarray | float,
    grid_size: int = DEFAULT_GRID_SIZE,
) -> np.ndarray:
    """Evaluate the aperture-normalized logarithmic cumulative mass."""
    radius_values = np.asarray(radius, float)
    if np.any(radius_values <= 0.0):
        raise ValueError("logarithmic CoG needs strictly positive radii")
    components = _components(expression, parameters, radius_values, grid_size)
    requested_coordinate = np.log1p(
        radius_values.reshape(-1) / float(components["core_radius"])
    )
    raw_mass = np.interp(
        requested_coordinate,
        np.asarray(components["coordinate"]),
        np.asarray(components["cumulative_mass_shape"]),
    )
    aperture_mass = 10.0 ** np.asarray(parameters, float)[0]
    mass = aperture_mass * raw_mass / float(components["aperture_mass_shape"])
    return np.log10(mass.reshape(radius_values.shape))


def surface_density(
    expression: ExpressionSpec,
    parameters: np.ndarray,
    radius: np.ndarray | float,
    grid_size: int = DEFAULT_GRID_SIZE,
) -> np.ndarray:
    """Evaluate projected density normalized by the aperture mass."""
    radius_values = np.asarray(radius, float)
    components = _components(expression, parameters, radius_values, grid_size)
    requested_coordinate = np.log1p(
        radius_values.reshape(-1) / float(components["core_radius"])
    )
    integrated_rate = np.interp(
        requested_coordinate,
        np.asarray(components["coordinate"]),
        np.asarray(components["integrated_rate"]),
    )
    scale = 10.0 ** np.asarray(parameters, float)[0] / float(
        components["aperture_mass_shape"]
    )
    return (scale * np.exp(-integrated_rate)).reshape(radius_values.shape)


def density_log_slope(
    expression: ExpressionSpec,
    parameters: np.ndarray,
    radius: np.ndarray | float,
) -> np.ndarray:
    """Return d ln(Sigma)/d ln(R), including its zero central limit."""
    radius_values = np.asarray(radius, float)
    if not np.all(np.isfinite(radius_values)) or np.any(radius_values < 0.0):
        raise ValueError("radius must be finite and non-negative")
    core_radius = 10.0 ** np.asarray(parameters, float)[1]
    coordinate = np.log1p(radius_values / core_radius)
    rate = logarithmic_density_rate(expression, parameters, coordinate)
    return -rate * (-np.expm1(-coordinate))


def shell_log_density(log_cog: np.ndarray, radius: np.ndarray) -> np.ndarray:
    """Log10 mean density in the central aperture and subsequent annuli."""
    log_values = np.asarray(log_cog, float)
    radius_values = np.asarray(radius, float)
    linear = 10.0**log_values
    shell_mass = np.concatenate(([linear[0]], np.diff(linear)))
    if np.any(shell_mass < -1.0e-8 * linear[-1]):
        raise ValueError("candidate CoG produced a negative shell")
    shell_mass = np.maximum(shell_mass, np.finfo(float).tiny)
    inner_squared = np.concatenate(([0.0], radius_values[:-1] ** 2))
    area = np.pi * (radius_values**2 - inner_squared)
    return np.log10(shell_mass / area)


def analytic_property_record(spec: ExpressionSpec) -> dict[str, object]:
    """Return the analytic/numerical classification for one expression."""
    elementary = spec.integration_class in {
        "elementary_squared_constant",
        "elementary_squared_harmonic",
    }
    return {
        "expression": spec.name,
        "finite_positive_center": True,
        "zero_central_derivative": True,
        "central_expansion": "Sigma(R)=Sigma(0)[1-C R^2+O(R^3)], C>0",
        "analytic_log_density_slope": True,
        "global_log_slope_bound": "d ln Sigma/d ln R <= 0",
        "outer_slope_lower_bound": "strictly steeper than R^-2.02",
        "finite_total_mass": True,
        "nonnegative_abel_deprojection": True,
        "integrated_density_rate": "elementary"
        if elementary
        else "numerical_quadrature",
        "projected_density": "elementary" if elementary else "numerical_quadrature",
        "cumulative_mass": "numerical_positive_quadrature",
        "enclosed_radii": "numerical_root",
        "radial_moments": "numerical_quadrature",
        "abel_deprojection": "numerical_abel_integral",
        "fourier_transform": "numerical_hankel_transform",
        "psf_convolution": "numerical_hankel_or_direct",
    }


def observed_half_aperture_radius(truth_log: np.ndarray, radius: np.ndarray) -> float:
    """Interpolate the radius enclosing half the measured aperture mass."""
    target = float(truth_log[-1] + np.log10(0.5))
    if target <= truth_log[0]:
        return float(radius[0] * 10.0 ** (0.5 * (target - truth_log[0])))
    return float(10.0 ** np.interp(target, truth_log, np.log10(radius)))


def synthetic_parameters(
    expression: ExpressionSpec, displacement: float = 0.0
) -> np.ndarray:
    """Return a nontrivial interior point for mechanics and recovery."""
    spec = expression
    lookup = {
        "log_mstar_148": 11.5,
        "log_core_radius": np.log10(12.0),
        "log_a": np.log(1.6),
        "signed_contrast": 1.3,
    }
    for index in range(1, len(spec.atoms) + 1):
        lookup[f"log_location_{index}"] = np.log(1.4 + 0.3 * index)
        lookup[f"log_scale_{index}"] = np.log(1.2 + 0.2 * index)
        lookup[f"log_frequency_{index}"] = np.log(1.4 + 0.4 * index)
        lookup[f"phase_{index}"] = 0.35 + 0.2 * index
        lookup[f"log_damping_{index}"] = np.log(0.5 + 0.15 * index)
    values = np.asarray([lookup[name] for name in spec.parameter_names], float)
    if displacement:
        lower = np.asarray(spec.lower)
        upper = np.asarray(spec.upper)
        direction = np.linspace(-1.0, 1.0, spec.n_parameter)
        values += displacement * 0.08 * (upper - lower) * direction
        values = np.clip(values, lower + 1.0e-5, upper - 1.0e-5)
    return values


def initial_parameters(
    spec: ExpressionSpec,
    truth_log: np.ndarray,
    radius: np.ndarray,
    n_starts: int,
    synthetic_truth_parameters: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Return deterministic separated starts."""
    lower = np.asarray(spec.lower)
    upper = np.asarray(spec.upper)
    if synthetic_truth_parameters is not None:
        truth = np.asarray(synthetic_truth_parameters, float)
        starts = []
        for fraction in (0.025, -0.05, 0.08, -0.12)[:n_starts]:
            direction = np.linspace(-1.0, 1.0, spec.n_parameter)
            start = truth + fraction * (upper - lower) * direction
            starts.append(np.clip(start, lower + 1.0e-6, upper - 1.0e-6))
        return starts
    half_radius = observed_half_aperture_radius(truth_log, radius)
    starts = []
    for radius_factor, a_value, contrast, phase_shift in (
        (0.6, 0.7, -2.0, -0.7),
        (1.0, 1.5, 2.0, 0.0),
        (1.7, 3.0, -5.0, 0.9),
        (0.35, 0.25, 5.0, 2.1),
    )[:n_starts]:
        defaults = synthetic_parameters(spec)
        lookup = dict(zip(spec.parameter_names, defaults, strict=True))
        lookup["log_mstar_148"] = float(truth_log[-1])
        lookup["log_core_radius"] = np.log10(
            np.clip(radius_factor * half_radius, 0.3, 300.0)
        )
        lookup["log_a"] = np.log(a_value)
        if spec.atoms:
            lookup["signed_contrast"] = contrast
        for name in spec.parameter_names:
            if name.startswith("phase_"):
                lookup[name] = phase_shift
        start = np.asarray([lookup[name] for name in spec.parameter_names])
        starts.append(np.clip(start, lower + 1.0e-6, upper - 1.0e-6))
    return starts


def _objective_residual(
    parameters: np.ndarray,
    spec: ExpressionSpec,
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


def fit_profile(
    expression: ExpressionSpec,
    radius: np.ndarray,
    truth_log: np.ndarray,
    *,
    objective: str = "log_cog",
    grid_size: int = DEFAULT_GRID_SIZE,
    n_starts: int = 2,
    max_nfev: int = 400,
    synthetic_truth_parameters: np.ndarray | None = None,
) -> dict[str, object]:
    """Fit one expression and return conditioning and multi-start diagnostics."""
    spec = expression
    radius_values = np.asarray(radius, float)
    truth_values = np.asarray(truth_log, float)
    lower = np.asarray(spec.lower)
    upper = np.asarray(spec.upper)
    solutions = []
    for start in initial_parameters(
        spec, truth_values, radius_values, n_starts, synthetic_truth_parameters
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
        solutions.append((float(np.mean(residual**2)), solution, prediction))
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
        "objective_rms": float(np.sqrt(score)),
        "cog_rms_dex": float(np.sqrt(np.mean((prediction - truth_values) ** 2))),
        "success": bool(best.success and np.all(np.isfinite(prediction))),
        "nfev": int(sum(item[1].nfev for item in solutions)),
        "jacobian_min_ratio": float(singular_values[-1] / singular_values[0])
        if singular_values[0] > 0.0
        else 0.0,
        "boundary_distance": float(np.min(np.minimum(position, 1.0 - position))),
        "near_solution_count": len(near),
        "near_parameter_spread": float(
            np.max(np.ptp(near_parameters, axis=0) / parameter_range)
        )
        if len(near) > 1
        else 0.0,
        "near_profile_spread_dex": float(
            np.max(np.sqrt(np.mean((near_predictions - prediction) ** 2, axis=1)))
        )
        if len(near) > 1
        else 0.0,
    }
