"""Hard-constrained projected densities built from a finite symbolic grammar."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import least_squares
from scipy.special import expit

from hongshao.tng_data import COG_RAD_KPC

RADII_KPC = np.asarray(COG_RAD_KPC, float)
APERTURE_RADIUS_KPC = float(RADII_KPC[-1])
TAIL_SLOPE_MARGIN = 0.02
DEFAULT_GRID_SIZE = 512

LOG_MASS_BOUNDS = (8.0, 14.0)
LOG_RADIUS_BOUNDS = (np.log10(0.3), np.log10(300.0))
BASELINE_BOUNDS = (-10.0, 10.0)
AMPLITUDE_BOUNDS = (-15.0, 15.0)
LOG_TRANSITION_BOUNDS = (np.log(0.05), np.log(12.0))


@dataclass(frozen=True)
class ExpressionSpec:
    """One expression tree and its optimizer coordinates."""

    name: str
    label: str
    atom_kinds: tuple[str, ...]
    atom_settings: tuple[tuple[float, ...], ...]
    parameter_names: tuple[str, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]

    @property
    def n_parameter(self) -> int:
        return len(self.parameter_names)

    @property
    def has_trigonometric_atom(self) -> bool:
        return "harmonic" in self.atom_kinds


def _format_number(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _make_spec(
    name: str,
    label: str,
    atoms: tuple[tuple[str, tuple[float, ...]], ...],
) -> ExpressionSpec:
    names = ["log_mstar_148", "log_core_radius", "baseline"]
    lower = [LOG_MASS_BOUNDS[0], LOG_RADIUS_BOUNDS[0], BASELINE_BOUNDS[0]]
    upper = [LOG_MASS_BOUNDS[1], LOG_RADIUS_BOUNDS[1], BASELINE_BOUNDS[1]]
    for index, (kind, _) in enumerate(atoms, start=1):
        names.append(f"amplitude_{index}")
        lower.append(AMPLITUDE_BOUNDS[0])
        upper.append(AMPLITUDE_BOUNDS[1])
        if kind in {"sigmoid", "rational"}:
            names.append(f"log_location_{index}")
            lower.append(LOG_TRANSITION_BOUNDS[0])
            upper.append(LOG_TRANSITION_BOUNDS[1])
    if len(names) > 6:
        raise ValueError(f"{name} exceeds the six-coordinate cap")
    return ExpressionSpec(
        name=name,
        label=label,
        atom_kinds=tuple(item[0] for item in atoms),
        atom_settings=tuple(item[1] for item in atoms),
        parameter_names=tuple(names),
        lower=tuple(lower),
        upper=tuple(upper),
    )


def round_a_specs() -> dict[str, ExpressionSpec]:
    """Return the frozen one-atom expression grid."""
    specs = {"constant": _make_spec("constant", "Constant slope warp", ())}
    for width in (0.5, 1.0, 2.0):
        label = _format_number(width)
        name = f"sigmoid_w{label}"
        specs[name] = _make_spec(
            name,
            f"Sigmoid transition, width {width:g}",
            (("sigmoid", (width,)),),
        )
    specs["rational"] = _make_spec(
        "rational", "Rational transition", (("rational", ()),)
    )
    for damping in (0.5, 1.0):
        for frequency in (1.0, 2.0):
            for phase_name, phase in (("sine", 0.0), ("cosine", np.pi / 2.0)):
                damping_label = _format_number(damping)
                frequency_label = _format_number(frequency)
                name = f"harmonic_{phase_name}_l{damping_label}_w{frequency_label}"
                specs[name] = _make_spec(
                    name,
                    (
                        f"Damped {phase_name}, damping {damping:g}, "
                        f"frequency {frequency:g}"
                    ),
                    (("harmonic", (damping, frequency, phase)),),
                )
    return specs


def build_round_b_specs(
    non_trigonometric: ExpressionSpec,
    first_trigonometric: ExpressionSpec,
    second_trigonometric: ExpressionSpec,
) -> dict[str, ExpressionSpec]:
    """Mechanically combine discovery-selected bounded atoms."""
    output: dict[str, ExpressionSpec] = {}
    if non_trigonometric.atom_kinds and first_trigonometric.atom_kinds:
        atoms = (
            (non_trigonometric.atom_kinds[0], non_trigonometric.atom_settings[0]),
            (
                first_trigonometric.atom_kinds[0],
                first_trigonometric.atom_settings[0],
            ),
        )
        name = f"combined_{non_trigonometric.name}_{first_trigonometric.name}"
        output[name] = _make_spec(
            name,
            f"{non_trigonometric.label} plus {first_trigonometric.label}",
            atoms,
        )
    if (
        first_trigonometric.atom_kinds
        and second_trigonometric.atom_kinds
        and first_trigonometric.name != second_trigonometric.name
    ):
        atoms = (
            (
                first_trigonometric.atom_kinds[0],
                first_trigonometric.atom_settings[0],
            ),
            (
                second_trigonometric.atom_kinds[0],
                second_trigonometric.atom_settings[0],
            ),
        )
        name = f"combined_{first_trigonometric.name}_{second_trigonometric.name}"
        output[name] = _make_spec(
            name,
            f"{first_trigonometric.label} plus {second_trigonometric.label}",
            atoms,
        )
    return output


def _all_known_specs() -> dict[str, ExpressionSpec]:
    return round_a_specs()


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


def expression_value(
    spec: ExpressionSpec, parameters: np.ndarray, coordinate: np.ndarray
) -> np.ndarray:
    """Evaluate the bounded symbolic expression g(u)."""
    values = _validate_parameters(spec, parameters)
    coordinate = np.asarray(coordinate, float)
    output = np.full_like(coordinate, values[2])
    parameter_index = 3
    for kind, settings in zip(spec.atom_kinds, spec.atom_settings, strict=True):
        amplitude = values[parameter_index]
        parameter_index += 1
        if kind == "sigmoid":
            location = np.exp(values[parameter_index])
            parameter_index += 1
            atom = expit((coordinate - location) / settings[0])
        elif kind == "rational":
            scale = np.exp(values[parameter_index])
            parameter_index += 1
            atom = coordinate / (coordinate + scale)
        elif kind == "harmonic":
            damping, frequency, phase = settings
            atom = np.exp(-damping * coordinate) * np.sin(
                frequency * coordinate + phase
            )
        else:
            raise ValueError(f"unknown atom kind {kind!r}")
        output += amplitude * atom
    return output


def logarithmic_density_rate(
    spec: ExpressionSpec, parameters: np.ndarray, coordinate: np.ndarray
) -> np.ndarray:
    """Return q(u)=dQ/du in the hard physical wrapper."""
    coordinate = np.asarray(coordinate, float)
    warp = expression_value(spec, parameters, coordinate)
    softplus = np.logaddexp(0.0, warp)
    return -np.expm1(-coordinate) * (2.0 + TAIL_SLOPE_MARGIN + softplus)


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
    rate = logarithmic_density_rate(spec, values, coordinate)
    log_density_shape = -cumulative_trapezoid(rate, coordinate, initial=0.0)
    density_shape = np.exp(log_density_shape)
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
        "log_density_shape": log_density_shape,
        "cumulative_mass_shape": cumulative_mass_shape,
        "aperture_mass_shape": aperture_mass_shape,
    }


def cog_log(
    expression: str | ExpressionSpec,
    parameters: np.ndarray,
    radius: np.ndarray | float,
    grid_size: int = DEFAULT_GRID_SIZE,
) -> np.ndarray:
    """Evaluate aperture-normalized log10 cumulative projected mass."""
    spec = _resolve_spec(expression)
    radius_values = np.asarray(radius, float)
    if np.any(radius_values <= 0.0):
        raise ValueError("logarithmic CoG needs strictly positive radii")
    components = _components(spec, parameters, radius_values, grid_size)
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
    expression: str | ExpressionSpec,
    parameters: np.ndarray,
    radius: np.ndarray | float,
    grid_size: int = DEFAULT_GRID_SIZE,
) -> np.ndarray:
    """Evaluate projected density normalized by the aperture mass coordinate."""
    spec = _resolve_spec(expression)
    radius_values = np.asarray(radius, float)
    components = _components(spec, parameters, radius_values, grid_size)
    requested_coordinate = np.log1p(
        radius_values.reshape(-1) / float(components["core_radius"])
    )
    log_shape = np.interp(
        requested_coordinate,
        np.asarray(components["coordinate"]),
        np.asarray(components["log_density_shape"]),
    )
    scale = 10.0 ** np.asarray(parameters, float)[0] / float(
        components["aperture_mass_shape"]
    )
    return (scale * np.exp(log_shape)).reshape(radius_values.shape)


def density_log_slope(
    expression: str | ExpressionSpec,
    parameters: np.ndarray,
    radius: np.ndarray | float,
) -> np.ndarray:
    """Return d ln(Sigma)/d ln(R), including its zero central limit."""
    spec = _resolve_spec(expression)
    radius_values = np.asarray(radius, float)
    if not np.all(np.isfinite(radius_values)) or np.any(radius_values < 0.0):
        raise ValueError("radius must be finite and non-negative")
    core_radius = 10.0 ** np.asarray(parameters, float)[1]
    coordinate = np.log1p(radius_values / core_radius)
    rate = logarithmic_density_rate(spec, parameters, coordinate)
    return -rate * (-np.expm1(-coordinate))


def shell_log_density(log_cog: np.ndarray, radius: np.ndarray) -> np.ndarray:
    """Log10 mean surface density in the central aperture and later annuli."""
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


def observed_half_aperture_radius(truth_log: np.ndarray, radius: np.ndarray) -> float:
    """Interpolate the radius enclosing half the measured aperture mass."""
    target = float(truth_log[-1] + np.log10(0.5))
    if target <= truth_log[0]:
        return float(radius[0] * 10.0 ** (0.5 * (target - truth_log[0])))
    return float(10.0 ** np.interp(target, truth_log, np.log10(radius)))


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
    starts: list[np.ndarray] = []
    if synthetic_truth_parameters is not None:
        truth = np.asarray(synthetic_truth_parameters, float)
        fractions = (0.08, -0.12, 0.20, -0.25)
        for fraction in fractions[:n_starts]:
            direction = np.linspace(-1.0, 1.0, spec.n_parameter)
            displaced = truth + fraction * (upper - lower) * direction
            starts.append(np.clip(displaced, lower + 1.0e-6, upper - 1.0e-6))
        return starts
    half_radius = observed_half_aperture_radius(truth_log, radius)
    configurations = (
        (0.7, -2.0, 0.0, 0.7),
        (1.0, 0.0, 2.0, 1.5),
        (1.8, 2.0, -3.0, 3.0),
        (0.4, 5.0, 5.0, 0.2),
    )
    for radius_factor, baseline, amplitude, location in configurations[:n_starts]:
        values = [
            float(truth_log[-1]),
            np.log10(np.clip(radius_factor * half_radius, 0.3, 300.0)),
            baseline,
        ]
        for kind in spec.atom_kinds:
            values.append(amplitude)
            if kind in {"sigmoid", "rational"}:
                values.append(np.log(location))
        starts.append(np.clip(np.asarray(values), lower + 1.0e-6, upper - 1.0e-6))
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
    expression: str | ExpressionSpec,
    radius: np.ndarray,
    truth_log: np.ndarray,
    *,
    objective: str = "log_cog",
    grid_size: int = DEFAULT_GRID_SIZE,
    n_starts: int = 4,
    max_nfev: int = 500,
    synthetic_truth_parameters: np.ndarray | None = None,
) -> dict[str, object]:
    """Fit one expression and report local and multi-start diagnostics."""
    spec = _resolve_spec(expression)
    radius_values = np.asarray(radius, float)
    truth_values = np.asarray(truth_log, float)
    lower = np.asarray(spec.lower)
    upper = np.asarray(spec.upper)
    solutions = []
    for start in initial_parameters(
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
        objective_residual = _objective_residual(
            solution.x,
            spec,
            radius_values,
            truth_values,
            objective,
            grid_size,
        )
        score = float(np.mean(objective_residual**2))
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


def synthetic_parameters(expression: str | ExpressionSpec) -> np.ndarray:
    """A nontrivial interior point for mechanics and recovery checks."""
    spec = _resolve_spec(expression)
    values = [11.5, np.log10(12.0), -0.5]
    for index, kind in enumerate(spec.atom_kinds):
        values.append(1.2 if index == 0 else -0.8)
        if kind in {"sigmoid", "rational"}:
            values.append(np.log(1.2 + index))
    return np.asarray(values)


def _resolve_spec(expression: str | ExpressionSpec) -> ExpressionSpec:
    if isinstance(expression, ExpressionSpec):
        return expression
    specs = _all_known_specs()
    try:
        return specs[expression]
    except KeyError as error:
        raise ValueError(f"unknown expression {expression!r}") from error
