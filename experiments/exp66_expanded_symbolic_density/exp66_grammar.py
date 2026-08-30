"""Expanded hard-constrained projected-density expression grammar for Exp66."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import least_squares

from hongshao.tng_data import COG_RAD_KPC

RADII_KPC = np.asarray(COG_RAD_KPC, float)
APERTURE_RADIUS_KPC = float(RADII_KPC[-1])
TAIL_SLOPE_MARGIN = 0.02
DEFAULT_GRID_SIZE = 512

LOG_MASS_BOUNDS = (8.0, 14.0)
LOG_RADIUS_BOUNDS = (np.log10(0.3), np.log10(300.0))
LOG_DELTA_BOUNDS = (np.log(0.05), np.log(20.0))
MODULATION_BOUNDS = (-3.0, 3.0)
LOG_LOCATION_BOUNDS = (np.log(0.05), np.log(12.0))
LOG_FREQUENCY_BOUNDS = (np.log(0.25), np.log(8.0))
PHASE_BOUNDS = (-np.pi, np.pi)
LOG_DAMPING_BOUNDS = (np.log(0.05), np.log(3.0))


@dataclass(frozen=True)
class AtomSpec:
    """One bounded atom and its fixed or fitted constants."""

    kind: str
    width: float | None = None
    frequency: float | None = None
    phase: float = 0.0
    damping: float | None = None
    chirp: float | None = None
    scale: float | None = None
    free_location: bool = False
    free_frequency: bool = False
    free_phase: bool = False
    free_damping: bool = False
    free_scale: bool = False


@dataclass(frozen=True)
class ExpressionSpec:
    """One bounded expression tree and its optimizer coordinates."""

    name: str
    label: str
    atoms: tuple[AtomSpec, ...]
    operation: str
    parameter_names: tuple[str, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    integration_class: str

    @property
    def n_parameter(self) -> int:
        return len(self.parameter_names)

    @property
    def atom_kinds(self) -> tuple[str, ...]:
        return tuple(atom.kind for atom in self.atoms)

    @property
    def has_trigonometric_atom(self) -> bool:
        return any(
            atom.kind
            in {
                "harmonic",
                "damped_harmonic",
                "wave_packet",
                "chirp",
                "log_harmonic",
            }
            for atom in self.atoms
        )

    @property
    def free_frequency(self) -> bool:
        return any(atom.free_frequency for atom in self.atoms)

    @property
    def free_phase(self) -> bool:
        return any(atom.free_phase for atom in self.atoms)

    @property
    def free_damping(self) -> bool:
        return any(atom.free_damping for atom in self.atoms)

    @property
    def tree_size(self) -> int:
        return len(self.atoms) + (0 if self.operation == "atom" else 1)


def _format_number(value: float) -> str:
    prefix = "n" if value < 0.0 else ""
    return prefix + f"{abs(value):g}".replace(".", "p")


def _parameter_layout(
    atoms: tuple[AtomSpec, ...], include_modulation: bool
) -> tuple[tuple[str, ...], tuple[float, ...], tuple[float, ...]]:
    names = ["log_mstar_148", "log_core_radius", "log_delta"]
    lower = [LOG_MASS_BOUNDS[0], LOG_RADIUS_BOUNDS[0], LOG_DELTA_BOUNDS[0]]
    upper = [LOG_MASS_BOUNDS[1], LOG_RADIUS_BOUNDS[1], LOG_DELTA_BOUNDS[1]]
    if include_modulation:
        names.append("modulation_coordinate")
        lower.append(MODULATION_BOUNDS[0])
        upper.append(MODULATION_BOUNDS[1])
    for index, atom in enumerate(atoms, start=1):
        if atom.free_location:
            names.append(f"log_location_{index}")
            lower.append(LOG_LOCATION_BOUNDS[0])
            upper.append(LOG_LOCATION_BOUNDS[1])
        if atom.free_scale:
            names.append(f"log_scale_{index}")
            lower.append(LOG_LOCATION_BOUNDS[0])
            upper.append(LOG_LOCATION_BOUNDS[1])
        if atom.free_frequency:
            names.append(f"log_frequency_{index}")
            lower.append(LOG_FREQUENCY_BOUNDS[0])
            upper.append(LOG_FREQUENCY_BOUNDS[1])
        if atom.free_phase:
            names.append(f"phase_{index}")
            lower.append(PHASE_BOUNDS[0])
            upper.append(PHASE_BOUNDS[1])
        if atom.free_damping:
            names.append(f"log_damping_{index}")
            lower.append(LOG_DAMPING_BOUNDS[0])
            upper.append(LOG_DAMPING_BOUNDS[1])
    return tuple(names), tuple(lower), tuple(upper)


def _make_spec(
    name: str,
    label: str,
    atoms: tuple[AtomSpec, ...],
    operation: str = "atom",
) -> ExpressionSpec:
    if operation not in {"atom", "mean", "product", "cube", "centered_square"}:
        raise ValueError(f"unknown expression operation {operation!r}")
    include_modulation = bool(atoms)
    names, lower, upper = _parameter_layout(atoms, include_modulation)
    if len(names) > (6 if operation == "atom" else 7):
        raise ValueError(f"{name} exceeds its coordinate cap")
    elementary = (
        operation == "atom"
        and len(atoms) == 1
        and atoms[0].kind in {"harmonic", "damped_harmonic"}
    )
    integration_class = "elementary_harmonic" if elementary else "numerical"
    if not atoms:
        integration_class = "elementary_constant"
    return ExpressionSpec(
        name=name,
        label=label,
        atoms=atoms,
        operation=operation,
        parameter_names=names,
        lower=lower,
        upper=upper,
        integration_class=integration_class,
    )


def round_a_specs() -> dict[str, ExpressionSpec]:
    """Return the frozen expanded one-atom grammar in deterministic order."""
    specs: dict[str, ExpressionSpec] = {
        "constant": _make_spec("constant", "Unmodulated monotone density", ())
    }
    for width in (0.35, 0.7, 1.4):
        width_label = _format_number(width)
        for kind, label in (
            ("sigmoid", "Sigmoid transition"),
            ("arctangent", "Arctangent transition"),
            ("shoulder", "Localized shoulder"),
        ):
            name = f"{kind}_w{width_label}"
            specs[name] = _make_spec(
                name,
                f"{label}, width {width:g}",
                (AtomSpec(kind, width=width, free_location=True),),
            )
    specs["rational"] = _make_spec(
        "rational",
        "Rational transition",
        (AtomSpec("rational", free_scale=True),),
    )
    phase_grid = (("sine", 0.0), ("cosine", np.pi / 2.0))
    for frequency in (0.5, 1.0, 2.0, 4.0):
        frequency_label = _format_number(frequency)
        for phase_name, phase in phase_grid:
            name = f"harmonic_{phase_name}_w{frequency_label}"
            specs[name] = _make_spec(
                name,
                f"{phase_name.title()} harmonic, frequency {frequency:g}",
                (AtomSpec("harmonic", frequency=frequency, phase=phase),),
            )
            for damping in (0.25, 0.75, 1.5):
                damping_label = _format_number(damping)
                name = f"damped_{phase_name}_l{damping_label}_w{frequency_label}"
                specs[name] = _make_spec(
                    name,
                    (
                        f"Damped {phase_name}, damping {damping:g}, "
                        f"frequency {frequency:g}"
                    ),
                    (
                        AtomSpec(
                            "damped_harmonic",
                            frequency=frequency,
                            phase=phase,
                            damping=damping,
                        ),
                    ),
                )
            for width in (0.5, 1.0):
                width_label = _format_number(width)
                name = f"wave_packet_{phase_name}_w{width_label}_w{frequency_label}"
                specs[name] = _make_spec(
                    name,
                    (
                        f"{phase_name.title()} wave packet, width {width:g}, "
                        f"frequency {frequency:g}"
                    ),
                    (
                        AtomSpec(
                            "wave_packet",
                            width=width,
                            frequency=frequency,
                            phase=phase,
                            free_location=True,
                        ),
                    ),
                )
            for chirp in (-0.25, 0.25):
                chirp_label = _format_number(chirp)
                name = f"chirp_{phase_name}_k{chirp_label}_w{frequency_label}"
                specs[name] = _make_spec(
                    name,
                    (
                        f"{phase_name.title()} chirp, curvature {chirp:g}, "
                        f"frequency {frequency:g}"
                    ),
                    (
                        AtomSpec(
                            "chirp",
                            frequency=frequency,
                            phase=phase,
                            chirp=chirp,
                        ),
                    ),
                )
            name = f"log_harmonic_{phase_name}_w{frequency_label}"
            specs[name] = _make_spec(
                name,
                f"Log-{phase_name} harmonic, frequency {frequency:g}",
                (
                    AtomSpec(
                        "log_harmonic",
                        frequency=frequency,
                        phase=phase,
                        free_scale=True,
                    ),
                ),
            )
    for phase_name, phase in phase_grid:
        name = f"free_frequency_{phase_name}"
        specs[name] = _make_spec(
            name,
            f"Fitted-frequency {phase_name}",
            (AtomSpec("harmonic", phase=phase, free_frequency=True),),
        )
        name = f"free_damped_{phase_name}"
        specs[name] = _make_spec(
            name,
            f"Fitted-frequency and damping {phase_name}",
            (
                AtomSpec(
                    "damped_harmonic",
                    phase=phase,
                    free_frequency=True,
                    free_damping=True,
                ),
            ),
        )
    specs["free_frequency_phase"] = _make_spec(
        "free_frequency_phase",
        "Fitted-frequency and phase harmonic",
        (AtomSpec("harmonic", free_frequency=True, free_phase=True),),
    )
    return specs


def _composition(
    first: ExpressionSpec,
    second: ExpressionSpec | None,
    operation: str,
) -> ExpressionSpec | None:
    atoms = first.atoms + (() if second is None else second.atoms)
    names = [first.name] if second is None else [first.name, second.name]
    name = f"{operation}__{'__'.join(names)}"
    label = f"{operation.replace('_', ' ').title()}: " + " + ".join(
        item.label for item in (first,) if item is not None
    )
    if second is not None:
        label += " and " + second.label
    try:
        return _make_spec(name, label, atoms, operation)
    except ValueError:
        return None


def build_round_b_specs(
    transitions: list[ExpressionSpec], trigonometric: list[ExpressionSpec]
) -> dict[str, ExpressionSpec]:
    """Generate the frozen bounded compositions from ranked Round-A inputs."""
    output: dict[str, ExpressionSpec] = {}

    def add(spec: ExpressionSpec | None) -> None:
        if spec is not None:
            output.setdefault(spec.name, spec)

    for transition in transitions[:2]:
        for trig in trigonometric[:4]:
            add(_composition(transition, trig, "mean"))
    for first, second in list(combinations(trigonometric[:4], 2))[:4]:
        add(_composition(first, second, "mean"))
    for transition in transitions[:2]:
        for trig in trigonometric[:2]:
            add(_composition(transition, trig, "product"))
    for selected in (transitions[:2], trigonometric[:2]):
        for spec in selected:
            add(_composition(spec, None, "cube"))
            add(_composition(spec, None, "centered_square"))
    return output


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


def _atom_values(
    atom: AtomSpec,
    atom_index: int,
    parameters: np.ndarray,
    spec: ExpressionSpec,
    coordinate: np.ndarray,
) -> np.ndarray:
    lookup = dict(zip(spec.parameter_names, parameters, strict=True))
    location = (
        np.exp(lookup[f"log_location_{atom_index}"]) if atom.free_location else 1.0
    )
    scale = (
        np.exp(lookup[f"log_scale_{atom_index}"])
        if atom.free_scale
        else float(atom.scale or 1.0)
    )
    frequency = (
        np.exp(lookup[f"log_frequency_{atom_index}"])
        if atom.free_frequency
        else float(atom.frequency or 1.0)
    )
    phase = lookup[f"phase_{atom_index}"] if atom.free_phase else atom.phase
    damping = (
        np.exp(lookup[f"log_damping_{atom_index}"])
        if atom.free_damping
        else float(atom.damping or 0.0)
    )
    if atom.kind == "sigmoid":
        return np.tanh((coordinate - location) / (2.0 * float(atom.width)))
    if atom.kind == "rational":
        return (coordinate - scale) / (coordinate + scale)
    if atom.kind == "arctangent":
        return (2.0 / np.pi) * np.arctan((coordinate - location) / float(atom.width))
    if atom.kind == "shoulder":
        argument = np.clip((coordinate - location) / float(atom.width), -700.0, 700.0)
        return 2.0 / np.cosh(argument) - 1.0
    if atom.kind == "harmonic":
        return np.sin(frequency * coordinate + phase)
    if atom.kind == "damped_harmonic":
        return np.exp(-damping * coordinate) * np.sin(frequency * coordinate + phase)
    if atom.kind == "wave_packet":
        shifted = coordinate - location
        argument = np.clip(shifted / float(atom.width), -700.0, 700.0)
        return np.sin(frequency * shifted + phase) / np.cosh(argument)
    if atom.kind == "chirp":
        return np.sin(
            frequency * coordinate + float(atom.chirp) * coordinate**2 + phase
        )
    if atom.kind == "log_harmonic":
        return np.sin(frequency * np.log1p(coordinate / scale) + phase)
    raise ValueError(f"unknown atom kind {atom.kind!r}")


def normalized_expression_value(
    expression: ExpressionSpec,
    parameters: np.ndarray,
    coordinate: np.ndarray,
) -> np.ndarray:
    """Evaluate T(u), whose absolute value is bounded by one algebraically."""
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
    """Return q(u)=dQ/du for the hard physical wrapper."""
    spec = expression
    values = _validate_parameters(spec, parameters)
    coordinate_values = np.asarray(coordinate, float)
    delta = np.exp(values[2])
    modulation = np.tanh(values[3]) if spec.atoms else 0.0
    transition = normalized_expression_value(spec, values, coordinate_values)
    bracket = 2.0 + TAIL_SLOPE_MARGIN + delta * (1.0 + modulation * transition)
    return -np.expm1(-coordinate_values) * bracket


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


def integrated_density_rate(
    expression: ExpressionSpec,
    parameters: np.ndarray,
    coordinate: np.ndarray,
) -> np.ndarray:
    """Return Q(u), using the exact route for every single harmonic."""
    spec = expression
    values = _validate_parameters(spec, parameters)
    coordinate_values = np.asarray(coordinate, float)
    delta = np.exp(values[2])
    base = (2.0 + TAIL_SLOPE_MARGIN + delta) * (
        coordinate_values + np.exp(-coordinate_values) - 1.0
    )
    if spec.integration_class == "elementary_constant":
        return base
    if spec.integration_class == "elementary_harmonic":
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
        amplitude = delta * np.tanh(values[3])
        modulation_integral = _exponential_sine_integral(
            damping, frequency, phase, coordinate_values
        ) - _exponential_sine_integral(
            damping + 1.0, frequency, phase, coordinate_values
        )
        return base + amplitude * modulation_integral
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
    """Evaluate aperture-normalized log10 cumulative projected mass."""
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
    """Evaluate projected density normalized by the aperture-mass coordinate."""
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
    """Return the predeclared analytic/numerical classification for a family."""
    elementary = spec.integration_class in {
        "elementary_constant",
        "elementary_harmonic",
    }
    return {
        "expression": spec.name,
        "finite_positive_center": True,
        "zero_central_derivative": True,
        "central_expansion": "Sigma(R)=Sigma(0)[1-C R^2+O(R^3)], C>0",
        "analytic_log_density_slope": True,
        "global_log_slope_bounds": "-(1-e^-u)^2[2.02+delta(1+|rho|)] <= slope <= 0",
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
    """Return a nontrivial interior point for mechanics and recovery checks."""
    spec = expression
    lookup = {
        "log_mstar_148": 11.5,
        "log_core_radius": np.log10(12.0),
        "log_delta": np.log(2.0),
        "modulation_coordinate": 0.9,
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
        values += displacement * 0.12 * (upper - lower) * direction
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
        for fraction in (0.04, -0.08, 0.12, -0.18)[:n_starts]:
            direction = np.linspace(-1.0, 1.0, spec.n_parameter)
            start = truth + fraction * (upper - lower) * direction
            starts.append(np.clip(start, lower + 1.0e-6, upper - 1.0e-6))
        return starts
    half_radius = observed_half_aperture_radius(truth_log, radius)
    starts = []
    for radius_factor, delta, modulation, phase_shift in (
        (0.6, 1.0, -0.8, -0.7),
        (1.0, 2.0, 0.8, 0.0),
        (1.7, 5.0, -1.5, 0.9),
        (0.35, 0.3, 1.5, 2.1),
    )[:n_starts]:
        defaults = synthetic_parameters(spec)
        lookup = dict(zip(spec.parameter_names, defaults, strict=True))
        lookup["log_mstar_148"] = float(truth_log[-1])
        lookup["log_core_radius"] = np.log10(
            np.clip(radius_factor * half_radius, 0.3, 300.0)
        )
        lookup["log_delta"] = np.log(delta)
        if spec.atoms:
            lookup["modulation_coordinate"] = modulation
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
    max_nfev: int = 300,
    synthetic_truth_parameters: np.ndarray | None = None,
) -> dict[str, object]:
    """Fit one expression and report local and multi-start diagnostics."""
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
