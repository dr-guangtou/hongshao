"""Analytic curve-of-growth families and identifiable fitting diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit, gammainc, gammaincinv


@dataclass(frozen=True)
class FamilySpec:
    """Definition of one analytic CoG family in optimizer coordinates."""

    name: str
    parameter_names: tuple[str, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]

    @property
    def n_parameter(self) -> int:
        return len(self.parameter_names)


LOG_MASS_BOUNDS = (8.0, 14.0)
LOG_RADIUS_BOUNDS = (np.log10(0.3), np.log10(300.0))

FAMILY_SPECS = {
    "sersic": FamilySpec(
        "sersic",
        ("log_mstar_148", "log_r50", "log_n"),
        (LOG_MASS_BOUNDS[0], LOG_RADIUS_BOUNDS[0], np.log(0.3)),
        (LOG_MASS_BOUNDS[1], LOG_RADIUS_BOUNDS[1], np.log(10.0)),
    ),
    "moffat": FamilySpec(
        "moffat",
        ("log_mstar_148", "log_r50", "log_gamma_minus_one"),
        (LOG_MASS_BOUNDS[0], LOG_RADIUS_BOUNDS[0], np.log(0.05)),
        (LOG_MASS_BOUNDS[1], LOG_RADIUS_BOUNDS[1], np.log(5.0)),
    ),
    "gompertz_log": FamilySpec(
        "gompertz_log",
        ("log_mstar_148", "log_r50", "log_c"),
        (LOG_MASS_BOUNDS[0], LOG_RADIUS_BOUNDS[0], np.log(0.1)),
        (LOG_MASS_BOUNDS[1], LOG_RADIUS_BOUNDS[1], np.log(8.0)),
    ),
    "richards": FamilySpec(
        "richards",
        ("log_mstar_148", "log_r50", "log_k", "log_nu"),
        (
            LOG_MASS_BOUNDS[0],
            LOG_RADIUS_BOUNDS[0],
            np.log(0.1),
            np.log(0.02),
        ),
        (
            LOG_MASS_BOUNDS[1],
            LOG_RADIUS_BOUNDS[1],
            np.log(8.0),
            np.log(20.0),
        ),
    ),
    "radial_sigmoid": FamilySpec(
        "radial_sigmoid",
        ("log_mstar_148", "beta_out", "log_delta_beta", "log_rc", "log_width"),
        (
            LOG_MASS_BOUNDS[0],
            0.0,
            np.log(0.01),
            LOG_RADIUS_BOUNDS[0],
            np.log(0.05),
        ),
        (
            LOG_MASS_BOUNDS[1],
            3.0,
            np.log(6.0),
            LOG_RADIUS_BOUNDS[1],
            np.log(3.0),
        ),
    ),
    "radial_sigmoid_beta0": FamilySpec(
        "radial_sigmoid_beta0",
        ("log_mstar_148", "log_delta_beta", "log_rc", "log_width"),
        (
            LOG_MASS_BOUNDS[0],
            np.log(0.01),
            LOG_RADIUS_BOUNDS[0],
            np.log(0.05),
        ),
        (
            LOG_MASS_BOUNDS[1],
            np.log(6.0),
            LOG_RADIUS_BOUNDS[1],
            np.log(3.0),
        ),
    ),
    "double_sersic_n2_n1": FamilySpec(
        "double_sersic_n2_n1",
        (
            "log_mstar_148",
            "logit_f_compact_148",
            "log_r_compact",
            "log_ratio_minus_one",
        ),
        (LOG_MASS_BOUNDS[0], -5.0, LOG_RADIUS_BOUNDS[0], np.log(0.03)),
        (LOG_MASS_BOUNDS[1], 5.0, LOG_RADIUS_BOUNDS[1], np.log(100.0)),
    ),
    "double_sersic_n4_n1": FamilySpec(
        "double_sersic_n4_n1",
        (
            "log_mstar_148",
            "logit_f_compact_148",
            "log_r_compact",
            "log_ratio_minus_one",
        ),
        (LOG_MASS_BOUNDS[0], -5.0, LOG_RADIUS_BOUNDS[0], np.log(0.03)),
        (LOG_MASS_BOUNDS[1], 5.0, LOG_RADIUS_BOUNDS[1], np.log(100.0)),
    ),
    "double_sersic_n4_n2": FamilySpec(
        "double_sersic_n4_n2",
        (
            "log_mstar_148",
            "logit_f_compact_148",
            "log_r_compact",
            "log_ratio_minus_one",
        ),
        (LOG_MASS_BOUNDS[0], -5.0, LOG_RADIUS_BOUNDS[0], np.log(0.03)),
        (LOG_MASS_BOUNDS[1], 5.0, LOG_RADIUS_BOUNDS[1], np.log(100.0)),
    ),
}

ONE_COMPONENT_FAMILIES = (
    "sersic",
    "moffat",
    "gompertz_log",
    "richards",
    "radial_sigmoid",
    "radial_sigmoid_beta0",
)
MIXTURE_FAMILIES = (
    "double_sersic_n2_n1",
    "double_sersic_n4_n1",
    "double_sersic_n4_n2",
)
ANALYTIC_FAMILIES = ONE_COMPONENT_FAMILIES + MIXTURE_FAMILIES


def sersic_fraction(radius: np.ndarray, r50: float, n_sersic: float) -> np.ndarray:
    """Infinite-aperture projected Sersic cumulative mass fraction."""
    b_n = gammaincinv(2.0 * n_sersic, 0.5)
    argument = b_n * (np.asarray(radius, float) / r50) ** (1.0 / n_sersic)
    return gammainc(2.0 * n_sersic, argument)


def _component_indices(family: str) -> tuple[float, float]:
    fields = family.removeprefix("double_sersic_n").split("_n")
    if len(fields) != 2:
        raise ValueError(f"cannot parse component indices from {family!r}")
    return float(fields[0]), float(fields[1])


def _radial_sigmoid_log_shape(radius: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    """Closed-form integral of the sigmoid logarithmic CoG slope."""
    beta_out = parameters[1]
    delta_beta = np.exp(parameters[2])
    r_c = 10.0 ** parameters[3]
    width = np.exp(parameters[4])
    log_radius = np.log(np.asarray(radius, float))
    scaled_radius = (log_radius - np.log(r_c)) / width
    sigmoid_integral = log_radius - width * np.logaddexp(0.0, scaled_radius)
    return (beta_out * log_radius + delta_beta * sigmoid_integral) / np.log(10.0)


def _radial_sigmoid_beta0_log_shape(
    radius: np.ndarray, parameters: np.ndarray
) -> np.ndarray:
    """Radial sigmoid with the physically asymptotic outer slope fixed to zero."""
    delta_beta = np.exp(parameters[1])
    r_c = 10.0 ** parameters[2]
    width = np.exp(parameters[3])
    log_radius = np.log(np.asarray(radius, float))
    scaled_radius = (log_radius - np.log(r_c)) / width
    sigmoid_integral = log_radius - width * np.logaddexp(0.0, scaled_radius)
    return delta_beta * sigmoid_integral / np.log(10.0)


def cog_log(family: str, parameters: np.ndarray, radius: np.ndarray) -> np.ndarray:
    """Evaluate log10 M*(<R) with amplitude defined at the last radius."""
    parameters = np.asarray(parameters, float)
    radius = np.asarray(radius, float)
    expected = FAMILY_SPECS[family].n_parameter
    if parameters.shape != (expected,):
        raise ValueError(
            f"{family} needs {expected} parameters, got {parameters.shape}"
        )

    log_mass_reference = parameters[0]
    r50_families = {"sersic", "moffat", "gompertz_log", "richards"}
    r50 = 10.0 ** parameters[1] if family in r50_families else None

    if family == "sersic":
        fraction = sersic_fraction(radius, r50, np.exp(parameters[2]))
    elif family == "moffat":
        gamma = 1.0 + np.exp(parameters[2])
        half_scale = np.sqrt(2.0 ** (1.0 / (gamma - 1.0)) - 1.0)
        r_core = r50 / half_scale
        fraction = 1.0 - (1.0 + (radius / r_core) ** 2) ** (1.0 - gamma)
    elif family == "gompertz_log":
        c_shape = np.exp(parameters[2])
        exponent = np.log(2.0) * (r50 / radius) ** c_shape
        fraction = np.exp(-np.clip(exponent, 0.0, 745.0))
    elif family == "richards":
        k_shape = np.exp(parameters[2])
        nu = np.exp(parameters[3])
        half_factor = np.expm1(nu * np.log(2.0)) / nu
        scale = r50 * half_factor ** (1.0 / k_shape)
        inverse_power = np.exp(
            np.clip(-k_shape * np.log(radius / scale), -700.0, 700.0)
        )
        fraction = np.exp(-np.log1p(nu * inverse_power) / nu)
    elif family == "radial_sigmoid":
        log_shape = _radial_sigmoid_log_shape(radius, parameters)
        return log_mass_reference + log_shape - log_shape[-1]
    elif family == "radial_sigmoid_beta0":
        log_shape = _radial_sigmoid_beta0_log_shape(radius, parameters)
        return log_mass_reference + log_shape - log_shape[-1]
    elif family.startswith("double_sersic"):
        n_compact, n_extended = _component_indices(family)
        compact_fraction_reference = expit(parameters[1])
        r_compact = 10.0 ** parameters[2]
        r_extended = r_compact * (1.0 + np.exp(parameters[3]))
        compact = sersic_fraction(radius, r_compact, n_compact)
        extended = sersic_fraction(radius, r_extended, n_extended)
        compact /= compact[-1]
        extended /= extended[-1]
        fraction = (
            compact_fraction_reference * compact
            + (1.0 - compact_fraction_reference) * extended
        )
    else:
        raise ValueError(f"unknown family {family!r}")

    fraction = np.clip(fraction, 1e-300, None)
    return log_mass_reference + np.log10(fraction / fraction[-1])


def _observed_r50(cog_log_values: np.ndarray, radius: np.ndarray) -> float:
    cumulative = 10.0 ** np.asarray(cog_log_values, float)
    target = 0.5 * cumulative[-1]
    if target <= cumulative[0]:
        slope = (np.log(cumulative[1]) - np.log(cumulative[0])) / (
            np.log(radius[1]) - np.log(radius[0])
        )
        estimate = radius[0] * np.exp(
            (np.log(target) - np.log(cumulative[0])) / max(slope, 1e-3)
        )
        return float(np.clip(estimate, 0.3, radius[-1]))
    return float(np.interp(target, cumulative, radius))


def initial_parameters(
    family: str, cog_log_values: np.ndarray, radius: np.ndarray
) -> list[np.ndarray]:
    """Widely separated, data-scaled starts for deterministic multi-start fits."""
    amplitude = float(cog_log_values[-1])
    r50 = _observed_r50(cog_log_values, radius)
    log_r50 = np.log10(r50)

    if family == "sersic":
        starts = [[amplitude, log_r50, np.log(n)] for n in (0.7, 2.0, 6.0)]
    elif family == "moffat":
        starts = [
            [amplitude, log_r50, np.log(gamma - 1.0)] for gamma in (1.15, 1.7, 4.0)
        ]
    elif family == "gompertz_log":
        starts = [[amplitude, log_r50, np.log(c)] for c in (0.3, 1.2, 4.0)]
    elif family == "richards":
        starts = [
            [amplitude, log_r50, np.log(k), np.log(nu)]
            for k, nu in ((0.4, 0.05), (1.2, 1.0), (3.5, 5.0), (1.0, 15.0))
        ]
    elif family == "radial_sigmoid":
        starts = [
            [amplitude, beta_out, np.log(delta_beta), log_rc, np.log(width)]
            for beta_out, delta_beta, log_rc, width in (
                (0.05, 1.0, np.log10(max(r50, 1.0)), 0.3),
                (0.3, 2.5, np.log10(max(r50 / 2.0, 0.3)), 1.0),
                (1.0, 0.3, np.log10(min(2.0 * r50, 300.0)), 2.0),
            )
        ]
    elif family == "radial_sigmoid_beta0":
        starts = [
            [amplitude, np.log(delta_beta), log_rc, np.log(width)]
            for delta_beta, log_rc, width in (
                (1.0, np.log10(max(r50, 1.0)), 0.3),
                (3.0, np.log10(max(r50 / 2.0, 0.3)), 1.0),
                (0.3, np.log10(min(2.0 * r50, 300.0)), 2.0),
            )
        ]
    elif family.startswith("double_sersic"):
        starts = []
        for compact_fraction, compact_factor, ratio in (
            (0.15, 0.15, 8.0),
            (0.5, 0.35, 3.0),
            (0.85, 0.7, 1.5),
            (0.5, 0.08, 20.0),
        ):
            compact_radius = np.clip(compact_factor * r50, 0.3, 250.0)
            starts.append(
                [
                    amplitude,
                    np.log(compact_fraction / (1.0 - compact_fraction)),
                    np.log10(compact_radius),
                    np.log(ratio - 1.0),
                ]
            )
    else:
        raise ValueError(family)

    spec = FAMILY_SPECS[family]
    lower = np.asarray(spec.lower)
    upper = np.asarray(spec.upper)
    return [
        np.clip(np.asarray(start, float), lower + 1e-8, upper - 1e-8)
        for start in starts
    ]


def residuals(
    parameters: np.ndarray, family: str, radius: np.ndarray, truth_log: np.ndarray
) -> np.ndarray:
    prediction = cog_log(family, parameters, radius)
    if not np.isfinite(prediction).all():
        return np.full(len(radius), 100.0)
    return prediction - truth_log


def _scaled_jacobian_ratio(jacobian: np.ndarray, spec: FamilySpec) -> float:
    parameter_range = np.asarray(spec.upper) - np.asarray(spec.lower)
    scaled = jacobian * parameter_range[None, :]
    singular_values = np.linalg.svd(scaled, compute_uv=False)
    if singular_values[0] <= 0.0:
        return 0.0
    return float(singular_values[-1] / singular_values[0])


def fit_cog(
    family: str,
    radius: np.ndarray,
    truth_log: np.ndarray,
    starts: list[np.ndarray] | None = None,
    max_nfev: int = 2000,
) -> dict:
    """Fit one CoG and return reconstruction plus degeneracy diagnostics."""
    spec = FAMILY_SPECS[family]
    lower = np.asarray(spec.lower)
    upper = np.asarray(spec.upper)
    starts = initial_parameters(family, truth_log, radius) if starts is None else starts
    solutions = []
    for start in starts:
        solution = least_squares(
            residuals,
            np.clip(np.asarray(start, float), lower + 1e-8, upper - 1e-8),
            args=(family, radius, truth_log),
            bounds=(lower, upper),
            method="trf",
            x_scale="jac",
            max_nfev=max_nfev,
        )
        prediction = cog_log(family, solution.x, radius)
        rss = float(np.sum((prediction - truth_log) ** 2))
        solutions.append((rss, solution, prediction))

    solutions.sort(key=lambda value: value[0])
    best_rss, best, prediction = solutions[0]
    near = [entry for entry in solutions if entry[0] <= 1.01 * best_rss + 1e-14]
    near_parameters = np.array([entry[1].x for entry in near])
    near_predictions = np.array([entry[2] for entry in near])
    parameter_range = upper - lower
    parameter_spread = (
        np.max(np.ptp(near_parameters, axis=0) / parameter_range)
        if len(near_parameters) > 1
        else 0.0
    )
    profile_spread = (
        np.max(np.sqrt(np.mean((near_predictions - prediction[None, :]) ** 2, axis=1)))
        if len(near_predictions) > 1
        else 0.0
    )
    fractional_position = (best.x - lower) / parameter_range
    boundary_distance = float(
        np.min(np.minimum(fractional_position, 1.0 - fractional_position))
    )

    return {
        "parameters": best.x,
        "prediction": prediction,
        "rms_dex": float(np.sqrt(best_rss / len(radius))),
        "success": bool(best.success),
        "nfev": int(best.nfev),
        "jacobian_min_ratio": _scaled_jacobian_ratio(best.jac, spec),
        "boundary_distance": boundary_distance,
        "near_solution_count": len(near),
        "near_parameter_spread": float(parameter_spread),
        "near_profile_spread_dex": float(profile_spread),
    }


def profile_scan(
    family: str,
    radius: np.ndarray,
    truth_log: np.ndarray,
    best_parameters: np.ndarray,
    parameter_index: int,
    values: np.ndarray,
) -> np.ndarray:
    """Profile one parameter while re-optimizing every other parameter."""
    spec = FAMILY_SPECS[family]
    lower = np.asarray(spec.lower)
    upper = np.asarray(spec.upper)
    free = np.arange(spec.n_parameter) != parameter_index
    output = np.empty(len(values))
    best_free = np.asarray(best_parameters)[free]
    previous = best_free.copy()

    for index, fixed_value in enumerate(values):

        def profiled_residual(free_parameters: np.ndarray) -> np.ndarray:
            parameters = np.empty(spec.n_parameter)
            parameters[free] = free_parameters
            parameters[parameter_index] = fixed_value
            return residuals(parameters, family, radius, truth_log)

        solutions = [
            least_squares(
                profiled_residual,
                np.clip(start, lower[free] + 1e-8, upper[free] - 1e-8),
                bounds=(lower[free], upper[free]),
                method="trf",
                x_scale="jac",
                max_nfev=3000,
            )
            for start in (best_free, previous)
        ]
        solution = min(solutions, key=lambda item: np.sum(item.fun**2))
        previous = solution.x
        output[index] = np.sqrt(np.mean(profiled_residual(solution.x) ** 2))
    return output


def demo(radius: np.ndarray) -> None:
    """Mechanical validity and exact synthetic-recovery checks."""
    template_mass = 11.5
    template_r50 = np.log10(12.0)
    truth_parameters = {
        "sersic": np.array([template_mass, template_r50, np.log(3.0)]),
        "moffat": np.array([template_mass, template_r50, np.log(0.8)]),
        "gompertz_log": np.array([template_mass, template_r50, np.log(1.2)]),
        "richards": np.array([template_mass, template_r50, np.log(1.4), np.log(0.7)]),
        "radial_sigmoid": np.array(
            [template_mass, 0.12, np.log(1.5), np.log10(15.0), np.log(0.6)]
        ),
        "radial_sigmoid_beta0": np.array(
            [template_mass, np.log(1.5), np.log10(15.0), np.log(0.6)]
        ),
        "double_sersic_n2_n1": np.array(
            [template_mass, 0.2, np.log10(4.0), np.log(4.0)]
        ),
        "double_sersic_n4_n1": np.array(
            [template_mass, 0.2, np.log10(4.0), np.log(4.0)]
        ),
        "double_sersic_n4_n2": np.array(
            [template_mass, 0.2, np.log10(4.0), np.log(4.0)]
        ),
    }

    for family, parameters in truth_parameters.items():
        synthetic = cog_log(family, parameters, radius)
        if not np.isfinite(synthetic).all() or np.any(np.diff(synthetic) < -1e-10):
            raise AssertionError(f"{family} does not produce a valid monotone CoG")
        result = fit_cog(family, radius, synthetic)
        if result["rms_dex"] > 1e-6:
            raise AssertionError(
                f"{family} synthetic recovery RMS is {result['rms_dex']:.3e} dex"
            )
    print(f"exp55 analytic-model demo OK for {len(truth_parameters)} families")
