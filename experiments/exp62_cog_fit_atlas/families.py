"""Analytic CoG families and one common bounded multi-start fitter.

All amplitudes and component fractions refer to the outermost measured
aperture. Optimizer coordinates are chosen to enforce positivity and component
ordering mechanically rather than through penalties.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit, gammainc, gammaincinv


@dataclass(frozen=True)
class FamilySpec:
    """One analytic family in optimizer coordinates."""

    name: str
    parameter_names: tuple[str, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    group: str

    @property
    def n_parameter(self) -> int:
        return len(self.parameter_names)


LOG_MASS_BOUNDS = (8.0, 14.0)
LOG_RADIUS_BOUNDS = (np.log10(0.3), np.log10(300.0))
LOG_INDEX_BOUNDS = (np.log(0.3), np.log(10.0))
LOGIT_FRACTION_BOUNDS = (-5.0, 5.0)
LOG_RATIO_MINUS_ONE_BOUNDS = (np.log(0.03), np.log(100.0))


def _single_spec(name: str, shape_name: str, shape_bounds: tuple[float, float]):
    return FamilySpec(
        name,
        ("log_mstar_148", "log_r50", shape_name),
        (LOG_MASS_BOUNDS[0], LOG_RADIUS_BOUNDS[0], shape_bounds[0]),
        (LOG_MASS_BOUNDS[1], LOG_RADIUS_BOUNDS[1], shape_bounds[1]),
        name,
    )


FAMILY_SPECS: dict[str, FamilySpec] = {
    "sersic": _single_spec("sersic", "log_n", LOG_INDEX_BOUNDS),
    "moffat": _single_spec(
        "moffat", "log_gamma_minus_one", (np.log(0.05), np.log(5.0))
    ),
    "gompertz_log": _single_spec("gompertz_log", "log_c", (np.log(0.1), np.log(8.0))),
    "gompertz_linear": _single_spec(
        "gompertz_linear", "log_c_per_kpc", (np.log(0.001), np.log(3.0))
    ),
    "log_logistic": _single_spec("log_logistic", "log_k", (np.log(0.1), np.log(8.0))),
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
        "richards",
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
        "radial_sigmoid_beta0",
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
        "radial_sigmoid",
    ),
    "double_sersic_free": FamilySpec(
        "double_sersic_free",
        (
            "log_mstar_148",
            "logit_f_compact_148",
            "log_r_compact",
            "log_ratio_minus_one",
            "log_n_compact",
            "log_n_extended",
        ),
        (
            LOG_MASS_BOUNDS[0],
            LOGIT_FRACTION_BOUNDS[0],
            LOG_RADIUS_BOUNDS[0],
            LOG_RATIO_MINUS_ONE_BOUNDS[0],
            LOG_INDEX_BOUNDS[0],
            LOG_INDEX_BOUNDS[0],
        ),
        (
            LOG_MASS_BOUNDS[1],
            LOGIT_FRACTION_BOUNDS[1],
            LOG_RADIUS_BOUNDS[1],
            LOG_RATIO_MINUS_ONE_BOUNDS[1],
            LOG_INDEX_BOUNDS[1],
            LOG_INDEX_BOUNDS[1],
        ),
        "double_sersic_free",
    ),
}

FIXED_DOUBLE_SERSIC_INDICES = ((1.0, 1.0), (2.0, 1.0), (4.0, 1.0), (4.0, 2.0))
SERSIC_MOFFAT_INDICES = tuple(
    (n_sersic, gamma)
    for n_sersic in (0.5, 0.75, 1.0, 1.25, 2.0)
    for gamma in (1.1, 1.25, 1.4, 1.7)
)


def _mixture_spec(name: str, group: str) -> FamilySpec:
    return FamilySpec(
        name,
        (
            "log_mstar_148",
            "logit_f_compact_148",
            "log_r_compact",
            "log_ratio_minus_one",
        ),
        (
            LOG_MASS_BOUNDS[0],
            LOGIT_FRACTION_BOUNDS[0],
            LOG_RADIUS_BOUNDS[0],
            LOG_RATIO_MINUS_ONE_BOUNDS[0],
        ),
        (
            LOG_MASS_BOUNDS[1],
            LOGIT_FRACTION_BOUNDS[1],
            LOG_RADIUS_BOUNDS[1],
            LOG_RATIO_MINUS_ONE_BOUNDS[1],
        ),
        group,
    )


def fixed_double_sersic_name(n_compact: float, n_extended: float) -> str:
    return f"double_sersic_n{n_compact:g}_n{n_extended:g}".replace(".", "p")


def sersic_moffat_name(n_sersic: float, gamma: float) -> str:
    return f"sersic{n_sersic:g}_moffat{gamma:g}".replace(".", "p")


for _indices in FIXED_DOUBLE_SERSIC_INDICES:
    _name = fixed_double_sersic_name(*_indices)
    FAMILY_SPECS[_name] = _mixture_spec(_name, "double_sersic_fixed")
for _indices in SERSIC_MOFFAT_INDICES:
    _name = sersic_moffat_name(*_indices)
    FAMILY_SPECS[_name] = _mixture_spec(_name, "sersic_moffat_fixed")


SINGLE_FAMILIES = (
    "sersic",
    "moffat",
    "gompertz_log",
    "gompertz_linear",
    "log_logistic",
    "richards",
    "radial_sigmoid_beta0",
    "radial_sigmoid",
)
FIXED_DOUBLE_SERSIC_FAMILIES = tuple(
    fixed_double_sersic_name(*indices) for indices in FIXED_DOUBLE_SERSIC_INDICES
)
SERSIC_MOFFAT_FAMILIES = tuple(
    sersic_moffat_name(*indices) for indices in SERSIC_MOFFAT_INDICES
)
FIT_VARIANTS = (
    SINGLE_FAMILIES
    + FIXED_DOUBLE_SERSIC_FAMILIES
    + ("double_sersic_free",)
    + SERSIC_MOFFAT_FAMILIES
)
ATLAS_GROUPS = SINGLE_FAMILIES + (
    "double_sersic_fixed",
    "double_sersic_free",
    "sersic_moffat_fixed",
)


def sersic_fraction(radius: np.ndarray, r50: float, n_sersic: float) -> np.ndarray:
    """Infinite-aperture projected Sersic cumulative fraction."""
    b_n = gammaincinv(2.0 * n_sersic, 0.5)
    argument = b_n * (np.asarray(radius, float) / r50) ** (1.0 / n_sersic)
    return gammainc(2.0 * n_sersic, argument)


def moffat_fraction(radius: np.ndarray, r50: float, gamma: float) -> np.ndarray:
    half_scale = np.sqrt(2.0 ** (1.0 / (gamma - 1.0)) - 1.0)
    r_core = r50 / half_scale
    return 1.0 - (1.0 + (np.asarray(radius, float) / r_core) ** 2) ** (1.0 - gamma)


def _fixed_indices(family: str) -> tuple[float, float]:
    for indices in FIXED_DOUBLE_SERSIC_INDICES:
        if family == fixed_double_sersic_name(*indices):
            return indices
    raise ValueError(f"cannot parse fixed double-Sersic family {family!r}")


def _sersic_moffat_indices(family: str) -> tuple[float, float]:
    for indices in SERSIC_MOFFAT_INDICES:
        if family == sersic_moffat_name(*indices):
            return indices
    raise ValueError(f"cannot parse Sersic-Moffat family {family!r}")


def _mixture_fraction(
    parameters: np.ndarray,
    radius: np.ndarray,
    compact_shape: np.ndarray,
    extended_shape: np.ndarray,
) -> np.ndarray:
    compact_shape = compact_shape / compact_shape[-1]
    extended_shape = extended_shape / extended_shape[-1]
    fraction = expit(parameters[1])
    return fraction * compact_shape + (1.0 - fraction) * extended_shape


def cog_log(family: str, parameters: np.ndarray, radius: np.ndarray) -> np.ndarray:
    """Evaluate log10 cumulative mass with amplitude defined at the last radius."""
    parameters = np.asarray(parameters, float)
    radius = np.asarray(radius, float)
    spec = FAMILY_SPECS[family]
    if parameters.shape != (spec.n_parameter,):
        raise ValueError(f"{family} needs {spec.n_parameter} parameters")

    log_mass = parameters[0]
    if family in SINGLE_FAMILIES[:6]:
        r50 = 10.0 ** parameters[1]
    if family == "sersic":
        fraction = sersic_fraction(radius, r50, np.exp(parameters[2]))
    elif family == "moffat":
        fraction = moffat_fraction(radius, r50, 1.0 + np.exp(parameters[2]))
    elif family == "gompertz_log":
        exponent = np.log(2.0) * (r50 / radius) ** np.exp(parameters[2])
        fraction = np.exp(-np.clip(exponent, 0.0, 745.0))
    elif family == "gompertz_linear":
        exponent = np.log(2.0) * np.exp(
            np.clip(np.exp(parameters[2]) * (r50 - radius), -700.0, 700.0)
        )
        fraction = np.exp(-np.clip(exponent, 0.0, 745.0))
    elif family == "log_logistic":
        inverse_power = np.exp(
            np.clip(-np.exp(parameters[2]) * np.log(radius / r50), -700.0, 700.0)
        )
        fraction = 1.0 / (1.0 + inverse_power)
    elif family == "richards":
        k_shape = np.exp(parameters[2])
        nu = np.exp(parameters[3])
        half_factor = np.expm1(nu * np.log(2.0)) / nu
        scale = r50 * half_factor ** (1.0 / k_shape)
        inverse_power = np.exp(
            np.clip(-k_shape * np.log(radius / scale), -700.0, 700.0)
        )
        fraction = np.exp(-np.log1p(nu * inverse_power) / nu)
    elif family == "radial_sigmoid_beta0":
        log_radius = np.log(radius)
        scaled = (log_radius - np.log(10.0 ** parameters[2])) / np.exp(parameters[3])
        log_shape = (
            np.exp(parameters[1])
            * (log_radius - np.exp(parameters[3]) * np.logaddexp(0.0, scaled))
            / np.log(10.0)
        )
        return log_mass + log_shape - log_shape[-1]
    elif family == "radial_sigmoid":
        log_radius = np.log(radius)
        scaled = (log_radius - np.log(10.0 ** parameters[3])) / np.exp(parameters[4])
        integral = log_radius - np.exp(parameters[4]) * np.logaddexp(0.0, scaled)
        log_shape = (
            parameters[1] * log_radius + np.exp(parameters[2]) * integral
        ) / np.log(10.0)
        return log_mass + log_shape - log_shape[-1]
    elif family in FIXED_DOUBLE_SERSIC_FAMILIES:
        n_compact, n_extended = _fixed_indices(family)
        r_compact = 10.0 ** parameters[2]
        r_extended = r_compact * (1.0 + np.exp(parameters[3]))
        fraction = _mixture_fraction(
            parameters,
            radius,
            sersic_fraction(radius, r_compact, n_compact),
            sersic_fraction(radius, r_extended, n_extended),
        )
    elif family == "double_sersic_free":
        r_compact = 10.0 ** parameters[2]
        r_extended = r_compact * (1.0 + np.exp(parameters[3]))
        fraction = _mixture_fraction(
            parameters,
            radius,
            sersic_fraction(radius, r_compact, np.exp(parameters[4])),
            sersic_fraction(radius, r_extended, np.exp(parameters[5])),
        )
    elif family in SERSIC_MOFFAT_FAMILIES:
        n_sersic, gamma = _sersic_moffat_indices(family)
        r_compact = 10.0 ** parameters[2]
        r_extended = r_compact * (1.0 + np.exp(parameters[3]))
        fraction = _mixture_fraction(
            parameters,
            radius,
            sersic_fraction(radius, r_compact, n_sersic),
            moffat_fraction(radius, r_extended, gamma),
        )
    else:
        raise ValueError(f"unknown family {family!r}")

    fraction = np.clip(fraction, 1e-300, None)
    return log_mass + np.log10(fraction / fraction[-1])


def observed_r50(truth_log: np.ndarray, radius: np.ndarray) -> float:
    mass = 10.0 ** np.asarray(truth_log, float)
    target = 0.5 * mass[-1]
    if target <= mass[0]:
        slope = np.diff(np.log(mass[:2]))[0] / np.diff(np.log(radius[:2]))[0]
        value = radius[0] * np.exp(
            (np.log(target) - np.log(mass[0])) / max(slope, 1e-3)
        )
        return float(np.clip(value, 0.3, radius[-1]))
    return float(np.interp(target, mass, radius))


def initial_parameters(family: str, truth_log: np.ndarray, radius: np.ndarray):
    """Widely separated, data-scaled deterministic starts."""
    amplitude = float(truth_log[-1])
    r50 = observed_r50(truth_log, radius)
    log_r50 = np.log10(r50)
    if family == "sersic":
        starts = [[amplitude, log_r50, np.log(value)] for value in (0.7, 2.0, 6.0)]
    elif family == "moffat":
        starts = [
            [amplitude, log_r50, np.log(value - 1.0)] for value in (1.15, 1.7, 4.0)
        ]
    elif family in ("gompertz_log", "log_logistic"):
        starts = [[amplitude, log_r50, np.log(value)] for value in (0.3, 1.2, 4.0)]
    elif family == "gompertz_linear":
        starts = [[amplitude, log_r50, np.log(value)] for value in (0.01, 0.1, 1.0)]
    elif family == "richards":
        starts = [
            [amplitude, log_r50, np.log(k_shape), np.log(nu)]
            for k_shape, nu in ((0.4, 0.05), (1.2, 1.0), (3.5, 5.0), (1.0, 15.0))
        ]
    elif family == "radial_sigmoid_beta0":
        starts = [
            [amplitude, np.log(delta), log_rc, np.log(width)]
            for delta, log_rc, width in (
                (1.0, np.log10(max(r50, 1.0)), 0.3),
                (3.0, np.log10(max(r50 / 2.0, 0.3)), 1.0),
                (0.3, np.log10(min(2.0 * r50, 300.0)), 2.0),
            )
        ]
    elif family == "radial_sigmoid":
        starts = [
            [amplitude, outer, np.log(delta), log_rc, np.log(width)]
            for outer, delta, log_rc, width in (
                (0.05, 1.0, np.log10(max(r50, 1.0)), 0.3),
                (0.3, 2.5, np.log10(max(r50 / 2.0, 0.3)), 1.0),
                (1.0, 0.3, np.log10(min(2.0 * r50, 300.0)), 2.0),
            )
        ]
    else:
        starts = []
        for compact_fraction, compact_factor, ratio in (
            (0.15, 0.15, 8.0),
            (0.5, 0.35, 3.0),
            (0.85, 0.7, 1.5),
            (0.5, 0.08, 20.0),
        ):
            base = [
                amplitude,
                np.log(compact_fraction / (1.0 - compact_fraction)),
                np.log10(np.clip(compact_factor * r50, 0.3, 250.0)),
                np.log(ratio - 1.0),
            ]
            if family == "double_sersic_free":
                starts.extend(
                    [
                        base + [np.log(ni), np.log(no)]
                        for ni, no in ((1.0, 1.0), (4.0, 1.0), (1.0, 4.0))
                    ]
                )
            else:
                starts.append(base)
    spec = FAMILY_SPECS[family]
    lower, upper = np.asarray(spec.lower), np.asarray(spec.upper)
    return [np.clip(np.asarray(start), lower + 1e-8, upper - 1e-8) for start in starts]


def quantity_values(cog_log_values: np.ndarray, radius: np.ndarray, quantity: str):
    mass = 10.0 ** np.asarray(cog_log_values, float)
    if quantity == "cog":
        return mass
    shells = np.concatenate([[mass[0]], np.diff(mass)])
    if quantity == "shells":
        return shells
    if quantity == "density":
        area = np.pi * np.diff(np.asarray(radius, float) ** 2)
        return shells[1:] / area
    raise ValueError(f"unknown quantity {quantity!r}")


def residual_vector(
    parameters: np.ndarray,
    family: str,
    radius: np.ndarray,
    truth_log: np.ndarray,
    quantity: str = "cog",
    residual: str = "log",
) -> np.ndarray:
    prediction_log = cog_log(family, parameters, radius)
    if not np.isfinite(prediction_log).all():
        size = len(radius) - (quantity == "density")
        return np.full(size, 100.0)
    model = quantity_values(prediction_log, radius, quantity)
    truth = quantity_values(truth_log, radius, quantity)
    if residual == "log":
        return np.log10(np.clip(model, 1.0e-30, None)) - np.log10(
            np.clip(truth, 1.0e-30, None)
        )
    if residual == "absolute":
        norm = truth_log[-1]
        if quantity == "density":
            norm = np.log10(np.max(truth))
        return (model - truth) / (10.0**norm)
    if residual == "fractional":
        return (model - truth) / np.clip(truth, 1.0e-30, None)
    raise ValueError(f"unknown residual {residual!r}")


def fit_cog(
    family: str,
    radius: np.ndarray,
    truth_log: np.ndarray,
    quantity: str = "cog",
    residual: str = "log",
    max_nfev: int = 1200,
    method: str = "trf",
) -> dict:
    """Fit one CoG and return the reconstruction and stability diagnostics."""
    spec = FAMILY_SPECS[family]
    lower, upper = np.asarray(spec.lower), np.asarray(spec.upper)
    solutions = []
    for start in initial_parameters(family, truth_log, radius):
        solution = least_squares(
            residual_vector,
            start,
            args=(family, radius, truth_log, quantity, residual),
            bounds=(lower, upper),
            method=method,
            x_scale="jac",
            max_nfev=max_nfev,
        )
        prediction = cog_log(family, solution.x, radius)
        score = float(
            np.mean(
                residual_vector(
                    solution.x, family, radius, truth_log, quantity, residual
                )
                ** 2
            )
        )
        solutions.append((score, solution, prediction))
    solutions.sort(key=lambda item: item[0])
    _, best, prediction = solutions[0]
    cog_rms = float(np.sqrt(np.mean((prediction - truth_log) ** 2)))
    near = [item for item in solutions if item[0] <= 1.01 * solutions[0][0] + 1e-14]
    near_parameters = np.asarray([item[1].x for item in near])
    near_predictions = np.asarray([item[2] for item in near])
    parameter_range = upper - lower
    scaled_jacobian = best.jac * parameter_range[None, :]
    singular_values = np.linalg.svd(scaled_jacobian, compute_uv=False)
    position = (best.x - lower) / parameter_range
    return {
        "parameters": best.x,
        "prediction": prediction,
        "cog_rms_dex": cog_rms,
        "success": bool(best.success),
        "nfev": int(best.nfev),
        "jacobian_min_ratio": float(singular_values[-1] / singular_values[0])
        if singular_values[0] > 0
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


def synthetic_parameters(family: str) -> np.ndarray:
    """An interior, non-trivial parameter vector for mechanical recovery tests."""
    amplitude, log_r50 = 11.5, np.log10(12.0)
    if family == "sersic":
        return np.array([amplitude, log_r50, np.log(3.0)])
    if family == "moffat":
        return np.array([amplitude, log_r50, np.log(0.8)])
    if family in ("gompertz_log", "log_logistic"):
        return np.array([amplitude, log_r50, np.log(1.2)])
    if family == "gompertz_linear":
        return np.array([amplitude, log_r50, np.log(0.12)])
    if family == "richards":
        return np.array([amplitude, log_r50, np.log(1.4), np.log(0.7)])
    if family == "radial_sigmoid_beta0":
        return np.array([amplitude, np.log(1.5), np.log10(15.0), np.log(0.6)])
    if family == "radial_sigmoid":
        return np.array([amplitude, 0.12, np.log(1.5), np.log10(15.0), np.log(0.6)])
    base = [amplitude, 0.2, np.log10(4.0), np.log(4.0)]
    return np.array(
        base + ([np.log(2.0), np.log(1.0)] if family == "double_sersic_free" else [])
    )


def self_check(radius: np.ndarray) -> None:
    """Analytic validity, nesting, and exact synthetic-recovery checks."""
    for family in FIT_VARIANTS:
        parameters = synthetic_parameters(family)
        synthetic = cog_log(family, parameters, radius)
        assert np.isfinite(synthetic).all(), family
        assert np.all(np.diff(synthetic) >= -1.0e-10), family
        assert abs(synthetic[-1] - parameters[0]) < 1.0e-12, family
        result = fit_cog(family, radius, synthetic)
        assert result["cog_rms_dex"] < 2.0e-5, (
            family,
            result["cog_rms_dex"],
        )

    r50, k_shape = 10.0, 1.2
    log_logistic = cog_log(
        "log_logistic", np.array([0.0, np.log10(r50), np.log(k_shape)]), radius
    )
    richards = cog_log(
        "richards",
        np.array([0.0, np.log10(r50), np.log(k_shape), 0.0]),
        radius,
    )
    assert np.max(np.abs(log_logistic - richards)) < 1.0e-12
    print(f"exp62 family mechanics OK for {len(FIT_VARIANTS)} fitted variants")
