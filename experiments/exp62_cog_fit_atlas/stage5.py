# %%
"""Exp62 Stage 5: search for compact, monotonic analytic CoG families.

Commands:

    uv run python experiments/exp62_cog_fit_atlas/stage5.py mechanics
    uv run python experiments/exp62_cog_fit_atlas/stage5.py demo
    uv run python experiments/exp62_cog_fit_atlas/stage5.py full
    uv run python experiments/exp62_cog_fit_atlas/stage5.py figures demo|full

The unrestricted double-Sersic fit is the accuracy ceiling. Candidate forms
contain four or five total per-profile parameters, including the finite-
aperture mass, and are monotonic by construction.
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.special import betainc, expit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from families import observed_r50  # noqa: E402
from hongshao.cubic_logit import cubic_logit_cumulative_fraction  # noqa: E402
from hongshao.plotting import OKABE_ITO, save_fig, set_style  # noqa: E402
from hongshao.profile_emulator import density_from_cog  # noqa: E402
from hongshao.provenance import write_manifest  # noqa: E402
from hongshao.qa import enclosed_radius  # noqa: E402
from run import OUTDIR, RADII, REDSHIFTS, profile_metrics  # noqa: E402
from visual_record import _binned_statistics, _log_annulus_mass, _mass_at_radius  # noqa: E402

set_style()

STAGE5_OUTDIR = OUTDIR / "stage5"
STAGE5_FIGDIR = HERE / "figures" / "stage5"
MAX_WORKERS = min(
    int(os.environ.get("EXP62_STAGE5_WORKERS", "10")), os.cpu_count() or 1
)


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    label: str
    parameter_names: tuple[str, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    objective: str = "log"

    @property
    def n_parameter(self) -> int:
        return len(self.parameter_names)


LOG_MASS_BOUNDS = (8.0, 14.0)
LOG_SCALE_BOUNDS = (np.log10(0.15), np.log10(500.0))
LOG_SLOPE_BOUNDS = (np.log(0.05), np.log(20.0))
CUBIC5_PARAMETER_NAMES = (
    "log_mstar_148",
    "log_scale_radius",
    "log_derivative_margin",
    "quadratic_warp",
    "log_cubic_warp",
)
CUBIC5_LOWER = (
    LOG_MASS_BOUNDS[0],
    LOG_SCALE_BOUNDS[0],
    LOG_SLOPE_BOUNDS[0],
    -8.0,
    np.log(1.0e-4),
)
CUBIC5_UPPER = (
    LOG_MASS_BOUNDS[1],
    LOG_SCALE_BOUNDS[1],
    LOG_SLOPE_BOUNDS[1],
    8.0,
    np.log(5.0),
)


def _spec(name: str, label: str, shape_names: tuple[str, ...]) -> CandidateSpec:
    lower = [LOG_MASS_BOUNDS[0], LOG_SCALE_BOUNDS[0]]
    upper = [LOG_MASS_BOUNDS[1], LOG_SCALE_BOUNDS[1]]
    for shape_name in shape_names:
        if shape_name == "log_transition_width":
            lower.append(np.log(0.15))
            upper.append(np.log(3.0))
        else:
            lower.append(LOG_SLOPE_BOUNDS[0])
            upper.append(LOG_SLOPE_BOUNDS[1])
    return CandidateSpec(
        name,
        label,
        ("log_mstar_148", "log_scale_radius", *shape_names),
        tuple(lower),
        tuple(upper),
    )


CANDIDATE_SPECS = {
    "burr4": _spec(
        "burr4",
        "Generalized Moffat / Burr (4 parameters)",
        ("log_inner_slope", "log_tail_ratio"),
    ),
    "beta_prime5": _spec(
        "beta_prime5",
        "Regularized-beta cumulative (5 parameters)",
        ("log_alpha", "log_beta", "log_radial_power"),
    ),
    "logit_warp4": _spec(
        "logit_warp4",
        "Fixed-width logit warp (4 parameters)",
        ("log_inner_logit_slope", "log_outer_logit_slope"),
    ),
    "logit_warp5": _spec(
        "logit_warp5",
        "Free-width logit warp (5 parameters)",
        (
            "log_inner_logit_slope",
            "log_outer_logit_slope",
            "log_transition_width",
        ),
    ),
    "logit_cubic4": _spec(
        "logit_cubic4",
        "Monotonic cubic logit (4 parameters)",
        ("log_linear_warp", "log_cubic_warp"),
    ),
    "logit_cubic5": CandidateSpec(
        "logit_cubic5",
        "Asymmetric monotonic cubic logit (5 parameters)",
        CUBIC5_PARAMETER_NAMES,
        CUBIC5_LOWER,
        CUBIC5_UPPER,
    ),
    "logit_cubic5_absolute": CandidateSpec(
        "logit_cubic5_absolute",
        "Asymmetric monotonic cubic logit, absolute-CoG fit (5 parameters)",
        CUBIC5_PARAMETER_NAMES,
        CUBIC5_LOWER,
        CUBIC5_UPPER,
        "absolute",
    ),
}

for _hybrid_weight in (0.1, 0.25, 0.5, 1.0):
    _weight_label = f"{_hybrid_weight:g}".replace(".", "p")
    _name = f"logit_cubic5_hybrid{_weight_label}"
    CANDIDATE_SPECS[_name] = CandidateSpec(
        _name,
        f"Asymmetric monotonic cubic logit, hybrid weight {_hybrid_weight:g} (5 parameters)",
        CUBIC5_PARAMETER_NAMES,
        CUBIC5_LOWER,
        CUBIC5_UPPER,
        f"hybrid:{_hybrid_weight:g}",
    )


def _base_family(family: str) -> str:
    if family == "logit_cubic5_absolute" or family.startswith("logit_cubic5_hybrid"):
        return "logit_cubic5"
    return family


def _positive_fraction(family: str, parameters: np.ndarray, radius: np.ndarray):
    family = _base_family(family)
    scale = 10.0 ** parameters[1]
    log_ratio = np.log(np.asarray(radius, float) / scale)
    if family == "burr4":
        inner_slope = np.exp(parameters[2])
        tail_ratio = np.exp(parameters[3])
        log_survival = -tail_ratio * np.logaddexp(0.0, inner_slope * log_ratio)
        return -np.expm1(log_survival)
    if family == "beta_prime5":
        alpha = np.exp(parameters[2])
        beta = np.exp(parameters[3])
        radial_power = np.exp(parameters[4])
        transformed_radius = expit(radial_power * log_ratio)
        return betainc(alpha, beta, transformed_radius)
    if family in ("logit_warp4", "logit_warp5"):
        inner_slope = np.exp(parameters[2])
        outer_slope = np.exp(parameters[3])
        width = 1.0 if family == "logit_warp4" else np.exp(parameters[4])
        transition = width * (np.logaddexp(0.0, log_ratio / width) - np.log(2.0))
        warp = inner_slope * log_ratio + (outer_slope - inner_slope) * transition
        return expit(warp)
    if family == "logit_cubic4":
        linear_warp = np.exp(parameters[2])
        cubic_warp = np.exp(parameters[3])
        return expit(linear_warp * log_ratio + cubic_warp * log_ratio**3)
    if family == "logit_cubic5":
        derivative_margin = np.exp(parameters[2])
        quadratic_warp = parameters[3]
        cubic_warp = np.exp(parameters[4])
        return cubic_logit_cumulative_fraction(
            radius,
            scale,
            derivative_margin,
            quadratic_warp,
            cubic_warp,
        )
    raise ValueError(f"unknown Stage 5 candidate {family!r}")


def candidate_cog_log(
    family: str, parameters: np.ndarray, radius: np.ndarray
) -> np.ndarray:
    """Evaluate a candidate with mass defined at the last measured radius."""
    spec = CANDIDATE_SPECS[family]
    parameters = np.asarray(parameters, float)
    if parameters.shape != (spec.n_parameter,):
        raise ValueError(f"{family} needs {spec.n_parameter} parameters")
    fraction = np.clip(_positive_fraction(family, parameters, radius), 1.0e-300, 1.0)
    return parameters[0] + np.log10(fraction / fraction[-1])


def candidate_initial_parameters(
    family: str, truth_log: np.ndarray, radius: np.ndarray
) -> list[np.ndarray]:
    base_family = _base_family(family)
    amplitude = float(truth_log[-1])
    r50 = observed_r50(truth_log, radius)
    if base_family == "burr4":
        configurations = (
            (0.5, 1.0, 0.8),
            (1.0, 2.0, 0.5),
            (1.5, 1.0, 2.0),
            (3.0, 0.6, 4.0),
        )
    elif base_family == "beta_prime5":
        configurations = (
            (0.5, 1.0, 1.0, 1.0),
            (1.0, 2.0, 1.0, 0.7),
            (1.5, 1.0, 2.0, 1.5),
            (3.0, 0.7, 3.0, 2.0),
        )
    elif base_family == "logit_warp4":
        configurations = (
            (0.5, 1.0, 1.0),
            (1.0, 2.5, 0.6),
            (1.5, 0.7, 2.5),
            (3.0, 4.0, 1.0),
        )
    elif base_family == "logit_warp5":
        configurations = (
            (0.5, 1.0, 1.0, 0.5),
            (1.0, 2.5, 0.6, 1.0),
            (1.5, 0.7, 2.5, 1.8),
            (3.0, 4.0, 1.0, 2.5),
        )
    elif base_family == "logit_cubic4":
        configurations = (
            (0.5, 1.0, 0.03),
            (1.0, 2.0, 0.01),
            (1.5, 0.7, 0.1),
            (3.0, 3.0, 0.3),
        )
    elif base_family == "logit_cubic5":
        configurations = (
            (0.5, 1.0, -0.5, 0.03),
            (1.0, 2.0, 0.0, 0.01),
            (1.5, 0.7, 0.5, 0.1),
            (3.0, 3.0, -1.5, 0.3),
        )
    else:
        raise ValueError(f"unknown Stage 5 candidate {family!r}")

    starts = []
    for configuration in configurations:
        scale_factor, *shape = configuration
        starts.append(
            np.asarray(
                [
                    amplitude,
                    np.log10(np.clip(scale_factor * r50, 0.15, 500.0)),
                    *(
                        [np.log(shape[0]), shape[1], np.log(shape[2])]
                        if base_family == "logit_cubic5"
                        else np.log(shape)
                    ),
                ]
            )
        )
    spec = CANDIDATE_SPECS[family]
    lower = np.asarray(spec.lower)
    upper = np.asarray(spec.upper)
    return [np.clip(start, lower + 1.0e-8, upper - 1.0e-8) for start in starts]


def _residual(
    parameters: np.ndarray, family: str, radius: np.ndarray, truth_log: np.ndarray
) -> np.ndarray:
    prediction = candidate_cog_log(family, parameters, radius)
    if not np.all(np.isfinite(prediction)):
        return np.full(len(radius), 100.0)
    objective = CANDIDATE_SPECS[family].objective
    if objective == "log":
        return prediction - truth_log
    if objective == "absolute":
        return (10.0**prediction - 10.0**truth_log) / 10.0 ** truth_log[-1]
    if objective.startswith("hybrid:"):
        weight = float(objective.partition(":")[2])
        log_residual = prediction - truth_log
        absolute_residual = (10.0**prediction - 10.0**truth_log) / 10.0 ** truth_log[-1]
        return np.concatenate([log_residual, weight * absolute_residual])
    raise ValueError(f"unknown Stage 5 objective {objective!r}")


def fit_candidate(
    family: str,
    radius: np.ndarray,
    truth_log: np.ndarray,
    max_nfev: int = 1000,
) -> dict:
    """Fit one candidate from separated starts and report stability diagnostics."""
    spec = CANDIDATE_SPECS[family]
    lower = np.asarray(spec.lower)
    upper = np.asarray(spec.upper)
    solutions = []
    for start in candidate_initial_parameters(family, truth_log, radius):
        solution = least_squares(
            _residual,
            start,
            args=(family, radius, truth_log),
            bounds=(lower, upper),
            method="trf",
            x_scale="jac",
            max_nfev=max_nfev,
        )
        prediction = candidate_cog_log(family, solution.x, radius)
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


def candidate_synthetic_parameters(family: str) -> np.ndarray:
    family = _base_family(family)
    base = [11.5, np.log10(12.0)]
    if family == "burr4":
        return np.asarray(base + [np.log(1.4), np.log(1.2)])
    if family == "beta_prime5":
        return np.asarray(base + [np.log(1.4), np.log(1.1), np.log(1.2)])
    if family == "logit_warp4":
        return np.asarray(base + [np.log(1.4), np.log(0.9)])
    if family == "logit_warp5":
        return np.asarray(base + [np.log(1.4), np.log(0.9), np.log(0.8)])
    if family == "logit_cubic4":
        return np.asarray(base + [np.log(1.4), np.log(0.08)])
    if family == "logit_cubic5":
        return np.asarray(base + [np.log(0.8), 0.4, np.log(0.08)])
    raise ValueError(f"unknown Stage 5 candidate {family!r}")


def mechanics() -> None:
    dense_radius = np.geomspace(0.01, 1000.0, 1000)
    for family in CANDIDATE_SPECS:
        parameters = candidate_synthetic_parameters(family)
        dense_profile = candidate_cog_log(family, parameters, dense_radius)
        assert np.all(np.isfinite(dense_profile)), family
        assert np.all(np.diff(dense_profile) > 0.0), family
        profile = candidate_cog_log(family, parameters, RADII)
        assert abs(profile[-1] - parameters[0]) < 1.0e-12, family
        fitted = fit_candidate(family, RADII, profile)
        assert fitted["cog_rms_dex"] < 2.0e-5, (
            family,
            fitted["cog_rms_dex"],
        )
    print(f"exp62 Stage 5 mechanics OK for {len(CANDIDATE_SPECS)} candidates")


def _fit_task(task):
    family, truth_log = task
    return fit_candidate(family, RADII, truth_log)


def _fit_family(
    family: str,
    truth_log: np.ndarray,
    executor: ProcessPoolExecutor | None,
) -> dict[str, np.ndarray]:
    tasks = [(family, profile) for profile in truth_log]
    if executor is None:
        fitted = [_fit_task(task) for task in tasks]
    else:
        fitted = list(executor.map(_fit_task, tasks, chunksize=8))
    return {
        "parameters": np.asarray([item["parameters"] for item in fitted]),
        "prediction": np.asarray([item["prediction"] for item in fitted]),
        "success": np.asarray([item["success"] for item in fitted]),
        "nfev": np.asarray([item["nfev"] for item in fitted]),
        "jacobian_min_ratio": np.asarray(
            [item["jacobian_min_ratio"] for item in fitted]
        ),
        "boundary_distance": np.asarray([item["boundary_distance"] for item in fitted]),
        "near_parameter_spread": np.asarray(
            [item["near_parameter_spread"] for item in fitted]
        ),
        "near_profile_spread_dex": np.asarray(
            [item["near_profile_spread_dex"] for item in fitted]
        ),
    }


def _median_by_epoch(values: np.ndarray, epochs: np.ndarray) -> list[float]:
    return [float(np.nanmedian(values[epochs == epoch])) for epoch in range(5)]


def _summarize_prediction(
    prediction: np.ndarray,
    truth_log: np.ndarray,
    epochs: np.ndarray,
    diagnostics: dict[str, np.ndarray] | None = None,
) -> dict:
    metrics = profile_metrics(prediction, truth_log)
    summary = {
        "median_full_cog_rms_dex": _median_by_epoch(
            metrics["cog_rms_full_dex"], epochs
        ),
        "median_cog_rms_5_30_dex": _median_by_epoch(
            metrics["cog_rms_5_30_dex"], epochs
        ),
        "median_density_rms_dex": _median_by_epoch(metrics["density_rms_dex"], epochs),
    }
    for column, fraction in enumerate((20, 50, 80, 90)):
        summary[f"median_absolute_r{fraction}_error_dex"] = _median_by_epoch(
            np.abs(metrics["size_error_dex"][:, column]), epochs
        )
    if diagnostics is not None:
        summary.update(
            {
                "boundary_fraction_within_one_percent": _median_by_epoch(
                    diagnostics["boundary_distance"] < 0.01, epochs
                ),
                "median_jacobian_min_ratio": _median_by_epoch(
                    diagnostics["jacobian_min_ratio"], epochs
                ),
                "failure_fraction": _median_by_epoch(~diagnostics["success"], epochs),
                "median_near_parameter_spread": _median_by_epoch(
                    diagnostics["near_parameter_spread"], epochs
                ),
                "median_near_profile_spread_dex": _median_by_epoch(
                    diagnostics["near_profile_spread_dex"], epochs
                ),
            }
        )
    return summary


def _tier_decision(summaries: dict[str, dict]) -> dict[str, dict]:
    reference = summaries["double_sersic_free"]
    pca = summaries["pca3"]
    output = {}
    for family in CANDIDATE_SPECS:
        if family not in summaries:
            continue
        candidate = summaries[family]
        replacement_metrics = (
            "median_full_cog_rms_dex",
            "median_cog_rms_5_30_dex",
            "median_density_rms_dex",
            "median_absolute_r50_error_dex",
            "median_absolute_r80_error_dex",
            "median_absolute_r90_error_dex",
        )
        replacement_accuracy = all(
            np.all(
                np.asarray(candidate[metric]) <= 1.10 * np.asarray(reference[metric])
            )
            for metric in replacement_metrics
        )
        better_conditioning = np.median(
            candidate["median_jacobian_min_ratio"]
        ) >= 10.0 * np.median(reference["median_jacobian_min_ratio"])
        lower_boundaries = np.median(
            candidate["boundary_fraction_within_one_percent"]
        ) <= 0.5 * np.median(reference["boundary_fraction_within_one_percent"])
        epochs_beating_pca = int(
            np.sum(
                np.asarray(candidate["median_full_cog_rms_dex"])
                <= np.asarray(pca["median_full_cog_rms_dex"])
            )
        )
        pca_local_match = all(
            np.median(candidate[metric]) <= np.median(pca[metric])
            for metric in (
                "median_density_rms_dex",
                "median_absolute_r80_error_dex",
                "median_absolute_r90_error_dex",
            )
        )
        output[family] = {
            "replacement_numerical_gate": bool(
                replacement_accuracy and better_conditioning and lower_boundaries
            ),
            "replacement_accuracy_gate": bool(replacement_accuracy),
            "better_conditioning_than_double_sersic": bool(better_conditioning),
            "lower_boundary_incidence_than_double_sersic": bool(lower_boundaries),
            "epochs_beating_pca3_full_cog": epochs_beating_pca,
            "pca3_density_outer_size_gate": bool(pca_local_match),
            "useful_compression_numerical_gate": bool(
                epochs_beating_pca >= 4 and pca_local_match
            ),
            "visual_qa_gate": None,
        }
    return output


def _load_base(mode: str):
    path = OUTDIR / mode / "atlas" / "predictions.npz"
    if not path.exists():
        raise FileNotFoundError(f"run Exp62 Stage 1 {mode!r} before Stage 5")
    return np.load(path)


def _validation_mask(
    rows: np.ndarray,
    epochs: np.ndarray,
    development_rows: np.ndarray,
    development_epochs: np.ndarray,
) -> np.ndarray:
    development_pairs = set(
        zip(
            development_rows.astype(int).tolist(),
            development_epochs.astype(int).tolist(),
        )
    )
    return np.asarray(
        [
            (int(row), int(epoch)) not in development_pairs
            for row, epoch in zip(rows, epochs)
        ],
        dtype=bool,
    )


def _selected_candidates(mode: str) -> tuple[str, ...]:
    if mode == "demo":
        return tuple(CANDIDATE_SPECS)
    demo_summary_path = STAGE5_OUTDIR / "demo" / "summary.json"
    if not demo_summary_path.exists():
        raise FileNotFoundError("run the Stage 5 demo before the full calculation")
    demo_summary = json.loads(demo_summary_path.read_text())
    selected = [
        family
        for family, decision in demo_summary["decisions"].items()
        if decision["replacement_accuracy_gate"]
        or decision["epochs_beating_pca3_full_cog"] >= 4
    ]
    hybrid = [family for family in selected if "_hybrid" in family]
    if len(hybrid) > 1:
        reference = demo_summary["summaries"]["double_sersic_free"]
        metrics = (
            "median_full_cog_rms_dex",
            "median_cog_rms_5_30_dex",
            "median_density_rms_dex",
            "median_absolute_r80_error_dex",
            "median_absolute_r90_error_dex",
        )

        def worst_ratio(family: str) -> float:
            candidate = demo_summary["summaries"][family]
            return max(
                np.max(np.asarray(candidate[metric]) / np.asarray(reference[metric]))
                for metric in metrics
            )

        winner = min(hybrid, key=worst_ratio)
        selected = [
            family for family in selected if family not in hybrid or family == winner
        ]
    selected = tuple(selected)
    if not selected:
        raise RuntimeError("no Stage 5 candidate passed the predeclared full-run gate")
    return selected


def run(mode: str, force: bool = False) -> dict:
    base = _load_base(mode)
    truth_log = np.asarray(base["truth_log"])
    epochs = np.asarray(base["epochs"], int)
    stage_dir = STAGE5_OUTDIR / mode
    stage_dir.mkdir(parents=True, exist_ok=True)
    selected_candidates = _selected_candidates(mode)
    print(
        f"exp62 Stage 5 {mode}: {len(truth_log)} profiles, "
        f"{len(selected_candidates)} candidates, {MAX_WORKERS} workers"
    )
    start = time.perf_counter()
    fitted = {}
    cache_path = stage_dir / "predictions.npz"
    if cache_path.exists() and not force:
        cached = np.load(cache_path)
        for family in selected_candidates:
            if f"prediction_{family}" not in cached.files:
                continue
            fitted[family] = {
                "parameters": np.asarray(cached[f"parameters_{family}"]),
                "prediction": np.asarray(cached[f"prediction_{family}"]),
                **{
                    diagnostic: np.asarray(cached[f"{diagnostic}_{family}"])
                    for diagnostic in (
                        "success",
                        "nfev",
                        "jacobian_min_ratio",
                        "boundary_distance",
                        "near_parameter_spread",
                        "near_profile_spread_dex",
                    )
                },
            }
    remaining = [family for family in selected_candidates if family not in fitted]
    if MAX_WORKERS > 1 and remaining:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for family in remaining:
                fitted[family] = _fit_family(family, truth_log, executor)
    else:
        for family in remaining:
            fitted[family] = _fit_family(family, truth_log, None)

    summaries = {
        family: _summarize_prediction(result["prediction"], truth_log, epochs, result)
        for family, result in fitted.items()
    }
    for reference in ("pca3", "pca5", "sersic_moffat_fixed", "double_sersic_free"):
        diagnostics = None
        if reference == "double_sersic_free":
            variant = np.load(OUTDIR / mode / "variants" / f"{reference}.npz")
            diagnostics = {
                key: np.asarray(variant[key])
                for key in (
                    "success",
                    "jacobian_min_ratio",
                    "boundary_distance",
                    "near_parameter_spread",
                    "near_profile_spread_dex",
                )
            }
        summaries[reference] = _summarize_prediction(
            np.asarray(base[f"prediction_{reference}"]),
            truth_log,
            epochs,
            diagnostics,
        )
    decisions = _tier_decision(summaries)
    validation_summaries = None
    validation_decisions = None
    n_validation_profile = None
    if mode == "full":
        development = np.load(STAGE5_OUTDIR / "demo" / "predictions.npz")
        validation = _validation_mask(
            np.asarray(base["rows"], int),
            epochs,
            np.asarray(development["rows"], int),
            np.asarray(development["epochs"], int),
        )
        n_validation_profile = int(np.count_nonzero(validation))
        validation_summaries = {}
        for family, result_for_family in fitted.items():
            diagnostics = {
                key: np.asarray(result_for_family[key])[validation]
                for key in (
                    "success",
                    "jacobian_min_ratio",
                    "boundary_distance",
                    "near_parameter_spread",
                    "near_profile_spread_dex",
                )
            }
            validation_summaries[family] = _summarize_prediction(
                result_for_family["prediction"][validation],
                truth_log[validation],
                epochs[validation],
                diagnostics,
            )
        for reference in (
            "pca3",
            "pca5",
            "sersic_moffat_fixed",
            "double_sersic_free",
        ):
            diagnostics = None
            if reference == "double_sersic_free":
                variant = np.load(OUTDIR / mode / "variants" / f"{reference}.npz")
                diagnostics = {
                    key: np.asarray(variant[key])[validation]
                    for key in (
                        "success",
                        "jacobian_min_ratio",
                        "boundary_distance",
                        "near_parameter_spread",
                        "near_profile_spread_dex",
                    )
                }
            validation_summaries[reference] = _summarize_prediction(
                np.asarray(base[f"prediction_{reference}"])[validation],
                truth_log[validation],
                epochs[validation],
                diagnostics,
            )
        validation_decisions = _tier_decision(validation_summaries)
    np.savez_compressed(
        stage_dir / "predictions.npz",
        rows=base["rows"],
        epochs=epochs,
        redshifts=REDSHIFTS,
        truth_log=truth_log,
        halo_mass=base["halo_mass"],
        **{
            f"prediction_{family}": result["prediction"]
            for family, result in fitted.items()
        },
        **{
            f"parameters_{family}": result["parameters"]
            for family, result in fitted.items()
        },
        **{
            f"{diagnostic}_{family}": result[diagnostic]
            for family, result in fitted.items()
            for diagnostic in (
                "success",
                "nfev",
                "jacobian_min_ratio",
                "boundary_distance",
                "near_parameter_spread",
                "near_profile_spread_dex",
            )
        },
    )
    elapsed = time.perf_counter() - start
    result = {
        "mode": mode,
        "n_profile": len(truth_log),
        "n_candidate": len(selected_candidates),
        "selected_candidates": list(selected_candidates),
        "wall_seconds_before_figures": elapsed,
        "subminute_gate_passed": bool(elapsed < 60.0) if mode == "demo" else None,
        "summaries": summaries,
        "decisions": decisions,
        "n_validation_profile": n_validation_profile,
        "validation_summaries": validation_summaries,
        "validation_decisions": validation_decisions,
    }
    (stage_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    write_manifest(
        stage_dir,
        {
            "experiment": "exp62_cog_fit_atlas",
            "stage": 5,
            "mode": mode,
            "n_profile": len(truth_log),
            "candidate_families": list(selected_candidates),
        },
    )
    make_figures(mode)
    elapsed_with_figures = time.perf_counter() - start
    result["wall_seconds_complete"] = elapsed_with_figures
    if mode == "demo":
        result["subminute_gate_passed"] = bool(elapsed_with_figures < 60.0)
    (stage_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if mode == "demo" and elapsed_with_figures >= 60.0:
        raise RuntimeError(
            f"Stage 5 demo took {elapsed_with_figures:.2f} seconds; full run forbidden"
        )
    return result


def _load_results(mode: str) -> dict[str, np.ndarray]:
    stage = np.load(STAGE5_OUTDIR / mode / "predictions.npz")
    base = _load_base(mode)
    candidate_names = tuple(
        name.removeprefix("prediction_")
        for name in stage.files
        if name.startswith("prediction_")
    )
    return {
        "rows": np.asarray(stage["rows"], int),
        "epochs": np.asarray(stage["epochs"], int),
        "truth_log": np.asarray(stage["truth_log"]),
        "halo_mass": np.asarray(stage["halo_mass"]),
        **{
            family: np.asarray(stage[f"prediction_{family}"])
            for family in candidate_names
        },
        **{
            f"parameters_{family}": np.asarray(stage[f"parameters_{family}"])
            for family in candidate_names
        },
        **{
            reference: np.asarray(base[f"prediction_{reference}"])
            for reference in (
                "pca3",
                "pca5",
                "sersic_moffat_fixed",
                "double_sersic_free",
            )
        },
    }


def _best_candidate(summary: dict) -> str:
    return min(
        [family for family in CANDIDATE_SPECS if family in summary["summaries"]],
        key=lambda family: np.median(
            summary["summaries"][family]["median_full_cog_rms_dex"]
        ),
    )


def frontier_figure(mode: str, data: dict, summary: dict) -> None:
    candidate_names = [
        family for family in CANDIDATE_SPECS if family in summary["summaries"]
    ]
    names = [
        *candidate_names,
        "pca3",
        "sersic_moffat_fixed",
        "double_sersic_free",
    ]
    labels = {
        **{name: spec.label for name, spec in CANDIDATE_SPECS.items()},
        "pca3": "PCA(3), 4 parameters",
        "sersic_moffat_fixed": "Constrained Sersic--Moffat, 4 parameters",
        "double_sersic_free": "Unrestricted double Sersic, 6 parameters",
    }
    candidate_colors = [OKABE_ITO[index % 6] for index in range(len(candidate_names))]
    colors = dict(zip(names, [*candidate_colors, OKABE_ITO[6], OKABE_ITO[2], "black"]))
    figure, axes = plt.subplots(2, 3, figsize=(13.5, 7.8))
    metrics = (
        ("median_full_cog_rms_dex", "Median full-CoG RMS (dex)"),
        ("median_cog_rms_5_30_dex", "Median 5--30 kpc CoG RMS (dex)"),
        ("median_density_rms_dex", "Median density RMS (dex)"),
        ("median_absolute_r50_error_dex", "Median |R50 error| (dex)"),
        ("median_absolute_r80_error_dex", "Median |R80 error| (dex)"),
        ("median_absolute_r90_error_dex", "Median |R90 error| (dex)"),
    )
    for axis, (metric, ylabel) in zip(axes.flat, metrics):
        for name in names:
            axis.plot(
                REDSHIFTS,
                summary["summaries"][name][metric],
                marker="o",
                linewidth=1.4,
                markersize=3.5,
                color=colors[name],
                label=labels[name],
            )
        axis.set(xlabel="Redshift", ylabel=ylabel)
        axis.invert_xaxis()
    axes[0, 0].legend(frameon=False, fontsize=6.3, ncol=2)
    figure.suptitle("Exp62 Stage 5 analytic-compression frontier; lower is better")
    figure.tight_layout()
    save_fig(figure, STAGE5_FIGDIR / mode / "exp62_stage5_candidate_frontier")
    plt.close(figure)


def _qa_candidates(mode: str, data: dict, best: str) -> tuple[str, ...]:
    available = tuple(family for family in CANDIDATE_SPECS if family in data)
    return available if mode == "full" else (best,)


def average_cog_figure(
    mode: str, data: dict, best: str, bin_kind: str = "halo"
) -> None:
    if bin_kind not in ("halo", "stellar"):
        raise ValueError("bin_kind must be halo or stellar")
    candidates = _qa_candidates(mode, data, best)
    comparisons = (*candidates, "pca3", "double_sersic_free")
    labels = {
        **{name: CANDIDATE_SPECS[name].label for name in candidates},
        "pca3": "PCA(3)",
        "double_sersic_free": "Unrestricted double Sersic",
    }
    colors = (*OKABE_ITO[: len(candidates)], OKABE_ITO[6], "black")
    figure, axes = plt.subplots(5, 3, figsize=(13.5, 15.0), sharex=True, sharey=True)
    truth = 10.0 ** data["truth_log"]
    for epoch, redshift in enumerate(REDSHIFTS):
        in_epoch = data["epochs"] == epoch
        bin_values = (
            data["halo_mass"][in_epoch]
            if bin_kind == "halo"
            else data["truth_log"][in_epoch, -1]
        )
        edges = np.quantile(
            bin_values[np.isfinite(bin_values)],
            [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0],
        )
        for mass_bin in range(3):
            axis = axes[epoch, mass_bin]
            all_bin_values = (
                data["halo_mass"] if bin_kind == "halo" else data["truth_log"][:, -1]
            )
            selected = (
                in_epoch
                & (all_bin_values >= edges[mass_bin])
                & (all_bin_values <= edges[mass_bin + 1])
            )
            for name, color in zip(comparisons, colors):
                prediction = 10.0 ** data[name][selected]
                pinned = prediction * truth[selected, -1:] / prediction[:, -1:]
                residual = 100.0 * np.median(
                    (pinned - truth[selected]) / truth[selected], axis=0
                )
                axis.plot(
                    RADII, residual, color=color, linewidth=1.5, label=labels[name]
                )
            axis.axhline(0.0, color="0.65", linewidth=0.8)
            axis.set_xscale("log")
            mass_symbol = r"M_{\rm h}" if bin_kind == "halo" else r"M_{*,148}"
            axis.set_title(
                rf"$z={redshift:g}$; $\log {mass_symbol}={edges[mass_bin]:.2f}$--${edges[mass_bin + 1]:.2f}$"
            )
            if mass_bin == 0:
                axis.set_ylabel("Median mass-pinned residual (%)")
            if epoch == 4:
                axis.set_xlabel("Semi-major axis (kpc)")
    axes[0, 0].legend(frameon=False, fontsize=6.5)
    noun = "concurrent halo-mass" if bin_kind == "halo" else "stellar-mass"
    figure.suptitle(f"Exp62 Stage 5 average CoG recovery in {noun} terciles")
    figure.tight_layout()
    save_fig(
        figure,
        STAGE5_FIGDIR / mode / f"exp62_stage5_average_cog_{bin_kind}_mass_bins",
    )
    plt.close(figure)


def density_qa_figure(mode: str, data: dict, best: str) -> None:
    candidates = _qa_candidates(mode, data, best)
    comparisons = (*candidates, "pca3", "double_sersic_free")
    labels = {
        **{name: CANDIDATE_SPECS[name].label for name in candidates},
        "pca3": "PCA(3)",
        "double_sersic_free": "Unrestricted double Sersic",
    }
    colors = (*OKABE_ITO[: len(candidates)], OKABE_ITO[6], "black")
    truth_density, density_radius = density_from_cog(data["truth_log"], RADII)
    figure, axes = plt.subplots(5, 3, figsize=(13.5, 15.0), sharex=True, sharey=True)
    for epoch, redshift in enumerate(REDSHIFTS):
        in_epoch = data["epochs"] == epoch
        mass = data["halo_mass"][in_epoch]
        edges = np.quantile(mass[np.isfinite(mass)], [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
        for mass_bin in range(3):
            selected = (
                in_epoch
                & (data["halo_mass"] >= edges[mass_bin])
                & (data["halo_mass"] <= edges[mass_bin + 1])
            )
            axis = axes[epoch, mass_bin]
            for name, color in zip(comparisons, colors):
                model_density, _ = density_from_cog(data[name][selected], RADII)
                residual = np.median(
                    model_density - truth_density[selected],
                    axis=0,
                )
                axis.plot(
                    density_radius,
                    residual,
                    color=color,
                    linewidth=1.4,
                    label=labels[name],
                )
            axis.axhline(0.0, color="0.65", linewidth=0.8)
            axis.set_xscale("log")
            axis.set_title(
                rf"$z={redshift:g}$; $\log M_{{\rm h}}={edges[mass_bin]:.2f}$--${edges[mass_bin + 1]:.2f}$"
            )
            if mass_bin == 0:
                axis.set_ylabel("Median log-density residual (dex)")
            if epoch == 4:
                axis.set_xlabel("Semi-major axis (kpc)")
    axes[0, 0].legend(frameon=False, fontsize=6.5)
    figure.suptitle("Exp62 Stage 5 density recovery in concurrent halo-mass terciles")
    figure.tight_layout()
    save_fig(figure, STAGE5_FIGDIR / mode / "exp62_stage5_density_halo_mass_bins")
    plt.close(figure)


def _mass_quantities(profile: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "m30": _mass_at_radius(profile, 30.0),
        "m30_50": _log_annulus_mass(profile, 30.0, 50.0),
        "m50_100": _log_annulus_mass(profile, 50.0, 100.0),
        "m100_148": _log_annulus_mass(profile, 100.0, RADII[-1]),
    }


def _empirical_cdf(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    ordered = np.sort(values[np.isfinite(values)])
    return np.searchsorted(ordered, grid, side="right") / len(ordered)


def mass_cdf_figure(mode: str, data: dict, best: str) -> None:
    candidates = _qa_candidates(mode, data, best)
    comparisons = (*candidates, "pca3", "double_sersic_free")
    labels = {
        **{name: CANDIDATE_SPECS[name].label for name in candidates},
        "pca3": "PCA(3)",
        "double_sersic_free": "Unrestricted double Sersic",
    }
    colors = (*OKABE_ITO[: len(candidates)], OKABE_ITO[6], "black")
    quantities = (
        ("m30", r"$M_*(<30)$"),
        ("m30_50", r"$M_*(30\!-\!50)$"),
        ("m50_100", r"$M_*(50\!-\!100)$"),
        ("m100_148", r"$M_*(100\!-\!148)$"),
    )
    figure, axes = plt.subplots(4, 5, figsize=(15.7, 11.7))
    for epoch, redshift in enumerate(REDSHIFTS):
        selected = data["epochs"] == epoch
        truth = _mass_quantities(data["truth_log"][selected])
        models = {name: _mass_quantities(data[name][selected]) for name in comparisons}
        for row, (quantity, label) in enumerate(quantities):
            axis = axes[row, epoch]
            display = truth[quantity][truth[quantity] > 5.0]
            grid = np.linspace(
                np.quantile(display, 0.005), np.quantile(display, 0.995), 160
            )
            truth_cdf = _empirical_cdf(truth[quantity], grid)
            for name, color in zip(comparisons, colors):
                axis.plot(
                    grid,
                    _empirical_cdf(models[name][quantity], grid) - truth_cdf,
                    color=color,
                    linewidth=1.4,
                    label=labels[name],
                )
            axis.axhline(0.0, color="0.65", linewidth=0.8)
            if epoch == 0:
                axis.set_ylabel(f"{label}\nmodel CDF - TNG CDF")
            if row == 0:
                axis.set_title(rf"$z={redshift:g}$")
            if row == 3:
                axis.set_xlabel(r"$\log_{10} M_*/M_\odot$")
    axes[0, 0].legend(frameon=False, fontsize=6.2)
    figure.suptitle("Exp62 Stage 5 aperture, annulus, and outskirt mass CDF residuals")
    figure.tight_layout()
    save_fig(figure, STAGE5_FIGDIR / mode / "exp62_stage5_mass_cdf_residuals")
    plt.close(figure)


def population_qa_figure(mode: str, data: dict, best: str) -> None:
    candidates = _qa_candidates(mode, data, best)
    comparisons = ("truth_log", *candidates, "pca3", "double_sersic_free")
    labels = {
        "truth_log": "TNG CoG",
        **{name: CANDIDATE_SPECS[name].label for name in candidates},
        "pca3": "PCA(3)",
        "double_sersic_free": "Unrestricted double Sersic",
    }
    colors = (
        "black",
        *OKABE_ITO[: len(candidates)],
        OKABE_ITO[6],
        OKABE_ITO[2],
    )
    figure, axes = plt.subplots(3, 5, figsize=(15.7, 9.2), sharex="col")
    for epoch, redshift in enumerate(REDSHIFTS):
        in_epoch = data["epochs"] == epoch
        for name, color in zip(comparisons, colors):
            profile = data[name][in_epoch]
            inner = _mass_at_radius(profile, 30.0)
            outer = _log_annulus_mass(profile, 50.0, 100.0)
            center, median, lower, upper = _binned_statistics(inner, outer)
            axes[0, epoch].plot(center, median, color=color, label=labels[name])
            if name == "truth_log":
                axes[0, epoch].fill_between(
                    center, lower, upper, color="0.75", alpha=0.3
                )

            linear = 10.0**profile
            mass = profile[:, -1]
            for row, fraction in enumerate((0.8, 0.9), start=1):
                size = np.log10(
                    [enclosed_radius(value, RADII, fraction) for value in linear]
                )
                center, median, lower, upper = _binned_statistics(mass, size)
                axes[row, epoch].plot(center, median, color=color, label=labels[name])
                if name == "truth_log":
                    axes[row, epoch].fill_between(
                        center, lower, upper, color="0.75", alpha=0.3
                    )
        axes[0, epoch].set_title(rf"$z={redshift:g}$")
        axes[0, epoch].set_xlabel(r"$\log_{10} M_*(<30\,{\rm kpc})$")
        axes[1, epoch].set_xlabel(r"$\log_{10} M_{*,148}$")
        axes[2, epoch].set_xlabel(r"$\log_{10} M_{*,148}$")
    axes[0, 0].set_ylabel(r"$\log_{10} M_*(50\!-!100\,{\rm kpc})$")
    axes[1, 0].set_ylabel(r"$\log_{10} R_{80}\ ({\rm kpc})$")
    axes[2, 0].set_ylabel(r"$\log_{10} R_{90}\ ({\rm kpc})$")
    axes[0, 0].legend(frameon=False, fontsize=6.2)
    figure.suptitle(
        "Exp62 Stage 5 stellar-mass and outer-size planes; grey bands show the measured 16--84% width"
    )
    figure.tight_layout()
    save_fig(figure, STAGE5_FIGDIR / mode / "exp62_stage5_population_qa")
    plt.close(figure)


def worst_cases_figure(mode: str, data: dict, family: str) -> None:
    metrics = profile_metrics(data[family], data["truth_log"])
    worst = np.argsort(metrics["cog_rms_full_dex"])[-10:][::-1]
    figure, axes = plt.subplots(5, 2, figsize=(10.5, 15.0), sharex=True)
    for axis, index in zip(axes.flat, worst):
        truth = data["truth_log"][index]
        prediction = data[family][index]
        axis.plot(RADII, truth - truth[-1], color="black", label="TNG")
        axis.plot(
            RADII,
            prediction - prediction[-1],
            color=OKABE_ITO[0],
            label=CANDIDATE_SPECS[family].label,
        )
        axis.plot(RADII, prediction - truth, color=OKABE_ITO[5], linestyle="--")
        axis.axhline(0.0, color="0.7", linewidth=0.7)
        axis.set_xscale("log")
        axis.set_title(
            rf"galaxy {data['rows'][index]}; $z={REDSHIFTS[data['epochs'][index]]:g}$; RMS={metrics['cog_rms_full_dex'][index]:.4f} dex"
        )
        axis.set_ylabel("Normalized log CoG / residual (dex)")
        axis.set_xlabel("Semi-major axis (kpc)")
    axes[0, 0].legend(frameon=False, fontsize=7)
    figure.suptitle(f"Exp62 Stage 5 ten worst fits: {CANDIDATE_SPECS[family].label}")
    figure.tight_layout()
    save_fig(
        figure,
        STAGE5_FIGDIR / mode / f"exp62_stage5_worst10_{family}",
    )
    plt.close(figure)


def formula_figure(mode: str, data: dict, family: str = "logit_cubic5") -> None:
    if family not in data:
        return
    parameters = np.median(data[f"parameters_{family}"], axis=0)
    baseline = candidate_cog_log(family, parameters, RADII) - parameters[0]
    variations = (
        (1, -0.18, 0.18, r"Scale radius $R_s$"),
        (2, -0.6, 0.6, r"Derivative margin $m$"),
        (3, -0.6, 0.6, r"Asymmetry $b$"),
        (4, -0.6, 0.6, r"Cubic curvature $c$"),
    )
    figure = plt.figure(figsize=(13.0, 6.8))
    grid = figure.add_gridspec(2, 4, height_ratios=(0.75, 1.0))
    equation_axis = figure.add_subplot(grid[0, :])
    equation_axis.axis("off")
    equation_axis.text(
        0.5,
        0.72,
        r"$M_*(<R)=M_{*,148}\,\sigma[g(x)]/\sigma[g(x_{148})],\quad x=\ln(R/R_s)$",
        ha="center",
        va="center",
        fontsize=17,
    )
    equation_axis.text(
        0.5,
        0.39,
        r"$g(x)=a x+b x^2+c x^3,\quad c=e^{p_c}>0,\quad a=e^{p_m}+b^2/(3c)$",
        ha="center",
        va="center",
        fontsize=17,
    )
    equation_axis.text(
        0.5,
        0.08,
        r"$g'(x)=e^{p_m}+3c[x+b/(3c)]^2>0$: monotonicity is exact; five fitted coordinates are $M_{*,148},R_s,p_m,b,p_c$.",
        ha="center",
        va="center",
        fontsize=12,
    )
    for column, (index, low, high, title) in enumerate(variations):
        axis = figure.add_subplot(grid[1, column])
        axis.plot(RADII, baseline, color="black", linewidth=1.8, label="Median fit")
        for delta, color, linestyle in (
            (low, OKABE_ITO[1], "--"),
            (high, OKABE_ITO[5], ":"),
        ):
            changed = parameters.copy()
            changed[index] += delta
            profile = candidate_cog_log(family, changed, RADII) - changed[0]
            axis.plot(
                RADII,
                profile,
                color=color,
                linestyle=linestyle,
                linewidth=1.5,
                label=f"coordinate {delta:+.2f}",
            )
        axis.set(xscale="log", xlabel="Semi-major axis (kpc)", title=title)
        if column == 0:
            axis.set_ylabel(r"$\log_{10}[M_*(<R)/M_{*,148}]$")
            axis.legend(frameon=False, fontsize=7)
    figure.suptitle("Exp62 Stage 5 compact analytic CoG and parameter responses")
    figure.tight_layout()
    save_fig(figure, STAGE5_FIGDIR / mode / "exp62_stage5_cubic_logit_recipe")
    plt.close(figure)


def conditioning_figure(mode: str) -> None:
    stage = np.load(STAGE5_OUTDIR / mode / "predictions.npz")
    reference = np.load(OUTDIR / mode / "variants" / "double_sersic_free.npz")
    families = tuple(
        family
        for family in CANDIDATE_SPECS
        if f"jacobian_min_ratio_{family}" in stage.files
    )
    labels = {
        **{family: CANDIDATE_SPECS[family].label for family in families},
        "double_sersic_free": "Unrestricted double Sersic",
    }
    colors = (*OKABE_ITO[: len(families)], "black")
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 3.8))
    for family, color in zip((*families, "double_sersic_free"), colors):
        source = stage if family in families else reference
        suffix = f"_{family}" if family in families else ""
        jacobian = np.asarray(source[f"jacobian_min_ratio{suffix}"])
        boundary = np.asarray(source[f"boundary_distance{suffix}"])
        spread = np.asarray(source[f"near_parameter_spread{suffix}"])
        axes[0].hist(
            np.log10(np.clip(jacobian, 1.0e-12, None)),
            bins=np.linspace(-7.0, -0.5, 50),
            density=True,
            histtype="step",
            linewidth=1.5,
            color=color,
            label=labels[family],
        )
        ordered = np.sort(boundary)
        axes[1].plot(
            ordered,
            np.linspace(0.0, 1.0, len(ordered)),
            color=color,
            linewidth=1.5,
            label=labels[family],
        )
        axes[2].scatter(
            np.median(jacobian),
            np.median(spread),
            color=color,
            s=45,
            label=labels[family],
        )
    axes[0].set(
        xlabel=r"$\log_{10}$ scaled Jacobian singular-value ratio",
        ylabel="Probability density",
    )
    axes[0].legend(frameon=False, fontsize=6.5)
    axes[1].set(
        xscale="log",
        xlabel="Nearest-bound distance / allowed range",
        ylabel="Cumulative fraction",
    )
    axes[2].set(
        xscale="log",
        yscale="log",
        xlabel="Median scaled Jacobian ratio",
        ylabel="Median near-optimal parameter spread",
    )
    figure.suptitle("Exp62 Stage 5 parameter conditioning and boundaries")
    figure.tight_layout()
    save_fig(figure, STAGE5_FIGDIR / mode / "exp62_stage5_conditioning")
    plt.close(figure)


def _refit_on_mask(
    family: str,
    parameters: np.ndarray,
    truth_log: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    spec = CANDIDATE_SPECS[family]
    solution = least_squares(
        _residual,
        parameters,
        args=(family, RADII[mask], truth_log[mask]),
        bounds=(np.asarray(spec.lower), np.asarray(spec.upper)),
        method="trf",
        x_scale="jac",
        max_nfev=600,
    )
    return solution.x


def _robustness_rows(data: dict, count_per_epoch: int = 18) -> np.ndarray:
    selected = []
    for epoch in range(len(REDSHIFTS)):
        in_epoch = np.flatnonzero(data["epochs"] == epoch)
        order = in_epoch[np.argsort(data["truth_log"][in_epoch, -1])]
        positions = np.linspace(0, len(order) - 1, count_per_epoch).round().astype(int)
        selected.extend(order[positions])
    return np.asarray(selected, int)


def robustness(mode: str) -> dict:
    data = _load_results(mode)
    families = _qa_candidates(
        mode,
        data,
        _best_candidate(
            json.loads((STAGE5_OUTDIR / mode / "summary.json").read_text())
        ),
    )
    rows = _robustness_rows(data)
    masks = {
        "alternating_even": np.asarray(
            [(index % 2 == 0) or index == len(RADII) - 1 for index in range(len(RADII))]
        ),
        "alternating_odd": np.asarray([index % 2 == 1 for index in range(len(RADII))]),
        "radius_ge_5_kpc": RADII >= 5.0,
    }
    output = {}
    for family in families:
        spec = CANDIDATE_SPECS[family]
        base_parameters = data[f"parameters_{family}"][rows]
        base_profiles = data[family][rows]
        family_result = {}
        for name, mask in masks.items():
            parameters = np.asarray(
                [
                    _refit_on_mask(family, start, truth, mask)
                    for start, truth in zip(base_parameters, data["truth_log"][rows])
                ]
            )
            profiles = np.asarray(
                [candidate_cog_log(family, value, RADII) for value in parameters]
            )
            parameter_movement = np.max(
                np.abs(parameters - base_parameters)
                / (np.asarray(spec.upper) - np.asarray(spec.lower)),
                axis=1,
            )
            profile_movement = np.sqrt(np.mean((profiles - base_profiles) ** 2, axis=1))
            family_result[name] = {
                "median_maximum_parameter_movement_fraction": float(
                    np.median(parameter_movement)
                ),
                "median_decoded_profile_movement_dex": float(
                    np.median(profile_movement)
                ),
                "p90_decoded_profile_movement_dex": float(
                    np.quantile(profile_movement, 0.9)
                ),
            }
        output[family] = family_result
    result = {
        "mode": mode,
        "n_profile": len(rows),
        "families": output,
    }
    path = STAGE5_OUTDIR / mode / "robustness.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    robustness_figure(mode, result)
    return result


def robustness_figure(mode: str, result: dict) -> None:
    families = tuple(result["families"])
    perturbations = tuple(next(iter(result["families"].values())))
    x = np.arange(len(perturbations))
    width = 0.8 / len(families)
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    for index, family in enumerate(families):
        values = result["families"][family]
        offset = (index - 0.5 * (len(families) - 1)) * width
        axes[0].bar(
            x + offset,
            [
                values[name]["median_decoded_profile_movement_dex"]
                for name in perturbations
            ],
            width,
            color=OKABE_ITO[index],
            label=CANDIDATE_SPECS[family].label,
        )
        axes[1].bar(
            x + offset,
            [
                values[name]["median_maximum_parameter_movement_fraction"]
                for name in perturbations
            ],
            width,
            color=OKABE_ITO[index],
            label=CANDIDATE_SPECS[family].label,
        )
    labels = [name.replace("_", "\n") for name in perturbations]
    for axis in axes:
        axis.set_xticks(x, labels)
    axes[0].set_ylabel("Median decoded-CoG movement (dex)")
    axes[1].set_ylabel("Median max parameter movement / range")
    axes[0].legend(frameon=False, fontsize=6.5)
    figure.suptitle(
        f"Exp62 Stage 5 radial-jackknife stability ({result['n_profile']} mass-stratified profiles)"
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.92), w_pad=3.0)
    save_fig(figure, STAGE5_FIGDIR / mode / "exp62_stage5_radial_jackknife")
    plt.close(figure)


def make_figures(mode: str) -> str:
    data = _load_results(mode)
    summary = json.loads((STAGE5_OUTDIR / mode / "summary.json").read_text())
    best = _best_candidate(summary)
    frontier_figure(mode, data, summary)
    average_cog_figure(mode, data, best, "halo")
    average_cog_figure(mode, data, best, "stellar")
    density_qa_figure(mode, data, best)
    population_qa_figure(mode, data, best)
    mass_cdf_figure(mode, data, best)
    formula_figure(mode, data)
    conditioning_figure(mode)
    for family in _qa_candidates(mode, data, best):
        worst_cases_figure(mode, data, family)
    return best


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "mechanics"
    if command == "mechanics":
        mechanics()
    elif command in ("demo", "full"):
        run(command, force="--force" in sys.argv)
    elif command == "figures":
        figure_mode = sys.argv[2] if len(sys.argv) > 2 else "demo"
        print("best Stage 5 candidate:", make_figures(figure_mode))
    elif command == "robustness":
        robustness_mode = sys.argv[2] if len(sys.argv) > 2 else "demo"
        print(json.dumps(robustness(robustness_mode), indent=2))
    else:
        raise SystemExit(__doc__)
