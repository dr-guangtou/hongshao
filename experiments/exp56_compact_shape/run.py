"""Exp56 Stage 1: globally fixed compact Sersic-index closure test.

Commands
--------
demo
    Run the complete representation path on 90 stratified galaxies. The
    measured wall time must remain below one minute.
run
    Run the full representation grid after a successful demo record exists.
"""

# %% Imports and frozen configuration
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP55 = ROOT / "experiments" / "exp55_analytic_cog"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP55))

from halo_map import RADII, folds_of, load_sample  # noqa: E402
from hongshao.plotting import OKABE_ITO, save_fig, set_style  # noqa: E402
from hongshao.profile_emulator import density_from_cog  # noqa: E402
from hongshao.provenance import write_manifest  # noqa: E402
from mixed import (  # noqa: E402
    LOWER,
    UPPER,
    fit_one,
    mixed_cog_log,
    mixed_component_cogs,
    starts_for,
)
from outer_envelope import size_relation, size_values  # noqa: E402
from smooth_outer import smooth_gamma  # noqa: E402

set_style()

COMPACT_INDICES = (0.5, 0.75, 1.0)
FIXED_GAMMA = 1.4
SMOOTH_WIDTH_DEX = 0.20
N_BOOTSTRAP = 1000
N_DEMO_GALAXY = 90
N_DIAGNOSTIC_GALAXY = 12
DEMO_TIME_LIMIT_SECONDS = 60.0
SEED = 5600

FIGURE_DIRECTORY = HERE / "figures"
OUTPUT_DIRECTORY = HERE / "outputs"


# %% Profile fitting and checkpointing
def compact_index_label(compact_index: float) -> str:
    return f"n{compact_index:.2f}".replace(".", "p")


def fit_checkpoint_path(
    output_prefix: str,
    outer_law: str,
    compact_index: float,
    fold_number: int | None = None,
) -> Path:
    directory = OUTPUT_DIRECTORY / f"{output_prefix}profile_fits"
    fold_suffix = "" if fold_number is None else f"_fold{fold_number}"
    return directory / (
        f"{outer_law}{fold_suffix}_{compact_index_label(compact_index)}.npz"
    )


def fit_profile_cell(
    truth_log: np.ndarray,
    galaxy_indices: np.ndarray,
    gamma: np.ndarray,
    compact_index: float,
    checkpoint: Path,
    force: bool,
) -> dict[str, np.ndarray]:
    if checkpoint.exists() and not force:
        saved = dict(np.load(checkpoint))
        if (
            np.array_equal(saved["galaxy_indices"], galaxy_indices)
            and np.allclose(saved["gamma"], gamma)
            and np.isclose(float(saved["compact_index"]), compact_index)
        ):
            return saved

    arrays = {
        "galaxy_indices": np.asarray(galaxy_indices, int),
        "gamma": np.asarray(gamma, float),
        "compact_index": np.asarray(compact_index),
        "parameters": np.empty((len(truth_log), 4)),
        "prediction_log": np.empty_like(truth_log),
        "profile_rms_dex": np.empty(len(truth_log)),
        "jacobian_min_ratio": np.empty(len(truth_log)),
        "boundary_distance": np.empty(len(truth_log)),
        "success": np.empty(len(truth_log), dtype=bool),
    }
    for galaxy, (truth, slope) in enumerate(zip(truth_log, gamma)):
        result = fit_one(
            truth,
            n_sersic=compact_index,
            gamma=float(slope),
        )
        arrays["parameters"][galaxy] = result["parameters"]
        arrays["prediction_log"][galaxy] = result["prediction"]
        arrays["profile_rms_dex"][galaxy] = result["rms_dex"]
        arrays["jacobian_min_ratio"][galaxy] = result["jacobian_min_ratio"]
        arrays["boundary_distance"][galaxy] = result["boundary_distance"]
        arrays["success"][galaxy] = result["success"]
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(checkpoint, **arrays)
    return arrays


def collate_smooth_evaluation(
    fold_cells: list[dict[float, dict[str, np.ndarray]]],
    compact_index: float,
    n_galaxy: int,
) -> dict[str, np.ndarray]:
    output = {
        "gamma": np.empty(n_galaxy),
        "parameters": np.empty((n_galaxy, 4)),
        "prediction_log": np.empty((n_galaxy, len(RADII))),
        "profile_rms_dex": np.empty(n_galaxy),
        "jacobian_min_ratio": np.empty(n_galaxy),
        "boundary_distance": np.empty(n_galaxy),
        "success": np.empty(n_galaxy, dtype=bool),
    }
    for test, cells in zip(folds_of(n_galaxy), fold_cells):
        cell = cells[compact_index]
        for name in output:
            output[name][test] = cell[name][test]
    return output


def fit_declared_grid(
    sample: dict[str, np.ndarray],
    output_prefix: str,
    force: bool,
) -> dict:
    truth_log = sample["truth_log"]
    galaxy_indices = sample["indices"]
    halo_mass = sample["features"][:, 0]
    fixed_gamma = np.full(len(truth_log), FIXED_GAMMA)
    fixed = {}
    for compact_index in COMPACT_INDICES:
        fixed[compact_index] = fit_profile_cell(
            truth_log,
            galaxy_indices,
            fixed_gamma,
            compact_index,
            fit_checkpoint_path(
                output_prefix,
                "fixed_gamma1p4",
                compact_index,
            ),
            force,
        )

    smooth_folds = []
    transitions = []
    all_rows = np.arange(len(truth_log))
    for fold_number, test in enumerate(folds_of(len(truth_log))):
        train = np.setdiff1d(all_rows, test)
        transition = float(np.quantile(halo_mass[train], 2.0 / 3.0))
        gamma = smooth_gamma(halo_mass, transition, SMOOTH_WIDTH_DEX)
        transitions.append(transition)
        cells = {}
        for compact_index in COMPACT_INDICES:
            cells[compact_index] = fit_profile_cell(
                truth_log,
                galaxy_indices,
                gamma,
                compact_index,
                fit_checkpoint_path(
                    output_prefix,
                    "smooth",
                    compact_index,
                    fold_number,
                ),
                force,
            )
        smooth_folds.append(cells)
    smooth = {
        compact_index: collate_smooth_evaluation(
            smooth_folds,
            compact_index,
            len(truth_log),
        )
        for compact_index in COMPACT_INDICES
    }
    return {
        "fixed_gamma1p4": fixed,
        "smooth": smooth,
        "smooth_folds": smooth_folds,
        "smooth_transition_mass": np.asarray(transitions),
    }


# %% Metrics and fold-clean selection
def density_values(profile_log: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values, midpoint = density_from_cog(profile_log, RADII)
    return np.asarray(values, float), np.asarray(midpoint, float)


def amplitude_pinned_residual(
    prediction_log: np.ndarray,
    truth_log: np.ndarray,
) -> np.ndarray:
    residual = prediction_log - truth_log
    return residual - residual[:, -1, None]


def interpolated_log_mass(profile_log: np.ndarray, radius: float) -> np.ndarray:
    return np.asarray(
        [np.interp(np.log(radius), np.log(RADII), row) for row in profile_log]
    )


def per_galaxy_metrics(
    cell: dict[str, np.ndarray],
    truth_log: np.ndarray,
) -> dict[str, np.ndarray]:
    prediction_log = cell["prediction_log"]
    residual = amplitude_pinned_residual(prediction_log, truth_log)
    radial_mask = (RADII >= 5.0) & (RADII <= 30.0)
    model_density, density_radius = density_values(prediction_log)
    truth_density, _ = density_values(truth_log)
    density_mask = (density_radius >= 5.0) & (density_radius <= 30.0)
    model_sizes = size_values(prediction_log)
    truth_sizes = size_values(truth_log)
    output = {
        "full_cog_rms_dex": np.sqrt(np.mean((prediction_log - truth_log) ** 2, axis=1)),
        "five_thirty_cog_rms_dex": np.sqrt(
            np.mean(residual[:, radial_mask] ** 2, axis=1)
        ),
        "full_density_rms_dex": np.sqrt(
            np.mean((model_density - truth_density) ** 2, axis=1)
        ),
        "five_thirty_density_rms_dex": np.sqrt(
            np.mean(
                (model_density[:, density_mask] - truth_density[:, density_mask]) ** 2,
                axis=1,
            )
        ),
        "log10_jacobian_min_ratio": np.log10(
            np.clip(cell["jacobian_min_ratio"], 1e-16, None)
        ),
        "near_boundary": np.asarray(cell["boundary_distance"] <= 0.01, float),
        "absolute_size_error": np.abs(model_sizes - truth_sizes),
    }
    for radius in (2.0, 5.0, 10.0):
        output[f"aperture_error_{radius:g}_kpc_dex"] = interpolated_log_mass(
            prediction_log, radius
        ) - interpolated_log_mass(truth_log, radius)
    return output


def bootstrap_median_interval(
    difference: np.ndarray,
    seed: int,
) -> list[float]:
    difference = np.asarray(difference, float)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(N_BOOTSTRAP)
    for sample_number in range(N_BOOTSTRAP):
        rows = rng.integers(0, len(difference), len(difference))
        bootstrap[sample_number] = np.median(difference[rows])
    return np.quantile(bootstrap, [0.16, 0.50, 0.84]).tolist()


def selection_for_fold(
    cells: dict[float, dict[str, np.ndarray]],
    truth_log: np.ndarray,
    train: np.ndarray,
    seed: int,
) -> dict:
    metrics = {
        compact_index: per_galaxy_metrics(cell, truth_log)
        for compact_index, cell in cells.items()
    }
    primary_medians = {
        compact_index: float(np.median(values["five_thirty_cog_rms_dex"][train]))
        for compact_index, values in metrics.items()
    }
    winner = min(COMPACT_INDICES, key=lambda value: (primary_medians[value], value))
    reference = metrics[1.0]
    candidate = metrics[winner]
    primary_interval = bootstrap_median_interval(
        candidate["five_thirty_cog_rms_dex"][train]
        - reference["five_thirty_cog_rms_dex"][train],
        seed,
    )
    comparison_intervals = {
        "full_cog_rms_dex": bootstrap_median_interval(
            candidate["full_cog_rms_dex"][train] - reference["full_cog_rms_dex"][train],
            seed + 1,
        ),
        "full_density_rms_dex": bootstrap_median_interval(
            candidate["full_density_rms_dex"][train]
            - reference["full_density_rms_dex"][train],
            seed + 2,
        ),
        "log10_jacobian_min_ratio": bootstrap_median_interval(
            candidate["log10_jacobian_min_ratio"][train]
            - reference["log10_jacobian_min_ratio"][train],
            seed + 3,
        ),
        "near_boundary": bootstrap_median_interval(
            candidate["near_boundary"][train] - reference["near_boundary"][train],
            seed + 4,
        ),
    }
    for column, size_name in enumerate(("R50", "R80", "R90")):
        comparison_intervals[f"absolute_{size_name}_error"] = bootstrap_median_interval(
            candidate["absolute_size_error"][train, column]
            - reference["absolute_size_error"][train, column],
            seed + 10 + column,
        )
    primary_resolved = winner == 1.0 or primary_interval[2] < 0.0
    safeguards = {
        "overall_cog_not_resolved_worse": comparison_intervals["full_cog_rms_dex"][0]
        <= 0.0,
        "overall_density_not_resolved_worse": comparison_intervals[
            "full_density_rms_dex"
        ][0]
        <= 0.0,
        "conditioning_not_resolved_worse": comparison_intervals[
            "log10_jacobian_min_ratio"
        ][2]
        >= 0.0,
        "boundary_incidence_not_resolved_worse": comparison_intervals["near_boundary"][
            0
        ]
        <= 0.0,
        "R50_not_resolved_worse": comparison_intervals["absolute_R50_error"][0] <= 0.0,
        "R80_not_resolved_worse": comparison_intervals["absolute_R80_error"][0] <= 0.0,
        "R90_not_resolved_worse": comparison_intervals["absolute_R90_error"][0] <= 0.0,
    }
    selected = winner if primary_resolved and all(safeguards.values()) else 1.0
    return {
        "primary_median_by_n": primary_medians,
        "numerical_winner": winner,
        "primary_candidate_minus_n1_bootstrap_16_50_84_dex": primary_interval,
        "primary_improvement_resolved": primary_resolved,
        "comparison_bootstrap_16_50_84": comparison_intervals,
        "safeguards": safeguards,
        "selected_n": selected,
    }


def fold_clean_selection(
    grid: dict,
    truth_log: np.ndarray,
) -> dict:
    output = {"fixed_gamma1p4": [], "smooth": []}
    all_rows = np.arange(len(truth_log))
    for fold_number, test in enumerate(folds_of(len(truth_log))):
        train = np.setdiff1d(all_rows, test)
        output["fixed_gamma1p4"].append(
            selection_for_fold(
                grid["fixed_gamma1p4"],
                truth_log,
                train,
                SEED + 100 * fold_number,
            )
        )
        output["smooth"].append(
            selection_for_fold(
                grid["smooth_folds"][fold_number],
                truth_log,
                train,
                SEED + 1000 + 100 * fold_number,
            )
        )
    return output


# %% Synthetic, multiple-start, and radial-jackknife diagnostics
def physical_parameters(parameters: np.ndarray) -> np.ndarray:
    parameters = np.atleast_2d(np.asarray(parameters, float))
    return np.column_stack(
        [
            parameters[:, 0],
            expit(parameters[:, 1]),
            parameters[:, 2],
            parameters[:, 2]
            + np.log10(1.0 + np.exp(np.clip(parameters[:, 3], -40.0, 40.0))),
        ]
    )


def diagnostic_rows(halo_mass: np.ndarray) -> np.ndarray:
    order = np.argsort(halo_mass)
    positions = np.linspace(0, len(order) - 1, N_DIAGNOSTIC_GALAXY).round().astype(int)
    return np.unique(order[positions])


def all_start_solutions(
    truth_log: np.ndarray,
    compact_index: float,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    def residual(parameters: np.ndarray) -> np.ndarray:
        return mixed_cog_log(parameters, RADII, compact_index, gamma) - truth_log

    solutions = [
        least_squares(
            residual,
            start,
            bounds=(LOWER, UPPER),
            method="trf",
            x_scale="jac",
            max_nfev=2000,
        )
        for start in starts_for(truth_log)
    ]
    parameters = np.asarray([solution.x for solution in solutions])
    rms = np.asarray([np.sqrt(np.mean(solution.fun**2)) for solution in solutions])
    return parameters, rms


def diagnostic_summary(
    cell: dict[str, np.ndarray],
    truth_log: np.ndarray,
    compact_index: float,
    rows: np.ndarray,
) -> dict:
    keep_radius = np.unique(np.append(np.arange(0, len(RADII), 2), len(RADII) - 1))
    exact_error = []
    jackknife_error = []
    near_best_fraction = []
    near_best_parameter_range = []
    for galaxy in rows:
        gamma = float(cell["gamma"][galaxy])
        parameters = cell["parameters"][galaxy]
        physical = physical_parameters(parameters)[0]
        synthetic = mixed_cog_log(parameters, RADII, compact_index, gamma)
        recovered = fit_one(synthetic, compact_index, gamma)["parameters"]
        jackknife = fit_one(
            truth_log[galaxy, keep_radius],
            compact_index,
            gamma,
            RADII[keep_radius],
        )["parameters"]
        exact_error.append(physical_parameters(recovered)[0] - physical)
        jackknife_error.append(physical_parameters(jackknife)[0] - physical)
        start_parameters, start_rms = all_start_solutions(
            truth_log[galaxy], compact_index, gamma
        )
        near_best = start_rms <= np.min(start_rms) + 1e-6
        near_best_fraction.append(np.mean(near_best))
        near_physical = physical_parameters(start_parameters[near_best])
        near_best_parameter_range.append(np.ptp(near_physical, axis=0))
    exact_error = np.asarray(exact_error)
    jackknife_error = np.asarray(jackknife_error)
    near_best_parameter_range = np.asarray(near_best_parameter_range)
    return {
        "n_galaxy": len(rows),
        "maximum_exact_synthetic_absolute_error": np.max(
            np.abs(exact_error), axis=0
        ).tolist(),
        "median_radial_jackknife_absolute_change": np.median(
            np.abs(jackknife_error), axis=0
        ).tolist(),
        "maximum_radial_jackknife_absolute_change": np.max(
            np.abs(jackknife_error), axis=0
        ).tolist(),
        "median_fraction_of_starts_at_best_profile": float(
            np.median(near_best_fraction)
        ),
        "median_near_best_parameter_range": np.median(
            near_best_parameter_range, axis=0
        ).tolist(),
    }


# %% Result summaries and figures
def population_summary(
    cell: dict[str, np.ndarray],
    truth_log: np.ndarray,
) -> dict:
    metrics = per_galaxy_metrics(cell, truth_log)
    prediction_log = cell["prediction_log"]
    residual = amplitude_pinned_residual(prediction_log, truth_log)
    model_density, density_radius = density_values(prediction_log)
    truth_density, _ = density_values(truth_log)
    sizes = size_values(prediction_log)
    truth_sizes = size_values(truth_log)
    size_stats = size_relation(prediction_log)
    truth_size_stats = size_relation(truth_log)
    output = {
        "median_full_cog_rms_dex": float(np.median(metrics["full_cog_rms_dex"])),
        "median_five_thirty_cog_rms_dex": float(
            np.median(metrics["five_thirty_cog_rms_dex"])
        ),
        "median_full_density_rms_dex": float(
            np.median(metrics["full_density_rms_dex"])
        ),
        "median_five_thirty_density_rms_dex": float(
            np.median(metrics["five_thirty_density_rms_dex"])
        ),
        "median_log10_jacobian_min_ratio": float(
            np.median(metrics["log10_jacobian_min_ratio"])
        ),
        "fraction_within_one_percent_of_bound": float(
            np.mean(metrics["near_boundary"])
        ),
        "fraction_success": float(np.mean(cell["success"])),
        "median_amplitude_pinned_residual_percent": (
            100.0 * np.median(10.0**residual - 1.0, axis=0)
        ).tolist(),
        "density_radius_kpc": density_radius.tolist(),
        "median_density_residual_dex": np.median(
            model_density - truth_density, axis=0
        ).tolist(),
        "median_size_offset_dex": np.median(sizes - truth_sizes, axis=0).tolist(),
        "r90_slope": size_stats["R90"]["slope"],
        "truth_r90_slope": truth_size_stats["R90"]["slope"],
    }
    for radius in (2.0, 5.0, 10.0):
        values = metrics[f"aperture_error_{radius:g}_kpc_dex"]
        output[f"median_aperture_error_{radius:g}_kpc_dex"] = float(np.median(values))
    return output


def comparison_figure(
    sample: dict[str, np.ndarray],
    grid: dict,
    output_prefix: str,
) -> None:
    truth_log = sample["truth_log"]
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.0))
    line_styles = ("--", "-.", "-")
    colors = (OKABE_ITO[0], OKABE_ITO[4], OKABE_ITO[2])
    for law_number, law in enumerate(("fixed_gamma1p4", "smooth")):
        for compact_index, color, style in zip(COMPACT_INDICES, colors, line_styles):
            cell = grid[law][compact_index]
            residual = amplitude_pinned_residual(cell["prediction_log"], truth_log)
            model_density, density_radius = density_values(cell["prediction_log"])
            truth_density, _ = density_values(truth_log)
            label = rf"$n={compact_index:g}$"
            axes[0, law_number].plot(
                RADII,
                100.0 * np.median(10.0**residual - 1.0, axis=0),
                style,
                color=color,
                label=label,
            )
            axes[1, law_number].plot(
                density_radius,
                np.median(model_density - truth_density, axis=0),
                style,
                color=color,
                label=label,
            )
        axes[0, law_number].axhline(0.0, color="0.5", linewidth=0.8)
        axes[1, law_number].axhline(0.0, color="0.5", linewidth=0.8)
        axes[0, law_number].axvspan(5.0, 30.0, color="0.85", alpha=0.4)
        axes[1, law_number].axvspan(5.0, 30.0, color="0.85", alpha=0.4)
        axes[0, law_number].set_xscale("log")
        axes[1, law_number].set_xscale("log")
        axes[0, law_number].set_title(
            r"Fixed $\gamma=1.4$" if law_number == 0 else "Frozen smooth outer law"
        )
        axes[0, law_number].set_ylabel(r"Median amplitude-pinned CoG residual (\%)")
        axes[1, law_number].set_ylabel("Median density residual (dex)")
        axes[1, law_number].set_xlabel("Radius (kpc)")
        axes[0, law_number].legend(frameon=False, fontsize=8)

    x = np.arange(len(COMPACT_INDICES))
    width = 0.34
    for law_number, (law, label, color) in enumerate(
        (
            ("fixed_gamma1p4", r"fixed $\gamma=1.4$", OKABE_ITO[1]),
            ("smooth", "smooth outer law", OKABE_ITO[2]),
        )
    ):
        cog_values = []
        cog_intervals = []
        density_values_summary = []
        density_intervals = []
        for compact_index_number, compact_index in enumerate(COMPACT_INDICES):
            metrics = per_galaxy_metrics(grid[law][compact_index], truth_log)
            cog_metric = metrics["five_thirty_cog_rms_dex"]
            density_metric = metrics["five_thirty_density_rms_dex"]
            cog_values.append(np.median(cog_metric))
            cog_intervals.append(
                bootstrap_median_interval(
                    cog_metric,
                    seed=5600 + 100 * law_number + compact_index_number,
                )
            )
            density_values_summary.append(np.median(density_metric))
            density_intervals.append(
                bootstrap_median_interval(
                    density_metric,
                    seed=5650 + 100 * law_number + compact_index_number,
                )
            )
        offset = (law_number - 0.5) * width
        cog_values_array = np.asarray(cog_values)
        cog_intervals_array = np.asarray(cog_intervals)
        density_values_array = np.asarray(density_values_summary)
        density_intervals_array = np.asarray(density_intervals)
        axes[0, 2].bar(
            x + offset,
            cog_values_array,
            width,
            color=color,
            label=label,
            yerr=np.vstack(
                (
                    cog_values_array - cog_intervals_array[:, 0],
                    cog_intervals_array[:, 2] - cog_values_array,
                )
            ),
            capsize=2,
        )
        axes[1, 2].bar(
            x + offset,
            density_values_array,
            width,
            color=color,
            label=label,
            yerr=np.vstack(
                (
                    density_values_array - density_intervals_array[:, 0],
                    density_intervals_array[:, 2] - density_values_array,
                )
            ),
            capsize=2,
        )
    axes[0, 2].set_ylabel("Median 5--30 kpc pinned CoG RMS (dex)")
    axes[1, 2].set_ylabel("Median 5--30 kpc density RMS (dex)")
    axes[1, 2].set_xlabel("Globally fixed compact Sérsic index")
    for axis in axes[:, 2]:
        axis.set_xticks(x, [f"{value:g}" for value in COMPACT_INDICES])
    axes[0, 2].legend(frameon=False, fontsize=8)
    for panel, axis in zip("ABCDEF", axes.ravel()):
        axis.text(-0.13, 1.04, panel, transform=axis.transAxes, fontweight="bold")
    fig.suptitle("exp56 — compact-index representation closure")
    fig.tight_layout()
    save_fig(fig, FIGURE_DIRECTORY / f"{output_prefix}exp56_compact_shape")
    plt.close(fig)


def component_figure(
    sample: dict[str, np.ndarray],
    grid: dict,
    output_prefix: str,
) -> None:
    halo_mass = sample["features"][:, 0]
    galaxy = int(np.argsort(halo_mass)[len(halo_mass) // 2])
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.0))
    for column, compact_index in enumerate(COMPACT_INDICES):
        cell = grid["fixed_gamma1p4"][compact_index]
        compact, extended = mixed_component_cogs(
            cell["parameters"][galaxy],
            RADII,
            compact_index,
            FIXED_GAMMA,
        )
        total = compact + extended
        axes[0, column].plot(
            RADII, compact / total[-1], color=OKABE_ITO[0], label="compact"
        )
        axes[0, column].plot(
            RADII, extended / total[-1], color=OKABE_ITO[1], label="extended"
        )
        axes[0, column].plot(RADII, total / total[-1], color="black", label="total")
        compact_density, density_radius = density_values(np.log10(compact[None]))
        extended_density, _ = density_values(np.log10(extended[None]))
        total_density, _ = density_values(np.log10(total[None]))
        axes[1, column].plot(
            density_radius,
            compact_density[0],
            color=OKABE_ITO[0],
            label="compact",
        )
        axes[1, column].plot(
            density_radius,
            extended_density[0],
            color=OKABE_ITO[1],
            label="extended",
        )
        axes[1, column].plot(
            density_radius,
            total_density[0],
            color="black",
            label="total",
        )
        axes[0, column].set_title(rf"Fixed $n={compact_index:g}$")
        axes[0, column].set_xscale("log")
        axes[1, column].set_xscale("log")
        axes[0, column].axvspan(5.0, 30.0, color="0.85", alpha=0.4)
        axes[1, column].axvspan(5.0, 30.0, color="0.85", alpha=0.4)
        axes[1, column].set_xlabel("Radius (kpc)")
    axes[0, 0].set_ylabel(r"Component $M_*(<R)/M_{*,148}$")
    axes[1, 0].set_ylabel(r"$\log_{10}\Sigma_*$ ($M_\odot\,{\rm kpc}^{-2}$)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[1, 0].legend(frameon=False, fontsize=8)
    for panel, axis in zip("ABCDEF", axes.ravel()):
        axis.text(-0.13, 1.04, panel, transform=axis.transAxes, fontweight="bold")
    fig.suptitle("exp56 — component contributions for a median-halo-mass galaxy")
    fig.tight_layout()
    save_fig(fig, FIGURE_DIRECTORY / f"{output_prefix}exp56_compact_components")
    plt.close(fig)


def individual_cases_figure(
    sample: dict[str, np.ndarray],
    grid: dict,
    output_prefix: str,
) -> None:
    truth_log = sample["truth_log"]
    halo_mass = sample["features"][:, 0]
    order = np.argsort(halo_mass)
    galaxies = order[[0, len(order) // 2, -1]]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.0))
    colors = (OKABE_ITO[0], OKABE_ITO[4], OKABE_ITO[2])
    styles = ("--", "-.", "-")
    for column, galaxy in enumerate(galaxies):
        truth_normalized = truth_log[galaxy] - truth_log[galaxy, -1]
        axes[0, column].plot(
            RADII,
            truth_normalized,
            "o-",
            color="black",
            ms=3,
            label="TNG",
        )
        for compact_index, color, style in zip(COMPACT_INDICES, colors, styles):
            prediction = grid["fixed_gamma1p4"][compact_index]["prediction_log"][galaxy]
            prediction_normalized = prediction - prediction[-1]
            axes[0, column].plot(
                RADII,
                prediction_normalized,
                style,
                color=color,
                label=rf"$n={compact_index:g}$",
            )
            axes[1, column].plot(
                RADII,
                100.0 * (10.0 ** (prediction_normalized - truth_normalized) - 1.0),
                style,
                color=color,
            )
        axes[0, column].set_xscale("log")
        axes[1, column].set_xscale("log")
        axes[1, column].axhline(0.0, color="0.5", linewidth=0.8)
        axes[1, column].axvspan(5.0, 30.0, color="0.85", alpha=0.4)
        axes[0, column].set_title(rf"$\log M_{{\rm peak}}={halo_mass[galaxy]:.2f}$")
        axes[1, column].set_xlabel("Radius (kpc)")
    axes[0, 0].set_ylabel(r"$\log_{10}[M_*(<R)/M_{*,148}]$")
    axes[1, 0].set_ylabel(r"Amplitude-pinned residual (\%)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    for panel, axis in zip("ABCDEF", axes.ravel()):
        axis.text(-0.13, 1.04, panel, transform=axis.transAxes, fontweight="bold")
    fig.suptitle(r"exp56 — direct fits for three halo masses at fixed $\gamma=1.4$")
    fig.tight_layout()
    save_fig(fig, FIGURE_DIRECTORY / f"{output_prefix}exp56_compact_cases")
    plt.close(fig)


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


# %% Complete experiment path
def run_experiment(
    n_max: int,
    output_prefix: str,
    force: bool,
) -> dict:
    started = time.perf_counter()
    sample = load_sample(n_max)
    if n_max and len(sample["indices"]) != n_max:
        raise AssertionError(
            f"requested {n_max} galaxies, loaded {len(sample['indices'])}"
        )
    grid = fit_declared_grid(sample, output_prefix, force)
    selection = fold_clean_selection(grid, sample["truth_log"])
    rows = diagnostic_rows(sample["features"][:, 0])
    summaries = {}
    diagnostics = {}
    for law in ("fixed_gamma1p4", "smooth"):
        summaries[law] = {}
        diagnostics[law] = {}
        for compact_index in COMPACT_INDICES:
            cell = grid[law][compact_index]
            summaries[law][compact_index] = population_summary(
                cell, sample["truth_log"]
            )
            diagnostics[law][compact_index] = diagnostic_summary(
                cell,
                sample["truth_log"],
                compact_index,
                rows,
            )
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    comparison_figure(sample, grid, output_prefix)
    component_figure(sample, grid, output_prefix)
    individual_cases_figure(sample, grid, output_prefix)
    elapsed_seconds = time.perf_counter() - started
    result = {
        "status": "validation" if output_prefix else "full_representation_grid",
        "n_galaxy": len(sample["indices"]),
        "compact_indices": COMPACT_INDICES,
        "fixed_gamma": FIXED_GAMMA,
        "smooth_width_dex": SMOOTH_WIDTH_DEX,
        "smooth_transition_mass": grid["smooth_transition_mass"],
        "selection": selection,
        "summaries": summaries,
        "diagnostics": diagnostics,
        "diagnostic_galaxy_indices": sample["indices"][rows],
        "elapsed_seconds": elapsed_seconds,
    }
    if output_prefix:
        result["complete_path_passed"] = bool(
            len(sample["indices"]) == N_DEMO_GALAXY
            and elapsed_seconds < DEMO_TIME_LIMIT_SECONDS
            and all(
                summary["fraction_success"] == 1.0
                for law in summaries.values()
                for summary in law.values()
            )
            and all(
                max(values["maximum_exact_synthetic_absolute_error"]) < 1e-5
                for law in diagnostics.values()
                for values in law.values()
            )
        )
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    result_path = OUTPUT_DIRECTORY / f"{output_prefix}results.json"
    result_path.write_text(json.dumps(json_safe(result), indent=2))
    np.savez_compressed(
        OUTPUT_DIRECTORY / f"{output_prefix}predictions.npz",
        galaxy_indices=sample["indices"],
        truth_log=sample["truth_log"],
        halo_mass=sample["features"][:, 0],
        **{
            f"{law}_{compact_index_label(compact_index)}_prediction_log": grid[law][
                compact_index
            ]["prediction_log"]
            for law in ("fixed_gamma1p4", "smooth")
            for compact_index in COMPACT_INDICES
        },
    )
    manifest_directory = (
        OUTPUT_DIRECTORY if not output_prefix else OUTPUT_DIRECTORY / "demo_manifest"
    )
    write_manifest(
        manifest_directory,
        params={
            "script": Path(__file__).name,
            "status": result["status"],
            "n_galaxy": len(sample["indices"]),
            "compact_indices": COMPACT_INDICES,
            "fixed_gamma": FIXED_GAMMA,
            "smooth_width_dex": SMOOTH_WIDTH_DEX,
            "output_prefix": output_prefix,
        },
    )
    return result


def validate_mass_stratification() -> None:
    full_sample = load_sample(0)
    demo_sample = load_sample(N_DEMO_GALAXY)
    edges = np.quantile(
        full_sample["features"][:, 0],
        [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0],
    )
    counts = np.histogram(demo_sample["features"][:, 0], bins=edges)[0]
    if np.any(counts < 20):
        raise AssertionError(f"90-galaxy sample is not mass stratified: {counts}")


def demo() -> None:
    validate_mass_stratification()
    result = run_experiment(
        n_max=N_DEMO_GALAXY,
        output_prefix="demo_",
        force=True,
    )
    if not result["complete_path_passed"]:
        raise AssertionError(
            f"Exp56 demo failed its complete-path gate in "
            f"{result['elapsed_seconds']:.2f} s"
        )
    print(
        f"Exp56 complete 90-galaxy path passed in {result['elapsed_seconds']:.2f} s",
        flush=True,
    )


def full_run() -> None:
    demo_path = OUTPUT_DIRECTORY / "demo_results.json"
    if not demo_path.exists():
        raise RuntimeError("run the 90-galaxy demo before the full representation grid")
    demo_result = json.loads(demo_path.read_text())
    if not demo_result.get("complete_path_passed", False):
        raise RuntimeError("the saved 90-galaxy complete-path validation did not pass")
    result = run_experiment(n_max=0, output_prefix="", force=False)
    print(
        f"Exp56 full representation grid complete: {result['n_galaxy']} galaxies "
        f"in {result['elapsed_seconds']:.1f} s",
        flush=True,
    )


def regenerate_figures() -> None:
    sample = load_sample(0)
    grid = fit_declared_grid(sample, output_prefix="", force=False)
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    comparison_figure(sample, grid, output_prefix="")
    component_figure(sample, grid, output_prefix="")
    individual_cases_figure(sample, grid, output_prefix="")
    print("Exp56 full-sample figures regenerated from cached fits", flush=True)


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if command == "demo":
        demo()
    elif command == "run":
        full_run()
    elif command == "figures":
        regenerate_figures()
    else:
        raise SystemExit(f"unknown command {command!r}; use demo, run, or figures")
