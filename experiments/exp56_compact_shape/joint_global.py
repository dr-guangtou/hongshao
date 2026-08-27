"""Exp56 Stage 1b: jointly profile one global Sersic n and Moffat gamma.

Commands
--------
demo
    Run the complete 25-cell path on 90 stratified galaxies and require less
    than one minute of measured wall time.
run
    Run the full sample only after the saved Stage 1b demo gate passes.
figures
    Regenerate the full-sample figures from cached Stage 1b fits.
"""

# %% Imports and frozen Stage 1b declaration
from __future__ import annotations

import json
import importlib.util
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

stage_1_spec = importlib.util.spec_from_file_location("exp56_stage_1", HERE / "run.py")
if stage_1_spec is None or stage_1_spec.loader is None:
    raise ImportError("could not load the Exp56 Stage 1 driver")
stage_1_module = importlib.util.module_from_spec(stage_1_spec)
sys.modules[stage_1_spec.name] = stage_1_module
stage_1_spec.loader.exec_module(stage_1_module)

from exp56_stage_1 import (  # noqa: E402
    FIGURE_DIRECTORY,
    FIXED_GAMMA,
    N_DEMO_GALAXY,
    OUTPUT_DIRECTORY,
    RADII,
    amplitude_pinned_residual,
    bootstrap_median_interval,
    density_values,
    diagnostic_rows,
    diagnostic_summary,
    fit_profile_cell,
    folds_of,
    json_safe,
    load_sample,
    mixed_component_cogs,
    per_galaxy_metrics,
    population_summary,
    size_values,
)
from hongshao.plotting import OKABE_ITO, save_fig  # noqa: E402
from hongshao.provenance import write_manifest  # noqa: E402

COMPACT_INDICES = np.array([0.750, 0.875, 1.000, 1.125, 1.250])
MOFFAT_INDICES = np.array([1.175, 1.250, 1.325, 1.400, 1.475])
REFERENCE_PAIR = (1.0, FIXED_GAMMA)
N_BOOTSTRAP = 1000
DEMO_TIME_LIMIT_SECONDS = 60.0
SEED = 5610


# %% Fits and inherited smooth-law reference
def pair_label(compact_index: float, moffat_index: float) -> str:
    compact_label = f"{compact_index:.3f}".replace(".", "p")
    moffat_label = f"{moffat_index:.3f}".replace(".", "p")
    return f"n{compact_label}_gamma{moffat_label}"


def checkpoint_path(
    artifact_prefix: str,
    compact_index: float,
    moffat_index: float,
) -> Path:
    return (
        OUTPUT_DIRECTORY
        / f"{artifact_prefix}profile_fits"
        / f"{pair_label(compact_index, moffat_index)}.npz"
    )


def fit_grid(
    sample: dict[str, np.ndarray],
    artifact_prefix: str,
    force: bool,
) -> dict[tuple[float, float], dict[str, np.ndarray]]:
    grid = {}
    for compact_index in COMPACT_INDICES:
        for moffat_index in MOFFAT_INDICES:
            pair = (float(compact_index), float(moffat_index))
            grid[pair] = fit_profile_cell(
                sample["truth_log"],
                sample["indices"],
                np.full(len(sample["indices"]), moffat_index),
                compact_index,
                checkpoint_path(artifact_prefix, *pair),
                force,
            )
    return grid


def load_smooth_reference(
    sample: dict[str, np.ndarray],
    demo: bool,
) -> dict[str, np.ndarray]:
    source_directory = OUTPUT_DIRECTORY / (
        "demo_profile_fits" if demo else "profile_fits"
    )
    output = {
        "gamma": np.empty(len(sample["indices"])),
        "parameters": np.empty((len(sample["indices"]), 4)),
        "prediction_log": np.empty_like(sample["truth_log"]),
        "profile_rms_dex": np.empty(len(sample["indices"])),
        "jacobian_min_ratio": np.empty(len(sample["indices"])),
        "boundary_distance": np.empty(len(sample["indices"])),
        "success": np.empty(len(sample["indices"]), dtype=bool),
    }
    for fold_number, test in enumerate(folds_of(len(sample["indices"]))):
        source = source_directory / f"smooth_fold{fold_number}_n1p00.npz"
        if not source.exists():
            raise RuntimeError(
                "Stage 1 smooth-reference fits are required before Stage 1b: "
                f"missing {source}"
            )
        saved = dict(np.load(source))
        if not np.array_equal(saved["galaxy_indices"], sample["indices"]):
            raise AssertionError(f"Stage 1 smooth reference does not match {source}")
        for name in output:
            output[name][test] = saved[name][test]
    return output


# %% Two-dimensional selection and conditional refinement
def metric_surface(
    grid: dict[tuple[float, float], dict[str, np.ndarray]],
    truth_log: np.ndarray,
    metric_name: str,
    rows: np.ndarray | None = None,
) -> np.ndarray:
    if rows is None:
        rows = np.arange(len(truth_log))
    surface = np.empty((len(COMPACT_INDICES), len(MOFFAT_INDICES)))
    for compact_number, compact_index in enumerate(COMPACT_INDICES):
        for moffat_number, moffat_index in enumerate(MOFFAT_INDICES):
            metrics = per_galaxy_metrics(
                grid[(float(compact_index), float(moffat_index))],
                truth_log,
            )
            values = metrics[metric_name][rows]
            surface[compact_number, moffat_number] = (
                np.mean(values) if metric_name == "near_boundary" else np.median(values)
            )
    return surface


def index_pair(surface: np.ndarray) -> tuple[int, int]:
    return tuple(
        int(value) for value in np.unravel_index(np.argmin(surface), surface.shape)
    )


def physical_pair(indices: tuple[int, int]) -> tuple[float, float]:
    return (
        float(COMPACT_INDICES[indices[0]]),
        float(MOFFAT_INDICES[indices[1]]),
    )


def fold_winners(
    grid: dict[tuple[float, float], dict[str, np.ndarray]],
    truth_log: np.ndarray,
) -> list[tuple[int, int]]:
    all_rows = np.arange(len(truth_log))
    output = []
    for test in folds_of(len(truth_log)):
        train = np.setdiff1d(all_rows, test)
        surface = metric_surface(
            grid,
            truth_log,
            "five_thirty_cog_rms_dex",
            train,
        )
        output.append(index_pair(surface))
    return output


def bootstrap_grid_selection(
    grid: dict[tuple[float, float], dict[str, np.ndarray]],
    truth_log: np.ndarray,
) -> np.ndarray:
    metric_stack = np.asarray(
        [
            per_galaxy_metrics(
                grid[(float(compact_index), float(moffat_index))],
                truth_log,
            )["five_thirty_cog_rms_dex"]
            for compact_index in COMPACT_INDICES
            for moffat_index in MOFFAT_INDICES
        ]
    )
    rng = np.random.default_rng(SEED)
    counts = np.zeros(metric_stack.shape[0], dtype=int)
    for _ in range(N_BOOTSTRAP):
        rows = rng.integers(0, len(truth_log), len(truth_log))
        counts[np.argmin(np.median(metric_stack[:, rows], axis=1))] += 1
    return counts.reshape(len(COMPACT_INDICES), len(MOFFAT_INDICES)) / N_BOOTSTRAP


def bootstrap_mean_interval(difference: np.ndarray, seed: int) -> list[float]:
    difference = np.asarray(difference, float)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(N_BOOTSTRAP)
    for sample_number in range(N_BOOTSTRAP):
        rows = rng.integers(0, len(difference), len(difference))
        bootstrap[sample_number] = np.mean(difference[rows])
    return np.quantile(bootstrap, [0.16, 0.50, 0.84]).tolist()


def quadratic_refinement(
    surface: np.ndarray,
    winner: tuple[int, int],
    winners_by_fold: list[tuple[int, int]],
) -> dict:
    compact_number, moffat_number = winner
    result = {
        "attempted": False,
        "valid": False,
        "reason": "",
        "grid_winner": physical_pair(winner),
    }
    if compact_number in (0, len(COMPACT_INDICES) - 1) or moffat_number in (
        0,
        len(MOFFAT_INDICES) - 1,
    ):
        result["reason"] = "the full-sample grid winner is on a grid boundary"
        return result
    if any(
        abs(fold_pair[0] - compact_number) > 1 or abs(fold_pair[1] - moffat_number) > 1
        for fold_pair in winners_by_fold
    ):
        result["reason"] = (
            "at least one training-fold winner is more than one grid step away"
        )
        return result

    result["attempted"] = True
    design = []
    values = []
    for compact_offset in (-1, 0, 1):
        for moffat_offset in (-1, 0, 1):
            x = float(compact_offset)
            y = float(moffat_offset)
            design.append([1.0, x, y, 0.5 * x**2, x * y, 0.5 * y**2])
            values.append(
                surface[
                    compact_number + compact_offset,
                    moffat_number + moffat_offset,
                ]
            )
    coefficients = np.linalg.lstsq(np.asarray(design), np.asarray(values), rcond=None)[
        0
    ]
    hessian = np.array(
        [
            [coefficients[3], coefficients[4]],
            [coefficients[4], coefficients[5]],
        ]
    )
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    result["quadratic_coefficients"] = coefficients
    result["hessian_eigenvalues"] = eigenvalues
    result["hessian_eigenvalue_ratio"] = float(eigenvalues[0] / eigenvalues[1])
    result["shallow_eigenvector_scaled_axes"] = eigenvectors[:, 0]
    if np.any(eigenvalues <= 0.0):
        result["reason"] = "the fitted local quadratic is not positive definite"
        return result

    stationary = -np.linalg.solve(hessian, coefficients[1:3])
    result["stationary_offset_grid_steps"] = stationary
    if np.any(np.abs(stationary) > 0.5):
        result["reason"] = (
            "the stationary point lies outside the central half-step rectangle"
        )
        return result
    candidate = (
        COMPACT_INDICES[compact_number]
        + stationary[0] * (COMPACT_INDICES[1] - COMPACT_INDICES[0]),
        MOFFAT_INDICES[moffat_number]
        + stationary[1] * (MOFFAT_INDICES[1] - MOFFAT_INDICES[0]),
    )
    result["candidate_pair"] = tuple(float(value) for value in candidate)
    result["valid"] = True
    result["reason"] = "valid positive-definite interior stationary point"
    return result


def comparison_with_reference(
    candidate: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    truth_log: np.ndarray,
    seed: int,
) -> dict:
    candidate_metrics = per_galaxy_metrics(candidate, truth_log)
    reference_metrics = per_galaxy_metrics(reference, truth_log)
    intervals = {}
    lower_is_better = (
        "five_thirty_cog_rms_dex",
        "full_cog_rms_dex",
        "five_thirty_density_rms_dex",
        "full_density_rms_dex",
    )
    for metric_number, metric_name in enumerate(lower_is_better):
        intervals[metric_name] = bootstrap_median_interval(
            candidate_metrics[metric_name] - reference_metrics[metric_name],
            seed + metric_number,
        )
    intervals["near_boundary"] = bootstrap_mean_interval(
        candidate_metrics["near_boundary"] - reference_metrics["near_boundary"],
        seed + 4,
    )
    intervals["log10_jacobian_min_ratio"] = bootstrap_median_interval(
        candidate_metrics["log10_jacobian_min_ratio"]
        - reference_metrics["log10_jacobian_min_ratio"],
        seed + 10,
    )
    for size_number, size_name in enumerate(("R50", "R80", "R90")):
        intervals[f"absolute_{size_name}_error"] = bootstrap_median_interval(
            candidate_metrics["absolute_size_error"][:, size_number]
            - reference_metrics["absolute_size_error"][:, size_number],
            seed + 20 + size_number,
        )
    safeguards = {
        metric_name: intervals[metric_name][0] <= 0.0
        for metric_name in (*lower_is_better[1:], "near_boundary")
    }
    safeguards["log10_jacobian_min_ratio"] = (
        intervals["log10_jacobian_min_ratio"][2] >= 0.0
    )
    for size_name in ("R50", "R80", "R90"):
        key = f"absolute_{size_name}_error"
        safeguards[key] = intervals[key][0] <= 0.0
    return {
        "bootstrap_candidate_minus_reference_16_50_84": intervals,
        "primary_improvement_resolved": intervals["five_thirty_cog_rms_dex"][2] < 0.0,
        "safeguards": safeguards,
        "all_safeguards_pass": all(safeguards.values()),
    }


def diagnostics_pass(
    selected: dict,
    reference: dict,
) -> dict:
    exact_pass = max(selected["maximum_exact_synthetic_absolute_error"]) < 1e-5
    jackknife_limits = np.maximum(
        0.05,
        2.0 * np.asarray(reference["maximum_radial_jackknife_absolute_change"]),
    )
    start_limits = np.maximum(
        1e-4,
        5.0 * np.asarray(reference["median_near_best_parameter_range"]),
    )
    jackknife_pass = np.all(
        np.asarray(selected["maximum_radial_jackknife_absolute_change"])
        <= jackknife_limits
    )
    starts_pass = np.all(
        np.asarray(selected["median_near_best_parameter_range"]) <= start_limits
    )
    return {
        "exact_recovery_pass": bool(exact_pass),
        "jackknife_limits": jackknife_limits,
        "radial_jackknife_pass": bool(jackknife_pass),
        "four_start_limits": start_limits,
        "four_start_pass": bool(starts_pass),
        "all_pass": bool(exact_pass and jackknife_pass and starts_pass),
    }


# %% Figures
def annotated_surface(
    axis: plt.Axes,
    values: np.ndarray,
    title: str,
    colorbar_label: str,
    fold_pairs: list[tuple[int, int]],
    selected_pair: tuple[float, float],
    value_format: str,
) -> None:
    image = axis.imshow(values, origin="lower", aspect="auto", cmap="cividis")
    low = float(np.min(values))
    span = max(float(np.max(values) - low), 1e-12)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            normalized = (values[row, column] - low) / span
            axis.text(
                column,
                row,
                format(values[row, column], value_format),
                color="white" if normalized < 0.55 else "black",
                ha="center",
                va="center",
                fontsize=7,
            )
    offsets = np.linspace(-0.16, 0.16, len(fold_pairs))
    for fold_number, ((row, column), offset) in enumerate(zip(fold_pairs, offsets)):
        axis.scatter(
            column + offset,
            row - 0.20,
            s=28,
            marker="o",
            facecolors="none",
            edgecolors=OKABE_ITO[fold_number],
            linewidths=1.2,
        )
    selected_column = np.interp(selected_pair[1], MOFFAT_INDICES, np.arange(5))
    selected_row = np.interp(selected_pair[0], COMPACT_INDICES, np.arange(5))
    axis.scatter(
        selected_column,
        selected_row,
        marker="*",
        s=130,
        facecolor="white",
        edgecolor="black",
        linewidth=1.0,
    )
    axis.set_xticks(np.arange(5), [f"{value:.3f}" for value in MOFFAT_INDICES])
    axis.set_yticks(np.arange(5), [f"{value:.3f}" for value in COMPACT_INDICES])
    axis.set_xlabel(r"Global Moffat $\gamma$")
    axis.set_ylabel(r"Global Sérsic $n$")
    axis.set_title(title)
    plt.colorbar(image, ax=axis, label=colorbar_label)


def surface_figure(
    grid: dict[tuple[float, float], dict[str, np.ndarray]],
    truth_log: np.ndarray,
    fold_pairs: list[tuple[int, int]],
    selected_pair: tuple[float, float],
    artifact_prefix: str,
) -> None:
    surfaces = (
        (
            metric_surface(grid, truth_log, "five_thirty_cog_rms_dex"),
            "5--30 kpc CoG accuracy",
            "Median RMS (dex)",
            ".4f",
        ),
        (
            metric_surface(grid, truth_log, "five_thirty_density_rms_dex"),
            "5--30 kpc density accuracy",
            "Median RMS (dex)",
            ".4f",
        ),
        (
            metric_surface(grid, truth_log, "log10_jacobian_min_ratio"),
            "Local parameter conditioning",
            r"Median $\log_{10}(s_{\min}/s_{\max})$",
            ".2f",
        ),
        (
            metric_surface(grid, truth_log, "near_boundary"),
            "Optimizer boundary incidence",
            "Boundary fraction",
            ".3f",
        ),
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.0))
    for panel, axis, arguments in zip("ABCD", axes.ravel(), surfaces):
        annotated_surface(
            axis,
            *arguments[:1],
            arguments[1],
            arguments[2],
            fold_pairs,
            selected_pair,
            arguments[3],
        )
        axis.text(-0.13, 1.04, panel, transform=axis.transAxes, fontweight="bold")
    fig.suptitle("exp56 Stage 1b — profiled global-shape surface")
    fig.tight_layout()
    save_fig(fig, FIGURE_DIRECTORY / f"{artifact_prefix}exp56_joint_surface")
    plt.close(fig)


def radial_figure(
    sample: dict[str, np.ndarray],
    selected: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    smooth: dict[str, np.ndarray],
    selected_pair: tuple[float, float],
    artifact_prefix: str,
) -> None:
    truth_log = sample["truth_log"]
    models = (
        (reference, r"fixed $n=1,\ \gamma=1.4$", OKABE_ITO[1], "--"),
        (
            selected,
            rf"selected $n={selected_pair[0]:.3f},\ \gamma={selected_pair[1]:.3f}$",
            OKABE_ITO[0],
            "-",
        ),
        (smooth, r"smooth $\gamma(\log M_{\rm peak})$", OKABE_ITO[2], "-."),
    )
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.0))
    truth_density, density_radius = density_values(truth_log)
    for model, label, color, style in models:
        residual = amplitude_pinned_residual(model["prediction_log"], truth_log)
        model_density, _ = density_values(model["prediction_log"])
        axes[0, 0].plot(
            RADII,
            100.0 * np.median(10.0**residual - 1.0, axis=0),
            style,
            color=color,
            label=label,
        )
        axes[0, 1].plot(
            density_radius,
            np.median(model_density - truth_density, axis=0),
            style,
            color=color,
            label=label,
        )
    for axis in axes[0, :2]:
        axis.axhline(0.0, color="0.5", linewidth=0.8)
        axis.axvspan(5.0, 30.0, color="0.85", alpha=0.4)
        axis.set_xscale("log")
        axis.set_xlabel("Radius (kpc)")
    axes[0, 0].set_ylabel("Median amplitude-pinned CoG residual (%)")
    axes[0, 1].set_ylabel("Median density residual (dex)")
    axes[0, 0].legend(frameon=False, fontsize=7)

    truth_sizes = size_values(truth_log)
    x = np.arange(3)
    width = 0.23
    for model_number, (model, label, color, _) in enumerate(models):
        offsets = size_values(model["prediction_log"]) - truth_sizes
        medians = np.median(offsets, axis=0)
        intervals = np.asarray(
            [
                bootstrap_median_interval(offsets[:, column], SEED + 100 + column)
                for column in range(3)
            ]
        )
        axes[0, 2].bar(
            x + (model_number - 1) * width,
            medians,
            width,
            color=color,
            label=label,
            yerr=np.vstack((medians - intervals[:, 0], intervals[:, 2] - medians)),
            capsize=2,
        )
    axes[0, 2].axhline(0.0, color="0.5", linewidth=0.8)
    axes[0, 2].set_xticks(x, ("R50", "R80", "R90"))
    axes[0, 2].set_ylabel("Median model-minus-TNG size (dex)")

    galaxy = int(np.argsort(sample["features"][:, 0])[len(truth_log) // 2])
    parameters = selected["parameters"][galaxy]
    compact, extended = mixed_component_cogs(
        parameters, RADII, selected_pair[0], selected_pair[1]
    )
    total = compact + extended
    for values, label, color in (
        (compact, "compact", OKABE_ITO[0]),
        (extended, "extended", OKABE_ITO[1]),
        (total, "total", "black"),
    ):
        axes[1, 0].plot(RADII, values / total[-1], color=color, label=label)
        density, component_radius = density_values(np.log10(values[None]))
        axes[1, 1].plot(component_radius, density[0], color=color, label=label)
    for axis in axes[1, :2]:
        axis.set_xscale("log")
        axis.axvspan(5.0, 30.0, color="0.85", alpha=0.4)
        axis.set_xlabel("Radius (kpc)")
    axes[1, 0].set_ylabel(r"Component $M_*(<R)/M_{*,148}$")
    axes[1, 1].set_ylabel(r"$\log_{10}\Sigma_*$ ($M_\odot\,{\rm kpc}^{-2}$)")
    axes[1, 0].legend(frameon=False, fontsize=8)

    selected_metric = per_galaxy_metrics(selected, truth_log)["five_thirty_cog_rms_dex"]
    fixed_metric = per_galaxy_metrics(reference, truth_log)["five_thirty_cog_rms_dex"]
    smooth_metric = per_galaxy_metrics(smooth, truth_log)["five_thirty_cog_rms_dex"]
    axes[1, 2].hist(
        selected_metric - fixed_metric,
        bins=40,
        histtype="step",
        color=OKABE_ITO[1],
        label=r"selected minus fixed $\gamma=1.4$",
    )
    axes[1, 2].hist(
        selected_metric - smooth_metric,
        bins=40,
        histtype="step",
        color=OKABE_ITO[2],
        label="selected minus smooth law",
    )
    axes[1, 2].axvline(0.0, color="black", linewidth=0.8)
    axes[1, 2].set_xlabel("Paired 5--30 kpc CoG RMS difference (dex)")
    axes[1, 2].set_ylabel("Number of galaxies")
    axes[1, 2].legend(frameon=False, fontsize=7)
    for panel, axis in zip("ABCDEF", axes.ravel()):
        axis.text(-0.13, 1.04, panel, transform=axis.transAxes, fontweight="bold")
    fig.suptitle("exp56 Stage 1b — selected global shape and direct safeguards")
    fig.tight_layout()
    save_fig(fig, FIGURE_DIRECTORY / f"{artifact_prefix}exp56_joint_radial")
    plt.close(fig)


def cases_figure(
    sample: dict[str, np.ndarray],
    selected: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    smooth: dict[str, np.ndarray],
    selected_pair: tuple[float, float],
    artifact_prefix: str,
) -> None:
    truth_log = sample["truth_log"]
    halo_mass = sample["features"][:, 0]
    order = np.argsort(halo_mass)
    galaxies = order[[0, len(order) // 2, -1]]
    models = (
        (reference, r"fixed $n=1,\ \gamma=1.4$", OKABE_ITO[1], "--"),
        (
            selected,
            rf"selected $n={selected_pair[0]:.3f},\ \gamma={selected_pair[1]:.3f}$",
            OKABE_ITO[0],
            "-",
        ),
        (smooth, "smooth outer law", OKABE_ITO[2], "-."),
    )
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.0))
    for column, galaxy in enumerate(galaxies):
        truth_normalized = truth_log[galaxy] - truth_log[galaxy, -1]
        axes[0, column].plot(
            RADII, truth_normalized, "o-", color="black", ms=3, label="TNG"
        )
        for model, label, color, style in models:
            prediction = model["prediction_log"][galaxy]
            prediction_normalized = prediction - prediction[-1]
            axes[0, column].plot(
                RADII,
                prediction_normalized,
                style,
                color=color,
                label=label,
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
    axes[1, 0].set_ylabel("Amplitude-pinned residual (%)")
    axes[0, 0].legend(frameon=False, fontsize=7)
    for panel, axis in zip("ABCDEF", axes.ravel()):
        axis.text(-0.13, 1.04, panel, transform=axis.transAxes, fontweight="bold")
    fig.suptitle("exp56 Stage 1b — direct fits for three halo masses")
    fig.tight_layout()
    save_fig(fig, FIGURE_DIRECTORY / f"{artifact_prefix}exp56_joint_cases")
    plt.close(fig)


# %% Complete Stage 1b path
def run_experiment(
    n_max: int,
    artifact_prefix: str,
    force: bool,
    recorded_elapsed_seconds: float | None = None,
) -> dict:
    started = time.perf_counter()
    sample = load_sample(n_max)
    if n_max and len(sample["indices"]) != n_max:
        raise AssertionError(
            f"requested {n_max} galaxies, loaded {len(sample['indices'])}"
        )
    grid = fit_grid(sample, artifact_prefix, force)
    truth_log = sample["truth_log"]
    primary_surface = metric_surface(grid, truth_log, "five_thirty_cog_rms_dex")
    winner_indices = index_pair(primary_surface)
    winners_by_fold = fold_winners(grid, truth_log)
    selection_frequency = bootstrap_grid_selection(grid, truth_log)
    refinement = quadratic_refinement(primary_surface, winner_indices, winners_by_fold)
    grid_winner_pair = physical_pair(winner_indices)
    grid_winner = grid[grid_winner_pair]
    selected_pair = grid_winner_pair
    selected = grid_winner
    if refinement["valid"]:
        candidate_pair = tuple(refinement["candidate_pair"])
        candidate = fit_profile_cell(
            truth_log,
            sample["indices"],
            np.full(len(truth_log), candidate_pair[1]),
            candidate_pair[0],
            checkpoint_path(artifact_prefix, *candidate_pair),
            force,
        )
        candidate_objective = float(
            np.median(
                per_galaxy_metrics(candidate, truth_log)["five_thirty_cog_rms_dex"]
            )
        )
        refinement["direct_candidate_objective_dex"] = candidate_objective
        refinement["grid_winner_objective_dex"] = float(primary_surface[winner_indices])
        refinement["direct_candidate_improves_grid"] = (
            candidate_objective < primary_surface[winner_indices]
        )
        if refinement["direct_candidate_improves_grid"]:
            selected_pair = candidate_pair
            selected = candidate
        else:
            refinement["valid"] = False
            refinement["reason"] = (
                "the directly evaluated candidate does not improve the grid winner"
            )

    reference = grid[REFERENCE_PAIR]
    smooth = load_smooth_reference(sample, demo=bool(n_max))
    fixed_comparison = comparison_with_reference(
        selected, reference, truth_log, SEED + 1000
    )
    smooth_comparison = comparison_with_reference(
        selected, smooth, truth_log, SEED + 2000
    )
    rows = diagnostic_rows(sample["features"][:, 0])
    selected_diagnostics = diagnostic_summary(
        selected, truth_log, selected_pair[0], rows
    )
    reference_diagnostics = diagnostic_summary(reference, truth_log, 1.0, rows)
    numerical_safeguards = diagnostics_pass(selected_diagnostics, reference_diagnostics)
    selected_summary = population_summary(selected, truth_log)
    reference_summary = population_summary(reference, truth_log)
    smooth_summary = population_summary(smooth, truth_log)
    r90_not_worse_than_smooth = abs(
        selected_summary["r90_slope"] - selected_summary["truth_r90_slope"]
    ) <= abs(smooth_summary["r90_slope"] - smooth_summary["truth_r90_slope"])
    adopt_over_fixed = bool(
        fixed_comparison["primary_improvement_resolved"]
        and fixed_comparison["all_safeguards_pass"]
        and numerical_safeguards["all_pass"]
    )
    replace_smooth = bool(
        smooth_comparison["primary_improvement_resolved"]
        and smooth_comparison["all_safeguards_pass"]
        and r90_not_worse_than_smooth
        and numerical_safeguards["all_pass"]
    )

    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    surface_figure(
        grid,
        truth_log,
        winners_by_fold,
        selected_pair,
        artifact_prefix,
    )
    radial_figure(
        sample,
        selected,
        reference,
        smooth,
        selected_pair,
        artifact_prefix,
    )
    cases_figure(
        sample,
        selected,
        reference,
        smooth,
        selected_pair,
        artifact_prefix,
    )

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_DIRECTORY / f"{artifact_prefix}predictions.npz",
        galaxy_indices=sample["indices"],
        truth_log=truth_log,
        halo_mass=sample["features"][:, 0],
        selected_pair=np.asarray(selected_pair),
        selected_prediction_log=selected["prediction_log"],
        reference_prediction_log=reference["prediction_log"],
        smooth_prediction_log=smooth["prediction_log"],
    )
    write_manifest(
        OUTPUT_DIRECTORY / f"{artifact_prefix}manifest",
        params={
            "script": Path(__file__).name,
            "n_galaxy": len(truth_log),
            "compact_indices": COMPACT_INDICES.tolist(),
            "moffat_indices": MOFFAT_INDICES.tolist(),
            "selected_pair": selected_pair,
            "artifact_prefix": artifact_prefix,
        },
    )
    measured_elapsed_seconds = time.perf_counter() - started
    elapsed_seconds = (
        measured_elapsed_seconds
        if recorded_elapsed_seconds is None
        else recorded_elapsed_seconds
    )
    grid_summaries = []
    for compact_index in COMPACT_INDICES:
        for moffat_index in MOFFAT_INDICES:
            pair = (float(compact_index), float(moffat_index))
            grid_summaries.append(
                {
                    "compact_index": pair[0],
                    "moffat_index": pair[1],
                    "summary": population_summary(grid[pair], truth_log),
                }
            )
    result = {
        "status": "joint_global_validation" if n_max else "joint_global_full",
        "n_galaxy": len(truth_log),
        "compact_indices": COMPACT_INDICES,
        "moffat_indices": MOFFAT_INDICES,
        "primary_surface_dex": primary_surface,
        "grid_winner": grid_winner_pair,
        "fold_winners": [physical_pair(pair) for pair in winners_by_fold],
        "bootstrap_grid_selection_frequency": selection_frequency,
        "refinement": refinement,
        "selected_pair": selected_pair,
        "fixed_reference_pair": REFERENCE_PAIR,
        "comparison_with_fixed_reference": fixed_comparison,
        "comparison_with_smooth_reference": smooth_comparison,
        "selected_summary": selected_summary,
        "fixed_reference_summary": reference_summary,
        "smooth_reference_summary": smooth_summary,
        "grid_summaries": grid_summaries,
        "diagnostics": {
            "selected": selected_diagnostics,
            "fixed_reference": reference_diagnostics,
            "safeguards": numerical_safeguards,
        },
        "r90_not_worse_than_smooth": r90_not_worse_than_smooth,
        "adopt_as_single_slope_baseline": adopt_over_fixed,
        "replace_frozen_smooth_law": replace_smooth,
        "elapsed_seconds": elapsed_seconds,
    }
    if recorded_elapsed_seconds is not None:
        result["cached_reanalysis_seconds"] = measured_elapsed_seconds
    if n_max:
        result["complete_path_passed"] = bool(
            len(truth_log) == N_DEMO_GALAXY
            and elapsed_seconds < DEMO_TIME_LIMIT_SECONDS
            and all(np.all(cell["success"]) for cell in grid.values())
            and max(selected_diagnostics["maximum_exact_synthetic_absolute_error"])
            < 1e-5
            and max(reference_diagnostics["maximum_exact_synthetic_absolute_error"])
            < 1e-5
        )
    (OUTPUT_DIRECTORY / f"{artifact_prefix}results.json").write_text(
        json.dumps(json_safe(result), indent=2)
    )
    return result


def demo() -> None:
    result = run_experiment(
        n_max=N_DEMO_GALAXY,
        artifact_prefix="joint_demo_",
        force=True,
    )
    if not result["complete_path_passed"]:
        raise AssertionError(
            "Exp56 Stage 1b demo failed its complete-path gate in "
            f"{result['elapsed_seconds']:.2f} s"
        )
    print(
        "Exp56 Stage 1b complete 90-galaxy path passed in "
        f"{result['elapsed_seconds']:.2f} s",
        flush=True,
    )


def full_run() -> None:
    demo_path = OUTPUT_DIRECTORY / "joint_demo_results.json"
    if not demo_path.exists():
        raise RuntimeError("run the Stage 1b 90-galaxy demo before the full grid")
    demo_result = json.loads(demo_path.read_text())
    if not demo_result.get("complete_path_passed", False):
        raise RuntimeError("the saved Stage 1b complete-path validation did not pass")
    result = run_experiment(n_max=0, artifact_prefix="joint_", force=False)
    print(
        f"Exp56 Stage 1b full grid complete: {result['n_galaxy']} galaxies "
        f"in {result['elapsed_seconds']:.1f} s",
        flush=True,
    )


def regenerate_figures() -> None:
    sample = load_sample(0)
    grid = fit_grid(sample, "joint_", force=False)
    result_path = OUTPUT_DIRECTORY / "joint_results.json"
    if not result_path.exists():
        raise RuntimeError("run the full Stage 1b grid before regenerating figures")
    result = json.loads(result_path.read_text())
    selected_pair = tuple(result["selected_pair"])
    selected_path = checkpoint_path("joint_", *selected_pair)
    if not selected_path.exists():
        raise RuntimeError(f"missing selected Stage 1b checkpoint {selected_path}")
    selected = dict(np.load(selected_path))
    reference = grid[REFERENCE_PAIR]
    smooth = load_smooth_reference(sample, demo=False)
    fold_pairs = [
        (
            int(np.where(np.isclose(COMPACT_INDICES, pair[0]))[0][0]),
            int(np.where(np.isclose(MOFFAT_INDICES, pair[1]))[0][0]),
        )
        for pair in result["fold_winners"]
    ]
    surface_figure(grid, sample["truth_log"], fold_pairs, selected_pair, "joint_")
    radial_figure(
        sample, selected, reference, smooth, selected_pair, artifact_prefix="joint_"
    )
    cases_figure(
        sample, selected, reference, smooth, selected_pair, artifact_prefix="joint_"
    )
    print("Exp56 Stage 1b full-sample figures regenerated from cached fits", flush=True)


def reanalyze_full() -> None:
    result_path = OUTPUT_DIRECTORY / "joint_results.json"
    if not result_path.exists():
        raise RuntimeError("run the full Stage 1b grid before reanalyzing")
    recorded_elapsed_seconds = float(
        json.loads(result_path.read_text())["elapsed_seconds"]
    )
    result = run_experiment(
        n_max=0,
        artifact_prefix="joint_",
        force=False,
        recorded_elapsed_seconds=recorded_elapsed_seconds,
    )
    print(
        "Exp56 Stage 1b cached reanalysis complete in "
        f"{result['cached_reanalysis_seconds']:.1f} s; preserved full-run time "
        f"{result['elapsed_seconds']:.1f} s",
        flush=True,
    )


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if command == "demo":
        demo()
    elif command == "run":
        full_run()
    elif command == "figures":
        regenerate_figures()
    elif command == "reanalyze":
        reanalyze_full()
    else:
        raise SystemExit(
            f"unknown command {command!r}; use demo, run, figures, or reanalyze"
        )
