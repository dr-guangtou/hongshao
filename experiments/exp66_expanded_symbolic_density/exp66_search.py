# %%
"""Exp66 expanded symbolic-density search.

Commands:

    uv run python experiments/exp66_expanded_symbolic_density/exp66_search.py mechanics
    uv run python experiments/exp66_expanded_symbolic_density/exp66_search.py gate --force
    uv run python experiments/exp66_expanded_symbolic_density/exp66_search.py broad
    uv run python experiments/exp66_expanded_symbolic_density/exp66_search.py run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from exp66_grammar import (  # noqa: E402
    RADII_KPC,
    ExpressionSpec,
    analytic_property_record,
    build_round_b_specs,
    cog_log,
    density_log_slope,
    fit_profile,
    round_a_specs,
    shell_log_density,
    surface_density,
    synthetic_parameters,
)
from hongshao.plotting import OKABE_ITO, save_fig, set_style  # noqa: E402
from hongshao.provenance import write_manifest  # noqa: E402
from hongshao.qa import enclosed_radius  # noqa: E402

set_style()
plt.rcParams["text.usetex"] = False

OUTDIR = HERE / "outputs"
FIGDIR = HERE / "figures"
DEFAULT_EXP62_ROOT = Path(
    "/Users/shuang/Dropbox/work/project/massive/hongshao_exp62_cog_fit_atlas"
)
EXP62_ROOT = Path(os.environ.get("HONGSHAO_EXP62_ROOT", DEFAULT_EXP62_ROOT))
EXP62_OUTDIR = EXP62_ROOT / "experiments" / "exp62_cog_fit_atlas" / "outputs"
MAX_WORKERS = min(int(os.environ.get("EXP66_WORKERS", "10")), os.cpu_count() or 1)
OBJECTIVES = ("log_cog", "absolute_cog", "log_shell")


def _source_path(mode: str, section: str) -> Path:
    paths = {
        "atlas": EXP62_OUTDIR / mode / "atlas" / "predictions.npz",
        "cubic": EXP62_OUTDIR / "stage5" / mode / "predictions.npz",
        "double": EXP62_OUTDIR / mode / "variants" / "double_sersic_free.npz",
    }
    return paths[section]


def load_references(mode: str) -> dict[str, np.ndarray]:
    """Load the aligned Exp62 measurements and paired model references."""
    paths = [_source_path(mode, key) for key in ("atlas", "cubic", "double")]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing Exp62 references: " + ", ".join(map(str, missing))
        )
    atlas = np.load(paths[0])
    cubic = np.load(paths[1])
    double = np.load(paths[2])
    if not np.array_equal(atlas["rows"], cubic["rows"]):
        raise ValueError("Exp62 reference rows do not match")
    if not np.array_equal(atlas["epochs"], cubic["epochs"]):
        raise ValueError("Exp62 reference epochs do not match")
    return {
        "rows": np.asarray(atlas["rows"], int),
        "epochs": np.asarray(atlas["epochs"], int),
        "redshifts": np.asarray(atlas["redshifts"], float),
        "truth_log": np.asarray(atlas["truth_log"], float),
        "halo_mass": np.asarray(atlas["halo_mass"], float),
        "prediction_double_sersic": np.asarray(
            atlas["prediction_double_sersic_free"], float
        ),
        "prediction_cubic_logit": np.asarray(cubic["prediction_logit_cubic5"], float),
        "prediction_pca3": np.asarray(atlas["prediction_pca3"], float),
        "jacobian_double_sersic": np.asarray(double["jacobian_min_ratio"], float),
        "boundary_double_sersic": np.asarray(double["boundary_distance"], float),
        "near_profile_double_sersic": np.asarray(
            double["near_profile_spread_dex"], float
        ),
        "jacobian_cubic_logit": np.asarray(
            cubic["jacobian_min_ratio_logit_cubic5"], float
        ),
        "boundary_cubic_logit": np.asarray(
            cubic["boundary_distance_logit_cubic5"], float
        ),
        "near_profile_cubic_logit": np.asarray(
            cubic["near_profile_spread_dex_logit_cubic5"], float
        ),
    }


def _galaxy_epoch_zero(
    reference: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    selected = reference["epochs"] == 0
    rows = reference["rows"][selected]
    masses = reference["halo_mass"][selected]
    if len(rows) != 3380 or len(np.unique(rows)) != 3380:
        raise ValueError("Exp66 requires 3,380 complete five-epoch galaxies")
    return rows, masses


def _indices_for_rows(reference: dict[str, np.ndarray], rows: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.isin(reference["rows"], rows))


def population_split(reference: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return the frozen mass-stratified galaxy-level Exp66 split."""
    rows, masses = _galaxy_epoch_zero(reference)
    order = np.argsort(masses, kind="stable")
    bins = np.array_split(order, 20)
    rng = np.random.default_rng(66)
    selected: dict[str, list[int]] = {
        "discovery": [],
        "selection": [],
        "validation": [],
    }
    for mass_bin in bins:
        shuffled = mass_bin.copy()
        rng.shuffle(shuffled)
        selected["discovery"].extend(rows[shuffled[:50]].tolist())
        selected["selection"].extend(rows[shuffled[50:100]].tolist())
        selected["validation"].extend(rows[shuffled[100:]].tolist())
    output = {
        name: _indices_for_rows(reference, np.asarray(values, int))
        for name, values in selected.items()
    }
    row_sets = [set(reference["rows"][indices].tolist()) for indices in output.values()]
    if any(
        first & second
        for first, second in (
            (row_sets[0], row_sets[1]),
            (row_sets[0], row_sets[2]),
            (row_sets[1], row_sets[2]),
        )
    ):
        raise RuntimeError("galaxy leakage across Exp66 population splits")
    return output


def broad_screen_indices(
    reference: dict[str, np.ndarray], discovery: np.ndarray
) -> np.ndarray:
    """Select the frozen 250-galaxy mass-stratified discovery screen."""
    epoch_zero = discovery[reference["epochs"][discovery] == 0]
    order = epoch_zero[np.argsort(reference["halo_mass"][epoch_zero], kind="stable")]
    bins = np.array_split(order, 20)
    rng = np.random.default_rng(660)
    rows: list[int] = []
    for index, mass_bin in enumerate(bins):
        shuffled = mass_bin.copy()
        rng.shuffle(shuffled)
        count = 13 if index < 10 else 12
        rows.extend(reference["rows"][shuffled[:count]].tolist())
    return _indices_for_rows(reference, np.asarray(rows, int))


def operational_gate_indices(
    reference: dict[str, np.ndarray], broad: np.ndarray
) -> np.ndarray:
    """Select 25 galaxies spanning the broad-screen halo-mass range."""
    epoch_zero = broad[reference["epochs"][broad] == 0]
    order = epoch_zero[np.argsort(reference["halo_mass"][epoch_zero], kind="stable")]
    positions = np.rint(np.linspace(0, len(order) - 1, 25)).astype(int)
    rows = reference["rows"][order[positions]]
    return _indices_for_rows(reference, rows)


def _shell_matrix(log_cog: np.ndarray) -> np.ndarray:
    return np.asarray([shell_log_density(profile, RADII_KPC) for profile in log_cog])


def profile_metrics(
    prediction_log: np.ndarray, truth_log: np.ndarray
) -> dict[str, np.ndarray]:
    """Return the common resolved-profile errors for each fitted profile."""
    prediction = np.asarray(prediction_log, float)
    truth = np.asarray(truth_log, float)
    residual = prediction - truth
    middle = (RADII_KPC >= 5.0) & (RADII_KPC <= 30.0)
    prediction_shell = _shell_matrix(prediction)
    truth_shell = _shell_matrix(truth)
    prediction_mass = 10.0**prediction
    truth_mass = 10.0**truth
    size_error = np.column_stack(
        [
            np.log10(
                [enclosed_radius(row, RADII_KPC, fraction) for row in prediction_mass]
            )
            - np.log10(
                [enclosed_radius(row, RADII_KPC, fraction) for row in truth_mass]
            )
            for fraction in (0.2, 0.5, 0.8, 0.9)
        ]
    )
    outer_shell_error = np.log10(
        np.clip(prediction_mass[:, -1] - prediction_mass[:, -2], 1.0, None)
    ) - np.log10(np.clip(truth_mass[:, -1] - truth_mass[:, -2], 1.0, None))
    return {
        "full_cog_rms_dex": np.sqrt(np.mean(residual**2, axis=1)),
        "middle_cog_rms_dex": np.sqrt(np.mean(residual[:, middle] ** 2, axis=1)),
        "shell_density_rms_dex": np.sqrt(
            np.mean((prediction_shell - truth_shell) ** 2, axis=1)
        ),
        "absolute_size_error_dex": np.abs(size_error),
        "absolute_outer_shell_error_dex": np.abs(outer_shell_error),
    }


def _fit_task(
    payload: tuple[ExpressionSpec, np.ndarray, str, int, int, int],
) -> dict[str, object]:
    spec, truth_log, objective, grid_size, n_starts, max_nfev = payload
    return fit_profile(
        spec,
        RADII_KPC,
        truth_log,
        objective=objective,
        grid_size=grid_size,
        n_starts=n_starts,
        max_nfev=max_nfev,
    )


def _fit_indexed_task(
    payload: tuple[str, int, ExpressionSpec, np.ndarray, str, int, int, int],
) -> tuple[str, int, dict[str, object]]:
    name, index, spec, truth_log, objective, grid_size, n_starts, max_nfev = payload
    return (
        name,
        index,
        _fit_task((spec, truth_log, objective, grid_size, n_starts, max_nfev)),
    )


def _pack_fits(fitted: list[dict[str, object]]) -> dict[str, np.ndarray]:
    """Pack scalar dictionaries from independent profile fits."""
    return {
        "parameters": np.asarray([item["parameters"] for item in fitted]),
        "prediction": np.asarray([item["prediction"] for item in fitted]),
        "objective_rms": np.asarray([item["objective_rms"] for item in fitted]),
        "cog_rms_dex": np.asarray([item["cog_rms_dex"] for item in fitted]),
        "success": np.asarray([item["success"] for item in fitted], bool),
        "nfev": np.asarray([item["nfev"] for item in fitted], int),
        "jacobian_min_ratio": np.asarray(
            [item["jacobian_min_ratio"] for item in fitted]
        ),
        "boundary_distance": np.asarray([item["boundary_distance"] for item in fitted]),
        "near_solution_count": np.asarray(
            [item["near_solution_count"] for item in fitted], int
        ),
        "near_parameter_spread": np.asarray(
            [item["near_parameter_spread"] for item in fitted]
        ),
        "near_profile_spread_dex": np.asarray(
            [item["near_profile_spread_dex"] for item in fitted]
        ),
    }


def fit_expression(
    spec: ExpressionSpec,
    truth_log: np.ndarray,
    *,
    objective: str,
    grid_size: int,
    n_starts: int,
    max_nfev: int,
    executor: ProcessPoolExecutor,
) -> dict[str, np.ndarray]:
    """Fit one expression across profiles and pack homogeneous arrays."""
    payloads = (
        (spec, profile, objective, grid_size, n_starts, max_nfev)
        for profile in truth_log
    )
    fitted = list(executor.map(_fit_task, payloads, chunksize=4))
    return _pack_fits(fitted)


def fit_expressions_flat(
    specs: dict[str, ExpressionSpec],
    truth_log: np.ndarray,
    *,
    objective: str,
    grid_size: int,
    n_starts: int,
    max_nfev: int,
    executor: ProcessPoolExecutor,
) -> dict[str, dict[str, np.ndarray]]:
    """Fit all expression/profile pairs in one queue to avoid idle workers."""
    payloads = (
        (name, index, spec, profile, objective, grid_size, n_starts, max_nfev)
        for name, spec in specs.items()
        for index, profile in enumerate(truth_log)
    )
    collected: dict[str, list[dict[str, object] | None]] = {
        name: [None] * len(truth_log) for name in specs
    }
    for name, index, fitted in executor.map(_fit_indexed_task, payloads, chunksize=4):
        collected[name][index] = fitted
    return {
        name: _pack_fits([item for item in fitted if item is not None])
        for name, fitted in collected.items()
    }


def _fit_expression_checkpointed(
    spec: ExpressionSpec,
    reference: dict[str, np.ndarray],
    indices: np.ndarray,
    stage: str,
    *,
    objective: str,
    grid_size: int,
    n_starts: int,
    max_nfev: int,
    executor: ProcessPoolExecutor,
    force: bool,
) -> tuple[dict[str, np.ndarray], float]:
    path = OUTDIR / stage / "expressions" / f"{spec.name}.npz"
    if path.exists() and not force:
        loaded = np.load(path)
        return {key: np.asarray(loaded[key]) for key in loaded.files}, 0.0
    started = time.perf_counter()
    result = fit_expression(
        spec,
        reference["truth_log"][indices],
        objective=objective,
        grid_size=grid_size,
        n_starts=n_starts,
        max_nfev=max_nfev,
        executor=executor,
    )
    elapsed = time.perf_counter() - started
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, indices=indices, elapsed_seconds=elapsed, **result)
    return result, elapsed


def _epoch_medians(values: np.ndarray, epochs: np.ndarray) -> list[float]:
    return [float(np.nanmedian(values[epochs == epoch])) for epoch in range(5)]


def candidate_record(
    spec: ExpressionSpec,
    result: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    indices: np.ndarray,
    boundary_limit: float,
    conditioning_fraction: float,
    near_profile_limit: float,
) -> dict[str, object]:
    """Summarize accuracy, conditioning, physical class, and eligibility."""
    truth = reference["truth_log"][indices]
    epochs = reference["epochs"][indices]
    metrics = profile_metrics(result["prediction"], truth)
    double_metrics = profile_metrics(
        reference["prediction_double_sersic"][indices], truth
    )
    full = _epoch_medians(metrics["full_cog_rms_dex"], epochs)
    double_full = _epoch_medians(double_metrics["full_cog_rms_dex"], epochs)
    ratios = np.asarray(full) / np.maximum(double_full, 1.0e-12)
    jacobian = _epoch_medians(result["jacobian_min_ratio"], epochs)
    cubic_jacobian = _epoch_medians(reference["jacobian_cubic_logit"][indices], epochs)
    conditioning_ratios = np.asarray(jacobian) / np.maximum(cubic_jacobian, 1.0e-16)
    success_fraction = float(np.mean(result["success"]))
    boundary_fraction = float(np.mean(result["boundary_distance"] < 0.01))
    maximum_near_profile = float(np.max(result["near_profile_spread_dex"]))
    eligible = bool(
        success_fraction >= 0.999
        and boundary_fraction < boundary_limit
        and np.count_nonzero(conditioning_ratios >= conditioning_fraction) >= 4
        and maximum_near_profile < near_profile_limit
    )
    return {
        "name": spec.name,
        "label": spec.label,
        "has_trigonometric_atom": spec.has_trigonometric_atom,
        "operation": spec.operation,
        "n_parameter": spec.n_parameter,
        "tree_size": spec.tree_size,
        "success_fraction": success_fraction,
        "boundary_fraction_within_one_percent": boundary_fraction,
        "maximum_near_profile_spread_dex": maximum_near_profile,
        "median_full_cog_rms_dex_by_epoch": full,
        "median_double_sersic_full_cog_rms_dex_by_epoch": double_full,
        "worst_epoch_full_cog_ratio_to_double_sersic": float(np.max(ratios)),
        "median_shell_density_rms_dex_by_epoch": _epoch_medians(
            metrics["shell_density_rms_dex"], epochs
        ),
        "median_jacobian_min_ratio_by_epoch": jacobian,
        "median_jacobian_ratio_to_cubic_by_epoch": conditioning_ratios.tolist(),
        "eligible": eligible,
        "analytic_properties": analytic_property_record(spec),
    }


def rank_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    """Apply the frozen lexicographic ranking, with eligible candidates first."""
    return sorted(
        records,
        key=lambda item: (
            not bool(item["eligible"]),
            float(item["worst_epoch_full_cog_ratio_to_double_sersic"]),
            max(item["median_shell_density_rms_dex_by_epoch"]),
            int(item["n_parameter"]),
            int(item["tree_size"]),
            str(item["name"]),
        ),
    )


def mechanics() -> dict[str, object]:
    """Run analytic, boundedness, resolution, and synthetic recovery checks."""
    specs = round_a_specs()
    representatives = [
        specs["constant"],
        specs["sigmoid_w0p7"],
        specs["harmonic_sine_w1"],
        specs["damped_cosine_l0p75_w2"],
        specs["wave_packet_sine_w0p5_w1"],
        specs["chirp_sine_k0p25_w2"],
        specs["log_harmonic_sine_w1"],
        specs["free_frequency_phase"],
    ]
    records = []
    for spec in representatives:
        parameters = synthetic_parameters(spec)
        truth = cog_log(spec, parameters, RADII_KPC, grid_size=2048)
        coarse = cog_log(spec, parameters, RADII_KPC, grid_size=512)
        fit = fit_profile(
            spec,
            RADII_KPC,
            truth,
            grid_size=1024,
            n_starts=2,
            max_nfev=400,
            synthetic_truth_parameters=parameters,
        )
        records.append(
            {
                "name": spec.name,
                "quadrature_maximum_difference_dex": float(
                    np.max(np.abs(coarse - truth))
                ),
                "synthetic_recovery_rms_dex": float(fit["cog_rms_dex"]),
                "synthetic_recovery_success": bool(fit["success"]),
                "analytic_properties": analytic_property_record(spec),
            }
        )
    passed = all(
        item["quadrature_maximum_difference_dex"] < 3.0e-4
        and item["synthetic_recovery_rms_dex"] < 3.0e-5
        and item["synthetic_recovery_success"]
        for item in records
    )
    return {"passed": passed, "expressions": records}


def _plot_frontier(records: list[dict[str, object]], stage: str) -> list[Path]:
    figure, axis = plt.subplots(figsize=(7.2, 5.0))
    for trig, marker, label, color in (
        (False, "o", "Non-trigonometric", OKABE_ITO[1]),
        (True, "^", "Contains trigonometric atom", OKABE_ITO[5]),
    ):
        selected = [item for item in records if item["has_trigonometric_atom"] == trig]
        axis.scatter(
            [max(item["median_full_cog_rms_dex_by_epoch"]) for item in selected],
            [min(item["median_jacobian_ratio_to_cubic_by_epoch"]) for item in selected],
            s=[18.0 + 7.0 * item["n_parameter"] for item in selected],
            marker=marker,
            color=color,
            alpha=0.78,
            label=label,
        )
    axis.axhline(
        0.25, color="0.35", lw=1.0, ls="--", label="Discovery conditioning gate"
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Worst-epoch median full-CoG RMS [dex]")
    axis.set_ylabel("Worst-epoch Jacobian ratio to cubic-logit")
    axis.set_title(f"Exp66 {stage}: accuracy and conditioning")
    axis.legend()
    figure.tight_layout()
    paths = save_fig(figure, FIGDIR / stage / "accuracy_conditioning_frontier")
    plt.close(figure)
    return paths


def _plot_direct_profiles(
    best_spec: ExpressionSpec,
    best_result: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    indices: np.ndarray,
    stage: str,
) -> list[Path]:
    truth = reference["truth_log"][indices]
    errors = profile_metrics(best_result["prediction"], truth)["full_cog_rms_dex"]
    order = np.argsort(errors)
    chosen = [order[0], order[len(order) // 2], order[-1]]
    labels = ("Best", "Typical", "Worst")
    figure, axes = plt.subplots(3, 3, figsize=(10.5, 8.5), sharex="col")
    for row, (local_index, label) in enumerate(zip(chosen, labels, strict=True)):
        parameters = best_result["parameters"][local_index]
        prediction = best_result["prediction"][local_index]
        dense_radius = np.geomspace(0.05, RADII_KPC[-1], 500)
        density = surface_density(best_spec, parameters, dense_radius, grid_size=1024)
        slope = density_log_slope(best_spec, parameters, dense_radius)
        axes[row, 0].plot(
            RADII_KPC, truth[local_index], "o", ms=3, label="Measured CoG"
        )
        axes[row, 0].plot(RADII_KPC, prediction, lw=1.5, label=best_spec.label)
        axes[row, 1].axhline(0.0, color="0.6", lw=0.8)
        axes[row, 1].plot(RADII_KPC, prediction - truth[local_index])
        axes[row, 2].plot(dense_radius, density / density[0], label="Density")
        twin = axes[row, 2].twinx()
        twin.plot(dense_radius, slope, color=OKABE_ITO[5], alpha=0.8, label="Slope")
        twin.set_ylabel("d ln Sigma / d ln R", color=OKABE_ITO[5])
        axes[row, 0].set_ylabel(f"{label}\nlog M(<R)")
        axes[row, 1].set_ylabel("Residual [dex]")
        axes[row, 2].set_ylabel("Sigma / Sigma(0)")
    for axis in axes[:, 0]:
        axis.set_xscale("log")
    for axis in axes[:, 1]:
        axis.set_xscale("log")
    for axis in axes[:, 2]:
        axis.set_xscale("log")
        axis.set_yscale("log")
    for axis in axes[-1]:
        axis.set_xlabel("Projected radius [kpc]")
    axes[0, 0].legend(loc="lower right")
    figure.suptitle(f"Exp66 {stage}: direct fits for {best_spec.label}")
    figure.tight_layout()
    paths = save_fig(figure, FIGDIR / stage / "direct_profile_fits")
    plt.close(figure)
    return paths


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def _write_summary(stage: str, summary: dict[str, object]) -> None:
    stage_dir = OUTDIR / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "summary.json").write_text(
        json.dumps(_json_ready(summary), indent=2) + "\n"
    )
    write_manifest(
        stage_dir,
        {
            "stage": stage,
            "max_workers": MAX_WORKERS,
            "source_exp62_outdir": str(EXP62_OUTDIR),
        },
    )


def run_stage(
    stage: str,
    specs: dict[str, ExpressionSpec],
    reference: dict[str, np.ndarray],
    indices: np.ndarray,
    *,
    n_starts: int,
    grid_size: int,
    max_nfev: int,
    boundary_limit: float,
    conditioning_fraction: float,
    near_profile_limit: float,
    force: bool = False,
    make_figures: bool = True,
    flat_schedule: bool = False,
) -> tuple[
    list[dict[str, object]], dict[str, dict[str, np.ndarray]], float, list[Path]
]:
    """Fit and checkpoint one frozen expression stage."""
    started = time.perf_counter()
    results: dict[str, dict[str, np.ndarray]] = {}
    expression_times = {}
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        if flat_schedule:
            results = fit_expressions_flat(
                specs,
                reference["truth_log"][indices],
                objective="log_cog",
                grid_size=grid_size,
                n_starts=n_starts,
                max_nfev=max_nfev,
                executor=executor,
            )
            for name, result in results.items():
                path = OUTDIR / stage / "expressions" / f"{name}.npz"
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, indices=indices, **result)
                expression_times[name] = None
                print(
                    f"[{stage}] {name}: median RMS "
                    f"{np.median(result['cog_rms_dex']):.5f} dex",
                    flush=True,
                )
        else:
            for number, (name, spec) in enumerate(specs.items(), start=1):
                result, elapsed = _fit_expression_checkpointed(
                    spec,
                    reference,
                    indices,
                    stage,
                    objective="log_cog",
                    grid_size=grid_size,
                    n_starts=n_starts,
                    max_nfev=max_nfev,
                    executor=executor,
                    force=force,
                )
                results[name] = result
                expression_times[name] = elapsed
                print(
                    f"[{stage}] {number}/{len(specs)} {name}: "
                    f"{elapsed:.2f} s, median RMS "
                    f"{np.median(result['cog_rms_dex']):.5f} dex",
                    flush=True,
                )
    records = rank_records(
        [
            candidate_record(
                spec,
                results[name],
                reference,
                indices,
                boundary_limit,
                conditioning_fraction,
                near_profile_limit,
            )
            for name, spec in specs.items()
        ]
    )
    elapsed = time.perf_counter() - started
    figure_paths: list[Path] = []
    if make_figures:
        figure_paths.extend(_plot_frontier(records, stage))
        best_name = str(records[0]["name"])
        figure_paths.extend(
            _plot_direct_profiles(
                specs[best_name], results[best_name], reference, indices, stage
            )
        )
    summary = {
        "stage": stage,
        "n_galaxy": int(len(np.unique(reference["rows"][indices]))),
        "n_profile": int(len(indices)),
        "n_expression": len(specs),
        "wall_seconds": elapsed,
        "expression_seconds": expression_times,
        "records": records,
        "figure_paths": [str(path) for path in figure_paths],
        "specifications": {name: asdict(spec) for name, spec in specs.items()},
    }
    _write_summary(stage, summary)
    return records, results, elapsed, figure_paths


def run_gate(force: bool = False) -> dict[str, object]:
    """Run the complete predeclared 125-profile operational gate."""
    gate_started = time.perf_counter()
    reference = load_references("full")
    split = population_split(reference)
    broad = broad_screen_indices(reference, split["discovery"])
    indices = operational_gate_indices(reference, broad)
    mechanics_result = mechanics()
    if not mechanics_result["passed"]:
        raise RuntimeError("Exp66 mechanics failed before the operational gate")
    records, _, fit_and_metric_seconds, figures = run_stage(
        "gate",
        round_a_specs(),
        reference,
        indices,
        n_starts=1,
        grid_size=128,
        max_nfev=180,
        boundary_limit=0.05,
        conditioning_fraction=0.25,
        near_profile_limit=0.003,
        force=force,
        flat_schedule=True,
    )
    elapsed = time.perf_counter() - gate_started
    passed = elapsed < 60.0
    summary = {
        "stage": "gate",
        "passed": passed,
        "operational_only": True,
        "wall_seconds": elapsed,
        "fit_and_metric_seconds": fit_and_metric_seconds,
        "fit_grid_size": 128,
        "mechanics_quadrature_grid_sizes": [512, 2048],
        "n_galaxy": 25,
        "n_profile": 125,
        "n_expression": len(records),
        "mechanics": mechanics_result,
        "best_record": records[0],
        "figure_paths": [str(path) for path in figures],
    }
    _write_summary("gate", summary)
    if not passed:
        raise RuntimeError(f"Exp66 operational gate took {elapsed:.2f} s, above 60 s")
    return summary


def run_broad(force: bool = False) -> dict[str, object]:
    """Run all Round-A expressions on 250 discovery galaxies at five epochs."""
    gate_path = OUTDIR / "gate" / "summary.json"
    if not gate_path.exists():
        run_gate(force=False)
    gate = json.loads(gate_path.read_text())
    if not gate["passed"]:
        raise RuntimeError("Exp66 broad screen is blocked by the operational gate")
    reference = load_references("full")
    split = population_split(reference)
    indices = broad_screen_indices(reference, split["discovery"])
    records, _, elapsed, figures = run_stage(
        "broad",
        round_a_specs(),
        reference,
        indices,
        n_starts=1,
        grid_size=256,
        max_nfev=220,
        boundary_limit=0.05,
        conditioning_fraction=0.25,
        near_profile_limit=0.003,
        force=force,
    )
    eligible = [record for record in records if record["eligible"]]
    survivors = [str(record["name"]) for record in eligible[:12]]
    summary = {
        "stage": "broad",
        "status": "complete" if survivors else "stopped_no_eligible_expression",
        "wall_seconds": elapsed,
        "n_galaxy": 250,
        "n_profile": 1250,
        "n_expression": len(records),
        "n_eligible": len(eligible),
        "round_a_survivors": survivors,
        "records": records,
        "figure_paths": [str(path) for path in figures],
    }
    _write_summary("broad", summary)
    return summary


def _load_stage_result(stage: str, name: str) -> dict[str, np.ndarray]:
    loaded = np.load(OUTDIR / stage / "expressions" / f"{name}.npz")
    return {key: np.asarray(loaded[key]) for key in loaded.files}


def run_complete_search(force: bool = False) -> dict[str, object]:
    """Run the predeclared discovery, selection, and untouched validation stages."""
    broad_summary = run_broad(force=force)
    survivors = broad_summary["round_a_survivors"]
    if not survivors:
        return broad_summary
    reference = load_references("full")
    split = population_split(reference)
    round_a = round_a_specs()
    survivor_specs = {name: round_a[name] for name in survivors}
    discovery_records, _, discovery_seconds, _ = run_stage(
        "discovery_round_a",
        survivor_specs,
        reference,
        split["discovery"],
        n_starts=2,
        grid_size=512,
        max_nfev=350,
        boundary_limit=0.05,
        conditioning_fraction=0.25,
        near_profile_limit=0.003,
        force=force,
    )
    eligible_round_a = [record for record in discovery_records if record["eligible"]]
    transitions = [
        round_a[str(record["name"])]
        for record in eligible_round_a
        if not record["has_trigonometric_atom"] and record["name"] != "constant"
    ][:2]
    trigonometric = [
        round_a[str(record["name"])]
        for record in eligible_round_a
        if record["has_trigonometric_atom"]
    ][:4]
    if len(transitions) < 2 or len(trigonometric) < 4:
        summary = {
            "stage": "discovery_round_a",
            "status": "stopped_insufficient_eligible_atoms",
            "n_eligible_transition": len(transitions),
            "n_eligible_trigonometric": len(trigonometric),
            "wall_seconds": discovery_seconds,
        }
        _write_summary("final", summary)
        return summary
    round_b = build_round_b_specs(transitions, trigonometric)
    round_b_records, _, round_b_seconds, _ = run_stage(
        "discovery_round_b",
        round_b,
        reference,
        split["discovery"],
        n_starts=2,
        grid_size=512,
        max_nfev=400,
        boundary_limit=0.05,
        conditioning_fraction=0.25,
        near_profile_limit=0.003,
        force=force,
    )
    all_specs = {**survivor_specs, **round_b}
    combined = rank_records(discovery_records + round_b_records)
    selection_names = [
        str(record["name"]) for record in combined if record["eligible"]
    ][:5]
    if not selection_names:
        summary = {"stage": "discovery", "status": "stopped_no_selection_candidate"}
        _write_summary("final", summary)
        return summary
    selection_specs = {name: all_specs[name] for name in selection_names}
    selection_records, _, selection_seconds, _ = run_stage(
        "selection",
        selection_specs,
        reference,
        split["selection"],
        n_starts=4,
        grid_size=512,
        max_nfev=500,
        boundary_limit=0.01,
        conditioning_fraction=0.5,
        near_profile_limit=0.001,
        force=force,
    )
    selection_eligible = [record for record in selection_records if record["eligible"]]
    if not selection_eligible:
        summary = {"stage": "selection", "status": "stopped_no_validation_finalist"}
        _write_summary("final", summary)
        return summary
    finalist_name = str(selection_eligible[0]["name"])
    finalist = selection_specs[finalist_name]
    objective_records = {}
    objective_seconds = {}
    for objective in OBJECTIVES:
        stage = f"validation_{objective}"
        started = time.perf_counter()
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            result, fit_seconds = _fit_expression_checkpointed(
                finalist,
                reference,
                split["validation"],
                stage,
                objective=objective,
                grid_size=1024,
                n_starts=4,
                max_nfev=600,
                executor=executor,
                force=force,
            )
        objective_seconds[objective] = time.perf_counter() - started
        objective_records[objective] = candidate_record(
            finalist,
            result,
            reference,
            split["validation"],
            0.01,
            0.5,
            0.001,
        )
        if objective == "log_cog":
            _plot_direct_profiles(
                finalist,
                result,
                reference,
                split["validation"],
                "validation",
            )
    summary = {
        "stage": "final",
        "status": "validation_complete",
        "finalist": finalist_name,
        "population_counts": {key: len(value) for key, value in split.items()},
        "stage_seconds": {
            "discovery_round_a": discovery_seconds,
            "discovery_round_b": round_b_seconds,
            "selection": selection_seconds,
            "validation_objectives": objective_seconds,
        },
        "objective_records": objective_records,
    }
    _write_summary("final", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("mechanics", "gate", "broad", "run"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.stage == "mechanics":
        result = mechanics()
    elif args.stage == "gate":
        result = run_gate(force=args.force)
    elif args.stage == "broad":
        result = run_broad(force=args.force)
    else:
        result = run_complete_search(force=args.force)
    print(json.dumps(_json_ready(result), indent=2))


if __name__ == "__main__":
    main()
