# %%
"""Exp62 Stage 2 robustness: recovery, radial jackknifes, and profile scans.

Commands:

    uv run python experiments/exp62_cog_fit_atlas/diagnostics.py demo
    uv run python experiments/exp62_cog_fit_atlas/diagnostics.py full
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from families import (  # noqa: E402
    FAMILY_SPECS,
    cog_log,
    residual_vector,
)
from hongshao.plotting import OKABE_ITO, save_fig, set_style  # noqa: E402
from hongshao.provenance import write_manifest  # noqa: E402
from run import FIGDIR, MAX_WORKERS, OUTDIR, RADII, REDSHIFTS  # noqa: E402
from stage2 import ROLES, cell_path  # noqa: E402

set_style()

OBJECTIVE_NAMES = ("cog_log", "cog_absolute")
TESTS = ("synthetic", "alternating_radii", "rmin_5", "dogbox")
N_FULL_SAMPLE = 200


def _objective_parts(name: str):
    return name.removeprefix("cog_")


def diagnostic_rows(truth_log: np.ndarray, epochs: np.ndarray, mode: str):
    if mode == "demo":
        return np.arange(len(truth_log))
    output = []
    per_epoch = N_FULL_SAMPLE // len(REDSHIFTS)
    for epoch in range(len(REDSHIFTS)):
        selected = np.flatnonzero(epochs == epoch)
        order = selected[np.argsort(truth_log[selected, -1])]
        positions = np.linspace(0, len(order) - 1, per_epoch).round().astype(int)
        output.extend(order[positions].tolist())
    return np.asarray(output, int)


def _diagnostic_task(task):
    family, baseline_parameters, truth_log, objective_name, test = task
    residual = _objective_parts(objective_name)
    if test == "synthetic":
        fit_radius = RADII
        fit_truth = cog_log(family, baseline_parameters, RADII)
        method = "trf"
    elif test == "alternating_radii":
        keep = np.unique(np.append(np.arange(0, len(RADII), 2), len(RADII) - 1))
        fit_radius = RADII[keep]
        fit_truth = truth_log[keep]
        method = "trf"
    elif test == "rmin_5":
        keep = RADII >= 5.0
        fit_radius = RADII[keep]
        fit_truth = truth_log[keep]
        method = "trf"
    elif test == "dogbox":
        fit_radius = RADII
        fit_truth = truth_log
        method = "dogbox"
    else:
        raise ValueError(test)
    spec = FAMILY_SPECS[family]
    lower = np.asarray(spec.lower)
    upper = np.asarray(spec.upper)
    solution = least_squares(
        residual_vector,
        np.clip(baseline_parameters, lower + 1.0e-8, upper - 1.0e-8),
        args=(family, fit_radius, fit_truth, "cog", residual),
        bounds=(lower, upper),
        method=method,
        x_scale="jac",
        max_nfev=1800,
    )
    decoded = cog_log(family, solution.x, RADII)
    reference = (
        cog_log(family, baseline_parameters, RADII)
        if test == "synthetic"
        else truth_log
    )
    return (
        solution.x,
        decoded,
        float(np.sqrt(np.mean((decoded - reference) ** 2))),
        bool(solution.success),
    )


def run_tests(mode: str, force: bool = False):
    atlas = np.load(OUTDIR / mode / "atlas" / "predictions.npz")
    truth_log = np.asarray(atlas["truth_log"])
    epochs = np.asarray(atlas["epochs"])
    rows = diagnostic_rows(truth_log, epochs, mode)
    output_path = OUTDIR / mode / "stage2" / "diagnostics" / "robustness.npz"
    if output_path.exists() and not force:
        return np.load(output_path), truth_log, epochs
    payload = {"rows": rows}
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for role in ROLES:
            for objective_name in OBJECTIVE_NAMES:
                baseline = np.load(
                    cell_path(mode, role, "cog", _objective_parts(objective_name))
                )
                variants = np.asarray(baseline["variants"]).astype(str)[rows]
                baseline_parameters = np.asarray(baseline["parameters"])[rows]
                baseline_prediction = np.asarray(baseline["prediction_log"])[rows]
                parameter_range = np.asarray(
                    [
                        np.asarray(FAMILY_SPECS[family].upper)
                        - np.asarray(FAMILY_SPECS[family].lower)
                        for family in variants
                    ]
                )
                payload[f"{role}_{objective_name}_baseline_parameters"] = (
                    baseline_parameters
                )
                payload[f"{role}_{objective_name}_baseline_prediction"] = (
                    baseline_prediction
                )
                for test in TESTS:
                    tasks = [
                        (family, parameters, truth_log[row], objective_name, test)
                        for family, parameters, row in zip(
                            variants, baseline_parameters, rows
                        )
                    ]
                    results = list(
                        executor.map(
                            _diagnostic_task,
                            tasks,
                            chunksize=max(len(tasks) // (8 * MAX_WORKERS), 1),
                        )
                    )
                    parameters = np.asarray([item[0] for item in results])
                    prediction = np.asarray([item[1] for item in results])
                    payload[f"{role}_{objective_name}_{test}_parameters"] = parameters
                    payload[f"{role}_{objective_name}_{test}_prediction"] = prediction
                    payload[f"{role}_{objective_name}_{test}_reference_rms_dex"] = (
                        np.asarray([item[2] for item in results])
                    )
                    payload[f"{role}_{objective_name}_{test}_success"] = np.asarray(
                        [item[3] for item in results]
                    )
                    payload[
                        f"{role}_{objective_name}_{test}_scaled_parameter_change"
                    ] = np.max(
                        np.abs(parameters - baseline_parameters) / parameter_range,
                        axis=1,
                    )
                    payload[
                        f"{role}_{objective_name}_{test}_baseline_profile_change_dex"
                    ] = np.sqrt(
                        np.mean((prediction - baseline_prediction) ** 2, axis=1)
                    )
                    print(
                        f"  {role:<22} {objective_name:<12} {test:<18} "
                        f"median profile change "
                        f"{np.median(payload[f'{role}_{objective_name}_{test}_baseline_profile_change_dex']):.5f} dex",
                        flush=True,
                    )
    payload["wall_seconds"] = time.perf_counter() - start
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    return np.load(output_path), truth_log, epochs


def profile_scan(
    family: str,
    truth_log: np.ndarray,
    best_parameters: np.ndarray,
    objective_name: str,
):
    spec = FAMILY_SPECS[family]
    lower, upper = np.asarray(spec.lower), np.asarray(spec.upper)
    residual = _objective_parts(objective_name)
    grids, losses, decoded_rms = [], [], []
    for parameter_index in range(spec.n_parameter):
        span = upper[parameter_index] - lower[parameter_index]
        grid = np.linspace(
            max(lower[parameter_index], best_parameters[parameter_index] - 0.15 * span),
            min(upper[parameter_index], best_parameters[parameter_index] + 0.15 * span),
            15,
        )
        free = np.arange(spec.n_parameter) != parameter_index
        previous = best_parameters[free].copy()
        parameter_loss, parameter_cog = [], []
        for fixed_value in grid:

            def profiled_residual(free_parameters):
                parameters = np.empty(spec.n_parameter)
                parameters[free] = free_parameters
                parameters[parameter_index] = fixed_value
                return residual_vector(
                    parameters,
                    family,
                    RADII,
                    truth_log,
                    "cog",
                    residual,
                )

            starts = (best_parameters[free], previous)
            solutions = [
                least_squares(
                    profiled_residual,
                    np.clip(start, lower[free] + 1.0e-8, upper[free] - 1.0e-8),
                    bounds=(lower[free], upper[free]),
                    x_scale="jac",
                    max_nfev=2400,
                )
                for start in starts
            ]
            solution = min(solutions, key=lambda item: np.mean(item.fun**2))
            previous = solution.x
            parameters = np.empty(spec.n_parameter)
            parameters[free] = solution.x
            parameters[parameter_index] = fixed_value
            prediction = cog_log(family, parameters, RADII)
            parameter_loss.append(float(np.sqrt(np.mean(solution.fun**2))))
            parameter_cog.append(float(np.sqrt(np.mean((prediction - truth_log) ** 2))))
        grids.append(grid)
        losses.append(np.asarray(parameter_loss))
        decoded_rms.append(np.asarray(parameter_cog))
    return grids, losses, decoded_rms


def _scan_task(task):
    family, truth_log, parameters, objective_name = task
    return profile_scan(family, truth_log, parameters, objective_name)


def run_profile_scans(mode: str, truth_log: np.ndarray, force: bool = False):
    scan_dir = OUTDIR / mode / "stage2" / "diagnostics" / "profile_scans"
    scan_dir.mkdir(parents=True, exist_ok=True)
    definitions = {}
    missing = []
    for role in ROLES:
        for objective_name in OBJECTIVE_NAMES:
            baseline = np.load(
                cell_path(mode, role, "cog", _objective_parts(objective_name))
            )
            errors = np.asarray(baseline["cog_rms_dex"])
            galaxy = np.argsort(errors)[len(errors) // 2]
            family = str(np.asarray(baseline["variants"]).astype(str)[galaxy])
            parameters = np.asarray(baseline["parameters"])[galaxy]
            cache = scan_dir / f"{role}_{objective_name}.npz"
            definitions[(role, objective_name)] = (
                family,
                galaxy,
                parameters,
                cache,
            )
            if force or not cache.exists():
                missing.append((role, objective_name))
    if missing:
        tasks = [
            (
                definitions[key][0],
                truth_log[definitions[key][1]],
                definitions[key][2],
                key[1],
            )
            for key in missing
        ]
        with ProcessPoolExecutor(max_workers=min(MAX_WORKERS, len(tasks))) as executor:
            scans = list(executor.map(_scan_task, tasks))
        for key, (grids, losses, decoded) in zip(missing, scans):
            family, galaxy, _, cache = definitions[key]
            np.savez(
                cache,
                family=family,
                galaxy=galaxy,
                grids=np.asarray(grids, dtype=object),
                losses=np.asarray(losses, dtype=object),
                decoded_rms=np.asarray(decoded, dtype=object),
            )

    for role in ROLES:
        figure, axes = plt.subplots(
            len(OBJECTIVE_NAMES),
            FAMILY_SPECS["double_sersic_free"].n_parameter,
            figsize=(15.0, 5.6),
            squeeze=False,
        )
        for row, objective_name in enumerate(OBJECTIVE_NAMES):
            family, _, parameters, cache = definitions[(role, objective_name)]
            stored = np.load(cache, allow_pickle=True)
            grids = list(stored["grids"])
            losses = list(stored["losses"])
            names = FAMILY_SPECS[family].parameter_names
            for column in range(axes.shape[1]):
                axis = axes[row, column]
                if column >= len(names):
                    axis.set_visible(False)
                    continue
                loss = np.asarray(losses[column], float)
                axis.plot(grids[column], loss - np.min(loss), color=OKABE_ITO[4])
                axis.axvline(parameters[column], color="black", linestyle=":")
                axis.set_title(names[column].replace("_", " "), fontsize=8)
                if column == 0:
                    axis.set_ylabel(
                        f"{objective_name.replace('_', ' ')}\nprofiled loss rise"
                    )
        figure.suptitle(
            f"exp62 Stage 2 {mode} — {role.replace('_', ' ')} profiled parameters"
        )
        figure.tight_layout()
        save_fig(
            figure,
            FIGDIR / f"stage2_{mode}" / f"exp62_profile_scans_{role}",
        )
        plt.close(figure)


def summarize_and_figure(mode: str, store):
    summary = {}
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))
    colors = {
        "gompertz_log": OKABE_ITO[0],
        "sersic_moffat_fixed": OKABE_ITO[2],
        "double_sersic_free": OKABE_ITO[4],
    }
    markers = {"cog_log": "o", "cog_absolute": "s"}
    for test_index, test in enumerate(TESTS):
        axis = axes.ravel()[test_index]
        for role in ROLES:
            for objective_name in OBJECTIVE_NAMES:
                profile_change = np.asarray(
                    store[f"{role}_{objective_name}_{test}_baseline_profile_change_dex"]
                )
                parameter_change = np.asarray(
                    store[f"{role}_{objective_name}_{test}_scaled_parameter_change"]
                )
                key = f"{role}_{objective_name}_{test}"
                summary[key] = {
                    "median_baseline_profile_change_dex": float(
                        np.median(profile_change)
                    ),
                    "p90_baseline_profile_change_dex": float(
                        np.quantile(profile_change, 0.9)
                    ),
                    "median_maximum_scaled_parameter_change": float(
                        np.median(parameter_change)
                    ),
                    "p90_maximum_scaled_parameter_change": float(
                        np.quantile(parameter_change, 0.9)
                    ),
                    "success_fraction": float(
                        np.mean(store[f"{role}_{objective_name}_{test}_success"])
                    ),
                }
                axis.scatter(
                    max(float(np.median(parameter_change)), 1.0e-12),
                    max(float(np.median(profile_change)), 1.0e-12),
                    color=colors[role],
                    marker=markers[objective_name],
                    s=55,
                    label=f"{role.replace('_', ' ')}, {objective_name.replace('_', ' ')}",
                )
        axis.set(
            xlabel="Median maximum parameter change / allowed range",
            ylabel="Median decoded-profile change (dex)",
            title=test.replace("_", " "),
        )
        axis.set_xscale("log")
        axis.set_yscale("log")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles, labels, frameon=False, loc="outside lower center", ncol=3, fontsize=7
    )
    figure.suptitle(f"exp62 Stage 2 {mode} — parameter motion versus CoG stability")
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    save_fig(figure, FIGDIR / f"stage2_{mode}" / "exp62_robustness_summary")
    plt.close(figure)
    path = OUTDIR / mode / "stage2" / "diagnostics" / "summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def run(mode: str, force: bool = False):
    start = time.perf_counter()
    store, truth_log, _ = run_tests(mode, force=force)
    summary = summarize_and_figure(mode, store)
    run_profile_scans(mode, truth_log, force=force)
    elapsed = time.perf_counter() - start
    result = {
        "mode": mode,
        "n_diagnostic_profile": len(store["rows"]),
        "wall_seconds": elapsed,
        "subminute_gate_passed": bool(elapsed < 60.0) if mode == "demo" else None,
        "n_summary_cell": len(summary),
    }
    output_dir = OUTDIR / mode / "stage2" / "diagnostics"
    (output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    write_manifest(
        output_dir,
        {
            "experiment": "exp62_cog_fit_atlas",
            "stage": "2_diagnostics",
            "mode": mode,
            "objectives": OBJECTIVE_NAMES,
            "tests": TESTS,
        },
    )
    print(json.dumps(result, indent=2))
    if mode == "demo" and elapsed >= 60.0:
        raise RuntimeError(
            f"Stage 2 diagnostics demo took {elapsed:.2f} seconds; full run is forbidden"
        )


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if command not in ("demo", "full"):
        raise SystemExit(__doc__)
    run(command, force="--force" in sys.argv)
