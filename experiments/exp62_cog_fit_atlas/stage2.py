# %%
"""Exp62 Stage 2: objective sensitivity for the three advanced family roles.

Commands:

    uv run python experiments/exp62_cog_fit_atlas/stage2.py demo
    uv run python experiments/exp62_cog_fit_atlas/stage2.py full
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from families import FAMILY_SPECS, fit_cog  # noqa: E402
from hongshao.plotting import OKABE_ITO, save_fig, set_style  # noqa: E402
from hongshao.provenance import write_manifest  # noqa: E402
from run import (  # noqa: E402
    FIGDIR,
    MAX_WORKERS,
    OUTDIR,
    RADII,
    REDSHIFTS,
    profile_metrics,
)

set_style()

ROLES = ("gompertz_log", "sersic_moffat_fixed", "double_sersic_free")
OBJECTIVES = tuple(
    (quantity, residual)
    for quantity in ("cog", "shells", "density")
    for residual in ("log", "absolute")
)


def objective_name(quantity: str, residual: str) -> str:
    return f"{quantity}_{residual}"


def _fit_task(task):
    family, truth_log, quantity, residual, radius = task
    result = fit_cog(
        family,
        np.asarray(radius),
        np.asarray(truth_log),
        quantity=quantity,
        residual=residual,
    )
    return (
        result["parameters"],
        result["prediction"],
        result["cog_rms_dex"],
        result["success"],
        result["nfev"],
        result["jacobian_min_ratio"],
        result["boundary_distance"],
        result["near_solution_count"],
        result["near_parameter_spread"],
        result["near_profile_spread_dex"],
    )


def load_stage1(mode: str):
    store = np.load(OUTDIR / mode / "atlas" / "predictions.npz")
    truth_log = np.asarray(store["truth_log"])
    variants = {
        role: np.asarray(store[f"selected_variant_{role}"]).astype(str)
        for role in ROLES
    }
    return store, truth_log, variants


def cell_path(mode: str, role: str, quantity: str, residual: str):
    return OUTDIR / mode / "stage2" / role / f"{objective_name(quantity, residual)}.npz"


def fit_cell(
    mode: str,
    role: str,
    variants: np.ndarray,
    truth_log: np.ndarray,
    quantity: str,
    residual: str,
    executor: ProcessPoolExecutor | None,
    force: bool,
):
    path = cell_path(mode, role, quantity, residual)
    if path.exists() and not force:
        return np.load(path)
    tasks = [
        (family, truth, quantity, residual, RADII)
        for family, truth in zip(variants, truth_log)
    ]
    start = time.perf_counter()
    if executor is None:
        results = [_fit_task(task) for task in tasks]
    else:
        results = list(
            executor.map(
                _fit_task,
                tasks,
                chunksize=max(len(tasks) // (8 * MAX_WORKERS), 1),
            )
        )
    elapsed = time.perf_counter() - start
    n_parameter = FAMILY_SPECS[str(variants[0])].n_parameter
    payload = {
        "parameters": np.asarray([item[0] for item in results]).reshape(
            -1, n_parameter
        ),
        "prediction_log": np.asarray([item[1] for item in results]),
        "cog_rms_dex": np.asarray([item[2] for item in results]),
        "success": np.asarray([item[3] for item in results]),
        "nfev": np.asarray([item[4] for item in results]),
        "jacobian_min_ratio": np.asarray([item[5] for item in results]),
        "boundary_distance": np.asarray([item[6] for item in results]),
        "near_solution_count": np.asarray([item[7] for item in results]),
        "near_parameter_spread": np.asarray([item[8] for item in results]),
        "near_profile_spread_dex": np.asarray([item[9] for item in results]),
        "fit_seconds": elapsed,
        "variants": variants,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    print(
        f"  {role:<22} {objective_name(quantity, residual):<18} "
        f"{elapsed:7.2f} s; median full-CoG RMS "
        f"{np.median(payload['cog_rms_dex']):.5f} dex",
        flush=True,
    )
    return np.load(path)


def summarize(results: dict, truth_log: np.ndarray, epochs: np.ndarray):
    output = {}
    for (role, name), cell in results.items():
        metrics = profile_metrics(np.asarray(cell["prediction_log"]), truth_log)
        output.setdefault(role, {})[name] = {"by_epoch": {}}
        for epoch, redshift in enumerate(REDSHIFTS):
            selected = epochs == epoch
            output[role][name]["by_epoch"][f"z{redshift:g}"] = {
                "median_cog_rms_full_dex": float(
                    np.median(metrics["cog_rms_full_dex"][selected])
                ),
                "median_cog_rms_5_30_dex": float(
                    np.median(metrics["cog_rms_5_30_dex"][selected])
                ),
                "median_density_rms_dex": float(
                    np.median(metrics["density_rms_dex"][selected])
                ),
                "median_shell_rms_dex": float(
                    np.median(metrics["shell_rms_dex"][selected])
                ),
                "median_absolute_r50_error_dex": float(
                    np.median(np.abs(metrics["size_error_dex"][selected, 1]))
                ),
                "median_absolute_r80_error_dex": float(
                    np.median(np.abs(metrics["size_error_dex"][selected, 2]))
                ),
                "median_absolute_r90_error_dex": float(
                    np.median(np.abs(metrics["size_error_dex"][selected, 3]))
                ),
                "failure_fraction": float(np.mean(~cell["success"][selected])),
                "boundary_fraction_within_one_percent": float(
                    np.mean(cell["boundary_distance"][selected] < 0.01)
                ),
                "median_jacobian_min_ratio": float(
                    np.median(cell["jacobian_min_ratio"][selected])
                ),
            }
    return output


def _median_over_epochs(summary, role, objective, metric):
    return float(
        np.median(
            [values[metric] for values in summary[role][objective]["by_epoch"].values()]
        )
    )


def matrix_figure(mode: str, summary: dict):
    objectives = [objective_name(*values) for values in OBJECTIVES]
    labels = [name.replace("_", " ") for name in objectives]
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.8))
    definitions = (
        ("median_cog_rms_full_dex", "Median full-CoG RMS (dex)"),
        ("median_density_rms_dex", "Median density RMS (dex)"),
        ("median_absolute_r90_error_dex", r"Median $|\Delta\log R_{90}|$ (dex)"),
    )
    for axis, (metric, title) in zip(axes, definitions):
        values = np.asarray(
            [
                [
                    _median_over_epochs(summary, role, objective, metric)
                    for objective in objectives
                ]
                for role in ROLES
            ]
        )
        image = axis.imshow(values, aspect="auto", cmap="cividis")
        axis.grid(False)
        axis.set_xticks(range(len(objectives)), labels, rotation=40, ha="right")
        axis.set_yticks(range(len(ROLES)), [role.replace("_", " ") for role in ROLES])
        axis.set_title(title)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    for panel, axis in zip("ABC", axes):
        axis.text(-0.17, 1.03, panel, transform=axis.transAxes, fontweight="bold")
    figure.suptitle(f"exp62 Stage 2 {mode} — fitting-objective sensitivity")
    figure.tight_layout()
    save_fig(figure, FIGDIR / f"stage2_{mode}" / "exp62_objective_matrix")
    plt.close(figure)


def direct_figure(mode: str, results: dict, truth_log: np.ndarray):
    objectives = [objective_name(*values) for values in OBJECTIVES]
    colors = [OKABE_ITO[index] for index in (0, 1, 2, 3, 4, 6)]
    figure, axes = plt.subplots(len(ROLES), 2, figsize=(10.0, 9.5), sharex=True)
    for row, role in enumerate(ROLES):
        baseline = np.asarray(results[(role, "cog_log")]["prediction_log"])
        error = np.sqrt(np.mean((baseline - truth_log) ** 2, axis=1))
        galaxy = np.argsort(error)[len(error) // 2]
        axes[row, 0].plot(
            RADII, truth_log[galaxy], "o", color="black", ms=3, label="TNG CoG"
        )
        for color, objective in zip(colors, objectives):
            prediction = np.asarray(results[(role, objective)]["prediction_log"])
            axes[row, 0].plot(
                RADII,
                prediction[galaxy],
                color=color,
                linewidth=1.2,
                label=objective.replace("_", " "),
            )
            axes[row, 1].plot(
                RADII,
                prediction[galaxy] - truth_log[galaxy],
                color=color,
                linewidth=1.2,
            )
        axes[row, 0].set_ylabel(r"$\log_{10}M_*(<R)\ [M_\odot]$")
        axes[row, 1].set_ylabel("Model - TNG (dex)")
        axes[row, 0].set_title(role.replace("_", " "))
        axes[row, 1].axhline(0.0, color="0.6", linewidth=0.8)
    for axis in axes.ravel():
        axis.set_xscale("log")
    axes[-1, 0].set_xlabel("Semi-major axis (kpc)")
    axes[-1, 1].set_xlabel("Semi-major axis (kpc)")
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=2)
    figure.suptitle(f"exp62 Stage 2 {mode} — one typical galaxy under all objectives")
    figure.tight_layout()
    save_fig(figure, FIGDIR / f"stage2_{mode}" / "exp62_objective_direct_profiles")
    plt.close(figure)


def run(mode: str, force: bool = False):
    store, truth_log, variants = load_stage1(mode)
    epochs = np.asarray(store["epochs"])
    print(
        f"exp62 Stage 2 {mode}: {len(truth_log)} galaxy-epoch CoGs, "
        f"{len(ROLES)} roles x {len(OBJECTIVES)} objectives"
    )
    start = time.perf_counter()
    results = {}
    if MAX_WORKERS > 1:
        context = ProcessPoolExecutor(max_workers=MAX_WORKERS)
    else:
        context = None
    try:
        for role in ROLES:
            for quantity, residual in OBJECTIVES:
                name = objective_name(quantity, residual)
                results[(role, name)] = fit_cell(
                    mode,
                    role,
                    variants[role],
                    truth_log,
                    quantity,
                    residual,
                    context,
                    force,
                )
    finally:
        if context is not None:
            context.shutdown()
    summary = summarize(results, truth_log, epochs)
    stage_dir = OUTDIR / mode / "stage2"
    (stage_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    matrix_figure(mode, summary)
    direct_figure(mode, results, truth_log)
    elapsed = time.perf_counter() - start
    payload = {
        "mode": mode,
        "n_profile": len(truth_log),
        "n_role": len(ROLES),
        "n_objective": len(OBJECTIVES),
        "wall_seconds": elapsed,
        "subminute_gate_passed": bool(elapsed < 60.0) if mode == "demo" else None,
    }
    (stage_dir / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    write_manifest(
        stage_dir,
        {
            "experiment": "exp62_cog_fit_atlas",
            "stage": 2,
            "mode": mode,
            "roles": ROLES,
            "objectives": [objective_name(*values) for values in OBJECTIVES],
        },
    )
    print(json.dumps(payload, indent=2))
    if mode == "demo" and elapsed >= 60.0:
        raise RuntimeError(
            f"Stage 2 demo took {elapsed:.2f} seconds; full run is forbidden"
        )


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if command not in ("demo", "full"):
        raise SystemExit(__doc__)
    run(command, force="--force" in sys.argv)
