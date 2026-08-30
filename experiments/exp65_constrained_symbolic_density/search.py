# %%
"""Exp65 constrained symbolic projected-density search.

Commands:

    uv run python experiments/exp65_constrained_symbolic_density/search.py mechanics
    uv run python experiments/exp65_constrained_symbolic_density/search.py demo --force
    uv run python experiments/exp65_constrained_symbolic_density/search.py full --force
"""

from __future__ import annotations

import json
import os
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

from grammar import (  # noqa: E402
    RADII_KPC,
    ExpressionSpec,
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
MAX_WORKERS = min(int(os.environ.get("EXP65_WORKERS", "10")), os.cpu_count() or 1)
OBJECTIVES = ("log_cog", "absolute_cog", "log_shell")


def _source_path(mode: str, section: str) -> Path:
    paths = {
        "atlas": EXP62_OUTDIR / mode / "atlas" / "predictions.npz",
        "cubic": EXP62_OUTDIR / "stage5" / mode / "predictions.npz",
        "double": EXP62_OUTDIR / mode / "variants" / "double_sersic_free.npz",
    }
    return paths[section]


def load_references(mode: str) -> dict[str, np.ndarray]:
    """Load aligned Exp62 truth and paired reference predictions."""
    missing = [
        _source_path(mode, section)
        for section in ("atlas", "cubic", "double")
        if not _source_path(mode, section).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Exp62 paired references are absent: " + ", ".join(map(str, missing))
        )
    atlas = np.load(_source_path(mode, "atlas"))
    cubic = np.load(_source_path(mode, "cubic"))
    double = np.load(_source_path(mode, "double"))
    if not np.array_equal(atlas["rows"], cubic["rows"]):
        raise ValueError("Exp62 reference rows do not match")
    if not np.array_equal(atlas["epochs"], cubic["epochs"]):
        raise ValueError("Exp62 reference epochs do not match")
    if np.max(np.abs(atlas["truth_log"] - cubic["truth_log"])) > 1.0e-12:
        raise ValueError("Exp62 reference CoGs do not match")
    return {
        "rows": np.asarray(atlas["rows"], int),
        "epochs": np.asarray(atlas["epochs"], int),
        "redshifts": np.asarray(atlas["redshifts"], float),
        "truth_log": np.asarray(atlas["truth_log"], float),
        "halo_mass": np.asarray(atlas["halo_mass"], float),
        "folds": np.asarray(atlas["folds"], int),
        "prediction_double_sersic": np.asarray(
            atlas["prediction_double_sersic_free"], float
        ),
        "prediction_cubic_logit": np.asarray(cubic["prediction_logit_cubic5"], float),
        "prediction_pca3": np.asarray(atlas["prediction_pca3"], float),
        "jacobian_double_sersic": np.asarray(double["jacobian_min_ratio"], float),
        "boundary_double_sersic": np.asarray(double["boundary_distance"], float),
        "near_parameter_double_sersic": np.asarray(
            double["near_parameter_spread"], float
        ),
        "near_profile_double_sersic": np.asarray(
            double["near_profile_spread_dex"], float
        ),
        "jacobian_cubic_logit": np.asarray(
            cubic["jacobian_min_ratio_logit_cubic5"], float
        ),
        "boundary_cubic_logit": np.asarray(
            cubic["boundary_distance_logit_cubic5"], float
        ),
        "near_parameter_cubic_logit": np.asarray(
            cubic["near_parameter_spread_logit_cubic5"], float
        ),
        "near_profile_cubic_logit": np.asarray(
            cubic["near_profile_spread_dex_logit_cubic5"], float
        ),
    }


def development_masks(folds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the frozen expression-discovery and development-holdout masks."""
    values = np.asarray(folds, int)
    discovery = np.isin(values, (0, 1, 2))
    holdout = np.isin(values, (3, 4))
    if not np.all(discovery ^ holdout):
        raise ValueError("development folds must be numbered zero through four")
    return discovery, holdout


def teacher_indices(
    reference: dict[str, np.ndarray], discovery: np.ndarray
) -> np.ndarray:
    """Select three double-Sersic compactness levels per epoch."""
    chosen: list[int] = []
    truth = reference["truth_log"]
    for epoch in range(5):
        candidates = np.flatnonzero(discovery & (reference["epochs"] == epoch))
        sizes = np.asarray(
            [
                enclosed_radius(10.0 ** truth[index], RADII_KPC, 0.5)
                for index in candidates
            ]
        )
        order = candidates[np.argsort(sizes)]
        positions = np.rint(np.linspace(0, len(order) - 1, 3)).astype(int)
        chosen.extend(order[positions].tolist())
    return np.asarray(chosen, int)


def _shell_matrix(log_cog: np.ndarray) -> np.ndarray:
    return np.asarray([shell_log_density(profile, RADII_KPC) for profile in log_cog])


def profile_metrics(
    prediction_log: np.ndarray, truth_log: np.ndarray
) -> dict[str, np.ndarray]:
    """Return the common Exp64 replacement metrics per profile."""
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
        "inner_cog_rms_dex": np.sqrt(np.mean(residual[:, middle] ** 2, axis=1)),
        "shell_density_rms_dex": np.sqrt(
            np.mean((prediction_shell - truth_shell) ** 2, axis=1)
        ),
        "absolute_size_error_dex": np.abs(size_error),
        "absolute_outer_shell_error_dex": np.abs(outer_shell_error),
    }


def _epoch_reduce(
    values: np.ndarray, epochs: np.ndarray, operation: str = "median"
) -> list[float] | list[list[float]]:
    output = []
    for epoch in range(5):
        selected = values[epochs == epoch]
        reduced = (
            np.nanmedian(selected, axis=0)
            if operation == "median"
            else np.nanmax(selected, axis=0)
        )
        output.append(
            float(reduced)
            if np.ndim(reduced) == 0
            else np.asarray(reduced, float).tolist()
        )
    return output


def summarize(
    prediction: np.ndarray,
    truth_log: np.ndarray,
    epochs: np.ndarray,
    diagnostics: dict[str, np.ndarray] | None = None,
) -> dict[str, object]:
    metrics = profile_metrics(prediction, truth_log)
    summary = {key: _epoch_reduce(value, epochs) for key, value in metrics.items()}
    if diagnostics is not None:
        summary.update(
            {
                "median_jacobian_min_ratio": _epoch_reduce(
                    diagnostics["jacobian_min_ratio"], epochs
                ),
                "boundary_fraction_within_one_percent": [
                    float(
                        np.mean(
                            diagnostics["boundary_distance"][epochs == epoch] < 0.01
                        )
                    )
                    for epoch in range(5)
                ],
                "median_near_parameter_spread": _epoch_reduce(
                    diagnostics["near_parameter_spread"], epochs
                ),
                "maximum_near_profile_spread_dex": _epoch_reduce(
                    diagnostics["near_profile_spread_dex"], epochs, "maximum"
                ),
            }
        )
    return summary


def _reference_diagnostics(
    reference: dict[str, np.ndarray], name: str
) -> dict[str, np.ndarray] | None:
    if name == "pca3":
        return None
    return {
        "jacobian_min_ratio": reference[f"jacobian_{name}"],
        "boundary_distance": reference[f"boundary_{name}"],
        "near_parameter_spread": reference[f"near_parameter_{name}"],
        "near_profile_spread_dex": reference[f"near_profile_{name}"],
    }


def mechanics() -> dict[str, float]:
    """Exercise hard constraints and synthetic recovery for all Round-A forms."""
    started = time.perf_counter()
    radius = np.concatenate(([0.0], np.geomspace(1.0e-6, 1.0e5, 1000)))
    for name in round_a_specs():
        parameters = synthetic_parameters(name)
        density = surface_density(name, parameters, radius, grid_size=2048)
        slope = density_log_slope(name, parameters, radius)
        if not (
            density[0] > 0.0
            and np.all(density > 0.0)
            and np.all(np.diff(density) <= 2.0e-10 * density[0])
            and np.all(slope <= 1.0e-12)
        ):
            raise AssertionError(f"physical mechanics failed for {name}")
        truth = cog_log(name, parameters, RADII_KPC, grid_size=1024)
        recovered = fit_profile(
            name,
            RADII_KPC,
            truth,
            grid_size=1024,
            synthetic_truth_parameters=parameters,
        )
        if recovered["cog_rms_dex"] >= 2.0e-5:
            raise AssertionError(f"synthetic recovery failed for {name}")
    elapsed = time.perf_counter() - started
    result = {"n_expression": len(round_a_specs()), "wall_seconds": elapsed}
    print(json.dumps(result, indent=2))
    return result


def _fit_task(
    task: tuple[ExpressionSpec, str, np.ndarray, int, int, int],
) -> dict[str, object]:
    spec, objective, truth_log, grid_size, n_starts, max_nfev = task
    return fit_profile(
        spec,
        RADII_KPC,
        truth_log,
        objective=objective,
        grid_size=grid_size,
        n_starts=n_starts,
        max_nfev=max_nfev,
    )


def _fit_specs(
    specs: dict[str, ExpressionSpec],
    truth_log: np.ndarray,
    objective: str,
    executor: ProcessPoolExecutor,
    *,
    grid_size: int,
    n_starts: int,
    max_nfev: int,
) -> dict[str, dict[str, np.ndarray]]:
    names = list(specs)
    tasks = [
        (specs[name], objective, truth, grid_size, n_starts, max_nfev)
        for name in names
        for truth in truth_log
    ]
    fitted = list(executor.map(_fit_task, tasks, chunksize=max(len(tasks) // 100, 1)))
    output = {}
    offset = 0
    for name in names:
        count = len(truth_log)
        items = fitted[offset : offset + count]
        offset += count
        output[name] = {
            "parameters": np.asarray([item["parameters"] for item in items]),
            "prediction": np.asarray([item["prediction"] for item in items]),
            "success": np.asarray([item["success"] for item in items], bool),
            "nfev": np.asarray([item["nfev"] for item in items], int),
            "jacobian_min_ratio": np.asarray(
                [item["jacobian_min_ratio"] for item in items], float
            ),
            "boundary_distance": np.asarray(
                [item["boundary_distance"] for item in items], float
            ),
            "near_parameter_spread": np.asarray(
                [item["near_parameter_spread"] for item in items], float
            ),
            "near_profile_spread_dex": np.asarray(
                [item["near_profile_spread_dex"] for item in items], float
            ),
        }
    return output


def _candidate_record(
    spec: ExpressionSpec,
    result: dict[str, np.ndarray],
    truth: np.ndarray,
    epochs: np.ndarray,
    reference: dict[str, np.ndarray],
) -> dict[str, object]:
    metrics = profile_metrics(result["prediction"], truth)
    candidate_epoch = np.asarray(_epoch_reduce(metrics["full_cog_rms_dex"], epochs))
    double_metrics = profile_metrics(reference["prediction_double_sersic"], truth)
    double_epoch = np.asarray(_epoch_reduce(double_metrics["full_cog_rms_dex"], epochs))
    candidate_jacobian = np.asarray(_epoch_reduce(result["jacobian_min_ratio"], epochs))
    cubic_jacobian = np.asarray(
        _epoch_reduce(reference["jacobian_cubic_logit"], epochs)
    )
    boundary_fraction = float(np.mean(result["boundary_distance"] < 0.01))
    conditioning_pass = int(
        np.count_nonzero(candidate_jacobian >= 0.25 * cubic_jacobian)
    )
    eligible = bool(
        np.all(result["success"])
        and boundary_fraction < 0.05
        and conditioning_pass >= 4
        and np.max(result["near_profile_spread_dex"]) < 0.003
    )
    return {
        "name": spec.name,
        "label": spec.label,
        "n_parameter": spec.n_parameter,
        "has_trigonometric_atom": spec.has_trigonometric_atom,
        "eligible": eligible,
        "rank_score_worst_epoch_ratio_to_double_sersic": float(
            np.max(candidate_epoch / double_epoch)
        ),
        "median_full_cog_rms_dex": _epoch_reduce(metrics["full_cog_rms_dex"], epochs),
        "median_jacobian_min_ratio": candidate_jacobian.tolist(),
        "boundary_fraction_within_one_percent": boundary_fraction,
        "conditioning_epochs_at_quarter_cubic": conditioning_pass,
        "maximum_near_profile_spread_dex": float(
            np.max(result["near_profile_spread_dex"])
        ),
    }


def _rank_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        records,
        key=lambda item: (
            not bool(item["eligible"]),
            float(item["rank_score_worst_epoch_ratio_to_double_sersic"]),
            int(item["n_parameter"]),
            str(item["name"]),
        ),
    )


def _subset_reference(
    reference: dict[str, np.ndarray], mask: np.ndarray
) -> dict[str, np.ndarray]:
    output = {}
    for key, value in reference.items():
        if isinstance(value, np.ndarray) and value.shape[:1] == mask.shape:
            output[key] = value[mask]
        else:
            output[key] = value
    return output


def _teacher_summary(
    specs: dict[str, ExpressionSpec],
    results: dict[str, dict[str, np.ndarray]],
    teacher_truth: np.ndarray,
) -> tuple[dict[str, dict[str, object]], bool]:
    summary = {}
    passed = False
    for name, spec in specs.items():
        rms = np.sqrt(
            np.mean((results[name]["prediction"] - teacher_truth) ** 2, axis=1)
        )
        item = {
            "label": spec.label,
            "median_candidate_to_teacher_rms_dex": float(np.median(rms)),
            "p90_candidate_to_teacher_rms_dex": float(np.percentile(rms, 90.0)),
            "all_fits_succeeded": bool(np.all(results[name]["success"])),
        }
        item["passes"] = bool(
            item["all_fits_succeeded"]
            and item["median_candidate_to_teacher_rms_dex"] < 0.003
            and item["p90_candidate_to_teacher_rms_dex"] < 0.010
        )
        passed |= item["passes"]
        summary[name] = item
    return summary, passed


def _spec_to_dict(spec: ExpressionSpec) -> dict[str, object]:
    return {
        "name": spec.name,
        "label": spec.label,
        "atom_kinds": list(spec.atom_kinds),
        "atom_settings": [list(values) for values in spec.atom_settings],
        "parameter_names": list(spec.parameter_names),
        "lower": list(spec.lower),
        "upper": list(spec.upper),
    }


def _spec_from_dict(values: dict[str, object]) -> ExpressionSpec:
    return ExpressionSpec(
        name=str(values["name"]),
        label=str(values["label"]),
        atom_kinds=tuple(values["atom_kinds"]),
        atom_settings=tuple(tuple(item) for item in values["atom_settings"]),
        parameter_names=tuple(values["parameter_names"]),
        lower=tuple(values["lower"]),
        upper=tuple(values["upper"]),
    )


def _replacement_decision(
    candidate: dict[str, object],
    double: dict[str, object],
    cubic: dict[str, object],
    diagnostics: dict[str, np.ndarray],
) -> dict[str, object]:
    accuracy_keys = ("full_cog_rms_dex", "inner_cog_rms_dex", "shell_density_rms_dex")
    accuracy_pass = all(
        np.all(np.asarray(candidate[key]) <= 1.10 * np.asarray(double[key]))
        for key in accuracy_keys
    )
    candidate_size = np.asarray(candidate["absolute_size_error_dex"])
    double_size = np.asarray(double["absolute_size_error_dex"])
    size_pass = bool(np.all(candidate_size[:, 1:] <= 1.10 * double_size[:, 1:]))
    conditioning_epoch = np.asarray(
        candidate["median_jacobian_min_ratio"]
    ) >= 0.5 * np.asarray(cubic["median_jacobian_min_ratio"])
    boundary_fraction = float(np.mean(diagnostics["boundary_distance"] < 0.01))
    near_parameter_pass = bool(
        np.all(np.asarray(candidate["median_near_parameter_spread"]) < 0.01)
    )
    near_profile_pass = bool(
        np.all(np.asarray(candidate["maximum_near_profile_spread_dex"]) < 0.001)
    )
    return {
        "accuracy_gate": bool(accuracy_pass and size_pass),
        "non_size_accuracy_gate": bool(accuracy_pass),
        "r50_r80_r90_gate": size_pass,
        "conditioning_epoch_pass": conditioning_epoch.tolist(),
        "conditioning_gate": bool(np.count_nonzero(conditioning_epoch) >= 4),
        "boundary_fraction_within_one_percent": boundary_fraction,
        "boundary_gate": bool(boundary_fraction < 0.01),
        "near_parameter_gate": near_parameter_pass,
        "near_profile_gate": near_profile_pass,
        "complete_numerical_gate": bool(
            accuracy_pass
            and size_pass
            and np.count_nonzero(conditioning_epoch) >= 4
            and boundary_fraction < 0.01
            and near_parameter_pass
            and near_profile_pass
        ),
        "visual_qa_gate": None,
    }


def _representative_indices(
    result: dict[str, np.ndarray], reference: dict[str, np.ndarray]
) -> list[int]:
    rms = np.sqrt(np.mean((result["prediction"] - reference["truth_log"]) ** 2, axis=1))
    order = np.argsort(rms)
    size = np.asarray(
        [enclosed_radius(10.0**row, RADII_KPC, 0.5) for row in reference["truth_log"]]
    )
    size_order = np.argsort(size)
    return list(
        dict.fromkeys(
            [
                int(order[0]),
                int(order[len(order) // 2]),
                int(order[-1]),
                int(size_order[len(size_order) // 6]),
                int(size_order[-1 - len(size_order) // 6]),
            ]
        )
    )


def _frontier_figure(
    mode: str,
    discovery_records: list[dict[str, object]],
    holdout_records: list[dict[str, object]],
    finalist_summary: dict[str, object],
    double_summary: dict[str, object],
    cubic_summary: dict[str, object],
    redshifts: np.ndarray,
) -> list[Path]:
    figure, axes = plt.subplots(2, 2, figsize=(10.0, 7.2))
    for records, marker, label in (
        (discovery_records, "o", "Expression discovery"),
        (holdout_records, "s", "Development holdout"),
    ):
        for item in records:
            color = OKABE_ITO[5] if item["has_trigonometric_atom"] else OKABE_ITO[1]
            axes[0, 0].scatter(
                item["n_parameter"],
                item["rank_score_worst_epoch_ratio_to_double_sersic"],
                color=color,
                marker=marker,
                alpha=0.8,
            )
            axes[0, 1].scatter(
                np.median(item["median_full_cog_rms_dex"]),
                np.median(item["median_jacobian_min_ratio"]),
                color=color,
                marker=marker,
                alpha=0.8,
            )
    axes[0, 0].axhline(1.10, color="black", linestyle=":", lw=1)
    axes[0, 0].set_xlabel("Total fitted coordinates")
    axes[0, 0].set_ylabel("Worst-epoch CoG RMS / double Sersic")
    axes[0, 1].set_xlabel("Median full-CoG RMS (dex)")
    axes[0, 1].set_ylabel("Median scaled Jacobian ratio")
    axes[0, 1].set_yscale("log")
    axes[0, 0].scatter([], [], color=OKABE_ITO[1], label="No trig atom")
    axes[0, 0].scatter([], [], color=OKABE_ITO[5], label="Includes damped trig")
    axes[0, 0].scatter([], [], color="black", marker="o", label="Discovery")
    axes[0, 0].scatter([], [], color="black", marker="s", label="Holdout")
    axes[0, 0].legend(fontsize=7)
    panels = (
        (axes[1, 0], "full_cog_rms_dex", "Median full-CoG RMS (dex)"),
        (axes[1, 1], "median_jacobian_min_ratio", "Median scaled Jacobian ratio"),
    )
    for axis, metric, ylabel in panels:
        for values, label, color, linestyle in (
            (finalist_summary[metric], "Symbolic finalist", OKABE_ITO[2], "-"),
            (double_summary[metric], "Double Sersic", OKABE_ITO[1], "--"),
            (cubic_summary[metric], "Cubic-logit", OKABE_ITO[5], ":"),
        ):
            axis.plot(
                redshifts,
                values,
                marker="o",
                color=color,
                linestyle=linestyle,
                label=label,
            )
        axis.set_xlabel("Redshift")
        axis.set_ylabel(ylabel)
        if metric == "median_jacobian_min_ratio":
            axis.set_yscale("log")
        axis.grid(alpha=0.2)
    axes[1, 0].legend(fontsize=7)
    figure.suptitle(
        f"Exp65 {mode}: constrained symbolic density frontier\n"
        "Errors reference measured CoG-derived masses; larger Jacobian ratio is better"
    )
    figure.tight_layout()
    paths = save_fig(figure, FIGDIR / mode / "exp65_symbolic_frontier")
    plt.close(figure)
    return paths


def _direct_figure(
    mode: str,
    spec: ExpressionSpec,
    result: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
) -> list[Path]:
    indices = _representative_indices(result, reference)
    figure, axes = plt.subplots(len(indices), 4, figsize=(13.0, 2.25 * len(indices)))
    shell_radius = np.sqrt(RADII_KPC * np.concatenate(([0.0], RADII_KPC[:-1])))
    shell_radius[0] = 0.5 * RADII_KPC[0]
    for row, index in enumerate(indices):
        truth = reference["truth_log"][index]
        axes[row, 0].plot(
            RADII_KPC, truth, color="black", marker="o", ms=2, label="TNG"
        )
        series = (
            (result["prediction"][index], "Symbolic finalist", OKABE_ITO[2], "-"),
            (
                reference["prediction_double_sersic"][index],
                "Double Sersic",
                OKABE_ITO[1],
                "--",
            ),
            (
                reference["prediction_cubic_logit"][index],
                "Cubic-logit",
                OKABE_ITO[5],
                ":",
            ),
        )
        truth_shell = shell_log_density(truth, RADII_KPC)
        for prediction, label, color, linestyle in series:
            axes[row, 0].plot(
                RADII_KPC, prediction, color=color, linestyle=linestyle, label=label
            )
            axes[row, 1].plot(
                RADII_KPC, prediction - truth, color=color, linestyle=linestyle
            )
            model_shell = shell_log_density(prediction, RADII_KPC)
            axes[row, 2].plot(
                shell_radius, model_shell, color=color, linestyle=linestyle
            )
            axes[row, 3].plot(
                shell_radius,
                model_shell - truth_shell,
                color=color,
                linestyle=linestyle,
            )
        axes[row, 2].plot(shell_radius, truth_shell, color="black", marker="o", ms=2)
        axes[row, 0].set_ylabel(
            f"z={reference['redshifts'][reference['epochs'][index]]:g}\nlog mass"
        )
        axes[row, 1].axhline(0.0, color="0.7", lw=0.8)
        axes[row, 3].axhline(0.0, color="0.7", lw=0.8)
        for axis in axes[row]:
            axis.set_xscale("log")
            axis.grid(alpha=0.15)
    axes[0, 0].legend(fontsize=7)
    labels = (
        "Radius (kpc)",
        "Radius (kpc)",
        "Annulus midpoint (kpc)",
        "Annulus midpoint (kpc)",
    )
    titles = (
        "Curve of growth",
        "Model - TNG CoG (dex)",
        "Log shell density",
        "Model - TNG density (dex)",
    )
    for axis, label, title in zip(axes[-1], labels, titles, strict=True):
        axis.set_xlabel(label)
        axis.set_title(title)
    figure.suptitle(
        f"Exp65 {mode}: {spec.label}\n"
        "Rows show best, typical, worst, compact, and extended development profiles"
    )
    figure.tight_layout()
    paths = save_fig(figure, FIGDIR / mode / "exp65_symbolic_direct_profiles")
    plt.close(figure)
    return paths


def _teacher_failure_figure(
    specs: dict[str, ExpressionSpec],
    results: dict[str, dict[str, np.ndarray]],
    teacher_truth: np.ndarray,
    teacher_summary: dict[str, dict[str, object]],
) -> list[Path]:
    """Show why the frozen grammar failed before measured-profile search."""
    names = list(specs)
    median = np.asarray(
        [teacher_summary[name]["median_candidate_to_teacher_rms_dex"] for name in names]
    )
    p90 = np.asarray(
        [teacher_summary[name]["p90_candidate_to_teacher_rms_dex"] for name in names]
    )
    normalized = np.maximum(median / 0.003, p90 / 0.010)
    best_name = names[int(np.argmin(normalized))]
    best_prediction = results[best_name]["prediction"]
    rms = np.sqrt(np.mean((best_prediction - teacher_truth) ** 2, axis=1))
    order = np.argsort(rms)
    selected = (int(order[0]), int(order[len(order) // 2]), int(order[-1]))

    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    positions = np.arange(len(names))
    colors = [
        OKABE_ITO[5] if specs[name].has_trigonometric_atom else OKABE_ITO[1]
        for name in names
    ]
    axes[0, 0].scatter(positions, median, c=colors)
    axes[0, 0].axhline(0.003, color="black", linestyle=":", label="Declared limit")
    axes[0, 0].set_ylabel("Median candidate-to-teacher CoG RMS (dex)")
    axes[0, 1].scatter(positions, p90, c=colors)
    axes[0, 1].axhline(0.010, color="black", linestyle=":", label="Declared limit")
    axes[0, 1].set_ylabel("90th-percentile CoG RMS (dex)")
    for axis in axes[0]:
        axis.set_xticks(positions)
        axis.set_xticklabels(names, rotation=70, ha="right", fontsize=6)
        axis.set_yscale("log")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)

    for index in selected:
        label = f"RMS={rms[index]:.4f} dex"
        axes[1, 0].plot(
            RADII_KPC, teacher_truth[index], color="black", marker="o", ms=2
        )
        axes[1, 0].plot(RADII_KPC, best_prediction[index], label=label)
        axes[1, 1].plot(
            RADII_KPC, best_prediction[index] - teacher_truth[index], label=label
        )
    axes[1, 0].set_ylabel("log10 enclosed stellar mass (Msun)")
    axes[1, 1].set_ylabel("Candidate - double Sersic CoG (dex)")
    axes[1, 1].axhline(0.0, color="0.7", lw=0.8)
    for axis in axes[1]:
        axis.set_xscale("log")
        axis.set_xlabel("Radius (kpc)")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    axes[1, 0].set_title(f"Best compromise: {specs[best_name].label}")
    axes[1, 1].set_title("Best, typical, and worst teacher residuals")
    figure.suptitle(
        "Exp65 teacher feasibility gate: no expression passes both limits\n"
        "Orange points contain damped sine/cosine atoms; blue points do not"
    )
    figure.tight_layout()
    paths = save_fig(figure, FIGDIR / "teacher" / "exp65_teacher_failure")
    plt.close(figure)
    return paths


def _close_teacher_failure(
    started: float,
    mechanics_result: dict[str, float],
    reference: dict[str, np.ndarray],
    teacher_rows: np.ndarray,
    round_a: dict[str, ExpressionSpec],
    teacher_results: dict[str, dict[str, np.ndarray]],
    teacher_summary: dict[str, dict[str, object]],
) -> dict[str, object]:
    teacher_truth = reference["prediction_double_sersic"][teacher_rows]
    figure_paths = _teacher_failure_figure(
        round_a, teacher_results, teacher_truth, teacher_summary
    )
    elapsed = time.perf_counter() - started
    stage_dir = OUTDIR / "teacher"
    stage_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "mode": "teacher",
        "status": "complete_null",
        "termination": "teacher_feasibility_gate_failed",
        "n_teacher_profile": len(teacher_rows),
        "teacher_rows": teacher_rows,
        "mechanics": mechanics_result,
        "teacher_gate_passed": False,
        "teacher_summary": teacher_summary,
        "wall_seconds_to_declared_stop": elapsed,
        "measured_profile_search_run": False,
        "population_validation_run": False,
        "figure_paths": [str(path) for path in figure_paths],
        "source_exp62_outdir": str(EXP62_OUTDIR),
    }
    (stage_dir / "summary.json").write_text(
        json.dumps(_json_ready(summary), indent=2) + "\n"
    )
    np.savez_compressed(
        stage_dir / "teacher_results.npz",
        teacher_rows=teacher_rows,
        teacher_truth=teacher_truth,
        **{
            f"prediction_{name}": result["prediction"]
            for name, result in teacher_results.items()
        },
    )
    write_manifest(
        stage_dir,
        {
            "mode": "teacher",
            "max_workers": MAX_WORKERS,
            "source_exp62_outdir": str(EXP62_OUTDIR),
        },
    )
    print(json.dumps(_json_ready(summary), indent=2))
    return summary


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


def _save_fit(path: Path, result: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **result)


def run_demo(force: bool = False) -> dict[str, object]:
    """Run the predeclared complete 90-profile development path."""
    summary_path = OUTDIR / "demo" / "summary.json"
    if summary_path.exists() and not force:
        return json.loads(summary_path.read_text())
    teacher_stop_path = OUTDIR / "teacher" / "summary.json"
    if teacher_stop_path.exists() and not force:
        return json.loads(teacher_stop_path.read_text())
    started = time.perf_counter()
    mechanics_result = mechanics()
    reference = load_references("demo")
    discovery, holdout = development_masks(reference["folds"])
    discovery_reference = _subset_reference(reference, discovery)
    holdout_reference = _subset_reference(reference, holdout)
    round_a = round_a_specs()
    teacher_rows = teacher_indices(reference, discovery)
    teacher_truth = reference["prediction_double_sersic"][teacher_rows]

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        teacher_results = _fit_specs(
            round_a,
            teacher_truth,
            "log_cog",
            executor,
            grid_size=256,
            n_starts=2,
            max_nfev=250,
        )
        teacher_summary, teacher_pass = _teacher_summary(
            round_a, teacher_results, teacher_truth
        )
        if not teacher_pass:
            return _close_teacher_failure(
                started,
                mechanics_result,
                reference,
                teacher_rows,
                round_a,
                teacher_results,
                teacher_summary,
            )
        round_a_results = _fit_specs(
            round_a,
            reference["truth_log"][discovery],
            "log_cog",
            executor,
            grid_size=256,
            n_starts=2,
            max_nfev=250,
        )
        round_a_records = [
            _candidate_record(
                spec,
                round_a_results[name],
                discovery_reference["truth_log"],
                discovery_reference["epochs"],
                discovery_reference,
            )
            for name, spec in round_a.items()
        ]
        ranked_round_a = _rank_records(round_a_records)
        non_trig = next(
            item for item in ranked_round_a if not item["has_trigonometric_atom"]
        )
        trig_records = [
            item for item in ranked_round_a if item["has_trigonometric_atom"]
        ]
        round_b = build_round_b_specs(
            round_a[str(non_trig["name"])],
            round_a[str(trig_records[0]["name"])],
            round_a[str(trig_records[1]["name"])],
        )
        round_b_results = _fit_specs(
            round_b,
            discovery_reference["truth_log"],
            "log_cog",
            executor,
            grid_size=256,
            n_starts=2,
            max_nfev=300,
        )
        round_b_records = [
            _candidate_record(
                spec,
                round_b_results[name],
                discovery_reference["truth_log"],
                discovery_reference["epochs"],
                discovery_reference,
            )
            for name, spec in round_b.items()
        ]
        all_specs = {**round_a, **round_b}
        discovery_records = _rank_records(round_a_records + round_b_records)
        eligible = [item for item in discovery_records if item["eligible"]]
        if not eligible:
            raise RuntimeError("No Exp65 expression passed discovery eligibility")
        finalist_pool_records = eligible[:3]
        finalist_pool = {
            str(item["name"]): all_specs[str(item["name"])]
            for item in finalist_pool_records
        }
        holdout_results = _fit_specs(
            finalist_pool,
            holdout_reference["truth_log"],
            "log_cog",
            executor,
            grid_size=256,
            n_starts=2,
            max_nfev=350,
        )
        holdout_records = _rank_records(
            [
                _candidate_record(
                    spec,
                    holdout_results[name],
                    holdout_reference["truth_log"],
                    holdout_reference["epochs"],
                    holdout_reference,
                )
                for name, spec in finalist_pool.items()
            ]
        )
        finalist_record = holdout_records[0]
        finalist = finalist_pool[str(finalist_record["name"])]
        objective_results = {}
        for objective in OBJECTIVES:
            fitted = _fit_specs(
                {finalist.name: finalist},
                reference["truth_log"],
                objective,
                executor,
                grid_size=512,
                n_starts=4,
                max_nfev=500,
            )[finalist.name]
            objective_results[objective] = fitted

    reference_summaries = {
        "double_sersic": summarize(
            reference["prediction_double_sersic"],
            reference["truth_log"],
            reference["epochs"],
            _reference_diagnostics(reference, "double_sersic"),
        ),
        "cubic_logit": summarize(
            reference["prediction_cubic_logit"],
            reference["truth_log"],
            reference["epochs"],
            _reference_diagnostics(reference, "cubic_logit"),
        ),
        "pca3": summarize(
            reference["prediction_pca3"], reference["truth_log"], reference["epochs"]
        ),
    }
    finalist_summaries = {
        objective: summarize(
            result["prediction"],
            reference["truth_log"],
            reference["epochs"],
            result,
        )
        for objective, result in objective_results.items()
    }
    decision = _replacement_decision(
        finalist_summaries["log_cog"],
        reference_summaries["double_sersic"],
        reference_summaries["cubic_logit"],
        objective_results["log_cog"],
    )
    figure_paths = []
    figure_paths.extend(
        _frontier_figure(
            "demo",
            discovery_records,
            holdout_records,
            finalist_summaries["log_cog"],
            reference_summaries["double_sersic"],
            reference_summaries["cubic_logit"],
            reference["redshifts"],
        )
    )
    figure_paths.extend(
        _direct_figure("demo", finalist, objective_results["log_cog"], reference)
    )
    elapsed = time.perf_counter() - started
    stage_dir = OUTDIR / "demo"
    stage_dir.mkdir(parents=True, exist_ok=True)
    for objective, result in objective_results.items():
        _save_fit(stage_dir / f"finalist_{objective}.npz", result)
    np.savez_compressed(
        stage_dir / "data.npz",
        rows=reference["rows"],
        epochs=reference["epochs"],
        truth_log=reference["truth_log"],
        halo_mass=reference["halo_mass"],
        folds=reference["folds"],
        prediction_double_sersic=reference["prediction_double_sersic"],
        prediction_cubic_logit=reference["prediction_cubic_logit"],
    )
    summary = {
        "mode": "demo",
        "n_profile": 90,
        "development_discovery_count": int(np.count_nonzero(discovery)),
        "development_holdout_count": int(np.count_nonzero(holdout)),
        "mechanics": mechanics_result,
        "teacher_gate_passed": teacher_pass,
        "teacher_summary": teacher_summary,
        "round_a_records": round_a_records,
        "round_b_records": round_b_records,
        "development_holdout_records": holdout_records,
        "finalist_spec": _spec_to_dict(finalist),
        "finalist_summaries": finalist_summaries,
        "reference_summaries": reference_summaries,
        "decision": decision,
        "wall_seconds_complete_path": elapsed,
        "subminute_gate_passed": bool(elapsed < 60.0),
        "figure_paths": [str(path) for path in figure_paths],
        "source_exp62_outdir": str(EXP62_OUTDIR),
    }
    summary_path.write_text(json.dumps(_json_ready(summary), indent=2) + "\n")
    write_manifest(
        stage_dir,
        {
            "mode": "demo",
            "max_workers": MAX_WORKERS,
            "objectives": list(OBJECTIVES),
            "source_exp62_outdir": str(EXP62_OUTDIR),
        },
    )
    print(json.dumps(_json_ready(summary), indent=2))
    if elapsed >= 60.0:
        raise RuntimeError(f"Exp65 development gate failed at {elapsed:.2f} seconds")
    return summary


def run_full(force: bool = False) -> dict[str, object]:
    """Run population validation only after the complete development gate passes."""
    demo_path = OUTDIR / "demo" / "summary.json"
    if not demo_path.exists():
        raise FileNotFoundError("run the Exp65 demo before population validation")
    demo = json.loads(demo_path.read_text())
    if (
        not demo["subminute_gate_passed"]
        or not demo["decision"]["complete_numerical_gate"]
    ):
        raise RuntimeError("Exp65 development gates do not authorize a full run")
    summary_path = OUTDIR / "full" / "summary.json"
    if summary_path.exists() and not force:
        return json.loads(summary_path.read_text())
    started = time.perf_counter()
    finalist = _spec_from_dict(demo["finalist_spec"])
    reference = load_references("full")
    demo_data = np.load(OUTDIR / "demo" / "data.npz")
    development_pairs = set(
        zip(demo_data["rows"].tolist(), demo_data["epochs"].tolist(), strict=True)
    )
    validation = np.asarray(
        [
            (int(row), int(epoch)) not in development_pairs
            for row, epoch in zip(reference["rows"], reference["epochs"], strict=True)
        ]
    )
    validation_reference = _subset_reference(reference, validation)
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        objective_results = {}
        for objective in OBJECTIVES:
            objective_results[objective] = _fit_specs(
                {finalist.name: finalist},
                validation_reference["truth_log"],
                objective,
                executor,
                grid_size=512,
                n_starts=4,
                max_nfev=500,
            )[finalist.name]
    reference_summaries = {
        "double_sersic": summarize(
            validation_reference["prediction_double_sersic"],
            validation_reference["truth_log"],
            validation_reference["epochs"],
            _reference_diagnostics(validation_reference, "double_sersic"),
        ),
        "cubic_logit": summarize(
            validation_reference["prediction_cubic_logit"],
            validation_reference["truth_log"],
            validation_reference["epochs"],
            _reference_diagnostics(validation_reference, "cubic_logit"),
        ),
        "pca3": summarize(
            validation_reference["prediction_pca3"],
            validation_reference["truth_log"],
            validation_reference["epochs"],
        ),
    }
    finalist_summaries = {
        objective: summarize(
            result["prediction"],
            validation_reference["truth_log"],
            validation_reference["epochs"],
            result,
        )
        for objective, result in objective_results.items()
    }
    decision = _replacement_decision(
        finalist_summaries["log_cog"],
        reference_summaries["double_sersic"],
        reference_summaries["cubic_logit"],
        objective_results["log_cog"],
    )
    figure_paths = []
    figure_paths.extend(
        _frontier_figure(
            "full",
            demo["round_a_records"] + demo["round_b_records"],
            demo["development_holdout_records"],
            finalist_summaries["log_cog"],
            reference_summaries["double_sersic"],
            reference_summaries["cubic_logit"],
            validation_reference["redshifts"],
        )
    )
    figure_paths.extend(
        _direct_figure(
            "full", finalist, objective_results["log_cog"], validation_reference
        )
    )
    elapsed = time.perf_counter() - started
    stage_dir = OUTDIR / "full"
    stage_dir.mkdir(parents=True, exist_ok=True)
    for objective, result in objective_results.items():
        _save_fit(stage_dir / f"finalist_{objective}.npz", result)
    summary = {
        "mode": "full",
        "n_validation_profile": int(np.count_nonzero(validation)),
        "finalist_spec": _spec_to_dict(finalist),
        "finalist_summaries": finalist_summaries,
        "reference_summaries": reference_summaries,
        "decision": decision,
        "wall_seconds_complete_path": elapsed,
        "figure_paths": [str(path) for path in figure_paths],
    }
    summary_path.write_text(json.dumps(_json_ready(summary), indent=2) + "\n")
    write_manifest(stage_dir, {"mode": "full", "max_workers": MAX_WORKERS})
    print(json.dumps(_json_ready(summary), indent=2))
    return summary


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "mechanics"
    force = "--force" in sys.argv[2:]
    if command == "mechanics":
        mechanics()
    elif command == "demo":
        run_demo(force=force)
    elif command == "full":
        run_full(force=force)
    else:
        raise SystemExit(f"unknown command {command!r}")
