# %%
"""Exp64 search for centrally finite analytic projected profiles.

Commands:

    uv run python experiments/exp64_no_central_hole/run.py mechanics
    uv run python experiments/exp64_no_central_hole/run.py demo
    uv run python experiments/exp64_no_central_hole/run.py full
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

from families import (  # noqa: E402
    FAMILY_SPECS,
    ONE_BREAK_FAMILIES,
    RADII_KPC,
    TWO_BREAK_GRID_FAMILIES,
    cog_log,
    density_log_slope,
    fit_profile,
    physical_parameters,
    projected_density,
    shell_log_density,
    synthetic_parameters,
)
from hongshao.plotting import OKABE_ITO, save_fig, set_style  # noqa: E402
from hongshao.provenance import write_manifest  # noqa: E402
from hongshao.qa import enclosed_radius  # noqa: E402

set_style()

OUTDIR = HERE / "outputs"
FIGDIR = HERE / "figures"
DEFAULT_EXP62_ROOT = Path(
    "/Users/shuang/Dropbox/work/project/massive/hongshao_exp62_cog_fit_atlas"
)
EXP62_ROOT = Path(os.environ.get("HONGSHAO_EXP62_ROOT", DEFAULT_EXP62_ROOT))
EXP62_OUTDIR = EXP62_ROOT / "experiments" / "exp62_cog_fit_atlas" / "outputs"
MAX_WORKERS = min(int(os.environ.get("EXP64_WORKERS", "10")), os.cpu_count() or 1)
OBJECTIVES = ("log_cog", "absolute_cog", "log_shell")
STRESS_FAMILY = "two_break_free_sharpness"


def _source_path(mode: str, section: str) -> Path:
    paths = {
        "atlas": EXP62_OUTDIR / mode / "atlas" / "predictions.npz",
        "cubic": EXP62_OUTDIR / "stage5" / mode / "predictions.npz",
        "double": EXP62_OUTDIR / mode / "variants" / "double_sersic_free.npz",
        "double_absolute": (
            EXP62_OUTDIR / mode / "stage2" / "double_sersic_free" / "cog_absolute.npz"
        ),
    }
    return paths[section]


def load_references(mode: str) -> dict[str, np.ndarray]:
    missing = [
        _source_path(mode, section)
        for section in ("atlas", "cubic", "double", "double_absolute")
        if not _source_path(mode, section).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Exp62 paired references are absent: " + ", ".join(map(str, missing))
        )
    atlas = np.load(_source_path(mode, "atlas"))
    cubic = np.load(_source_path(mode, "cubic"))
    double = np.load(_source_path(mode, "double"))
    double_absolute = np.load(_source_path(mode, "double_absolute"))
    for source in (cubic,):
        if not np.array_equal(atlas["rows"], source["rows"]):
            raise ValueError("Exp62 reference rows do not match")
        if not np.array_equal(atlas["epochs"], source["epochs"]):
            raise ValueError("Exp62 reference epochs do not match")
        if np.max(np.abs(atlas["truth_log"] - source["truth_log"])) > 1.0e-12:
            raise ValueError("Exp62 reference CoGs do not match")
    return {
        "rows": np.asarray(atlas["rows"], int),
        "epochs": np.asarray(atlas["epochs"], int),
        "redshifts": np.asarray(atlas["redshifts"], float),
        "truth_log": np.asarray(atlas["truth_log"], float),
        "halo_mass": np.asarray(atlas["halo_mass"], float),
        "folds": np.asarray(atlas["folds"], int),
        "prediction_pca3": np.asarray(atlas["prediction_pca3"], float),
        "prediction_double_sersic": np.asarray(
            atlas["prediction_double_sersic_free"], float
        ),
        "prediction_double_sersic_absolute": np.asarray(
            double_absolute["prediction_log"], float
        ),
        "prediction_cubic_logit": np.asarray(cubic["prediction_logit_cubic5"], float),
        "prediction_cubic_logit_absolute": np.asarray(
            cubic["prediction_logit_cubic5_absolute"], float
        ),
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


def mechanics() -> None:
    radius = np.concatenate(([0.0], np.geomspace(1.0e-5, 1.0e4, 400)))
    for family in FAMILY_SPECS:
        parameters = synthetic_parameters(family)
        density = projected_density(family, parameters, radius)
        slope = density_log_slope(family, parameters, radius)
        assert density[0] > 0.0
        assert np.all(density > 0.0)
        assert np.all(np.diff(density) <= 1.0e-10 * density[0])
        assert np.all(slope <= 0.0)
        assert physical_parameters(family, parameters).outer_slope > 2.0
        truth = cog_log(family, parameters, RADII_KPC)
        recovered = fit_profile(
            family,
            RADII_KPC,
            truth,
            synthetic_truth_parameters=parameters,
        )
        assert recovered["cog_rms_dex"] < 2.0e-6, family
    print(f"Exp64 mechanics passed for {len(FAMILY_SPECS)} families")


def _fit_task(task: tuple[str, str, np.ndarray]) -> dict[str, object]:
    family, objective, truth_log = task
    return fit_profile(family, RADII_KPC, truth_log, objective=objective)


def _fit_path(mode: str, family_label: str, objective: str) -> Path:
    return OUTDIR / mode / "fits" / f"{family_label}_{objective}.npz"


def fit_profiles(
    mode: str,
    family_label: str,
    families: np.ndarray,
    objective: str,
    truth_log: np.ndarray,
    executor: ProcessPoolExecutor,
    force: bool,
) -> dict[str, np.ndarray]:
    path = _fit_path(mode, family_label, objective)
    if path.exists() and not force:
        return {key: np.asarray(value) for key, value in np.load(path).items()}
    tasks = [
        (str(family), objective, truth)
        for family, truth in zip(families, truth_log, strict=True)
    ]
    fitted = list(executor.map(_fit_task, tasks, chunksize=max(len(tasks) // 80, 1)))
    maximum_parameters = max(
        FAMILY_SPECS[str(family)].n_parameter for family in families
    )
    parameters = np.full((len(fitted), maximum_parameters), np.nan)
    for index, item in enumerate(fitted):
        values = np.asarray(item["parameters"], float)
        parameters[index, : len(values)] = values
    result = {
        "families": np.asarray(families, str),
        "parameters": parameters,
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
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **result)
    return result


def _fold_selected_families(
    family_results: dict[str, dict[str, np.ndarray]],
    truth_log: np.ndarray,
    epochs: np.ndarray,
    folds: np.ndarray,
) -> tuple[np.ndarray, dict[str, str]]:
    selected = np.empty(len(truth_log), dtype="U64")
    choices = {}
    for epoch in range(5):
        for fold in range(5):
            training = (epochs == epoch) & (folds != fold)
            score_by_family = {}
            for family, result in family_results.items():
                residual = result["prediction"][training] - truth_log[training]
                score_by_family[family] = float(
                    np.median(np.sqrt(np.mean(residual**2, axis=1)))
                )
            winner = min(score_by_family, key=score_by_family.get)
            selected[(epochs == epoch) & (folds == fold)] = winner
            choices[f"epoch_{epoch}_fold_{fold}"] = winner
    return selected, choices


def _assemble_selected(
    selected_families: np.ndarray,
    family_results: dict[str, dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    output = {}
    for key in (
        "prediction",
        "success",
        "nfev",
        "jacobian_min_ratio",
        "boundary_distance",
        "near_parameter_spread",
        "near_profile_spread_dex",
    ):
        example = next(iter(family_results.values()))[key]
        values = np.empty_like(example)
        for family, result in family_results.items():
            use = selected_families == family
            values[use] = result[key][use]
        output[key] = values
    maximum_parameters = max(
        result["parameters"].shape[1] for result in family_results.values()
    )
    parameters = np.full((len(selected_families), maximum_parameters), np.nan)
    for family, result in family_results.items():
        use = selected_families == family
        parameters[use, : result["parameters"].shape[1]] = result["parameters"][use]
    output["parameters"] = parameters
    output["families"] = selected_families
    return output


def profile_metrics(
    prediction_log: np.ndarray, truth_log: np.ndarray
) -> dict[str, np.ndarray]:
    residual = prediction_log - truth_log
    middle = (RADII_KPC >= 5.0) & (RADII_KPC <= 30.0)
    prediction_shell = np.asarray(
        [
            shell_log_density(profile, RADII_KPC, allow_nonpositive_floor=True)
            for profile in prediction_log
        ]
    )
    truth_shell = np.asarray(
        [
            shell_log_density(profile, RADII_KPC, allow_nonpositive_floor=True)
            for profile in truth_log
        ]
    )
    prediction_mass = 10.0**prediction_log
    truth_mass = 10.0**truth_log
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
    shell_outer_error = np.log10(
        np.clip(prediction_mass[:, -1] - prediction_mass[:, -2], 1.0, None)
    ) - np.log10(np.clip(truth_mass[:, -1] - truth_mass[:, -2], 1.0, None))
    return {
        "full_cog_rms_dex": np.sqrt(np.mean(residual**2, axis=1)),
        "inner_cog_rms_dex": np.sqrt(np.mean(residual[:, middle] ** 2, axis=1)),
        "shell_density_rms_dex": np.sqrt(
            np.mean((prediction_shell - truth_shell) ** 2, axis=1)
        ),
        "absolute_size_error_dex": np.abs(size_error),
        "absolute_outer_shell_error_dex": np.abs(shell_outer_error),
    }


def _epoch_medians(
    values: np.ndarray, epochs: np.ndarray
) -> list[float] | list[list[float]]:
    if values.ndim == 1:
        return [float(np.nanmedian(values[epochs == epoch])) for epoch in range(5)]
    return [
        np.nanmedian(values[epochs == epoch], axis=0).astype(float).tolist()
        for epoch in range(5)
    ]


def summarize(
    prediction: np.ndarray,
    truth_log: np.ndarray,
    epochs: np.ndarray,
    diagnostics: dict[str, np.ndarray] | None = None,
) -> dict[str, object]:
    metrics = profile_metrics(prediction, truth_log)
    summary = {key: _epoch_medians(value, epochs) for key, value in metrics.items()}
    if diagnostics is not None:
        summary.update(
            {
                "median_jacobian_min_ratio": _epoch_medians(
                    diagnostics["jacobian_min_ratio"], epochs
                ),
                "boundary_fraction_within_one_percent": _epoch_medians(
                    diagnostics["boundary_distance"] < 0.01, epochs
                ),
                "median_near_parameter_spread": _epoch_medians(
                    diagnostics["near_parameter_spread"], epochs
                ),
                "maximum_near_profile_spread_dex": [
                    float(
                        np.nanmax(
                            diagnostics["near_profile_spread_dex"][epochs == epoch]
                        )
                    )
                    for epoch in range(5)
                ],
            }
        )
        summary["boundary_fraction_within_one_percent"] = [
            float(np.mean(diagnostics["boundary_distance"][epochs == epoch] < 0.01))
            for epoch in range(5)
        ]
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


def _tradeoff_figure(
    mode: str,
    redshifts: np.ndarray,
    summaries: dict[str, dict[str, object]],
) -> list[Path]:
    labels = {
        "double_sersic": "Unrestricted double Sersic",
        "cubic_logit": "Cubic-logit",
        "pca3": "PCA(3)",
        "one_break": "One-break density",
        "two_break_log_cog": "Two-break density",
        "two_break_free_sharpness": "Free-sharpness stress test",
    }
    figure, axes = plt.subplots(2, 2, figsize=(10.0, 7.0), sharex=True)
    panels = (
        ("full_cog_rms_dex", "Median full-CoG RMS (dex)"),
        ("shell_density_rms_dex", "Median shell-density RMS (dex)"),
        ("median_jacobian_min_ratio", "Median scaled Jacobian ratio"),
        ("boundary_fraction_within_one_percent", "Fraction within 1% of a bound"),
    )
    for panel, (metric, ylabel) in zip(axes.flat, panels, strict=True):
        for index, (name, label) in enumerate(labels.items()):
            if name not in summaries or metric not in summaries[name]:
                continue
            values = np.asarray(summaries[name][metric], float)
            panel.plot(
                redshifts,
                values,
                marker="o",
                linestyle=("-" if index < 3 else "--"),
                label=label,
            )
        panel.set_ylabel(ylabel)
        panel.set_xlabel("Redshift")
        if metric == "median_jacobian_min_ratio":
            panel.set_yscale("log")
        panel.grid(alpha=0.2)
    axes[0, 0].legend(ncol=2, fontsize=7)
    figure.suptitle(
        f"Exp64 {mode}: accuracy and conditioning on paired TNG CoGs\n"
        "Errors reference measured CoG-derived masses; smaller error and larger Jacobian ratio are better"
    )
    figure.tight_layout()
    paths = save_fig(figure, FIGDIR / mode / "exp64_accuracy_conditioning")
    plt.close(figure)
    return paths


def _representative_indices(
    result: dict[str, np.ndarray], truth_log: np.ndarray, halo_mass: np.ndarray
) -> list[int]:
    rms = np.sqrt(np.mean((result["prediction"] - truth_log) ** 2, axis=1))
    order = np.argsort(rms)
    finite_mass = np.flatnonzero(np.isfinite(halo_mass))
    mass_order = finite_mass[np.argsort(halo_mass[finite_mass])]
    candidates = [
        int(order[0]),
        int(order[len(order) // 2]),
        int(order[-1]),
        int(mass_order[len(mass_order) // 6]),
        int(mass_order[-1 - len(mass_order) // 6]),
    ]
    return list(dict.fromkeys(candidates))


def _direct_fit_figure(
    mode: str,
    selected: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
) -> list[Path]:
    indices = _representative_indices(
        selected, reference["truth_log"], reference["halo_mass"]
    )
    figure, axes = plt.subplots(len(indices), 3, figsize=(10.5, 2.25 * len(indices)))
    model_series = (
        (selected["prediction"], "Two-break", OKABE_ITO[2], "-"),
        (reference["prediction_double_sersic"], "Double Sersic", OKABE_ITO[1], "--"),
        (reference["prediction_cubic_logit"], "Cubic-logit", OKABE_ITO[5], ":"),
    )
    for row, index in enumerate(indices):
        truth = reference["truth_log"][index]
        axes[row, 0].plot(
            RADII_KPC, truth, color="black", marker="o", ms=2, label="TNG"
        )
        for prediction, label, color, linestyle in model_series:
            axes[row, 0].plot(
                RADII_KPC,
                prediction[index],
                color=color,
                linestyle=linestyle,
                label=label,
            )
            axes[row, 1].plot(
                RADII_KPC,
                prediction[index] - truth,
                color=color,
                linestyle=linestyle,
            )
            axes[row, 2].plot(
                RADII_KPC,
                shell_log_density(prediction[index], RADII_KPC)
                - shell_log_density(truth, RADII_KPC),
                color=color,
                linestyle=linestyle,
            )
        axes[row, 1].axhline(0.0, color="0.5", linewidth=0.7)
        axes[row, 2].axhline(0.0, color="0.5", linewidth=0.7)
        redshift = reference["redshifts"][reference["epochs"][index]]
        axes[row, 0].set_ylabel(
            f"log M(<R)\nz={redshift:g}, row={reference['rows'][index]}"
        )
        for panel in axes[row]:
            panel.set_xscale("log")
            panel.grid(alpha=0.15)
    axes[0, 0].legend(ncol=2, fontsize=7)
    axes[0, 0].set_title("Measured and fitted CoG")
    axes[0, 1].set_title("Model - TNG CoG (dex)")
    axes[0, 2].set_title("Model - TNG shell density (dex)")
    for panel in axes[-1]:
        panel.set_xlabel("Projected radius (kpc)")
    figure.suptitle(
        f"Exp64 {mode}: best, typical, worst, low-mass, and high-mass direct fits\n"
        "Reference data are the 24-point TNG CoG-derived stellar-mass profiles"
    )
    figure.tight_layout()
    paths = save_fig(figure, FIGDIR / mode / "exp64_direct_profile_fits")
    plt.close(figure)
    return paths


def run(mode: str, force: bool = False) -> dict[str, object]:
    if mode not in ("demo", "full"):
        raise ValueError("mode must be 'demo' or 'full'")
    started = time.perf_counter()
    mechanics()
    reference = load_references(mode)
    truth_log = reference["truth_log"]
    epochs = reference["epochs"]
    folds = reference["folds"]
    one_results = {}
    two_results = {}
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for family in ONE_BREAK_FAMILIES:
            one_results[family] = fit_profiles(
                mode,
                family,
                np.full(len(truth_log), family),
                "log_cog",
                truth_log,
                executor,
                force,
            )
        for family in TWO_BREAK_GRID_FAMILIES:
            two_results[family] = fit_profiles(
                mode,
                family,
                np.full(len(truth_log), family),
                "log_cog",
                truth_log,
                executor,
                force,
            )

        one_families, one_choices = _fold_selected_families(
            one_results, truth_log, epochs, folds
        )
        two_families, two_choices = _fold_selected_families(
            two_results, truth_log, epochs, folds
        )
        one_selected = _assemble_selected(one_families, one_results)
        two_selected = _assemble_selected(two_families, two_results)
        objective_results = {"log_cog": two_selected}
        for objective in ("absolute_cog", "log_shell"):
            objective_results[objective] = fit_profiles(
                mode,
                "two_break_selected",
                two_families,
                objective,
                truth_log,
                executor,
                force,
            )
        stress = fit_profiles(
            mode,
            STRESS_FAMILY,
            np.full(len(truth_log), STRESS_FAMILY),
            "log_cog",
            truth_log,
            executor,
            force,
        )

    summaries = {
        "double_sersic": summarize(
            reference["prediction_double_sersic"],
            truth_log,
            epochs,
            _reference_diagnostics(reference, "double_sersic"),
        ),
        "cubic_logit": summarize(
            reference["prediction_cubic_logit"],
            truth_log,
            epochs,
            _reference_diagnostics(reference, "cubic_logit"),
        ),
        "pca3": summarize(reference["prediction_pca3"], truth_log, epochs),
        "one_break": summarize(
            one_selected["prediction"], truth_log, epochs, one_selected
        ),
        "two_break_log_cog": summarize(
            two_selected["prediction"], truth_log, epochs, two_selected
        ),
        "two_break_absolute_cog": summarize(
            objective_results["absolute_cog"]["prediction"],
            truth_log,
            epochs,
            objective_results["absolute_cog"],
        ),
        "two_break_log_shell": summarize(
            objective_results["log_shell"]["prediction"],
            truth_log,
            epochs,
            objective_results["log_shell"],
        ),
        "two_break_free_sharpness": summarize(
            stress["prediction"], truth_log, epochs, stress
        ),
    }
    figure_paths = []
    figure_paths.extend(_tradeoff_figure(mode, reference["redshifts"], summaries))
    figure_paths.extend(_direct_fit_figure(mode, two_selected, reference))
    elapsed = time.perf_counter() - started
    stage_dir = OUTDIR / mode
    stage_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        stage_dir / "selected_predictions.npz",
        rows=reference["rows"],
        epochs=epochs,
        redshifts=reference["redshifts"],
        truth_log=truth_log,
        halo_mass=reference["halo_mass"],
        selected_one_break_family=one_families,
        selected_two_break_family=two_families,
        prediction_one_break=one_selected["prediction"],
        prediction_two_break_log_cog=two_selected["prediction"],
        prediction_two_break_absolute_cog=objective_results["absolute_cog"][
            "prediction"
        ],
        prediction_two_break_log_shell=objective_results["log_shell"]["prediction"],
        parameters_two_break_log_cog=two_selected["parameters"],
        jacobian_two_break_log_cog=two_selected["jacobian_min_ratio"],
        boundary_two_break_log_cog=two_selected["boundary_distance"],
    )
    result = {
        "mode": mode,
        "n_profile": len(truth_log),
        "wall_seconds_complete_path": elapsed,
        "subminute_gate_passed": bool(elapsed < 60.0) if mode == "demo" else None,
        "one_break_fold_choices": one_choices,
        "two_break_fold_choices": two_choices,
        "summaries": summaries,
        "figure_paths": [str(path) for path in figure_paths],
        "source_exp62_outdir": str(EXP62_OUTDIR),
    }
    (stage_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    write_manifest(
        stage_dir,
        {
            "mode": mode,
            "n_profile": len(truth_log),
            "max_workers": MAX_WORKERS,
            "source_exp62_outdir": str(EXP62_OUTDIR),
            "sharpness_grid": list(TWO_BREAK_GRID_FAMILIES),
            "objectives": list(OBJECTIVES),
        },
    )
    print(json.dumps(result, indent=2))
    if mode == "demo" and elapsed >= 60.0:
        raise RuntimeError(f"Exp64 development gate failed at {elapsed:.2f} seconds")
    return result


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "mechanics"
    if command == "mechanics":
        mechanics()
    elif command in ("demo", "full"):
        run(command, force="--force" in sys.argv[2:])
    else:
        raise SystemExit(f"unknown command {command!r}")
