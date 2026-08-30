# %%
"""Exp64 Stage 2: fit the monotone-density envelope of cubic-logit."""

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

from envelope import (  # noqa: E402
    CUBIC_LOWER,
    CUBIC_UPPER,
    base_cubic_aperture_density,
    envelope_diagnostics,
    envelope_surface_density,
    fit_envelope,
)
from families import RADII_KPC, shell_log_density  # noqa: E402
from hongshao.plotting import OKABE_ITO, save_fig, set_style  # noqa: E402
from hongshao.provenance import write_manifest  # noqa: E402
from run import (  # noqa: E402
    EXP62_OUTDIR,
    _reference_diagnostics,
    load_references,
    summarize,
)

set_style()

OUTDIR = HERE / "outputs" / "stage2_envelope"
FIGDIR = HERE / "figures" / "stage2_envelope"
OBJECTIVES = ("log_cog", "absolute_cog", "log_shell")
MAX_WORKERS = min(
    int(os.environ.get("EXP64_ENVELOPE_WORKERS", "10")), os.cpu_count() or 1
)


def _load_cubic_parameters(mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = EXP62_OUTDIR / "stage5" / mode / "predictions.npz"
    if not path.exists():
        raise FileNotFoundError(f"missing Exp62 cubic-logit record: {path}")
    data = np.load(path)
    return (
        np.asarray(data["rows"], int),
        np.asarray(data["parameters_logit_cubic5"], float),
        np.asarray(data["parameters_logit_cubic5_absolute"], float),
    )


def mechanics() -> None:
    from envelope import synthetic_envelope_parameters

    parameters = synthetic_envelope_parameters()
    radius = np.concatenate(([0.0], np.geomspace(1.0e-5, 1.0e4, 1000)))
    density = envelope_surface_density(parameters, radius, grid_size=4096)
    assert density[0] > 0.0
    assert np.all(density > 0.0)
    assert np.all(np.diff(density) <= 1.0e-10 * density[0])
    print("Exp64 Stage 2 envelope mechanics passed")


def _fit_task(task: tuple[str, np.ndarray, np.ndarray]) -> dict[str, object]:
    objective, truth_log, initial_parameters = task
    return fit_envelope(
        RADII_KPC,
        truth_log,
        objective=objective,
        initial_parameters=initial_parameters,
    )


def _fit_path(mode: str, objective: str) -> Path:
    return OUTDIR / mode / f"{objective}.npz"


def fit_objective(
    mode: str,
    objective: str,
    truth_log: np.ndarray,
    initial_parameters: np.ndarray,
    executor: ProcessPoolExecutor,
    force: bool,
) -> dict[str, np.ndarray]:
    path = _fit_path(mode, objective)
    if path.exists() and not force:
        return {key: np.asarray(value) for key, value in np.load(path).items()}
    tasks = [
        (objective, truth, initial)
        for truth, initial in zip(truth_log, initial_parameters, strict=True)
    ]
    fitted = list(executor.map(_fit_task, tasks, chunksize=max(len(tasks) // 80, 1)))
    result = {
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
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **result)
    return result


def _envelope_population_diagnostics(
    parameters: np.ndarray,
) -> dict[str, np.ndarray]:
    central_density = np.empty(len(parameters))
    affected_mass_fraction = np.empty(len(parameters))
    changed_measured_density = np.empty(len(parameters), int)
    envelope_transitions = np.empty(len(parameters), int)
    for index, values in enumerate(parameters):
        diagnostic = envelope_diagnostics(values, RADII_KPC)
        central_density[index] = diagnostic["central_density"]
        affected_mass_fraction[index] = diagnostic["affected_mass_fraction_148"]
        changed_measured_density[index] = diagnostic["n_changed_measured_density"]
        envelope_transitions[index] = diagnostic["n_envelope_transitions"]
    return {
        "central_density": central_density,
        "affected_mass_fraction_148": affected_mass_fraction,
        "n_changed_measured_density": changed_measured_density,
        "n_envelope_transitions": envelope_transitions,
    }


def _decision(
    envelope_summary: dict[str, object],
    double_summary: dict[str, object],
    cubic_summary: dict[str, object],
    result: dict[str, np.ndarray],
) -> dict[str, object]:
    accuracy_metrics = (
        "full_cog_rms_dex",
        "inner_cog_rms_dex",
        "shell_density_rms_dex",
    )
    accuracy_pass = all(
        np.all(
            np.asarray(envelope_summary[metric])
            <= 1.10 * np.asarray(double_summary[metric])
        )
        for metric in accuracy_metrics
    )
    envelope_size = np.asarray(envelope_summary["absolute_size_error_dex"])
    double_size = np.asarray(double_summary["absolute_size_error_dex"])
    size_pass = bool(np.all(envelope_size[:, 1:] <= 1.10 * double_size[:, 1:]))
    conditioning_epoch = np.asarray(
        envelope_summary["median_jacobian_min_ratio"]
    ) >= 0.5 * np.asarray(cubic_summary["median_jacobian_min_ratio"])
    boundary_fraction = float(np.mean(result["boundary_distance"] < 0.01))
    near_parameter_pass = bool(
        np.all(np.asarray(envelope_summary["median_near_parameter_spread"]) < 0.01)
    )
    near_profile_pass = bool(
        np.all(np.asarray(envelope_summary["maximum_near_profile_spread_dex"]) < 0.001)
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


def _validation_mask(
    mode: str,
    rows: np.ndarray,
    epochs: np.ndarray,
) -> np.ndarray:
    if mode == "demo":
        return np.ones(len(rows), bool)
    development_path = OUTDIR / "demo" / "envelope_results.npz"
    if not development_path.exists():
        raise FileNotFoundError("run the Stage 2 envelope demo before full validation")
    development = np.load(development_path)
    development_pairs = set(
        zip(
            np.asarray(development["rows"], int).tolist(),
            np.asarray(development["epochs"], int).tolist(),
            strict=True,
        )
    )
    return np.asarray(
        [
            (int(row), int(epoch)) not in development_pairs
            for row, epoch in zip(rows, epochs, strict=True)
        ],
        bool,
    )


def _accuracy_figure(
    mode: str,
    redshifts: np.ndarray,
    summaries: dict[str, dict[str, object]],
) -> list[Path]:
    labels = {
        "double_sersic": "Unrestricted double Sersic",
        "cubic_logit": "Original cubic-logit",
        "envelope_log_cog": "Envelope, log-CoG fit",
        "envelope_absolute_cog": "Envelope, absolute-CoG fit",
        "envelope_log_shell": "Envelope, log-shell fit",
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
            summary = summaries[name]
            if metric not in summary:
                continue
            panel.plot(
                redshifts,
                np.asarray(summary[metric]),
                marker="o",
                linestyle=("-" if index < 2 else "--"),
                label=label,
            )
        panel.set_xlabel("Redshift")
        panel.set_ylabel(ylabel)
        panel.grid(alpha=0.2)
        if metric == "median_jacobian_min_ratio":
            panel.set_yscale("log")
    axes[0, 0].legend(ncol=2, fontsize=7)
    figure.suptitle(
        f"Exp64 Stage 2 {mode}: no-hole envelope accuracy and conditioning\n"
        "Errors reference measured TNG CoGs; smaller error and larger Jacobian ratio are better"
    )
    figure.tight_layout()
    paths = save_fig(figure, FIGDIR / mode / "exp64_envelope_accuracy_conditioning")
    plt.close(figure)
    return paths


def _representative_indices(
    prediction: np.ndarray,
    truth_log: np.ndarray,
    halo_mass: np.ndarray,
) -> list[int]:
    rms = np.sqrt(np.mean((prediction - truth_log) ** 2, axis=1))
    order = np.argsort(rms)
    finite = np.flatnonzero(np.isfinite(halo_mass))
    mass_order = finite[np.argsort(halo_mass[finite])]
    return list(
        dict.fromkeys(
            [
                int(order[0]),
                int(order[len(order) // 2]),
                int(order[-1]),
                int(mass_order[len(mass_order) // 6]),
                int(mass_order[-1 - len(mass_order) // 6]),
            ]
        )
    )


def _direct_figure(
    mode: str,
    reference: dict[str, np.ndarray],
    result: dict[str, np.ndarray],
) -> list[Path]:
    indices = _representative_indices(
        result["prediction"], reference["truth_log"], reference["halo_mass"]
    )
    figure, axes = plt.subplots(len(indices), 3, figsize=(10.5, 2.3 * len(indices)))
    inner_radius = np.geomspace(1.0e-3, RADII_KPC[-1], 500)
    for row, index in enumerate(indices):
        truth = reference["truth_log"][index]
        parameters = result["parameters"][index]
        axes[row, 0].plot(
            inner_radius,
            base_cubic_aperture_density(parameters, inner_radius),
            color=OKABE_ITO[5],
            linestyle=":",
            label="Original cubic-logit",
        )
        axes[row, 0].plot(
            inner_radius,
            envelope_surface_density(parameters, inner_radius),
            color=OKABE_ITO[2],
            label="Monotone envelope",
        )
        axes[row, 0].axvline(2.0, color="0.5", linewidth=0.8)
        axes[row, 0].set_yscale("log")
        model_series = (
            (result["prediction"], "Envelope", OKABE_ITO[2], "-"),
            (
                reference["prediction_double_sersic"],
                "Double Sersic",
                OKABE_ITO[1],
                "--",
            ),
            (reference["prediction_cubic_logit"], "Cubic-logit", OKABE_ITO[5], ":"),
        )
        for prediction, label, color, linestyle in model_series:
            axes[row, 1].plot(
                RADII_KPC,
                prediction[index] - truth,
                color=color,
                linestyle=linestyle,
                label=label,
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
            f"Surface density\nz={redshift:g}, row={reference['rows'][index]}"
        )
        for panel in axes[row]:
            panel.set_xscale("log")
            panel.grid(alpha=0.15)
    axes[0, 0].set_title("Central density repair; 2 kpc limit shown")
    axes[0, 1].set_title("Model - TNG CoG (dex)")
    axes[0, 2].set_title("Model - TNG shell density (dex)")
    axes[0, 0].legend(fontsize=7)
    axes[0, 1].legend(ncol=2, fontsize=7)
    for panel in axes[-1]:
        panel.set_xlabel("Projected radius (kpc)")
    figure.suptitle(
        f"Exp64 Stage 2 {mode}: best, typical, worst, low-mass, and high-mass fits\n"
        "The envelope is the least non-increasing density above the original cubic-logit density"
    )
    figure.tight_layout()
    paths = save_fig(figure, FIGDIR / mode / "exp64_envelope_direct_profiles")
    plt.close(figure)
    return paths


def run(mode: str, force: bool = False) -> dict[str, object]:
    if mode not in ("demo", "full"):
        raise ValueError("mode must be 'demo' or 'full'")
    fit_cache_used = not force and all(
        _fit_path(mode, objective).exists() for objective in OBJECTIVES
    )
    started = time.perf_counter()
    mechanics()
    reference = load_references(mode)
    rows, log_initial, absolute_initial = _load_cubic_parameters(mode)
    if not np.array_equal(rows, reference["rows"]):
        raise ValueError("Exp62 cubic-logit parameter rows do not match references")
    initial_by_objective = {
        "log_cog": log_initial,
        "absolute_cog": absolute_initial,
        "log_shell": log_initial,
    }
    fitted = {}
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for objective in OBJECTIVES:
            fitted[objective] = fit_objective(
                mode,
                objective,
                reference["truth_log"],
                initial_by_objective[objective],
                executor,
                force,
            )

    validation = _validation_mask(mode, reference["rows"], reference["epochs"])
    validation_truth = reference["truth_log"][validation]
    validation_epochs = reference["epochs"][validation]
    summaries = {
        "double_sersic": summarize(
            reference["prediction_double_sersic"][validation],
            validation_truth,
            validation_epochs,
            {
                key: value[validation]
                for key, value in _reference_diagnostics(
                    reference, "double_sersic"
                ).items()
            },
        ),
        "cubic_logit": summarize(
            reference["prediction_cubic_logit"][validation],
            validation_truth,
            validation_epochs,
            {
                key: value[validation]
                for key, value in _reference_diagnostics(
                    reference, "cubic_logit"
                ).items()
            },
        ),
    }
    for objective, result in fitted.items():
        summaries[f"envelope_{objective}"] = summarize(
            result["prediction"][validation],
            validation_truth,
            validation_epochs,
            {
                key: value[validation]
                for key, value in result.items()
                if key
                in (
                    "jacobian_min_ratio",
                    "boundary_distance",
                    "near_parameter_spread",
                    "near_profile_spread_dex",
                )
            },
        )
    population_diagnostics = _envelope_population_diagnostics(
        fitted["log_cog"]["parameters"]
    )
    normalized_displacement = np.abs(fitted["log_cog"]["parameters"] - log_initial) / (
        CUBIC_UPPER - CUBIC_LOWER
    )
    decision = _decision(
        summaries["envelope_log_cog"],
        summaries["double_sersic"],
        summaries["cubic_logit"],
        {
            key: value[validation]
            for key, value in fitted["log_cog"].items()
            if key
            in (
                "jacobian_min_ratio",
                "boundary_distance",
                "near_parameter_spread",
                "near_profile_spread_dex",
            )
        },
    )
    figure_paths = []
    figure_paths.extend(_accuracy_figure(mode, reference["redshifts"], summaries))
    figure_paths.extend(_direct_figure(mode, reference, fitted["log_cog"]))
    elapsed = time.perf_counter() - started
    output_directory = OUTDIR / mode
    output_directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_directory / "envelope_results.npz",
        rows=reference["rows"],
        epochs=reference["epochs"],
        redshifts=reference["redshifts"],
        truth_log=reference["truth_log"],
        halo_mass=reference["halo_mass"],
        **{
            f"prediction_{objective}": result["prediction"]
            for objective, result in fitted.items()
        },
        **{
            f"parameters_{objective}": result["parameters"]
            for objective, result in fitted.items()
        },
        **population_diagnostics,
        normalized_parameter_displacement=normalized_displacement,
    )
    result_summary = {
        "mode": mode,
        "n_profile": len(reference["truth_log"]),
        "n_validation_profile": int(np.count_nonzero(validation)),
        "wall_seconds_complete_path": None if fit_cache_used else elapsed,
        "wall_seconds_cached_refresh": elapsed if fit_cache_used else None,
        "subminute_gate_passed": (
            bool(elapsed < 60.0) if mode == "demo" and not fit_cache_used else None
        ),
        "summaries": summaries,
        "decision": decision,
        "envelope_diagnostics": {
            "median_affected_mass_fraction_148": float(
                np.median(population_diagnostics["affected_mass_fraction_148"])
            ),
            "maximum_affected_mass_fraction_148": float(
                np.max(population_diagnostics["affected_mass_fraction_148"])
            ),
            "median_changed_measured_density_points": float(
                np.median(population_diagnostics["n_changed_measured_density"])
            ),
            "median_envelope_transitions": float(
                np.median(population_diagnostics["n_envelope_transitions"])
            ),
            "median_maximum_normalized_parameter_displacement": float(
                np.median(np.max(normalized_displacement, axis=1))
            ),
        },
        "figure_paths": [str(path) for path in figure_paths],
    }
    (output_directory / "summary.json").write_text(
        json.dumps(result_summary, indent=2) + "\n"
    )
    write_manifest(
        output_directory,
        {
            "mode": mode,
            "n_profile": len(reference["truth_log"]),
            "max_workers": MAX_WORKERS,
            "objectives": list(OBJECTIVES),
            "source_exp62_outdir": str(EXP62_OUTDIR),
        },
    )
    print(json.dumps(result_summary, indent=2))
    if mode == "demo" and elapsed >= 60.0:
        raise RuntimeError(
            f"Exp64 Stage 2 development gate failed at {elapsed:.2f} seconds"
        )
    return result_summary


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "mechanics"
    if command == "mechanics":
        mechanics()
    elif command in ("demo", "full"):
        run(command, force="--force" in sys.argv[2:])
    else:
        raise SystemExit(f"unknown command {command!r}")
