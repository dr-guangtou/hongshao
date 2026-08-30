# %%
"""Exp70 population-wide damping experiment driver."""

from __future__ import annotations

import argparse
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
from scipy.integrate import cumulative_trapezoid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP66_DIR = HERE.parent / "exp66_expanded_symbolic_density"
EXP67_DIR = HERE.parent / "exp67_squared_slope_symbolic_density"
EXP69_DIR = HERE.parent / "exp69_damped_cosine_reparameterization"
for path in (ROOT, HERE, EXP66_DIR, EXP67_DIR, EXP69_DIR):
    sys.path.insert(0, str(path))

os.environ.setdefault(
    "HONGSHAO_EXP62_ROOT",
    "/Users/shuang/Dropbox/work/project/massive/hongshao_master",
)

from exp66_search import load_references, profile_metrics  # noqa: E402
from exp67_search import population_split  # noqa: E402
from exp69_diagnostics import continuity_metrics, weakest_direction_path  # noqa: E402
from exp69_family import (  # noqa: E402
    CandidateSpec,
    cog_log,
    descriptor_bounds,
    fit_profile,
    integrated_density_rate,
    logarithmic_density_rate,
    physical_descriptors,
    synthetic_parameters,
)
from exp70_family import (  # noqa: E402
    RADII_KPC,
    all_specs,
    free_reference_spec,
    shared_damping_specs,
)
from hongshao.plotting import OKABE_ITO, save_fig, set_style  # noqa: E402
from hongshao.provenance import write_manifest  # noqa: E402

set_style()
plt.rcParams["text.usetex"] = False

OUTDIR = HERE / "outputs"
FIGDIR = HERE / "figures"
MAX_WORKERS = min(int(os.environ.get("EXP70_WORKERS", "10")), os.cpu_count() or 1)
ARCHIVE_RELATIVE_PATH = (
    Path("experiments")
    / "exp67_squared_slope_symbolic_density"
    / "outputs"
    / "discovery_round_a"
    / "expressions"
    / "free_damped_cosine.npz"
)
STAGE_SETTINGS = {
    "gate": (4, 128, 180),
    "discovery": (8, 256, 350),
    "selection": (8, 256, 350),
    "validation": (8, 256, 350),
}


def _exp67_root() -> Path:
    configured = os.environ.get("HONGSHAO_EXP67_ROOT")
    if not configured:
        raise RuntimeError(
            "HONGSHAO_EXP67_ROOT must point to the preserved Exp67 worktree"
        )
    return Path(configured).resolve()


def load_reference_data() -> dict[str, np.ndarray]:
    """Load references and authenticate only the preserved discovery archive."""
    archive_path = _exp67_root() / ARCHIVE_RELATIVE_PATH
    if not archive_path.exists():
        raise FileNotFoundError(
            f"missing preserved Exp67 discovery archive: {archive_path}"
        )
    reference = load_references("full")
    discovery = np.asarray(population_split(reference)["discovery"], int)
    with np.load(archive_path) as archive:
        archived_indices = np.asarray(archive["indices"], int)
    if not np.array_equal(archived_indices, discovery):
        raise ValueError("preserved Exp67 archive does not contain discovery only")
    reference["exp67_discovery_archive"] = np.asarray(str(archive_path))
    return reference


def stage_indices(reference: dict[str, np.ndarray], stage: str) -> np.ndarray:
    """Return one explicitly authorized frozen population stage."""
    if stage not in {"discovery", "selection", "validation"}:
        raise ValueError(f"unknown population stage: {stage}")
    return np.asarray(population_split(reference)[stage], int)


def operational_gate_indices(
    reference: dict[str, np.ndarray], discovery: np.ndarray
) -> np.ndarray:
    """Return the frozen 25-galaxy discovery-only operational sample."""
    epoch_zero = discovery[reference["epochs"][discovery] == 0]
    masses = reference["halo_mass"][epoch_zero]
    finite = epoch_zero[np.isfinite(masses)]
    missing = epoch_zero[~np.isfinite(masses)]
    finite = finite[np.argsort(reference["halo_mass"][finite], kind="stable")]
    positions = np.rint(np.linspace(0, len(finite) - 1, 20)).astype(int)
    missing_selected = np.random.default_rng(670).choice(missing, size=5, replace=False)
    rows = np.concatenate(
        (reference["rows"][finite[positions]], reference["rows"][missing_selected])
    )
    selected = np.flatnonzero(np.isin(reference["rows"], rows))
    if not set(selected.tolist()) <= set(discovery.tolist()):
        raise RuntimeError("operational sample escaped the discovery population")
    if len(selected) != 125 or len(np.unique(reference["rows"][selected])) != 25:
        raise RuntimeError("operational sample is not 25 complete galaxies")
    return selected


def _fit_task(
    payload: tuple[CandidateSpec, np.ndarray, str, int, int, int],
) -> dict[str, object]:
    spec, truth, objective, grid_size, n_starts, max_nfev = payload
    return fit_profile(
        spec,
        RADII_KPC,
        truth,
        objective=objective,
        grid_size=grid_size,
        n_starts=n_starts,
        max_nfev=max_nfev,
    )


def _fit_indexed_task(
    payload: tuple[str, int, CandidateSpec, np.ndarray, str, int, int, int],
) -> tuple[str, int, dict[str, object]]:
    name, index, spec, truth, objective, grid_size, n_starts, max_nfev = payload
    result = _fit_task((spec, truth, objective, grid_size, n_starts, max_nfev))
    return name, index, result


def _pack_fits(fitted: list[dict[str, object]]) -> dict[str, np.ndarray]:
    keys = (
        "parameters",
        "prediction",
        "objective_rms",
        "cog_rms_dex",
        "success",
        "nfev",
        "jacobian_min_ratio",
        "singular_values",
        "weakest_direction",
        "boundary_distance",
        "near_solution_count",
        "near_parameter_spread",
        "near_profile_spread_dex",
        "all_scores",
        "all_parameters",
        "all_predictions",
        "all_success",
        "all_singular_minimum",
        "all_singular_values",
        "all_descriptors",
    )
    return {key: np.asarray([item[key] for item in fitted]) for key in keys}


def _fit_specs_flat(
    specs: dict[str, CandidateSpec],
    truth: np.ndarray,
    *,
    objective: str,
    n_starts: int,
    grid_size: int,
    max_nfev: int,
) -> dict[str, dict[str, np.ndarray]]:
    payloads = [
        (name, index, spec, profile, objective, grid_size, n_starts, max_nfev)
        for name, spec in specs.items()
        for index, profile in enumerate(truth)
    ]
    collected: dict[str, list[dict[str, object] | None]] = {
        name: [None] * len(truth) for name in specs
    }
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for name, index, result in executor.map(
            _fit_indexed_task, payloads, chunksize=2
        ):
            collected[name][index] = result
    return {
        name: _pack_fits([item for item in values if item is not None])
        for name, values in collected.items()
    }


def _archive_path(stage: str, objective: str, name: str) -> Path:
    return OUTDIR / stage / objective / f"{name}.npz"


def _fit_stage_checkpointed(
    specs: dict[str, CandidateSpec],
    reference: dict[str, np.ndarray],
    indices: np.ndarray,
    *,
    stage: str,
    objective: str,
    force: bool,
) -> dict[str, dict[str, np.ndarray]]:
    n_starts, grid_size, max_nfev = STAGE_SETTINGS[stage]
    results = {}
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for number, (name, spec) in enumerate(specs.items(), start=1):
            path = _archive_path(stage, objective, name)
            started = time.perf_counter()
            if path.exists() and not force:
                with np.load(path) as archive:
                    archived_indices = np.asarray(archive["indices"], int)
                    if not np.array_equal(archived_indices, indices):
                        raise ValueError(f"{path} contains the wrong population stage")
                    result = {
                        key: np.asarray(archive[key])
                        for key in archive.files
                        if key != "indices"
                    }
            else:
                payloads = (
                    (spec, profile, objective, grid_size, n_starts, max_nfev)
                    for profile in reference["truth_log"][indices]
                )
                result = _pack_fits(
                    list(executor.map(_fit_task, payloads, chunksize=4))
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, indices=indices, **result)
            results[name] = result
            print(
                f"[{stage}/{objective}] {number}/{len(specs)} {name}: "
                f"{time.perf_counter() - started:.2f} s; median CoG RMS "
                f"{np.median(result['cog_rms_dex']):.5f} dex",
                flush=True,
            )
    return results


def _common_shape_descriptors(
    spec: CandidateSpec, parameters: np.ndarray
) -> np.ndarray:
    """Return the same five normalized shape coordinates for every system."""
    physical = np.asarray(
        [physical_descriptors(spec, parameter) for parameter in parameters], float
    )
    # physical = log mass, log core radius, a^2, pivot contrast, damping, phase.
    selected = physical[:, [1, 2, 3, 5]]
    signed_contrast = np.asarray(parameters)[
        :, spec.parameter_names.index("signed_contrast")
    ]
    values = np.column_stack((selected, signed_contrast))
    common_lower, common_upper = descriptor_bounds(free_reference_spec())
    lower = np.concatenate((common_lower[[1, 2, 3, 5]], [-20.0]))
    upper = np.concatenate((common_upper[[1, 2, 3, 5]], [20.0]))
    return (values - lower) / (upper - lower)


def _epoch_medians(values: np.ndarray, epochs: np.ndarray) -> list[float]:
    return [float(np.nanmedian(values[epochs == epoch])) for epoch in range(5)]


def _continuity(
    spec: CandidateSpec,
    result: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    indices: np.ndarray,
) -> dict[str, object]:
    return continuity_metrics(
        _common_shape_descriptors(spec, result["parameters"]),
        reference["truth_log"][indices],
        reference["rows"][indices],
        reference["epochs"][indices],
    )


def _candidate_record(
    spec: CandidateSpec,
    result: dict[str, np.ndarray],
    free_result: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    indices: np.ndarray,
) -> dict[str, object]:
    truth = reference["truth_log"][indices]
    epochs = reference["epochs"][indices]
    metrics = profile_metrics(result["prediction"], truth)
    free_metrics = profile_metrics(free_result["prediction"], truth)
    continuity = _continuity(spec, result, reference, indices)
    free_continuity = _continuity(
        free_reference_spec(), free_result, reference, indices
    )

    def ratio(key: str) -> list[float]:
        numerator = np.asarray(_epoch_medians(metrics[key], epochs))
        denominator = np.asarray(_epoch_medians(free_metrics[key], epochs))
        return (numerator / np.maximum(denominator, 1.0e-12)).tolist()

    jacobian = np.asarray(_epoch_medians(result["jacobian_min_ratio"], epochs))
    cubic = np.asarray(
        _epoch_medians(reference["jacobian_cubic_logit"][indices], epochs)
    )
    free_jacobian = np.asarray(
        _epoch_medians(free_result["jacobian_min_ratio"], epochs)
    )
    return {
        "name": spec.name,
        "fixed_damping": spec.fixed_damping,
        "success_fraction": float(np.mean(result["success"])),
        "boundary_fraction": float(np.mean(result["boundary_distance"] < 0.01)),
        "maximum_near_profile_spread_dex": float(
            np.max(result["near_profile_spread_dex"])
        ),
        "full_cog_rms_dex_by_epoch": _epoch_medians(
            metrics["full_cog_rms_dex"], epochs
        ),
        "middle_cog_rms_dex_by_epoch": _epoch_medians(
            metrics["middle_cog_rms_dex"], epochs
        ),
        "shell_rms_dex_by_epoch": _epoch_medians(
            metrics["shell_density_rms_dex"], epochs
        ),
        "full_cog_ratio_to_free_by_epoch": ratio("full_cog_rms_dex"),
        "middle_cog_ratio_to_free_by_epoch": ratio("middle_cog_rms_dex"),
        "shell_ratio_to_free_by_epoch": ratio("shell_density_rms_dex"),
        "jacobian_ratio_to_cubic_by_epoch": (
            jacobian / np.maximum(cubic, 1.0e-16)
        ).tolist(),
        "jacobian_factor_to_free_by_epoch": (
            jacobian / np.maximum(free_jacobian, 1.0e-16)
        ).tolist(),
        "nearest_continuity_median": float(continuity["nearest_median"]),
        "nearest_continuity_p95": float(continuity["nearest_p95"]),
        "epoch_continuity_median": float(continuity["epoch_median"]),
        "epoch_continuity_p95": float(continuity["epoch_p95"]),
        "nearest_continuity_ratio_to_free": float(
            continuity["nearest_p95"]
            / max(float(free_continuity["nearest_p95"]), 1e-16)
        ),
        "epoch_continuity_ratio_to_free": float(
            continuity["epoch_p95"] / max(float(free_continuity["epoch_p95"]), 1e-16)
        ),
        "nearest_continuity_median_ratio_to_free": float(
            continuity["nearest_median"]
            / max(float(free_continuity["nearest_median"]), 1e-16)
        ),
        "epoch_continuity_median_ratio_to_free": float(
            continuity["epoch_median"]
            / max(float(free_continuity["epoch_median"]), 1e-16)
        ),
    }


def candidate_is_eligible(record: dict[str, object], stage: str) -> bool:
    """Apply the frozen discovery or selection gate without ranking."""
    if stage not in {"discovery", "selection"}:
        raise ValueError("eligibility exists only for discovery and selection")
    boundary_limit = 0.05 if stage == "discovery" else 0.01
    near_limit = 0.003 if stage == "discovery" else 0.001
    jacobian = np.asarray(record["jacobian_ratio_to_cubic_by_epoch"], float)
    return bool(
        float(record["success_fraction"]) >= 0.999
        and float(record["boundary_fraction"]) < boundary_limit
        and float(record["maximum_near_profile_spread_dex"]) < near_limit
        and np.all(jacobian >= 0.15)
        and np.count_nonzero(jacobian >= 0.25) >= 4
        and np.all(np.asarray(record["jacobian_factor_to_free_by_epoch"]) >= 4.0)
        and np.all(np.asarray(record["full_cog_ratio_to_free_by_epoch"]) <= 1.10)
        and np.all(np.asarray(record["middle_cog_ratio_to_free_by_epoch"]) <= 1.10)
        and np.all(np.asarray(record["shell_ratio_to_free_by_epoch"]) <= 1.10)
        and float(record["nearest_continuity_ratio_to_free"]) <= 0.75
        and float(record["epoch_continuity_ratio_to_free"]) <= 0.75
        and float(record["nearest_continuity_median_ratio_to_free"]) <= 1.0
        and float(record["epoch_continuity_median_ratio_to_free"]) <= 1.0
    )


def choose_global_damping(records: list[dict[str, object]]) -> dict[str, object]:
    """Return the frozen lexicographic discovery winner."""
    eligible = [
        record for record in records if candidate_is_eligible(record, "discovery")
    ]
    if not eligible:
        raise ValueError("no fixed damping value passes every discovery gate")
    return min(
        eligible,
        key=lambda record: (
            max(record["full_cog_ratio_to_free_by_epoch"]),
            max(record["shell_ratio_to_free_by_epoch"]),
            max(
                float(record["nearest_continuity_ratio_to_free"]),
                float(record["epoch_continuity_ratio_to_free"]),
            ),
            float(record["fixed_damping"]),
        ),
    )


def mechanics() -> dict[str, object]:
    """Run the frozen twenty-case analytic, numerical, and recovery audit."""
    records = []
    for name, spec in all_specs().items():
        for case_index in range(4):
            parameters = synthetic_parameters(spec, case_index)
            coordinate = np.linspace(0.0, 9.0, 60001)
            exact = integrated_density_rate(spec, parameters, coordinate)
            numerical = cumulative_trapezoid(
                logarithmic_density_rate(spec, parameters, coordinate),
                coordinate,
                initial=0.0,
            )
            truth = cog_log(spec, parameters, RADII_KPC, grid_size=2048)
            coarse = cog_log(spec, parameters, RADII_KPC, grid_size=512)
            fitted = fit_profile(
                spec,
                RADII_KPC,
                truth,
                n_starts=4,
                max_nfev=600,
                grid_size=512,
                synthetic_truth_parameters=parameters,
            )
            truth_descriptor = physical_descriptors(spec, parameters)
            fit_descriptor = physical_descriptors(spec, fitted["parameters"])
            descriptor_lower, descriptor_upper = descriptor_bounds(spec)
            descriptor_scale = np.maximum(descriptor_upper - descriptor_lower, 1.0e-12)
            records.append(
                {
                    "name": name,
                    "case_index": case_index,
                    "integral_maximum_difference": float(
                        np.max(np.abs(exact - numerical))
                    ),
                    "quadrature_maximum_difference_dex": float(
                        np.max(np.abs(truth - coarse))
                    ),
                    "recovery_cog_rms_dex": float(fitted["cog_rms_dex"]),
                    "recovery_descriptor_error": float(
                        np.max(
                            np.abs(fit_descriptor - truth_descriptor) / descriptor_scale
                        )
                    ),
                    "strictly_increasing_cog": bool(np.all(np.diff(truth) > 0.0)),
                    "positive_density_rate": bool(
                        np.all(
                            logarithmic_density_rate(spec, parameters, coordinate)
                            >= 0.0
                        )
                    ),
                    "success": bool(fitted["success"]),
                }
            )
    passed = all(
        item["integral_maximum_difference"] < 3e-7
        and item["quadrature_maximum_difference_dex"] < 4e-4
        and item["recovery_cog_rms_dex"] < 3e-5
        and item["recovery_descriptor_error"] < 0.03
        and item["strictly_increasing_cog"]
        and item["positive_density_rate"]
        and item["success"]
        for item in records
    )
    return {"passed": passed, "records": records}


def _save(figure: plt.Figure, stage: str, stem: str) -> list[Path]:
    paths = save_fig(figure, FIGDIR / stage / stem)
    plt.close(figure)
    return paths


def _plot_synthetic(mechanical: dict[str, object], stage: str) -> list[Path]:
    names = tuple(all_specs())
    records = mechanical["records"]
    arrays = []
    for metric in ("recovery_cog_rms_dex", "recovery_descriptor_error"):
        arrays.append(
            np.asarray(
                [
                    [
                        next(
                            item[metric]
                            for item in records
                            if item["name"] == name and item["case_index"] == case
                        )
                        for case in range(4)
                    ]
                    for name in names
                ]
            )
        )
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    for axis, values, title in zip(
        axes,
        arrays,
        ("Noiseless CoG recovery [dex]", "Descriptor error [range fraction]"),
        strict=True,
    ):
        image = axis.imshow(values, aspect="auto", cmap="cividis")
        axis.set_yticks(range(len(names)), [all_specs()[name].label for name in names])
        axis.set_xticks(range(4), [f"Case {index + 1}" for index in range(4)])
        axis.set_title(title)
        figure.colorbar(image, ax=axis)
    figure.suptitle("Twenty predeclared synthetic recovery cases")
    return _save(figure, stage, "synthetic_recovery")


def _plot_trade(records: list[dict[str, object]], stage: str) -> list[Path]:
    values = np.asarray([record["fixed_damping"] for record in records], float)
    figure, axes = plt.subplots(2, 4, figsize=(14.5, 7.0), constrained_layout=True)
    series = (
        ("full_cog_ratio_to_free_by_epoch", "Worst epoch CoG RMS / free", "max"),
        ("shell_ratio_to_free_by_epoch", "Worst epoch shell RMS / free", "max"),
        ("jacobian_factor_to_free_by_epoch", "Weakest epoch Jacobian / free", "min"),
        ("jacobian_ratio_to_cubic_by_epoch", "Weakest epoch Jacobian / cubic", "min"),
        ("success_fraction", "Converged fraction", None),
        ("boundary_fraction", "Near-bound fraction", None),
        (
            "maximum_near_profile_spread_dex",
            "Maximum near-solution CoG spread [dex]",
            None,
        ),
        ("epoch_continuity_ratio_to_free", "Adjacent-epoch p95 / free", None),
    )
    for axis, (key, label, reducer) in zip(axes.ravel(), series, strict=True):
        raw = [record[key] for record in records]
        y = np.asarray(
            [
                max(item)
                if reducer == "max"
                else min(item)
                if reducer == "min"
                else item
                for item in raw
            ],
            float,
        )
        axis.plot(values, y, "o-", color=OKABE_ITO[0])
        axis.set_xlabel(r"Shared damping $\lambda$")
        axis.set_ylabel(label)
    figure.suptitle(
        "Population-wide damping trade relative to newly refit free damping"
    )
    return _save(figure, stage, "shared_damping_trade")


def _representative_indices(
    results: dict[str, dict[str, np.ndarray]], truth: np.ndarray
) -> tuple[int, int, int]:
    errors = np.sqrt(
        np.mean((results["free_reference"]["prediction"] - truth) ** 2, axis=1)
    )
    order = np.argsort(errors)
    return int(order[0]), int(order[len(order) // 2]), int(order[-1])


def _plot_multi_start(
    results: dict[str, dict[str, np.ndarray]],
    reference: dict[str, np.ndarray],
    indices: np.ndarray,
    stage: str,
) -> list[Path]:
    truth = reference["truth_log"][indices]
    chosen = _representative_indices(results, truth)
    figure, axes = plt.subplots(3, 2, figsize=(10.5, 8.6), sharex=True)
    for row, profile_index in enumerate(chosen):
        axes[row, 0].plot(RADII_KPC, truth[profile_index], "o", color="black", ms=3)
        axes[row, 1].axhline(0.0, color="0.6", lw=0.8)
        for color_index, (name, result) in enumerate(results.items()):
            color = OKABE_ITO[color_index % len(OKABE_ITO)]
            scores = result["all_scores"][profile_index]
            near = scores <= 1.01 * np.min(scores) + 1e-14
            for start, prediction in enumerate(
                result["all_predictions"][profile_index]
            ):
                alpha = 0.7 if near[start] else 0.12
                axes[row, 0].plot(
                    RADII_KPC, prediction, color=color, alpha=alpha, lw=0.9
                )
                axes[row, 1].plot(
                    RADII_KPC,
                    prediction - truth[profile_index],
                    color=color,
                    alpha=alpha,
                    lw=0.9,
                )
        axes[row, 0].set_ylabel(("Best", "Typical", "Worst")[row] + "\nlog M(<R)")
        axes[row, 1].set_ylabel("Residual [dex]")
    for axis in axes.ravel():
        axis.set_xscale("log")
    for axis in axes[-1]:
        axis.set_xlabel("Projected radius [kpc]")
    handles = [
        plt.Line2D([], [], color=OKABE_ITO[index], label=spec.label)
        for index, spec in enumerate(all_specs().values())
    ]
    figure.suptitle(
        "All starts for common best, typical, and worst free-family fits", y=0.995
    )
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=3,
        fontsize=7,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.89))
    return _save(figure, stage, "multi_start_solution_overlays")


def _plot_weakest(
    results: dict[str, dict[str, np.ndarray]],
    reference: dict[str, np.ndarray],
    indices: np.ndarray,
    stage: str,
) -> list[Path]:
    truth = reference["truth_log"][indices]
    figure, axes = plt.subplots(
        1, len(results), figsize=(15.0, 3.5), sharex=True, sharey=True
    )
    for axis, (name, result) in zip(axes, results.items(), strict=True):
        spec = all_specs()[name]
        profile_index = int(np.argmin(result["jacobian_min_ratio"]))
        path = weakest_direction_path(
            result["parameters"][profile_index],
            result["weakest_direction"][profile_index],
            np.asarray(spec.lower),
            np.asarray(spec.upper),
            n_point=21,
        )
        for path_index, parameter in enumerate(path):
            prediction = cog_log(spec, parameter, RADII_KPC, grid_size=128)
            axis.plot(
                RADII_KPC,
                prediction - truth[profile_index],
                color=plt.cm.cividis(path_index / 20),
                alpha=0.8,
            )
        axis.axhline(0.0, color="0.6", lw=0.8)
        axis.set_xscale("log")
        axis.set_title(spec.label, fontsize=9)
        axis.set_xlabel("R [kpc]")
    axes[0].set_ylabel("CoG residual [dex]")
    figure.suptitle(
        "Decoded profiles along each system's weakest scaled-Jacobian direction",
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    return _save(figure, stage, "weakest_jacobian_directions")


def _plot_continuity(
    results: dict[str, dict[str, np.ndarray]],
    reference: dict[str, np.ndarray],
    indices: np.ndarray,
    stage: str,
) -> list[Path]:
    truth = reference["truth_log"][indices]
    diagnostics = {
        name: _continuity(all_specs()[name], result, reference, indices)
        for name, result in results.items()
    }
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), constrained_layout=True)
    for color_index, (name, diagnostic) in enumerate(diagnostics.items()):
        for axis, key in (
            (axes[0, 0], "nearest_distance"),
            (axes[0, 1], "epoch_distance"),
        ):
            distribution = np.sort(diagnostic[key])
            axis.plot(
                distribution,
                np.linspace(0.0, 1.0, len(distribution)),
                color=OKABE_ITO[color_index],
                label=all_specs()[name].label,
            )
    axes[0, 0].set_xlabel("Nearest-profile descriptor separation")
    axes[0, 1].set_xlabel("Adjacent-epoch descriptor separation")
    for axis in axes[0]:
        axis.set_ylabel("Cumulative fraction")
    axes[0, 0].legend(fontsize=7)
    free = diagnostics["free_reference"]
    for axis, pairs_key, distance_key, title in (
        (
            axes[1, 0],
            "nearest_pairs",
            "nearest_distance",
            "Largest nearest-profile jump",
        ),
        (axes[1, 1], "epoch_pairs", "epoch_distance", "Largest adjacent-epoch jump"),
    ):
        pair = free[pairs_key][int(np.argmax(free[distance_key]))]
        for profile_index, marker in zip(pair, ("o", "s"), strict=True):
            axis.plot(
                RADII_KPC,
                truth[profile_index] - truth[profile_index, -1],
                marker=marker,
                label=f"Profile {profile_index}",
            )
        axis.set_xscale("log")
        axis.set_xlabel("Projected radius [kpc]")
        axis.set_ylabel("Amplitude-pinned log M(<R)")
        axis.set_title(title)
        axis.legend(fontsize=8)
    figure.suptitle("Coordinate continuity using identical five-descriptor distances")
    return _save(figure, stage, "coordinate_continuity")


def _diagnostic_figures(
    results: dict[str, dict[str, np.ndarray]],
    records: list[dict[str, object]],
    reference: dict[str, np.ndarray],
    indices: np.ndarray,
    mechanical: dict[str, object],
    stage: str,
) -> list[Path]:
    paths = _plot_synthetic(mechanical, stage)
    paths += _plot_trade(records, stage)
    paths += _plot_multi_start(results, reference, indices, stage)
    paths += _plot_weakest(results, reference, indices, stage)
    paths += _plot_continuity(results, reference, indices, stage)
    return paths


def _tercile_masks(values: np.ndarray) -> list[np.ndarray]:
    finite = np.isfinite(values)
    edges = np.quantile(values[finite], (0.0, 1 / 3, 2 / 3, 1.0))
    return [
        finite & (values >= lower) & (values <= upper if index == 2 else values < upper)
        for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:], strict=True))
    ]


def _mass_at_radius(profiles: np.ndarray, radius: float) -> np.ndarray:
    return np.asarray(
        [
            np.interp(np.log10(radius), np.log10(RADII_KPC), profile)
            for profile in profiles
        ]
    )


def _annulus_mass(
    profiles: np.ndarray, inner_radius: float, outer_radius: float
) -> np.ndarray:
    inner = 10.0 ** _mass_at_radius(profiles, inner_radius)
    outer = 10.0 ** _mass_at_radius(profiles, outer_radius)
    return np.log10(np.maximum(outer - inner, np.finfo(float).tiny))


def _mass_quantities(profiles: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "m30": _mass_at_radius(profiles, 30.0),
        "m30_50": _annulus_mass(profiles, 30.0, 50.0),
        "m50_100": _annulus_mass(profiles, 50.0, 100.0),
        "m100_148": _annulus_mass(profiles, 100.0, RADII_KPC[-1]),
    }


def _binned_median(
    x_values: np.ndarray, y_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    valid = (
        np.isfinite(x_values) & np.isfinite(y_values) & (x_values > 5) & (y_values > 5)
    )
    edges = np.unique(np.quantile(x_values[valid], np.linspace(0.0, 1.0, 8)))
    centers = []
    medians = []
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        selected = (
            valid
            & (x_values >= lower)
            & (x_values <= upper if index == len(edges) - 2 else x_values < upper)
        )
        centers.append(np.median(x_values[selected]))
        medians.append(np.median(y_values[selected]))
    return np.asarray(centers), np.asarray(medians)


def _plot_average_cogs(
    results: dict[str, dict[str, np.ndarray]],
    winner: str,
    reference: dict[str, np.ndarray],
    indices: np.ndarray,
    stage: str,
) -> list[Path]:
    epoch_zero = reference["epochs"][indices] == 0
    truth_log = reference["truth_log"][indices][epoch_zero]
    truth = 10.0**truth_log
    free = 10.0 ** results["free_reference"]["prediction"][epoch_zero]
    fixed = 10.0 ** results[winner]["prediction"][epoch_zero]
    halo_mass = reference["halo_mass"][indices][epoch_zero]
    stellar_mass = truth_log[:, -1]
    bin_definitions = (
        ("Halo-mass terciles (finite subset)", _tercile_masks(halo_mass)),
        ("Stellar-mass terciles", _tercile_masks(stellar_mass)),
    )
    colors = (OKABE_ITO[1], OKABE_ITO[2], OKABE_ITO[5])
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), sharex=True)
    for column, (title, masks) in enumerate(bin_definitions):
        for bin_index, (mask, color) in enumerate(zip(masks, colors, strict=True)):
            normalization = truth[:, -1:]
            pinned_truth = truth / normalization
            pinned_free = free / free[:, -1:]
            pinned_fixed = fixed / fixed[:, -1:]
            axes[0, column].plot(
                RADII_KPC,
                np.median(np.log10(pinned_truth[mask]), axis=0),
                color=color,
                lw=2.0,
                label=("low", "middle", "high")[bin_index],
            )
            axes[0, column].plot(
                RADII_KPC,
                np.median(np.log10(pinned_free[mask]), axis=0),
                color=color,
                ls=":",
            )
            axes[0, column].plot(
                RADII_KPC,
                np.median(np.log10(pinned_fixed[mask]), axis=0),
                color=color,
                ls="--",
            )
            for prediction, linestyle, model_label in (
                (free, ":", "Free damping"),
                (fixed, "--", "Shared damping"),
            ):
                residual = (
                    100.0
                    * (prediction * normalization / prediction[:, -1:] - truth)
                    / truth
                )
                axes[1, column].plot(
                    RADII_KPC,
                    np.median(residual[mask], axis=0),
                    color=color,
                    ls=linestyle,
                    label=model_label if bin_index == 0 else None,
                )
        axes[0, column].set_title(title)
        axes[1, column].axhline(0.0, color="0.6", lw=0.8)
    for axis in axes.ravel():
        axis.set_xscale("log")
    axes[0, 0].set_ylabel(r"Median $\log[M(<R)/M(<148.2)]$")
    axes[1, 0].set_ylabel("Median shape residual [%]")
    for axis in axes[-1]:
        axis.set_xlabel("Projected radius [kpc]")
    axes[0, 0].legend(title="Mass tercile", fontsize=8)
    axes[1, 0].legend(fontsize=8)
    figure.suptitle(
        "Discovery z=0.4 average CoGs: solid measured, dotted free, dashed shared damping"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(figure, stage, "average_cog_halo_and_stellar_mass_bins")


def _plot_stellar_mass_planes(
    results: dict[str, dict[str, np.ndarray]],
    winner: str,
    reference: dict[str, np.ndarray],
    indices: np.ndarray,
    stage: str,
) -> list[Path]:
    epoch_zero = reference["epochs"][indices] == 0
    profiles = {
        "Measured CoG": reference["truth_log"][indices][epoch_zero],
        "Free damping": results["free_reference"]["prediction"][epoch_zero],
        "Shared damping": results[winner]["prediction"][epoch_zero],
    }
    quantities = {name: _mass_quantities(values) for name, values in profiles.items()}
    planes = (
        (
            "m30",
            "m30_50",
            r"$\log M_*(<30\,\mathrm{kpc})$",
            r"$\log M_*(30\!-\!50\,\mathrm{kpc})$",
        ),
        (
            "m30",
            "m50_100",
            r"$\log M_*(<30\,\mathrm{kpc})$",
            r"$\log M_*(50\!-\!100\,\mathrm{kpc})$",
        ),
        (
            "m30_50",
            "m100_148",
            r"$\log M_*(30\!-\!50\,\mathrm{kpc})$",
            r"$\log M_*(100\!-\!148.2\,\mathrm{kpc})$",
        ),
    )
    styles = {
        "Measured CoG": ("black", "-", 2.2),
        "Free damping": (OKABE_ITO[0], ":", 1.8),
        "Shared damping": (OKABE_ITO[1], "--", 1.8),
    }
    figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.6))
    for axis, (x_name, y_name, x_label, y_label) in zip(axes, planes, strict=True):
        for name, values in quantities.items():
            center, median = _binned_median(values[x_name], values[y_name])
            color, linestyle, width = styles[name]
            axis.plot(center, median, color=color, ls=linestyle, lw=width, label=name)
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
    axes[0].legend(fontsize=8)
    figure.suptitle("Discovery z=0.4 stellar-mass planes: binned population medians")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(figure, stage, "stellar_mass_planes_z0p4")


def _minimum_population_qa(
    results: dict[str, dict[str, np.ndarray]],
    winner: str,
    reference: dict[str, np.ndarray],
    indices: np.ndarray,
    stage: str,
) -> list[Path]:
    return _plot_average_cogs(
        results, winner, reference, indices, stage
    ) + _plot_stellar_mass_planes(results, winner, reference, indices, stage)


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
    path = OUTDIR / stage / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(summary), indent=2) + "\n")


def _records(
    results: dict[str, dict[str, np.ndarray]],
    reference: dict[str, np.ndarray],
    indices: np.ndarray,
) -> list[dict[str, object]]:
    free = results["free_reference"]
    return [
        _candidate_record(spec, results[name], free, reference, indices)
        for name, spec in shared_damping_specs().items()
    ]


def run_gate(force: bool = False) -> dict[str, object]:
    started = time.perf_counter()
    reference = load_reference_data()
    discovery = stage_indices(reference, "discovery")
    indices = operational_gate_indices(reference, discovery)
    mechanical = mechanics()
    results = _fit_specs_flat(
        all_specs(),
        reference["truth_log"][indices],
        objective="log_cog",
        n_starts=4,
        grid_size=128,
        max_nfev=180,
    )
    for name, result in results.items():
        path = _archive_path("gate", "log_cog", name)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, indices=indices, **result)
    records = _records(results, reference, indices)
    figures = _diagnostic_figures(
        results, records, reference, indices, mechanical, "gate"
    )
    wall_seconds = time.perf_counter() - started
    summary = {
        "stage": "gate",
        "passed": bool(wall_seconds < 60.0 and mechanical["passed"]),
        "operational_only": True,
        "wall_seconds": wall_seconds,
        "n_galaxy": 25,
        "n_profile": 125,
        "mechanics": mechanical,
        "records": records,
        "figure_paths": [str(path) for path in figures],
    }
    _write_summary("gate", summary)
    write_manifest(
        OUTDIR,
        {
            "experiment": "exp70_global_damping",
            "stage": "gate",
            "n_starts": 4,
            "grid_size": 128,
        },
    )
    if not summary["passed"]:
        raise RuntimeError(
            f"Exp70 operational gate failed after {wall_seconds:.2f} seconds"
        )
    return summary


def run_discovery(force: bool = False) -> dict[str, object]:
    gate_path = OUTDIR / "gate" / "summary.json"
    if not gate_path.exists() or not json.loads(gate_path.read_text())["passed"]:
        raise RuntimeError(
            "Exp70 discovery is blocked until the operational gate passes"
        )
    started = time.perf_counter()
    reference = load_reference_data()
    indices = stage_indices(reference, "discovery")
    results = _fit_stage_checkpointed(
        all_specs(),
        reference,
        indices,
        stage="discovery",
        objective="log_cog",
        force=force,
    )
    records = _records(results, reference, indices)
    eligible = [
        record for record in records if candidate_is_eligible(record, "discovery")
    ]
    winner = choose_global_damping(records) if eligible else None
    figures = _diagnostic_figures(
        results, records, reference, indices, mechanics(), "discovery"
    )
    if winner is not None:
        figures += _minimum_population_qa(
            results, str(winner["name"]), reference, indices, "discovery"
        )
    summary = {
        "stage": "discovery",
        "status": "eligible_global_damping_frozen"
        if winner
        else "stopped_no_eligible_global_damping",
        "wall_seconds": time.perf_counter() - started,
        "n_galaxy": 1200,
        "n_profile": 6000,
        "n_eligible": len(eligible),
        "frozen_candidate": None if winner is None else winner["name"],
        "records": records,
        "figure_paths": [str(path) for path in figures],
    }
    _write_summary("discovery", summary)
    write_manifest(
        OUTDIR / "discovery",
        {
            "experiment": "exp70_global_damping",
            "stage": "discovery",
            "n_starts": 8,
            "grid_size": 256,
            "n_profile": 6000,
        },
    )
    return summary


def run_selection(force: bool = False) -> dict[str, object]:
    discovery_path = OUTDIR / "discovery" / "summary.json"
    if not discovery_path.exists():
        raise RuntimeError("selection is blocked until discovery completes")
    discovery = json.loads(discovery_path.read_text())
    frozen = discovery.get("frozen_candidate")
    if not frozen:
        raise RuntimeError("selection is blocked because discovery froze no candidate")
    reference = load_reference_data()
    indices = stage_indices(reference, "selection")
    specs = {
        "free_reference": free_reference_spec(),
        frozen: shared_damping_specs()[frozen],
    }
    started = time.perf_counter()
    results = _fit_stage_checkpointed(
        specs,
        reference,
        indices,
        stage="selection",
        objective="log_cog",
        force=force,
    )
    record = _candidate_record(
        specs[frozen], results[frozen], results["free_reference"], reference, indices
    )
    passed = candidate_is_eligible(record, "selection")
    summary = {
        "stage": "selection",
        "status": "passed" if passed else "stopped_selection_gate_failed",
        "wall_seconds": time.perf_counter() - started,
        "frozen_candidate": frozen,
        "record": record,
    }
    _write_summary("selection", summary)
    write_manifest(
        OUTDIR / "selection",
        {
            "experiment": "exp70_global_damping",
            "stage": "selection",
            "n_starts": 8,
            "grid_size": 256,
            "n_profile": 5000,
        },
    )
    return summary


def run_validation(force: bool = False) -> dict[str, object]:
    selection_path = OUTDIR / "selection" / "summary.json"
    if not selection_path.exists():
        raise RuntimeError("validation is blocked until selection completes")
    selection = json.loads(selection_path.read_text())
    if selection["status"] != "passed":
        raise RuntimeError("validation is blocked because selection did not pass")
    frozen = str(selection["frozen_candidate"])
    reference = load_reference_data()
    indices = stage_indices(reference, "validation")
    specs = {
        "free_reference": free_reference_spec(),
        frozen: shared_damping_specs()[frozen],
    }
    started = time.perf_counter()
    objective_records = {}
    passed = True
    for objective in ("log_cog", "absolute_cog", "log_shell"):
        results = _fit_stage_checkpointed(
            specs,
            reference,
            indices,
            stage="validation",
            objective=objective,
            force=force,
        )
        record = _candidate_record(
            specs[frozen],
            results[frozen],
            results["free_reference"],
            reference,
            indices,
        )
        double_metrics = profile_metrics(
            reference["prediction_double_sersic"][indices],
            reference["truth_log"][indices],
        )
        candidate_metrics = profile_metrics(
            results[frozen]["prediction"], reference["truth_log"][indices]
        )
        epochs = reference["epochs"][indices]
        double_full = np.asarray(
            _epoch_medians(double_metrics["full_cog_rms_dex"], epochs)
        )
        candidate_full = np.asarray(
            _epoch_medians(candidate_metrics["full_cog_rms_dex"], epochs)
        )
        ratio_to_double = (candidate_full / np.maximum(double_full, 1e-12)).tolist()
        record["full_cog_ratio_to_double_sersic_by_epoch"] = ratio_to_double
        objective_passed = bool(
            max(record["full_cog_ratio_to_free_by_epoch"]) <= 1.10
            and max(record["middle_cog_ratio_to_free_by_epoch"]) <= 1.10
            and max(record["shell_ratio_to_free_by_epoch"]) <= 1.10
            and max(ratio_to_double) <= 1.10
        )
        objective_records[objective] = {"passed": objective_passed, "record": record}
        passed &= objective_passed
    summary = {
        "stage": "validation",
        "status": "production_candidate" if passed else "not_for_production",
        "wall_seconds": time.perf_counter() - started,
        "frozen_candidate": frozen,
        "objective_records": objective_records,
    }
    _write_summary("validation", summary)
    write_manifest(
        OUTDIR / "validation",
        {
            "experiment": "exp70_global_damping",
            "stage": "validation",
            "n_starts": 8,
            "grid_size": 256,
            "n_profile": 5900,
            "objectives": ["log_cog", "absolute_cog", "log_shell"],
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=("mechanics", "gate", "discovery", "selection", "validation")
    )
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    if arguments.stage == "mechanics":
        print(json.dumps(_json_ready(mechanics()), indent=2))
    else:
        result = globals()[f"run_{arguments.stage}"](force=arguments.force)
        print(json.dumps(_json_ready(result), indent=2))


if __name__ == "__main__":
    main()
