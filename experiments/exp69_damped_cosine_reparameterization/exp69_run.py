# %%
"""Exp69 damped-cosine reparameterization driver.

Commands:

    uv run python experiments/exp69_damped_cosine_reparameterization/exp69_run.py mechanics
    uv run python experiments/exp69_damped_cosine_reparameterization/exp69_run.py gate --force
"""

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
from scipy.optimize import least_squares
from scipy.integrate import cumulative_trapezoid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP66_DIR = HERE.parent / "exp66_expanded_symbolic_density"
EXP67_DIR = HERE.parent / "exp67_squared_slope_symbolic_density"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXP66_DIR))
sys.path.insert(0, str(EXP67_DIR))

os.environ.setdefault(
    "HONGSHAO_EXP62_ROOT",
    "/Users/shuang/Dropbox/work/project/massive/hongshao_master",
)

from exp66_search import load_references, profile_metrics  # noqa: E402
from exp67_search import population_split  # noqa: E402
from exp69_diagnostics import continuity_metrics, weakest_direction_path  # noqa: E402
from exp69_family import (  # noqa: E402
    RADII_KPC,
    CandidateSpec,
    _objective_residual,
    candidate_specs,
    cog_log,
    descriptor_bounds,
    fit_profile,
    integrated_density_rate,
    logarithmic_density_rate,
    physical_descriptors,
    synthetic_parameters,
)
from hongshao.plotting import OKABE_ITO, save_fig, set_style  # noqa: E402
from hongshao.provenance import write_manifest  # noqa: E402

set_style()
plt.rcParams["text.usetex"] = False

OUTDIR = HERE / "outputs"
FIGDIR = HERE / "figures"
MAX_WORKERS = min(int(os.environ.get("EXP69_WORKERS", "10")), os.cpu_count() or 1)
OBJECTIVES = ("log_cog", "absolute_cog", "log_shell")
ARCHIVE_RELATIVE_PATH = (
    Path("experiments")
    / "exp67_squared_slope_symbolic_density"
    / "outputs"
    / "discovery_round_a"
    / "expressions"
    / "free_damped_cosine.npz"
)


def _exp67_root() -> Path:
    configured = os.environ.get("HONGSHAO_EXP67_ROOT")
    if not configured:
        raise RuntimeError(
            "HONGSHAO_EXP67_ROOT must point to the preserved Exp67 worktree"
        )
    return Path(configured).resolve()


def _archive_path() -> Path:
    return _exp67_root() / ARCHIVE_RELATIVE_PATH


def discovery_artifacts_available() -> bool:
    """Return whether the explicitly configured discovery archive exists."""
    try:
        path = _archive_path()
    except RuntimeError:
        return False
    return path.exists()


def load_discovery_bundle(limit_galaxies: int | None = None) -> dict[str, object]:
    """Load the frozen Exp67 discovery profiles and no later-stage fit archive."""
    path = _archive_path()
    if not path.exists():
        raise FileNotFoundError(f"missing preserved Exp67 discovery archive: {path}")
    reference = load_references("full")
    discovery = np.asarray(population_split(reference)["discovery"], int)
    with np.load(path) as archive:
        archive_indices = np.asarray(archive["indices"], int)
        if not np.array_equal(archive_indices, discovery):
            raise ValueError("preserved free-damped-cosine archive is not discovery")
        archived = {key: np.asarray(archive[key]) for key in archive.files}
    local = np.arange(len(discovery))
    if limit_galaxies is not None:
        epoch_zero = np.flatnonzero(reference["epochs"][discovery] == 0)
        selected_rows = reference["rows"][discovery[epoch_zero[:limit_galaxies]]]
        local = np.flatnonzero(np.isin(reference["rows"][discovery], selected_rows))
    selected = discovery[local]
    return {
        "stage": "discovery",
        "archive_path": str(path),
        "archive_indices": archive_indices,
        "discovery_indices": discovery,
        "selected_global_indices": selected,
        "rows": np.asarray(reference["rows"][selected], int),
        "epochs": np.asarray(reference["epochs"][selected], int),
        "redshifts": np.asarray(
            reference["redshifts"][reference["epochs"][selected]], float
        ),
        "halo_mass": np.asarray(reference["halo_mass"][selected], float),
        "truth_log": np.asarray(reference["truth_log"][selected], float),
        "prediction_double_sersic": np.asarray(
            reference["prediction_double_sersic"][selected], float
        ),
        "prediction_cubic_logit": np.asarray(
            reference["prediction_cubic_logit"][selected], float
        ),
        "prediction_pca3": np.asarray(reference["prediction_pca3"][selected], float),
        "jacobian_cubic_logit": np.asarray(
            reference["jacobian_cubic_logit"][selected], float
        ),
        "archived_original_parameters": np.asarray(archived["parameters"])[local],
        "archived_original_prediction": np.asarray(archived["prediction"])[local],
        "archived_original_jacobian": np.asarray(archived["jacobian_min_ratio"])[local],
    }


def operational_gate_indices(bundle: dict[str, object]) -> np.ndarray:
    """Return local positions for the frozen 25-galaxy Exp67 gate sample."""
    rows = np.asarray(bundle["rows"], int)
    epochs = np.asarray(bundle["epochs"], int)
    halo_mass = np.asarray(bundle["halo_mass"], float)
    epoch_zero = np.flatnonzero(epochs == 0)
    finite = epoch_zero[np.isfinite(halo_mass[epoch_zero])]
    missing = epoch_zero[~np.isfinite(halo_mass[epoch_zero])]
    finite = finite[np.argsort(halo_mass[finite], kind="stable")]
    positions = np.rint(np.linspace(0, len(finite) - 1, 20)).astype(int)
    rng = np.random.default_rng(670)
    missing_selected = rng.choice(missing, size=5, replace=False)
    selected_rows = np.concatenate((rows[finite[positions]], rows[missing_selected]))
    selected = np.flatnonzero(np.isin(rows, selected_rows))
    if len(selected) != 125 or len(np.unique(rows[selected])) != 25:
        raise RuntimeError("operational sample does not contain 25 complete galaxies")
    return selected


def _descriptor_bounds(spec: CandidateSpec) -> tuple[np.ndarray, np.ndarray]:
    return descriptor_bounds(spec)


def _normalized_descriptors(spec: CandidateSpec, parameters: np.ndarray) -> np.ndarray:
    lower, upper = _descriptor_bounds(spec)
    span = np.maximum(upper - lower, 1.0e-12)
    return (
        np.asarray([physical_descriptors(spec, item) for item in parameters]) - lower
    ) / span


def _fit_task(
    payload: tuple[CandidateSpec, np.ndarray, str, int, int, int],
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


def _pack_fits(fitted: list[dict[str, object]]) -> dict[str, np.ndarray]:
    keys = (
        "parameters",
        "prediction",
        "canonical_parameters",
        "canonical_prediction",
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
        "canonical_index",
        "all_scores",
        "all_parameters",
        "all_predictions",
        "all_success",
        "all_singular_minimum",
        "all_singular_values",
        "all_descriptors",
    )
    return {key: np.asarray([item[key] for item in fitted]) for key in keys}


def fit_candidates(
    truth_log: np.ndarray,
    *,
    objective: str,
    grid_size: int,
    n_starts: int,
    max_nfev: int,
) -> dict[str, dict[str, np.ndarray]]:
    """Fit every frozen system with a flat process schedule."""
    specs = candidate_specs()
    collected: dict[str, list[dict[str, object]]] = {name: [] for name in specs}
    indexed_payloads = []
    for name, spec in specs.items():
        for index, truth in enumerate(truth_log):
            indexed_payloads.append(
                (name, index, (spec, truth, objective, grid_size, n_starts, max_nfev))
            )

    def ordered_payloads():
        for _, _, payload in indexed_payloads:
            yield payload

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        fitted = executor.map(_fit_task, ordered_payloads(), chunksize=2)
        for (name, _, _), result in zip(indexed_payloads, fitted, strict=True):
            collected[name].append(result)
    return {name: _pack_fits(values) for name, values in collected.items()}


def _synthetic_mechanics() -> dict[str, object]:
    records = []
    for name, spec in candidate_specs().items():
        for case_index in range(6):
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
                grid_size=512,
                n_starts=4,
                max_nfev=600,
                synthetic_truth_parameters=parameters,
            )
            descriptor_lower, descriptor_upper = _descriptor_bounds(spec)
            descriptor_error = float(
                np.max(
                    np.abs(
                        physical_descriptors(spec, fitted["parameters"])
                        - physical_descriptors(spec, parameters)
                    )
                    / np.maximum(descriptor_upper - descriptor_lower, 1.0e-12)
                )
            )
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
                    "recovery_descriptor_error": descriptor_error,
                    "success": bool(fitted["success"]),
                }
            )
    passed = all(
        record["integral_maximum_difference"] < 3.0e-7
        and record["quadrature_maximum_difference_dex"] < 4.0e-4
        and record["recovery_cog_rms_dex"] < 3.0e-5
        and record["recovery_descriptor_error"] < 0.03
        and record["success"]
        for record in records
    )
    return {"passed": passed, "records": records}


def _profiled_pair_scan(
    spec: CandidateSpec,
    truth_log: np.ndarray,
    center: np.ndarray,
    first_index: int,
    second_index: int,
    *,
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower = np.asarray(spec.lower)
    upper = np.asarray(spec.upper)
    first_grid = np.linspace(lower[first_index], upper[first_index], 9)
    second_grid = np.linspace(lower[second_index], upper[second_index], 9)
    free = np.asarray(
        [
            index
            for index in range(spec.n_parameter)
            if index not in {first_index, second_index}
        ]
    )
    loss = np.empty((9, 9))
    previous = np.asarray(center, float).copy()
    for row, first_value in enumerate(first_grid):
        for column, second_value in enumerate(second_grid):
            fixed = previous.copy()
            fixed[first_index] = first_value
            fixed[second_index] = second_value

            def residual(free_values: np.ndarray) -> np.ndarray:
                trial = fixed.copy()
                trial[free] = free_values
                return _objective_residual(
                    trial, spec, RADII_KPC, truth_log, "log_cog", grid_size
                )

            solution = least_squares(
                residual,
                fixed[free],
                bounds=(lower[free], upper[free]),
                method="trf",
                x_scale="jac",
                max_nfev=80,
            )
            fixed[free] = solution.x
            previous = fixed
            loss[row, column] = np.sqrt(np.mean(residual(solution.x) ** 2))
    return first_grid, second_grid, loss


def _candidate_records(
    results: dict[str, dict[str, np.ndarray]], bundle: dict[str, object]
) -> list[dict[str, object]]:
    truth = np.asarray(bundle["truth_log"], float)
    epochs = np.asarray(bundle["epochs"], int)
    records = []
    for name, result in _candidate_output_views(results, bundle).items():
        metrics = profile_metrics(result["prediction"], truth)
        continuity = continuity_metrics(
            _normalized_descriptors(_output_spec(name), result["parameters"]),
            truth,
            np.asarray(bundle["rows"], int),
            epochs,
        )
        records.append(
            {
                "name": name,
                "success_fraction": float(np.mean(result["success"])),
                "boundary_fraction_within_one_percent": float(
                    np.mean(result["boundary_distance"] < 0.01)
                ),
                "maximum_near_profile_spread_dex": float(
                    np.max(result["near_profile_spread_dex"])
                ),
                "median_full_cog_rms_dex": float(
                    np.median(metrics["full_cog_rms_dex"])
                ),
                "median_shell_density_rms_dex": float(
                    np.median(metrics["shell_density_rms_dex"])
                ),
                "median_jacobian_min_ratio": float(
                    np.median(result["jacobian_min_ratio"])
                ),
                "median_jacobian_ratio_to_cubic": float(
                    np.median(
                        result["jacobian_min_ratio"]
                        / np.maximum(
                            np.asarray(bundle["jacobian_cubic_logit"], float), 1.0e-16
                        )
                    )
                ),
                "continuity": {
                    key: value
                    for key, value in continuity.items()
                    if not isinstance(value, np.ndarray)
                },
            }
        )
    return records


def _output_spec(name: str) -> CandidateSpec:
    base_name = name.removesuffix("_canonical")
    return candidate_specs()[base_name]


def _output_label(name: str) -> str:
    label = _output_spec(name).label
    return f"{label}, canonical branch" if name.endswith("_canonical") else label


def _candidate_output_views(
    results: dict[str, dict[str, np.ndarray]],
    bundle: dict[str, object] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    views = dict(results)
    for base_name in ("original_free", "window_pivot"):
        result = results[base_name]
        spec = candidate_specs()[base_name]
        row = np.arange(len(result["canonical_index"]))
        chosen = np.asarray(result["canonical_index"], int)
        parameters = np.asarray(result["canonical_parameters"])
        position = (parameters - np.asarray(spec.lower)[None, :]) / (
            np.asarray(spec.upper) - np.asarray(spec.lower)
        )[None, :]
        if "all_singular_values" in result:
            singular_values = np.asarray(result["all_singular_values"])[row, chosen]
        elif bundle is not None:
            truth = np.asarray(bundle["truth_log"], float)
            singular_values = np.asarray(
                [
                    _scaled_singular_values(spec, parameter, truth_log)
                    for parameter, truth_log in zip(parameters, truth, strict=True)
                ]
            )
        else:
            singular_values = np.column_stack(
                (
                    np.ones(len(parameters)),
                    np.asarray(result["all_singular_minimum"])[row, chosen],
                )
            )
        maximum = singular_values[:, 0]
        jacobian_ratio = np.divide(
            singular_values[:, -1],
            maximum,
            out=np.zeros_like(maximum),
            where=maximum > 0.0,
        )
        canonical = dict(result)
        canonical.update(
            {
                "parameters": parameters,
                "prediction": np.asarray(result["canonical_prediction"]),
                "success": np.asarray(result["all_success"])[row, chosen],
                "jacobian_min_ratio": jacobian_ratio,
                "boundary_distance": np.min(
                    np.minimum(position, 1.0 - position), axis=1
                ),
            }
        )
        views[f"{base_name}_canonical"] = canonical
    return views


def _scaled_singular_values(
    spec: CandidateSpec, parameters: np.ndarray, truth_log: np.ndarray
) -> np.ndarray:
    values = np.asarray(parameters, float)
    lower = np.asarray(spec.lower)
    upper = np.asarray(spec.upper)
    parameter_range = upper - lower
    residual = _objective_residual(values, spec, RADII_KPC, truth_log, "log_cog", 128)
    jacobian = np.empty((len(residual), spec.n_parameter))
    for index in range(spec.n_parameter):
        step = np.sqrt(np.finfo(float).eps) * max(abs(values[index]), 1.0)
        direction = 1.0 if values[index] + step <= upper[index] else -1.0
        trial = values.copy()
        trial[index] = np.clip(
            values[index] + direction * step, lower[index], upper[index]
        )
        actual_step = trial[index] - values[index]
        shifted = _objective_residual(trial, spec, RADII_KPC, truth_log, "log_cog", 128)
        jacobian[:, index] = (shifted - residual) / actual_step
    return np.linalg.svd(jacobian * parameter_range[None, :], compute_uv=False)


def _plot_synthetic(mechanics: dict[str, object], stage: str) -> list[Path]:
    specs = candidate_specs()
    records = mechanics["records"]
    recovery = np.asarray(
        [
            [
                next(
                    record["recovery_cog_rms_dex"]
                    for record in records
                    if record["name"] == name and record["case_index"] == case
                )
                for case in range(6)
            ]
            for name in specs
        ]
    )
    descriptor = np.asarray(
        [
            [
                next(
                    record["recovery_descriptor_error"]
                    for record in records
                    if record["name"] == name and record["case_index"] == case
                )
                for case in range(6)
            ]
            for name in specs
        ]
    )
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), constrained_layout=True)
    for axis, values, title, label in (
        (axes[0], recovery, "Noiseless CoG recovery", "Full-CoG RMS [dex]"),
        (
            axes[1],
            descriptor,
            "Common physical-descriptor recovery",
            "Maximum error / declared range",
        ),
    ):
        image = axis.imshow(values, aspect="auto", cmap="cividis")
        axis.set_xticks(range(6), [f"Case {index + 1}" for index in range(6)])
        axis.set_yticks(range(len(specs)), [spec.label for spec in specs.values()])
        axis.set_title(title)
        figure.colorbar(image, ax=axis, label=label)
    return save_fig(figure, FIGDIR / stage / "synthetic_recovery")


def _plot_frontier(records: list[dict[str, object]], stage: str) -> list[Path]:
    figure, axis = plt.subplots(figsize=(7.4, 5.2))
    for index, record in enumerate(records):
        axis.scatter(
            record["median_full_cog_rms_dex"],
            record["median_jacobian_ratio_to_cubic"],
            s=65,
            color=OKABE_ITO[index % len(OKABE_ITO)],
            label=_output_label(record["name"]),
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Median full-CoG RMS [dex]")
    axis.set_ylabel("Median scaled-Jacobian strength / cubic-logit")
    axis.set_title("Exp69 operational sample: accuracy and local conditioning")
    axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    return save_fig(figure, FIGDIR / stage / "accuracy_conditioning")


def _plot_multi_start(
    results: dict[str, dict[str, np.ndarray]], bundle: dict[str, object], stage: str
) -> list[Path]:
    truth = np.asarray(bundle["truth_log"], float)
    original = results["original_free"]
    errors = np.sqrt(np.mean((original["prediction"] - truth) ** 2, axis=1))
    order = np.argsort(errors)
    chosen = (order[0], order[len(order) // 2], order[-1])
    figure, axes = plt.subplots(3, 2, figsize=(10.0, 8.4), sharex=True)
    colors = {
        name: OKABE_ITO[index % len(OKABE_ITO)] for index, name in enumerate(results)
    }
    for row, index in enumerate(chosen):
        axes[row, 0].plot(RADII_KPC, truth[index], "o", color="black", ms=3)
        axes[row, 1].axhline(0.0, color="0.6", lw=0.8)
        for name, result in results.items():
            scores = result["all_scores"][index]
            near = scores <= 1.01 * np.min(scores) + 1.0e-14
            canonical_index = int(result["canonical_index"][index])
            for start, prediction in enumerate(result["all_predictions"][index]):
                alpha = 0.65 if near[start] else 0.12
                width = (
                    1.8 if start == canonical_index else (1.0 if near[start] else 0.6)
                )
                axes[row, 0].plot(
                    RADII_KPC, prediction, color=colors[name], alpha=alpha, lw=width
                )
                axes[row, 1].plot(
                    RADII_KPC,
                    prediction - truth[index],
                    color=colors[name],
                    alpha=alpha,
                    lw=width,
                )
        axes[row, 0].set_ylabel(("Best", "Typical", "Worst")[row] + "\nlog M(<R)")
        axes[row, 1].set_ylabel("Residual [dex]")
    for axis in axes[-1]:
        axis.set_xlabel("Projected radius [kpc]")
    for axis in axes.ravel():
        axis.set_xscale("log")
    handles = [
        plt.Line2D([], [], color=colors[name], label=spec.label)
        for name, spec in candidate_specs().items()
    ]
    figure.legend(handles=handles, loc="upper center", ncol=4, fontsize=7)
    figure.suptitle(
        "All four-start solutions; solutions within 1% of minimum loss are opaque",
        y=0.96,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    return save_fig(figure, FIGDIR / stage / "multi_start_solution_overlays")


def _scan_set(
    spec: CandidateSpec,
    truth: np.ndarray,
    parameters: np.ndarray,
    grid_size: int,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    damping_index = spec.parameter_names.index("log_damping")
    contrast_index = spec.parameter_names.index("signed_contrast")
    core_index = spec.parameter_names.index("log_core_radius")
    return {
        "damping_contrast": _profiled_pair_scan(
            spec,
            truth,
            parameters,
            damping_index,
            contrast_index,
            grid_size=grid_size,
        ),
        "damping_core": _profiled_pair_scan(
            spec,
            truth,
            parameters,
            damping_index,
            core_index,
            grid_size=grid_size,
        ),
    }


def _plot_weakest_and_scans(
    results: dict[str, dict[str, np.ndarray]], bundle: dict[str, object], stage: str
) -> list[Path]:
    spec = candidate_specs()["original_free"]
    original = results["original_free"]
    measured_index = int(np.argmin(original["jacobian_min_ratio"]))
    synthetic_parameters_value = synthetic_parameters(spec, 1)
    synthetic_truth = cog_log(spec, synthetic_parameters_value, RADII_KPC, 512)
    synthetic_fit = fit_profile(
        spec,
        RADII_KPC,
        synthetic_truth,
        n_starts=4,
        grid_size=128,
        max_nfev=400,
        synthetic_truth_parameters=synthetic_parameters_value,
    )
    cases = (
        (
            "Synthetic",
            synthetic_truth,
            np.asarray(synthetic_fit["parameters"]),
            np.asarray(synthetic_fit["weakest_direction"]),
        ),
        (
            "Weakest measured",
            np.asarray(bundle["truth_log"])[measured_index],
            original["parameters"][measured_index],
            original["weakest_direction"][measured_index],
        ),
    )
    figure, axes = plt.subplots(2, 3, figsize=(13.0, 8.0), constrained_layout=True)
    for row, (label, truth, parameters, direction) in enumerate(cases):
        path = weakest_direction_path(
            parameters,
            direction,
            np.asarray(spec.lower),
            np.asarray(spec.upper),
            n_point=21,
        )
        path_predictions = np.asarray(
            [cog_log(spec, point, RADII_KPC, grid_size=128) for point in path]
        )
        for index, prediction in enumerate(path_predictions):
            axes[row, 0].plot(
                RADII_KPC,
                prediction - truth,
                color=plt.cm.cividis(index / (len(path_predictions) - 1)),
                alpha=0.8,
            )
        axes[row, 0].axhline(0.0, color="0.5", lw=0.8)
        axes[row, 0].set_xscale("log")
        axes[row, 0].set_ylabel(f"{label}\nCoG residual [dex]")
        scans = _scan_set(spec, truth, parameters, 128)
        for column, (name, scan) in enumerate(scans.items(), start=1):
            first, second, loss = scan
            image = axes[row, column].imshow(
                loss.T,
                origin="lower",
                aspect="auto",
                extent=(first[0], first[-1], second[0], second[-1]),
                cmap="cividis",
            )
            axes[row, column].set_xlabel("log damping")
            axes[row, column].set_ylabel(
                "Signed contrast" if name == "damping_contrast" else "log R_c"
            )
            figure.colorbar(image, ax=axes[row, column], label="Profiled CoG RMS [dex]")
    axes[-1, 0].set_xlabel("Projected radius [kpc]")
    figure.suptitle("Weakest-Jacobian paths and profiled damping degeneracy scans")
    return save_fig(figure, FIGDIR / stage / "weakest_direction_and_profiled_scans")


def _plot_continuity(
    results: dict[str, dict[str, np.ndarray]], bundle: dict[str, object], stage: str
) -> list[Path]:
    truth = np.asarray(bundle["truth_log"], float)
    rows = np.asarray(bundle["rows"], int)
    epochs = np.asarray(bundle["epochs"], int)
    diagnostics = {}
    for name, result in _candidate_output_views(results, bundle).items():
        diagnostics[name] = continuity_metrics(
            _normalized_descriptors(_output_spec(name), result["parameters"]),
            truth,
            rows,
            epochs,
        )
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), constrained_layout=True)
    for index, (name, values) in enumerate(diagnostics.items()):
        color = OKABE_ITO[index % len(OKABE_ITO)]
        for axis, key in (
            (axes[0, 0], "nearest_distance"),
            (axes[0, 1], "epoch_distance"),
        ):
            distribution = np.sort(values[key])
            axis.plot(
                distribution,
                np.linspace(0.0, 1.0, len(distribution), endpoint=True),
                color=color,
                label=_output_label(name),
            )
    axes[0, 0].set_xlabel("Nearest-profile descriptor separation")
    axes[0, 1].set_xlabel("Adjacent-epoch descriptor separation")
    for axis in axes[0]:
        axis.set_ylabel("Cumulative fraction")
    axes[0, 0].legend(fontsize=7)
    original = diagnostics["original_free"]
    for axis, pair_key, distance_key, title in (
        (
            axes[1, 0],
            "nearest_pairs",
            "nearest_distance",
            "Largest nearest-profile coordinate jump",
        ),
        (
            axes[1, 1],
            "epoch_pairs",
            "epoch_distance",
            "Largest adjacent-epoch coordinate jump",
        ),
    ):
        largest = int(np.argmax(original[distance_key]))
        first, second = original[pair_key][largest]
        axis.plot(RADII_KPC, truth[first] - truth[first, -1], "o-", label="First")
        axis.plot(RADII_KPC, truth[second] - truth[second, -1], "s-", label="Second")
        axis.set_xscale("log")
        axis.set_xlabel("Projected radius [kpc]")
        axis.set_ylabel("Amplitude-pinned log M(<R)")
        axis.set_title(title)
        axis.legend()
    figure.suptitle("Exp69 coordinate continuity on fixed measured-profile pairs")
    return save_fig(figure, FIGDIR / stage / "coordinate_continuity")


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
    write_manifest(stage_dir, {"stage": stage, "max_workers": MAX_WORKERS})


def mechanics() -> dict[str, object]:
    """Run the frozen synthetic and analytic checks without measured fitting."""
    return _synthetic_mechanics()


def _gate_bundle() -> dict[str, object]:
    bundle_full = load_discovery_bundle()
    gate_local = operational_gate_indices(bundle_full)
    return {
        key: (
            value[gate_local]
            if isinstance(value, np.ndarray) and value.ndim > 0 and len(value) == 6000
            else value
        )
        for key, value in bundle_full.items()
    }


def run_gate(force: bool = False) -> dict[str, object]:
    """Run the complete predeclared 25-galaxy operational path."""
    started = time.perf_counter()
    bundle = _gate_bundle()
    mechanics_result = mechanics()
    if not mechanics_result["passed"]:
        summary = {
            "stage": "gate",
            "passed": False,
            "status": "stopped_failed_mechanics",
            "wall_seconds": time.perf_counter() - started,
            "mechanics": mechanics_result,
        }
        _write_summary("gate", summary)
        raise RuntimeError("Exp69 synthetic mechanics failed")
    results = fit_candidates(
        np.asarray(bundle["truth_log"]),
        objective="log_cog",
        grid_size=128,
        n_starts=4,
        max_nfev=180,
    )
    stage_dir = OUTDIR / "gate" / "candidates"
    stage_dir.mkdir(parents=True, exist_ok=True)
    for name, result in results.items():
        np.savez_compressed(stage_dir / f"{name}.npz", **result)
    records = _candidate_records(results, bundle)
    figures = []
    figures.extend(_plot_synthetic(mechanics_result, "gate"))
    figures.extend(_plot_frontier(records, "gate"))
    figures.extend(_plot_multi_start(results, bundle, "gate"))
    figures.extend(_plot_weakest_and_scans(results, bundle, "gate"))
    figures.extend(_plot_continuity(results, bundle, "gate"))
    wall_seconds = time.perf_counter() - started
    summary = {
        "stage": "gate",
        "status": "complete" if wall_seconds < 60.0 else "stopped_over_time_limit",
        "passed": wall_seconds < 60.0,
        "operational_only": True,
        "wall_seconds": wall_seconds,
        "n_galaxy": 25,
        "n_profile": 125,
        "n_coordinate_system": len(results),
        "n_candidate_output": len(_candidate_output_views(results)),
        "n_starts": 4,
        "mechanics": mechanics_result,
        "records": records,
        "figure_paths": [str(path) for path in figures],
        "exp67_archive": str(_archive_path()),
    }
    _write_summary("gate", summary)
    if not summary["passed"]:
        raise RuntimeError(f"Exp69 gate took {wall_seconds:.2f} seconds")
    return summary


def refresh_gate_report() -> dict[str, object]:
    """Correct reporting from stored gate fits without refitting or retiming."""
    summary_path = OUTDIR / "gate" / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError("the Exp69 gate summary does not exist")
    bundle = _gate_bundle()
    results = {}
    for name in candidate_specs():
        path = OUTDIR / "gate" / "candidates" / f"{name}.npz"
        if not path.exists():
            raise FileNotFoundError(f"missing stored gate candidate: {path}")
        with np.load(path) as archive:
            results[name] = {key: np.asarray(archive[key]) for key in archive.files}
    records = _candidate_records(results, bundle)
    refreshed = []
    refreshed.extend(_plot_frontier(records, "gate"))
    refreshed.extend(_plot_multi_start(results, bundle, "gate"))
    refreshed.extend(_plot_continuity(results, bundle, "gate"))
    summary = json.loads(summary_path.read_text())
    summary.update(
        {
            "records": records,
            "n_coordinate_system": len(results),
            "n_candidate_output": len(_candidate_output_views(results, bundle)),
            "reporting_correction": (
                "Canonical original and pivot outputs were added from stored "
                "multi-start fits; no profile was refit and gate timing/status "
                "remain unchanged."
            ),
            "refreshed_figure_paths": [str(path) for path in refreshed],
        }
    )
    summary.pop("n_candidate", None)
    _write_summary("gate", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("mechanics", "gate", "refresh_gate"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.stage == "mechanics":
        result = mechanics()
    elif args.stage == "gate":
        result = run_gate(args.force)
    else:
        result = refresh_gate_report()
    concise = {
        key: value
        for key, value in result.items()
        if key not in {"mechanics", "records"}
    }
    print(json.dumps(_json_ready(concise), indent=2))


if __name__ == "__main__":
    main()
