# %%
"""Exp67 squared-slope symbolic-density search driver."""

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
EXP66_DIR = HERE.parent / "exp66_expanded_symbolic_density"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXP66_DIR))

from exp66_search import (  # noqa: E402
    load_references,
    profile_metrics,
)
from exp67_grammar import (  # noqa: E402
    RADII_KPC,
    ExpressionSpec,
    analytic_property_record,
    build_round_b_specs,
    cog_log,
    density_log_slope,
    fit_profile,
    round_a_specs,
    surface_density,
    synthetic_parameters,
)
from hongshao.plotting import OKABE_ITO, save_fig, set_style  # noqa: E402
from hongshao.provenance import write_manifest  # noqa: E402

set_style()
plt.rcParams["text.usetex"] = False

OUTDIR = HERE / "outputs"
FIGDIR = HERE / "figures"
MAX_WORKERS = min(int(os.environ.get("EXP67_WORKERS", "10")), os.cpu_count() or 1)
TARGET_COUNTS = {
    "finite_halo_mass": {"discovery": 850, "selection": 708, "validation": 837},
    "missing_halo_mass": {"discovery": 350, "selection": 292, "validation": 343},
}
OBJECTIVES = ("log_cog", "absolute_cog", "log_shell")


def _indices_for_rows(reference: dict[str, np.ndarray], rows: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.isin(reference["rows"], rows))


def _largest_remainder_counts(group_sizes: np.ndarray, total: int) -> np.ndarray:
    ideal = group_sizes * total / np.sum(group_sizes)
    counts = np.floor(ideal).astype(int)
    remainder = total - int(np.sum(counts))
    order = np.argsort(-(ideal - counts), kind="stable")
    counts[order[:remainder]] += 1
    return counts


def _allocate_groups(
    groups: list[np.ndarray],
    discovery_total: int,
    selection_total: int,
    rng: np.random.Generator,
) -> dict[str, list[int]]:
    sizes = np.asarray([len(group) for group in groups], int)
    discovery_counts = _largest_remainder_counts(sizes, discovery_total)
    selection_counts = _largest_remainder_counts(sizes, selection_total)
    output = {"discovery": [], "selection": [], "validation": []}
    for group, n_discovery, n_selection in zip(
        groups, discovery_counts, selection_counts, strict=True
    ):
        shuffled = group.copy()
        rng.shuffle(shuffled)
        first = int(n_discovery)
        second = first + int(n_selection)
        if second > len(shuffled):
            raise RuntimeError("stratified allocation exceeds a group")
        output["discovery"].extend(shuffled[:first].tolist())
        output["selection"].extend(shuffled[first:second].tolist())
        output["validation"].extend(shuffled[second:].tolist())
    return output


def population_split(reference: dict[str, np.ndarray]) -> dict[str, object]:
    """Return the frozen finite-mass and missing-mass galaxy-level split."""
    epoch_zero = reference["epochs"] == 0
    rows = reference["rows"][epoch_zero]
    masses = reference["halo_mass"][epoch_zero]
    if len(rows) != 3380 or len(np.unique(rows)) != 3380:
        raise ValueError("Exp67 requires 3,380 complete five-epoch galaxies")
    finite = np.flatnonzero(np.isfinite(masses))
    missing = np.flatnonzero(~np.isfinite(masses))
    if len(finite) != 2395 or len(missing) != 985:
        raise ValueError("Exp67 finite/missing halo-mass counts changed")
    finite_order = finite[np.argsort(masses[finite], kind="stable")]
    finite_groups = [rows[group] for group in np.array_split(finite_order, 20)]
    missing_groups = [rows[missing]]
    rng = np.random.default_rng(67)
    finite_split = _allocate_groups(finite_groups, 850, 708, rng)
    missing_split = _allocate_groups(missing_groups, 350, 292, rng)
    output: dict[str, object] = {}
    for name in ("discovery", "selection", "validation"):
        selected_rows = np.asarray(finite_split[name] + missing_split[name], int)
        output[name] = _indices_for_rows(reference, selected_rows)
    output["stratum_counts"] = TARGET_COUNTS
    return output


def operational_gate_indices(
    reference: dict[str, np.ndarray], discovery: np.ndarray
) -> np.ndarray:
    """Select 20 finite-mass quantiles and five missing-mass galaxies."""
    epoch_zero = discovery[reference["epochs"][discovery] == 0]
    masses = reference["halo_mass"][epoch_zero]
    finite = epoch_zero[np.isfinite(masses)]
    missing = epoch_zero[~np.isfinite(masses)]
    finite = finite[np.argsort(reference["halo_mass"][finite], kind="stable")]
    positions = np.rint(np.linspace(0, len(finite) - 1, 20)).astype(int)
    rng = np.random.default_rng(670)
    missing_selected = rng.choice(missing, size=5, replace=False)
    rows = np.concatenate(
        (reference["rows"][finite[positions]], reference["rows"][missing_selected])
    )
    return _indices_for_rows(reference, rows)


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
    payloads = (
        (spec, profile, objective, grid_size, n_starts, max_nfev)
        for profile in truth_log
    )
    return _pack_fits(list(executor.map(_fit_task, payloads, chunksize=4)))


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
    payloads = (
        (name, index, spec, profile, objective, grid_size, n_starts, max_nfev)
        for name, spec in specs.items()
        for index, profile in enumerate(truth_log)
    )
    collected: dict[str, list[dict[str, object] | None]] = {
        name: [None] * len(truth_log) for name in specs
    }
    for name, index, result in executor.map(_fit_indexed_task, payloads, chunksize=4):
        collected[name][index] = result
    return {
        name: _pack_fits([item for item in fitted if item is not None])
        for name, fitted in collected.items()
    }


def _epoch_medians(values: np.ndarray, epochs: np.ndarray) -> list[float]:
    return [float(np.nanmedian(values[epochs == epoch])) for epoch in range(5)]


def candidate_record(
    spec: ExpressionSpec,
    result: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    indices: np.ndarray,
    *,
    boundary_limit: float,
    conditioning_fraction: float,
    near_profile_limit: float,
    accuracy_ratio_limit: float,
) -> dict[str, object]:
    truth = reference["truth_log"][indices]
    epochs = reference["epochs"][indices]
    metrics = profile_metrics(result["prediction"], truth)
    double_metrics = profile_metrics(
        reference["prediction_double_sersic"][indices], truth
    )
    full = _epoch_medians(metrics["full_cog_rms_dex"], epochs)
    double_full = _epoch_medians(double_metrics["full_cog_rms_dex"], epochs)
    accuracy_ratios = np.asarray(full) / np.maximum(double_full, 1.0e-12)
    jacobian = _epoch_medians(result["jacobian_min_ratio"], epochs)
    cubic_jacobian = _epoch_medians(reference["jacobian_cubic_logit"][indices], epochs)
    conditioning_ratios = np.asarray(jacobian) / np.maximum(cubic_jacobian, 1.0e-16)
    success_fraction = float(np.mean(result["success"]))
    boundary_fraction = float(np.mean(result["boundary_distance"] < 0.01))
    near_profile = float(np.max(result["near_profile_spread_dex"]))
    worst_accuracy_ratio = float(np.max(accuracy_ratios))
    eligible = bool(
        success_fraction >= 0.999
        and boundary_fraction < boundary_limit
        and np.count_nonzero(conditioning_ratios >= conditioning_fraction) >= 4
        and near_profile < near_profile_limit
        and worst_accuracy_ratio < accuracy_ratio_limit
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
        "maximum_near_profile_spread_dex": near_profile,
        "median_full_cog_rms_dex_by_epoch": full,
        "median_double_sersic_full_cog_rms_dex_by_epoch": double_full,
        "worst_epoch_full_cog_ratio_to_double_sersic": worst_accuracy_ratio,
        "median_shell_density_rms_dex_by_epoch": _epoch_medians(
            metrics["shell_density_rms_dex"], epochs
        ),
        "median_jacobian_min_ratio_by_epoch": jacobian,
        "median_jacobian_ratio_to_cubic_by_epoch": conditioning_ratios.tolist(),
        "eligible": eligible,
        "analytic_properties": analytic_property_record(spec),
    }


def rank_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
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
    specs = round_a_specs()
    names = (
        "constant",
        "sigmoid_w0p7",
        "rational",
        "arctangent_w0p7",
        "shoulder_w0p7",
        "harmonic_sine_w1",
        "damped_cosine_l0p75_w2",
        "wave_packet_sine_w1_w2",
        "chirp_sine_k0p25_w2",
        "log_harmonic_cosine_w2",
        "free_frequency_phase",
    )
    records = []
    for name in names:
        spec = specs[name]
        parameters = synthetic_parameters(spec)
        truth = cog_log(spec, parameters, RADII_KPC, grid_size=2048)
        coarse = cog_log(spec, parameters, RADII_KPC, grid_size=512)
        result = fit_profile(
            spec,
            RADII_KPC,
            truth,
            grid_size=1024,
            n_starts=2,
            max_nfev=500,
            synthetic_truth_parameters=parameters,
        )
        records.append(
            {
                "name": name,
                "quadrature_maximum_difference_dex": float(
                    np.max(np.abs(coarse - truth))
                ),
                "synthetic_recovery_rms_dex": float(result["cog_rms_dex"]),
                "synthetic_recovery_success": bool(result["success"]),
                "analytic_properties": analytic_property_record(spec),
            }
        )
    passed = all(
        item["quadrature_maximum_difference_dex"] < 4.0e-4
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
    axis.axhline(0.25, color="0.35", lw=1.0, ls="--")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Worst-epoch median full-CoG RMS [dex]")
    axis.set_ylabel("Worst-epoch Jacobian ratio to cubic-logit")
    axis.set_title(f"Exp67 {stage}: squared-slope accuracy and conditioning")
    axis.legend()
    figure.tight_layout()
    paths = save_fig(figure, FIGDIR / stage / "accuracy_conditioning_frontier")
    plt.close(figure)
    return paths


def _plot_direct(
    spec: ExpressionSpec,
    result: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    indices: np.ndarray,
    stage: str,
) -> list[Path]:
    truth = reference["truth_log"][indices]
    errors = profile_metrics(result["prediction"], truth)["full_cog_rms_dex"]
    order = np.argsort(errors)
    chosen = [order[0], order[len(order) // 2], order[-1]]
    figure, axes = plt.subplots(3, 3, figsize=(10.5, 8.5), sharex="col")
    for row, (local_index, label) in enumerate(
        zip(chosen, ("Best", "Typical", "Worst"), strict=True)
    ):
        parameters = result["parameters"][local_index]
        prediction = result["prediction"][local_index]
        dense_radius = np.geomspace(0.05, RADII_KPC[-1], 500)
        density = surface_density(spec, parameters, dense_radius, grid_size=1024)
        slope = density_log_slope(spec, parameters, dense_radius)
        axes[row, 0].plot(RADII_KPC, truth[local_index], "o", ms=3, label="Measured")
        axes[row, 0].plot(RADII_KPC, prediction, lw=1.5, label=spec.label)
        axes[row, 1].axhline(0.0, color="0.6", lw=0.8)
        axes[row, 1].plot(RADII_KPC, prediction - truth[local_index])
        axes[row, 2].plot(dense_radius, density / density[0])
        twin = axes[row, 2].twinx()
        twin.plot(dense_radius, slope, color=OKABE_ITO[5], alpha=0.8)
        twin.set_ylabel("d ln Sigma / d ln R", color=OKABE_ITO[5])
        axes[row, 0].set_ylabel(f"{label}\nlog M(<R)")
        axes[row, 1].set_ylabel("Residual [dex]")
        axes[row, 2].set_ylabel("Sigma / Sigma(0)")
    for axis in axes.ravel():
        axis.set_xscale("log")
    for axis in axes[:, 2]:
        axis.set_yscale("log")
    for axis in axes[-1]:
        axis.set_xlabel("Projected radius [kpc]")
    axes[0, 0].legend(loc="lower right")
    figure.suptitle(f"Exp67 {stage}: direct fits for {spec.label}")
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
    write_manifest(stage_dir, {"stage": stage, "max_workers": MAX_WORKERS})


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
    accuracy_ratio_limit: float,
    force: bool,
    flat_schedule: bool,
) -> tuple[
    list[dict[str, object]], dict[str, dict[str, np.ndarray]], float, list[Path]
]:
    started = time.perf_counter()
    results: dict[str, dict[str, np.ndarray]] = {}
    expression_seconds: dict[str, float | None] = {}
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
                expression_seconds[name] = None
                print(
                    f"[{stage}] {name}: median RMS "
                    f"{np.median(result['cog_rms_dex']):.5f} dex",
                    flush=True,
                )
        else:
            for number, (name, spec) in enumerate(specs.items(), start=1):
                path = OUTDIR / stage / "expressions" / f"{name}.npz"
                if path.exists() and not force:
                    loaded = np.load(path)
                    result = {key: np.asarray(loaded[key]) for key in loaded.files}
                    elapsed = 0.0
                else:
                    expression_started = time.perf_counter()
                    result = fit_expression(
                        spec,
                        reference["truth_log"][indices],
                        objective="log_cog",
                        grid_size=grid_size,
                        n_starts=n_starts,
                        max_nfev=max_nfev,
                        executor=executor,
                    )
                    elapsed = time.perf_counter() - expression_started
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(path, indices=indices, **result)
                results[name] = result
                expression_seconds[name] = elapsed
                print(
                    f"[{stage}] {number}/{len(specs)} {name}: {elapsed:.2f} s, "
                    f"median RMS {np.median(result['cog_rms_dex']):.5f} dex",
                    flush=True,
                )
    records = rank_records(
        [
            candidate_record(
                spec,
                results[name],
                reference,
                indices,
                boundary_limit=boundary_limit,
                conditioning_fraction=conditioning_fraction,
                near_profile_limit=near_profile_limit,
                accuracy_ratio_limit=accuracy_ratio_limit,
            )
            for name, spec in specs.items()
        ]
    )
    figures = _plot_frontier(records, stage)
    best_name = str(records[0]["name"])
    figures.extend(
        _plot_direct(specs[best_name], results[best_name], reference, indices, stage)
    )
    elapsed = time.perf_counter() - started
    summary = {
        "stage": stage,
        "n_galaxy": int(len(np.unique(reference["rows"][indices]))),
        "n_profile": int(len(indices)),
        "n_expression": len(specs),
        "wall_seconds": elapsed,
        "expression_seconds": expression_seconds,
        "records": records,
        "figure_paths": [str(path) for path in figures],
        "specifications": {name: asdict(spec) for name, spec in specs.items()},
    }
    _write_summary(stage, summary)
    return records, results, elapsed, figures


def run_gate(force: bool = False) -> dict[str, object]:
    gate_started = time.perf_counter()
    reference = load_references("full")
    split = population_split(reference)
    indices = operational_gate_indices(reference, split["discovery"])
    mechanics_result = mechanics()
    if not mechanics_result["passed"]:
        raise RuntimeError("Exp67 mechanics failed")
    records, _, stage_seconds, figures = run_stage(
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
        accuracy_ratio_limit=1.5,
        force=force,
        flat_schedule=True,
    )
    wall_seconds = time.perf_counter() - gate_started
    summary = {
        "stage": "gate",
        "passed": wall_seconds < 60.0,
        "operational_only": True,
        "wall_seconds": wall_seconds,
        "stage_seconds": stage_seconds,
        "n_galaxy": 25,
        "n_profile": 125,
        "n_expression": len(records),
        "mechanics": mechanics_result,
        "best_record": records[0],
        "figure_paths": [str(path) for path in figures],
    }
    _write_summary("gate", summary)
    if not summary["passed"]:
        raise RuntimeError(f"Exp67 gate took {wall_seconds:.2f} seconds")
    return summary


def run_discovery(force: bool = False) -> dict[str, object]:
    gate_path = OUTDIR / "gate" / "summary.json"
    if not gate_path.exists():
        run_gate()
    if not json.loads(gate_path.read_text())["passed"]:
        raise RuntimeError("Exp67 discovery is blocked by its operational gate")
    reference = load_references("full")
    split = population_split(reference)
    round_a = round_a_specs()
    records, results, elapsed, figures = run_stage(
        "discovery_round_a",
        round_a,
        reference,
        split["discovery"],
        n_starts=2,
        grid_size=256,
        max_nfev=350,
        boundary_limit=0.05,
        conditioning_fraction=0.25,
        near_profile_limit=0.003,
        accuracy_ratio_limit=1.5,
        force=force,
        flat_schedule=False,
    )
    eligible = [record for record in records if record["eligible"]]
    survivors = [str(record["name"]) for record in eligible[:12]]
    summary = {
        "stage": "discovery_round_a",
        "status": "complete" if survivors else "stopped_no_eligible_expression",
        "wall_seconds": elapsed,
        "n_galaxy": 1200,
        "n_profile": 6000,
        "n_expression": len(records),
        "n_eligible": len(eligible),
        "survivors": survivors,
        "records": records,
        "figure_paths": [str(path) for path in figures],
    }
    _write_summary("discovery_round_a", summary)
    if not survivors:
        return summary
    transitions = [
        round_a[str(record["name"])]
        for record in eligible
        if not record["has_trigonometric_atom"] and record["name"] != "constant"
    ][:2]
    trigonometric = [
        round_a[str(record["name"])]
        for record in eligible
        if record["has_trigonometric_atom"]
    ][:4]
    if len(transitions) < 2 or len(trigonometric) < 4:
        summary["status"] = "stopped_insufficient_atom_classes_for_round_b"
        _write_summary("discovery_round_a", summary)
        return summary
    round_b = build_round_b_specs(transitions, trigonometric)
    round_b_records, _, round_b_seconds, round_b_figures = run_stage(
        "discovery_round_b",
        round_b,
        reference,
        split["discovery"],
        n_starts=2,
        grid_size=256,
        max_nfev=400,
        boundary_limit=0.05,
        conditioning_fraction=0.25,
        near_profile_limit=0.003,
        accuracy_ratio_limit=1.5,
        force=force,
        flat_schedule=False,
    )
    all_records = rank_records(records + round_b_records)
    selection_names = [
        str(record["name"]) for record in all_records if record["eligible"]
    ][:6]
    summary.update(
        {
            "status": "round_b_complete",
            "round_b_seconds": round_b_seconds,
            "round_b_expression_count": len(round_b),
            "selection_candidates": selection_names,
            "round_b_figure_paths": [str(path) for path in round_b_figures],
        }
    )
    _write_summary("discovery", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("mechanics", "gate", "discovery"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.stage == "mechanics":
        result = mechanics()
    elif args.stage == "gate":
        result = run_gate(force=args.force)
    else:
        result = run_discovery(force=args.force)
    concise = {key: value for key, value in result.items() if key != "records"}
    print(json.dumps(_json_ready(concise), indent=2))


if __name__ == "__main__":
    main()
