# %%
"""Exp62 Stages 0-1: build the multi-epoch analytic CoG fitting atlas.

Commands:

    uv run python experiments/exp62_cog_fit_atlas/run.py mechanics
    uv run python experiments/exp62_cog_fit_atlas/run.py demo
    uv run python experiments/exp62_cog_fit_atlas/run.py full
    uv run python experiments/exp62_cog_fit_atlas/run.py figures demo|full

The demo uses 90 unique galaxies, 18 at each epoch, stratified by concurrent
halo mass. It exercises every fitted variant, fold-clean global-shape
selection, metrics, references, and figures before a full run is allowed.
"""

from __future__ import annotations

import importlib.util
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
    ATLAS_GROUPS,
    FAMILY_SPECS,
    FIT_VARIANTS,
    FIXED_DOUBLE_SERSIC_FAMILIES,
    SERSIC_MOFFAT_FAMILIES,
    fit_cog,
    self_check,
)
from hongshao.plotting import OKABE_ITO, save_fig, set_style  # noqa: E402
from hongshao.profile_emulator import density_from_cog  # noqa: E402
from hongshao.provenance import write_manifest  # noqa: E402
from hongshao.qa import enclosed_radius  # noqa: E402
from hongshao.tng_data import ANCHORS, COG_RAD_KPC, N_GAL, load_aper  # noqa: E402

set_style()

RADII = np.asarray(COG_RAD_KPC, float)
REDSHIFTS = np.asarray(sorted(ANCHORS), float)
OUTDIR = HERE / "outputs"
FIGDIR = HERE / "figures"
SOURCE_EXP51 = OUTDIR / "source" / "exp51_predictions.npz"
POPULATION_CACHE = OUTDIR / "population.npz"
DEFAULT_DATA_DIR = Path(
    os.environ.get("HONGSHAO_DATA_DIR", "/Users/shuang/Desktop/tng300_mah_mprof")
)
N_DEMO_PER_EPOCH = 18
N_FOLD = 5
MAX_WORKERS = min(int(os.environ.get("EXP62_WORKERS", "10")), os.cpu_count() or 1)


def _load_raw_galaxy(task):
    """Load and validate one galaxy in a worker process."""
    galaxy, data_dir = task
    output = np.full((len(REDSHIFTS), len(RADII)), np.nan)
    flags = np.zeros((4, len(REDSHIFTS)), bool)
    try:
        aperture = load_aper(galaxy, data_dir=Path(data_dir))
    except (FileNotFoundError, KeyError, ValueError, OSError):
        return galaxy, output, flags
    for epoch, redshift in enumerate(REDSHIFTS):
        key = ANCHORS[float(redshift)][0]
        if key not in aperture or "cog" not in aperture[key]:
            continue
        values = np.asarray(aperture[key]["cog"], float)
        if values.shape != RADII.shape:
            continue
        flags[0, epoch] = True
        flags[1, epoch] = np.isfinite(values).all()
        flags[2, epoch] = flags[1, epoch] and (values > 0).all()
        tolerance = 1.0e-10 * values[-1] if flags[2, epoch] else 0.0
        flags[3, epoch] = flags[2, epoch] and np.all(np.diff(values) >= -tolerance)
        if flags[3, epoch]:
            output[epoch] = np.log10(np.maximum.accumulate(values))
    return galaxy, output, flags


def build_population(force: bool = False, data_dir: Path = DEFAULT_DATA_DIR):
    """Build the per-epoch maximum valid sample directly from the raw CoGs."""
    if POPULATION_CACHE.exists() and not force:
        return np.load(POPULATION_CACHE)
    cogs_log = np.full((N_GAL, len(REDSHIFTS), len(RADII)), np.nan)
    finite = np.zeros((N_GAL, len(REDSHIFTS)), bool)
    positive = np.zeros_like(finite)
    monotonic = np.zeros_like(finite)
    present = np.zeros_like(finite)
    tasks = [(galaxy, str(data_dir)) for galaxy in range(N_GAL)]
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        loaded = executor.map(_load_raw_galaxy, tasks, chunksize=8)
        for completed, (galaxy, values, flags) in enumerate(loaded, start=1):
            cogs_log[galaxy] = values
            present[galaxy], finite[galaxy], positive[galaxy], monotonic[galaxy] = flags
            if completed % 500 == 0:
                print(f"  loaded {completed}/{N_GAL} raw CoG files", flush=True)

    valid = present & finite & positive & monotonic
    halo_mass = np.full((N_GAL, len(REDSHIFTS)), np.nan)
    matched_track = np.zeros(N_GAL, bool)
    if SOURCE_EXP51.exists():
        source = np.load(SOURCE_EXP51)
        source_indices = np.asarray(source["indices"], int)
        halo_mass[source_indices] = np.asarray(source["epoch_halo_mass"], float)
        matched_track[source_indices] = True
        maximum_difference = np.nanmax(
            np.abs(cogs_log[source_indices] - source["truth_cogs_log"])
        )
        if maximum_difference > 1.0e-8:
            raise ValueError(
                "raw and Exp51 CoGs disagree by "
                f"{maximum_difference:.3e} dex on the matched sample"
            )

    OUTDIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        POPULATION_CACHE,
        indices=np.arange(N_GAL),
        cogs_log=cogs_log,
        valid=valid,
        present=present,
        finite=finite,
        positive=positive,
        monotonic=monotonic,
        epoch_halo_mass=halo_mass,
        matched_track=matched_track,
        redshifts=REDSHIFTS,
        radii=RADII,
    )
    attrition = {
        "n_catalog": N_GAL,
        "redshifts": REDSHIFTS.tolist(),
        "n_present": present.sum(axis=0).tolist(),
        "n_finite": (present & finite).sum(axis=0).tolist(),
        "n_positive": (present & finite & positive).sum(axis=0).tolist(),
        "n_monotonic_valid": valid.sum(axis=0).tolist(),
        "n_valid_all_epochs": int(valid.all(axis=1).sum()),
        "n_matched_quality_clean": int(matched_track.sum()),
    }
    (OUTDIR / "attrition.json").write_text(json.dumps(attrition, indent=2) + "\n")
    print("population attrition:", json.dumps(attrition))
    return np.load(POPULATION_CACHE)


def _stratified_positions(values: np.ndarray, count: int) -> np.ndarray:
    order = np.argsort(values)
    positions = np.linspace(0, len(order) - 1, count).round().astype(int)
    return order[positions]


def selected_profiles(population, mode: str):
    """Return profile rows, epoch indices, truth, and concurrent halo masses."""
    valid = np.asarray(population["valid"], bool)
    cogs_log = np.asarray(population["cogs_log"], float)
    halo_mass = np.asarray(population["epoch_halo_mass"], float)
    if mode == "full":
        rows, epochs = np.nonzero(valid)
    elif mode == "demo":
        rows_out, epochs_out = [], []
        used = set()
        for epoch in range(len(REDSHIFTS)):
            candidates = np.flatnonzero(
                valid[:, epoch] & np.isfinite(halo_mass[:, epoch])
            )
            candidates = np.asarray([row for row in candidates if int(row) not in used])
            selected = candidates[
                _stratified_positions(halo_mass[candidates, epoch], N_DEMO_PER_EPOCH)
            ]
            rows_out.extend(selected.tolist())
            epochs_out.extend([epoch] * len(selected))
            used.update(int(row) for row in selected)
        rows, epochs = np.asarray(rows_out), np.asarray(epochs_out)
        if len(np.unique(rows)) != N_DEMO_PER_EPOCH * len(REDSHIFTS):
            raise AssertionError("demo must use 90 unique galaxies")
    else:
        raise ValueError("mode must be 'demo' or 'full'")
    return rows, epochs, cogs_log[rows, epochs], halo_mass[rows, epochs]


def _fit_task(task):
    family, truth_log = task
    result = fit_cog(family, RADII, truth_log)
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


def variant_path(mode: str, family: str) -> Path:
    return OUTDIR / mode / "variants" / f"{family}.npz"


def fit_variant(
    mode: str,
    family: str,
    truth_log: np.ndarray,
    force: bool = False,
    executor: ProcessPoolExecutor | None = None,
):
    path = variant_path(mode, family)
    if path.exists() and not force:
        return np.load(path)
    start = time.perf_counter()
    tasks = [(family, row) for row in truth_log]
    if executor is not None:
        results = list(
            executor.map(
                _fit_task,
                tasks,
                chunksize=max(len(tasks) // (8 * MAX_WORKERS), 1),
            )
        )
    elif MAX_WORKERS > 1:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as local_executor:
            results = list(
                local_executor.map(
                    _fit_task,
                    tasks,
                    chunksize=max(len(tasks) // (8 * MAX_WORKERS), 1),
                )
            )
    else:
        results = [_fit_task(task) for task in tasks]
    elapsed = time.perf_counter() - start
    spec = FAMILY_SPECS[family]
    payload = {
        "parameters": np.asarray([item[0] for item in results]).reshape(
            -1, spec.n_parameter
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
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    print(
        f"  {family:<29} {elapsed:7.2f} s; median CoG RMS "
        f"{np.median(payload['cog_rms_dex']):.5f} dex",
        flush=True,
    )
    return np.load(path)


def mass_stratified_folds(truth_log: np.ndarray, epochs: np.ndarray) -> np.ndarray:
    folds = np.empty(len(truth_log), int)
    for epoch in range(len(REDSHIFTS)):
        selected = np.flatnonzero(epochs == epoch)
        order = selected[np.argsort(truth_log[selected, -1])]
        folds[order] = np.arange(len(order)) % N_FOLD
    return folds


def select_variant_group(
    variants: tuple[str, ...],
    results: dict[str, dict],
    truth_log: np.ndarray,
    epochs: np.ndarray,
    folds: np.ndarray,
):
    """Training-only global-shape selection and held-out reconstruction."""
    prediction = np.empty_like(truth_log)
    n_parameter = FAMILY_SPECS[variants[0]].n_parameter
    parameters = np.empty((len(truth_log), n_parameter))
    selected_variant = np.empty(len(truth_log), dtype="U40")
    choices: dict[str, dict[str, str]] = {}
    for epoch in range(len(REDSHIFTS)):
        choices[f"z{REDSHIFTS[epoch]:g}"] = {}
        in_epoch = epochs == epoch
        for fold in range(N_FOLD):
            test = in_epoch & (folds == fold)
            train = in_epoch & (folds != fold)
            winner = min(
                variants,
                key=lambda name: float(np.median(results[name]["cog_rms_dex"][train])),
            )
            prediction[test] = results[winner]["prediction_log"][test]
            parameters[test] = results[winner]["parameters"][test]
            selected_variant[test] = winner
            choices[f"z{REDSHIFTS[epoch]:g}"][f"fold{fold}"] = winner
    return prediction, parameters, selected_variant, choices


def pca_reconstruction(truth_log: np.ndarray, epochs: np.ndarray, n_mode: int):
    output = np.empty_like(truth_log)
    for epoch in range(len(REDSHIFTS)):
        selected = epochs == epoch
        shape = truth_log[selected] - truth_log[selected, -1:]
        mean = np.mean(shape, axis=0)
        _, _, vectors = np.linalg.svd(shape - mean, full_matrices=False)
        basis = vectors[:n_mode]
        coefficient = (shape - mean) @ basis.T
        output[selected] = truth_log[selected, -1:] + mean + coefficient @ basis
    return output


def readable_reconstruction(truth_log: np.ndarray):
    path = ROOT / "experiments" / "exp51_direct_cog_redshift" / "model.py"
    spec = importlib.util.spec_from_file_location("exp62_exp51_coordinates", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    parameters = module.compress_cogs(truth_log, RADII, "quantile3")
    return module.reconstruct_cogs(parameters, RADII, "quantile3")


def profile_metrics(prediction_log: np.ndarray, truth_log: np.ndarray):
    residual = prediction_log - truth_log
    linear_prediction = 10.0**prediction_log
    linear_truth = 10.0**truth_log
    fractional = (linear_prediction - linear_truth) / linear_truth
    density_prediction, middle = density_from_cog(prediction_log, RADII)
    density_truth, _ = density_from_cog(truth_log, RADII)
    shell_prediction = np.column_stack(
        [linear_prediction[:, 0], np.diff(linear_prediction, axis=1)]
    )
    shell_truth = np.column_stack([linear_truth[:, 0], np.diff(linear_truth, axis=1)])
    outside = RADII >= 5.0
    middle_region = (RADII >= 5.0) & (RADII <= 30.0)
    size_fractions = (0.2, 0.5, 0.8, 0.9)
    size_error = np.column_stack(
        [
            np.log10(
                [enclosed_radius(row, RADII, fraction) for row in linear_prediction]
            )
            - np.log10([enclosed_radius(row, RADII, fraction) for row in linear_truth])
            for fraction in size_fractions
        ]
    )
    return {
        "cog_rms_full_dex": np.sqrt(np.mean(residual**2, axis=1)),
        "cog_rms_rge5_dex": np.sqrt(np.mean(residual[:, outside] ** 2, axis=1)),
        "cog_rms_5_30_dex": np.sqrt(np.mean(residual[:, middle_region] ** 2, axis=1)),
        "maximum_fractional_full": np.max(np.abs(fractional), axis=1),
        "maximum_fractional_rge5": np.max(np.abs(fractional[:, outside]), axis=1),
        "density_rms_dex": np.sqrt(
            np.mean((density_prediction - density_truth) ** 2, axis=1)
        ),
        "shell_rms_dex": np.sqrt(
            np.mean(
                (
                    np.log10(np.clip(shell_prediction, 1.0, None))
                    - np.log10(np.clip(shell_truth, 1.0, None))
                )
                ** 2,
                axis=1,
            )
        ),
        "log_mstar_148_error_dex": residual[:, -1],
        "size_error_dex": size_error,
        "density_middle_radius": middle,
    }


def summarize_group(name: str, metrics: dict, epochs: np.ndarray, diagnostics=None):
    summary = {"name": name, "by_epoch": {}}
    for epoch, redshift in enumerate(REDSHIFTS):
        selected = epochs == epoch
        values = {}
        for metric, array in metrics.items():
            if metric == "density_middle_radius":
                continue
            array = np.asarray(array)
            if array.ndim == 1:
                values[f"median_{metric}"] = float(np.nanmedian(array[selected]))
            elif metric == "size_error_dex":
                for column, fraction in enumerate((20, 50, 80, 90)):
                    values[f"median_absolute_r{fraction}_error_dex"] = float(
                        np.nanmedian(np.abs(array[selected, column]))
                    )
        if diagnostics is not None:
            values["failure_fraction"] = float(
                np.mean(~np.asarray(diagnostics["success"])[selected])
            )
            values["boundary_fraction_within_one_percent"] = float(
                np.mean(np.asarray(diagnostics["boundary_distance"])[selected] < 0.01)
            )
            values["median_jacobian_min_ratio"] = float(
                np.median(np.asarray(diagnostics["jacobian_min_ratio"])[selected])
            )
        summary["by_epoch"][f"z{redshift:g}"] = values
    return summary


def aggregate_atlas(mode: str, rows, epochs, truth_log, halo_mass, variant_results):
    folds = mass_stratified_folds(truth_log, epochs)
    predictions = {}
    parameters = {}
    selected_variants = {}
    choices = {}
    diagnostics = {}
    for group in ATLAS_GROUPS:
        if group == "double_sersic_fixed":
            variants = FIXED_DOUBLE_SERSIC_FAMILIES
        elif group == "sersic_moffat_fixed":
            variants = SERSIC_MOFFAT_FAMILIES
        else:
            variants = (group,)
        if len(variants) == 1:
            result = variant_results[variants[0]]
            predictions[group] = np.asarray(result["prediction_log"])
            parameters[group] = np.asarray(result["parameters"])
            selected_variants[group] = np.full(len(truth_log), variants[0], dtype="U40")
            diagnostics[group] = result
        else:
            prediction, fitted, selected, group_choices = select_variant_group(
                variants, variant_results, truth_log, epochs, folds
            )
            predictions[group] = prediction
            parameters[group] = fitted
            selected_variants[group] = selected
            choices[group] = group_choices
            diagnostics[group] = {
                key: np.choose(
                    [list(variants).index(name) for name in selected],
                    [np.asarray(variant_results[variant][key]) for variant in variants],
                )
                for key in ("success", "boundary_distance", "jacobian_min_ratio")
            }

    predictions["pca3"] = pca_reconstruction(truth_log, epochs, 3)
    predictions["pca5"] = pca_reconstruction(truth_log, epochs, 5)
    predictions["readable5"] = readable_reconstruction(truth_log)
    summaries = {}
    metric_store = {}
    for name, prediction in predictions.items():
        metrics = profile_metrics(prediction, truth_log)
        metric_store[name] = metrics
        summaries[name] = summarize_group(name, metrics, epochs, diagnostics.get(name))
    atlas_dir = OUTDIR / mode / "atlas"
    atlas_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        atlas_dir / "predictions.npz",
        rows=rows,
        epochs=epochs,
        redshifts=REDSHIFTS,
        truth_log=truth_log,
        halo_mass=halo_mass,
        folds=folds,
        **{f"prediction_{name}": value for name, value in predictions.items()},
        **{f"parameters_{name}": value for name, value in parameters.items()},
        **{
            f"selected_variant_{name}": value
            for name, value in selected_variants.items()
        },
    )
    (atlas_dir / "summary.json").write_text(
        json.dumps({"groups": summaries, "global_shape_choices": choices}, indent=2)
        + "\n"
    )
    return predictions, metric_store, summaries


def _metric_matrix(summaries: dict, names: list[str], metric: str):
    return np.asarray(
        [
            [
                summaries[name]["by_epoch"][f"z{redshift:g}"].get(metric, np.nan)
                for redshift in REDSHIFTS
            ]
            for name in names
        ]
    )


def atlas_figure(mode: str, summaries: dict):
    names = list(ATLAS_GROUPS) + ["pca3", "pca5", "readable5"]
    labels = [name.replace("_", " ") for name in names]
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 6.3))
    definitions = (
        ("median_cog_rms_full_dex", "Median full-CoG RMS (dex)"),
        ("median_density_rms_dex", "Median density RMS (dex)"),
        (
            "median_absolute_r90_error_dex",
            r"Median $|\Delta\log R_{90}|$ (dex)",
        ),
    )
    for axis, (metric, title) in zip(axes, definitions):
        values = _metric_matrix(summaries, names, metric)
        image = axis.imshow(values, aspect="auto", cmap="cividis")
        axis.grid(False)
        axis.set_xticks(range(len(REDSHIFTS)), [f"{z:g}" for z in REDSHIFTS])
        axis.set_yticks(range(len(names)), labels)
        axis.set_xlabel("Redshift")
        axis.set_title(title)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    for panel, axis in zip("ABC", axes):
        axis.text(-0.17, 1.03, panel, transform=axis.transAxes, fontweight="bold")
    figure.suptitle(f"exp62 {mode} — common analytic CoG fitting atlas")
    figure.tight_layout()
    path = FIGDIR / mode / "exp62_family_atlas"
    save_fig(figure, path)
    plt.close(figure)


def direct_profile_figure(mode, prediction_log, truth_log, group_name, metrics):
    errors = metrics["cog_rms_full_dex"]
    order = np.argsort(errors)
    chosen = [order[0], order[len(order) // 2], order[-1]]
    labels = ("Best", "Typical", "Worst")
    figure, axes = plt.subplots(3, 2, figsize=(8.5, 9.0), sharex="col")
    for row, (index, label) in enumerate(zip(chosen, labels)):
        axes[row, 0].plot(
            RADII, truth_log[index], "o", color="black", ms=3, label="TNG CoG"
        )
        axes[row, 0].plot(
            RADII,
            prediction_log[index],
            color=OKABE_ITO[2],
            label=group_name.replace("_", " "),
        )
        axes[row, 1].axhline(0.0, color="0.6", linewidth=0.8)
        axes[row, 1].plot(
            RADII, prediction_log[index] - truth_log[index], color=OKABE_ITO[1]
        )
        axes[row, 0].set_ylabel(r"$\log_{10} M_*(<R)\ [M_\odot]$")
        axes[row, 1].set_ylabel("Model - TNG (dex)")
        axes[row, 0].set_title(f"{label}: full-CoG RMS = {errors[index]:.4f} dex")
        axes[row, 1].set_title(
            f"Maximum |fractional error| = {metrics['maximum_fractional_full'][index]:.3f}"
        )
    for axis in axes.ravel():
        axis.set_xscale("log")
    axes[-1, 0].set_xlabel("Semi-major axis (kpc)")
    axes[-1, 1].set_xlabel("Semi-major axis (kpc)")
    axes[0, 0].legend(frameon=False)
    figure.suptitle(
        f"exp62 {mode} — direct individual fits for {group_name.replace('_', ' ')}"
    )
    figure.tight_layout()
    save_fig(figure, FIGDIR / mode / "exp62_direct_profiles")
    plt.close(figure)


def _interpolated_mass(cogs_log: np.ndarray, radius: float):
    return np.asarray([np.interp(radius, RADII, row) for row in cogs_log])


def minimum_qa_figure(mode, prediction_log, truth_log, epochs, halo_mass, group_name):
    figure, axes = plt.subplots(len(REDSHIFTS), 2, figsize=(9.5, 15.0))
    colors = (OKABE_ITO[0], OKABE_ITO[2], OKABE_ITO[4])
    for epoch, redshift in enumerate(REDSHIFTS):
        selected = (epochs == epoch) & np.isfinite(halo_mass)
        mass = halo_mass[selected]
        truth = truth_log[selected]
        model = prediction_log[selected]
        edges = np.quantile(mass, [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
        for mass_bin, color in enumerate(colors):
            in_bin = (mass >= edges[mass_bin]) & (mass <= edges[mass_bin + 1])
            truth_shape = truth[in_bin] - truth[in_bin, -1:]
            model_shape = model[in_bin] - model[in_bin, -1:]
            axes[epoch, 0].plot(RADII, np.median(truth_shape, axis=0), ":", color=color)
            axes[epoch, 0].plot(
                RADII,
                np.median(model_shape, axis=0),
                color=color,
                label=rf"${edges[mass_bin]:.2f}<\log M_h<{edges[mass_bin + 1]:.2f}$",
            )
        truth_inner = _interpolated_mass(truth, 30.0)
        model_inner = _interpolated_mass(model, 30.0)
        truth_outer = np.log10(
            np.clip(
                10.0 ** _interpolated_mass(truth, 100.0)
                - 10.0 ** _interpolated_mass(truth, 50.0),
                1.0,
                None,
            )
        )
        model_outer = np.log10(
            np.clip(
                10.0 ** _interpolated_mass(model, 100.0)
                - 10.0 ** _interpolated_mass(model, 50.0),
                1.0,
                None,
            )
        )
        axes[epoch, 1].scatter(
            truth_inner, truth_outer, s=8, alpha=0.35, color="black", label="TNG"
        )
        axes[epoch, 1].scatter(
            model_inner,
            model_outer,
            s=8,
            alpha=0.35,
            color=OKABE_ITO[2],
            label=group_name.replace("_", " "),
        )
        axes[epoch, 0].set_xscale("log")
        axes[epoch, 0].set_ylabel(rf"$z={redshift:g}$  $\log[M(<R)/M_{{148}}]$")
        axes[epoch, 1].set_ylabel(r"$\log M_*(50\!-!100\,\mathrm{kpc})$")
    axes[0, 0].legend(frameon=False, fontsize=7)
    axes[0, 1].legend(frameon=False, fontsize=7)
    axes[-1, 0].set_xlabel("Semi-major axis (kpc)")
    axes[-1, 1].set_xlabel(r"$\log M_*(<30\,\mathrm{kpc})$")
    figure.suptitle(
        f"exp62 {mode} — mandatory visual gate: halo-mass-bin CoGs and stellar-mass plane"
    )
    figure.tight_layout()
    save_fig(figure, FIGDIR / mode / "exp62_minimum_standard_qa")
    plt.close(figure)


def make_figures(mode: str):
    store = np.load(OUTDIR / mode / "atlas" / "predictions.npz")
    summaries = json.loads((OUTDIR / mode / "atlas" / "summary.json").read_text())[
        "groups"
    ]
    truth_log = np.asarray(store["truth_log"])
    epochs = np.asarray(store["epochs"])
    analytic_names = list(ATLAS_GROUPS)
    best_group = min(
        analytic_names,
        key=lambda name: np.median(
            profile_metrics(store[f"prediction_{name}"], truth_log)["cog_rms_full_dex"]
        ),
    )
    prediction = np.asarray(store[f"prediction_{best_group}"])
    metrics = profile_metrics(prediction, truth_log)
    atlas_figure(mode, summaries)
    direct_profile_figure(mode, prediction, truth_log, best_group, metrics)
    minimum_qa_figure(
        mode,
        prediction,
        truth_log,
        epochs,
        np.asarray(store["halo_mass"]),
        best_group,
    )
    return best_group


def run(mode: str, force: bool = False):
    population = build_population()
    rows, epochs, truth_log, halo_mass = selected_profiles(population, mode)
    print(
        f"exp62 Stage 1 {mode}: {len(truth_log)} galaxy-epoch CoGs, "
        f"{len(np.unique(rows))} unique galaxies, {MAX_WORKERS} workers"
    )
    start = time.perf_counter()
    variant_results = {}
    if MAX_WORKERS > 1:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for family in FIT_VARIANTS:
                variant_results[family] = fit_variant(
                    mode, family, truth_log, force=force, executor=executor
                )
    else:
        for family in FIT_VARIANTS:
            variant_results[family] = fit_variant(mode, family, truth_log, force=force)
    predictions, metrics, summaries = aggregate_atlas(
        mode, rows, epochs, truth_log, halo_mass, variant_results
    )
    best_group = make_figures(mode)
    elapsed = time.perf_counter() - start
    result = {
        "mode": mode,
        "n_profile": len(truth_log),
        "n_unique_galaxy": len(np.unique(rows)),
        "n_fitted_variant": len(FIT_VARIANTS),
        "wall_seconds": elapsed,
        "best_analytic_group_by_median_full_cog_rms": best_group,
        "subminute_gate_passed": bool(elapsed < 60.0) if mode == "demo" else None,
        "median_full_cog_rms_dex": float(
            np.median(metrics[best_group]["cog_rms_full_dex"])
        ),
    }
    result_path = OUTDIR / mode / "results.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    write_manifest(
        OUTDIR / mode,
        {
            "experiment": "exp62_cog_fit_atlas",
            "stage": 1,
            "mode": mode,
            "n_profile": len(truth_log),
            "redshifts": REDSHIFTS.tolist(),
            "n_variants": len(FIT_VARIANTS),
        },
    )
    print(json.dumps(result, indent=2))
    if mode == "demo" and elapsed >= 60.0:
        raise RuntimeError(
            f"Stage 1 demo took {elapsed:.2f} seconds; full run is forbidden"
        )
    return result


def mechanics():
    self_check(RADII)
    population = build_population()
    assert np.array_equal(population["radii"], RADII)
    assert np.array_equal(population["redshifts"], REDSHIFTS)
    rows, epochs, truth, halo_mass = selected_profiles(population, "demo")
    assert len(rows) == 90 and len(np.unique(rows)) == 90
    assert np.isfinite(truth).all() and np.isfinite(halo_mass).all()
    exact = profile_metrics(truth, truth)
    assert np.max(np.abs(exact["cog_rms_full_dex"])) == 0.0
    assert np.max(np.abs(exact["density_rms_dex"])) == 0.0
    print("exp62 data and metric mechanics OK; 90-galaxy sample spans all epochs")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "mechanics"
    if command == "mechanics":
        mechanics()
    elif command in ("demo", "full"):
        run(command, force="--force" in sys.argv)
    elif command == "figures":
        figure_mode = sys.argv[2] if len(sys.argv) > 2 else "demo"
        print("best analytic group:", make_figures(figure_mode))
    else:
        raise SystemExit(__doc__)
