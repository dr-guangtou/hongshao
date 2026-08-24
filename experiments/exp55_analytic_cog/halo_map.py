"""exp55 stage 3 — predict analytic CoG coordinates from halo assembly.

The fixed global profile family is the stage-2 compact Sersic n=1 plus
extended cored-power-law (Moffat gamma=1.25) representation. This stage maps
portable z=0.4 halo features to the four per-galaxy profile coordinates in
held-out folds, then judges the reconstructed profiles rather than only the
intermediate coordinates.

Commands
--------
demo
    Mechanical checks plus a small real-data held-out fit.
fit
    Full five-fold model/control comparison and standard QA.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from scipy.special import expit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from hongshao import qa  # noqa: E402
from hongshao.emulator import fit as emulator_fit  # noqa: E402
from hongshao.plotting import OKABE_ITO, save_fig, set_style  # noqa: E402
from hongshao.profile_emulator import density_from_cog  # noqa: E402
from hongshao.provenance import write_manifest  # noqa: E402
from mixed import mixed_cog_log  # noqa: E402
from qa_figures import (  # noqa: E402
    density_by_halo_mass_figure,
    single_epoch_planes_figure,
    single_epoch_size_figure,
)
from run import FIGDIR, OUTDIR, RADII, source_table  # noqa: E402

set_style()

MAP_FIGDIR = FIGDIR / "halo_map"
MAP_OUTDIR = OUTDIR / "halo_map"
STANDARD_QA_DIR = MAP_FIGDIR / "standard_qa"
N_SERSIC = 1.0
MOFFAT_GAMMA = 1.25
N_FOLD = 5
SEED = 0
N_DRAW = int(os.environ.get("EXP55_MAP_N_DRAW", 32))
N_MAX = int(os.environ.get("EXP55_MAP_NMAX", 0))

FEATURE_NAMES = ("logmp", "logtc", "early", "late", "c200c")
RAW_TARGET_NAMES = (
    "log_mstar_148",
    "logit_f_compact_148",
    "log_r_compact",
    "log_ratio_minus_one",
)
PHYSICAL_TARGET_NAMES = (
    "log_mstar_148",
    "f_compact_148",
    "log_r_compact",
    "log_r_extended",
)
PHYSICAL_TARGET_LABELS = (
    r"$\log_{10}M_{*,148}$",
    r"$f_{\rm compact,148}$",
    r"$\log_{10}R_{\rm compact}$ (kpc)",
    r"$\log_{10}R_{\rm extended}$ (kpc)",
)


def folds_of(n_galaxy: int) -> list[np.ndarray]:
    """The same deterministic five folds used by exp50 and exp51."""
    order = np.random.default_rng(SEED).permutation(n_galaxy)
    return list(np.array_split(order, N_FOLD))


def physical_parameters(raw_parameters: np.ndarray) -> np.ndarray:
    """Convert optimizer coordinates to four readable physical coordinates."""
    raw_parameters = np.atleast_2d(np.asarray(raw_parameters, float))
    compact_fraction = expit(raw_parameters[:, 1])
    log_r_compact = raw_parameters[:, 2]
    log_r_extended = log_r_compact + np.log10(
        1.0 + np.exp(np.clip(raw_parameters[:, 3], -40.0, 40.0))
    )
    return np.column_stack(
        [raw_parameters[:, 0], compact_fraction, log_r_compact, log_r_extended]
    )


def decode_profiles(raw_parameters: np.ndarray) -> np.ndarray:
    """Decode optimizer coordinates to full log10 CoGs."""
    raw_parameters = np.atleast_2d(np.asarray(raw_parameters, float))
    return np.asarray(
        [
            mixed_cog_log(parameters, RADII, N_SERSIC, MOFFAT_GAMMA)
            for parameters in raw_parameters
        ]
    )


def _stratified_indices(
    features: np.ndarray, targets: np.ndarray, n_max: int
) -> np.ndarray:
    """Select a small sample across halo mass and compact fraction."""
    if not n_max or n_max >= len(features):
        return np.arange(len(features))
    mass_rank = np.argsort(np.argsort(features[:, 0])) / max(len(features) - 1, 1)
    fraction = expit(targets[:, 1])
    fraction_rank = np.argsort(np.argsort(fraction)) / max(len(fraction) - 1, 1)
    side = int(np.ceil(np.sqrt(n_max)))
    targets_grid = np.array(
        [
            (mass_position, fraction_position)
            for mass_position in np.linspace(0.0, 1.0, side)
            for fraction_position in np.linspace(0.0, 1.0, side)
        ]
    )
    available = np.ones(len(features), bool)
    selected = []
    for target in targets_grid:
        distance = (mass_rank - target[0]) ** 2 + (fraction_rank - target[1]) ** 2
        distance[~available] = np.inf
        choice = int(np.argmin(distance))
        selected.append(choice)
        available[choice] = False
        if len(selected) == n_max:
            break
    return np.asarray(selected, int)


def load_sample(n_max: int = N_MAX) -> dict[str, np.ndarray]:
    """Join selected analytic targets to portable halo features by galaxy id."""
    selected = np.load(OUTDIR / "mixed_grid" / "selected.npz")
    indices = np.asarray(selected["indices"], int)
    table = Table.read(source_table())
    row_by_index = {int(index): row for row, index in enumerate(table["index"])}
    rows = np.asarray([row_by_index[int(index)] for index in indices], int)
    features = np.column_stack(
        [
            np.asarray(table[column][rows], float)
            for column in (
                "dmah_logmp",
                "dmah_logtc",
                "dmah_early",
                "dmah_late",
                "c_200c",
            )
        ]
    )
    raw_targets = np.asarray(selected["best_parameters"], float)
    truth_log = np.asarray(selected["truth_log"], float)
    representation_log = np.asarray(selected["best_prediction"], float)
    condition_ratio = np.asarray(
        np.load(OUTDIR / "mixed_grid" / "sersic1_moffat1p25.npz")["jacobian_min_ratio"],
        float,
    )
    good = np.isfinite(features).all(axis=1)
    good &= np.isfinite(raw_targets).all(axis=1) & np.isfinite(truth_log).all(axis=1)
    values = {
        "indices": indices[good],
        "features": features[good],
        "raw_targets": raw_targets[good],
        "truth_log": truth_log[good],
        "representation_log": representation_log[good],
        "condition_ratio": condition_ratio[good],
    }
    sample = _stratified_indices(values["features"], values["raw_targets"], n_max)
    return {name: value[sample] for name, value in values.items()}


def shuffled_within_mass_bins(
    features: np.ndarray,
    columns: tuple[int, ...],
    seed: int,
    n_bin: int = 10,
) -> np.ndarray:
    """Break feature-galaxy pairing while preserving final-mass dependence."""
    output = np.asarray(features, float).copy()
    edges = np.quantile(features[:, 0], np.linspace(0.0, 1.0, n_bin + 1))
    mass_bin = np.digitize(features[:, 0], edges[1:-1])
    rng = np.random.default_rng(seed)
    for bin_number in np.unique(mass_bin):
        members = np.where(mass_bin == bin_number)[0]
        if len(members) > 1:
            permutation = rng.permutation(members)
            output[np.ix_(members, columns)] = features[np.ix_(permutation, columns)]
    return output


def cv_map(
    features: np.ndarray,
    raw_targets: np.ndarray,
    mean: str,
    n_draw: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Fold-clean joint parameter predictions and decoded profile draws."""
    n_galaxy, n_target = raw_targets.shape
    mean_raw = np.empty_like(raw_targets)
    sigma_raw = np.empty_like(raw_targets)
    covariance_raw = np.empty((n_galaxy, n_target, n_target))
    mean_log = np.empty((n_galaxy, len(RADII)))
    draw_raw = np.empty((n_draw, n_galaxy, n_target)) if n_draw else None
    draw_log = np.empty((n_draw, n_galaxy, len(RADII))) if n_draw else None
    fold_id = np.empty(n_galaxy, int)
    for fold_number, fold in enumerate(folds_of(n_galaxy)):
        train = np.setdiff1d(np.arange(n_galaxy), fold)
        emulator = emulator_fit(features[train], raw_targets[train], mean=mean)
        mean_raw[fold], sigma_raw[fold], covariance_raw[fold] = emulator.predict(
            features[fold]
        )
        mean_log[fold] = decode_profiles(mean_raw[fold])
        fold_id[fold] = fold_number
        if n_draw:
            draws = emulator.sample(features[fold], size=n_draw, rng=seed + fold_number)
            draw_raw[:, fold] = draws
            draw_log[:, fold] = decode_profiles(draws.reshape(-1, n_target)).reshape(
                n_draw, len(fold), -1
            )
    return {
        "mean_raw": mean_raw,
        "sigma_raw": sigma_raw,
        "covariance_raw": covariance_raw,
        "mean_log": mean_log,
        "draw_raw": draw_raw,
        "draw_log": draw_log,
        "fold_id": fold_id,
    }


def empirical_crps(truth: np.ndarray, draws: np.ndarray) -> np.ndarray:
    """Empirical CRPS for arrays shaped (draw, galaxy, target)."""
    ordered = np.sort(np.asarray(draws, float), axis=0)
    truth = np.asarray(truth, float)
    n_draw = len(ordered)
    weights = (2 * np.arange(n_draw) - n_draw + 1).reshape(n_draw, 1, 1)
    pair_term = np.sum(weights * ordered, axis=0) / n_draw**2
    return np.mean(np.abs(ordered - truth[None]), axis=0) - pair_term


def aperture_targets(cogs_log: np.ndarray) -> np.ndarray:
    """CoG-derived core and three annular stellar masses in log10 Msun."""
    cumulative_log = np.column_stack(
        [
            [np.interp(radius, RADII, row) for row in cogs_log]
            for radius in (10.0, 30.0, 50.0, 100.0)
        ]
    )
    cumulative = 10.0**cumulative_log
    return np.column_stack(
        [
            cumulative_log[:, 0],
            np.log10(np.clip(cumulative[:, 1] - cumulative[:, 0], 1.0, None)),
            np.log10(np.clip(cumulative[:, 2] - cumulative[:, 1], 1.0, None)),
            np.log10(np.clip(cumulative[:, 3] - cumulative[:, 2], 1.0, None)),
        ]
    )


def summarize_result(
    name: str,
    result: dict[str, np.ndarray],
    raw_targets: np.ndarray,
    truth_log: np.ndarray,
) -> dict:
    """Parameter and reconstructed-profile performance for one halo map."""
    mean_log = result["mean_log"]
    draw_log = result["draw_log"]
    physical_truth = physical_parameters(raw_targets)
    physical_mean = physical_parameters(result["mean_raw"])
    target_variance = np.var(physical_truth, axis=0)
    parameter_r2 = 1.0 - np.mean(
        (physical_mean - physical_truth) ** 2, axis=0
    ) / np.clip(target_variance, 1e-20, None)
    profile_error = mean_log - truth_log
    shape_error = profile_error - profile_error[:, -1:]
    mean_density, _ = density_from_cog(mean_log, RADII)
    truth_density, _ = density_from_cog(truth_log, RADII)
    truth_apertures = aperture_targets(truth_log)
    mean_apertures = aperture_targets(mean_log)
    relative = np.abs((10.0**mean_log - 10.0**truth_log) / 10.0**truth_log)
    summary = {
        "name": name,
        "parameter_r2": dict(zip(PHYSICAL_TARGET_NAMES, parameter_r2.tolist())),
        "median_profile_rms_dex": float(
            np.median(np.sqrt(np.mean(profile_error**2, axis=1)))
        ),
        "median_shape_rms_dex": float(
            np.median(np.sqrt(np.mean(shape_error**2, axis=1)))
        ),
        "median_density_rms_dex": float(
            np.median(np.sqrt(np.mean((mean_density - truth_density) ** 2, axis=1)))
        ),
        "aperture_rms_dex": np.sqrt(
            np.mean((mean_apertures - truth_apertures) ** 2, axis=0)
        ).tolist(),
        "median_maximum_fractional_error_r_gt_5_kpc": float(
            np.median(np.max(relative[:, RADII > 5.0], axis=1))
        ),
    }
    if draw_log is not None:
        profile_crps = empirical_crps(truth_log, draw_log)
        physical_draws = physical_parameters(
            result["draw_raw"].reshape(-1, raw_targets.shape[1])
        ).reshape(len(draw_log), len(raw_targets), -1)
        parameter_crps = empirical_crps(physical_truth, physical_draws)
        summary.update(
            {
                "profile_crps_dex": float(np.mean(profile_crps)),
                "profile_crps_by_radius_dex": np.mean(profile_crps, axis=0).tolist(),
                "parameter_crps": dict(
                    zip(PHYSICAL_TARGET_NAMES, np.mean(parameter_crps, axis=0).tolist())
                ),
                "profile_coverage_68": float(
                    np.mean(
                        (truth_log >= np.quantile(draw_log, 0.16, axis=0))
                        & (truth_log <= np.quantile(draw_log, 0.84, axis=0))
                    )
                ),
                "profile_coverage_90": float(
                    np.mean(
                        (truth_log >= np.quantile(draw_log, 0.05, axis=0))
                        & (truth_log <= np.quantile(draw_log, 0.95, axis=0))
                    )
                ),
            }
        )
    return summary


def representation_summary(
    representation_log: np.ndarray, truth_log: np.ndarray
) -> dict[str, float]:
    error = representation_log - truth_log
    density_model, _ = density_from_cog(representation_log, RADII)
    density_truth, _ = density_from_cog(truth_log, RADII)
    return {
        "median_profile_rms_dex": float(np.median(np.sqrt(np.mean(error**2, axis=1)))),
        "median_density_rms_dex": float(
            np.median(np.sqrt(np.mean((density_model - density_truth) ** 2, axis=1)))
        ),
    }


def load_exp50_reference(indices: np.ndarray) -> dict[str, np.ndarray] | None:
    """Load the existing fold-clean readable-coordinate prediction if present."""
    source_root = Path(
        os.environ.get(
            "EXP50_SOURCE_ROOT", ROOT.parent / "hongshao_exp50_direct_cog_map"
        )
    ).expanduser()
    path = (
        source_root
        / "experiments"
        / "exp50_direct_cog_map"
        / "outputs"
        / "predictions.npz"
    )
    if N_MAX or not path.exists():
        return None
    data = np.load(path)
    row_by_index = {int(index): row for row, index in enumerate(data["indices"])}
    if any(int(index) not in row_by_index for index in indices):
        return None
    rows = np.asarray([row_by_index[int(index)] for index in indices], int)
    return {
        "mean_log": np.asarray(data["selected_mean_cogs_log"][rows], float),
        "draw_log": np.asarray(data["selected_draw_cogs_log"][:, rows], float),
    }


def skill_figure(summaries: dict[str, dict]) -> None:
    order = [
        "final_mass_only",
        "diffmah",
        "diffmah_c_linear",
        "mah_shape_shuffled",
        "c_shuffled",
        "diffmah_c_poly2",
    ]
    labels = [
        "final mass\nonly",
        "DiffMAH",
        "DiffMAH\n+ c linear",
        "MAH shape\nshuffled",
        "concentration\nshuffled",
        "DiffMAH\n+ c degree-2",
    ]
    colors = [
        "0.55",
        OKABE_ITO[0],
        OKABE_ITO[1],
        OKABE_ITO[6],
        OKABE_ITO[5],
        OKABE_ITO[2],
    ]
    metrics = (
        ("profile_crps_dex", "Mean profile CRPS (dex)"),
        ("median_profile_rms_dex", "Median CoG RMS (dex)"),
        ("median_density_rms_dex", "Median density RMS (dex)"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.3))
    for axis, (metric, ylabel) in zip(axes, metrics):
        values = [summaries[name][metric] for name in order]
        axis.bar(np.arange(len(order)), values, color=colors)
        axis.set_xticks(np.arange(len(order)), labels, fontsize=7)
        axis.set_ylabel(ylabel)
        axis.set_ylim(0.0, max(values) * 1.15)
        for position, value in enumerate(values):
            axis.text(
                position, value, f"{value:.4f}", ha="center", va="bottom", fontsize=6.5
            )
    for panel, axis in zip("ABC", axes):
        axis.text(-0.14, 1.04, panel, transform=axis.transAxes, fontweight="bold")
    fig.suptitle("exp55 — held-out halo-to-analytic-profile skill and null controls")
    fig.tight_layout()
    save_fig(fig, MAP_FIGDIR / "exp55_halo_map_skill")
    plt.close(fig)


def parameter_figure(
    raw_targets: np.ndarray,
    selected_result: dict[str, np.ndarray],
) -> None:
    truth = physical_parameters(raw_targets)
    prediction = physical_parameters(selected_result["mean_raw"])
    fig, axes = plt.subplots(2, 4, figsize=(14.5, 7.1), sharex="col")
    for column, label in enumerate(PHYSICAL_TARGET_LABELS):
        residual = prediction[:, column] - truth[:, column]
        axes[0, column].scatter(
            truth[:, column],
            prediction[:, column],
            s=5,
            alpha=0.22,
            color=OKABE_ITO[4],
            edgecolors="none",
        )
        low = min(truth[:, column].min(), prediction[:, column].min())
        high = max(truth[:, column].max(), prediction[:, column].max())
        axes[0, column].plot([low, high], [low, high], "k--", linewidth=0.9)
        axes[0, column].set_ylabel("Held-out conditional mean")
        variance = np.var(truth[:, column])
        r2 = 1.0 - np.mean(residual**2) / variance
        axes[0, column].set_title(f"{label}\nvariance explained $R^2={r2:.2f}$")
        axes[1, column].scatter(
            truth[:, column],
            residual,
            s=5,
            alpha=0.22,
            color=OKABE_ITO[5],
            edgecolors="none",
        )
        axes[1, column].axhline(0.0, color="0.5", linewidth=0.8)
        axes[1, column].set_xlabel(f"TNG profile fit: {label}")
        axes[1, column].set_ylabel("Prediction - profile fit")
    for panel, axis in zip("ABCDEFGH", axes.ravel()):
        axis.text(-0.15, 1.05, panel, transform=axis.transAxes, fontweight="bold")
    fig.suptitle(
        "exp55 — what halo assembly predicts about the four analytic CoG coordinates"
    )
    fig.tight_layout()
    save_fig(fig, MAP_FIGDIR / "exp55_halo_map_parameters")
    plt.close(fig)


def radial_figure(
    truth_log: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> None:
    styles = {
        "final mass only": ("0.45", ":"),
        "shuffled MAH shape": (OKABE_ITO[6], "-."),
        "analytic halo map": (OKABE_ITO[2], "-"),
        "Exp50 readable map": (OKABE_ITO[4], "--"),
        "analytic representation floor": (OKABE_ITO[0], ":"),
    }
    truth_density, middle_radius = density_from_cog(truth_log, RADII)
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2))
    for name, prediction in predictions.items():
        color, line_style = styles[name]
        error = prediction - truth_log
        density, _ = density_from_cog(prediction, RADII)
        axes[0].plot(
            RADII, np.median(np.abs(error), axis=0), line_style, color=color, label=name
        )
        axes[1].plot(
            middle_radius,
            np.median(np.abs(density - truth_density), axis=0),
            line_style,
            color=color,
        )
        axes[2].plot(
            RADII,
            np.median(error - error[:, -1:], axis=0),
            line_style,
            color=color,
        )
    axes[0].set(
        xscale="log",
        yscale="log",
        xlabel="Radius (kpc)",
        ylabel="Median absolute CoG error (dex)",
        title="Absolute profile",
    )
    axes[1].set(
        xscale="log",
        yscale="log",
        xlabel="Annulus midpoint (kpc)",
        ylabel="Median absolute density error (dex)",
        title="Differential profile",
    )
    axes[2].axhline(0.0, color="0.6", linewidth=0.8)
    axes[2].set(
        xscale="log",
        xlabel="Radius (kpc)",
        ylabel="Median amplitude-removed residual (dex)",
        title="Profile shape bias",
    )
    axes[0].legend(fontsize=7)
    for panel, axis in zip("ABC", axes):
        axis.text(-0.14, 1.04, panel, transform=axis.transAxes, fontweight="bold")
    fig.suptitle("exp55 — radial diagnosis of the halo-to-analytic-profile map")
    fig.tight_layout()
    save_fig(fig, MAP_FIGDIR / "exp55_halo_map_radial")
    plt.close(fig)


def standard_summary(result: dict) -> dict:
    """Small JSON-safe extract from the standard QA return value."""
    output = {"mass_quantities": {}, "planes": {}, "sizes": {}}
    for key in result["keys"]:
        truth = result["truth"][key][:, 0]
        model = result["model"][key][:, 0]
        good = np.isfinite(truth) & np.isfinite(model) & (truth > 0.0) & (model > 0.0)
        if not np.any(good):
            output["mass_quantities"][key] = None
            continue
        log_ratio = np.log10(model[good] / truth[good])
        output["mass_quantities"][key] = {
            "n": int(np.count_nonzero(good)),
            "median_fractional_bias": float(np.median(model[good] / truth[good] - 1.0)),
            "scatter_log10_model_over_truth_dex": float(np.std(log_ratio)),
        }
    for (x_name, y_name), values in result["planes"].items():
        truth_stats, model_stats = values[0]
        output["planes"][f"{x_name}__{y_name}"] = {
            "truth": truth_stats,
            "model": model_stats,
        }
    for (radius_name, epoch), values in result["sizes"].items():
        if epoch == 0:
            output["sizes"][radius_name] = {"truth": values[0], "model": values[1]}
    output["profile"] = {
        "median_maximum_fractional_error_all_radii": float(
            np.median(result["mr_all"][:, 0])
        ),
        "median_maximum_fractional_error_r_gt_5_kpc": float(
            np.median(result["mr_out"][:, 0])
        ),
    }
    return output


def generative_population_summary(
    draw_log: np.ndarray,
    truth_log: np.ndarray,
    n_evaluate: int = 4,
) -> dict:
    """Population-plane fidelity averaged over measured stochastic draws."""
    truth_cogs = 10.0 ** truth_log[:, None, :]
    _, truth_measures, _, _ = qa.measure_all(truth_cogs, truth_cogs, RADII)
    evaluated = min(n_evaluate, len(draw_log))
    plane_values = {f"{x_name}__{y_name}": [] for x_name, y_name in qa.PLANES}
    size_values = {name: [] for name in ("R50", "R80", "R90")}
    for draw in draw_log[:evaluated]:
        draw_cogs = 10.0 ** draw[:, None, :]
        _, draw_measures, _, _ = qa.measure_all(draw_cogs, truth_cogs, RADII)
        for x_name, y_name in qa.PLANES:
            truth_x = np.log10(np.clip(truth_measures[x_name][:, 0], 1.0, None))
            truth_y = np.log10(np.clip(truth_measures[y_name][:, 0], 1.0, None))
            draw_x = np.log10(np.clip(draw_measures[x_name][:, 0], 1.0, None))
            draw_y = np.log10(np.clip(draw_measures[y_name][:, 0], 1.0, None))
            statistics = qa.plane_stats(draw_x, draw_y)
            statistics.update(
                qa.plane_energy(
                    np.column_stack([truth_x, truth_y]),
                    np.column_stack([draw_x, draw_y]),
                )
            )
            plane_values[f"{x_name}__{y_name}"].append(statistics)
        sizes = qa.size_planes(draw_cogs, truth_cogs, RADII)
        for radius_name in size_values:
            size_values[radius_name].append(sizes[(radius_name, 0)][1])

    def average(entries: list[dict]) -> dict[str, float]:
        return {
            key: float(np.mean([entry[key] for entry in entries])) for key in entries[0]
        }

    return {
        "n_draw_evaluated": evaluated,
        "planes": {name: average(entries) for name, entries in plane_values.items()},
        "sizes": {name: average(entries) for name, entries in size_values.items()},
    }


def model_definitions(features: np.ndarray) -> dict[str, tuple[np.ndarray, str, int]]:
    """Predeclared real and shuffled feature sets."""
    mah_shuffled = shuffled_within_mass_bins(features, (1, 2, 3), seed=551)
    concentration_shuffled = shuffled_within_mass_bins(features, (4,), seed=552)
    return {
        "final_mass_only": (features[:, :1], "linear", 1100),
        "diffmah": (features[:, :4], "linear", 1200),
        "diffmah_c_linear": (features, "linear", 1300),
        "mah_shape_shuffled": (mah_shuffled, "poly2", 1400),
        "c_shuffled": (concentration_shuffled, "poly2", 1500),
        "diffmah_c_poly2": (features, "poly2", 1600),
    }


def demo() -> None:
    started = time.perf_counter()
    synthetic = np.array(
        [
            [11.2, 0.0, np.log10(2.5), np.log(13.0)],
            [11.7, -1.0, np.log10(4.0), np.log(7.0)],
        ]
    )
    profiles = decode_profiles(synthetic)
    assert np.isfinite(profiles).all()
    assert np.all(np.diff(profiles, axis=1) >= -1e-10)
    recovered = physical_parameters(synthetic)
    assert np.all(recovered[:, 3] > recovered[:, 2])
    sample = load_sample(60)
    result = cv_map(
        sample["features"], sample["raw_targets"], "linear", n_draw=4, seed=100
    )
    assert np.isfinite(result["mean_log"]).all()
    assert np.isfinite(result["draw_log"]).all()
    assert np.all(np.diff(result["mean_log"], axis=1) >= -1e-10)
    elapsed = time.perf_counter() - started
    print(
        f"exp55 halo-map demo OK: 60 galaxies, five folds, four draws in {elapsed:.2f} s"
    )


def fit_experiment() -> None:
    started = time.perf_counter()
    sample = load_sample()
    features = sample["features"]
    raw_targets = sample["raw_targets"]
    truth_log = sample["truth_log"]
    print(
        f"exp55 halo map: n={len(features)}, folds={N_FOLD}, draws={N_DRAW}; "
        f"features={FEATURE_NAMES}"
    )
    results = {}
    summaries = {}
    for name, (model_features, mean, seed) in model_definitions(features).items():
        print(f"  fitting {name}: {model_features.shape[1]} features, {mean} mean")
        results[name] = cv_map(
            model_features, raw_targets, mean=mean, n_draw=N_DRAW, seed=seed
        )
        summaries[name] = summarize_result(name, results[name], raw_targets, truth_log)

    selected_name = "diffmah_c_poly2"
    selected_result = results[selected_name]
    representation = representation_summary(sample["representation_log"], truth_log)
    exp50 = load_exp50_reference(sample["indices"])
    exp50_comparison = None
    if exp50 is not None:
        profile_error = exp50["mean_log"] - truth_log
        exp50_density, _ = density_from_cog(exp50["mean_log"], RADII)
        truth_density, _ = density_from_cog(truth_log, RADII)
        analytic_profile_rms = np.sqrt(
            np.mean((selected_result["mean_log"] - truth_log) ** 2, axis=1)
        )
        exp50_profile_rms = np.sqrt(np.mean(profile_error**2, axis=1))
        analytic_density, _ = density_from_cog(selected_result["mean_log"], RADII)
        analytic_density_rms = np.sqrt(
            np.mean((analytic_density - truth_density) ** 2, axis=1)
        )
        exp50_density_rms = np.sqrt(
            np.mean((exp50_density - truth_density) ** 2, axis=1)
        )
        analytic_crps = np.mean(
            empirical_crps(truth_log, selected_result["draw_log"]), axis=1
        )
        exp50_crps = np.mean(empirical_crps(truth_log, exp50["draw_log"]), axis=1)
        exp50_comparison = {
            "median_analytic_minus_exp50_profile_rms_dex": float(
                np.median(analytic_profile_rms - exp50_profile_rms)
            ),
            "median_analytic_minus_exp50_density_rms_dex": float(
                np.median(analytic_density_rms - exp50_density_rms)
            ),
            "mean_analytic_minus_exp50_profile_crps_dex": float(
                np.mean(analytic_crps - exp50_crps)
            ),
            "fraction_analytic_mean_profile_better": float(
                np.mean(analytic_profile_rms < exp50_profile_rms)
            ),
        }
        summaries["exp50_readable_map"] = {
            "median_profile_rms_dex": float(
                np.median(np.sqrt(np.mean(profile_error**2, axis=1)))
            ),
            "median_density_rms_dex": float(
                np.median(
                    np.sqrt(np.mean((exp50_density - truth_density) ** 2, axis=1))
                )
            ),
            "profile_crps_dex": float(
                np.mean(empirical_crps(truth_log, exp50["draw_log"]))
            ),
        }

    MAP_OUTDIR.mkdir(parents=True, exist_ok=True)
    MAP_FIGDIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        MAP_OUTDIR / "predictions.npz",
        indices=sample["indices"],
        features=features,
        raw_targets=raw_targets,
        truth_log=truth_log,
        representation_log=sample["representation_log"],
        selected_mean_raw=selected_result["mean_raw"],
        selected_sigma_raw=selected_result["sigma_raw"],
        selected_mean_log=selected_result["mean_log"],
        selected_draw_raw=selected_result["draw_raw"],
        selected_draw_log=selected_result["draw_log"],
        mass_only_mean_log=results["final_mass_only"]["mean_log"],
        shuffled_mah_mean_log=results["mah_shape_shuffled"]["mean_log"],
    )

    skill_figure(summaries)
    parameter_figure(raw_targets, selected_result)
    radial_predictions = {
        "final mass only": results["final_mass_only"]["mean_log"],
        "shuffled MAH shape": results["mah_shape_shuffled"]["mean_log"],
        "analytic halo map": selected_result["mean_log"],
        "analytic representation floor": sample["representation_log"],
    }
    if exp50 is not None:
        radial_predictions["Exp50 readable map"] = exp50["mean_log"]
    radial_figure(truth_log, radial_predictions)

    qa_result = qa.evaluate(
        10.0 ** selected_result["mean_log"][:, None, :],
        10.0 ** truth_log[:, None, :],
        RADII,
        [0.4],
        name="exp55_halo_map",
        figdir=STANDARD_QA_DIR,
        bin_by=features[:, 0],
        bin_label=r"DiffMAH $\log M_{\rm peak}$",
        draw_cogs=10.0 ** selected_result["draw_log"][:, :, None, :],
    )
    standard = standard_summary(qa_result)
    generative = generative_population_summary(selected_result["draw_log"], truth_log)
    density_by_halo_mass_figure(
        selected_result["mean_log"],
        truth_log,
        features[:, 0],
        figdir=STANDARD_QA_DIR,
        model_name="exp55_halo_map",
        model_label="analytic halo map",
    )
    single_epoch_planes_figure(
        qa_result,
        figdir=STANDARD_QA_DIR,
        model_name="exp55_halo_map",
        model_label="analytic halo-map mean",
    )
    single_epoch_size_figure(
        selected_result["mean_log"],
        truth_log,
        qa_result,
        figdir=STANDARD_QA_DIR,
        model_name="exp55_halo_map",
        model_label="analytic halo-map mean",
    )

    paired_profile_rms = np.sqrt(
        np.mean((selected_result["mean_log"] - truth_log) ** 2, axis=1)
    )
    shuffle_profile_rms = np.sqrt(
        np.mean((results["mah_shape_shuffled"]["mean_log"] - truth_log) ** 2, axis=1)
    )
    paired_improvement = shuffle_profile_rms - paired_profile_rms
    rng = np.random.default_rng(55)
    bootstrap = np.median(
        paired_improvement[
            rng.integers(
                0, len(paired_improvement), size=(1000, len(paired_improvement))
            )
        ],
        axis=1,
    )
    comparison = {
        "median_profile_rms_improvement_vs_shuffled_mah_dex": float(
            np.median(paired_improvement)
        ),
        "bootstrap_16_84_dex": np.quantile(bootstrap, [0.16, 0.84]).tolist(),
        "fraction_real_mah_better": float(np.mean(paired_improvement > 0.0)),
    }
    selected_crps = summaries[selected_name]["profile_crps_dex"]
    model_comparisons = {
        "profile_crps_improvement_vs_final_mass_percent": 100.0
        * (summaries["final_mass_only"]["profile_crps_dex"] - selected_crps)
        / summaries["final_mass_only"]["profile_crps_dex"],
        "profile_crps_improvement_vs_shuffled_mah_percent": 100.0
        * (summaries["mah_shape_shuffled"]["profile_crps_dex"] - selected_crps)
        / summaries["mah_shape_shuffled"]["profile_crps_dex"],
        "profile_crps_improvement_vs_shuffled_concentration_percent": 100.0
        * (summaries["c_shuffled"]["profile_crps_dex"] - selected_crps)
        / summaries["c_shuffled"]["profile_crps_dex"],
        "profile_crps_improvement_vs_linear_mean_percent": 100.0
        * (summaries["diffmah_c_linear"]["profile_crps_dex"] - selected_crps)
        / summaries["diffmah_c_linear"]["profile_crps_dex"],
    }
    payload = {
        "n_galaxy": len(features),
        "n_draw": N_DRAW,
        "feature_names": FEATURE_NAMES,
        "raw_target_names": RAW_TARGET_NAMES,
        "physical_target_names": PHYSICAL_TARGET_NAMES,
        "selected_model": selected_name,
        "representation_floor": representation,
        "models": summaries,
        "paired_real_vs_shuffled_mah": comparison,
        "model_comparisons": model_comparisons,
        "exp50_comparison": exp50_comparison,
        "standard_qa": standard,
        "generative_population_qa": generative,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (MAP_OUTDIR / "results.json").write_text(json.dumps(payload, indent=2))
    write_manifest(
        MAP_OUTDIR,
        {
            "experiment": "exp55_analytic_cog_halo_map",
            "n_galaxy": len(features),
            "n_draw": N_DRAW,
            "n_fold": N_FOLD,
            "selected_model": selected_name,
            "profile_family": {"n_sersic": N_SERSIC, "moffat_gamma": MOFFAT_GAMMA},
        },
    )
    print("\nSelected analytic halo map")
    print(json.dumps(summaries[selected_name], indent=2))
    print("\nReal versus shuffled MAH-shape pairing")
    print(json.dumps(comparison, indent=2))
    print(f"wrote halo-map outputs to {MAP_OUTDIR}")
    print(f"wrote halo-map figures to {MAP_FIGDIR}")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if command == "demo":
        demo()
    elif command == "fit":
        fit_experiment()
    else:
        raise SystemExit(f"unknown command {command!r}; use demo or fit")
