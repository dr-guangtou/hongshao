"""Exp56 Stage 4: radial information in the held-out halo–CoG relation.

Commands
--------
demo
    Run the complete predeclared path on 90 mass-stratified galaxies.
run
    Run all 2,539 galaxies with 32 draws after the demo gate passes.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP55 = ROOT / "experiments" / "exp55_analytic_cog"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP55))
sys.path.insert(0, str(HERE))

import stage3_halo_cog as stage3  # noqa: E402
from halo_map import (  # noqa: E402
    empirical_crps,
    folds_of,
    shuffled_within_mass_bins,
    source_table,
    standard_summary,
)
from hongshao import qa  # noqa: E402
from hongshao.diffmah import log_mah  # noqa: E402
from hongshao.plotting import OKABE_ITO, save_fig, set_style  # noqa: E402
from hongshao.profile_emulator import density_from_cog  # noqa: E402
from hongshao.provenance import write_manifest  # noqa: E402
from hongshao.tng_data import load_cosmic_time  # noqa: E402
from qa_figures import (  # noqa: E402
    density_by_halo_mass_figure,
    single_epoch_planes_figure,
    single_epoch_size_figure,
)
from refine_halo_map import fit_conditional_model  # noqa: E402
from smooth_outer import neighbour_draws  # noqa: E402

set_style()
matplotlib.rcParams["text.usetex"] = False

N_DEMO_GALAXY = 90
N_DEMO_DRAW = 8
N_FULL_DRAW = 32
N_BOOTSTRAP = 1000
DEMO_TIME_LIMIT_SECONDS = 60.0
SEED = 5640
MASS_TERMS = ((0, 0),)
SCALE_GRID = np.asarray([1.00, 1.16, 1.32, 1.48, 1.64])

REAL_RELATIONS = (
    "final_mass",
    "mass_recent",
    "mass_recent_early",
    "complete_diffmah",
    "complete_c",
)
CONTROL_RELATIONS = (
    "mass_recent_shuffled",
    "mass_recent_early_shuffled",
    "complete_diffmah_shuffled",
    "complete_c_shuffled",
)
RELATION_LABELS = {
    "final_mass": r"$M_{\rm peak}$",
    "mass_recent": r"$M_{\rm peak}$ + recent growth",
    "mass_recent_early": r"$M_{\rm peak}$ + recent + early fraction",
    "complete_diffmah": "complete DiffMAH",
    "complete_c": r"complete DiffMAH + $c_{200c}$",
}
RELATION_COLORS = dict(zip(REAL_RELATIONS, OKABE_ITO[:5]))
ADDITIONS = {
    "recent_growth": (
        "mass_recent",
        "final_mass",
        "mass_recent_shuffled",
    ),
    "early_assembled_fraction": (
        "mass_recent_early",
        "mass_recent",
        "mass_recent_early_shuffled",
    ),
    "complete_diffmah_history": (
        "complete_diffmah",
        "mass_recent_early",
        "complete_diffmah_shuffled",
    ),
    "concentration": (
        "complete_c",
        "complete_diffmah",
        "complete_c_shuffled",
    ),
}
ADDITION_COLORS = dict(zip(ADDITIONS, OKABE_ITO[:4]))
REGIONS = {
    "inner_2_10_kpc": (2.0, 10.0),
    "middle_10_30_kpc": (10.0, 30.0),
    "outer_30_148_kpc": (30.0, 148.0),
}

FIGURE_DIRECTORY = HERE / "figures" / "stage4"
OUTPUT_DIRECTORY = HERE / "outputs" / "stage4"


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def artifact_prefix(n_max: int) -> str:
    return "demo_" if n_max else ""


def assembly_summaries(features: np.ndarray) -> dict[str, np.ndarray]:
    cosmic_time = load_cosmic_time()
    evaluation_time = cosmic_time[[33, 50, 72]]
    history = log_mah(
        np.log10(evaluation_time),
        *features[:, :4].T,
        logt0=np.log10(evaluation_time[-1]),
    )
    recent_growth = history[:, 2] - history[:, 1]
    log_early_fraction = history[:, 0] - history[:, 2]
    return {
        "log_mass_z2": history[:, 0],
        "log_mass_z1": history[:, 1],
        "log_mass_z0p4": history[:, 2],
        "recent_growth_dex": recent_growth,
        "log_early_assembled_fraction": log_early_fraction,
        "early_assembled_fraction": 10.0**log_early_fraction,
        "cosmic_time_gyr": evaluation_time,
    }


def feature_diagnostics(
    sample: dict[str, np.ndarray], summaries: dict[str, np.ndarray]
) -> dict:
    table = Table.read(source_table())
    row_by_index = {int(index): row for row, index in enumerate(table["index"])}
    rows = np.asarray([row_by_index[int(index)] for index in sample["indices"]], int)
    raw_recent = np.asarray(table["dlogm_z1_z0p4"][rows], float)
    raw_early = np.asarray(table["f_early"][rows], float)

    def finite_spearman(first: np.ndarray, second: np.ndarray) -> tuple[float, int]:
        finite = np.isfinite(first) & np.isfinite(second)
        return float(spearmanr(first[finite], second[finite]).statistic), int(
            np.count_nonzero(finite)
        )

    recent_correlation, recent_count = finite_spearman(
        summaries["recent_growth_dex"], raw_recent
    )
    early_correlation, early_count = finite_spearman(
        summaries["early_assembled_fraction"], raw_early
    )
    values = {
        "recent_growth_raw_history_spearman": recent_correlation,
        "recent_growth_raw_history_n_finite": recent_count,
        "early_fraction_raw_history_spearman": early_correlation,
        "early_fraction_raw_history_n_finite": early_count,
    }
    named = {
        "final_mass": sample["features"][:, 0],
        "recent_growth": summaries["recent_growth_dex"],
        "log_early_fraction": summaries["log_early_assembled_fraction"],
        "concentration": sample["features"][:, 4],
    }
    values["spearman_correlations"] = {
        f"{first}__{second}": float(spearmanr(named[first], named[second]).statistic)
        for index, first in enumerate(named)
        for second in tuple(named)[index + 1 :]
    }
    values["ranges"] = {
        name: [float(np.min(array)), float(np.median(array)), float(np.max(array))]
        for name, array in named.items()
    }
    return values


def relation_features(
    features: np.ndarray, summaries: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    final_mass = features[:, :1]
    recent = summaries["recent_growth_dex"][:, None]
    early = summaries["log_early_assembled_fraction"][:, None]
    real = {
        "final_mass": final_mass,
        "mass_recent": np.column_stack([final_mass, recent]),
        "mass_recent_early": np.column_stack([final_mass, recent, early]),
        "complete_diffmah": features[:, :4],
        "complete_c": features,
    }
    output = dict(real)
    output["mass_recent_shuffled"] = shuffled_within_mass_bins(
        real["mass_recent"], (1,), seed=SEED + 1
    )
    output["mass_recent_early_shuffled"] = shuffled_within_mass_bins(
        real["mass_recent_early"], (2,), seed=SEED + 2
    )
    output["complete_diffmah_shuffled"] = shuffled_within_mass_bins(
        real["complete_diffmah"], (1, 2, 3), seed=SEED + 3
    )
    output["complete_c_shuffled"] = shuffled_within_mass_bins(
        real["complete_c"], (4,), seed=SEED + 4
    )
    return output


def calibrated_draws(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    train_truth_log: np.ndarray,
    test_features: np.ndarray,
    model,
    n_draw: int,
    seed: int,
) -> tuple[np.ndarray, dict]:
    order = np.random.default_rng(seed).permutation(len(train_features))
    inner_mean = np.empty_like(train_targets)
    inner_draw = np.empty((n_draw, *train_targets.shape))
    all_rows = np.arange(len(train_features))
    for inner_number, validation in enumerate(np.array_split(order, 4)):
        training = np.setdiff1d(all_rows, validation)
        inner_model = fit_conditional_model(
            train_features[training], train_targets[training], MASS_TERMS
        )
        inner_mean[validation] = inner_model.predict_mean(train_features[validation])
        inner_draw[:, validation] = neighbour_draws(
            inner_model,
            train_features[training],
            train_targets[training],
            train_features[validation],
            n_draw,
            seed + inner_number,
        )
    inner_gamma = np.full(len(train_features), stage3.stage_1.FIXED_GAMMA)
    coverage_by_scale = {}
    for scale in SCALE_GRID:
        draw_log = stage3.decode_profiles(
            inner_mean[None] + scale * (inner_draw - inner_mean[None]), inner_gamma
        )
        lower, upper = np.quantile(draw_log, (0.16, 0.84), axis=0)
        coverage_by_scale[f"{scale:.2f}"] = float(
            np.mean((train_truth_log >= lower) & (train_truth_log <= upper))
        )
    selected_scale = min(
        SCALE_GRID,
        key=lambda scale: (
            abs(coverage_by_scale[f"{scale:.2f}"] - 0.68),
            scale,
        ),
    )
    base = neighbour_draws(
        model,
        train_features,
        train_targets,
        test_features,
        n_draw,
        seed + 100,
    )
    mean = model.predict_mean(test_features)[None]
    return mean + selected_scale * (base - mean), {
        "selected_residual_scale": float(selected_scale),
        "inner_coverage_by_scale": coverage_by_scale,
    }


def cross_validated_relations(
    sample: dict[str, np.ndarray],
    feature_sets: dict[str, np.ndarray],
    n_draw: int,
    demo: bool,
) -> dict[str, dict]:
    cell = stage3.load_profile_cell(sample, "fixed_gamma1p4", 0, demo)
    targets = np.asarray(cell["parameters"], float)
    gamma = np.full(len(targets), stage3.stage_1.FIXED_GAMMA)
    relation_names = REAL_RELATIONS + CONTROL_RELATIONS
    relations = {
        name: {
            "mean_parameters": np.empty_like(targets),
            "draw_parameters": np.empty((n_draw, *targets.shape)),
            "metadata": [],
        }
        for name in relation_names
    }
    all_rows = np.arange(len(targets))
    for fold_number, test in enumerate(folds_of(len(targets))):
        train = np.setdiff1d(all_rows, test)
        for relation_number, name in enumerate(relation_names):
            values = feature_sets[name]
            model = fit_conditional_model(values[train], targets[train], MASS_TERMS)
            relations[name]["mean_parameters"][test] = model.predict_mean(values[test])
            draws, calibration = calibrated_draws(
                values[train],
                targets[train],
                sample["truth_log"][train],
                values[test],
                model,
                n_draw,
                SEED + 10000 * relation_number + 1000 * fold_number,
            )
            relations[name]["draw_parameters"][:, test] = draws
            relations[name]["metadata"].append(
                {
                    "fold": fold_number,
                    "n_train": len(train),
                    "n_test": len(test),
                    **calibration,
                }
            )
    for name in relation_names:
        relations[name]["mean_log"] = stage3.decode_profiles(
            relations[name]["mean_parameters"], gamma
        )
        relations[name]["draw_log"] = stage3.decode_profiles(
            relations[name]["draw_parameters"], gamma
        )
    return relations


def annular_log_mass(profile_log: np.ndarray) -> np.ndarray:
    linear = 10.0 ** np.asarray(profile_log, float)
    return np.log10(np.clip(np.diff(linear, axis=-1), 1e-30, None))


def metric_arrays(
    mean_log: np.ndarray, draw_log: np.ndarray, truth_log: np.ndarray
) -> dict[str, np.ndarray]:
    mean_density, density_radius = density_from_cog(mean_log, stage3.stage_1.RADII)
    truth_density, _ = density_from_cog(truth_log, stage3.stage_1.RADII)
    draw_density = np.asarray(
        [density_from_cog(draw, stage3.stage_1.RADII)[0] for draw in draw_log]
    )
    mean_annulus = annular_log_mass(mean_log)
    truth_annulus = annular_log_mass(truth_log)
    draw_annulus = annular_log_mass(draw_log)
    return {
        "cog_squared_error": (mean_log - truth_log) ** 2,
        "cog_crps": empirical_crps(truth_log, draw_log),
        "density_squared_error": (mean_density - truth_density) ** 2,
        "density_crps": empirical_crps(truth_density, draw_density),
        "annulus_squared_error": (mean_annulus - truth_annulus) ** 2,
        "annulus_crps": empirical_crps(truth_annulus, draw_annulus),
        "density_radius": density_radius,
        "annulus_radius": np.sqrt(stage3.stage_1.RADII[:-1] * stage3.stage_1.RADII[1:]),
    }


def summarize_relation(
    relation: dict, truth_log: np.ndarray, halo_mass: np.ndarray
) -> dict:
    arrays = metric_arrays(relation["mean_log"], relation["draw_log"], truth_log)
    lower_68, upper_68 = np.quantile(relation["draw_log"], (0.16, 0.84), axis=0)
    lower_90, upper_90 = np.quantile(relation["draw_log"], (0.05, 0.95), axis=0)
    summary = {
        "profile_crps_dex": float(np.mean(arrays["cog_crps"])),
        "median_profile_rms_dex": float(
            np.median(np.sqrt(np.mean(arrays["cog_squared_error"], axis=1)))
        ),
        "profile_coverage_68": float(
            np.mean((truth_log >= lower_68) & (truth_log <= upper_68))
        ),
        "profile_coverage_90": float(
            np.mean((truth_log >= lower_90) & (truth_log <= upper_90))
        ),
        "cog_rms_by_radius_dex": np.sqrt(np.mean(arrays["cog_squared_error"], axis=0)),
        "cog_crps_by_radius_dex": np.mean(arrays["cog_crps"], axis=0),
        "density_rms_by_radius_dex": np.sqrt(
            np.mean(arrays["density_squared_error"], axis=0)
        ),
        "density_crps_by_radius_dex": np.mean(arrays["density_crps"], axis=0),
        "annulus_rms_by_radius_dex": np.sqrt(
            np.mean(arrays["annulus_squared_error"], axis=0)
        ),
        "annulus_crps_by_radius_dex": np.mean(arrays["annulus_crps"], axis=0),
        "sizes": stage3.size_relation(relation["mean_log"]),
        "point": stage3.profile_point_summary(
            relation["mean_log"], truth_log, halo_mass
        ),
    }
    return summary


def bootstrap_curve_difference(
    current: np.ndarray,
    reference: np.ndarray,
    rms: bool,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    difference = np.empty((N_BOOTSTRAP, current.shape[1]))
    for bootstrap_number in range(N_BOOTSTRAP):
        rows = rng.integers(0, len(current), len(current))
        if rms:
            difference[bootstrap_number] = np.sqrt(np.mean(current[rows], axis=0))
            difference[bootstrap_number] -= np.sqrt(np.mean(reference[rows], axis=0))
        else:
            difference[bootstrap_number] = np.median(
                current[rows] - reference[rows], axis=0
            )
    return np.quantile(difference, (0.16, 0.50, 0.84), axis=0).T


def bootstrap_region_difference(
    current: np.ndarray,
    reference: np.ndarray,
    radius: np.ndarray,
    rms: bool,
    seed: int,
) -> dict[str, list[float]]:
    output = {}
    for region_number, (name, limits) in enumerate(REGIONS.items()):
        mask = (radius >= limits[0]) & (radius <= limits[1])
        if rms:
            current_value = np.sqrt(np.mean(current[:, mask], axis=1))
            reference_value = np.sqrt(np.mean(reference[:, mask], axis=1))
        else:
            current_value = np.mean(current[:, mask], axis=1)
            reference_value = np.mean(reference[:, mask], axis=1)
        output[name] = stage3.bootstrap_median_interval(
            current_value - reference_value, seed + region_number
        )
    return output


def adjacent_resolved(real_interval: np.ndarray, shuffle_interval: np.ndarray) -> bool:
    passing = (real_interval[:, 2] < 0.0) & (shuffle_interval[:, 2] < 0.0)
    return bool(np.any(passing[:-1] & passing[1:]))


def compare_information(
    relations: dict[str, dict], truth_log: np.ndarray
) -> tuple[dict, dict[str, dict[str, np.ndarray]]]:
    arrays = {
        name: metric_arrays(value["mean_log"], value["draw_log"], truth_log)
        for name, value in relations.items()
    }
    comparison = {}
    for addition_number, (addition, names) in enumerate(ADDITIONS.items()):
        current_name, preceding_name, shuffled_name = names
        current = arrays[current_name]
        preceding = arrays[preceding_name]
        shuffled = arrays[shuffled_name]
        record = {"models": names, "curves": {}, "regions": {}, "resolved": {}}
        metric_definitions = {
            "cog_rms": ("cog_squared_error", stage3.stage_1.RADII, True),
            "cog_crps": ("cog_crps", stage3.stage_1.RADII, False),
            "density_rms": (
                "density_squared_error",
                current["density_radius"],
                True,
            ),
            "density_crps": ("density_crps", current["density_radius"], False),
            "annulus_rms": (
                "annulus_squared_error",
                current["annulus_radius"],
                True,
            ),
            "annulus_crps": ("annulus_crps", current["annulus_radius"], False),
        }
        for metric_number, (metric, definition) in enumerate(
            metric_definitions.items()
        ):
            key, radius, is_rms = definition
            seed = SEED + 100 * addition_number + 10 * metric_number
            versus_preceding = bootstrap_curve_difference(
                current[key], preceding[key], is_rms, seed
            )
            versus_shuffled = bootstrap_curve_difference(
                current[key], shuffled[key], is_rms, seed + 1
            )
            record["curves"][metric] = {
                "radius_kpc": radius,
                "real_minus_preceding_16_50_84": versus_preceding,
                "real_minus_shuffled_16_50_84": versus_shuffled,
            }
            record["regions"][metric] = {
                "real_minus_preceding_16_50_84": bootstrap_region_difference(
                    current[key], preceding[key], radius, is_rms, seed + 2
                ),
                "real_minus_shuffled_16_50_84": bootstrap_region_difference(
                    current[key], shuffled[key], radius, is_rms, seed + 3
                ),
            }
            record["resolved"][f"{metric}_two_adjacent_bins"] = adjacent_resolved(
                versus_preceding, versus_shuffled
            )
        comparison[addition] = record
    return comparison, arrays


def stage3_adequacy(
    prefix: str, relation: dict, truth_log: np.ndarray
) -> dict[str, float | bool]:
    saved = np.load(stage3.OUTPUT_DIRECTORY / f"{prefix}predictions.npz")
    stage3_mean = np.asarray(saved["fixed_gamma1p4_mean_log"], float)
    stage3_draw = np.asarray(saved["fixed_gamma1p4_draw_log"], float)
    stage3_rms = float(
        np.median(np.sqrt(np.mean((stage3_mean - truth_log) ** 2, axis=1)))
    )
    stage4_rms = float(
        np.median(np.sqrt(np.mean((relation["mean_log"] - truth_log) ** 2, axis=1)))
    )
    stage3_crps = float(np.mean(empirical_crps(truth_log, stage3_draw)))
    stage4_crps = float(np.mean(empirical_crps(truth_log, relation["draw_log"])))
    return {
        "stage3_profile_crps_dex": stage3_crps,
        "stage4_profile_crps_dex": stage4_crps,
        "stage4_over_stage3_crps_ratio": stage4_crps / stage3_crps,
        "stage3_median_profile_rms_dex": stage3_rms,
        "stage4_median_profile_rms_dex": stage4_rms,
        "stage4_over_stage3_rms_ratio": stage4_rms / stage3_rms,
        "crps_within_one_percent": stage4_crps <= 1.01 * stage3_crps,
        "rms_within_one_percent": stage4_rms <= 1.01 * stage3_rms,
    }


def radial_accuracy_figure(summaries: dict[str, dict], prefix: str) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex="col")
    definitions = (
        ("cog_rms_by_radius_dex", "CoG RMS (dex)", stage3.stage_1.RADII),
        ("cog_crps_by_radius_dex", "CoG CRPS (dex)", stage3.stage_1.RADII),
        (
            "density_rms_by_radius_dex",
            "CoG-derived density RMS (dex)",
            np.sqrt(stage3.stage_1.RADII[:-1] * stage3.stage_1.RADII[1:]),
        ),
        (
            "density_crps_by_radius_dex",
            "CoG-derived density CRPS (dex)",
            np.sqrt(stage3.stage_1.RADII[:-1] * stage3.stage_1.RADII[1:]),
        ),
    )
    for panel, (axis, definition) in enumerate(zip(axes.flat, definitions)):
        key, ylabel, radius = definition
        for relation in REAL_RELATIONS:
            axis.plot(
                radius,
                summaries[relation][key],
                label=RELATION_LABELS[relation],
                color=RELATION_COLORS[relation],
                marker="o",
                markersize=2.5,
                linewidth=1.3,
            )
        axis.set_xscale("log")
        axis.set_ylabel(ylabel)
        axis.text(
            -0.12,
            1.04,
            chr(65 + panel),
            transform=axis.transAxes,
            fontweight="bold",
        )
    axes[1, 0].set_xlabel("Radius (kpc)")
    axes[1, 1].set_xlabel("Radius (kpc)")
    axes[0, 0].legend(ncol=2, fontsize=7)
    figure.suptitle("Exp56 Stage 4 — held-out radial accuracy")
    figure.tight_layout()
    save_fig(figure, FIGURE_DIRECTORY / f"{prefix}exp56_stage4_radial_accuracy")
    plt.close(figure)


def annular_accuracy_figure(summaries: dict[str, dict], prefix: str) -> None:
    radius = np.sqrt(stage3.stage_1.RADII[:-1] * stage3.stage_1.RADII[1:])
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 3.7), sharex=True)
    for axis, key, ylabel in zip(
        axes,
        ("annulus_rms_by_radius_dex", "annulus_crps_by_radius_dex"),
        ("Differential annulus RMS (dex)", "Differential annulus CRPS (dex)"),
    ):
        for relation in REAL_RELATIONS:
            axis.plot(
                radius,
                summaries[relation][key],
                label=RELATION_LABELS[relation],
                color=RELATION_COLORS[relation],
                marker="o",
                markersize=2.5,
            )
        axis.set_xscale("log")
        axis.set_xlabel("Annulus midpoint (kpc)")
        axis.set_ylabel(ylabel)
    axes[0].legend(ncol=2, fontsize=7)
    figure.suptitle("Exp56 Stage 4 — held-out differential masses")
    figure.tight_layout()
    save_fig(figure, FIGURE_DIRECTORY / f"{prefix}exp56_stage4_annular_accuracy")
    plt.close(figure)


def information_increment_figure(comparison: dict, prefix: str) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    definitions = (
        ("cog_rms", "Cumulative CoG RMS difference (dex)"),
        ("cog_crps", "Cumulative CoG CRPS difference (dex)"),
        ("annulus_rms", "Annular-mass RMS difference (dex)"),
        ("annulus_crps", "Annular-mass CRPS difference (dex)"),
    )
    for panel, (axis, definition) in enumerate(zip(axes.flat, definitions)):
        metric, ylabel = definition
        for addition in ADDITIONS:
            curves = comparison[addition]["curves"][metric]
            radius = np.asarray(curves["radius_kpc"])
            preceding = np.asarray(curves["real_minus_preceding_16_50_84"])
            shuffled = np.asarray(curves["real_minus_shuffled_16_50_84"])
            color = ADDITION_COLORS[addition]
            axis.plot(
                radius, preceding[:, 1], color=color, label=addition.replace("_", " ")
            )
            axis.fill_between(
                radius, preceding[:, 0], preceding[:, 2], color=color, alpha=0.12
            )
            axis.plot(radius, shuffled[:, 1], color=color, linestyle="--", alpha=0.8)
        axis.axhline(0.0, color="0.35", linewidth=0.8)
        axis.set_xscale("log")
        axis.set_xlabel("Radius or annulus midpoint (kpc)")
        axis.set_ylabel(ylabel)
        axis.text(
            -0.12,
            1.04,
            chr(65 + panel),
            transform=axis.transAxes,
            fontweight="bold",
        )
    axes[0, 0].legend(fontsize=7)
    figure.suptitle(
        "Exp56 Stage 4 — added information (solid: previous; dashed: shuffled)"
    )
    figure.tight_layout()
    save_fig(figure, FIGURE_DIRECTORY / f"{prefix}exp56_stage4_information_increment")
    plt.close(figure)


def feature_figure(
    sample: dict[str, np.ndarray], summaries: dict[str, np.ndarray], prefix: str
) -> None:
    mass = sample["features"][:, 0]
    recent = summaries["recent_growth_dex"]
    early = summaries["early_assembled_fraction"]
    concentration = sample["features"][:, 4]
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    definitions = (
        (mass, recent, r"$\log_{10}M_{\rm peak}$", r"recent growth (dex)"),
        (
            mass,
            early,
            r"$\log_{10}M_{\rm peak}$",
            r"$M_{\rm peak}(z=2)/M_{\rm peak}(z=0.4)$",
        ),
        (
            recent,
            early,
            "recent growth (dex)",
            r"$M_{\rm peak}(z=2)/M_{\rm peak}(z=0.4)$",
        ),
    )
    for panel, (axis, definition) in enumerate(zip(axes, definitions)):
        x_value, y_value, xlabel, ylabel = definition
        points = axis.scatter(
            x_value,
            y_value,
            c=concentration,
            cmap="cividis",
            s=7,
            alpha=0.45,
            linewidths=0,
        )
        rho = spearmanr(x_value, y_value).statistic
        axis.set(xlabel=xlabel, ylabel=ylabel, title=rf"Spearman $\rho={rho:+.2f}$")
        axis.text(
            -0.15,
            1.06,
            chr(65 + panel),
            transform=axis.transAxes,
            fontweight="bold",
        )
    colorbar = figure.colorbar(points, ax=axes, fraction=0.025, pad=0.02)
    colorbar.set_label(r"$c_{200c}$")
    figure.suptitle("Exp56 Stage 4 — tested halo summaries")
    figure.subplots_adjust(left=0.08, right=0.92, bottom=0.16, top=0.80, wspace=0.34)
    save_fig(figure, FIGURE_DIRECTORY / f"{prefix}exp56_stage4_features")
    plt.close(figure)


def cases_figure(
    sample: dict[str, np.ndarray], relations: dict[str, dict], prefix: str
) -> None:
    truth = sample["truth_log"]
    primary = relations["complete_c"]["mean_log"]
    rms = np.sqrt(np.mean((primary - truth) ** 2, axis=1))
    ordered = np.argsort(rms)
    cases = (ordered[0], ordered[len(ordered) // 2], ordered[-1])
    figure, axes = plt.subplots(2, 3, figsize=(10.5, 6.5), sharex=True)
    for column, galaxy in enumerate(cases):
        anchor = truth[galaxy, -1]
        axes[0, column].plot(
            stage3.stage_1.RADII,
            truth[galaxy] - anchor,
            color="black",
            marker="o",
            markersize=2.5,
            label="TNG",
        )
        for relation in REAL_RELATIONS:
            prediction = relations[relation]["mean_log"][galaxy]
            axes[0, column].plot(
                stage3.stage_1.RADII,
                prediction - prediction[-1],
                color=RELATION_COLORS[relation],
                linewidth=1.1,
                label=RELATION_LABELS[relation],
            )
            residual = 100.0 * (
                10.0 ** ((prediction - truth[galaxy]) - (prediction[-1] - anchor)) - 1.0
            )
            axes[1, column].plot(
                stage3.stage_1.RADII,
                residual,
                color=RELATION_COLORS[relation],
                linewidth=1.1,
            )
        axes[0, column].set_title(
            ("best", "typical", "worst")[column]
            + rf": $\log M_{{\rm peak}}={sample['features'][galaxy, 0]:.2f}$"
        )
        axes[1, column].axhline(0.0, color="0.35", linewidth=0.8)
        axes[1, column].axvspan(10.0, 30.0, color="0.92", zorder=-1)
        axes[1, column].set_xlabel("Radius (kpc)")
        for row in range(2):
            axes[row, column].set_xscale("log")
    axes[0, 0].set_ylabel(r"$\log_{10}[M_*(<R)/M_{*,148}]$")
    axes[1, 0].set_ylabel("Amplitude-pinned CoG residual (%)")
    axes[0, 0].legend(fontsize=6, ncol=2)
    figure.suptitle("Exp56 Stage 4 — direct held-out galaxies")
    figure.tight_layout()
    save_fig(figure, FIGURE_DIRECTORY / f"{prefix}exp56_stage4_cases")
    plt.close(figure)


def generate_standard_qa(
    sample: dict[str, np.ndarray], relation: dict, prefix: str
) -> dict:
    model_name = f"exp56_stage4_{prefix}complete_c"
    figure_directory = FIGURE_DIRECTORY / f"{prefix}complete_c_standard_qa"
    figure_directory.mkdir(parents=True, exist_ok=True)
    result = qa.evaluate(
        10.0 ** relation["mean_log"][:, None, :],
        10.0 ** sample["truth_log"][:, None, :],
        stage3.stage_1.RADII,
        [0.4],
        name=model_name,
        figdir=figure_directory,
        bin_by=sample["features"][:, 0],
        bin_label=r"DiffMAH $\log M_{\rm peak}$",
        draw_cogs=10.0 ** relation["draw_log"][:, :, None, :],
    )
    density_by_halo_mass_figure(
        relation["mean_log"],
        sample["truth_log"],
        sample["features"][:, 0],
        figdir=figure_directory,
        model_name=model_name,
        model_label=RELATION_LABELS["complete_c"],
    )
    single_epoch_planes_figure(
        result,
        figdir=figure_directory,
        model_name=model_name,
        model_label=RELATION_LABELS["complete_c"],
    )
    single_epoch_size_figure(
        relation["mean_log"],
        sample["truth_log"],
        result,
        figdir=figure_directory,
        model_name=model_name,
        model_label=RELATION_LABELS["complete_c"],
    )
    return standard_summary(result)


def save_predictions(
    sample: dict[str, np.ndarray],
    relations: dict[str, dict],
    summaries: dict[str, np.ndarray],
    prefix: str,
) -> None:
    arrays = {
        "indices": sample["indices"],
        "features": sample["features"],
        "truth_log": sample["truth_log"],
        "recent_growth_dex": summaries["recent_growth_dex"],
        "log_early_assembled_fraction": summaries["log_early_assembled_fraction"],
    }
    for name, relation in relations.items():
        arrays[f"{name}_mean_parameters"] = relation["mean_parameters"]
        arrays[f"{name}_draw_parameters"] = relation["draw_parameters"]
        arrays[f"{name}_mean_log"] = relation["mean_log"]
        arrays[f"{name}_draw_log"] = relation["draw_log"]
    np.savez_compressed(OUTPUT_DIRECTORY / f"{prefix}predictions.npz", **arrays)


def run_experiment(n_max: int, n_draw: int) -> dict:
    started = time.perf_counter()
    prefix = artifact_prefix(n_max)
    sample = stage3.stage_1.load_sample(n_max)
    demo = bool(n_max)
    if demo and len(sample["indices"]) != N_DEMO_GALAXY:
        raise RuntimeError("Stage 4 demo did not load exactly 90 galaxies")
    summaries = assembly_summaries(sample["features"])
    diagnostics = feature_diagnostics(sample, summaries)
    feature_sets = relation_features(sample["features"], summaries)
    relations = cross_validated_relations(sample, feature_sets, n_draw, demo)
    relation_summaries = {
        name: summarize_relation(
            relation, sample["truth_log"], sample["features"][:, 0]
        )
        for name, relation in relations.items()
    }
    comparison, _ = compare_information(relations, sample["truth_log"])
    adequacy = stage3_adequacy(prefix, relations["complete_c"], sample["truth_log"])

    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    radial_accuracy_figure(relation_summaries, prefix)
    annular_accuracy_figure(relation_summaries, prefix)
    information_increment_figure(comparison, prefix)
    feature_figure(sample, summaries, prefix)
    cases_figure(sample, relations, prefix)
    standard = generate_standard_qa(sample, relations["complete_c"], prefix)
    save_predictions(sample, relations, summaries, prefix)
    elapsed_seconds = time.perf_counter() - started

    primary = relation_summaries["complete_c"]
    finite = all(
        np.isfinite(relation["mean_log"]).all()
        and np.isfinite(relation["draw_log"]).all()
        for relation in relations.values()
    )
    monotonic = all(
        np.all(np.diff(relation["mean_log"], axis=1) >= -1e-12)
        and np.all(np.diff(relation["draw_log"], axis=2) >= -1e-12)
        for relation in relations.values()
    )
    result = {
        "status": "stage4_validation" if demo else "stage4_full",
        "n_galaxy": len(sample["indices"]),
        "n_draw": n_draw,
        "elapsed_seconds": elapsed_seconds,
        "feature_diagnostics": diagnostics,
        "relations": relation_summaries,
        "comparison": comparison,
        "stage3_adequacy": adequacy,
        "standard_qa": standard,
        "fold_metadata": {
            name: relation["metadata"] for name, relation in relations.items()
        },
        "integrity": {"finite": finite, "monotonic_cogs": monotonic},
    }
    if demo:
        result["complete_path_passed"] = bool(
            elapsed_seconds < DEMO_TIME_LIMIT_SECONDS
            and finite
            and monotonic
            and all(len(relation["metadata"]) == 5 for relation in relations.values())
            and abs(primary["profile_coverage_68"] - 0.68) <= 0.03
        )
    (OUTPUT_DIRECTORY / f"{prefix}results.json").write_text(
        json.dumps(json_safe(result), indent=2)
    )
    write_manifest(
        OUTPUT_DIRECTORY / f"{prefix}manifest",
        params={
            "script": Path(__file__).name,
            "status": result["status"],
            "n_galaxy": len(sample["indices"]),
            "n_draw": n_draw,
            "relations": REAL_RELATIONS,
            "controls": CONTROL_RELATIONS,
            "terms": MASS_TERMS,
            "compact_index": 1.0,
            "moffat_gamma": stage3.stage_1.FIXED_GAMMA,
        },
    )
    return result


def demo() -> None:
    result = run_experiment(N_DEMO_GALAXY, N_DEMO_DRAW)
    if not result["complete_path_passed"]:
        raise AssertionError(
            "Exp56 Stage 4 demo failed: "
            f"runtime={result['elapsed_seconds']:.2f} s, "
            f"coverage={result['relations']['complete_c']['profile_coverage_68']:.3f}, "
            f"integrity={result['integrity']}"
        )
    print(
        f"Exp56 Stage 4 complete 90-galaxy path passed in "
        f"{result['elapsed_seconds']:.2f} s",
        flush=True,
    )


def full_run() -> None:
    validation_path = OUTPUT_DIRECTORY / "demo_results.json"
    if not validation_path.exists():
        raise RuntimeError("run the Stage 4 demo before the full calculation")
    validation = json.loads(validation_path.read_text())
    if not validation.get("complete_path_passed", False):
        raise RuntimeError("the saved Stage 4 demo did not pass its complete gate")
    result = run_experiment(0, N_FULL_DRAW)
    print(
        f"Exp56 Stage 4 full information test complete: {result['n_galaxy']} "
        f"galaxies in {result['elapsed_seconds']:.1f} s",
        flush=True,
    )


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if command == "demo":
        demo()
    elif command == "run":
        full_run()
    else:
        raise SystemExit("usage: stage4_halo_information.py {demo|run}")
