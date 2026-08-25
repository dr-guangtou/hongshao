"""Exp56 Stage 3: refit the held-out halo–CoG relation.

Commands
--------
demo
    Run both retained representations, controls, stochastic calibration, and
    standard QA on the predeclared 90-galaxy sample with eight draws.
run
    Run the full 2,539-galaxy calculation with 32 draws after the demo passes.
figures
    Regenerate the custom and standard QA figures from saved full predictions.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP55 = ROOT / "experiments" / "exp55_analytic_cog"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP55))

from halo_map import (  # noqa: E402
    empirical_crps,
    folds_of,
    physical_parameters,
    shuffled_within_mass_bins,
    standard_summary,
)
from hongshao import qa  # noqa: E402
from hongshao.plotting import OKABE_ITO, save_fig, set_style  # noqa: E402
from hongshao.profile_emulator import density_from_cog  # noqa: E402
from hongshao.provenance import write_manifest  # noqa: E402
from outer_envelope import (  # noqa: E402
    PLANE_KEYS,
    individual_population_metrics,
    profile_draw_summary,
    profile_point_summary,
    size_relation,
    size_values,
    stochastic_population_summary,
    summarize_individual_populations,
)
from qa_figures import (  # noqa: E402
    density_by_halo_mass_figure,
    single_epoch_planes_figure,
    single_epoch_size_figure,
)
from refine_halo_map import SPARSE_TERMS, fit_conditional_model  # noqa: E402
from smooth_outer import (  # noqa: E402
    augmented_features,
    calibrated_fold_draws,
    decode_profiles,
)

stage_1_spec = importlib.util.spec_from_file_location(
    "exp56_stage_1_for_stage_3", HERE / "run.py"
)
if stage_1_spec is None or stage_1_spec.loader is None:
    raise RuntimeError("cannot load the Exp56 Stage 1 driver")
stage_1 = importlib.util.module_from_spec(stage_1_spec)
stage_1_spec.loader.exec_module(stage_1)

set_style()
# The user's global matplotlib configuration enables external LaTeX.  MathText
# renders this driver's notation correctly and keeps the complete demo path
# inside the predeclared one-minute validation budget.
matplotlib.rcParams["text.usetex"] = False

REPRESENTATIONS = ("fixed_gamma1p4", "smooth")
REPRESENTATION_LABELS = {
    "fixed_gamma1p4": r"fixed $\gamma=1.4$",
    "smooth": r"smooth $\gamma(\log M_{\rm peak})$",
}
CONTROL_NAMES = (
    "final_mass_only",
    "complete",
    "mah_shape_shuffled",
    "concentration_shuffled",
)
N_DEMO_GALAXY = 90
N_DEMO_DRAW = 8
N_FULL_DRAW = 32
N_BOOTSTRAP = 1000
DEMO_TIME_LIMIT_SECONDS = 60.0
SEED = 5630

FIGURE_DIRECTORY = HERE / "figures" / "stage3"
OUTPUT_DIRECTORY = HERE / "outputs" / "stage3"


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


def profile_checkpoint(
    representation: str,
    fold_number: int,
    demo: bool,
) -> Path:
    directory = HERE / "outputs" / ("demo_profile_fits" if demo else "profile_fits")
    if representation == "fixed_gamma1p4":
        return directory / "fixed_gamma1p4_n1p00.npz"
    if representation == "smooth":
        return directory / f"smooth_fold{fold_number}_n1p00.npz"
    raise ValueError(f"unknown representation {representation!r}")


def load_profile_cell(
    sample: dict[str, np.ndarray],
    representation: str,
    fold_number: int,
    demo: bool,
) -> dict[str, np.ndarray]:
    checkpoint = profile_checkpoint(representation, fold_number, demo)
    if not checkpoint.exists():
        raise RuntimeError(f"missing completed Stage 1 checkpoint {checkpoint}")
    cell = dict(np.load(checkpoint))
    if not np.array_equal(cell["galaxy_indices"], sample["indices"]):
        raise RuntimeError(f"Stage 1 checkpoint indices do not match: {checkpoint}")
    if not np.isclose(float(cell["compact_index"]), 1.0):
        raise RuntimeError(f"Stage 3 requires global n=1: {checkpoint}")
    return cell


def control_feature_sets(features: np.ndarray) -> dict[str, tuple[np.ndarray, tuple]]:
    return {
        "final_mass_only": (features[:, :1], ()),
        "complete": (features, SPARSE_TERMS),
        "mah_shape_shuffled": (
            shuffled_within_mass_bins(features, (1, 2, 3), seed=SEED + 1),
            SPARSE_TERMS,
        ),
        "concentration_shuffled": (
            shuffled_within_mass_bins(features, (4,), seed=SEED + 2),
            SPARSE_TERMS,
        ),
    }


def cross_validated_relation(
    sample: dict[str, np.ndarray],
    representation: str,
    n_draw: int,
    demo: bool,
) -> dict[str, np.ndarray | list[dict]]:
    features = sample["features"]
    truth_log = sample["truth_log"]
    n_galaxy = len(features)
    mean_parameters = np.empty((n_galaxy, 4))
    draw_parameters = np.empty((n_draw, n_galaxy, 4))
    fitted_parameters = np.empty((n_galaxy, 4))
    representation_log = np.empty_like(truth_log)
    gamma_by_galaxy = np.empty(n_galaxy)
    fold_id = np.empty(n_galaxy, int)
    control_predictions = {name: np.empty_like(truth_log) for name in CONTROL_NAMES}
    variants = control_feature_sets(features)
    metadata = []
    all_rows = np.arange(n_galaxy)

    for fold_number, test in enumerate(folds_of(n_galaxy)):
        train = np.setdiff1d(all_rows, test)
        cell = load_profile_cell(sample, representation, fold_number, demo)
        gamma = np.asarray(cell["gamma"], float)
        targets = np.asarray(cell["parameters"], float)
        fitted_parameters[test] = targets[test]
        representation_log[test] = cell["prediction_log"][test]
        gamma_by_galaxy[test] = gamma[test]
        fold_id[test] = fold_number

        complete_train = augmented_features(features[train], gamma[train])
        complete_test = augmented_features(features[test], gamma[test])
        complete_model = fit_conditional_model(
            complete_train,
            targets[train],
            SPARSE_TERMS,
        )
        mean_parameters[test] = complete_model.predict_mean(complete_test)
        draw_parameters[:, test], calibration = calibrated_fold_draws(
            complete_train,
            targets[train],
            gamma[train],
            truth_log[train],
            complete_test,
            gamma[test],
            complete_model,
            n_draw,
            SEED + 1000 * fold_number,
        )

        for name, (feature_values, terms) in variants.items():
            train_values = augmented_features(feature_values[train], gamma[train])
            test_values = augmented_features(feature_values[test], gamma[test])
            if name == "complete":
                model = complete_model
            else:
                model = fit_conditional_model(
                    train_values,
                    targets[train],
                    terms,
                )
            control_predictions[name][test] = decode_profiles(
                model.predict_mean(test_values), gamma[test]
            )

        metadata.append(
            {
                "fold": fold_number,
                "n_train": len(train),
                "n_test": len(test),
                "gamma_min": float(np.min(gamma)),
                "gamma_max": float(np.max(gamma)),
                **calibration,
            }
        )

    return {
        "mean_parameters": mean_parameters,
        "draw_parameters": draw_parameters,
        "fitted_parameters": fitted_parameters,
        "mean_log": decode_profiles(mean_parameters, gamma_by_galaxy),
        "draw_log": decode_profiles(draw_parameters, gamma_by_galaxy),
        "representation_log": representation_log,
        "gamma": gamma_by_galaxy,
        "fold_id": fold_id,
        "control_predictions": control_predictions,
        "metadata": metadata,
    }


def per_galaxy_metrics(
    truth_log: np.ndarray,
    mean_log: np.ndarray,
    draw_log: np.ndarray,
) -> dict[str, np.ndarray]:
    error = mean_log - truth_log
    pinned_error = error - error[:, -1:]
    radial_mask = (stage_1.RADII >= 5.0) & (stage_1.RADII <= 30.0)
    model_density, density_radius = density_from_cog(mean_log, stage_1.RADII)
    truth_density, _ = density_from_cog(truth_log, stage_1.RADII)
    density_error = model_density - truth_density
    density_mask = (density_radius >= 5.0) & (density_radius <= 30.0)
    size_error = size_values(mean_log) - size_values(truth_log)
    return {
        "profile_crps_dex": np.mean(empirical_crps(truth_log, draw_log), axis=1),
        "full_cog_rms_dex": np.sqrt(np.mean(error**2, axis=1)),
        "five_thirty_cog_rms_dex": np.sqrt(
            np.mean(pinned_error[:, radial_mask] ** 2, axis=1)
        ),
        "full_density_rms_dex": np.sqrt(np.mean(density_error**2, axis=1)),
        "five_thirty_density_rms_dex": np.sqrt(
            np.mean(density_error[:, density_mask] ** 2, axis=1)
        ),
        "absolute_R50_error": np.abs(size_error[:, 0]),
        "absolute_R80_error": np.abs(size_error[:, 1]),
        "absolute_R90_error": np.abs(size_error[:, 2]),
    }


def bootstrap_median_interval(difference: np.ndarray, seed: int) -> list[float]:
    difference = np.asarray(difference, float)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(N_BOOTSTRAP)
    for bootstrap_number in range(N_BOOTSTRAP):
        rows = rng.integers(0, len(difference), len(difference))
        bootstrap[bootstrap_number] = np.median(difference[rows])
    return np.quantile(bootstrap, (0.16, 0.50, 0.84)).tolist()


def worst_mass_bin_shape_bootstrap(
    truth_log: np.ndarray,
    fixed_log: np.ndarray,
    smooth_log: np.ndarray,
    halo_mass: np.ndarray,
) -> list[float]:
    order = np.argsort(halo_mass)
    bins = np.array_split(order, 3)
    fixed_shape = 100.0 * (
        10.0 ** ((fixed_log - truth_log) - (fixed_log - truth_log)[:, -1:]) - 1.0
    )
    smooth_shape = 100.0 * (
        10.0 ** ((smooth_log - truth_log) - (smooth_log - truth_log)[:, -1:]) - 1.0
    )
    rng = np.random.default_rng(SEED + 50)
    difference = np.empty(N_BOOTSTRAP)
    for bootstrap_number in range(N_BOOTSTRAP):
        fixed_curves = []
        smooth_curves = []
        for members in bins:
            rows = members[rng.integers(0, len(members), len(members))]
            fixed_curves.append(np.median(fixed_shape[rows], axis=0))
            smooth_curves.append(np.median(smooth_shape[rows], axis=0))
        difference[bootstrap_number] = np.max(np.abs(smooth_curves)) - np.max(
            np.abs(fixed_curves)
        )
    return np.quantile(difference, (0.16, 0.50, 0.84)).tolist()


def coordinate_r2(
    fitted_parameters: np.ndarray,
    mean_parameters: np.ndarray,
) -> dict[str, float]:
    truth = physical_parameters(fitted_parameters)
    prediction = physical_parameters(mean_parameters)
    names = (
        "log_mstar_148",
        "compact_fraction_148",
        "log_r_compact",
        "log_r_extended",
    )
    variance = np.var(truth, axis=0)
    values = 1.0 - np.mean((prediction - truth) ** 2, axis=0) / np.clip(
        variance, 1e-20, None
    )
    return dict(zip(names, values.tolist()))


def summarize_model(
    sample: dict[str, np.ndarray],
    relation: dict,
    n_population_draw: int,
) -> dict:
    truth_log = sample["truth_log"]
    mean_log = relation["mean_log"]
    draw_log = relation["draw_log"]
    point = profile_point_summary(mean_log, truth_log, sample["features"][:, 0])
    draw = profile_draw_summary(draw_log, truth_log)
    population = stochastic_population_summary(
        draw_log,
        truth_log,
        n_population_draw=n_population_draw,
    )
    individual_populations = individual_population_metrics(
        draw_log,
        truth_log,
        min(8, len(draw_log)),
    )
    metrics = per_galaxy_metrics(truth_log, mean_log, draw_log)
    control_summaries = {
        name: profile_point_summary(
            relation["control_predictions"][name],
            truth_log,
            sample["features"][:, 0],
        )
        for name in CONTROL_NAMES
    }
    return {
        "point": point,
        "draw": draw,
        "coordinate_r2": coordinate_r2(
            relation["fitted_parameters"], relation["mean_parameters"]
        ),
        "median_five_thirty_cog_rms_dex": float(
            np.median(metrics["five_thirty_cog_rms_dex"])
        ),
        "median_five_thirty_density_rms_dex": float(
            np.median(metrics["five_thirty_density_rms_dex"])
        ),
        "size_relation": size_relation(mean_log),
        "population": {
            "planes": population["planes"],
            "sizes": population["sizes"],
            "draw_intervals": summarize_individual_populations(individual_populations),
        },
        "controls": control_summaries,
    }


def compare_representations(
    sample: dict[str, np.ndarray],
    relations: dict[str, dict],
    summaries: dict[str, dict],
) -> dict:
    truth_log = sample["truth_log"]
    fixed = relations["fixed_gamma1p4"]
    smooth = relations["smooth"]
    fixed_metrics = per_galaxy_metrics(truth_log, fixed["mean_log"], fixed["draw_log"])
    smooth_metrics = per_galaxy_metrics(
        truth_log, smooth["mean_log"], smooth["draw_log"]
    )
    paired = {
        name: bootstrap_median_interval(
            smooth_metrics[name] - fixed_metrics[name],
            SEED + index,
        )
        for index, name in enumerate(fixed_metrics)
    }
    paired["worst_mass_bin_shape_residual_percent"] = worst_mass_bin_shape_bootstrap(
        truth_log,
        fixed["mean_log"],
        smooth["mean_log"],
        sample["features"][:, 0],
    )
    fixed_summary = summaries["fixed_gamma1p4"]
    smooth_summary = summaries["smooth"]
    smooth_coverage = smooth_summary["draw"]["profile_coverage_68"]
    gates = {
        "coverage_within_three_percentage_points": abs(smooth_coverage - 0.68) <= 0.03,
        "profile_crps_within_one_percent": smooth_summary["draw"]["profile_crps_dex"]
        <= 1.01 * fixed_summary["draw"]["profile_crps_dex"],
        "full_cog_not_resolved_worse": paired["full_cog_rms_dex"][0] <= 0.0,
        "full_density_not_resolved_worse": paired["full_density_rms_dex"][0] <= 0.0,
        "mass_bin_shape_not_resolved_worse": paired[
            "worst_mass_bin_shape_residual_percent"
        ][0]
        <= 0.0,
        "R50_not_resolved_worse": paired["absolute_R50_error"][0] <= 0.0,
        "R80_not_resolved_worse": paired["absolute_R80_error"][0] <= 0.0,
        "R90_not_resolved_worse": paired["absolute_R90_error"][0] <= 0.0,
        "generated_R90_distance_improves": smooth_summary["population"]["sizes"]["R90"][
            "energy_ratio"
        ]
        < fixed_summary["population"]["sizes"]["R90"]["energy_ratio"],
        "self_consistent_size_plane_improves": smooth_summary["population"]["planes"][
            PLANE_KEYS[2]
        ]["energy_ratio"]
        < fixed_summary["population"]["planes"][PLANE_KEYS[2]]["energy_ratio"],
    }
    return {
        "paired_smooth_minus_fixed_bootstrap_16_50_84": paired,
        "numeric_gates": gates,
        "all_numeric_gates_pass": bool(all(gates.values())),
        "stage4_primary_before_visual_review": (
            "smooth" if all(gates.values()) else "fixed_gamma1p4"
        ),
    }


def comparison_figure(
    sample: dict[str, np.ndarray],
    relations: dict[str, dict],
    summaries: dict[str, dict],
    prefix: str,
) -> None:
    truth_log = sample["truth_log"]
    halo_mass = sample["features"][:, 0]
    truth_density, density_radius = density_from_cog(truth_log, stage_1.RADII)
    truth_sizes = size_values(truth_log)
    colors = {"fixed_gamma1p4": OKABE_ITO[1], "smooth": OKABE_ITO[2]}
    figure, axes = plt.subplots(2, 3, figsize=(15.2, 8.3))

    metrics = (
        ("profile_crps_dex", "Profile CRPS (dex)"),
        ("median_profile_rms_dex", "Median CoG RMS (dex)"),
        ("median_density_rms_dex", "Median density RMS (dex)"),
    )
    for position, (metric, label) in enumerate(metrics):
        values = [
            summaries[name]["draw"].get(metric, summaries[name]["point"].get(metric))
            for name in REPRESENTATIONS
        ]
        axes[0, 0].bar(
            position + np.array([-0.18, 0.18]),
            values,
            0.34,
            color=[colors[name] for name in REPRESENTATIONS],
        )
    axes[0, 0].set_xticks(np.arange(3), [label for _, label in metrics], rotation=15)
    axes[0, 0].set_ylabel("Held-out error")

    for name in REPRESENTATIONS:
        error = relations[name]["mean_log"] - truth_log
        axes[0, 1].plot(
            stage_1.RADII,
            100.0 * np.median(10.0 ** (error - error[:, -1:]) - 1.0, axis=0),
            color=colors[name],
            label=REPRESENTATION_LABELS[name],
        )
        density, _ = density_from_cog(relations[name]["mean_log"], stage_1.RADII)
        axes[1, 1].plot(
            density_radius,
            np.median(density - truth_density, axis=0),
            color=colors[name],
        )
    axes[0, 1].set(xscale="log", ylabel="Median pinned CoG residual (%)")
    axes[1, 1].set(
        xscale="log",
        xlabel="Radius (kpc)",
        ylabel="Median density residual (dex)",
    )
    axes[0, 1].legend(frameon=False, fontsize=8)
    for axis in axes[:, 1]:
        axis.axhline(0.0, color="0.5", linewidth=0.8)
        axis.axvspan(5.0, 30.0, color="0.85", alpha=0.35)

    control_labels = (
        "mass only",
        "complete",
        "MAH shuffled",
        "concentration shuffled",
    )
    x = np.arange(len(CONTROL_NAMES))
    for offset, name in zip((-0.18, 0.18), REPRESENTATIONS):
        values = [
            summaries[name]["controls"][control]["median_profile_rms_dex"]
            for control in CONTROL_NAMES
        ]
        axes[0, 2].bar(
            x + offset,
            values,
            0.34,
            color=colors[name],
            label=REPRESENTATION_LABELS[name],
        )
    axes[0, 2].set_xticks(x, control_labels, rotation=18)
    axes[0, 2].set_ylabel("Median held-out CoG RMS (dex)")
    axes[0, 2].legend(frameon=False, fontsize=7)

    size_x = np.arange(3)
    for offset, name in zip((-0.18, 0.18), REPRESENTATIONS):
        size_error = size_values(relations[name]["mean_log"]) - truth_sizes
        axes[1, 0].bar(
            size_x + offset,
            np.median(size_error, axis=0),
            0.34,
            color=colors[name],
        )
    axes[1, 0].axhline(0.0, color="0.5", linewidth=0.8)
    axes[1, 0].set_xticks(size_x, ("R50", "R80", "R90"))
    axes[1, 0].set_ylabel("Median model-minus-TNG size (dex)")

    population_labels = ("self-consistent\nsize plane", "R90")
    for offset, name in zip((-0.18, 0.18), REPRESENTATIONS):
        population_values = (
            summaries[name]["population"]["planes"][PLANE_KEYS[2]]["energy_ratio"],
            summaries[name]["population"]["sizes"]["R90"]["energy_ratio"],
        )
        axes[1, 2].bar(
            np.arange(2) + offset,
            population_values,
            0.34,
            color=colors[name],
        )
    axes[1, 2].axhline(1.0, color="0.5", linestyle="--", linewidth=0.8)
    axes[1, 2].set_xticks(np.arange(2), population_labels)
    axes[1, 2].set_ylabel("Energy distance / TNG sampling floor")

    for panel, axis in zip("ABCDEF", axes.ravel()):
        axis.text(-0.13, 1.04, panel, transform=axis.transAxes, fontweight="bold")
    axes[0, 0].set_title(
        rf"$N={len(halo_mass)}$ held-out galaxies; all values decoded to CoGs"
    )
    figure.suptitle("exp56 Stage 3 — held-out halo–CoG relation")
    figure.tight_layout()
    save_fig(figure, FIGURE_DIRECTORY / f"{prefix}exp56_stage3_comparison")
    plt.close(figure)


def coordinate_figure(
    relations: dict[str, dict],
    prefix: str,
) -> None:
    names = (
        r"$\log_{10}M_{*,148}$",
        r"$f_{\rm compact,148}$",
        r"$\log_{10}R_{\rm compact}$",
        r"$\log_{10}R_{\rm extended}$",
    )
    figure, axes = plt.subplots(2, 4, figsize=(15.0, 7.0), sharex="col")
    colors = {"fixed_gamma1p4": OKABE_ITO[1], "smooth": OKABE_ITO[2]}
    for column, label in enumerate(names):
        for representation in REPRESENTATIONS:
            truth = physical_parameters(relations[representation]["fitted_parameters"])[
                :, column
            ]
            prediction = physical_parameters(
                relations[representation]["mean_parameters"]
            )[:, column]
            axes[0, column].scatter(
                truth,
                prediction,
                s=5,
                alpha=0.18,
                color=colors[representation],
                edgecolors="none",
                label=REPRESENTATION_LABELS[representation],
            )
            axes[1, column].scatter(
                truth,
                prediction - truth,
                s=5,
                alpha=0.18,
                color=colors[representation],
                edgecolors="none",
            )
        low, high = axes[0, column].get_xlim()
        axes[0, column].plot([low, high], [low, high], "k--", linewidth=0.8)
        axes[1, column].axhline(0.0, color="0.5", linewidth=0.8)
        axes[0, column].set_title(label)
        axes[1, column].set_xlabel(f"fitted coordinate: {label}")
    axes[0, 0].set_ylabel("Held-out conditional mean")
    axes[1, 0].set_ylabel("Prediction minus fitted coordinate")
    axes[0, 0].legend(frameon=False, fontsize=7)
    for panel, axis in zip("ABCDEFGH", axes.ravel()):
        axis.text(-0.15, 1.05, panel, transform=axis.transAxes, fontweight="bold")
    figure.suptitle("exp56 Stage 3 — prediction of the four profile coordinates")
    figure.tight_layout()
    save_fig(figure, FIGURE_DIRECTORY / f"{prefix}exp56_stage3_coordinates")
    plt.close(figure)


def cases_figure(
    sample: dict[str, np.ndarray],
    relations: dict[str, dict],
    primary: str,
    prefix: str,
) -> None:
    truth_log = sample["truth_log"]
    primary_error = np.sqrt(
        np.mean((relations[primary]["mean_log"] - truth_log) ** 2, axis=1)
    )
    order = np.argsort(primary_error)
    galaxies = order[[0, len(order) // 2, -1]]
    figure, axes = plt.subplots(2, 3, figsize=(13.8, 7.0))
    colors = {"fixed_gamma1p4": OKABE_ITO[1], "smooth": OKABE_ITO[2]}
    labels = ("best", "typical", "worst")
    for column, (galaxy, case_label) in enumerate(zip(galaxies, labels)):
        truth = truth_log[galaxy] - truth_log[galaxy, -1]
        axes[0, column].plot(
            stage_1.RADII,
            truth,
            "o-",
            color="black",
            ms=3,
            label="TNG",
        )
        for representation in REPRESENTATIONS:
            prediction = relations[representation]["mean_log"][galaxy]
            prediction = prediction - prediction[-1]
            axes[0, column].plot(
                stage_1.RADII,
                prediction,
                color=colors[representation],
                label=REPRESENTATION_LABELS[representation],
            )
            axes[1, column].plot(
                stage_1.RADII,
                100.0 * (10.0 ** (prediction - truth) - 1.0),
                color=colors[representation],
            )
        axes[0, column].set_xscale("log")
        axes[1, column].set_xscale("log")
        axes[1, column].axhline(0.0, color="0.5", linewidth=0.8)
        axes[1, column].axvspan(5.0, 30.0, color="0.85", alpha=0.35)
        axes[0, column].set_title(
            rf"{case_label}: $\log M_{{\rm peak}}="
            rf"{sample['features'][galaxy, 0]:.2f}$"
        )
        axes[1, column].set_xlabel("Radius (kpc)")
    axes[0, 0].set_ylabel(r"$\log_{10}[M_*(<R)/M_{*,148}]$")
    axes[1, 0].set_ylabel("Amplitude-pinned residual (%)")
    axes[0, 0].legend(frameon=False, fontsize=7)
    for panel, axis in zip("ABCDEF", axes.ravel()):
        axis.text(-0.13, 1.04, panel, transform=axis.transAxes, fontweight="bold")
    figure.suptitle("exp56 Stage 3 — direct held-out galaxies")
    figure.tight_layout()
    save_fig(figure, FIGURE_DIRECTORY / f"{prefix}exp56_stage3_cases")
    plt.close(figure)


def generate_standard_qa(
    sample: dict[str, np.ndarray],
    relations: dict[str, dict],
    prefix: str,
) -> dict[str, dict]:
    output = {}
    for representation in REPRESENTATIONS:
        truth_log = sample["truth_log"]
        mean_log = relations[representation]["mean_log"]
        draw_log = relations[representation]["draw_log"]
        halo_mass = sample["features"][:, 0]
        model_name = f"exp56_stage3_{prefix}{representation}"
        figure_directory = FIGURE_DIRECTORY / f"{prefix}{representation}_standard_qa"
        figure_directory.mkdir(parents=True, exist_ok=True)
        qa_result = qa.evaluate(
            10.0 ** mean_log[:, None, :],
            10.0 ** truth_log[:, None, :],
            stage_1.RADII,
            [0.4],
            name=model_name,
            figdir=figure_directory,
            bin_by=halo_mass,
            bin_label=r"DiffMAH $\log M_{\rm peak}$",
            draw_cogs=10.0 ** draw_log[:, :, None, :],
        )
        density_by_halo_mass_figure(
            mean_log,
            truth_log,
            halo_mass,
            figdir=figure_directory,
            model_name=model_name,
            model_label=REPRESENTATION_LABELS[representation],
        )
        single_epoch_planes_figure(
            qa_result,
            figdir=figure_directory,
            model_name=model_name,
            model_label=REPRESENTATION_LABELS[representation],
        )
        single_epoch_size_figure(
            mean_log,
            truth_log,
            qa_result,
            figdir=figure_directory,
            model_name=model_name,
            model_label=REPRESENTATION_LABELS[representation],
        )
        output[representation] = standard_summary(qa_result)
    return output


def save_predictions(
    sample: dict[str, np.ndarray],
    relations: dict[str, dict],
    prefix: str,
) -> None:
    arrays = {
        "indices": sample["indices"],
        "features": sample["features"],
        "truth_log": sample["truth_log"],
    }
    for representation in REPRESENTATIONS:
        relation = relations[representation]
        for name in (
            "mean_parameters",
            "draw_parameters",
            "fitted_parameters",
            "mean_log",
            "draw_log",
            "representation_log",
            "gamma",
            "fold_id",
        ):
            arrays[f"{representation}_{name}"] = relation[name]
        for control in CONTROL_NAMES:
            arrays[f"{representation}_{control}_mean_log"] = relation[
                "control_predictions"
            ][control]
    np.savez_compressed(OUTPUT_DIRECTORY / f"{prefix}predictions.npz", **arrays)


def run_experiment(
    n_max: int,
    n_draw: int,
    n_population_draw: int,
) -> dict:
    started = time.perf_counter()
    prefix = artifact_prefix(n_max)
    sample = stage_1.load_sample(n_max)
    demo = bool(n_max)
    if demo and len(sample["indices"]) != N_DEMO_GALAXY:
        raise RuntimeError("Stage 3 demo did not load exactly 90 galaxies")
    relations = {
        representation: cross_validated_relation(
            sample,
            representation,
            n_draw,
            demo,
        )
        for representation in REPRESENTATIONS
    }
    summaries = {
        representation: summarize_model(
            sample,
            relations[representation],
            n_population_draw,
        )
        for representation in REPRESENTATIONS
    }
    comparison = compare_representations(sample, relations, summaries)

    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    comparison_figure(sample, relations, summaries, prefix)
    coordinate_figure(relations, prefix)
    cases_figure(
        sample,
        relations,
        comparison["stage4_primary_before_visual_review"],
        prefix,
    )
    standard = generate_standard_qa(sample, relations, prefix)
    save_predictions(sample, relations, prefix)
    elapsed_seconds = time.perf_counter() - started

    result = {
        "status": "stage3_validation" if demo else "stage3_full",
        "n_galaxy": len(sample["indices"]),
        "n_draw": n_draw,
        "n_population_draw": n_population_draw,
        "representations": summaries,
        "comparison": comparison,
        "standard_qa": standard,
        "fold_metadata": {
            representation: relations[representation]["metadata"]
            for representation in REPRESENTATIONS
        },
        "elapsed_seconds": elapsed_seconds,
    }
    if demo:
        result["complete_path_passed"] = bool(
            len(sample["indices"]) == N_DEMO_GALAXY
            and elapsed_seconds < DEMO_TIME_LIMIT_SECONDS
            and all(
                np.isfinite(relations[name]["mean_log"]).all()
                and np.isfinite(relations[name]["draw_log"]).all()
                and len(relations[name]["metadata"]) == 5
                for name in REPRESENTATIONS
            )
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
            "n_population_draw": n_population_draw,
            "representations": REPRESENTATIONS,
            "compact_index": 1.0,
            "smooth_width_dex": stage_1.SMOOTH_WIDTH_DEX,
        },
    )
    return result


def demo() -> None:
    result = run_experiment(
        n_max=N_DEMO_GALAXY,
        n_draw=N_DEMO_DRAW,
        n_population_draw=1,
    )
    if not result["complete_path_passed"]:
        raise AssertionError(
            "Exp56 Stage 3 complete demo failed its sub-minute gate: "
            f"{result['elapsed_seconds']:.2f} s"
        )
    print(
        f"Exp56 Stage 3 complete 90-galaxy path passed in "
        f"{result['elapsed_seconds']:.2f} s",
        flush=True,
    )


def full_run() -> None:
    demo_path = OUTPUT_DIRECTORY / "demo_results.json"
    if not demo_path.exists():
        raise RuntimeError("run the Stage 3 demo before the full calculation")
    demo_result = json.loads(demo_path.read_text())
    if not demo_result.get("complete_path_passed", False):
        raise RuntimeError("the saved Stage 3 complete-path validation did not pass")
    result = run_experiment(n_max=0, n_draw=N_FULL_DRAW, n_population_draw=4)
    print(
        f"Exp56 Stage 3 full relation complete: {result['n_galaxy']} galaxies "
        f"in {result['elapsed_seconds']:.1f} s",
        flush=True,
    )


def regenerate_figures() -> None:
    prefix = ""
    prediction_path = OUTPUT_DIRECTORY / "predictions.npz"
    result_path = OUTPUT_DIRECTORY / "results.json"
    if not prediction_path.exists() or not result_path.exists():
        raise RuntimeError("run the full Stage 3 calculation before figures")
    saved = np.load(prediction_path)
    sample = {
        "indices": saved["indices"],
        "features": saved["features"],
        "truth_log": saved["truth_log"],
    }
    relations = {}
    for representation in REPRESENTATIONS:
        relation = {
            name: saved[f"{representation}_{name}"]
            for name in (
                "mean_parameters",
                "draw_parameters",
                "fitted_parameters",
                "mean_log",
                "draw_log",
                "representation_log",
                "gamma",
                "fold_id",
            )
        }
        relation["control_predictions"] = {
            control: saved[f"{representation}_{control}_mean_log"]
            for control in CONTROL_NAMES
        }
        relations[representation] = relation
    result = json.loads(result_path.read_text())
    summaries = result["representations"]
    comparison_figure(sample, relations, summaries, prefix)
    coordinate_figure(relations, prefix)
    cases_figure(
        sample,
        relations,
        result["comparison"]["stage4_primary_before_visual_review"],
        prefix,
    )
    generate_standard_qa(sample, relations, prefix)
    print("Exp56 Stage 3 figures regenerated from saved predictions", flush=True)


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if command == "demo":
        demo()
    elif command == "run":
        full_run()
    elif command == "figures":
        regenerate_figures()
    else:
        raise SystemExit(f"unknown command {command!r}; use demo, run, or figures")
