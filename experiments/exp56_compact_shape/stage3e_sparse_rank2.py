"""Exp56 Stage 3e: exhaustive whole-term sparsification inside rank 2.

Commands
--------
checks
    Run support-grid and complete-rank-2 closure checks.
demo
    Run the complete predeclared path on 90 mass-stratified galaxies.
run
    Run the full five-fold held-out calculation after the demo gate passes.
figures
    Rebuild all Stage 3e figures from saved full predictions.
"""

from __future__ import annotations

import itertools
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
sys.path.insert(0, str(HERE))

import stage3_halo_cog as stage3  # noqa: E402
import stage3c_sparse_relation as stage3c  # noqa: E402
import stage3d_low_rank_relation as stage3d  # noqa: E402
from hongshao.plotting import OKABE_ITO, save_fig, set_style  # noqa: E402
from refine_halo_map import SPARSE_TERMS  # noqa: E402

set_style()
matplotlib.rcParams["text.usetex"] = False

RANK = 2
N_DEMO_GALAXY = 90
N_DEMO_DRAW = 8
N_FULL_DRAW = 32
N_POPULATION_DRAW = 4
DEMO_TIME_LIMIT_SECONDS = 60.0
MAXIMUM_EFFECTIVE_COUNT = 24
SEED = 5690

FIGURE_DIRECTORY = HERE / "figures" / "stage3e"
OUTPUT_DIRECTORY = HERE / "outputs" / "stage3e"
RANK2_OUTPUT_DIRECTORY = HERE / "outputs" / "stage3d"


def candidate_supports() -> tuple[np.ndarray, ...]:
    """Return all 128 whole-term nonlinear supports in deterministic order."""
    return tuple(
        np.asarray(values, dtype=bool)
        for values in itertools.product((False, True), repeat=len(SPARSE_TERMS))
    )


def terms_for_support(support: np.ndarray) -> tuple[tuple, ...]:
    support = np.asarray(support, bool)
    if support.shape != (len(SPARSE_TERMS),):
        raise ValueError("support must have one entry per established nonlinear term")
    return tuple(term for term, keep in zip(SPARSE_TERMS, support) if keep)


def effective_count_for_support(support: np.ndarray) -> int:
    """Identifiable rank-2 count with five mandatory linear predictors."""
    n_predictor = 5 + int(np.count_nonzero(support))
    return stage3d.effective_coefficient_count(RANK, n_predictor, 4)


def support_code(support: np.ndarray) -> str:
    return "".join("1" if value else "0" for value in np.asarray(support, bool))


def fit_supported_rank2_model(
    features: np.ndarray,
    targets: np.ndarray,
    support: np.ndarray,
) -> stage3d.ReducedRankModel:
    """Fit rank 2 with all linear terms and the selected nonlinear columns."""
    return stage3d.fit_reduced_rank_model(
        features,
        targets,
        terms_for_support(support),
        rank=RANK,
    )


def run_checks() -> dict:
    supports = candidate_supports()
    encoded = {support_code(support) for support in supports}
    rng = np.random.default_rng(SEED)
    features = rng.normal(size=(500, 5))
    targets = rng.normal(size=(500, 4))
    reference = stage3d.fit_reduced_rank_model(
        features, targets, SPARSE_TERMS, rank=RANK
    )
    candidate = fit_supported_rank2_model(
        features, targets, np.ones(len(SPARSE_TERMS), dtype=bool)
    )
    closure = float(
        np.max(
            np.abs(candidate.predict_mean(features) - reference.predict_mean(features))
        )
    )
    counts = sorted({effective_count_for_support(support) for support in supports})
    values = {
        "candidate_support_count": len(supports),
        "unique_support_count": len(encoded),
        "effective_coefficient_counts": counts,
        "complete_support_rank2_maximum_prediction_difference": closure,
        "passed": bool(
            len(supports) == 128
            and len(encoded) == 128
            and counts == [18, 20, 22, 24, 26, 28, 30, 32]
            and closure < 1e-12
        ),
    }
    print(json.dumps(stage3.json_safe(values), indent=2))
    return values


def candidate_passes_inner_envelope(
    metrics: dict[str, float], reference: dict[str, float]
) -> bool:
    return bool(
        metrics["full_cog_rms_dex"] <= 1.01 * reference["full_cog_rms_dex"]
        and metrics["five_thirty_cog_rms_dex"]
        <= reference["five_thirty_cog_rms_dex"] + 0.001
        and metrics["full_density_rms_dex"] <= reference["full_density_rms_dex"] + 0.001
        and metrics["five_thirty_density_rms_dex"]
        <= reference["five_thirty_density_rms_dex"] + 0.001
        and metrics["absolute_R50_error"] <= reference["absolute_R50_error"] + 0.001
        and metrics["absolute_R80_error"] <= reference["absolute_R80_error"] + 0.001
        and metrics["absolute_R90_error"] <= reference["absolute_R90_error"] + 0.001
    )


def select_nonlinear_support(
    features: np.ndarray,
    targets: np.ndarray,
    truth_log: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, list[dict]]:
    """Select the smallest decoded-profile-safe support in four inner folds."""
    folds = stage3c.deterministic_inner_folds(len(features), seed)
    all_rows = np.arange(len(features))
    gamma = np.full(len(features), 1.4)
    records = []
    for support in candidate_supports():
        mean_parameters = np.empty_like(targets)
        condition_numbers = []
        for validation in folds:
            training = np.setdiff1d(all_rows, validation)
            model = fit_supported_rank2_model(
                features[training], targets[training], support
            )
            mean_parameters[validation] = model.predict_mean(features[validation])
            condition_numbers.append(model.design_condition_number)
        metrics = stage3c.point_metric_summary(
            truth_log,
            stage3.decode_profiles(mean_parameters, gamma),
        )
        records.append(
            {
                "support": support,
                "support_code": support_code(support),
                "nonlinear_term_count": int(np.count_nonzero(support)),
                "effective_coefficient_count": effective_count_for_support(support),
                "median_design_condition_number": float(np.median(condition_numbers)),
                **metrics,
            }
        )
    reference = next(
        record for record in records if record["nonlinear_term_count"] == 7
    )
    for record in records:
        record["passes_inner_accuracy_envelope"] = candidate_passes_inner_envelope(
            record, reference
        )
    passing = [record for record in records if record["passes_inner_accuracy_envelope"]]
    selected = min(
        passing,
        key=lambda record: (
            record["effective_coefficient_count"],
            record["full_cog_rms_dex"],
            record["five_thirty_cog_rms_dex"],
            record["support_code"],
        ),
    )
    for record in records:
        record["selected"] = bool(record is selected)
    return np.asarray(selected["support"], bool), records


def calibrated_draws(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    train_truth_log: np.ndarray,
    test_features: np.ndarray,
    model: stage3d.ReducedRankModel,
    support: np.ndarray,
    n_draw: int,
    seed: int,
) -> tuple[np.ndarray, dict]:
    """Calibrate residual inflation inside the outer fold for one support."""
    scale_grid = np.asarray([1.00, 1.08, 1.16, 1.24, 1.32])
    folds = stage3c.deterministic_inner_folds(len(train_features), seed)
    all_rows = np.arange(len(train_features))
    inner_mean = np.empty_like(train_targets)
    inner_draw = np.empty((n_draw, *train_targets.shape))
    for inner_number, validation in enumerate(folds):
        training = np.setdiff1d(all_rows, validation)
        inner_model = fit_supported_rank2_model(
            train_features[training], train_targets[training], support
        )
        inner_mean[validation] = inner_model.predict_mean(train_features[validation])
        inner_draw[:, validation] = stage3c.neighbour_draws(
            inner_model,
            train_features[training],
            train_targets[training],
            train_features[validation],
            n_draw,
            seed + inner_number,
        )
    gamma = np.full(len(train_features), 1.4)
    coverage = {}
    for scale in scale_grid:
        draw_log = stage3.decode_profiles(
            inner_mean[None] + scale * (inner_draw - inner_mean[None]), gamma
        )
        coverage[f"{scale:.2f}"] = float(
            np.mean(
                (train_truth_log >= np.quantile(draw_log, 0.16, axis=0))
                & (train_truth_log <= np.quantile(draw_log, 0.84, axis=0))
            )
        )
    selected_scale = min(
        scale_grid,
        key=lambda value: (abs(coverage[f"{value:.2f}"] - 0.68), value),
    )
    base = stage3c.neighbour_draws(
        model,
        train_features,
        train_targets,
        test_features,
        n_draw,
        seed + 100,
    )
    mean = model.predict_mean(test_features)[None]
    return mean + selected_scale * (base - mean), {
        "selected_scale": float(selected_scale),
        "inner_coverage_by_scale": coverage,
    }


def cross_validated_relation(
    sample: dict[str, np.ndarray], n_draw: int, demo: bool
) -> dict:
    """Nested support selection and five-fold held-out rank-2 prediction."""
    features = sample["features"]
    truth_log = sample["truth_log"]
    n_galaxy = len(features)
    gamma = np.full(n_galaxy, 1.4)
    mean_parameters = np.empty((n_galaxy, 4))
    draw_parameters = np.empty((n_draw, n_galaxy, 4))
    fitted_parameters = np.empty((n_galaxy, 4))
    representation_log = np.empty_like(truth_log)
    fold_id = np.empty(n_galaxy, int)
    control_predictions = {
        name: np.empty_like(truth_log) for name in stage3.CONTROL_NAMES
    }
    variants = stage3.control_feature_sets(features)
    metadata = []
    all_rows = np.arange(n_galaxy)
    for fold_number, test in enumerate(stage3.folds_of(n_galaxy)):
        train = np.setdiff1d(all_rows, test)
        cell = stage3.load_profile_cell(sample, "fixed_gamma1p4", fold_number, demo)
        targets = np.asarray(cell["parameters"], float)
        fitted_parameters[test] = targets[test]
        representation_log[test] = cell["prediction_log"][test]
        fold_id[test] = fold_number
        support, selection_records = select_nonlinear_support(
            features[train],
            targets[train],
            truth_log[train],
            SEED + 1000 * fold_number,
        )
        model = fit_supported_rank2_model(features[train], targets[train], support)
        mean_parameters[test] = model.predict_mean(features[test])
        draw_parameters[:, test], calibration = calibrated_draws(
            features[train],
            targets[train],
            truth_log[train],
            features[test],
            model,
            support,
            n_draw,
            SEED + 10_000 + 1000 * fold_number,
        )
        selected_terms = terms_for_support(support)
        for name, (feature_values, _) in variants.items():
            if name == "complete":
                control_parameters = mean_parameters[test]
            else:
                terms = () if name == "final_mass_only" else selected_terms
                control_model = stage3d.fit_reduced_rank_model(
                    feature_values[train], targets[train], terms, rank=RANK
                )
                control_parameters = control_model.predict_mean(feature_values[test])
            control_predictions[name][test] = stage3.decode_profiles(
                control_parameters, gamma[test]
            )
        metadata.append(
            {
                "fold": fold_number,
                "n_train": len(train),
                "n_test": len(test),
                "selected_support": support,
                "selected_support_code": support_code(support),
                "selected_terms": selected_terms,
                "effective_coefficient_count": effective_count_for_support(support),
                "model": stage3d.model_metadata(model),
                "calibration": calibration,
                "selection_records": selection_records,
            }
        )
        print(
            f"fold {fold_number}: support {support_code(support)}, "
            f"{effective_count_for_support(support)} effective coefficients",
            flush=True,
        )
    return {
        "mean_parameters": mean_parameters,
        "draw_parameters": draw_parameters,
        "fitted_parameters": fitted_parameters,
        "mean_log": stage3.decode_profiles(mean_parameters, gamma),
        "draw_log": stage3.decode_profiles(draw_parameters, gamma),
        "representation_log": representation_log,
        "gamma": gamma,
        "fold_id": fold_id,
        "control_predictions": control_predictions,
        "metadata": metadata,
    }


def load_rank2_reference(
    sample: dict[str, np.ndarray], demo: bool
) -> tuple[dict, dict]:
    """Load the exact complete-basis Stage 3d rank-2 held-out product."""
    prefix = "demo_" if demo else ""
    prediction_path = RANK2_OUTPUT_DIRECTORY / f"{prefix}predictions.npz"
    result_path = RANK2_OUTPUT_DIRECTORY / f"{prefix}results.json"
    if not prediction_path.exists() or not result_path.exists():
        raise RuntimeError("matching complete-rank-2 Stage 3d products are required")
    saved = np.load(prediction_path)
    if not np.array_equal(saved["indices"], sample["indices"]):
        raise RuntimeError("Stage 3e and complete-rank-2 galaxy indices do not match")
    relation = {
        name: saved[f"rank2_{name}"]
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
        control: saved[f"rank2_{control}_mean_log"] for control in stage3.CONTROL_NAMES
    }
    result = json.loads(result_path.read_text())
    relation["metadata"] = result["fold_metadata"]["2"]
    return relation, result


def support_summary(metadata: list[dict]) -> dict:
    supports = np.asarray([fold["selected_support"] for fold in metadata], dtype=bool)
    counts = np.asarray(
        [fold["effective_coefficient_count"] for fold in metadata], dtype=int
    )
    return {
        "selected_support_codes": [support_code(support) for support in supports],
        "selected_effective_coefficient_counts": counts,
        "nonlinear_term_selection_frequency": np.mean(supports, axis=0),
        "all_folds_at_most_24_effective_coefficients": bool(
            np.all(counts <= MAXIMUM_EFFECTIVE_COUNT)
        ),
    }


def compare_with_rank2(
    sample: dict[str, np.ndarray],
    relation: dict,
    summary: dict,
    reference: dict,
    reference_result: dict,
) -> dict:
    """Apply paired observable gates against saved complete rank 2."""
    truth_log = sample["truth_log"]
    candidate_metrics = stage3.per_galaxy_metrics(
        truth_log, relation["mean_log"], relation["draw_log"]
    )
    reference_metrics = stage3.per_galaxy_metrics(
        truth_log, reference["mean_log"], reference["draw_log"]
    )
    paired = {
        name: stage3.bootstrap_median_interval(
            candidate_metrics[name] - reference_metrics[name], SEED + index
        )
        for index, name in enumerate(candidate_metrics)
    }
    mass_bin = stage3.worst_mass_bin_shape_bootstrap(
        truth_log,
        reference["mean_log"],
        relation["mean_log"],
        sample["features"][:, 0],
    )
    paired["worst_mass_bin_shape_residual_percent"] = mass_bin
    candidate_point = stage3c.point_metric_summary(truth_log, relation["mean_log"])
    reference_point = stage3c.point_metric_summary(truth_log, reference["mean_log"])
    candidate_radial = stage3c.radial_subset_metrics(truth_log, relation["mean_log"])
    reference_radial = stage3c.radial_subset_metrics(truth_log, reference["mean_log"])
    outside = np.any(
        (relation["mean_parameters"] < stage3.stage_1.LOWER)
        | (relation["mean_parameters"] > stage3.stage_1.UPPER),
        axis=1,
    )
    complexity = support_summary(relation["metadata"])
    reference_summary = reference_result["rank_summaries"]["2"]
    gates = {
        "every_fold_at_most_24_effective_coefficients": complexity[
            "all_folds_at_most_24_effective_coefficients"
        ],
        "profile_crps_within_one_percent": summary["draw"]["profile_crps_dex"]
        <= 1.01 * reference_summary["draw"]["profile_crps_dex"],
        "full_cog_rms_within_one_percent": candidate_point["full_cog_rms_dex"]
        <= 1.01 * reference_point["full_cog_rms_dex"],
        "full_density_rms_increase_at_most_0p001_dex": candidate_point[
            "full_density_rms_dex"
        ]
        <= reference_point["full_density_rms_dex"] + 0.001,
        "five_thirty_density_rms_increase_at_most_0p001_dex": candidate_point[
            "five_thirty_density_rms_dex"
        ]
        <= reference_point["five_thirty_density_rms_dex"] + 0.001,
        "five_thirty_cog_not_resolved_worse": paired["five_thirty_cog_rms_dex"][0]
        <= 0.0,
        "R50_not_resolved_worse": paired["absolute_R50_error"][0] <= 0.0,
        "R80_not_resolved_worse": paired["absolute_R80_error"][0] <= 0.0,
        "R90_not_resolved_worse": paired["absolute_R90_error"][0] <= 0.0,
        "mass_bin_shape_increase_below_0p5_percentage_point": mass_bin[2] < 0.5,
        "no_mean_coordinate_outside_profile_bounds": not np.any(outside),
        "even_radius_subset_within_one_percent": candidate_radial[
            "even_radius_pinned_cog_rms_dex"
        ]
        <= 1.01 * reference_radial["even_radius_pinned_cog_rms_dex"],
        "odd_radius_subset_within_one_percent": candidate_radial[
            "odd_radius_pinned_cog_rms_dex"
        ]
        <= 1.01 * reference_radial["odd_radius_pinned_cog_rms_dex"],
        "coverage_68_not_worse_by_more_than_0p03": abs(
            summary["draw"]["profile_coverage_68"] - 0.68
        )
        <= abs(reference_summary["draw"]["profile_coverage_68"] - 0.68) + 0.03,
        "complete_relation_beats_all_null_controls": all(
            summary["controls"]["complete"]["median_profile_rms_dex"]
            < summary["controls"][name]["median_profile_rms_dex"]
            for name in (
                "final_mass_only",
                "mah_shape_shuffled",
                "concentration_shuffled",
            )
        ),
    }
    return {
        "candidate_point_metrics": candidate_point,
        "complete_rank2_point_metrics": reference_point,
        "paired_sparse_minus_complete_rank2_bootstrap_16_50_84": paired,
        "candidate_radial_subset_metrics": candidate_radial,
        "complete_rank2_radial_subset_metrics": reference_radial,
        "outside_bound_fraction": float(np.mean(outside)),
        "support_summary": complexity,
        "numeric_gates": gates,
        "all_adoption_gates_pass": bool(all(gates.values())),
    }


def support_frontier_figure(relation: dict, prefix: str) -> None:
    """Show the nested accuracy-complexity frontier and selected term frequency."""
    figure, axes = plt.subplots(1, 3, figsize=(15.4, 4.5))
    for fold in relation["metadata"]:
        records = fold["selection_records"]
        reference = next(
            record for record in records if record["nonlinear_term_count"] == 7
        )
        counts = np.asarray(
            [record["effective_coefficient_count"] for record in records]
        )
        delta_cog = 1000.0 * np.asarray(
            [
                record["full_cog_rms_dex"] - reference["full_cog_rms_dex"]
                for record in records
            ]
        )
        delta_density = 1000.0 * np.asarray(
            [
                record["five_thirty_density_rms_dex"]
                - reference["five_thirty_density_rms_dex"]
                for record in records
            ]
        )
        axes[0].scatter(counts, delta_cog, s=7, color="0.7", alpha=0.20)
        axes[1].scatter(counts, delta_density, s=7, color="0.7", alpha=0.20)
        selected = next(record for record in records if record["selected"])
        axes[0].scatter(
            selected["effective_coefficient_count"],
            1000.0 * (selected["full_cog_rms_dex"] - reference["full_cog_rms_dex"]),
            s=45,
            color=OKABE_ITO[1],
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )
        axes[1].scatter(
            selected["effective_coefficient_count"],
            1000.0
            * (
                selected["five_thirty_density_rms_dex"]
                - reference["five_thirty_density_rms_dex"]
            ),
            s=45,
            color=OKABE_ITO[1],
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )
    for axis, ylabel, title in (
        (
            axes[0],
            r"inner $\Delta$ full-CoG RMS [$10^{-3}$ dex]",
            "Cumulative-profile envelope",
        ),
        (
            axes[1],
            r"inner $\Delta$ 5--30 kpc density RMS [$10^{-3}$ dex]",
            "Local-structure envelope",
        ),
    ):
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.axvline(MAXIMUM_EFFECTIVE_COUNT, color=OKABE_ITO[3], linestyle="--")
        axis.set(xlabel="effective mean coefficients", ylabel=ylabel, title=title)
    supports = np.asarray(
        [fold["selected_support"] for fold in relation["metadata"]], dtype=float
    )
    image = axes[2].imshow(
        supports,
        cmap="cividis",
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
        interpolation="nearest",
    )
    axes[2].set_xticks(
        range(len(stage3c.NONLINEAR_TERM_LABELS)),
        stage3c.NONLINEAR_TERM_LABELS,
        rotation=45,
        ha="right",
        fontsize=8,
    )
    axes[2].set_yticks(range(5), [f"outer fold {fold}" for fold in range(5)])
    axes[2].set_title("Whole nonlinear terms selected")
    figure.colorbar(image, ax=axes[2], ticks=(0, 1), label="excluded / retained")
    figure.suptitle("Nested sparse-rank-2 selection; blue points are chosen supports")
    figure.tight_layout()
    save_fig(figure, FIGURE_DIRECTORY / f"{prefix}support_frontier")
    plt.close(figure)


def average_cog_by_halo_mass_figure(
    sample: dict[str, np.ndarray],
    relation: dict,
    reference: dict,
    stage3_baseline: dict,
    prefix: str,
) -> None:
    """Compare average held-out CoGs in halo-mass bins."""
    truth = sample["truth_log"]
    halo_mass = sample["features"][:, 0]
    bins = np.array_split(np.argsort(halo_mass), 3)
    models = (
        ("sparse rank 2", relation["mean_log"], OKABE_ITO[1], "-"),
        ("complete rank 2", reference["mean_log"], "0.50", "--"),
        ("unrestricted Stage 3", stage3_baseline["mean_log"], OKABE_ITO[2], ":"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(14.8, 7.4), sharex=True)
    for column, rows in enumerate(bins):
        truth_median = np.median(truth[rows], axis=0)
        axes[0, column].plot(
            stage3.stage_1.RADII,
            truth_median,
            color="black",
            linewidth=2.2,
            label="TNG",
        )
        for label, values, color, linestyle in models:
            model_median = np.median(values[rows], axis=0)
            axes[0, column].plot(
                stage3.stage_1.RADII,
                model_median,
                color=color,
                linestyle=linestyle,
                label=label,
            )
            residual = 100.0 * (
                10.0
                ** (
                    (model_median - truth_median)
                    - (model_median[-1] - truth_median[-1])
                )
                - 1.0
            )
            axes[1, column].plot(
                stage3.stage_1.RADII,
                residual,
                color=color,
                linestyle=linestyle,
            )
        axes[0, column].set_xscale("log")
        axes[1, column].set_xscale("log")
        axes[1, column].axhline(0.0, color="black", linewidth=0.8)
        axes[0, column].set_title(
            rf"$\log M_{{\rm peak}}={np.min(halo_mass[rows]):.2f}$--"
            rf"${np.max(halo_mass[rows]):.2f}$  ($n={len(rows)}$)"
        )
        axes[1, column].set_xlabel("projected radius [kpc]")
    axes[0, 0].set_ylabel(r"median $\log_{10}M_\star(<R)$ [$M_\odot$]")
    axes[1, 0].set_ylabel("mass-pinned model minus TNG [%]")
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle("Held-out average CoGs in halo-mass bins")
    figure.tight_layout()
    save_fig(figure, FIGURE_DIRECTORY / f"{prefix}average_cog_by_halo_mass")
    plt.close(figure)


def stellar_mass_planes_figure(
    sample: dict[str, np.ndarray], relation: dict, reference: dict, prefix: str
) -> None:
    """Directly compare population planes for sparse and complete rank 2."""
    truth_cog = 10.0 ** sample["truth_log"][:, None, :]
    sparse_cog = 10.0 ** relation["mean_log"][:, None, :]
    reference_cog = 10.0 ** reference["mean_log"][:, None, :]
    truth, sparse, _, _ = stage3.qa.measure_all(
        sparse_cog, truth_cog, stage3.stage_1.RADII
    )
    _, complete, _, _ = stage3.qa.measure_all(
        reference_cog, truth_cog, stage3.stage_1.RADII
    )
    labels = {
        "kpc:M(<30)": r"$\log M_\star(<30\,{\rm kpc})$",
        "kpc:M(30-50)": r"$\log M_\star(30\!-\!50\,{\rm kpc})$",
        "kpc:M(50-100)": r"$\log M_\star(50\!-\!100\,{\rm kpc})$",
        "Re:M(<2Re)": r"$\log M_\star(<2R_{\rm e})$",
        "Re:M(2-4Re)": r"$\log M_\star(2\!-\!4R_{\rm e})$",
    }
    figure, axes = plt.subplots(1, 3, figsize=(12.8, 4.1))
    for axis, (x_name, y_name) in zip(axes, stage3.qa.PLANES):
        values = {}
        for name, source in (
            ("TNG", truth),
            ("complete rank 2", complete),
            ("sparse rank 2", sparse),
        ):
            values[name] = (
                np.log10(np.clip(source[x_name][:, 0], 1.0, None)),
                np.log10(np.clip(source[y_name][:, 0], 1.0, None)),
            )
        axis.scatter(
            *values["TNG"],
            s=8,
            color="0.65",
            alpha=0.35,
            edgecolors="none",
            label="TNG",
        )
        axis.scatter(
            *values["complete rank 2"],
            s=10,
            facecolors="none",
            edgecolors="0.35",
            alpha=0.35,
            linewidth=0.45,
            label="complete rank 2",
        )
        axis.scatter(
            *values["sparse rank 2"],
            s=10,
            facecolors="none",
            edgecolors=OKABE_ITO[1],
            alpha=0.45,
            linewidth=0.55,
            label="sparse rank 2",
        )
        statistics = {
            name: stage3.qa.plane_stats(*coordinate)
            for name, coordinate in values.items()
        }
        axis.set_title(
            "scatter TNG / complete / sparse\n"
            f"{statistics['TNG']['scatter']:.3f} / "
            f"{statistics['complete rank 2']['scatter']:.3f} / "
            f"{statistics['sparse rank 2']['scatter']:.3f}",
            fontsize=9,
        )
        axis.set_xlabel(labels[x_name])
        axis.set_ylabel(labels[y_name])
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle("Held-out stellar aperture and annular mass planes")
    figure.tight_layout()
    save_fig(figure, FIGURE_DIRECTORY / f"{prefix}stellar_mass_planes")
    plt.close(figure)


def cases_figure(
    sample: dict[str, np.ndarray], relation: dict, reference: dict, prefix: str
) -> None:
    """Show individual galaxies where sparse rank 2 helps, matches, or hurts."""
    truth = sample["truth_log"]
    sparse = relation["mean_log"]
    complete = reference["mean_log"]
    change = np.sqrt(np.mean((sparse - truth) ** 2, axis=1)) - np.sqrt(
        np.mean((complete - truth) ** 2, axis=1)
    )
    ordered = np.argsort(change)
    rows = (ordered[0], ordered[len(ordered) // 2], ordered[-1])
    titles = ("largest improvement", "typical change", "largest degradation")
    figure, axes = plt.subplots(2, 3, figsize=(14.2, 7.2), sharex=True)
    for column, (row, title) in enumerate(zip(rows, titles)):
        axes[0, column].plot(
            stage3.stage_1.RADII, truth[row], color="black", label="TNG"
        )
        axes[0, column].plot(
            stage3.stage_1.RADII,
            complete[row],
            color="0.5",
            linestyle="--",
            label="complete rank 2",
        )
        axes[0, column].plot(
            stage3.stage_1.RADII, sparse[row], color=OKABE_ITO[1], label="sparse rank 2"
        )
        axes[0, column].set_title(f"{title}\n" + rf"$\Delta$RMS={change[row]:+.3f} dex")
        truth_pinned = truth[row] - truth[row, -1]
        axes[1, column].axhline(0.0, color="black", linewidth=0.8)
        axes[1, column].plot(
            stage3.stage_1.RADII,
            complete[row] - complete[row, -1] - truth_pinned,
            color="0.5",
            linestyle="--",
        )
        axes[1, column].plot(
            stage3.stage_1.RADII,
            sparse[row] - sparse[row, -1] - truth_pinned,
            color=OKABE_ITO[1],
        )
        for axis in axes[:, column]:
            axis.set_xscale("log")
        axes[1, column].set_xlabel("projected radius [kpc]")
    axes[0, 0].set_ylabel(r"$\log_{10}M_\star(<R)$ [$M_\odot$]")
    axes[1, 0].set_ylabel("mass-pinned CoG residual [dex]")
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle("Held-out galaxies: sparse versus complete rank 2")
    figure.tight_layout()
    save_fig(figure, FIGURE_DIRECTORY / f"{prefix}individual_cases")
    plt.close(figure)


def generate_standard_qa(
    sample: dict[str, np.ndarray], relation: dict, prefix: str
) -> dict:
    """Generate the complete standard single-epoch QA battery."""
    truth_log = sample["truth_log"]
    mean_log = relation["mean_log"]
    draw_log = relation["draw_log"]
    halo_mass = sample["features"][:, 0]
    model_name = f"exp56_stage3e_{prefix}sparse_rank2"
    figure_directory = FIGURE_DIRECTORY / f"{prefix}sparse_rank2_standard_qa"
    figure_directory.mkdir(parents=True, exist_ok=True)
    qa_result = stage3.qa.evaluate(
        10.0 ** mean_log[:, None, :],
        10.0 ** truth_log[:, None, :],
        stage3.stage_1.RADII,
        [0.4],
        name=model_name,
        figdir=figure_directory,
        bin_by=halo_mass,
        bin_label=r"DiffMAH $\log M_{\rm peak}$",
        draw_cogs=10.0 ** draw_log[:, :, None, :],
    )
    model_label = "selected sparse rank 2"
    stage3.density_by_halo_mass_figure(
        mean_log,
        truth_log,
        halo_mass,
        figdir=figure_directory,
        model_name=model_name,
        model_label=model_label,
        experiment_label="exp56 Stage 3e",
    )
    stage3.single_epoch_planes_figure(
        qa_result,
        figdir=figure_directory,
        model_name=model_name,
        model_label=model_label,
        experiment_label="exp56 Stage 3e",
    )
    stage3.single_epoch_size_figure(
        mean_log,
        truth_log,
        qa_result,
        figdir=figure_directory,
        model_name=model_name,
        model_label=model_label,
        experiment_label="exp56 Stage 3e",
    )
    return stage3.standard_summary(qa_result)


def save_predictions(
    sample: dict[str, np.ndarray], relation: dict, prefix: str
) -> None:
    arrays = {
        "indices": sample["indices"],
        "features": sample["features"],
        "truth_log": sample["truth_log"],
    }
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
        arrays[name] = relation[name]
    for control in stage3.CONTROL_NAMES:
        arrays[f"{control}_mean_log"] = relation["control_predictions"][control]
    np.savez_compressed(OUTPUT_DIRECTORY / f"{prefix}predictions.npz", **arrays)


def run_experiment(n_max: int, n_draw: int, n_population_draw: int) -> dict:
    started = time.perf_counter()
    demo = bool(n_max)
    prefix = "demo_" if demo else ""
    checks = run_checks()
    if not checks["passed"]:
        raise RuntimeError(f"Stage 3e numerical checks failed: {checks}")
    sample = stage3.stage_1.load_sample(n_max)
    if demo and len(sample["indices"]) != N_DEMO_GALAXY:
        raise RuntimeError("Stage 3e demo did not load exactly 90 galaxies")
    relation = cross_validated_relation(sample, n_draw, demo)
    summary = stage3.summarize_model(sample, relation, n_population_draw)
    reference, reference_result = load_rank2_reference(sample, demo)
    stage3_baseline, _ = stage3c.load_baseline(sample, demo)
    comparison = compare_with_rank2(
        sample, relation, summary, reference, reference_result
    )

    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    support_frontier_figure(relation, prefix)
    average_cog_by_halo_mass_figure(
        sample, relation, reference, stage3_baseline, prefix
    )
    stellar_mass_planes_figure(sample, relation, reference, prefix)
    cases_figure(sample, relation, reference, prefix)
    standard = generate_standard_qa(sample, relation, prefix)
    save_predictions(sample, relation, prefix)
    elapsed_seconds = time.perf_counter() - started
    result = {
        "status": "stage3e_validation" if demo else "stage3e_full",
        "n_galaxy": len(sample["indices"]),
        "n_draw": n_draw,
        "n_population_draw": n_population_draw,
        "numerical_checks": checks,
        "sparse_rank2_summary": summary,
        "sparse_minus_complete_rank2": comparison,
        "standard_qa": standard,
        "fold_metadata": relation["metadata"],
        "elapsed_seconds": elapsed_seconds,
    }
    if demo:
        result["complete_path_passed"] = bool(
            len(sample["indices"]) == N_DEMO_GALAXY
            and elapsed_seconds < DEMO_TIME_LIMIT_SECONDS
            and np.isfinite(relation["mean_log"]).all()
            and np.isfinite(relation["draw_log"]).all()
            and len(relation["metadata"]) == 5
        )
    (OUTPUT_DIRECTORY / f"{prefix}results.json").write_text(
        json.dumps(stage3.json_safe(result), indent=2)
    )
    stage3.write_manifest(
        OUTPUT_DIRECTORY / f"{prefix}manifest",
        params={
            "script": Path(__file__).name,
            "status": result["status"],
            "n_galaxy": len(sample["indices"]),
            "n_draw": n_draw,
            "n_population_draw": n_population_draw,
            "rank": RANK,
            "compact_index": 1.0,
            "gamma": 1.4,
            "candidate_support_count": len(candidate_supports()),
            "mandatory_linear_term_count": 5,
            "optional_nonlinear_term_count": len(SPARSE_TERMS),
            "maximum_effective_count": MAXIMUM_EFFECTIVE_COUNT,
            "inner_fold_count": 4,
            "outer_fold_count": 5,
        },
    )
    return result


def demo() -> None:
    result = run_experiment(N_DEMO_GALAXY, N_DEMO_DRAW, 1)
    if not result["complete_path_passed"]:
        raise AssertionError(
            "Exp56 Stage 3e complete demo failed its sub-minute gate: "
            f"{result['elapsed_seconds']:.2f} s"
        )
    print(
        f"Exp56 Stage 3e complete 90-galaxy path passed in "
        f"{result['elapsed_seconds']:.2f} s",
        flush=True,
    )


def full_run() -> None:
    demo_path = OUTPUT_DIRECTORY / "demo_results.json"
    if not demo_path.exists():
        raise RuntimeError("run the Stage 3e demo before the full calculation")
    demo_result = json.loads(demo_path.read_text())
    if not demo_result.get("complete_path_passed", False):
        raise RuntimeError("the saved Stage 3e complete-path validation did not pass")
    result = run_experiment(0, N_FULL_DRAW, N_POPULATION_DRAW)
    print(
        f"Exp56 Stage 3e full relation complete: {result['n_galaxy']} galaxies "
        f"in {result['elapsed_seconds']:.1f} s",
        flush=True,
    )


def regenerate_figures() -> None:
    prediction_path = OUTPUT_DIRECTORY / "predictions.npz"
    result_path = OUTPUT_DIRECTORY / "results.json"
    if not prediction_path.exists() or not result_path.exists():
        raise RuntimeError("run the full Stage 3e calculation before figures")
    saved = np.load(prediction_path)
    sample = stage3.stage_1.load_sample(0)
    if not np.array_equal(saved["indices"], sample["indices"]):
        raise RuntimeError("saved Stage 3e indices do not match the current sample")
    relation = {
        name: saved[name]
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
        control: saved[f"{control}_mean_log"] for control in stage3.CONTROL_NAMES
    }
    result = json.loads(result_path.read_text())
    relation["metadata"] = result["fold_metadata"]
    reference, _ = load_rank2_reference(sample, demo=False)
    stage3_baseline, _ = stage3c.load_baseline(sample, demo=False)
    support_frontier_figure(relation, "")
    average_cog_by_halo_mass_figure(sample, relation, reference, stage3_baseline, "")
    stellar_mass_planes_figure(sample, relation, reference, "")
    cases_figure(sample, relation, reference, "")
    generate_standard_qa(sample, relation, "")


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {
        "checks",
        "demo",
        "run",
        "figures",
    }:
        raise SystemExit("usage: stage3e_sparse_rank2.py [checks|demo|run|figures]")
    if sys.argv[1] == "checks":
        values = run_checks()
        if not values["passed"]:
            raise RuntimeError("Stage 3e numerical checks failed")
    elif sys.argv[1] == "demo":
        demo()
    elif sys.argv[1] == "run":
        full_run()
    else:
        regenerate_figures()


if __name__ == "__main__":
    main()
