"""Exp56 Stage 3d: shared low-rank halo responses for analytic CoGs.

Commands
--------
checks
    Run coefficient-count, rank-4 closure, and synthetic recovery checks.
demo
    Run the complete predeclared path on 90 mass-stratified galaxies.
run
    Run the full five-fold held-out calculation after the demo gate passes.
figures
    Rebuild Stage 3d figures from the saved full predictions and results.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import subspace_angles

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP55 = ROOT / "experiments" / "exp55_analytic_cog"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP55))
sys.path.insert(0, str(HERE))

import stage3_halo_cog as stage3  # noqa: E402
import stage3c_sparse_relation as stage3c  # noqa: E402
from hongshao.plotting import OKABE_ITO, save_fig, set_style  # noqa: E402
from hongshao.profile_emulator import density_from_cog  # noqa: E402
from refine_halo_map import SPARSE_TERMS, design_matrix, fit_conditional_model  # noqa: E402

set_style()
matplotlib.rcParams["text.usetex"] = False

RANKS = (1, 2, 3, 4)
PRIMARY_RANK = 2
N_DEMO_GALAXY = 90
N_DEMO_DRAW = 8
N_FULL_DRAW = 32
N_POPULATION_DRAW = 4
DEMO_TIME_LIMIT_SECONDS = 60.0
MAXIMUM_SUBSPACE_ANGLE_DEGREES = 20.0
SEED = 5680

HALO_BASIS_LABELS = (
    r"$\log M_{\rm peak}$",
    r"$\log t_c$",
    "early",
    "late",
    r"$c_{200c}$",
    *stage3c.NONLINEAR_TERM_LABELS,
)
OUTPUT_LABELS = (
    r"$\log M_{\star,148}$",
    r"$\mathrm{logit}\,f_c$",
    r"$\log R_c$",
    r"$\log(R_e/R_c-1)$",
)

FIGURE_DIRECTORY = HERE / "figures" / "stage3d"
OUTPUT_DIRECTORY = HERE / "outputs" / "stage3d"


def effective_coefficient_count(rank: int, n_predictor: int, n_output: int) -> int:
    """Identifiable intercept-plus-rank-r matrix dimension."""
    rank = min(rank, n_predictor, n_output)
    return n_output + rank * (n_predictor + n_output - rank)


@dataclass(frozen=True)
class ReducedRankModel:
    """Training-standardized reduced-rank multivariate relation."""

    feature_mean: np.ndarray
    feature_standard_deviation: np.ndarray
    basis_mean: np.ndarray
    target_mean: np.ndarray
    target_standard_deviation: np.ndarray
    standardized_coefficients: np.ndarray
    halo_loadings: np.ndarray
    standardized_output_loadings: np.ndarray
    singular_values: np.ndarray
    score_standard_deviation: np.ndarray
    terms: tuple[tuple, ...]
    requested_rank: int
    design_condition_number: float

    def standardized(self, features: np.ndarray) -> np.ndarray:
        return (np.asarray(features, float) - self.feature_mean) / (
            self.feature_standard_deviation
        )

    def centered_basis(self, features: np.ndarray) -> np.ndarray:
        matrix = design_matrix(self.standardized(features), self.terms)[:, 1:]
        return matrix - self.basis_mean

    def predict_mean(self, features: np.ndarray) -> np.ndarray:
        standardized_prediction = (
            self.centered_basis(features) @ self.standardized_coefficients
        )
        return self.target_mean + (
            standardized_prediction * self.target_standard_deviation
        )

    def raw_output_subspace(self) -> np.ndarray:
        scaled = (
            self.target_standard_deviation[:, None] * self.standardized_output_loadings
        )
        return np.linalg.qr(scaled)[0][:, : self.standardized_output_loadings.shape[1]]


def fit_reduced_rank_model(
    features: np.ndarray,
    targets: np.ndarray,
    terms: tuple[tuple, ...],
    rank: int,
) -> ReducedRankModel:
    """Fit the least-squares response after projecting to rank ``rank``."""
    features = np.asarray(features, float)
    targets = np.asarray(targets, float)
    feature_mean = np.mean(features, axis=0)
    feature_standard_deviation = np.std(features, axis=0)
    feature_standard_deviation = np.where(
        feature_standard_deviation > 1e-12,
        feature_standard_deviation,
        1.0,
    )
    standardized_features = (features - feature_mean) / feature_standard_deviation
    basis = design_matrix(standardized_features, terms)[:, 1:]
    basis_mean = np.mean(basis, axis=0)
    centered_basis = basis - basis_mean
    target_mean = np.mean(targets, axis=0)
    target_standard_deviation = np.std(targets, axis=0)
    target_standard_deviation = np.where(
        target_standard_deviation > 1e-12,
        target_standard_deviation,
        1.0,
    )
    standardized_targets = (targets - target_mean) / target_standard_deviation
    ordinary_coefficients, *_ = np.linalg.lstsq(
        centered_basis, standardized_targets, rcond=None
    )
    fitted = centered_basis @ ordinary_coefficients
    _, singular_values, right_vectors = np.linalg.svd(fitted, full_matrices=False)
    fitted_rank = min(rank, ordinary_coefficients.shape[0], targets.shape[1])
    output_loadings = right_vectors[:fitted_rank].T
    for axis in range(fitted_rank):
        pivot = np.argmax(np.abs(output_loadings[:, axis]))
        if output_loadings[pivot, axis] < 0.0:
            output_loadings[:, axis] *= -1.0
    halo_loadings = ordinary_coefficients @ output_loadings
    standardized_coefficients = halo_loadings @ output_loadings.T
    scores = centered_basis @ halo_loadings
    singular = np.linalg.svd(centered_basis, compute_uv=False)
    condition = float(singular[0] / max(singular[-1], 1e-15))
    return ReducedRankModel(
        feature_mean=feature_mean,
        feature_standard_deviation=feature_standard_deviation,
        basis_mean=basis_mean,
        target_mean=target_mean,
        target_standard_deviation=target_standard_deviation,
        standardized_coefficients=standardized_coefficients,
        halo_loadings=halo_loadings,
        standardized_output_loadings=output_loadings,
        singular_values=singular_values,
        score_standard_deviation=np.std(scores, axis=0),
        terms=terms,
        requested_rank=rank,
        design_condition_number=condition,
    )


def model_metadata(model: ReducedRankModel) -> dict:
    variance = model.singular_values**2
    return {
        "requested_rank": model.requested_rank,
        "effective_coefficient_count": effective_coefficient_count(
            model.requested_rank,
            model.standardized_coefficients.shape[0],
            model.standardized_coefficients.shape[1],
        ),
        "singular_values": model.singular_values,
        "fitted_response_variance_fraction": variance / np.sum(variance),
        "halo_loadings": model.halo_loadings,
        "standardized_output_loadings": model.standardized_output_loadings,
        "raw_output_subspace": model.raw_output_subspace(),
        "target_mean": model.target_mean,
        "target_standard_deviation": model.target_standard_deviation,
        "score_standard_deviation": model.score_standard_deviation,
        "design_condition_number": model.design_condition_number,
    }


def run_checks() -> dict:
    """Numerical closure and exact low-rank synthetic recovery."""
    rng = np.random.default_rng(SEED)
    features = rng.normal(size=(600, 5))
    targets = rng.normal(size=(600, 4))
    reference = fit_conditional_model(features, targets, SPARSE_TERMS)
    full = fit_reduced_rank_model(features, targets, SPARSE_TERMS, rank=4)
    closure = float(
        np.max(np.abs(full.predict_mean(features) - reference.predict_mean(features)))
    )

    design, _, _ = stage3c.standardized_design(features)
    centered_basis = design[:, 1:] - np.mean(design[:, 1:], axis=0)
    halo_loadings = rng.normal(size=(centered_basis.shape[1], 2))
    output_loadings = rng.normal(size=(2, 4))
    synthetic_targets = rng.normal(scale=0.2, size=4) + centered_basis @ (
        halo_loadings @ output_loadings
    )
    recovered = fit_reduced_rank_model(
        features, synthetic_targets, SPARSE_TERMS, rank=2
    )
    synthetic_error = float(
        np.max(np.abs(recovered.predict_mean(features) - synthetic_targets))
    )
    counts = [effective_coefficient_count(rank, 12, 4) for rank in RANKS]
    values = {
        "effective_coefficient_counts_rank_1_to_4": counts,
        "rank4_maximum_coordinate_prediction_difference": closure,
        "synthetic_rank2_maximum_coordinate_prediction_error": synthetic_error,
        "synthetic_recovered_matrix_rank": int(
            np.linalg.matrix_rank(recovered.standardized_coefficients, tol=1e-10)
        ),
        "passed": bool(
            counts == [19, 32, 43, 52]
            and closure < 1e-10
            and synthetic_error < 1e-10
            and np.linalg.matrix_rank(recovered.standardized_coefficients, tol=1e-10)
            == 2
        ),
    }
    print(json.dumps(stage3.json_safe(values), indent=2))
    return values


def calibrated_draws(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    train_truth_log: np.ndarray,
    test_features: np.ndarray,
    model: ReducedRankModel,
    rank: int,
    n_draw: int,
    seed: int,
) -> tuple[np.ndarray, dict]:
    """Calibrate complete-coordinate residual draws inside the outer fold."""
    scale_grid = np.asarray([1.00, 1.08, 1.16, 1.24, 1.32])
    folds = stage3c.deterministic_inner_folds(len(train_features), seed)
    all_rows = np.arange(len(train_features))
    inner_mean = np.empty_like(train_targets)
    inner_draw = np.empty((n_draw, *train_targets.shape))
    for inner_number, validation in enumerate(folds):
        training = np.setdiff1d(all_rows, validation)
        inner_model = fit_reduced_rank_model(
            train_features[training],
            train_targets[training],
            SPARSE_TERMS,
            rank,
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
    sample: dict[str, np.ndarray], rank: int, n_draw: int, demo: bool
) -> dict:
    """Five-fold low-rank mean, controls, and calibrated stochastic draws."""
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
        model = fit_reduced_rank_model(
            features[train], targets[train], SPARSE_TERMS, rank
        )
        mean_parameters[test] = model.predict_mean(features[test])
        draw_parameters[:, test], calibration = calibrated_draws(
            features[train],
            targets[train],
            truth_log[train],
            features[test],
            model,
            rank,
            n_draw,
            SEED + 10_000 * rank + 1000 * fold_number,
        )
        for name, (feature_values, terms) in variants.items():
            if name == "complete":
                control_parameters = mean_parameters[test]
            else:
                control_model = fit_reduced_rank_model(
                    feature_values[train], targets[train], terms, rank
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
                "model": model_metadata(model),
                "calibration": calibration,
            }
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


def subspace_summary(metadata: list[dict], rank: int) -> dict:
    subspaces = [
        np.asarray(fold["model"]["raw_output_subspace"], float) for fold in metadata
    ]
    maximum_angles = []
    all_angles = []
    for first in range(len(subspaces)):
        for second in range(first + 1, len(subspaces)):
            degrees = np.degrees(subspace_angles(subspaces[first], subspaces[second]))
            all_angles.append(degrees)
            maximum_angles.append(float(np.max(degrees)))
    variance_fraction = np.asarray(
        [fold["model"]["fitted_response_variance_fraction"] for fold in metadata]
    )
    return {
        "rank": rank,
        "pairwise_principal_angles_degrees": all_angles,
        "median_maximum_pairwise_angle_degrees": float(np.median(maximum_angles)),
        "maximum_pairwise_angle_degrees": float(np.max(maximum_angles)),
        "median_fitted_response_variance_fraction": np.median(
            variance_fraction, axis=0
        ),
        "median_design_condition_number": float(
            np.median([fold["model"]["design_condition_number"] for fold in metadata])
        ),
    }


def compare_with_baseline(
    sample: dict[str, np.ndarray],
    relation: dict,
    summary: dict,
    baseline: dict,
    baseline_result: dict,
    rank: int,
) -> dict:
    """Paired held-out comparison with the unrestricted Stage 3 relation."""
    truth_log = sample["truth_log"]
    candidate_metrics = stage3.per_galaxy_metrics(
        truth_log, relation["mean_log"], relation["draw_log"]
    )
    baseline_metrics = stage3.per_galaxy_metrics(
        truth_log, baseline["mean_log"], baseline["draw_log"]
    )
    paired = {
        name: stage3.bootstrap_median_interval(
            candidate_metrics[name] - baseline_metrics[name],
            SEED + 100 * rank + metric_index,
        )
        for metric_index, name in enumerate(candidate_metrics)
    }
    mass_bin = stage3.worst_mass_bin_shape_bootstrap(
        truth_log,
        baseline["mean_log"],
        relation["mean_log"],
        sample["features"][:, 0],
    )
    paired["worst_mass_bin_shape_residual_percent"] = mass_bin
    candidate_point = stage3c.point_metric_summary(truth_log, relation["mean_log"])
    baseline_point = stage3c.point_metric_summary(truth_log, baseline["mean_log"])
    candidate_radial = stage3c.radial_subset_metrics(truth_log, relation["mean_log"])
    baseline_radial = stage3c.radial_subset_metrics(truth_log, baseline["mean_log"])
    outside = np.any(
        (relation["mean_parameters"] < stage3.stage_1.LOWER)
        | (relation["mean_parameters"] > stage3.stage_1.UPPER),
        axis=1,
    )
    structure = subspace_summary(relation["metadata"], rank)
    baseline_summary = baseline_result["representations"]["fixed_gamma1p4"]
    gates = {
        "effective_coefficient_count_at_most_32": effective_coefficient_count(
            rank, 12, 4
        )
        <= 32,
        "profile_crps_within_one_percent": summary["draw"]["profile_crps_dex"]
        <= 1.01 * baseline_summary["draw"]["profile_crps_dex"],
        "full_cog_rms_within_one_percent": candidate_point["full_cog_rms_dex"]
        <= 1.01 * baseline_point["full_cog_rms_dex"],
        "full_density_rms_increase_at_most_0p001_dex": candidate_point[
            "full_density_rms_dex"
        ]
        <= baseline_point["full_density_rms_dex"] + 0.001,
        "five_thirty_density_rms_increase_at_most_0p001_dex": candidate_point[
            "five_thirty_density_rms_dex"
        ]
        <= baseline_point["five_thirty_density_rms_dex"] + 0.001,
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
        <= 1.01 * baseline_radial["even_radius_pinned_cog_rms_dex"],
        "odd_radius_subset_within_one_percent": candidate_radial[
            "odd_radius_pinned_cog_rms_dex"
        ]
        <= 1.01 * baseline_radial["odd_radius_pinned_cog_rms_dex"],
        "subspace_maximum_angle_below_20_degrees": structure[
            "maximum_pairwise_angle_degrees"
        ]
        < MAXIMUM_SUBSPACE_ANGLE_DEGREES,
    }
    return {
        "rank": rank,
        "effective_coefficient_count": effective_coefficient_count(rank, 12, 4),
        "candidate_point_metrics": candidate_point,
        "baseline_point_metrics": baseline_point,
        "paired_rank_minus_stage3_bootstrap_16_50_84": paired,
        "candidate_radial_subset_metrics": candidate_radial,
        "baseline_radial_subset_metrics": baseline_radial,
        "outside_bound_fraction": float(np.mean(outside)),
        "subspace": structure,
        "numeric_gates": gates,
        "all_adoption_gates_pass": bool(all(gates.values())),
    }


def rank_comparison_figure(
    relations: dict[int, dict], comparisons: dict[int, dict], prefix: str
) -> None:
    """Show the held-out accuracy and variance retained by each rank."""
    figure, axes = plt.subplots(1, 3, figsize=(15.2, 4.4))
    counts = [effective_coefficient_count(rank, 12, 4) for rank in RANKS]
    full_cog = [
        comparisons[rank]["candidate_point_metrics"]["full_cog_rms_dex"]
        for rank in RANKS
    ]
    density = [
        comparisons[rank]["candidate_point_metrics"]["full_density_rms_dex"]
        for rank in RANKS
    ]
    axes[0].plot(counts, full_cog, "o-", color=OKABE_ITO[1], label="CoG")
    axes[0].plot(counts, density, "s--", color=OKABE_ITO[2], label="density")
    axes[0].axvline(32, color="0.4", linestyle=":", label="rank-2 target")
    axes[0].set(
        xlabel="effective mean coefficients",
        ylabel="median held-out RMS [dex]",
        title="Accuracy–complexity relation",
    )
    axes[0].legend(frameon=False)

    fractions = np.asarray(
        [
            comparisons[rank]["subspace"]["median_fitted_response_variance_fraction"]
            for rank in RANKS
        ]
    )
    cumulative = np.cumsum(fractions[0])
    axes[1].bar(np.arange(1, 5), fractions[0], color=OKABE_ITO[1], label="individual")
    axes[1].plot(
        np.arange(1, 5), cumulative, "o-", color=OKABE_ITO[3], label="cumulative"
    )
    axes[1].set(
        xlabel="shared response direction",
        ylabel="fitted-response variance fraction",
        title="Unrestricted Stage 3 response spectrum",
        xticks=np.arange(1, 5),
        ylim=(0.0, 1.05),
    )
    axes[1].legend(frameon=False)

    angles = [
        comparisons[rank]["subspace"]["maximum_pairwise_angle_degrees"]
        for rank in (1, 2, 3)
    ]
    axes[2].bar((1, 2, 3), angles, color=OKABE_ITO[2])
    axes[2].axhline(
        MAXIMUM_SUBSPACE_ANGLE_DEGREES,
        color=OKABE_ITO[3],
        linestyle="--",
        label="rank-2 stability limit",
    )
    axes[2].set(
        xlabel="retained rank",
        ylabel="largest pairwise principal angle [degree]",
        title="Fold-to-fold output-subspace stability",
        xticks=(1, 2, 3),
    )
    axes[2].legend(frameon=False)
    figure.tight_layout()
    save_fig(figure, FIGURE_DIRECTORY / f"{prefix}rank_comparison")
    plt.close(figure)


def average_cog_by_halo_mass_figure(
    sample: dict[str, np.ndarray], relations: dict[int, dict], prefix: str
) -> None:
    """Directly compare average CoGs and shape residuals in halo-mass bins."""
    truth = sample["truth_log"]
    halo_mass = sample["features"][:, 0]
    bins = np.array_split(np.argsort(halo_mass), 3)
    styles = {
        1: (OKABE_ITO[0], ":"),
        2: (OKABE_ITO[1], "-"),
        3: (OKABE_ITO[2], "-."),
        4: ("0.45", "--"),
    }
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
        for rank in RANKS:
            color, linestyle = styles[rank]
            model_median = np.median(relations[rank]["mean_log"][rows], axis=0)
            axes[0, column].plot(
                stage3.stage_1.RADII,
                model_median,
                color=color,
                linestyle=linestyle,
                label=f"rank {rank}",
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


def latent_direction_figure(relation: dict, prefix: str) -> None:
    """Visualize the halo weights and decoded profile response of rank 2."""
    metadata = relation["metadata"]
    halo = []
    output = []
    cog_response = []
    density_response = []
    density_radius = None
    for fold in metadata:
        model = fold["model"]
        halo_loading = np.asarray(model["halo_loadings"], float)
        halo_loading /= np.maximum(np.linalg.norm(halo_loading, axis=0), 1e-15)
        halo.append(halo_loading)
        raw_output = (
            np.asarray(model["target_standard_deviation"], float)[:, None]
            * np.asarray(model["standardized_output_loadings"], float)
            * np.asarray(model["score_standard_deviation"], float)[None]
        )
        output.append(raw_output)
        base = np.asarray(model["target_mean"], float)
        base_log = stage3.decode_profiles(base[None], np.asarray([1.4]))[0]
        base_density, density_radius = density_from_cog(
            base_log[None], stage3.stage_1.RADII
        )
        fold_cog = []
        fold_density = []
        for axis in range(PRIMARY_RANK):
            shifted_log = stage3.decode_profiles(
                (base + raw_output[:, axis])[None], np.asarray([1.4])
            )[0]
            shifted_density, _ = density_from_cog(
                shifted_log[None], stage3.stage_1.RADII
            )
            fold_cog.append(shifted_log - base_log)
            fold_density.append(shifted_density[0] - base_density[0])
        cog_response.append(fold_cog)
        density_response.append(fold_density)
    halo_mean = np.mean(halo, axis=0)
    output_mean = np.mean(output, axis=0)
    maximum = max(np.max(np.abs(halo_mean)), 1e-12)
    figure, axes = plt.subplots(2, 2, figsize=(13.6, 8.2))
    halo_image = axes[0, 0].imshow(
        halo_mean.T, cmap="PuOr", vmin=-maximum, vmax=maximum, aspect="auto"
    )
    axes[0, 0].set_xticks(
        range(len(HALO_BASIS_LABELS)),
        HALO_BASIS_LABELS,
        rotation=45,
        ha="right",
        fontsize=8,
    )
    axes[0, 0].set_yticks((0, 1), ("direction 1", "direction 2"))
    axes[0, 0].set_title("Normalized halo-basis weights")
    figure.colorbar(halo_image, ax=axes[0, 0], label="signed normalized weight")

    output_maximum = max(np.max(np.abs(output_mean)), 1e-12)
    output_image = axes[0, 1].imshow(
        output_mean.T,
        cmap="PuOr",
        vmin=-output_maximum,
        vmax=output_maximum,
        aspect="auto",
    )
    axes[0, 1].set_xticks(range(4), OUTPUT_LABELS, rotation=25, ha="right")
    axes[0, 1].set_yticks((0, 1), ("direction 1", "direction 2"))
    axes[0, 1].set_title(r"Analytic-coordinate change for $+1\sigma$ score")
    figure.colorbar(output_image, ax=axes[0, 1], label="coordinate change")

    colors = (OKABE_ITO[1], OKABE_ITO[2])
    for axis, color in enumerate(colors):
        cog = np.asarray(cog_response)[:, axis]
        density = np.asarray(density_response)[:, axis]
        axes[1, 0].plot(
            stage3.stage_1.RADII,
            np.mean(cog, axis=0),
            color=color,
            label=f"direction {axis + 1}",
        )
        axes[1, 0].fill_between(
            stage3.stage_1.RADII,
            np.min(cog, axis=0),
            np.max(cog, axis=0),
            color=color,
            alpha=0.15,
        )
        axes[1, 1].plot(
            density_radius,
            np.mean(density, axis=0),
            color=color,
            label=f"direction {axis + 1}",
        )
        axes[1, 1].fill_between(
            density_radius,
            np.min(density, axis=0),
            np.max(density, axis=0),
            color=color,
            alpha=0.15,
        )
    for axis in axes[1]:
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xscale("log")
        axis.set_xlabel("projected radius [kpc]")
        axis.legend(frameon=False)
    axes[1, 0].set_ylabel(r"$\Delta\log_{10}M_\star(<R)$ [dex]")
    axes[1, 0].set_title(r"Decoded CoG response to $+1\sigma$ score")
    axes[1, 1].set_ylabel(r"$\Delta\log_{10}\Sigma_\star$ [dex]")
    axes[1, 1].set_title(r"Decoded density response to $+1\sigma$ score")
    figure.tight_layout()
    save_fig(figure, FIGURE_DIRECTORY / f"{prefix}latent_directions")
    plt.close(figure)


def cases_figure(
    sample: dict[str, np.ndarray], relation: dict, baseline: dict, prefix: str
) -> None:
    """Show direct galaxies where rank 2 helps, changes little, or hurts."""
    truth = sample["truth_log"]
    low_rank = relation["mean_log"]
    reference = baseline["mean_log"]
    low_rank_rms = np.sqrt(np.mean((low_rank - truth) ** 2, axis=1))
    reference_rms = np.sqrt(np.mean((reference - truth) ** 2, axis=1))
    change = low_rank_rms - reference_rms
    ordered = np.argsort(change)
    rows = (ordered[0], ordered[len(ordered) // 2], ordered[-1])
    titles = ("largest improvement", "typical change", "largest degradation")
    figure, axes = plt.subplots(2, 3, figsize=(14.2, 7.2), sharex=True)
    for column, (row, title) in enumerate(zip(rows, titles)):
        axes[0, column].plot(
            stage3.stage_1.RADII, truth[row], color="black", label="TNG"
        )
        axes[0, column].plot(
            stage3.stage_1.RADII, reference[row], color="0.55", label="rank 4 / Stage 3"
        )
        axes[0, column].plot(
            stage3.stage_1.RADII, low_rank[row], color=OKABE_ITO[1], label="rank 2"
        )
        axes[0, column].set_title(f"{title}\n" + rf"$\Delta$RMS={change[row]:+.3f} dex")
        truth_pinned = truth[row] - truth[row, -1]
        axes[1, column].axhline(0.0, color="black", linewidth=0.8)
        axes[1, column].plot(
            stage3.stage_1.RADII,
            reference[row] - reference[row, -1] - truth_pinned,
            color="0.55",
        )
        axes[1, column].plot(
            stage3.stage_1.RADII,
            low_rank[row] - low_rank[row, -1] - truth_pinned,
            color=OKABE_ITO[1],
        )
        for axis in axes[:, column]:
            axis.set_xscale("log")
        axes[1, column].set_xlabel("projected radius [kpc]")
    axes[0, 0].set_ylabel(r"$\log_{10}M_\star(<R)$ [$M_\odot$]")
    axes[1, 0].set_ylabel("mass-pinned CoG residual [dex]")
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle("Held-out galaxies: two shared halo-response directions")
    figure.tight_layout()
    save_fig(figure, FIGURE_DIRECTORY / f"{prefix}rank2_individual_cases")
    plt.close(figure)


def generate_standard_qa(
    sample: dict[str, np.ndarray], relation: dict, prefix: str
) -> dict:
    """Complete standard QA, including halo-bin CoGs and stellar-mass planes."""
    truth_log = sample["truth_log"]
    mean_log = relation["mean_log"]
    draw_log = relation["draw_log"]
    halo_mass = sample["features"][:, 0]
    model_name = f"exp56_stage3d_{prefix}rank2"
    figure_directory = FIGURE_DIRECTORY / f"{prefix}rank2_standard_qa"
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
    model_label = "two shared halo-response directions"
    stage3.density_by_halo_mass_figure(
        mean_log,
        truth_log,
        halo_mass,
        figdir=figure_directory,
        model_name=model_name,
        model_label=model_label,
        experiment_label="exp56 Stage 3d",
    )
    stage3.single_epoch_planes_figure(
        qa_result,
        figdir=figure_directory,
        model_name=model_name,
        model_label=model_label,
        experiment_label="exp56 Stage 3d",
    )
    stage3.single_epoch_size_figure(
        mean_log,
        truth_log,
        qa_result,
        figdir=figure_directory,
        model_name=model_name,
        model_label=model_label,
        experiment_label="exp56 Stage 3d",
    )
    return stage3.standard_summary(qa_result)


def save_predictions(
    sample: dict[str, np.ndarray], relations: dict[int, dict], prefix: str
) -> None:
    arrays = {
        "indices": sample["indices"],
        "features": sample["features"],
        "truth_log": sample["truth_log"],
    }
    for rank, relation in relations.items():
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
            arrays[f"rank{rank}_{name}"] = relation[name]
        for control in stage3.CONTROL_NAMES:
            arrays[f"rank{rank}_{control}_mean_log"] = relation["control_predictions"][
                control
            ]
    np.savez_compressed(OUTPUT_DIRECTORY / f"{prefix}predictions.npz", **arrays)


def run_experiment(n_max: int, n_draw: int, n_population_draw: int) -> dict:
    started = time.perf_counter()
    demo = bool(n_max)
    prefix = "demo_" if demo else ""
    checks = run_checks()
    if not checks["passed"]:
        raise RuntimeError(f"Stage 3d numerical checks failed: {checks}")
    sample = stage3.stage_1.load_sample(n_max)
    if demo and len(sample["indices"]) != N_DEMO_GALAXY:
        raise RuntimeError("Stage 3d demo did not load exactly 90 galaxies")
    relations = {
        rank: cross_validated_relation(sample, rank, n_draw, demo) for rank in RANKS
    }
    summaries = {
        rank: stage3.summarize_model(sample, relation, n_population_draw)
        for rank, relation in relations.items()
    }
    baseline, baseline_result = stage3c.load_baseline(sample, demo)
    comparisons = {
        rank: compare_with_baseline(
            sample,
            relations[rank],
            summaries[rank],
            baseline,
            baseline_result,
            rank,
        )
        for rank in RANKS
    }
    rank4_profile_closure = float(
        np.max(np.abs(relations[4]["mean_log"] - baseline["mean_log"]))
    )

    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    rank_comparison_figure(relations, comparisons, prefix)
    average_cog_by_halo_mass_figure(sample, relations, prefix)
    latent_direction_figure(relations[PRIMARY_RANK], prefix)
    cases_figure(sample, relations[PRIMARY_RANK], baseline, prefix)
    standard = generate_standard_qa(sample, relations[PRIMARY_RANK], prefix)
    save_predictions(sample, relations, prefix)
    elapsed_seconds = time.perf_counter() - started
    result = {
        "status": "stage3d_validation" if demo else "stage3d_full",
        "n_galaxy": len(sample["indices"]),
        "n_draw": n_draw,
        "n_population_draw": n_population_draw,
        "numerical_and_synthetic_checks": checks,
        "rank4_maximum_saved_stage3_profile_difference_dex": rank4_profile_closure,
        "rank_summaries": summaries,
        "rank_minus_stage3_comparisons": comparisons,
        "rank2_standard_qa": standard,
        "fold_metadata": {str(rank): relations[rank]["metadata"] for rank in RANKS},
        "elapsed_seconds": elapsed_seconds,
    }
    if demo:
        result["complete_path_passed"] = bool(
            len(sample["indices"]) == N_DEMO_GALAXY
            and elapsed_seconds < DEMO_TIME_LIMIT_SECONDS
            and rank4_profile_closure < 1e-10
            and all(
                np.isfinite(relations[rank]["mean_log"]).all()
                and np.isfinite(relations[rank]["draw_log"]).all()
                and len(relations[rank]["metadata"]) == 5
                for rank in RANKS
            )
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
            "ranks": RANKS,
            "primary_rank": PRIMARY_RANK,
            "compact_index": 1.0,
            "gamma": 1.4,
            "maximum_subspace_angle_degrees": MAXIMUM_SUBSPACE_ANGLE_DEGREES,
        },
    )
    return result


def demo() -> None:
    result = run_experiment(N_DEMO_GALAXY, N_DEMO_DRAW, 1)
    if not result["complete_path_passed"]:
        raise AssertionError(
            "Exp56 Stage 3d complete demo failed its sub-minute gate: "
            f"{result['elapsed_seconds']:.2f} s"
        )
    print(
        f"Exp56 Stage 3d complete 90-galaxy path passed in "
        f"{result['elapsed_seconds']:.2f} s",
        flush=True,
    )


def full_run() -> None:
    demo_path = OUTPUT_DIRECTORY / "demo_results.json"
    if not demo_path.exists():
        raise RuntimeError("run the Stage 3d demo before the full calculation")
    demo_result = json.loads(demo_path.read_text())
    if not demo_result.get("complete_path_passed", False):
        raise RuntimeError("the saved Stage 3d complete-path validation did not pass")
    result = run_experiment(0, N_FULL_DRAW, N_POPULATION_DRAW)
    print(
        f"Exp56 Stage 3d full relation complete: {result['n_galaxy']} galaxies "
        f"in {result['elapsed_seconds']:.1f} s",
        flush=True,
    )


def regenerate_figures() -> None:
    prediction_path = OUTPUT_DIRECTORY / "predictions.npz"
    result_path = OUTPUT_DIRECTORY / "results.json"
    if not prediction_path.exists() or not result_path.exists():
        raise RuntimeError("run the full Stage 3d calculation before figures")
    saved = np.load(prediction_path)
    sample = stage3.stage_1.load_sample(0)
    if not np.array_equal(saved["indices"], sample["indices"]):
        raise RuntimeError("saved Stage 3d indices do not match the current sample")
    result = json.loads(result_path.read_text())
    relations = {}
    for rank in RANKS:
        relation = {
            name: saved[f"rank{rank}_{name}"]
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
            control: saved[f"rank{rank}_{control}_mean_log"]
            for control in stage3.CONTROL_NAMES
        }
        relation["metadata"] = result["fold_metadata"][str(rank)]
        relations[rank] = relation
    baseline, _ = stage3c.load_baseline(sample, demo=False)
    comparisons = {
        int(rank): value
        for rank, value in result["rank_minus_stage3_comparisons"].items()
    }
    rank_comparison_figure(relations, comparisons, "")
    average_cog_by_halo_mass_figure(sample, relations, "")
    latent_direction_figure(relations[PRIMARY_RANK], "")
    cases_figure(sample, relations[PRIMARY_RANK], baseline, "")
    generate_standard_qa(sample, relations[PRIMARY_RANK], "")


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"checks", "demo", "run", "figures"}:
        raise SystemExit(
            "usage: stage3d_low_rank_relation.py [checks|demo|run|figures]"
        )
    if sys.argv[1] == "checks":
        values = run_checks()
        if not values["passed"]:
            raise RuntimeError("Stage 3d numerical checks failed")
    elif sys.argv[1] == "demo":
        demo()
    elif sys.argv[1] == "run":
        full_run()
    else:
        regenerate_figures()


if __name__ == "__main__":
    main()
