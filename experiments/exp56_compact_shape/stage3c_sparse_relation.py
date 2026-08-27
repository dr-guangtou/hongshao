"""Exp56 Stage 3c: nested structured sparsification of the halo–CoG mean.

Commands
--------
checks
    Run numerical sparse-solver and zero-penalty closure checks.
demo
    Run the complete predeclared path on 90 mass-stratified galaxies.
run
    Run the full five-fold held-out calculation after the demo gate passes.
figures
    Rebuild Stage 3c figures from the saved full predictions and results.
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
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP55 = ROOT / "experiments" / "exp55_analytic_cog"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP55))
sys.path.insert(0, str(HERE))

import stage3_halo_cog as stage3  # noqa: E402
from hongshao.plotting import OKABE_ITO, save_fig, set_style  # noqa: E402
from refine_halo_map import (  # noqa: E402
    SPARSE_TERMS,
    design_matrix,
    fit_conditional_model,
)

set_style()
matplotlib.rcParams["text.usetex"] = False

CORE_COLUMN_COUNT = 6
CORE_COEFFICIENT_COUNT = 4 * CORE_COLUMN_COUNT
ALPHA_GRID = (0.25, 0.50, 0.75, 1.00)
PENALTY_FRACTION_GRID = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0)
SUPPORT_THRESHOLD = 1e-7
MAXIMUM_ITERATION = 5_000
CONVERGENCE_TOLERANCE = 1e-7
N_DEMO_GALAXY = 90
N_DEMO_DRAW = 8
N_FULL_DRAW = 32
N_POPULATION_DRAW = 4
DEMO_TIME_LIMIT_SECONDS = 60.0
SEED = 5650
NONLINEAR_TERM_LABELS = (
    r"$\log^2 M_{\rm peak}$",
    r"$\log t_c\,{\times}\,\mathrm{late}$",
    r"$\mathrm{early}^2$",
    r"$\mathrm{late}\,{\times}\,c_{200c}$",
    r"$\mathrm{late}^2$",
    r"$\log M_{\rm peak}\,{\times}\,\mathrm{early}$",
    r"$\log M_{\rm peak}\,{\times}\,\mathrm{late}$",
)

FIGURE_DIRECTORY = HERE / "figures" / "stage3c"
OUTPUT_DIRECTORY = HERE / "outputs" / "stage3c"
BASELINE_OUTPUT_DIRECTORY = HERE / "outputs" / "stage3"


@dataclass(frozen=True)
class SparseCoordinateModel:
    """Debiased sparse polynomial mean in raw analytic coordinates."""

    feature_mean: np.ndarray
    feature_standard_deviation: np.ndarray
    coefficients: np.ndarray
    nonlinear_support: np.ndarray

    def standardized(self, features: np.ndarray) -> np.ndarray:
        return (np.asarray(features, float) - self.feature_mean) / (
            self.feature_standard_deviation
        )

    def predict_mean(self, features: np.ndarray) -> np.ndarray:
        matrix = design_matrix(self.standardized(features), SPARSE_TERMS)
        return matrix @ self.coefficients.T


def sparse_group_proximal(
    coefficients: np.ndarray,
    penalty: float,
    alpha: float,
) -> np.ndarray:
    """Elementwise soft threshold followed by row-group shrinkage."""
    coefficients = np.asarray(coefficients, float)
    soft = np.sign(coefficients) * np.maximum(
        np.abs(coefficients) - penalty * alpha,
        0.0,
    )
    if alpha >= 1.0:
        return soft
    row_norm = np.linalg.norm(soft, axis=1, keepdims=True)
    shrinkage = np.maximum(
        1.0 - penalty * (1.0 - alpha) / np.maximum(row_norm, 1e-30),
        0.0,
    )
    return soft * shrinkage


def standardized_design(
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Training-only standardization and the established 13-column basis."""
    features = np.asarray(features, float)
    feature_mean = np.mean(features, axis=0)
    feature_standard_deviation = np.std(features, axis=0)
    feature_standard_deviation = np.where(
        feature_standard_deviation > 1e-12,
        feature_standard_deviation,
        1.0,
    )
    standardized = (features - feature_mean) / feature_standard_deviation
    return (
        design_matrix(standardized, SPARSE_TERMS),
        feature_mean,
        feature_standard_deviation,
    )


def debiased_coefficients(
    design: np.ndarray,
    targets: np.ndarray,
    nonlinear_support: np.ndarray,
) -> np.ndarray:
    """Refit each selected equation by ordinary least squares."""
    output = np.zeros((targets.shape[1], design.shape[1]))
    for target_index in range(targets.shape[1]):
        active = np.concatenate(
            [
                np.ones(CORE_COLUMN_COUNT, dtype=bool),
                nonlinear_support[:, target_index],
            ]
        )
        output[target_index, active], *_ = np.linalg.lstsq(
            design[:, active],
            targets[:, target_index],
            rcond=None,
        )
    return output


def fit_sparse_group_model(
    features: np.ndarray,
    targets: np.ndarray,
    alpha: float,
    penalty_fraction: float,
) -> tuple[SparseCoordinateModel, dict]:
    """Fit, threshold, and debias one structured-sparse coordinate mean."""
    features = np.asarray(features, float)
    targets = np.asarray(targets, float)
    if alpha not in ALPHA_GRID:
        raise ValueError(f"alpha must be one of {ALPHA_GRID}")
    if penalty_fraction not in PENALTY_FRACTION_GRID:
        raise ValueError(f"penalty_fraction must be one of {PENALTY_FRACTION_GRID}")
    design, feature_mean, feature_standard_deviation = standardized_design(features)
    target_mean = np.mean(targets, axis=0)
    target_standard_deviation = np.std(targets, axis=0)
    target_standard_deviation = np.where(
        target_standard_deviation > 1e-12,
        target_standard_deviation,
        1.0,
    )
    standardized_targets = (targets - target_mean) / target_standard_deviation

    if penalty_fraction == 0.0:
        nonlinear_support = np.ones(
            (len(SPARSE_TERMS), targets.shape[1]),
            dtype=bool,
        )
        coefficients = debiased_coefficients(design, targets, nonlinear_support)
        model = SparseCoordinateModel(
            feature_mean=feature_mean,
            feature_standard_deviation=feature_standard_deviation,
            coefficients=coefficients,
            nonlinear_support=nonlinear_support,
        )
        return model, {
            "alpha": alpha,
            "penalty_fraction": penalty_fraction,
            "penalty_value": 0.0,
            "active_mean_coefficient_count": int(coefficients.size),
            "n_iteration": 0,
            "converged": True,
        }

    coefficients = np.zeros((design.shape[1], targets.shape[1]))
    coefficients[:CORE_COLUMN_COUNT], *_ = np.linalg.lstsq(
        design[:, :CORE_COLUMN_COUNT],
        standardized_targets,
        rcond=None,
    )
    residual = design @ coefficients - standardized_targets
    nonlinear_gradient = design[:, CORE_COLUMN_COUNT:].T @ residual / len(design)
    penalty_scale = max(
        float(np.max(np.abs(nonlinear_gradient))),
        float(np.max(np.linalg.norm(nonlinear_gradient, axis=1))),
        1e-12,
    )
    penalty_value = penalty_fraction * penalty_scale
    lipschitz = float(np.linalg.norm(design, ord=2) ** 2 / len(design))
    step = 1.0 / max(lipschitz, 1e-12)
    converged = False
    for iteration in range(1, MAXIMUM_ITERATION + 1):
        residual = design @ coefficients - standardized_targets
        gradient = design.T @ residual / len(design)
        candidate = coefficients - step * gradient
        candidate[CORE_COLUMN_COUNT:] = sparse_group_proximal(
            candidate[CORE_COLUMN_COUNT:],
            penalty=step * penalty_value,
            alpha=alpha,
        )
        maximum_change = float(np.max(np.abs(candidate - coefficients)))
        coefficients = candidate
        if maximum_change < CONVERGENCE_TOLERANCE:
            converged = True
            break
    nonlinear_support = np.abs(coefficients[CORE_COLUMN_COUNT:]) > SUPPORT_THRESHOLD
    raw_coefficients = debiased_coefficients(design, targets, nonlinear_support)
    model = SparseCoordinateModel(
        feature_mean=feature_mean,
        feature_standard_deviation=feature_standard_deviation,
        coefficients=raw_coefficients,
        nonlinear_support=nonlinear_support,
    )
    return model, {
        "alpha": alpha,
        "penalty_fraction": penalty_fraction,
        "penalty_value": penalty_value,
        "active_mean_coefficient_count": int(
            CORE_COEFFICIENT_COUNT + np.count_nonzero(nonlinear_support)
        ),
        "n_iteration": iteration,
        "converged": converged,
    }


def fit_declared_support_model(
    features: np.ndarray,
    targets: np.ndarray,
    nonlinear_support: np.ndarray,
) -> SparseCoordinateModel:
    """Fit a frozen nonlinear support, used for matched shuffle controls."""
    design, feature_mean, feature_standard_deviation = standardized_design(features)
    nonlinear_support = np.asarray(nonlinear_support, bool)
    return SparseCoordinateModel(
        feature_mean=feature_mean,
        feature_standard_deviation=feature_standard_deviation,
        coefficients=debiased_coefficients(design, targets, nonlinear_support),
        nonlinear_support=nonlinear_support,
    )


def candidate_grid() -> tuple[tuple[float, float], ...]:
    """The predeclared grid with the duplicated zero-penalty cases removed."""
    values = [
        (alpha, fraction)
        for alpha in ALPHA_GRID
        for fraction in PENALTY_FRACTION_GRID
        if fraction > 0.0
    ]
    values.append((0.5, 0.0))
    return tuple(values)


def point_metric_summary(
    truth_log: np.ndarray,
    mean_log: np.ndarray,
) -> dict[str, float]:
    """Median per-galaxy quantities used by the inner simplicity gate."""
    metrics = stage3.per_galaxy_metrics(truth_log, mean_log, mean_log[None])
    return {
        name: float(np.median(values))
        for name, values in metrics.items()
        if name != "profile_crps_dex"
    }


def deterministic_inner_folds(n_galaxy: int, seed: int) -> list[np.ndarray]:
    order = np.random.default_rng(seed).permutation(n_galaxy)
    return list(np.array_split(order, 4))


def candidate_passes_inner_envelope(
    metrics: dict[str, float],
    reference: dict[str, float],
) -> bool:
    """Apply the profile-level simplicity envelope declared in the README."""
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


def select_sparse_candidate(
    features: np.ndarray,
    targets: np.ndarray,
    truth_log: np.ndarray,
    gamma: np.ndarray,
    seed: int,
) -> tuple[tuple[float, float], list[dict]]:
    """Choose sparsity entirely from four inner held-out folds."""
    folds = deterministic_inner_folds(len(features), seed)
    all_rows = np.arange(len(features))
    records = []
    for alpha, penalty_fraction in candidate_grid():
        mean_parameters = np.empty_like(targets)
        active_counts = []
        convergence = []
        for validation in folds:
            training = np.setdiff1d(all_rows, validation)
            model, metadata = fit_sparse_group_model(
                features[training],
                targets[training],
                alpha,
                penalty_fraction,
            )
            mean_parameters[validation] = model.predict_mean(features[validation])
            active_counts.append(metadata["active_mean_coefficient_count"])
            convergence.append(metadata["converged"])
        mean_log = stage3.decode_profiles(mean_parameters, gamma)
        records.append(
            {
                "alpha": alpha,
                "penalty_fraction": penalty_fraction,
                "median_active_mean_coefficient_count": float(np.median(active_counts)),
                "minimum_active_mean_coefficient_count": int(np.min(active_counts)),
                "maximum_active_mean_coefficient_count": int(np.max(active_counts)),
                "all_solver_fits_converged": bool(np.all(convergence)),
                **point_metric_summary(truth_log, mean_log),
            }
        )
    reference = next(record for record in records if record["penalty_fraction"] == 0.0)
    for record in records:
        record["passes_inner_accuracy_envelope"] = (
            candidate_passes_inner_envelope(record, reference)
            and record["all_solver_fits_converged"]
        )
    passing = [record for record in records if record["passes_inner_accuracy_envelope"]]
    if not passing:
        selected = reference
    else:
        selected = min(
            passing,
            key=lambda record: (
                record["median_active_mean_coefficient_count"],
                record["full_cog_rms_dex"],
                record["alpha"],
                -record["penalty_fraction"],
            ),
        )
    for record in records:
        record["selected"] = bool(record is selected)
    return (selected["alpha"], selected["penalty_fraction"]), records


def neighbour_draws(
    model: SparseCoordinateModel,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    n_draw: int,
    seed: int,
    n_neighbour: int = 128,
) -> np.ndarray:
    """Resample complete coordinate residuals from nearby training halos."""
    residual = train_targets - model.predict_mean(train_features)
    tree = cKDTree(model.standardized(train_features))
    _, neighbours = tree.query(
        model.standardized(test_features),
        k=min(n_neighbour, len(train_features)),
    )
    if neighbours.ndim == 1:
        neighbours = neighbours[:, None]
    choices = np.random.default_rng(seed).integers(
        0,
        neighbours.shape[1],
        (n_draw, len(test_features)),
    )
    selected = neighbours[np.arange(len(test_features))[None], choices]
    return model.predict_mean(test_features)[None] + residual[selected]


def calibrated_sparse_draws(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    train_truth_log: np.ndarray,
    test_features: np.ndarray,
    model: SparseCoordinateModel,
    alpha: float,
    penalty_fraction: float,
    n_draw: int,
    seed: int,
) -> tuple[np.ndarray, dict]:
    """Select residual inflation from inner held-out profile coverage."""
    scale_grid = np.asarray([1.00, 1.08, 1.16, 1.24, 1.32])
    folds = deterministic_inner_folds(len(train_features), seed)
    all_rows = np.arange(len(train_features))
    inner_mean = np.empty_like(train_targets)
    inner_draw = np.empty((n_draw, *train_targets.shape))
    inner_metadata = []
    for inner_number, validation in enumerate(folds):
        training = np.setdiff1d(all_rows, validation)
        inner_model, metadata = fit_sparse_group_model(
            train_features[training],
            train_targets[training],
            alpha,
            penalty_fraction,
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
        inner_metadata.append(metadata)
    gamma = np.full(len(train_features), 1.4)
    coverage = {}
    for scale in scale_grid:
        draw_log = stage3.decode_profiles(
            inner_mean[None] + scale * (inner_draw - inner_mean[None]),
            gamma,
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
        "selected_scale": float(selected_scale),
        "inner_coverage_by_scale": coverage,
        "inner_fit_metadata": inner_metadata,
    }


def cross_validated_sparse_relation(
    sample: dict[str, np.ndarray],
    n_draw: int,
    demo: bool,
) -> dict:
    """Nested selection, outer means, controls, and stochastic draws."""
    features = sample["features"]
    truth_log = sample["truth_log"]
    n_galaxy = len(features)
    mean_parameters = np.empty((n_galaxy, 4))
    draw_parameters = np.empty((n_draw, n_galaxy, 4))
    fitted_parameters = np.empty((n_galaxy, 4))
    representation_log = np.empty_like(truth_log)
    gamma = np.full(n_galaxy, 1.4)
    fold_id = np.empty(n_galaxy, int)
    control_predictions = {
        name: np.empty_like(truth_log) for name in stage3.CONTROL_NAMES
    }
    variants = stage3.control_feature_sets(features)
    metadata = []
    all_rows = np.arange(n_galaxy)
    for fold_number, test in enumerate(stage3.folds_of(n_galaxy)):
        train = np.setdiff1d(all_rows, test)
        cell = stage3.load_profile_cell(
            sample,
            "fixed_gamma1p4",
            fold_number,
            demo,
        )
        targets = np.asarray(cell["parameters"], float)
        fitted_parameters[test] = targets[test]
        representation_log[test] = cell["prediction_log"][test]
        fold_id[test] = fold_number
        selected, frontier = select_sparse_candidate(
            features[train],
            targets[train],
            truth_log[train],
            gamma[train],
            SEED + 1000 * fold_number,
        )
        alpha, penalty_fraction = selected
        model, fit_metadata = fit_sparse_group_model(
            features[train],
            targets[train],
            alpha,
            penalty_fraction,
        )
        mean_parameters[test] = model.predict_mean(features[test])
        draw_parameters[:, test], calibration = calibrated_sparse_draws(
            features[train],
            targets[train],
            truth_log[train],
            features[test],
            model,
            alpha,
            penalty_fraction,
            n_draw,
            SEED + 10_000 + 1000 * fold_number,
        )
        for name, (feature_values, terms) in variants.items():
            if name == "complete":
                control_parameters = mean_parameters[test]
            elif name == "final_mass_only":
                control_model = fit_conditional_model(
                    feature_values[train],
                    targets[train],
                    terms,
                )
                control_parameters = control_model.predict_mean(feature_values[test])
            else:
                control_model = fit_declared_support_model(
                    feature_values[train],
                    targets[train],
                    model.nonlinear_support,
                )
                control_parameters = control_model.predict_mean(feature_values[test])
            control_predictions[name][test] = stage3.decode_profiles(
                control_parameters,
                gamma[test],
            )
        metadata.append(
            {
                "fold": fold_number,
                "n_train": len(train),
                "n_test": len(test),
                "selected_alpha": alpha,
                "selected_penalty_fraction": penalty_fraction,
                "fit": fit_metadata,
                "nonlinear_support": model.nonlinear_support,
                "frontier": frontier,
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


def run_checks() -> dict:
    """Standalone numerical checks used before either data calculation."""
    rng = np.random.default_rng(SEED)
    features = rng.normal(size=(300, 5))
    targets = rng.normal(size=(300, 4))
    sparse, metadata = fit_sparse_group_model(features, targets, 0.5, 0.0)
    reference = fit_conditional_model(features, targets, SPARSE_TERMS)
    closure = float(
        np.max(np.abs(sparse.predict_mean(features) - reference.predict_mean(features)))
    )
    synthetic_features = rng.normal(size=(800, 5))
    synthetic_design, _, _ = standardized_design(synthetic_features)
    synthetic_coefficients = np.zeros((4, synthetic_design.shape[1]))
    synthetic_coefficients[:, :CORE_COLUMN_COUNT] = rng.normal(
        scale=0.2, size=(4, CORE_COLUMN_COUNT)
    )
    synthetic_support = np.zeros((len(SPARSE_TERMS), 4), dtype=bool)
    for term_index, output_index in (
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 0),
        (5, 1),
        (6, 2),
    ):
        synthetic_support[term_index, output_index] = True
    synthetic_coefficients[:, CORE_COLUMN_COUNT:] = np.where(
        synthetic_support.T,
        0.25 * rng.choice((-1.0, 1.0), size=(4, len(SPARSE_TERMS))),
        0.0,
    )
    synthetic_targets = synthetic_design @ synthetic_coefficients.T
    recovered, recovery_metadata = fit_sparse_group_model(
        synthetic_features,
        synthetic_targets,
        alpha=0.5,
        penalty_fraction=0.125,
    )
    synthetic_prediction_error = float(
        np.max(np.abs(recovered.predict_mean(synthetic_features) - synthetic_targets))
    )
    support_recovered = bool(
        np.array_equal(recovered.nonlinear_support, synthetic_support)
    )
    values = {
        "zero_penalty_maximum_prediction_difference": closure,
        "zero_penalty_active_mean_coefficient_count": metadata[
            "active_mean_coefficient_count"
        ],
        "synthetic_support_recovered_exactly": support_recovered,
        "synthetic_maximum_prediction_error": synthetic_prediction_error,
        "synthetic_active_mean_coefficient_count": recovery_metadata[
            "active_mean_coefficient_count"
        ],
        "passed": bool(
            closure < 1e-10
            and metadata["active_mean_coefficient_count"] == 52
            and support_recovered
            and synthetic_prediction_error < 1e-10
        ),
    }
    print(json.dumps(stage3.json_safe(values), indent=2))
    return values


def load_baseline(sample: dict[str, np.ndarray], demo: bool) -> tuple[dict, dict]:
    """Load the exact Stage 3 fixed-gamma predictions for paired comparison."""
    prefix = "demo_" if demo else ""
    prediction_path = BASELINE_OUTPUT_DIRECTORY / f"{prefix}predictions.npz"
    result_path = BASELINE_OUTPUT_DIRECTORY / f"{prefix}results.json"
    if not prediction_path.exists() or not result_path.exists():
        raise RuntimeError(
            "The matching Stage 3 predictions and results are required as baseline"
        )
    saved = np.load(prediction_path)
    if not np.array_equal(saved["indices"], sample["indices"]):
        raise RuntimeError("Stage 3c and Stage 3 galaxy indices do not match")
    baseline = {
        "mean_parameters": saved["fixed_gamma1p4_mean_parameters"],
        "fitted_parameters": saved["fixed_gamma1p4_fitted_parameters"],
        "mean_log": saved["fixed_gamma1p4_mean_log"],
        "draw_log": saved["fixed_gamma1p4_draw_log"],
        "representation_log": saved["fixed_gamma1p4_representation_log"],
    }
    return baseline, json.loads(result_path.read_text())


def radial_subset_metrics(
    truth_log: np.ndarray,
    prediction_log: np.ndarray,
) -> dict[str, float]:
    """Evaluate the same held-out mean on alternating radial samples."""
    pinned = (prediction_log - truth_log) - (prediction_log[:, -1:] - truth_log[:, -1:])
    return {
        "even_radius_pinned_cog_rms_dex": float(
            np.median(np.sqrt(np.mean(pinned[:, ::2] ** 2, axis=1)))
        ),
        "odd_radius_pinned_cog_rms_dex": float(
            np.median(np.sqrt(np.mean(pinned[:, 1::2] ** 2, axis=1)))
        ),
    }


def support_stability(metadata: list[dict]) -> dict:
    supports = [np.asarray(fold["nonlinear_support"], bool) for fold in metadata]
    frequencies = np.mean(supports, axis=0)
    jaccard = []
    for first in range(len(supports)):
        for second in range(first + 1, len(supports)):
            union = np.count_nonzero(supports[first] | supports[second])
            intersection = np.count_nonzero(supports[first] & supports[second])
            jaccard.append(intersection / union if union else 1.0)
    return {
        "selection_frequency": frequencies,
        "median_pairwise_jaccard": float(np.median(jaccard)),
        "minimum_pairwise_jaccard": float(np.min(jaccard)),
    }


def compare_with_baseline(
    sample: dict[str, np.ndarray],
    relation: dict,
    summary: dict,
    baseline: dict,
    baseline_result: dict,
) -> dict:
    """Apply the predeclared held-out accuracy and simplicity gates."""
    truth_log = sample["truth_log"]
    candidate_metrics = stage3.per_galaxy_metrics(
        truth_log, relation["mean_log"], relation["draw_log"]
    )
    baseline_metrics = stage3.per_galaxy_metrics(
        truth_log, baseline["mean_log"], baseline["draw_log"]
    )
    paired = {
        name: stage3.bootstrap_median_interval(
            candidate_metrics[name] - baseline_metrics[name], SEED + index
        )
        for index, name in enumerate(candidate_metrics)
    }
    mass_bin = stage3.worst_mass_bin_shape_bootstrap(
        truth_log,
        baseline["mean_log"],
        relation["mean_log"],
        sample["features"][:, 0],
    )
    paired["worst_mass_bin_shape_residual_percent"] = mass_bin
    candidate_point = point_metric_summary(truth_log, relation["mean_log"])
    baseline_point = point_metric_summary(truth_log, baseline["mean_log"])
    candidate_radial = radial_subset_metrics(truth_log, relation["mean_log"])
    baseline_radial = radial_subset_metrics(truth_log, baseline["mean_log"])
    outside = np.any(
        (relation["mean_parameters"] < stage3.stage_1.LOWER)
        | (relation["mean_parameters"] > stage3.stage_1.UPPER),
        axis=1,
    )
    active_counts = [
        int(fold["fit"]["active_mean_coefficient_count"])
        for fold in relation["metadata"]
    ]
    stability = support_stability(relation["metadata"])
    baseline_summary = baseline_result["representations"]["fixed_gamma1p4"]
    gates = {
        "every_outer_fold_has_at_most_32_mean_coefficients": max(active_counts) <= 32,
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
    }
    return {
        "active_mean_coefficient_counts_by_outer_fold": active_counts,
        "median_active_mean_coefficient_count": float(np.median(active_counts)),
        "candidate_point_metrics": candidate_point,
        "baseline_point_metrics": baseline_point,
        "paired_sparse_minus_52_coefficient_bootstrap_16_50_84": paired,
        "candidate_radial_subset_metrics": candidate_radial,
        "baseline_radial_subset_metrics": baseline_radial,
        "outside_bound_fraction": float(np.mean(outside)),
        "support_stability": stability,
        "numeric_gates": gates,
        "all_adoption_gates_pass": bool(all(gates.values())),
        "complexity_decision": (
            "adopt"
            if all(gates.values())
            else "partial_reduction"
            if max(active_counts) <= 39
            else "not_meaningful"
        ),
    }


def complexity_figure(
    relation: dict,
    comparison: dict,
    prefix: str,
) -> None:
    """Show inner-fold accuracy, selected support, and outer-fold complexity."""
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.5))
    colors = dict(zip(ALPHA_GRID, OKABE_ITO[:4]))
    for fold in relation["metadata"]:
        for record in fold["frontier"]:
            axes[0].scatter(
                record["median_active_mean_coefficient_count"],
                record["full_cog_rms_dex"],
                color=colors[record["alpha"]],
                alpha=0.18,
                s=22,
            )
            if record["selected"]:
                axes[0].scatter(
                    record["median_active_mean_coefficient_count"],
                    record["full_cog_rms_dex"],
                    facecolor=colors[record["alpha"]],
                    edgecolor="black",
                    linewidth=0.8,
                    s=62,
                    zorder=5,
                )
    for alpha, color in colors.items():
        axes[0].scatter([], [], color=color, label=rf"$\alpha={alpha:.2g}$")
    axes[0].set(
        xlabel="active mean coefficients",
        ylabel="inner held-out CoG RMS [dex]",
        title="Nested sparsity selection",
    )
    axes[0].legend(frameon=False, fontsize=8, ncol=2)

    frequency = np.asarray(comparison["support_stability"]["selection_frequency"])
    image = axes[1].imshow(frequency.T, vmin=0.0, vmax=1.0, cmap="cividis")
    axes[1].set_xticks(range(len(SPARSE_TERMS)))
    axes[1].set_xticklabels(NONLINEAR_TERM_LABELS, rotation=45, ha="right", fontsize=8)
    axes[1].set_yticks(range(4))
    axes[1].set_yticklabels(
        [r"$\log M_{\star,148}$", r"$f_{\rm compact}$", r"$\log R_c$", r"$\log R_e$"]
    )
    axes[1].set_title("Nonlinear-term selection frequency")
    figure.colorbar(image, ax=axes[1], label="fraction of outer folds")

    counts = comparison["active_mean_coefficient_counts_by_outer_fold"]
    axes[2].bar(np.arange(5), counts, color=OKABE_ITO[1])
    axes[2].axhline(32, color=OKABE_ITO[3], linestyle="--", label="adoption limit")
    axes[2].axhline(52, color="0.35", linestyle=":", label="Stage 3 reference")
    axes[2].set(
        xlabel="outer held-out fold",
        ylabel="active mean coefficients",
        xticks=np.arange(5),
        title="Complexity on each outer fit",
        ylim=(0, 57),
    )
    axes[2].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    save_fig(figure, FIGURE_DIRECTORY / f"{prefix}sparsity_frontier")
    plt.close(figure)


def paired_metric_figure(
    comparison: dict,
    prefix: str,
) -> None:
    """Display paired held-out changes relative to the 52-coefficient model."""
    names = (
        "full_cog_rms_dex",
        "five_thirty_cog_rms_dex",
        "full_density_rms_dex",
        "five_thirty_density_rms_dex",
        "absolute_R50_error",
        "absolute_R80_error",
        "absolute_R90_error",
    )
    labels = ("CoG", "CoG 5–30", "density", "density 5–30", "R50", "R80", "R90")
    paired = comparison["paired_sparse_minus_52_coefficient_bootstrap_16_50_84"]
    interval = np.asarray([paired[name] for name in names])
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    axis.axhline(0.0, color="0.35", linewidth=1)
    axis.errorbar(
        np.arange(len(names)),
        interval[:, 1],
        yerr=np.vstack(
            (interval[:, 1] - interval[:, 0], interval[:, 2] - interval[:, 1])
        ),
        fmt="o",
        color=OKABE_ITO[1],
        capsize=4,
    )
    axis.set_xticks(np.arange(len(names)), labels, rotation=25, ha="right")
    axis.set_ylabel("sparse minus 52-coefficient median error [dex]")
    axis.set_title("Paired held-out accuracy; lower is better")
    figure.tight_layout()
    save_fig(figure, FIGURE_DIRECTORY / f"{prefix}paired_accuracy")
    plt.close(figure)


def cases_figure(
    sample: dict[str, np.ndarray],
    relation: dict,
    baseline: dict,
    prefix: str,
) -> None:
    """Show direct individual CoGs and residuals where sparsification helps or hurts."""
    truth = sample["truth_log"]
    sparse = relation["mean_log"]
    reference = baseline["mean_log"]
    sparse_rms = np.sqrt(np.mean((sparse - truth) ** 2, axis=1))
    reference_rms = np.sqrt(np.mean((reference - truth) ** 2, axis=1))
    change = sparse_rms - reference_rms
    ordered = np.argsort(change)
    rows = [ordered[0], ordered[len(ordered) // 2], ordered[-1]]
    titles = ("largest improvement", "typical change", "largest degradation")
    figure, axes = plt.subplots(2, 3, figsize=(14.2, 7.2), sharex=True)
    for column, (row, title) in enumerate(zip(rows, titles)):
        axes[0, column].plot(
            stage3.stage_1.RADII, truth[row], color="black", label="measured"
        )
        axes[0, column].plot(
            stage3.stage_1.RADII, reference[row], color="0.55", label="52 coefficients"
        )
        axes[0, column].plot(
            stage3.stage_1.RADII, sparse[row], color=OKABE_ITO[1], label="sparse"
        )
        axes[0, column].set_xscale("log")
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
            sparse[row] - sparse[row, -1] - truth_pinned,
            color=OKABE_ITO[1],
        )
        axes[1, column].set_xscale("log")
        axes[1, column].set_xlabel("projected radius [kpc]")
    axes[0, 0].set_ylabel(r"$\log_{10} M_\star(<R)$ [$M_\odot$]")
    axes[1, 0].set_ylabel("total-mass-pinned CoG residual [dex]")
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle("Held-out galaxies: structured sparse halo–CoG relation")
    figure.tight_layout()
    save_fig(figure, FIGURE_DIRECTORY / f"{prefix}individual_cases")
    plt.close(figure)


def generate_standard_qa(
    sample: dict[str, np.ndarray], relation: dict, prefix: str
) -> dict:
    truth_log = sample["truth_log"]
    mean_log = relation["mean_log"]
    draw_log = relation["draw_log"]
    halo_mass = sample["features"][:, 0]
    model_name = f"exp56_stage3c_{prefix}fixed_gamma1p4"
    figure_directory = FIGURE_DIRECTORY / f"{prefix}fixed_gamma1p4_standard_qa"
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
    model_label = r"structured sparse, fixed $\gamma=1.4$"
    stage3.density_by_halo_mass_figure(
        mean_log,
        truth_log,
        halo_mass,
        figdir=figure_directory,
        model_name=model_name,
        model_label=model_label,
        experiment_label="exp56 Stage 3c",
    )
    stage3.single_epoch_planes_figure(
        qa_result,
        figdir=figure_directory,
        model_name=model_name,
        model_label=model_label,
        experiment_label="exp56 Stage 3c",
    )
    stage3.single_epoch_size_figure(
        mean_log,
        truth_log,
        qa_result,
        figdir=figure_directory,
        model_name=model_name,
        model_label=model_label,
        experiment_label="exp56 Stage 3c",
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
        raise RuntimeError(f"Stage 3c numerical checks failed: {checks}")
    sample = stage3.stage_1.load_sample(n_max)
    if demo and len(sample["indices"]) != N_DEMO_GALAXY:
        raise RuntimeError("Stage 3c demo did not load exactly 90 galaxies")
    relation = cross_validated_sparse_relation(sample, n_draw, demo)
    summary = stage3.summarize_model(sample, relation, n_population_draw)
    baseline, baseline_result = load_baseline(sample, demo)
    comparison = compare_with_baseline(
        sample, relation, summary, baseline, baseline_result
    )

    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    complexity_figure(relation, comparison, prefix)
    paired_metric_figure(comparison, prefix)
    cases_figure(sample, relation, baseline, prefix)
    standard = generate_standard_qa(sample, relation, prefix)
    save_predictions(sample, relation, prefix)
    elapsed_seconds = time.perf_counter() - started
    result = {
        "status": "stage3c_validation" if demo else "stage3c_full",
        "n_galaxy": len(sample["indices"]),
        "n_draw": n_draw,
        "n_population_draw": n_population_draw,
        "numerical_and_synthetic_checks": checks,
        "sparse_representation": summary,
        "sparse_minus_52_coefficient_baseline": comparison,
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
            "compact_index": 1.0,
            "gamma": 1.4,
            "alpha_grid": ALPHA_GRID,
            "penalty_fraction_grid": PENALTY_FRACTION_GRID,
            "inner_fold_count": 4,
            "outer_fold_count": 5,
        },
    )
    return result


def demo() -> None:
    result = run_experiment(N_DEMO_GALAXY, N_DEMO_DRAW, 1)
    if not result["complete_path_passed"]:
        raise AssertionError(
            "Exp56 Stage 3c complete demo failed its sub-minute gate: "
            f"{result['elapsed_seconds']:.2f} s"
        )
    print(
        f"Exp56 Stage 3c complete 90-galaxy path passed in "
        f"{result['elapsed_seconds']:.2f} s",
        flush=True,
    )


def full_run() -> None:
    demo_path = OUTPUT_DIRECTORY / "demo_results.json"
    if not demo_path.exists():
        raise RuntimeError("run the Stage 3c demo before the full calculation")
    demo_result = json.loads(demo_path.read_text())
    if not demo_result.get("complete_path_passed", False):
        raise RuntimeError("the saved Stage 3c complete-path validation did not pass")
    result = run_experiment(0, N_FULL_DRAW, N_POPULATION_DRAW)
    print(
        f"Exp56 Stage 3c full relation complete: {result['n_galaxy']} galaxies "
        f"in {result['elapsed_seconds']:.1f} s",
        flush=True,
    )


def regenerate_figures() -> None:
    prediction_path = OUTPUT_DIRECTORY / "predictions.npz"
    result_path = OUTPUT_DIRECTORY / "results.json"
    if not prediction_path.exists() or not result_path.exists():
        raise RuntimeError("run the full Stage 3c calculation before figures")
    saved = np.load(prediction_path)
    sample = stage3.stage_1.load_sample(0)
    if not np.array_equal(saved["indices"], sample["indices"]):
        raise RuntimeError("saved Stage 3c indices do not match the current sample")
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
    baseline, _ = load_baseline(sample, demo=False)
    comparison = result["sparse_minus_52_coefficient_baseline"]
    complexity_figure(relation, comparison, "")
    paired_metric_figure(comparison, "")
    cases_figure(sample, relation, baseline, "")
    generate_standard_qa(sample, relation, "")


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"checks", "demo", "run", "figures"}:
        raise SystemExit("usage: stage3c_sparse_relation.py [checks|demo|run|figures]")
    if sys.argv[1] == "checks":
        values = run_checks()
        if not values["passed"]:
            raise RuntimeError("Stage 3c numerical checks failed")
        return
    if sys.argv[1] == "demo":
        demo()
    elif sys.argv[1] == "run":
        full_run()
    else:
        regenerate_figures()


if __name__ == "__main__":
    main()
