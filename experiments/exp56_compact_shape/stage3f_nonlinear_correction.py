"""Exp56 Stage 3f: rank-2 linear core plus rank-1 nonlinear correction.

Commands
--------
checks
    Run coefficient-count, orthogonality, closure, and synthetic checks.
demo
    Run the complete predeclared path on 90 mass-stratified galaxies.
run
    Run the full five-fold held-out calculation after the demo gate passes.
figures
    Rebuild Stage 3f figures from the saved full predictions and results.
"""

from __future__ import annotations

import itertools
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
import stage3d_low_rank_relation as stage3d  # noqa: E402
from hongshao.plotting import OKABE_ITO, save_fig, set_style  # noqa: E402
from hongshao.profile_emulator import density_from_cog  # noqa: E402
from refine_halo_map import SPARSE_TERMS, design_matrix  # noqa: E402

set_style()
matplotlib.rcParams["text.usetex"] = False

LINEAR_RANK = 2
NONLINEAR_RANK = 1
N_DEMO_GALAXY = 90
N_DEMO_DRAW = 8
N_FULL_DRAW = 32
N_POPULATION_DRAW = 4
DEMO_TIME_LIMIT_SECONDS = 60.0
MAXIMUM_CORRECTION_ANGLE_DEGREES = 30.0
SEED = 5710

FIGURE_DIRECTORY = HERE / "figures" / "stage3f"
OUTPUT_DIRECTORY = HERE / "outputs" / "stage3f"
RANK2_OUTPUT_DIRECTORY = HERE / "outputs" / "stage3d"


def candidate_supports() -> tuple[np.ndarray, ...]:
    """All one-, two-, and three-term nonlinear supports."""
    supports = []
    for count in (1, 2, 3):
        for selected in itertools.combinations(range(len(SPARSE_TERMS)), count):
            support = np.zeros(len(SPARSE_TERMS), dtype=bool)
            support[list(selected)] = True
            supports.append(support)
    return tuple(supports)


def terms_for_support(support: np.ndarray) -> tuple[tuple, ...]:
    support = np.asarray(support, bool)
    if support.shape != (len(SPARSE_TERMS),):
        raise ValueError("support must have one entry per established nonlinear term")
    return tuple(term for term, keep in zip(SPARSE_TERMS, support) if keep)


def support_code(support: np.ndarray) -> str:
    return "".join("1" if value else "0" for value in np.asarray(support, bool))


def effective_count_for_support(support: np.ndarray) -> int:
    """Identifiable target-relation count for the two additive blocks."""
    nonlinear_term_count = int(np.count_nonzero(support))
    linear_count = 4 + LINEAR_RANK * (5 + 4 - LINEAR_RANK)
    if nonlinear_term_count == 0:
        return linear_count
    return linear_count + NONLINEAR_RANK * (nonlinear_term_count + 4 - NONLINEAR_RANK)


def project_coefficient_rank(
    basis: np.ndarray, targets: np.ndarray, rank: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Least-squares coefficients truncated by fitted-response rank."""
    if basis.shape[1] == 0:
        return (
            np.empty((0, targets.shape[1])),
            np.empty((0, 0)),
            np.empty((targets.shape[1], 0)),
            np.empty(0),
        )
    ordinary, *_ = np.linalg.lstsq(basis, targets, rcond=None)
    fitted = basis @ ordinary
    _, singular_values, right_vectors = np.linalg.svd(fitted, full_matrices=False)
    fitted_rank = min(rank, basis.shape[1], targets.shape[1])
    output_loadings = right_vectors[:fitted_rank].T
    for axis in range(fitted_rank):
        pivot = np.argmax(np.abs(output_loadings[:, axis]))
        if output_loadings[pivot, axis] < 0.0:
            output_loadings[:, axis] *= -1.0
    halo_loadings = ordinary @ output_loadings
    coefficients = halo_loadings @ output_loadings.T
    return coefficients, halo_loadings, output_loadings, singular_values


@dataclass(frozen=True)
class BlockRankModel:
    """Training-standardized additive block-rank halo–CoG relation."""

    feature_mean: np.ndarray
    feature_standard_deviation: np.ndarray
    target_mean: np.ndarray
    target_standard_deviation: np.ndarray
    nonlinear_residualization_coefficients: np.ndarray
    linear_coefficients: np.ndarray
    nonlinear_coefficients: np.ndarray
    linear_halo_loadings: np.ndarray
    linear_output_loadings: np.ndarray
    nonlinear_halo_loadings: np.ndarray
    nonlinear_output_loadings: np.ndarray
    linear_singular_values: np.ndarray
    nonlinear_singular_values: np.ndarray
    nonlinear_score_standard_deviation: np.ndarray
    support: np.ndarray
    terms: tuple[tuple, ...]
    linear_condition_number: float
    nonlinear_condition_number: float

    def standardized(self, features: np.ndarray) -> np.ndarray:
        return (np.asarray(features, float) - self.feature_mean) / (
            self.feature_standard_deviation
        )

    def predict_mean(self, features: np.ndarray) -> np.ndarray:
        linear = self.standardized(features)
        nonlinear = nonlinear_block(self, features)
        standardized_prediction = linear @ self.linear_coefficients
        if nonlinear.shape[1]:
            standardized_prediction += nonlinear @ self.nonlinear_coefficients
        return self.target_mean + (
            standardized_prediction * self.target_standard_deviation
        )

    def raw_nonlinear_output_direction(self) -> np.ndarray:
        if self.nonlinear_output_loadings.shape[1] == 0:
            return np.empty((4, 0))
        scaled = (
            self.target_standard_deviation[:, None] * self.nonlinear_output_loadings
        )
        return np.linalg.qr(scaled)[0][:, :1]


def nonlinear_raw_basis(
    standardized_features: np.ndarray, terms: tuple[tuple, ...]
) -> np.ndarray:
    if not terms:
        return np.empty((len(standardized_features), 0))
    return design_matrix(standardized_features, terms)[:, 6:]


def nonlinear_block(model: BlockRankModel, features: np.ndarray) -> np.ndarray:
    """Apply the training-frozen residualization to nonlinear columns."""
    linear = model.standardized(features)
    raw = nonlinear_raw_basis(linear, model.terms)
    if raw.shape[1] == 0:
        return raw
    augmented = np.column_stack([np.ones(len(linear)), linear])
    return raw - augmented @ model.nonlinear_residualization_coefficients


def fit_block_rank_model(
    features: np.ndarray,
    targets: np.ndarray,
    support: np.ndarray,
) -> BlockRankModel:
    """Fit orthogonal rank-2 linear and rank-1 nonlinear response blocks."""
    features = np.asarray(features, float)
    targets = np.asarray(targets, float)
    support = np.asarray(support, bool)
    terms = terms_for_support(support)
    feature_mean = np.mean(features, axis=0)
    feature_standard_deviation = np.std(features, axis=0)
    feature_standard_deviation = np.where(
        feature_standard_deviation > 1e-12,
        feature_standard_deviation,
        1.0,
    )
    linear = (features - feature_mean) / feature_standard_deviation
    target_mean = np.mean(targets, axis=0)
    target_standard_deviation = np.std(targets, axis=0)
    target_standard_deviation = np.where(
        target_standard_deviation > 1e-12,
        target_standard_deviation,
        1.0,
    )
    standardized_targets = (targets - target_mean) / target_standard_deviation
    raw_nonlinear = nonlinear_raw_basis(linear, terms)
    augmented = np.column_stack([np.ones(len(linear)), linear])
    if raw_nonlinear.shape[1]:
        residualization_coefficients, *_ = np.linalg.lstsq(
            augmented, raw_nonlinear, rcond=None
        )
        nonlinear = raw_nonlinear - augmented @ residualization_coefficients
    else:
        residualization_coefficients = np.empty((6, 0))
        nonlinear = raw_nonlinear
    (
        linear_coefficients,
        linear_halo_loadings,
        linear_output_loadings,
        linear_singular_values,
    ) = project_coefficient_rank(linear, standardized_targets, LINEAR_RANK)
    (
        nonlinear_coefficients,
        nonlinear_halo_loadings,
        nonlinear_output_loadings,
        nonlinear_singular_values,
    ) = project_coefficient_rank(nonlinear, standardized_targets, NONLINEAR_RANK)
    nonlinear_scores = nonlinear @ nonlinear_halo_loadings
    linear_singular = np.linalg.svd(linear, compute_uv=False)
    linear_condition = float(linear_singular[0] / max(linear_singular[-1], 1e-15))
    if nonlinear.shape[1]:
        nonlinear_singular = np.linalg.svd(nonlinear, compute_uv=False)
        nonlinear_condition = float(
            nonlinear_singular[0] / max(nonlinear_singular[-1], 1e-15)
        )
    else:
        nonlinear_condition = 1.0
    return BlockRankModel(
        feature_mean=feature_mean,
        feature_standard_deviation=feature_standard_deviation,
        target_mean=target_mean,
        target_standard_deviation=target_standard_deviation,
        nonlinear_residualization_coefficients=residualization_coefficients,
        linear_coefficients=linear_coefficients,
        nonlinear_coefficients=nonlinear_coefficients,
        linear_halo_loadings=linear_halo_loadings,
        linear_output_loadings=linear_output_loadings,
        nonlinear_halo_loadings=nonlinear_halo_loadings,
        nonlinear_output_loadings=nonlinear_output_loadings,
        linear_singular_values=linear_singular_values,
        nonlinear_singular_values=nonlinear_singular_values,
        nonlinear_score_standard_deviation=np.std(nonlinear_scores, axis=0),
        support=support,
        terms=terms,
        linear_condition_number=linear_condition,
        nonlinear_condition_number=nonlinear_condition,
    )


def model_metadata(model: BlockRankModel) -> dict:
    return {
        "support": model.support,
        "support_code": support_code(model.support),
        "effective_coefficient_count": effective_count_for_support(model.support),
        "feature_mean": model.feature_mean,
        "feature_standard_deviation": model.feature_standard_deviation,
        "target_mean": model.target_mean,
        "target_standard_deviation": model.target_standard_deviation,
        "linear_coefficients": model.linear_coefficients,
        "nonlinear_coefficients": model.nonlinear_coefficients,
        "linear_halo_loadings": model.linear_halo_loadings,
        "linear_output_loadings": model.linear_output_loadings,
        "nonlinear_halo_loadings": model.nonlinear_halo_loadings,
        "nonlinear_output_loadings": model.nonlinear_output_loadings,
        "raw_nonlinear_output_direction": model.raw_nonlinear_output_direction(),
        "nonlinear_score_standard_deviation": model.nonlinear_score_standard_deviation,
        "linear_singular_values": model.linear_singular_values,
        "nonlinear_singular_values": model.nonlinear_singular_values,
        "linear_condition_number": model.linear_condition_number,
        "nonlinear_condition_number": model.nonlinear_condition_number,
    }


def run_checks() -> dict:
    """Run the predeclared closure, orthogonality, and recovery checks."""
    supports = candidate_supports()
    rng = np.random.default_rng(SEED)
    features = rng.normal(size=(900, 5))
    targets = rng.normal(size=(900, 4))
    empty_support = np.zeros(len(SPARSE_TERMS), dtype=bool)
    reference = stage3d.fit_reduced_rank_model(features, targets, (), rank=2)
    linear_core = fit_block_rank_model(features, targets, empty_support)
    linear_closure = float(
        np.max(
            np.abs(
                linear_core.predict_mean(features) - reference.predict_mean(features)
            )
        )
    )
    synthetic_support = np.asarray(
        [True, True, False, False, True, False, False], dtype=bool
    )
    seed_model = fit_block_rank_model(features, targets, synthetic_support)
    linear = seed_model.standardized(features)
    nonlinear = nonlinear_block(seed_model, features)
    augmented = np.column_stack([np.ones(len(features)), linear])
    orthogonality = float(np.max(np.abs(augmented.T @ nonlinear / len(features))))
    linear_coefficients = rng.normal(size=(5, 2)) @ rng.normal(size=(2, 4))
    nonlinear_coefficients = rng.normal(size=(3, 1)) @ rng.normal(size=(1, 4))
    synthetic_targets = (
        rng.normal(size=4)
        + linear @ linear_coefficients
        + nonlinear @ nonlinear_coefficients
    )
    recovered = fit_block_rank_model(features, synthetic_targets, synthetic_support)
    synthetic_error = float(
        np.max(np.abs(recovered.predict_mean(features) - synthetic_targets))
    )
    counts = sorted({effective_count_for_support(support) for support in supports})
    values = {
        "candidate_support_count": len(supports),
        "unique_support_count": len({support_code(value) for value in supports}),
        "effective_coefficient_counts": counts,
        "linear_core_maximum_prediction_difference": linear_closure,
        "maximum_training_block_cross_product": orthogonality,
        "synthetic_maximum_coordinate_prediction_error": synthetic_error,
        "passed": bool(
            len(supports) == 63
            and len({support_code(value) for value in supports}) == 63
            and counts == [22, 23, 24]
            and linear_closure < 1e-12
            and orthogonality < 1e-10
            and synthetic_error < 1e-10
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
) -> tuple[np.ndarray, list[dict], dict, bool]:
    """Select one to three nonlinear terms using inner held-out profiles."""
    folds = stage3c.deterministic_inner_folds(len(features), seed)
    all_rows = np.arange(len(features))
    gamma = np.full(len(features), 1.4)
    reference_parameters = np.empty_like(targets)
    for validation in folds:
        training = np.setdiff1d(all_rows, validation)
        reference_model = stage3d.fit_reduced_rank_model(
            features[training], targets[training], SPARSE_TERMS, rank=2
        )
        reference_parameters[validation] = reference_model.predict_mean(
            features[validation]
        )
    reference_metrics = stage3c.point_metric_summary(
        truth_log, stage3.decode_profiles(reference_parameters, gamma)
    )
    records = []
    for support in candidate_supports():
        mean_parameters = np.empty_like(targets)
        for validation in folds:
            training = np.setdiff1d(all_rows, validation)
            model = fit_block_rank_model(features[training], targets[training], support)
            mean_parameters[validation] = model.predict_mean(features[validation])
        metrics = stage3c.point_metric_summary(
            truth_log, stage3.decode_profiles(mean_parameters, gamma)
        )
        records.append(
            {
                "support": support,
                "support_code": support_code(support),
                "nonlinear_term_count": int(np.count_nonzero(support)),
                "effective_coefficient_count": effective_count_for_support(support),
                **metrics,
            }
        )
    for record in records:
        record["passes_inner_accuracy_envelope"] = candidate_passes_inner_envelope(
            record, reference_metrics
        )
    passing = [record for record in records if record["passes_inner_accuracy_envelope"]]
    if passing:
        selected = min(
            passing,
            key=lambda record: (
                record["nonlinear_term_count"],
                record["full_cog_rms_dex"],
                record["five_thirty_cog_rms_dex"],
                record["support_code"],
            ),
        )
    else:
        selected = min(
            records,
            key=lambda record: (
                record["full_cog_rms_dex"],
                record["five_thirty_cog_rms_dex"],
                record["support_code"],
            ),
        )
    for record in records:
        record["selected"] = bool(record is selected)
    return (
        np.asarray(selected["support"], bool),
        records,
        reference_metrics,
        bool(passing),
    )


def calibrated_draws(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    train_truth_log: np.ndarray,
    test_features: np.ndarray,
    model: BlockRankModel,
    support: np.ndarray,
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
        inner_model = fit_block_rank_model(
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
    """Nested selection and outer held-out block-rank prediction."""
    features = sample["features"]
    truth_log = sample["truth_log"]
    n_galaxy = len(features)
    gamma = np.full(n_galaxy, 1.4)
    mean_parameters = np.empty((n_galaxy, 4))
    draw_parameters = np.empty((n_draw, n_galaxy, 4))
    fitted_parameters = np.empty((n_galaxy, 4))
    representation_log = np.empty_like(truth_log)
    linear_core_parameters = np.empty((n_galaxy, 4))
    all_nonlinear_parameters = np.empty((n_galaxy, 4))
    fold_id = np.empty(n_galaxy, int)
    control_predictions = {
        name: np.empty_like(truth_log) for name in stage3.CONTROL_NAMES
    }
    variants = stage3.control_feature_sets(features)
    metadata = []
    all_rows = np.arange(n_galaxy)
    empty_support = np.zeros(len(SPARSE_TERMS), dtype=bool)
    full_support = np.ones(len(SPARSE_TERMS), dtype=bool)
    for fold_number, test in enumerate(stage3.folds_of(n_galaxy)):
        train = np.setdiff1d(all_rows, test)
        cell = stage3.load_profile_cell(sample, "fixed_gamma1p4", fold_number, demo)
        targets = np.asarray(cell["parameters"], float)
        fitted_parameters[test] = targets[test]
        representation_log[test] = cell["prediction_log"][test]
        fold_id[test] = fold_number
        support, records, inner_reference, envelope_available = (
            select_nonlinear_support(
                features[train],
                targets[train],
                truth_log[train],
                SEED + 1000 * fold_number,
            )
        )
        model = fit_block_rank_model(features[train], targets[train], support)
        linear_model = fit_block_rank_model(
            features[train], targets[train], empty_support
        )
        full_model = fit_block_rank_model(features[train], targets[train], full_support)
        mean_parameters[test] = model.predict_mean(features[test])
        linear_core_parameters[test] = linear_model.predict_mean(features[test])
        all_nonlinear_parameters[test] = full_model.predict_mean(features[test])
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
        for name, (feature_values, _) in variants.items():
            if name == "complete":
                control_parameters = mean_parameters[test]
            elif name == "final_mass_only":
                control_model = stage3d.fit_reduced_rank_model(
                    feature_values[train], targets[train], (), rank=2
                )
                control_parameters = control_model.predict_mean(feature_values[test])
            else:
                control_model = fit_block_rank_model(
                    feature_values[train], targets[train], support
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
                "selected_terms": terms_for_support(support),
                "effective_coefficient_count": effective_count_for_support(support),
                "inner_accuracy_envelope_available": envelope_available,
                "inner_complete_rank2_metrics": inner_reference,
                "model": model_metadata(model),
                "linear_core_model": model_metadata(linear_model),
                "all_nonlinear_model": model_metadata(full_model),
                "calibration": calibration,
                "selection_records": records,
            }
        )
        print(
            f"fold {fold_number}: support {support_code(support)}, "
            f"{effective_count_for_support(support)} effective coefficients, "
            f"inner envelope {'available' if envelope_available else 'failed'}",
            flush=True,
        )
    return {
        "mean_parameters": mean_parameters,
        "draw_parameters": draw_parameters,
        "fitted_parameters": fitted_parameters,
        "mean_log": stage3.decode_profiles(mean_parameters, gamma),
        "draw_log": stage3.decode_profiles(draw_parameters, gamma),
        "representation_log": representation_log,
        "linear_core_mean_log": stage3.decode_profiles(linear_core_parameters, gamma),
        "all_nonlinear_mean_log": stage3.decode_profiles(
            all_nonlinear_parameters, gamma
        ),
        "gamma": gamma,
        "fold_id": fold_id,
        "control_predictions": control_predictions,
        "metadata": metadata,
    }


def load_rank2_reference(
    sample: dict[str, np.ndarray], demo: bool
) -> tuple[dict, dict]:
    """Load the saved complete-basis Stage 3d rank-2 relation."""
    prefix = "demo_" if demo else ""
    prediction_path = RANK2_OUTPUT_DIRECTORY / f"{prefix}predictions.npz"
    result_path = RANK2_OUTPUT_DIRECTORY / f"{prefix}results.json"
    if not prediction_path.exists() or not result_path.exists():
        raise RuntimeError("matching complete-rank-2 Stage 3d products are required")
    saved = np.load(prediction_path)
    if not np.array_equal(saved["indices"], sample["indices"]):
        raise RuntimeError("Stage 3f and complete-rank-2 galaxy indices do not match")
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


def correction_stability(metadata: list[dict]) -> dict:
    """Fold-to-fold principal angles for the independent output direction."""
    directions = [
        np.asarray(fold["model"]["raw_nonlinear_output_direction"], float)
        for fold in metadata
    ]
    angles = []
    for first in range(len(directions)):
        for second in range(first + 1, len(directions)):
            angles.append(
                float(
                    np.degrees(subspace_angles(directions[first], directions[second]))[
                        0
                    ]
                )
            )
    return {
        "pairwise_angles_degrees": angles,
        "median_pairwise_angle_degrees": float(np.median(angles)),
        "maximum_pairwise_angle_degrees": float(np.max(angles)),
    }


def support_summary(metadata: list[dict]) -> dict:
    supports = np.asarray([fold["selected_support"] for fold in metadata], dtype=bool)
    return {
        "selected_support_codes": [support_code(value) for value in supports],
        "selected_effective_coefficient_counts": [
            int(fold["effective_coefficient_count"]) for fold in metadata
        ],
        "nonlinear_term_selection_frequency": np.mean(supports, axis=0),
        "all_inner_accuracy_envelopes_available": bool(
            all(fold["inner_accuracy_envelope_available"] for fold in metadata)
        ),
        "correction_direction_stability": correction_stability(metadata),
    }


def compare_with_rank2(
    sample: dict[str, np.ndarray],
    relation: dict,
    summary: dict,
    reference: dict,
    reference_result: dict,
) -> dict:
    """Apply paired outer-held-out gates against complete rank 2."""
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
    diagnostic_points = {
        "linear_rank2_core": stage3c.point_metric_summary(
            truth_log, relation["linear_core_mean_log"]
        ),
        "all_seven_rank1_correction": stage3c.point_metric_summary(
            truth_log, relation["all_nonlinear_mean_log"]
        ),
    }
    candidate_radial = stage3c.radial_subset_metrics(truth_log, relation["mean_log"])
    reference_radial = stage3c.radial_subset_metrics(truth_log, reference["mean_log"])
    outside = np.any(
        (relation["mean_parameters"] < stage3.stage_1.LOWER)
        | (relation["mean_parameters"] > stage3.stage_1.UPPER),
        axis=1,
    )
    structure = support_summary(relation["metadata"])
    reference_summary = reference_result["rank_summaries"]["2"]
    gates = {
        "all_inner_accuracy_envelopes_available": structure[
            "all_inner_accuracy_envelopes_available"
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
        "correction_direction_maximum_angle_below_30_degrees": structure[
            "correction_direction_stability"
        ]["maximum_pairwise_angle_degrees"]
        < MAXIMUM_CORRECTION_ANGLE_DEGREES,
    }
    return {
        "candidate_point_metrics": candidate_point,
        "complete_rank2_point_metrics": reference_point,
        "diagnostic_point_metrics": diagnostic_points,
        "paired_candidate_minus_complete_rank2_bootstrap_16_50_84": paired,
        "candidate_radial_subset_metrics": candidate_radial,
        "complete_rank2_radial_subset_metrics": reference_radial,
        "outside_bound_fraction": float(np.mean(outside)),
        "structure": structure,
        "numeric_gates": gates,
        "all_adoption_gates_pass": bool(all(gates.values())),
    }


def selection_figure(relation: dict, comparison: dict, prefix: str) -> None:
    """Show nested accuracy, selected supports, and diagnostic ceilings."""
    figure, axes = plt.subplots(1, 4, figsize=(18.0, 4.5))
    for fold in relation["metadata"]:
        records = fold["selection_records"]
        reference = fold["inner_complete_rank2_metrics"]
        counts = np.asarray(
            [record["effective_coefficient_count"] for record in records]
        )
        delta_cog = 1000.0 * np.asarray(
            [
                record["full_cog_rms_dex"] - reference["full_cog_rms_dex"]
                for record in records
            ]
        )
        delta_size = 1000.0 * np.asarray(
            [
                np.mean(
                    [
                        record[f"absolute_R{fraction}_error"]
                        - reference[f"absolute_R{fraction}_error"]
                        for fraction in (50, 80, 90)
                    ]
                )
                for record in records
            ]
        )
        axes[0].scatter(counts, delta_cog, s=9, color="0.7", alpha=0.25)
        axes[1].scatter(counts, delta_size, s=9, color="0.7", alpha=0.25)
        selected = next(record for record in records if record["selected"])
        axes[0].scatter(
            selected["effective_coefficient_count"],
            1000.0 * (selected["full_cog_rms_dex"] - reference["full_cog_rms_dex"]),
            s=48,
            color=OKABE_ITO[1],
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )
        axes[1].scatter(
            selected["effective_coefficient_count"],
            np.mean(
                [
                    1000.0
                    * (
                        selected[f"absolute_R{fraction}_error"]
                        - reference[f"absolute_R{fraction}_error"]
                    )
                    for fraction in (50, 80, 90)
                ]
            ),
            s=48,
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
            r"inner mean $\Delta |\log R_q|$ [$10^{-3}$ dex]",
            "R50/R80/R90 envelope",
        ),
    ):
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set(
            xlabel="effective mean coefficients",
            ylabel=ylabel,
            title=title,
            xticks=(22, 23, 24),
        )
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
    axes[2].set_title("Selected nonlinear terms")
    figure.colorbar(image, ax=axes[2], ticks=(0, 1), label="excluded / retained")

    labels = (
        "linear\ncore",
        "selected\n22--24",
        "all seven\nterms",
        "complete\nrank 2",
    )
    points = comparison["diagnostic_point_metrics"]
    values = (
        points["linear_rank2_core"]["full_cog_rms_dex"],
        comparison["candidate_point_metrics"]["full_cog_rms_dex"],
        points["all_seven_rank1_correction"]["full_cog_rms_dex"],
        comparison["complete_rank2_point_metrics"]["full_cog_rms_dex"],
    )
    axes[3].bar(
        range(4), values, color=(OKABE_ITO[3], OKABE_ITO[1], OKABE_ITO[2], "0.55")
    )
    axes[3].set_xticks(range(4), labels)
    axes[3].set_ylabel("median held-out full-CoG RMS [dex]")
    axes[3].set_title("Outer diagnostic ceilings")
    figure.suptitle(
        "Rank-2 linear core plus an independent rank-1 nonlinear correction"
    )
    figure.tight_layout()
    save_fig(figure, FIGURE_DIRECTORY / f"{prefix}selection_and_ceiling")
    plt.close(figure)


def aligned_correction_metadata(
    fold: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return sign-aligned full halo weights, raw output direction, and response."""
    model = fold["model"]
    support = np.asarray(fold["selected_support"], bool)
    output = (
        np.asarray(model["target_standard_deviation"], float)
        * np.asarray(model["nonlinear_output_loadings"], float)[:, 0]
    )
    pivot = int(np.argmax(np.abs(output)))
    sign = 1.0 if output[pivot] >= 0.0 else -1.0
    output *= sign
    output /= max(np.linalg.norm(output), 1e-15)
    halo = np.zeros(len(SPARSE_TERMS))
    selected_halo = np.asarray(model["nonlinear_halo_loadings"], float)[:, 0] * sign
    selected_halo /= max(np.linalg.norm(selected_halo), 1e-15)
    halo[support] = selected_halo
    response = (
        np.asarray(model["target_standard_deviation"], float)
        * np.asarray(model["nonlinear_output_loadings"], float)[:, 0]
        * float(np.asarray(model["nonlinear_score_standard_deviation"], float)[0])
        * sign
    )
    return halo, output, response


def correction_response_figure(relation: dict, comparison: dict, prefix: str) -> None:
    """Show the fitted nonlinear weights and decoded one-sigma response."""
    halo_weights = []
    output_directions = []
    cog_responses = []
    density_responses = []
    density_radius = None
    for fold in relation["metadata"]:
        halo, output, response = aligned_correction_metadata(fold)
        halo_weights.append(halo)
        output_directions.append(output)
        base = np.asarray(fold["model"]["target_mean"], float)
        base_log = stage3.decode_profiles(base[None], np.asarray([1.4]))[0]
        shifted_log = stage3.decode_profiles(
            (base + response)[None], np.asarray([1.4])
        )[0]
        base_density, density_radius = density_from_cog(
            base_log[None], stage3.stage_1.RADII
        )
        shifted_density, _ = density_from_cog(shifted_log[None], stage3.stage_1.RADII)
        cog_responses.append(shifted_log - base_log)
        density_responses.append(shifted_density[0] - base_density[0])
    halo_weights = np.asarray(halo_weights)
    output_directions = np.asarray(output_directions)
    cog_responses = np.asarray(cog_responses)
    density_responses = np.asarray(density_responses)
    figure, axes = plt.subplots(2, 2, figsize=(13.8, 8.2))
    halo_maximum = max(np.max(np.abs(halo_weights)), 1e-12)
    halo_image = axes[0, 0].imshow(
        halo_weights,
        cmap="PuOr",
        vmin=-halo_maximum,
        vmax=halo_maximum,
        aspect="auto",
    )
    axes[0, 0].set_xticks(
        range(len(stage3c.NONLINEAR_TERM_LABELS)),
        stage3c.NONLINEAR_TERM_LABELS,
        rotation=45,
        ha="right",
        fontsize=8,
    )
    axes[0, 0].set_yticks(range(5), [f"fold {value}" for value in range(5)])
    axes[0, 0].set_title("Normalized nonlinear halo weights")
    figure.colorbar(halo_image, ax=axes[0, 0], label="signed normalized weight")
    output_maximum = max(np.max(np.abs(output_directions)), 1e-12)
    output_image = axes[0, 1].imshow(
        output_directions,
        cmap="PuOr",
        vmin=-output_maximum,
        vmax=output_maximum,
        aspect="auto",
    )
    axes[0, 1].set_xticks(range(4), stage3d.OUTPUT_LABELS, rotation=25, ha="right")
    axes[0, 1].set_yticks(range(5), [f"fold {value}" for value in range(5)])
    maximum_angle = comparison["structure"]["correction_direction_stability"][
        "maximum_pairwise_angle_degrees"
    ]
    axes[0, 1].set_title(
        f"Independent output direction; max angle {maximum_angle:.1f} degrees"
    )
    figure.colorbar(output_image, ax=axes[0, 1], label="signed normalized loading")
    for fold, (cog, density) in enumerate(zip(cog_responses, density_responses)):
        axes[1, 0].plot(stage3.stage_1.RADII, cog, color=OKABE_ITO[1], alpha=0.25)
        axes[1, 1].plot(density_radius, density, color=OKABE_ITO[2], alpha=0.25)
    axes[1, 0].plot(
        stage3.stage_1.RADII,
        np.median(cog_responses, axis=0),
        color=OKABE_ITO[1],
        linewidth=2.2,
        label="fold median",
    )
    axes[1, 1].plot(
        density_radius,
        np.median(density_responses, axis=0),
        color=OKABE_ITO[2],
        linewidth=2.2,
        label="fold median",
    )
    for axis in axes[1]:
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xscale("log")
        axis.set_xlabel("projected radius [kpc]")
        axis.legend(frameon=False)
    axes[1, 0].set_ylabel(r"$\Delta\log_{10}M_\star(<R)$ [dex]")
    axes[1, 0].set_title(r"Decoded response to a $+1\sigma$ nonlinear score")
    axes[1, 1].set_ylabel(r"$\Delta\log_{10}\Sigma_\star$ [dex]")
    axes[1, 1].set_title(r"Density response to a $+1\sigma$ nonlinear score")
    figure.suptitle("Independent nonlinear correction; no radial sign imposed")
    figure.tight_layout()
    save_fig(figure, FIGURE_DIRECTORY / f"{prefix}correction_response")
    plt.close(figure)


def average_cog_by_halo_mass_figure(
    sample: dict[str, np.ndarray],
    relation: dict,
    reference: dict,
    stage3_baseline: dict,
    prefix: str,
) -> None:
    """Compare candidate, diagnostic ceilings, and references in halo bins."""
    truth = sample["truth_log"]
    halo_mass = sample["features"][:, 0]
    bins = np.array_split(np.argsort(halo_mass), 3)
    models = (
        ("rank-2 + correction", relation["mean_log"], OKABE_ITO[1], "-"),
        ("complete rank 2", reference["mean_log"], "0.45", "--"),
        ("linear rank-2 core", relation["linear_core_mean_log"], OKABE_ITO[3], ":"),
        (
            "all-seven correction",
            relation["all_nonlinear_mean_log"],
            OKABE_ITO[2],
            "-.",
        ),
        (
            "unrestricted Stage 3",
            stage3_baseline["mean_log"],
            OKABE_ITO[0],
            (0, (3, 2)),
        ),
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
    axes[0, 0].legend(frameon=False, fontsize=7)
    figure.suptitle("Held-out average CoGs in halo-mass bins")
    figure.tight_layout()
    save_fig(figure, FIGURE_DIRECTORY / f"{prefix}average_cog_by_halo_mass")
    plt.close(figure)


def stellar_mass_planes_figure(
    sample: dict[str, np.ndarray], relation: dict, reference: dict, prefix: str
) -> None:
    """Directly compare candidate and complete-rank-2 population planes."""
    truth_cog = 10.0 ** sample["truth_log"][:, None, :]
    candidate_cog = 10.0 ** relation["mean_log"][:, None, :]
    reference_cog = 10.0 ** reference["mean_log"][:, None, :]
    truth, candidate, _, _ = stage3.qa.measure_all(
        candidate_cog, truth_cog, stage3.stage_1.RADII
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
            ("rank-2 + correction", candidate),
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
            *values["rank-2 + correction"],
            s=10,
            facecolors="none",
            edgecolors=OKABE_ITO[1],
            alpha=0.45,
            linewidth=0.55,
            label="rank-2 + correction",
        )
        statistics = {
            name: stage3.qa.plane_stats(*coordinate)
            for name, coordinate in values.items()
        }
        axis.set_title(
            "scatter TNG / complete / correction\n"
            f"{statistics['TNG']['scatter']:.3f} / "
            f"{statistics['complete rank 2']['scatter']:.3f} / "
            f"{statistics['rank-2 + correction']['scatter']:.3f}",
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
    """Show galaxies where the independent correction helps or hurts."""
    truth = sample["truth_log"]
    candidate = relation["mean_log"]
    complete = reference["mean_log"]
    change = np.sqrt(np.mean((candidate - truth) ** 2, axis=1)) - np.sqrt(
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
            stage3.stage_1.RADII,
            candidate[row],
            color=OKABE_ITO[1],
            label="rank-2 + correction",
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
            candidate[row] - candidate[row, -1] - truth_pinned,
            color=OKABE_ITO[1],
        )
        for axis in axes[:, column]:
            axis.set_xscale("log")
        axes[1, column].set_xlabel("projected radius [kpc]")
    axes[0, 0].set_ylabel(r"$\log_{10}M_\star(<R)$ [$M_\odot$]")
    axes[1, 0].set_ylabel("mass-pinned CoG residual [dex]")
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle("Held-out galaxies: independent nonlinear correction")
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
    model_name = f"exp56_stage3f_{prefix}nonlinear_correction"
    figure_directory = FIGURE_DIRECTORY / f"{prefix}standard_qa"
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
    model_label = "rank-2 linear + rank-1 nonlinear"
    stage3.density_by_halo_mass_figure(
        mean_log,
        truth_log,
        halo_mass,
        figdir=figure_directory,
        model_name=model_name,
        model_label=model_label,
        experiment_label="exp56 Stage 3f",
    )
    stage3.single_epoch_planes_figure(
        qa_result,
        figdir=figure_directory,
        model_name=model_name,
        model_label=model_label,
        experiment_label="exp56 Stage 3f",
    )
    stage3.single_epoch_size_figure(
        mean_log,
        truth_log,
        qa_result,
        figdir=figure_directory,
        model_name=model_name,
        model_label=model_label,
        experiment_label="exp56 Stage 3f",
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
        "linear_core_mean_log",
        "all_nonlinear_mean_log",
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
        raise RuntimeError(f"Stage 3f numerical checks failed: {checks}")
    sample = stage3.stage_1.load_sample(n_max)
    if demo and len(sample["indices"]) != N_DEMO_GALAXY:
        raise RuntimeError("Stage 3f demo did not load exactly 90 galaxies")
    relation = cross_validated_relation(sample, n_draw, demo)
    summary = stage3.summarize_model(sample, relation, n_population_draw)
    reference, reference_result = load_rank2_reference(sample, demo)
    stage3_baseline, _ = stage3c.load_baseline(sample, demo)
    comparison = compare_with_rank2(
        sample, relation, summary, reference, reference_result
    )

    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    selection_figure(relation, comparison, prefix)
    correction_response_figure(relation, comparison, prefix)
    average_cog_by_halo_mass_figure(
        sample, relation, reference, stage3_baseline, prefix
    )
    stellar_mass_planes_figure(sample, relation, reference, prefix)
    cases_figure(sample, relation, reference, prefix)
    standard = generate_standard_qa(sample, relation, prefix)
    save_predictions(sample, relation, prefix)
    elapsed_seconds = time.perf_counter() - started
    result = {
        "status": "stage3f_validation" if demo else "stage3f_full",
        "n_galaxy": len(sample["indices"]),
        "n_draw": n_draw,
        "n_population_draw": n_population_draw,
        "numerical_checks": checks,
        "candidate_summary": summary,
        "candidate_minus_complete_rank2": comparison,
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
            "linear_rank": LINEAR_RANK,
            "nonlinear_rank": NONLINEAR_RANK,
            "compact_index": 1.0,
            "gamma": 1.4,
            "candidate_support_count": len(candidate_supports()),
            "candidate_nonlinear_term_counts": (1, 2, 3),
            "inner_fold_count": 4,
            "outer_fold_count": 5,
            "maximum_correction_angle_degrees": (MAXIMUM_CORRECTION_ANGLE_DEGREES),
        },
    )
    return result


def demo() -> None:
    result = run_experiment(N_DEMO_GALAXY, N_DEMO_DRAW, 1)
    if not result["complete_path_passed"]:
        raise AssertionError(
            "Exp56 Stage 3f complete demo failed its sub-minute gate: "
            f"{result['elapsed_seconds']:.2f} s"
        )
    print(
        f"Exp56 Stage 3f complete 90-galaxy path passed in "
        f"{result['elapsed_seconds']:.2f} s",
        flush=True,
    )


def full_run() -> None:
    demo_path = OUTPUT_DIRECTORY / "demo_results.json"
    if not demo_path.exists():
        raise RuntimeError("run the Stage 3f demo before the full calculation")
    demo_result = json.loads(demo_path.read_text())
    if not demo_result.get("complete_path_passed", False):
        raise RuntimeError("the saved Stage 3f complete-path validation did not pass")
    result = run_experiment(0, N_FULL_DRAW, N_POPULATION_DRAW)
    print(
        f"Exp56 Stage 3f full relation complete: {result['n_galaxy']} galaxies "
        f"in {result['elapsed_seconds']:.1f} s",
        flush=True,
    )


def regenerate_figures() -> None:
    prediction_path = OUTPUT_DIRECTORY / "predictions.npz"
    result_path = OUTPUT_DIRECTORY / "results.json"
    if not prediction_path.exists() or not result_path.exists():
        raise RuntimeError("run the full Stage 3f calculation before figures")
    saved = np.load(prediction_path)
    sample = stage3.stage_1.load_sample(0)
    if not np.array_equal(saved["indices"], sample["indices"]):
        raise RuntimeError("saved Stage 3f indices do not match the current sample")
    relation = {
        name: saved[name]
        for name in (
            "mean_parameters",
            "draw_parameters",
            "fitted_parameters",
            "mean_log",
            "draw_log",
            "representation_log",
            "linear_core_mean_log",
            "all_nonlinear_mean_log",
            "gamma",
            "fold_id",
        )
    }
    relation["control_predictions"] = {
        control: saved[f"{control}_mean_log"] for control in stage3.CONTROL_NAMES
    }
    result = json.loads(result_path.read_text())
    relation["metadata"] = result["fold_metadata"]
    comparison = result["candidate_minus_complete_rank2"]
    reference, _ = load_rank2_reference(sample, demo=False)
    stage3_baseline, _ = stage3c.load_baseline(sample, demo=False)
    selection_figure(relation, comparison, "")
    correction_response_figure(relation, comparison, "")
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
        raise SystemExit(
            "usage: stage3f_nonlinear_correction.py [checks|demo|run|figures]"
        )
    if sys.argv[1] == "checks":
        values = run_checks()
        if not values["passed"]:
            raise RuntimeError("Stage 3f numerical checks failed")
    elif sys.argv[1] == "demo":
        demo()
    elif sys.argv[1] == "run":
        full_run()
    else:
        regenerate_figures()


if __name__ == "__main__":
    main()
