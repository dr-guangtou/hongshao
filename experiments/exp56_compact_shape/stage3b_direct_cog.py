"""Exp56 Stage 3b: fit the shared halo relation in decoded CoG space.

Commands
--------
checks
    Run decoder, Jacobian, and synthetic shared-relation checks.
demo
    Run the complete predeclared path on 90 galaxies with eight draws.
run
    Run the full held-out calculation after the demo passes.
figures
    Regenerate custom and standard-QA figures from saved full predictions.
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
from scipy.special import expit, gammainc, gammaincinv

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP55 = ROOT / "experiments" / "exp55_analytic_cog"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP55))
sys.path.insert(0, str(HERE))

import stage3_halo_cog as stage3  # noqa: E402
from hongshao.plotting import OKABE_ITO, save_fig, set_style  # noqa: E402
from refine_halo_map import design_matrix, fit_conditional_model  # noqa: E402

set_style()
matplotlib.rcParams["text.usetex"] = False

N_DEMO_GALAXY = 90
N_DEMO_DRAW = 8
N_FULL_DRAW = 32
N_POPULATION_DRAW = 4
DEMO_TIME_LIMIT_SECONDS = 60.0
SEED = 5631
SCALE_GRID = np.asarray([1.00, 1.08, 1.16, 1.24, 1.32])
JACOBIAN_STEP = np.asarray([0.0, 1e-5, 1e-5, 1e-5])

FIGURE_DIRECTORY = HERE / "figures" / "stage3b"
OUTPUT_DIRECTORY = HERE / "outputs" / "stage3b"
BASELINE_OUTPUT_DIRECTORY = HERE / "outputs" / "stage3"


@dataclass(frozen=True)
class DecodedFitResult:
    coefficients: np.ndarray
    start_rms_dex: float
    fitted_rms_dex: float
    n_function_evaluation: int
    optimality: float
    success: bool


@dataclass(frozen=True)
class DirectCogModel:
    terms: tuple[tuple, ...]
    feature_mean: np.ndarray
    feature_standard_deviation: np.ndarray
    coefficients: np.ndarray

    def standardized(self, features: np.ndarray) -> np.ndarray:
        return (np.asarray(features, float) - self.feature_mean) / (
            self.feature_standard_deviation
        )

    def predict_mean(self, features: np.ndarray) -> np.ndarray:
        matrix = design_matrix(self.standardized(features), self.terms)
        return matrix @ self.coefficients.T


def vectorized_decode(
    parameters: np.ndarray,
    gamma: np.ndarray,
    radii: np.ndarray = stage3.stage_1.RADII,
) -> np.ndarray:
    """Decode all galaxies at once using the established n=1 mixture."""
    parameters = np.atleast_2d(np.asarray(parameters, float))
    gamma = np.asarray(gamma, float)
    radii = np.asarray(radii, float)
    if len(parameters) != len(gamma):
        raise ValueError("gamma array does not match the galaxy axis")
    log_mass = parameters[:, 0]
    compact_fraction = expit(parameters[:, 1])
    compact_radius = 10.0 ** np.clip(parameters[:, 2], -6.0, 6.0)
    radius_ratio = 1.0 + np.exp(np.clip(parameters[:, 3], -40.0, 40.0))
    extended_radius = compact_radius * radius_ratio

    radius_grid = radii[None, :]
    sersic_constant = gammaincinv(2.0, 0.5)
    compact = gammainc(2.0, sersic_constant * radius_grid / compact_radius[:, None])
    compact /= compact[:, -1:]

    half_scale = np.sqrt(2.0 ** (1.0 / (gamma - 1.0)) - 1.0)
    core_radius = extended_radius / half_scale
    extended = 1.0 - (
        1.0 + (radius_grid / core_radius[:, None]) ** 2
    ) ** (1.0 - gamma[:, None])
    extended /= extended[:, -1:]

    normalized_cog = (
        compact_fraction[:, None] * compact
        + (1.0 - compact_fraction[:, None]) * extended
    )
    return log_mass[:, None] + np.log10(np.clip(normalized_cog, 1e-300, None))


def local_decoder_jacobian(
    parameters: np.ndarray,
    gamma: np.ndarray,
    radii: np.ndarray,
) -> np.ndarray:
    """Profile derivative with respect to each galaxy's four coordinates."""
    parameters = np.asarray(parameters, float)
    output = np.empty((len(parameters), len(radii), 4))
    output[:, :, 0] = 1.0
    for coordinate in range(1, 4):
        step = JACOBIAN_STEP[coordinate]
        offset = np.zeros_like(parameters)
        offset[:, coordinate] = step
        output[:, :, coordinate] = (
            vectorized_decode(parameters + offset, gamma, radii)
            - vectorized_decode(parameters - offset, gamma, radii)
        ) / (2.0 * step)
    return output


class DecodedCogLeastSquares:
    """Least-squares residual and chained Jacobian for shared coefficients."""

    def __init__(
        self,
        design: np.ndarray,
        truth_log: np.ndarray,
        gamma: np.ndarray,
        radii: np.ndarray = stage3.stage_1.RADII,
    ) -> None:
        self.design = np.asarray(design, float)
        self.truth_log = np.asarray(truth_log, float)
        self.gamma = np.asarray(gamma, float)
        self.radii = np.asarray(radii, float)
        self.n_coefficient = self.design.shape[1]

    def parameters(self, flat_coefficients: np.ndarray) -> np.ndarray:
        coefficients = np.asarray(flat_coefficients, float).reshape(
            4, self.n_coefficient
        )
        return self.design @ coefficients.T

    def residual(self, flat_coefficients: np.ndarray) -> np.ndarray:
        prediction = vectorized_decode(
            self.parameters(flat_coefficients), self.gamma, self.radii
        )
        return (prediction - self.truth_log).ravel()

    def jacobian(self, flat_coefficients: np.ndarray) -> np.ndarray:
        local = local_decoder_jacobian(
            self.parameters(flat_coefficients), self.gamma, self.radii
        )
        chained = np.einsum("nrk,np->nrkp", local, self.design, optimize=True)
        return chained.reshape(len(self.truth_log) * len(self.radii), -1)


def fit_decoded_coefficients(
    design: np.ndarray,
    truth_log: np.ndarray,
    gamma: np.ndarray,
    start_coefficients: np.ndarray,
    radii: np.ndarray = stage3.stage_1.RADII,
) -> DecodedFitResult:
    """Refit decoded profiles with feasible Gauss-Newton line searches."""
    objective = DecodedCogLeastSquares(design, truth_log, gamma, radii)
    coefficients = np.asarray(start_coefficients, float).ravel().copy()
    start_residual = objective.residual(coefficients)
    residual = start_residual.copy()
    cost = float(np.dot(residual, residual))
    n_function_evaluation = 1
    optimality = np.inf
    success = np.isfinite(residual).all()
    for _ in range(100):
        jacobian = objective.jacobian(coefficients)
        gradient = jacobian.T @ residual
        optimality = float(np.max(np.abs(gradient)) / max(len(residual), 1))
        if optimality < 1e-10:
            break
        step, *_ = np.linalg.lstsq(jacobian, -residual, rcond=1e-10)
        if np.linalg.norm(step) <= 1e-10 * (1.0 + np.linalg.norm(coefficients)):
            break
        accepted = False
        relative_improvement = 0.0
        for line_search_number in range(25):
            scale = 0.5**line_search_number
            candidate = coefficients + scale * step
            candidate_parameters = objective.parameters(candidate)
            within_bounds = np.all(
                (candidate_parameters >= stage3.stage_1.LOWER)
                & (candidate_parameters <= stage3.stage_1.UPPER)
            )
            if not within_bounds:
                continue
            candidate_residual = objective.residual(candidate)
            n_function_evaluation += 1
            candidate_cost = float(np.dot(candidate_residual, candidate_residual))
            if np.isfinite(candidate_cost) and candidate_cost < cost:
                relative_improvement = (cost - candidate_cost) / max(cost, 1e-30)
                coefficients = candidate
                residual = candidate_residual
                cost = candidate_cost
                accepted = True
                break
        if not accepted or relative_improvement < 1e-10:
            break
    return DecodedFitResult(
        coefficients=coefficients.reshape(4, design.shape[1]),
        start_rms_dex=float(np.sqrt(np.mean(start_residual**2))),
        fitted_rms_dex=float(np.sqrt(np.mean(residual**2))),
        n_function_evaluation=n_function_evaluation,
        optimality=optimality,
        success=bool(success and np.isfinite(coefficients).all()),
    )


def fit_direct_model(
    features: np.ndarray,
    targets: np.ndarray,
    truth_log: np.ndarray,
    gamma: np.ndarray,
    terms: tuple[tuple, ...],
    radius_indices: np.ndarray | None = None,
) -> tuple[DirectCogModel, dict]:
    """Initialize from coordinate regression, then minimize decoded CoG error."""
    baseline = fit_conditional_model(features, targets, terms)
    standardized = baseline.standardized(features)
    matrix = design_matrix(standardized, terms)
    if radius_indices is None:
        radius_indices = np.arange(len(stage3.stage_1.RADII))
    radius_indices = np.asarray(radius_indices, int)
    fit = fit_decoded_coefficients(
        matrix,
        truth_log[:, radius_indices],
        gamma,
        baseline.coefficients,
        stage3.stage_1.RADII[radius_indices],
    )
    model = DirectCogModel(
        terms=terms,
        feature_mean=baseline.feature_mean,
        feature_standard_deviation=baseline.feature_standard_deviation,
        coefficients=fit.coefficients,
    )
    return model, {
        "start_rms_dex": fit.start_rms_dex,
        "fitted_rms_dex": fit.fitted_rms_dex,
        "n_function_evaluation": fit.n_function_evaluation,
        "optimality": fit.optimality,
        "success": fit.success,
    }


def neighbour_draws(
    model: DirectCogModel,
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
        0, neighbours.shape[1], (n_draw, len(test_features))
    )
    selected = neighbours[np.arange(len(test_features))[None], choices]
    return model.predict_mean(test_features)[None] + residual[selected]


def calibrated_fold_draws(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    train_gamma: np.ndarray,
    train_truth_log: np.ndarray,
    test_features: np.ndarray,
    test_gamma: np.ndarray,
    model: DirectCogModel,
    terms: tuple[tuple, ...],
    n_draw: int,
    seed: int,
) -> tuple[np.ndarray, dict]:
    """Select residual inflation with four direct-fit inner validation folds."""
    order = np.random.default_rng(seed).permutation(len(train_features))
    all_rows = np.arange(len(train_features))
    inner_mean = np.empty_like(train_targets)
    inner_draw = np.empty((n_draw, *train_targets.shape))
    inner_fit_metadata = []
    for inner_number, validation in enumerate(np.array_split(order, 4)):
        training = np.setdiff1d(all_rows, validation)
        inner_model, fit_metadata = fit_direct_model(
            train_features[training],
            train_targets[training],
            train_truth_log[training],
            train_gamma[training],
            terms,
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
        inner_fit_metadata.append(fit_metadata)
    coverage = {}
    for scale in SCALE_GRID:
        scaled_parameters = (
            inner_mean[None] + scale * (inner_draw - inner_mean[None])
        ).reshape(-1, 4)
        draw_log = vectorized_decode(
            scaled_parameters,
            np.tile(train_gamma, n_draw),
        ).reshape(n_draw, len(train_features), -1)
        coverage[f"{scale:.2f}"] = float(
            np.mean(
                (train_truth_log >= np.quantile(draw_log, 0.16, axis=0))
                & (train_truth_log <= np.quantile(draw_log, 0.84, axis=0))
            )
        )
    selected_scale = min(
        SCALE_GRID,
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
        "inner_direct_fits": inner_fit_metadata,
    }


def cross_validated_relation(
    sample: dict[str, np.ndarray],
    representation: str,
    n_draw: int,
    demo: bool,
) -> dict:
    """Five-fold direct-CoG means, controls, draws, and radial jackknife."""
    features = sample["features"]
    truth_log = sample["truth_log"]
    n_galaxy = len(features)
    mean_parameters = np.empty((n_galaxy, 4))
    draw_parameters = np.empty((n_draw, n_galaxy, 4))
    fitted_parameters = np.empty((n_galaxy, 4))
    representation_log = np.empty_like(truth_log)
    jackknife_mean_log = np.empty_like(truth_log)
    gamma_by_galaxy = np.empty(n_galaxy)
    fold_id = np.empty(n_galaxy, int)
    control_predictions = {
        name: np.empty_like(truth_log) for name in stage3.CONTROL_NAMES
    }
    variants = stage3.control_feature_sets(features)
    metadata = []
    all_rows = np.arange(n_galaxy)
    radial_jackknife = np.unique(
        np.append(np.arange(0, len(stage3.stage_1.RADII), 2), len(stage3.stage_1.RADII) - 1)
    )

    for fold_number, test in enumerate(stage3.folds_of(n_galaxy)):
        train = np.setdiff1d(all_rows, test)
        cell = stage3.load_profile_cell(sample, representation, fold_number, demo)
        gamma = np.asarray(cell["gamma"], float)
        targets = np.asarray(cell["parameters"], float)
        fitted_parameters[test] = targets[test]
        representation_log[test] = cell["prediction_log"][test]
        gamma_by_galaxy[test] = gamma[test]
        fold_id[test] = fold_number

        complete_train = stage3.augmented_features(features[train], gamma[train])
        complete_test = stage3.augmented_features(features[test], gamma[test])
        complete_model, complete_fit = fit_direct_model(
            complete_train,
            targets[train],
            truth_log[train],
            gamma[train],
            stage3.SPARSE_TERMS,
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
            stage3.SPARSE_TERMS,
            n_draw,
            SEED + 1000 * fold_number,
        )
        jackknife_model, jackknife_fit = fit_direct_model(
            complete_train,
            targets[train],
            truth_log[train],
            gamma[train],
            stage3.SPARSE_TERMS,
            radial_jackknife,
        )
        jackknife_mean_log[test] = vectorized_decode(
            jackknife_model.predict_mean(complete_test), gamma[test]
        )

        control_fit_metadata = {}
        for name, (feature_values, terms) in variants.items():
            train_values = stage3.augmented_features(feature_values[train], gamma[train])
            test_values = stage3.augmented_features(feature_values[test], gamma[test])
            if name == "complete":
                model = complete_model
                fit_metadata = complete_fit
            else:
                model, fit_metadata = fit_direct_model(
                    train_values,
                    targets[train],
                    truth_log[train],
                    gamma[train],
                    terms,
                )
            control_predictions[name][test] = vectorized_decode(
                model.predict_mean(test_values), gamma[test]
            )
            control_fit_metadata[name] = fit_metadata

        metadata.append(
            {
                "fold": fold_number,
                "n_train": len(train),
                "n_test": len(test),
                "complete_direct_fit": complete_fit,
                "radial_jackknife_direct_fit": jackknife_fit,
                "control_direct_fits": control_fit_metadata,
                **calibration,
            }
        )

    mean_log = vectorized_decode(mean_parameters, gamma_by_galaxy)
    draw_log = vectorized_decode(
        draw_parameters.reshape(-1, 4),
        np.tile(gamma_by_galaxy, n_draw),
    ).reshape(n_draw, n_galaxy, -1)
    return {
        "mean_parameters": mean_parameters,
        "draw_parameters": draw_parameters,
        "fitted_parameters": fitted_parameters,
        "mean_log": mean_log,
        "draw_log": draw_log,
        "representation_log": representation_log,
        "jackknife_mean_log": jackknife_mean_log,
        "gamma": gamma_by_galaxy,
        "fold_id": fold_id,
        "control_predictions": control_predictions,
        "metadata": metadata,
    }


def load_baseline(sample: dict[str, np.ndarray]) -> tuple[dict, dict]:
    prediction_path = BASELINE_OUTPUT_DIRECTORY / "predictions.npz"
    result_path = BASELINE_OUTPUT_DIRECTORY / "results.json"
    if not prediction_path.exists() or not result_path.exists():
        raise RuntimeError("Stage 3 full predictions are required as the frozen baseline")
    saved = np.load(prediction_path)
    if not np.array_equal(saved["indices"], sample["indices"]):
        raise RuntimeError("Stage 3 baseline galaxy indices do not match Stage 3b")
    relations = {}
    for representation in stage3.REPRESENTATIONS:
        relations[representation] = {
            "mean_parameters": saved[f"{representation}_mean_parameters"],
            "mean_log": saved[f"{representation}_mean_log"],
            "draw_log": saved[f"{representation}_draw_log"],
        }
    return relations, json.loads(result_path.read_text())


def compare_with_baseline(
    sample: dict[str, np.ndarray],
    direct_relations: dict[str, dict],
    direct_summaries: dict[str, dict],
    baseline_relations: dict[str, dict],
    baseline_result: dict,
) -> dict:
    """Paired held-out comparison of direct and coordinate-space means."""
    output = {}
    truth_log = sample["truth_log"]
    for representation_index, representation in enumerate(stage3.REPRESENTATIONS):
        direct = direct_relations[representation]
        baseline = baseline_relations[representation]
        direct_metrics = stage3.per_galaxy_metrics(
            truth_log, direct["mean_log"], direct["draw_log"]
        )
        baseline_metrics = stage3.per_galaxy_metrics(
            truth_log, baseline["mean_log"], baseline["draw_log"]
        )
        paired = {
            name: stage3.bootstrap_median_interval(
                direct_metrics[name] - baseline_metrics[name],
                SEED + 100 * representation_index + metric_index,
            )
            for metric_index, name in enumerate(direct_metrics)
        }
        mass_bin = stage3.worst_mass_bin_shape_bootstrap(
            truth_log,
            baseline["mean_log"],
            direct["mean_log"],
            sample["features"][:, 0],
        )
        paired["worst_mass_bin_shape_residual_percent"] = mass_bin
        parameters = direct["mean_parameters"]
        outside = np.any(
            (parameters < stage3.stage_1.LOWER)
            | (parameters > stage3.stage_1.UPPER),
            axis=1,
        )
        jackknife_rms = np.sqrt(
            np.mean((direct["jackknife_mean_log"] - direct["mean_log"]) ** 2, axis=1)
        )
        representation_floor = float(
            np.median(
                np.sqrt(
                    np.mean((direct["representation_log"] - truth_log) ** 2, axis=1)
                )
            )
        )
        direct_summary = direct_summaries[representation]
        baseline_summary = baseline_result["representations"][representation]
        gates = {
            "mass_bin_shape_resolved_improvement": mass_bin[2] < 0.0,
            "full_cog_not_resolved_worse": paired["full_cog_rms_dex"][0] <= 0.0,
            "five_thirty_cog_not_resolved_worse": paired[
                "five_thirty_cog_rms_dex"
            ][0]
            <= 0.0,
            "full_density_not_resolved_worse": paired["full_density_rms_dex"][0]
            <= 0.0,
            "five_thirty_density_not_resolved_worse": paired[
                "five_thirty_density_rms_dex"
            ][0]
            <= 0.0,
            "R50_not_resolved_worse": paired["absolute_R50_error"][0] <= 0.0,
            "R80_not_resolved_worse": paired["absolute_R80_error"][0] <= 0.0,
            "R90_not_resolved_worse": paired["absolute_R90_error"][0] <= 0.0,
            "profile_crps_within_one_percent": direct_summary["draw"][
                "profile_crps_dex"
            ]
            <= 1.01 * baseline_summary["draw"]["profile_crps_dex"],
            "coverage_within_three_percentage_points": abs(
                direct_summary["draw"]["profile_coverage_68"] - 0.68
            )
            <= 0.03,
            "no_mean_coordinate_outside_profile_bounds": not np.any(outside),
            "radial_jackknife_within_representation_floor": float(
                np.median(jackknife_rms)
            )
            <= representation_floor,
        }
        output[representation] = {
            "paired_direct_minus_coordinate_bootstrap_16_50_84": paired,
            "outside_bound_fraction": float(np.mean(outside)),
            "median_radial_jackknife_profile_change_dex": float(
                np.median(jackknife_rms)
            ),
            "representation_median_profile_rms_dex": representation_floor,
            "numeric_gates": gates,
            "all_numeric_gates_pass": bool(all(gates.values())),
        }
    return output


def comparison_figure(
    sample: dict[str, np.ndarray],
    direct_relations: dict[str, dict],
    baseline_relations: dict[str, dict],
    prefix: str,
) -> None:
    """Show where direct training changes the held-out radial relation."""
    truth = sample["truth_log"]
    halo_mass = sample["features"][:, 0]
    high_mass = np.array_split(np.argsort(halo_mass), 3)[-1]
    colors = {"coordinate": OKABE_ITO[1], "direct": OKABE_ITO[2]}
    figure, axes = plt.subplots(2, 3, figsize=(14.5, 7.8), sharex="col")
    for column, representation in enumerate(stage3.REPRESENTATIONS):
        for label, relation in (
            ("coordinate", baseline_relations[representation]),
            ("direct", direct_relations[representation]),
        ):
            error = relation["mean_log"] - truth
            pinned = 100.0 * (10.0 ** (error - error[:, -1:]) - 1.0)
            axes[0, column].plot(
                stage3.stage_1.RADII,
                np.median(pinned, axis=0),
                color=colors[label],
                label=f"{label}-trained mean",
            )
            axes[1, column].plot(
                stage3.stage_1.RADII,
                np.median(pinned[high_mass], axis=0),
                color=colors[label],
            )
        axes[0, column].set_title(stage3.REPRESENTATION_LABELS[representation])
        axes[0, column].legend(frameon=False, fontsize=8)
    for axis in axes[:, :2].ravel():
        axis.axhline(0.0, color="0.5", linewidth=0.8)
        axis.axvspan(5.0, 30.0, color="0.85", alpha=0.35)
        axis.set_xscale("log")
    axes[0, 0].set_ylabel("All-galaxy median pinned residual (%)")
    axes[1, 0].set_ylabel("Highest-halo-mass tercile median residual (%)")
    for axis in axes[1, :2]:
        axis.set_xlabel("Radius (kpc)")

    x = np.arange(2)
    for row, representation in enumerate(stage3.REPRESENTATIONS):
        baseline_rms = np.sqrt(
            np.mean((baseline_relations[representation]["mean_log"] - truth) ** 2, axis=1)
        )
        direct_rms = np.sqrt(
            np.mean((direct_relations[representation]["mean_log"] - truth) ** 2, axis=1)
        )
        values = [np.median(baseline_rms), np.median(direct_rms)]
        rng = np.random.default_rng(SEED + row)
        intervals = []
        for metric in (baseline_rms, direct_rms):
            samples = np.median(
                metric[rng.integers(0, len(metric), (1000, len(metric)))], axis=1
            )
            intervals.append(np.quantile(samples, (0.16, 0.84)))
        errors = np.asarray(
            [[value - interval[0], interval[1] - value] for value, interval in zip(values, intervals)]
        ).T
        axes[row, 2].bar(x, values, color=[colors["coordinate"], colors["direct"]])
        axes[row, 2].errorbar(x, values, yerr=errors, fmt="none", color="black", capsize=3)
        axes[row, 2].set_xticks(x, ("coordinate", "direct"))
        axes[row, 2].set_ylabel("Median held-out CoG RMS (dex)")
        axes[row, 2].set_title(stage3.REPRESENTATION_LABELS[representation])
    for panel, axis in zip("ABCDEF", axes.ravel()):
        axis.text(-0.14, 1.04, panel, transform=axis.transAxes, fontweight="bold")
    figure.suptitle(
        f"Exp56 Stage 3b — coordinate bridge versus direct decoded-CoG fit; N={len(truth)}"
    )
    figure.tight_layout()
    save_fig(figure, FIGURE_DIRECTORY / f"{prefix}exp56_stage3b_comparison")
    plt.close(figure)


def cases_figure(
    sample: dict[str, np.ndarray],
    direct_relations: dict[str, dict],
    baseline_relations: dict[str, dict],
    representation: str,
    prefix: str,
) -> None:
    """Display direct held-out profiles for best, typical, and worst galaxies."""
    truth = sample["truth_log"]
    direct = direct_relations[representation]["mean_log"]
    baseline = baseline_relations[representation]["mean_log"]
    order = np.argsort(np.sqrt(np.mean((direct - truth) ** 2, axis=1)))
    galaxies = order[[0, len(order) // 2, -1]]
    figure, axes = plt.subplots(2, 3, figsize=(13.8, 7.0))
    for column, (galaxy, label) in enumerate(zip(galaxies, ("best", "typical", "worst"))):
        truth_shape = truth[galaxy] - truth[galaxy, -1]
        baseline_shape = baseline[galaxy] - baseline[galaxy, -1]
        direct_shape = direct[galaxy] - direct[galaxy, -1]
        axes[0, column].plot(stage3.stage_1.RADII, truth_shape, "o-", color="black", ms=3, label="TNG")
        axes[0, column].plot(stage3.stage_1.RADII, baseline_shape, color=OKABE_ITO[1], label="coordinate")
        axes[0, column].plot(stage3.stage_1.RADII, direct_shape, color=OKABE_ITO[2], label="direct CoG")
        axes[1, column].plot(
            stage3.stage_1.RADII,
            100.0 * (10.0 ** (baseline_shape - truth_shape) - 1.0),
            color=OKABE_ITO[1],
        )
        axes[1, column].plot(
            stage3.stage_1.RADII,
            100.0 * (10.0 ** (direct_shape - truth_shape) - 1.0),
            color=OKABE_ITO[2],
        )
        axes[0, column].set_title(
            rf"{label}: $\log M_{{\rm peak}}={sample['features'][galaxy, 0]:.2f}$"
        )
        for axis in axes[:, column]:
            axis.set_xscale("log")
        axes[1, column].axhline(0.0, color="0.5", linewidth=0.8)
        axes[1, column].axvspan(5.0, 30.0, color="0.85", alpha=0.35)
        axes[1, column].set_xlabel("Radius (kpc)")
    axes[0, 0].set_ylabel(r"$\log_{10}[M_*(<R)/M_{*,148}]$")
    axes[1, 0].set_ylabel("Amplitude-pinned residual (%)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    for panel, axis in zip("ABCDEF", axes.ravel()):
        axis.text(-0.13, 1.04, panel, transform=axis.transAxes, fontweight="bold")
    figure.suptitle(
        f"Exp56 Stage 3b — direct held-out galaxies; {stage3.REPRESENTATION_LABELS[representation]}"
    )
    figure.tight_layout()
    save_fig(figure, FIGURE_DIRECTORY / f"{prefix}exp56_stage3b_{representation}_cases")
    plt.close(figure)


def generate_standard_qa(
    sample: dict[str, np.ndarray],
    relations: dict[str, dict],
    prefix: str,
) -> dict[str, dict]:
    output = {}
    for representation in stage3.REPRESENTATIONS:
        truth_log = sample["truth_log"]
        mean_log = relations[representation]["mean_log"]
        draw_log = relations[representation]["draw_log"]
        halo_mass = sample["features"][:, 0]
        model_name = f"exp56_stage3b_{prefix}{representation}"
        figure_directory = FIGURE_DIRECTORY / f"{prefix}{representation}_standard_qa"
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
        stage3.density_by_halo_mass_figure(
            mean_log,
            truth_log,
            halo_mass,
            figdir=figure_directory,
            model_name=model_name,
            model_label=f"direct {stage3.REPRESENTATION_LABELS[representation]}",
            experiment_label="exp56 Stage 3b",
        )
        stage3.single_epoch_planes_figure(
            qa_result,
            figdir=figure_directory,
            model_name=model_name,
            model_label=f"direct {stage3.REPRESENTATION_LABELS[representation]}",
            experiment_label="exp56 Stage 3b",
        )
        stage3.single_epoch_size_figure(
            mean_log,
            truth_log,
            qa_result,
            figdir=figure_directory,
            model_name=model_name,
            model_label=f"direct {stage3.REPRESENTATION_LABELS[representation]}",
            experiment_label="exp56 Stage 3b",
        )
        output[representation] = stage3.standard_summary(qa_result)
    return output


def mechanical_checks() -> dict:
    """Run the three predeclared decoder and optimizer checks."""
    rng = np.random.default_rng(SEED)
    parameters = np.column_stack(
        [
            rng.uniform(10.8, 12.2, 12),
            rng.uniform(-1.5, 0.2, 12),
            rng.uniform(0.25, 0.65, 12),
            rng.uniform(1.2, 3.8, 12),
        ]
    )
    gamma = rng.uniform(1.25, 1.4, len(parameters))
    decoder_error = float(
        np.max(np.abs(vectorized_decode(parameters, gamma) - stage3.decode_profiles(parameters, gamma)))
    )

    features = rng.normal(size=(9, 3))
    matrix = np.column_stack([np.ones(len(features)), features])
    coefficients = np.asarray(
        [
            [11.5, 0.08, -0.03, 0.02],
            [-0.6, 0.05, 0.03, -0.02],
            [0.42, 0.02, -0.01, 0.01],
            [2.3, -0.04, 0.03, 0.02],
        ]
    )
    gamma_jacobian = np.linspace(1.4, 1.25, len(features))
    truth = vectorized_decode(matrix @ coefficients.T, gamma_jacobian)
    objective = DecodedCogLeastSquares(matrix, truth, gamma_jacobian)
    chained = objective.jacobian(coefficients.ravel())
    independent = np.empty_like(chained)
    step = 1e-6
    for column in range(coefficients.size):
        offset = np.zeros(coefficients.size)
        offset[column] = step
        independent[:, column] = (
            objective.residual(coefficients.ravel() + offset)
            - objective.residual(coefficients.ravel() - offset)
        ) / (2.0 * step)
    jacobian_error = float(
        np.linalg.norm(chained - independent) / np.linalg.norm(independent)
    )

    synthetic_features = rng.normal(size=(36, 3))
    synthetic_matrix = np.column_stack(
        [np.ones(len(synthetic_features)), synthetic_features]
    )
    synthetic_gamma = np.linspace(1.4, 1.25, len(synthetic_features))
    synthetic_truth = vectorized_decode(
        synthetic_matrix @ coefficients.T, synthetic_gamma
    )
    start = coefficients + rng.normal(scale=0.01, size=coefficients.shape)
    synthetic_fit = fit_decoded_coefficients(
        synthetic_matrix, synthetic_truth, synthetic_gamma, start
    )
    synthetic_prediction = vectorized_decode(
        synthetic_matrix @ synthetic_fit.coefficients.T, synthetic_gamma
    )
    synthetic_error = float(np.max(np.abs(synthetic_prediction - synthetic_truth)))
    passed = bool(
        decoder_error < 1e-10
        and jacobian_error < 1e-5
        and synthetic_error < 1e-6
        and synthetic_fit.success
    )
    return {
        "vectorized_decoder_maximum_error_dex": decoder_error,
        "coefficient_jacobian_relative_error": jacobian_error,
        "synthetic_maximum_profile_error_dex": synthetic_error,
        "passed": passed,
    }


def save_predictions(
    sample: dict[str, np.ndarray], relations: dict[str, dict], prefix: str
) -> None:
    arrays = {
        "indices": sample["indices"],
        "features": sample["features"],
        "truth_log": sample["truth_log"],
    }
    for representation in stage3.REPRESENTATIONS:
        relation = relations[representation]
        for name in (
            "mean_parameters",
            "draw_parameters",
            "fitted_parameters",
            "mean_log",
            "draw_log",
            "representation_log",
            "jackknife_mean_log",
            "gamma",
            "fold_id",
        ):
            arrays[f"{representation}_{name}"] = relation[name]
        for control in stage3.CONTROL_NAMES:
            arrays[f"{representation}_{control}_mean_log"] = relation[
                "control_predictions"
            ][control]
    np.savez_compressed(OUTPUT_DIRECTORY / f"{prefix}predictions.npz", **arrays)


def run_experiment(n_max: int, n_draw: int, n_population_draw: int) -> dict:
    started = time.perf_counter()
    demo = bool(n_max)
    prefix = "demo_" if demo else ""
    checks = mechanical_checks()
    if not checks["passed"]:
        raise RuntimeError(f"Stage 3b mechanical checks failed: {checks}")
    sample = stage3.stage_1.load_sample(n_max)
    if demo and len(sample["indices"]) != N_DEMO_GALAXY:
        raise RuntimeError("Stage 3b demo did not load exactly 90 galaxies")
    direct_relations = {
        representation: cross_validated_relation(
            sample, representation, n_draw, demo
        )
        for representation in stage3.REPRESENTATIONS
    }
    direct_summaries = {
        representation: stage3.summarize_model(
            sample, direct_relations[representation], n_population_draw
        )
        for representation in stage3.REPRESENTATIONS
    }
    baseline_relations, baseline_result = load_baseline(
        stage3.stage_1.load_sample(0) if demo else sample
    )
    if demo:
        full_indices = stage3.stage_1.load_sample(0)["indices"]
        baseline_row = {int(index): row for row, index in enumerate(full_indices)}
        rows = np.asarray([baseline_row[int(index)] for index in sample["indices"]])
        baseline_relations = {
            representation: {
                name: values[rows] if name != "draw_log" else values[:, rows]
                for name, values in relation.items()
            }
            for representation, relation in baseline_relations.items()
        }
        baseline_result = {
            "representations": {
                representation: stage3.summarize_model(
                    sample,
                    {
                        **baseline_relations[representation],
                        "fitted_parameters": direct_relations[representation][
                            "fitted_parameters"
                        ],
                        "control_predictions": direct_relations[representation][
                            "control_predictions"
                        ],
                    },
                    1,
                )
                for representation in stage3.REPRESENTATIONS
            }
        }
    comparisons = compare_with_baseline(
        sample,
        direct_relations,
        direct_summaries,
        baseline_relations,
        baseline_result,
    )
    representation_comparison = stage3.compare_representations(
        sample, direct_relations, direct_summaries
    )

    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    comparison_figure(sample, direct_relations, baseline_relations, prefix)
    for representation in stage3.REPRESENTATIONS:
        cases_figure(
            sample,
            direct_relations,
            baseline_relations,
            representation,
            prefix,
        )
    standard = generate_standard_qa(sample, direct_relations, prefix)
    save_predictions(sample, direct_relations, prefix)
    elapsed_seconds = time.perf_counter() - started
    result = {
        "status": "stage3b_validation" if demo else "stage3b_full",
        "n_galaxy": len(sample["indices"]),
        "n_draw": n_draw,
        "n_population_draw": n_population_draw,
        "mechanical_checks": checks,
        "direct_representations": direct_summaries,
        "direct_minus_coordinate_baseline": comparisons,
        "direct_representation_comparison": representation_comparison,
        "standard_qa": standard,
        "fold_metadata": {
            representation: direct_relations[representation]["metadata"]
            for representation in stage3.REPRESENTATIONS
        },
        "elapsed_seconds": elapsed_seconds,
    }
    if demo:
        result["complete_path_passed"] = bool(
            elapsed_seconds < DEMO_TIME_LIMIT_SECONDS
            and all(
                np.isfinite(direct_relations[name]["mean_log"]).all()
                and np.isfinite(direct_relations[name]["draw_log"]).all()
                and len(direct_relations[name]["metadata"]) == 5
                for name in stage3.REPRESENTATIONS
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
            "objective": "unweighted decoded log10 CoG residual at 24 radii",
            "representations": stage3.REPRESENTATIONS,
        },
    )
    return result


def demo() -> None:
    result = run_experiment(N_DEMO_GALAXY, N_DEMO_DRAW, 1)
    if not result["complete_path_passed"]:
        raise AssertionError(
            "Exp56 Stage 3b complete demo failed its sub-minute gate: "
            f"{result['elapsed_seconds']:.2f} s"
        )
    print(
        f"Exp56 Stage 3b complete 90-galaxy path passed in "
        f"{result['elapsed_seconds']:.2f} s",
        flush=True,
    )


def full_run() -> None:
    demo_path = OUTPUT_DIRECTORY / "demo_results.json"
    if not demo_path.exists():
        raise RuntimeError("run the Stage 3b demo before the full calculation")
    demo_result = json.loads(demo_path.read_text())
    if not demo_result.get("complete_path_passed", False):
        raise RuntimeError("the saved Stage 3b complete-path validation did not pass")
    result = run_experiment(0, N_FULL_DRAW, N_POPULATION_DRAW)
    print(
        f"Exp56 Stage 3b full relation complete: {result['n_galaxy']} galaxies "
        f"in {result['elapsed_seconds']:.1f} s",
        flush=True,
    )


def regenerate_figures() -> None:
    prediction_path = OUTPUT_DIRECTORY / "predictions.npz"
    result_path = OUTPUT_DIRECTORY / "results.json"
    if not prediction_path.exists() or not result_path.exists():
        raise RuntimeError("run the full Stage 3b calculation before figures")
    saved = np.load(prediction_path)
    sample = {
        "indices": saved["indices"],
        "features": saved["features"],
        "truth_log": saved["truth_log"],
    }
    relations = {}
    for representation in stage3.REPRESENTATIONS:
        relation = {
            name: saved[f"{representation}_{name}"]
            for name in (
                "mean_parameters",
                "draw_parameters",
                "fitted_parameters",
                "mean_log",
                "draw_log",
                "representation_log",
                "jackknife_mean_log",
                "gamma",
                "fold_id",
            )
        }
        relation["control_predictions"] = {
            control: saved[f"{representation}_{control}_mean_log"]
            for control in stage3.CONTROL_NAMES
        }
        relations[representation] = relation
    baseline_relations, _ = load_baseline(sample)
    comparison_figure(sample, relations, baseline_relations, "")
    for representation in stage3.REPRESENTATIONS:
        cases_figure(sample, relations, baseline_relations, representation, "")
    generate_standard_qa(sample, relations, "")
    print("Exp56 Stage 3b figures regenerated from saved predictions", flush=True)


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if command == "checks":
        check_result = mechanical_checks()
        print(json.dumps(check_result, indent=2))
        if not check_result["passed"]:
            raise SystemExit(1)
    elif command == "demo":
        demo()
    elif command == "run":
        full_run()
    elif command == "figures":
        regenerate_figures()
    else:
        raise SystemExit(f"unknown command {command!r}; use checks, demo, run, or figures")
