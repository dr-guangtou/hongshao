"""exp55 Stages 1--4: diagnose and refine the analytic halo map.

The fixed profile family remains compact Sersic n=1 plus extended cored
power-law (Moffat gamma=1.25). This script tests the halo-to-parameter mean and
the conditional residual distribution without changing that representation.

Commands
--------
demo
    Run mechanical checks and the candidate means on 90 stratified galaxies.
run
    Execute all four stages on the complete redshift-0.4 sample.
"""

from __future__ import annotations

import json
import os
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from hongshao import qa  # noqa: E402
from hongshao.emulator import _fit_logvar  # noqa: E402
from hongshao.plotting import OKABE_ITO, save_fig, set_style  # noqa: E402
from hongshao.profile_emulator import density_from_cog  # noqa: E402
from hongshao.provenance import write_manifest  # noqa: E402
from halo_map import (  # noqa: E402
    N_FOLD,
    RADII,
    decode_profiles,
    empirical_crps,
    folds_of,
    load_sample,
    shuffled_within_mass_bins,
    standard_summary,
)
from qa_figures import (  # noqa: E402
    density_by_halo_mass_figure,
    single_epoch_planes_figure,
    single_epoch_size_figure,
)

set_style()

FIGURE_DIRECTORY = HERE / "figures" / "halo_map_refinement"
OUTPUT_DIRECTORY = HERE / "outputs" / "halo_map_refinement"
STANDARD_QA_DIRECTORY = FIGURE_DIRECTORY / "standard_qa"
N_DRAW = int(os.environ.get("EXP55_REFINE_N_DRAW", 32))
N_MAX = int(os.environ.get("EXP55_REFINE_NMAX", 0))
SEED = 5500

FEATURE_NAMES = ("logmp", "logtc", "early", "late", "c200c")
TARGET_LABELS = (
    r"$\log_{10}M_{*,148}$",
    r"$\mathrm{logit}\,f_{\rm compact}$",
    r"$\log_{10}R_{\rm compact}$",
    r"$\ln(R_{\rm extended}/R_{\rm compact}-1)$",
)

SPARSE_TERMS = (
    (0, 0),
    (1, 3),
    (2, 2),
    (3, 4),
    (3, 3),
    (0, 2),
    (0, 3),
)
MASS_INTERACTION_TERMS = SPARSE_TERMS + ((0, 1), (0, 4))
FULL_DEGREE_TWO_TERMS = tuple(
    (first, second) for first in range(5) for second in range(first, 5)
)
FULL_DEGREE_TWO_MASS_CUBIC_TERMS = FULL_DEGREE_TWO_TERMS + ((0, 0, 0),)

MEAN_CANDIDATES = {
    "sparse_degree_two": SPARSE_TERMS,
    "mass_interactions": MASS_INTERACTION_TERMS,
    "full_degree_two": FULL_DEGREE_TWO_TERMS,
    "full_degree_two_mass_cubic": FULL_DEGREE_TWO_MASS_CUBIC_TERMS,
}


def design_matrix(standardized_features: np.ndarray, terms: tuple[tuple, ...]):
    """Intercept, all linear features, then the declared product terms."""
    columns = [np.ones(len(standardized_features))]
    columns.extend(
        standardized_features[:, index]
        for index in range(standardized_features.shape[1])
    )
    for term in terms:
        value = np.ones(len(standardized_features))
        for index in term:
            value = value * standardized_features[:, index]
        columns.append(value)
    return np.column_stack(columns)


def stable_cholesky(correlation: np.ndarray) -> np.ndarray:
    """Return a Cholesky factor after the minimum eigenvalue repair."""
    correlation = np.asarray(correlation, float)
    correlation = 0.5 * (correlation + correlation.T)
    np.fill_diagonal(correlation, 1.0)
    try:
        return np.linalg.cholesky(correlation)
    except np.linalg.LinAlgError:
        minimum = float(np.linalg.eigvalsh(correlation).min())
        shrinkage = max((1e-8 - minimum) / max(1.0 - minimum, 1e-12), 0.0)
        repaired = (1.0 - shrinkage) * correlation + shrinkage * np.eye(
            len(correlation)
        )
        return np.linalg.cholesky(repaired)


@dataclass
class ConditionalModel:
    """Analytic mean plus heteroscedastic Gaussian coordinate residuals."""

    terms: tuple[tuple, ...]
    feature_mean: np.ndarray
    feature_standard_deviation: np.ndarray
    coefficients: np.ndarray
    log_variance_coefficients: np.ndarray
    residual_correlation: np.ndarray

    def standardized(self, features: np.ndarray) -> np.ndarray:
        return (features - self.feature_mean) / self.feature_standard_deviation

    def predict_mean(self, features: np.ndarray) -> np.ndarray:
        standardized = self.standardized(np.asarray(features, float))
        return design_matrix(standardized, self.terms) @ self.coefficients.T

    def predict_sigma(self, features: np.ndarray) -> np.ndarray:
        standardized = self.standardized(np.asarray(features, float))
        variance_design = np.column_stack([np.ones(len(standardized)), standardized])
        return np.exp(0.5 * (variance_design @ self.log_variance_coefficients.T))

    def sample(self, features: np.ndarray, n_draw: int, seed: int) -> np.ndarray:
        mean = self.predict_mean(features)
        sigma = self.predict_sigma(features)
        normal = np.random.default_rng(seed).standard_normal(
            (n_draw, len(features), mean.shape[1])
        )
        cholesky = stable_cholesky(self.residual_correlation)
        return mean[None] + sigma[None] * (normal @ cholesky.T)


def fit_conditional_model(
    features: np.ndarray,
    targets: np.ndarray,
    terms: tuple[tuple, ...],
) -> ConditionalModel:
    """Fit a declared analytic mean and the incumbent scatter prescription."""
    feature_mean = np.mean(features, axis=0)
    feature_standard_deviation = np.std(features, axis=0)
    feature_standard_deviation = np.where(
        feature_standard_deviation > 0.0, feature_standard_deviation, 1.0
    )
    standardized = (features - feature_mean) / feature_standard_deviation
    mean_design = design_matrix(standardized, terms)
    variance_design = np.column_stack([np.ones(len(standardized)), standardized])
    coefficients = np.empty((targets.shape[1], mean_design.shape[1]))
    log_variance_coefficients = np.empty((targets.shape[1], variance_design.shape[1]))
    residual = np.empty_like(targets)
    sigma = np.empty_like(targets)
    for target_index in range(targets.shape[1]):
        coefficients[target_index], *_ = np.linalg.lstsq(
            mean_design, targets[:, target_index], rcond=None
        )
        residual[:, target_index] = (
            targets[:, target_index] - mean_design @ coefficients[target_index]
        )
        log_variance_coefficients[target_index] = _fit_logvar(
            residual[:, target_index], standardized, ridge=2.0
        )
        sigma[:, target_index] = np.exp(
            0.5 * (variance_design @ log_variance_coefficients[target_index])
        )
    normalized_residual = residual / np.clip(sigma, 1e-12, None)
    residual_correlation = np.corrcoef(normalized_residual.T)
    return ConditionalModel(
        terms,
        feature_mean,
        feature_standard_deviation,
        coefficients,
        log_variance_coefficients,
        residual_correlation,
    )


def decode_draws(draw_parameters: np.ndarray) -> np.ndarray:
    """Decode arrays shaped (draw, galaxy, coordinate) to full log CoGs."""
    n_draw, n_galaxy, n_target = draw_parameters.shape
    return decode_profiles(draw_parameters.reshape(-1, n_target)).reshape(
        n_draw, n_galaxy, -1
    )


def cross_validated_model(
    features: np.ndarray,
    targets: np.ndarray,
    terms: tuple[tuple, ...],
    n_draw: int,
    seed: int,
) -> dict:
    """Five-fold conditional predictions, draws, and fitted fold models."""
    mean_parameters = np.empty_like(targets)
    draw_parameters = np.empty((n_draw, *targets.shape))
    fold_models = []
    fold_records = []
    for fold_number, test in enumerate(folds_of(len(features))):
        train = np.setdiff1d(np.arange(len(features)), test)
        model = fit_conditional_model(features[train], targets[train], terms)
        mean_parameters[test] = model.predict_mean(features[test])
        if n_draw:
            draw_parameters[:, test] = model.sample(
                features[test], n_draw, seed + fold_number
            )
        fold_models.append(model)
        fold_records.append((train, test))
    return {
        "mean_parameters": mean_parameters,
        "mean_log": decode_profiles(mean_parameters),
        "draw_parameters": draw_parameters,
        "draw_log": decode_draws(draw_parameters) if n_draw else None,
        "fold_models": fold_models,
        "fold_records": fold_records,
    }


def equal_number_bins(values: np.ndarray, n_bin: int = 3) -> list[np.ndarray]:
    """Indices for deterministic equal-number bins in one property."""
    ordered = np.argsort(values)
    return [np.asarray(group, int) for group in np.array_split(ordered, n_bin)]


def fractional_profile_residual(
    model_log: np.ndarray, truth_log: np.ndarray, remove_amplitude: bool = False
) -> np.ndarray:
    """Fractional model-minus-truth residual, optionally pinned at Rmax."""
    residual = model_log - truth_log
    if remove_amplitude:
        residual = residual - residual[:, -1:]
    return 10.0**residual - 1.0


def profile_jacobians(raw_parameters: np.ndarray, step: float = 1e-4) -> np.ndarray:
    """Central finite-difference derivative of log CoG with raw coordinates."""
    jacobian = np.empty((len(raw_parameters), len(RADII), raw_parameters.shape[1]))
    for coordinate in range(raw_parameters.shape[1]):
        upper = raw_parameters.copy()
        lower = raw_parameters.copy()
        upper[:, coordinate] += step
        lower[:, coordinate] -= step
        jacobian[:, :, coordinate] = (
            decode_profiles(upper) - decode_profiles(lower)
        ) / (2.0 * step)
    return jacobian


def bootstrap_median_curve(
    values: np.ndarray, seed: int, n_bootstrap: int = 300
) -> tuple[np.ndarray, np.ndarray]:
    """16th--84th percentile interval for a median radial curve."""
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    medians = np.median(values[indices], axis=1)
    return tuple(np.quantile(medians, [0.16, 0.84], axis=0))


def stage_one_diagnostics(sample: dict, baseline: dict) -> dict:
    """Representation/mapping split and Jacobian parameter contributions."""
    truth_log = sample["truth_log"]
    representation_log = sample["representation_log"]
    mean_log = baseline["mean_log"]
    raw_targets = sample["raw_targets"]
    mean_parameters = baseline["mean_parameters"]
    jacobian = profile_jacobians(raw_targets)
    parameter_difference = mean_parameters - raw_targets
    contributions = jacobian * parameter_difference[:, None, :]
    linear_mapping = np.sum(contributions, axis=2)
    exact_mapping = mean_log - representation_log
    closure = exact_mapping - linear_mapping
    bins = equal_number_bins(sample["features"][:, 0])

    summary = {
        "median_mapping_rms_dex": float(
            np.median(np.sqrt(np.mean(exact_mapping**2, axis=1)))
        ),
        "median_representation_rms_dex": float(
            np.median(np.sqrt(np.mean((representation_log - truth_log) ** 2, axis=1)))
        ),
        "median_linearization_closure_rms_dex": float(
            np.median(np.sqrt(np.mean(closure**2, axis=1)))
        ),
        "mass_bins": [],
    }
    for indices in bins:
        shape = fractional_profile_residual(
            mean_log[indices], truth_log[indices], remove_amplitude=True
        )
        summary["mass_bins"].append(
            {
                "n_galaxy": len(indices),
                "logmp_range": [
                    float(np.min(sample["features"][indices, 0])),
                    float(np.max(sample["features"][indices, 0])),
                ],
                "maximum_absolute_median_shape_residual_percent": float(
                    100.0 * np.max(np.abs(np.median(shape, axis=0)))
                ),
                "radius_of_maximum_kpc": float(
                    RADII[np.argmax(np.abs(np.median(shape, axis=0)))]
                ),
            }
        )

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.4), sharex=True)
    parameter_colors = [OKABE_ITO[index] for index in (0, 2, 4, 5)]
    for bin_number, indices in enumerate(bins):
        total = fractional_profile_residual(mean_log[indices], truth_log[indices], True)
        representation = fractional_profile_residual(
            representation_log[indices], truth_log[indices], True
        )
        mapping = fractional_profile_residual(
            mean_log[indices], representation_log[indices], True
        )
        mapping_interval = bootstrap_median_curve(100.0 * mapping, SEED + bin_number)
        axes[0, bin_number].fill_between(
            RADII,
            mapping_interval[0],
            mapping_interval[1],
            color=OKABE_ITO[2],
            alpha=0.18,
            linewidth=0.0,
            label="mapping median 16--84% bootstrap" if bin_number == 0 else None,
        )
        axes[0, bin_number].plot(
            RADII,
            100.0 * np.median(total, axis=0),
            color="0.15",
            linewidth=1.8,
            label="total" if bin_number == 0 else None,
        )
        axes[0, bin_number].plot(
            RADII,
            100.0 * np.median(representation, axis=0),
            color=OKABE_ITO[0],
            linestyle="--",
            label="analytic representation" if bin_number == 0 else None,
        )
        axes[0, bin_number].plot(
            RADII,
            100.0 * np.median(mapping, axis=0),
            color=OKABE_ITO[2],
            linestyle="-.",
            label="halo mapping" if bin_number == 0 else None,
        )
        for coordinate, color in enumerate(parameter_colors):
            contribution = contributions[indices, :, coordinate]
            contribution = contribution - contribution[:, -1:]
            axes[1, bin_number].plot(
                RADII,
                100.0 * np.log(10.0) * np.median(contribution, axis=0),
                color=color,
                label=TARGET_LABELS[coordinate] if bin_number == 0 else None,
            )
        closure_shape = closure[indices] - closure[indices, -1:]
        axes[1, bin_number].plot(
            RADII,
            100.0 * np.log(10.0) * np.median(closure_shape, axis=0),
            color="0.35",
            linestyle=":",
            label="non-linear closure" if bin_number == 0 else None,
        )
        low = np.min(sample["features"][indices, 0])
        high = np.max(sample["features"][indices, 0])
        axes[0, bin_number].set_title(
            rf"$\log M_{{\rm peak}}={low:.2f}$--${high:.2f}$; "
            rf"$n={len(indices)}$"
        )
        for row in range(2):
            axes[row, bin_number].axhline(0.0, color="0.65", linewidth=0.8)
            axes[row, bin_number].set_xscale("log")
            axes[row, bin_number].set_xlabel("Radius (kpc)")
    axes[0, 0].set_ylabel(r"Median shape residual (\%)")
    axes[1, 0].set_ylabel(r"Linearized contribution (\%)")
    axes[0, 0].legend(fontsize=7, frameon=False)
    axes[1, 0].legend(fontsize=7, frameon=False)
    for panel, axis in zip("ABCDEF", axes.ravel()):
        axis.text(-0.14, 1.05, panel, transform=axis.transAxes, fontweight="bold")
    fig.suptitle(
        "exp55 — halo-mass-dependent CoG residuals arise mainly in the halo map"
    )
    fig.tight_layout()
    save_fig(fig, FIGURE_DIRECTORY / "exp55_refinement_decomposition")
    plt.close(fig)
    return summary


def candidate_summary(result: dict, truth_log: np.ndarray, mass: np.ndarray) -> dict:
    """Profile-level metrics used to select a conditional-mean basis."""
    mean_log = result["mean_log"]
    draw_log = result["draw_log"]
    profile_error = mean_log - truth_log
    mean_density, _ = density_from_cog(mean_log, RADII)
    truth_density, _ = density_from_cog(truth_log, RADII)
    bin_residuals = []
    for indices in equal_number_bins(mass):
        shape = fractional_profile_residual(
            mean_log[indices], truth_log[indices], remove_amplitude=True
        )
        bin_residuals.append(100.0 * np.median(shape, axis=0))
    profile_crps = empirical_crps(truth_log, draw_log)
    return {
        "profile_crps_dex": float(np.mean(profile_crps)),
        "median_profile_rms_dex": float(
            np.median(np.sqrt(np.mean(profile_error**2, axis=1)))
        ),
        "median_density_rms_dex": float(
            np.median(np.sqrt(np.mean((mean_density - truth_density) ** 2, axis=1)))
        ),
        "worst_mass_bin_median_shape_residual_percent": float(
            np.max(np.abs(bin_residuals))
        ),
        "mass_bin_median_shape_residual_percent": [
            residual.tolist() for residual in bin_residuals
        ],
        "profile_coverage_68": float(
            np.mean(
                (truth_log >= np.quantile(draw_log, 0.16, axis=0))
                & (truth_log <= np.quantile(draw_log, 0.84, axis=0))
            )
        ),
    }


def select_mean_candidate(summaries: dict[str, dict]) -> tuple[str, dict]:
    """Apply the predeclared profile-level complexity gate."""
    baseline = summaries["sparse_degree_two"]
    passing = []
    for name, summary in summaries.items():
        if name == "sparse_degree_two":
            continue
        crps_gain = (
            100.0
            * (baseline["profile_crps_dex"] - summary["profile_crps_dex"])
            / baseline["profile_crps_dex"]
        )
        residual_gain = (
            baseline["worst_mass_bin_median_shape_residual_percent"]
            - summary["worst_mass_bin_median_shape_residual_percent"]
        )
        density_change = (
            summary["median_density_rms_dex"] - baseline["median_density_rms_dex"]
        )
        summary["crps_improvement_vs_baseline_percent"] = crps_gain
        summary["worst_shape_residual_improvement_percentage_point"] = residual_gain
        summary["density_rms_change_vs_baseline_dex"] = density_change
        if crps_gain >= 0.5 and residual_gain >= 0.5 and density_change <= 0.001:
            passing.append(name)
    if not passing:
        return "sparse_degree_two", {"reason": "no added basis passed all gates"}
    best_crps = min(summaries[name]["profile_crps_dex"] for name in passing)
    within_tolerance = [
        name
        for name in passing
        if 100.0 * (summaries[name]["profile_crps_dex"] - best_crps) / best_crps <= 0.25
    ]
    selected = min(within_tolerance, key=lambda name: len(MEAN_CANDIDATES[name]))
    return selected, {
        "reason": "passed all gates; simplest within 0.25% of best CRPS",
        "passing": passing,
    }


def stage_two_figure(summaries: dict[str, dict], selected_name: str) -> None:
    """Candidate scores and the mass-binned shape residuals."""
    names = list(MEAN_CANDIDATES)
    labels = [
        "sparse\ndegree-2",
        "+ mass\ninteractions",
        "full\ndegree-2",
        "full degree-2\n+ mass cubic",
    ]
    colors = [
        OKABE_ITO[0],
        OKABE_ITO[1],
        OKABE_ITO[2],
        OKABE_ITO[4],
    ]
    fig = plt.figure(figsize=(13.5, 7.4))
    grid = fig.add_gridspec(2, 3, height_ratios=(0.9, 1.2))
    metric_definitions = (
        ("profile_crps_dex", "Mean profile CRPS (dex)"),
        (
            "worst_mass_bin_median_shape_residual_percent",
            r"Worst mass-bin median shape residual (\%)",
        ),
        ("median_density_rms_dex", "Median density RMS (dex)"),
    )
    for column, (metric, label) in enumerate(metric_definitions):
        axis = fig.add_subplot(grid[0, column])
        values = [summaries[name][metric] for name in names]
        bars = axis.bar(np.arange(len(names)), values, color=colors)
        selected_index = names.index(selected_name)
        bars[selected_index].set_edgecolor("black")
        bars[selected_index].set_linewidth(2.0)
        axis.set_xticks(np.arange(len(names)), labels, fontsize=7)
        axis.set_ylabel(label)
        axis.set_ylim(0.0, max(values) * 1.18)
        for position, value in enumerate(values):
            axis.text(
                position,
                value,
                f"{value:.4f}" if value < 1.0 else f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
        axis.text(
            -0.14, 1.05, "ABC"[column], transform=axis.transAxes, fontweight="bold"
        )
    for bin_number in range(3):
        axis = fig.add_subplot(grid[1, bin_number])
        for name, color, label in zip(names, colors, labels):
            residual = np.asarray(
                summaries[name]["mass_bin_median_shape_residual_percent"][bin_number]
            )
            axis.plot(
                RADII,
                residual,
                color=color,
                linewidth=2.0 if name == selected_name else 1.2,
                linestyle="-" if name == selected_name else "--",
                label=label.replace("\n", " ") if bin_number == 0 else None,
            )
        axis.axhline(0.0, color="0.65", linewidth=0.8)
        axis.set_xscale("log")
        axis.set_xlabel("Radius (kpc)")
        axis.set_title(f"Halo-mass tercile {bin_number + 1}")
        axis.text(
            -0.14, 1.05, "DEF"[bin_number], transform=axis.transAxes, fontweight="bold"
        )
    fig.axes[3].set_ylabel(r"Median amplitude-removed residual (\%)")
    fig.axes[3].legend(fontsize=7, frameon=False)
    fig.suptitle(
        "exp55 — held-out analytic conditional-mean candidates; selected outlined"
    )
    fig.tight_layout()
    save_fig(fig, FIGURE_DIRECTORY / "exp55_refinement_mean_candidates")
    plt.close(fig)


def profile_metric_modes(
    raw_parameters: np.ndarray,
    residual_parameters: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Profile-metric residual modes and their variance fractions."""
    jacobian = profile_jacobians(raw_parameters)
    metric = np.mean(np.einsum("nri,nrj->nij", jacobian, jacobian) / len(RADII), axis=0)
    metric_value, metric_vector = np.linalg.eigh(metric)
    metric_value = np.clip(metric_value, 1e-10, None)
    square_root_metric = np.diag(np.sqrt(metric_value)) @ metric_vector.T
    metric_residual = residual_parameters @ square_root_metric.T
    covariance = np.cov(metric_residual.T)
    mode_value, mode_vector = np.linalg.eigh(covariance)
    order = np.argsort(mode_value)[::-1]
    mode_value = mode_value[order]
    mode_vector = mode_vector[:, order]
    transformation = mode_vector.T @ square_root_metric
    mode_residual = residual_parameters @ transformation.T
    for mode in range(mode_residual.shape[1]):
        radial_signature = np.median(
            jacobian
            @ np.linalg.solve(
                transformation,
                np.eye(transformation.shape[0])[:, mode],
            ),
            axis=0,
        )
        if radial_signature[0] < 0.0:
            transformation[mode] *= -1.0
            mode_residual[:, mode] *= -1.0
    variance_fraction = np.var(mode_residual, axis=0)
    variance_fraction /= np.sum(variance_fraction)
    return transformation, mode_residual, variance_fraction, metric


def cross_validated_mode_r2(
    features: np.ndarray,
    raw_targets: np.ndarray,
    transformation: np.ndarray,
    terms: tuple[tuple, ...],
) -> np.ndarray:
    """Held-out R2 of fixed transformed profile coordinates."""
    targets = raw_targets @ transformation.T
    prediction = np.empty_like(targets)
    for test in folds_of(len(features)):
        train = np.setdiff1d(np.arange(len(features)), test)
        model = fit_conditional_model(features[train], targets[train], terms)
        prediction[test] = model.predict_mean(features[test])
    return 1.0 - np.mean((prediction - targets) ** 2, axis=0) / np.var(targets, axis=0)


def stage_three_diagnostics(
    sample: dict, selected_mean: dict, selected_terms: tuple[tuple, ...]
) -> tuple[dict, np.ndarray]:
    """Build and visualize global profile-metric residual coordinates."""
    residual = sample["raw_targets"] - selected_mean["mean_parameters"]
    transformation, mode_residual, variance_fraction, metric = profile_metric_modes(
        sample["raw_targets"], residual
    )
    mode_r2 = cross_validated_mode_r2(
        sample["features"], sample["raw_targets"], transformation, selected_terms
    )
    inverse = np.linalg.inv(transformation)
    reference = np.median(sample["raw_targets"], axis=0)
    signatures = []
    for mode in range(4):
        amplitude = np.std(mode_residual[:, mode])
        parameter_shift = amplitude * inverse[:, mode]
        positive = decode_profiles((reference + parameter_shift)[None])[0]
        negative = decode_profiles((reference - parameter_shift)[None])[0]
        signatures.append(0.5 * (positive - negative))
    signatures = np.asarray(signatures)
    correlations = np.column_stack(
        [
            [
                np.corrcoef(mode_residual[:, mode], sample["features"][:, feature])[
                    0, 1
                ]
                for feature in range(sample["features"].shape[1])
            ]
            for mode in range(4)
        ]
    ).T

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4))
    mode_colors = [OKABE_ITO[index] for index in (0, 2, 4, 5)]
    positions = np.arange(4)
    axes[0].bar(positions - 0.18, variance_fraction, width=0.36, color=mode_colors)
    axes[0].bar(
        positions + 0.18,
        np.clip(mode_r2, -0.2, 1.0),
        width=0.36,
        color="none",
        edgecolor=mode_colors,
        linewidth=1.6,
        hatch="//",
    )
    axes[0].axhline(0.0, color="0.6", linewidth=0.8)
    axes[0].set_xticks(positions, [f"mode {index + 1}" for index in positions])
    axes[0].set_ylabel("Fraction or held-out $R^2$")
    axes[0].set_title("Filled: residual profile variance\nHatched: halo predictability")
    for mode, color in enumerate(mode_colors):
        axes[1].plot(
            RADII,
            signatures[mode],
            color=color,
            label=f"mode {mode + 1}",
        )
    axes[1].axhline(0.0, color="0.6", linewidth=0.8)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Radius (kpc)")
    axes[1].set_ylabel(r"One-$\sigma$ $\Delta\log_{10}M_*(<R)$")
    axes[1].set_title("Decoded radial signature")
    axes[1].legend(frameon=False, fontsize=8)
    correlation_limit = max(0.005, 1.1 * float(np.max(np.abs(correlations))))
    image = axes[2].imshow(
        correlations,
        vmin=-correlation_limit,
        vmax=correlation_limit,
        cmap="PuOr_r",
        aspect="auto",
    )
    axes[2].set_xticks(np.arange(5), FEATURE_NAMES, rotation=30, ha="right")
    axes[2].set_yticks(np.arange(4), [f"mode {index + 1}" for index in range(4)])
    axes[2].set_title("Residual-mode correlation with halo inputs")
    for mode in range(4):
        for feature in range(5):
            axes[2].text(
                feature,
                mode,
                f"{correlations[mode, feature]:+.3f}",
                ha="center",
                va="center",
                fontsize=7,
            )
    fig.colorbar(image, ax=axes[2], label="Pearson correlation")
    for panel, axis in zip("ABC", axes):
        axis.text(-0.14, 1.05, panel, transform=axis.transAxes, fontweight="bold")
    fig.suptitle("exp55 — profile-metric residual modes")
    fig.tight_layout()
    save_fig(fig, FIGURE_DIRECTORY / "exp55_refinement_residual_modes")
    plt.close(fig)

    summary = {
        "profile_variance_fraction": variance_fraction.tolist(),
        "held_out_mode_r2": mode_r2.tolist(),
        "mode_halo_feature_correlation": correlations.tolist(),
        "transformation": transformation.tolist(),
        "profile_metric": metric.tolist(),
    }
    return summary, transformation


def mode_draws_for_fold(
    model: ConditionalModel,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    n_draw: int,
    seed: int,
    correlation_kind: str,
) -> np.ndarray:
    """Draw residual modes with diagonal or mass-binned correlations."""
    train_mean = model.predict_mean(train_features)
    residual = train_targets - train_mean
    transformation, mode_residual, _, _ = profile_metric_modes(train_targets, residual)
    standardized_train = model.standardized(train_features)
    standardized_test = model.standardized(test_features)
    variance_design_train = np.column_stack(
        [np.ones(len(standardized_train)), standardized_train]
    )
    variance_design_test = np.column_stack(
        [np.ones(len(standardized_test)), standardized_test]
    )
    gamma = np.empty((mode_residual.shape[1], variance_design_train.shape[1]))
    train_sigma = np.empty_like(mode_residual)
    test_sigma = np.empty((len(test_features), mode_residual.shape[1]))
    for mode in range(mode_residual.shape[1]):
        gamma[mode] = _fit_logvar(mode_residual[:, mode], standardized_train, ridge=2.0)
        train_sigma[:, mode] = np.exp(0.5 * (variance_design_train @ gamma[mode]))
        test_sigma[:, mode] = np.exp(0.5 * (variance_design_test @ gamma[mode]))
    normalized = mode_residual / np.clip(train_sigma, 1e-12, None)
    rng = np.random.default_rng(seed)
    normal = rng.standard_normal((n_draw, len(test_features), mode_residual.shape[1]))
    if correlation_kind == "diagonal":
        mode_draw = normal * test_sigma[None]
    elif correlation_kind == "mass_binned":
        edges = np.quantile(train_features[:, 0], [1.0 / 3.0, 2.0 / 3.0])
        train_bin = np.digitize(train_features[:, 0], edges)
        test_bin = np.digitize(test_features[:, 0], edges)
        mode_draw = np.empty_like(normal)
        for bin_number in range(3):
            train_members = train_bin == bin_number
            test_members = test_bin == bin_number
            correlation = np.corrcoef(normalized[train_members].T)
            cholesky = stable_cholesky(correlation)
            mode_draw[:, test_members] = (
                normal[:, test_members] @ cholesky.T
            ) * test_sigma[None, test_members]
    else:
        raise ValueError(f"unknown mode correlation {correlation_kind!r}")
    inverse = np.linalg.inv(transformation)
    parameter_residual = mode_draw @ inverse.T
    return model.predict_mean(test_features)[None] + parameter_residual


def nearest_neighbour_draws_for_fold(
    model: ConditionalModel,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    n_draw: int,
    seed: int,
    n_neighbour: int = 128,
    residual_library: np.ndarray | None = None,
) -> np.ndarray:
    """Empirical residual draws from nearby training halos."""
    train_standardized = model.standardized(train_features)
    test_standardized = model.standardized(test_features)
    residual = (
        train_targets - model.predict_mean(train_features)
        if residual_library is None
        else np.asarray(residual_library, float)
    )
    tree = cKDTree(train_standardized)
    _, neighbours = tree.query(
        test_standardized, k=min(n_neighbour, len(train_features))
    )
    if neighbours.ndim == 1:
        neighbours = neighbours[:, None]
    rng = np.random.default_rng(seed)
    choices = rng.integers(0, neighbours.shape[1], size=(n_draw, len(test_features)))
    selected = neighbours[np.arange(len(test_features))[None], choices]
    return model.predict_mean(test_features)[None] + residual[selected]


def cross_fitted_residual_library(
    features: np.ndarray,
    targets: np.ndarray,
    terms: tuple[tuple, ...],
    seed: int,
    n_inner_fold: int = 4,
) -> np.ndarray:
    """Residual library whose members were not used to fit their own means."""
    order = np.random.default_rng(seed).permutation(len(features))
    prediction = np.empty_like(targets)
    for validation in np.array_split(order, n_inner_fold):
        training = np.setdiff1d(np.arange(len(features)), validation)
        inner_model = fit_conditional_model(
            features[training], targets[training], terms
        )
        prediction[validation] = inner_model.predict_mean(features[validation])
    return targets - prediction


def calibrated_neighbour_draws_for_fold(
    model: ConditionalModel,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    train_truth_log: np.ndarray,
    test_features: np.ndarray,
    n_draw: int,
    seed: int,
    terms: tuple[tuple, ...],
) -> tuple[np.ndarray, float, dict[str, float]]:
    """Choose residual inflation on inner held-out profiles, then draw outer."""
    scale_grid = np.asarray([1.00, 1.08, 1.16, 1.24, 1.32])
    order = np.random.default_rng(seed).permutation(len(train_features))
    inner_mean = np.empty_like(train_targets)
    inner_draw = np.empty((n_draw, *train_targets.shape))
    for inner_number, validation in enumerate(np.array_split(order, 4)):
        training = np.setdiff1d(np.arange(len(train_features)), validation)
        inner_model = fit_conditional_model(
            train_features[training], train_targets[training], terms
        )
        inner_mean[validation] = inner_model.predict_mean(train_features[validation])
        inner_draw[:, validation] = nearest_neighbour_draws_for_fold(
            inner_model,
            train_features[training],
            train_targets[training],
            train_features[validation],
            n_draw,
            seed + 10 + inner_number,
        )
    coverage_by_scale = {}
    for scale in scale_grid:
        scaled_log = decode_draws(
            inner_mean[None] + scale * (inner_draw - inner_mean[None])
        )
        coverage_by_scale[f"{scale:.2f}"] = float(
            np.mean(
                (train_truth_log >= np.quantile(scaled_log, 0.16, axis=0))
                & (train_truth_log <= np.quantile(scaled_log, 0.84, axis=0))
            )
        )
    selected_scale = min(
        scale_grid,
        key=lambda scale: (
            abs(coverage_by_scale[f"{scale:.2f}"] - 0.68),
            scale,
        ),
    )
    outer_base = nearest_neighbour_draws_for_fold(
        model,
        train_features,
        train_targets,
        test_features,
        n_draw,
        seed + 100,
    )
    outer_mean = model.predict_mean(test_features)[None]
    calibrated = outer_mean + selected_scale * (outer_base - outer_mean)
    return calibrated, float(selected_scale), coverage_by_scale


def stochastic_candidates(
    features: np.ndarray,
    targets: np.ndarray,
    truth_log: np.ndarray,
    selected_mean: dict,
    selected_terms: tuple[tuple, ...],
    n_draw: int,
) -> tuple[dict[str, np.ndarray], dict]:
    """Generate the four predeclared residual-distribution candidates."""
    candidates = {
        "raw_gaussian": selected_mean["draw_parameters"].copy(),
        "mode_diagonal": np.empty((n_draw, *targets.shape)),
        "mode_mass_binned_correlation": np.empty((n_draw, *targets.shape)),
        "neighbour_resampling": np.empty((n_draw, *targets.shape)),
        "cross_fitted_neighbour_resampling": np.empty((n_draw, *targets.shape)),
        "nested_calibrated_neighbour_resampling": np.empty((n_draw, *targets.shape)),
    }
    metadata = {"nested_calibrated_neighbour_resampling": {"folds": []}}
    for fold_number, ((train, test), model) in enumerate(
        zip(selected_mean["fold_records"], selected_mean["fold_models"])
    ):
        candidates["mode_diagonal"][:, test] = mode_draws_for_fold(
            model,
            features[train],
            targets[train],
            features[test],
            n_draw,
            SEED + 100 + fold_number,
            "diagonal",
        )
        candidates["mode_mass_binned_correlation"][:, test] = mode_draws_for_fold(
            model,
            features[train],
            targets[train],
            features[test],
            n_draw,
            SEED + 200 + fold_number,
            "mass_binned",
        )
        candidates["neighbour_resampling"][:, test] = nearest_neighbour_draws_for_fold(
            model,
            features[train],
            targets[train],
            features[test],
            n_draw,
            SEED + 300 + fold_number,
        )
        cross_fitted_residual = cross_fitted_residual_library(
            features[train],
            targets[train],
            selected_terms,
            SEED + 400 + fold_number,
        )
        candidates["cross_fitted_neighbour_resampling"][:, test] = (
            nearest_neighbour_draws_for_fold(
                model,
                features[train],
                targets[train],
                features[test],
                n_draw,
                SEED + 500 + fold_number,
                residual_library=cross_fitted_residual,
            )
        )
        calibrated, selected_scale, coverage_by_scale = (
            calibrated_neighbour_draws_for_fold(
                model,
                features[train],
                targets[train],
                truth_log[train],
                features[test],
                n_draw,
                SEED + 600 + fold_number,
                selected_terms,
            )
        )
        candidates["nested_calibrated_neighbour_resampling"][:, test] = calibrated
        metadata["nested_calibrated_neighbour_resampling"]["folds"].append(
            {
                "fold": fold_number,
                "selected_scale": selected_scale,
                "inner_coverage_by_scale": coverage_by_scale,
            }
        )
    return candidates, metadata


def average_dicts(entries: list[dict]) -> dict[str, float]:
    """Average matching scalar dictionaries."""
    return {
        key: float(np.mean([entry[key] for entry in entries])) for key in entries[0]
    }


def stochastic_population_summary(
    draw_log: np.ndarray,
    truth_log: np.ndarray,
    n_population_draw: int = 4,
) -> dict:
    """Marginal scores and derived-plane fidelity of population draws."""
    evaluated = min(n_population_draw, len(draw_log))
    truth_cogs = 10.0 ** truth_log[:, None, :]
    _, truth_measures, _, _ = qa.measure_all(truth_cogs, truth_cogs, RADII)
    plane_values = {f"{x}__{y}": [] for x, y in qa.PLANES}
    size_values = {name: [] for name in ("R50", "R80", "R90")}
    for draw in draw_log[:evaluated]:
        draw_cogs = 10.0 ** draw[:, None, :]
        _, draw_measures, _, _ = qa.measure_all(draw_cogs, truth_cogs, RADII)
        _, draw_measures_own_radius, _, _ = qa.measure_all(draw_cogs, draw_cogs, RADII)
        for x_name, y_name in qa.PLANES:
            truth_x = np.log10(np.clip(truth_measures[x_name][:, 0], 1.0, None))
            truth_y = np.log10(np.clip(truth_measures[y_name][:, 0], 1.0, None))
            population_measures = (
                draw_measures_own_radius if x_name.startswith("Re:") else draw_measures
            )
            draw_x = np.log10(np.clip(population_measures[x_name][:, 0], 1.0, None))
            draw_y = np.log10(np.clip(population_measures[y_name][:, 0], 1.0, None))
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
    profile_crps = empirical_crps(truth_log, draw_log)
    return {
        "n_population_draw": evaluated,
        "profile_crps_dex": float(np.mean(profile_crps)),
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
        "planes": {
            name: average_dicts(entries) for name, entries in plane_values.items()
        },
        "sizes": {
            name: average_dicts(entries) for name, entries in size_values.items()
        },
    }


def select_stochastic_candidate(summaries: dict[str, dict]) -> tuple[str, dict]:
    """Apply the predeclared joint-population calibration gate."""
    baseline = summaries["raw_gaussian"]
    re_key = "Re:M(<2Re)__Re:M(2-4Re)"
    fixed_keys = (
        "kpc:M(<30)__kpc:M(30-50)",
        "kpc:M(<30)__kpc:M(50-100)",
    )
    baseline_re = baseline["planes"][re_key]["energy_ratio"]
    baseline_fixed = max(baseline["planes"][key]["energy_ratio"] for key in fixed_keys)
    passing = []
    gate_details = {}
    for name, summary in summaries.items():
        re_energy = summary["planes"][re_key]["energy_ratio"]
        fixed_energy = max(summary["planes"][key]["energy_ratio"] for key in fixed_keys)
        gates = {
            "lower_re_energy": re_energy < baseline_re,
            "fixed_kpc_within_10_percent": fixed_energy <= 1.1 * baseline_fixed,
            "coverage_within_3_percentage_points": abs(
                summary["profile_coverage_68"] - 0.68
            )
            <= 0.03,
            "crps_within_1_percent": summary["profile_crps_dex"]
            <= 1.01 * baseline["profile_crps_dex"],
        }
        gate_details[name] = gates
        if name != "raw_gaussian" and all(gates.values()):
            passing.append(name)
    if not passing:
        return "raw_gaussian", {
            "reason": "no alternative passed all joint-population gates",
            "gates": gate_details,
        }
    selected = min(
        passing,
        key=lambda name: summaries[name]["planes"][re_key]["energy_ratio"],
    )
    return selected, {
        "reason": "passed all gates and minimized size-scaled-plane energy",
        "passing": passing,
        "gates": gate_details,
    }


def stage_four_figure(summaries: dict[str, dict], selected_name: str) -> None:
    """Compare stochastic candidates in coverage, CRPS, and population planes."""
    names = list(summaries)
    labels = [
        "raw Gaussian",
        "mode diagonal",
        "mode mass-correlation",
        "neighbour",
        "cross-fit neighbour",
        "nested calibrated",
    ]
    colors = [OKABE_ITO[index] for index in (0, 1, 2, 4, 5, 6)]
    re_key = "Re:M(<2Re)__Re:M(2-4Re)"
    fixed_one = "kpc:M(<30)__kpc:M(30-50)"
    fixed_two = "kpc:M(<30)__kpc:M(50-100)"
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 5.2), sharey=True)
    values_by_panel = [
        [summaries[name]["profile_crps_dex"] for name in names],
        [summaries[name]["profile_coverage_68"] for name in names],
        [summaries[name]["planes"][re_key]["energy_ratio"] for name in names],
        [
            max(
                summaries[name]["planes"][fixed_one]["energy_ratio"],
                summaries[name]["planes"][fixed_two]["energy_ratio"],
            )
            for name in names
        ],
    ]
    ylabels = (
        "Mean profile CRPS (dex)",
        "Pointwise 68% coverage",
        "Size-scaled plane E / floor",
        "Worst fixed-kpc plane E / floor",
    )
    reference_lines = (None, 0.68, 1.0, 1.0)
    for panel, (axis, values, ylabel, reference) in enumerate(
        zip(axes, values_by_panel, ylabels, reference_lines)
    ):
        positions = np.arange(len(names))
        bars = axis.barh(positions, values, color=colors)
        selected_index = names.index(selected_name)
        bars[selected_index].set_edgecolor("black")
        bars[selected_index].set_linewidth(2.0)
        axis.set_yticks(positions, labels, fontsize=7)
        axis.set_xlabel(ylabel)
        axis.set_xlim(0.0, max(values) * 1.25)
        axis.invert_yaxis()
        if reference is not None:
            axis.axvline(reference, color="0.25", linestyle="--", linewidth=0.9)
        for position, value in enumerate(values):
            axis.text(
                value,
                position,
                f"{value:.3f}",
                ha="left",
                va="center",
                fontsize=7,
            )
        axis.text(
            -0.14, 1.05, "ABCD"[panel], transform=axis.transAxes, fontweight="bold"
        )
    fig.suptitle(
        "exp55 — stochastic population calibration; selected candidate outlined"
    )
    fig.tight_layout()
    save_fig(fig, FIGURE_DIRECTORY / "exp55_refinement_stochastic_candidates")
    plt.close(fig)


def population_draw_figure(draw_log: np.ndarray, truth_log: np.ndarray) -> None:
    """Direct population-plane comparison for one selected stochastic draw."""
    draw_cogs = 10.0 ** draw_log[0, :, None, :]
    truth_cogs = 10.0 ** truth_log[:, None, :]
    truth_measures, draw_fixed, _, _ = qa.measure_all(draw_cogs, truth_cogs, RADII)
    _, draw_own_radius, _, _ = qa.measure_all(draw_cogs, draw_cogs, RADII)
    plane_labels = {
        "kpc:M(<30)": r"$\log_{10}M_*(<30\,\mathrm{kpc})$",
        "kpc:M(30-50)": r"$\log_{10}M_*(30$--$50\,\mathrm{kpc})$",
        "kpc:M(50-100)": r"$\log_{10}M_*(50$--$100\,\mathrm{kpc})$",
        "Re:M(<2Re)": r"$\log_{10}M_*(<2R_e)$",
        "Re:M(2-4Re)": r"$\log_{10}M_*(2$--$4R_e)$",
    }
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.4))
    for column, (x_name, y_name) in enumerate(qa.PLANES):
        draw_measures = draw_own_radius if x_name.startswith("Re:") else draw_fixed
        truth_x = np.log10(np.clip(truth_measures[x_name][:, 0], 1.0, None))
        truth_y = np.log10(np.clip(truth_measures[y_name][:, 0], 1.0, None))
        draw_x = np.log10(np.clip(draw_measures[x_name][:, 0], 1.0, None))
        draw_y = np.log10(np.clip(draw_measures[y_name][:, 0], 1.0, None))
        axes[0, column].scatter(
            truth_x,
            truth_y,
            s=5,
            color="0.25",
            alpha=0.18,
            edgecolors="none",
            label="TNG" if column == 0 else None,
        )
        axes[0, column].scatter(
            draw_x,
            draw_y,
            s=6,
            facecolors="none",
            edgecolors=OKABE_ITO[6],
            alpha=0.24,
            linewidths=0.45,
            label="one generated population" if column == 0 else None,
        )
        truth_statistics = qa.plane_stats(truth_x, truth_y)
        draw_statistics = qa.plane_stats(draw_x, draw_y)
        axes[0, column].set_title(
            "slope/scatter: "
            f"{truth_statistics['slope']:.2f}/{truth_statistics['scatter']:.3f}"
            r" $\rightarrow$ "
            f"{draw_statistics['slope']:.2f}/{draw_statistics['scatter']:.3f} dex"
        )
        axes[0, column].set_xlabel(plane_labels[x_name])
        axes[0, column].set_ylabel(plane_labels[y_name])

    truth_mass = truth_log[:, -1]
    draw_mass = draw_log[0, :, -1]
    for column, fraction in enumerate((0.5, 0.8, 0.9)):
        truth_radius = np.asarray(
            [qa.enclosed_radius(cog, RADII, fraction) for cog in truth_cogs[:, 0]]
        )
        draw_radius = np.asarray(
            [qa.enclosed_radius(cog, RADII, fraction) for cog in draw_cogs[:, 0]]
        )
        truth_log_radius = np.log10(truth_radius)
        draw_log_radius = np.log10(draw_radius)
        axes[1, column].scatter(
            truth_mass,
            truth_log_radius,
            s=5,
            color="0.25",
            alpha=0.18,
            edgecolors="none",
        )
        axes[1, column].scatter(
            draw_mass,
            draw_log_radius,
            s=6,
            facecolors="none",
            edgecolors=OKABE_ITO[6],
            alpha=0.24,
            linewidths=0.45,
        )
        truth_statistics = qa.plane_stats(truth_mass, truth_log_radius)
        draw_statistics = qa.plane_stats(draw_mass, draw_log_radius)
        axes[1, column].set_title(
            f"R{int(100 * fraction)} slope/scatter: "
            f"{truth_statistics['slope']:.2f}/{truth_statistics['scatter']:.3f}"
            r" $\rightarrow$ "
            f"{draw_statistics['slope']:.2f}/{draw_statistics['scatter']:.3f} dex"
        )
        axes[1, column].set_xlabel(r"$\log_{10}M_*(<148.2\,\mathrm{kpc})$")
        axes[1, column].set_ylabel(
            rf"$\log_{{10}}R_{{{int(100 * fraction)}}}\;[\mathrm{{kpc}}]$"
        )
    axes[0, 0].legend(frameon=False, fontsize=8)
    for panel, axis in zip("ABCDEF", axes.ravel()):
        axis.text(-0.14, 1.05, panel, transform=axis.transAxes, fontweight="bold")
    fig.suptitle(
        "exp55 — one nested-calibrated generated population versus TNG at z=0.4"
    )
    fig.tight_layout()
    save_fig(fig, FIGURE_DIRECTORY / "exp55_refinement_population_draws")
    plt.close(fig)


def control_summaries(
    features: np.ndarray,
    targets: np.ndarray,
    truth_log: np.ndarray,
    terms: tuple[tuple, ...],
    n_draw: int,
) -> dict:
    """Selected-basis final-mass and mass-conditioned shuffle controls."""
    definitions = {
        "final_mass_only": (features[:, :1], tuple()),
        "mah_shape_shuffled": (
            shuffled_within_mass_bins(features, (1, 2, 3), seed=5551),
            terms,
        ),
        "concentration_shuffled": (
            shuffled_within_mass_bins(features, (4,), seed=5552),
            terms,
        ),
        "complete": (features, terms),
    }
    summaries = {}
    for name, (model_features, model_terms) in definitions.items():
        result = cross_validated_model(
            model_features,
            targets,
            model_terms,
            n_draw,
            SEED + 600,
        )
        summaries[name] = candidate_summary(result, truth_log, features[:, 0])
    return summaries


def json_safe(value):
    """Convert nested NumPy values for the experiment result file."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def run_experiment(n_max: int = N_MAX, n_draw: int = N_DRAW) -> dict:
    """Execute all four stages and write regenerable outputs and figures."""
    started = time.perf_counter()
    sample = load_sample(n_max)
    features = sample["features"]
    targets = sample["raw_targets"]
    truth_log = sample["truth_log"]
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    print(
        f"exp55 refinement: n={len(features)}, folds={N_FOLD}, draws={n_draw}",
        flush=True,
    )

    candidate_results = {}
    candidate_summaries = {}
    for name, terms in MEAN_CANDIDATES.items():
        print(f"  mean candidate {name}: {6 + len(terms)} coefficients", flush=True)
        result = cross_validated_model(
            features,
            targets,
            terms,
            n_draw,
            SEED,
        )
        candidate_results[name] = result
        candidate_summaries[name] = candidate_summary(result, truth_log, features[:, 0])
    selected_mean_name, mean_selection = select_mean_candidate(candidate_summaries)
    selected_mean = candidate_results[selected_mean_name]
    selected_terms = MEAN_CANDIDATES[selected_mean_name]
    print(f"  selected mean: {selected_mean_name}", flush=True)

    stage_one = stage_one_diagnostics(sample, candidate_results["sparse_degree_two"])
    stage_two_figure(candidate_summaries, selected_mean_name)
    stage_three, transformation = stage_three_diagnostics(
        sample, selected_mean, selected_terms
    )

    draw_parameters, stochastic_metadata = stochastic_candidates(
        features,
        targets,
        truth_log,
        selected_mean,
        selected_terms,
        n_draw,
    )
    draw_logs = {
        name: decode_draws(parameters) for name, parameters in draw_parameters.items()
    }
    stochastic_summaries = {}
    for name, draws in draw_logs.items():
        print(f"  stochastic candidate {name}", flush=True)
        stochastic_summaries[name] = stochastic_population_summary(draws, truth_log)
    selected_stochastic_name, stochastic_selection = select_stochastic_candidate(
        stochastic_summaries
    )
    print(f"  selected stochastic model: {selected_stochastic_name}", flush=True)
    stage_four_figure(stochastic_summaries, selected_stochastic_name)

    controls = control_summaries(
        features,
        targets,
        truth_log,
        selected_terms,
        n_draw,
    )

    selected_draw_log = draw_logs[selected_stochastic_name]
    population_draw_figure(selected_draw_log, truth_log)
    STANDARD_QA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    qa_result = qa.evaluate(
        10.0 ** selected_mean["mean_log"][:, None, :],
        10.0 ** truth_log[:, None, :],
        RADII,
        [0.4],
        name="exp55_refined_halo_map",
        figdir=STANDARD_QA_DIRECTORY,
        bin_by=features[:, 0],
        bin_label=r"DiffMAH $\log M_{\rm peak}$",
        draw_cogs=10.0 ** selected_draw_log[:, :, None, :],
    )
    standard = standard_summary(qa_result)
    density_by_halo_mass_figure(
        selected_mean["mean_log"],
        truth_log,
        features[:, 0],
        figdir=STANDARD_QA_DIRECTORY,
        model_name="exp55_refined_halo_map",
        model_label="refined analytic halo map",
    )
    single_epoch_planes_figure(
        qa_result,
        figdir=STANDARD_QA_DIRECTORY,
        model_name="exp55_refined_halo_map",
        model_label="refined analytic halo-map mean",
    )
    single_epoch_size_figure(
        selected_mean["mean_log"],
        truth_log,
        qa_result,
        figdir=STANDARD_QA_DIRECTORY,
        model_name="exp55_refined_halo_map",
        model_label="refined analytic halo-map mean",
    )

    np.savez_compressed(
        OUTPUT_DIRECTORY / "predictions.npz",
        indices=sample["indices"],
        features=features,
        truth_log=truth_log,
        representation_log=sample["representation_log"],
        raw_targets=targets,
        selected_mean_name=selected_mean_name,
        selected_mean_parameters=selected_mean["mean_parameters"],
        selected_mean_log=selected_mean["mean_log"],
        selected_stochastic_name=selected_stochastic_name,
        selected_draw_parameters=draw_parameters[selected_stochastic_name],
        selected_draw_log=selected_draw_log,
        residual_mode_transformation=transformation,
    )
    payload = {
        "n_galaxy": len(features),
        "n_draw": n_draw,
        "stage_one": stage_one,
        "mean_candidates": candidate_summaries,
        "selected_mean": selected_mean_name,
        "mean_selection": mean_selection,
        "residual_modes": stage_three,
        "stochastic_candidates": stochastic_summaries,
        "stochastic_metadata": stochastic_metadata,
        "selected_stochastic": selected_stochastic_name,
        "stochastic_selection": stochastic_selection,
        "selected_mean_controls": controls,
        "standard_qa": standard,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (OUTPUT_DIRECTORY / "results.json").write_text(
        json.dumps(json_safe(payload), indent=2)
    )
    write_manifest(
        OUTPUT_DIRECTORY,
        {
            "experiment": "exp55_halo_map_refinement",
            "n_galaxy": len(features),
            "n_draw": n_draw,
            "n_fold": N_FOLD,
            "selected_mean": selected_mean_name,
            "selected_stochastic": selected_stochastic_name,
        },
    )
    print(json.dumps(json_safe(payload), indent=2), flush=True)
    return payload


def demo() -> None:
    """Sub-minute validation of transformations, candidates, and draws."""
    started = time.perf_counter()
    sample = load_sample(90)
    for name, terms in MEAN_CANDIDATES.items():
        result = cross_validated_model(
            sample["features"],
            sample["raw_targets"],
            terms,
            n_draw=4,
            seed=SEED,
        )
        assert np.isfinite(result["mean_log"]).all(), name
        assert np.isfinite(result["draw_log"]).all(), name
        assert np.all(np.diff(result["mean_log"], axis=1) >= -1e-10), name
    jacobian = profile_jacobians(sample["raw_targets"][:5])
    assert jacobian.shape == (5, len(RADII), 4)
    print(
        f"exp55 refinement demo OK: four means, 90 galaxies, four draws in "
        f"{time.perf_counter() - started:.2f} s"
    )


def regenerate_refinement_figures() -> None:
    """Rebuild the four refinement figures from saved metrics and current data."""
    payload = json.loads((OUTPUT_DIRECTORY / "results.json").read_text())
    sample = load_sample()
    if payload["n_galaxy"] != len(sample["features"]):
        raise ValueError("saved refinement results are not the full-sample run")
    selected_name = payload["selected_mean"]
    selected = cross_validated_model(
        sample["features"],
        sample["raw_targets"],
        MEAN_CANDIDATES[selected_name],
        n_draw=0,
        seed=SEED,
    )
    stage_one_diagnostics(sample, selected)
    stage_two_figure(payload["mean_candidates"], selected_name)
    stage_three_diagnostics(sample, selected, MEAN_CANDIDATES[selected_name])
    stage_four_figure(payload["stochastic_candidates"], payload["selected_stochastic"])
    saved_predictions = np.load(OUTPUT_DIRECTORY / "predictions.npz")
    population_draw_figure(
        np.asarray(saved_predictions["selected_draw_log"], float),
        np.asarray(saved_predictions["truth_log"], float),
    )
    print(f"rebuilt refinement figures in {FIGURE_DIRECTORY}")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if command == "demo":
        demo()
    elif command == "run":
        run_experiment()
    elif command == "figures":
        regenerate_refinement_figures()
    else:
        raise SystemExit(f"unknown command {command!r}; use demo, run, or figures")
