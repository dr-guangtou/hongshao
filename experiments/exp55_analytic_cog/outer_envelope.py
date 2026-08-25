"""exp55 Stages 5--8: stochastic robustness and outer-envelope diagnosis.

This continuation asks why the generated mass--R90 relation remains wrong.
It separates the analytic representation, halo-map mean, and stochastic layer;
tests Monte Carlo and nearest-neighbour choices; and compares the already-fit
Moffat gamma=1.25 and gamma=1.4 representations. Non-portable MAH measurements
are used only as diagnostic ceilings.

Commands
--------
demo
    Run the mechanical path on 90 stratified galaxies and four draws.
run
    Run the complete Stage 5--8 analysis and save figures and outputs.
figures
    Rebuild figures from the complete saved analysis.
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
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from hongshao import qa  # noqa: E402
from hongshao.diffmah import log_mah  # noqa: E402
from hongshao.plotting import OKABE_ITO, save_fig, set_style  # noqa: E402
from hongshao.profile_emulator import density_from_cog  # noqa: E402
from hongshao.provenance import write_manifest  # noqa: E402
from hongshao.tng_data import ANCHORS, load_cosmic_time  # noqa: E402
from halo_map import (  # noqa: E402
    RADII,
    empirical_crps,
    folds_of,
    load_sample,
    shuffled_within_mass_bins,
    source_table,
)
from mixed import mixed_cog_log  # noqa: E402
from qa_figures import (  # noqa: E402
    density_by_halo_mass_figure,
    single_epoch_planes_figure,
    single_epoch_size_figure,
)
from refine_halo_map import (  # noqa: E402
    SPARSE_TERMS,
    equal_number_bins,
    fit_conditional_model,
    fractional_profile_residual,
    stochastic_population_summary,
)
from run import OUTDIR  # noqa: E402

set_style()

FIGURE_DIRECTORY = HERE / "figures" / "outer_envelope"
OUTPUT_DIRECTORY = HERE / "outputs" / "outer_envelope"
STANDARD_QA_DIRECTORY = FIGURE_DIRECTORY / "standard_qa_gamma1p4"
REFINEMENT_OUTPUT = HERE / "outputs" / "halo_map_refinement"

N_MAX = int(os.environ.get("EXP55_OUTER_NMAX", 0))
N_DRAW = int(os.environ.get("EXP55_OUTER_N_DRAW", 32))
N_POPULATION_DRAW = int(os.environ.get("EXP55_OUTER_N_POPULATION_DRAW", 4))
SEED = 7550

BASE_FEATURE_NAMES = ("logmp", "logtc", "early", "late", "c200c")
SIZE_NAMES = ("R50", "R80", "R90")
SIZE_FRACTIONS = (0.5, 0.8, 0.9)
PLANE_KEYS = (
    "kpc:M(<30)__kpc:M(30-50)",
    "kpc:M(<30)__kpc:M(50-100)",
    "Re:M(<2Re)__Re:M(2-4Re)",
)


def decode_fixed_gamma(parameters: np.ndarray, gamma: float) -> np.ndarray:
    """Decode raw mixture coordinates for one fixed outer slope."""
    parameters = np.asarray(parameters, float)
    original_shape = parameters.shape
    flat = parameters.reshape(-1, original_shape[-1])
    decoded = np.asarray([mixed_cog_log(row, RADII, 1.0, gamma) for row in flat], float)
    return decoded.reshape(*original_shape[:-1], len(RADII))


def aligned_profile_family(sample: dict, gamma_label: str) -> dict:
    """Load one fitted profile family on the halo-map sample order."""
    base = np.load(OUTDIR / "mixed_grid" / "selected.npz")
    row_by_index = {int(value): row for row, value in enumerate(base["indices"])}
    rows = np.asarray([row_by_index[int(value)] for value in sample["indices"]], int)
    cell = np.load(OUTDIR / "mixed_grid" / f"sersic1_moffat{gamma_label}.npz")
    return {
        "parameters": np.asarray(cell["parameters"][rows], float),
        "prediction_log": np.asarray(cell["prediction"][rows], float),
        "profile_rms_dex": np.asarray(cell["rms_dex"][rows], float),
        "jacobian_min_ratio": np.asarray(cell["jacobian_min_ratio"][rows], float),
        "boundary_distance": np.asarray(cell["boundary_distance"][rows], float),
    }


def fit_fold_models(features: np.ndarray, targets: np.ndarray) -> dict:
    """Fit the sparse conditional mean in the fixed five outer folds."""
    mean_parameters = np.empty_like(targets)
    models = []
    records = []
    all_rows = np.arange(len(features))
    for test in folds_of(len(features)):
        train = np.setdiff1d(all_rows, test)
        model = fit_conditional_model(features[train], targets[train], SPARSE_TERMS)
        mean_parameters[test] = model.predict_mean(features[test])
        models.append(model)
        records.append((train, test))
    return {
        "mean_parameters": mean_parameters,
        "models": models,
        "records": records,
    }


def neighbour_draws(
    model,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    n_draw: int,
    seed: int,
    n_neighbour: int,
    distance_columns: tuple[int, ...],
) -> np.ndarray:
    """Draw complete residual vectors from nearby training halos."""
    standardized_train = model.standardized(train_features)[:, distance_columns]
    standardized_test = model.standardized(test_features)[:, distance_columns]
    residual = train_targets - model.predict_mean(train_features)
    tree = cKDTree(standardized_train)
    _, neighbours = tree.query(
        standardized_test, k=min(n_neighbour, len(train_features))
    )
    if neighbours.ndim == 1:
        neighbours = neighbours[:, None]
    rng = np.random.default_rng(seed)
    choices = rng.integers(0, neighbours.shape[1], (n_draw, len(test_features)))
    selected = neighbours[np.arange(len(test_features))[None], choices]
    return model.predict_mean(test_features)[None] + residual[selected]


def calibrated_neighbour_generator(
    features: np.ndarray,
    targets: np.ndarray,
    truth_log: np.ndarray,
    fold_fit: dict,
    n_draw: int,
    seed: int,
    n_neighbour: int = 128,
    distance_columns: tuple[int, ...] = (0, 1, 2, 3, 4),
    gamma: float = 1.25,
) -> tuple[np.ndarray, list[dict]]:
    """Nested coverage calibration for a declared neighbour prescription."""
    output = np.empty((n_draw, *targets.shape))
    metadata = []
    scale_grid = np.asarray([1.00, 1.08, 1.16, 1.24, 1.32])
    for fold_number, ((train, test), model) in enumerate(
        zip(fold_fit["records"], fold_fit["models"])
    ):
        train_features = features[train]
        train_targets = targets[train]
        order = np.random.default_rng(seed + fold_number).permutation(len(train))
        inner_mean = np.empty_like(train_targets)
        inner_draw = np.empty((n_draw, *train_targets.shape))
        all_inner = np.arange(len(train))
        for inner_number, validation in enumerate(np.array_split(order, 4)):
            training = np.setdiff1d(all_inner, validation)
            inner_model = fit_conditional_model(
                train_features[training], train_targets[training], SPARSE_TERMS
            )
            inner_mean[validation] = inner_model.predict_mean(
                train_features[validation]
            )
            inner_draw[:, validation] = neighbour_draws(
                inner_model,
                train_features[training],
                train_targets[training],
                train_features[validation],
                n_draw,
                seed + 100 * fold_number + inner_number,
                n_neighbour,
                distance_columns,
            )
        coverage = {}
        for scale in scale_grid:
            draw_log = decode_fixed_gamma(
                inner_mean[None] + scale * (inner_draw - inner_mean[None]), gamma
            )
            coverage[f"{scale:.2f}"] = float(
                np.mean(
                    (truth_log[train] >= np.quantile(draw_log, 0.16, axis=0))
                    & (truth_log[train] <= np.quantile(draw_log, 0.84, axis=0))
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
            features[test],
            n_draw,
            seed + 1000 + fold_number,
            n_neighbour,
            distance_columns,
        )
        outer_mean = model.predict_mean(features[test])[None]
        output[:, test] = outer_mean + selected_scale * (base - outer_mean)
        metadata.append(
            {
                "fold": fold_number,
                "selected_scale": float(selected_scale),
                "inner_coverage_by_scale": coverage,
            }
        )
    return output, metadata


def profile_point_summary(
    point_log: np.ndarray,
    truth_log: np.ndarray,
    halo_mass: np.ndarray,
) -> dict:
    """Held-out profile and coherent radial metrics for one point prediction."""
    profile_rms = np.sqrt(np.mean((point_log - truth_log) ** 2, axis=1))
    model_density, _ = density_from_cog(point_log, RADII)
    truth_density, _ = density_from_cog(truth_log, RADII)
    density_rms = np.sqrt(np.mean((model_density - truth_density) ** 2, axis=1))
    bin_residual = mass_bin_shape_residual(point_log, truth_log, halo_mass)
    absolute_fractional = np.abs(10.0 ** (point_log - truth_log) - 1.0)
    return {
        "median_profile_rms_dex": float(np.median(profile_rms)),
        "median_density_rms_dex": float(np.median(density_rms)),
        "worst_mass_bin_shape_residual_percent": float(np.max(np.abs(bin_residual))),
        "mass_bin_shape_residual_percent": bin_residual,
        "median_maximum_fractional_error": float(
            np.median(np.max(absolute_fractional, axis=1))
        ),
    }


def mass_bin_shape_residual(
    point_log: np.ndarray,
    truth_log: np.ndarray,
    halo_mass: np.ndarray,
) -> np.ndarray:
    """Amplitude-removed median fractional residual in three halo-mass bins."""
    output = []
    for indices in equal_number_bins(halo_mass):
        residual = fractional_profile_residual(
            point_log[indices], truth_log[indices], remove_amplitude=True
        )
        output.append(100.0 * np.median(residual, axis=0))
    return np.asarray(output)


def profile_draw_summary(draw_log: np.ndarray, truth_log: np.ndarray) -> dict:
    """CRPS and pointwise interval coverage using all available draws."""
    return {
        "profile_crps_dex": float(np.mean(empirical_crps(truth_log, draw_log))),
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


def size_values(profile_log: np.ndarray) -> np.ndarray:
    """Return log10 R50, R80, and R90 for one profile population."""
    cogs = 10.0 ** np.asarray(profile_log)[:, None, :]
    return np.column_stack(
        [
            np.log10(qa._safe_rhalf(cogs, RADII, fraction)[:, 0])
            for fraction in SIZE_FRACTIONS
        ]
    )


def size_relation(profile_log: np.ndarray) -> dict:
    """Mass--size slopes, scatter, and medians for one profile population."""
    sizes = size_values(profile_log)
    mass = np.asarray(profile_log)[:, -1]
    output = {}
    for column, name in enumerate(SIZE_NAMES):
        coefficients = np.polyfit(mass, sizes[:, column], 1)
        output[name] = {
            "slope": float(coefficients[0]),
            "scatter_dex": float(
                np.std(sizes[:, column] - np.polyval(coefficients, mass))
            ),
            "median_log_radius": float(np.median(sizes[:, column])),
        }
    return output


def individual_population_metrics(
    draw_log: np.ndarray,
    truth_log: np.ndarray,
    n_evaluate: int,
) -> list[dict]:
    """Exact joint-population metrics for separate complete realizations."""
    output = []
    for draw_number in range(min(n_evaluate, len(draw_log))):
        summary = stochastic_population_summary(
            draw_log[draw_number : draw_number + 1],
            truth_log,
            n_population_draw=1,
        )
        output.append(
            {
                "draw": draw_number,
                "planes": summary["planes"],
                "sizes": summary["sizes"],
            }
        )
    return output


def distribution_interval(values: list[float]) -> dict:
    """Median and 16th--84th percentile interval across population draws."""
    values = np.asarray(values, float)
    return {
        "median": float(np.median(values)),
        "percentile_16": float(np.quantile(values, 0.16)),
        "percentile_84": float(np.quantile(values, 0.84)),
    }


def summarize_individual_populations(entries: list[dict]) -> dict:
    """Compress repeated full-population geometry into uncertainty intervals."""
    output = {"planes": {}, "sizes": {}}
    for key in PLANE_KEYS:
        output["planes"][key] = {}
        for metric in ("slope", "scatter", "energy_ratio", "energy_ratio_centered"):
            output["planes"][key][metric] = distribution_interval(
                [entry["planes"][key][metric] for entry in entries]
            )
    for name in SIZE_NAMES:
        output["sizes"][name] = {}
        for metric in ("slope", "scatter", "energy_ratio", "median_dlogR"):
            output["sizes"][name][metric] = distribution_interval(
                [entry["sizes"][name][metric] for entry in entries]
            )
    return output


def representation_grid(sample: dict, truth_log: np.ndarray) -> dict:
    """Measure outer-size behavior across the already-fitted 3-by-4 grid."""
    base = np.load(OUTDIR / "mixed_grid" / "selected.npz")
    row_by_index = {int(value): row for row, value in enumerate(base["indices"])}
    rows = np.asarray([row_by_index[int(value)] for value in sample["indices"]], int)
    truth_size = size_values(truth_log)
    output = {"truth": size_relation(truth_log), "cells": {}}
    for sersic_index in (1, 2, 4):
        for gamma_label in ("1p1", "1p25", "1p4", "1p7"):
            name = f"sersic{sersic_index}_moffat{gamma_label}"
            cell = np.load(OUTDIR / "mixed_grid" / f"{name}.npz")
            prediction = np.asarray(cell["prediction"][rows], float)
            sizes = size_values(prediction)
            error = sizes[:, 2] - truth_size[:, 2]
            output["cells"][name] = {
                "size_relation": size_relation(prediction),
                "median_r90_error_dex": float(np.median(error)),
                "r90_error_halo_mass_slope": float(
                    np.polyfit(sample["features"][:, 0], error, 1)[0]
                ),
                "median_profile_rms_dex": float(np.median(cell["rms_dex"][rows])),
                "median_density_rms_dex": float(
                    np.median(
                        np.sqrt(
                            np.mean(
                                (
                                    density_from_cog(prediction, RADII)[0]
                                    - density_from_cog(truth_log, RADII)[0]
                                )
                                ** 2,
                                axis=1,
                            )
                        )
                    )
                ),
                "median_log10_jacobian_min_ratio": float(
                    np.median(np.log10(cell["jacobian_min_ratio"][rows]))
                ),
                "boundary_fraction": float(
                    np.mean(cell["boundary_distance"][rows] < 0.01)
                ),
            }
    return output


def fold_representation_gates(
    incumbent: dict,
    candidate: dict,
    truth_log: np.ndarray,
) -> list[dict]:
    """Apply representation gates using training galaxies in every outer fold."""
    output = []
    all_rows = np.arange(len(truth_log))
    for fold_number, test in enumerate(folds_of(len(truth_log))):
        train = np.setdiff1d(all_rows, test)
        truth = truth_log[train]
        truth_density, _ = density_from_cog(truth, RADII)
        truth_r90_slope = size_relation(truth)["R90"]["slope"]
        summaries = []
        for family in (incumbent, candidate):
            prediction = family["prediction_log"][train]
            density, _ = density_from_cog(prediction, RADII)
            summaries.append(
                {
                    "median_profile_rms_dex": float(
                        np.median(np.sqrt(np.mean((prediction - truth) ** 2, axis=1)))
                    ),
                    "median_density_rms_dex": float(
                        np.median(
                            np.sqrt(np.mean((density - truth_density) ** 2, axis=1))
                        )
                    ),
                    "absolute_r90_slope_error": float(
                        abs(size_relation(prediction)["R90"]["slope"] - truth_r90_slope)
                    ),
                    "boundary_fraction": float(
                        np.mean(family["boundary_distance"][train] < 0.01)
                    ),
                }
            )
        first, second = summaries
        gates = {
            "profile_rms_within_0p0001_dex": second["median_profile_rms_dex"]
            <= first["median_profile_rms_dex"] + 0.0001,
            "density_rms_better": second["median_density_rms_dex"]
            < first["median_density_rms_dex"],
            "r90_slope_error_better": second["absolute_r90_slope_error"]
            < first["absolute_r90_slope_error"],
            "boundary_fraction_lower": second["boundary_fraction"]
            < first["boundary_fraction"],
        }
        output.append(
            {
                "fold": fold_number,
                "incumbent": first,
                "candidate": second,
                "gates": gates,
                "all_gates_pass": bool(all(gates.values())),
            }
        )
    return output


def halo_diagnostic_features(sample: dict) -> tuple[dict[str, np.ndarray], dict]:
    """Construct portable and explicitly non-portable halo diagnostics."""
    table = Table.read(source_table())
    row_by_index = {int(value): row for row, value in enumerate(table["index"])}
    rows = np.asarray([row_by_index[int(value)] for value in sample["indices"]], int)
    time = load_cosmic_time()
    anchor_time = time[ANCHORS[0.4][2]]
    log_anchor_time = np.log10(anchor_time)
    parameters = sample["features"][:, :4]
    predicted_growth = []
    actual_growth = []
    diffmah_miss = []
    for redshift in (0.7, 1.0, 1.5, 2.0):
        log_time = np.log10(time[ANCHORS[redshift][2]])
        predicted = (
            log_mah(
                np.asarray([log_time]),
                parameters[:, 0],
                parameters[:, 1],
                parameters[:, 2],
                parameters[:, 3],
                log_anchor_time,
            )[:, 0]
            - parameters[:, 0]
        )
        column = f"logmpeak_z{redshift:g}".replace(".", "p")
        actual = np.asarray(table[column][rows], float) - np.asarray(
            table["logmpeak_z0p4"][rows], float
        )
        predicted_growth.append(predicted)
        actual_growth.append(actual)
        diffmah_miss.append(actual - predicted)
    blocks = {
        "portable_diffmah_growth": np.column_stack(predicted_growth),
        "actual_anchor_growth_nonportable": np.column_stack(actual_growth),
        "diffmah_miss_nonportable": np.column_stack(diffmah_miss),
        "diffmah_fit_rms_nonportable": np.asarray(table["dmah_rms"][rows], float)[
            :, None
        ],
        "formation_redshift_nonportable": np.asarray(table["z50"][rows], float)[
            :, None
        ],
        "concurrent_accretion_rate": np.asarray(table["acc_rate"][rows], float)[
            :, None
        ],
        "halo_axis_ratio": np.asarray(table["c_to_a_3d"][rows], float)[:, None],
    }
    metadata = {
        "redshifts": [0.7, 1.0, 1.5, 2.0],
        "finite_fraction": {
            name: float(np.mean(np.isfinite(values).all(axis=1)))
            for name, values in blocks.items()
        },
    }
    return blocks, metadata


def diagnostic_correlations(
    sample: dict,
    truth_log: np.ndarray,
    mean_log: np.ndarray,
    blocks: dict[str, np.ndarray],
) -> dict:
    """Correlate outer-size residuals with each diagnostic information block."""
    truth_r90 = size_values(truth_log)[:, 2]
    mean_r90 = size_values(mean_log)[:, 2]
    error = mean_r90 - truth_r90
    output = {}
    for name, values in blocks.items():
        correlations = []
        for column in range(values.shape[1]):
            good = np.isfinite(values[:, column]) & np.isfinite(error)
            correlations.append(
                float(spearmanr(error[good], values[good, column]).statistic)
            )
        output[name] = correlations
    return output


def cross_validated_augmented_mean(
    base_features: np.ndarray,
    added_features: np.ndarray,
    targets: np.ndarray,
    gamma: float,
) -> dict:
    """Fit one added information block and its mass-conditioned shuffle."""
    complete = np.column_stack([base_features, added_features])
    good = np.isfinite(complete).all(axis=1)
    features = complete[good]
    local_targets = targets[good]

    def one(feature_values: np.ndarray) -> np.ndarray:
        prediction = np.empty_like(local_targets)
        all_rows = np.arange(len(feature_values))
        for test in folds_of(len(feature_values)):
            train = np.setdiff1d(all_rows, test)
            model = fit_conditional_model(
                feature_values[train], local_targets[train], SPARSE_TERMS
            )
            prediction[test] = model.predict_mean(feature_values[test])
        return decode_fixed_gamma(prediction, gamma)

    real_log = one(features)
    columns = tuple(range(base_features.shape[1], features.shape[1]))
    shuffled = shuffled_within_mass_bins(features, columns, seed=SEED + 33)
    shuffled_log = one(shuffled)
    return {
        "good": good,
        "real_log": real_log,
        "shuffled_log": shuffled_log,
    }


def augmented_feature_audit(
    sample: dict,
    targets: np.ndarray,
    truth_log: np.ndarray,
    blocks: dict[str, np.ndarray],
    gamma: float,
) -> dict:
    """Measure real versus shuffled incremental information for each block."""
    baseline = fit_fold_models(sample["features"], targets)
    baseline_log = decode_fixed_gamma(baseline["mean_parameters"], gamma)
    output = {
        "baseline": profile_point_summary(
            baseline_log, truth_log, sample["features"][:, 0]
        ),
        "blocks": {},
    }
    for name, values in blocks.items():
        result = cross_validated_augmented_mean(
            sample["features"], values, targets, gamma
        )
        good = result["good"]
        real = profile_point_summary(
            result["real_log"], truth_log[good], sample["features"][good, 0]
        )
        shuffled = profile_point_summary(
            result["shuffled_log"], truth_log[good], sample["features"][good, 0]
        )
        output["blocks"][name] = {
            "n_galaxy": int(np.count_nonzero(good)),
            "real": real,
            "shuffled": shuffled,
            "real_minus_shuffled_profile_rms_dex": real["median_profile_rms_dex"]
            - shuffled["median_profile_rms_dex"],
        }
    return output


def json_safe(value):
    """Convert nested NumPy values for the result record."""
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
    return value


def outer_source_figure(result: dict) -> None:
    """Show where the R90 error enters and how outer slope changes it."""
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
    names = ["TNG", "analytic fit", "halo-map mean", "generated"]
    keys = ["truth", "representation", "mean", "generated"]
    colors = ["black", OKABE_ITO[1], OKABE_ITO[2], OKABE_ITO[5]]
    for axis, metric, ylabel in (
        (axes[0, 0], "slope", "Slope of stellar mass--radius relation"),
        (axes[0, 1], "scatter_dex", "Vertical scatter (dex)"),
    ):
        x = np.arange(3)
        width = 0.18
        for index, (name, key, color) in enumerate(zip(names, keys, colors)):
            values = [
                result["size_decomposition"][key][size][metric] for size in SIZE_NAMES
            ]
            axis.bar(x + (index - 1.5) * width, values, width, label=name, color=color)
        axis.set_xticks(x, SIZE_NAMES)
        axis.set_ylabel(ylabel)
    axes[0, 0].legend(frameon=False, fontsize=8)

    grid = result["representation_grid"]["cells"]
    for gamma_label, color in zip(("1p1", "1p25", "1p4", "1p7"), OKABE_ITO[:4]):
        cell = grid[f"sersic1_moffat{gamma_label}"]
        axes[1, 0].scatter(
            cell["median_profile_rms_dex"],
            abs(
                cell["size_relation"]["R90"]["slope"]
                - result["representation_grid"]["truth"]["R90"]["slope"]
            ),
            color=color,
            s=65,
            label=rf"$\gamma={gamma_label.replace('p', '.')} $",
        )
    axes[1, 0].set_xlabel("Median representation CoG RMS (dex)")
    axes[1, 0].set_ylabel("Absolute R90 slope error")
    axes[1, 0].legend(frameon=False, fontsize=8)

    correlation = result["diagnostic_correlations"]
    labels = []
    values = []
    for name, entries in correlation.items():
        labels.append(name.replace("_", " "))
        values.append(max(entries, key=abs))
    order = np.argsort(np.abs(values))
    axes[1, 1].barh(
        np.arange(len(order)), np.asarray(values)[order], color=OKABE_ITO[4]
    )
    axes[1, 1].set_yticks(np.arange(len(order)), np.asarray(labels)[order], fontsize=7)
    axes[1, 1].axvline(0.0, color="0.4", linewidth=0.8)
    axes[1, 1].set_xlabel("Spearman correlation with halo-map R90 error")
    for panel, axis in zip("ABCD", axes.ravel()):
        axis.text(-0.13, 1.04, panel, transform=axis.transAxes, fontweight="bold")
    fig.suptitle("exp55 — the fixed outer profile contributes the R90 failure")
    fig.tight_layout()
    save_fig(fig, FIGURE_DIRECTORY / "exp55_outer_failure_source")
    plt.close(fig)


def robustness_figure(result: dict) -> None:
    """Compare neighbour settings and seeds against the incumbent generator."""
    configurations = result["stochastic_robustness"]["configurations"]
    names = list(configurations)
    labels = [configurations[name]["label"] for name in names]
    metrics = (
        ("profile_crps_dex", "Profile CRPS (dex)"),
        ("profile_coverage_68", "Nominal 68% coverage"),
        ("re_energy", "Size-scaled plane E / floor"),
        ("r90_energy", "R90 plane E / floor"),
    )
    fig, axes = plt.subplots(1, 4, figsize=(16, 6.4), sharey=True)
    y = np.arange(len(names))
    for panel, (axis, (key, xlabel)) in enumerate(zip(axes, metrics)):
        values = [configurations[name][key] for name in names]
        axis.barh(y, values, color=OKABE_ITO[1])
        axis.set_yticks(y, labels, fontsize=7)
        axis.invert_yaxis()
        axis.set_xlabel(xlabel)
        if key == "profile_coverage_68":
            axis.axvline(0.68, color="0.25", linestyle="--", linewidth=0.9)
        elif key.endswith("energy"):
            axis.axvline(1.0, color="0.25", linestyle="--", linewidth=0.9)
        baseline = result["stochastic_robustness"]["raw_gaussian_reference"]
        baseline_value = {
            "profile_crps_dex": baseline["profile_crps_dex"],
            "profile_coverage_68": baseline["profile_coverage_68"],
            "re_energy": baseline["re_energy"],
            "r90_energy": baseline["r90_energy"],
        }[key]
        axis.axvline(
            baseline_value,
            color=OKABE_ITO[5],
            linestyle=":",
            linewidth=1.4,
            label="raw Gaussian" if panel == 0 else None,
        )
        if key in ("re_energy", "r90_energy"):
            group = "planes" if key == "re_energy" else "sizes"
            item = PLANE_KEYS[2] if key == "re_energy" else "R90"
            interval = result["stochastic_robustness"]["saved_selected_draw_intervals"][
                group
            ][item]["energy_ratio"]
            axis.axvspan(
                interval["percentile_16"],
                interval["percentile_84"],
                color=OKABE_ITO[2],
                alpha=0.12,
            )
            axis.axvline(interval["median"], color=OKABE_ITO[2], linewidth=1.1)
        for row, value in enumerate(values):
            axis.text(value, row, f" {value:.3f}", va="center", fontsize=7)
        axis.text(
            -0.14, 1.04, "ABCD"[panel], transform=axis.transAxes, fontweight="bold"
        )
    axes[0].legend(frameon=False, fontsize=7, loc="lower right")
    fig.suptitle("exp55 — stochastic robustness across seeds and neighbour definitions")
    fig.tight_layout()
    save_fig(fig, FIGURE_DIRECTORY / "exp55_stochastic_robustness")
    plt.close(fig)


def feature_audit_figure(result: dict) -> None:
    """Show real-versus-shuffled gains for targeted additional information."""
    audits = (
        (result["feature_audit_gamma1p25"], r"fixed $\gamma=1.25$"),
        (result["feature_audit_gamma1p4"], r"fixed $\gamma=1.4$"),
    )
    blocks = audits[0][0]["blocks"]
    names = list(blocks)
    labels = [name.replace("_", " ") for name in names]
    y = np.arange(len(names))
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.0), sharey=True)
    for row, (audit, title) in enumerate(audits):
        local = audit["blocks"]
        real = [local[name]["real"]["median_profile_rms_dex"] for name in names]
        shuffled = [local[name]["shuffled"]["median_profile_rms_dex"] for name in names]
        worst_real = [
            local[name]["real"]["worst_mass_bin_shape_residual_percent"]
            for name in names
        ]
        worst_shuffled = [
            local[name]["shuffled"]["worst_mass_bin_shape_residual_percent"]
            for name in names
        ]
        axes[row, 0].plot(real, y, "o", color=OKABE_ITO[2], label="real")
        axes[row, 0].plot(
            shuffled,
            y,
            "s",
            color=OKABE_ITO[5],
            label="mass-conditioned shuffle",
        )
        axes[row, 1].plot(worst_real, y, "o", color=OKABE_ITO[2])
        axes[row, 1].plot(worst_shuffled, y, "s", color=OKABE_ITO[5])
        axes[row, 0].set_title(title)
        axes[row, 1].set_title(title)
        axes[row, 0].set_yticks(y, labels, fontsize=7)
        axes[row, 0].invert_yaxis()
        axes[row, 0].set_xlabel("Median held-out CoG RMS (dex)")
        axes[row, 1].set_xlabel(r"Worst mass-bin shape residual (\%)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    for panel, axis in zip("ABCD", axes.ravel()):
        axis.text(-0.12, 1.04, panel, transform=axis.transAxes, fontweight="bold")
    fig.suptitle("exp55 — targeted halo information, real versus shuffled")
    fig.tight_layout()
    save_fig(fig, FIGURE_DIRECTORY / "exp55_outer_feature_audit")
    plt.close(fig)


def gamma_comparison_figure(result: dict) -> None:
    """Compare representation, parameter mean, and draw median for two slopes."""
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 7.8), sharex=False)
    for column, gamma_name in enumerate(("gamma1p25", "gamma1p4")):
        values = result["gamma_comparison"][gamma_name]
        for label, key, color, style in (
            ("analytic representation", "representation_log", OKABE_ITO[1], "-"),
            ("decoded parameter mean", "parameter_mean_log", OKABE_ITO[3], "--"),
            ("median generated profile", "draw_median_log", OKABE_ITO[2], "-."),
        ):
            residual = mass_bin_shape_residual(
                np.asarray(values[key]),
                np.asarray(result["truth_log"]),
                np.asarray(result["halo_mass"]),
            )
            for row, mass_bin in enumerate((0, 1)):
                axes[row, column].plot(
                    RADII,
                    residual[mass_bin],
                    color=color,
                    linestyle=style,
                    label=label if mass_bin == 0 else None,
                )
        axes[0, column].set_title(values["label"])
        axes[0, column].axhline(0.0, color="0.5", linewidth=0.8)
        axes[1, column].axhline(0.0, color="0.5", linewidth=0.8)
        axes[0, column].set_xscale("log")
        axes[1, column].set_xscale("log")
        axes[0, column].set_xlim(RADII[0], RADII[-1])
        axes[1, column].set_xlim(RADII[0], RADII[-1])
        axes[0, column].set_ylabel(r"Low-mass residual (\%)")
        axes[1, column].set_ylabel(r"Middle-mass residual (\%)")
        axes[1, column].set_xlabel("Radius (kpc)")
    # Use the spare third column for the high-mass residuals and R90 slopes.
    for gamma_name, color in zip(
        ("gamma1p25", "gamma1p4"), (OKABE_ITO[0], OKABE_ITO[4])
    ):
        values = result["gamma_comparison"][gamma_name]
        residual = mass_bin_shape_residual(
            np.asarray(values["draw_median_log"]),
            np.asarray(result["truth_log"]),
            np.asarray(result["halo_mass"]),
        )
        axes[0, 2].plot(RADII, residual[2], color=color, label=values["label"])
    axes[0, 2].axhline(0.0, color="0.5", linewidth=0.8)
    axes[0, 2].set_xscale("log")
    axes[0, 2].set_xlim(RADII[0], RADII[-1])
    axes[0, 2].set_ylabel(r"High-mass draw-median residual (\%)")
    axes[0, 2].set_xlabel("Radius (kpc)")
    axes[0, 2].legend(frameon=False, fontsize=8)
    names = ["TNG", r"$\gamma=1.25$", r"$\gamma=1.4$"]
    slopes = [
        result["size_decomposition"]["truth"]["R90"]["slope"],
        result["gamma_comparison"]["gamma1p25"]["draw_size"]["R90"]["slope"],
        result["gamma_comparison"]["gamma1p4"]["draw_size"]["R90"]["slope"],
    ]
    positions = np.arange(3)
    axes[1, 2].bar(positions, slopes, color=["black", OKABE_ITO[0], OKABE_ITO[4]])
    axes[1, 2].set_xticks(positions, names, rotation=15)
    axes[1, 2].set_ylabel("Mass--R90 slope")
    for panel, axis in zip("ABCDEF", axes.ravel()):
        axis.text(-0.13, 1.04, panel, transform=axis.transAxes, fontweight="bold")
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle("exp55 — outer-slope comparison through the full halo generator")
    fig.tight_layout()
    save_fig(fig, FIGURE_DIRECTORY / "exp55_gamma_generator_comparison")
    plt.close(fig)


def run_experiment(
    n_max: int = N_MAX,
    n_draw: int = N_DRAW,
    n_population_draw: int = N_POPULATION_DRAW,
) -> dict:
    """Run Stages 5--8 and write the regenerable record."""
    started = time.perf_counter()
    sample = load_sample(n_max)
    truth_log = sample["truth_log"]
    halo_mass = sample["features"][:, 0]
    family_1p25 = aligned_profile_family(sample, "1p25")
    family_1p4 = aligned_profile_family(sample, "1p4")
    saved = np.load(REFINEMENT_OUTPUT / "predictions.npz")
    if len(sample["indices"]) == len(saved["indices"]):
        if not np.array_equal(sample["indices"], saved["indices"]):
            raise ValueError("saved refinement sample is not aligned")
        baseline_mean_log = np.asarray(saved["selected_mean_log"], float)
        baseline_draw_log = np.asarray(saved["selected_draw_log"], float)
    else:
        baseline_fit = fit_fold_models(sample["features"], family_1p25["parameters"])
        baseline_mean_log = decode_fixed_gamma(baseline_fit["mean_parameters"], 1.25)
        baseline_raw, _ = calibrated_neighbour_generator(
            sample["features"],
            family_1p25["parameters"],
            truth_log,
            baseline_fit,
            n_draw,
            SEED,
        )
        baseline_draw_log = decode_fixed_gamma(baseline_raw, 1.25)

    generated_reference = np.median(baseline_draw_log, axis=0)
    result = {
        "n_galaxy": len(truth_log),
        "n_draw": n_draw,
        "n_population_draw": n_population_draw,
        "size_decomposition": {
            "truth": size_relation(truth_log),
            "representation": size_relation(family_1p25["prediction_log"]),
            "mean": size_relation(baseline_mean_log),
            "generated": size_relation(generated_reference),
        },
        "representation_grid": representation_grid(sample, truth_log),
    }

    blocks, halo_metadata = halo_diagnostic_features(sample)
    result["halo_diagnostic_metadata"] = halo_metadata
    result["diagnostic_correlations"] = diagnostic_correlations(
        sample, truth_log, baseline_mean_log, blocks
    )

    print("Stage 5: stochastic robustness", flush=True)
    baseline_fit = fit_fold_models(sample["features"], family_1p25["parameters"])
    robustness_configurations = {}
    configurations = [
        ("seed_7550", "seed 7550; 128; all", 7550, 128, (0, 1, 2, 3, 4)),
        ("seed_7650", "seed 7650; 128; all", 7650, 128, (0, 1, 2, 3, 4)),
        ("seed_7750", "seed 7750; 128; all", 7750, 128, (0, 1, 2, 3, 4)),
        ("neighbour_32", "seed 7550; 32; all", 7550, 32, (0, 1, 2, 3, 4)),
        ("neighbour_64", "seed 7550; 64; all", 7550, 64, (0, 1, 2, 3, 4)),
        ("neighbour_256", "seed 7550; 256; all", 7550, 256, (0, 1, 2, 3, 4)),
        ("distance_mass", "seed 7550; 128; mass", 7550, 128, (0,)),
        ("distance_diffmah", "seed 7550; 128; DiffMAH", 7550, 128, (0, 1, 2, 3)),
    ]
    for name, label, seed, n_neighbour, columns in configurations:
        print(f"  {label}", flush=True)
        raw_draw, metadata = calibrated_neighbour_generator(
            sample["features"],
            family_1p25["parameters"],
            truth_log,
            baseline_fit,
            n_draw,
            seed,
            n_neighbour,
            columns,
            1.25,
        )
        draw_log = decode_fixed_gamma(raw_draw, 1.25)
        marginal = profile_draw_summary(draw_log, truth_log)
        population = stochastic_population_summary(
            draw_log, truth_log, n_population_draw=n_population_draw
        )
        robustness_configurations[name] = {
            "label": label,
            "n_neighbour": n_neighbour,
            "distance_columns": columns,
            "metadata": metadata,
            **marginal,
            "re_energy": population["planes"][PLANE_KEYS[2]]["energy_ratio"],
            "fixed_one_energy": population["planes"][PLANE_KEYS[0]]["energy_ratio"],
            "fixed_two_energy": population["planes"][PLANE_KEYS[1]]["energy_ratio"],
            "r90_energy": population["sizes"]["R90"]["energy_ratio"],
            "r90_slope": population["sizes"]["R90"]["slope"],
            "r90_median_dlogr": population["sizes"]["R90"]["median_dlogR"],
        }
    saved_individual = individual_population_metrics(
        baseline_draw_log,
        truth_log,
        len(baseline_draw_log) if not n_max else min(4, len(baseline_draw_log)),
    )
    initial_result = json.loads((REFINEMENT_OUTPUT / "results.json").read_text())
    raw_reference = initial_result["stochastic_candidates"]["raw_gaussian"]
    result["stochastic_robustness"] = {
        "configurations": robustness_configurations,
        "saved_selected_draw_intervals": summarize_individual_populations(
            saved_individual
        ),
        "n_saved_population_draw": len(saved_individual),
        "raw_gaussian_reference": {
            "profile_crps_dex": raw_reference["profile_crps_dex"],
            "profile_coverage_68": raw_reference["profile_coverage_68"],
            "re_energy": raw_reference["planes"][PLANE_KEYS[2]]["energy_ratio"],
            "r90_energy": raw_reference["sizes"]["R90"]["energy_ratio"],
        },
    }

    print("Stage 6/7: fixed outer-slope comparison", flush=True)
    gamma_comparison = {}
    draw_logs = {}
    for name, label, gamma, family in (
        ("gamma1p25", r"fixed $\gamma=1.25$", 1.25, family_1p25),
        ("gamma1p4", r"fixed $\gamma=1.4$", 1.4, family_1p4),
    ):
        fold_fit = fit_fold_models(sample["features"], family["parameters"])
        raw_draw, metadata = calibrated_neighbour_generator(
            sample["features"],
            family["parameters"],
            truth_log,
            fold_fit,
            n_draw,
            SEED + (0 if gamma == 1.25 else 500),
            128,
            (0, 1, 2, 3, 4),
            gamma,
        )
        draw_log = decode_fixed_gamma(raw_draw, gamma)
        draw_logs[name] = draw_log
        parameter_mean_log = decode_fixed_gamma(fold_fit["mean_parameters"], gamma)
        draw_median_log = np.median(draw_log, axis=0)
        population = stochastic_population_summary(
            draw_log, truth_log, n_population_draw=n_population_draw
        )
        gamma_comparison[name] = {
            "label": label,
            "representation_log": family["prediction_log"],
            "parameter_mean_log": parameter_mean_log,
            "draw_median_log": draw_median_log,
            "representation": profile_point_summary(
                family["prediction_log"], truth_log, halo_mass
            ),
            "parameter_mean": profile_point_summary(
                parameter_mean_log, truth_log, halo_mass
            ),
            "draw_median": profile_point_summary(draw_median_log, truth_log, halo_mass),
            "draw_marginal": profile_draw_summary(draw_log, truth_log),
            "draw_population": population,
            "draw_size": size_relation(draw_median_log),
            "metadata": metadata,
            "median_log10_jacobian_min_ratio": float(
                np.median(np.log10(family["jacobian_min_ratio"]))
            ),
            "boundary_fraction": float(np.mean(family["boundary_distance"] < 0.01)),
        }
    result["gamma_comparison"] = gamma_comparison
    result["gamma1p4_representation_folds"] = fold_representation_gates(
        family_1p25, family_1p4, truth_log
    )

    print("Stage 7: targeted halo-feature audit", flush=True)
    result["feature_audit_gamma1p25"] = augmented_feature_audit(
        sample, family_1p25["parameters"], truth_log, blocks, 1.25
    )
    result["feature_audit_gamma1p4"] = augmented_feature_audit(
        sample, family_1p4["parameters"], truth_log, blocks, 1.4
    )

    baseline_gamma = gamma_comparison["gamma1p25"]
    candidate_gamma = gamma_comparison["gamma1p4"]
    incumbent_grid = result["representation_grid"]["cells"]["sersic1_moffat1p25"]
    candidate_grid = result["representation_grid"]["cells"]["sersic1_moffat1p4"]
    truth_r90_slope = result["size_decomposition"]["truth"]["R90"]["slope"]
    gates = {
        "representation_cog_within_0p0001_dex": candidate_grid["median_profile_rms_dex"]
        <= incumbent_grid["median_profile_rms_dex"] + 0.0001,
        "representation_density_better": candidate_grid["median_density_rms_dex"]
        < incumbent_grid["median_density_rms_dex"],
        "representation_r90_slope_error_better": abs(
            candidate_grid["size_relation"]["R90"]["slope"] - truth_r90_slope
        )
        < abs(incumbent_grid["size_relation"]["R90"]["slope"] - truth_r90_slope),
        "draw_crps_within_1_percent": candidate_gamma["draw_marginal"][
            "profile_crps_dex"
        ]
        <= 1.01 * baseline_gamma["draw_marginal"]["profile_crps_dex"],
        "draw_median_shape_within_10_percent": candidate_gamma["draw_median"][
            "worst_mass_bin_shape_residual_percent"
        ]
        <= 1.10
        * baseline_gamma["draw_median"]["worst_mass_bin_shape_residual_percent"],
        "parameter_mean_maximum_error_within_10_percent": candidate_gamma[
            "parameter_mean"
        ]["median_maximum_fractional_error"]
        <= 1.10 * baseline_gamma["parameter_mean"]["median_maximum_fractional_error"],
        "fixed_kpc_population_within_10_percent": max(
            candidate_gamma["draw_population"]["planes"][PLANE_KEYS[0]]["energy_ratio"],
            candidate_gamma["draw_population"]["planes"][PLANE_KEYS[1]]["energy_ratio"],
        )
        <= 1.10
        * max(
            baseline_gamma["draw_population"]["planes"][PLANE_KEYS[0]]["energy_ratio"],
            baseline_gamma["draw_population"]["planes"][PLANE_KEYS[1]]["energy_ratio"],
        ),
        "size_scaled_population_better": candidate_gamma["draw_population"]["planes"][
            PLANE_KEYS[2]
        ]["energy_ratio"]
        < baseline_gamma["draw_population"]["planes"][PLANE_KEYS[2]]["energy_ratio"],
        "r90_population_below_incumbent_draw_variation": candidate_gamma[
            "draw_population"
        ]["sizes"]["R90"]["energy_ratio"]
        < result["stochastic_robustness"]["saved_selected_draw_intervals"]["sizes"][
            "R90"
        ]["energy_ratio"]["percentile_16"],
        "coverage_within_3_percentage_points": abs(
            candidate_gamma["draw_marginal"]["profile_coverage_68"] - 0.68
        )
        <= 0.03,
        "representation_gates_pass_in_every_training_fold": all(
            fold["all_gates_pass"] for fold in result["gamma1p4_representation_folds"]
        ),
        "boundary_fraction_lower": candidate_gamma["boundary_fraction"]
        < baseline_gamma["boundary_fraction"],
    }
    result["gamma1p4_gates"] = gates
    result["gamma1p4_selected"] = bool(all(gates.values()))

    result["truth_log"] = truth_log
    result["halo_mass"] = halo_mass
    result["elapsed_seconds"] = time.perf_counter() - started
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIRECTORY / "results.json").write_text(
        json.dumps(json_safe(result), indent=2)
    )
    np.savez_compressed(
        OUTPUT_DIRECTORY / "predictions.npz",
        indices=sample["indices"],
        truth_log=truth_log,
        halo_mass=halo_mass,
        gamma1p25_draw_log=draw_logs["gamma1p25"],
        gamma1p4_draw_log=draw_logs["gamma1p4"],
        gamma1p25_draw_median_log=gamma_comparison["gamma1p25"]["draw_median_log"],
        gamma1p4_draw_median_log=gamma_comparison["gamma1p4"]["draw_median_log"],
        gamma1p25_parameter_mean_log=gamma_comparison["gamma1p25"][
            "parameter_mean_log"
        ],
        gamma1p4_parameter_mean_log=gamma_comparison["gamma1p4"]["parameter_mean_log"],
    )
    write_manifest(
        OUTPUT_DIRECTORY,
        params={
            "script": Path(__file__).name,
            "n_galaxy": len(truth_log),
            "n_draw": n_draw,
            "n_population_draw": n_population_draw,
        },
    )
    generate_figures(result)
    return result


def generate_figures(result: dict) -> None:
    """Write the Stage 5--8 custom figures."""
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    outer_source_figure(result)
    robustness_figure(result)
    feature_audit_figure(result)
    gamma_comparison_figure(result)


def generate_standard_qa(result: dict, predictions: dict) -> None:
    """Run standard QA only if gamma=1.4 passes every predeclared gate."""
    if not result["gamma1p4_selected"]:
        return
    truth_log = np.asarray(predictions["truth_log"], float)
    point_log = np.asarray(predictions["gamma1p4_parameter_mean_log"], float)
    draw_log = np.asarray(predictions["gamma1p4_draw_log"], float)
    halo_mass = np.asarray(predictions["halo_mass"], float)
    STANDARD_QA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    qa_result = qa.evaluate(
        10.0 ** point_log[:, None, :],
        10.0 ** truth_log[:, None, :],
        RADII,
        [0.4],
        name="exp55_gamma1p4_halo_map",
        figdir=STANDARD_QA_DIRECTORY,
        bin_by=halo_mass,
        bin_label=r"DiffMAH $\log M_{\rm peak}$",
        draw_cogs=10.0 ** draw_log[:, :, None, :],
    )
    density_by_halo_mass_figure(
        point_log,
        truth_log,
        halo_mass,
        figdir=STANDARD_QA_DIRECTORY,
        model_name="exp55_gamma1p4_halo_map",
        model_label=r"analytic halo map, $\gamma=1.4$",
    )
    single_epoch_planes_figure(
        qa_result,
        figdir=STANDARD_QA_DIRECTORY,
        model_name="exp55_gamma1p4_halo_map",
        model_label=r"analytic halo map, $\gamma=1.4$",
    )
    single_epoch_size_figure(
        point_log,
        truth_log,
        qa_result,
        figdir=STANDARD_QA_DIRECTORY,
        model_name="exp55_gamma1p4_halo_map",
        model_label=r"analytic halo map, $\gamma=1.4$",
    )


def demo() -> None:
    """Mechanical small-sample validation before the complete run."""
    result = run_experiment(n_max=90, n_draw=4, n_population_draw=1)
    assert result["n_galaxy"] == 90
    assert set(result["gamma_comparison"]) == {"gamma1p25", "gamma1p4"}
    assert len(result["stochastic_robustness"]["configurations"]) == 8
    print("exp55 outer-envelope demo OK", flush=True)


def regenerate_figures() -> None:
    """Rebuild figures and conditional standard QA from saved outputs."""
    result = json.loads((OUTPUT_DIRECTORY / "results.json").read_text())
    predictions = np.load(OUTPUT_DIRECTORY / "predictions.npz")
    generate_figures(result)
    generate_standard_qa(result, predictions)


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if command == "demo":
        demo()
    elif command == "run":
        result = run_experiment()
        predictions = np.load(OUTPUT_DIRECTORY / "predictions.npz")
        generate_standard_qa(result, predictions)
        print(
            f"exp55 outer-envelope run complete in {result['elapsed_seconds']:.1f} s",
            flush=True,
        )
    elif command == "figures":
        regenerate_figures()
    else:
        raise SystemExit(f"unknown command {command!r}; use demo, run, or figures")
