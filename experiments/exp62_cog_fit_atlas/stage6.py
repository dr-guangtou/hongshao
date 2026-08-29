# %%
"""Exp62 Stage 6: halo information in cubic-logit CoG shape coordinates.

Commands:

    uv run python experiments/exp62_cog_fit_atlas/stage6.py mechanics
    uv run python experiments/exp62_cog_fit_atlas/stage6.py demo
    uv run python experiments/exp62_cog_fit_atlas/stage6.py full
    uv run python experiments/exp62_cog_fit_atlas/stage6.py figures demo|full

Stellar mass is held at its fitted value and is never a regression target.
The calculation measures descriptive, held-out halo--shape associations.
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from hongshao.plotting import OKABE_ITO, save_fig, set_style  # noqa: E402
from hongshao.provenance import write_manifest  # noqa: E402
from hongshao.qa import enclosed_radius  # noqa: E402
from run import OUTDIR, RADII, REDSHIFTS, profile_metrics  # noqa: E402
from stage3 import _log_annulus_mass, _mass_at_radius  # noqa: E402
from stage4 import (  # noqa: E402
    bootstrap_coefficients,
    galaxy_folds,
    mass_conditioned_shuffle,
)
from stage5 import (  # noqa: E402
    CANDIDATE_SPECS,
    STAGE5_OUTDIR,
    candidate_cog_log,
)
from visual_record import _binned_statistics  # noqa: E402

set_style()

SOURCE = OUTDIR / "source" / "exp51_epoch_local_diffmah.npz"
STAGE6_OUTDIR = OUTDIR / "stage6"
STAGE6_FIGDIR = HERE / "figures" / "stage6"
PREDICTOR_NAMES = ("logmp", "logtc", "early_index", "late_index", "c_200c")
SHAPE_NAMES = (
    "log_scale_radius",
    "log_derivative_margin",
    "quadratic_warp",
    "log_cubic_warp",
)
OBJECTIVE_NAMES = ("log", "absolute")
RELATION_NAMES = (
    "diffmah_linear",
    "concentration_linear",
    "complete_linear",
    "complete_additive_quadratic",
)
N_DEMO = 90
N_BOOTSTRAP = {"demo": 40, "full": 200}
N_SHUFFLE = {"demo": 8, "full": 32}


def _mass_design(train_mass: np.ndarray, evaluation_mass: np.ndarray) -> np.ndarray:
    mean = np.mean(train_mass)
    scale = max(np.std(train_mass), 1.0e-8)
    standardized = (evaluation_mass - mean) / scale
    return np.column_stack([np.ones(len(standardized)), standardized, standardized**2])


def _feature_basis(
    train_features: np.ndarray,
    evaluation_features: np.ndarray,
    degree: int,
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(train_features, axis=0)
    scale = np.clip(np.std(train_features, axis=0), 1.0e-8, None)
    train_standardized = (train_features - mean) / scale
    evaluation_standardized = (evaluation_features - mean) / scale
    train_columns = []
    evaluation_columns = []
    for feature in range(train_features.shape[1]):
        for power in range(1, degree + 1):
            train_column = train_standardized[:, feature] ** power
            evaluation_column = evaluation_standardized[:, feature] ** power
            column_mean = np.mean(train_column)
            column_scale = max(np.std(train_column), 1.0e-8)
            train_columns.append((train_column - column_mean) / column_scale)
            evaluation_columns.append((evaluation_column - column_mean) / column_scale)
    return np.column_stack(train_columns), np.column_stack(evaluation_columns)


def cross_fitted_shape_relation(
    targets: np.ndarray,
    mass: np.ndarray,
    features: np.ndarray,
    folds: np.ndarray,
    degree: int = 1,
) -> dict[str, np.ndarray]:
    """Predict mass-residual targets with additive feature powers in held-out folds."""
    targets = np.asarray(targets, float)
    mass = np.asarray(mass, float)
    features = np.asarray(features, float)
    unique_folds = np.unique(folds)
    mass_prediction = np.empty_like(targets)
    full_prediction = np.empty_like(targets)
    fold_coefficients = np.empty(
        (len(unique_folds), features.shape[1] * degree, targets.shape[1])
    )
    for fold_index, fold in enumerate(unique_folds):
        train = folds != fold
        test = folds == fold
        train_mass_design = _mass_design(mass[train], mass[train])
        test_mass_design = _mass_design(mass[train], mass[test])
        target_mass_coefficient = np.linalg.lstsq(
            train_mass_design, targets[train], rcond=None
        )[0]
        target_residual = targets[train] - train_mass_design @ target_mass_coefficient
        train_basis_raw, test_basis_raw = _feature_basis(
            features[train], features[test], degree
        )
        basis_mass_coefficient = np.linalg.lstsq(
            train_mass_design, train_basis_raw, rcond=None
        )[0]
        train_basis = train_basis_raw - train_mass_design @ basis_mass_coefficient
        test_basis = test_basis_raw - test_mass_design @ basis_mass_coefficient
        basis_mean = np.mean(train_basis, axis=0)
        basis_scale = np.clip(np.std(train_basis, axis=0), 1.0e-8, None)
        train_basis = (train_basis - basis_mean) / basis_scale
        test_basis = (test_basis - basis_mean) / basis_scale
        coefficient = np.linalg.lstsq(
            np.column_stack([np.ones(train.sum()), train_basis]),
            target_residual,
            rcond=None,
        )[0]
        fold_coefficients[fold_index] = coefficient[1:]
        mass_prediction[test] = test_mass_design @ target_mass_coefficient
        full_prediction[test] = (
            mass_prediction[test]
            + np.column_stack([np.ones(test.sum()), test_basis]) @ coefficient
        )
    mass_residual = targets - mass_prediction
    full_residual = targets - full_prediction
    mass_rmse = np.sqrt(np.mean(mass_residual**2, axis=0))
    full_rmse = np.sqrt(np.mean(full_residual**2, axis=0))
    return {
        "mass_prediction": mass_prediction,
        "full_prediction": full_prediction,
        "mass_rmse": mass_rmse,
        "full_rmse": full_rmse,
        "residual_r2": 1.0
        - np.sum(full_residual**2, axis=0)
        / np.clip(np.sum(mass_residual**2, axis=0), 1.0e-30, None),
        "fold_coefficients": fold_coefficients,
    }


def align_cubic_parameters(
    stage_rows: np.ndarray,
    stage_epochs: np.ndarray,
    stage_parameters: np.ndarray,
    galaxy_ids: np.ndarray,
    n_epoch: int = 5,
) -> np.ndarray:
    lookup = {
        (int(row), int(epoch)): index
        for index, (row, epoch) in enumerate(zip(stage_rows, stage_epochs))
    }
    output = np.empty((len(galaxy_ids), n_epoch, stage_parameters.shape[1]))
    for galaxy, galaxy_id in enumerate(galaxy_ids):
        for epoch in range(n_epoch):
            output[galaxy, epoch] = stage_parameters[lookup[(int(galaxy_id), epoch)]]
    return output


def decode_cubic_shape_predictions(
    log_mstar_148: np.ndarray,
    shape_parameters: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    log_mstar_148 = np.asarray(log_mstar_148, float)
    shape_parameters = np.asarray(shape_parameters, float)
    original_shape = shape_parameters.shape[:-1]
    mass_flat = log_mstar_148.reshape(-1)
    shape_flat = shape_parameters.reshape(-1, len(SHAPE_NAMES))
    spec = CANDIDATE_SPECS["logit_cubic5"]
    lower = np.asarray(spec.lower)
    upper = np.asarray(spec.upper)
    parameters = np.column_stack([mass_flat, shape_flat])
    bounded = np.clip(parameters, lower + 1.0e-8, upper - 1.0e-8)
    clipped = np.any(bounded != parameters, axis=1)
    profiles = np.asarray(
        [candidate_cog_log("logit_cubic5", value, RADII) for value in bounded]
    )
    return profiles.reshape(*original_shape, len(RADII)), clipped.reshape(
        original_shape
    )


def _stratified_rows(values: np.ndarray, n_row: int = N_DEMO) -> np.ndarray:
    order = np.argsort(values)
    positions = np.linspace(0, len(order) - 1, n_row).round().astype(int)
    return order[positions]


def load_data(mode: str) -> dict[str, np.ndarray]:
    source = np.load(SOURCE)
    stage = np.load(STAGE5_OUTDIR / "full" / "predictions.npz")
    galaxy_ids = np.asarray(source["indices"], int)
    selection = np.arange(len(galaxy_ids))
    if mode == "demo":
        selection = _stratified_rows(np.asarray(source["epoch_halo_mass"])[:, 0])
    galaxy_ids = galaxy_ids[selection]
    parameters = {}
    for objective, key in (
        ("log", "parameters_logit_cubic5"),
        ("absolute", "parameters_logit_cubic5_absolute"),
    ):
        parameters[objective] = align_cubic_parameters(
            np.asarray(stage["rows"], int),
            np.asarray(stage["epochs"], int),
            np.asarray(stage[key]),
            galaxy_ids,
        )
    diffmah = np.asarray(source["parameters"])[selection]
    concentration = np.asarray(source["concentration"])[selection]
    return {
        "galaxy_ids": galaxy_ids,
        "halo_mass": np.asarray(source["epoch_halo_mass"])[selection],
        "features": np.concatenate([diffmah, concentration[..., None]], axis=-1),
        "truth_log": np.asarray(source["cogs_log"])[selection],
        **{f"parameters_{name}": value for name, value in parameters.items()},
    }


def _relation_specification(name: str) -> tuple[np.ndarray, int]:
    if name == "diffmah_linear":
        return np.arange(4), 1
    if name == "concentration_linear":
        return np.asarray([4]), 1
    if name == "complete_linear":
        return np.arange(5), 1
    if name == "complete_additive_quadratic":
        return np.arange(5), 2
    raise ValueError(name)


def _median_metrics_by_epoch(
    prediction: np.ndarray, truth: np.ndarray
) -> dict[str, np.ndarray]:
    keys = (
        "cog_rms_full_dex",
        "cog_rms_5_30_dex",
        "density_rms_dex",
    )
    output = {key: np.empty(len(REDSHIFTS)) for key in keys}
    for radius_name in ("r50", "r80", "r90"):
        output[f"absolute_{radius_name}_error_dex"] = np.empty(len(REDSHIFTS))
    for epoch in range(len(REDSHIFTS)):
        metrics = profile_metrics(prediction[:, epoch], truth[:, epoch])
        for key in keys:
            output[key][epoch] = np.median(metrics[key])
        for column, radius_name in zip((1, 2, 3), ("r50", "r80", "r90")):
            output[f"absolute_{radius_name}_error_dex"][epoch] = np.median(
                np.abs(metrics["size_error_dex"][:, column])
            )
    return output


def _cog_error_tail_by_epoch(
    prediction: np.ndarray, truth: np.ndarray
) -> dict[str, np.ndarray]:
    per_galaxy = np.sqrt(np.mean((prediction - truth) ** 2, axis=-1))
    return {
        "p90_cog_rms_dex": np.quantile(per_galaxy, 0.90, axis=0),
        "p99_cog_rms_dex": np.quantile(per_galaxy, 0.99, axis=0),
        "maximum_cog_rms_dex": np.max(per_galaxy, axis=0),
        "fraction_above_0p2_dex": np.mean(per_galaxy > 0.2, axis=0),
    }


def _predictor_design_condition_numbers(
    mass: np.ndarray,
    features: np.ndarray,
    folds: np.ndarray,
) -> np.ndarray:
    unique_folds = np.unique(folds)
    output = np.empty((mass.shape[1], len(unique_folds)))
    for epoch in range(mass.shape[1]):
        for fold_index, fold in enumerate(unique_folds):
            train = folds != fold
            mass_design = _mass_design(mass[train, epoch], mass[train, epoch])
            feature_basis, _ = _feature_basis(
                features[train, epoch], features[train, epoch], degree=1
            )
            mass_coefficient = np.linalg.lstsq(mass_design, feature_basis, rcond=None)[
                0
            ]
            residual_basis = feature_basis - mass_design @ mass_coefficient
            residual_basis = (
                residual_basis - np.mean(residual_basis, axis=0)
            ) / np.clip(np.std(residual_basis, axis=0), 1.0e-8, None)
            design = np.column_stack([np.ones(train.sum()), residual_basis])
            output[epoch, fold_index] = np.linalg.cond(design)
    return output


def _select_relation(
    linear_metrics: dict[str, np.ndarray],
    quadratic_metrics: dict[str, np.ndarray],
) -> tuple[str, dict[str, object]]:
    full_improvement = 1.0 - (
        quadratic_metrics["cog_rms_full_dex"] / linear_metrics["cog_rms_full_dex"]
    )
    local_nonworse = np.column_stack(
        [
            quadratic_metrics[key] <= linear_metrics[key]
            for key in (
                "density_rms_dex",
                "absolute_r80_error_dex",
                "absolute_r90_error_dex",
            )
        ]
    )
    improvement_gate = int(np.count_nonzero(full_improvement >= 0.02)) >= 3
    local_gate = int(np.count_nonzero(np.all(local_nonworse, axis=1))) >= 3
    selected = (
        "complete_additive_quadratic"
        if improvement_gate and local_gate
        else "complete_linear"
    )
    return selected, {
        "full_cog_improvement_fraction_by_epoch": full_improvement.tolist(),
        "epochs_improving_full_cog_by_two_percent": int(
            np.count_nonzero(full_improvement >= 0.02)
        ),
        "epochs_nonworse_in_density_r80_r90": int(
            np.count_nonzero(np.all(local_nonworse, axis=1))
        ),
        "selected_relation": selected,
    }


def _bootstrap_linear(
    targets: np.ndarray,
    mass: np.ndarray,
    features: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    return bootstrap_coefficients(targets, mass, features, n_bootstrap, seed)


def _decoded_linear_responses(
    parameters: np.ndarray,
    bootstrap_beta: np.ndarray,
    direct_beta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_bootstrap = bootstrap_beta.shape[1]
    raw_response = np.empty(
        (len(REDSHIFTS), n_bootstrap, len(PREDICTOR_NAMES), len(RADII))
    )
    direct_response = np.zeros((len(REDSHIFTS), len(PREDICTOR_NAMES), len(RADII)))
    direct_response[..., :-1] = direct_beta
    correlation = np.empty((len(REDSHIFTS), len(PREDICTOR_NAMES)))
    sign_agreement = np.empty_like(correlation)
    for epoch in range(len(REDSHIFTS)):
        baseline = np.median(parameters[:, epoch], axis=0)
        baseline_log = candidate_cog_log("logit_cubic5", baseline, RADII)
        baseline_shape = baseline_log - baseline_log[-1]
        for bootstrap in range(n_bootstrap):
            for predictor in range(len(PREDICTOR_NAMES)):
                changed = baseline.copy()
                changed[1:] += bootstrap_beta[epoch, bootstrap, predictor]
                profile, _ = decode_cubic_shape_predictions(
                    np.asarray([baseline[0]]), changed[None, 1:]
                )
                raw_response[epoch, bootstrap, predictor] = (
                    profile[0] - profile[0, -1] - baseline_shape
                )
        median_raw = np.median(raw_response[epoch], axis=0)
        for predictor in range(len(PREDICTOR_NAMES)):
            correlation[epoch, predictor] = np.corrcoef(
                median_raw[predictor, :-1], direct_response[epoch, predictor, :-1]
            )[0, 1]
            sign_agreement[epoch, predictor] = np.mean(
                np.sign(median_raw[predictor, :-1])
                == np.sign(direct_response[epoch, predictor, :-1])
            )
    return raw_response, direct_response, correlation, sign_agreement


def _empirical_cdf(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    ordered = np.sort(values[np.isfinite(values)])
    return np.searchsorted(ordered, grid, side="right") / len(ordered)


def _mass_quantities(profile: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "m30": _mass_at_radius(profile, 30.0),
        "m30_50": _log_annulus_mass(profile, 30.0, 50.0),
        "m50_100": _log_annulus_mass(profile, 50.0, 100.0),
        "m100_148": _log_annulus_mass(profile, 100.0, RADII[-1]),
    }


def parameter_information_figure(mode: str, result: dict[str, np.ndarray]) -> None:
    gain = 100.0 * (
        1.0 - result["relation_full_rmse_log"] / result["relation_mass_rmse_log"]
    )
    figure, axes = plt.subplots(
        1, len(REDSHIFTS), figsize=(17.5, 4.5), sharey=True, layout="constrained"
    )
    limit = max(5.0, min(30.0, float(np.quantile(np.abs(gain), 0.98))))
    for epoch, axis in enumerate(axes):
        image = axis.imshow(
            gain[:, epoch],
            aspect="auto",
            cmap="PuOr_r",
            vmin=-limit,
            vmax=limit,
        )
        axis.set_title(rf"$z={REDSHIFTS[epoch]:g}$")
        axis.set_xticks(
            np.arange(len(SHAPE_NAMES)), SHAPE_NAMES, rotation=70, ha="right"
        )
        axis.set_yticks(np.arange(len(RELATION_NAMES)), RELATION_NAMES)
    figure.colorbar(
        image,
        ax=axes,
        label=r"Held-out shape-coordinate RMS reduction from halo mass only (\%)",
        shrink=0.78,
    )
    figure.suptitle(
        f"Exp62 Stage 6 {mode}: incremental halo information; color range clipped at +/-{limit:g}%"
    )
    save_fig(figure, STAGE6_FIGDIR / mode / "exp62_stage6_parameter_information")
    plt.close(figure)


def coefficient_figure(mode: str, result: dict[str, np.ndarray]) -> None:
    beta = result["linear_beta_standardized"]
    bootstrap = result["linear_bootstrap_beta_standardized"]
    limit = float(np.quantile(np.abs(beta), 0.98))
    figure, axes = plt.subplots(
        1, len(REDSHIFTS), figsize=(17.5, 4.5), sharey=True, layout="constrained"
    )
    for epoch, axis in enumerate(axes):
        image = axis.imshow(
            beta[epoch], aspect="auto", cmap="PuOr_r", vmin=-limit, vmax=limit
        )
        lower, upper = np.quantile(bootstrap[epoch], [0.025, 0.975], axis=0)
        for predictor in range(len(PREDICTOR_NAMES)):
            for target in range(len(SHAPE_NAMES)):
                if lower[predictor, target] * upper[predictor, target] > 0.0:
                    axis.text(
                        target, predictor, "*", ha="center", va="center", fontsize=8
                    )
        axis.set_title(rf"$z={REDSHIFTS[epoch]:g}$")
        axis.set_xticks(
            np.arange(len(SHAPE_NAMES)), SHAPE_NAMES, rotation=70, ha="right"
        )
        axis.set_yticks(np.arange(len(PREDICTOR_NAMES)), PREDICTOR_NAMES)
    figure.colorbar(
        image,
        ax=axes,
        label="Conditional response / mass-residual target scatter",
        shrink=0.78,
    )
    figure.suptitle(
        f"Exp62 Stage 6 {mode}: linear mass-conditioned associations; star = bootstrap 95% interval excludes zero"
    )
    save_fig(figure, STAGE6_FIGDIR / mode / "exp62_stage6_parameter_associations")
    plt.close(figure)


def null_control_figure(mode: str, result: dict[str, np.ndarray]) -> None:
    mass_rmse = result["selected_mass_rmse"]
    real = 100.0 * (1.0 - result["selected_full_rmse"] / mass_rmse)
    shuffled_mah = 100.0 * (
        1.0 - np.median(result["shuffle_mah_rmse"], axis=0) / mass_rmse
    )
    shuffled_c = 100.0 * (1.0 - np.median(result["shuffle_c_rmse"], axis=0) / mass_rmse)
    values = np.stack([real, shuffled_mah, shuffled_c], axis=0)
    limit = max(1.0, float(np.quantile(np.abs(values), 0.98)))
    figure, axes = plt.subplots(
        1, len(REDSHIFTS), figsize=(17.5, 3.8), sharey=True, layout="constrained"
    )
    labels = (
        "Real DiffMAH + c",
        "Shuffled DiffMAH + real c",
        "Real DiffMAH + shuffled c",
    )
    for epoch, axis in enumerate(axes):
        image = axis.imshow(
            values[:, epoch], aspect="auto", cmap="PuOr_r", vmin=-limit, vmax=limit
        )
        axis.set_title(rf"$z={REDSHIFTS[epoch]:g}$")
        axis.set_xticks(
            np.arange(len(SHAPE_NAMES)), SHAPE_NAMES, rotation=70, ha="right"
        )
        axis.set_yticks(np.arange(3), labels)
    figure.colorbar(
        image,
        ax=axes,
        label=r"Held-out RMS reduction from halo mass only (\%)",
        shrink=0.78,
    )
    figure.suptitle(f"Exp62 Stage 6 {mode}: mass-conditioned block-shuffle controls")
    save_fig(figure, STAGE6_FIGDIR / mode / "exp62_stage6_null_controls")
    plt.close(figure)


def radial_response_figure(mode: str, result: dict[str, np.ndarray]) -> None:
    raw = result["raw_radial_response"]
    direct = result["direct_radial_response"]
    colors = (OKABE_ITO[0], OKABE_ITO[1], OKABE_ITO[2], OKABE_ITO[4], OKABE_ITO[5])
    figure, axes = plt.subplots(
        1, len(PREDICTOR_NAMES), figsize=(17.5, 3.8), sharey=True
    )
    for predictor, axis in enumerate(axes):
        for epoch, color in enumerate(colors):
            median = np.median(raw[epoch, :, predictor], axis=0)
            lower, upper = np.quantile(raw[epoch, :, predictor], [0.16, 0.84], axis=0)
            axis.plot(RADII, median, color=color, label=rf"$z={REDSHIFTS[epoch]:g}$")
            axis.fill_between(RADII, lower, upper, color=color, alpha=0.08)
            axis.plot(
                RADII,
                direct[epoch, predictor],
                color=color,
                linestyle="--",
                linewidth=1.0,
            )
        axis.axhline(0.0, color="0.6", linewidth=0.7)
        axis.set_xscale("log")
        axis.set_title(PREDICTOR_NAMES[predictor])
        axis.set_xlabel("Semi-major axis (kpc)")
    axes[0].set_ylabel("Normalized CoG response (dex)")
    axes[0].legend(frameon=False, fontsize=7, title="Solid: cubic; dashed: direct")
    figure.suptitle(
        f"Exp62 Stage 6 {mode}: response to +1 mass-conditioned predictor SD; bands are bootstrap 16--84%"
    )
    figure.tight_layout()
    save_fig(figure, STAGE6_FIGDIR / mode / "exp62_stage6_decoded_radial_responses")
    plt.close(figure)


def average_cog_figure(
    mode: str,
    data: dict[str, np.ndarray],
    mass_profile: np.ndarray,
    full_profile: np.ndarray,
    bin_kind: str,
) -> None:
    figure, axes = plt.subplots(
        len(REDSHIFTS), 3, figsize=(13.5, 16.0), sharex=True, sharey=True
    )
    truth = data["truth_log"]
    for epoch, redshift in enumerate(REDSHIFTS):
        values = (
            data["halo_mass"][:, epoch] if bin_kind == "halo" else truth[:, epoch, -1]
        )
        edges = np.quantile(values, [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
        for mass_bin, axis in enumerate(axes[epoch]):
            selected = (values >= edges[mass_bin]) & (values <= edges[mass_bin + 1])
            truth_shape = truth[selected, epoch] - truth[selected, epoch, -1:]
            for profile, color, style, label in (
                (mass_profile, "0.45", "--", "Concurrent halo mass only"),
                (full_profile, OKABE_ITO[2], "-", "Selected complete relation"),
            ):
                model_shape = profile[selected, epoch] - profile[selected, epoch, -1:]
                residual = 100.0 * np.median(
                    10.0 ** (model_shape - truth_shape) - 1.0, axis=0
                )
                axis.plot(RADII, residual, color=color, linestyle=style, label=label)
            axis.axhline(0.0, color="0.65", linewidth=0.7)
            axis.set_xscale("log")
            symbol = r"M_{\rm h}" if bin_kind == "halo" else r"M_{*,148}"
            axis.set_title(
                rf"$z={redshift:g};\ \log {symbol}={edges[mass_bin]:.2f}$--${edges[mass_bin + 1]:.2f}$"
            )
            if mass_bin == 0:
                axis.set_ylabel("Median normalized-CoG residual (%)")
            if epoch == len(REDSHIFTS) - 1:
                axis.set_xlabel("Semi-major axis (kpc)")
    axes[0, 0].legend(frameon=False, fontsize=7)
    figure.suptitle(f"Exp62 Stage 6 {mode}: average CoGs in {bin_kind}-mass terciles")
    figure.tight_layout()
    save_fig(
        figure, STAGE6_FIGDIR / mode / f"exp62_stage6_average_cog_{bin_kind}_mass_bins"
    )
    plt.close(figure)


def population_qa_figure(
    mode: str,
    data: dict[str, np.ndarray],
    mass_profile: np.ndarray,
    full_profile: np.ndarray,
) -> None:
    truth = data["truth_log"]
    models = {"TNG": truth, "Halo mass only": mass_profile, "Complete": full_profile}
    colors = {"TNG": "black", "Halo mass only": "0.55", "Complete": OKABE_ITO[2]}
    styles = {"TNG": ":", "Halo mass only": "--", "Complete": "-"}
    figure, axes = plt.subplots(3, len(REDSHIFTS), figsize=(15.7, 9.7))
    for epoch, redshift in enumerate(REDSHIFTS):
        mass_148 = truth[:, epoch, -1]
        for name, profiles in models.items():
            profile = profiles[:, epoch]
            quantities = _mass_quantities(profile)
            x, y, _, _ = _binned_statistics(quantities["m30"], quantities["m50_100"])
            axes[0, epoch].plot(
                x, y, color=colors[name], linestyle=styles[name], label=name
            )
            linear = 10.0**profile
            for row, fraction in enumerate((0.8, 0.9), start=1):
                radius = np.log10(
                    [enclosed_radius(value, RADII, fraction) for value in linear]
                )
                x, y, _, _ = _binned_statistics(mass_148, radius)
                axes[row, epoch].plot(x, y, color=colors[name], linestyle=styles[name])
        axes[0, epoch].set_title(rf"$z={redshift:g}$")
        axes[0, epoch].set_xlabel(r"$\log_{10} M_*(<30\,\mathrm{kpc})$")
        axes[1, epoch].set_xlabel(r"$\log_{10} M_{*,148}$")
        axes[2, epoch].set_xlabel(r"$\log_{10} M_{*,148}$")
    axes[0, 0].set_ylabel(r"$\log_{10} M_*(50\!-\!100\,\mathrm{kpc})$")
    axes[1, 0].set_ylabel(r"$\log_{10} R_{80}\,(\mathrm{kpc})$")
    axes[2, 0].set_ylabel(r"$\log_{10} R_{90}\,(\mathrm{kpc})$")
    axes[0, 0].legend(frameon=False, fontsize=7)
    figure.suptitle(f"Exp62 Stage 6 {mode}: stellar-mass and outer-size planes")
    figure.tight_layout()
    save_fig(figure, STAGE6_FIGDIR / mode / "exp62_stage6_population_qa")
    plt.close(figure)


def mass_cdf_figure(
    mode: str,
    data: dict[str, np.ndarray],
    mass_profile: np.ndarray,
    full_profile: np.ndarray,
) -> None:
    quantities = (
        ("m30", r"$M_*(<30)$"),
        ("m30_50", r"$M_*(30\!-\!50)$"),
        ("m50_100", r"$M_*(50\!-\!100)$"),
        ("m100_148", r"$M_*(100\!-\!148)$"),
    )
    figure, axes = plt.subplots(4, len(REDSHIFTS), figsize=(15.7, 11.7))
    for epoch, redshift in enumerate(REDSHIFTS):
        truth = _mass_quantities(data["truth_log"][:, epoch])
        models = {
            "Halo mass only": _mass_quantities(mass_profile[:, epoch]),
            "Complete": _mass_quantities(full_profile[:, epoch]),
        }
        for row, (quantity, label) in enumerate(quantities):
            display = truth[quantity][truth[quantity] > 5.0]
            grid = np.linspace(
                np.quantile(display, 0.005), np.quantile(display, 0.995), 160
            )
            truth_cdf = _empirical_cdf(truth[quantity], grid)
            for name, color, style in (
                ("Halo mass only", "0.5", "--"),
                ("Complete", OKABE_ITO[2], "-"),
            ):
                axes[row, epoch].plot(
                    grid,
                    _empirical_cdf(models[name][quantity], grid) - truth_cdf,
                    color=color,
                    linestyle=style,
                    label=name,
                )
            axes[row, epoch].axhline(0.0, color="0.65", linewidth=0.7)
            if epoch == 0:
                axes[row, epoch].set_ylabel(f"{label}\nmodel CDF - TNG CDF")
            if row == 0:
                axes[row, epoch].set_title(rf"$z={redshift:g}$")
            if row == 3:
                axes[row, epoch].set_xlabel(r"$\log_{10} M_*/M_\odot$")
    axes[0, 0].legend(frameon=False, fontsize=7)
    figure.suptitle(
        f"Exp62 Stage 6 {mode}: aperture, annulus, and outskirt mass CDF residuals"
    )
    figure.tight_layout()
    save_fig(figure, STAGE6_FIGDIR / mode / "exp62_stage6_mass_cdf_residuals")
    plt.close(figure)


def worst_case_figure(
    mode: str,
    data: dict[str, np.ndarray],
    full_profile: np.ndarray,
) -> None:
    residual = full_profile - data["truth_log"]
    rms = np.sqrt(np.mean(residual**2, axis=-1))
    flat = np.argsort(rms.ravel())[-10:][::-1]
    figure, axes = plt.subplots(5, 2, figsize=(11.5, 15.0))
    for axis, index in zip(axes.flat, flat):
        galaxy, epoch = np.unravel_index(index, rms.shape)
        truth_shape = (
            data["truth_log"][galaxy, epoch] - data["truth_log"][galaxy, epoch, -1]
        )
        model_shape = full_profile[galaxy, epoch] - full_profile[galaxy, epoch, -1]
        axis.plot(RADII, truth_shape, color="black", label="TNG")
        axis.plot(RADII, model_shape, color=OKABE_ITO[2], label="Complete relation")
        axis.plot(
            RADII,
            model_shape - truth_shape,
            color=OKABE_ITO[5],
            linestyle="--",
            label="Residual",
        )
        axis.axhline(0.0, color="0.7", linewidth=0.7)
        axis.set_xscale("log")
        axis.set_title(
            rf"galaxy {data['galaxy_ids'][galaxy]}; $z={REDSHIFTS[epoch]:g}$; RMS={rms[galaxy, epoch]:.3f} dex"
        )
        axis.set_xlabel("Semi-major axis (kpc)")
        axis.set_ylabel("Normalized log CoG / residual (dex)")
    axes[0, 0].legend(frameon=False, fontsize=7)
    figure.suptitle(f"Exp62 Stage 6 {mode}: ten largest complete-relation CoG errors")
    figure.tight_layout()
    save_fig(figure, STAGE6_FIGDIR / mode / "exp62_stage6_worst10")
    plt.close(figure)


def make_figures(mode: str) -> None:
    data = load_data(mode)
    result = dict(np.load(STAGE6_OUTDIR / mode / "associations.npz"))
    mass_profile = result["selected_mass_profile"]
    full_profile = result["selected_full_profile"]
    parameter_information_figure(mode, result)
    coefficient_figure(mode, result)
    null_control_figure(mode, result)
    radial_response_figure(mode, result)
    average_cog_figure(mode, data, mass_profile, full_profile, "halo")
    average_cog_figure(mode, data, mass_profile, full_profile, "stellar")
    population_qa_figure(mode, data, mass_profile, full_profile)
    mass_cdf_figure(mode, data, mass_profile, full_profile)
    worst_case_figure(mode, data, full_profile)


def run(mode: str) -> dict[str, object]:
    start = time.perf_counter()
    data = load_data(mode)
    folds = galaxy_folds(data["halo_mass"])
    n_galaxy = len(data["galaxy_ids"])
    n_bootstrap = N_BOOTSTRAP[mode]
    n_shuffle = N_SHUFFLE[mode]
    relations = {
        objective: {name: [None] * len(REDSHIFTS) for name in RELATION_NAMES}
        for objective in OBJECTIVE_NAMES
    }
    for objective in OBJECTIVE_NAMES:
        targets = data[f"parameters_{objective}"][..., 1:]
        for epoch in range(len(REDSHIFTS)):
            for relation_name in RELATION_NAMES:
                columns, degree = _relation_specification(relation_name)
                relations[objective][relation_name][epoch] = (
                    cross_fitted_shape_relation(
                        targets[:, epoch],
                        data["halo_mass"][:, epoch],
                        data["features"][:, epoch, columns],
                        folds,
                        degree,
                    )
                )

    decoded = {}
    decoded_metrics = {}
    for objective in OBJECTIVE_NAMES:
        mass_coordinate = data[f"parameters_{objective}"][..., 0]
        decoded[objective] = {}
        decoded_metrics[objective] = {}
        for relation_name in ("complete_linear", "complete_additive_quadratic"):
            mass_shape = np.stack(
                [
                    relations[objective][relation_name][epoch]["mass_prediction"]
                    for epoch in range(len(REDSHIFTS))
                ],
                axis=1,
            )
            full_shape = np.stack(
                [
                    relations[objective][relation_name][epoch]["full_prediction"]
                    for epoch in range(len(REDSHIFTS))
                ],
                axis=1,
            )
            mass_profile, mass_clipped = decode_cubic_shape_predictions(
                mass_coordinate, mass_shape
            )
            full_profile, full_clipped = decode_cubic_shape_predictions(
                mass_coordinate, full_shape
            )
            decoded[objective][relation_name] = {
                "mass_profile": mass_profile,
                "full_profile": full_profile,
                "mass_clipped": mass_clipped,
                "full_clipped": full_clipped,
            }
            decoded_metrics[objective][relation_name] = {
                "mass": _median_metrics_by_epoch(mass_profile, data["truth_log"]),
                "full": _median_metrics_by_epoch(full_profile, data["truth_log"]),
            }

    if mode == "demo":
        selected_relation, selection = _select_relation(
            decoded_metrics["log"]["complete_linear"]["full"],
            decoded_metrics["log"]["complete_additive_quadratic"]["full"],
        )
    else:
        demo_summary = json.loads((STAGE6_OUTDIR / "demo" / "summary.json").read_text())
        selected_relation = demo_summary["selected_relation"]
        selection = demo_summary["relation_selection"]
    selected_degree = _relation_specification(selected_relation)[1]
    selected = relations["log"][selected_relation]
    selected_mass_shape = np.stack(
        [item["mass_prediction"] for item in selected], axis=1
    )
    selected_full_shape = np.stack(
        [item["full_prediction"] for item in selected], axis=1
    )
    selected_mass_profile, selected_mass_clipped = decode_cubic_shape_predictions(
        data["parameters_log"][..., 0], selected_mass_shape
    )
    selected_full_profile, selected_full_clipped = decode_cubic_shape_predictions(
        data["parameters_log"][..., 0], selected_full_shape
    )
    selected_mass_metrics = _median_metrics_by_epoch(
        selected_mass_profile, data["truth_log"]
    )
    selected_full_metrics = _median_metrics_by_epoch(
        selected_full_profile, data["truth_log"]
    )
    predictor_condition_numbers = _predictor_design_condition_numbers(
        data["halo_mass"], data["features"], folds
    )
    diffmah_slopes = data["features"][..., 2:4]
    diffmah_slope_at_boundary = np.any(
        (diffmah_slopes <= 0.05001) | (diffmah_slopes >= 11.99999), axis=-1
    )

    linear_beta = np.empty((len(REDSHIFTS), len(PREDICTOR_NAMES), len(SHAPE_NAMES)))
    linear_bootstrap = np.empty(
        (len(REDSHIFTS), n_bootstrap, len(PREDICTOR_NAMES), len(SHAPE_NAMES))
    )
    linear_bootstrap_standardized = np.empty_like(linear_bootstrap)
    direct_beta = np.empty((len(REDSHIFTS), len(PREDICTOR_NAMES), len(RADII) - 1))
    for epoch in range(len(REDSHIFTS)):
        linear_result = relations["log"]["complete_linear"][epoch]
        linear_beta[epoch] = np.mean(linear_result["fold_coefficients"], axis=0)
        raw, standardized = _bootstrap_linear(
            data["parameters_log"][:, epoch, 1:],
            data["halo_mass"][:, epoch],
            data["features"][:, epoch],
            n_bootstrap,
            6600 + epoch,
        )
        linear_bootstrap[epoch] = raw
        linear_bootstrap_standardized[epoch] = standardized
        truth_shape = (
            data["truth_log"][:, epoch, :-1] - data["truth_log"][:, epoch, -1:]
        )
        direct_result = cross_fitted_shape_relation(
            truth_shape,
            data["halo_mass"][:, epoch],
            data["features"][:, epoch],
            folds,
            degree=1,
        )
        direct_beta[epoch] = np.mean(direct_result["fold_coefficients"], axis=0)
    target_scale = np.empty((len(REDSHIFTS), len(SHAPE_NAMES)))
    for epoch in range(len(REDSHIFTS)):
        target_scale[epoch] = np.std(
            data["parameters_log"][:, epoch, 1:]
            - relations["log"]["complete_linear"][epoch]["mass_prediction"],
            axis=0,
        )
    linear_beta_standardized = linear_beta / np.clip(
        target_scale[:, None], 1.0e-8, None
    )
    absolute_beta = np.empty_like(linear_beta)
    absolute_target_scale = np.empty_like(target_scale)
    for epoch in range(len(REDSHIFTS)):
        absolute_relation = relations["absolute"]["complete_linear"][epoch]
        absolute_beta[epoch] = np.mean(absolute_relation["fold_coefficients"], axis=0)
        absolute_target_scale[epoch] = np.std(
            data["parameters_absolute"][:, epoch, 1:]
            - absolute_relation["mass_prediction"],
            axis=0,
        )
    absolute_beta_standardized = absolute_beta / np.clip(
        absolute_target_scale[:, None], 1.0e-8, None
    )
    objective_beta_correlation = np.asarray(
        [
            np.corrcoef(
                linear_beta_standardized[epoch].ravel(),
                absolute_beta_standardized[epoch].ravel(),
            )[0, 1]
            for epoch in range(len(REDSHIFTS))
        ]
    )
    objective_beta_sign_agreement = np.mean(
        np.sign(linear_beta_standardized) == np.sign(absolute_beta_standardized),
        axis=(1, 2),
    )
    raw_response, direct_response, response_correlation, response_sign_agreement = (
        _decoded_linear_responses(data["parameters_log"], linear_bootstrap, direct_beta)
    )

    rng = np.random.default_rng(6260)
    selected_mass_rmse = np.stack([item["mass_rmse"] for item in selected], axis=0)
    selected_full_rmse = np.stack([item["full_rmse"] for item in selected], axis=0)
    shuffle_mah_rmse = np.empty((n_shuffle, len(REDSHIFTS), len(SHAPE_NAMES)))
    shuffle_c_rmse = np.empty_like(shuffle_mah_rmse)
    shuffle_mah_cog_rms = np.empty((n_shuffle, len(REDSHIFTS)))
    shuffle_c_cog_rms = np.empty_like(shuffle_mah_cog_rms)
    for shuffle in range(n_shuffle):
        for epoch in range(len(REDSHIFTS)):
            for control, columns, store_rmse, store_cog in (
                ("mah", (0, 1, 2, 3), shuffle_mah_rmse, shuffle_mah_cog_rms),
                ("concentration", (4,), shuffle_c_rmse, shuffle_c_cog_rms),
            ):
                shuffled = mass_conditioned_shuffle(
                    data["features"][:, epoch],
                    data["halo_mass"][:, epoch],
                    columns,
                    rng,
                )
                relation = cross_fitted_shape_relation(
                    data["parameters_log"][:, epoch, 1:],
                    data["halo_mass"][:, epoch],
                    shuffled,
                    folds,
                    selected_degree,
                )
                store_rmse[shuffle, epoch] = relation["full_rmse"]
                profile, _ = decode_cubic_shape_predictions(
                    data["parameters_log"][:, epoch, 0], relation["full_prediction"]
                )
                store_cog[shuffle, epoch] = np.median(
                    profile_metrics(profile, data["truth_log"][:, epoch])[
                        "cog_rms_full_dex"
                    ]
                )

    arrays = {
        "galaxy_ids": data["galaxy_ids"],
        "redshifts": REDSHIFTS,
        "predictor_names": np.asarray(PREDICTOR_NAMES),
        "shape_names": np.asarray(SHAPE_NAMES),
        "folds": folds,
        "relation_mass_rmse_log": np.stack(
            [
                [
                    relations["log"][name][epoch]["mass_rmse"]
                    for epoch in range(len(REDSHIFTS))
                ]
                for name in RELATION_NAMES
            ]
        ),
        "relation_full_rmse_log": np.stack(
            [
                [
                    relations["log"][name][epoch]["full_rmse"]
                    for epoch in range(len(REDSHIFTS))
                ]
                for name in RELATION_NAMES
            ]
        ),
        "relation_mass_rmse_absolute": np.stack(
            [
                [
                    relations["absolute"][name][epoch]["mass_rmse"]
                    for epoch in range(len(REDSHIFTS))
                ]
                for name in RELATION_NAMES
            ]
        ),
        "relation_full_rmse_absolute": np.stack(
            [
                [
                    relations["absolute"][name][epoch]["full_rmse"]
                    for epoch in range(len(REDSHIFTS))
                ]
                for name in RELATION_NAMES
            ]
        ),
        "selected_mass_rmse": selected_mass_rmse,
        "selected_full_rmse": selected_full_rmse,
        "selected_mass_profile": selected_mass_profile,
        "selected_full_profile": selected_full_profile,
        "selected_mass_clipped": selected_mass_clipped,
        "selected_full_clipped": selected_full_clipped,
        "linear_beta_standardized": linear_beta_standardized,
        "absolute_beta_standardized": absolute_beta_standardized,
        "objective_beta_correlation": objective_beta_correlation,
        "objective_beta_sign_agreement": objective_beta_sign_agreement,
        "linear_bootstrap_beta": linear_bootstrap,
        "linear_bootstrap_beta_standardized": linear_bootstrap_standardized,
        "raw_radial_response": raw_response,
        "direct_radial_response": direct_response,
        "response_correlation": response_correlation,
        "response_sign_agreement": response_sign_agreement,
        "predictor_design_condition_numbers": predictor_condition_numbers,
        "diffmah_slope_at_boundary": diffmah_slope_at_boundary,
        "shuffle_mah_rmse": shuffle_mah_rmse,
        "shuffle_c_rmse": shuffle_c_rmse,
        "shuffle_mah_cog_rms": shuffle_mah_cog_rms,
        "shuffle_c_cog_rms": shuffle_c_cog_rms,
    }
    output_dir = STAGE6_OUTDIR / mode
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "associations.npz", **arrays)

    make_figures(mode)
    elapsed = time.perf_counter() - start
    selected_mass_tail = _cog_error_tail_by_epoch(
        selected_mass_profile, data["truth_log"]
    )
    selected_full_tail = _cog_error_tail_by_epoch(
        selected_full_profile, data["truth_log"]
    )
    decoded_metric_record = {
        objective: {
            relation_name: {
                prediction_name: {
                    metric_name: metric_value.tolist()
                    for metric_name, metric_value in metric_values.items()
                }
                for prediction_name, metric_values in relation_values.items()
            }
            for relation_name, relation_values in objective_values.items()
        }
        for objective, objective_values in decoded_metrics.items()
    }
    summary = {
        "mode": mode,
        "n_galaxy": n_galaxy,
        "n_bootstrap": n_bootstrap,
        "n_shuffle": n_shuffle,
        "selected_relation": selected_relation,
        "relation_selection": selection,
        "selected_mass_only_metrics_by_epoch": {
            key: value.tolist() for key, value in selected_mass_metrics.items()
        },
        "selected_complete_metrics_by_epoch": {
            key: value.tolist() for key, value in selected_full_metrics.items()
        },
        "decoded_candidate_metrics_by_objective": decoded_metric_record,
        "selected_mass_only_cog_error_tail_by_epoch": {
            key: value.tolist() for key, value in selected_mass_tail.items()
        },
        "selected_complete_cog_error_tail_by_epoch": {
            key: value.tolist() for key, value in selected_full_tail.items()
        },
        "raw_coordinate_residual_r2_complete_linear_by_epoch": np.stack(
            [item["residual_r2"] for item in relations["log"]["complete_linear"]]
        ).tolist(),
        "selected_parameter_clip_fraction_mass_only": float(
            np.mean(selected_mass_clipped)
        ),
        "selected_parameter_clip_fraction_complete": float(
            np.mean(selected_full_clipped)
        ),
        "predictor_design_condition_number_median_by_epoch": np.median(
            predictor_condition_numbers, axis=1
        ).tolist(),
        "diffmah_slope_boundary_fraction_by_epoch": np.mean(
            diffmah_slope_at_boundary, axis=0
        ).tolist(),
        "diffmah_slope_boundary_fraction_any_epoch": float(
            np.mean(np.any(diffmah_slope_at_boundary, axis=1))
        ),
        "log_absolute_objective_beta_correlation_by_epoch": objective_beta_correlation.tolist(),
        "log_absolute_objective_beta_sign_agreement_by_epoch": objective_beta_sign_agreement.tolist(),
        "median_raw_direct_response_correlation": float(
            np.nanmedian(response_correlation)
        ),
        "median_raw_direct_response_sign_agreement": float(
            np.nanmedian(response_sign_agreement)
        ),
        "wall_seconds": elapsed,
        "subminute_gate_passed": bool(elapsed < 60.0) if mode == "demo" else None,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_manifest(
        output_dir,
        {
            "experiment": "exp62_cog_fit_atlas",
            "stage": "6_cubic_logit_halo_information",
            "mode": mode,
            "n_galaxy": n_galaxy,
            "selected_relation": selected_relation,
            "stellar_mass_is_target": False,
        },
    )
    print(json.dumps(summary, indent=2))
    if mode == "demo" and elapsed >= 60.0:
        raise RuntimeError(
            f"Stage 6 demo took {elapsed:.2f} seconds; full run forbidden"
        )
    return summary


def mechanics() -> None:
    data = load_data("demo")
    assert len(data["galaxy_ids"]) == N_DEMO
    assert data["parameters_log"].shape == (N_DEMO, len(REDSHIFTS), 5)
    profiles, clipped = decode_cubic_shape_predictions(
        data["parameters_log"][:, 0, 0], data["parameters_log"][:, 0, 1:]
    )
    assert np.mean(clipped) < 0.02
    assert np.allclose(profiles[:, -1], data["parameters_log"][:, 0, 0])
    assert np.all(np.diff(profiles, axis=1) > 0.0)
    print("exp62 Stage 6 mechanics OK")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "mechanics"
    if command == "mechanics":
        mechanics()
    elif command in ("demo", "full"):
        run(command)
    elif command == "figures":
        figure_mode = sys.argv[2] if len(sys.argv) > 2 else "demo"
        make_figures(figure_mode)
    else:
        raise SystemExit(__doc__)
