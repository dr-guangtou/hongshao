# %%
"""Exp62 Stage 7: halo information in mass- and R50-normalized CoG diversity.

Commands:

    uv run python experiments/exp62_cog_fit_atlas/stage7.py mechanics
    uv run python experiments/exp62_cog_fit_atlas/stage7.py demo
    uv run python experiments/exp62_cog_fit_atlas/stage7.py full
    uv run python experiments/exp62_cog_fit_atlas/stage7.py figures demo|full

Measured Mstar(<148 kpc) and R50 are used only to remove amplitude and radial
scale. The calculation is a descriptive held-out information audit, not a
portable halo-to-profile emulator.
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
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import NullFormatter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from hongshao import qa  # noqa: E402
from hongshao.plotting import OKABE_ITO, save_fig, set_style  # noqa: E402
from hongshao.provenance import write_manifest  # noqa: E402
from run import OUTDIR, RADII, REDSHIFTS  # noqa: E402
from shape_diversity import (  # noqa: E402
    _common_support,
    _normalized_grid,
    normalized_cogs,
)
from stage4 import (  # noqa: E402
    galaxy_folds,
    mass_conditioned_shuffle,
    predictor_diagnostics,
)

set_style()

SOURCE = OUTDIR / "source" / "exp51_epoch_local_diffmah.npz"
STAGE7_OUTDIR = OUTDIR / "stage7"
STAGE7_FIGDIR = HERE / "figures" / "stage7"
PREDICTOR_NAMES = ("logmp", "logtc", "early_index", "late_index", "c_200c")
REGION_NAMES = ("inner", "outer", "equal_weight")
RELATION_NAMES = (
    "diffmah_linear",
    "concentration_linear",
    "complete_linear",
    "complete_additive_quadratic",
)
PRIMARY_Q_MINIMUM = 0.7
SENSITIVITY_Q_MINIMUM = 0.85
Q_MAXIMUM = 3.0
N_DEMO = 90
N_BOOTSTRAP = {"demo": 40, "full": 200}
N_SHUFFLE = {"demo": 8, "full": 32}
PCA_MODE = 3


def _baseline_design(
    train_mass: np.ndarray,
    evaluation_mass: np.ndarray,
    train_controls: np.ndarray | None,
    evaluation_controls: np.ndarray | None,
) -> np.ndarray:
    mass_mean = np.mean(train_mass)
    mass_scale = max(np.std(train_mass), 1.0e-8)
    standardized_mass = (evaluation_mass - mass_mean) / mass_scale
    columns = [np.ones(len(evaluation_mass)), standardized_mass, standardized_mass**2]
    if train_controls is not None:
        control_mean = np.mean(train_controls, axis=0)
        control_scale = np.clip(np.std(train_controls, axis=0), 1.0e-8, None)
        standardized_controls = (evaluation_controls - control_mean) / control_scale
        columns.extend(standardized_controls.T)
    return np.column_stack(columns)


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


def cross_fitted_profile_relation(
    targets: np.ndarray,
    mass: np.ndarray,
    controls: np.ndarray | None,
    features: np.ndarray,
    folds: np.ndarray,
    degree: int = 1,
    n_pca: int | None = None,
) -> dict[str, np.ndarray]:
    """Predict normalized profiles in held-out galaxy folds."""
    targets = np.asarray(targets, float)
    mass = np.asarray(mass, float)
    features = np.asarray(features, float)
    if controls is not None:
        controls = np.asarray(controls, float)
    unique_folds = np.unique(folds)
    baseline_prediction = np.empty_like(targets)
    full_prediction = np.empty_like(targets)
    fold_beta = np.empty(
        (len(unique_folds), features.shape[1] * degree, targets.shape[1])
    )
    for fold_index, fold in enumerate(unique_folds):
        train = folds != fold
        test = folds == fold
        train_controls = None if controls is None else controls[train]
        test_controls = None if controls is None else controls[test]
        train_design = _baseline_design(
            mass[train], mass[train], train_controls, train_controls
        )
        test_design = _baseline_design(
            mass[train], mass[test], train_controls, test_controls
        )

        if n_pca is None:
            target_mean = np.zeros(targets.shape[1])
            basis = np.eye(targets.shape[1])
        else:
            target_mean = np.mean(targets[train], axis=0)
            _, _, vectors = np.linalg.svd(
                targets[train] - target_mean, full_matrices=False
            )
            basis = vectors[:n_pca]
        train_representation = (targets[train] - target_mean) @ basis.T

        baseline_coefficient = np.linalg.lstsq(
            train_design, train_representation, rcond=None
        )[0]
        baseline_train = train_design @ baseline_coefficient
        baseline_test = test_design @ baseline_coefficient
        target_residual = train_representation - baseline_train

        train_basis_raw, test_basis_raw = _feature_basis(
            features[train], features[test], degree
        )
        basis_mass_coefficient = np.linalg.lstsq(
            train_design, train_basis_raw, rcond=None
        )[0]
        train_basis = train_basis_raw - train_design @ basis_mass_coefficient
        test_basis = test_basis_raw - test_design @ basis_mass_coefficient
        basis_mean = np.mean(train_basis, axis=0)
        basis_scale = np.clip(np.std(train_basis, axis=0), 1.0e-8, None)
        train_basis = (train_basis - basis_mean) / basis_scale
        test_basis = (test_basis - basis_mean) / basis_scale
        relation_coefficient = np.linalg.lstsq(
            np.column_stack([np.ones(train.sum()), train_basis]),
            target_residual,
            rcond=None,
        )[0]

        baseline_prediction[test] = target_mean + baseline_test @ basis
        full_representation = (
            baseline_test
            + np.column_stack([np.ones(test.sum()), test_basis]) @ relation_coefficient
        )
        full_prediction[test] = target_mean + full_representation @ basis
        fold_beta[fold_index] = relation_coefficient[1:] @ basis

    return {
        "baseline_prediction": baseline_prediction,
        "full_prediction": full_prediction,
        "fold_beta": fold_beta,
    }


def _region_loss(error: np.ndarray, coordinate: np.ndarray) -> np.ndarray:
    inner = coordinate < 1.0
    outer = coordinate > 1.0
    if not np.any(inner) or not np.any(outer):
        raise ValueError("both inner and outer coordinates are required")
    inner_loss = np.mean(error[..., inner] ** 2, axis=-1)
    outer_loss = np.mean(error[..., outer] ** 2, axis=-1)
    return np.stack([inner_loss, outer_loss, 0.5 * (inner_loss + outer_loss)], axis=-1)


def diversity_r2(
    truth: np.ndarray,
    baseline_prediction: np.ndarray,
    full_prediction: np.ndarray,
    coordinate: np.ndarray,
) -> np.ndarray:
    """Fraction of baseline residual diversity explained in three regions."""
    baseline_loss = np.sum(
        _region_loss(baseline_prediction - truth, coordinate), axis=0
    )
    full_loss = np.sum(_region_loss(full_prediction - truth, coordinate), axis=0)
    return 1.0 - full_loss / np.clip(baseline_loss, 1.0e-30, None)


def radial_diversity_r2(
    truth: np.ndarray,
    baseline_prediction: np.ndarray,
    full_prediction: np.ndarray,
) -> np.ndarray:
    baseline = np.sum((baseline_prediction - truth) ** 2, axis=0)
    full = np.sum((full_prediction - truth) ** 2, axis=0)
    output = 1.0 - full / np.clip(baseline, 1.0e-30, None)
    output[baseline < 1.0e-24] = np.nan
    return output


def coarse_annular_fractions(
    profiles_log: np.ndarray,
    normalized_radius: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
    profiles_log = np.asarray(profiles_log, float)
    original_shape = profiles_log.shape[:-1]
    flattened = profiles_log.reshape(-1, profiles_log.shape[-1])
    cumulative = np.asarray(
        [np.interp(edges, normalized_radius, 10.0**profile) for profile in flattened]
    )
    return np.diff(cumulative, axis=-1).reshape(*original_shape, len(edges) - 1)


def _stratified_rows(values: np.ndarray, count: int = N_DEMO) -> np.ndarray:
    order = np.argsort(values)
    positions = np.linspace(0, len(order) - 1, count).round().astype(int)
    return order[positions]


def _restore_anchor(values: np.ndarray, normalized_radius: np.ndarray) -> np.ndarray:
    anchor = int(np.flatnonzero(np.isclose(normalized_radius, 1.0))[0])
    output = np.empty((*values.shape[:-1], len(normalized_radius)))
    output[..., :anchor] = values[..., :anchor]
    output[..., anchor] = np.log10(0.5)
    output[..., anchor + 1 :] = values[..., anchor:]
    return output


def _without_anchor(values: np.ndarray, normalized_radius: np.ndarray) -> np.ndarray:
    return values[..., ~np.isclose(normalized_radius, 1.0)]


def load_data(mode: str, q_minimum: float = PRIMARY_Q_MINIMUM) -> dict[str, np.ndarray]:
    source = np.load(SOURCE)
    galaxy_ids = np.asarray(source["indices"], int)
    cogs_log = np.asarray(source["cogs_log"], float)
    r50 = qa._safe_rhalf(10.0**cogs_log, RADII, 0.5)
    selected = np.flatnonzero(_common_support(r50, q_minimum, Q_MAXIMUM))
    if mode == "demo":
        positions = _stratified_rows(np.asarray(source["epoch_halo_mass"])[selected, 0])
        selected = selected[positions]
    normalized_radius = _normalized_grid(q_minimum, Q_MAXIMUM)
    profiles = np.stack(
        [
            normalized_cogs(
                cogs_log[selected, epoch], r50[selected, epoch], normalized_radius
            )
            for epoch in range(len(REDSHIFTS))
        ],
        axis=1,
    )
    if not np.all(np.isfinite(profiles)):
        raise ValueError("common-support normalized profiles contain missing values")
    anchor = int(np.flatnonzero(np.isclose(normalized_radius, 1.0))[0])
    if not np.allclose(profiles[..., anchor], np.log10(0.5)):
        raise ValueError("R50-normalized profiles do not share the half-mass anchor")
    diffmah = np.asarray(source["parameters"])[selected]
    concentration = np.asarray(source["concentration"])[selected]
    log_mstar_148 = cogs_log[selected, :, -1]
    return {
        "galaxy_ids": galaxy_ids[selected],
        "halo_mass": np.asarray(source["epoch_halo_mass"])[selected],
        "features": np.concatenate([diffmah, concentration[..., None]], axis=-1),
        "strict_controls": np.stack([log_mstar_148, np.log10(r50[selected])], axis=-1),
        "profiles": profiles,
        "targets": _without_anchor(profiles, normalized_radius),
        "normalized_radius": normalized_radius,
        "target_radius": normalized_radius[~np.isclose(normalized_radius, 1.0)],
        "r50": r50[selected],
        "log_mstar_148": log_mstar_148,
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


def _bootstrap_diversity_r2(
    truth: np.ndarray,
    baseline_prediction: np.ndarray,
    full_prediction: np.ndarray,
    coordinate: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    output = np.empty((n_bootstrap, len(REGION_NAMES)))
    for bootstrap in range(n_bootstrap):
        rows = rng.integers(0, len(truth), len(truth))
        output[bootstrap] = diversity_r2(
            truth[rows],
            baseline_prediction[rows],
            full_prediction[rows],
            coordinate,
        )
    return output


def _bootstrap_radial_beta(
    targets: np.ndarray,
    mass: np.ndarray,
    features: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> np.ndarray:
    baseline_design = _baseline_design(mass, mass, None, None)
    target_coefficient = np.linalg.lstsq(baseline_design, targets, rcond=None)[0]
    target_residual = targets - baseline_design @ target_coefficient
    feature_coefficient = np.linalg.lstsq(baseline_design, features, rcond=None)[0]
    feature_residual = features - baseline_design @ feature_coefficient
    feature_residual = (feature_residual - np.mean(feature_residual, axis=0)) / np.clip(
        np.std(feature_residual, axis=0), 1.0e-8, None
    )
    rng = np.random.default_rng(seed)
    output = np.empty((n_bootstrap, features.shape[1], targets.shape[1]))
    for bootstrap in range(n_bootstrap):
        rows = rng.integers(0, len(targets), len(targets))
        design = np.column_stack([np.ones(len(rows)), feature_residual[rows]])
        output[bootstrap] = np.linalg.lstsq(design, target_residual[rows], rcond=None)[
            0
        ][1:]
    return output


def _standardized_feature_residuals(
    mass: np.ndarray, features: np.ndarray
) -> np.ndarray:
    design = _baseline_design(mass, mass, None, None)
    coefficient = np.linalg.lstsq(design, features, rcond=None)[0]
    residual = features - design @ coefficient
    return (residual - np.mean(residual, axis=0)) / np.clip(
        np.std(residual, axis=0), 1.0e-8, None
    )


def _fit_primary_sample(
    data: dict[str, np.ndarray], mode: str
) -> dict[str, np.ndarray]:
    n_epoch = len(REDSHIFTS)
    n_bootstrap = N_BOOTSTRAP[mode]
    n_shuffle = N_SHUFFLE[mode]
    folds = galaxy_folds(data["halo_mass"])
    n_galaxy, _, n_target = data["targets"].shape
    n_radius = len(data["normalized_radius"])
    n_region = len(REGION_NAMES)
    relation_r2 = np.empty((len(RELATION_NAMES), n_epoch, n_region))
    relation_predictions = np.empty((len(RELATION_NAMES), n_galaxy, n_epoch, n_radius))
    baseline_prediction = np.empty((n_galaxy, n_epoch, n_radius))
    strict_baseline_prediction = np.empty_like(baseline_prediction)
    strict_complete_prediction = np.empty_like(baseline_prediction)
    pca_baseline_prediction = np.empty_like(baseline_prediction)
    pca_complete_prediction = np.empty_like(baseline_prediction)
    pca_strict_baseline_prediction = np.empty_like(baseline_prediction)
    pca_strict_complete_prediction = np.empty_like(baseline_prediction)
    strict_r2 = np.empty((n_epoch, n_region))
    pca_r2 = np.empty_like(strict_r2)
    pca_strict_r2 = np.empty_like(strict_r2)
    quadratic_pca_r2 = np.empty_like(strict_r2)
    quadratic_strict_r2 = np.empty_like(strict_r2)
    radial_r2 = np.empty((n_epoch, n_radius))
    bootstrap_r2 = np.empty((n_epoch, n_bootstrap, n_region))
    quadratic_bootstrap_r2 = np.empty_like(bootstrap_r2)
    bootstrap_beta = np.empty((n_epoch, n_bootstrap, len(PREDICTOR_NAMES), n_target))
    shuffle_mah_r2 = np.empty((n_epoch, n_shuffle, n_region))
    shuffle_c_r2 = np.empty_like(shuffle_mah_r2)
    quadratic_shuffle_mah_r2 = np.empty_like(shuffle_mah_r2)
    quadratic_shuffle_c_r2 = np.empty_like(shuffle_mah_r2)
    predictor_condition = np.empty(n_epoch)
    feature_residuals = np.empty_like(data["features"])

    edges = np.asarray([data["normalized_radius"][0], 0.85, 1.0, 1.5, 2.0, Q_MAXIMUM])
    edges = np.unique(edges)
    annular_coordinate = np.sqrt(edges[:-1] * edges[1:])
    annular_r2 = np.empty((n_epoch, n_region))
    annular_strict_r2 = np.empty_like(annular_r2)
    quadratic_annular_r2 = np.empty_like(annular_r2)
    rng = np.random.default_rng(6270)

    for epoch, redshift in enumerate(REDSHIFTS):
        truth_target = data["targets"][:, epoch]
        truth_profile = data["profiles"][:, epoch]
        mass = data["halo_mass"][:, epoch]
        features = data["features"][:, epoch]
        feature_residuals[:, epoch] = _standardized_feature_residuals(mass, features)
        predictor_condition[epoch] = predictor_diagnostics(mass, features)[1]

        epoch_results = {}
        for relation_index, relation_name in enumerate(RELATION_NAMES):
            columns, degree = _relation_specification(relation_name)
            relation = cross_fitted_profile_relation(
                truth_target,
                mass,
                None,
                features[:, columns],
                folds,
                degree=degree,
            )
            epoch_results[relation_name] = relation
            baseline = _restore_anchor(
                relation["baseline_prediction"], data["normalized_radius"]
            )
            full = _restore_anchor(
                relation["full_prediction"], data["normalized_radius"]
            )
            relation_predictions[relation_index, :, epoch] = full
            relation_r2[relation_index, epoch] = diversity_r2(
                truth_profile, baseline, full, data["normalized_radius"]
            )
            if relation_name == "complete_linear":
                baseline_prediction[:, epoch] = baseline
                radial_r2[epoch] = radial_diversity_r2(truth_profile, baseline, full)

        complete = relation_predictions[
            RELATION_NAMES.index("complete_linear"), :, epoch
        ]
        quadratic_complete = relation_predictions[
            RELATION_NAMES.index("complete_additive_quadratic"), :, epoch
        ]
        strict = cross_fitted_profile_relation(
            truth_target,
            mass,
            data["strict_controls"][:, epoch],
            features,
            folds,
            degree=1,
        )
        strict_baseline_prediction[:, epoch] = _restore_anchor(
            strict["baseline_prediction"], data["normalized_radius"]
        )
        strict_complete_prediction[:, epoch] = _restore_anchor(
            strict["full_prediction"], data["normalized_radius"]
        )
        strict_r2[epoch] = diversity_r2(
            truth_profile,
            strict_baseline_prediction[:, epoch],
            strict_complete_prediction[:, epoch],
            data["normalized_radius"],
        )

        pca = cross_fitted_profile_relation(
            truth_target,
            mass,
            None,
            features,
            folds,
            degree=1,
            n_pca=PCA_MODE,
        )
        pca_baseline_prediction[:, epoch] = _restore_anchor(
            pca["baseline_prediction"], data["normalized_radius"]
        )
        pca_complete_prediction[:, epoch] = _restore_anchor(
            pca["full_prediction"], data["normalized_radius"]
        )
        pca_r2[epoch] = diversity_r2(
            truth_profile,
            pca_baseline_prediction[:, epoch],
            pca_complete_prediction[:, epoch],
            data["normalized_radius"],
        )

        quadratic_pca = cross_fitted_profile_relation(
            truth_target,
            mass,
            None,
            features,
            folds,
            degree=2,
            n_pca=PCA_MODE,
        )
        quadratic_pca_baseline = _restore_anchor(
            quadratic_pca["baseline_prediction"], data["normalized_radius"]
        )
        quadratic_pca_complete = _restore_anchor(
            quadratic_pca["full_prediction"], data["normalized_radius"]
        )
        quadratic_pca_r2[epoch] = diversity_r2(
            truth_profile,
            quadratic_pca_baseline,
            quadratic_pca_complete,
            data["normalized_radius"],
        )

        pca_strict = cross_fitted_profile_relation(
            truth_target,
            mass,
            data["strict_controls"][:, epoch],
            features,
            folds,
            degree=1,
            n_pca=PCA_MODE,
        )
        pca_strict_baseline_prediction[:, epoch] = _restore_anchor(
            pca_strict["baseline_prediction"], data["normalized_radius"]
        )
        pca_strict_complete_prediction[:, epoch] = _restore_anchor(
            pca_strict["full_prediction"], data["normalized_radius"]
        )
        pca_strict_r2[epoch] = diversity_r2(
            truth_profile,
            pca_strict_baseline_prediction[:, epoch],
            pca_strict_complete_prediction[:, epoch],
            data["normalized_radius"],
        )

        quadratic_strict = cross_fitted_profile_relation(
            truth_target,
            mass,
            data["strict_controls"][:, epoch],
            features,
            folds,
            degree=2,
        )
        quadratic_strict_baseline = _restore_anchor(
            quadratic_strict["baseline_prediction"], data["normalized_radius"]
        )
        quadratic_strict_complete = _restore_anchor(
            quadratic_strict["full_prediction"], data["normalized_radius"]
        )
        quadratic_strict_r2[epoch] = diversity_r2(
            truth_profile,
            quadratic_strict_baseline,
            quadratic_strict_complete,
            data["normalized_radius"],
        )

        bootstrap_r2[epoch] = _bootstrap_diversity_r2(
            truth_profile,
            baseline_prediction[:, epoch],
            complete,
            data["normalized_radius"],
            n_bootstrap,
            6700 + epoch,
        )
        quadratic_bootstrap_r2[epoch] = _bootstrap_diversity_r2(
            truth_profile,
            baseline_prediction[:, epoch],
            quadratic_complete,
            data["normalized_radius"],
            n_bootstrap,
            6750 + epoch,
        )
        bootstrap_beta[epoch] = _bootstrap_radial_beta(
            truth_target,
            mass,
            features,
            n_bootstrap,
            6800 + epoch,
        )

        truth_annular = coarse_annular_fractions(
            truth_profile, data["normalized_radius"], edges
        )
        baseline_annular = coarse_annular_fractions(
            baseline_prediction[:, epoch], data["normalized_radius"], edges
        )
        complete_annular = coarse_annular_fractions(
            complete, data["normalized_radius"], edges
        )
        quadratic_annular = coarse_annular_fractions(
            quadratic_complete, data["normalized_radius"], edges
        )
        strict_baseline_annular = coarse_annular_fractions(
            strict_baseline_prediction[:, epoch], data["normalized_radius"], edges
        )
        strict_complete_annular = coarse_annular_fractions(
            strict_complete_prediction[:, epoch], data["normalized_radius"], edges
        )
        annular_r2[epoch] = diversity_r2(
            truth_annular,
            baseline_annular,
            complete_annular,
            annular_coordinate,
        )
        quadratic_annular_r2[epoch] = diversity_r2(
            truth_annular,
            baseline_annular,
            quadratic_annular,
            annular_coordinate,
        )
        annular_strict_r2[epoch] = diversity_r2(
            truth_annular,
            strict_baseline_annular,
            strict_complete_annular,
            annular_coordinate,
        )

        for shuffle in range(n_shuffle):
            shuffled_mah = mass_conditioned_shuffle(features, mass, (0, 1, 2, 3), rng)
            shuffled_c = mass_conditioned_shuffle(features, mass, (4,), rng)
            for shuffled, store in (
                (shuffled_mah, shuffle_mah_r2),
                (shuffled_c, shuffle_c_r2),
            ):
                shuffled_result = cross_fitted_profile_relation(
                    truth_target,
                    mass,
                    None,
                    shuffled,
                    folds,
                    degree=1,
                )
                shuffled_prediction = _restore_anchor(
                    shuffled_result["full_prediction"],
                    data["normalized_radius"],
                )
                store[epoch, shuffle] = diversity_r2(
                    truth_profile,
                    baseline_prediction[:, epoch],
                    shuffled_prediction,
                    data["normalized_radius"],
                )
            for shuffled, store in (
                (shuffled_mah, quadratic_shuffle_mah_r2),
                (shuffled_c, quadratic_shuffle_c_r2),
            ):
                shuffled_result = cross_fitted_profile_relation(
                    truth_target,
                    mass,
                    None,
                    shuffled,
                    folds,
                    degree=2,
                )
                shuffled_prediction = _restore_anchor(
                    shuffled_result["full_prediction"],
                    data["normalized_radius"],
                )
                store[epoch, shuffle] = diversity_r2(
                    truth_profile,
                    baseline_prediction[:, epoch],
                    shuffled_prediction,
                    data["normalized_radius"],
                )
        print(
            f"  z={redshift:g}: direct complete diversity R2 "
            f"inner={relation_r2[2, epoch, 0]:+.3f}, "
            f"outer={relation_r2[2, epoch, 1]:+.3f}; "
            f"PCA inner={pca_r2[epoch, 0]:+.3f}, "
            f"outer={pca_r2[epoch, 1]:+.3f}",
            flush=True,
        )

    diffmah_slopes = data["features"][..., 2:4]
    diffmah_boundary = np.any(
        (diffmah_slopes <= 0.05001) | (diffmah_slopes >= 11.99999), axis=-1
    )
    return {
        "folds": folds,
        "relation_r2": relation_r2,
        "relation_predictions": relation_predictions,
        "baseline_prediction": baseline_prediction,
        "strict_baseline_prediction": strict_baseline_prediction,
        "strict_complete_prediction": strict_complete_prediction,
        "pca_baseline_prediction": pca_baseline_prediction,
        "pca_complete_prediction": pca_complete_prediction,
        "pca_strict_baseline_prediction": pca_strict_baseline_prediction,
        "pca_strict_complete_prediction": pca_strict_complete_prediction,
        "strict_r2": strict_r2,
        "pca_r2": pca_r2,
        "pca_strict_r2": pca_strict_r2,
        "quadratic_pca_r2": quadratic_pca_r2,
        "quadratic_strict_r2": quadratic_strict_r2,
        "radial_r2": radial_r2,
        "bootstrap_r2": bootstrap_r2,
        "quadratic_bootstrap_r2": quadratic_bootstrap_r2,
        "bootstrap_beta": bootstrap_beta,
        "shuffle_mah_r2": shuffle_mah_r2,
        "shuffle_c_r2": shuffle_c_r2,
        "quadratic_shuffle_mah_r2": quadratic_shuffle_mah_r2,
        "quadratic_shuffle_c_r2": quadratic_shuffle_c_r2,
        "predictor_condition": predictor_condition,
        "feature_residuals": feature_residuals,
        "annular_edges": edges,
        "annular_coordinate": annular_coordinate,
        "annular_r2": annular_r2,
        "annular_strict_r2": annular_strict_r2,
        "quadratic_annular_r2": quadratic_annular_r2,
        "diffmah_boundary": diffmah_boundary,
    }


def _fit_sensitivity_sample(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    folds = galaxy_folds(data["halo_mass"])
    direct_r2 = np.empty((len(REDSHIFTS), len(REGION_NAMES)))
    strict_r2 = np.empty_like(direct_r2)
    pca_r2 = np.empty_like(direct_r2)
    quadratic_r2 = np.empty_like(direct_r2)
    quadratic_pca_r2 = np.empty_like(direct_r2)
    for epoch in range(len(REDSHIFTS)):
        truth = data["profiles"][:, epoch]
        for controls, pca_modes, store in (
            (None, None, direct_r2),
            (data["strict_controls"][:, epoch], None, strict_r2),
            (None, PCA_MODE, pca_r2),
        ):
            relation = cross_fitted_profile_relation(
                data["targets"][:, epoch],
                data["halo_mass"][:, epoch],
                controls,
                data["features"][:, epoch],
                folds,
                degree=1,
                n_pca=pca_modes,
            )
            baseline = _restore_anchor(
                relation["baseline_prediction"], data["normalized_radius"]
            )
            full = _restore_anchor(
                relation["full_prediction"], data["normalized_radius"]
            )
            store[epoch] = diversity_r2(
                truth, baseline, full, data["normalized_radius"]
            )
        for pca_modes, store in (
            (None, quadratic_r2),
            (PCA_MODE, quadratic_pca_r2),
        ):
            relation = cross_fitted_profile_relation(
                data["targets"][:, epoch],
                data["halo_mass"][:, epoch],
                None,
                data["features"][:, epoch],
                folds,
                degree=2,
                n_pca=pca_modes,
            )
            baseline = _restore_anchor(
                relation["baseline_prediction"], data["normalized_radius"]
            )
            full = _restore_anchor(
                relation["full_prediction"], data["normalized_radius"]
            )
            store[epoch] = diversity_r2(
                truth, baseline, full, data["normalized_radius"]
            )
    return {
        "direct_r2": direct_r2,
        "strict_r2": strict_r2,
        "pca_r2": pca_r2,
        "quadratic_r2": quadratic_r2,
        "quadratic_pca_r2": quadratic_pca_r2,
    }


def _diversity_support_figure(mode: str, data: dict[str, np.ndarray]) -> None:
    source = np.load(SOURCE)
    cogs = np.asarray(source["cogs_log"])
    r50 = qa._safe_rhalf(10.0**cogs, RADII, 0.5)
    display_radius = _normalized_grid(0.5, 5.0, 40)
    colors = (OKABE_ITO[0], OKABE_ITO[1], OKABE_ITO[2], OKABE_ITO[4], OKABE_ITO[5])
    figure, axes = plt.subplots(2, len(REDSHIFTS), figsize=(17.0, 7.0), sharex=True)
    for epoch, (redshift, color) in enumerate(zip(REDSHIFTS, colors)):
        profiles = normalized_cogs(cogs[:, epoch], r50[:, epoch], display_radius)
        coverage = np.mean(np.isfinite(profiles), axis=0)
        lower, median, upper = np.nanquantile(profiles, [0.16, 0.5, 0.84], axis=0)
        axes[0, epoch].fill_between(
            display_radius, lower, upper, color=color, alpha=0.18
        )
        axes[0, epoch].plot(display_radius, median, color=color, linewidth=1.8)
        axes[1, epoch].plot(display_radius, coverage, color=color, linewidth=1.8)
        axes[1, epoch].axhline(0.9, color="0.5", linestyle=":", linewidth=0.8)
        axes[0, epoch].set_title(rf"$z={redshift:g}$")
        for axis in axes[:, epoch]:
            axis.axvspan(PRIMARY_Q_MINIMUM, Q_MAXIMUM, color="0.7", alpha=0.08)
            axis.axvline(1.0, color="black", linewidth=0.7)
            axis.set_xscale("log")
            axis.set_xlim(display_radius[0], display_radius[-1])
            axis.set_xticks((0.5, 0.7, 1.0, 2.0, 3.0, 5.0))
            axis.xaxis.set_minor_formatter(NullFormatter())
    axes[0, 0].set_ylabel(r"$\log_{10}[M_*(<R)/M_{*,148}]$")
    axes[1, 0].set_ylabel("Usable fraction")
    for axis in axes[1]:
        axis.set_ylim(0.0, 1.03)
        axis.set_xlabel(r"Normalized radius $R/R_{50}$")
    figure.suptitle(
        f"Exp62 Stage 7 {mode}: measured scale-free CoG diversity and radial support"
    )
    figure.text(
        0.5,
        0.01,
        f"Gray span is the frozen {PRIMARY_Q_MINIMUM:g}--{Q_MAXIMUM:g} R50 interval; primary common-support N={len(data['galaxy_ids'])}.",
        ha="center",
        fontsize=7.5,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 1))
    save_fig(figure, STAGE7_FIGDIR / mode / "exp62_stage7_diversity_support")
    plt.close(figure)


def _radial_information_figure(
    mode: str, data: dict[str, np.ndarray], result: dict[str, np.ndarray]
) -> None:
    quadratic = result["relation_predictions"][
        RELATION_NAMES.index("complete_additive_quadratic")
    ]
    quadratic_radial = np.stack(
        [
            radial_diversity_r2(
                data["profiles"][:, epoch],
                result["baseline_prediction"][:, epoch],
                quadratic[:, epoch],
            )
            for epoch in range(len(REDSHIFTS))
        ]
    )
    pca_radial = np.stack(
        [
            radial_diversity_r2(
                data["profiles"][:, epoch],
                result["pca_baseline_prediction"][:, epoch],
                result["pca_complete_prediction"][:, epoch],
            )
            for epoch in range(len(REDSHIFTS))
        ]
    )
    strict_radial = np.stack(
        [
            radial_diversity_r2(
                data["profiles"][:, epoch],
                result["strict_baseline_prediction"][:, epoch],
                result["strict_complete_prediction"][:, epoch],
            )
            for epoch in range(len(REDSHIFTS))
        ]
    )
    values = (result["radial_r2"], quadratic_radial, strict_radial, pca_radial)
    labels = (
        "Direct; halo-mass baseline",
        "Direct; additive quadratic",
        "Direct; strict baseline",
        "Fold-local PCA(3)",
    )
    finite = np.concatenate([value[np.isfinite(value)] for value in values])
    limit = max(float(np.quantile(np.abs(finite), 0.98)), 0.02)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    figure, axes = plt.subplots(
        4, 1, figsize=(12.0, 9.2), sharex=True, layout="constrained"
    )
    for axis, value, label in zip(axes, values, labels):
        image = axis.pcolormesh(
            data["normalized_radius"],
            REDSHIFTS,
            value,
            cmap="PuOr",
            norm=norm,
            shading="nearest",
        )
        axis.axvline(1.0, color="black", linewidth=0.8)
        axis.set_ylabel("Redshift")
        axis.set_title(label)
    axes[-1].set_xscale("log")
    axes[-1].set_xlabel(r"Normalized radius $R/R_{50}$")
    figure.colorbar(image, ax=axes, label=r"Held-out diversity $R^2$", shrink=0.82)
    figure.suptitle(
        f"Exp62 Stage 7 {mode}: radial location of assembly-predictable scale-free diversity"
    )
    save_fig(figure, STAGE7_FIGDIR / mode / "exp62_stage7_radial_information")
    plt.close(figure)


def _integrated_information_figure(mode: str, result: dict[str, np.ndarray]) -> None:
    complete_r2 = result["relation_r2"][RELATION_NAMES.index("complete_linear")]
    quadratic_r2 = result["relation_r2"][
        RELATION_NAMES.index("complete_additive_quadratic")
    ]
    lower, upper = np.quantile(result["quadratic_bootstrap_r2"], [0.025, 0.975], axis=1)
    shuffle_upper = np.quantile(result["quadratic_shuffle_mah_r2"], 0.95, axis=1)
    figure, axes = plt.subplots(
        1, 3, figsize=(14.5, 4.6), sharex=True, layout="constrained"
    )
    for region, axis in enumerate(axes):
        axis.fill_between(
            REDSHIFTS,
            lower[:, region],
            upper[:, region],
            color=OKABE_ITO[2],
            alpha=0.16,
            label="Quadratic bootstrap 95%",
        )
        axis.plot(
            REDSHIFTS,
            complete_r2[:, region],
            "o--",
            color=OKABE_ITO[2],
            label="Direct linear",
        )
        axis.plot(
            REDSHIFTS,
            quadratic_r2[:, region],
            "d-",
            color=OKABE_ITO[4],
            label="Direct quadratic",
        )
        axis.plot(
            REDSHIFTS,
            result["quadratic_pca_r2"][:, region],
            "s:",
            color=OKABE_ITO[1],
            label="Quadratic PCA(3)",
        )
        axis.plot(
            REDSHIFTS,
            result["quadratic_strict_r2"][:, region],
            "^-.",
            color=OKABE_ITO[5],
            label="Quadratic strict baseline",
        )
        axis.plot(
            REDSHIFTS,
            shuffle_upper[:, region],
            ":",
            color="0.35",
            label="Quadratic shuffled-DiffMAH 95th",
        )
        axis.axhline(0.0, color="0.5", linewidth=0.8)
        axis.axhline(0.05, color="0.5", linestyle="--", linewidth=0.8)
        axis.set_title(REGION_NAMES[region].replace("_", " "))
        axis.set_xlabel("Redshift")
        axis.invert_xaxis()
    axes[0].set_ylabel(r"Held-out diversity $R^2$")
    axes[0].legend(frameon=False, fontsize=7.0)
    figure.suptitle(
        f"Exp62 Stage 7 {mode}: scale-free CoG diversity explained beyond the baseline"
    )
    save_fig(figure, STAGE7_FIGDIR / mode / "exp62_stage7_integrated_information")
    plt.close(figure)


def _average_mass_bin_figure(
    mode: str, data: dict[str, np.ndarray], result: dict[str, np.ndarray]
) -> None:
    complete = result["relation_predictions"][RELATION_NAMES.index("complete_linear")]
    quadratic = result["relation_predictions"][
        RELATION_NAMES.index("complete_additive_quadratic")
    ]
    figure, axes = plt.subplots(
        2, len(REDSHIFTS), figsize=(17.0, 7.0), sharex=True, sharey="row"
    )
    colors = (OKABE_ITO[0], OKABE_ITO[2], OKABE_ITO[4])
    for epoch, redshift in enumerate(REDSHIFTS):
        mass = data["halo_mass"][:, epoch]
        edges = np.quantile(mass, [0.0, 1 / 3, 2 / 3, 1.0])
        for mass_bin, color in enumerate(colors):
            selected = (mass >= edges[mass_bin]) & (mass <= edges[mass_bin + 1])
            truth = np.median(data["profiles"][selected, epoch], axis=0)
            baseline = np.median(result["baseline_prediction"][selected, epoch], axis=0)
            full = np.median(complete[selected, epoch], axis=0)
            quadratic_full = np.median(quadratic[selected, epoch], axis=0)
            label = rf"${edges[mass_bin]:.2f}<\log M_h<{edges[mass_bin + 1]:.2f}$"
            axes[0, epoch].plot(data["normalized_radius"], truth, ":", color=color)
            axes[0, epoch].plot(data["normalized_radius"], baseline, "--", color=color)
            axes[0, epoch].plot(
                data["normalized_radius"], full, color=color, label=label
            )
            axes[0, epoch].plot(
                data["normalized_radius"], quadratic_full, "-.", color=color
            )
            axes[1, epoch].plot(
                data["normalized_radius"], baseline - truth, "--", color=color
            )
            axes[1, epoch].plot(data["normalized_radius"], full - truth, color=color)
            axes[1, epoch].plot(
                data["normalized_radius"],
                quadratic_full - truth,
                "-.",
                color=color,
            )
        axes[0, epoch].set_title(rf"$z={redshift:g}$")
        for axis in axes[:, epoch]:
            axis.axvline(1.0, color="black", linewidth=0.7)
            axis.axhline(0.0, color="0.6", linewidth=0.6)
            axis.set_xscale("log")
    axes[0, 0].set_ylabel("Normalized log CoG")
    axes[1, 0].set_ylabel("Median residual (dex)")
    for axis in axes[1]:
        axis.set_xlabel(r"$R/R_{50}$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        fontsize=7.0,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=3,
    )
    figure.text(
        0.5,
        0.905,
        "Solid: linear; dash-dot: quadratic; dashed: halo mass only; dotted: TNG",
        ha="center",
        fontsize=8.0,
    )
    figure.suptitle(
        f"Exp62 Stage 7 {mode}: average normalized CoGs in concurrent halo-mass bins",
        y=0.995,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    save_fig(figure, STAGE7_FIGDIR / mode / "exp62_stage7_average_cog_halo_mass_bins")
    plt.close(figure)


def _predictor_quantile_figure(
    mode: str, data: dict[str, np.ndarray], result: dict[str, np.ndarray]
) -> None:
    complete = result["relation_predictions"][RELATION_NAMES.index("complete_linear")]
    quadratic = result["relation_predictions"][
        RELATION_NAMES.index("complete_additive_quadratic")
    ]
    figure, axes = plt.subplots(
        4, len(REDSHIFTS), figsize=(17.0, 11.5), sharex=True, sharey="row"
    )
    for predictor in range(4):
        for epoch, redshift in enumerate(REDSHIFTS):
            coordinate = result["feature_residuals"][:, epoch, predictor]
            low, high = np.quantile(coordinate, [0.25, 0.75])
            for selected, color, label in (
                (coordinate <= low, OKABE_ITO[1], "bottom quartile"),
                (coordinate >= high, OKABE_ITO[5], "top quartile"),
            ):
                truth_residual = (
                    data["profiles"][selected, epoch]
                    - result["baseline_prediction"][selected, epoch]
                )
                model_residual = (
                    complete[selected, epoch]
                    - result["baseline_prediction"][selected, epoch]
                )
                quadratic_residual = (
                    quadratic[selected, epoch]
                    - result["baseline_prediction"][selected, epoch]
                )
                axes[predictor, epoch].plot(
                    data["normalized_radius"],
                    np.median(truth_residual, axis=0),
                    color=color,
                    label=label,
                )
                axes[predictor, epoch].plot(
                    data["normalized_radius"],
                    np.median(model_residual, axis=0),
                    "--",
                    color=color,
                )
                axes[predictor, epoch].plot(
                    data["normalized_radius"],
                    np.median(quadratic_residual, axis=0),
                    ":",
                    color=color,
                )
            axes[predictor, epoch].axhline(0.0, color="0.6", linewidth=0.6)
            axes[predictor, epoch].axvline(1.0, color="black", linewidth=0.7)
            axes[predictor, epoch].set_xscale("log")
            if predictor == 0:
                axes[predictor, epoch].set_title(rf"$z={redshift:g}$")
            if epoch == 0:
                axes[predictor, epoch].set_ylabel(
                    f"{PREDICTOR_NAMES[predictor]}\nresidual (dex)"
                )
    for axis in axes[-1]:
        axis.set_xlabel(r"$R/R_{50}$")
    axes[0, 0].legend(
        frameon=False,
        fontsize=6.8,
        title="Solid: TNG; dashed: linear; dotted: quadratic",
    )
    figure.suptitle(
        f"Exp62 Stage 7 {mode}: normalized-CoG residuals split by mass-conditioned DiffMAH quartile"
    )
    figure.tight_layout()
    save_fig(figure, STAGE7_FIGDIR / mode / "exp62_stage7_diffmah_quantile_profiles")
    plt.close(figure)


def _radial_response_figure(
    mode: str, data: dict[str, np.ndarray], result: dict[str, np.ndarray]
) -> None:
    colors = (OKABE_ITO[0], OKABE_ITO[1], OKABE_ITO[2], OKABE_ITO[4], OKABE_ITO[5])
    figure, axes = plt.subplots(
        1, len(PREDICTOR_NAMES), figsize=(17.0, 4.2), sharey=True
    )
    for predictor, axis in enumerate(axes):
        for epoch, (redshift, color) in enumerate(zip(REDSHIFTS, colors)):
            beta = result["bootstrap_beta"][epoch, :, predictor]
            median = np.median(beta, axis=0)
            lower, upper = np.quantile(beta, [0.16, 0.84], axis=0)
            axis.plot(
                data["target_radius"], median, color=color, label=rf"$z={redshift:g}$"
            )
            axis.fill_between(
                data["target_radius"], lower, upper, color=color, alpha=0.10
            )
        axis.axhline(0.0, color="0.5", linewidth=0.7)
        axis.axvline(1.0, color="black", linewidth=0.7)
        axis.set_xscale("log")
        axis.set_title(PREDICTOR_NAMES[predictor])
        axis.set_xlabel(r"$R/R_{50}$")
    axes[0].set_ylabel("Normalized CoG response (dex)")
    axes[0].legend(frameon=False, fontsize=7.0)
    figure.suptitle(
        f"Exp62 Stage 7 {mode}: response to +1 mass-conditioned halo-predictor SD; bands are bootstrap 16--84%"
    )
    figure.tight_layout()
    save_fig(figure, STAGE7_FIGDIR / mode / "exp62_stage7_radial_responses")
    plt.close(figure)


def _cdf_residual_figure(
    mode: str, data: dict[str, np.ndarray], result: dict[str, np.ndarray]
) -> None:
    complete = result["relation_predictions"][RELATION_NAMES.index("complete_linear")]
    quadratic = result["relation_predictions"][
        RELATION_NAMES.index("complete_additive_quadratic")
    ]
    q_minimum = data["normalized_radius"][0]
    quantity_names = (
        rf"M(<{q_minimum:g}R50)",
        "M(<2R50)",
        "M(0.7--1R50)",
        "M(1--3R50)",
    )
    colors = (OKABE_ITO[0], OKABE_ITO[1], OKABE_ITO[2], OKABE_ITO[5])
    figure, axes = plt.subplots(1, len(REDSHIFTS), figsize=(17.0, 3.8), sharey=True)
    for epoch, axis in enumerate(axes):
        sets = []
        for profiles in (
            data["profiles"][:, epoch],
            result["baseline_prediction"][:, epoch],
            complete[:, epoch],
            quadratic[:, epoch],
        ):
            fraction = 10.0**profiles
            q_values = np.asarray(
                [
                    np.interp(
                        [q_minimum, 1.0, 2.0, 3.0], data["normalized_radius"], row
                    )
                    for row in fraction
                ]
            )
            sets.append(
                np.column_stack(
                    [
                        q_values[:, 0],
                        q_values[:, 2],
                        q_values[:, 1] - q_values[:, 0],
                        q_values[:, 3] - q_values[:, 1],
                    ]
                )
            )
        truth, baseline, full, quadratic_full = sets
        for quantity, (name, color) in enumerate(zip(quantity_names, colors)):
            low = np.quantile(
                np.concatenate(
                    [
                        truth[:, quantity],
                        baseline[:, quantity],
                        full[:, quantity],
                        quadratic_full[:, quantity],
                    ]
                ),
                0.005,
            )
            high = np.quantile(
                np.concatenate(
                    [
                        truth[:, quantity],
                        baseline[:, quantity],
                        full[:, quantity],
                        quadratic_full[:, quantity],
                    ]
                ),
                0.995,
            )
            x = np.linspace(low, high, 180)
            truth_cdf = np.mean(truth[:, quantity, None] <= x[None, :], axis=0)
            baseline_cdf = np.mean(baseline[:, quantity, None] <= x[None, :], axis=0)
            full_cdf = np.mean(full[:, quantity, None] <= x[None, :], axis=0)
            quadratic_cdf = np.mean(
                quadratic_full[:, quantity, None] <= x[None, :], axis=0
            )
            axis.plot(x, baseline_cdf - truth_cdf, "--", color=color)
            axis.plot(x, full_cdf - truth_cdf, color=color, label=name)
            axis.plot(x, quadratic_cdf - truth_cdf, ":", color=color)
        axis.axhline(0.0, color="0.5", linewidth=0.7)
        axis.set_title(rf"$z={REDSHIFTS[epoch]:g}$")
        axis.set_xlabel("Normalized stellar-mass fraction")
    axes[0].set_ylabel("Model CDF - TNG CDF")
    axes[0].legend(
        frameon=False,
        fontsize=6.5,
        title="Solid: linear; dotted: quadratic; dashed: mass only",
    )
    figure.suptitle(
        f"Exp62 Stage 7 {mode}: normalized aperture and annular mass CDF residuals"
    )
    figure.tight_layout()
    save_fig(figure, STAGE7_FIGDIR / mode / "exp62_stage7_mass_fraction_cdf_residuals")
    plt.close(figure)


def _robustness_figure(
    mode: str,
    result: dict[str, np.ndarray],
    sensitivity: dict[str, np.ndarray],
) -> None:
    direct = result["relation_r2"][RELATION_NAMES.index("complete_linear")]
    quadratic = result["relation_r2"][
        RELATION_NAMES.index("complete_additive_quadratic")
    ]
    figure, axes = plt.subplots(
        1, 3, figsize=(14.5, 4.2), sharex=True, layout="constrained"
    )
    for region, axis in enumerate(axes):
        for values, label, color, style in (
            (direct, "Linear; 0.7--3 R50", OKABE_ITO[2], "o--"),
            (quadratic, "Quadratic; 0.7--3 R50", OKABE_ITO[4], "d-"),
            (
                result["quadratic_strict_r2"],
                "Quadratic strict baseline",
                OKABE_ITO[5],
                "^-.",
            ),
            (
                result["quadratic_pca_r2"],
                "Quadratic PCA(3)",
                OKABE_ITO[1],
                "s:",
            ),
            (
                sensitivity["quadratic_r2"],
                "Quadratic; 0.85--3 R50",
                OKABE_ITO[0],
                "v:",
            ),
        ):
            axis.plot(REDSHIFTS, values[:, region], style, color=color, label=label)
        axis.axhline(0.0, color="0.5", linewidth=0.7)
        axis.set_title(REGION_NAMES[region].replace("_", " "))
        axis.set_xlabel("Redshift")
        axis.invert_xaxis()
    axes[0].set_ylabel(r"Held-out diversity $R^2$")
    axes[0].legend(frameon=False, fontsize=6.8)
    figure.suptitle(
        f"Exp62 Stage 7 {mode}: representation, baseline, and radial-support checks"
    )
    save_fig(figure, STAGE7_FIGDIR / mode / "exp62_stage7_robustness_comparison")
    plt.close(figure)


def _worst_case_figure(
    mode: str, data: dict[str, np.ndarray], result: dict[str, np.ndarray]
) -> None:
    complete = result["relation_predictions"][RELATION_NAMES.index("complete_linear")]
    quadratic = result["relation_predictions"][
        RELATION_NAMES.index("complete_additive_quadratic")
    ]
    error = quadratic - data["profiles"]
    per_pair = _region_loss(error, data["normalized_radius"])[..., 2]
    selected = np.argsort(per_pair.ravel())[-10:][::-1]
    figure, axes = plt.subplots(5, 2, figsize=(12.0, 16.0), sharex=True)
    for axis, flat in zip(axes.ravel(), selected):
        galaxy, epoch = np.unravel_index(flat, per_pair.shape)
        axis.plot(
            data["normalized_radius"],
            data["profiles"][galaxy, epoch],
            color="black",
            label="TNG",
        )
        axis.plot(
            data["normalized_radius"],
            result["baseline_prediction"][galaxy, epoch],
            "--",
            color=OKABE_ITO[0],
            label="Halo mass only",
        )
        axis.plot(
            data["normalized_radius"],
            complete[galaxy, epoch],
            color=OKABE_ITO[2],
            label="Linear",
        )
        axis.plot(
            data["normalized_radius"],
            quadratic[galaxy, epoch],
            color=OKABE_ITO[4],
            label="Quadratic",
        )
        axis.axvline(1.0, color="0.5", linewidth=0.7)
        axis.set_xscale("log")
        axis.set_title(
            f"galaxy {data['galaxy_ids'][galaxy]}; z={REDSHIFTS[epoch]:g}; quadratic equal-weight RMS={np.sqrt(per_pair[galaxy, epoch]):.3f} dex"
        )
        axis.set_ylabel("Normalized log CoG")
    for axis in axes[-1]:
        axis.set_xlabel(r"$R/R_{50}$")
    axes[0, 0].legend(frameon=False, fontsize=7.0)
    figure.suptitle(f"Exp62 Stage 7 {mode}: ten largest held-out normalized-CoG errors")
    figure.tight_layout()
    save_fig(figure, STAGE7_FIGDIR / mode / "exp62_stage7_worst10")
    plt.close(figure)


def make_figures(
    mode: str,
    data: dict[str, np.ndarray],
    result: dict[str, np.ndarray],
    sensitivity: dict[str, np.ndarray],
) -> None:
    _diversity_support_figure(mode, data)
    _radial_information_figure(mode, data, result)
    _integrated_information_figure(mode, result)
    _average_mass_bin_figure(mode, data, result)
    _predictor_quantile_figure(mode, data, result)
    _radial_response_figure(mode, data, result)
    _cdf_residual_figure(mode, data, result)
    _robustness_figure(mode, result, sensitivity)
    _worst_case_figure(mode, data, result)


def _decision(result: dict[str, np.ndarray]) -> dict[str, object]:
    def classify(
        direct: np.ndarray,
        pca: np.ndarray,
        shuffled: np.ndarray,
    ) -> dict[str, object]:
        material_by_region = []
        detectable_by_region = []
        for region in (0, 1):
            separated = direct[:, region] > shuffled[:, region]
            material = (
                separated & (direct[:, region] >= 0.05) & (pca[:, region] >= 0.05)
            )
            detectable = separated & (direct[:, region] > 0.0) & (pca[:, region] > 0.0)
            material_by_region.append(int(np.count_nonzero(material)))
            detectable_by_region.append(int(np.count_nonzero(detectable)))
        if max(material_by_region) >= 3:
            classification = "material"
        elif max(detectable_by_region) >= 3:
            classification = "detectable_but_minor"
        else:
            classification = "not_demonstrated"
        return {
            "classification": classification,
            "material_epoch_count_inner_outer": material_by_region,
            "detectable_epoch_count_inner_outer": detectable_by_region,
        }

    linear_decision = classify(
        result["relation_r2"][RELATION_NAMES.index("complete_linear")],
        result["pca_r2"],
        np.quantile(result["shuffle_mah_r2"], 0.95, axis=1),
    )
    quadratic_decision = classify(
        result["relation_r2"][RELATION_NAMES.index("complete_additive_quadratic")],
        result["quadratic_pca_r2"],
        np.quantile(result["quadratic_shuffle_mah_r2"], 0.95, axis=1),
    )
    if quadratic_decision["classification"] == "material":
        classification = "material_nonlinear_association"
    elif linear_decision["classification"] == "material":
        classification = "material_linear_association"
    elif "detectable_but_minor" in (
        linear_decision["classification"],
        quadratic_decision["classification"],
    ):
        classification = "detectable_but_minor"
    else:
        classification = "not_demonstrated"
    return {
        "classification": classification,
        "linear": linear_decision,
        "additive_quadratic": quadratic_decision,
    }


def run(mode: str) -> dict[str, object]:
    start = time.perf_counter()
    data = load_data(mode, PRIMARY_Q_MINIMUM)
    sensitivity_data = load_data(mode, SENSITIVITY_Q_MINIMUM)
    result = _fit_primary_sample(data, mode)
    sensitivity = _fit_sensitivity_sample(sensitivity_data)
    output_dir = STAGE7_OUTDIR / mode
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "associations.npz",
        galaxy_ids=data["galaxy_ids"],
        redshifts=REDSHIFTS,
        normalized_radius=data["normalized_radius"],
        predictor_names=np.asarray(PREDICTOR_NAMES),
        relation_names=np.asarray(RELATION_NAMES),
        region_names=np.asarray(REGION_NAMES),
        truth_profiles=data["profiles"],
        **result,
        sensitivity_galaxy_ids=sensitivity_data["galaxy_ids"],
        sensitivity_normalized_radius=sensitivity_data["normalized_radius"],
        sensitivity_direct_r2=sensitivity["direct_r2"],
        sensitivity_strict_r2=sensitivity["strict_r2"],
        sensitivity_pca_r2=sensitivity["pca_r2"],
        sensitivity_quadratic_r2=sensitivity["quadratic_r2"],
        sensitivity_quadratic_pca_r2=sensitivity["quadratic_pca_r2"],
    )
    make_figures(mode, data, result, sensitivity)
    elapsed = time.perf_counter() - start
    decision = _decision(result)
    complete_r2 = result["relation_r2"][RELATION_NAMES.index("complete_linear")]
    diffmah_r2 = result["relation_r2"][RELATION_NAMES.index("diffmah_linear")]
    concentration_r2 = result["relation_r2"][
        RELATION_NAMES.index("concentration_linear")
    ]
    quadratic_r2 = result["relation_r2"][
        RELATION_NAMES.index("complete_additive_quadratic")
    ]
    summary = {
        "mode": mode,
        "n_galaxy": len(data["galaxy_ids"]),
        "sensitivity_n_galaxy": len(sensitivity_data["galaxy_ids"]),
        "primary_normalized_radius_range": [PRIMARY_Q_MINIMUM, Q_MAXIMUM],
        "sensitivity_normalized_radius_range": [SENSITIVITY_Q_MINIMUM, Q_MAXIMUM],
        "n_bootstrap": N_BOOTSTRAP[mode],
        "n_shuffle": N_SHUFFLE[mode],
        "complete_linear_diversity_r2_by_epoch_inner_outer_equal": complete_r2.tolist(),
        "diffmah_linear_diversity_r2_by_epoch_inner_outer_equal": diffmah_r2.tolist(),
        "concentration_linear_diversity_r2_by_epoch_inner_outer_equal": concentration_r2.tolist(),
        "complete_additive_quadratic_diversity_r2_by_epoch_inner_outer_equal": quadratic_r2.tolist(),
        "strict_complete_diversity_r2_by_epoch_inner_outer_equal": result[
            "strict_r2"
        ].tolist(),
        "pca3_complete_diversity_r2_by_epoch_inner_outer_equal": result[
            "pca_r2"
        ].tolist(),
        "annular_complete_diversity_r2_by_epoch_inner_outer_equal": result[
            "annular_r2"
        ].tolist(),
        "quadratic_pca3_diversity_r2_by_epoch_inner_outer_equal": result[
            "quadratic_pca_r2"
        ].tolist(),
        "quadratic_strict_diversity_r2_by_epoch_inner_outer_equal": result[
            "quadratic_strict_r2"
        ].tolist(),
        "quadratic_annular_diversity_r2_by_epoch_inner_outer_equal": result[
            "quadratic_annular_r2"
        ].tolist(),
        "quadratic_bootstrap_95_interval_by_epoch_inner_outer_equal": np.quantile(
            result["quadratic_bootstrap_r2"], [0.025, 0.975], axis=1
        ).tolist(),
        "quadratic_shuffled_diffmah_95th_by_epoch_inner_outer_equal": np.quantile(
            result["quadratic_shuffle_mah_r2"], 0.95, axis=1
        ).tolist(),
        "sensitivity_complete_diversity_r2_by_epoch_inner_outer_equal": sensitivity[
            "direct_r2"
        ].tolist(),
        "sensitivity_quadratic_diversity_r2_by_epoch_inner_outer_equal": sensitivity[
            "quadratic_r2"
        ].tolist(),
        "predictor_condition_by_epoch": result["predictor_condition"].tolist(),
        "diffmah_slope_boundary_fraction_by_epoch": np.mean(
            result["diffmah_boundary"], axis=0
        ).tolist(),
        "decision": decision,
        "wall_seconds": elapsed,
        "subminute_gate_passed": bool(elapsed < 60.0) if mode == "demo" else None,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_manifest(
        output_dir,
        {
            "experiment": "exp62_cog_fit_atlas",
            "stage": "7_scale_free_cog_diversity_information",
            "mode": mode,
            "n_galaxy": len(data["galaxy_ids"]),
            "measured_mstar_and_r50_are_normalizers": True,
            "portable_emulator": False,
        },
    )
    print(json.dumps(summary, indent=2))
    if mode == "demo" and elapsed >= 60.0:
        raise RuntimeError(
            f"Stage 7 demo took {elapsed:.2f} seconds; full run forbidden"
        )
    return summary


def figures(mode: str) -> None:
    data = load_data(mode, PRIMARY_Q_MINIMUM)
    arrays = dict(np.load(STAGE7_OUTDIR / mode / "associations.npz"))
    sensitivity = {
        "direct_r2": arrays.pop("sensitivity_direct_r2"),
        "strict_r2": arrays.pop("sensitivity_strict_r2"),
        "pca_r2": arrays.pop("sensitivity_pca_r2"),
        "quadratic_r2": arrays.pop("sensitivity_quadratic_r2"),
        "quadratic_pca_r2": arrays.pop("sensitivity_quadratic_pca_r2"),
    }
    make_figures(mode, data, arrays, sensitivity)


def mechanics() -> None:
    data = load_data("demo", PRIMARY_Q_MINIMUM)
    assert len(data["galaxy_ids"]) == N_DEMO
    assert data["profiles"].shape[:2] == (N_DEMO, len(REDSHIFTS))
    anchor = np.flatnonzero(np.isclose(data["normalized_radius"], 1.0))
    assert len(anchor) == 1
    assert np.allclose(data["profiles"][..., anchor[0]], np.log10(0.5))
    assert np.all(np.diff(data["profiles"], axis=-1) >= 0.0)
    print("exp62 Stage 7 mechanics OK")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "mechanics"
    if command == "mechanics":
        mechanics()
    elif command in ("demo", "full"):
        run(command)
    elif command == "figures":
        figure_mode = sys.argv[2] if len(sys.argv) > 2 else "full"
        figures(figure_mode)
    else:
        raise SystemExit(__doc__)
