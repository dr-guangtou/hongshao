# %%
"""Exp62 Stage 4: descriptive epoch-local halo--CoG associations.

Commands:

    uv run python experiments/exp62_cog_fit_atlas/stage4.py demo
    uv run python experiments/exp62_cog_fit_atlas/stage4.py full

The regressions are deliberately simple and cross-fitted.  They are used to
measure mass-conditioned associations, not to define a portable emulator or
to establish causal physics.
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

from families import FAMILY_SPECS, cog_log, quantity_values  # noqa: E402
from hongshao.plotting import OKABE_ITO, save_fig, set_style  # noqa: E402
from hongshao.profile_emulator import density_from_cog  # noqa: E402
from hongshao.provenance import write_manifest  # noqa: E402
from run import FIGDIR, OUTDIR, RADII, REDSHIFTS, profile_metrics  # noqa: E402
from stage2 import ROLES  # noqa: E402
from stage3 import (  # noqa: E402
    DECODED_NAMES,
    OBJECTIVE_NAMES,
    _log_annulus_mass,
    _mass_at_radius,
)

set_style()

SOURCE = OUTDIR / "source" / "exp51_epoch_local_diffmah.npz"
STAGE3 = OUTDIR / "full" / "stage3" / "scaling_and_evolution.npz"
PREDICTOR_NAMES = ("logmp", "logtc", "early_index", "late_index", "c_200c")
N_FOLD = 5
N_DEMO = 90
N_BOOTSTRAP = {"demo": 40, "full": 200}
N_SHUFFLE = {"demo": 8, "full": 32}
PRIMARY_RELATION = "sersic_moffat_fixed_cog_log"
PRIMARY_DECODED = f"{PRIMARY_RELATION}_decoded"
PRIMARY_RAW = f"{PRIMARY_RELATION}_raw"


def _demo_rows(halo_mass: np.ndarray) -> np.ndarray:
    order = np.argsort(halo_mass[:, 0])
    positions = np.linspace(0, len(order) - 1, N_DEMO).round().astype(int)
    return order[positions]


def galaxy_folds(halo_mass: np.ndarray) -> np.ndarray:
    order = np.argsort(halo_mass[:, 0])
    folds = np.empty(len(order), int)
    folds[order] = np.arange(len(order)) % N_FOLD
    return folds


def load_data(mode: str):
    source = np.load(SOURCE)
    stage3 = np.load(STAGE3)
    source_ids = np.asarray(source["indices"], int)
    stage3_lookup = {
        int(galaxy): row
        for row, galaxy in enumerate(np.asarray(stage3["galaxy_ids"], int))
    }
    stage3_rows = np.asarray([stage3_lookup[int(galaxy)] for galaxy in source_ids])
    select = np.arange(len(source_ids))
    if mode == "demo":
        select = _demo_rows(np.asarray(source["epoch_halo_mass"]))
    source_ids = source_ids[select]
    stage3_rows = stage3_rows[select]
    diffmah = np.asarray(source["parameters"])[select]
    concentration = np.asarray(source["concentration"])[select]
    features = np.concatenate([diffmah, concentration[..., None]], axis=-1)
    blocks = {}
    target_names = []

    def add_block(name: str, names: tuple[str, ...], values: np.ndarray):
        start = len(target_names)
        target_names.extend([f"{name}::{target}" for target in names])
        blocks[name] = slice(start, len(target_names))
        return np.asarray(values)[stage3_rows]

    target_arrays = [
        add_block(
            "truth_decoded",
            DECODED_NAMES,
            np.asarray(stage3["truth_decoded"]),
        )
    ]
    for role in ROLES:
        for objective_name in OBJECTIVE_NAMES:
            key = f"{role}_{objective_name}"
            target_arrays.append(
                add_block(
                    f"{key}_decoded",
                    DECODED_NAMES,
                    np.asarray(stage3[f"{key}_decoded"]),
                )
            )
    for role in ("gompertz_log", "sersic_moffat_fixed"):
        for objective_name in OBJECTIVE_NAMES:
            key = f"{role}_{objective_name}"
            names = tuple(np.asarray(stage3[f"{key}_parameter_names"]).astype(str))
            target_arrays.append(
                add_block(
                    f"{key}_raw",
                    names,
                    np.asarray(stage3[f"{key}_parameters"]),
                )
            )
    targets = np.concatenate(target_arrays, axis=-1)
    return {
        "galaxy_ids": source_ids,
        "halo_mass": np.asarray(source["epoch_halo_mass"])[select],
        "features": features,
        "targets": targets,
        "target_names": np.asarray(target_names),
        "blocks": blocks,
        "truth_log": np.asarray(source["cogs_log"])[select],
        "primary_variants": np.asarray(stage3[f"{PRIMARY_RELATION}_variants"]).astype(
            str
        )[stage3_rows],
    }


def _mass_design(train_mass: np.ndarray, evaluation_mass: np.ndarray):
    mean = np.mean(train_mass)
    scale = max(np.std(train_mass), 1.0e-8)
    standardized = (evaluation_mass - mean) / scale
    return np.column_stack([np.ones(len(standardized)), standardized, standardized**2])


def cross_fitted_association(
    targets: np.ndarray,
    mass: np.ndarray,
    features: np.ndarray,
    folds: np.ndarray,
    return_predictions: bool = True,
):
    n_target = targets.shape[1]
    mass_prediction = np.empty_like(targets)
    full_prediction = np.empty_like(targets)
    fold_beta = np.empty((N_FOLD, features.shape[1], n_target))
    fold_marginal = np.empty_like(fold_beta)
    for fold in range(N_FOLD):
        train = folds != fold
        test = folds == fold
        train_design = _mass_design(mass[train], mass[train])
        test_design = _mass_design(mass[train], mass[test])
        target_mass_coefficient = np.linalg.lstsq(
            train_design, targets[train], rcond=None
        )[0]
        feature_mass_coefficient = np.linalg.lstsq(
            train_design, features[train], rcond=None
        )[0]
        target_residual_train = targets[train] - train_design @ target_mass_coefficient
        feature_residual_train = (
            features[train] - train_design @ feature_mass_coefficient
        )
        feature_residual_test = features[test] - test_design @ feature_mass_coefficient
        feature_mean = np.mean(feature_residual_train, axis=0)
        feature_scale = np.clip(np.std(feature_residual_train, axis=0), 1.0e-8, None)
        feature_train = (feature_residual_train - feature_mean) / feature_scale
        feature_test = (feature_residual_test - feature_mean) / feature_scale
        complete_coefficient = np.linalg.lstsq(
            np.column_stack([np.ones(train.sum()), feature_train]),
            target_residual_train,
            rcond=None,
        )[0]
        fold_beta[fold] = complete_coefficient[1:]
        for predictor in range(features.shape[1]):
            x = feature_train[:, predictor]
            centered_target = target_residual_train - np.mean(
                target_residual_train, axis=0
            )
            fold_marginal[fold, predictor] = (
                x @ centered_target / np.clip(x @ x, 1.0e-12, None)
            )
        mass_prediction[test] = test_design @ target_mass_coefficient
        full_prediction[test] = (
            mass_prediction[test]
            + np.column_stack([np.ones(test.sum()), feature_test])
            @ complete_coefficient
        )
    mass_residual = targets - mass_prediction
    full_residual = targets - full_prediction
    target_scale = np.clip(np.std(mass_residual, axis=0), 1.0e-8, None)
    output = {
        "mass_rmse": np.sqrt(np.mean(mass_residual**2, axis=0)),
        "full_rmse": np.sqrt(np.mean(full_residual**2, axis=0)),
        "residual_r2": 1.0
        - np.sum(full_residual**2, axis=0)
        / np.clip(np.sum(mass_residual**2, axis=0), 1.0e-30, None),
        "target_residual_scale": target_scale,
        "conditional_beta": np.mean(fold_beta, axis=0),
        "conditional_beta_standardized": np.mean(fold_beta, axis=0)
        / target_scale[None, :],
        "marginal_beta_standardized": np.mean(fold_marginal, axis=0)
        / target_scale[None, :],
        "fold_beta_standardized": fold_beta / target_scale[None, None, :],
    }
    if return_predictions:
        output["mass_prediction"] = mass_prediction
        output["full_prediction"] = full_prediction
    return output


def _full_residualized_system(
    targets: np.ndarray, mass: np.ndarray, features: np.ndarray
):
    design = _mass_design(mass, mass)
    target_coefficient = np.linalg.lstsq(design, targets, rcond=None)[0]
    feature_coefficient = np.linalg.lstsq(design, features, rcond=None)[0]
    target_residual = targets - design @ target_coefficient
    feature_residual = features - design @ feature_coefficient
    feature_residual -= np.mean(feature_residual, axis=0)
    feature_scale = np.clip(np.std(feature_residual, axis=0), 1.0e-8, None)
    feature_residual /= feature_scale
    target_scale = np.clip(np.std(target_residual, axis=0), 1.0e-8, None)
    return target_residual, feature_residual, target_scale


def bootstrap_coefficients(
    targets: np.ndarray,
    mass: np.ndarray,
    features: np.ndarray,
    n_bootstrap: int,
    seed: int,
):
    target_residual, feature_residual, target_scale = _full_residualized_system(
        targets, mass, features
    )
    rng = np.random.default_rng(seed)
    raw = np.empty((n_bootstrap, features.shape[1], targets.shape[1]))
    for bootstrap in range(n_bootstrap):
        rows = rng.integers(0, len(targets), len(targets))
        design = np.column_stack([np.ones(len(rows)), feature_residual[rows]])
        raw[bootstrap] = np.linalg.lstsq(design, target_residual[rows], rcond=None)[0][
            1:
        ]
    return raw, raw / target_scale[None, None, :]


def mass_conditioned_shuffle(
    features: np.ndarray,
    mass: np.ndarray,
    columns: tuple[int, ...],
    rng: np.random.Generator,
):
    output = features.copy()
    edges = np.unique(np.quantile(mass, np.linspace(0.0, 1.0, 11)))
    for low, high in zip(edges[:-1], edges[1:]):
        rows = np.flatnonzero((mass >= low) & (mass <= high))
        if len(rows) > 1:
            shuffled = rng.permutation(rows)
            output[np.ix_(rows, columns)] = features[np.ix_(shuffled, columns)]
    return output


def predictor_diagnostics(mass: np.ndarray, features: np.ndarray):
    _, residual, _ = _full_residualized_system(np.zeros((len(mass), 1)), mass, features)
    correlation = np.corrcoef(residual.T)
    singular_values = np.linalg.svd(residual, compute_uv=False)
    condition = float(singular_values[0] / singular_values[-1])
    return correlation, condition


def association_heatmap(
    mode: str,
    standardized_beta: np.ndarray,
    bootstrap_standardized: np.ndarray,
    decoded_slice: slice,
):
    chosen = (1, 2, 3, 4, 5, 7, 9, 10)
    labels = [DECODED_NAMES[index].replace("_", " ") for index in chosen]
    values = standardized_beta[:, :, decoded_slice][:, :, chosen]
    bootstrap = bootstrap_standardized[:, :, :, decoded_slice][:, :, :, chosen]
    limit = np.quantile(np.abs(values), 0.98)
    figure, axes = plt.subplots(
        1,
        len(REDSHIFTS),
        figsize=(20.0, 4.8),
        sharey=True,
        layout="constrained",
    )
    for epoch, axis in enumerate(axes):
        image = axis.imshow(
            values[epoch],
            aspect="auto",
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
        )
        lower = np.quantile(bootstrap[epoch], 0.025, axis=0)
        upper = np.quantile(bootstrap[epoch], 0.975, axis=0)
        for predictor in range(len(PREDICTOR_NAMES)):
            for target in range(len(chosen)):
                if lower[predictor, target] * upper[predictor, target] > 0.0:
                    axis.text(
                        target, predictor, "*", ha="center", va="center", fontsize=8
                    )
        axis.set_title(rf"$z={REDSHIFTS[epoch]:g}$")
        axis.set_xticks(
            np.arange(len(labels)), labels, rotation=70, ha="right", fontsize=7
        )
        axis.set_yticks(np.arange(len(PREDICTOR_NAMES)), PREDICTOR_NAMES)
    figure.colorbar(
        image,
        ax=axes,
        label="Conditional response / mass-residual target scatter",
        shrink=0.78,
        pad=0.015,
    )
    figure.suptitle(
        f"exp62 Stage 4 {mode} — mass-conditioned epoch-local halo–CoG associations; "
        r"star = bootstrap 95\% interval excludes zero"
    )
    save_fig(figure, FIGDIR / f"stage4_{mode}" / "exp62_halo_cog_associations")
    plt.close(figure)


def control_figure(
    mode: str,
    mass_rmse: np.ndarray,
    full_rmse: np.ndarray,
    shuffle_mah_rmse: np.ndarray,
    shuffle_c_rmse: np.ndarray,
    decoded_slice: slice,
):
    chosen = (1, 2, 3, 4, 5, 7, 9, 10)
    labels = [DECODED_NAMES[index].replace("_", " ") for index in chosen]
    mass = mass_rmse[:, decoded_slice][:, chosen]
    real = 100.0 * (mass - full_rmse[:, decoded_slice][:, chosen]) / mass
    mah = (
        100.0
        * (
            mass
            - np.median(shuffle_mah_rmse[:, :, decoded_slice][:, :, chosen], axis=1)
        )
        / mass
    )
    concentration = (
        100.0
        * (mass - np.median(shuffle_c_rmse[:, :, decoded_slice][:, :, chosen], axis=1))
        / mass
    )
    values = np.stack([real, mah, concentration], axis=1)
    limit = np.quantile(np.abs(values), 0.98)
    figure, axes = plt.subplots(
        1,
        len(REDSHIFTS),
        figsize=(20.0, 4.2),
        sharey=True,
        layout="constrained",
    )
    row_labels = (
        "Real DiffMAH + c",
        "Shuffled DiffMAH + real c",
        "Real DiffMAH + shuffled c",
    )
    for epoch, axis in enumerate(axes):
        image = axis.imshow(
            values[epoch], aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit
        )
        axis.set_title(rf"$z={REDSHIFTS[epoch]:g}$")
        axis.set_xticks(
            np.arange(len(labels)), labels, rotation=70, ha="right", fontsize=7
        )
        axis.set_yticks(np.arange(3), row_labels)
    figure.colorbar(
        image,
        ax=axes,
        label=r"Held-out RMS reduction from current-halo-mass only (\%)",
        shrink=0.78,
        pad=0.015,
    )
    figure.suptitle(f"exp62 Stage 4 {mode} — mass-conditioned shuffle controls")
    save_fig(figure, FIGDIR / f"stage4_{mode}" / "exp62_halo_cog_controls")
    plt.close(figure)


def decoded_radial_responses(
    mode: str,
    raw_bootstrap: np.ndarray,
    raw_slice: slice,
    primary_parameters: np.ndarray,
    primary_variants: np.ndarray,
):
    figure, axes = plt.subplots(3, len(PREDICTOR_NAMES), figsize=(16.0, 9.5))
    colors = (OKABE_ITO[0], OKABE_ITO[1], OKABE_ITO[2], OKABE_ITO[4], OKABE_ITO[5])
    quantity_labels = ("Cumulative CoG", "Differential shell mass", "Surface density")
    response_cache = []
    for epoch, color in enumerate(colors):
        variants, counts = np.unique(primary_variants[:, epoch], return_counts=True)
        family = str(variants[np.argmax(counts)])
        spec = FAMILY_SPECS[family]
        baseline_parameters = np.median(primary_parameters[:, epoch], axis=0)
        baseline_log = cog_log(family, baseline_parameters, RADII)
        baseline_shell = np.log10(
            np.clip(quantity_values(baseline_log, RADII, "shells"), 1.0, None)
        )
        baseline_density, density_radius = density_from_cog(
            baseline_log[None, :], RADII
        )
        for predictor in range(len(PREDICTOR_NAMES)):
            responses = [[], [], []]
            for beta in raw_bootstrap[epoch, :, predictor, raw_slice]:
                parameters = np.clip(
                    baseline_parameters + beta,
                    np.asarray(spec.lower) + 1.0e-8,
                    np.asarray(spec.upper) - 1.0e-8,
                )
                changed = cog_log(family, parameters, RADII)
                changed_shell = np.log10(
                    np.clip(quantity_values(changed, RADII, "shells"), 1.0, None)
                )
                changed_density, _ = density_from_cog(changed[None, :], RADII)
                responses[0].append(changed - baseline_log)
                responses[1].append(changed_shell - baseline_shell)
                responses[2].append(changed_density[0] - baseline_density[0])
            radii = (RADII, RADII, density_radius)
            for quantity, (response, radius) in enumerate(zip(responses, radii)):
                response = np.asarray(response)
                median = np.median(response, axis=0)
                lower, upper = np.quantile(response, [0.16, 0.84], axis=0)
                axes[quantity, predictor].plot(
                    radius, median, color=color, label=rf"$z={REDSHIFTS[epoch]:g}$"
                )
                axes[quantity, predictor].fill_between(
                    radius, lower, upper, color=color, alpha=0.08
                )
                response_cache.append(median)
    for quantity in range(3):
        for predictor in range(len(PREDICTOR_NAMES)):
            axes[quantity, predictor].axhline(0.0, color="0.5", linewidth=0.7)
            axes[quantity, predictor].set_xscale("log")
            axes[quantity, predictor].set_title(PREDICTOR_NAMES[predictor])
            axes[quantity, predictor].set_xlabel("Semi-major axis (kpc)")
            if predictor == 0:
                axes[quantity, predictor].set_ylabel(
                    f"{quantity_labels[quantity]}\nchange (dex)"
                )
    axes[0, 0].legend(frameon=False, fontsize=7)
    figure.suptitle(
        f"exp62 Stage 4 {mode} — decoded response to +1 mass-conditioned predictor SD; "
        r"bands are galaxy-bootstrap 16–84\%"
    )
    figure.tight_layout()
    save_fig(figure, FIGDIR / f"stage4_{mode}" / "exp62_decoded_radial_responses")
    plt.close(figure)
    return response_cache


def _decode_predictions(variants: np.ndarray, parameters: np.ndarray):
    output = np.empty((*parameters.shape[:2], len(RADII)))
    clipped = np.zeros(parameters.shape[:2], bool)
    for galaxy in range(len(parameters)):
        for epoch in range(len(REDSHIFTS)):
            family = str(variants[galaxy, epoch])
            spec = FAMILY_SPECS[family]
            bounded = np.clip(
                parameters[galaxy, epoch],
                np.asarray(spec.lower) + 1.0e-8,
                np.asarray(spec.upper) - 1.0e-8,
            )
            clipped[galaxy, epoch] = not np.array_equal(
                bounded, parameters[galaxy, epoch]
            )
            output[galaxy, epoch] = cog_log(family, bounded, RADII)
    return output, clipped


def standard_qa_figure(
    mode: str,
    truth_log: np.ndarray,
    halo_mass: np.ndarray,
    mass_prediction: np.ndarray,
    full_prediction: np.ndarray,
):
    figure, axes = plt.subplots(len(REDSHIFTS), 2, figsize=(10.5, 15.0))
    colors = (OKABE_ITO[0], OKABE_ITO[2], OKABE_ITO[4])
    for epoch, redshift in enumerate(REDSHIFTS):
        mass = halo_mass[:, epoch]
        edges = np.quantile(mass, [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
        for mass_bin, color in enumerate(colors):
            selected = (mass >= edges[mass_bin]) & (mass <= edges[mass_bin + 1])
            truth_shape = truth_log[selected, epoch] - truth_log[selected, epoch, -1:]
            mass_shape = (
                mass_prediction[selected, epoch] - mass_prediction[selected, epoch, -1:]
            )
            full_shape = (
                full_prediction[selected, epoch] - full_prediction[selected, epoch, -1:]
            )
            axes[epoch, 0].plot(RADII, np.median(truth_shape, axis=0), ":", color=color)
            axes[epoch, 0].plot(RADII, np.median(mass_shape, axis=0), "--", color=color)
            axes[epoch, 0].plot(
                RADII,
                np.median(full_shape, axis=0),
                color=color,
                label=rf"${edges[mass_bin]:.2f}<\log M_h<{edges[mass_bin + 1]:.2f}$",
            )
        for profiles, color, label, alpha in (
            (truth_log[:, epoch], "black", "TNG", 0.16),
            (mass_prediction[:, epoch], "0.6", "Current halo mass only", 0.12),
            (full_prediction[:, epoch], OKABE_ITO[2], "DiffMAH + concentration", 0.16),
        ):
            axes[epoch, 1].scatter(
                _mass_at_radius(profiles, 30.0),
                _log_annulus_mass(profiles, 50.0, 100.0),
                s=5,
                alpha=alpha,
                color=color,
                label=label,
            )
        axes[epoch, 0].set_xscale("log")
        axes[epoch, 0].set_ylabel(rf"$z={redshift:g}$  $\log[M(<R)/M_{{148}}]$")
        axes[epoch, 1].set_ylabel(r"$\log M_*(50\!-\!100\,\mathrm{kpc})$")
    axes[0, 0].legend(frameon=False, fontsize=7, title="Solid: full; dashed: mass only")
    axes[0, 1].legend(frameon=False, fontsize=7)
    axes[-1, 0].set_xlabel("Semi-major axis (kpc)")
    axes[-1, 1].set_xlabel(r"$\log M_*(<30\,\mathrm{kpc})$")
    figure.suptitle(
        f"exp62 Stage 4 {mode} — cross-fitted descriptive halo–CoG relation"
    )
    figure.tight_layout()
    save_fig(figure, FIGDIR / f"stage4_{mode}" / "exp62_stage4_standard_qa")
    plt.close(figure)


def run(mode: str):
    start = time.perf_counter()
    data = load_data(mode)
    folds = galaxy_folds(data["halo_mass"])
    n_target = data["targets"].shape[-1]
    n_bootstrap = N_BOOTSTRAP[mode]
    n_shuffle = N_SHUFFLE[mode]
    payload = {
        "galaxy_ids": data["galaxy_ids"],
        "target_names": data["target_names"],
        "predictor_names": np.asarray(PREDICTOR_NAMES),
        "folds": folds,
        "redshifts": REDSHIFTS,
    }
    arrays = {
        "mass_rmse": np.empty((len(REDSHIFTS), n_target)),
        "full_rmse": np.empty((len(REDSHIFTS), n_target)),
        "residual_r2": np.empty((len(REDSHIFTS), n_target)),
        "conditional_beta": np.empty((len(REDSHIFTS), len(PREDICTOR_NAMES), n_target)),
        "conditional_beta_standardized": np.empty(
            (len(REDSHIFTS), len(PREDICTOR_NAMES), n_target)
        ),
        "marginal_beta_standardized": np.empty(
            (len(REDSHIFTS), len(PREDICTOR_NAMES), n_target)
        ),
        "fold_beta_standardized": np.empty(
            (len(REDSHIFTS), N_FOLD, len(PREDICTOR_NAMES), n_target)
        ),
        "mass_prediction": np.empty_like(data["targets"]),
        "full_prediction": np.empty_like(data["targets"]),
        "bootstrap_beta": np.empty(
            (len(REDSHIFTS), n_bootstrap, len(PREDICTOR_NAMES), n_target)
        ),
        "bootstrap_beta_standardized": np.empty(
            (len(REDSHIFTS), n_bootstrap, len(PREDICTOR_NAMES), n_target)
        ),
        "shuffle_mah_rmse": np.empty((len(REDSHIFTS), n_shuffle, n_target)),
        "shuffle_c_rmse": np.empty((len(REDSHIFTS), n_shuffle, n_target)),
        "predictor_correlation": np.empty(
            (len(REDSHIFTS), len(PREDICTOR_NAMES), len(PREDICTOR_NAMES))
        ),
        "predictor_condition": np.empty(len(REDSHIFTS)),
    }
    rng = np.random.default_rng(6104)
    for epoch, redshift in enumerate(REDSHIFTS):
        result = cross_fitted_association(
            data["targets"][:, epoch],
            data["halo_mass"][:, epoch],
            data["features"][:, epoch],
            folds,
        )
        for key in (
            "mass_rmse",
            "full_rmse",
            "residual_r2",
            "conditional_beta",
            "conditional_beta_standardized",
            "marginal_beta_standardized",
            "fold_beta_standardized",
        ):
            arrays[key][epoch] = result[key]
        arrays["mass_prediction"][:, epoch] = result["mass_prediction"]
        arrays["full_prediction"][:, epoch] = result["full_prediction"]
        raw_bootstrap, standardized_bootstrap = bootstrap_coefficients(
            data["targets"][:, epoch],
            data["halo_mass"][:, epoch],
            data["features"][:, epoch],
            n_bootstrap,
            6200 + epoch,
        )
        arrays["bootstrap_beta"][epoch] = raw_bootstrap
        arrays["bootstrap_beta_standardized"][epoch] = standardized_bootstrap
        correlation, condition = predictor_diagnostics(
            data["halo_mass"][:, epoch], data["features"][:, epoch]
        )
        arrays["predictor_correlation"][epoch] = correlation
        arrays["predictor_condition"][epoch] = condition
        for shuffle in range(n_shuffle):
            shuffled_mah = mass_conditioned_shuffle(
                data["features"][:, epoch],
                data["halo_mass"][:, epoch],
                (0, 1, 2, 3),
                rng,
            )
            shuffled_c = mass_conditioned_shuffle(
                data["features"][:, epoch],
                data["halo_mass"][:, epoch],
                (4,),
                rng,
            )
            arrays["shuffle_mah_rmse"][epoch, shuffle] = cross_fitted_association(
                data["targets"][:, epoch],
                data["halo_mass"][:, epoch],
                shuffled_mah,
                folds,
                return_predictions=False,
            )["full_rmse"]
            arrays["shuffle_c_rmse"][epoch, shuffle] = cross_fitted_association(
                data["targets"][:, epoch],
                data["halo_mass"][:, epoch],
                shuffled_c,
                folds,
                return_predictions=False,
            )["full_rmse"]
        primary = data["blocks"][PRIMARY_DECODED]
        gain = (
            100.0
            * (
                np.median(arrays["mass_rmse"][epoch, primary])
                - np.median(arrays["full_rmse"][epoch, primary])
            )
            / np.median(arrays["mass_rmse"][epoch, primary])
        )
        print(
            f"  z={redshift:g}: median decoded-target held-out RMS gain over current halo mass = {gain:+.2f}%; "
            f"predictor condition={condition:.2f}",
            flush=True,
        )
    payload.update(arrays)

    primary_decoded = data["blocks"][PRIMARY_DECODED]
    primary_raw = data["blocks"][PRIMARY_RAW]
    association_heatmap(
        mode,
        arrays["conditional_beta_standardized"],
        arrays["bootstrap_beta_standardized"],
        primary_decoded,
    )
    control_figure(
        mode,
        arrays["mass_rmse"],
        arrays["full_rmse"],
        arrays["shuffle_mah_rmse"],
        arrays["shuffle_c_rmse"],
        primary_decoded,
    )
    decoded_radial_responses(
        mode,
        arrays["bootstrap_beta"],
        primary_raw,
        data["targets"][:, :, primary_raw],
        data["primary_variants"],
    )
    mass_profile, mass_clipped = _decode_predictions(
        data["primary_variants"], arrays["mass_prediction"][:, :, primary_raw]
    )
    full_profile, full_clipped = _decode_predictions(
        data["primary_variants"], arrays["full_prediction"][:, :, primary_raw]
    )
    standard_qa_figure(
        mode,
        data["truth_log"],
        data["halo_mass"],
        mass_profile,
        full_profile,
    )
    mass_metrics = profile_metrics(
        mass_profile.reshape(-1, len(RADII)), data["truth_log"].reshape(-1, len(RADII))
    )
    full_metrics = profile_metrics(
        full_profile.reshape(-1, len(RADII)), data["truth_log"].reshape(-1, len(RADII))
    )
    result = {
        "mode": mode,
        "n_galaxy": len(data["galaxy_ids"]),
        "n_target": n_target,
        "n_bootstrap": n_bootstrap,
        "n_shuffle": n_shuffle,
        "primary_relation": PRIMARY_RELATION,
        "excluded_raw_parameter_family": "double_sersic_free",
        "exclusion_reason": "Stage 2 profiled scans and Jacobians show raw component-shape degeneracy even when decoded CoGs are accurate.",
        "mass_only_median_full_cog_rms_dex": float(
            np.median(mass_metrics["cog_rms_full_dex"])
        ),
        "complete_median_full_cog_rms_dex": float(
            np.median(full_metrics["cog_rms_full_dex"])
        ),
        "mass_only_median_density_rms_dex": float(
            np.median(mass_metrics["density_rms_dex"])
        ),
        "complete_median_density_rms_dex": float(
            np.median(full_metrics["density_rms_dex"])
        ),
        "mass_only_parameter_clip_fraction": float(np.mean(mass_clipped)),
        "complete_parameter_clip_fraction": float(np.mean(full_clipped)),
        "predictor_condition_by_epoch": arrays["predictor_condition"].tolist(),
        "wall_seconds": time.perf_counter() - start,
    }
    result["subminute_gate_passed"] = (
        bool(result["wall_seconds"] < 60.0) if mode == "demo" else None
    )
    output_dir = OUTDIR / mode / "stage4"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "associations.npz", **payload)
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    write_manifest(
        output_dir,
        {
            "experiment": "exp62_cog_fit_atlas",
            "stage": "4_epoch_local_halo_cog_associations",
            "mode": mode,
            "n_galaxy": len(data["galaxy_ids"]),
            "n_target": n_target,
            "n_bootstrap": n_bootstrap,
            "n_shuffle": n_shuffle,
        },
    )
    print(json.dumps(result, indent=2))
    if mode == "demo" and result["wall_seconds"] >= 60.0:
        raise RuntimeError(
            f"Stage 4 demo took {result['wall_seconds']:.2f} seconds; full run is forbidden"
        )


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if command not in ("demo", "full"):
        raise SystemExit(__doc__)
    run(command)
