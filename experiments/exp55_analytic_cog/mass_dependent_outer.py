"""exp55 Stage 9: a fold-local mass-dependent outer-slope boundary test.

The rule is deliberately simple and predeclared: fit the gamma=1.4 profile
below the outer-training sample's 2/3 halo-mass quantile and gamma=1.25 above
it. This discontinuous model tests whether the mass dependence seen in the
representation survives held-out halo mapping and stochastic generation. It is
not proposed as the final smooth outer-slope law.

Commands
--------
demo
    Validate the complete path on 90 galaxies and four draws.
run
    Run the full five-fold test, figures, and conditional standard QA.
figures
    Rebuild all figures from saved outputs.
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
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from hongshao import qa  # noqa: E402
from hongshao.plotting import OKABE_ITO, save_fig, set_style  # noqa: E402
from hongshao.provenance import write_manifest  # noqa: E402
from halo_map import (  # noqa: E402
    RADII,
    folds_of,
    load_sample,
    shuffled_within_mass_bins,
)
from outer_envelope import (  # noqa: E402
    PLANE_KEYS,
    SIZE_NAMES,
    aligned_profile_family,
    decode_fixed_gamma,
    individual_population_metrics,
    mass_bin_shape_residual,
    profile_draw_summary,
    profile_point_summary,
    size_relation,
    stochastic_population_summary,
    summarize_individual_populations,
)
from qa_figures import (  # noqa: E402
    density_by_halo_mass_figure,
    single_epoch_planes_figure,
    single_epoch_size_figure,
)
from refine_halo_map import SPARSE_TERMS, fit_conditional_model  # noqa: E402

set_style()

FIGURE_DIRECTORY = HERE / "figures" / "mass_dependent_outer"
OUTPUT_DIRECTORY = HERE / "outputs" / "mass_dependent_outer"
STANDARD_QA_DIRECTORY = FIGURE_DIRECTORY / "standard_qa"
OUTER_OUTPUT = HERE / "outputs" / "outer_envelope"

N_MAX = int(os.environ.get("EXP55_MASS_GAMMA_NMAX", 0))
N_DRAW = int(os.environ.get("EXP55_MASS_GAMMA_N_DRAW", 32))
N_POPULATION_DRAW = int(os.environ.get("EXP55_MASS_GAMMA_N_POPULATION_DRAW", 4))
SEED = 9550


def decode_variable_gamma(
    parameters: np.ndarray,
    gamma_by_galaxy: np.ndarray,
) -> np.ndarray:
    """Decode raw profile coordinates using one known gamma per galaxy."""
    parameters = np.asarray(parameters, float)
    gamma_by_galaxy = np.asarray(gamma_by_galaxy, float)
    if parameters.ndim == 2:
        if len(parameters) != len(gamma_by_galaxy):
            raise ValueError("gamma array does not match the galaxy axis")
        output = np.empty((len(parameters), len(RADII)))
        for gamma in (1.25, 1.4):
            selected = np.isclose(gamma_by_galaxy, gamma)
            output[selected] = decode_fixed_gamma(parameters[selected], gamma)
        return output
    if parameters.ndim == 3:
        return np.asarray(
            [decode_variable_gamma(draw, gamma_by_galaxy) for draw in parameters]
        )
    raise ValueError("parameters must have galaxy or draw-by-galaxy shape")


def family_selection(
    mass: np.ndarray,
    threshold: float,
    family_1p25: dict,
    family_1p4: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return known gamma, fitted coordinates, and representations."""
    use_1p4 = np.asarray(mass) < threshold
    gamma = np.where(use_1p4, 1.4, 1.25)
    parameters = np.where(
        use_1p4[:, None], family_1p4["parameters"], family_1p25["parameters"]
    )
    prediction_log = np.where(
        use_1p4[:, None],
        family_1p4["prediction_log"],
        family_1p25["prediction_log"],
    )
    return gamma, parameters, prediction_log


def augmented_features(features: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """Append the known profile-family indicator to the halo vector."""
    return np.column_stack([features, np.isclose(gamma, 1.4).astype(float)])


def same_regime_neighbour_draws(
    model,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    train_gamma: np.ndarray,
    test_features: np.ndarray,
    test_gamma: np.ndarray,
    n_draw: int,
    seed: int,
    n_neighbour: int = 128,
) -> np.ndarray:
    """Resample complete residuals only within the same outer-slope regime."""
    residual = train_targets - model.predict_mean(train_features)
    output = np.empty((n_draw, len(test_features), train_targets.shape[1]))
    rng = np.random.default_rng(seed)
    standardized_train = model.standardized(train_features)
    standardized_test = model.standardized(test_features)
    for gamma in (1.25, 1.4):
        train_selected = np.where(np.isclose(train_gamma, gamma))[0]
        test_selected = np.where(np.isclose(test_gamma, gamma))[0]
        if not len(test_selected):
            continue
        tree = cKDTree(standardized_train[train_selected, :5])
        _, neighbours = tree.query(
            standardized_test[test_selected, :5],
            k=min(n_neighbour, len(train_selected)),
        )
        if neighbours.ndim == 1:
            neighbours = neighbours[:, None]
        choices = rng.integers(0, neighbours.shape[1], (n_draw, len(test_selected)))
        chosen = neighbours[np.arange(len(test_selected))[None], choices]
        residual_rows = train_selected[chosen]
        output[:, test_selected] = (
            model.predict_mean(test_features[test_selected])[None]
            + residual[residual_rows]
        )
    return output


def calibrate_outer_fold(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    train_gamma: np.ndarray,
    train_truth_log: np.ndarray,
    test_features: np.ndarray,
    test_gamma: np.ndarray,
    model,
    n_draw: int,
    seed: int,
) -> tuple[np.ndarray, dict]:
    """Select one residual scale on inner held-out galaxies."""
    scale_grid = np.asarray([1.00, 1.08, 1.16, 1.24, 1.32])
    order = np.random.default_rng(seed).permutation(len(train_features))
    inner_mean = np.empty_like(train_targets)
    inner_draw = np.empty((n_draw, *train_targets.shape))
    all_rows = np.arange(len(train_features))
    for inner_number, validation in enumerate(np.array_split(order, 4)):
        training = np.setdiff1d(all_rows, validation)
        inner_model = fit_conditional_model(
            train_features[training], train_targets[training], SPARSE_TERMS
        )
        inner_mean[validation] = inner_model.predict_mean(train_features[validation])
        inner_draw[:, validation] = same_regime_neighbour_draws(
            inner_model,
            train_features[training],
            train_targets[training],
            train_gamma[training],
            train_features[validation],
            train_gamma[validation],
            n_draw,
            seed + inner_number,
        )
    coverage = {}
    for scale in scale_grid:
        draw_log = decode_variable_gamma(
            inner_mean[None] + scale * (inner_draw - inner_mean[None]), train_gamma
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
    base = same_regime_neighbour_draws(
        model,
        train_features,
        train_targets,
        train_gamma,
        test_features,
        test_gamma,
        n_draw,
        seed + 100,
    )
    mean = model.predict_mean(test_features)[None]
    return mean + selected_scale * (base - mean), {
        "selected_scale": float(selected_scale),
        "inner_coverage_by_scale": coverage,
    }


def cross_validated_hybrid(
    sample: dict,
    family_1p25: dict,
    family_1p4: dict,
    n_draw: int,
) -> dict:
    """Fit, calibrate, and generate the fold-local two-regime profile map."""
    features = sample["features"]
    truth_log = sample["truth_log"]
    mean_parameters = np.empty_like(family_1p25["parameters"])
    gamma_by_galaxy = np.empty(len(features))
    representation_log = np.empty_like(truth_log)
    draw_parameters = np.empty((n_draw, len(features), mean_parameters.shape[1]))
    metadata = []
    all_rows = np.arange(len(features))
    for fold_number, test in enumerate(folds_of(len(features))):
        train = np.setdiff1d(all_rows, test)
        threshold = float(np.quantile(features[train, 0], 2.0 / 3.0))
        train_gamma, train_targets, train_representation = family_selection(
            features[train, 0],
            threshold,
            {name: values[train] for name, values in family_1p25.items()},
            {name: values[train] for name, values in family_1p4.items()},
        )
        test_gamma, _, test_representation = family_selection(
            features[test, 0],
            threshold,
            {name: values[test] for name, values in family_1p25.items()},
            {name: values[test] for name, values in family_1p4.items()},
        )
        train_augmented = augmented_features(features[train], train_gamma)
        test_augmented = augmented_features(features[test], test_gamma)
        model = fit_conditional_model(train_augmented, train_targets, SPARSE_TERMS)
        mean_parameters[test] = model.predict_mean(test_augmented)
        gamma_by_galaxy[test] = test_gamma
        representation_log[test] = test_representation
        draw_parameters[:, test], calibration = calibrate_outer_fold(
            train_augmented,
            train_targets,
            train_gamma,
            truth_log[train],
            test_augmented,
            test_gamma,
            model,
            n_draw,
            SEED + 1000 * fold_number,
        )
        metadata.append(
            {
                "fold": fold_number,
                "halo_mass_threshold": threshold,
                "n_train_gamma1p4": int(np.count_nonzero(train_gamma == 1.4)),
                "n_test_gamma1p4": int(np.count_nonzero(test_gamma == 1.4)),
                **calibration,
            }
        )
    return {
        "mean_parameters": mean_parameters,
        "mean_log": decode_variable_gamma(mean_parameters, gamma_by_galaxy),
        "draw_parameters": draw_parameters,
        "draw_log": decode_variable_gamma(draw_parameters, gamma_by_galaxy),
        "gamma": gamma_by_galaxy,
        "representation_log": representation_log,
        "metadata": metadata,
    }


def mean_control(
    sample: dict,
    family_1p25: dict,
    family_1p4: dict,
    control_features: np.ndarray,
    terms: tuple[tuple, ...],
) -> np.ndarray:
    """Held-out point map for one real or shuffled halo-information control."""
    mass = sample["features"][:, 0]
    prediction = np.empty_like(sample["truth_log"])
    all_rows = np.arange(len(mass))
    for test in folds_of(len(mass)):
        train = np.setdiff1d(all_rows, test)
        threshold = float(np.quantile(mass[train], 2.0 / 3.0))
        train_gamma, train_targets, _ = family_selection(
            mass[train],
            threshold,
            {name: values[train] for name, values in family_1p25.items()},
            {name: values[train] for name, values in family_1p4.items()},
        )
        test_gamma, _, _ = family_selection(
            mass[test],
            threshold,
            {name: values[test] for name, values in family_1p25.items()},
            {name: values[test] for name, values in family_1p4.items()},
        )
        train_values = augmented_features(control_features[train], train_gamma)
        test_values = augmented_features(control_features[test], test_gamma)
        model = fit_conditional_model(train_values, train_targets, terms)
        prediction[test] = decode_variable_gamma(
            model.predict_mean(test_values), test_gamma
        )
    return prediction


def control_audit(sample: dict, family_1p25: dict, family_1p4: dict) -> dict:
    """Retain mass-only and mass-conditioned assembly null controls."""
    features = sample["features"]
    variants = {
        "complete": (features, SPARSE_TERMS),
        "final_mass_only": (features[:, :1], ()),
        "mah_shape_shuffled": (
            shuffled_within_mass_bins(features, (1, 2, 3), seed=SEED + 1),
            SPARSE_TERMS,
        ),
        "concentration_shuffled": (
            shuffled_within_mass_bins(features, (4,), seed=SEED + 2),
            SPARSE_TERMS,
        ),
    }
    output = {}
    for name, (values, terms) in variants.items():
        prediction = mean_control(sample, family_1p25, family_1p4, values, terms)
        output[name] = profile_point_summary(
            prediction, sample["truth_log"], features[:, 0]
        )
    return output


def comparison_figure(result: dict) -> None:
    """Direct representation and generated-population comparison."""
    truth_log = np.asarray(result["truth_log"])
    halo_mass = np.asarray(result["halo_mass"])
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.5))

    for name, label, color in (
        ("gamma1p25", r"fixed $\gamma=1.25$", OKABE_ITO[0]),
        ("gamma1p4", r"fixed $\gamma=1.4$", OKABE_ITO[1]),
        ("hybrid", "mass-dependent boundary", OKABE_ITO[2]),
    ):
        error = np.asarray(result["models"][name]["representation_r90_error"])
        order = np.argsort(halo_mass)
        chunks = np.array_split(order, 12)
        axes[0, 0].plot(
            [np.median(halo_mass[group]) for group in chunks],
            [np.median(error[group]) for group in chunks],
            marker="o",
            color=color,
            label=label,
        )
    axes[0, 0].axhline(0.0, color="0.5", linewidth=0.8)
    axes[0, 0].set_xlabel(r"DiffMAH $\log M_{\rm peak}$")
    axes[0, 0].set_ylabel(r"Representation $\Delta\log R_{90}$ (dex)")
    axes[0, 0].legend(frameon=False, fontsize=8)

    for name, label, color in (
        ("gamma1p4", r"fixed $\gamma=1.4$", OKABE_ITO[1]),
        ("hybrid", "mass-dependent boundary", OKABE_ITO[2]),
    ):
        residual = mass_bin_shape_residual(
            np.asarray(result["models"][name]["point_log"]), truth_log, halo_mass
        )
        axes[0, 1].plot(RADII, residual[0], color=color, label=label)
        axes[1, 1].plot(RADII, residual[2], color=color, label=label)
    axes[0, 1].set_title("Lowest halo-mass bin")
    axes[1, 1].set_title("Highest halo-mass bin")
    for axis in axes[:, 1]:
        axis.axhline(0.0, color="0.5", linewidth=0.8)
        axis.set_xscale("log")
        axis.set_xlabel("Radius (kpc)")
        axis.set_ylabel(r"Median shape residual (\%)")
    axes[0, 1].legend(frameon=False, fontsize=8)
    axes[1, 1].legend(frameon=False, fontsize=8)

    metrics = ("fixed_one_energy", "fixed_two_energy", "re_energy", "r90_energy")
    metric_labels = ("30--50", "50--100", "2--4 Re", "R90")
    x = np.arange(len(metrics))
    width = 0.25
    for index, (name, label, color) in enumerate(
        (
            ("gamma1p25", r"$\gamma=1.25$", OKABE_ITO[0]),
            ("gamma1p4", r"$\gamma=1.4$", OKABE_ITO[1]),
            ("hybrid", "mass-dependent", OKABE_ITO[2]),
        )
    ):
        values = [result["models"][name][metric] for metric in metrics]
        axes[0, 2].bar(x + (index - 1) * width, values, width, color=color, label=label)
    axes[0, 2].axhline(1.0, color="0.4", linestyle="--", linewidth=0.8)
    axes[0, 2].set_xticks(x, metric_labels, rotation=20)
    axes[0, 2].set_ylabel("Energy distance / TNG sampling floor")
    axes[0, 2].legend(frameon=False, fontsize=8)

    names = ("gamma1p25", "gamma1p4", "hybrid")
    labels = (r"$\gamma=1.25$", r"$\gamma=1.4$", "mass-dependent")
    axes[1, 0].bar(
        np.arange(3),
        [result["models"][name]["profile_crps_dex"] for name in names],
        color=(OKABE_ITO[0], OKABE_ITO[1], OKABE_ITO[2]),
    )
    axes[1, 0].set_xticks(np.arange(3), labels, rotation=15)
    axes[1, 0].set_ylabel("Profile CRPS (dex)")

    thresholds = [entry["halo_mass_threshold"] for entry in result["metadata"]]
    axes[1, 2].scatter(np.arange(5), thresholds, color=OKABE_ITO[2], s=55)
    axes[1, 2].axhline(np.mean(thresholds), color="0.4", linestyle="--")
    axes[1, 2].set_xlabel("Outer fold")
    axes[1, 2].set_ylabel(r"Training 2/3 quantile in $\log M_{\rm peak}$")
    axes[1, 2].set_xticks(np.arange(5))
    axes[1, 2].ticklabel_format(axis="y", style="plain", useOffset=False)
    for panel, axis in zip("ABCDEF", axes.ravel()):
        axis.text(-0.13, 1.04, panel, transform=axis.transAxes, fontweight="bold")
    fig.suptitle("exp55 — fold-local mass-dependent outer-slope boundary test")
    fig.tight_layout()
    save_fig(fig, FIGURE_DIRECTORY / "exp55_mass_dependent_outer")
    plt.close(fig)


def json_safe(value):
    """Convert nested NumPy values for JSON."""
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


def run_experiment(
    n_max: int = N_MAX,
    n_draw: int = N_DRAW,
    n_population_draw: int = N_POPULATION_DRAW,
) -> dict:
    """Execute the mass-dependent boundary test."""
    started = time.perf_counter()
    sample = load_sample(n_max)
    family_1p25 = aligned_profile_family(sample, "1p25")
    family_1p4 = aligned_profile_family(sample, "1p4")
    hybrid = cross_validated_hybrid(sample, family_1p25, family_1p4, n_draw)
    hybrid_population = stochastic_population_summary(
        hybrid["draw_log"],
        sample["truth_log"],
        n_population_draw=n_population_draw,
    )
    hybrid_individual = individual_population_metrics(
        hybrid["draw_log"], sample["truth_log"], min(8, n_draw)
    )

    outer_predictions = np.load(OUTER_OUTPUT / "predictions.npz")
    outer_results = json.loads((OUTER_OUTPUT / "results.json").read_text())
    models = {}
    truth_r90 = size_relation(sample["truth_log"])["R90"]
    for name, label in (("gamma1p25", "gamma1p25"), ("gamma1p4", "gamma1p4")):
        point_log = np.asarray(outer_predictions[f"{name}_parameter_mean_log"], float)
        family = family_1p25 if name == "gamma1p25" else family_1p4
        population = outer_results["gamma_comparison"][name]["draw_population"]
        models[name] = {
            "point_log": point_log,
            "representation_r90_error": size_relation_error(
                family["prediction_log"], sample["truth_log"], "R90"
            ),
            "profile_crps_dex": outer_results["gamma_comparison"][name][
                "draw_marginal"
            ]["profile_crps_dex"],
            "profile_coverage_68": outer_results["gamma_comparison"][name][
                "draw_marginal"
            ]["profile_coverage_68"],
            "fixed_one_energy": population["planes"][PLANE_KEYS[0]]["energy_ratio"],
            "fixed_two_energy": population["planes"][PLANE_KEYS[1]]["energy_ratio"],
            "re_energy": population["planes"][PLANE_KEYS[2]]["energy_ratio"],
            "r90_energy": population["sizes"]["R90"]["energy_ratio"],
            "r90_slope": population["sizes"]["R90"]["slope"],
        }
    hybrid_point = profile_point_summary(
        hybrid["mean_log"], sample["truth_log"], sample["features"][:, 0]
    )
    hybrid_representation = profile_point_summary(
        hybrid["representation_log"],
        sample["truth_log"],
        sample["features"][:, 0],
    )
    models["hybrid"] = {
        "point_log": hybrid["mean_log"],
        "representation_r90_error": size_relation_error(
            hybrid["representation_log"], sample["truth_log"], "R90"
        ),
        "point": hybrid_point,
        "representation": hybrid_representation,
        **profile_draw_summary(hybrid["draw_log"], sample["truth_log"]),
        "fixed_one_energy": hybrid_population["planes"][PLANE_KEYS[0]]["energy_ratio"],
        "fixed_two_energy": hybrid_population["planes"][PLANE_KEYS[1]]["energy_ratio"],
        "re_energy": hybrid_population["planes"][PLANE_KEYS[2]]["energy_ratio"],
        "r90_energy": hybrid_population["sizes"]["R90"]["energy_ratio"],
        "r90_slope": hybrid_population["sizes"]["R90"]["slope"],
        "r90_scatter": hybrid_population["sizes"]["R90"]["scatter"],
        "r90_median_dlogr": hybrid_population["sizes"]["R90"]["median_dlogR"],
        "population_draw_intervals": summarize_individual_populations(
            hybrid_individual
        ),
    }
    comparison = models["gamma1p4"]
    gates = {
        "representation_profile_rms_better": hybrid_representation[
            "median_profile_rms_dex"
        ]
        < outer_results["gamma_comparison"]["gamma1p4"]["representation"][
            "median_profile_rms_dex"
        ],
        "representation_density_rms_better": hybrid_representation[
            "median_density_rms_dex"
        ]
        < outer_results["gamma_comparison"]["gamma1p4"]["representation"][
            "median_density_rms_dex"
        ],
        "coverage_within_3_percentage_points": abs(
            models["hybrid"]["profile_coverage_68"] - 0.68
        )
        <= 0.03,
        "crps_within_1_percent": models["hybrid"]["profile_crps_dex"]
        <= 1.01 * comparison["profile_crps_dex"],
        "fixed_one_within_10_percent": models["hybrid"]["fixed_one_energy"]
        <= 1.10 * comparison["fixed_one_energy"],
        "fixed_two_within_10_percent": models["hybrid"]["fixed_two_energy"]
        <= 1.10 * comparison["fixed_two_energy"],
        "re_within_10_percent": models["hybrid"]["re_energy"]
        <= 1.10 * comparison["re_energy"],
        "r90_better": models["hybrid"]["r90_energy"] < comparison["r90_energy"],
    }
    controls = control_audit(sample, family_1p25, family_1p4)
    result = {
        "n_galaxy": len(sample["indices"]),
        "n_draw": n_draw,
        "n_population_draw": n_population_draw,
        "truth_r90": truth_r90,
        "models": models,
        "metadata": hybrid["metadata"],
        "controls": controls,
        "gates": gates,
        "selected": bool(all(gates.values())),
        "truth_log": sample["truth_log"],
        "halo_mass": sample["features"][:, 0],
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIRECTORY / "results.json").write_text(
        json.dumps(json_safe(result), indent=2)
    )
    np.savez_compressed(
        OUTPUT_DIRECTORY / "predictions.npz",
        indices=sample["indices"],
        truth_log=sample["truth_log"],
        halo_mass=sample["features"][:, 0],
        gamma=hybrid["gamma"],
        representation_log=hybrid["representation_log"],
        mean_log=hybrid["mean_log"],
        draw_log=hybrid["draw_log"],
    )
    write_manifest(
        OUTPUT_DIRECTORY,
        params={
            "script": Path(__file__).name,
            "n_galaxy": len(sample["indices"]),
            "n_draw": n_draw,
            "n_population_draw": n_population_draw,
            "rule": "gamma=1.4 below outer-train 2/3 mass quantile; 1.25 above",
        },
    )
    comparison_figure(result)
    return result


def size_relation_error(
    model_log: np.ndarray,
    truth_log: np.ndarray,
    size_name: str,
) -> np.ndarray:
    """Per-galaxy log-radius difference for one enclosed-mass fraction."""
    column = SIZE_NAMES.index(size_name)
    from outer_envelope import size_values

    return size_values(model_log)[:, column] - size_values(truth_log)[:, column]


def generate_standard_qa(result: dict, predictions: dict) -> None:
    """Generate standard QA only for a candidate passing all Stage 9 gates."""
    if not result["selected"]:
        return
    truth_log = np.asarray(predictions["truth_log"], float)
    mean_log = np.asarray(predictions["mean_log"], float)
    draw_log = np.asarray(predictions["draw_log"], float)
    halo_mass = np.asarray(predictions["halo_mass"], float)
    STANDARD_QA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    qa_result = qa.evaluate(
        10.0 ** mean_log[:, None, :],
        10.0 ** truth_log[:, None, :],
        RADII,
        [0.4],
        name="exp55_mass_dependent_outer",
        figdir=STANDARD_QA_DIRECTORY,
        bin_by=halo_mass,
        bin_label=r"DiffMAH $\log M_{\rm peak}$",
        draw_cogs=10.0 ** draw_log[:, :, None, :],
    )
    density_by_halo_mass_figure(
        mean_log,
        truth_log,
        halo_mass,
        figdir=STANDARD_QA_DIRECTORY,
        model_name="exp55_mass_dependent_outer",
        model_label="mass-dependent outer slope",
    )
    single_epoch_planes_figure(
        qa_result,
        figdir=STANDARD_QA_DIRECTORY,
        model_name="exp55_mass_dependent_outer",
        model_label="mass-dependent outer slope",
    )
    single_epoch_size_figure(
        mean_log,
        truth_log,
        qa_result,
        figdir=STANDARD_QA_DIRECTORY,
        model_name="exp55_mass_dependent_outer",
        model_label="mass-dependent outer slope",
    )


def regenerate_figures() -> None:
    """Rebuild custom and conditional standard QA from saved results."""
    result = json.loads((OUTPUT_DIRECTORY / "results.json").read_text())
    predictions = np.load(OUTPUT_DIRECTORY / "predictions.npz")
    comparison_figure(result)
    generate_standard_qa(result, predictions)


def demo() -> None:
    """Small complete validation before the full boundary test."""
    result = run_experiment(n_max=90, n_draw=4, n_population_draw=1)
    assert result["n_galaxy"] == 90
    assert len(result["metadata"]) == 5
    assert set(result["gates"])
    print("exp55 mass-dependent outer demo OK", flush=True)


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if command == "demo":
        demo()
    elif command == "run":
        result = run_experiment()
        predictions = np.load(OUTPUT_DIRECTORY / "predictions.npz")
        generate_standard_qa(result, predictions)
        print(
            f"exp55 mass-dependent outer run complete in "
            f"{result['elapsed_seconds']:.1f} s",
            flush=True,
        )
    elif command == "figures":
        regenerate_figures()
    else:
        raise SystemExit(f"unknown command {command!r}; use demo, run, or figures")
