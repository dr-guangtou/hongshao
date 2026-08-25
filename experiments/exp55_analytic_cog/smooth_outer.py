"""exp55 Stage 10: fold-clean smooth outer-slope feasibility test.

The extended-component Moffat slope follows a fixed logistic function of
DiffMAH final peak halo mass. The transition is the outer-training sample's
2/3 mass quantile. Stage 10 compares three predeclared widths; the separately
declared Stage 11 closure adds three wider values. Every galaxy is refitted at
its prescribed slope, so this is an analytic profile rather than an
interpolation between final profiles.

Commands
--------
demo
    Run the complete path on 90 galaxies and four draws.
run
    Run the full five-fold experiment and conditional standard QA.
figures
    Rebuild figures from saved full-sample outputs.
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
from hongshao.profile_emulator import density_from_cog  # noqa: E402
from hongshao.provenance import write_manifest  # noqa: E402
from halo_map import RADII, folds_of, load_sample, shuffled_within_mass_bins  # noqa: E402
from mass_dependent_outer import (  # noqa: E402
    OUTPUT_DIRECTORY as BOUNDARY_OUTPUT,
)
from mixed import fit_one, mixed_cog_log  # noqa: E402
from outer_envelope import (  # noqa: E402
    PLANE_KEYS,
    aligned_profile_family,
    individual_population_metrics,
    profile_draw_summary,
    profile_point_summary,
    size_relation,
    size_values,
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

FIGURE_DIRECTORY = HERE / "figures" / "smooth_outer"
OUTPUT_DIRECTORY = HERE / "outputs" / "smooth_outer"
CHECKPOINT_DIRECTORY = OUTPUT_DIRECTORY / "profile_fits"
STANDARD_QA_DIRECTORY = FIGURE_DIRECTORY / "standard_qa"

WIDTHS = (0.05, 0.10, 0.20, 0.30, 0.50, 0.80)
N_MAX = int(os.environ.get("EXP55_SMOOTH_NMAX", 0))
N_DRAW = int(os.environ.get("EXP55_SMOOTH_N_DRAW", 32))
N_POPULATION_DRAW = int(os.environ.get("EXP55_SMOOTH_N_POPULATION_DRAW", 4))
SEED = 10550


def smooth_gamma(
    halo_mass: np.ndarray, transition_mass: float, width: float
) -> np.ndarray:
    """Logistic slope from 1.4 at low mass to 1.25 at high mass."""
    argument = np.clip((np.asarray(halo_mass) - transition_mass) / width, -50, 50)
    return 1.25 + 0.15 / (1.0 + np.exp(argument))


def decode_profiles(parameters: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """Decode arbitrary per-galaxy outer slopes into complete log CoGs."""
    parameters = np.asarray(parameters, float)
    gamma = np.asarray(gamma, float)
    if parameters.ndim == 2:
        if len(parameters) != len(gamma):
            raise ValueError("gamma array does not match the galaxy axis")
        return np.asarray(
            [
                mixed_cog_log(row, RADII, n_sersic=1.0, gamma=float(slope))
                for row, slope in zip(parameters, gamma)
            ]
        )
    if parameters.ndim == 3:
        return np.asarray([decode_profiles(draw, gamma) for draw in parameters])
    raise ValueError("parameters must have galaxy or draw-by-galaxy shape")


def fit_profile_candidate(
    truth_log: np.ndarray,
    gamma: np.ndarray,
    fold_number: int,
    width: float,
) -> dict:
    """Fit and checkpoint all galaxy profiles for one fold-local smooth law."""
    path = CHECKPOINT_DIRECTORY / f"fold{fold_number}_width{width:.2f}.npz"
    if path.exists():
        saved = dict(np.load(path))
        if len(saved["parameters"]) == len(truth_log) and np.allclose(
            saved["gamma"], gamma
        ):
            return saved
    arrays = {
        "parameters": np.empty((len(truth_log), 4)),
        "prediction_log": np.empty_like(truth_log),
        "profile_rms_dex": np.empty(len(truth_log)),
        "jacobian_min_ratio": np.empty(len(truth_log)),
        "boundary_distance": np.empty(len(truth_log)),
        "success": np.empty(len(truth_log), dtype=bool),
        "gamma": np.asarray(gamma),
    }
    started = time.perf_counter()
    for galaxy, (truth, slope) in enumerate(zip(truth_log, gamma)):
        result = fit_one(truth, n_sersic=1.0, gamma=float(slope))
        arrays["parameters"][galaxy] = result["parameters"]
        arrays["prediction_log"][galaxy] = result["prediction"]
        arrays["profile_rms_dex"][galaxy] = result["rms_dex"]
        arrays["jacobian_min_ratio"][galaxy] = result["jacobian_min_ratio"]
        arrays["boundary_distance"][galaxy] = result["boundary_distance"]
        arrays["success"][galaxy] = result["success"]
    CHECKPOINT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    print(
        f"fold {fold_number}, width {width:.2f}: {len(truth_log)} profiles in "
        f"{time.perf_counter() - started:.1f} s",
        flush=True,
    )
    return arrays


def representation_metrics(
    candidate: dict,
    truth_log: np.ndarray,
    rows: np.ndarray,
) -> dict:
    """Training-only selection metrics for one smooth-law candidate."""
    prediction = candidate["prediction_log"][rows]
    truth = truth_log[rows]
    model_density, _ = density_from_cog(prediction, RADII)
    truth_density, _ = density_from_cog(truth, RADII)
    model_slope = size_relation(prediction)["R90"]["slope"]
    truth_slope = size_relation(truth)["R90"]["slope"]
    return {
        "median_profile_rms_dex": float(np.median(candidate["profile_rms_dex"][rows])),
        "median_density_rms_dex": float(
            np.median(np.sqrt(np.mean((model_density - truth_density) ** 2, axis=1)))
        ),
        "boundary_fraction": float(
            np.mean(candidate["boundary_distance"][rows] <= 0.01)
        ),
        "r90_slope": model_slope,
        "truth_r90_slope": truth_slope,
        "absolute_r90_slope_error": abs(model_slope - truth_slope),
    }


def augmented_features(features: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """Append the deterministic local profile slope to the halo vector."""
    return np.column_stack([features, gamma])


def neighbour_draws(
    model,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    n_draw: int,
    seed: int,
    n_neighbour: int = 128,
) -> np.ndarray:
    """Resample complete residual vectors from nearby augmented halo inputs."""
    residual = train_targets - model.predict_mean(train_features)
    standardized_train = model.standardized(train_features)
    standardized_test = model.standardized(test_features)
    tree = cKDTree(standardized_train)
    _, neighbours = tree.query(
        standardized_test, k=min(n_neighbour, len(train_features))
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
    model,
    n_draw: int,
    seed: int,
) -> tuple[np.ndarray, dict]:
    """Select one residual inflation using inner held-out profile coverage."""
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
        inner_draw[:, validation] = neighbour_draws(
            inner_model,
            train_features[training],
            train_targets[training],
            train_features[validation],
            n_draw,
            seed + inner_number,
        )
    coverage = {}
    for scale in scale_grid:
        draw_log = decode_profiles(
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
    }


def select_width(
    candidates: dict[float, dict],
    truth_log: np.ndarray,
    train: np.ndarray,
    fixed_1p25: dict,
    fixed_1p4: dict,
) -> tuple[float | None, dict]:
    """Apply the predeclared ordered representation gates on training rows."""
    reference_1p4 = representation_metrics(fixed_1p4, truth_log, train)
    reference_1p25 = representation_metrics(fixed_1p25, truth_log, train)
    records = {}
    passing = []
    for width, candidate in candidates.items():
        metrics = representation_metrics(candidate, truth_log, train)
        gates = {
            "profile_rms_within_0p0001_dex": metrics["median_profile_rms_dex"]
            <= reference_1p4["median_profile_rms_dex"] + 0.0001,
            "density_rms_not_worse": metrics["median_density_rms_dex"]
            <= reference_1p4["median_density_rms_dex"],
            "boundary_below_original": metrics["boundary_fraction"]
            <= reference_1p25["boundary_fraction"],
        }
        records[f"{width:.2f}"] = {"metrics": metrics, "gates": gates}
        if all(gates.values()):
            passing.append(width)
    selected = (
        min(
            passing,
            key=lambda width: (
                records[f"{width:.2f}"]["metrics"]["absolute_r90_slope_error"],
                width,
            ),
        )
        if passing
        else None
    )
    return selected, {
        "fixed_gamma1p4": reference_1p4,
        "fixed_gamma1p25": reference_1p25,
        "candidates": records,
    }


def cross_validated_smooth(sample: dict, n_draw: int) -> dict:
    """Select widths, fit the halo map, and generate every outer fold."""
    features = sample["features"]
    halo_mass = features[:, 0]
    truth_log = sample["truth_log"]
    fixed_1p25 = aligned_profile_family(sample, "1p25")
    fixed_1p4 = aligned_profile_family(sample, "1p4")
    mean_parameters = np.empty((len(features), 4))
    representation_log = np.empty_like(truth_log)
    gamma_by_galaxy = np.empty(len(features))
    draw_parameters = np.empty((n_draw, len(features), 4))
    control_predictions = {
        name: np.empty_like(truth_log)
        for name in (
            "complete",
            "final_mass_only",
            "mah_shape_shuffled",
            "concentration_shuffled",
        )
    }
    shuffled_mah = shuffled_within_mass_bins(features, (1, 2, 3), seed=SEED + 1)
    shuffled_concentration = shuffled_within_mass_bins(features, (4,), seed=SEED + 2)
    variants = {
        "complete": (features, SPARSE_TERMS),
        "final_mass_only": (features[:, :1], ()),
        "mah_shape_shuffled": (shuffled_mah, SPARSE_TERMS),
        "concentration_shuffled": (shuffled_concentration, SPARSE_TERMS),
    }
    metadata = []
    all_rows = np.arange(len(features))
    for fold_number, test in enumerate(folds_of(len(features))):
        train = np.setdiff1d(all_rows, test)
        transition = float(np.quantile(halo_mass[train], 2.0 / 3.0))
        candidates = {}
        for width in WIDTHS:
            gamma = smooth_gamma(halo_mass, transition, width)
            candidates[width] = fit_profile_candidate(
                truth_log, gamma, fold_number, width
            )
        selected_width, selection = select_width(
            candidates,
            truth_log,
            train,
            fixed_1p25,
            fixed_1p4,
        )
        if selected_width is None:
            raise RuntimeError(f"no smooth width passed in outer fold {fold_number}")
        candidate = candidates[selected_width]
        gamma = candidate["gamma"]
        train_targets = candidate["parameters"][train]
        representation_log[test] = candidate["prediction_log"][test]
        gamma_by_galaxy[test] = gamma[test]
        complete_train = augmented_features(features[train], gamma[train])
        complete_test = augmented_features(features[test], gamma[test])
        complete_model = fit_conditional_model(
            complete_train, train_targets, SPARSE_TERMS
        )
        mean_parameters[test] = complete_model.predict_mean(complete_test)
        draw_parameters[:, test], calibration = calibrated_fold_draws(
            complete_train,
            train_targets,
            gamma[train],
            truth_log[train],
            complete_test,
            gamma[test],
            complete_model,
            n_draw,
            SEED + 1000 * fold_number,
        )
        for name, (control_features, terms) in variants.items():
            train_values = augmented_features(control_features[train], gamma[train])
            test_values = augmented_features(control_features[test], gamma[test])
            model = fit_conditional_model(train_values, train_targets, terms)
            control_predictions[name][test] = decode_profiles(
                model.predict_mean(test_values), gamma[test]
            )
        metadata.append(
            {
                "fold": fold_number,
                "transition_mass": transition,
                "selected_width": selected_width,
                "selection": selection,
                **calibration,
            }
        )
    return {
        "mean_parameters": mean_parameters,
        "mean_log": decode_profiles(mean_parameters, gamma_by_galaxy),
        "draw_parameters": draw_parameters,
        "draw_log": decode_profiles(draw_parameters, gamma_by_galaxy),
        "gamma": gamma_by_galaxy,
        "representation_log": representation_log,
        "control_predictions": control_predictions,
        "metadata": metadata,
    }


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


def comparison_figure(result: dict) -> None:
    """Show fold selection and final boundary-versus-smooth performance."""
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.5))
    width_colors = {
        width: plt.colormaps["cividis"](index / max(len(WIDTHS) - 1, 1))
        for index, width in enumerate(WIDTHS)
    }
    for fold in result["metadata"]:
        for width in WIDTHS:
            record = fold["selection"]["candidates"][f"{width:.2f}"]["metrics"]
            axes[0, 0].scatter(
                width,
                record["median_profile_rms_dex"],
                color=width_colors[width],
                alpha=0.7,
            )
            axes[0, 1].scatter(
                width,
                record["median_density_rms_dex"],
                color=width_colors[width],
                alpha=0.7,
            )
            axes[0, 2].scatter(
                width,
                record["absolute_r90_slope_error"],
                color=width_colors[width],
                alpha=0.7,
            )
    axes[0, 0].set_ylabel("Training median CoG RMS (dex)")
    axes[0, 1].set_ylabel("Training median density RMS (dex)")
    axes[0, 2].set_ylabel("Training |R90 slope error|")
    for axis in axes[0]:
        axis.set_xlabel("Logistic width (dex)")
        axis.set_xticks(WIDTHS)

    boundary = result["references"]["boundary"]
    smooth = result["smooth"]
    metrics = (
        "profile_crps_dex",
        "fixed_one_energy",
        "fixed_two_energy",
        "re_energy",
        "r90_energy",
    )
    labels = ("profile\nCRPS", "30--50", "50--100", "2--4 Re", "R90")
    boundary_values = np.asarray([boundary[name] for name in metrics])
    smooth_values = np.asarray([smooth[name] for name in metrics])
    axes[1, 0].bar(
        np.arange(len(metrics)) - 0.18,
        np.ones(len(metrics)),
        0.36,
        label="hard boundary",
        color=OKABE_ITO[1],
    )
    axes[1, 0].bar(
        np.arange(len(metrics)) + 0.18,
        smooth_values / boundary_values,
        0.36,
        label="smooth law",
        color=OKABE_ITO[2],
    )
    axes[1, 0].set_xticks(np.arange(len(metrics)), labels)
    axes[1, 0].axhline(1.0, color="0.4", linewidth=0.8)
    axes[1, 0].set_ylabel("Metric / hard-boundary value")
    axes[1, 0].legend(frameon=False, fontsize=8)

    halo_mass = np.asarray(result["halo_mass"])
    truth_r90 = size_values(np.asarray(result["truth_log"]))[:, 2]
    for name, values, color in (
        ("TNG", truth_r90, "0.25"),
        ("hard boundary", np.asarray(result["boundary_r90"]), OKABE_ITO[1]),
        ("smooth law", np.asarray(result["smooth_r90"]), OKABE_ITO[2]),
    ):
        chunks = np.array_split(np.argsort(halo_mass), 12)
        axes[1, 1].plot(
            [np.median(halo_mass[group]) for group in chunks],
            [np.median(values[group]) for group in chunks],
            marker="o",
            color=color,
            label=name,
        )
    axes[1, 1].set_xlabel(r"DiffMAH $\log M_{\rm peak}$")
    axes[1, 1].set_ylabel(r"Median $\log R_{90}$ (kpc)")
    axes[1, 1].legend(frameon=False, fontsize=8)

    selected = [entry["selected_width"] for entry in result["metadata"]]
    axes[1, 2].bar(np.arange(5), selected, color=OKABE_ITO[2])
    axes[1, 2].set_xlabel("Outer fold")
    axes[1, 2].set_ylabel("Selected logistic width (dex)")
    axes[1, 2].set_xticks(np.arange(5))
    for panel, axis in zip("ABCDEF", axes.ravel()):
        axis.text(-0.13, 1.04, panel, transform=axis.transAxes, fontweight="bold")
    fig.suptitle("exp55 — smooth halo-mass-dependent outer-slope feasibility")
    fig.tight_layout()
    save_fig(fig, FIGURE_DIRECTORY / "exp55_smooth_outer")
    plt.close(fig)


def run_experiment(
    n_max: int = N_MAX,
    n_draw: int = N_DRAW,
    n_population_draw: int = N_POPULATION_DRAW,
) -> dict:
    """Execute the full smooth outer-slope test."""
    started = time.perf_counter()
    sample = load_sample(n_max)
    smooth = cross_validated_smooth(sample, n_draw)
    population = stochastic_population_summary(
        smooth["draw_log"],
        sample["truth_log"],
        n_population_draw=n_population_draw,
    )
    individuals = individual_population_metrics(
        smooth["draw_log"], sample["truth_log"], min(32, n_draw)
    )
    point = profile_point_summary(
        smooth["mean_log"], sample["truth_log"], sample["features"][:, 0]
    )
    representation = profile_point_summary(
        smooth["representation_log"],
        sample["truth_log"],
        sample["features"][:, 0],
    )
    draw_summary = profile_draw_summary(smooth["draw_log"], sample["truth_log"])
    smooth_metrics = {
        "point": point,
        "representation": representation,
        **draw_summary,
        "fixed_one_energy": population["planes"][PLANE_KEYS[0]]["energy_ratio"],
        "fixed_two_energy": population["planes"][PLANE_KEYS[1]]["energy_ratio"],
        "re_energy": population["planes"][PLANE_KEYS[2]]["energy_ratio"],
        "r90_energy": population["sizes"]["R90"]["energy_ratio"],
        "r90_slope": population["sizes"]["R90"]["slope"],
        "r90_scatter": population["sizes"]["R90"]["scatter"],
        "r90_median_dlogr": population["sizes"]["R90"]["median_dlogR"],
        "population_draw_intervals": summarize_individual_populations(individuals),
    }
    boundary_result = json.loads((BOUNDARY_OUTPUT / "results.json").read_text())
    boundary_predictions = np.load(BOUNDARY_OUTPUT / "predictions.npz")
    boundary_row_by_index = {
        int(value): row
        for row, value in enumerate(np.asarray(boundary_predictions["indices"]))
    }
    boundary_rows = np.asarray(
        [boundary_row_by_index[int(value)] for value in sample["indices"]], int
    )
    boundary_draw_log = np.asarray(boundary_predictions["draw_log"])[:, boundary_rows]
    boundary_individuals = individual_population_metrics(
        boundary_draw_log,
        sample["truth_log"],
        min(32, boundary_draw_log.shape[0]),
    )
    boundary_model = boundary_result["models"]["hybrid"]
    boundary = {
        "profile_crps_dex": boundary_model["profile_crps_dex"],
        "profile_coverage_68": boundary_model["profile_coverage_68"],
        "fixed_one_energy": boundary_model["fixed_one_energy"],
        "fixed_two_energy": boundary_model["fixed_two_energy"],
        "re_energy": boundary_model["re_energy"],
        "r90_energy": boundary_model["r90_energy"],
        "worst_mass_bin_shape_residual_percent": boundary_model["point"][
            "worst_mass_bin_shape_residual_percent"
        ],
    }
    fixed_1p4_r90 = boundary_result["models"]["gamma1p4"]["r90_energy"]
    gates = {
        "coverage_within_3_percentage_points": abs(
            smooth_metrics["profile_coverage_68"] - 0.68
        )
        <= 0.03,
        "crps_within_1_percent_of_boundary": smooth_metrics["profile_crps_dex"]
        <= 1.01 * boundary["profile_crps_dex"],
        "fixed_one_within_10_percent_of_boundary": smooth_metrics["fixed_one_energy"]
        <= 1.10 * boundary["fixed_one_energy"],
        "fixed_two_within_10_percent_of_boundary": smooth_metrics["fixed_two_energy"]
        <= 1.10 * boundary["fixed_two_energy"],
        "re_within_10_percent_of_boundary": smooth_metrics["re_energy"]
        <= 1.10 * boundary["re_energy"],
        "shape_residual_not_worse_than_boundary": point[
            "worst_mass_bin_shape_residual_percent"
        ]
        <= boundary["worst_mass_bin_shape_residual_percent"],
        "r90_within_10_percent_of_boundary": smooth_metrics["r90_energy"]
        <= 1.10 * boundary["r90_energy"],
        "r90_better_than_fixed_gamma1p4": smooth_metrics["r90_energy"] < fixed_1p4_r90,
    }
    controls = {
        name: profile_point_summary(
            prediction, sample["truth_log"], sample["features"][:, 0]
        )
        for name, prediction in smooth["control_predictions"].items()
    }
    result = {
        "n_galaxy": len(sample["indices"]),
        "n_draw": n_draw,
        "n_population_draw": n_population_draw,
        "smooth": smooth_metrics,
        "references": {
            "boundary": boundary,
            "boundary_population_draw_intervals": summarize_individual_populations(
                boundary_individuals
            ),
            "fixed_gamma1p4_r90_energy": fixed_1p4_r90,
        },
        "metadata": smooth["metadata"],
        "controls": controls,
        "gates": gates,
        "selected": bool(all(gates.values())),
        "truth_log": sample["truth_log"],
        "halo_mass": sample["features"][:, 0],
        "smooth_r90": size_values(smooth["draw_log"][0])[:, 2],
        "boundary_r90": size_values(boundary_draw_log[0])[:, 2],
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
        gamma=smooth["gamma"],
        representation_log=smooth["representation_log"],
        mean_log=smooth["mean_log"],
        draw_log=smooth["draw_log"],
    )
    write_manifest(
        OUTPUT_DIRECTORY,
        params={
            "script": Path(__file__).name,
            "n_galaxy": len(sample["indices"]),
            "n_draw": n_draw,
            "n_population_draw": n_population_draw,
            "widths": WIDTHS,
            "transition": "outer-training 2/3 halo-mass quantile",
        },
    )
    comparison_figure(result)
    return result


def generate_standard_qa(result: dict, predictions: dict) -> None:
    """Generate standard QA for this scientifically informative candidate."""
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
        name="exp55_smooth_outer",
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
        model_name="exp55_smooth_outer",
        model_label="smooth outer slope",
    )
    single_epoch_planes_figure(
        qa_result,
        figdir=STANDARD_QA_DIRECTORY,
        model_name="exp55_smooth_outer",
        model_label="smooth outer slope",
    )
    single_epoch_size_figure(
        mean_log,
        truth_log,
        qa_result,
        figdir=STANDARD_QA_DIRECTORY,
        model_name="exp55_smooth_outer",
        model_label="smooth outer slope",
    )


def regenerate_figures() -> None:
    """Rebuild custom and conditional standard QA from saved outputs."""
    result = json.loads((OUTPUT_DIRECTORY / "results.json").read_text())
    predictions = np.load(OUTPUT_DIRECTORY / "predictions.npz")
    comparison_figure(result)
    generate_standard_qa(result, predictions)


def demo() -> None:
    """Small complete validation before the full smooth-law test."""
    result = run_experiment(n_max=90, n_draw=4, n_population_draw=1)
    assert result["n_galaxy"] == 90
    assert len(result["metadata"]) == 5
    assert set(result["gates"])
    print("exp55 smooth outer demo OK", flush=True)


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if command == "demo":
        demo()
    elif command == "run":
        result = run_experiment()
        predictions = np.load(OUTPUT_DIRECTORY / "predictions.npz")
        generate_standard_qa(result, predictions)
        print(
            f"exp55 smooth outer run complete in {result['elapsed_seconds']:.1f} s",
            flush=True,
        )
    elif command == "figures":
        regenerate_figures()
    else:
        raise SystemExit(f"unknown command {command!r}; use demo, run, or figures")
