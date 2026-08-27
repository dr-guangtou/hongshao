"""Expose the Stage 3b paired errors and fitted halo–CoG coefficients.

Commands
--------
demo
    Refit all relations on the predeclared 90-galaxy sample in under one minute.
run
    Refit the five held-out relations and one full-sample descriptive relation.
figures
    Build the paired-error and coefficient figures from saved artifacts.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP55 = ROOT / "experiments" / "exp55_analytic_cog"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP55))
sys.path.insert(0, str(HERE))

import stage3_halo_cog as stage3  # noqa: E402
import stage3b_direct_cog as stage3b  # noqa: E402
from hongshao.plotting import OKABE_ITO, save_fig, set_style  # noqa: E402

set_style()
matplotlib.rcParams["text.usetex"] = False

OUTPUT_DIRECTORY = HERE / "outputs" / "stage3b_relation_diagnostics"
FIGURE_DIRECTORY = HERE / "figures" / "stage3b"
N_DEMO_GALAXY = 90
DEMO_TIME_LIMIT_SECONDS = 60.0
SMOOTH_WIDTH_DEX = 0.20
FEATURE_NAMES = (
    "log_m_peak",
    "log_t_c",
    "early_diffmah",
    "late_diffmah",
    "c_200c",
    "gamma",
)
FEATURE_LABELS = (
    r"$\log M_{\rm peak}$",
    r"$\log t_c$",
    "early",
    "late",
    r"$c_{200c}$",
    r"$\gamma$",
)
TARGET_NAMES = (
    "log_m_star_148",
    "logit_compact_fraction",
    "log_r_compact",
    "log_extended_compact_ratio_minus_one",
)
TARGET_LABELS = (
    r"$\log M_{*,148}$",
    r"$\mathrm{logit}(f_c)$",
    r"$\log R_c$",
    r"$\ln(R_e/R_c-1)$",
)
METRIC_NAMES = (
    "full_density_rms_dex",
    "five_thirty_density_rms_dex",
    "absolute_R50_error",
    "absolute_R80_error",
    "absolute_R90_error",
)
METRIC_LABELS = (
    "Density RMS\n(all annuli)",
    "Density RMS\n(5–30 kpc)",
    r"$|\Delta\log R_{50}|$",
    r"$|\Delta\log R_{80}|$",
    r"$|\Delta\log R_{90}|$",
)


def smooth_gamma(halo_mass: np.ndarray, transition_mass: float) -> np.ndarray:
    """Frozen logistic outer-slope law used by the smooth representation."""
    argument = np.clip(
        (np.asarray(halo_mass, float) - transition_mass) / SMOOTH_WIDTH_DEX,
        -50.0,
        50.0,
    )
    return 1.25 + 0.15 / (1.0 + np.exp(argument))


def basis_names(n_feature: int, terms: tuple[tuple, ...]) -> tuple[str, ...]:
    """Names in the exact order used by the fitted design matrix."""
    names = ["intercept", *FEATURE_NAMES[:n_feature]]
    for term in terms:
        names.append(" × ".join(FEATURE_NAMES[index] for index in term))
    return tuple(names)


def collated_profile_coordinates(
    sample: dict[str, np.ndarray], representation: str, demo: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble the fold-local fitted coordinates used by held-out QA."""
    n_galaxy = len(sample["indices"])
    parameters = np.empty((n_galaxy, 4))
    gamma = np.empty(n_galaxy)
    for fold_number, test in enumerate(stage3.folds_of(n_galaxy)):
        cell = stage3.load_profile_cell(sample, representation, fold_number, demo)
        parameters[test] = cell["parameters"][test]
        gamma[test] = cell["gamma"][test]
    return parameters, gamma


def fit_relation_coefficients(n_max: int, demo: bool) -> dict:
    """Fit the five QA relations plus a descriptive full-sample relation."""
    sample = stage3.stage_1.load_sample(n_max)
    base_features = sample["features"]
    truth_log = sample["truth_log"]
    all_rows = np.arange(len(base_features))
    output = {
        "n_galaxy": len(base_features),
        "feature_names": FEATURE_NAMES,
        "target_names": TARGET_NAMES,
        "representations": {},
    }
    for representation in stage3.REPRESENTATIONS:
        parameters, fold_gamma = collated_profile_coordinates(
            sample, representation, demo
        )
        folds = []
        for fold_number, test in enumerate(stage3.folds_of(len(base_features))):
            train = np.setdiff1d(all_rows, test)
            cell = stage3.load_profile_cell(sample, representation, fold_number, demo)
            gamma = np.asarray(cell["gamma"], float)
            augmented = stage3.augmented_features(base_features, gamma)
            model, fit_metadata = stage3b.fit_direct_model(
                augmented[train],
                np.asarray(cell["parameters"], float)[train],
                truth_log[train],
                gamma[train],
                stage3.SPARSE_TERMS,
            )
            folds.append(
                {
                    "fold": fold_number,
                    "n_train": len(train),
                    "n_test": len(test),
                    "feature_mean": model.feature_mean,
                    "feature_standard_deviation": model.feature_standard_deviation,
                    "coefficients": model.coefficients,
                    "fit": fit_metadata,
                }
            )

        if representation == "fixed_gamma1p4":
            full_gamma = np.full(len(base_features), 1.4)
            full_features = base_features
            transition_mass = None
        else:
            transition_mass = float(np.quantile(base_features[:, 0], 2.0 / 3.0))
            full_gamma = smooth_gamma(base_features[:, 0], transition_mass)
            full_features = stage3.augmented_features(base_features, full_gamma)
        full_model, full_fit_metadata = stage3b.fit_direct_model(
            full_features,
            parameters,
            truth_log,
            full_gamma,
            stage3.SPARSE_TERMS,
        )
        output["representations"][representation] = {
            "gamma_transition_log_m_peak": transition_mass,
            "profile_coordinate_initialization": (
                "fold-collated analytic profile coordinates; the final mean "
                "coefficients minimize decoded CoG error"
            ),
            "fold_gamma_minimum": float(np.min(fold_gamma)),
            "fold_gamma_maximum": float(np.max(fold_gamma)),
            "folds": folds,
            "full_sample": {
                "role": (
                    "identifiable descriptive/deployment refit; not used to "
                    "calculate held-out QA"
                ),
                "basis_names": basis_names(full_features.shape[1], stage3.SPARSE_TERMS),
                "feature_mean": full_model.feature_mean,
                "feature_standard_deviation": (full_model.feature_standard_deviation),
                "coefficients": full_model.coefficients,
                "fit": full_fit_metadata,
            },
        }
    return output


def write_coefficient_artifacts(result: dict, prefix: str = "") -> None:
    """Write exact machine-readable coefficients and convenient CSV tables."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIRECTORY / f"{prefix}relation_coefficients.json"
    json_path.write_text(json.dumps(stage3.json_safe(result), indent=2) + "\n")

    full_path = OUTPUT_DIRECTORY / f"{prefix}full_sample_coefficients.csv"
    with full_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["representation", "target", "basis_term", "coefficient"])
        for representation, values in result["representations"].items():
            full = values["full_sample"]
            coefficients = np.asarray(full["coefficients"])
            for target_index, target in enumerate(TARGET_NAMES):
                for basis_index, basis in enumerate(full["basis_names"]):
                    writer.writerow(
                        [
                            representation,
                            target,
                            basis,
                            coefficients[target_index, basis_index],
                        ]
                    )

    fold_path = OUTPUT_DIRECTORY / f"{prefix}heldout_fold_coefficients.csv"
    with fold_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["representation", "fold", "target", "basis_term", "coefficient"]
        )
        for representation, values in result["representations"].items():
            for fold in values["folds"]:
                coefficients = np.asarray(fold["coefficients"])
                for target_index, target in enumerate(TARGET_NAMES):
                    for basis_index, basis in enumerate(
                        basis_names(6, stage3.SPARSE_TERMS)
                    ):
                        writer.writerow(
                            [
                                representation,
                                fold["fold"],
                                target,
                                basis,
                                coefficients[target_index, basis_index],
                            ]
                        )

    scaling_path = OUTPUT_DIRECTORY / f"{prefix}full_sample_feature_scaling.csv"
    with scaling_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["representation", "feature", "mean", "standard_deviation"])
        for representation, values in result["representations"].items():
            full = values["full_sample"]
            active_features = FEATURE_NAMES[: len(full["feature_mean"])]
            for feature, mean, standard_deviation in zip(
                active_features,
                full["feature_mean"],
                full["feature_standard_deviation"],
            ):
                writer.writerow([representation, feature, mean, standard_deviation])


def paired_error_values() -> tuple[dict, dict]:
    """Calculate matched Stage 3b-minus-Stage 3 errors for every galaxy."""
    coordinate = np.load(HERE / "outputs" / "stage3" / "predictions.npz")
    direct = np.load(HERE / "outputs" / "stage3b" / "predictions.npz")
    if not np.array_equal(coordinate["indices"], direct["indices"]):
        raise RuntimeError("Stage 3 and Stage 3b galaxy indices differ")
    output = {}
    summary = {}
    for representation in stage3.REPRESENTATIONS:
        coordinate_metrics = stage3.per_galaxy_metrics(
            direct["truth_log"],
            coordinate[f"{representation}_mean_log"],
            coordinate[f"{representation}_draw_log"],
        )
        direct_metrics = stage3.per_galaxy_metrics(
            direct["truth_log"],
            direct[f"{representation}_mean_log"],
            direct[f"{representation}_draw_log"],
        )
        output[representation] = {
            metric: direct_metrics[metric] - coordinate_metrics[metric]
            for metric in METRIC_NAMES
        }
        summary[representation] = {}
        for metric_index, metric in enumerate(METRIC_NAMES):
            difference = output[representation][metric]
            summary[representation][metric] = {
                "coordinate_median_error_dex": float(
                    np.median(coordinate_metrics[metric])
                ),
                "direct_median_error_dex": float(np.median(direct_metrics[metric])),
                "paired_median_difference_dex": float(np.median(difference)),
                "fraction_individually_worse": float(np.mean(difference > 0.0)),
                "bootstrap_median_difference_16_50_84_dex": (
                    stage3.bootstrap_median_interval(
                        difference,
                        stage3b.SEED
                        + 100 * stage3.REPRESENTATIONS.index(representation)
                        + list(
                            stage3.per_galaxy_metrics(
                                direct["truth_log"],
                                direct[f"{representation}_mean_log"],
                                direct[f"{representation}_draw_log"],
                            )
                        ).index(metric),
                    )
                ),
            }
    return output, summary


def write_paired_error_artifacts(values: dict, summary: dict) -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    arrays = {
        f"{representation}_{metric}": difference
        for representation, metrics in values.items()
        for metric, difference in metrics.items()
    }
    np.savez_compressed(OUTPUT_DIRECTORY / "paired_error_differences.npz", **arrays)
    (OUTPUT_DIRECTORY / "paired_error_summary.json").write_text(
        json.dumps(stage3.json_safe(summary), indent=2) + "\n"
    )


def paired_error_figure(values: dict, summary: dict) -> None:
    """Show raw paired error changes and uncertainty on their medians."""
    figure, axes = plt.subplots(2, 1, figsize=(13.0, 8.6), sharex=True)
    colors = (OKABE_ITO[1], OKABE_ITO[2])
    positions = np.arange(len(METRIC_NAMES))
    for axis, representation, color in zip(axes, stage3.REPRESENTATIONS, colors):
        metric_values = [values[representation][metric] for metric in METRIC_NAMES]
        violins = axis.violinplot(
            metric_values,
            positions=positions,
            widths=0.78,
            showextrema=False,
            showmedians=False,
            points=200,
        )
        for body in violins["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.30)
        for position, metric, differences in zip(
            positions, METRIC_NAMES, metric_values
        ):
            low, median, high = summary[representation][metric][
                "bootstrap_median_difference_16_50_84_dex"
            ]
            axis.errorbar(
                position,
                median,
                yerr=np.asarray([[median - low], [high - median]]),
                fmt="o",
                color="black",
                markerfacecolor=color,
                markersize=6,
                capsize=4,
                linewidth=1.5,
                zorder=5,
            )
            fraction = np.mean(differences > 0.0)
            axis.text(
                position,
                np.quantile(differences, 0.98),
                f"{100.0 * fraction:.1f}% worse",
                ha="center",
                va="bottom",
                fontsize=8,
                color="0.25",
            )
            axis.text(
                position,
                np.quantile(differences, 0.015),
                f"median {median:+.4f}\n[{low:+.4f}, {high:+.4f}]",
                ha="center",
                va="top",
                fontsize=7.3,
                color="0.25",
            )
        axis.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
        axis.set_ylabel("Direct-CoG fit minus coordinate fit (dex)")
        axis.set_title(stage3.REPRESENTATION_LABELS[representation], loc="left")
        axis.grid(axis="y", color="0.90", linewidth=0.7)
    axes[-1].set_xticks(positions, METRIC_LABELS)
    figure.suptitle(
        "Directly minimizing cumulative CoG error trades away local density and sizes",
        fontsize=14,
        y=0.995,
    )
    figure.text(
        0.5,
        0.955,
        (
            "Violin: all 2,539 matched held-out galaxies. Dot and bar: median "
            "paired change and its 16–84% bootstrap interval. Positive is worse."
        ),
        ha="center",
        va="top",
        fontsize=9,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.925))
    save_fig(figure, FIGURE_DIRECTORY / "exp56_stage3b_paired_error_evidence")
    plt.close(figure)


def coefficient_label(value: float) -> str:
    if abs(value) >= 100.0:
        return f"{value:.0f}"
    if abs(value) >= 10.0:
        return f"{value:.1f}"
    return f"{value:.2f}"


def relation_coefficient_figure(result: dict) -> None:
    """Display the full-sample free coefficients and required normalization."""
    fixed = result["representations"]["fixed_gamma1p4"]["full_sample"]
    smooth = result["representations"]["smooth"]["full_sample"]
    basis = tuple(smooth["basis_names"])
    matrices = []
    for full in (fixed, smooth):
        source = np.asarray(full["coefficients"])
        source_basis = tuple(full["basis_names"])
        matrix = np.full((4, len(basis)), np.nan)
        for source_column, name in enumerate(source_basis):
            matrix[:, basis.index(name)] = source[:, source_column]
        matrices.append(matrix)
    nonzero = np.abs(
        np.concatenate([matrix[np.isfinite(matrix)].ravel() for matrix in matrices])
    )
    nonzero = nonzero[nonzero > 0.0]
    maximum = float(np.max(nonzero))
    linear_threshold = max(float(np.quantile(nonzero, 0.30)), 0.02)
    color_map = LinearSegmentedColormap.from_list(
        "blue_white_orange", (OKABE_ITO[4], "white", OKABE_ITO[0])
    )
    normalization = SymLogNorm(
        linthresh=linear_threshold,
        linscale=1.0,
        vmin=-maximum,
        vmax=maximum,
    )

    figure = plt.figure(figsize=(21.0, 14.5), layout="constrained")
    grid = figure.add_gridspec(
        4,
        2,
        height_ratios=(0.16, 0.42, 0.42, 0.19),
        width_ratios=(1.0, 0.31),
        hspace=0.38,
        wspace=0.08,
    )
    formula_axis = figure.add_subplot(grid[0, :])
    formula_axis.axis("off")
    formula_axis.text(
        0.5,
        1.02,
        (
            "Exp56 Stage 3b: free parameters of the DiffMAH + concentration "
            "halo–CoG relation"
        ),
        fontsize=15,
        ha="center",
        va="top",
    )
    formula_axis.text(
        0.0,
        0.60,
        r"Actual fitted mean relation:  $z_j=(x_j-\mu_j)/s_j$,   "
        r"$\widehat{\theta}_k=\sum_p B_{kp}\,\phi_p(\mathbf{z})$,   "
        r"$\widehat{\mathrm{CoG}}=\mathrm{Decode}(\widehat{\boldsymbol{\theta}},\gamma)$",
        fontsize=16,
        va="top",
    )
    formula_axis.text(
        0.0,
        0.08,
        (
            r"$\mathbf{x}=(\log M_{\rm peak},\log t_c,\mathrm{early},"
            r"\mathrm{late},c_{200c},\gamma)$.  Each colored cell is one "
            r"free coefficient $B_{kp}$; the table gives $\mu_j$ and $s_j$."
        ),
        fontsize=10.5,
        va="top",
    )

    heat_axes = [figure.add_subplot(grid[index, 0]) for index in (1, 2)]
    titles = (
        r"Fixed $\gamma=1.4$: full-sample descriptive refit",
        r"Smooth $\gamma(\log M_{\rm peak})$: full-sample descriptive refit",
    )
    image = None
    for axis, matrix, title in zip(heat_axes, matrices, titles):
        image = axis.imshow(
            np.ma.masked_invalid(matrix),
            aspect="auto",
            cmap=color_map,
            norm=normalization,
        )
        axis.set_xticks(np.arange(len(basis)), basis, rotation=42, ha="right")
        axis.set_yticks(np.arange(4), TARGET_LABELS)
        axis.set_title(title, loc="left", fontsize=12)
        axis.set_xlabel(r"Basis function $\phi_p(\mathbf{z})$")
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                if not np.isfinite(value):
                    axis.text(
                        column,
                        row,
                        "fixed",
                        ha="center",
                        va="center",
                        fontsize=6.7,
                        color="0.35",
                    )
                    continue
                axis.text(
                    column,
                    row,
                    coefficient_label(value),
                    ha="center",
                    va="center",
                    fontsize=6.7,
                    color="black",
                )
    if image is not None:
        color_bar = figure.colorbar(
            image, ax=heat_axes, location="right", shrink=0.78, pad=0.015
        )
        color_bar.set_label("Coefficient value (symmetric-log color scale)")

    for table_axis, representation, label in zip(
        (figure.add_subplot(grid[1, 1]), figure.add_subplot(grid[2, 1])),
        stage3.REPRESENTATIONS,
        (r"fixed $\gamma$", r"smooth $\gamma$"),
    ):
        table_axis.axis("off")
        values = result["representations"][representation]
        full = values["full_sample"]
        rows = [
            [feature_label, f"{mean:.5g}", f"{scale:.5g}"]
            for feature_label, mean, scale in zip(
                FEATURE_LABELS[: len(full["feature_mean"])],
                full["feature_mean"],
                full["feature_standard_deviation"],
            )
        ]
        table_axis.text(
            0.0,
            1.00,
            f"{label}: feature normalization",
            fontsize=11,
            fontweight="bold",
            va="top",
        )
        table = table_axis.table(
            cellText=rows,
            colLabels=("feature", r"$\mu_j$", r"$s_j$"),
            cellLoc="right",
            colLoc="right",
            bbox=(0.0, 0.28, 1.0, 0.62),
            colWidths=(0.46, 0.27, 0.27),
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.2)
        for (row, _), cell in table.get_celld().items():
            cell.set_edgecolor("0.83")
            if row == 0:
                cell.set_facecolor("0.93")
                cell.set_text_props(fontweight="bold")
        fit = full["fit"]
        extra = (
            f"Decoded training CoG RMS: {fit['start_rms_dex']:.5f} to "
            f"{fit['fitted_rms_dex']:.5f} dex"
        )
        if representation == "smooth":
            transition = values["gamma_transition_log_m_peak"]
            extra += (
                "\n"
                + r"$\gamma=1.25+0.15/[1+\exp((\log M_{\rm peak}-"
                + f"{transition:.4f}"
                + r")/0.20)]$"
            )
        else:
            extra += "\nGamma is fixed in the decoder, so it is not a fitted predictor."
        table_axis.text(0.0, 0.20, extra, fontsize=8.7, va="top")

    note_axis = figure.add_subplot(grid[3, :])
    note_axis.axis("off")
    note_axis.text(
        0.0,
        0.98,
        "What is—and is not—shown",
        fontsize=11,
        fontweight="bold",
        va="top",
    )
    note_axis.text(
        0.0,
        0.78,
        (
            "These are the single full-sample coefficients to use as a readable "
            "model recipe. The held-out QA was produced by five separate fits, "
            "each trained on four fifths of the galaxies; every one of those exact "
            "coefficient matrices is exported in heldout_fold_coefficients.csv. "
            "The full-sample refit was not used to score its own training galaxies."
        ),
        fontsize=9.2,
        va="top",
        wrap=True,
    )
    save_fig(figure, FIGURE_DIRECTORY / "exp56_stage3b_relation_coefficients")
    plt.close(figure)


def run_demo() -> None:
    started = time.perf_counter()
    result = fit_relation_coefficients(N_DEMO_GALAXY, demo=True)
    elapsed = time.perf_counter() - started
    if elapsed >= DEMO_TIME_LIMIT_SECONDS:
        raise RuntimeError(
            f"90-galaxy coefficient path took {elapsed:.2f} s, exceeding 60 s"
        )
    result["elapsed_seconds"] = elapsed
    write_coefficient_artifacts(result, prefix="demo_")
    print(f"90-galaxy complete coefficient path: {elapsed:.2f} s")


def run_full() -> None:
    started = time.perf_counter()
    result = fit_relation_coefficients(0, demo=False)
    result["elapsed_seconds"] = time.perf_counter() - started
    write_coefficient_artifacts(result)
    print(
        f"full coefficient export: {result['n_galaxy']} galaxies in "
        f"{result['elapsed_seconds']:.2f} s"
    )


def build_figures() -> None:
    coefficient_path = OUTPUT_DIRECTORY / "relation_coefficients.json"
    if not coefficient_path.exists():
        raise RuntimeError("run the full coefficient export before figures")
    result = json.loads(coefficient_path.read_text())
    values, summary = paired_error_values()
    write_paired_error_artifacts(values, summary)
    paired_error_figure(values, summary)
    relation_coefficient_figure(result)


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"demo", "run", "figures"}:
        raise SystemExit("usage: stage3b_relation_diagnostics.py [demo|run|figures]")
    command = sys.argv[1]
    if command == "demo":
        run_demo()
    elif command == "run":
        run_full()
    else:
        build_figures()


if __name__ == "__main__":
    main()
