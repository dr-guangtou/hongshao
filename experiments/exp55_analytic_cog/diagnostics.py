"""exp55 stage 3 — deep identifiability audit of the selected analytic mixture."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from hongshao.plotting import OKABE_ITO, save_fig, set_style  # noqa: E402
from hongshao.provenance import write_manifest  # noqa: E402
from mixed import (  # noqa: E402
    LOWER,
    UPPER,
    fit_one,
    mixed_cog_log,
)
from run import (  # noqa: E402
    FIGDIR,
    OUTDIR,
    RADII,
    load_sample,
    representative_indices,
)

set_style()


def selected_shape() -> tuple[str, float, float]:
    summary = json.loads((OUTDIR / "mixed_grid" / "summary.json").read_text())
    name = summary["best_full_sample_shape_pair"]
    match = re.fullmatch(r"sersic([0-9p]+)_moffat([0-9p]+)", name)
    if match is None:
        raise ValueError(f"cannot parse {name!r}")
    return (
        name,
        float(match.group(1).replace("p", ".")),
        float(match.group(2).replace("p", ".")),
    )


def physical_parameters(parameters: np.ndarray) -> np.ndarray:
    parameters = np.atleast_2d(parameters)
    compact_fraction = expit(parameters[:, 1])
    log_r_compact = parameters[:, 2]
    log_r_extended = log_r_compact + np.log10(1.0 + np.exp(parameters[:, 3]))
    return np.column_stack(
        [parameters[:, 0], compact_fraction, log_r_compact, log_r_extended]
    )


def profiled_scan(
    truth_log: np.ndarray,
    best_parameters: np.ndarray,
    parameter_index: int,
    values: np.ndarray,
    n_sersic: float,
    gamma: float,
) -> np.ndarray:
    free = np.arange(4) != parameter_index
    output = np.empty(len(values))
    previous = best_parameters[free]
    for index, fixed_value in enumerate(values):

        def residual(free_parameters: np.ndarray) -> np.ndarray:
            parameters = np.empty(4)
            parameters[free] = free_parameters
            parameters[parameter_index] = fixed_value
            return mixed_cog_log(parameters, RADII, n_sersic, gamma) - truth_log

        solutions = [
            least_squares(
                residual,
                np.clip(start, LOWER[free] + 1e-8, UPPER[free] - 1e-8),
                bounds=(LOWER[free], UPPER[free]),
                x_scale="jac",
                max_nfev=3000,
            )
            for start in (best_parameters[free], previous)
        ]
        best = min(solutions, key=lambda solution: np.sum(solution.fun**2))
        previous = best.x
        output[index] = np.sqrt(np.mean(best.fun**2))
    return output


def recovery_sample(full_indices: np.ndarray, n_sample: int = 200) -> np.ndarray:
    _, selected_indices = load_sample(n_sample)
    row_of = {int(index): row for row, index in enumerate(full_indices)}
    return np.asarray([row_of[int(index)] for index in selected_indices], int)


def scan_figure(
    truth: np.ndarray,
    parameters: np.ndarray,
    n_sersic: float,
    gamma: float,
) -> dict:
    galaxies = representative_indices(truth)
    fraction_grid = np.linspace(LOWER[1], UPPER[1], 41)
    ratio_grid = np.linspace(LOWER[3], UPPER[3], 41)
    output = {"galaxies": np.asarray(galaxies)}
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.8), sharey="row")
    for column, galaxy in enumerate(galaxies):
        best = parameters[galaxy]
        fraction_rms = profiled_scan(
            truth[galaxy], best, 1, fraction_grid, n_sersic, gamma
        )
        ratio_rms = profiled_scan(truth[galaxy], best, 3, ratio_grid, n_sersic, gamma)
        axes[0, column].plot(expit(fraction_grid), fraction_rms, color=OKABE_ITO[4])
        axes[0, column].axvline(expit(best[1]), color="black", linestyle=":")
        axes[1, column].plot(1.0 + np.exp(ratio_grid), ratio_rms, color=OKABE_ITO[2])
        axes[1, column].axvline(1.0 + np.exp(best[3]), color="black", linestyle=":")
        axes[1, column].set_xscale("log")
        compactness = 10.0 ** (truth[galaxy, 0] - truth[galaxy, -1])
        axes[0, column].set_title(rf"$M_*(<2)/M_{{*,148}}={compactness:.2f}$")
        axes[0, column].set_xlabel("Compact component fraction within 148 kpc")
        axes[1, column].set_xlabel(r"$R_{\rm extended}/R_{\rm compact}$")
        output[f"galaxy_{galaxy}_fraction_rms"] = fraction_rms
        output[f"galaxy_{galaxy}_ratio_rms"] = ratio_rms
    axes[0, 0].set_ylabel("Profiled CoG RMS (dex)")
    axes[1, 0].set_ylabel("Profiled CoG RMS (dex)")
    for panel, axis in zip("ABCDEF", axes.ravel()):
        axis.text(
            -0.14, 1.04, panel, transform=axis.transAxes, fontweight="bold", fontsize=11
        )
    fig.suptitle(
        "exp55 — selected mixture profile scans; all other parameters re-optimized"
    )
    fig.tight_layout()
    save_fig(fig, FIGDIR / "exp55_selected_mixture_scans")
    plt.close(fig)
    output["logit_fraction_grid"] = fraction_grid
    output["log_ratio_minus_one_grid"] = ratio_grid
    return output


def run() -> None:
    name, n_sersic, gamma = selected_shape()
    base = np.load(OUTDIR / "mixed_grid" / "selected.npz")
    truth = base["truth_log"]
    indices = base["indices"]
    parameters = base["best_parameters"]
    sample = recovery_sample(indices)

    exact_recovered = np.empty((len(sample), 4))
    jackknife_recovered = np.empty((len(sample), 4))
    keep_radius = np.unique(np.append(np.arange(0, len(RADII), 2), len(RADII) - 1))
    for output_row, galaxy in enumerate(sample):
        synthetic = mixed_cog_log(parameters[galaxy], RADII, n_sersic, gamma)
        exact_recovered[output_row] = fit_one(synthetic, n_sersic, gamma)["parameters"]
        jackknife_recovered[output_row] = fit_one(
            truth[galaxy, keep_radius], n_sersic, gamma, RADII[keep_radius]
        )["parameters"]

    physical_full = physical_parameters(parameters[sample])
    physical_exact = physical_parameters(exact_recovered)
    physical_jackknife = physical_parameters(jackknife_recovered)
    names = [
        "log_mstar_148",
        "compact_fraction_148",
        "log_r_compact",
        "log_r_extended",
    ]
    exact_error = physical_exact - physical_full
    jackknife_error = physical_jackknife - physical_full
    correlation = spearmanr(physical_parameters(parameters)).statistic

    summary = {
        "selected_family": name,
        "n_sersic": n_sersic,
        "moffat_gamma": gamma,
        "n_recovery_galaxy": len(sample),
        "radial_jackknife_points": int(len(keep_radius)),
        "exact_synthetic_median_absolute_error": {
            parameter: float(value)
            for parameter, value in zip(names, np.median(np.abs(exact_error), axis=0))
        },
        "exact_synthetic_maximum_absolute_error": {
            parameter: float(value)
            for parameter, value in zip(names, np.max(np.abs(exact_error), axis=0))
        },
        "radial_jackknife_median_absolute_change": {
            parameter: float(value)
            for parameter, value in zip(
                names, np.median(np.abs(jackknife_error), axis=0)
            )
        },
        "radial_jackknife_84th_percentile_absolute_change": {
            parameter: float(value)
            for parameter, value in zip(
                names, np.quantile(np.abs(jackknife_error), 0.84, axis=0)
            )
        },
        "radial_jackknife_95th_percentile_absolute_change": {
            parameter: float(value)
            for parameter, value in zip(
                names, np.quantile(np.abs(jackknife_error), 0.95, axis=0)
            )
        },
        "full_sample_spearman_correlation": correlation.tolist(),
    }
    diagnostics_dir = OUTDIR / "mixed_diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    (diagnostics_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    scans = scan_figure(truth, parameters, n_sersic, gamma)
    np.savez(
        diagnostics_dir / "diagnostics.npz",
        sample=sample,
        physical_full=physical_full,
        physical_exact=physical_exact,
        physical_jackknife=physical_jackknife,
        correlation=correlation,
        **scans,
    )

    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.0))
    labels = [
        r"$\log M_{*,148}$",
        r"$f_{\rm compact,148}$",
        r"$\log R_{\rm compact}$",
        r"$\log R_{\rm extended}$",
    ]
    for parameter_index, axis in enumerate(axes.ravel()[:4]):
        axis.scatter(
            physical_full[:, parameter_index],
            physical_jackknife[:, parameter_index],
            s=10,
            alpha=0.5,
            color=OKABE_ITO[2],
        )
        limits = [
            min(
                physical_full[:, parameter_index].min(),
                physical_jackknife[:, parameter_index].min(),
            ),
            max(
                physical_full[:, parameter_index].max(),
                physical_jackknife[:, parameter_index].max(),
            ),
        ]
        axis.plot(limits, limits, ":", color="black")
        median_change = np.median(np.abs(jackknife_error[:, parameter_index]))
        axis.set(
            xlabel=f"24-radius fit {labels[parameter_index]}",
            ylabel=f"12-radius fit {labels[parameter_index]}",
            title=f"Median absolute change = {median_change:.3g}",
        )

    ratio = np.load(OUTDIR / "mixed_grid" / f"{name}.npz")["jacobian_min_ratio"]
    axes[1, 1].hist(-np.log10(np.clip(ratio, 1e-16, None)), bins=35, color=OKABE_ITO[4])
    axes[1, 1].set(
        xlabel=r"$-\log_{10}(s_{\min}/s_{\max})$",
        ylabel="Number of galaxies",
        title="Full-sample local conditioning",
    )
    image = axes[1, 2].imshow(correlation, vmin=-1.0, vmax=1.0, cmap="PuOr_r")
    axes[1, 2].set_xticks(range(4), labels, rotation=35, ha="right")
    axes[1, 2].set_yticks(range(4), labels)
    axes[1, 2].set_title("Spearman parameter correlation")
    fig.colorbar(image, ax=axes[1, 2], label=r"Spearman $\rho$")
    for panel, axis in zip("ABCDEF", axes.ravel()):
        axis.text(
            -0.14, 1.04, panel, transform=axis.transAxes, fontweight="bold", fontsize=11
        )
    fig.suptitle("exp55 — selected mixture parameter stability")
    fig.tight_layout()
    save_fig(fig, FIGDIR / "exp55_selected_mixture_identifiability")
    plt.close(fig)
    write_manifest(
        diagnostics_dir,
        {
            "experiment": "exp55_analytic_cog_mixed_diagnostics",
            "selected_family": name,
            "n_recovery_galaxy": len(sample),
            "radial_jackknife_points": len(keep_radius),
        },
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()
