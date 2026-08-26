"""Standard and profile-specific QA for the selected exp55 analytic CoG.

The standard battery is the repository's ``hongshao.qa.evaluate`` layout.
Two additions answer questions specific to an analytic decomposition:

1. a population radial summary in cumulative and differential form;
2. the compact and extended components shown separately.

Run:
    PYTHONPATH=. uv run python experiments/exp55_analytic_cog/qa_figures.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from hongshao import qa  # noqa: E402
from hongshao.plotting import OKABE_ITO, save_fig, set_style  # noqa: E402
from hongshao.profile_emulator import density_from_cog  # noqa: E402
from hongshao.provenance import write_manifest  # noqa: E402
from mixed import mixed_component_cogs  # noqa: E402
from run import FIGDIR, OUTDIR, RADII, representative_indices, source_table  # noqa: E402

set_style()

QA_FIGDIR = FIGDIR / "standard_qa"
QA_OUTDIR = OUTDIR / "standard_qa"
MODEL_NAME = "exp55_analytic_cog"
SELECTED_N_SERSIC = 1.0
SELECTED_MOFFAT_GAMMA = 1.25


def load_inputs() -> dict[str, np.ndarray]:
    selected = np.load(OUTDIR / "mixed_grid" / "selected.npz")
    indices = np.asarray(selected["indices"], int)
    table = Table.read(source_table())
    row_by_index = {int(index): row for row, index in enumerate(table["index"])}
    rows = np.asarray([row_by_index[int(index)] for index in indices], int)
    values = {
        "indices": indices,
        "truth_log": np.asarray(selected["truth_log"], float),
        "prediction_log": np.asarray(selected["cv_prediction"], float),
        "best_prediction_log": np.asarray(selected["best_prediction"], float),
        "best_parameters": np.asarray(selected["best_parameters"], float),
        "log_halo_mass": np.asarray(table["logmh_z0p4"][rows], float),
    }
    n_max = int(os.environ.get("EXP55_QA_NMAX", 0))
    if 0 < n_max < len(indices):
        selected_rows = np.unique(np.linspace(0, len(indices) - 1, n_max, dtype=int))
        values = {name: value[selected_rows] for name, value in values.items()}
    return values


def standard_summary(
    result: dict, model_log: np.ndarray, truth_log: np.ndarray
) -> dict:
    summary = {"mass_quantities": {}, "observational_planes": {}, "sizes": {}}
    for key in result["keys"]:
        truth = result["truth"][key][:, 0]
        model = result["model"][key][:, 0]
        good = np.isfinite(truth) & np.isfinite(model) & (truth > 0.0) & (model > 0.0)
        if not np.any(good):
            summary["mass_quantities"][key] = None
            continue
        ratio = model[good] / truth[good]
        summary["mass_quantities"][key] = {
            "n": int(np.count_nonzero(good)),
            "median_fractional_bias": float(np.median(ratio - 1.0)),
            "scatter_log10_model_over_truth_dex": float(np.std(np.log10(ratio))),
        }
    for (x_name, y_name), values in result["planes"].items():
        truth_stats, model_stats = values[0]
        summary["observational_planes"][f"{x_name}__{y_name}"] = {
            "truth": truth_stats,
            "model": model_stats,
        }
    for (radius_name, epoch), values in result["sizes"].items():
        if epoch != 0:
            continue
        truth_stats, model_stats = values
        summary["sizes"][radius_name] = {
            "truth": truth_stats,
            "model": model_stats,
        }
    fractional = np.abs((10.0**model_log - 10.0**truth_log) / 10.0**truth_log)
    summary["profile"] = {
        "median_maximum_fractional_error_all_radii": float(
            np.median(np.max(fractional, axis=1))
        ),
        "median_maximum_fractional_error_r_gt_5_kpc": float(
            np.median(np.max(fractional[:, RADII > 5.0], axis=1))
        ),
    }
    return summary


def density_by_halo_mass_figure(
    model_log: np.ndarray,
    truth_log: np.ndarray,
    log_halo_mass: np.ndarray,
    figdir: Path = QA_FIGDIR,
    model_name: str = MODEL_NAME,
    model_label: str = "analytic CoG",
    experiment_label: str = "exp55",
) -> None:
    """Exp54-style surface-density QA split by a model input."""
    edges = np.quantile(log_halo_mass, [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
    model_density, middle_radius = density_from_cog(model_log, RADII)
    truth_density, _ = density_from_cog(truth_log, RADII)
    x = middle_radius**0.25
    fig, axes = plt.subplots(
        2, 3, figsize=(15.5, 7.5), sharex=True, height_ratios=[2, 1]
    )
    maximum_residual = 0.0
    guide = None
    for column in range(3):
        selected = (log_halo_mass >= edges[column]) & (
            log_halo_mass <= edges[column + 1] + 1e-9
        )
        median_truth = np.median(truth_density[selected], axis=0)
        median_model = np.median(model_density[selected], axis=0)
        residual = np.median(model_density[selected] - truth_density[selected], axis=0)
        maximum_residual = max(maximum_residual, float(np.max(np.abs(residual))))
        axes[0, column].plot(x, median_truth, color="black", linewidth=1.9, label="TNG")
        axes[0, column].plot(
            x,
            median_model,
            "--",
            color=OKABE_ITO[2],
            linewidth=1.6,
            label=model_label,
        )
        axes[1, column].plot(x, residual, color=OKABE_ITO[2], linewidth=1.6)
        axes[1, column].axhline(0.0, color="0.55", linewidth=0.8)
        axes[0, column].set_title(
            rf"$\log M_{{\rm halo}}={edges[column]:.2f}$--"
            rf"${edges[column + 1]:.2f}$  ($N={np.count_nonzero(selected)}$)"
        )
        if guide is None:
            coefficients = np.polyfit(x, median_truth, 1)
            guide = np.polyval(coefficients, x)
        axes[0, column].plot(
            x,
            guide,
            ":",
            color="0.45",
            linewidth=1.1,
            label=r"$R^{1/4}$ guide" if column == 0 else None,
        )
        axes[1, column].set_xticks(
            [radius**0.25 for radius in (2, 5, 10, 20, 50, 100, 150)]
        )
        axes[1, column].set_xticklabels(["2", "5", "10", "20", "50", "100", "150"])
        axes[1, column].set_xlabel(r"Radius (kpc), on an $R^{1/4}$ axis")
    limit = float(np.clip(1.25 * maximum_residual, 0.15, 0.9))
    for column in range(3):
        axes[1, column].set_ylim(-limit, limit)
    axes[0, 0].set_ylabel(r"Median $\log_{10}\Sigma_*$ ($M_\odot\,\mathrm{kpc}^{-2}$)")
    axes[1, 0].set_ylabel(
        r"Median $\log_{10}(\Sigma_{\rm model}/\Sigma_{\rm TNG})$ (dex)"
    )
    axes[0, 0].legend(fontsize=8, loc="upper right")
    for panel, axis in zip("ABCDEF", axes.ravel()):
        axis.text(
            -0.12,
            1.04,
            panel,
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=11,
        )
    fig.suptitle(
        f"{experiment_label} — surface-density QA by halo-mass tercile; "
        f"TNG solid, {model_label} dashed"
    )
    fig.text(
        0.5,
        0.004,
        "Surface density is the finite-annulus derivative of the CoG. Its residual is in dex: "
        "0.10 dex is a factor 1.26 and 0.20 dex is a factor 1.58. The cumulative CoG can look "
        "excellent while the derivative exposes a local outer-slope mismatch.",
        ha="center",
        va="bottom",
        fontsize=7.2,
        color="0.35",
        style="italic",
        wrap=True,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_fig(fig, figdir / f"qa_dens_{model_name}")
    plt.close(fig)


def single_epoch_planes_figure(
    result: dict,
    figdir: Path = QA_FIGDIR,
    model_name: str = MODEL_NAME,
    model_label: str = "analytic CoG",
    experiment_label: str = "exp55",
) -> None:
    """Replace the multi-epoch QA layout with a readable one-row z=0.4 view."""
    axis_labels = {
        "kpc:M(<30)": r"$\log_{10}M_*(<30\,\mathrm{kpc})$",
        "kpc:M(30-50)": r"$\log_{10}M_*(30\!-\!50\,\mathrm{kpc})$",
        "kpc:M(50-100)": r"$\log_{10}M_*(50\!-\!100\,\mathrm{kpc})$",
        "Re:M(<2Re)": r"$\log_{10}M_*(<2R_{\rm e})$",
        "Re:M(2-4Re)": r"$\log_{10}M_*(2\!-\!4R_{\rm e})$",
    }
    fig, axes = plt.subplots(1, len(qa.PLANES), figsize=(12.0, 3.8))
    for axis, (x_name, y_name) in zip(axes, qa.PLANES):
        truth_x = np.log10(np.clip(result["truth"][x_name][:, 0], 1.0, None))
        truth_y = np.log10(np.clip(result["truth"][y_name][:, 0], 1.0, None))
        model_x = np.log10(np.clip(result["model"][x_name][:, 0], 1.0, None))
        model_y = np.log10(np.clip(result["model"][y_name][:, 0], 1.0, None))
        axis.scatter(
            truth_x,
            truth_y,
            s=8,
            color=OKABE_ITO[4],
            alpha=0.45,
            edgecolors="none",
            label="TNG",
        )
        axis.scatter(
            model_x,
            model_y,
            s=10,
            facecolors="none",
            edgecolors="0.2",
            linewidth=0.45,
            alpha=0.45,
            label=model_label,
        )
        truth_stats, model_stats = result["planes"][(x_name, y_name)][0]
        axis.set_title(
            f"slope {truth_stats['slope']:.2f} $\\rightarrow$ "
            f"{model_stats['slope']:.2f}; scatter "
            f"{truth_stats['scatter']:.3f} $\\rightarrow$ "
            f"{model_stats['scatter']:.3f}",
            fontsize=9,
        )
        axis.set_xlabel(axis_labels[x_name])
        axis.set_ylabel(axis_labels[y_name])
    axes[0].legend(fontsize=8, markerscale=1.7)
    for panel, axis in zip("ABC", axes):
        axis.text(
            -0.14,
            1.05,
            panel,
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=11,
        )
    fig.suptitle(
        f"{experiment_label} — standard observational planes at z=0.4; "
        f"TNG filled, {model_label} open"
    )
    fig.tight_layout()
    save_fig(fig, figdir / f"qa_planes_{model_name}")
    plt.close(fig)


def enclosed_radius_array(cogs: np.ndarray, fraction: float) -> np.ndarray:
    return np.asarray(
        [qa.enclosed_radius(profile, RADII, fraction) for profile in cogs], float
    )


def single_epoch_size_figure(
    model_log: np.ndarray,
    truth_log: np.ndarray,
    result: dict,
    figdir: Path = QA_FIGDIR,
    model_name: str = MODEL_NAME,
    model_label: str = "analytic CoG",
    experiment_label: str = "exp55",
) -> None:
    """Replace the generic one-column multi-epoch size layout."""
    model = 10.0**model_log
    truth = 10.0**truth_log
    log_mass_truth = truth_log[:, -1]
    log_mass_model = model_log[:, -1]
    truth_radii = {
        name: enclosed_radius_array(truth, fraction)
        for name, fraction in zip(("R50", "R80", "R90"), (0.5, 0.8, 0.9))
    }
    model_radii = {
        name: enclosed_radius_array(model, fraction)
        for name, fraction in zip(("R50", "R80", "R90"), (0.5, 0.8, 0.9))
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].scatter(
        log_mass_truth,
        np.log10(truth_radii["R50"]),
        s=7,
        color="0.65",
        alpha=0.4,
        edgecolors="none",
        label="TNG",
    )
    axes[0].scatter(
        log_mass_model,
        np.log10(model_radii["R50"]),
        s=7,
        color=OKABE_ITO[4],
        alpha=0.35,
        edgecolors="none",
        label=model_label,
    )
    truth_stats, model_stats = result["sizes"][("R50", 0)]
    axes[0].set(
        xlabel=r"$\log_{10}M_{*,148}$",
        ylabel=r"$\log_{10}R_{50}$ (kpc)",
        title=(
            f"R50 slope {truth_stats['slope']:.2f} $\\rightarrow$ "
            f"{model_stats['slope']:.2f}; median offset "
            f"{model_stats['median_dlogR']:+.3f} dex"
        ),
    )
    axes[0].legend(fontsize=8)
    edges = np.quantile(log_mass_truth, np.linspace(0.02, 0.98, 8))
    for name, color, style, marker in (
        ("R50", OKABE_ITO[4], "-", "o"),
        ("R80", OKABE_ITO[0], "--", "s"),
        ("R90", OKABE_ITO[2], ":", "^"),
    ):
        centers = []
        offsets = []
        for lower, upper in zip(edges[:-1], edges[1:]):
            selected = (log_mass_truth >= lower) & (log_mass_truth < upper)
            centers.append(0.5 * (lower + upper))
            offsets.append(
                np.median(
                    np.log10(model_radii[name][selected])
                    - np.log10(truth_radii[name][selected])
                )
            )
        axes[1].plot(
            centers,
            offsets,
            style,
            marker=marker,
            color=color,
            linewidth=1.7,
            markersize=4,
            label=name,
        )
    axes[1].axhline(0.0, color="0.55", linewidth=0.8)
    axes[1].set(
        xlabel=r"TNG $\log_{10}M_{*,148}$",
        ylabel=r"Median $\Delta\log_{10}R$ (model - TNG)",
        title="Size offsets across the stellar-mass range",
    )
    axes[1].legend(fontsize=8, ncol=3)
    for panel, axis in zip("AB", axes):
        axis.text(
            -0.12,
            1.05,
            panel,
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=11,
        )
    fig.suptitle(f"{experiment_label} — standard mass–size QA at z=0.4")
    fig.tight_layout()
    save_fig(fig, figdir / f"qa_size_{model_name}")
    plt.close(fig)


def radial_summary_figure(model_log: np.ndarray, truth_log: np.ndarray) -> dict:
    """Show why small cumulative errors can become larger density errors."""
    model_density, middle_radius = density_from_cog(model_log, RADII)
    truth_density, _ = density_from_cog(truth_log, RADII)
    normalized_model = model_log - model_log[:, -1:]
    normalized_truth = truth_log - truth_log[:, -1:]
    cumulative_error = model_log - truth_log
    density_error = model_density - truth_density

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.3))
    axes[0, 0].plot(
        RADII, np.median(normalized_truth, axis=0), color="black", label="TNG"
    )
    axes[0, 0].plot(
        RADII,
        np.median(normalized_model, axis=0),
        "--",
        color=OKABE_ITO[2],
        label="analytic CoG",
    )
    axes[0, 0].set(
        xscale="log",
        xlabel="Radius (kpc)",
        ylabel=r"Median $\log_{10}[M_*(<R)/M_{*,148}]$",
        title="Cumulative profile",
    )
    axes[0, 0].legend(fontsize=8)

    cumulative_median = np.median(cumulative_error, axis=0)
    cumulative_low, cumulative_high = np.quantile(
        cumulative_error, [0.16, 0.84], axis=0
    )
    axes[1, 0].plot(RADII, cumulative_median, color=OKABE_ITO[2])
    axes[1, 0].fill_between(
        RADII, cumulative_low, cumulative_high, color=OKABE_ITO[2], alpha=0.18
    )
    axes[1, 0].axhline(0.0, color="0.5", linewidth=0.8)
    axes[1, 0].set(
        xscale="log",
        xlabel="Radius (kpc)",
        ylabel="Model - TNG (dex)",
        title="Cumulative residual; band is 16th--84th percentile",
    )

    axes[0, 1].plot(
        middle_radius**0.25,
        np.median(truth_density, axis=0),
        color="black",
        label="TNG",
    )
    axes[0, 1].plot(
        middle_radius**0.25,
        np.median(model_density, axis=0),
        "--",
        color=OKABE_ITO[2],
        label="analytic CoG",
    )
    axes[0, 1].set(
        xlabel=r"Radius (kpc), on an $R^{1/4}$ axis",
        ylabel=r"Median $\log_{10}\Sigma_*$ ($M_\odot\,\mathrm{kpc}^{-2}$)",
        title="Finite-annulus surface density",
    )
    density_median = np.median(density_error, axis=0)
    density_low, density_high = np.quantile(density_error, [0.16, 0.84], axis=0)
    axes[1, 1].plot(middle_radius**0.25, density_median, color=OKABE_ITO[2])
    axes[1, 1].fill_between(
        middle_radius**0.25,
        density_low,
        density_high,
        color=OKABE_ITO[2],
        alpha=0.18,
    )
    axes[1, 1].axhline(0.0, color="0.5", linewidth=0.8)
    axes[1, 1].set(
        xlabel=r"Radius (kpc), on an $R^{1/4}$ axis",
        ylabel=r"$\log_{10}(\Sigma_{\rm model}/\Sigma_{\rm TNG})$ (dex)",
        title="Density residual; band is 16th--84th percentile",
    )
    for axis in (axes[0, 1], axes[1, 1]):
        axis.set_xticks([radius**0.25 for radius in (2, 5, 10, 20, 50, 100, 150)])
        axis.set_xticklabels(["2", "5", "10", "20", "50", "100", "150"])
    for panel, axis in zip("ABCD", axes.ravel()):
        axis.text(
            -0.12,
            1.04,
            panel,
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=11,
        )
    fig.suptitle("exp55 — cumulative agreement does not imply differential agreement")
    fig.text(
        0.5,
        0.004,
        r"Each density point uses $[M(<R_{out})-M(<R_{in})]/[\pi(R_{out}^2-R_{in}^2)]$. "
        "In the outskirts this subtracts two similar enclosed masses, amplifying small radial "
        "changes in the cumulative residual. Both TNG and model pass through the same operator.",
        ha="center",
        va="bottom",
        fontsize=7.2,
        color="0.35",
        style="italic",
        wrap=True,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_fig(fig, QA_FIGDIR / "exp55_qa_radial_summary")
    plt.close(fig)

    truth_linear = 10.0**truth_log
    annular_fraction = np.diff(truth_linear, axis=1) / truth_linear[:, 1:]
    rows = []
    for index in range(len(middle_radius)):
        rows.append(
            {
                "annulus_kpc": [float(RADII[index]), float(RADII[index + 1])],
                "midpoint_kpc": float(middle_radius[index]),
                "median_truth_annular_mass_over_enclosed_outer_mass": float(
                    np.median(annular_fraction[:, index])
                ),
                "median_absolute_cumulative_error_at_outer_edge_dex": float(
                    np.median(np.abs(cumulative_error[:, index + 1]))
                ),
                "median_absolute_density_error_dex": float(
                    np.median(np.abs(density_error[:, index]))
                ),
                "median_signed_density_error_dex": float(
                    np.median(density_error[:, index])
                ),
            }
        )
    return {"radial_rows": rows}


def component_figure(
    truth_log: np.ndarray, parameters: np.ndarray, indices: np.ndarray
) -> dict:
    """Show component CoGs and densities for diffuse, typical, and compact cases."""
    galaxies = representative_indices(truth_log)
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.1))
    records = []
    for column, galaxy in enumerate(galaxies):
        compact, extended = mixed_component_cogs(
            parameters[galaxy],
            RADII,
            SELECTED_N_SERSIC,
            SELECTED_MOFFAT_GAMMA,
        )
        total = compact + extended
        truth = 10.0 ** truth_log[galaxy]
        aperture_mass = total[-1]
        for label, values, color, style, width in (
            ("TNG total", truth / truth[-1], "black", "-", 1.9),
            ("model total", total / aperture_mass, OKABE_ITO[2], "--", 1.8),
            ("compact contribution", compact / aperture_mass, OKABE_ITO[0], ":", 2.0),
            (
                "extended contribution",
                extended / aperture_mass,
                OKABE_ITO[4],
                "-.",
                1.8,
            ),
        ):
            axes[0, column].plot(
                RADII, values, style, color=color, linewidth=width, label=label
            )
        component_logs = [
            np.log10(np.clip(values, 1.0, None))[None, :]
            for values in (truth, total, compact, extended)
        ]
        densities = [density_from_cog(values, RADII)[0][0] for values in component_logs]
        middle_radius = density_from_cog(component_logs[0], RADII)[1]
        for label, density, color, style, width in zip(
            (
                "TNG total",
                "model total",
                "compact contribution",
                "extended contribution",
            ),
            densities,
            ("black", OKABE_ITO[2], OKABE_ITO[0], OKABE_ITO[4]),
            ("-", "--", ":", "-."),
            (1.9, 1.8, 2.0, 1.8),
        ):
            axes[1, column].plot(
                middle_radius,
                density,
                style,
                color=color,
                linewidth=width,
                label=label,
            )
        central_fraction = truth[0] / truth[-1]
        compact_fraction = compact[-1] / aperture_mass
        crossing = np.nan
        difference = compact - extended
        changed = np.where(np.sign(difference[:-1]) != np.sign(difference[1:]))[0]
        if len(changed):
            crossing = float(
                np.interp(
                    0.0,
                    difference[changed[0] : changed[0] + 2],
                    RADII[changed[0] : changed[0] + 2],
                )
            )
        axes[0, column].set(
            xscale="log",
            xlabel="Radius (kpc)",
            ylabel=r"Contribution to $M_*(<R)/M_{*,148}$" if column == 0 else None,
            title=(
                rf"TNG $M_*(<2)/M_{{*,148}}={central_fraction:.2f}$"
                "\n"
                rf"fit $f_{{\rm compact,148}}={compact_fraction:.2f}$"
            ),
        )
        axes[1, column].set(
            xscale="log",
            yscale="linear",
            ylim=(4.5, 9.5),
            xlabel="Annulus midpoint (kpc)",
            ylabel=r"$\log_{10}\Sigma_*$ ($M_\odot\,\mathrm{kpc}^{-2}$)"
            if column == 0
            else None,
        )
        records.append(
            {
                "catalog_index": int(indices[galaxy]),
                "central_fraction_truth": float(central_fraction),
                "compact_fraction_148": float(compact_fraction),
                "compact_r50_kpc": float(10.0 ** parameters[galaxy, 2]),
                "extended_r50_kpc": float(
                    10.0 ** parameters[galaxy, 2]
                    * (1.0 + np.exp(parameters[galaxy, 3]))
                ),
                "cumulative_component_crossing_kpc": crossing,
            }
        )
    axes[0, 0].legend(fontsize=7.5, loc="lower right")
    axes[1, 0].legend(fontsize=7.5, loc="upper right")
    for panel, axis in zip("ABCDEF", axes.ravel()):
        axis.text(
            -0.12,
            1.04,
            panel,
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=11,
        )
    fig.suptitle(
        "exp55 — compact and extended contributions for representative galaxies"
    )
    fig.text(
        0.5,
        0.004,
        "The two colored component curves add exactly to the green total. Fractions are defined "
        "within 148.2 kpc. 'Compact' and 'extended' describe the fitted radial terms only and "
        "must not be identified with in-situ and ex-situ stars without an independent test.",
        ha="center",
        va="bottom",
        fontsize=7.2,
        color="0.35",
        style="italic",
        wrap=True,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_fig(fig, QA_FIGDIR / "exp55_component_decomposition")
    plt.close(fig)
    return {"representative_galaxies": records}


def run() -> None:
    values = load_inputs()
    truth_log = values["truth_log"]
    prediction_log = values["prediction_log"]
    truth = 10.0 ** truth_log[:, None, :]
    prediction = 10.0 ** prediction_log[:, None, :]
    QA_FIGDIR.mkdir(parents=True, exist_ok=True)
    QA_OUTDIR.mkdir(parents=True, exist_ok=True)

    result = qa.evaluate(
        prediction,
        truth,
        RADII,
        [0.4],
        name=MODEL_NAME,
        figdir=QA_FIGDIR,
        bin_by=values["log_halo_mass"],
        bin_label=r"$\log_{10}M_{\rm halo}(z=0.4)/M_\odot$",
    )
    single_epoch_planes_figure(result)
    single_epoch_size_figure(prediction_log, truth_log, result)
    density_by_halo_mass_figure(prediction_log, truth_log, values["log_halo_mass"])
    radial = radial_summary_figure(prediction_log, truth_log)
    components = component_figure(
        truth_log, values["best_parameters"], values["indices"]
    )
    summary = {
        "n_galaxy": len(truth_log),
        "model_prediction": "fold-selected global component shapes",
        "standard_qa": standard_summary(result, prediction_log, truth_log),
        "cumulative_to_density": radial,
        "component_decomposition": components,
    }
    (QA_OUTDIR / "summary.json").write_text(json.dumps(summary, indent=2))
    write_manifest(
        QA_OUTDIR,
        {
            "experiment": "exp55_analytic_cog_standard_qa",
            "n_galaxy": len(truth_log),
            "model_name": MODEL_NAME,
            "selected_n_sersic": SELECTED_N_SERSIC,
            "selected_moffat_gamma": SELECTED_MOFFAT_GAMMA,
        },
    )
    print(f"Wrote standard QA to {QA_FIGDIR}")
    print(f"Wrote QA summary to {QA_OUTDIR / 'summary.json'}")


if __name__ == "__main__":
    run()
