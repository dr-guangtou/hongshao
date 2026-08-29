# %%
"""Exp62 final visual record: standard QA and scientific synthesis.

Commands:

    uv run python experiments/exp62_cog_fit_atlas/visual_record.py demo
    uv run python experiments/exp62_cog_fit_atlas/visual_record.py full

This driver only reads completed Exp62 artifacts. It does not refit profiles or
alter any scientific selection.
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
from scipy.ndimage import gaussian_filter
from scipy.special import expit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from families import _sersic_moffat_indices  # noqa: E402
from hongshao import qa  # noqa: E402
from hongshao.plotting import OKABE_ITO, save_fig, set_style  # noqa: E402
from hongshao.provenance import write_manifest  # noqa: E402
from run import FIGDIR, OUTDIR, RADII, REDSHIFTS, profile_metrics  # noqa: E402
from stage3 import DECODED_NAMES, _log_annulus_mass, _mass_at_radius  # noqa: E402
from stage4 import (  # noqa: E402
    PRIMARY_RAW,
    _decode_predictions,
    load_data as load_halo_data,
)

set_style()

N_DEMO = 90
REPRESENTATIONS = {
    "gompertz_log": "Log-radius Gompertz (3 parameters)",
    "sersic_moffat_fixed": "Constrained Sersic--Moffat (4 parameters)",
    "double_sersic_free": "Unrestricted double Sersic (6 parameters)",
}
PRIMARY_REPRESENTATION = "sersic_moffat_fixed"
STAGE3_KEYS = {name: f"{name}_cog_log_decoded" for name in REPRESENTATIONS}


def _cube(
    atlas: np.lib.npyio.NpzFile, key: str, allow_nonfinite: bool = False
) -> np.ndarray:
    rows = np.asarray(atlas["rows"], int)
    epochs = np.asarray(atlas["epochs"], int)
    values = np.asarray(atlas[key])
    unique_rows, compact_rows = np.unique(rows, return_inverse=True)
    output = np.full((len(unique_rows), len(REDSHIFTS), *values.shape[1:]), np.nan)
    output[compact_rows, epochs] = values
    if not allow_nonfinite and not np.all(np.isfinite(output)):
        raise ValueError(f"{key} does not cover the complete galaxy--epoch cube")
    return output


def _stratified_rows(values: np.ndarray, n_row: int = N_DEMO) -> np.ndarray:
    order = np.argsort(values)
    positions = np.linspace(0, len(order) - 1, n_row).round().astype(int)
    return order[positions]


def load_atlas(mode: str) -> dict[str, np.ndarray]:
    atlas = np.load(OUTDIR / "full" / "atlas" / "predictions.npz")
    galaxy_ids = np.unique(np.asarray(atlas["rows"], int))
    truth_log = _cube(atlas, "truth_log")
    halo_mass = _cube(atlas, "halo_mass", allow_nonfinite=True)
    rows = np.flatnonzero(np.all(np.isfinite(halo_mass), axis=1))
    if mode == "demo":
        rows = rows[_stratified_rows(halo_mass[rows, 0])]
    return {
        "galaxy_ids": galaxy_ids[rows],
        "truth_log": truth_log[rows],
        "halo_mass": halo_mass[rows],
        **{name: _cube(atlas, f"prediction_{name}")[rows] for name in REPRESENTATIONS},
    }


def performance_summary(atlas: dict[str, np.ndarray]) -> dict:
    output = {}
    for name, label in REPRESENTATIONS.items():
        metrics = profile_metrics(
            atlas[name].reshape(-1, len(RADII)),
            atlas["truth_log"].reshape(-1, len(RADII)),
        )
        output[name] = {
            "label": label,
            "median_full_cog_rms_dex": float(np.median(metrics["cog_rms_full_dex"])),
            "median_density_rms_dex": float(np.median(metrics["density_rms_dex"])),
            "median_absolute_r50_error_dex": float(
                np.median(np.abs(metrics["size_error_dex"][:, 1]))
            ),
        }
    return output


def representation_qa_comparison(mode: str, atlas: dict[str, np.ndarray]) -> None:
    """Put the decisive CoG, density, size, and mass-plane checks side by side."""
    epoch_colors = (
        OKABE_ITO[0],
        OKABE_ITO[1],
        OKABE_ITO[2],
        OKABE_ITO[5],
        OKABE_ITO[6],
    )
    figure, axes = plt.subplots(
        len(REPRESENTATIONS), 4, figsize=(16.5, 10.5), squeeze=False
    )
    truth_log = atlas["truth_log"]
    truth_linear = 10.0**truth_log
    truth_density, density_radius = qa.density_from_cog(
        truth_log.reshape(-1, len(RADII)), RADII
    )
    truth_density = truth_density.reshape(
        len(truth_log), len(REDSHIFTS), len(density_radius)
    )
    truth_sizes = {
        fraction: qa._safe_rhalf(truth_linear, RADII, fraction)
        for fraction in (0.5, 0.8, 0.9)
    }
    for row, (name, label) in enumerate(REPRESENTATIONS.items()):
        prediction_log = atlas[name]
        prediction_linear = 10.0**prediction_log
        prediction_density, _ = qa.density_from_cog(
            prediction_log.reshape(-1, len(RADII)), RADII
        )
        prediction_density = prediction_density.reshape(truth_density.shape)
        for epoch, (redshift, color) in enumerate(zip(REDSHIFTS, epoch_colors)):
            pinned = (
                prediction_linear[:, epoch]
                * truth_linear[:, epoch, -1:]
                / prediction_linear[:, epoch, -1:]
            )
            residual = 100.0 * np.median(
                (pinned - truth_linear[:, epoch]) / truth_linear[:, epoch], axis=0
            )
            axes[row, 0].plot(RADII, residual, color=color, label=rf"$z={redshift:g}$")
            density_residual = np.median(
                prediction_density[:, epoch] - truth_density[:, epoch], axis=0
            )
            axes[row, 1].plot(density_radius, density_residual, color=color)
        for fraction, linestyle, marker in (
            (0.5, "-", "o"),
            (0.8, "--", "s"),
            (0.9, ":", "^"),
        ):
            prediction_size = qa._safe_rhalf(prediction_linear, RADII, fraction)
            error = np.median(
                np.abs(np.log10(prediction_size) - np.log10(truth_sizes[fraction])),
                axis=0,
            )
            axes[row, 2].plot(
                REDSHIFTS,
                error,
                linestyle=linestyle,
                marker=marker,
                color=OKABE_ITO[2],
                label=rf"$R_{{{int(100 * fraction)}}}$",
            )
        truth_inner = _mass_at_radius(truth_log[:, 0], 30.0)
        truth_outer = _log_annulus_mass(truth_log[:, 0], 50.0, 100.0)
        model_inner = _mass_at_radius(prediction_log[:, 0], 30.0)
        model_outer = _log_annulus_mass(prediction_log[:, 0], 50.0, 100.0)
        axes[row, 3].scatter(
            truth_inner, truth_outer, s=4, color="0.55", alpha=0.20, label="TNG"
        )
        axes[row, 3].scatter(
            model_inner,
            model_outer,
            s=4,
            color=OKABE_ITO[2],
            alpha=0.20,
            label="analytic fit",
        )
        axes[row, 0].axhline(0.0, color="0.6", linewidth=0.8)
        axes[row, 1].axhline(0.0, color="0.6", linewidth=0.8)
        axes[row, 0].set_xscale("log")
        axes[row, 1].set_xscale("log")
        axes[row, 2].invert_xaxis()
        axes[row, 0].set_ylabel(f"{label}\nmedian residual " + r"(\%)")
        axes[row, 1].set_ylabel("median density residual (dex)")
        axes[row, 2].set_ylabel("median absolute size error (dex)")
        axes[row, 3].set_ylabel(r"$\log M_*(50\!-!100\,\mathrm{kpc})$")
    for column, title in enumerate(
        (
            "Amplitude-pinned CoG shape",
            "Differenced surface density",
            "Enclosed-mass radii",
            r"$z=0.4$ inner--outer mass plane",
        )
    ):
        axes[0, column].set_title(title)
    axes[0, 0].legend(frameon=False, fontsize=7)
    axes[0, 2].legend(frameon=False, fontsize=7)
    axes[0, 3].legend(frameon=False, fontsize=7)
    for row in range(len(REPRESENTATIONS)):
        axes[row, 0].set_xlabel("Semi-major axis (kpc)")
        axes[row, 1].set_xlabel("Semi-major axis (kpc)")
        axes[row, 2].set_xlabel("Redshift")
        axes[row, 3].set_xlabel(r"$\log M_*(<30\,\mathrm{kpc})$")
    figure.suptitle(
        "Exp62 common representation QA: the same galaxies, epochs, and diagnostics"
    )
    figure.tight_layout()
    save_fig(
        figure,
        FIGDIR / "visual_record" / mode / "exp62_representation_qa_comparison",
    )
    plt.close(figure)


def average_cog_bin_figure(
    mode: str, atlas: dict[str, np.ndarray], bin_kind: str
) -> None:
    """Compare median amplitude-pinned CoG residuals in population bins."""
    if bin_kind not in ("halo", "stellar"):
        raise ValueError("bin_kind must be halo or stellar")
    colors = (OKABE_ITO[0], OKABE_ITO[2], OKABE_ITO[5])
    figure, axes = plt.subplots(
        len(REDSHIFTS), 3, figsize=(13.8, 15.5), sharex=True, sharey=True
    )
    truth_linear = 10.0 ** atlas["truth_log"]
    for epoch, redshift in enumerate(REDSHIFTS):
        bin_values = (
            atlas["halo_mass"][:, epoch]
            if bin_kind == "halo"
            else atlas["truth_log"][:, epoch, -1]
        )
        edges = np.quantile(bin_values, [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
        for mass_bin in range(3):
            axis = axes[epoch, mass_bin]
            selected = (bin_values >= edges[mass_bin]) & (
                bin_values <= edges[mass_bin + 1]
            )
            for (name, label), color in zip(REPRESENTATIONS.items(), colors):
                prediction = 10.0 ** atlas[name][selected, epoch]
                truth = truth_linear[selected, epoch]
                pinned = prediction * truth[:, -1:] / prediction[:, -1:]
                residual = 100.0 * np.median((pinned - truth) / truth, axis=0)
                axis.plot(RADII, residual, color=color, linewidth=1.5, label=label)
            axis.axhline(0.0, color="0.55", linewidth=0.8)
            axis.set_xscale("log")
            axis.set_title(
                rf"$z={redshift:g}$; ${edges[mass_bin]:.2f}$--${edges[mass_bin + 1]:.2f}$"
            )
            if mass_bin == 0:
                axis.set_ylabel(r"Median shape residual (\%)")
            if epoch == len(REDSHIFTS) - 1:
                axis.set_xlabel("Semi-major axis (kpc)")
    axes[0, 0].legend(frameon=False, fontsize=6.8)
    noun = "current halo mass" if bin_kind == "halo" else "current stellar mass"
    figure.suptitle(
        f"Exp62 average CoG recovery in {noun} terciles; models are pinned to the measured 148-kpc mass"
    )
    figure.tight_layout()
    save_fig(
        figure,
        FIGDIR / "visual_record" / mode / f"exp62_average_cog_{bin_kind}_mass_bins",
    )
    plt.close(figure)


def _mass_quantities(cogs_log: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "m30": _mass_at_radius(cogs_log, 30.0),
        "m30_50": _log_annulus_mass(cogs_log, 30.0, 50.0),
        "m50_100": _log_annulus_mass(cogs_log, 50.0, 100.0),
        "m100_148": _log_annulus_mass(cogs_log, 100.0, RADII[-1]),
    }


def _binned_statistics(x: np.ndarray, y: np.ndarray, n_bin: int = 7):
    edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, n_bin + 1)))
    center, median, lower, upper = [], [], [], []
    for low, high in zip(edges[:-1], edges[1:]):
        selected = (x >= low) & (x <= high)
        center.append(np.median(x[selected]))
        median.append(np.median(y[selected]))
        lower.append(np.quantile(y[selected], 0.16))
        upper.append(np.quantile(y[selected], 0.84))
    return tuple(np.asarray(value) for value in (center, median, lower, upper))


def stellar_mass_planes_figure(mode: str, atlas: dict[str, np.ndarray]) -> None:
    planes = (
        ("m30", "m30_50", r"$M_*(<30)$", r"$M_*(30\!-!50)$"),
        ("m30", "m50_100", r"$M_*(<30)$", r"$M_*(50\!-!100)$"),
        ("m30_50", "m100_148", r"$M_*(30\!-!50)$", r"$M_*(100\!-!148)$"),
    )
    colors = {
        "TNG CoG": "black",
        **dict(zip(REPRESENTATIONS, (OKABE_ITO[0], OKABE_ITO[2], OKABE_ITO[5]))),
    }
    labels = {"TNG CoG": "TNG", **REPRESENTATIONS}
    figure, axes = plt.subplots(3, len(REDSHIFTS), figsize=(15.7, 9.7))
    for epoch, redshift in enumerate(REDSHIFTS):
        profiles = {"TNG CoG": atlas["truth_log"][:, epoch]}
        profiles.update({name: atlas[name][:, epoch] for name in REPRESENTATIONS})
        quantities = {name: _mass_quantities(value) for name, value in profiles.items()}
        for row, (x_name, y_name, x_label, y_label) in enumerate(planes):
            axis = axes[row, epoch]
            for name, values in quantities.items():
                valid = (values[x_name] > 5.0) & (values[y_name] > 5.0)
                center, median, lower, upper = _binned_statistics(
                    values[x_name][valid], values[y_name][valid]
                )
                axis.plot(center, median, color=colors[name], label=labels[name])
                if name == "TNG CoG":
                    axis.fill_between(center, lower, upper, color="0.7", alpha=0.25)
            axis.set_xlabel(rf"$\log_{{10}}${x_label}")
            if epoch == 0:
                axis.set_ylabel(rf"$\log_{{10}}${y_label}")
            if row == 0:
                axis.set_title(rf"$z={redshift:g}$")
    axes[0, 0].legend(frameon=False, fontsize=6.8)
    figure.suptitle(
        r"Exp62 stellar-mass planes: binned medians; grey bands are the measured 16--84\% width"
    )
    figure.tight_layout()
    save_fig(figure, FIGDIR / "visual_record" / mode / "exp62_stellar_mass_planes")
    plt.close(figure)


def mass_size_relations_figure(mode: str, atlas: dict[str, np.ndarray]) -> None:
    colors = {
        "TNG CoG": "black",
        **dict(zip(REPRESENTATIONS, (OKABE_ITO[0], OKABE_ITO[2], OKABE_ITO[5]))),
    }
    labels = {"TNG CoG": "TNG", **REPRESENTATIONS}
    fractions = (0.5, 0.8, 0.9)
    figure, axes = plt.subplots(3, len(REDSHIFTS), figsize=(15.7, 9.7), sharex="col")
    for epoch, redshift in enumerate(REDSHIFTS):
        profiles = {"TNG CoG": atlas["truth_log"][:, epoch]}
        profiles.update({name: atlas[name][:, epoch] for name in REPRESENTATIONS})
        for row, fraction in enumerate(fractions):
            axis = axes[row, epoch]
            for name, profile in profiles.items():
                linear = 10.0 ** profile[:, None, :]
                size = np.log10(qa._safe_rhalf(linear, RADII, fraction)[:, 0])
                mass = profile[:, -1]
                center, median, lower, upper = _binned_statistics(mass, size)
                axis.plot(center, median, color=colors[name], label=labels[name])
                if name == "TNG CoG":
                    axis.fill_between(center, lower, upper, color="0.7", alpha=0.25)
            if epoch == 0:
                axis.set_ylabel(
                    rf"$\log_{{10}}R_{{{int(100 * fraction)}}}/\mathrm{{kpc}}$"
                )
            if row == 0:
                axis.set_title(rf"$z={redshift:g}$")
            if row == len(fractions) - 1:
                axis.set_xlabel(r"$\log_{10}M_{*,148}/M_\odot$")
    axes[0, 0].legend(frameon=False, fontsize=6.8)
    figure.suptitle(
        r"Exp62 mass--size relations: binned medians; grey bands are the measured 16--84\% width"
    )
    figure.tight_layout()
    save_fig(figure, FIGDIR / "visual_record" / mode / "exp62_mass_size_relations")
    plt.close(figure)


def _empirical_cdf(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    ordered = np.sort(values[np.isfinite(values)])
    return np.searchsorted(ordered, grid, side="right") / len(ordered)


def mass_cdf_residuals_figure(mode: str, atlas: dict[str, np.ndarray]) -> None:
    quantity_names = ("m30", "m30_50", "m50_100", "m100_148")
    quantity_labels = (
        r"$M_*(<30)$",
        r"$M_*(30\!-!50)$",
        r"$M_*(50\!-!100)$",
        r"$M_*(100\!-!148)$",
    )
    colors = (OKABE_ITO[0], OKABE_ITO[2], OKABE_ITO[5])
    figure, axes = plt.subplots(4, len(REDSHIFTS), figsize=(15.7, 11.7))
    for epoch, redshift in enumerate(REDSHIFTS):
        truth = _mass_quantities(atlas["truth_log"][:, epoch])
        models = {
            name: _mass_quantities(atlas[name][:, epoch]) for name in REPRESENTATIONS
        }
        for row, (quantity, label) in enumerate(zip(quantity_names, quantity_labels)):
            axis = axes[row, epoch]
            display_values = truth[quantity][truth[quantity] > 5.0]
            grid = np.linspace(
                np.quantile(display_values, 0.005),
                np.quantile(display_values, 0.995),
                160,
            )
            truth_cdf = _empirical_cdf(truth[quantity], grid)
            for (name, model), color in zip(models.items(), colors):
                axis.plot(
                    grid,
                    _empirical_cdf(model[quantity], grid) - truth_cdf,
                    color=color,
                    label=REPRESENTATIONS[name],
                )
            axis.axhline(0.0, color="0.55", linewidth=0.8)
            if epoch == 0:
                axis.set_ylabel(f"{label}\nmodel CDF - TNG CDF")
            if row == 0:
                axis.set_title(rf"$z={redshift:g}$")
            if row == len(quantity_names) - 1:
                axis.set_xlabel(r"$\log_{10}M_*/M_\odot$")
    axes[0, 0].legend(frameon=False, fontsize=6.5)
    figure.suptitle("Exp62 aperture and outskirt stellar-mass CDF residuals")
    figure.tight_layout()
    save_fig(figure, FIGDIR / "visual_record" / mode / "exp62_mass_cdf_residuals")
    plt.close(figure)


def worst_cases_figure(mode: str, atlas: dict[str, np.ndarray], name: str) -> None:
    residual = atlas[name] - atlas["truth_log"]
    rms = np.sqrt(np.mean(residual**2, axis=-1))
    score = np.max(rms, axis=1)
    worst = np.argsort(score)[-10:][::-1]
    colors = (
        OKABE_ITO[0],
        OKABE_ITO[1],
        OKABE_ITO[2],
        OKABE_ITO[5],
        OKABE_ITO[6],
    )
    figure, axes = plt.subplots(2, 5, figsize=(16.0, 6.6), sharex=True, sharey=True)
    for axis, row in zip(axes.ravel(), worst):
        pinned = (
            atlas[name][row]
            - atlas[name][row, :, -1:]
            - atlas["truth_log"][row]
            + atlas["truth_log"][row, :, -1:]
        )
        for epoch, (redshift, color) in enumerate(zip(REDSHIFTS, colors)):
            axis.plot(RADII, pinned[epoch], color=color, label=rf"$z={redshift:g}$")
        axis.axhline(0.0, color="0.55", linewidth=0.8)
        axis.set_xscale("log")
        axis.set_title(
            f"galaxy {atlas['galaxy_ids'][row]}; worst RMS={score[row]:.3f} dex",
            fontsize=8,
        )
        axis.set_xlabel("Radius (kpc)")
    axes[0, 0].set_ylabel("Mass-pinned CoG residual (dex)")
    axes[1, 0].set_ylabel("Mass-pinned CoG residual (dex)")
    axes[0, 0].legend(frameon=False, fontsize=6.5)
    figure.suptitle(f"Exp62 ten worst multi-epoch cases: {REPRESENTATIONS[name]}")
    figure.tight_layout()
    save_fig(figure, FIGDIR / "visual_record" / mode / f"exp62_worst10_{name}")
    plt.close(figure)


def epoch_performance_figure(mode: str, atlas: dict[str, np.ndarray]) -> None:
    colors = (OKABE_ITO[0], OKABE_ITO[2], OKABE_ITO[5])
    figure, axes = plt.subplots(2, 3, figsize=(13.0, 7.8), sharex=True)
    size_labels = ("R50", "R80", "R90")
    for (name, label), color in zip(REPRESENTATIONS.items(), colors):
        cog, density = [], []
        sizes = np.empty((len(REDSHIFTS), 3))
        mass_errors = np.empty((len(REDSHIFTS), 4))
        for epoch in range(len(REDSHIFTS)):
            metrics = profile_metrics(
                atlas[name][:, epoch], atlas["truth_log"][:, epoch]
            )
            cog.append(np.median(metrics["cog_rms_full_dex"]))
            density.append(np.median(metrics["density_rms_dex"]))
            sizes[epoch] = np.median(np.abs(metrics["size_error_dex"][:, 1:]), axis=0)
            truth_mass = _mass_quantities(atlas["truth_log"][:, epoch])
            model_mass = _mass_quantities(atlas[name][:, epoch])
            for quantity, quantity_name in enumerate(
                ("m30", "m30_50", "m50_100", "m100_148")
            ):
                mass_errors[epoch, quantity] = np.median(
                    np.abs(model_mass[quantity_name] - truth_mass[quantity_name])
                )
        axes[0, 0].plot(REDSHIFTS, cog, "o-", color=color, label=label)
        axes[0, 1].plot(REDSHIFTS, density, "o-", color=color)
        for size, linestyle in enumerate(("-", "--", ":")):
            axes[0, 2].plot(
                REDSHIFTS,
                sizes[:, size],
                linestyle=linestyle,
                marker="o",
                color=color,
                alpha=1.0 - 0.18 * size,
                label=(size_labels[size] if name == PRIMARY_REPRESENTATION else None),
            )
        for quantity in range(3):
            axes[1, quantity].plot(
                REDSHIFTS,
                mass_errors[:, quantity + 1],
                "o-",
                color=color,
                label=label if quantity == 0 else None,
            )
    titles = (
        "Full CoG RMS",
        "Differenced density RMS",
        "R50/R80/R90 errors",
        "30--50 kpc mass error",
        "50--100 kpc mass error",
        "100--148 kpc mass error",
    )
    for axis, title in zip(axes.ravel(), titles):
        axis.set_title(title)
        axis.set_xlabel("Redshift")
        axis.set_ylabel("Median absolute error (dex)")
        axis.invert_xaxis()
    axes[0, 0].legend(frameon=False, fontsize=6.7)
    axes[0, 2].legend(frameon=False, fontsize=6.7, title="Line style")
    figure.suptitle("Exp62 model performance across epochs")
    figure.tight_layout()
    save_fig(figure, FIGDIR / "visual_record" / mode / "exp62_epoch_performance")
    plt.close(figure)


def _mass_residual(values: np.ndarray, halo_mass: np.ndarray) -> np.ndarray:
    centered = (halo_mass - np.mean(halo_mass)) / np.std(halo_mass)
    design = np.column_stack([np.ones(len(centered)), centered, centered**2])
    return values - design @ np.linalg.lstsq(design, values, rcond=None)[0]


def parameter_diffmah_scaling_figure(mode: str) -> None:
    data = load_halo_data(mode)
    parameters = data["targets"][:, :, data["blocks"][PRIMARY_RAW]]
    parameter_labels = (
        r"$\log M_{*,148}$",
        r"$\mathrm{logit}(f_c)$",
        r"$\log R_c$",
        r"$\log(R_e/R_c-1)$",
    )
    predictor_labels = (
        r"$\log M_{\rm peak}$",
        r"$\log t_c$",
        "early index",
        "late index",
        r"$c_{200c}$",
    )
    epoch_colors = (
        OKABE_ITO[0],
        OKABE_ITO[1],
        OKABE_ITO[2],
        OKABE_ITO[5],
        OKABE_ITO[6],
    )
    figure, axes = plt.subplots(4, 5, figsize=(16.0, 12.0), sharex=True, sharey="row")
    for epoch, (redshift, color) in enumerate(zip(REDSHIFTS, epoch_colors)):
        x_residual = _mass_residual(
            data["features"][:, epoch], data["halo_mass"][:, epoch]
        )
        y_residual = _mass_residual(parameters[:, epoch], data["halo_mass"][:, epoch])
        x_residual /= np.std(x_residual, axis=0)
        y_residual /= np.std(y_residual, axis=0)
        for parameter in range(4):
            for predictor in range(5):
                center, median, lower, upper = _binned_statistics(
                    x_residual[:, predictor], y_residual[:, parameter], n_bin=8
                )
                axis = axes[parameter, predictor]
                axis.plot(center, median, color=color, label=rf"$z={redshift:g}$")
                axis.fill_between(center, lower, upper, color=color, alpha=0.035)
    for parameter in range(4):
        for predictor in range(5):
            axis = axes[parameter, predictor]
            axis.axhline(0.0, color="0.65", linewidth=0.7)
            axis.axvline(0.0, color="0.8", linewidth=0.7)
            if parameter == 0:
                axis.set_title(predictor_labels[predictor])
            if predictor == 0:
                axis.set_ylabel(f"{parameter_labels[parameter]}\nmass-residual SD")
            if parameter == 3:
                axis.set_xlabel("Mass-conditioned predictor residual (SD)")
    axes[0, 0].legend(frameon=False, fontsize=6.5)
    figure.suptitle(
        "Exp62 direct parameter--DiffMAH scaling: constrained Sersic--Moffat binned medians"
    )
    figure.text(
        0.5,
        0.005,
        r"Both axes have had their quadratic concurrent-halo-mass trend removed and are standardized within each epoch; shaded bands show the 16--84\% parameter distribution.",
        ha="center",
        fontsize=7.3,
        color="0.35",
    )
    figure.tight_layout(rect=(0, 0.025, 1, 1))
    save_fig(
        figure,
        FIGDIR / "visual_record" / mode / "exp62_parameter_diffmah_scaling",
    )
    plt.close(figure)


def parameter_diffmah_raw_scaling_figure(mode: str) -> None:
    """Show the unconditioned halo--profile parameter scaling relations."""
    data = load_halo_data(mode)
    raw_parameters = data["targets"][:, :, data["blocks"][PRIMARY_RAW]]
    physical_parameters = np.stack(
        [
            raw_parameters[..., 0],
            expit(raw_parameters[..., 1]),
            10.0 ** raw_parameters[..., 2],
            1.0 + np.exp(raw_parameters[..., 3]),
        ],
        axis=-1,
    )
    parameter_labels = (
        r"$\log_{10}M_{*,148}/M_\odot$",
        r"Compact fraction $f_c$",
        r"Compact radius $R_c$ (kpc)",
        r"Component radius ratio $R_e/R_c$",
    )
    predictor_labels = (
        r"$\log_{10}M_{\rm peak}/M_\odot$",
        r"$\log_{10}(t_c/\mathrm{Gyr})$",
        r"Early index $\alpha_{\rm early}$",
        r"Late index $\alpha_{\rm late}$",
        r"Concentration $c_{200c}$",
    )
    epoch_colors = (
        OKABE_ITO[0],
        OKABE_ITO[1],
        OKABE_ITO[2],
        OKABE_ITO[5],
        OKABE_ITO[6],
    )
    figure, axes = plt.subplots(4, 5, figsize=(16.0, 12.0), sharex="col", sharey="row")
    for epoch, (redshift, color) in enumerate(zip(REDSHIFTS, epoch_colors)):
        for parameter in range(4):
            for predictor in range(5):
                center, median, lower, upper = _binned_statistics(
                    data["features"][:, epoch, predictor],
                    physical_parameters[:, epoch, parameter],
                    n_bin=8,
                )
                axis = axes[parameter, predictor]
                axis.plot(center, median, color=color, label=rf"$z={redshift:g}$")
                axis.fill_between(center, lower, upper, color=color, alpha=0.035)
    for parameter in range(4):
        for predictor in range(5):
            axis = axes[parameter, predictor]
            if parameter == 0:
                axis.set_title(predictor_labels[predictor])
            if predictor == 0:
                axis.set_ylabel(parameter_labels[parameter])
            if parameter == 3:
                axis.set_xlabel(predictor_labels[predictor])
    axes[0, 0].legend(frameon=False, fontsize=6.5)
    figure.suptitle(
        "Exp62 traditional parameter--parameter scaling: unconditioned binned medians"
    )
    figure.text(
        0.5,
        0.005,
        r"Axes show the measured epoch-local values without halo-mass subtraction or standardization; shaded bands are the 16--84\% profile-parameter distributions.",
        ha="center",
        fontsize=7.3,
        color="0.35",
    )
    figure.tight_layout(rect=(0, 0.025, 1, 1))
    save_fig(
        figure,
        FIGDIR / "visual_record" / mode / "exp62_parameter_diffmah_raw_scaling",
    )
    plt.close(figure)


def halo_mass_sufficiency_figure(mode: str) -> None:
    """Test held-out profile information beyond concurrent halo mass."""
    stored = np.load(OUTDIR / mode / "stage4" / "associations.npz")
    data = load_halo_data(mode)
    target_names = np.asarray(stored["target_names"]).astype(str)
    target_lookup = {name: index for index, name in enumerate(target_names)}
    selected_targets = (
        ("truth_decoded::log_mstar_148", r"$M_{*,148}$", "black", "-"),
        ("truth_decoded::log_r50", r"$R_{50}$", OKABE_ITO[0], "-"),
        ("truth_decoded::log_r80", r"$R_{80}$", OKABE_ITO[2], "--"),
        ("truth_decoded::log_r90", r"$R_{90}$", OKABE_ITO[5], "-."),
        (
            "truth_decoded::log_fraction_inside_2",
            r"$M_*(<2)/M_{*,148}$",
            OKABE_ITO[1],
            ":",
        ),
        (
            "truth_decoded::cog_slope_10_30",
            "CoG slope, 10--30 kpc",
            OKABE_ITO[6],
            "-",
        ),
        (
            "truth_decoded::cog_slope_30_100",
            "CoG slope, 30--100 kpc",
            OKABE_ITO[3],
            "--",
        ),
    )
    primary_raw = data["blocks"][PRIMARY_RAW]
    mass_profile, _ = _decode_predictions(
        data["primary_variants"], stored["mass_prediction"][:, :, primary_raw]
    )
    complete_profile, _ = _decode_predictions(
        data["primary_variants"], stored["full_prediction"][:, :, primary_raw]
    )
    truth_profile = data["truth_log"]

    def pinned_metrics(profile: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pinned = profile - profile[..., -1, None] + truth_profile[..., -1, None]
        cog_rms = np.sqrt(np.mean((pinned - truth_profile) ** 2, axis=-1))
        density, _ = qa.density_from_cog(pinned.reshape(-1, len(RADII)), RADII)
        truth_density, _ = qa.density_from_cog(
            truth_profile.reshape(-1, len(RADII)), RADII
        )
        density_rms = np.sqrt(np.mean((density - truth_density) ** 2, axis=-1))
        return cog_rms, density_rms.reshape(cog_rms.shape)

    mass_cog_rms, mass_density_rms = pinned_metrics(mass_profile)
    complete_cog_rms, complete_density_rms = pinned_metrics(complete_profile)
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    for name, label, color, linestyle in selected_targets:
        index = target_lookup[name]
        gain = (
            100.0
            * (stored["mass_rmse"][:, index] - stored["full_rmse"][:, index])
            / stored["mass_rmse"][:, index]
        )
        axes[0].plot(
            REDSHIFTS,
            gain,
            "o",
            color=color,
            linestyle=linestyle,
            linewidth=1.5,
            markersize=4,
            label=label,
        )
    axes[0].axhline(0.0, color="0.6", linewidth=0.8)
    axes[0].set_ylabel("Held-out RMS reduction beyond halo mass (%)")
    axes[0].legend(frameon=False, fontsize=6.3, ncol=2)
    for axis, mass_values, complete_values, ylabel in (
        (
            axes[1],
            mass_cog_rms,
            complete_cog_rms,
            "Median amplitude-pinned CoG RMS (dex)",
        ),
        (
            axes[2],
            mass_density_rms,
            complete_density_rms,
            "Median amplitude-pinned density RMS (dex)",
        ),
    ):
        axis.plot(
            REDSHIFTS,
            np.median(mass_values, axis=0),
            "o--",
            color="0.35",
            label="Concurrent halo mass only",
        )
        axis.plot(
            REDSHIFTS,
            np.median(complete_values, axis=0),
            "o-",
            color=OKABE_ITO[2],
            label="Halo mass + DiffMAH + concentration",
        )
        axis.set_ylabel(ylabel)
        axis.legend(frameon=False, fontsize=6.7)
    for axis in axes:
        axis.set_xlabel("Redshift")
        axis.invert_xaxis()
    figure.suptitle(
        "Exp62 held-out test: how much CoG information lies beyond concurrent halo mass?"
    )
    figure.text(
        0.5,
        0.005,
        "Five-fold predictions; the middle and right panels force every predicted CoG to the measured M*148 before scoring its shape.",
        ha="center",
        fontsize=7.3,
        color="0.35",
    )
    figure.tight_layout(rect=(0, 0.035, 1, 1))
    save_fig(
        figure,
        FIGDIR / "visual_record" / mode / "exp62_halo_mass_sufficiency",
    )
    plt.close(figure)


def _corner_density_contours(
    axis: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    color: str,
) -> None:
    histogram, x_edges, y_edges = np.histogram2d(
        x, y, bins=38, range=(x_limits, y_limits)
    )
    density = gaussian_filter(histogram.T, 1.0)
    ordered = np.sort(density.ravel())[::-1]
    cumulative = np.cumsum(ordered)
    if cumulative[-1] <= 0.0:
        return
    cumulative /= cumulative[-1]
    thresholds = []
    for fraction in (0.90, 0.50):
        index = min(np.searchsorted(cumulative, fraction), len(ordered) - 1)
        thresholds.append(ordered[index])
    levels = np.unique(np.sort(thresholds))
    if len(levels) == 0 or levels[-1] <= 0.0:
        return
    x_center = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_center = 0.5 * (y_edges[:-1] + y_edges[1:])
    linestyles = ("--", "-")[-len(levels) :]
    axis.contour(
        x_center,
        y_center,
        density,
        levels=levels,
        colors=(color,),
        linewidths=1.0,
        linestyles=linestyles,
    )


def profile_parameter_corner_figures(mode: str, rows: np.ndarray | None = None) -> None:
    """Show per-family parameter ranges and degeneracies at every epoch."""
    stored = np.load(OUTDIR / "full" / "stage3" / "scaling_and_evolution.npz")
    configurations = (
        (
            "gompertz_log",
            "Log-radius Gompertz",
            (
                r"$\log_{10}(M_{*,148}/M_\odot)$",
                r"$\log_{10}(R_{50}/\mathrm{kpc})$",
                r"$\log_{10}c$",
            ),
            lambda raw: np.stack(
                [raw[..., 0], raw[..., 1], raw[..., 2] / np.log(10.0)], axis=-1
            ),
        ),
        (
            "sersic_moffat_fixed",
            "Constrained Sersic--Moffat",
            (
                r"$\log_{10}(M_{*,148}/M_\odot)$",
                r"$f_c$",
                r"$\log_{10}(R_c/\mathrm{kpc})$",
                r"$\log_{10}(R_e/R_c)$",
            ),
            lambda raw: np.stack(
                [
                    raw[..., 0],
                    expit(raw[..., 1]),
                    raw[..., 2],
                    np.log10(1.0 + np.exp(raw[..., 3])),
                ],
                axis=-1,
            ),
        ),
        (
            "double_sersic_free",
            "Unrestricted double Sersic",
            (
                r"$\log_{10}(M_{*,148}/M_\odot)$",
                r"$f_c$",
                r"$\log_{10}(R_c/\mathrm{kpc})$",
                r"$\log_{10}(R_e/R_c)$",
                r"$\log_{10}n_c$",
                r"$\log_{10}n_e$",
            ),
            lambda raw: np.stack(
                [
                    raw[..., 0],
                    expit(raw[..., 1]),
                    raw[..., 2],
                    np.log10(1.0 + np.exp(raw[..., 3])),
                    raw[..., 4] / np.log(10.0),
                    raw[..., 5] / np.log(10.0),
                ],
                axis=-1,
            ),
        ),
    )
    epoch_colors = (
        OKABE_ITO[0],
        OKABE_ITO[1],
        OKABE_ITO[2],
        OKABE_ITO[5],
        OKABE_ITO[6],
    )
    for family, title, labels, transform in configurations:
        raw = np.asarray(stored[f"{family}_cog_log_parameters"])
        if rows is not None:
            raw = raw[rows]
        parameters = transform(raw)
        n_parameter = parameters.shape[-1]
        limits = []
        for parameter in range(n_parameter):
            lower, upper = np.quantile(parameters[..., parameter], [0.001, 0.999])
            padding = max(0.04 * (upper - lower), 1.0e-4)
            limits.append((float(lower - padding), float(upper + padding)))
        figure, axes = plt.subplots(
            n_parameter,
            n_parameter,
            figsize=(2.65 * n_parameter, 2.65 * n_parameter),
            squeeze=False,
        )
        for row in range(n_parameter):
            for column in range(n_parameter):
                axis = axes[row, column]
                if column > row:
                    axis.axis("off")
                    continue
                if row == column:
                    bins = np.linspace(*limits[column], 35)
                    for epoch, (redshift, color) in enumerate(
                        zip(REDSHIFTS, epoch_colors)
                    ):
                        axis.hist(
                            parameters[:, epoch, column],
                            bins=bins,
                            density=True,
                            histtype="step",
                            linewidth=1.3,
                            color=color,
                            label=rf"$z={redshift:g}$",
                        )
                    axis.set_yticks([])
                else:
                    for epoch, color in enumerate(epoch_colors):
                        _corner_density_contours(
                            axis,
                            parameters[:, epoch, column],
                            parameters[:, epoch, row],
                            limits[column],
                            limits[row],
                            color,
                        )
                axis.set_xlim(limits[column])
                if row != column:
                    axis.set_ylim(limits[row])
                if row == n_parameter - 1:
                    axis.set_xlabel(labels[column])
                else:
                    axis.set_xticklabels([])
                if column == 0 and row > 0:
                    axis.set_ylabel(labels[row])
                elif column > 0:
                    axis.set_yticklabels([])
        axes[0, 0].legend(
            frameon=False,
            fontsize=7.0,
            bbox_to_anchor=(1.02, 1.0),
            loc="upper left",
        )
        figure.suptitle(
            f"Exp62 {title}: fitted parameter distributions and covariances"
        )
        figure.text(
            0.5,
            0.005,
            r"Diagonal: normalized distributions. Solid/dashed contours enclose 50\%/90\%. Limits: pooled 0.1--99.9 percentiles.",
            ha="center",
            fontsize=7.2,
            color="0.35",
        )
        figure.tight_layout(rect=(0, 0.025, 1, 1))
        save_fig(
            figure,
            FIGDIR / "visual_record" / mode / f"exp62_corner_{family}",
        )
        plt.close(figure)


def _decoded_quantity(values: np.ndarray, quantity: str) -> np.ndarray:
    index = {name: position for position, name in enumerate(DECODED_NAMES)}
    if quantity == "log_outer_fraction_50_100":
        return values[..., index["log_mass_50_100"]] - values[..., 0]
    return values[..., index[quantity]]


def common_scaling_figure(mode: str, rows: np.ndarray | None = None) -> None:
    stored = np.load(OUTDIR / "full" / "stage3" / "scaling_and_evolution.npz")
    decoded = {"TNG CoG": np.asarray(stored["truth_decoded"])}
    decoded.update(
        {
            REPRESENTATIONS[name]: np.asarray(stored[key])
            for name, key in STAGE3_KEYS.items()
        }
    )
    if rows is not None:
        decoded = {name: values[rows] for name, values in decoded.items()}
    quantities = (
        ("log_fraction_inside_2", r"$\log_{10}[M_*(<2)/M_{*,148}]$"),
        ("log_r50", r"$\log_{10}R_{50}\ (\mathrm{kpc})$"),
        ("log_r90", r"$\log_{10}R_{90}\ (\mathrm{kpc})$"),
        (
            "log_outer_fraction_50_100",
            r"$\log_{10}[M_*(50\!-!100)/M_{*,148}]$",
        ),
    )
    epochs = (0, 2, 4)
    styles = {
        "TNG CoG": ("black", "-", 2.2),
        REPRESENTATIONS["gompertz_log"]: (OKABE_ITO[0], ":", 1.8),
        REPRESENTATIONS["sersic_moffat_fixed"]: (OKABE_ITO[2], "--", 1.8),
        REPRESENTATIONS["double_sersic_free"]: (OKABE_ITO[5], "-.", 1.8),
    }
    figure, axes = plt.subplots(
        len(quantities), len(epochs), figsize=(13.5, 12.5), sharex="col"
    )
    for row, (quantity, ylabel) in enumerate(quantities):
        for column, epoch in enumerate(epochs):
            axis = axes[row, column]
            x = decoded["TNG CoG"][:, epoch, 0]
            for label, values in decoded.items():
                y = _decoded_quantity(values[:, epoch], quantity)
                center, median, lower, upper = _binned_statistics(x, y)
                color, linestyle, linewidth = styles[label]
                if label == "TNG CoG":
                    axis.fill_between(center, lower, upper, color="0.75", alpha=0.32)
                axis.plot(
                    center,
                    median,
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    label=label,
                )
            if row == 0:
                axis.set_title(rf"$z={REDSHIFTS[epoch]:g}$")
            if column == 0:
                axis.set_ylabel(ylabel)
            if row == len(quantities) - 1:
                axis.set_xlabel(r"Measured $\log_{10}M_{*,148}/M_\odot$")
    axes[0, 0].legend(frameon=False, fontsize=7.3)
    figure.suptitle(
        "Exp62 common decoded scaling relations: measured CoGs and three analytic representations"
    )
    figure.tight_layout()
    save_fig(
        figure,
        FIGDIR / "visual_record" / mode / "exp62_common_scaling_relations",
    )
    plt.close(figure)


def _physical_parameters(stored: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    gompertz = np.asarray(stored["gompertz_log_cog_log_parameters"])
    mixture = np.asarray(stored["sersic_moffat_fixed_cog_log_parameters"])
    variants = np.asarray(stored["sersic_moffat_fixed_cog_log_variants"]).astype(str)
    n_sersic = np.empty(variants.shape)
    gamma = np.empty(variants.shape)
    for family in np.unique(variants):
        selected = variants == family
        n_value, gamma_value = _sersic_moffat_indices(str(family))
        n_sersic[selected] = n_value
        gamma[selected] = gamma_value
    compact_radius = 10.0 ** mixture[..., 2]
    return {
        "log_mstar_148": gompertz[..., 0],
        "gompertz_r50": 10.0 ** gompertz[..., 1],
        "gompertz_c": np.exp(gompertz[..., 2]),
        "compact_fraction": expit(mixture[..., 1]),
        "compact_radius": compact_radius,
        "extended_radius": compact_radius * (1.0 + np.exp(mixture[..., 3])),
        "sersic_n": n_sersic,
        "moffat_gamma": gamma,
    }


def parameter_evolution_figure(mode: str, rows: np.ndarray | None = None) -> None:
    stored = np.load(OUTDIR / "full" / "stage3" / "scaling_and_evolution.npz")
    parameters = _physical_parameters(stored)
    final_mass = np.asarray(stored["truth_decoded"])[..., 0][:, 0]
    if rows is not None:
        parameters = {name: values[rows] for name, values in parameters.items()}
        final_mass = final_mass[rows]
    edges = np.quantile(final_mass, [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
    panels = (
        ("log_mstar_148", r"$\log_{10}M_{*,148}/M_\odot$", False),
        ("gompertz_r50", r"Gompertz $R_{50}$ (kpc)", True),
        ("gompertz_c", r"Gompertz $c$", False),
        ("compact_fraction", r"Sersic compact fraction at 148 kpc", False),
        ("compact_radius", r"Sersic $R_{50,\mathrm{compact}}$ (kpc)", True),
        ("extended_radius", r"Moffat $R_{50,\mathrm{extended}}$ (kpc)", True),
        ("sersic_n", r"Fold-selected global Sersic $n$", False),
        ("moffat_gamma", r"Fold-selected global Moffat $\gamma$", False),
    )
    colors = (OKABE_ITO[0], OKABE_ITO[2], OKABE_ITO[5])
    figure, axes = plt.subplots(2, 4, figsize=(15.5, 7.7), sharex=True)
    for axis, (name, ylabel, logarithmic) in zip(axes.ravel(), panels):
        values = parameters[name]
        for mass_bin, color in enumerate(colors):
            selected = (final_mass >= edges[mass_bin]) & (
                final_mass <= edges[mass_bin + 1]
            )
            median = np.median(values[selected], axis=0)
            lower, upper = np.quantile(values[selected], [0.16, 0.84], axis=0)
            label = (
                rf"${edges[mass_bin]:.2f}<\log M_{{*,148}}(z=0.4)"
                rf"<{edges[mass_bin + 1]:.2f}$"
            )
            axis.plot(REDSHIFTS, median, "o-", color=color, label=label)
            axis.fill_between(REDSHIFTS, lower, upper, color=color, alpha=0.10)
        if logarithmic:
            axis.set_yscale("log")
        axis.set_ylabel(ylabel)
        axis.set_xlabel("Redshift")
        axis.invert_xaxis()
    axes[0, 0].legend(frameon=False, fontsize=6.8)
    figure.suptitle(
        r"Exp62 matched-track parameter evolution; bands are population 16--84\% ranges"
    )
    figure.text(
        0.5,
        0.005,
        "The unrestricted double-Sersic raw parameters are omitted because Stage 2 shows that they move along degeneracy valleys at nearly fixed CoG.",
        ha="center",
        fontsize=7.4,
        color="0.35",
    )
    figure.tight_layout(rect=(0, 0.025, 1, 1))
    save_fig(
        figure,
        FIGDIR / "visual_record" / mode / "exp62_parameter_redshift_evolution",
    )
    plt.close(figure)


def symbolic_halo_cog_figure(mode: str) -> None:
    stored = np.load(OUTDIR / "full" / "stage4" / "associations.npz")
    target_names = np.asarray(stored["target_names"]).astype(str)
    raw = np.flatnonzero(
        np.char.startswith(target_names, "sersic_moffat_fixed_cog_log_raw::")
    )
    if len(raw) != 4:
        raise ValueError("the primary Sersic--Moffat raw target block is incomplete")
    beta = np.asarray(stored["conditional_beta_standardized"])[..., raw]
    bootstrap = np.asarray(stored["bootstrap_beta_standardized"])[..., raw]
    limit = float(np.quantile(np.abs(beta), 0.98))
    predictor_labels = (
        r"$\log M_{\rm peak}$",
        r"$\log t_c$",
        "early index",
        "late index",
        r"$c_{200c}$",
    )
    parameter_labels = (
        r"$\log M_{*,148}$",
        r"$\mathrm{logit}(f_c)$",
        r"$\log R_c$",
        r"$\log(R_e/R_c-1)$",
    )
    figure = plt.figure(figsize=(17.5, 9.0))
    grid = figure.add_gridspec(2, 5, height_ratios=(1.05, 1.0), hspace=0.28)
    equation_axis = figure.add_subplot(grid[0, :])
    equation_axis.axis("off")
    box = dict(boxstyle="round,pad=0.55", facecolor="white", edgecolor="0.3")
    equation_axis.text(
        0.015,
        0.57,
        r"$\mathbf{x}=(\log M_{\rm peak},\log t_c,e,l,c_{200c})$"
        "\n"
        r"$\widetilde{x}_j=\mathrm{std}\{x_j-\langle x_j\mid\log M_h(z)\rangle\}$",
        transform=equation_axis.transAxes,
        ha="left",
        va="center",
        fontsize=13,
        bbox=box,
    )
    equation_axis.text(
        0.365,
        0.57,
        r"$\theta_k=a_k[\log M_h(z)]+\sum_{j=1}^{5}\beta_{kj}(z)\widetilde{x}_j$"
        "\n"
        r"$\theta=(\log M_{*,148},\mathrm{logit}f_c,\log R_c,"
        r"\log[R_e/R_c-1])$",
        transform=equation_axis.transAxes,
        ha="left",
        va="center",
        fontsize=13,
        bbox=box,
    )
    equation_axis.text(
        0.735,
        0.57,
        r"$M_*(<R)=10^{\theta_M}\,[f_cP_n(R\mid R_c)"
        r"+(1-f_c)Q_\gamma(R\mid R_e)]$"
        "\n"
        r"$f_c=\sigma(\theta_f),\ R_e=R_c(1+e^{\theta_e});\quad P_n(148)=Q_\gamma(148)=1$",
        transform=equation_axis.transAxes,
        ha="left",
        va="center",
        fontsize=12.2,
        bbox=box,
    )
    for start, end in ((0.315, 0.36), (0.685, 0.73)):
        equation_axis.annotate(
            "",
            xy=(end, 0.57),
            xytext=(start, 0.57),
            xycoords="axes fraction",
            arrowprops=dict(arrowstyle="->", linewidth=1.8, color="0.25"),
        )
    equation_axis.text(
        0.5,
        0.09,
        "Mass trend and halo coordinates  →  four analytic coordinates  →  the complete CoG",
        transform=equation_axis.transAxes,
        ha="center",
        fontsize=12,
        weight="bold",
    )

    heatmap_axes = []
    for epoch, redshift in enumerate(REDSHIFTS):
        axis = figure.add_subplot(grid[1, epoch])
        heatmap_axes.append(axis)
        values = beta[epoch].T
        image = axis.imshow(
            values,
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            aspect="auto",
        )
        lower, upper = np.quantile(bootstrap[epoch], [0.025, 0.975], axis=0)
        for parameter in range(4):
            for predictor in range(5):
                if lower[predictor, parameter] * upper[predictor, parameter] > 0.0:
                    axis.text(predictor, parameter, "*", ha="center", va="center")
        axis.set_xticks(np.arange(5), predictor_labels, rotation=55, ha="right")
        axis.set_yticks(np.arange(4), parameter_labels if epoch == 0 else ())
        axis.set_title(rf"$z={redshift:g}$")
    figure.colorbar(
        image,
        ax=heatmap_axes,
        shrink=0.78,
        pad=0.015,
        label="Conditional response / mass-residual parameter scatter",
    )
    figure.suptitle(
        r"Exp62 symbolic epoch-local halo--CoG relation; stars mark bootstrap 95\% intervals excluding zero",
        y=0.985,
    )
    figure.text(
        0.5,
        0.005,
        "The coefficients are cross-fitted descriptive associations. The diagram does not impose or establish an inner--early-MAH or outer--recent-assembly connection.",
        ha="center",
        fontsize=7.5,
        color="0.35",
    )
    save_fig(
        figure,
        FIGDIR / "visual_record" / mode / "exp62_symbolic_halo_cog_relation",
    )
    plt.close(figure)


def run(mode: str) -> None:
    started = time.perf_counter()
    atlas = load_atlas(mode)
    qa_summary = performance_summary(atlas)
    representation_qa_comparison(mode, atlas)
    average_cog_bin_figure(mode, atlas, "halo")
    average_cog_bin_figure(mode, atlas, "stellar")
    stellar_mass_planes_figure(mode, atlas)
    mass_size_relations_figure(mode, atlas)
    mass_cdf_residuals_figure(mode, atlas)
    epoch_performance_figure(mode, atlas)
    for representation in REPRESENTATIONS:
        worst_cases_figure(mode, atlas, representation)
    stage3 = np.load(OUTDIR / "full" / "stage3" / "scaling_and_evolution.npz")
    stage3_rows = None
    if mode == "demo":
        stage3_rows = _stratified_rows(np.asarray(stage3["truth_decoded"])[:, 0, 0])
    common_scaling_figure(mode, stage3_rows)
    parameter_evolution_figure(mode, stage3_rows)
    parameter_diffmah_scaling_figure(mode)
    parameter_diffmah_raw_scaling_figure(mode)
    halo_mass_sufficiency_figure(mode)
    profile_parameter_corner_figures(mode, stage3_rows)
    symbolic_halo_cog_figure(mode)
    wall_seconds = time.perf_counter() - started
    summary = {
        "mode": mode,
        "n_galaxy_qa": len(atlas["truth_log"]),
        "n_track_scaling": N_DEMO if mode == "demo" else len(stage3["galaxy_ids"]),
        "population_qa": qa_summary,
        "wall_seconds": wall_seconds,
        "subminute_gate_passed": wall_seconds < 60.0 if mode == "demo" else None,
    }
    output_directory = OUTDIR / mode / "visual_record"
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_manifest(
        output_directory,
        {
            "experiment": "exp62_cog_fit_atlas",
            "stage": "final_visual_record",
            "mode": mode,
            "n_galaxy_qa": len(atlas["truth_log"]),
        },
    )
    print(json.dumps(summary, indent=2))
    if mode == "demo" and wall_seconds >= 60.0:
        raise RuntimeError(
            f"visual-record demo took {wall_seconds:.2f} seconds; full run is forbidden"
        )


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if command not in ("demo", "full"):
        raise SystemExit(__doc__)
    run(command)
