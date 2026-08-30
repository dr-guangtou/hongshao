# %%
"""Read-only comparative QA for the scientifically informative Exp67 families.

Commands:

    uv run python experiments/exp67_squared_slope_symbolic_density/exp67_qa.py smoke
    uv run python experiments/exp67_squared_slope_symbolic_density/exp67_qa.py full

The driver only reads the completed Exp62 and Exp67 artifacts. It does not
refit profiles, rerank expressions, or inspect Exp67 selection/validation data.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP66_DIR = HERE.parent / "exp66_expanded_symbolic_density"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXP66_DIR))

from exp66_search import load_references, profile_metrics  # noqa: E402
from exp67_grammar import RADII_KPC, shell_log_density  # noqa: E402
from exp67_search import population_split  # noqa: E402
from hongshao.plotting import OKABE_ITO, save_fig, set_style  # noqa: E402
from hongshao.provenance import write_manifest  # noqa: E402

set_style()
plt.rcParams["text.usetex"] = False

FIGDIR = HERE / "figures" / "discovery_comparative_qa"
OUTDIR = HERE / "outputs" / "discovery_comparative_qa"
EXPRESSION_DIR = HERE / "outputs" / "discovery_round_a" / "expressions"
EXP62_OUTDIR = Path(
    "/Users/shuang/Dropbox/work/project/massive/hongshao_exp62_cog_fit_atlas/"
    "experiments/exp62_cog_fit_atlas/outputs"
)
DOUBLE_SERSIC_PATH = EXP62_OUTDIR / "full" / "variants" / "double_sersic_free.npz"
REDSHIFTS = np.asarray((0.4, 0.7, 1.0, 1.5, 2.0))
EPOCH_COLORS = (OKABE_ITO[0], OKABE_ITO[1], OKABE_ITO[2], OKABE_ITO[5], OKABE_ITO[6])

COMMON_WRAPPER = (
    r"New families: $u=\ln(1+R/R_c)$, $\Sigma=\Sigma_0e^{-Q(u)}$, "
    r"$Q'(u)=(1-e^{-u})\{2.02+[a+bT(u)]^2\}$."
)
DOUBLE_SERSIC_FORMULA = (
    r"$\Sigma=\Sigma_c e^{-b_{n_c}(R/R_c)^{1/n_c}}"
    r"+\Sigma_e e^{-b_{n_e}(R/R_e)^{1/n_e}}$"
)


@dataclass(frozen=True)
class ModelRecord:
    label: str
    short_label: str
    formula: str
    n_parameter: int
    analytic_status: str
    color: str
    linestyle: str


MODELS = {
    "double_sersic_free": ModelRecord(
        "Unrestricted double Sersic",
        "Double Sersic",
        DOUBLE_SERSIC_FORMULA,
        6,
        r"$\Sigma$ exact; CoG uses incomplete gamma",
        OKABE_ITO[7],
        "--",
    ),
    "free_damped_cosine": ModelRecord(
        "Free damped cosine",
        "Damped cosine",
        r"$T(u)=e^{-\lambda u}\cos(\omega u)$",
        6,
        r"$Q$ and $\Sigma$ elementary",
        OKABE_ITO[5],
        "-",
    ),
    "wave_packet_sine_w0p5_w0p5": ModelRecord(
        "Fixed sine wave packet",
        "Sine packet",
        r"$T(u)=\mathrm{sech}[(u-t)/0.5]\sin[0.5(u-t)]$",
        5,
        r"$Q$ and $\Sigma$ require quadrature",
        OKABE_ITO[2],
        "-",
    ),
    "log_harmonic_sine_w4": ModelRecord(
        "Log-periodic sine",
        "Log-periodic sine",
        r"$T(u)=\sin\{4\ln(1+u/s)\}$",
        5,
        r"$Q$ and $\Sigma$ require quadrature",
        OKABE_ITO[4],
        "-",
    ),
}


def _candidate_path(name: str) -> Path:
    return EXPRESSION_DIR / f"{name}.npz"


def _select_complete_galaxies(
    reference: dict[str, np.ndarray], indices: np.ndarray, count: int
) -> np.ndarray:
    epoch_zero = indices[reference["epochs"][indices] == 0]
    rows = reference["rows"][epoch_zero[:count]]
    return indices[np.isin(reference["rows"][indices], rows)]


def load_discovery_data(limit_galaxies: int | None = None) -> dict[str, object]:
    """Load aligned stored fits for the frozen Exp67 discovery population."""
    reference = load_references("full")
    discovery = np.asarray(population_split(reference)["discovery"], int)
    candidate_archives = {
        name: np.load(_candidate_path(name))
        for name in MODELS
        if name != "double_sersic_free"
    }
    for name, archive in candidate_archives.items():
        if not np.array_equal(np.asarray(archive["indices"], int), discovery):
            raise ValueError(
                f"stored {name} rows do not match the frozen discovery split"
            )

    if not DOUBLE_SERSIC_PATH.exists():
        raise FileNotFoundError(
            f"missing double-Sersic diagnostics: {DOUBLE_SERSIC_PATH}"
        )
    double_archive = np.load(DOUBLE_SERSIC_PATH)
    local = np.arange(len(discovery))
    if limit_galaxies is not None:
        selected = _select_complete_galaxies(reference, discovery, limit_galaxies)
        local = np.flatnonzero(np.isin(discovery, selected))
        discovery = discovery[local]

    predictions = {
        "double_sersic_free": np.asarray(
            reference["prediction_double_sersic"][discovery], float
        ),
        **{
            name: np.asarray(archive["prediction"], float)[local]
            for name, archive in candidate_archives.items()
        },
    }
    diagnostics = {
        "double_sersic_free": {
            key: np.asarray(double_archive[key])[discovery]
            for key in (
                "success",
                "jacobian_min_ratio",
                "boundary_distance",
                "near_profile_spread_dex",
            )
        },
        **{
            name: {
                key: np.asarray(archive[key])[local]
                for key in (
                    "success",
                    "jacobian_min_ratio",
                    "boundary_distance",
                    "near_profile_spread_dex",
                )
            }
            for name, archive in candidate_archives.items()
        },
    }
    output: dict[str, object] = {
        "indices": discovery,
        "rows": np.asarray(reference["rows"][discovery], int),
        "epochs": np.asarray(reference["epochs"][discovery], int),
        "truth_log": np.asarray(reference["truth_log"][discovery], float),
        "halo_mass": np.asarray(reference["halo_mass"][discovery], float),
        "cubic_jacobian": np.asarray(
            reference["jacobian_cubic_logit"][discovery], float
        ),
        "predictions": predictions,
        "diagnostics": diagnostics,
    }
    return output


def _mass_at_radius(cogs_log: np.ndarray, radius: float) -> np.ndarray:
    profiles = np.asarray(cogs_log, float)
    return np.asarray(
        [np.interp(np.log10(radius), np.log10(RADII_KPC), row) for row in profiles]
    )


def _log_annulus_mass(
    cogs_log: np.ndarray, inner_radius: float, outer_radius: float
) -> np.ndarray:
    inner = 10.0 ** _mass_at_radius(cogs_log, inner_radius)
    outer = 10.0 ** _mass_at_radius(cogs_log, outer_radius)
    return np.log10(np.maximum(outer - inner, np.finfo(float).tiny))


def mass_quantities(cogs_log: np.ndarray) -> dict[str, np.ndarray]:
    """Return the standard fixed-kpc aperture and annulus masses."""
    return {
        "m30": _mass_at_radius(cogs_log, 30.0),
        "m30_50": _log_annulus_mass(cogs_log, 30.0, 50.0),
        "m50_100": _log_annulus_mass(cogs_log, 50.0, 100.0),
        "m100_148": _log_annulus_mass(cogs_log, 100.0, RADII_KPC[-1]),
    }


def _shell_matrix(cogs_log: np.ndarray) -> np.ndarray:
    return np.asarray([shell_log_density(row, RADII_KPC) for row in cogs_log])


def _epoch_medians(values: np.ndarray, epochs: np.ndarray) -> np.ndarray:
    return np.asarray([np.nanmedian(values[epochs == epoch]) for epoch in range(5)])


def performance_summary(data: dict[str, object]) -> dict[str, dict[str, object]]:
    truth = np.asarray(data["truth_log"])
    epochs = np.asarray(data["epochs"])
    cubic_jacobian = np.asarray(data["cubic_jacobian"])
    output = {}
    for name in MODELS:
        prediction = np.asarray(data["predictions"][name])
        diagnostic = data["diagnostics"][name]
        metrics = profile_metrics(prediction, truth)
        jacobian_ratio = np.asarray(diagnostic["jacobian_min_ratio"]) / np.maximum(
            cubic_jacobian, 1.0e-16
        )
        output[name] = {
            "full_cog_rms_dex_by_epoch": _epoch_medians(
                metrics["full_cog_rms_dex"], epochs
            ),
            "shell_density_rms_dex_by_epoch": _epoch_medians(
                metrics["shell_density_rms_dex"], epochs
            ),
            "jacobian_ratio_to_cubic_by_epoch": _epoch_medians(jacobian_ratio, epochs),
            "success_fraction": float(np.mean(diagnostic["success"])),
            "boundary_fraction": float(
                np.mean(np.asarray(diagnostic["boundary_distance"]) < 0.01)
            ),
            "maximum_near_profile_spread_dex": float(
                np.max(diagnostic["near_profile_spread_dex"])
            ),
        }
    return output


def _formula_footer(figure: plt.Figure) -> None:
    figure.text(0.01, 0.012, COMMON_WRAPPER, ha="left", va="bottom", fontsize=7.0)
    figure.text(
        0.99,
        0.012,
        "Discovery sample only; stored fits; no refitting or gate revision.",
        ha="right",
        va="bottom",
        fontsize=7.0,
        color="0.35",
    )


def _save_or_close(figure: plt.Figure, stem: str, save: bool) -> list[Path]:
    paths = save_fig(figure, FIGDIR / stem) if save else []
    plt.close(figure)
    return paths


def formula_scorecard_figure(
    data: dict[str, object], save: bool
) -> tuple[list[Path], dict[str, dict[str, object]]]:
    """Show equations, raw errors, conditioning, and failure diagnostics."""
    summary = performance_summary(data)
    figure = plt.figure(figsize=(13.0, 10.2))
    grid = figure.add_gridspec(3, 3, height_ratios=(1.55, 1.0, 1.05))
    formula_axis = figure.add_subplot(grid[0, :])
    formula_axis.axis("off")
    formula_axis.set_title(
        "Exp67 comparative QA: mathematical definitions and empirical status",
        loc="left",
        fontsize=13,
        pad=8,
    )
    y_positions = np.linspace(0.86, 0.12, len(MODELS))
    for y_position, (name, record) in zip(y_positions, MODELS.items(), strict=True):
        formula_axis.text(
            0.01,
            y_position,
            f"{record.label} ({record.n_parameter} fitted coordinates)",
            color=record.color,
            weight="bold",
            va="center",
            fontsize=9.5,
        )
        formula_axis.text(0.34, y_position, record.formula, va="center", fontsize=10)
        formula_axis.text(
            0.76,
            y_position,
            record.analytic_status,
            va="center",
            fontsize=8.2,
            color="0.25",
        )
    formula_axis.text(0.01, 0.0, COMMON_WRAPPER, va="bottom", fontsize=8)

    cog_axis = figure.add_subplot(grid[1, 0])
    density_axis = figure.add_subplot(grid[1, 1])
    jacobian_axis = figure.add_subplot(grid[1, 2])
    for name, record in MODELS.items():
        values = summary[name]
        style = dict(
            color=record.color,
            linestyle=record.linestyle,
            marker="o",
            linewidth=1.7,
            markersize=4,
            label=record.short_label,
        )
        cog_axis.plot(REDSHIFTS, values["full_cog_rms_dex_by_epoch"], **style)
        density_axis.plot(REDSHIFTS, values["shell_density_rms_dex_by_epoch"], **style)
        jacobian_axis.plot(
            REDSHIFTS, values["jacobian_ratio_to_cubic_by_epoch"], **style
        )
    cog_axis.set_title("Median full-CoG RMS")
    cog_axis.set_ylabel("RMS of log M(<R) residual [dex]")
    density_axis.set_title("Median shell-density RMS")
    density_axis.set_ylabel("RMS of log mean shell density residual [dex]")
    jacobian_axis.set_title("Local conditioning relative to cubic-logit")
    jacobian_axis.set_ylabel("Scaled minimum-Jacobian ratio")
    jacobian_axis.set_yscale("log")
    jacobian_axis.axhline(0.25, color="0.55", linestyle=":", linewidth=1)
    for axis in (cog_axis, density_axis, jacobian_axis):
        axis.set_xlabel("Redshift")
        axis.set_xticks(REDSHIFTS)
        axis.invert_xaxis()
    cog_axis.legend(fontsize=7)

    table_axis = figure.add_subplot(grid[2, :])
    table_axis.axis("off")
    columns = (
        "Model",
        "Converged fits",
        "Fits near a bound",
        "Maximum near-optimal\nCoG difference",
        "Worst-epoch CoG RMS\nrelative to double Sersic",
        "Interpretation",
    )
    double_by_epoch = np.asarray(
        summary["double_sersic_free"]["full_cog_rms_dex_by_epoch"]
    )
    interpretations = {
        "double_sersic_free": "accuracy reference",
        "free_damped_cosine": "raw-error leader; nearly null damping direction",
        "wave_packet_sine_w0p5_w0p5": "best stable compromise; rare branch ambiguity",
        "log_harmonic_sine_w4": "distinct stretched oscillation; rare severe branch",
    }
    rows = []
    for name, record in MODELS.items():
        values = summary[name]
        worst_ratio = np.max(
            np.asarray(values["full_cog_rms_dex_by_epoch"]) / double_by_epoch
        )
        rows.append(
            (
                record.short_label,
                f"{100.0 * values['success_fraction']:.3f}%",
                f"{100.0 * values['boundary_fraction']:.2f}%",
                f"{values['maximum_near_profile_spread_dex']:.5f} dex",
                f"{worst_ratio:.3f}",
                interpretations[name],
            )
        )
    table = table_axis.table(
        cellText=rows,
        colLabels=columns,
        loc="center",
        cellLoc="left",
        colLoc="left",
        colWidths=(0.14, 0.12, 0.13, 0.16, 0.16, 0.29),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.6)
    table.scale(1.0, 1.75)
    for column in range(len(columns)):
        table[(0, column)].set_facecolor("0.91")
    figure.subplots_adjust(left=0.07, right=0.98, top=0.97, bottom=0.055, hspace=0.43)
    return _save_or_close(figure, "formula_and_performance_scorecard", save), summary


def resolved_residual_figure(data: dict[str, object], save: bool) -> list[Path]:
    """Show radial CoG and independent shell-density residual structure."""
    truth = np.asarray(data["truth_log"])
    epochs = np.asarray(data["epochs"])
    truth_shell = _shell_matrix(truth)
    figure, axes = plt.subplots(
        len(MODELS), 2, figsize=(12.0, 11.0), sharex="col", sharey="col"
    )
    for row, (name, record) in enumerate(MODELS.items()):
        prediction = np.asarray(data["predictions"][name])
        cog_residual = prediction - truth
        density_residual = _shell_matrix(prediction) - truth_shell
        for epoch, (redshift, color) in enumerate(
            zip(REDSHIFTS, EPOCH_COLORS, strict=True)
        ):
            selected = epochs == epoch
            for axis, residual in zip(
                axes[row], (cog_residual, density_residual), strict=True
            ):
                median = np.median(residual[selected], axis=0)
                lower, upper = np.quantile(residual[selected], (0.16, 0.84), axis=0)
                axis.plot(RADII_KPC, median, color=color, label=rf"$z={redshift:g}$")
                axis.fill_between(RADII_KPC, lower, upper, color=color, alpha=0.07)
                axis.axhline(0.0, color="0.65", linewidth=0.8)
                axis.set_xscale("log")
        axes[row, 0].set_ylabel(f"{record.short_label}\nresidual [dex]")
        axes[row, 0].text(
            0.02,
            0.91,
            record.formula,
            transform=axes[row, 0].transAxes,
            va="top",
            fontsize=7.4,
            color=record.color,
        )
    axes[0, 0].set_title("Cumulative mass: model minus measured CoG")
    axes[0, 1].set_title("Independent annular density: model minus measured")
    axes[0, 0].legend(ncol=3, fontsize=7)
    axes[-1, 0].set_xlabel("Projected radius [kpc]")
    axes[-1, 1].set_xlabel("Outer radius of aperture or annulus [kpc]")
    figure.suptitle(
        "Exp67 resolved-profile QA: median and 16--84% residuals for 6,000 discovery profiles",
        fontsize=13,
    )
    _formula_footer(figure)
    figure.subplots_adjust(
        left=0.11, right=0.98, top=0.94, bottom=0.07, hspace=0.22, wspace=0.22
    )
    return _save_or_close(figure, "resolved_cog_and_density_residuals", save)


def _tercile_masks(
    values: np.ndarray, valid: np.ndarray | None = None
) -> list[np.ndarray]:
    finite = np.isfinite(values) if valid is None else valid & np.isfinite(values)
    edges = np.quantile(values[finite], (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0))
    masks = []
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        upper_test = values <= upper if index == 2 else values < upper
        masks.append(finite & (values >= lower) & upper_test)
    return masks


def average_cog_mass_bins_figure(data: dict[str, object], save: bool) -> list[Path]:
    """Show z=0.4 average CoG shape residuals in halo and stellar mass bins."""
    epochs = np.asarray(data["epochs"])
    selected = epochs == 0
    truth_log = np.asarray(data["truth_log"])[selected]
    truth = 10.0**truth_log
    halo_mass = np.asarray(data["halo_mass"])[selected]
    stellar_mass = truth_log[:, -1]
    masks_by_kind = {
        "Halo-mass terciles (finite-mass subset)": _tercile_masks(halo_mass),
        "Stellar-mass terciles (all discovery galaxies)": _tercile_masks(stellar_mass),
    }
    bin_colors = (OKABE_ITO[1], OKABE_ITO[2], OKABE_ITO[5])
    figure, axes = plt.subplots(
        len(MODELS), 2, figsize=(12.0, 10.5), sharex=True, sharey=True
    )
    for row, (name, record) in enumerate(MODELS.items()):
        prediction = 10.0 ** np.asarray(data["predictions"][name])[selected]
        pinned = prediction * truth[:, -1:] / prediction[:, -1:]
        fractional = 100.0 * (pinned - truth) / truth
        for column, (title, masks) in enumerate(masks_by_kind.items()):
            axis = axes[row, column]
            for bin_index, (mask, color) in enumerate(
                zip(masks, bin_colors, strict=True)
            ):
                axis.plot(
                    RADII_KPC,
                    np.median(fractional[mask], axis=0),
                    color=color,
                    label=("low", "middle", "high")[bin_index],
                )
            axis.axhline(0.0, color="0.6", linewidth=0.8)
            axis.set_xscale("log")
            if row == 0:
                axis.set_title(title)
        axes[row, 0].set_ylabel(f"{record.short_label}\nmedian shape residual [%]")
        axes[row, 0].text(
            0.02,
            0.91,
            record.formula,
            transform=axes[row, 0].transAxes,
            va="top",
            fontsize=7.2,
            color=record.color,
        )
    axes[0, 0].legend(title="tercile", ncol=3, fontsize=7, title_fontsize=7)
    for axis in axes[-1]:
        axis.set_xlabel("Projected radius [kpc]")
    figure.suptitle(
        "Exp67 z=0.4 average CoG recovery; every model is pinned to the measured 148.2-kpc mass",
        fontsize=13,
    )
    _formula_footer(figure)
    figure.subplots_adjust(
        left=0.12, right=0.98, top=0.94, bottom=0.07, hspace=0.20, wspace=0.14
    )
    return _save_or_close(figure, "average_cog_halo_and_stellar_mass_bins", save)


def _binned_statistics(
    x_values: np.ndarray, y_values: np.ndarray
) -> tuple[np.ndarray, ...]:
    valid = (
        np.isfinite(x_values)
        & np.isfinite(y_values)
        & (x_values > 5.0)
        & (y_values > 5.0)
    )
    x_values = x_values[valid]
    y_values = y_values[valid]
    edges = np.unique(np.quantile(x_values, np.linspace(0.0, 1.0, 8)))
    centers, medians, lowers, uppers = [], [], [], []
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        upper_test = x_values <= upper if index == len(edges) - 2 else x_values < upper
        selected = (x_values >= lower) & upper_test
        centers.append(np.median(x_values[selected]))
        medians.append(np.median(y_values[selected]))
        lowers.append(np.quantile(y_values[selected], 0.16))
        uppers.append(np.quantile(y_values[selected], 0.84))
    return tuple(np.asarray(item) for item in (centers, medians, lowers, uppers))


def stellar_mass_planes_figure(data: dict[str, object], save: bool) -> list[Path]:
    """Compare the three standard z=0.4 inner--outer stellar-mass planes."""
    selected = np.asarray(data["epochs"]) == 0
    profiles = {"Measured CoG": np.asarray(data["truth_log"])[selected]}
    profiles.update(
        {name: np.asarray(data["predictions"][name])[selected] for name in MODELS}
    )
    quantities = {name: mass_quantities(values) for name, values in profiles.items()}
    planes = (
        (
            "m30",
            "m30_50",
            r"$\log M_*(<30\,\mathrm{kpc})$",
            r"$\log M_*(30\!-\!50\,\mathrm{kpc})$",
        ),
        (
            "m30",
            "m50_100",
            r"$\log M_*(<30\,\mathrm{kpc})$",
            r"$\log M_*(50\!-\!100\,\mathrm{kpc})$",
        ),
        (
            "m30_50",
            "m100_148",
            r"$\log M_*(30\!-\!50\,\mathrm{kpc})$",
            r"$\log M_*(100\!-\!148.2\,\mathrm{kpc})$",
        ),
    )
    figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.8))
    for axis, (x_name, y_name, x_label, y_label) in zip(axes, planes, strict=True):
        for name, values in quantities.items():
            center, median, lower, upper = _binned_statistics(
                values[x_name], values[y_name]
            )
            if name == "Measured CoG":
                axis.fill_between(center, lower, upper, color="0.75", alpha=0.32)
                axis.plot(
                    center, median, color="black", linewidth=2.2, label="Measured CoG"
                )
            else:
                record = MODELS[name]
                axis.plot(
                    center,
                    median,
                    color=record.color,
                    linestyle=record.linestyle,
                    label=record.short_label,
                )
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
    axes[0].legend(fontsize=7)
    figure.suptitle(
        "Exp67 z=0.4 stellar-mass planes: binned medians; grey bands are the measured 16--84% width",
        fontsize=12.5,
    )
    formula_lines = "   ".join(
        f"{record.short_label}: {record.formula}" for record in MODELS.values()
    )
    figure.text(0.5, 0.075, formula_lines, ha="center", va="top", fontsize=6.6)
    _formula_footer(figure)
    figure.subplots_adjust(left=0.07, right=0.985, top=0.88, bottom=0.23, wspace=0.27)
    return _save_or_close(figure, "stellar_mass_planes_z0p4", save)


def representative_profiles_figure(data: dict[str, object], save: bool) -> list[Path]:
    """Show common best, typical, and worst profiles for every compared model."""
    truth = np.asarray(data["truth_log"])
    candidate_names = tuple(MODELS)[1:]
    candidate_rms = np.asarray(
        [
            np.sqrt(
                np.mean((np.asarray(data["predictions"][name]) - truth) ** 2, axis=1)
            )
            for name in candidate_names
        ]
    )
    shared_difficulty = np.max(candidate_rms, axis=0)
    order = np.argsort(shared_difficulty)
    chosen = (order[0], order[len(order) // 2], order[-1])
    labels = ("Best", "Typical", "Worst")
    figure, axes = plt.subplots(3, 2, figsize=(11.5, 9.5), sharex=True)
    rows = np.asarray(data["rows"])
    epochs = np.asarray(data["epochs"])
    for row, (profile_index, label) in enumerate(zip(chosen, labels, strict=True)):
        axes[row, 0].plot(
            RADII_KPC,
            truth[profile_index],
            "o",
            color="0.45",
            ms=3,
            label="Measured CoG",
        )
        axes[row, 1].axhline(0.0, color="0.6", linewidth=0.8)
        for name, record in MODELS.items():
            prediction = np.asarray(data["predictions"][name])[profile_index]
            axes[row, 0].plot(
                RADII_KPC,
                prediction,
                color=record.color,
                linestyle=record.linestyle,
                linewidth=1.5,
                label=record.short_label,
            )
            axes[row, 1].plot(
                RADII_KPC,
                prediction - truth[profile_index],
                color=record.color,
                linestyle=record.linestyle,
                linewidth=1.5,
            )
        axes[row, 0].set_ylabel(f"{label}\nlog M(<R) [dex]")
        axes[row, 1].set_ylabel("model - measured [dex]")
        axes[row, 0].set_title(
            rf"galaxy row {rows[profile_index]}, $z={REDSHIFTS[epochs[profile_index]]:g}$; "
            rf"shared difficulty={shared_difficulty[profile_index]:.4f} dex",
            fontsize=8.5,
        )
        axes[row, 1].set_title("Same object and model colors", fontsize=8.5)
        for axis in axes[row]:
            axis.set_xscale("log")
    axes[0, 0].legend(ncol=2, fontsize=7)
    for axis in axes[-1]:
        axis.set_xlabel("Projected radius [kpc]")
    figure.suptitle(
        "Exp67 common representative profiles selected by the worst RMS among the three new families",
        fontsize=12.5,
        y=0.995,
    )
    formula_rows = (
        "   ".join(
            f"{MODELS[name].short_label}: {MODELS[name].formula}"
            for name in tuple(MODELS)[:2]
        ),
        "   ".join(
            f"{MODELS[name].short_label}: {MODELS[name].formula}"
            for name in tuple(MODELS)[2:]
        ),
    )
    figure.text(0.5, 0.958, formula_rows[0], ha="center", va="top", fontsize=6.6)
    figure.text(0.5, 0.936, formula_rows[1], ha="center", va="top", fontsize=6.6)
    _formula_footer(figure)
    figure.subplots_adjust(
        left=0.11, right=0.98, top=0.88, bottom=0.075, hspace=0.30, wspace=0.20
    )
    return _save_or_close(figure, "representative_profiles", save)


def render_all(
    data: dict[str, object], save: bool
) -> tuple[list[Path], dict[str, object]]:
    paths, performance = formula_scorecard_figure(data, save)
    paths.extend(resolved_residual_figure(data, save))
    paths.extend(average_cog_mass_bins_figure(data, save))
    paths.extend(stellar_mass_planes_figure(data, save))
    paths.extend(representative_profiles_figure(data, save))
    summary = {
        "population": "Exp67 discovery only",
        "n_galaxy": int(len(np.unique(data["rows"]))),
        "n_profile": int(len(data["rows"])),
        "models": {
            name: {
                "label": MODELS[name].label,
                "formula": MODELS[name].formula,
                "n_parameter": MODELS[name].n_parameter,
                **{
                    key: value.tolist() if isinstance(value, np.ndarray) else value
                    for key, value in performance[name].items()
                },
            }
            for name in MODELS
        },
    }
    return paths, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("smoke", "full"))
    args = parser.parse_args()
    started = time.perf_counter()
    data = load_discovery_data(limit_galaxies=25 if args.mode == "smoke" else None)
    paths, summary = render_all(data, save=args.mode == "full")
    elapsed = time.perf_counter() - started
    summary["measured_wall_seconds"] = elapsed
    summary["mode"] = args.mode
    if args.mode == "smoke" and elapsed >= 60.0:
        raise RuntimeError(f"sub-minute figure gate failed in {elapsed:.2f} seconds")
    if args.mode == "full":
        OUTDIR.mkdir(parents=True, exist_ok=True)
        (OUTDIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        write_manifest(
            OUTDIR,
            {
                "stage": "exp67_discovery_comparative_qa",
                "population": "discovery_only",
                "n_galaxy": summary["n_galaxy"],
                "n_profile": summary["n_profile"],
            },
        )
    print(
        f"Exp67 comparative QA {args.mode}: {summary['n_galaxy']} galaxies, "
        f"{summary['n_profile']} profiles, {elapsed:.2f} seconds"
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
