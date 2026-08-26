"""Illustrate the two Exp56 Stage 3b profile recipes and their fitted outputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit, gammainc, gammaincinv

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP55 = ROOT / "experiments" / "exp55_analytic_cog"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP55))
sys.path.insert(0, str(HERE))

import stage3_halo_cog as stage3  # noqa: E402
from hongshao.plotting import OKABE_ITO, save_fig, set_style  # noqa: E402
from hongshao.profile_emulator import density_from_cog  # noqa: E402

set_style()
matplotlib.rcParams["text.usetex"] = False

STAGE3_DIRECTORY = HERE / "outputs" / "stage3"
STAGE3B_DIRECTORY = HERE / "outputs" / "stage3b"
FIGURE_DIRECTORY = HERE / "figures" / "stage3b"


def smooth_gamma(log_mass: np.ndarray, transition: float) -> np.ndarray:
    argument = np.clip((np.asarray(log_mass) - transition) / 0.20, -50.0, 50.0)
    return 1.25 + 0.15 / (1.0 + np.exp(argument))


def component_fractions(
    raw_parameters: np.ndarray,
    gamma: float,
    radius: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    compact_fraction = float(expit(raw_parameters[1]))
    compact_radius = 10.0 ** raw_parameters[2]
    extended_radius = compact_radius * (1.0 + np.exp(raw_parameters[3]))
    sersic_constant = gammaincinv(2.0, 0.5)
    compact = gammainc(2.0, sersic_constant * radius / compact_radius)
    compact /= compact[-1]
    half_scale = np.sqrt(2.0 ** (1.0 / (gamma - 1.0)) - 1.0)
    core_radius = extended_radius / half_scale
    extended = 1.0 - (1.0 + (radius / core_radius) ** 2) ** (1.0 - gamma)
    extended /= extended[-1]
    return compact_fraction * compact, (1.0 - compact_fraction) * extended


def physical_parameter_intervals(raw_parameters: np.ndarray) -> dict[str, np.ndarray]:
    physical = stage3.physical_parameters(raw_parameters)
    return {
        "log_mass": np.quantile(physical[:, 0], (0.16, 0.50, 0.84)),
        "compact_fraction": np.quantile(physical[:, 1], (0.16, 0.50, 0.84)),
        "compact_radius": 10.0 ** np.quantile(physical[:, 2], (0.16, 0.50, 0.84)),
        "extended_radius": 10.0 ** np.quantile(physical[:, 3], (0.16, 0.50, 0.84)),
    }


def representative_trade_galaxy(
    truth_log: np.ndarray,
    coordinate_log: np.ndarray,
    direct_log: np.ndarray,
) -> tuple[int, int]:
    coordinate_density, _ = density_from_cog(coordinate_log, stage3.stage_1.RADII)
    direct_density, _ = density_from_cog(direct_log, stage3.stage_1.RADII)
    truth_density, _ = density_from_cog(truth_log, stage3.stage_1.RADII)
    coordinate_cog_rms = np.sqrt(np.mean((coordinate_log - truth_log) ** 2, axis=1))
    direct_cog_rms = np.sqrt(np.mean((direct_log - truth_log) ** 2, axis=1))
    coordinate_density_rms = np.sqrt(
        np.mean((coordinate_density - truth_density) ** 2, axis=1)
    )
    direct_density_rms = np.sqrt(np.mean((direct_density - truth_density) ** 2, axis=1))
    cog_change = direct_cog_rms - coordinate_cog_rms
    density_change = direct_density_rms - coordinate_density_rms
    trade = np.flatnonzero((cog_change < 0.0) & (density_change > 0.0))
    normalized_cog = (cog_change[trade] - np.median(cog_change[trade])) / np.std(
        cog_change[trade]
    )
    normalized_density = (
        density_change[trade] - np.median(density_change[trade])
    ) / np.std(density_change[trade])
    selected = int(trade[np.argmin(normalized_cog**2 + normalized_density**2)])
    return selected, len(trade)


def interval_text(values: np.ndarray, digits: int = 3) -> str:
    low, median, high = values
    return f"{median:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def build_figure() -> None:
    coordinate = np.load(STAGE3_DIRECTORY / "predictions.npz")
    direct = np.load(STAGE3B_DIRECTORY / "predictions.npz")
    result = json.loads((STAGE3B_DIRECTORY / "results.json").read_text())
    if not np.array_equal(coordinate["indices"], direct["indices"]):
        raise RuntimeError("Stage 3 and Stage 3b galaxy indices differ")

    truth_log = direct["truth_log"]
    features = direct["features"]
    halo_mass = features[:, 0]
    all_rows = np.arange(len(halo_mass))
    transitions = []
    for test in stage3.folds_of(len(halo_mass)):
        train = np.setdiff1d(all_rows, test)
        transitions.append(float(np.quantile(halo_mass[train], 2.0 / 3.0)))
    transitions = np.asarray(transitions)

    fixed_intervals = physical_parameter_intervals(
        direct["fixed_gamma1p4_mean_parameters"]
    )
    smooth_intervals = physical_parameter_intervals(direct["smooth_mean_parameters"])
    upper_mass_rows = np.array_split(np.argsort(halo_mass), 3)[-1]
    fixed_upper_raw = np.median(
        direct["fixed_gamma1p4_mean_parameters"][upper_mass_rows], axis=0
    )
    smooth_upper_raw = np.median(
        direct["smooth_mean_parameters"][upper_mass_rows], axis=0
    )
    fixed_upper_physical = stage3.physical_parameters(fixed_upper_raw[None])[0]
    smooth_upper_physical = stage3.physical_parameters(smooth_upper_raw[None])[0]
    smooth_upper_gamma = float(np.median(direct["smooth_gamma"][upper_mass_rows]))

    trade_row, n_trade = representative_trade_galaxy(
        truth_log,
        coordinate["fixed_gamma1p4_mean_log"],
        direct["fixed_gamma1p4_mean_log"],
    )

    fixed_color = OKABE_ITO[1]
    smooth_color = OKABE_ITO[2]
    coordinate_color = "0.35"
    direct_color = OKABE_ITO[2]
    figure = plt.figure(figsize=(17.0, 9.4))
    grid = figure.add_gridspec(
        2,
        3,
        width_ratios=(1.0, 1.0, 1.28),
        hspace=0.32,
        wspace=0.30,
    )
    gamma_axis = figure.add_subplot(grid[0, 0])
    component_axis = figure.add_subplot(grid[0, 1])
    cog_axis = figure.add_subplot(grid[1, 0])
    density_axis = figure.add_subplot(grid[1, 1])
    recipe_axis = figure.add_subplot(grid[:, 2])
    recipe_axis.axis("off")

    mass_grid = np.linspace(np.min(halo_mass), np.max(halo_mass), 500)
    gamma_axis.plot(
        mass_grid,
        np.full_like(mass_grid, 1.4),
        color=fixed_color,
        linewidth=2.5,
        label=r"fixed $\gamma=1.4$",
    )
    for transition in transitions:
        gamma_axis.plot(
            mass_grid,
            smooth_gamma(mass_grid, transition),
            color=smooth_color,
            alpha=0.24,
            linewidth=1.2,
        )
    gamma_axis.plot(
        mass_grid,
        smooth_gamma(mass_grid, float(np.median(transitions))),
        color=smooth_color,
        linewidth=2.5,
        label=r"smooth $\gamma(\log M_{\rm peak})$",
    )
    gamma_axis.axvspan(
        float(np.min(transitions)),
        float(np.max(transitions)),
        color="0.75",
        alpha=0.45,
        label="five fold transitions",
    )
    gamma_axis.set(
        xlabel=r"DiffMAH $\log_{10}(M_{\rm peak}/M_\odot)$",
        ylabel=r"Moffat outer index $\gamma$",
        ylim=(1.235, 1.415),
        title="Only the outer-slope prescription differs globally",
    )
    gamma_axis.legend(frameon=False, fontsize=8, loc="lower left")

    dense_radius = np.geomspace(stage3.stage_1.RADII[0], stage3.stage_1.RADII[-1], 300)
    fixed_compact, fixed_extended = component_fractions(
        fixed_upper_raw, 1.4, dense_radius
    )
    smooth_compact, smooth_extended = component_fractions(
        smooth_upper_raw, smooth_upper_gamma, dense_radius
    )
    for color, compact, extended, label in (
        (fixed_color, fixed_compact, fixed_extended, r"fixed $\gamma$"),
        (smooth_color, smooth_compact, smooth_extended, r"smooth $\gamma$"),
    ):
        component_axis.plot(
            dense_radius,
            compact + extended,
            color=color,
            linewidth=2.4,
            label=f"{label}: total",
        )
        component_axis.plot(
            dense_radius,
            extended,
            color=color,
            linestyle="--",
            linewidth=1.8,
            label=f"{label}: extended",
        )
        component_axis.plot(
            dense_radius,
            compact,
            color=color,
            linestyle=":",
            linewidth=1.5,
            label=f"{label}: compact",
        )
    component_axis.set(
        xscale="log",
        yscale="log",
        xlabel="Radius (kpc)",
        ylabel=r"Component contribution to $M_*(<R)/M_{*,148}$",
        ylim=(0.015, 1.08),
        title="Upper halo-mass tercile: median predicted profile",
    )
    component_axis.text(
        0.02,
        0.97,
        (
            rf"fixed: $(f_c,R_c,R_e,\gamma)=({fixed_upper_physical[1]:.3f},"
            rf"{10.0 ** fixed_upper_physical[2]:.2f},"
            rf"{10.0 ** fixed_upper_physical[3]:.1f},1.400)$"
            "\n"
            rf"smooth: $({smooth_upper_physical[1]:.3f},"
            rf"{10.0 ** smooth_upper_physical[2]:.2f},"
            rf"{10.0 ** smooth_upper_physical[3]:.1f},{smooth_upper_gamma:.3f})$"
        ),
        transform=component_axis.transAxes,
        fontsize=7.6,
        va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78},
    )
    component_axis.legend(frameon=False, fontsize=7.4, ncol=2, loc="lower right")

    truth = truth_log[trade_row]
    coordinate_profile = coordinate["fixed_gamma1p4_mean_log"][trade_row]
    direct_profile = direct["fixed_gamma1p4_mean_log"][trade_row]
    for profile, color, style, label in (
        (coordinate_profile, coordinate_color, "--", "coordinate-trained mean"),
        (direct_profile, direct_color, "-", "direct-CoG-trained mean"),
    ):
        pinned = (profile - truth) - (profile[-1] - truth[-1])
        cog_axis.plot(
            stage3.stage_1.RADII,
            100.0 * (10.0**pinned - 1.0),
            color=color,
            linestyle=style,
            linewidth=2.0,
            label=label,
        )
    cog_axis.axhline(0.0, color="0.55", linewidth=0.8)
    cog_axis.axvspan(5.0, 30.0, color="0.85", alpha=0.45)
    cog_axis.set(
        xscale="log",
        xlabel="Radius (kpc)",
        ylabel="Amplitude-pinned CoG residual (%)",
        title=(
            "One representative held-out trade: cumulative profile\n"
            rf"catalog row {int(direct['indices'][trade_row])}, "
            rf"$\log M_{{\rm peak}}={halo_mass[trade_row]:.2f}$"
        ),
    )
    cog_axis.legend(frameon=False, fontsize=8)

    truth_density, middle_radius = density_from_cog(
        truth[None], stage3.stage_1.RADII
    )
    coordinate_density, _ = density_from_cog(
        coordinate_profile[None], stage3.stage_1.RADII
    )
    direct_density, _ = density_from_cog(
        direct_profile[None], stage3.stage_1.RADII
    )
    density_axis.plot(
        middle_radius,
        (coordinate_density - truth_density)[0],
        color=coordinate_color,
        linestyle="--",
        linewidth=2.0,
        label="coordinate-trained mean",
    )
    density_axis.plot(
        middle_radius,
        (direct_density - truth_density)[0],
        color=direct_color,
        linewidth=2.0,
        label="direct-CoG-trained mean",
    )
    density_axis.axhline(0.0, color="0.55", linewidth=0.8)
    density_axis.axvspan(5.0, 30.0, color="0.85", alpha=0.45)
    density_axis.set(
        xscale="log",
        xlabel="Annulus midpoint (kpc)",
        ylabel=r"$\log_{10}(\Sigma_{\rm model}/\Sigma_{\rm TNG})$ (dex)",
        title=(
            "The same galaxy after adjacent cumulative masses are differenced\n"
            f"{n_trade}/{len(truth_log)} galaxies ({100.0 * n_trade / len(truth_log):.1f}%) "
            "show this CoG-better / density-worse trade"
        ),
    )
    density_axis.legend(frameon=False, fontsize=8)

    recipe_axis.text(
        0.0,
        0.985,
        "Shared analytic profile",
        transform=recipe_axis.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
    )
    recipe_axis.text(
        0.0,
        0.942,
        (
            r"$M_*(<R)=M_{*,148}\,[f_c\,S_1(R;R_c)+(1-f_c)\,E_\gamma(R;R_e)]$"
            "\n"
            r"$S_1= P(2,b_1R/R_c)/P(2,b_1R_{148}/R_c),\quad b_1=1.678$"
            "\n"
            r"$E_\gamma=\dfrac{1-[1+(R/r_0)^2]^{1-\gamma}}"
            r"{1-[1+(R_{148}/r_0)^2]^{1-\gamma}},\quad "
            r"r_0=\dfrac{R_e}{\sqrt{2^{1/(\gamma-1)}-1}}$"
        ),
        transform=recipe_axis.transAxes,
        fontsize=10.1,
        va="top",
        linespacing=1.65,
    )
    recipe_axis.text(
        0.0,
        0.805,
        "Fold-local halo–CoG relation",
        transform=recipe_axis.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
    )
    recipe_axis.text(
        0.0,
        0.765,
        (
            r"$\widehat{\theta}_k=\beta^{(f)}_{k0}+\boldsymbol{\beta}^{(f)}_k\!\cdot\!\tilde{\mathbf{x}}"
            r"+\sum_{(a,b)\in\mathcal{Q}}\beta^{(f)}_{kab}\tilde{x}_a\tilde{x}_b$"
            "\n"
            r"$\theta=(\log M_{*,148},\,\mathrm{logit}\,f_c,\,\log R_c,\,\ln[R_e/R_c-1])$"
            "\n"
            r"$\mathbf{x}=(\log M_{\rm peak},\log t_c,\mathrm{early},\mathrm{late},c_{200c},\gamma)$"
            "\n"
            r"$\mathcal{Q}=$ {(0,0), (1,3), (2,2), (3,4), (3,3), (0,2), (0,3)};"
            " five outer-fold fits"
            "\n"
            r"$\mathcal{L}_{\rm train}=\sum_g\sum_{j=1}^{24}[\log M_{\rm model}(<R_j)-"
            r"\log M_{\rm TNG}(<R_j)]^2$"
        ),
        transform=recipe_axis.transAxes,
        fontsize=9.7,
        va="top",
        linespacing=1.5,
    )

    recipe_axis.text(
        0.0,
        0.565,
        "Global and fold-fitted values",
        transform=recipe_axis.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
    )
    fixed_scales = [
        item["selected_scale"]
        for item in result["fold_metadata"]["fixed_gamma1p4"]
    ]
    smooth_scales = [
        item["selected_scale"] for item in result["fold_metadata"]["smooth"]
    ]
    table_rows = [
        ["Quantity", r"fixed $\gamma$", r"smooth $\gamma(m)$"],
        [r"Sérsic $n$", "1.00", "1.00"],
        [r"Moffat $\gamma$", "1.400", r"1.40 $\rightarrow$ 1.25"],
        [r"logistic width", "—", "0.20 dex"],
        [
            r"transition $\log M_{\rm peak}$",
            "—",
            f"{np.median(transitions):.3f} "
            f"[{np.min(transitions):.3f}, {np.max(transitions):.3f}]",
        ],
        ["active mean coefficients / fold", "52", "56"],
        [
            "residual scale across folds",
            f"{min(fixed_scales):.2f}–{max(fixed_scales):.2f}",
            f"{min(smooth_scales):.2f}–{max(smooth_scales):.2f}",
        ],
    ]
    table = recipe_axis.table(
        cellText=table_rows[1:],
        colLabels=table_rows[0],
        cellLoc="left",
        colLoc="left",
        bbox=(0.0, 0.365, 1.0, 0.175),
        colWidths=(0.46, 0.22, 0.32),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.6)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("0.82")
        if row == 0:
            cell.set_facecolor("0.92")
            cell.set_text_props(fontweight="bold")

    recipe_axis.text(
        0.0,
        0.337,
        "Held-out predicted coordinates: median [16th, 84th percentile]",
        transform=recipe_axis.transAxes,
        fontsize=10.5,
        fontweight="bold",
        va="top",
    )
    coordinate_rows = [
        ["Coordinate", r"fixed $\gamma$", r"smooth $\gamma(m)$"],
        [
            r"$\log M_{*,148}$",
            interval_text(fixed_intervals["log_mass"]),
            interval_text(smooth_intervals["log_mass"]),
        ],
        [
            r"$f_c$",
            interval_text(fixed_intervals["compact_fraction"]),
            interval_text(smooth_intervals["compact_fraction"]),
        ],
        [
            r"$R_c$ (kpc)",
            interval_text(fixed_intervals["compact_radius"], 2),
            interval_text(smooth_intervals["compact_radius"], 2),
        ],
        [
            r"$R_e$ (kpc)",
            interval_text(fixed_intervals["extended_radius"], 1),
            interval_text(smooth_intervals["extended_radius"], 1),
        ],
    ]
    coordinate_table = recipe_axis.table(
        cellText=coordinate_rows[1:],
        colLabels=coordinate_rows[0],
        cellLoc="left",
        colLoc="left",
        bbox=(0.0, 0.188, 1.0, 0.135),
        colWidths=(0.29, 0.35, 0.36),
    )
    coordinate_table.auto_set_font_size(False)
    coordinate_table.set_fontsize(8.4)
    for (row, _), cell in coordinate_table.get_celld().items():
        cell.set_edgecolor("0.82")
        if row == 0:
            cell.set_facecolor("0.92")
            cell.set_text_props(fontweight="bold")

    fixed_summary = result["direct_representations"]["fixed_gamma1p4"]
    smooth_summary = result["direct_representations"]["smooth"]
    fixed_r90 = result["standard_qa"]["fixed_gamma1p4"]["sizes"]["R90"]["model"]
    smooth_r90 = result["standard_qa"]["smooth"]["sizes"]["R90"]["model"]
    recipe_axis.text(
        0.0,
        0.160,
        "Held-out performance (model versus TNG)",
        transform=recipe_axis.transAxes,
        fontsize=10.5,
        fontweight="bold",
        va="top",
    )
    performance_rows = [
        ["Metric", r"fixed $\gamma$", r"smooth $\gamma(m)$"],
        [
            "median full-CoG RMS (dex)",
            f"{fixed_summary['point']['median_profile_rms_dex']:.5f}",
            f"{smooth_summary['point']['median_profile_rms_dex']:.5f}",
        ],
        [
            "median density RMS (dex)",
            f"{fixed_summary['point']['median_density_rms_dex']:.5f}",
            f"{smooth_summary['point']['median_density_rms_dex']:.5f}",
        ],
        [
            "worst halo-bin shape residual",
            f"{fixed_summary['point']['worst_mass_bin_shape_residual_percent']:.2f}%",
            f"{smooth_summary['point']['worst_mass_bin_shape_residual_percent']:.2f}%",
        ],
        [
            r"median $\Delta\log R_{90}$ (dex)",
            f"{fixed_r90['median_dlogR']:+.3f}",
            f"{smooth_r90['median_dlogR']:+.3f}",
        ],
    ]
    performance_table = recipe_axis.table(
        cellText=performance_rows[1:],
        colLabels=performance_rows[0],
        cellLoc="left",
        colLoc="left",
        bbox=(0.0, 0.012, 1.0, 0.125),
        colWidths=(0.54, 0.22, 0.24),
    )
    performance_table.auto_set_font_size(False)
    performance_table.set_fontsize(8.4)
    for (row, _), cell in performance_table.get_celld().items():
        cell.set_edgecolor("0.82")
        if row == 0:
            cell.set_facecolor("0.92")
            cell.set_text_props(fontweight="bold")

    for panel, axis in zip("ABCD", (gamma_axis, component_axis, cog_axis, density_axis)):
        axis.text(
            -0.14,
            1.05,
            panel,
            transform=axis.transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
        )
    recipe_axis.text(
        -0.04,
        1.005,
        "E",
        transform=recipe_axis.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
    )
    figure.suptitle(
        "Exp56 Stage 3b — two direct-fit halo–CoG recipes and why cumulative loss is not the whole profile",
        fontsize=15,
        y=0.995,
    )
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    save_fig(figure, FIGURE_DIRECTORY / "exp56_stage3b_model_recipes")
    plt.close(figure)


if __name__ == "__main__":
    build_figure()
