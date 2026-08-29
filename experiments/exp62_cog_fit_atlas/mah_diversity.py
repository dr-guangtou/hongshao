# %%
"""DiffMAH diversity after endpoint-mass and half-mass-time normalization.

Commands:

    uv run python experiments/exp62_cog_fit_atlas/mah_diversity.py demo
    uv run python experiments/exp62_cog_fit_atlas/mah_diversity.py full

Each epoch uses the DiffMAH fit truncated at that epoch, so no later halo
growth enters an earlier history.
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
from matplotlib.ticker import NullFormatter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from hongshao.diffmah import log_mah  # noqa: E402
from hongshao.plotting import OKABE_ITO, save_fig, set_style  # noqa: E402
from hongshao.provenance import write_manifest  # noqa: E402
from run import FIGDIR, OUTDIR, REDSHIFTS  # noqa: E402
from shape_diversity import (  # noqa: E402
    _bootstrap_diversity,
    _diversity_statistics,
    _normalized_grid,
    _stratified_rows,
)

set_style()

SOURCE = OUTDIR / "source" / "exp51_epoch_local_diffmah.npz"
STAGE3 = OUTDIR / "full" / "stage3" / "scaling_and_evolution.npz"
T_MINIMUM_GYR = 1.0
N_DEMO = 90
N_BOOTSTRAP = {"demo": 50, "full": 200}
PRIMARY_Q_MINIMUM = 0.6
PRIMARY_Q_MAXIMUM = 1.3
INNER_Q_MINIMUMS = (0.5, 0.6, 0.7, 0.8)
OUTER_Q_MAXIMUMS = (1.2, 1.3, 1.4, 1.5)


def half_mass_times(parameters: np.ndarray, endpoint_time_gyr: float) -> np.ndarray:
    """Solve the monotonic DiffMAH curve for half its endpoint mass."""
    parameters = np.asarray(parameters, float)
    log_time_low = np.full(len(parameters), np.log10(T_MINIMUM_GYR))
    log_time_high = np.full(len(parameters), np.log10(endpoint_time_gyr))
    target = parameters[:, 0] - np.log10(2.0)

    def evaluate(log_time: np.ndarray) -> np.ndarray:
        return log_mah(
            log_time[:, None],
            parameters[:, 0],
            parameters[:, 1],
            parameters[:, 2],
            parameters[:, 3],
            np.log10(endpoint_time_gyr),
        )[:, 0]

    valid = evaluate(log_time_low) <= target
    for _ in range(60):
        midpoint = 0.5 * (log_time_low + log_time_high)
        below_target = evaluate(midpoint) < target
        log_time_low = np.where(below_target, midpoint, log_time_low)
        log_time_high = np.where(below_target, log_time_high, midpoint)
    output = np.full(len(parameters), np.nan)
    output[valid] = 10.0 ** (0.5 * (log_time_low[valid] + log_time_high[valid]))
    return output


def normalized_mahs(
    parameters: np.ndarray,
    endpoint_time_gyr: float,
    half_mass_time_gyr: np.ndarray,
    normalized_time: np.ndarray,
) -> np.ndarray:
    """Evaluate endpoint-mass-normalized histories without time extrapolation."""
    parameters = np.asarray(parameters, float)
    evaluation_time = half_mass_time_gyr[:, None] * normalized_time[None, :]
    valid = (evaluation_time >= T_MINIMUM_GYR) & (evaluation_time <= endpoint_time_gyr)
    output = (
        log_mah(
            np.log10(np.clip(evaluation_time, T_MINIMUM_GYR, endpoint_time_gyr)),
            parameters[:, 0],
            parameters[:, 1],
            parameters[:, 2],
            parameters[:, 3],
            np.log10(endpoint_time_gyr),
        )
        - parameters[:, 0, None]
    )
    output[~valid] = np.nan
    at_half_mass = np.flatnonzero(np.isclose(normalized_time, 1.0))
    if len(at_half_mass) == 1:
        measured = np.isfinite(output[:, at_half_mass[0]])
        output[measured, at_half_mass[0]] = -np.log10(2.0)
    return output


def _common_support(
    half_mass_time_gyr: np.ndarray,
    endpoint_time_gyr: np.ndarray,
    q_minimum: float,
    q_maximum: float,
) -> np.ndarray:
    supported = (q_minimum * half_mass_time_gyr >= T_MINIMUM_GYR) & (
        q_maximum * half_mass_time_gyr <= endpoint_time_gyr[None, :]
    )
    return np.all(supported, axis=1)


def _profile_figure(
    mode: str,
    parameters: np.ndarray,
    endpoint_time_gyr: np.ndarray,
    half_mass_time_gyr: np.ndarray,
) -> None:
    normalized_time = _normalized_grid(0.3, 2.5, 40)
    epoch_colors = (
        OKABE_ITO[0],
        OKABE_ITO[1],
        OKABE_ITO[2],
        OKABE_ITO[5],
        OKABE_ITO[6],
    )
    figure, axes = plt.subplots(
        2, len(REDSHIFTS), figsize=(16.0, 7.2), sharex=True, sharey="row"
    )
    for epoch, (redshift, color) in enumerate(zip(REDSHIFTS, epoch_colors)):
        histories = normalized_mahs(
            parameters[:, epoch],
            endpoint_time_gyr[epoch],
            half_mass_time_gyr[:, epoch],
            normalized_time,
        )
        coverage = np.mean(np.isfinite(histories), axis=0)
        lower = np.full(len(normalized_time), np.nan)
        median = np.full(len(normalized_time), np.nan)
        upper = np.full(len(normalized_time), np.nan)
        supported_columns = coverage > 0.0
        (
            lower[supported_columns],
            median[supported_columns],
            upper[supported_columns],
        ) = np.nanquantile(histories[:, supported_columns], [0.16, 0.5, 0.84], axis=0)
        sample_rows = _stratified_rows(
            parameters[:, epoch, 0], min(36, len(parameters))
        )
        for halo in sample_rows:
            axes[0, epoch].plot(
                normalized_time,
                histories[halo],
                color="0.72",
                alpha=0.28,
                linewidth=0.65,
            )
        axes[0, epoch].fill_between(
            normalized_time, lower, upper, color=color, alpha=0.18
        )
        axes[0, epoch].plot(normalized_time, median, color=color, linewidth=2.0)
        robust_scatter = 0.5 * (upper - lower)
        reliable = coverage >= 0.5
        axes[1, epoch].plot(
            normalized_time[reliable],
            robust_scatter[reliable],
            color=color,
            linewidth=2.0,
            label=r"Half-width of 16--84\% range",
        )
        coverage_axis = axes[1, epoch].twinx()
        coverage_axis.plot(
            normalized_time,
            coverage,
            color="0.55",
            linestyle=":",
            linewidth=1.0,
        )
        coverage_axis.set_ylim(0.0, 1.05)
        if epoch == len(REDSHIFTS) - 1:
            coverage_axis.set_ylabel("Usable fraction", color="0.45")
        else:
            coverage_axis.set_yticklabels([])
        axes[0, epoch].set_title(rf"$z={redshift:g}$; $N={len(parameters)}$")
        for row in range(2):
            axes[row, epoch].axvline(1.0, color="black", linewidth=0.8)
            axes[row, epoch].set_xscale("log")
            axes[row, epoch].set_xlim(normalized_time[0], normalized_time[-1])
            axes[row, epoch].set_xticks(
                (0.3, 0.5, 1.0, 1.5, 2.0, 2.5),
                ("0.3", "0.5", "1", "1.5", "2", "2.5"),
            )
            axes[row, epoch].xaxis.set_minor_formatter(NullFormatter())
    axes[0, 0].set_ylabel(r"$\log_{10}[M_{\rm peak}(t)/M_{\rm peak}(t_z)]$")
    axes[0, 0].set_ylim(-2.5, 0.05)
    axes[1, 0].set_ylabel("Population MAH-shape scatter (dex)")
    for axis in axes[1]:
        axis.set_xlabel(r"Normalized cosmic time $t/t_{50}$")
    axes[1, 0].legend(frameon=False, fontsize=7.0)
    figure.suptitle(
        "Exp62 epoch-local DiffMAH diversity after removing endpoint mass and t50"
    )
    figure.text(
        0.5,
        0.005,
        r"Every supported history passes through $(t/t_{50},M/M_z)=(1,0.5)$; dotted gray curves show support within each fit's 1 Gyr--epoch interval. Top panels are clipped at $-2.5$ dex for readability.",
        ha="center",
        fontsize=7.4,
        color="0.35",
    )
    figure.tight_layout(rect=(0, 0.035, 1, 1))
    save_fig(
        figure,
        FIGDIR / "visual_record" / mode / "exp62_normalized_diffmah_diversity_profiles",
    )
    plt.close(figure)


def _measure_selection(
    parameters: np.ndarray,
    endpoint_time_gyr: np.ndarray,
    half_mass_time_gyr: np.ndarray,
    q_minimum: float,
    q_maximum: float,
    mode: str,
) -> tuple[np.ndarray, int]:
    supported = _common_support(
        half_mass_time_gyr,
        endpoint_time_gyr,
        q_minimum,
        q_maximum,
    )
    rows = np.flatnonzero(supported)
    if mode == "demo":
        positions = _stratified_rows(parameters[rows, 0, 0], N_DEMO)
        rows = rows[positions]
    grid = _normalized_grid(q_minimum, q_maximum)
    diversity = np.empty((len(REDSHIFTS), 2))
    for epoch in range(len(REDSHIFTS)):
        histories = normalized_mahs(
            parameters[rows, epoch],
            endpoint_time_gyr[epoch],
            half_mass_time_gyr[rows, epoch],
            grid,
        )
        statistics = _diversity_statistics(histories, grid)
        diversity[epoch] = (
            statistics["inner_robust_diversity_dex"],
            statistics["outer_robust_diversity_dex"],
        )
    return diversity, len(rows)


def _evolution_figure(
    mode: str,
    primary: dict[str, np.ndarray],
    inner_sensitivity: dict[float, dict[str, np.ndarray | int]],
    outer_sensitivity: dict[float, dict[str, np.ndarray | int]],
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharex=True)
    styles = (
        (0, "Early: 0.6--1 t50", OKABE_ITO[0]),
        (1, "Late: 1--1.3 t50", OKABE_ITO[2]),
    )
    for region, label, color in styles:
        values = primary["diversity"][:, region]
        lower = primary["bootstrap_lower"][:, region]
        upper = primary["bootstrap_upper"][:, region]
        axes[0].errorbar(
            REDSHIFTS,
            values,
            yerr=np.vstack([values - lower, upper - values]),
            marker="o",
            color=color,
            capsize=2,
            label=label,
        )
    axes[0].set_ylabel("Robust integrated MAH diversity (dex)")
    axes[0].legend(frameon=False, fontsize=7.0)
    sensitivity_colors = (OKABE_ITO[5], OKABE_ITO[0], OKABE_ITO[2], OKABE_ITO[6])
    for q_minimum, color in zip(INNER_Q_MINIMUMS, sensitivity_colors):
        values = inner_sensitivity[q_minimum]["diversity"]
        n_halo = inner_sensitivity[q_minimum]["n_halo"]
        axes[1].plot(
            REDSHIFTS,
            values[:, 0],
            "o-",
            color=color,
            label=rf"$q_{{\min}}={q_minimum:g}$; $N={n_halo}$",
        )
    for q_maximum, color in zip(OUTER_Q_MAXIMUMS, sensitivity_colors):
        values = outer_sensitivity[q_maximum]["diversity"]
        n_halo = outer_sensitivity[q_maximum]["n_halo"]
        axes[2].plot(
            REDSHIFTS,
            values[:, 1],
            "o-",
            color=color,
            label=rf"$q_{{\max}}={q_maximum:g}$; $N={n_halo}$",
        )
    axes[1].set_ylabel("Early MAH diversity (dex)")
    axes[2].set_ylabel("Late MAH diversity (dex)")
    axes[1].legend(frameon=False, fontsize=6.5)
    axes[2].legend(frameon=False, fontsize=6.5)
    for axis in axes:
        axis.set_xlabel("Endpoint redshift")
        axis.invert_xaxis()
    axes[0].set_title("Primary common-support sample")
    axes[1].set_title("Early-time-range sensitivity")
    axes[2].set_title("Late-time-range sensitivity")
    figure.suptitle(
        "Exp62 redshift evolution of endpoint-mass- and t50-normalized DiffMAH diversity"
    )
    figure.text(
        0.5,
        0.005,
        r"Diversity is the RMS across normalized time of half the population 16--84\% width; every line uses the same halos at all endpoints.",
        ha="center",
        fontsize=7.4,
        color="0.35",
    )
    figure.tight_layout(rect=(0, 0.035, 1, 1))
    save_fig(
        figure,
        FIGDIR
        / "visual_record"
        / mode
        / "exp62_normalized_diffmah_diversity_evolution",
    )
    plt.close(figure)


def run(mode: str) -> None:
    start = time.perf_counter()
    source = np.load(SOURCE)
    stage3 = np.load(STAGE3)
    parameters = np.asarray(source["parameters"], float)
    endpoint_time_gyr = np.asarray(stage3["cosmic_time_gyr"], float)
    half_mass_time_gyr = np.stack(
        [
            half_mass_times(parameters[:, epoch], endpoint_time_gyr[epoch])
            for epoch in range(len(REDSHIFTS))
        ],
        axis=1,
    )
    if not np.all(np.isfinite(half_mass_time_gyr)):
        raise ValueError("some half-mass times lie before the 1 Gyr fit boundary")

    primary_supported = _common_support(
        half_mass_time_gyr,
        endpoint_time_gyr,
        PRIMARY_Q_MINIMUM,
        PRIMARY_Q_MAXIMUM,
    )
    primary_rows = np.flatnonzero(primary_supported)
    if mode == "demo":
        positions = _stratified_rows(parameters[primary_rows, 0, 0], N_DEMO)
        primary_rows = primary_rows[positions]
        display_rows = primary_rows
    else:
        display_rows = np.arange(len(parameters))
    _profile_figure(
        mode,
        parameters[display_rows],
        endpoint_time_gyr,
        half_mass_time_gyr[display_rows],
    )

    primary_grid = _normalized_grid(PRIMARY_Q_MINIMUM, PRIMARY_Q_MAXIMUM)
    primary_histories = np.stack(
        [
            normalized_mahs(
                parameters[primary_rows, epoch],
                endpoint_time_gyr[epoch],
                half_mass_time_gyr[primary_rows, epoch],
                primary_grid,
            )
            for epoch in range(len(REDSHIFTS))
        ],
        axis=1,
    )
    if not np.all(np.isfinite(primary_histories)):
        raise ValueError("primary common-support histories contain missing values")
    primary_diversity = np.empty((len(REDSHIFTS), 2))
    primary_standard = np.empty_like(primary_diversity)
    bootstrap = np.empty((len(REDSHIFTS), N_BOOTSTRAP[mode], 2))
    for epoch in range(len(REDSHIFTS)):
        statistics = _diversity_statistics(primary_histories[:, epoch], primary_grid)
        primary_diversity[epoch] = (
            statistics["inner_robust_diversity_dex"],
            statistics["outer_robust_diversity_dex"],
        )
        primary_standard[epoch] = (
            statistics["inner_standard_diversity_dex"],
            statistics["outer_standard_diversity_dex"],
        )
        bootstrap[epoch] = _bootstrap_diversity(
            primary_histories[:, epoch],
            primary_grid,
            N_BOOTSTRAP[mode],
            6362 + epoch,
        )
    primary = {
        "diversity": primary_diversity,
        "standard": primary_standard,
        "bootstrap_lower": np.quantile(bootstrap, 0.025, axis=1),
        "bootstrap_upper": np.quantile(bootstrap, 0.975, axis=1),
    }

    inner_sensitivity = {}
    for q_minimum in INNER_Q_MINIMUMS:
        diversity, n_halo = _measure_selection(
            parameters,
            endpoint_time_gyr,
            half_mass_time_gyr,
            q_minimum,
            PRIMARY_Q_MAXIMUM,
            mode,
        )
        inner_sensitivity[q_minimum] = {
            "diversity": diversity,
            "n_halo": n_halo,
        }
    outer_sensitivity = {}
    for q_maximum in OUTER_Q_MAXIMUMS:
        diversity, n_halo = _measure_selection(
            parameters,
            endpoint_time_gyr,
            half_mass_time_gyr,
            PRIMARY_Q_MINIMUM,
            q_maximum,
            mode,
        )
        outer_sensitivity[q_maximum] = {
            "diversity": diversity,
            "n_halo": n_halo,
        }
    _evolution_figure(mode, primary, inner_sensitivity, outer_sensitivity)

    summary = {
        "mode": mode,
        "definition": "RMS across log-spaced normalized time of half the 16--84% population width in log10[Mpeak(t)/Mpeak(endpoint)]",
        "fit_time_minimum_gyr": T_MINIMUM_GYR,
        "primary_normalized_time_range": [
            PRIMARY_Q_MINIMUM,
            PRIMARY_Q_MAXIMUM,
        ],
        "primary_n_halo": len(primary_rows),
        "redshifts": REDSHIFTS.tolist(),
        "endpoint_time_gyr": endpoint_time_gyr.tolist(),
        "half_mass_time_median_gyr": np.median(half_mass_time_gyr, axis=0).tolist(),
        "early_robust_diversity_dex": primary_diversity[:, 0].tolist(),
        "late_robust_diversity_dex": primary_diversity[:, 1].tolist(),
        "early_standard_diversity_dex": primary_standard[:, 0].tolist(),
        "late_standard_diversity_dex": primary_standard[:, 1].tolist(),
        "bootstrap_95_lower_dex": primary["bootstrap_lower"].tolist(),
        "bootstrap_95_upper_dex": primary["bootstrap_upper"].tolist(),
        "inner_sensitivity_n_halo": {
            str(q_minimum): int(inner_sensitivity[q_minimum]["n_halo"])
            for q_minimum in INNER_Q_MINIMUMS
        },
        "outer_sensitivity_n_halo": {
            str(q_maximum): int(outer_sensitivity[q_maximum]["n_halo"])
            for q_maximum in OUTER_Q_MAXIMUMS
        },
        "wall_seconds": time.perf_counter() - start,
    }
    summary["subminute_gate_passed"] = (
        bool(summary["wall_seconds"] < 60.0) if mode == "demo" else None
    )
    output_directory = OUTDIR / mode / "mah_diversity"
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_manifest(
        output_directory,
        {
            "experiment": "exp62_cog_fit_atlas",
            "stage": "quick_normalized_diffmah_diversity",
            "mode": mode,
            "n_halo": len(primary_rows),
        },
    )
    print(json.dumps(summary, indent=2))
    if mode == "demo" and summary["wall_seconds"] >= 60.0:
        raise RuntimeError(
            f"DiffMAH-diversity demo took {summary['wall_seconds']:.2f} seconds"
        )


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if command not in ("demo", "full"):
        raise SystemExit(__doc__)
    run(command)
