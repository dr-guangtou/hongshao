# %%
"""Quick test of CoG-shape diversity after stellar-mass and size normalization.

Commands:

    uv run python experiments/exp62_cog_fit_atlas/shape_diversity.py demo
    uv run python experiments/exp62_cog_fit_atlas/shape_diversity.py full

The comparison uses measured TNG CoGs only. It does not fit or decode an
analytic profile family.
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

from hongshao import qa  # noqa: E402
from hongshao.plotting import OKABE_ITO, save_fig, set_style  # noqa: E402
from hongshao.provenance import write_manifest  # noqa: E402
from run import FIGDIR, OUTDIR, RADII, REDSHIFTS  # noqa: E402

set_style()

N_DEMO = 90
N_BOOTSTRAP = {"demo": 50, "full": 200}
PRIMARY_Q_MINIMUM = 0.7
PRIMARY_Q_MAXIMUM = 3.0
SENSITIVITY_Q_MINIMUMS = (0.5, 0.7, 0.8, 0.9)


def _cube(atlas: np.lib.npyio.NpzFile, key: str) -> tuple[np.ndarray, np.ndarray]:
    galaxy_ids, compact_rows = np.unique(
        np.asarray(atlas["rows"], int), return_inverse=True
    )
    epochs = np.asarray(atlas["epochs"], int)
    values = np.asarray(atlas[key])
    output = np.full((len(galaxy_ids), len(REDSHIFTS), *values.shape[1:]), np.nan)
    output[compact_rows, epochs] = values
    if not np.all(np.isfinite(output)):
        raise ValueError(f"{key} does not cover the complete galaxy--epoch cube")
    return galaxy_ids, output


def normalized_cogs(
    cogs_log: np.ndarray, r50: np.ndarray, normalized_radius: np.ndarray
) -> np.ndarray:
    """Interpolate log mass fractions without extrapolating the measured CoG."""
    cogs_log = np.asarray(cogs_log, float)
    r50 = np.asarray(r50, float)
    normalized_radius = np.asarray(normalized_radius, float)
    output = np.full((len(cogs_log), len(normalized_radius)), np.nan)
    fractions = 10.0 ** (cogs_log - cogs_log[:, -1, None])
    for galaxy in range(len(cogs_log)):
        evaluation_radius = normalized_radius * r50[galaxy]
        valid = (evaluation_radius >= RADII[0]) & (evaluation_radius <= RADII[-1])
        output[galaxy, valid] = np.log10(
            np.clip(
                np.interp(evaluation_radius[valid], RADII, fractions[galaxy]),
                1.0e-12,
                None,
            )
        )
    at_r50 = np.flatnonzero(np.isclose(normalized_radius, 1.0))
    if len(at_r50) == 1:
        measured = np.isfinite(output[:, at_r50[0]])
        output[measured, at_r50[0]] = np.log10(0.5)
    return output


def _normalized_grid(q_minimum: float, q_maximum: float, n_side: int = 32):
    inner = np.geomspace(q_minimum, 1.0, n_side, endpoint=False)
    outer = np.geomspace(1.0, q_maximum, n_side + 1)
    return np.concatenate([inner, outer])


def _common_support(r50: np.ndarray, q_minimum: float, q_maximum: float):
    supported = (q_minimum * r50 >= RADII[0]) & (q_maximum * r50 <= RADII[-1])
    return np.all(supported, axis=1)


def _stratified_rows(values: np.ndarray, n_row: int) -> np.ndarray:
    order = np.argsort(values)
    positions = np.linspace(0, len(order) - 1, n_row).round().astype(int)
    return order[positions]


def _diversity_statistics(
    profiles: np.ndarray, normalized_radius: np.ndarray
) -> dict[str, np.ndarray | float]:
    lower, median, upper = np.quantile(profiles, [0.16, 0.5, 0.84], axis=0)
    robust_scatter = 0.5 * (upper - lower)
    standard_scatter = np.std(profiles - median[None, :], axis=0)
    inner = normalized_radius < 1.0
    outer = normalized_radius > 1.0
    return {
        "median": median,
        "robust_scatter": robust_scatter,
        "standard_scatter": standard_scatter,
        "inner_robust_diversity_dex": float(
            np.sqrt(np.mean(robust_scatter[inner] ** 2))
        ),
        "outer_robust_diversity_dex": float(
            np.sqrt(np.mean(robust_scatter[outer] ** 2))
        ),
        "inner_standard_diversity_dex": float(
            np.sqrt(np.mean(standard_scatter[inner] ** 2))
        ),
        "outer_standard_diversity_dex": float(
            np.sqrt(np.mean(standard_scatter[outer] ** 2))
        ),
    }


def _bootstrap_diversity(
    profiles: np.ndarray,
    normalized_radius: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    output = np.empty((n_bootstrap, 2))
    for bootstrap in range(n_bootstrap):
        rows = rng.integers(0, len(profiles), len(profiles))
        statistics = _diversity_statistics(profiles[rows], normalized_radius)
        output[bootstrap] = (
            statistics["inner_robust_diversity_dex"],
            statistics["outer_robust_diversity_dex"],
        )
    return output


def _profile_figure(
    mode: str,
    cogs_log: np.ndarray,
    r50: np.ndarray,
    galaxy_ids: np.ndarray,
) -> None:
    normalized_radius = _normalized_grid(0.3, 5.0, 40)
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
        profiles = normalized_cogs(cogs_log[:, epoch], r50[:, epoch], normalized_radius)
        coverage = np.mean(np.isfinite(profiles), axis=0)
        lower, median, upper = np.nanquantile(profiles, [0.16, 0.5, 0.84], axis=0)
        sample_rows = _stratified_rows(cogs_log[:, epoch, -1], min(36, len(cogs_log)))
        for galaxy in sample_rows:
            axes[0, epoch].plot(
                normalized_radius,
                profiles[galaxy],
                color="0.72",
                alpha=0.28,
                linewidth=0.65,
            )
        axes[0, epoch].fill_between(
            normalized_radius, lower, upper, color=color, alpha=0.18
        )
        axes[0, epoch].plot(normalized_radius, median, color=color, linewidth=2.0)
        robust_scatter = 0.5 * (upper - lower)
        reliable = coverage >= 0.5
        axes[1, epoch].plot(
            normalized_radius[reliable],
            robust_scatter[reliable],
            color=color,
            linewidth=2.0,
            label=r"Half-width of 16--84\% range",
        )
        coverage_axis = axes[1, epoch].twinx()
        coverage_axis.plot(
            normalized_radius,
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
        axes[0, epoch].set_title(rf"$z={redshift:g}$; $N={len(galaxy_ids)}$")
        for row in range(2):
            axes[row, epoch].axvline(1.0, color="black", linewidth=0.8)
            axes[row, epoch].set_xscale("log")
            axes[row, epoch].set_xlim(normalized_radius[0], normalized_radius[-1])
            axes[row, epoch].set_xticks(
                (0.3, 0.5, 1.0, 2.0, 3.0, 5.0),
                ("0.3", "0.5", "1", "2", "3", "5"),
            )
            axes[row, epoch].xaxis.set_minor_formatter(NullFormatter())
    axes[0, 0].set_ylabel(r"$\log_{10}[M_*(<R)/M_{*,148}]$")
    axes[1, 0].set_ylabel("Population shape scatter (dex)")
    for axis in axes[1]:
        axis.set_xlabel(r"Normalized radius $R/R_{50}$")
    axes[1, 0].legend(frameon=False, fontsize=7.0)
    figure.suptitle(
        "Exp62 measured CoG diversity after removing total stellar mass and R50"
    )
    figure.text(
        0.5,
        0.005,
        r"Every usable profile passes through $(R/R_{50},M_*/M_{*,148})=(1,0.5)$; dotted gray curves show the fraction supported by the measured 2--148 kpc grid.",
        ha="center",
        fontsize=7.4,
        color="0.35",
    )
    figure.tight_layout(rect=(0, 0.035, 1, 1))
    save_fig(
        figure,
        FIGDIR / "visual_record" / mode / "exp62_normalized_cog_diversity_profiles",
    )
    plt.close(figure)


def _evolution_figure(
    mode: str,
    primary: dict[str, np.ndarray],
    sensitivity: dict[float, dict[str, np.ndarray | int]],
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharex=True)
    styles = (
        (0, "Inner: 0.7--1 R50", OKABE_ITO[0]),
        (1, "Outer: 1--3 R50", OKABE_ITO[2]),
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
    axes[0].set_ylabel("Robust integrated shape diversity (dex)")
    axes[0].legend(frameon=False, fontsize=7.0)
    sensitivity_colors = (OKABE_ITO[5], OKABE_ITO[0], OKABE_ITO[2], OKABE_ITO[6])
    for q_minimum, color in zip(SENSITIVITY_Q_MINIMUMS, sensitivity_colors):
        values = sensitivity[q_minimum]["diversity"]
        n_galaxy = sensitivity[q_minimum]["n_galaxy"]
        axes[1].plot(
            REDSHIFTS,
            values[:, 0],
            "o-",
            color=color,
            label=rf"$q_{{\min}}={q_minimum:g}$; $N={n_galaxy}$",
        )
        axes[2].plot(
            REDSHIFTS,
            values[:, 1],
            "o-",
            color=color,
            label=rf"$q_{{\min}}={q_minimum:g}$; $N={n_galaxy}$",
        )
    axes[1].set_ylabel("Inner shape diversity (dex)")
    axes[2].set_ylabel("Outer shape diversity (dex)")
    axes[1].legend(frameon=False, fontsize=6.5)
    axes[2].legend(frameon=False, fontsize=6.5)
    for axis in axes:
        axis.set_xlabel("Redshift")
        axis.invert_xaxis()
    axes[0].set_title("Primary common-support sample")
    axes[1].set_title("Inner-range and selection sensitivity")
    axes[2].set_title("Outer robustness to the same selections")
    figure.suptitle(
        "Exp62 redshift evolution of mass- and R50-normalized CoG diversity"
    )
    figure.text(
        0.5,
        0.005,
        r"Diversity is the RMS across normalized radius of half the population 16--84\% width; every line uses the same galaxies at all epochs.",
        ha="center",
        fontsize=7.4,
        color="0.35",
    )
    figure.tight_layout(rect=(0, 0.035, 1, 1))
    save_fig(
        figure,
        FIGDIR / "visual_record" / mode / "exp62_normalized_cog_diversity_evolution",
    )
    plt.close(figure)


def run(mode: str) -> None:
    start = time.perf_counter()
    atlas = np.load(OUTDIR / "full" / "atlas" / "predictions.npz")
    galaxy_ids, cogs_log = _cube(atlas, "truth_log")
    r50 = qa._safe_rhalf(10.0**cogs_log, RADII, 0.5)
    primary_supported = _common_support(r50, PRIMARY_Q_MINIMUM, PRIMARY_Q_MAXIMUM)
    selected = np.flatnonzero(primary_supported)
    if mode == "demo":
        demo_positions = _stratified_rows(cogs_log[selected, 0, -1], N_DEMO)
        selected = selected[demo_positions]
        display_rows = selected
    else:
        display_rows = np.arange(len(galaxy_ids))
    _profile_figure(
        mode,
        cogs_log[display_rows],
        r50[display_rows],
        galaxy_ids[display_rows],
    )
    galaxy_ids = galaxy_ids[selected]
    cogs_log = cogs_log[selected]
    r50 = r50[selected]

    primary_grid = _normalized_grid(PRIMARY_Q_MINIMUM, PRIMARY_Q_MAXIMUM)
    primary_profiles = np.stack(
        [
            normalized_cogs(cogs_log[:, epoch], r50[:, epoch], primary_grid)
            for epoch in range(len(REDSHIFTS))
        ],
        axis=1,
    )
    if not np.all(np.isfinite(primary_profiles)):
        raise ValueError("primary common-support profiles contain missing values")
    primary_diversity = np.empty((len(REDSHIFTS), 2))
    primary_standard = np.empty_like(primary_diversity)
    bootstrap = np.empty((len(REDSHIFTS), N_BOOTSTRAP[mode], 2))
    for epoch in range(len(REDSHIFTS)):
        statistics = _diversity_statistics(primary_profiles[:, epoch], primary_grid)
        primary_diversity[epoch] = (
            statistics["inner_robust_diversity_dex"],
            statistics["outer_robust_diversity_dex"],
        )
        primary_standard[epoch] = (
            statistics["inner_standard_diversity_dex"],
            statistics["outer_standard_diversity_dex"],
        )
        bootstrap[epoch] = _bootstrap_diversity(
            primary_profiles[:, epoch],
            primary_grid,
            N_BOOTSTRAP[mode],
            6262 + epoch,
        )
    primary = {
        "diversity": primary_diversity,
        "standard": primary_standard,
        "bootstrap_lower": np.quantile(bootstrap, 0.025, axis=1),
        "bootstrap_upper": np.quantile(bootstrap, 0.975, axis=1),
    }

    full_atlas = np.load(OUTDIR / "full" / "atlas" / "predictions.npz")
    _, all_cogs_log = _cube(full_atlas, "truth_log")
    all_r50 = qa._safe_rhalf(10.0**all_cogs_log, RADII, 0.5)
    sensitivity = {}
    for q_minimum in SENSITIVITY_Q_MINIMUMS:
        supported = _common_support(all_r50, q_minimum, PRIMARY_Q_MAXIMUM)
        rows = np.flatnonzero(supported)
        if mode == "demo":
            positions = _stratified_rows(all_cogs_log[rows, 0, -1], N_DEMO)
            rows = rows[positions]
        grid = _normalized_grid(q_minimum, PRIMARY_Q_MAXIMUM)
        diversity = np.empty((len(REDSHIFTS), 2))
        for epoch in range(len(REDSHIFTS)):
            profiles = normalized_cogs(
                all_cogs_log[rows, epoch], all_r50[rows, epoch], grid
            )
            statistics = _diversity_statistics(profiles, grid)
            diversity[epoch] = (
                statistics["inner_robust_diversity_dex"],
                statistics["outer_robust_diversity_dex"],
            )
        sensitivity[q_minimum] = {
            "n_galaxy": len(rows),
            "diversity": diversity,
        }
    _evolution_figure(mode, primary, sensitivity)

    summary = {
        "mode": mode,
        "definition": "RMS across log-spaced normalized radius of half the 16--84% population width in log10[M(<R)/M148]",
        "primary_normalized_radius_range": [
            PRIMARY_Q_MINIMUM,
            PRIMARY_Q_MAXIMUM,
        ],
        "primary_n_galaxy": len(cogs_log),
        "redshifts": REDSHIFTS.tolist(),
        "inner_robust_diversity_dex": primary_diversity[:, 0].tolist(),
        "outer_robust_diversity_dex": primary_diversity[:, 1].tolist(),
        "inner_standard_diversity_dex": primary_standard[:, 0].tolist(),
        "outer_standard_diversity_dex": primary_standard[:, 1].tolist(),
        "bootstrap_95_lower_dex": primary["bootstrap_lower"].tolist(),
        "bootstrap_95_upper_dex": primary["bootstrap_upper"].tolist(),
        "sensitivity_n_galaxy": {
            str(q_minimum): int(sensitivity[q_minimum]["n_galaxy"])
            for q_minimum in SENSITIVITY_Q_MINIMUMS
        },
        "wall_seconds": time.perf_counter() - start,
    }
    summary["subminute_gate_passed"] = (
        bool(summary["wall_seconds"] < 60.0) if mode == "demo" else None
    )
    output_directory = OUTDIR / mode / "shape_diversity"
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_manifest(
        output_directory,
        {
            "experiment": "exp62_cog_fit_atlas",
            "stage": "quick_normalized_cog_shape_diversity",
            "mode": mode,
            "n_galaxy": len(cogs_log),
        },
    )
    print(json.dumps(summary, indent=2))
    if mode == "demo" and summary["wall_seconds"] >= 60.0:
        raise RuntimeError(
            f"shape-diversity demo took {summary['wall_seconds']:.2f} seconds"
        )


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if command not in ("demo", "full"):
        raise SystemExit(__doc__)
    run(command)
