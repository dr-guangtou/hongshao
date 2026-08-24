"""exp55 stage 2 — compact Sersic plus power-law-tailed Moffat envelope."""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from hongshao.plotting import OKABE_ITO, save_fig, set_style  # noqa: E402
from hongshao.profile_emulator import density_from_cog  # noqa: E402
from hongshao.provenance import write_manifest  # noqa: E402
from model import sersic_fraction  # noqa: E402
from run import (  # noqa: E402
    FIGDIR,
    OUTDIR,
    RADII,
    load_sample,
    outer_density_slopes,
    representative_indices,
)

set_style()

SERSIC_INDICES = np.array([1.0, 2.0, 4.0])
MOFFAT_INDICES = np.array([1.1, 1.25, 1.4, 1.7])
LOWER = np.array([8.0, -5.0, np.log10(0.3), np.log(0.03)])
UPPER = np.array([14.0, 5.0, np.log10(300.0), np.log(100.0)])


def label(n_sersic: float, gamma: float) -> str:
    return f"sersic{n_sersic:g}_moffat{gamma:g}".replace(".", "p")


def moffat_fraction(radius: np.ndarray, r50: float, gamma: float) -> np.ndarray:
    half_scale = np.sqrt(2.0 ** (1.0 / (gamma - 1.0)) - 1.0)
    r_core = r50 / half_scale
    return 1.0 - (1.0 + (radius / r_core) ** 2) ** (1.0 - gamma)


def mixed_cog_log(
    parameters: np.ndarray,
    radius: np.ndarray,
    n_sersic: float,
    gamma: float,
) -> np.ndarray:
    """Four-parameter CoG with component fraction defined within 148 kpc."""
    compact_mass, extended_mass = mixed_component_cogs(
        parameters, radius, n_sersic, gamma
    )
    return np.log10(np.clip(compact_mass + extended_mass, 1e-300, None))


def mixed_component_cogs(
    parameters: np.ndarray,
    radius: np.ndarray,
    n_sersic: float,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return compact and extended cumulative masses in physical solar masses."""
    log_mass, logit_fraction, log_r_compact, log_ratio_minus_one = parameters
    compact_fraction = expit(logit_fraction)
    r_compact = 10.0**log_r_compact
    r_extended = r_compact * (1.0 + np.exp(log_ratio_minus_one))
    compact = sersic_fraction(radius, r_compact, n_sersic)
    extended = moffat_fraction(radius, r_extended, gamma)
    compact /= compact[-1]
    extended /= extended[-1]
    total_mass = 10.0**log_mass
    return (
        total_mass * compact_fraction * compact,
        total_mass * (1.0 - compact_fraction) * extended,
    )


def observed_r50(cog_log_values: np.ndarray, radius: np.ndarray = RADII) -> float:
    cumulative = 10.0**cog_log_values
    target = 0.5 * cumulative[-1]
    if target <= cumulative[0]:
        slope = np.diff(np.log(cumulative[:2]))[0] / np.diff(np.log(radius[:2]))[0]
        value = radius[0] * np.exp(
            (np.log(target) - np.log(cumulative[0])) / max(slope, 1e-3)
        )
        return float(np.clip(value, 0.3, radius[-1]))
    return float(np.interp(target, cumulative, radius))


def starts_for(truth_log: np.ndarray, radius: np.ndarray = RADII) -> list[np.ndarray]:
    amplitude = truth_log[-1]
    r50 = observed_r50(truth_log, radius)
    starts = []
    for fraction, compact_factor, ratio in (
        (0.15, 0.15, 8.0),
        (0.5, 0.35, 3.0),
        (0.85, 0.7, 1.5),
        (0.5, 0.08, 20.0),
    ):
        starts.append(
            np.array(
                [
                    amplitude,
                    np.log(fraction / (1.0 - fraction)),
                    np.log10(np.clip(compact_factor * r50, 0.3, 250.0)),
                    np.log(ratio - 1.0),
                ]
            )
        )
    return [np.clip(start, LOWER + 1e-8, UPPER - 1e-8) for start in starts]


def fit_one(
    truth_log: np.ndarray,
    n_sersic: float,
    gamma: float,
    radius: np.ndarray = RADII,
) -> dict:
    def residual(parameters: np.ndarray) -> np.ndarray:
        return mixed_cog_log(parameters, radius, n_sersic, gamma) - truth_log

    solutions = [
        least_squares(
            residual,
            start,
            bounds=(LOWER, UPPER),
            method="trf",
            x_scale="jac",
            max_nfev=2000,
        )
        for start in starts_for(truth_log, radius)
    ]
    best = min(solutions, key=lambda solution: np.sum(solution.fun**2))
    prediction = mixed_cog_log(best.x, radius, n_sersic, gamma)
    scaled_jacobian = best.jac * (UPPER - LOWER)[None, :]
    singular_values = np.linalg.svd(scaled_jacobian, compute_uv=False)
    positions = (best.x - LOWER) / (UPPER - LOWER)
    return {
        "parameters": best.x,
        "prediction": prediction,
        "rms_dex": np.sqrt(np.mean((prediction - truth_log) ** 2)),
        "jacobian_min_ratio": singular_values[-1] / singular_values[0],
        "boundary_distance": np.min(np.minimum(positions, 1.0 - positions)),
        "success": best.success,
    }


def checkpoint_path(n_sersic: float, gamma: float) -> Path:
    return OUTDIR / "mixed_grid" / f"{label(n_sersic, gamma)}.npz"


def fit_cell(cogs_log: np.ndarray, n_sersic: float, gamma: float) -> dict:
    path = checkpoint_path(n_sersic, gamma)
    if path.exists():
        return dict(np.load(path))
    arrays = {
        "parameters": np.empty((len(cogs_log), 4)),
        "prediction": np.empty_like(cogs_log),
        "rms_dex": np.empty(len(cogs_log)),
        "jacobian_min_ratio": np.empty(len(cogs_log)),
        "boundary_distance": np.empty(len(cogs_log)),
        "success": np.empty(len(cogs_log), dtype=bool),
    }
    started = time.perf_counter()
    for galaxy, truth_log in enumerate(cogs_log):
        result = fit_one(truth_log, n_sersic, gamma)
        for name in arrays:
            arrays[name][galaxy] = result[name]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    print(
        f"  {label(n_sersic, gamma)}: {len(cogs_log)} galaxies in "
        f"{time.perf_counter() - started:.1f} s",
        flush=True,
    )
    return arrays


def folds_of(n_galaxy: int) -> list[np.ndarray]:
    order = np.random.default_rng(0).permutation(n_galaxy)
    return list(np.array_split(order, 5))


def density_rms(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    model_density, _ = density_from_cog(prediction, RADII)
    truth_density, _ = density_from_cog(truth, RADII)
    return np.sqrt(np.mean((model_density - truth_density) ** 2, axis=1))


def paired_median_comparison(
    candidate_values: np.ndarray, reference_values: np.ndarray, seed: int
) -> dict:
    difference = np.asarray(candidate_values) - np.asarray(reference_values)
    rng = np.random.default_rng(seed)
    bootstrap = np.median(
        difference[rng.integers(0, len(difference), size=(1000, len(difference)))],
        axis=1,
    )
    return {
        "median_candidate_minus_reference": float(np.median(difference)),
        "bootstrap_16_84": np.quantile(bootstrap, [0.16, 0.84]).tolist(),
        "fraction_candidate_better": float(np.mean(difference < 0.0)),
    }


def choose_by_fold(
    results: dict[str, dict], truth: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    prediction = np.empty_like(truth)
    selected = []
    all_indices = np.arange(len(truth))
    for fold in folds_of(len(truth)):
        train = np.setdiff1d(all_indices, fold)
        winner = min(
            results, key=lambda name: np.median(results[name]["rms_dex"][train])
        )
        prediction[fold] = results[winner]["prediction"][fold]
        selected.append(winner)
    return prediction, selected


def grid_figure(
    truth: np.ndarray,
    results: dict[str, dict],
    cv_prediction: np.ndarray,
    selected: list[str],
    best_name: str,
) -> None:
    cog_grid = np.empty((len(SERSIC_INDICES), len(MOFFAT_INDICES)))
    density_grid = np.empty_like(cog_grid)
    condition_grid = np.empty_like(cog_grid)
    for row, n_sersic in enumerate(SERSIC_INDICES):
        for column, gamma in enumerate(MOFFAT_INDICES):
            name = label(n_sersic, gamma)
            cog_grid[row, column] = np.median(results[name]["rms_dex"])
            density_grid[row, column] = np.median(
                density_rms(results[name]["prediction"], truth)
            )
            condition_grid[row, column] = np.median(
                -np.log10(np.clip(results[name]["jacobian_min_ratio"], 1e-16, None))
            )

    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.0))
    for axis, values, title, colorbar_label in (
        (axes[0, 0], cog_grid, "Cumulative-profile accuracy", "Median CoG RMS (dex)"),
        (
            axes[0, 1],
            density_grid,
            "Differential-profile accuracy",
            "Median density RMS (dex)",
        ),
        (
            axes[0, 2],
            condition_grid,
            "Local parameter degeneracy",
            r"Median $-\log_{10}(s_{\min}/s_{\max})$",
        ),
    ):
        image = axis.imshow(values, cmap="cividis", aspect="auto", origin="lower")
        axis.set_xticks(
            range(len(MOFFAT_INDICES)), [f"{value:g}" for value in MOFFAT_INDICES]
        )
        axis.set_yticks(
            range(len(SERSIC_INDICES)), [f"{value:g}" for value in SERSIC_INDICES]
        )
        axis.set(
            xlabel=r"Extended Moffat $\gamma$",
            ylabel=r"Compact Sersic $n$",
            title=title,
        )
        fig.colorbar(image, ax=axis, label=colorbar_label)

    counts = Counter(selected)
    names = sorted(counts)
    axes[1, 0].barh(
        [name.replace("_", " + ") for name in names],
        [counts[name] for name in names],
        color=OKABE_ITO[2],
    )
    axes[1, 0].set(
        xlabel="Held-out folds out of five",
        title="Shape pair selected on training galaxies",
    )

    base = np.load(OUTDIR / "representation.npz")
    comparisons = {
        "PCA-3": base["reconstruction_pca3"],
        "five coordinates": base["reconstruction_five_coordinates"],
        "radial sigmoid": base["reconstruction_radial_sigmoid"],
        "fold-selected mixture": cv_prediction,
    }
    colors = ["black", OKABE_ITO[0], OKABE_ITO[4], OKABE_ITO[2]]
    styles = [":", "-.", "-", "--"]
    for (name, prediction), color, style in zip(comparisons.items(), colors, styles):
        axes[1, 1].plot(
            RADII,
            np.median(np.abs(prediction - truth), axis=0),
            style,
            color=color,
            label=name,
        )
    axes[1, 1].set(
        xscale="log",
        yscale="log",
        xlabel="Radius (kpc)",
        ylabel="Median absolute CoG error (dex)",
        title="Full radial comparison",
    )
    axes[1, 1].legend(fontsize=7)

    for name, result in results.items():
        axes[1, 2].scatter(
            np.median(-np.log10(np.clip(result["jacobian_min_ratio"], 1e-16, None))),
            np.median(result["rms_dex"]),
            color=OKABE_ITO[6],
            alpha=0.75,
            s=35,
        )
    best = results[best_name]
    axes[1, 2].scatter(
        np.median(-np.log10(np.clip(best["jacobian_min_ratio"], 1e-16, None))),
        np.median(best["rms_dex"]),
        color="black",
        marker="*",
        s=100,
        label=best_name.replace("_", " + "),
    )
    axes[1, 2].set(
        xlabel=r"Median $-\log_{10}(s_{\min}/s_{\max})$",
        ylabel="Median CoG RMS (dex)",
        title="Accuracy and conditioning",
    )
    axes[1, 2].legend(fontsize=7)

    for panel, axis in zip("ABCDEF", axes.ravel()):
        axis.text(
            -0.14, 1.04, panel, transform=axis.transAxes, fontweight="bold", fontsize=11
        )
    fig.suptitle("exp55 — compact Sersic plus extended Moffat grid")
    fig.tight_layout()
    save_fig(fig, FIGDIR / "exp55_mixed_grid")
    plt.close(fig)


def gallery_figure(
    truth: np.ndarray, best_prediction: np.ndarray, best_name: str
) -> None:
    base = np.load(OUTDIR / "representation.npz")
    radial = base["reconstruction_radial_sigmoid"]
    five = base["reconstruction_five_coordinates"]
    galaxies = representative_indices(truth)
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.8))
    for column, galaxy in enumerate(galaxies):
        for name, prediction, color, style in (
            ("TNG", truth, "black", "-"),
            ("five coordinates", five, OKABE_ITO[0], "-."),
            ("radial sigmoid", radial, OKABE_ITO[4], "-"),
            (best_name.replace("_", " + "), best_prediction, OKABE_ITO[2], "--"),
        ):
            normalized = prediction[galaxy] - prediction[galaxy, -1]
            axes[0, column].plot(
                RADII,
                normalized,
                style,
                color=color,
                marker="o" if name == "TNG" else None,
                ms=3,
                label=name,
            )
        density_truth, middle = density_from_cog(truth[galaxy : galaxy + 1], RADII)
        for name, prediction, color, style in (
            ("radial sigmoid", radial, OKABE_ITO[4], "-"),
            (best_name.replace("_", " + "), best_prediction, OKABE_ITO[2], "--"),
        ):
            density_model, _ = density_from_cog(prediction[galaxy : galaxy + 1], RADII)
            axes[1, column].plot(
                middle,
                (density_model - density_truth)[0],
                style,
                color=color,
                label=name,
            )
        central_fraction = 10.0 ** (truth[galaxy, 0] - truth[galaxy, -1])
        axes[0, column].set(
            xscale="log",
            xlabel="Radius (kpc)",
            title=rf"$M_*(<2)/M_{{*,148}}={central_fraction:.2f}$",
        )
        axes[1, column].axhline(0.0, color="0.5", linestyle=":")
        axes[1, column].set(xscale="log", xlabel="Annulus midpoint (kpc)")
    axes[0, 0].set_ylabel(r"$\log_{10}[M_*(<R)/M_{*,148}]$")
    axes[1, 0].set_ylabel(r"Model $-$ TNG $\log_{10}\Sigma_*$ (dex)")
    axes[0, 0].legend(fontsize=7)
    axes[1, 0].legend(fontsize=7)
    for panel, axis in zip("ABCDEF", axes.ravel()):
        axis.text(
            -0.14, 1.04, panel, transform=axis.transAxes, fontweight="bold", fontsize=11
        )
    fig.suptitle("exp55 — direct cumulative and differential profile check")
    fig.tight_layout()
    save_fig(fig, FIGDIR / "exp55_mixed_profiles")
    plt.close(fig)


def demo() -> None:
    cogs_log, _ = load_sample(8)
    for n_sersic, gamma in ((1.0, 1.1), (2.0, 1.4), (4.0, 1.7)):
        result = fit_one(cogs_log[0], n_sersic, gamma)
        assert result["success"]
        synthetic = mixed_cog_log(result["parameters"], RADII, n_sersic, gamma)
        recovery = fit_one(synthetic, n_sersic, gamma)
        assert recovery["rms_dex"] < 1e-6
    print("exp55 mixed-family demo OK")


def run() -> None:
    cogs_log, indices = load_sample()
    results = {}
    print(
        f"exp55 mixed-family grid: n={len(cogs_log)}, cells={len(SERSIC_INDICES) * len(MOFFAT_INDICES)}"
    )
    for n_sersic in SERSIC_INDICES:
        for gamma in MOFFAT_INDICES:
            name = label(n_sersic, gamma)
            results[name] = fit_cell(cogs_log, n_sersic, gamma)

    cv_prediction, selected = choose_by_fold(results, cogs_log)
    best_name = min(results, key=lambda name: np.median(results[name]["rms_dex"]))
    best = results[best_name]
    base = np.load(OUTDIR / "representation.npz")
    candidate_cog_rms = np.sqrt(np.mean((cv_prediction - cogs_log) ** 2, axis=1))
    candidate_density_rms = density_rms(cv_prediction, cogs_log)
    references = {
        "pca3": base["reconstruction_pca3"],
        "five_coordinates": base["reconstruction_five_coordinates"],
        "radial_sigmoid": base["reconstruction_radial_sigmoid"],
    }
    summary = {
        "n_galaxy": len(cogs_log),
        "grid": {
            name: {
                "median_cog_rms_dex": float(np.median(result["rms_dex"])),
                "median_density_rms_dex": float(
                    np.median(density_rms(result["prediction"], cogs_log))
                ),
                "median_log10_jacobian_min_ratio": float(
                    np.median(
                        np.log10(np.clip(result["jacobian_min_ratio"], 1e-16, None))
                    )
                ),
                "fraction_within_one_percent_of_bound": float(
                    np.mean(result["boundary_distance"] < 0.01)
                ),
                "median_absolute_outer_slope_error": float(
                    np.median(
                        np.abs(
                            outer_density_slopes(result["prediction"])
                            - outer_density_slopes(cogs_log)
                        )
                    )
                ),
            }
            for name, result in results.items()
        },
        "best_full_sample_shape_pair": best_name,
        "fold_selected_shape_pairs": selected,
        "fold_selected_median_cog_rms_dex": float(
            np.median(np.sqrt(np.mean((cv_prediction - cogs_log) ** 2, axis=1)))
        ),
        "fold_selected_median_density_rms_dex": float(np.median(candidate_density_rms)),
        "paired_comparisons": {
            reference_name: {
                "cog_rms_dex": paired_median_comparison(
                    candidate_cog_rms,
                    np.sqrt(np.mean((reference_prediction - cogs_log) ** 2, axis=1)),
                    seed=100 + reference_index,
                ),
                "density_rms_dex": paired_median_comparison(
                    candidate_density_rms,
                    density_rms(reference_prediction, cogs_log),
                    seed=200 + reference_index,
                ),
            }
            for reference_index, (reference_name, reference_prediction) in enumerate(
                references.items()
            )
        },
    }
    mixed_dir = OUTDIR / "mixed_grid"
    (mixed_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    np.savez(
        mixed_dir / "selected.npz",
        truth_log=cogs_log,
        indices=indices,
        cv_prediction=cv_prediction,
        best_prediction=best["prediction"],
        best_parameters=best["parameters"],
        selected=np.asarray(selected),
    )
    grid_figure(cogs_log, results, cv_prediction, selected, best_name)
    gallery_figure(cogs_log, cv_prediction, best_name)
    write_manifest(
        mixed_dir,
        {
            "experiment": "exp55_analytic_cog_mixed_grid",
            "n_galaxy": len(cogs_log),
            "sersic_indices": SERSIC_INDICES.tolist(),
            "moffat_indices": MOFFAT_INDICES.tolist(),
            "best_full_sample_shape_pair": best_name,
            "fold_selected_shape_pairs": selected,
        },
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if command == "demo":
        demo()
    elif command == "run":
        run()
    else:
        raise SystemExit(__doc__)
