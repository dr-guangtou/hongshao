"""exp55 — analytic CoG representation and identifiability shootout."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from scipy.special import expit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from hongshao.plotting import OKABE_ITO, save_fig, set_style  # noqa: E402
from hongshao.profile_emulator import density_from_cog  # noqa: E402
from hongshao.provenance import write_manifest  # noqa: E402
from hongshao.tng_data import COG_RAD_KPC  # noqa: E402
from model import (  # noqa: E402
    ANALYTIC_FAMILIES,
    FAMILY_SPECS,
    MIXTURE_FAMILIES,
    ONE_COMPONENT_FAMILIES,
    demo as model_demo,
    fit_cog,
    profile_scan,
)

set_style()

FIGDIR = HERE / "figures"
OUTDIR = HERE / "outputs"
RADII = np.asarray(COG_RAD_KPC, float)
N_MAX = int(os.environ.get("EXP55_NMAX", 0))


def source_table() -> Path:
    """Resolve the existing processed table without writing outside Exp55."""
    local = ROOT / "data" / "processed" / "tng300_072_z0p4.fits"
    if local.exists():
        return local
    source_root = Path(
        os.environ.get("EXP55_SOURCE_ROOT", ROOT.parent / "hongshao")
    ).expanduser()
    fallback = source_root / "data" / "processed" / local.name
    if not fallback.exists():
        raise FileNotFoundError(
            f"missing {local.name}; set EXP55_SOURCE_ROOT to a checkout containing it"
        )
    return fallback


def _stratified_indices(cogs_log: np.ndarray, n_max: int) -> np.ndarray:
    """Select across total mass and central fraction, not mass-ordered rows."""
    if not n_max or n_max >= len(cogs_log):
        return np.arange(len(cogs_log))
    total = cogs_log[:, -1]
    central_fraction = 10.0 ** (cogs_log[:, 0] - total)
    mass_rank = np.argsort(np.argsort(total)) / max(len(total) - 1, 1)
    compact_rank = np.argsort(np.argsort(central_fraction)) / max(len(total) - 1, 1)
    target_side = int(np.ceil(np.sqrt(n_max)))
    targets = np.array(
        [
            (mass_position, compact_position)
            for mass_position in np.linspace(0.0, 1.0, target_side)
            for compact_position in np.linspace(0.0, 1.0, target_side)
        ]
    )
    available = np.ones(len(cogs_log), dtype=bool)
    selected = []
    for target in targets:
        distance = (mass_rank - target[0]) ** 2 + (compact_rank - target[1]) ** 2
        distance[~available] = np.inf
        choice = int(np.argmin(distance))
        selected.append(choice)
        available[choice] = False
        if len(selected) == n_max:
            break
    return np.asarray(selected, int)


def load_sample(n_max: int = 0) -> tuple[np.ndarray, np.ndarray]:
    table = Table.read(source_table())
    table = table[table["use"]]
    cogs_log = np.asarray(table["logmstar_cog"], float)
    indices = np.asarray(table["index"], int)
    finite = np.isfinite(cogs_log).all(axis=1)
    monotone = np.all(np.diff(cogs_log, axis=1) >= -1e-10, axis=1)
    cogs_log, indices = cogs_log[finite & monotone], indices[finite & monotone]
    selected = _stratified_indices(cogs_log, n_max)
    return cogs_log[selected], indices[selected]


def exp50_model():
    path = ROOT / "experiments" / "exp50_direct_cog_map" / "model.py"
    spec = importlib.util.spec_from_file_location("exp50_coordinate_model", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def pca_reconstruction(cogs_log: np.ndarray, n_component: int) -> np.ndarray:
    amplitude = cogs_log[:, -1]
    shape = cogs_log[:, :-1] - amplitude[:, None]
    mean_shape = np.mean(shape, axis=0)
    _, _, modes = np.linalg.svd(shape - mean_shape, full_matrices=False)
    basis = modes[:n_component]
    reconstructed = (shape - mean_shape) @ basis.T @ basis + mean_shape
    return np.column_stack([reconstructed + amplitude[:, None], amplitude])


def reference_reconstructions(cogs_log: np.ndarray) -> dict[str, np.ndarray]:
    coordinate_model = exp50_model()
    parameters = coordinate_model.compress_cogs(cogs_log, RADII, "quantile3")
    return {
        "pca3": pca_reconstruction(cogs_log, 3),
        "pca5": pca_reconstruction(cogs_log, 5),
        "five_coordinates": coordinate_model.reconstruct_cogs(
            parameters, RADII, "quantile3"
        ),
    }


def family_output_path(family: str, n_galaxy: int) -> Path:
    suffix = "full" if not N_MAX else f"n{n_galaxy}"
    return OUTDIR / f"{family}_{suffix}.npz"


def fit_family(family: str, cogs_log: np.ndarray) -> dict[str, np.ndarray]:
    n_galaxy = len(cogs_log)
    n_parameter = FAMILY_SPECS[family].n_parameter
    arrays = {
        "parameters": np.empty((n_galaxy, n_parameter)),
        "prediction": np.empty_like(cogs_log),
        "rms_dex": np.empty(n_galaxy),
        "success": np.empty(n_galaxy, dtype=bool),
        "nfev": np.empty(n_galaxy, dtype=int),
        "jacobian_min_ratio": np.empty(n_galaxy),
        "boundary_distance": np.empty(n_galaxy),
        "near_solution_count": np.empty(n_galaxy, dtype=int),
        "near_parameter_spread": np.empty(n_galaxy),
        "near_profile_spread_dex": np.empty(n_galaxy),
    }
    started = time.perf_counter()
    for galaxy, truth in enumerate(cogs_log):
        result = fit_cog(family, RADII, truth)
        for key in arrays:
            arrays[key][galaxy] = result[key]
        if (galaxy + 1) % 100 == 0 or galaxy + 1 == n_galaxy:
            elapsed = time.perf_counter() - started
            print(
                f"  {family}: {galaxy + 1}/{n_galaxy} galaxies in {elapsed:.1f} s",
                flush=True,
            )
    OUTDIR.mkdir(parents=True, exist_ok=True)
    np.savez(family_output_path(family, n_galaxy), **arrays)
    return arrays


def annular_masses(cogs_log: np.ndarray) -> np.ndarray:
    cumulative_log = np.column_stack(
        [
            [np.interp(radius, RADII, profile) for profile in cogs_log]
            for radius in (10.0, 30.0, 50.0, 100.0)
        ]
    )
    cumulative = 10.0**cumulative_log
    return np.column_stack(
        [
            cumulative_log[:, 0],
            np.log10(np.clip(cumulative[:, 1] - cumulative[:, 0], 1.0, None)),
            np.log10(np.clip(cumulative[:, 2] - cumulative[:, 1], 1.0, None)),
            np.log10(np.clip(cumulative[:, 3] - cumulative[:, 2], 1.0, None)),
        ]
    )


def outer_density_slopes(cogs_log: np.ndarray) -> np.ndarray:
    density, middle_radius = density_from_cog(cogs_log, RADII)
    outer = middle_radius >= 50.0
    design = np.column_stack(
        [np.ones(np.count_nonzero(outer)), np.log10(middle_radius[outer])]
    )
    coefficient = np.linalg.lstsq(design, density[:, outer].T, rcond=None)[0]
    return coefficient[1]


def bootstrap_median_interval(values: np.ndarray, seed: int = 0) -> list[float]:
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    bootstrap = np.median(
        values[rng.integers(0, len(values), size=(500, len(values)))], axis=1
    )
    return np.quantile(bootstrap, [0.16, 0.84]).tolist()


def summarize(
    name: str,
    prediction: np.ndarray,
    truth: np.ndarray,
    fit_result: dict[str, np.ndarray] | None = None,
) -> dict:
    error = prediction - truth
    per_galaxy_rms = np.sqrt(np.mean(error**2, axis=1))
    density_prediction, _ = density_from_cog(prediction, RADII)
    density_truth, _ = density_from_cog(truth, RADII)
    per_galaxy_density_rms = np.sqrt(
        np.mean((density_prediction - density_truth) ** 2, axis=1)
    )
    fractional = np.abs((10.0**prediction - 10.0**truth) / 10.0**truth)
    outside = RADII > 5.0
    annulus_error = annular_masses(prediction) - annular_masses(truth)
    slope_error = outer_density_slopes(prediction) - outer_density_slopes(truth)
    summary = {
        "name": name,
        "n_parameter": int(FAMILY_SPECS[name].n_parameter)
        if name in FAMILY_SPECS
        else None,
        "median_cog_rms_dex": float(np.median(per_galaxy_rms)),
        "median_cog_rms_16_84_bootstrap_dex": bootstrap_median_interval(per_galaxy_rms),
        "median_density_rms_dex": float(np.median(per_galaxy_density_rms)),
        "median_absolute_central_error_dex": float(np.median(np.abs(error[:, 0]))),
        "median_max_fractional_error_r_gt_5": float(
            np.median(np.max(fractional[:, outside], axis=1))
        ),
        "annular_rms_dex": np.sqrt(np.mean(annulus_error**2, axis=0)).tolist(),
        "annular_median_absolute_error_dex": np.median(
            np.abs(annulus_error), axis=0
        ).tolist(),
        "median_absolute_outer_slope_error": float(np.median(np.abs(slope_error))),
        "fraction_nonmonotone": float(
            np.mean(np.any(np.diff(prediction, axis=1) < -1e-10, axis=1))
        ),
    }
    if fit_result is not None:
        ratio = np.clip(fit_result["jacobian_min_ratio"], 1e-16, None)
        spec = FAMILY_SPECS[name]
        lower = np.asarray(spec.lower)
        upper = np.asarray(spec.upper)
        positions = (fit_result["parameters"] - lower) / (upper - lower)
        per_parameter_bound_fraction = np.mean(
            np.minimum(positions, 1.0 - positions) < 0.01, axis=0
        )
        summary.update(
            {
                "success_fraction": float(np.mean(fit_result["success"])),
                "median_log10_jacobian_min_ratio": float(np.median(np.log10(ratio))),
                "fraction_within_one_percent_of_bound": float(
                    np.mean(fit_result["boundary_distance"] < 0.01)
                ),
                "median_near_parameter_spread_fraction": float(
                    np.median(fit_result["near_parameter_spread"])
                ),
                "median_near_profile_spread_dex": float(
                    np.median(fit_result["near_profile_spread_dex"])
                ),
                "fraction_within_one_percent_of_bound_by_parameter": {
                    parameter: float(fraction)
                    for parameter, fraction in zip(
                        spec.parameter_names, per_parameter_bound_fraction
                    )
                },
            }
        )
    return summary


def selected_families(summaries: dict[str, dict]) -> tuple[str, str]:
    best_one = min(
        ONE_COMPONENT_FAMILIES,
        key=lambda family: summaries[family]["median_cog_rms_dex"],
    )
    best_mixture = min(
        MIXTURE_FAMILIES,
        key=lambda family: summaries[family]["median_cog_rms_dex"],
    )
    return best_one, best_mixture


def representative_indices(cogs_log: np.ndarray) -> list[int]:
    compactness = cogs_log[:, 0] - cogs_log[:, -1]
    return [
        int(np.argmin(np.abs(compactness - value)))
        for value in np.quantile(compactness, [0.1, 0.5, 0.9])
    ]


def representation_figure(
    truth: np.ndarray,
    reconstructions: dict[str, np.ndarray],
    summaries: dict[str, dict],
    best_one: str,
    best_mixture: str,
) -> None:
    shown = ["pca3", "pca5", "five_coordinates", best_one, best_mixture]
    colors = {
        "pca3": "black",
        "pca5": "0.55",
        "five_coordinates": OKABE_ITO[0],
        best_one: OKABE_ITO[4],
        best_mixture: OKABE_ITO[2],
    }
    styles = {
        "pca3": ":",
        "pca5": "--",
        "five_coordinates": "-.",
        best_one: "-",
        best_mixture: "-",
    }
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.2))

    for name in shown:
        error = np.abs(reconstructions[name] - truth)
        axes[0, 0].plot(
            RADII,
            np.median(error, axis=0),
            styles[name],
            color=colors[name],
            label=name.replace("_", " "),
        )
        density_model, middle = density_from_cog(reconstructions[name], RADII)
        density_truth, _ = density_from_cog(truth, RADII)
        axes[0, 1].plot(
            middle,
            np.median(np.abs(density_model - density_truth), axis=0),
            styles[name],
            color=colors[name],
            label=name.replace("_", " "),
        )
    axes[0, 0].set(
        xscale="log",
        yscale="log",
        xlabel="Radius (kpc)",
        ylabel="Median absolute CoG error (dex)",
        title="Cumulative profile",
    )
    axes[0, 0].legend(fontsize=7)
    axes[0, 1].set(
        xscale="log",
        yscale="log",
        xlabel="Annulus midpoint (kpc)",
        ylabel="Median absolute density error (dex)",
        title="Differential profile",
    )

    ordered = sorted(summaries, key=lambda name: summaries[name]["median_cog_rms_dex"])
    values = [summaries[name]["median_cog_rms_dex"] for name in ordered]
    bar_colors = [
        colors.get(name, OKABE_ITO[6] if name in MIXTURE_FAMILIES else OKABE_ITO[1])
        for name in ordered
    ]
    axes[0, 2].barh(
        [name.replace("_", " ") for name in ordered], values, color=bar_colors
    )
    axes[0, 2].invert_yaxis()
    axes[0, 2].set(xlabel="Median CoG RMS (dex)", title="Representation ranking")

    for axis, galaxy in zip(axes[1], representative_indices(truth)):
        normalized_truth = truth[galaxy] - truth[galaxy, -1]
        axis.plot(RADII, normalized_truth, "o", color="black", ms=3, label="TNG")
        for name in ("five_coordinates", best_one, best_mixture):
            normalized = (
                reconstructions[name][galaxy] - reconstructions[name][galaxy, -1]
            )
            axis.plot(
                RADII,
                normalized,
                styles[name],
                color=colors[name],
                label=name.replace("_", " "),
            )
        central_fraction = 10.0 ** normalized_truth[0]
        axis.set(
            xscale="log",
            xlabel="Radius (kpc)",
            title=rf"$M_*(<2)/M_{{*,148}}={central_fraction:.2f}$",
        )
    axes[1, 0].set_ylabel(r"$log_{10}[M_*(<R)/M_{*,148}]$")
    axes[1, 0].legend(fontsize=7)

    for label, axis in zip("ABCDEF", axes.ravel()):
        axis.text(
            -0.14, 1.04, label, transform=axis.transAxes, fontweight="bold", fontsize=11
        )
    fig.suptitle(f"exp55 — analytic representation of TNG300 CoGs (n={len(truth)})")
    fig.tight_layout()
    save_fig(fig, FIGDIR / "exp55_representation")
    plt.close(fig)


def identifiability_figure(
    truth: np.ndarray,
    fits: dict[str, dict[str, np.ndarray]],
    summaries: dict[str, dict],
    best_one: str,
    best_mixture: str,
) -> None:
    families = list(ANALYTIC_FAMILIES)
    colors = [
        OKABE_ITO[1] if name in ONE_COMPONENT_FAMILIES else OKABE_ITO[6]
        for name in families
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.2))

    x_condition = [
        -summaries[name]["median_log10_jacobian_min_ratio"] for name in families
    ]
    y_error = [summaries[name]["median_cog_rms_dex"] for name in families]
    for name, x_value, y_value, color in zip(families, x_condition, y_error, colors):
        axes[0, 0].scatter(x_value, y_value, color=color, s=45)
        axes[0, 0].annotate(
            name.replace("double_sersic_", "2S "), (x_value, y_value), fontsize=7
        )
    axes[0, 0].set(
        xlabel=r"Median $-\log_{10}(s_{\min}/s_{\max})$",
        ylabel="Median CoG RMS (dex)",
        title="Accuracy versus local degeneracy",
    )

    boundary = [
        summaries[name]["fraction_within_one_percent_of_bound"] for name in families
    ]
    axes[0, 1].barh(
        [name.replace("_", " ") for name in families], boundary, color=colors
    )
    axes[0, 1].invert_yaxis()
    axes[0, 1].set(xlabel="Fraction of galaxies", title="Near any parameter bound")

    spread = [
        summaries[name]["median_near_parameter_spread_fraction"] for name in families
    ]
    axes[0, 2].barh([name.replace("_", " ") for name in families], spread, color=colors)
    axes[0, 2].invert_yaxis()
    axes[0, 2].set(
        xlabel="Median parameter-range fraction",
        title="Multi-start near-optimum spread",
    )

    for axis, family, color in (
        (axes[1, 0], best_one, OKABE_ITO[4]),
        (axes[1, 1], best_mixture, OKABE_ITO[2]),
    ):
        ratio = np.clip(fits[family]["jacobian_min_ratio"], 1e-16, None)
        axis.hist(-np.log10(ratio), bins=30, color=color, alpha=0.8)
        axis.set(
            xlabel=r"$-\log_{10}(s_{\min}/s_{\max})$",
            ylabel="Number of galaxies",
            title=family.replace("_", " "),
        )

    compactness = truth[:, 0] - truth[:, -1]
    for family, color, marker in (
        (best_one, OKABE_ITO[4], "o"),
        (best_mixture, OKABE_ITO[2], "s"),
    ):
        bins = np.quantile(compactness, np.linspace(0.0, 1.0, 7))
        centers, medians = [], []
        for lower, upper in zip(bins[:-1], bins[1:]):
            members = (compactness >= lower) & (compactness <= upper)
            centers.append(np.median(10.0 ** compactness[members]))
            medians.append(np.median(fits[family]["rms_dex"][members]))
        axes[1, 2].plot(
            centers, medians, marker=marker, color=color, label=family.replace("_", " ")
        )
    axes[1, 2].set(
        xscale="log",
        xlabel=r"$M_*(<2)/M_{*,148}$",
        ylabel="Median CoG RMS (dex)",
        title="Accuracy across compactness",
    )
    axes[1, 2].legend(fontsize=7)

    for label, axis in zip("ABCDEF", axes.ravel()):
        axis.text(
            -0.14, 1.04, label, transform=axis.transAxes, fontweight="bold", fontsize=11
        )
    fig.suptitle("exp55 — residual quality is not parameter identifiability")
    fig.tight_layout()
    save_fig(fig, FIGDIR / "exp55_identifiability")
    plt.close(fig)


def profile_scan_figure(
    truth: np.ndarray,
    mixture_family: str,
    mixture_fit: dict[str, np.ndarray],
) -> dict:
    galaxies = representative_indices(truth)
    spec = FAMILY_SPECS[mixture_family]
    logit_values = np.linspace(spec.lower[1], spec.upper[1], 31)
    ratio_values = np.linspace(spec.lower[3], spec.upper[3], 31)
    scan_output = {}
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.8), sharey="row")
    for column, galaxy in enumerate(galaxies):
        best = mixture_fit["parameters"][galaxy]
        fraction_rms = profile_scan(
            mixture_family, RADII, truth[galaxy], best, 1, logit_values
        )
        ratio_rms = profile_scan(
            mixture_family, RADII, truth[galaxy], best, 3, ratio_values
        )
        fraction = expit(logit_values)
        ratio = 1.0 + np.exp(ratio_values)
        axes[0, column].plot(fraction, fraction_rms, color=OKABE_ITO[4])
        axes[0, column].axvline(expit(best[1]), color="black", linestyle=":")
        axes[1, column].plot(ratio, ratio_rms, color=OKABE_ITO[2])
        axes[1, column].axvline(1.0 + np.exp(best[3]), color="black", linestyle=":")
        axes[1, column].set_xscale("log")
        central_fraction = 10.0 ** (truth[galaxy, 0] - truth[galaxy, -1])
        axes[0, column].set_title(rf"$M_*(<2)/M_{{*,148}}={central_fraction:.2f}$")
        axes[0, column].set_xlabel("Compact fraction within 148 kpc")
        axes[1, column].set_xlabel(r"$R_{\rm extended}/R_{\rm compact}$")
        scan_output[f"galaxy_{galaxy}_fraction_rms"] = fraction_rms
        scan_output[f"galaxy_{galaxy}_ratio_rms"] = ratio_rms
    axes[0, 0].set_ylabel("Profiled CoG RMS (dex)")
    axes[1, 0].set_ylabel("Profiled CoG RMS (dex)")
    for label, axis in zip("ABCDEF", axes.ravel()):
        axis.text(
            -0.14, 1.04, label, transform=axis.transAxes, fontweight="bold", fontsize=11
        )
    fig.suptitle(
        "exp55 — mixture profile likelihoods; every other parameter re-optimized"
    )
    fig.tight_layout()
    save_fig(fig, FIGDIR / "exp55_profile_scans")
    plt.close(fig)
    scan_output.update(
        {
            "galaxies": np.asarray(galaxies),
            "logit_fraction_grid": logit_values,
            "log_ratio_minus_one_grid": ratio_values,
        }
    )
    return scan_output


def demo() -> None:
    started = time.perf_counter()
    model_demo(RADII)
    cogs_log, _ = load_sample(12)
    for family in ANALYTIC_FAMILIES:
        result = fit_cog(family, RADII, cogs_log[0])
        assert result["success"]
        assert np.isfinite(result["prediction"]).all()
    elapsed = time.perf_counter() - started
    print(f"exp55 real-profile preflight OK in {elapsed:.2f} s")


def run_fit() -> None:
    cogs_log, indices = load_sample(N_MAX)
    print(
        f"exp55 analytic CoG shootout: n={len(cogs_log)}, "
        f"source={source_table()}, families={len(ANALYTIC_FAMILIES)}"
    )
    references = reference_reconstructions(cogs_log)
    fits = {}
    for family in ANALYTIC_FAMILIES:
        path = family_output_path(family, len(cogs_log))
        if path.exists():
            print(f"  loading completed checkpoint {path.name}")
            fits[family] = dict(np.load(path))
        else:
            fits[family] = fit_family(family, cogs_log)

    reconstructions = dict(references)
    reconstructions.update(
        {family: result["prediction"] for family, result in fits.items()}
    )
    summaries = {
        name: summarize(name, prediction, cogs_log, fits.get(name))
        for name, prediction in reconstructions.items()
    }
    best_one, best_mixture = selected_families(summaries)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "summary.json").write_text(json.dumps(summaries, indent=2))
    np.savez(
        OUTDIR / "representation.npz",
        truth_log=cogs_log,
        indices=indices,
        radii=RADII,
        **{f"reconstruction_{name}": value for name, value in reconstructions.items()},
    )
    representation_figure(cogs_log, reconstructions, summaries, best_one, best_mixture)
    identifiability_figure(cogs_log, fits, summaries, best_one, best_mixture)
    scan_output = profile_scan_figure(cogs_log, best_mixture, fits[best_mixture])
    np.savez(OUTDIR / "profile_scans.npz", **scan_output)
    write_manifest(
        OUTDIR,
        {
            "experiment": "exp55_analytic_cog",
            "n_galaxy": len(cogs_log),
            "n_max": N_MAX,
            "families": list(ANALYTIC_FAMILIES),
            "best_one_component_by_cog_rms": best_one,
            "best_mixture_by_cog_rms": best_mixture,
            "source_table": str(source_table()),
        },
    )

    print("\nRepresentation summary")
    print(
        f"{'family':30s} {'npar':>4s} {'CoG RMS':>10s} {'density RMS':>12s} "
        f"{'log10 smin/smax':>16s} {'bound':>8s}"
    )
    for name in sorted(
        summaries, key=lambda item: summaries[item]["median_cog_rms_dex"]
    ):
        summary = summaries[name]
        condition = summary.get("median_log10_jacobian_min_ratio", np.nan)
        bound = summary.get("fraction_within_one_percent_of_bound", np.nan)
        n_parameter = (
            summary["n_parameter"] if summary["n_parameter"] is not None else 0
        )
        print(
            f"{name:30s} {n_parameter:4d} {summary['median_cog_rms_dex']:10.5f} "
            f"{summary['median_density_rms_dex']:12.5f} {condition:16.3f} {bound:8.3f}"
        )
    print(f"\nBest one-component family by CoG RMS: {best_one}")
    print(f"Best restricted mixture by CoG RMS: {best_mixture}")
    print(f"Wrote figures to {FIGDIR}")
    print(f"Wrote outputs to {OUTDIR}")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if command == "demo":
        demo()
    elif command == "fit":
        run_fit()
    else:
        raise SystemExit(__doc__)
