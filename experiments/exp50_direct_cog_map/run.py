"""exp50 — direct DiffMAH-to-CoG conversion at z=0.4.

The experiment predicts stable CoG coordinates (total mass, central fraction,
and ordered enclosed-mass radii) from DiffMAH+c200c, reconstructs a monotone
CoG, and compares it with the PCA-3 profile emulator on identical folds.
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
from hongshao.emulator import fit as emulator_fit  # noqa: E402
from hongshao.plotting import OKABE_ITO, save_fig, set_style  # noqa: E402
from hongshao.profile_emulator import fit_profile  # noqa: E402
from hongshao.provenance import write_manifest  # noqa: E402
from hongshao.tng_data import COG_RAD_KPC  # noqa: E402
from model import (  # noqa: E402
    QUANTILE_FRACTIONS,
    compress_cogs,
    empirical_crps,
    reconstruct_cogs,
    target_names,
)

set_style()

TABLE = ROOT / "data" / "processed" / "tng300_072_z0p4.fits"
FIGDIR = HERE / "figures"
OUTDIR = HERE / "outputs"
RADII = np.asarray(COG_RAD_KPC, float)
FEATURE_NAMES = ["logmp", "logtc", "early", "late", "c200c"]
N_FOLD = 5
SEED = 0
N_MAX = int(os.environ.get("EXP50_NMAX", 0))
N_DRAW = int(os.environ.get("EXP50_N_DRAW", 32))


def load_sample(n_max: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the z=0.4 sample and optionally take a mass-stratified subset."""
    table = Table.read(TABLE)
    table = table[table["use"]]
    cogs_log = np.asarray(table["logmstar_cog"], float)
    features = np.column_stack(
        [
            np.asarray(table[column], float)
            for column in (
                "dmah_logmp",
                "dmah_logtc",
                "dmah_early",
                "dmah_late",
                "c_200c",
            )
        ]
    )
    indices = np.asarray(table["index"], int)
    good = np.isfinite(cogs_log).all(axis=1) & np.isfinite(features).all(axis=1)
    cogs_log, features, indices = cogs_log[good], features[good], indices[good]
    if n_max and n_max < len(features):
        mass_order = np.argsort(features[:, 0])
        positions = np.linspace(0, len(features) - 1, n_max).round().astype(int)
        selected = mass_order[positions]
        cogs_log, features, indices = (
            cogs_log[selected],
            features[selected],
            indices[selected],
        )
    return features, cogs_log, indices


def folds_of(n_galaxy: int) -> list[np.ndarray]:
    order = np.random.default_rng(SEED).permutation(n_galaxy)
    return list(np.array_split(order, N_FOLD))


def shuffled_within_mass_bins(
    features: np.ndarray, columns: tuple[int, ...], seed: int, n_bin: int = 10
) -> np.ndarray:
    """Shuffle selected features within final-halo-mass deciles."""
    output = features.copy()
    edges = np.quantile(features[:, 0], np.linspace(0.0, 1.0, n_bin + 1))
    rng = np.random.default_rng(seed)
    for lower, upper in zip(edges[:-1], edges[1:]):
        members = np.where((features[:, 0] >= lower) & (features[:, 0] <= upper))[0]
        if len(members) > 1:
            permutation = rng.permutation(members)
            output[np.ix_(members, columns)] = features[np.ix_(permutation, columns)]
    return output


def cv_quantile(
    features: np.ndarray,
    targets: np.ndarray,
    mode: str,
    mean: str = "linear",
    n_draw: int = N_DRAW,
) -> dict:
    """Out-of-fold quantile-coordinate predictions and monotone CoG draws."""
    n_galaxy, n_target = targets.shape
    mean_parameters = np.empty_like(targets)
    sigma_parameters = np.empty_like(targets)
    mean_cogs = np.empty((n_galaxy, len(RADII)))
    draw_cogs = np.empty((n_draw, n_galaxy, len(RADII))) if n_draw else None
    fold_id = np.empty(n_galaxy, dtype=int)
    mean_dropped = np.empty(n_galaxy, dtype=int)
    draw_dropped = np.empty((n_draw, n_galaxy), dtype=int) if n_draw else None

    for fold_number, fold in enumerate(folds_of(n_galaxy)):
        train = np.setdiff1d(np.arange(n_galaxy), fold)
        emulator = emulator_fit(features[train], targets[train], mean=mean)
        mean_fold, sigma_fold, _ = emulator.predict(features[fold])
        mean_parameters[fold] = mean_fold
        sigma_parameters[fold] = sigma_fold
        mean_cogs[fold], mean_dropped[fold] = reconstruct_cogs(
            mean_fold, RADII, mode, return_dropped=True
        )
        fold_id[fold] = fold_number
        if n_draw:
            parameter_draws = emulator.sample(
                features[fold], size=n_draw, rng=1000 + fold_number
            )
            decoded, dropped = reconstruct_cogs(
                parameter_draws.reshape(-1, n_target), RADII, mode, True
            )
            draw_cogs[:, fold] = decoded.reshape(n_draw, len(fold), -1)
            draw_dropped[:, fold] = dropped.reshape(n_draw, len(fold))

    return {
        "mean_parameters": mean_parameters,
        "sigma_parameters": sigma_parameters,
        "mean_cogs": mean_cogs,
        "draw_cogs": draw_cogs,
        "fold_id": fold_id,
        "mean_dropped": mean_dropped,
        "draw_dropped": draw_dropped,
    }


def cv_pca(
    features: np.ndarray,
    cogs_log: np.ndarray,
    mean: str = "linear",
    n_draw: int = N_DRAW,
) -> dict:
    """Like-for-like out-of-fold PCA-3 profile reference."""
    n_galaxy = len(cogs_log)
    mean_cogs = np.empty_like(cogs_log)
    draw_cogs = np.empty((n_draw, *cogs_log.shape)) if n_draw else None
    fold_id = np.empty(n_galaxy, dtype=int)

    for fold_number, fold in enumerate(folds_of(n_galaxy)):
        train = np.setdiff1d(np.arange(n_galaxy), fold)
        profile_emulator = fit_profile(
            features[train],
            cogs_log[train, :-1],
            cogs_log[train, -1],
            RADII[:-1],
            n_modes=3,
            mean=mean,
        )
        mean_inner, _ = profile_emulator.predict(features[fold])
        mean_scores, _, _ = profile_emulator.emu.predict(features[fold])
        mean_cogs[fold] = np.column_stack([mean_inner, mean_scores[:, 0]])
        fold_id[fold] = fold_number
        if n_draw:
            score_draws = profile_emulator.emu.sample(
                features[fold], size=n_draw, rng=2000 + fold_number
            )
            inner_draws = (
                score_draws @ profile_emulator._design.T + profile_emulator.mean_shape
            )
            draw_cogs[:, fold] = np.concatenate(
                [inner_draws, score_draws[:, :, 0:1]], axis=2
            )

    return {
        "mean_cogs": mean_cogs,
        "draw_cogs": draw_cogs,
        "fold_id": fold_id,
        "mean_dropped": np.zeros(n_galaxy, dtype=int),
        "draw_dropped": None,
    }


def pca_representation(cogs_log: np.ndarray, n_mode: int = 3) -> np.ndarray:
    """In-sample PCA compression floor, not a halo prediction."""
    anchor = cogs_log[:, -1]
    shape = cogs_log[:, :-1] - anchor[:, None]
    mean_shape = shape.mean(axis=0)
    _, _, modes = np.linalg.svd(shape - mean_shape, full_matrices=False)
    basis = modes[:n_mode]
    reconstructed = (shape - mean_shape) @ basis.T @ basis + mean_shape
    return np.column_stack([reconstructed + anchor[:, None], anchor])


def aperture_targets(cogs_log: np.ndarray) -> np.ndarray:
    """M(<10), M(10-30), M(30-50), M(50-100), all in log10 Msun."""
    cumulative_log = np.column_stack(
        [
            [np.interp(radius, RADII, row) for row in cogs_log]
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


def enclosed_radii(cogs_log: np.ndarray, fractions=(0.5, 0.8, 0.9)) -> np.ndarray:
    linear = 10.0**cogs_log
    return np.array(
        [
            [qa.enclosed_radius(cog, RADII, fraction) for fraction in fractions]
            for cog in linear
        ]
    )


def summarize_model(name: str, result: dict, truth_log: np.ndarray) -> dict:
    mean_log = result["mean_cogs"]
    error = mean_log - truth_log
    relative = (10.0**mean_log - 10.0**truth_log) / 10.0**truth_log
    outside = RADII > 5.0
    truth_apertures = aperture_targets(truth_log)
    mean_apertures = aperture_targets(mean_log)
    summary = {
        "name": name,
        "median_profile_rms_dex": float(np.median(np.sqrt(np.mean(error**2, axis=1)))),
        "median_shape_rms_dex": float(
            np.median(np.sqrt(np.mean((error - error[:, -1:]) ** 2, axis=1)))
        ),
        "median_max_relative_r_gt_5": float(
            np.median(np.max(np.abs(relative[:, outside]), axis=1))
        ),
        "aperture_rms_dex": np.sqrt(
            np.mean((mean_apertures - truth_apertures) ** 2, axis=0)
        ).tolist(),
        "mean_dropped_quantile_knots": float(np.mean(result["mean_dropped"])),
    }
    draws = result.get("draw_cogs")
    if draws is not None:
        score = empirical_crps(truth_log, draws)
        summary["profile_crps_dex"] = float(score.mean())
        summary["profile_crps_by_radius"] = score.mean(axis=0).tolist()
        summary["coverage_68_by_radius"] = np.mean(
            (truth_log >= np.quantile(draws, 0.16, axis=0))
            & (truth_log <= np.quantile(draws, 0.84, axis=0)),
            axis=0,
        ).tolist()
        summary["coverage_90_by_radius"] = np.mean(
            (truth_log >= np.quantile(draws, 0.05, axis=0))
            & (truth_log <= np.quantile(draws, 0.95, axis=0)),
            axis=0,
        ).tolist()
        fold_scores = []
        for fold_number in range(N_FOLD):
            selected = result["fold_id"] == fold_number
            fold_scores.append(float(score[selected].mean()))
        summary["profile_crps_by_fold"] = fold_scores
        if result.get("draw_dropped") is not None:
            summary["draw_dropped_quantile_knots"] = float(
                np.mean(result["draw_dropped"])
            )
    return summary


def population_planes(result: dict, truth_log: np.ndarray) -> dict:
    """Mass-size and central-outer energy distances for mean and draws."""
    mean_log = result["mean_cogs"]
    truth_size = np.log10(enclosed_radii(truth_log, (0.5,))[:, 0])
    mean_size = np.log10(enclosed_radii(mean_log, (0.5,))[:, 0])
    truth_mass_size = np.column_stack([truth_log[:, -1], truth_size])
    mean_mass_size = np.column_stack([mean_log[:, -1], mean_size])
    truth_apertures = aperture_targets(truth_log)
    mean_apertures = aperture_targets(mean_log)
    truth_central_outer = np.column_stack(
        [truth_apertures[:, 0], truth_apertures[:, -1]]
    )
    mean_central_outer = np.column_stack([mean_apertures[:, 0], mean_apertures[:, -1]])
    output = {
        "mass_size_mean": qa.plane_energy(truth_mass_size, mean_mass_size),
        "central_outer_mean": qa.plane_energy(truth_central_outer, mean_central_outer),
    }
    draws = result.get("draw_cogs")
    if draws is not None:
        mass_size_draw, central_outer_draw = [], []
        for draw in draws[: min(8, len(draws))]:
            draw_size = np.log10(enclosed_radii(draw, (0.5,))[:, 0])
            mass_size_draw.append(
                qa.plane_energy(
                    truth_mass_size, np.column_stack([draw[:, -1], draw_size])
                )
            )
            draw_apertures = aperture_targets(draw)
            central_outer_draw.append(
                qa.plane_energy(
                    truth_central_outer,
                    np.column_stack([draw_apertures[:, 0], draw_apertures[:, -1]]),
                )
            )
        for key, values in (
            ("mass_size_draw", mass_size_draw),
            ("central_outer_draw", central_outer_draw),
        ):
            output[key] = {
                metric: float(np.mean([value[metric] for value in values]))
                for metric in values[0]
            }
    return output


def representation_figure(
    cogs_log: np.ndarray, reconstructions: dict[str, np.ndarray]
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.0), height_ratios=[1.0, 1.4])
    styles = {
        "quantile3": (OKABE_ITO[0], "-"),
        "quantile4": (OKABE_ITO[2], "--"),
        "pca3": (OKABE_ITO[4], ":"),
    }
    axis = axes[0, 0]
    for name, reconstructed in reconstructions.items():
        absolute = np.abs(reconstructed - cogs_log)
        color, line_style = styles[name]
        axis.plot(
            RADII, np.median(absolute, axis=0), line_style, color=color, label=name
        )
        axis.fill_between(
            RADII,
            np.quantile(absolute, 0.16, axis=0),
            np.quantile(absolute, 0.84, axis=0),
            color=color,
            alpha=0.12,
        )
    axis.set(
        xscale="log",
        yscale="log",
        xlabel="Radius (kpc)",
        ylabel="Absolute reconstruction error (dex)",
        title="Representation floor",
    )
    axis.legend()

    axis = axes[0, 1]
    labels, values = [], []
    for name, reconstructed in reconstructions.items():
        labels.append(name)
        values.append(
            np.median(np.sqrt(np.mean((reconstructed - cogs_log) ** 2, axis=1)))
        )
    axis.bar(labels, values, color=[styles[name][0] for name in labels])
    axis.set(ylabel="Median CoG RMS (dex)", title="Whole-profile compression")

    axis = axes[0, 2]
    for name, reconstructed in reconstructions.items():
        rel = np.abs((10.0**reconstructed - 10.0**cogs_log) / 10.0**cogs_log)
        axis.plot(
            RADII,
            np.median(rel, axis=0),
            styles[name][1],
            color=styles[name][0],
            label=name,
        )
    axis.set(
        xscale="log",
        yscale="log",
        xlabel="Radius (kpc)",
        ylabel="Median absolute fractional error",
        title="Fractional profile error",
    )

    total = cogs_log[:, -1]
    picks = [
        np.argmin(np.abs(total - quantile))
        for quantile in np.quantile(total, [0.1, 0.5, 0.9])
    ]
    for column, galaxy in enumerate(picks):
        axis = axes[1, column]
        axis.plot(RADII, cogs_log[galaxy], "o-", color="black", ms=3, label="TNG")
        for name, reconstructed in reconstructions.items():
            axis.plot(
                RADII,
                reconstructed[galaxy],
                styles[name][1],
                color=styles[name][0],
                label=name,
            )
        axis.set(
            xscale="log", xlabel="Radius (kpc)", title=f"log M*={total[galaxy]:.2f}"
        )
        if column == 0:
            axis.set_ylabel(r"log$_{10}$ M$_*(<R)$ (M$_\odot$)")
            axis.legend(fontsize=7)
    for label, axis in zip("ABCDEF", axes.ravel()):
        axis.text(
            -0.14, 1.04, label, transform=axis.transAxes, fontsize=11, fontweight="bold"
        )
    fig.suptitle(f"exp50 — stable quantile representation (n={len(cogs_log)})")
    fig.tight_layout()
    save_fig(fig, FIGDIR / "exp50_representation")
    plt.close(fig)


def skill_figure(
    summaries: dict[str, dict], results: dict[str, dict], truth_log: np.ndarray
) -> None:
    compared = [
        "pca_linear",
        "pca_poly2",
        "quantile3_linear",
        "quantile3_poly2",
        "quantile4_linear",
    ]
    colors = [OKABE_ITO[7], OKABE_ITO[4], OKABE_ITO[0], OKABE_ITO[2], OKABE_ITO[5]]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))
    for name, color in zip(compared, colors):
        axes[0, 0].plot(
            RADII,
            summaries[name]["profile_crps_by_radius"],
            color=color,
            label=name.replace("_", " "),
        )
        axes[0, 1].plot(
            RADII,
            np.median(np.abs(results[name]["mean_cogs"] - truth_log), axis=0),
            color=color,
            label=name.replace("_", " "),
        )
        axes[1, 0].plot(RADII, summaries[name]["coverage_68_by_radius"], color=color)
    axes[0, 0].set(
        xscale="log",
        xlabel="Radius (kpc)",
        ylabel="Empirical CRPS (dex)",
        title="Predictive profile distribution",
    )
    axes[0, 0].legend(fontsize=7)
    axes[0, 1].set(
        xscale="log",
        xlabel="Radius (kpc)",
        ylabel="Median |mean−truth| (dex)",
        title="Conditional-mean error",
    )
    axes[1, 0].axhline(0.68, color="0.5", linestyle=":")
    axes[1, 0].set(
        xscale="log",
        xlabel="Radius (kpc)",
        ylabel="Empirical coverage",
        ylim=(0.45, 0.9),
        title="Nominal central interval (0.68)",
    )

    control_names = [
        "mass_only",
        "diffmah",
        "mah_shuffled",
        "c_shuffled",
        "quantile3_linear",
        "quantile3_poly2",
        "pca_poly2",
    ]
    values = [summaries[name]["median_profile_rms_dex"] for name in control_names]
    axes[1, 1].barh(
        [name.replace("_", " ") for name in control_names],
        values,
        color=[
            OKABE_ITO[3],
            OKABE_ITO[1],
            OKABE_ITO[6],
            OKABE_ITO[5],
            OKABE_ITO[0],
            OKABE_ITO[2],
            OKABE_ITO[4],
        ],
    )
    axes[1, 1].invert_yaxis()
    axes[1, 1].set(
        xlabel="Median held-out CoG RMS (dex)", title="Features and null controls"
    )
    for label, axis in zip("ABCD", axes.ravel()):
        axis.text(
            -0.14, 1.04, label, transform=axis.transAxes, fontsize=11, fontweight="bold"
        )
    fig.suptitle(f"exp50 — end-to-end halo-to-CoG prediction (n={len(truth_log)})")
    fig.tight_layout()
    save_fig(fig, FIGDIR / "exp50_predictive_skill")
    plt.close(fig)


def parameter_figure(truth_parameters: np.ndarray, result: dict, mode: str) -> None:
    predicted = result["mean_parameters"]
    names = target_names(mode)
    labels = {
        "log_mstar_total": r"$\log M_{*,148}$",
        "logit_f2": r"$\mathrm{logit}[M_*(<2)/M_{*,148}]$",
        "log_r20": r"$\log R_{20}$",
        "log_dlogr_20_50": r"$\log \Delta\log R_{20-50}$",
        "log_dlogr_50_80": r"$\log \Delta\log R_{50-80}$",
        "log_dlogr_80_90": r"$\log \Delta\log R_{80-90}$",
    }
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.0))
    for index, (axis, name) in enumerate(zip(axes.ravel(), names)):
        axis.hexbin(
            truth_parameters[:, index],
            predicted[:, index],
            gridsize=35,
            mincnt=1,
            cmap="cividis",
            rasterized=True,
        )
        limits = [
            min(truth_parameters[:, index].min(), predicted[:, index].min()),
            max(truth_parameters[:, index].max(), predicted[:, index].max()),
        ]
        axis.plot(limits, limits, ":", color="white", lw=1.2)
        residual = predicted[:, index] - truth_parameters[:, index]
        r_squared = 1.0 - np.mean(residual**2) / np.var(truth_parameters[:, index])
        axis.set(
            xlabel=f"Measured {labels[name]}",
            ylabel=f"Predicted {labels[name]}",
            title=f"R²={r_squared:+.2f}",
        )
    for axis in axes.ravel()[len(names) :]:
        axis.axis("off")
    for label, axis in zip("ABCDE", axes.ravel()[: len(names)]):
        axis.text(
            -0.14, 1.04, label, transform=axis.transAxes, fontsize=11, fontweight="bold"
        )
    fig.suptitle("exp50 — held-out quantile-coordinate predictions (poly2 mean)")
    fig.tight_layout()
    save_fig(fig, FIGDIR / "exp50_parameter_predictions")
    plt.close(fig)


def planes_figure(result: dict, truth_log: np.ndarray) -> None:
    mean_log = result["mean_cogs"]
    draw_log = result["draw_cogs"][0]
    sets = [("TNG", truth_log), ("Conditional mean", mean_log), ("One draw", draw_log)]
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.0), sharex="row", sharey="row")
    for column, (label, cogs) in enumerate(sets):
        sizes = np.log10(enclosed_radii(cogs, (0.5,))[:, 0])
        axes[0, column].hexbin(
            cogs[:, -1],
            sizes,
            gridsize=35,
            mincnt=1,
            cmap="cividis",
            rasterized=True,
        )
        axes[0, column].set(
            xlabel=r"log$_{10}$ M$_*(<148)$ (M$_\odot$)",
            ylabel=r"log$_{10}$ R$_{50}$ (kpc)" if column == 0 else "",
            title=label,
        )
        apertures = aperture_targets(cogs)
        axes[1, column].hexbin(
            apertures[:, 0],
            apertures[:, -1],
            gridsize=35,
            mincnt=1,
            cmap="cividis",
            rasterized=True,
        )
        axes[1, column].set(
            xlabel=r"log$_{10}$ M$_*(<10)$ (M$_\odot$)",
            ylabel=r"log$_{10}$ M$_*(50$–$100)$ (M$_\odot$)" if column == 0 else "",
        )
    for label, axis in zip("ABCDEF", axes.ravel()):
        axis.text(
            -0.14, 1.04, label, transform=axis.transAxes, fontsize=11, fontweight="bold"
        )
    fig.suptitle(
        "exp50 — population structure: data, conditional mean, and generative draw"
    )
    fig.tight_layout()
    save_fig(fig, FIGDIR / "exp50_population_planes")
    plt.close(fig)


def regenerate_saved_figures() -> None:
    """Rebuild diagnostic figures from the saved held-out predictions."""
    prediction_path = OUTDIR / "predictions.npz"
    if not prediction_path.exists():
        raise FileNotFoundError(
            f"missing {prediction_path}; run the full Exp50 fit before rebuilding figures"
        )
    with np.load(prediction_path) as saved:
        truth_log = saved["truth_cogs_log"]
        mean_cogs = saved["selected_mean_cogs_log"]
        result = {
            "mean_cogs": mean_cogs,
            "draw_cogs": saved["selected_draw_cogs_log"],
            "mean_parameters": compress_cogs(mean_cogs, RADII, "quantile3"),
        }
        parameter_figure(saved["quantile3_targets"], result, "quantile3")
        planes_figure(result, truth_log)
    print(f"rebuilt saved diagnostic figures in {FIGDIR}")


def mass_binned_prediction_figure(
    features: np.ndarray, result: dict, truth_log: np.ndarray
) -> None:
    """Show predicted profiles and residuals directly in halo-mass bins."""
    mean_log = result["mean_cogs"]
    draw_log = result["draw_cogs"][0]
    mass_edges = np.quantile(features[:, 0], [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.0), sharex=True)
    for column, (lower, upper) in enumerate(zip(mass_edges[:-1], mass_edges[1:])):
        if column == 2:
            selected = (features[:, 0] >= lower) & (features[:, 0] <= upper)
        else:
            selected = (features[:, 0] >= lower) & (features[:, 0] < upper)

        axis = axes[0, column]
        axis.plot(
            RADII,
            np.median(truth_log[selected], axis=0),
            "o-",
            color="black",
            markersize=3,
            label="TNG median",
        )
        axis.plot(
            RADII,
            np.median(mean_log[selected], axis=0),
            "--",
            color=OKABE_ITO[0],
            label="Conditional-mean median",
        )
        axis.plot(
            RADII,
            np.median(draw_log[selected], axis=0),
            ":",
            color=OKABE_ITO[2],
            label="One generative-draw median",
        )
        axis.set(
            xscale="log",
            title=(
                rf"${lower:.2f} \leq \log M_{{\rm peak}} < {upper:.2f}$"
                if column < 2
                else rf"${lower:.2f} \leq \log M_{{\rm peak}} \leq {upper:.2f}$"
            ),
            ylabel=r"Median log$_{10}$ M$_*(<R)$ (M$_\odot$)" if column == 0 else "",
        )
        if column == 0:
            axis.legend(fontsize=7)

        axis = axes[1, column]
        mean_relative = (
            10.0 ** mean_log[selected] - 10.0 ** truth_log[selected]
        ) / 10.0 ** truth_log[selected]
        draw_relative = (
            10.0 ** draw_log[selected] - 10.0 ** truth_log[selected]
        ) / 10.0 ** truth_log[selected]
        axis.axhline(0.0, color="0.5", linestyle=":")
        axis.plot(
            RADII,
            100.0 * np.median(mean_relative, axis=0),
            "--",
            color=OKABE_ITO[0],
            label="Conditional mean",
        )
        axis.plot(
            RADII,
            100.0 * np.median(draw_relative, axis=0),
            ":",
            color=OKABE_ITO[2],
            label="One draw",
        )
        axis.fill_between(
            RADII,
            100.0 * np.quantile(mean_relative, 0.16, axis=0),
            100.0 * np.quantile(mean_relative, 0.84, axis=0),
            color=OKABE_ITO[0],
            alpha=0.12,
        )
        axis.set(
            xscale="log",
            xlabel="Radius (kpc)",
            ylabel="Median relative residual (%)" if column == 0 else "",
        )
    for label, axis in zip("ABCDEF", axes.ravel()):
        axis.text(
            -0.14, 1.04, label, transform=axis.transAxes, fontsize=11, fontweight="bold"
        )
    fig.suptitle(
        "exp50 — direct CoG predictions in final halo-mass terciles "
        f"(n={len(truth_log)})"
    )
    fig.tight_layout()
    save_fig(fig, FIGDIR / "exp50_mass_binned_predictions")
    plt.close(fig)


def fit_experiment() -> None:
    features, truth_log, indices = load_sample(N_MAX)
    print(f"exp50: z=0.4 direct halo-to-CoG map on n={len(features)} galaxies")
    print(f"  draws per predictive model: {N_DRAW}")
    targets = {
        mode: compress_cogs(truth_log, RADII, mode) for mode in QUANTILE_FRACTIONS
    }

    representations = {
        mode: reconstruct_cogs(parameters, RADII, mode)
        for mode, parameters in targets.items()
    }
    representations["pca3"] = pca_representation(truth_log)
    representation_figure(truth_log, representations)
    print("\n[representation floors]")
    for name, reconstructed in representations.items():
        rms = np.sqrt(np.mean((reconstructed - truth_log) ** 2, axis=1))
        print(f"  {name:12s}: median CoG RMS {np.median(rms):.4f} dex")

    shuffled_mah = shuffled_within_mass_bins(features, (1, 2, 3), seed=11)
    shuffled_c = shuffled_within_mass_bins(features, (4,), seed=12)
    results = {
        "mass_only": cv_quantile(
            features[:, :1], targets["quantile3"], "quantile3", n_draw=0
        ),
        "diffmah": cv_quantile(
            features[:, :4], targets["quantile3"], "quantile3", n_draw=0
        ),
        "mah_shuffled": cv_quantile(
            shuffled_mah, targets["quantile3"], "quantile3", n_draw=0
        ),
        "c_shuffled": cv_quantile(
            shuffled_c, targets["quantile3"], "quantile3", n_draw=0
        ),
        "quantile3_linear": cv_quantile(features, targets["quantile3"], "quantile3"),
        "quantile3_poly2": cv_quantile(
            features, targets["quantile3"], "quantile3", mean="poly2"
        ),
        "quantile4_linear": cv_quantile(features, targets["quantile4"], "quantile4"),
        "pca_linear": cv_pca(features, truth_log),
        "pca_poly2": cv_pca(features, truth_log, mean="poly2"),
    }
    summaries = {
        name: summarize_model(name, result, truth_log)
        for name, result in results.items()
    }

    print("\n[end-to-end held-out results]")
    for name, summary in summaries.items():
        crps_text = (
            f", profile CRPS {summary['profile_crps_dex']:.4f} dex"
            if "profile_crps_dex" in summary
            else ""
        )
        print(
            f"  {name:18s}: median CoG RMS {summary['median_profile_rms_dex']:.4f} dex"
            f", median max|relative| R>5 kpc "
            f"{100 * summary['median_max_relative_r_gt_5']:.1f}%{crps_text}"
        )

    quantile_candidates = ["quantile3_linear", "quantile3_poly2", "quantile4_linear"]
    chosen = min(
        quantile_candidates, key=lambda name: summaries[name]["profile_crps_dex"]
    )
    pca_reference = "pca_poly2" if chosen.endswith("poly2") else "pca_linear"
    quantile_crps = summaries[chosen]["profile_crps_dex"]
    pca_crps = summaries[pca_reference]["profile_crps_dex"]
    pca_fold_spread = float(
        np.std(summaries[pca_reference]["profile_crps_by_fold"], ddof=1)
    )
    parity = quantile_crps <= pca_crps + pca_fold_spread
    print(f"\n[decision] selected {chosen}; matched reference {pca_reference}")
    print(f"  profile CRPS: quantile {quantile_crps:.4f} dex vs PCA {pca_crps:.4f} dex")
    print(f"  PCA fold-to-fold CRPS spread: {pca_fold_spread:.4f} dex")
    print(f"  interpretability-with-parity gate: {'PASS' if parity else 'FAIL'}")

    planes = population_planes(results[chosen], truth_log)
    skill_figure(summaries, results, truth_log)
    parameter_figure(targets["quantile3"], results["quantile3_poly2"], "quantile3")
    planes_figure(results[chosen], truth_log)
    mass_binned_prediction_figure(features, results[chosen], truth_log)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_galaxy": len(features),
        "n_draw": N_DRAW,
        "selected_quantile_model": chosen,
        "pca_reference": pca_reference,
        "parity": parity,
        "pca_fold_spread_dex": pca_fold_spread,
        "representation": {
            name: {
                "median_profile_rms_dex": float(
                    np.median(
                        np.sqrt(np.mean((reconstructed - truth_log) ** 2, axis=1))
                    )
                )
            }
            for name, reconstructed in representations.items()
        },
        "models": summaries,
        "selected_population_planes": planes,
    }
    (OUTDIR / "results.json").write_text(json.dumps(payload, indent=2))
    np.savez_compressed(
        OUTDIR / "predictions.npz",
        indices=indices,
        features=features,
        truth_cogs_log=truth_log,
        quantile3_targets=targets["quantile3"],
        selected_mean_cogs_log=results[chosen]["mean_cogs"],
        selected_draw_cogs_log=results[chosen]["draw_cogs"],
        pca_mean_cogs_log=results[pca_reference]["mean_cogs"],
        pca_draw_cogs_log=results[pca_reference]["draw_cogs"],
    )
    write_manifest(
        OUTDIR,
        {
            "n_galaxy": len(features),
            "n_draw": N_DRAW,
            "selected_quantile_model": chosen,
            "pca_reference": pca_reference,
            "parity": parity,
        },
    )
    print(f"\nwrote figures to {FIGDIR}")
    print(f"wrote outputs to {OUTDIR}")


def demo() -> None:
    from model import demo as coordinate_demo

    coordinate_demo()
    features, truth_log, _ = load_sample(80)
    targets = compress_cogs(truth_log, RADII, "quantile3")
    result = cv_quantile(features, targets, "quantile3", n_draw=3)
    assert np.isfinite(result["mean_cogs"]).all()
    assert np.all(np.diff(result["mean_cogs"], axis=1) >= 0.0)
    assert result["draw_cogs"].shape == (3, len(features), len(RADII))
    shuffled = shuffled_within_mass_bins(features, (1, 2, 3), seed=3, n_bin=4)
    assert np.allclose(shuffled[:, 0], features[:, 0])
    assert not np.allclose(shuffled[:, 1:4], features[:, 1:4])
    score = summarize_model("demo", result, truth_log)
    assert np.isfinite(score["profile_crps_dex"])
    print("exp50 end-to-end demo OK")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "fit"
    if command == "demo":
        demo()
    elif command == "fit":
        fit_experiment()
    elif command == "figures":
        regenerate_saved_figures()
    else:
        raise SystemExit(f"unknown command {command!r}; use demo, fit, or figures")
