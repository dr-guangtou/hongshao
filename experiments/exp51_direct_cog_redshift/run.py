"""exp51 — redshift evolution of the readable DiffMAH-to-CoG conversion.

Commands
--------
demo
    Synthetic mechanics plus a small real-data fit.
fit
    Five-epoch descendant-conditioned comparison, held-epoch closure, and
    cross-epoch coherent draws.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP50 = ROOT / "experiments" / "exp50_direct_cog_map"
sys.path.insert(0, str(ROOT))

from hongshao.emulator import fit as emulator_fit  # noqa: E402
from hongshao.multi_epoch import MultiEpochEmulator, fit_rho, interp_emulator  # noqa: E402
from hongshao.plotting import OKABE_ITO, save_fig, set_style  # noqa: E402
from hongshao.provenance import write_manifest  # noqa: E402
from hongshao.tng_data import COG_RAD_KPC  # noqa: E402


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EXP50_MODEL = load_module("exp50_coordinate_model", EXP50 / "model.py")
# exp50/run.py imports its coordinate model as ``model``. Make that validated
# module visible under the expected short name only while the module is loaded.
sys.modules["model"] = EXP50_MODEL
EXP50_RUN = load_module("exp50_run", EXP50 / "run.py")
EXP51_MODEL = load_module("exp51_coordinate_model", HERE / "model.py")

compress_cogs = EXP51_MODEL.compress_cogs
decode_coordinates = EXP51_MODEL.decode_coordinates
reconstruct_cogs = EXP51_MODEL.reconstruct_cogs
# Reuse exp50's cross-validation driver with exp51's high-redshift-safe decoder.
EXP50_RUN.reconstruct_cogs = reconstruct_cogs
cv_quantile = EXP50_RUN.cv_quantile
cv_pca = EXP50_RUN.cv_pca
folds_of = EXP50_RUN.folds_of
pca_representation = EXP50_RUN.pca_representation
population_planes = EXP50_RUN.population_planes
shuffled_within_mass_bins = EXP50_RUN.shuffled_within_mass_bins
summarize_model = EXP50_RUN.summarize_model

set_style()

FIGDIR = HERE / "figures"
OUTDIR = HERE / "outputs"
TABLE = ROOT / "data" / "processed" / "tng300_072_z0p4.fits"
RADII = np.asarray(COG_RAD_KPC, float)
REDSHIFTS = np.array([0.4, 0.7, 1.0, 1.5, 2.0])
FEATURE_NAMES = ["logmp", "logtc", "early", "late", "c200c(z=0.4)"]
TARGET_LABELS = [
    r"$\log M_{*,148}$",
    r"$\mathrm{logit}\,f_{*,2}$",
    r"$\log R_{20,\mathrm{out}}$",
    r"$\log\Delta R_{20-50,\mathrm{out}}$",
    r"$\log\Delta R_{50-80,\mathrm{out}}$",
]
N_DRAW = int(os.environ.get("EXP51_N_DRAW", 32))
N_MAX = int(os.environ.get("EXP51_NMAX", 0))


def source_output(experiment: str, filename: str) -> Path:
    candidates = [
        ROOT / "experiments" / experiment / "outputs" / filename,
        ROOT.parent / "hongshao" / "experiments" / experiment / "outputs" / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"missing {filename}; checked {candidates}")


def progenitor_quality_mask(data: np.ndarray) -> np.ndarray:
    """Use exp37's measured break to remove broken main-branch matches."""
    total_log = np.log10(data[..., -1])
    maximum_drop = np.max(total_log[:, :1] - total_log, axis=1)
    return (maximum_drop <= 2.0) & (np.min(total_log, axis=1) >= 9.0)


def load_sample(
    n_max: int = N_MAX,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return features, five log-CoGs, ids, epoch masses, and population rows."""
    population = np.load(source_output("exp32_full_population", "population.npz"))
    table = Table.read(TABLE)
    table_row = {int(index): row for row, index in enumerate(table["index"])}
    population_rows = []
    features = []
    for row, index in enumerate(population["index"]):
        source_row = table_row.get(int(index))
        if source_row is None:
            continue
        values = np.array(
            [
                table[column][source_row]
                for column in (
                    "dmah_logmp",
                    "dmah_logtc",
                    "dmah_early",
                    "dmah_late",
                    "c_200c",
                )
            ],
            float,
        )
        if np.isfinite(values).all():
            population_rows.append(row)
            features.append(values)
    population_rows = np.asarray(population_rows, int)
    features = np.asarray(features, float)
    data = np.asarray(population["data"][population_rows], float)
    indices = np.asarray(population["index"][population_rows], int)
    epoch_mass = np.asarray(population["logmh_zk_real"][population_rows], float)
    keep = progenitor_quality_mask(data)
    keep &= np.isfinite(data).all(axis=(1, 2)) & np.isfinite(epoch_mass).all(axis=1)
    features, data, indices, epoch_mass, population_rows = (
        features[keep],
        data[keep],
        indices[keep],
        epoch_mass[keep],
        population_rows[keep],
    )
    if n_max and n_max < len(features):
        order = np.argsort(features[:, 0])
        positions = np.linspace(0, len(features) - 1, n_max).round().astype(int)
        selected = order[positions]
        features, data, indices, epoch_mass, population_rows = (
            features[selected],
            data[selected],
            indices[selected],
            epoch_mass[selected],
            population_rows[selected],
        )
    return features, np.log10(data), indices, epoch_mass, population_rows


def representation_summary(cogs_log: np.ndarray) -> tuple[dict, dict]:
    reconstructions = {"quantile": [], "pca": []}
    summaries = {"quantile": [], "pca": []}
    for epoch in range(len(REDSHIFTS)):
        targets = compress_cogs(cogs_log[:, epoch], RADII, "quantile3")
        quantile = reconstruct_cogs(targets, RADII, "quantile3")
        pca = pca_representation(cogs_log[:, epoch])
        for name, reconstructed in (("quantile", quantile), ("pca", pca)):
            error = np.sqrt(np.mean((reconstructed - cogs_log[:, epoch]) ** 2, axis=1))
            reconstructions[name].append(reconstructed)
            summaries[name].append(float(np.median(error)))
    return summaries, {
        name: np.stack(value, axis=1) for name, value in reconstructions.items()
    }


def fit_independent_epochs(
    features: np.ndarray, cogs_log: np.ndarray
) -> tuple[np.ndarray, dict, dict]:
    targets = np.stack(
        [compress_cogs(cogs_log[:, epoch], RADII, "quantile3") for epoch in range(5)],
        axis=1,
    )
    results = {name: [] for name in ("quantile_linear", "quantile_poly2", "pca_poly2")}
    summaries = {name: [] for name in results}
    for epoch, redshift in enumerate(REDSHIFTS):
        print(f"  fitting independent z={redshift:.1f} models")
        epoch_results = {
            "quantile_linear": cv_quantile(
                features, targets[:, epoch], "quantile3", n_draw=N_DRAW
            ),
            "quantile_poly2": cv_quantile(
                features,
                targets[:, epoch],
                "quantile3",
                mean="poly2",
                n_draw=N_DRAW,
            ),
            "pca_poly2": cv_pca(
                features, cogs_log[:, epoch], mean="poly2", n_draw=N_DRAW
            ),
        }
        for name, result in epoch_results.items():
            results[name].append(result)
            summaries[name].append(summarize_model(name, result, cogs_log[:, epoch]))
    return targets, results, summaries


def z2_controls(
    features: np.ndarray,
    epoch_mass: np.ndarray,
    targets_z2: np.ndarray,
    truth_z2: np.ndarray,
) -> dict:
    shuffled = shuffled_within_mass_bins(features, (1, 2, 3), seed=51)
    definitions = {
        "descendant_final_mass_only": features[:, :1],
        "z2_halo_mass_only": epoch_mass[:, 4:5],
        "descendant_diffmah": features[:, :4],
        "descendant_diffmah_shuffled": shuffled[:, :4],
        "descendant_diffmah_c": features,
    }
    output = {}
    for name, model_features in definitions.items():
        result = cv_quantile(
            model_features, targets_z2, "quantile3", mean="linear", n_draw=0
        )
        output[name] = summarize_model(name, result, truth_z2)
    return output


def closure_result(
    features: np.ndarray,
    targets: np.ndarray,
    held_epoch: int,
    method: str,
) -> dict:
    """Fold-clean held-epoch prediction by interpolation or nearest transfer."""
    n_galaxy, _, n_target = targets.shape
    mean_parameters = np.empty((n_galaxy, n_target))
    sigma_parameters = np.empty_like(mean_parameters)
    mean_cogs = np.empty((n_galaxy, len(RADII)))
    draw_cogs = np.empty((N_DRAW, n_galaxy, len(RADII)))
    mean_dropped = np.empty(n_galaxy, int)
    draw_dropped = np.empty((N_DRAW, n_galaxy), int)
    fold_id = np.empty(n_galaxy, int)
    other_epochs = [value for value in range(5) if value != held_epoch]
    nearest_epoch = min(other_epochs, key=lambda value: abs(value - held_epoch))
    for fold_number, fold in enumerate(folds_of(n_galaxy)):
        train = np.setdiff1d(np.arange(n_galaxy), fold)
        emulators = [
            emulator_fit(features[train], targets[train, epoch], mean="poly2")
            for epoch in other_epochs
        ]
        if method == "interpolated":
            emulator = interp_emulator(
                emulators, REDSHIFTS[other_epochs], REDSHIFTS[held_epoch]
            )
        elif method == "nearest":
            emulator = emulators[other_epochs.index(nearest_epoch)]
        else:
            raise ValueError(f"unknown closure method {method}")
        mean_parameters[fold], sigma_parameters[fold], _ = emulator.predict(
            features[fold]
        )
        mean_cogs[fold], mean_dropped[fold] = reconstruct_cogs(
            mean_parameters[fold], RADII, "quantile3", True
        )
        parameter_draws = emulator.sample(
            features[fold], size=N_DRAW, rng=4100 + fold_number
        )
        decoded, dropped = reconstruct_cogs(
            parameter_draws.reshape(-1, n_target), RADII, "quantile3", True
        )
        draw_cogs[:, fold] = decoded.reshape(N_DRAW, len(fold), -1)
        draw_dropped[:, fold] = dropped.reshape(N_DRAW, len(fold))
        fold_id[fold] = fold_number
    return {
        "mean_parameters": mean_parameters,
        "sigma_parameters": sigma_parameters,
        "mean_cogs": mean_cogs,
        "draw_cogs": draw_cogs,
        "mean_dropped": mean_dropped,
        "draw_dropped": draw_dropped,
        "fold_id": fold_id,
    }


def held_epoch_closure(
    features: np.ndarray,
    targets: np.ndarray,
    cogs_log: np.ndarray,
    direct_summaries: list[dict],
) -> tuple[dict, dict]:
    summaries = {}
    results = {}
    for epoch in (1, 2, 3):
        key = f"z{REDSHIFTS[epoch]:.1f}"
        summaries[key] = {"direct": direct_summaries[epoch]}
        results[key] = {}
        for method in ("interpolated", "nearest"):
            result = closure_result(features, targets, epoch, method)
            results[key][method] = result
            summaries[key][method] = summarize_model(method, result, cogs_log[:, epoch])
    return summaries, results


def residual_correlation(
    targets: np.ndarray, epoch_results: list[dict]
) -> tuple[np.ndarray, np.ndarray, float]:
    residuals = np.stack(
        [
            (targets[:, epoch] - epoch_results[epoch]["mean_parameters"])
            / epoch_results[epoch]["sigma_parameters"]
            for epoch in range(5)
        ],
        axis=1,
    )
    by_target = np.empty((residuals.shape[2], 5, 5))
    for target in range(residuals.shape[2]):
        for first in range(5):
            for second in range(5):
                by_target[target, first, second] = spearmanr(
                    residuals[:, first, target], residuals[:, second, target]
                ).statistic
    average = np.mean(by_target, axis=0)
    return average, by_target, fit_rho(average)


def coherent_draws(features: np.ndarray, targets: np.ndarray, rho: float) -> np.ndarray:
    """Fold-clean AR(1)-coherent profile draws with one measured rho."""
    n_galaxy, n_epoch, n_target = targets.shape
    output = np.empty((N_DRAW, n_galaxy, n_epoch, len(RADII)))
    for fold_number, fold in enumerate(folds_of(n_galaxy)):
        train = np.setdiff1d(np.arange(n_galaxy), fold)
        emulators = [
            emulator_fit(features[train], targets[train, epoch], mean="poly2")
            for epoch in range(n_epoch)
        ]
        multi_epoch = MultiEpochEmulator(REDSHIFTS, emulators, rho)
        parameter_draws = multi_epoch.sample_epochs(
            features[fold], size=N_DRAW, rng=5100 + fold_number
        )
        for epoch in range(n_epoch):
            decoded = reconstruct_cogs(
                parameter_draws[:, :, epoch].reshape(-1, n_target),
                RADII,
                "quantile3",
            )
            output[:, fold, epoch] = decoded.reshape(N_DRAW, len(fold), -1)
    return output


def enclosed_r50(cogs_log: np.ndarray) -> np.ndarray:
    linear = 10.0**cogs_log
    output = np.empty(linear.shape[:-1])
    for index in np.ndindex(linear.shape[:-1]):
        output[index] = EXP50_RUN.qa.enclosed_radius(linear[index], RADII, 0.5)
    return output


def rank_correlation_matrix(values: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [
                spearmanr(values[:, first], values[:, second]).statistic
                for second in range(5)
            ]
            for first in range(5)
        ]
    )


def coherence_summary(
    truth_log: np.ndarray, mean_log: np.ndarray, draws_log: np.ndarray
) -> dict:
    quantities = {
        "total_mass": (
            truth_log[..., -1],
            mean_log[..., -1],
            draws_log[..., -1],
        ),
        "r50": (
            np.log10(enclosed_r50(truth_log)),
            np.log10(enclosed_r50(mean_log)),
            np.log10(np.stack([enclosed_r50(draw) for draw in draws_log], axis=0)),
        ),
        "central_fraction": (
            truth_log[..., 0] - truth_log[..., -1],
            mean_log[..., 0] - mean_log[..., -1],
            draws_log[..., 0] - draws_log[..., -1],
        ),
    }
    output = {}
    for name, (truth, mean, draws) in quantities.items():
        draw_matrices = np.stack([rank_correlation_matrix(draw) for draw in draws])
        output[name] = {
            "truth": rank_correlation_matrix(truth).tolist(),
            "mean": rank_correlation_matrix(mean).tolist(),
            "draw_mean": np.mean(draw_matrices, axis=0).tolist(),
            "draw_standard_deviation": np.std(draw_matrices, axis=0).tolist(),
        }
    return output


def coefficient_evolution(features: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            emulator_fit(features, targets[:, epoch], mean="linear").beta[:, 1:]
            for epoch in range(5)
        ],
        axis=0,
    )


def main_figure(
    representation: dict,
    summaries: dict,
    coefficients: np.ndarray,
    closure: dict,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.8))
    axes[0, 0].plot(
        REDSHIFTS,
        representation["quantile"],
        "o-",
        color=OKABE_ITO[0],
        label="Readable",
    )
    axes[0, 0].plot(
        REDSHIFTS, representation["pca"], "s--", color=OKABE_ITO[4], label="PCA-3"
    )
    axes[0, 0].set(
        xlabel="Redshift",
        ylabel="Median reconstruction RMS (dex)",
        title="Representation floor",
    )
    axes[0, 0].legend()

    for name, color, style in (
        ("quantile_linear", OKABE_ITO[0], "--"),
        ("quantile_poly2", OKABE_ITO[2], "-"),
        ("pca_poly2", OKABE_ITO[4], ":"),
    ):
        axes[0, 1].plot(
            REDSHIFTS,
            [value["profile_crps_dex"] for value in summaries[name]],
            marker="o",
            linestyle=style,
            color=color,
            label=name.replace("_", " "),
        )
    axes[0, 1].set(
        xlabel="Redshift",
        ylabel="Held-out profile CRPS (dex)",
        title="Predictive distribution",
    )
    axes[0, 1].legend(fontsize=7)

    image = axes[1, 0].imshow(
        coefficients[:, :, 0].T,
        aspect="auto",
        cmap="cividis",
        extent=(REDSHIFTS[0], REDSHIFTS[-1], len(TARGET_LABELS) - 0.5, -0.5),
    )
    axes[1, 0].set_yticks(range(len(TARGET_LABELS)), TARGET_LABELS)
    axes[1, 0].set(
        xlabel="Redshift",
        title=r"Standardized $\log M_{\rm peak}$ coefficient",
    )
    fig.colorbar(image, ax=axes[1, 0], label="Linear coefficient")

    closure_redshifts = [0.7, 1.0, 1.5]
    direct = [
        closure[f"z{z:.1f}"]["direct"]["profile_crps_dex"] for z in closure_redshifts
    ]
    interpolated = [
        closure[f"z{z:.1f}"]["interpolated"]["profile_crps_dex"]
        for z in closure_redshifts
    ]
    nearest = [
        closure[f"z{z:.1f}"]["nearest"]["profile_crps_dex"] for z in closure_redshifts
    ]
    x = np.arange(3)
    width = 0.25
    axes[1, 1].bar(x - width, direct, width, color=OKABE_ITO[4], label="Direct fit")
    axes[1, 1].bar(x, interpolated, width, color=OKABE_ITO[2], label="Interpolated")
    axes[1, 1].bar(x + width, nearest, width, color=OKABE_ITO[6], label="Nearest epoch")
    axes[1, 1].set_xticks(x, [f"z={z}" for z in closure_redshifts])
    axes[1, 1].set(ylabel="Profile CRPS (dex)", title="Held-epoch closure")
    axes[1, 1].legend(fontsize=7)
    for label, axis in zip("ABCD", axes.ravel()):
        axis.text(
            -0.14, 1.04, label, transform=axis.transAxes, fontsize=11, fontweight="bold"
        )
    fig.suptitle("exp51 — readable halo-to-CoG map across redshift")
    fig.tight_layout()
    save_fig(fig, FIGDIR / "exp51_redshift_summary")
    plt.close(fig)


def z2_profile_figure(
    epoch_mass: np.ndarray,
    truth: np.ndarray,
    mean: np.ndarray,
    draw: np.ndarray,
    *,
    title: str = "exp51 — descendant-conditioned z=2 CoG predictions",
    filename: str = "exp51_z2_predictions",
    draw_label: str = "One coherent draw",
) -> None:
    mass = epoch_mass[:, 4]
    edges = np.quantile(mass, [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.0), sharex=True)
    for column, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        selected = (mass >= lower) & (mass < upper if column < 2 else mass <= upper)
        axes[0, column].plot(
            RADII,
            np.median(truth[selected], axis=0),
            "o-",
            color="black",
            ms=3,
            label="TNG",
        )
        axes[0, column].plot(
            RADII,
            np.median(mean[selected], axis=0),
            "--",
            color=OKABE_ITO[0],
            label="Conditional mean",
        )
        axes[0, column].plot(
            RADII,
            np.median(draw[selected], axis=0),
            ":",
            color=OKABE_ITO[2],
            label=draw_label,
        )
        axes[0, column].set(
            xscale="log",
            title=rf"${lower:.2f}\leq\log M_h(z=2){'<' if column < 2 else r'\leq'}{upper:.2f}$",
            ylabel=r"Median log$_{10}$ M$_*(<R)$ (M$_\odot$)" if column == 0 else "",
        )
        if column == 0:
            axes[0, column].legend(fontsize=7)
        relative = (10.0 ** mean[selected] - 10.0 ** truth[selected]) / 10.0 ** truth[
            selected
        ]
        axes[1, column].axhline(0.0, color="0.5", linestyle=":")
        axes[1, column].plot(
            RADII,
            100.0 * np.median(relative, axis=0),
            color=OKABE_ITO[0],
        )
        axes[1, column].fill_between(
            RADII,
            100.0 * np.quantile(relative, 0.16, axis=0),
            100.0 * np.quantile(relative, 0.84, axis=0),
            color=OKABE_ITO[0],
            alpha=0.12,
        )
        axes[1, column].set(
            xscale="log",
            xlabel="Radius (kpc)",
            ylabel="Relative residual (percent)" if column == 0 else "",
        )
    for label, axis in zip("ABCDEF", axes.ravel()):
        axis.text(
            -0.14, 1.04, label, transform=axis.transAxes, fontsize=11, fontweight="bold"
        )
    fig.suptitle(title)
    fig.tight_layout()
    save_fig(fig, FIGDIR / filename)
    plt.close(fig)


def correlation_figure(residual: np.ndarray, rho: float, coherence: dict) -> None:
    r50 = coherence["r50"]
    matrices = [
        ("Standardized coordinate residuals", residual),
        ("Measured R50 ranks", np.asarray(r50["truth"])),
        ("Conditional-mean R50 ranks", np.asarray(r50["mean"])),
        ("Coherent-draw R50 ranks", np.asarray(r50["draw_mean"])),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14.5, 3.6))
    for label, axis, (title, matrix) in zip("ABCD", axes, matrices):
        axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="cividis")
        axis.set_xticks(range(5), REDSHIFTS)
        axis.set_yticks(range(5), REDSHIFTS)
        axis.set(xlabel="Redshift", ylabel="Redshift", title=title)
        axis.text(
            -0.16,
            1.05,
            label,
            transform=axis.transAxes,
            fontsize=11,
            fontweight="bold",
        )
        for row in range(5):
            for column in range(5):
                color = "white" if matrix[row, column] < 0.55 else "black"
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]:.2f}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=7,
                )
    fig.suptitle(f"exp51 — cross-epoch coherence; residual AR(1) rho={rho:.2f}")
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.16, top=0.80, wspace=0.35)
    save_fig(fig, FIGDIR / "exp51_cross_epoch_coherence")
    plt.close(fig)


def coefficient_figure(coefficients: np.ndarray) -> None:
    """Show every standardized linear coefficient across redshift."""
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.0), sharex=True)
    colors = [OKABE_ITO[4], OKABE_ITO[1], OKABE_ITO[2], OKABE_ITO[6], OKABE_ITO[0]]
    for target, axis in enumerate(axes.ravel()[:5]):
        for feature, (name, color) in enumerate(zip(FEATURE_NAMES, colors)):
            axis.plot(
                REDSHIFTS,
                coefficients[:, target, feature],
                "o-",
                color=color,
                label=name,
            )
        axis.axhline(0.0, color="0.5", linestyle=":")
        axis.set(
            xlabel="Redshift" if target >= 2 else "",
            ylabel="Standardized linear coefficient" if target in (0, 3) else "",
            title=TARGET_LABELS[target],
        )
        if target == 0:
            axis.legend(fontsize=7, ncol=2)
    axes.ravel()[5].axis("off")
    for label, axis in zip("ABCDE", axes.ravel()[:5]):
        axis.text(
            -0.14,
            1.04,
            label,
            transform=axis.transAxes,
            fontsize=11,
            fontweight="bold",
        )
    fig.suptitle("exp51 — evolution of the readable halo-to-profile coefficients")
    fig.tight_layout()
    save_fig(fig, FIGDIR / "exp51_coefficient_evolution")
    plt.close(fig)


def serializable_epoch_summaries(summaries: dict) -> dict:
    return {
        name: {f"z{redshift:.1f}": value for redshift, value in zip(REDSHIFTS, values)}
        for name, values in summaries.items()
    }


def fit_experiment() -> None:
    features, cogs_log, indices, epoch_mass, population_rows = load_sample()
    print(f"exp51 descendant-conditioned five-epoch map: n={len(features)}")
    print(f"  redshifts: {REDSHIFTS.tolist()}, coherent draws: {N_DRAW}")

    representation, _ = representation_summary(cogs_log)
    targets, results, summaries = fit_independent_epochs(features, cogs_log)
    controls = z2_controls(features, epoch_mass, targets[:, 4], cogs_log[:, 4])
    closure, _ = held_epoch_closure(
        features, targets, cogs_log, summaries["quantile_poly2"]
    )
    residual, residual_by_target, rho = residual_correlation(
        targets, results["quantile_poly2"]
    )
    draws = coherent_draws(features, targets, rho)
    mean_log = np.stack(
        [result["mean_cogs"] for result in results["quantile_poly2"]], axis=1
    )
    coherence = coherence_summary(cogs_log, mean_log, draws)
    coefficients = coefficient_evolution(features, targets)

    parity = []
    print("\n[per-epoch readable map versus PCA]")
    for epoch, redshift in enumerate(REDSHIFTS):
        readable = summaries["quantile_poly2"][epoch]
        pca = summaries["pca_poly2"][epoch]
        fold_spread = float(np.std(pca["profile_crps_by_fold"], ddof=1))
        difference = readable["profile_crps_dex"] - pca["profile_crps_dex"]
        passed = difference <= fold_spread
        parity.append(
            {
                "redshift": float(redshift),
                "readable_crps_dex": readable["profile_crps_dex"],
                "pca_crps_dex": pca["profile_crps_dex"],
                "difference_dex": difference,
                "pca_fold_spread_dex": fold_spread,
                "passed": bool(passed),
            }
        )
        print(
            f"  z={redshift:.1f}: readable CRPS {readable['profile_crps_dex']:.5f} dex; "
            f"PCA {pca['profile_crps_dex']:.5f} dex; difference {difference:+.5f}; "
            f"PCA fold spread {fold_spread:.5f}; {'PASS' if passed else 'FAIL'}"
        )

    print("\n[z=2 assembly controls: median profile RMS]")
    for name, summary in controls.items():
        print(f"  {name:32s}: {summary['median_profile_rms_dex']:.5f} dex")
    print(f"\n[cross-epoch standardized-residual AR(1)] rho={rho:.4f}")
    print("\n[held-epoch closure]")
    for key, values in closure.items():
        direct = values["direct"]["profile_crps_dex"]
        interpolated = values["interpolated"]["profile_crps_dex"]
        nearest = values["nearest"]["profile_crps_dex"]
        print(
            f"  {key}: direct {direct:.5f}, interpolated {interpolated:.5f} "
            f"({100.0 * (interpolated / direct - 1.0):+.2f}%), nearest {nearest:.5f}"
        )

    main_figure(representation, summaries, coefficients, closure)
    coefficient_figure(coefficients)
    z2_profile_figure(
        epoch_mass,
        cogs_log[:, 4],
        results["quantile_poly2"][4]["mean_cogs"],
        draws[0, :, 4],
    )
    correlation_figure(residual, rho, coherence)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_galaxy": len(features),
        "redshifts": REDSHIFTS.tolist(),
        "n_draw": N_DRAW,
        "prediction_scope": "descendant-conditioned complete DiffMAH and z=0.4 concentration",
        "representation_median_profile_rms_dex": representation,
        "models": serializable_epoch_summaries(summaries),
        "parity": parity,
        "z2_controls": controls,
        "held_epoch_closure": closure,
        "residual_correlation": residual.tolist(),
        "residual_correlation_by_target": residual_by_target.tolist(),
        "residual_ar1_rho": rho,
        "coherence": coherence,
        "linear_coefficients": coefficients.tolist(),
        "feature_names": FEATURE_NAMES,
        "target_labels": TARGET_LABELS,
    }
    (OUTDIR / "results.json").write_text(json.dumps(payload, indent=2))
    np.savez_compressed(
        OUTDIR / "predictions.npz",
        indices=indices,
        population_rows=population_rows,
        features=features,
        epoch_halo_mass=epoch_mass,
        truth_cogs_log=cogs_log,
        targets=targets,
        mean_cogs_log=mean_log,
        coherent_draw_cogs_log=draws,
    )
    write_manifest(
        OUTDIR,
        {
            "n_galaxy": len(features),
            "redshifts": REDSHIFTS.tolist(),
            "n_draw": N_DRAW,
            "residual_ar1_rho": rho,
            "parity_passed": [value["passed"] for value in parity],
        },
    )
    print(f"\nwrote figures to {FIGDIR}")
    print(f"wrote outputs to {OUTDIR}")


def demo() -> None:
    EXP51_MODEL.demo()
    features, cogs_log, _, _, _ = load_sample(100)
    targets = np.stack(
        [compress_cogs(cogs_log[:, epoch], RADII, "quantile3") for epoch in range(5)],
        axis=1,
    )
    result = cv_quantile(features, targets[:, 4], "quantile3", mean="linear", n_draw=3)
    assert np.isfinite(result["mean_cogs"]).all()
    assert np.all(np.diff(result["mean_cogs"], axis=1) >= 0.0)
    average, by_target, rho = residual_correlation(targets, [result] * 5)
    assert average.shape == (5, 5) and by_target.shape == (5, 5, 5)
    assert 0.0 < rho <= 1.0
    print("exp51 mechanics and 100-galaxy z=2 fit OK")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "fit"
    if command == "demo":
        demo()
    elif command == "fit":
        fit_experiment()
    else:
        raise SystemExit(f"unknown command {command!r}; use demo or fit")
