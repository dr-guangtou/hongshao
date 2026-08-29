# %%
"""Exp62 Stage 3: scaling relations and matched-track smooth evolution.

Commands:

    uv run python experiments/exp62_cog_fit_atlas/stage3.py demo
    uv run python experiments/exp62_cog_fit_atlas/stage3.py full

The demo uses 90 halo-mass-stratified matched galaxies.  The full calculation
uses every quality-clean track available in the Exp51 bridge artifact.
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
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from families import FAMILY_SPECS, cog_log  # noqa: E402
from hongshao.plotting import OKABE_ITO, save_fig, set_style  # noqa: E402
from hongshao.profile_emulator import density_from_cog  # noqa: E402
from hongshao.provenance import write_manifest  # noqa: E402
from hongshao.qa import enclosed_radius  # noqa: E402
from hongshao.tng_data import ANCHORS, load_cosmic_time  # noqa: E402
from run import FIGDIR, OUTDIR, RADII, REDSHIFTS, profile_metrics  # noqa: E402
from stage2 import ROLES, cell_path  # noqa: E402

set_style()

OBJECTIVE_NAMES = ("cog_log", "cog_absolute")
HELD_EPOCHS = (1, 2, 3)
TIME_DEGREES = (0, 1, 2)
SOURCE_TRACKS = OUTDIR / "source" / "exp51_predictions.npz"
SOURCE_CONCENTRATION = OUTDIR / "source" / "exp51_concentration_sample.npz"
N_DEMO = 90

DECODED_NAMES = (
    "log_mstar_148",
    "log_fraction_inside_2",
    "log_r20",
    "log_r50",
    "log_r80",
    "log_r90",
    "log_mass_2_10",
    "log_mass_10_30",
    "log_mass_30_50",
    "log_mass_50_100",
    "log_mass_100_148",
    "cog_slope_2_10",
    "cog_slope_10_30",
    "cog_slope_30_100",
    "density_slope_2_10",
    "density_slope_10_30",
    "density_slope_30_100",
)


def _mass_at_radius(cogs_log: np.ndarray, radius: float) -> np.ndarray:
    log_radius = np.log10(RADII)
    target = np.log10(radius)
    return np.asarray([np.interp(target, log_radius, row) for row in cogs_log])


def _log_annulus_mass(cogs_log: np.ndarray, inner: float, outer: float):
    inner_mass = 10.0 ** _mass_at_radius(cogs_log, inner)
    outer_mass = 10.0 ** _mass_at_radius(cogs_log, outer)
    return np.log10(np.clip(outer_mass - inner_mass, 1.0, None))


def _fixed_slope(values: np.ndarray, radius: np.ndarray, low: float, high: float):
    selected = (radius >= low) & (radius <= high)
    x = np.log10(radius[selected])
    centered = x - np.mean(x)
    return (values[:, selected] @ centered) / np.sum(centered**2)


def decoded_observables(cogs_log: np.ndarray) -> np.ndarray:
    """Common physical summaries decoded from any analytic CoG."""
    cogs_log = np.asarray(cogs_log, float)
    linear = 10.0**cogs_log
    total = cogs_log[:, -1]
    size = np.column_stack(
        [
            np.log10([enclosed_radius(row, RADII, fraction) for row in linear])
            for fraction in (0.2, 0.5, 0.8, 0.9)
        ]
    )
    annuli = np.column_stack(
        [
            _log_annulus_mass(cogs_log, inner, outer)
            for inner, outer in (
                (2.0, 10.0),
                (10.0, 30.0),
                (30.0, 50.0),
                (50.0, 100.0),
                (100.0, RADII[-1]),
            )
        ]
    )
    cog_slopes = np.column_stack(
        [
            _fixed_slope(cogs_log, RADII, low, high)
            for low, high in ((2.0, 10.0), (10.0, 30.0), (30.0, 100.0))
        ]
    )
    density_log, density_radius = density_from_cog(cogs_log, RADII)
    density_slopes = np.column_stack(
        [
            _fixed_slope(density_log, density_radius, low, high)
            for low, high in ((2.0, 10.0), (10.0, 30.0), (30.0, 100.0))
        ]
    )
    return np.column_stack(
        [
            total,
            _mass_at_radius(cogs_log, 2.0) - total,
            size,
            annuli,
            cog_slopes,
            density_slopes,
        ]
    )


def _atlas_row_map(atlas) -> np.ndarray:
    rows = np.asarray(atlas["rows"], int)
    epochs = np.asarray(atlas["epochs"], int)
    output = np.full((int(rows.max()) + 1, len(REDSHIFTS)), -1, int)
    output[rows, epochs] = np.arange(len(rows))
    return output


def _demo_indices(halo_mass: np.ndarray) -> np.ndarray:
    order = np.argsort(halo_mass[:, 0])
    positions = np.linspace(0, len(order) - 1, N_DEMO).round().astype(int)
    return order[positions]


def load_tracks(mode: str):
    atlas = np.load(OUTDIR / "full" / "atlas" / "predictions.npz")
    source = np.load(SOURCE_TRACKS)
    population = np.load(OUTDIR / "population.npz")
    galaxy_ids = np.asarray(source["indices"], int)
    select = np.arange(len(galaxy_ids))
    if mode == "demo":
        select = _demo_indices(np.asarray(source["epoch_halo_mass"]))
    galaxy_ids = galaxy_ids[select]
    atlas_rows = _atlas_row_map(atlas)[galaxy_ids]
    if np.any(atlas_rows < 0):
        raise ValueError("a matched track is absent from the full Stage 1 atlas")

    concentration = np.full((len(galaxy_ids), len(REDSHIFTS)), np.nan)
    concentration_source = np.load(SOURCE_CONCENTRATION)
    concentration_lookup = {
        int(galaxy): row
        for row, galaxy in enumerate(np.asarray(concentration_source["indices"], int))
    }
    for row, galaxy in enumerate(galaxy_ids):
        if int(galaxy) in concentration_lookup:
            concentration[row] = concentration_source["concentration_history"][
                concentration_lookup[int(galaxy)]
            ]

    truth_log = np.asarray(population["cogs_log"])[galaxy_ids]
    halo_mass = np.asarray(source["epoch_halo_mass"])[select]
    return {
        "atlas": atlas,
        "atlas_rows": atlas_rows,
        "galaxy_ids": galaxy_ids,
        "truth_log": truth_log,
        "halo_mass": halo_mass,
        "concentration": concentration,
    }


def load_relation_tracks(role: str, objective_name: str, atlas_rows: np.ndarray):
    stored = np.load(
        cell_path("full", role, "cog", objective_name.removeprefix("cog_"))
    )
    return {
        "parameters": np.asarray(stored["parameters"])[atlas_rows],
        "prediction_log": np.asarray(stored["prediction_log"])[atlas_rows],
        "variants": np.asarray(stored["variants"]).astype(str)[atlas_rows],
    }


def _rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 10:
        return np.nan
    ranked_x = rankdata(x[valid])
    ranked_y = rankdata(y[valid])
    return float(np.corrcoef(ranked_x, ranked_y)[0, 1])


def _scaling_statistics(decoded: np.ndarray, predictors: np.ndarray) -> np.ndarray:
    output = np.full((predictors.shape[-1], len(REDSHIFTS), decoded.shape[-1]), np.nan)
    for predictor in range(predictors.shape[-1]):
        for epoch in range(len(REDSHIFTS)):
            for target in range(decoded.shape[-1]):
                output[predictor, epoch, target] = _rank_correlation(
                    predictors[:, epoch, predictor], decoded[:, epoch, target]
                )
    return output


def _binned_trend(x: np.ndarray, y: np.ndarray, n_bin: int = 6):
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, n_bin + 1)))
    center, median, lower, upper = [], [], [], []
    for low, high in zip(edges[:-1], edges[1:]):
        selected = (x >= low) & (x <= high)
        center.append(np.median(x[selected]))
        median.append(np.median(y[selected]))
        lower.append(np.quantile(y[selected], 0.16))
        upper.append(np.quantile(y[selected], 0.84))
    return map(np.asarray, (center, median, lower, upper))


def _polynomial_prediction(
    values: np.ndarray, times: np.ndarray, held: int, degree: int
):
    keep = np.arange(len(times)) != held
    design = np.vander(times[keep], degree + 1, increasing=True)
    flattened = values[:, keep].reshape(len(values), keep.sum(), -1)
    coefficient = np.linalg.lstsq(
        design, flattened.transpose(1, 0, 2).reshape(keep.sum(), -1), rcond=None
    )[0]
    target = np.vander(np.asarray([times[held]]), degree + 1, increasing=True)[0]
    return (target @ coefficient).reshape(len(values), *values.shape[2:])


def _decode_parameters(
    variants: np.ndarray, parameters: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    prediction = np.empty((len(parameters), len(RADII)))
    clipped = np.zeros(len(parameters), bool)
    for row, (family, values) in enumerate(zip(variants, parameters)):
        spec = FAMILY_SPECS[str(family)]
        lower = np.asarray(spec.lower)
        upper = np.asarray(spec.upper)
        bounded = np.clip(values, lower + 1.0e-8, upper - 1.0e-8)
        clipped[row] = not np.array_equal(bounded, values)
        prediction[row] = cog_log(str(family), bounded, RADII)
    return prediction, clipped


def evaluate_evolution(
    relation: dict,
    truth_log: np.ndarray,
    times: np.ndarray,
):
    parameters = relation["parameters"]
    direct = relation["prediction_log"]
    variants = relation["variants"]
    n_galaxy = len(parameters)
    n_held = len(HELD_EPOCHS)
    output = {
        "truth": truth_log[:, HELD_EPOCHS],
        "direct": direct[:, HELD_EPOCHS],
        "nearest": np.empty((n_galaxy, n_held, len(RADII))),
        "profile_degree_2": np.empty((n_galaxy, n_held, len(RADII))),
        "truth_degree_2": np.empty((n_galaxy, n_held, len(RADII))),
    }
    clip_fraction = {}
    for position, held in enumerate(HELD_EPOCHS):
        available = [epoch for epoch in range(len(times)) if epoch != held]
        nearest = min(available, key=lambda epoch: abs(times[epoch] - times[held]))
        output["nearest"][:, position] = direct[:, nearest]
        output["profile_degree_2"][:, position] = _polynomial_prediction(
            direct, times, held, 2
        )
        output["truth_degree_2"][:, position] = _polynomial_prediction(
            truth_log, times, held, 2
        )
        for degree in TIME_DEGREES:
            key = f"degree_{degree}"
            output.setdefault(key, np.empty((n_galaxy, n_held, len(RADII))))
            interpolated = _polynomial_prediction(parameters, times, held, degree)
            decoded, clipped = _decode_parameters(variants[:, held], interpolated)
            output[key][:, position] = decoded
            clip_fraction[f"{key}_z{REDSHIFTS[held]:g}"] = float(np.mean(clipped))

    summary = {"clip_fraction": clip_fraction, "methods": {}}
    flattened_truth = output["truth"].reshape(-1, len(RADII))
    for method in (
        "direct",
        "nearest",
        "degree_0",
        "degree_1",
        "degree_2",
        "profile_degree_2",
        "truth_degree_2",
    ):
        flattened = output[method].reshape(-1, len(RADII))
        metrics = profile_metrics(flattened, flattened_truth)
        by_epoch = {}
        for position, held in enumerate(HELD_EPOCHS):
            # The flattening order is galaxy-major; use an explicit epoch mask.
            mask = np.tile(np.arange(n_held), n_galaxy) == position
            by_epoch[f"z{REDSHIFTS[held]:g}"] = {
                "median_full_cog_rms_dex": float(
                    np.median(metrics["cog_rms_full_dex"][mask])
                ),
                "median_density_rms_dex": float(
                    np.median(metrics["density_rms_dex"][mask])
                ),
                "median_absolute_r50_error_dex": float(
                    np.median(np.abs(metrics["size_error_dex"][mask, 1]))
                ),
                "median_absolute_r90_error_dex": float(
                    np.median(np.abs(metrics["size_error_dex"][mask, 3]))
                ),
            }
        summary["methods"][method] = {
            "median_full_cog_rms_dex": float(np.median(metrics["cog_rms_full_dex"])),
            "median_density_rms_dex": float(np.median(metrics["density_rms_dex"])),
            "median_absolute_r50_error_dex": float(
                np.median(np.abs(metrics["size_error_dex"][:, 1]))
            ),
            "median_absolute_r90_error_dex": float(
                np.median(np.abs(metrics["size_error_dex"][:, 3]))
            ),
            "by_epoch": by_epoch,
        }
    return output, summary


def coordinate_smoothness(values: np.ndarray, times: np.ndarray):
    errors = []
    for held in HELD_EPOCHS:
        prediction = _polynomial_prediction(values, times, held, 2)
        scale = np.subtract(*np.quantile(values[:, held], [0.75, 0.25], axis=0))
        errors.append(
            np.median(np.abs(prediction - values[:, held]), axis=0)
            / np.clip(scale, 1.0e-8, None)
        )
    return np.asarray(errors)


def scaling_figure(
    mode: str,
    role: str,
    decoded: np.ndarray,
    predictors: np.ndarray,
):
    target_indices = (1, 3, 5, 9)
    target_labels = (
        r"$\log[M_*(<2)/M_{*,148}]$",
        r"$\log R_{50}$",
        r"$\log R_{90}$",
        r"$\log M_*(50\!-\!100\,\mathrm{kpc})$",
    )
    predictor_labels = (
        r"$\log M_{*,148}$",
        r"$\log M_{200c}$",
        r"$c_{200c}$",
    )
    figure, axes = plt.subplots(3, 4, figsize=(14.0, 9.0), sharex="row")
    colors = (OKABE_ITO[0], OKABE_ITO[1], OKABE_ITO[2], OKABE_ITO[4], OKABE_ITO[5])
    for predictor in range(3):
        for column, target in enumerate(target_indices):
            axis = axes[predictor, column]
            for epoch, (redshift, color) in enumerate(zip(REDSHIFTS, colors)):
                x, median, lower, upper = _binned_trend(
                    predictors[:, epoch, predictor], decoded[:, epoch, target]
                )
                axis.plot(x, median, color=color, label=rf"$z={redshift:g}$")
                axis.fill_between(x, lower, upper, color=color, alpha=0.10)
            if predictor == 0:
                axis.set_title(target_labels[column])
            if column == 0:
                axis.set_ylabel("Decoded CoG observable")
            axis.set_xlabel(predictor_labels[predictor])
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=2)
    figure.suptitle(
        f"exp62 Stage 3 {mode} — {role.replace('_', ' ')}, log-CoG fit: "
        r"median scaling relations and 16–84\% spread"
    )
    figure.tight_layout()
    save_fig(figure, FIGDIR / f"stage3_{mode}" / f"exp62_scaling_{role}")
    plt.close(figure)


def evolution_figure(mode: str, summaries: dict):
    figure, axes = plt.subplots(len(ROLES), len(OBJECTIVE_NAMES), figsize=(11.0, 10.0))
    methods = (
        "direct",
        "nearest",
        "degree_1",
        "degree_2",
        "profile_degree_2",
        "truth_degree_2",
    )
    labels = (
        "Direct epoch fit",
        "Nearest analytic epoch",
        "Linear analytic parameters",
        "Quadratic analytic parameters",
        "Quadratic analytic CoG values",
        "Quadratic measured CoG values",
    )
    colors = (
        "black",
        OKABE_ITO[7],
        OKABE_ITO[2],
        OKABE_ITO[4],
        OKABE_ITO[0],
        OKABE_ITO[1],
    )
    markers = ("o", "X", "^", "D", "s", "v")
    for row, role in enumerate(ROLES):
        for column, objective_name in enumerate(OBJECTIVE_NAMES):
            axis = axes[row, column]
            summary = summaries[f"{role}_{objective_name}"]["methods"]
            for method, label, color, marker in zip(methods, labels, colors, markers):
                values = [
                    summary[method]["by_epoch"][f"z{REDSHIFTS[held]:g}"][
                        "median_full_cog_rms_dex"
                    ]
                    for held in HELD_EPOCHS
                ]
                axis.plot(
                    REDSHIFTS[list(HELD_EPOCHS)],
                    values,
                    marker=marker,
                    color=color,
                    label=label,
                )
            axis.set_title(
                f"{role.replace('_', ' ')} — {objective_name.replace('_', ' ')}"
            )
            axis.set_ylabel("Median held-epoch CoG RMS (dex)")
            axis.set_xlabel("Held redshift")
    axes[0, 0].legend(frameon=False, fontsize=7)
    figure.suptitle(
        f"exp62 Stage 3 {mode} — leave-one-interior-epoch-out decoded-CoG closure"
    )
    figure.tight_layout()
    save_fig(figure, FIGDIR / f"stage3_{mode}" / "exp62_evolution_closure")
    plt.close(figure)


def standard_qa_figure(
    mode: str,
    label: str,
    prediction: np.ndarray,
    truth: np.ndarray,
    halo_mass: np.ndarray,
):
    figure, axes = plt.subplots(len(HELD_EPOCHS), 2, figsize=(10.5, 10.5))
    colors = (OKABE_ITO[0], OKABE_ITO[2], OKABE_ITO[4])
    for position, held in enumerate(HELD_EPOCHS):
        mass = halo_mass[:, held]
        edges = np.quantile(mass, [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
        for mass_bin, color in enumerate(colors):
            selected = (mass >= edges[mass_bin]) & (mass <= edges[mass_bin + 1])
            truth_shape = truth[selected, position] - truth[selected, position, -1:]
            model_shape = (
                prediction[selected, position] - prediction[selected, position, -1:]
            )
            axes[position, 0].plot(
                RADII,
                np.median(truth_shape, axis=0),
                ":",
                color=color,
            )
            axes[position, 0].plot(
                RADII,
                np.median(model_shape, axis=0),
                color=color,
                label=rf"${edges[mass_bin]:.2f}<\log M_h<{edges[mass_bin + 1]:.2f}$",
            )
        truth_inner = _mass_at_radius(truth[:, position], 30.0)
        model_inner = _mass_at_radius(prediction[:, position], 30.0)
        truth_outer = _log_annulus_mass(truth[:, position], 50.0, 100.0)
        model_outer = _log_annulus_mass(prediction[:, position], 50.0, 100.0)
        axes[position, 1].scatter(
            truth_inner,
            truth_outer,
            color="black",
            s=6,
            alpha=0.22,
            label="TNG",
        )
        axes[position, 1].scatter(
            model_inner,
            model_outer,
            color=OKABE_ITO[2],
            s=6,
            alpha=0.22,
            label="time-interpolated analytic CoG",
        )
        axes[position, 0].set_xscale("log")
        axes[position, 0].set_ylabel(
            rf"$z={REDSHIFTS[held]:g}$  $\log[M(<R)/M_{{148}}]$"
        )
        axes[position, 1].set_ylabel(r"$\log M_*(50\!-\!100\,\mathrm{kpc})$")
    axes[0, 0].legend(frameon=False, fontsize=7)
    axes[0, 1].legend(frameon=False, fontsize=7)
    axes[-1, 0].set_xlabel("Semi-major axis (kpc)")
    axes[-1, 1].set_xlabel(r"$\log M_*(<30\,\mathrm{kpc})$")
    figure.suptitle(f"exp62 Stage 3 {mode} — {label}; quadratic-in-time held-epoch QA")
    figure.tight_layout()
    save_fig(figure, FIGDIR / f"stage3_{mode}" / "exp62_stage3_standard_qa")
    plt.close(figure)


def run(mode: str):
    start = time.perf_counter()
    tracks = load_tracks(mode)
    snapshots = [ANCHORS[float(redshift)][2] for redshift in REDSHIFTS]
    times = load_cosmic_time()[snapshots]
    truth_decoded = decoded_observables(
        tracks["truth_log"].reshape(-1, len(RADII))
    ).reshape(len(tracks["galaxy_ids"]), len(REDSHIFTS), -1)
    predictors = np.stack(
        [truth_decoded[..., 0], tracks["halo_mass"], tracks["concentration"]],
        axis=-1,
    )

    summaries = {}
    evolution_outputs = {}
    payload = {
        "galaxy_ids": tracks["galaxy_ids"],
        "redshifts": REDSHIFTS,
        "cosmic_time_gyr": times,
        "decoded_names": np.asarray(DECODED_NAMES),
        "predictor_names": np.asarray(("log_mstar_148", "log_halo_mass", "c_200c")),
        "truth_decoded": truth_decoded,
        "predictors": predictors,
    }
    for role in ROLES:
        for objective_name in OBJECTIVE_NAMES:
            key = f"{role}_{objective_name}"
            relation = load_relation_tracks(role, objective_name, tracks["atlas_rows"])
            decoded = decoded_observables(
                relation["prediction_log"].reshape(-1, len(RADII))
            ).reshape(len(tracks["galaxy_ids"]), len(REDSHIFTS), -1)
            evolution, summary = evaluate_evolution(
                relation, tracks["truth_log"], times
            )
            summaries[key] = summary
            evolution_outputs[key] = evolution
            spec = FAMILY_SPECS[str(relation["variants"][0, 0])]
            payload[f"{key}_parameter_names"] = np.asarray(spec.parameter_names)
            payload[f"{key}_parameters"] = relation["parameters"]
            payload[f"{key}_variants"] = relation["variants"]
            payload[f"{key}_decoded"] = decoded
            payload[f"{key}_decoded_rank_correlation"] = _scaling_statistics(
                decoded, predictors
            )
            payload[f"{key}_raw_quadratic_scaled_error"] = coordinate_smoothness(
                relation["parameters"], times
            )
            payload[f"{key}_decoded_quadratic_scaled_error"] = coordinate_smoothness(
                decoded, times
            )
            payload[f"{key}_held_truth"] = evolution["truth"]
            for method in (
                "direct",
                "nearest",
                "degree_0",
                "degree_1",
                "degree_2",
                "profile_degree_2",
                "truth_degree_2",
            ):
                payload[f"{key}_{method}"] = evolution[method]
            if objective_name == "cog_log":
                scaling_figure(mode, role, decoded, predictors)
            print(
                f"  {key:<38} direct {summary['methods']['direct']['median_full_cog_rms_dex']:.5f} dex; "
                f"quadratic parameters {summary['methods']['degree_2']['median_full_cog_rms_dex']:.5f} dex; "
                f"quadratic CoG {summary['methods']['profile_degree_2']['median_full_cog_rms_dex']:.5f} dex",
                flush=True,
            )

    evolution_figure(mode, summaries)
    best_key = min(
        summaries,
        key=lambda key: summaries[key]["methods"]["degree_2"][
            "median_full_cog_rms_dex"
        ],
    )
    standard_qa_figure(
        mode,
        best_key.replace("_", " "),
        evolution_outputs[best_key]["degree_2"],
        evolution_outputs[best_key]["truth"],
        tracks["halo_mass"],
    )
    result = {
        "mode": mode,
        "n_track": len(tracks["galaxy_ids"]),
        "n_concentration_track": int(
            np.all(np.isfinite(tracks["concentration"]), axis=1).sum()
        ),
        "held_epochs": [float(REDSHIFTS[epoch]) for epoch in HELD_EPOCHS],
        "best_quadratic_relation": best_key,
        "relations": summaries,
        "wall_seconds": time.perf_counter() - start,
    }
    result["subminute_gate_passed"] = (
        bool(result["wall_seconds"] < 60.0) if mode == "demo" else None
    )
    output_dir = OUTDIR / mode / "stage3"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "scaling_and_evolution.npz", **payload)
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    write_manifest(
        output_dir,
        {
            "experiment": "exp62_cog_fit_atlas",
            "stage": "3_scaling_and_evolution",
            "mode": mode,
            "n_track": len(tracks["galaxy_ids"]),
            "objectives": OBJECTIVE_NAMES,
            "roles": ROLES,
        },
    )
    print(json.dumps(result, indent=2))
    if mode == "demo" and result["wall_seconds"] >= 60.0:
        raise RuntimeError(
            f"Stage 3 demo took {result['wall_seconds']:.2f} seconds; full run is forbidden"
        )


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if command not in ("demo", "full"):
        raise SystemExit(__doc__)
    run(command)
