"""exp51 epoch-local test using DiffMAH fits truncated at each profile epoch.

For a profile at redshift z_k, each DiffMAH fit sees only the official
main-branch Mpeak measurements at cosmic times t <= t(z_k). This makes the
input usable for an independently selected halo at that epoch and prevents
future descendant growth from entering its MAH parameters.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from hongshao.diffmah import fit_mah  # noqa: E402
from hongshao.plotting import OKABE_ITO, save_fig, set_style  # noqa: E402
from hongshao.provenance import write_manifest  # noqa: E402
from hongshao.tng_data import H  # noqa: E402
from concentration import load_concentration_sample, shuffled_within_epoch_mass  # noqa: E402
from run import (  # noqa: E402
    N_MAX,
    RADII,
    REDSHIFTS,
    compress_cogs,
    cv_quantile,
    source_output,
    summarize_model,
    z2_profile_figure,
)

set_style()

FIGDIR = HERE / "figures"
OUTDIR = HERE / "outputs" / "epoch_local"
N_DRAW = int(os.environ.get("EXP51_LOCAL_N_DRAW", 32))
LOCAL_N_MAX = int(os.environ.get("EXP51_LOCAL_NMAX", N_MAX))
T_MIN_GYR = 1.0
SNAPSHOTS = np.array([72, 59, 50, 40, 33])
PARAMETER_NAMES = ["logmp", "logtc", "early", "late"]


def feature_cache_path() -> Path:
    suffix = f"_n{LOCAL_N_MAX}" if LOCAL_N_MAX else ""
    return OUTDIR / f"truncated_diffmah{suffix}.npz"


def fit_epoch_local_features(force: bool = False):
    cache = feature_cache_path()
    if cache.exists() and not force:
        return np.load(cache)
    features, cogs_log, indices, epoch_mass, concentration = load_concentration_sample()
    if LOCAL_N_MAX and LOCAL_N_MAX < len(features):
        order = np.argsort(features[:, 0])
        positions = np.linspace(0, len(features) - 1, LOCAL_N_MAX).round().astype(int)
        selected = order[positions]
        features, cogs_log, indices, epoch_mass, concentration = (
            features[selected],
            cogs_log[selected],
            indices[selected],
            epoch_mass[selected],
            concentration[selected],
        )
    official = np.load(source_output("exp27_tng_api_crossmatch", "official_mah.npz"))
    official_row = {int(index): row for row, index in enumerate(official["index"])}
    cosmic_time = np.asarray(official["cosmic_time_gyr"], float)
    mass_unit_correction = np.log10(1.0 / H)
    parameters = np.full((len(indices), 5, 4), np.nan)
    rms = np.full((len(indices), 5), np.nan)
    success = np.zeros((len(indices), 5), bool)
    start = time.perf_counter()
    for galaxy, index in enumerate(indices):
        row = official_row.get(int(index))
        if row is None or not official["matched"][row]:
            continue
        history = np.asarray(official["diffmah_log_mah_sim"][row], float)
        for epoch, snapshot in enumerate(SNAPSHOTS):
            use = np.isfinite(history)
            use &= np.arange(len(history)) <= snapshot
            use &= cosmic_time >= T_MIN_GYR
            result = fit_mah(
                cosmic_time[use],
                history[use] + mass_unit_correction,
                cosmic_time[snapshot],
                t_min=T_MIN_GYR,
            )
            parameters[galaxy, epoch] = [result[name] for name in PARAMETER_NAMES]
            rms[galaxy, epoch] = result["rms"]
            success[galaxy, epoch] = result["success"]
    elapsed = time.perf_counter() - start
    keep = success.all(axis=1) & np.isfinite(parameters).all(axis=(1, 2))
    OUTDIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        indices=indices[keep],
        descendant_features=features[keep],
        cogs_log=cogs_log[keep],
        epoch_halo_mass=epoch_mass[keep],
        concentration=concentration[keep],
        parameters=parameters[keep],
        rms=rms[keep],
        fit_seconds=elapsed,
        n_attempted=len(indices),
    )
    print(
        f"fitted truncated DiffMAH at five epochs for {len(indices)} halos in "
        f"{elapsed:.2f} seconds; {keep.sum()} have five valid fits"
    )
    return np.load(cache)


def fit_experiment() -> None:
    data = fit_epoch_local_features()
    descendant = np.asarray(data["descendant_features"], float)
    cogs_log = np.asarray(data["cogs_log"], float)
    epoch_mass = np.asarray(data["epoch_halo_mass"], float)
    concentration = np.asarray(data["concentration"], float)
    local_parameters = np.asarray(data["parameters"], float)
    fit_rms = np.asarray(data["rms"], float)
    print(
        f"exp51 epoch-local DiffMAH test: n={len(descendant)}, draws={N_DRAW}; "
        f"feature fits took {float(data['fit_seconds']):.2f} seconds"
    )
    summaries = {}
    z2_local_result = None
    for epoch, redshift in enumerate(REDSHIFTS):
        print(f"  fitting epoch-local prediction variants at z={redshift:.1f}")
        targets = compress_cogs(cogs_log[:, epoch], RADII, "quantile3")
        local_diffmah = local_parameters[:, epoch]
        local_complete = np.column_stack([local_diffmah, concentration[:, epoch]])
        local_shape_shuffled = shuffled_within_epoch_mass(
            local_complete,
            epoch_mass[:, epoch],
            (1, 2, 3),
            seed=800 + epoch,
        )
        local_c_shuffled = shuffled_within_epoch_mass(
            local_complete,
            epoch_mass[:, epoch],
            (4,),
            seed=900 + epoch,
        )
        descendant_concurrent = np.column_stack(
            [descendant[:, :4], concentration[:, epoch]]
        )
        definitions = {
            "epoch_mass_only": epoch_mass[:, epoch : epoch + 1],
            "local_diffmah": local_diffmah,
            "local_diffmah_concurrent_c": local_complete,
            "local_shape_shuffled": local_shape_shuffled,
            "local_c_shuffled": local_c_shuffled,
            "descendant_diffmah_concurrent_c": descendant_concurrent,
        }
        summaries[f"z{redshift:.1f}"] = {}
        for name, model_features in definitions.items():
            result = cv_quantile(
                model_features,
                targets,
                "quantile3",
                mean="poly2" if model_features.shape[1] >= 5 else "linear",
                n_draw=N_DRAW,
            )
            summaries[f"z{redshift:.1f}"][name] = summarize_model(
                name, result, cogs_log[:, epoch]
            )
            if epoch == 4 and name == "local_diffmah_concurrent_c":
                z2_local_result = result

    gains = {}
    print("\n[epoch-local MAH and concurrent-concentration gains]")
    for redshift in REDSHIFTS:
        key = f"z{redshift:.1f}"
        values = summaries[key]
        mass = values["epoch_mass_only"]["profile_crps_dex"]
        local = values["local_diffmah_concurrent_c"]["profile_crps_dex"]
        shape_shuffle = values["local_shape_shuffled"]["profile_crps_dex"]
        c_shuffle = values["local_c_shuffled"]["profile_crps_dex"]
        descendant_score = values["descendant_diffmah_concurrent_c"]["profile_crps_dex"]
        gains[key] = {
            "local_vs_epoch_mass_percent": 100.0 * (mass - local) / mass,
            "local_shape_shuffle_vs_epoch_mass_percent": 100.0
            * (mass - shape_shuffle)
            / mass,
            "local_c_shuffle_vs_epoch_mass_percent": 100.0 * (mass - c_shuffle) / mass,
            "local_vs_descendant_percent": 100.0
            * (descendant_score - local)
            / descendant_score,
        }
        print(
            f"  {key}: local vs epoch mass {gains[key]['local_vs_epoch_mass_percent']:+.2f}% "
            f"(shape shuffle {gains[key]['local_shape_shuffle_vs_epoch_mass_percent']:+.2f}%, "
            f"c shuffle {gains[key]['local_c_shuffle_vs_epoch_mass_percent']:+.2f}%); "
            f"local vs descendant-conditioned {gains[key]['local_vs_descendant_percent']:+.2f}%"
        )

    figure(summaries, gains, fit_rms, len(descendant))
    if z2_local_result is None:
        raise RuntimeError("z=2 epoch-local result was not retained")
    z2_profile_figure(
        epoch_mass,
        cogs_log[:, 4],
        z2_local_result["mean_cogs"],
        z2_local_result["draw_cogs"][0],
        title="exp51 — epoch-local z=2 CoG predictions",
        filename="exp51_epoch_local_z2_predictions",
        draw_label="One stochastic draw",
    )
    payload = {
        "n_galaxy": len(descendant),
        "redshifts": REDSHIFTS.tolist(),
        "n_draw": N_DRAW,
        "t_min_gyr": T_MIN_GYR,
        "feature_fit_seconds": float(data["fit_seconds"]),
        "diffmah_fit_rms_median_dex": np.median(fit_rms, axis=0).tolist(),
        "diffmah_fit_rms_90th_dex": np.quantile(fit_rms, 0.9, axis=0).tolist(),
        "models": summaries,
        "gains": gains,
    }
    (OUTDIR / "results.json").write_text(json.dumps(payload, indent=2))
    write_manifest(
        OUTDIR,
        {
            "n_galaxy": len(descendant),
            "redshifts": REDSHIFTS.tolist(),
            "n_draw": N_DRAW,
            "t_min_gyr": T_MIN_GYR,
        },
    )
    print(f"wrote epoch-local outputs to {OUTDIR}")


def figure(summaries: dict, gains: dict, fit_rms: np.ndarray, n_galaxy: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9))
    axes[0].plot(
        REDSHIFTS,
        np.median(fit_rms, axis=0),
        "o-",
        color=OKABE_ITO[0],
        label="median",
    )
    axes[0].plot(
        REDSHIFTS,
        np.quantile(fit_rms, 0.9, axis=0),
        "s--",
        color=OKABE_ITO[1],
        label="90th percentile",
    )
    axes[0].set(
        xlabel="Fit endpoint redshift",
        ylabel="Truncated DiffMAH fit RMS (dex)",
        title="Past-only MAH compression",
    )
    axes[0].legend(fontsize=7)

    definitions = (
        ("epoch_mass_only", "epoch halo mass only", OKABE_ITO[7], ":"),
        ("local_diffmah_concurrent_c", "epoch-local DiffMAH + c", OKABE_ITO[2], "-"),
        (
            "descendant_diffmah_concurrent_c",
            "descendant DiffMAH + concurrent c",
            OKABE_ITO[4],
            "--",
        ),
        ("local_shape_shuffled", "shuffled local MAH shape", OKABE_ITO[6], "-."),
    )
    for name, label, color, style in definitions:
        axes[1].plot(
            REDSHIFTS,
            [summaries[f"z{z:.1f}"][name]["profile_crps_dex"] for z in REDSHIFTS],
            marker="o",
            linestyle=style,
            color=color,
            label=label,
        )
    axes[1].set(
        xlabel="Profile redshift",
        ylabel="Held-out profile CRPS (dex)",
        title="Portable versus descendant-conditioned inputs",
    )
    axes[1].legend(fontsize=7)

    axes[2].axhline(0.0, color="0.5", linestyle=":")
    axes[2].plot(
        REDSHIFTS,
        [gains[f"z{z:.1f}"]["local_vs_epoch_mass_percent"] for z in REDSHIFTS],
        "o-",
        color=OKABE_ITO[2],
        label="real local DiffMAH + c",
    )
    axes[2].plot(
        REDSHIFTS,
        [
            gains[f"z{z:.1f}"]["local_shape_shuffle_vs_epoch_mass_percent"]
            for z in REDSHIFTS
        ],
        "s--",
        color=OKABE_ITO[6],
        label="shuffled local MAH shape",
    )
    axes[2].plot(
        REDSHIFTS,
        [gains[f"z{z:.1f}"]["local_vs_descendant_percent"] for z in REDSHIFTS],
        "^-.",
        color=OKABE_ITO[1],
        label="local relative to descendant-conditioned",
    )
    axes[2].set(
        xlabel="Profile redshift",
        ylabel="Profile CRPS improvement (percent)",
        title="Information available by the epoch",
    )
    axes[2].legend(fontsize=7)
    for label, axis in zip("ABC", axes):
        axis.text(
            -0.15,
            1.05,
            label,
            transform=axis.transAxes,
            fontsize=11,
            fontweight="bold",
        )
    fig.suptitle(f"exp51 — epoch-local DiffMAH conversion (n={n_galaxy})")
    fig.tight_layout()
    save_fig(fig, FIGDIR / "exp51_epoch_local")
    plt.close(fig)


def demo() -> None:
    data = fit_epoch_local_features()
    parameters = np.asarray(data["parameters"])
    assert parameters.ndim == 3 and parameters.shape[1:] == (5, 4)
    assert np.isfinite(parameters).all()
    assert np.isfinite(data["rms"]).all()
    print(
        f"exp51 epoch-local mechanics OK: n={len(parameters)}, "
        f"five-epoch fits in {float(data['fit_seconds']):.2f} seconds"
    )


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "fit"
    if command == "demo":
        demo()
    elif command == "fit":
        fit_experiment()
    elif command == "fit_features":
        fit_epoch_local_features(force=True)
    else:
        raise SystemExit(f"unknown command {command!r}; use demo, fit_features, or fit")
