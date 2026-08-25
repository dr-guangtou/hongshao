"""exp51 stage 4 — concentration measured at each profile epoch.

The complete DiffMAH fit remains descendant-conditioned in this diagnostic.
Only the concentration information changes: z=0.4 concentration, concurrent
concentration, or the concentration measurements available at and before the
profile epoch. Every increment has an epoch-mass-conditioned shuffle control.
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from hongshao.plotting import OKABE_ITO, save_fig, set_style  # noqa: E402
from hongshao.provenance import write_manifest  # noqa: E402
from run import (  # noqa: E402
    N_MAX,
    RADII,
    REDSHIFTS,
    compress_cogs,
    cv_quantile,
    load_sample,
    source_output,
    summarize_model,
)

set_style()

FIGDIR = HERE / "figures"
OUTDIR = HERE / "outputs" / "concentration"
N_DRAW = int(os.environ.get("EXP51_CONC_N_DRAW", 32))


def shuffled_within_epoch_mass(
    features: np.ndarray,
    epoch_mass: np.ndarray,
    columns: tuple[int, ...],
    seed: int,
    n_bin: int = 10,
) -> np.ndarray:
    output = features.copy()
    edges = np.quantile(epoch_mass, np.linspace(0.0, 1.0, n_bin + 1))
    rng = np.random.default_rng(seed)
    for lower, upper in zip(edges[:-1], edges[1:]):
        members = np.where((epoch_mass >= lower) & (epoch_mass <= upper))[0]
        if len(members) > 1:
            permutation = rng.permutation(members)
            output[np.ix_(members, columns)] = features[np.ix_(permutation, columns)]
    return output


def load_concentration_sample():
    features, cogs_log, indices, epoch_mass, population_rows = load_sample(N_MAX)
    history = np.load(source_output("exp46_highz_ridge", "conc_history.npz"))
    concentration_all = np.where(history["GroupFlag"] == 1, history["c200c"], np.nan)
    concentration = concentration_all[population_rows]
    keep = np.isfinite(concentration).all(axis=1)
    features, cogs_log, indices, epoch_mass, concentration = (
        features[keep],
        cogs_log[keep],
        indices[keep],
        epoch_mass[keep],
        concentration[keep],
    )
    maximum_difference = np.max(np.abs(features[:, 4] - concentration[:, 0]))
    if maximum_difference > 1e-9:
        raise ValueError(
            f"z=0.4 concentration join mismatch: maximum {maximum_difference}"
        )
    return features, cogs_log, indices, epoch_mass, concentration


def fit_experiment() -> None:
    features, cogs_log, indices, epoch_mass, concentration = load_concentration_sample()
    print(f"exp51 concurrent-concentration test: n={len(features)}, draws={N_DRAW}")
    summaries = {}
    for epoch, redshift in enumerate(REDSHIFTS):
        print(f"  fitting concentration variants at z={redshift:.1f}")
        targets = compress_cogs(cogs_log[:, epoch], RADII, "quantile3")
        baseline = features
        concurrent = np.column_stack([features[:, :4], concentration[:, epoch]])
        concurrent_shuffled = shuffled_within_epoch_mass(
            concurrent, epoch_mass[:, epoch], (4,), seed=600 + epoch
        )
        # In ascending-redshift storage order, columns epoch: are the current
        # concentration and measurements from earlier cosmic times.
        available = np.column_stack([features[:, :4], concentration[:, epoch:]])
        available_columns = tuple(range(4, available.shape[1]))
        available_shuffled = shuffled_within_epoch_mass(
            available,
            epoch_mass[:, epoch],
            available_columns,
            seed=700 + epoch,
        )
        definitions = {
            "baseline_poly2": (baseline, "poly2"),
            "concurrent_poly2": (concurrent, "poly2"),
            "concurrent_shuffled_poly2": (concurrent_shuffled, "poly2"),
            "concurrent_linear": (concurrent, "linear"),
            "available_history_linear": (available, "linear"),
            "available_history_shuffled_linear": (available_shuffled, "linear"),
        }
        summaries[f"z{redshift:.1f}"] = {}
        for name, (model_features, mean) in definitions.items():
            result = cv_quantile(
                model_features,
                targets,
                "quantile3",
                mean=mean,
                n_draw=N_DRAW,
            )
            summaries[f"z{redshift:.1f}"][name] = summarize_model(
                name, result, cogs_log[:, epoch]
            )

    gains = {}
    print("\n[concurrent concentration and available-history gains]")
    for epoch, redshift in enumerate(REDSHIFTS):
        key = f"z{redshift:.1f}"
        values = summaries[key]
        baseline = values["baseline_poly2"]["profile_crps_dex"]
        concurrent = values["concurrent_poly2"]["profile_crps_dex"]
        concurrent_shuffled = values["concurrent_shuffled_poly2"]["profile_crps_dex"]
        current_linear = values["concurrent_linear"]["profile_crps_dex"]
        history = values["available_history_linear"]["profile_crps_dex"]
        history_shuffled = values["available_history_shuffled_linear"][
            "profile_crps_dex"
        ]
        gains[key] = {
            "concurrent_vs_z0p4_percent": 100.0 * (baseline - concurrent) / baseline,
            "concurrent_shuffle_vs_z0p4_percent": 100.0
            * (baseline - concurrent_shuffled)
            / baseline,
            "available_history_vs_concurrent_percent": 100.0
            * (current_linear - history)
            / current_linear,
            "available_history_shuffle_vs_concurrent_percent": 100.0
            * (current_linear - history_shuffled)
            / current_linear,
        }
        print(
            f"  z={redshift:.1f}: concurrent vs z=0.4 c "
            f"{gains[key]['concurrent_vs_z0p4_percent']:+.2f}% "
            f"(shuffle {gains[key]['concurrent_shuffle_vs_z0p4_percent']:+.2f}%); "
            f"available history vs concurrent {gains[key]['available_history_vs_concurrent_percent']:+.2f}% "
            f"(shuffle {gains[key]['available_history_shuffle_vs_concurrent_percent']:+.2f}%)"
        )

    figure(summaries, gains, len(features))
    OUTDIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_galaxy": len(features),
        "redshifts": REDSHIFTS.tolist(),
        "n_draw": N_DRAW,
        "models": summaries,
        "gains": gains,
    }
    (OUTDIR / "results.json").write_text(json.dumps(payload, indent=2))
    np.savez_compressed(
        OUTDIR / "sample.npz",
        indices=indices,
        baseline_features=features,
        epoch_halo_mass=epoch_mass,
        concentration_history=concentration,
    )
    write_manifest(
        OUTDIR,
        {
            "n_galaxy": len(features),
            "redshifts": REDSHIFTS.tolist(),
            "n_draw": N_DRAW,
        },
    )
    print(f"wrote concentration outputs to {OUTDIR}")


def figure(summaries: dict, gains: dict, n_galaxy: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9))
    definitions = (
        ("baseline_poly2", "z=0.4 concentration", OKABE_ITO[7], "--"),
        ("concurrent_poly2", "concurrent concentration", OKABE_ITO[2], "-"),
        (
            "concurrent_shuffled_poly2",
            "shuffled concurrent concentration",
            OKABE_ITO[6],
            ":",
        ),
    )
    for name, label, color, style in definitions:
        axes[0].plot(
            REDSHIFTS,
            [summaries[f"z{z:.1f}"][name]["profile_crps_dex"] for z in REDSHIFTS],
            marker="o",
            linestyle=style,
            color=color,
            label=label,
        )
    axes[0].set(
        xlabel="Profile redshift",
        ylabel="Held-out profile CRPS (dex)",
        title="Concentration at the profile epoch",
    )
    axes[0].legend(fontsize=7)

    concurrent_gain = [
        gains[f"z{z:.1f}"]["concurrent_vs_z0p4_percent"] for z in REDSHIFTS
    ]
    concurrent_shuffle = [
        gains[f"z{z:.1f}"]["concurrent_shuffle_vs_z0p4_percent"] for z in REDSHIFTS
    ]
    axes[1].axhline(0.0, color="0.5", linestyle=":")
    axes[1].plot(
        REDSHIFTS,
        concurrent_gain,
        "o-",
        color=OKABE_ITO[2],
        label="real concurrent concentration",
    )
    axes[1].plot(
        REDSHIFTS,
        concurrent_shuffle,
        "s--",
        color=OKABE_ITO[6],
        label="shuffled concurrent concentration",
    )
    axes[1].set(
        xlabel="Profile redshift",
        ylabel="CRPS improvement over z=0.4 c (percent)",
        title="Value of concurrent concentration",
    )
    axes[1].legend(fontsize=7)

    history_gain = [
        gains[f"z{z:.1f}"]["available_history_vs_concurrent_percent"] for z in REDSHIFTS
    ]
    history_shuffle = [
        gains[f"z{z:.1f}"]["available_history_shuffle_vs_concurrent_percent"]
        for z in REDSHIFTS
    ]
    axes[2].axhline(0.0, color="0.5", linestyle=":")
    axes[2].plot(
        REDSHIFTS,
        history_gain,
        "o-",
        color=OKABE_ITO[1],
        label="real available history",
    )
    axes[2].plot(
        REDSHIFTS,
        history_shuffle,
        "s--",
        color=OKABE_ITO[6],
        label="shuffled available history",
    )
    axes[2].set(
        xlabel="Profile redshift",
        ylabel="CRPS improvement over concurrent c (percent)",
        title="Additional value of earlier concentrations",
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
    fig.suptitle(f"exp51 — halo concentration at the correct epoch (n={n_galaxy})")
    fig.tight_layout()
    save_fig(fig, FIGDIR / "exp51_concentration_redshift")
    plt.close(fig)


def demo() -> None:
    features, cogs_log, _, epoch_mass, concentration = load_concentration_sample()
    selected = np.linspace(0, len(features) - 1, min(100, len(features))).astype(int)
    shuffled = shuffled_within_epoch_mass(
        np.column_stack([features[selected, :4], concentration[selected, 4]]),
        epoch_mass[selected, 4],
        (4,),
        seed=1,
        n_bin=4,
    )
    assert np.allclose(shuffled[:, :4], features[selected, :4])
    assert not np.allclose(shuffled[:, 4], concentration[selected, 4])
    targets = compress_cogs(cogs_log[selected, 4], RADII, "quantile3")
    assert np.isfinite(targets).all()
    print("exp51 concentration mechanics OK")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "fit"
    if command == "demo":
        demo()
    elif command == "fit":
        fit_experiment()
    else:
        raise SystemExit(f"unknown command {command!r}; use demo or fit")
