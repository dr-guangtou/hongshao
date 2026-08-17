"""Gated exp50 test: does fitted concentration history improve the direct CoG map?"""

from __future__ import annotations

import json
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
from model import compress_cogs  # noqa: E402
from run import (  # noqa: E402
    RADII,
    cv_quantile,
    load_sample,
    shuffled_within_mass_bins,
    summarize_model,
)

set_style()

FIGDIR = HERE / "figures"
OUTDIR = HERE / "outputs" / "concentration_history"
N_DRAW = 32


def source_output(experiment: str, filename: str) -> Path:
    candidates = [
        ROOT / "experiments" / experiment / "outputs" / filename,
        ROOT.parent / "hongshao" / "experiments" / experiment / "outputs" / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"missing {filename}; checked {candidates}")


def load_joined() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features, truth_log, indices = load_sample()
    history_file = source_output("exp46_highz_ridge", "conc_history.npz")
    population_file = source_output("exp32_full_population", "population.npz")
    history = np.load(history_file)
    population = np.load(population_file)
    concentration = np.where(history["GroupFlag"] == 1, history["c200c"], np.nan)
    index_to_row = {int(index): row for row, index in enumerate(indices)}
    sample_rows, history_rows = [], []
    for history_row, index in enumerate(population["index"]):
        sample_row = index_to_row.get(int(index))
        if sample_row is not None and np.isfinite(concentration[history_row]).all():
            sample_rows.append(sample_row)
            history_rows.append(history_row)
    sample_rows = np.asarray(sample_rows, int)
    history_rows = np.asarray(history_rows, int)
    features = features[sample_rows]
    truth_log = truth_log[sample_rows]
    indices = indices[sample_rows]
    concentration = concentration[history_rows]
    difference = np.max(np.abs(features[:, 4] - concentration[:, 0]))
    if difference > 1e-9:
        raise ValueError(
            f"z=0.4 concentration join mismatch: max difference {difference}"
        )
    return features, truth_log, indices, concentration


def run() -> None:
    features, truth_log, indices, concentration = load_joined()
    targets = compress_cogs(truth_log, RADII, "quantile3")
    # Current c200c is already feature 4. Add only the four earlier-epoch
    # measurements, so the increment asks about history rather than duplicating
    # the established concurrent concentration feature.
    history_features = np.column_stack([features, concentration[:, 1:]])
    shuffled_history = shuffled_within_mass_bins(
        history_features, (5, 6, 7, 8), seed=50
    )
    results = {
        "current_linear": cv_quantile(features, targets, "quantile3", n_draw=N_DRAW),
        "history_linear": cv_quantile(
            history_features, targets, "quantile3", n_draw=N_DRAW
        ),
        "shuffled_linear": cv_quantile(
            shuffled_history, targets, "quantile3", n_draw=N_DRAW
        ),
        "current_poly2": cv_quantile(
            features, targets, "quantile3", mean="poly2", n_draw=N_DRAW
        ),
        "history_poly2": cv_quantile(
            history_features, targets, "quantile3", mean="poly2", n_draw=N_DRAW
        ),
        "shuffled_poly2": cv_quantile(
            shuffled_history, targets, "quantile3", mean="poly2", n_draw=N_DRAW
        ),
    }
    summaries = {
        name: summarize_model(name, result, truth_log)
        for name, result in results.items()
    }
    linear_baseline = summaries["current_linear"]["profile_crps_dex"]
    linear_history = summaries["history_linear"]["profile_crps_dex"]
    linear_shuffled = summaries["shuffled_linear"]["profile_crps_dex"]
    baseline = summaries["current_poly2"]["profile_crps_dex"]
    history_score = summaries["history_poly2"]["profile_crps_dex"]
    shuffled_score = summaries["shuffled_poly2"]["profile_crps_dex"]
    linear_gain_percent = 100.0 * (linear_baseline - linear_history) / linear_baseline
    linear_shuffle_gain_percent = (
        100.0 * (linear_baseline - linear_shuffled) / linear_baseline
    )
    gain_percent = 100.0 * (baseline - history_score) / baseline
    shuffle_gain_percent = 100.0 * (baseline - shuffled_score) / baseline

    print(f"exp50 concentration-history gate: n={len(features)}")
    print(f"  linear current      profile CRPS {linear_baseline:.5f} dex")
    print(
        f"  linear + history    profile CRPS {linear_history:.5f} dex "
        f"({linear_gain_percent:+.2f}% vs current)"
    )
    print(
        f"  linear + shuffled   profile CRPS {linear_shuffled:.5f} dex "
        f"({linear_shuffle_gain_percent:+.2f}% vs current)"
    )
    print(f"  poly2 current       profile CRPS {baseline:.5f} dex")
    print(
        f"  poly2 + history     profile CRPS {history_score:.5f} dex "
        f"({gain_percent:+.2f}% vs current)"
    )
    print(
        f"  poly2 + shuffled    profile CRPS {shuffled_score:.5f} dex "
        f"({shuffle_gain_percent:+.2f}% vs current)"
    )

    parameter_r_squared = {}
    for name, result in results.items():
        residual = result["mean_parameters"] - targets
        parameter_r_squared[name] = (
            1.0 - np.mean(residual**2, axis=0) / np.var(targets, axis=0)
        ).tolist()

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.9))
    names = [
        "current_linear",
        "history_linear",
        "current_poly2",
        "history_poly2",
        "shuffled_poly2",
    ]
    labels = [
        "linear current",
        "linear + history",
        "poly2 current",
        "poly2 + history",
        "poly2 + shuffled",
    ]
    colors = [OKABE_ITO[7], OKABE_ITO[1], OKABE_ITO[4], OKABE_ITO[2], OKABE_ITO[6]]
    axes[0].bar(
        labels, [summaries[name]["profile_crps_dex"] for name in names], color=colors
    )
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].set(ylabel="Profile CRPS (dex)", title="Incremental predictive value")
    for name, color in zip(names, colors):
        axes[1].plot(
            RADII,
            summaries[name]["profile_crps_by_radius"],
            color=color,
            label=name.replace("_", " "),
        )
    axes[1].set(
        xscale="log",
        xlabel="Radius (kpc)",
        ylabel="CRPS (dex)",
        title="Radial predictive score",
    )
    axes[1].legend(fontsize=7)
    for label, axis in zip("AB", axes):
        axis.text(
            -0.15, 1.05, label, transform=axis.transAxes, fontsize=11, fontweight="bold"
        )
    fig.suptitle(f"exp50 — fitted halo-concentration history (n={len(features)})")
    fig.tight_layout()
    save_fig(fig, FIGDIR / "exp50_concentration_history")
    plt.close(fig)

    payload = {
        "n_galaxy": len(features),
        "epochs": [0.4, 0.7, 1.0, 1.5, 2.0],
        "models": summaries,
        "parameter_r_squared": parameter_r_squared,
        "linear_history_gain_percent": linear_gain_percent,
        "linear_shuffled_history_gain_percent": linear_shuffle_gain_percent,
        "history_gain_percent": gain_percent,
        "shuffled_history_gain_percent": shuffle_gain_percent,
    }
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "results.json").write_text(json.dumps(payload, indent=2))
    np.savez_compressed(
        OUTDIR / "joined_features.npz",
        indices=indices,
        baseline_features=features,
        concentration_history=concentration,
    )
    write_manifest(
        OUTDIR,
        {
            "n_galaxy": len(features),
            "history_gain_percent": gain_percent,
            "shuffled_history_gain_percent": shuffle_gain_percent,
        },
    )
    print(f"wrote concentration-history outputs to {OUTDIR}")


if __name__ == "__main__":
    run()
