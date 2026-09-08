# %%
"""Assemble disjoint held-out folds, apply frozen gates and plot direct QA."""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from run import OUTPUT, REDSHIFTS, coordinate_targets, decode, load_inputs
from prepare import file_hash

import matplotlib.pyplot as plt
import numpy as np
from hongshao.plotting import save_fig, set_style

HERE = Path(__file__).resolve().parent
NAMES = ("baseline", "intercept", "hybrid", "mass_only", "shuffled", "direct")
STYLES = {
    "baseline": ("#0072B2", "--"),
    "hybrid": ("#D55E00", "-"),
    "direct": ("#009E73", ":"),
}


def write_manifest():
    paths = sorted(HERE.glob("*.py")) + [HERE / "README.md"]
    paths += sorted(path for path in OUTPUT.iterdir() if path.name != "manifest.json")
    paths += sorted(path for path in (HERE / "figures").rglob("*") if path.is_file())
    manifest = {
        "experiment": "exp75_halo_residual_correction",
        "stage": "discovery_complete_not_production",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=HERE, text=True
        ).strip(),
        "protected_selection_validation_role": "none",
        "files_sha256": {
            str(path.relative_to(HERE)): file_hash(path) for path in paths
        },
        "note": "Fold records retain fitting-time git SHA, code hashes, choices, roles and runtime; this manifest inventories the final report.",
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def assemble(stage):
    sample = load_inputs(150 if stage == "pilot" else 0)
    predictions = {name: np.full_like(sample["target"], np.nan) for name in NAMES}
    seen = np.zeros(len(sample["indices"]), int)
    records = []
    for rotation in range(5):
        record = json.loads((OUTPUT / f"{stage}_fold{rotation}.json").read_text())
        if not record["passed"]:
            raise ValueError("unsettled baseline: no scientific comparison")
        with np.load(OUTPUT / f"{stage}_fold{rotation}.npz") as archive:
            np.testing.assert_array_equal(archive["indices"], sample["indices"])
            held = archive["evaluation"]
            if np.intersect1d(
                held, np.r_[archive["training"], archive["calibration"]]
            ).size:
                raise ValueError("evaluation overlap with fitting galaxies")
            np.testing.assert_array_equal(archive["truth"], sample["target"][held])
            seen[held] += 1
            for name in NAMES:
                predictions[name][held] = archive[name]
        records.append(record)
    if not np.all(seen == 1):
        raise ValueError("each galaxy must be evaluated exactly once")
    return sample, predictions, records


def statistics(sample, predictions):
    truth = sample["target"]
    errors = {
        name: np.log10(prediction / truth) for name, prediction in predictions.items()
    }
    rms = {name: np.sqrt(np.mean(error**2, axis=-1)) for name, error in errors.items()}
    bins = np.searchsorted(
        np.quantile(sample["halo"][:, 0], [1 / 3, 2 / 3]), sample["halo"][:, 0]
    )
    biases = {
        name: np.stack([np.median(error[bins == group], axis=0) for group in range(3)])
        for name, error in errors.items()
    }
    result = {
        name: {
            "pooled_mean_galaxy_rms_dex": float(value.mean()),
            "epoch_mean_galaxy_rms_dex": value.mean(0).tolist(),
            "epoch_p90_galaxy_rms_dex": np.percentile(value, 90, axis=0).tolist(),
            "maximum_absolute_mass_bin_median_bias_dex": float(
                np.max(np.abs(biases[name]))
            ),
        }
        for name, value in rms.items()
    }
    rng = np.random.default_rng(75001)
    improvement = []
    for _ in range(1000):
        rows = rng.integers(0, len(truth), len(truth))
        improvement.append(
            1 - rms["hybrid"][rows].mean() / rms["baseline"][rows].mean()
        )
    interval = np.percentile(improvement, [2.5, 97.5])
    improvement = float(1 - rms["hybrid"].mean() / rms["baseline"].mean())
    checks = {
        "pooled_improvement_at_least_five_percent": improvement >= 0.05,
        "paired_bootstrap_interval_above_zero": bool(interval[0] > 0),
        "better_than_intercept_only": bool(
            rms["hybrid"].mean() < rms["intercept"].mean()
        ),
        "no_epoch_more_than_two_percent_worse": bool(
            np.max(rms["hybrid"].mean(0) / rms["baseline"].mean(0)) <= 1.02
        ),
        "mass_bin_bias_no_more_than_point_zero_one_dex_worse": bool(
            np.max(np.abs(biases["hybrid"]))
            <= np.max(np.abs(biases["baseline"])) + 0.01
        ),
    }
    result["decision"] = {
        "checks": checks,
        "eligible_for_further_work": all(checks.values()),
        "fractional_rms_improvement": improvement,
        "galaxy_bootstrap_95_percent_interval": interval.tolist(),
    }
    return result, errors, rms, bins


def figures(sample, predictions, errors, rms, bins, stage):
    set_style()
    plt.rcParams["text.usetex"] = False
    paths = []
    radii, truth = sample["radii"], sample["target"]
    fig, axes = plt.subplots(4, 3, figsize=(12, 12), sharex=True)
    for row, epoch in enumerate((0, 4)):
        for column in range(3):
            selected = bins == column
            measured = np.log10(truth[selected, epoch])
            axes[2 * row, column].plot(
                radii,
                np.log10(np.mean(truth[selected, epoch], axis=0)),
                "k-",
                label="Measured mean",
            )
            low, high = np.percentile(measured, [16, 84], axis=0)
            axes[2 * row, column].fill_between(
                radii, low, high, color="0.5", alpha=0.12
            )
            for name, (color, linestyle) in STYLES.items():
                axes[2 * row, column].plot(
                    radii,
                    np.log10(np.mean(predictions[name][selected, epoch], axis=0)),
                    color=color,
                    ls=linestyle,
                    label=name,
                )
                axes[2 * row + 1, column].plot(
                    radii,
                    np.median(errors[name][selected, epoch], axis=0),
                    color=color,
                    ls=linestyle,
                )
            axes[2 * row, column].set_title(
                f"z={REDSHIFTS[epoch]}; halo-mass third {column + 1}; n={selected.sum()}"
            )
            axes[2 * row + 1, column].axhline(0, color="0.5", lw=0.7)
        axes[2 * row, 0].set_ylabel(r"$\log_{10}[\langle M_*(<R)\rangle/M_\odot]$")
        axes[2 * row + 1, 0].set_ylabel("Median log(prediction/data) (dex)")
    for axis in axes.flat:
        axis.set_xscale("log")
    for axis in axes[-1]:
        axis.set_xlabel("Semi-major radius (kpc)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        f"Exp75 {stage}: independent halo-only predictions; n={len(truth)}\n"
        "Gray bands: measured 16–84% population range, not uncertainty on the mean.",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    paths += save_fig(fig, HERE / "figures" / f"{stage}_mass_binned_cogs")
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    inner = int(np.argmin(np.abs(radii - 10)))
    lower = int(np.argmin(np.abs(radii - 50)))
    upper = int(np.argmin(np.abs(radii - 100)))
    for row, epoch in enumerate((0, 4)):
        for column, name in enumerate(STYLES):
            axis = axes[row, column]
            for data, color, label in (
                (truth, "0.65", "Measured"),
                (predictions[name], STYLES[name][0], name),
            ):
                axis.scatter(
                    np.log10(data[:, epoch, inner]),
                    np.log10(
                        np.maximum(data[:, epoch, upper] - data[:, epoch, lower], 1)
                    ),
                    s=7,
                    color=color,
                    alpha=0.35,
                    label=label,
                )
            axis.set_title(f"{name}; z={REDSHIFTS[epoch]}")
            axis.set_xlabel(
                rf"$\log_{{10}}[M_*(<{radii[inner]:.2f}\,\mathrm{{kpc}})/M_\odot]$"
            )
            axis.set_ylabel(
                rf"$\log_{{10}}[M_*({radii[lower]:.2f}\!:\!{radii[upper]:.2f}\,\mathrm{{kpc}})/M_\odot]$"
            )
            axis.legend()
    for row in range(2):
        xlim = (
            min(axis.get_xlim()[0] for axis in axes[row]),
            max(axis.get_xlim()[1] for axis in axes[row]),
        )
        ylim = (
            min(axis.get_ylim()[0] for axis in axes[row]),
            max(axis.get_ylim()[1] for axis in axes[row]),
        )
        for axis in axes[row]:
            axis.set(xlim=xlim, ylim=ylim)
    fig.suptitle(
        f"Exp75 {stage}: stellar-mass planes on identical held-out galaxies\n"
        "Colored points are point predictions, not random populations; narrowness alone is not a failure.",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    paths += save_fig(fig, HERE / "figures" / f"{stage}_stellar_mass_planes")
    plt.close(fig)

    ranking = np.argsort(rms["hybrid"].mean(1))
    selected = ranking[[0, len(ranking) // 2, -1]]
    fig, axes = plt.subplots(3, 3, figsize=(11, 9))
    for row, galaxy in enumerate(selected):
        for column, epoch in enumerate((0, 2, 4)):
            axis = axes[row, column]
            axis.plot(
                radii, np.log10(truth[galaxy, epoch]), "ko", ms=3, label="Measured"
            )
            for name, (color, linestyle) in STYLES.items():
                axis.plot(
                    radii,
                    np.log10(predictions[name][galaxy, epoch]),
                    color=color,
                    ls=linestyle,
                    label=name,
                )
            axis.set_xscale("log")
            axis.set_title(
                f"{('Best', 'Typical', 'Worst')[row]}: ID {sample['indices'][galaxy]}, z={REDSHIFTS[epoch]}"
            )
    for axis in axes[-1]:
        axis.set_xlabel("Semi-major radius (kpc)")
    for axis in axes[:, 0]:
        axis.set_ylabel(r"$\log_{10}[M_*(<R)/M_\odot]$")
    axes[0, 0].legend()
    fig.suptitle(
        f"Exp75 {stage}: best, median and worst hybrid errors averaged over all five epochs\n"
        "The same three galaxies are followed across redshift; no amplitude pin.",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    paths += save_fig(fig, HERE / "figures" / f"{stage}_ranked_individuals")
    plt.close(fig)
    return [str(path) for path in paths]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("pilot", "discovery"))
    args = parser.parse_args()
    sample, predictions, records = assemble(args.stage)
    result, errors, rms, bins = statistics(sample, predictions)
    result["figures"] = figures(sample, predictions, errors, rms, bins, args.stage)
    result["stage"] = args.stage
    result["n_galaxy"] = len(sample["indices"])
    result["fold_seconds"] = [record["seconds"] for record in records]
    result["scientific_result"] = args.stage == "discovery"
    represented = decode(
        coordinate_targets(sample["target"], sample["radii"]), sample["radii"]
    )
    representation_error = np.sqrt(
        np.mean(np.log10(represented / sample["target"]) ** 2, axis=-1)
    )
    result["direct_decoder_representation_only"] = {
        "note": "Measured stellar coordinates supplied only to diagnose representation; not a halo prediction or a scored reference.",
        "pooled_mean_galaxy_rms_dex": float(representation_error.mean()),
        "epoch_mean_galaxy_rms_dex": representation_error.mean(0).tolist(),
    }
    (OUTPUT / f"{args.stage}_summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    if args.stage == "discovery":
        write_manifest()
    print(json.dumps(result, indent=2))
