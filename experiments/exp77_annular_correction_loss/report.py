# %%
"""Frozen held-out Exp77 comparisons and directly inspectable QA."""

# ruff: noqa: E402
import argparse
import json
import time

from annular import HERE, REF, annular_matrix, scores

DRIVER = REF.CONT.BASE.load_module("exp77_report_driver", HERE / "run.py")

import matplotlib.pyplot as plt
import numpy as np
from threadpoolctl import threadpool_limits
from hongshao import qa
from hongshao.plotting import save_fig, set_style

OUTPUT, FIGURES = DRIVER.OUTPUT, DRIVER.FIGURES
STYLES = {
    "reference": ("#0072B2", "--", "Cumulative-loss correction"),
    "selected": ("#D55E00", "-", "Calibration-selected annular loss"),
}


def assemble(stage="discovery"):
    sample, subset = DRIVER.load_sample(150 if stage == "pilot" else 0)
    predictions = {name: np.full_like(sample["target"], np.nan) for name in STYLES}
    seen = np.zeros(len(sample["indices"]), int)
    records = []
    for fold in range(5):
        stem = OUTPUT / f"{stage}_fold{fold}"
        record = json.loads(stem.with_suffix(".json").read_text())
        if (
            not record["passed"]
            or REF.file_hash(stem.with_suffix(".npz")) != record["archive_sha256"]
        ):
            raise ValueError("Invalid fold checkpoint")
        with np.load(stem.with_suffix(".npz")) as archive:
            np.testing.assert_array_equal(archive["indices"], sample["indices"])
            held, calibration = archive["evaluation"], archive["calibration"]
            np.testing.assert_array_equal(
                held,
                np.flatnonzero(
                    (sample["folds"] == fold) & np.isin(np.arange(len(seen)), subset)
                ),
            )
            if np.intersect1d(calibration, held).size:
                raise ValueError("Calibration and evaluation overlap")
            for name in predictions:
                predictions[name][held] = archive[name]
            seen[held] += 1
        records.append(record)
    if not np.all(seen[subset] == 1):
        raise ValueError("Each galaxy must be evaluated exactly once")
    for name in predictions:
        predictions[name] = predictions[name][subset]
    count = len(sample["indices"])
    sample = {
        key: value[subset] if value.ndim and value.shape[0] == count else value
        for key, value in sample.items()
    }
    return sample, predictions, records


def future_difference(sample, predictions):
    rng = np.random.default_rng(77101)
    result = {}
    for epoch in range(1, 5):
        mass = sample["measured_epoch_mass"][:, epoch]
        growth = sample["measured_epoch_mass"][:, 0] - mass
        valid = np.isfinite(mass) & np.isfinite(growth)
        mass, growth = mass[valid] - 13, growth[valid]
        design = np.column_stack((np.ones(len(mass)), mass, mass**2, growth))
        truth = REF.aperture(sample["target"], sample["radii"], 100)[valid, epoch]
        labels = np.column_stack(
            [
                np.log10(
                    REF.aperture(predictions[name], sample["radii"], 100)[valid, epoch]
                    / truth
                )
                for name in STYLES
            ]
        )
        slopes = np.linalg.lstsq(design, labels, rcond=None)[0][-1]
        changes = []
        for _ in range(1000):
            rows = rng.integers(0, len(mass), len(mass))
            values = np.linalg.lstsq(design[rows], labels[rows], rcond=None)[0][-1]
            changes.append(abs(values[1]) - abs(values[0]))
        interval = np.percentile(changes, [2.5, 97.5])
        result[str(REF.CONT.BASE.REDSHIFTS[epoch])] = {
            "residual_slope_reference_selected_dex_per_dex": slopes,
            "increase_in_absolute_residual_slope": abs(slopes[1]) - abs(slopes[0]),
            "paired_bootstrap_95_interval": interval,
            "significantly_worse": bool(interval[0] > 0),
            "valid_galaxies": int(valid.sum()),
        }
    return result


def summary():
    destination = OUTPUT / "summary.json"
    if destination.exists():
        raise FileExistsError(destination)
    sample, predictions, records = assemble()
    truth, radii = sample["target"], sample["radii"]
    radial = REF.radial_summary(sample, predictions)
    score = {name: scores(value, truth, radii) for name, value in predictions.items()}
    relative = {
        name: ((value - truth) @ annular_matrix(radii))[..., -2:] / truth[..., -1, None]
        for name, value in predictions.items()
    }
    rng = np.random.default_rng(77001)
    improvements = []
    for _ in range(1000):
        rows = rng.integers(0, len(truth), len(truth))
        improvements.append(
            1
            - np.sqrt(
                np.mean(relative["selected"][rows] ** 2)
                / np.mean(relative["reference"][rows] ** 2)
            )
        )
    interval = np.percentile(improvements, [2.5, 97.5])
    gain = (
        1
        - score["selected"]["outer_relative_rms"]
        / score["reference"]["outer_relative_rms"]
    )
    edges = np.quantile(sample["measured_epoch_mass"][:, 0], [0, 1 / 3, 2 / 3, 1])
    bins = np.searchsorted(edges[1:-1], sample["measured_epoch_mass"][:, 0])
    biases = {
        name: float(
            np.max(
                np.abs(
                    [
                        np.median(
                            np.log10(value[bins == number] / truth[bins == number]),
                            axis=0,
                        )
                        for number in range(3)
                    ]
                )
            )
        )
        for name, value in predictions.items()
    }
    size_increase = {}
    for fraction in ("0.5", "0.8", "0.9"):
        key = "median_log_prediction_over_data_dex_by_epoch"
        size_increase[fraction] = np.abs(
            radial["selected"]["sizes"][fraction][key]
        ) - np.abs(radial["reference"]["sizes"][fraction][key])
    future = future_difference(sample, predictions)
    criteria = {
        "outer_improvement_at_least_10_percent": bool(gain >= 0.1),
        "positive_paired_outer_interval": bool(interval[0] > 0),
        "cog_rms_no_more_than_2_percent_worse": bool(
            score["selected"]["cog_mean_radial_rms_dex"]
            <= 1.02 * score["reference"]["cog_mean_radial_rms_dex"]
        ),
        "halo_bin_bias_no_more_than_001_dex_worse": bool(
            biases["selected"] <= biases["reference"] + 0.01
        ),
        "all_size_biases_no_more_than_001_dex_worse": bool(
            all(np.max(value) <= 0.01 for value in size_increase.values())
        ),
        "no_significantly_increased_future_response_error": not any(
            value["significantly_worse"] for value in future.values()
        ),
    }
    result = {
        **DRIVER.provenance(),
        "galaxies": len(truth),
        "scores": score,
        "outer_fractional_improvement": gain,
        "outer_improvement_paired_95_interval": interval,
        "maximum_halo_bin_median_bias_dex": biases,
        "size_absolute_bias_increase_dex_by_epoch": size_increase,
        "criteria": criteria,
        "advancement_passed": all(criteria.values()),
        "radial_diagnostics": radial,
        "future_difference": future,
        "future_growth": REF.future_growth(sample, predictions),
        "fold_choices": [record["selection"]["selected"] for record in records],
        "reference_parity_dex_by_fold": [
            record["reference_max_error_dex"] for record in records
        ],
        "note": "Fixed discovery population reused for an exploratory loss test. Intervals condition on fitted models; no production qualification.",
    }
    DRIVER.write_record(destination, result)
    print(
        json.dumps(
            REF.STANDARD.json_ready(
                {
                    key: result[key]
                    for key in (
                        "scores",
                        "outer_fractional_improvement",
                        "outer_improvement_paired_95_interval",
                        "criteria",
                        "fold_choices",
                    )
                }
            )
        ),
        flush=True,
    )


def save(figure, name, caption):
    figure.suptitle(caption, fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    paths = save_fig(figure, FIGURES / name)
    plt.close(figure)
    return [str(path) for path in paths]


def figures():
    destination = OUTPUT / "comparison_figures.json"
    if destination.exists():
        raise FileExistsError(destination)
    sample, predictions, _ = assemble()
    truth, radii = sample["target"], sample["radii"]
    set_style()
    plt.rcParams["text.usetex"] = False
    paths = []
    edges = np.quantile(sample["measured_epoch_mass"][:, 0], [0, 1 / 3, 2 / 3, 1])
    bins = np.searchsorted(edges[1:-1], sample["measured_epoch_mass"][:, 0])
    fig, axes = plt.subplots(4, 3, figsize=(11, 10), sharex=True)
    for row, epoch in enumerate((0, 4)):
        for column in range(3):
            held = bins == column
            axes[2 * row, column].plot(
                radii,
                np.log10(truth[held, epoch].mean(0)),
                "k",
                label="Measured mean CoG",
            )
            for name, (color, style, label) in STYLES.items():
                curve = predictions[name][held, epoch]
                axes[2 * row, column].plot(
                    radii, np.log10(curve.mean(0)), c=color, ls=style, label=label
                )
                axes[2 * row + 1, column].plot(
                    radii,
                    np.median(np.log10(curve / truth[held, epoch]), axis=0),
                    c=color,
                    ls=style,
                )
            axes[2 * row, column].set_title(
                f"z={REF.CONT.BASE.REDSHIFTS[epoch]}; halo-mass third {column + 1}; n={held.sum()}"
            )
            axes[2 * row, column].set_ylabel("log mean stellar mass (Msun)")
            axes[2 * row + 1, column].set_ylabel("Median log prediction/data (dex)")
            axes[2 * row + 1, column].axhline(0, c="0.5", lw=0.7)
    axes[0, 0].legend(fontsize=7)
    for axis in axes.flat:
        axis.set_xscale("log")
    for axis in axes[-1]:
        axis.set_xlabel("Semi-major radius (kpc)")
    paths += save(
        fig,
        "mass_binned_cogs",
        "Exp77: same held-out galaxies and measured halo histories\nDoes changing the correction loss preserve the average cumulative profiles?",
    )

    fig, axes = plt.subplots(2, 2, figsize=(9, 8), sharex=True, sharey=True)
    for row, epoch in enumerate((0, 4)):
        for column, (name, (color, _, label)) in enumerate(STYLES.items()):
            axis = axes[row, column]
            for values, shade, text in (
                (truth, "0.6", "Measured galaxies"),
                (predictions[name], color, label),
            ):
                inner = REF.aperture(values, radii, 30)[:, epoch]
                outer = (
                    REF.aperture(values, radii, 100) - REF.aperture(values, radii, 50)
                )[:, epoch]
                valid = (inner > 0) & (outer > 0)
                axis.scatter(
                    np.log10(inner[valid]),
                    np.log10(outer[valid]),
                    c=shade,
                    s=6,
                    alpha=0.4,
                    label=f"{text}; n={valid.sum()}",
                )
            axis.set_title(f"z={REF.CONT.BASE.REDSHIFTS[epoch]}")
            axis.set_xlabel("log stellar mass inside 30 kpc (Msun)")
            axis.set_ylabel("log stellar mass in 50–100 kpc (Msun)")
            axis.legend(fontsize=7)
    paths += save(
        fig,
        "exact_stellar_mass_planes",
        "Exp77: exact physical apertures, halo-only point predictions versus measured CoGs\nCompare slope and location; population width requires a separate scatter model",
    )

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for column, epoch in enumerate((0, 4)):
        for name, (color, style, label) in STYLES.items():
            error = np.log10(predictions[name] / truth)
            annuli = ((predictions[name] - truth) @ annular_matrix(radii)) / truth[
                ..., -1, None
            ]
            axes[0, column].plot(
                radii,
                np.sqrt(np.mean(error[:, epoch] ** 2, axis=0)),
                c=color,
                ls=style,
                label=label,
            )
            axes[1, column].plot(
                np.arange(6),
                np.sqrt(np.mean(annuli[:, epoch] ** 2, axis=0)),
                "o-",
                c=color,
            )
        axes[0, column].set(
            xscale="log",
            xlabel="Semi-major radius (kpc)",
            ylabel="RMS log CoG error (dex)",
            title=f"z={REF.CONT.BASE.REDSHIFTS[epoch]}",
        )
        axes[1, column].set_xticks(
            np.arange(6),
            ["0–2", "2–10", "10–30", "30–50", "50–100", "100–148.22"],
            rotation=35,
        )
        axes[1, column].set(
            xlabel="Annulus (kpc)", ylabel="RMS annular mass error / true total"
        )
    axes[0, 0].legend(fontsize=8)
    paths += save(
        fig,
        "radial_and_annular_errors",
        "Exp77: cumulative and annular prediction errors against the same measured profiles\nOuter-mass gains must not hide changes elsewhere in the galaxy",
    )

    fixed = np.array(
        [np.flatnonzero(sample["indices"] == index)[0] for index in (479, 1869, 3386)]
    )
    paths += DRIVER.comparison_figure(
        sample,
        fixed,
        {name: value[fixed] for name, value in predictions.items()},
        "fixed_individual_cogs",
    )
    error = {
        name: np.mean(
            (((value - truth) @ annular_matrix(radii))[..., -2:] / truth[..., -1, None])
            ** 2,
            axis=(1, 2),
        )
        for name, value in predictions.items()
    }
    gain = error["reference"] - error["selected"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 7), sharex=True)
    for row, position in enumerate((int(np.argmax(gain)), int(np.argmin(gain)))):
        for column, epoch in enumerate((0, 2, 4)):
            axis = axes[row, column]
            axis.plot(
                radii,
                np.log10(truth[position, epoch]),
                "ko",
                ms=3,
                label="Measured CoG",
            )
            for name, (color, style, label) in STYLES.items():
                axis.plot(
                    radii,
                    np.log10(predictions[name][position, epoch]),
                    c=color,
                    ls=style,
                    label=label,
                )
            axis.set(
                xscale="log",
                xlabel="Semi-major radius (kpc)",
                ylabel="log stellar mass within R (Msun)",
                title=f"{'Most improved' if row == 0 else 'Most worsened'} ID {sample['indices'][position]}; z={REF.CONT.BASE.REDSHIFTS[epoch]}",
            )
    axes[0, 0].legend(fontsize=7)
    paths += save(
        fig,
        "most_changed_individuals",
        "Exp77: largest improvement and deterioration in outer-annular squared error\nThe same galaxy IDs are followed across epochs; this is not an average-profile guarantee",
    )
    DRIVER.write_record(destination, {"figures": paths})
    print(json.dumps(paths), flush=True)


def standard():
    destination = OUTPUT / "standard_qa.json"
    if destination.exists():
        raise FileExistsError(destination)
    sample, predictions, _ = assemble()
    started = time.perf_counter()
    plt.rcParams["text.usetex"] = False
    metrics = qa.evaluate(
        predictions["selected"],
        sample["target"],
        sample["radii"],
        REF.CONT.BASE.REDSHIFTS,
        name="exp77_annular_selected",
        figdir=FIGURES / "standard_qa",
        figures=True,
        verbose=False,
        bin_by=sample["measured_epoch_mass"][:, 0],
        bin_label=r"$\log M_{200c}(z=0.4)$",
        halo_mass_epochs=sample["measured_epoch_mass"],
    )
    paths = [str(path) for path in sorted((FIGURES / "standard_qa").glob("*.png"))]
    DRIVER.write_record(
        destination,
        {
            "metrics": metrics,
            "seconds": time.perf_counter() - started,
            "figures": paths,
            "caption": "Exp77 calibration-selected annular-loss correction versus the same held-out measured CoGs at five epochs. All stellar amplitude and shape are predicted from halos; these are point predictions, not a sampled population. Exact apertures and QA masks match Exp75; final grid edge is 148.22 kpc. See README for interpretation.",
        },
    )
    print(
        json.dumps({"seconds": time.perf_counter() - started, "figures": paths}),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("summary", "figures", "standard"))
    arguments = parser.parse_args()
    with threadpool_limits(limits=1):
        REF.CONT.check_deadline()
        {"summary": summary, "figures": figures, "standard": standard}[
            arguments.action
        ]()
