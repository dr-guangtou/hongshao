# %%
"""Representation trade-off, safeguards and representative held-out profiles."""

# ruff: noqa: E402
import json

from annular import HERE, REF, annular_matrix

REPORT = REF.CONT.BASE.load_module("exp77_diagnostic_report", HERE / "report.py")

import matplotlib.pyplot as plt
import numpy as np
from hongshao.plotting import set_style


def main():
    destination = REPORT.OUTPUT / "diagnostic_figures.json"
    if destination.exists():
        raise FileExistsError(destination)
    sample, predictions, _ = REPORT.assemble()
    truth, radii = sample["target"], sample["radii"]
    summary = json.loads((REPORT.OUTPUT / "summary.json").read_text())
    set_style()
    plt.rcParams["text.usetex"] = False
    paths = []

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    with np.load(REPORT.OUTPUT / "discovery_representation.npz") as archive:
        np.testing.assert_array_equal(archive["indices"], sample["indices"])
        for weight, color in (("0", "#0072B2"), ("0.25", "#D55E00"), ("1", "#009E73")):
            value = archive[weight]
            errors = np.log10(value / truth)
            annuli = ((value - truth) @ annular_matrix(radii)) / truth[..., -1, None]
            for column, epoch in enumerate((0, 4)):
                axes[0, column].plot(
                    radii,
                    np.sqrt(np.mean(errors[:, epoch] ** 2, axis=0)),
                    c=color,
                    label=f"Annular weight {weight}",
                )
                axes[1, column].plot(
                    np.arange(6),
                    np.sqrt(np.mean(annuli[:, epoch] ** 2, axis=0)),
                    "o-",
                    c=color,
                )
    for column, epoch in enumerate((0, 4)):
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
    paths += REPORT.save(
        fig,
        "representation_tradeoff",
        "Exp77: four corrections fitted directly to each true stellar CoG\nRepresentation diagnostic ONLY, not halo prediction: better annuli cost small absolute cumulative accuracy",
    )

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for name, (color, style, label) in REPORT.STYLES.items():
        values = np.sort(np.log10(predictions[name] / truth).ravel())
        axes[0, 0].plot(
            values,
            np.arange(1, len(values) + 1) / len(values),
            c=color,
            ls=style,
            label=label,
        )
        for fraction, marker in (("0.5", "o"), ("0.9", "s")):
            values = summary["radial_diagnostics"][name]["sizes"][fraction][
                "median_log_prediction_over_data_dex_by_epoch"
            ]
            axes[0, 1].plot(
                REF.CONT.BASE.REDSHIFTS,
                values,
                c=color,
                ls=style,
                marker=marker,
                label=f"{'R50' if fraction == '0.5' else 'R90'}: {label}",
            )
    axes[0, 0].set(
        xlabel="log prediction/data CoG residual (dex)",
        ylabel="Cumulative fraction of galaxy-epoch-radius entries",
        xlim=(-0.5, 0.5),
    )
    axes[0, 0].legend(fontsize=7)
    axes[0, 1].set(xlabel="Redshift", ylabel="Median log size prediction/data (dex)")
    axes[0, 1].axhline(0, c="0.5", lw=0.7)
    axes[0, 1].legend(fontsize=6)
    epochs = list(REF.CONT.BASE.REDSHIFTS[1:])
    values = [summary["future_difference"][str(epoch)] for epoch in epochs]
    for column, name in enumerate(REPORT.STYLES):
        color, style, label = REPORT.STYLES[name]
        axes[1, 0].plot(
            epochs,
            [
                value["residual_slope_reference_selected_dex_per_dex"][column]
                for value in values
            ],
            c=color,
            ls=style,
            marker="o",
            label=label,
        )
    axes[1, 0].axhline(0, c="0.5", lw=0.7)
    axes[1, 0].set(
        xlabel="Redshift", ylabel="Model minus measured growth response (dex/dex)"
    )
    axes[1, 0].legend(fontsize=7)
    center = np.array(
        [value["increase_in_absolute_residual_slope"] for value in values]
    )
    interval = np.array([value["paired_bootstrap_95_interval"] for value in values])
    axes[1, 1].vlines(epochs, interval[:, 0], interval[:, 1], color="#D55E00")
    axes[1, 1].plot(epochs, center, "o", c="#D55E00")
    axes[1, 1].axhline(0, c="0.5", lw=0.7)
    axes[1, 1].set(
        xlabel="Redshift",
        ylabel="Increase in absolute growth-response error\n(dex/dex; paired 95% interval) ",
    )
    paths += REPORT.save(
        fig,
        "residuals_sizes_and_growth",
        "Exp77: the outer-mass gain must also preserve sizes and conditional halo trends\nFuture halo growth is allowed; disagreement with the measured response is the diagnostic",
    )

    galaxy_rms = np.sqrt(
        np.mean(np.log10(predictions["selected"] / truth) ** 2, axis=-1)
    ).mean(1)
    order = np.argsort(galaxy_rms)
    chosen = order[[0, len(order) // 2, -1]]
    fig, axes = plt.subplots(2, 3, figsize=(11, 7), sharex=True)
    for column, (position, rank) in enumerate(
        zip(chosen, ("Best", "Typical", "Worst"), strict=True)
    ):
        epoch = int(
            np.argmax(
                np.sqrt(
                    np.mean(
                        np.log10(predictions["selected"][position] / truth[position])
                        ** 2,
                        axis=-1,
                    )
                )
            )
        )
        axes[0, column].plot(
            radii, np.log10(truth[position, epoch]), "ko", ms=3, label="Measured CoG"
        )
        for name, (color, style, label) in REPORT.STYLES.items():
            curve = predictions[name][position, epoch]
            axes[0, column].plot(radii, np.log10(curve), c=color, ls=style, label=label)
            axes[1, column].plot(
                radii, np.log10(curve / truth[position, epoch]), c=color, ls=style
            )
        axes[0, column].set_title(
            f"{rank} ID {sample['indices'][position]}; z={REF.CONT.BASE.REDSHIFTS[epoch]}"
        )
        axes[0, column].set_ylabel("log stellar mass within R (Msun)")
        axes[1, column].set(
            xlabel="Semi-major radius (kpc)", ylabel="log prediction/data (dex)"
        )
        axes[1, column].axhline(0, c="0.5", lw=0.7)
        for axis in axes[:, column]:
            axis.set_xscale("log")
    axes[0, 0].legend(fontsize=7)
    paths += REPORT.save(
        fig,
        "ranked_individuals",
        "Exp77: best, median-ranked and worst galaxy by mean radial log-CoG RMS\nEach panel shows that galaxy's worst epoch; better outer masses do not eliminate individual failures",
    )
    REPORT.DRIVER.write_record(
        destination,
        {"figures": paths, "ranked_global_ids": sample["indices"][chosen].tolist()},
    )
    print(json.dumps(paths), flush=True)


if __name__ == "__main__":
    main()
