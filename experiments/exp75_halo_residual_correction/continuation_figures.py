# %%
"""Plain-language comparisons at exact apertures, with fixed individual IDs."""

# ruff: noqa: E402
import json

from continuation_report import CONT, FIGURES, OUTPUT, aperture, annuli, assemble

import matplotlib.pyplot as plt
import numpy as np
from hongshao.plotting import save_fig, set_style

STYLES = {
    "baseline": ("#0072B2", "--", "Physical baseline"),
    "hybrid": ("#D55E00", "-", "Baseline + halo correction"),
    "direct": ("#009E73", ":", "Direct halo prediction"),
}


def save(figure, name, caption):
    figure.suptitle(caption, fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    paths = save_fig(figure, FIGURES / name)
    plt.close(figure)
    return [str(path) for path in paths]


def main():
    CONT.check_deadline()
    destination = OUTPUT / "diagnostic_figures.json"
    if destination.exists():
        raise FileExistsError(destination)
    sample, predictions, _ = assemble()
    summary = json.loads((OUTPUT / "summary.json").read_text())
    truth, radii = sample["target"], sample["radii"]
    set_style()
    plt.rcParams["text.usetex"] = False
    paths = []

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True)
    for row, epoch in enumerate((0, 4)):
        for column, (name, (color, _, label)) in enumerate(STYLES.items()):
            axis = axes[row, column]
            for value, shade, legend in (
                (truth, "0.55", "Measured galaxies"),
                (predictions[name], color, label),
            ):
                inner = aperture(value, radii, 30)[:, epoch]
                outer = (aperture(value, radii, 100) - aperture(value, radii, 50))[
                    :, epoch
                ]
                valid = (inner > 0) & (outer > 0)
                axis.scatter(
                    np.log10(inner[valid]),
                    np.log10(outer[valid]),
                    s=6,
                    alpha=0.35,
                    c=shade,
                    label=f"{legend}; n={valid.sum()}",
                )
            axis.set_title(f"{label}; z={CONT.BASE.REDSHIFTS[epoch]}")
            axis.set_xlabel(r"$\log_{10}[M_*(<30\,\mathrm{kpc})/M_\odot]$")
            axis.set_ylabel(r"$\log_{10}[M_*(50-100\,\mathrm{kpc})/M_\odot]$")
            axis.legend(fontsize=7)
    paths += save(
        fig,
        "exact_stellar_mass_planes",
        "Same held-out galaxies; exact 30, 50 and 100 kpc interpolation\nColored points are halo-only point predictions, not sampled populations; inspect slope and location separately from width",
    )

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True)
    for column, (galaxy, epoch) in enumerate(((3362, 0), (1802, 2), (9, 4))):
        row = int(np.flatnonzero(sample["indices"] == galaxy)[0])
        axes[0, column].plot(
            radii, np.log10(truth[row, epoch]), "ko", ms=3, label="Measured CoG"
        )
        for name, (color, style, label) in STYLES.items():
            value = predictions[name][row, epoch]
            axes[0, column].plot(radii, np.log10(value), c=color, ls=style, label=label)
            axes[1, column].plot(
                radii, np.log10(value / truth[row, epoch]), c=color, ls=style
            )
        axes[0, column].set_title(f"Galaxy {galaxy}; z={CONT.BASE.REDSHIFTS[epoch]}")
        axes[0, column].set_ylabel(r"$\log_{10}[M_*(<R)/M_\odot]$")
        axes[1, column].set_ylabel("log prediction/data (dex)")
        axes[1, column].set_xlabel("Semi-major radius (kpc)")
        axes[1, column].axhline(0, c="0.5", lw=0.7)
        for axis in axes[:, column]:
            axis.set_xscale("log")
    axes[0, 0].legend(fontsize=7)
    paths += save(
        fig,
        "fixed_individual_cogs",
        "Full-discovery predictions for the same three galaxy IDs shown in the operational check\nEvery displayed galaxy was excluded from both its baseline fit and correction calibration",
    )

    baseline_error = np.sqrt(
        np.mean(np.log10(predictions["baseline"] / truth) ** 2, axis=-1)
    ).mean(1)
    hybrid_error = np.sqrt(
        np.mean(np.log10(predictions["hybrid"] / truth) ** 2, axis=-1)
    ).mean(1)
    gain = baseline_error - hybrid_error
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True)
    for row, galaxy in enumerate((int(np.argmax(gain)), int(np.argmin(gain)))):
        for column, epoch in enumerate((0, 2, 4)):
            axis = axes[row, column]
            axis.plot(
                radii, np.log10(truth[galaxy, epoch]), "ko", ms=3, label="Measured CoG"
            )
            for name, (color, style, label) in STYLES.items():
                axis.plot(
                    radii,
                    np.log10(predictions[name][galaxy, epoch]),
                    c=color,
                    ls=style,
                    label=label,
                )
            axis.set_xscale("log")
            axis.set_title(
                f"{'Most improved' if row == 0 else 'Most worsened'} ID {sample['indices'][galaxy]}; z={CONT.BASE.REDSHIFTS[epoch]}"
            )
            axis.set_xlabel("Semi-major radius (kpc)")
            axis.set_ylabel(r"$\log_{10}[M_*(<R)/M_\odot]$")
    axes[0, 0].legend(fontsize=7)
    paths += save(
        fig,
        "most_changed_individuals",
        "Which held-out galaxies gain or lose most when the halo correction is added?\nRanked by change in galaxy-averaged radial log-CoG RMS across all five epochs; the same IDs are followed across redshift",
    )

    with np.load(OUTPUT / "representation.npz") as archive:
        oracle = archive["oracle"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    true_annuli = annuli(truth, radii)
    for column, epoch in enumerate((0, 4)):
        for name, value, color in (
            ("Physical baseline", predictions["baseline"], "#0072B2"),
            ("Halo-predicted correction", predictions["hybrid"], "#D55E00"),
            ("Correction fitted to each true CoG", oracle, "#CC79A7"),
        ):
            error = np.log10(value / truth)
            relative = (annuli(value, radii) - true_annuli) / truth[..., -1, None]
            axes[0, column].plot(
                radii,
                np.sqrt(np.mean(error[:, epoch] ** 2, axis=0)),
                c=color,
                label=name,
            )
            axes[1, column].plot(
                np.arange(6),
                np.sqrt(np.mean(relative[:, epoch] ** 2, axis=0)),
                "o-",
                c=color,
            )
        axes[0, column].set_title(f"z={CONT.BASE.REDSHIFTS[epoch]}")
        axes[0, column].set(
            xscale="log",
            xlabel="Semi-major radius (kpc)",
            ylabel="RMS log CoG error (dex)",
        )
        axes[1, column].set_xticks(
            np.arange(6),
            ["0–2", "2–10", "10–30", "30–50", "50–100", "100–148"],
            rotation=30,
        )
        axes[1, column].set(
            xlabel="Annulus (kpc; final edge 148.22)",
            ylabel="RMS annular error / true total",
        )
    axes[0, 0].legend(fontsize=7)
    paths += save(
        fig,
        "representation_vs_halo_prediction",
        "Can four profile corrections represent the missing structure, or are they hard to predict from halos?\nPurple uses the galaxy's true stellar profile: a representation diagnostic, NEVER a halo prediction",
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    redshifts = list(CONT.BASE.REDSHIFTS[1:])
    for name, color, label in (
        ("data", "k", "Measured galaxies"),
        *[(name, value[0], value[2]) for name, value in STYLES.items()],
    ):
        entries = [summary["future_growth"][str(z)]["models"][name] for z in redshifts]
        for column, (key, interval_key) in enumerate(
            (
                ("partial_slope_dex_per_dex", "partial_slope_bootstrap_95_interval"),
                (
                    "residual_slope_model_minus_data",
                    "residual_slope_bootstrap_95_interval",
                ),
            )
        ):
            center = np.array([entry[key] for entry in entries])
            bounds = np.array([entry[interval_key] for entry in entries])
            axes[column].plot(redshifts, center, "o-", c=color, label=label)
            axes[column].fill_between(
                redshifts, bounds[:, 0], bounds[:, 1], color=color, alpha=0.1
            )
    axes[0].set_ylabel("Stellar-mass response to later halo growth (dex/dex)")
    axes[1].set_ylabel("Model response minus measured response (dex/dex)")
    axes[1].axhline(0, c="0.5", lw=0.7)
    axes[0].legend(fontsize=7)
    for axis in axes:
        axis.set_xlabel("Redshift")
    paths += save(
        fig,
        "future_growth_response",
        "Mstar(<100 kpc) versus later M200c growth, holding quadratic epoch halo mass fixed\nFull halo history is allowed: the target is the measured response, not zero; shaded bands are paired galaxy-bootstrap 95% intervals",
    )
    CONT.write_record(destination, {"figures": paths})
    print(json.dumps(paths, indent=2), flush=True)


if __name__ == "__main__":
    main()
