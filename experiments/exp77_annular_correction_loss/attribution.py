# %%
"""Post-result controls: annular coefficient targets versus map selection."""

# ruff: noqa: E402
import json

from annular import (
    HERE,
    REFERENCE,
    REF,
    apply_correction,
    fit_labels,
    fit_map,
    scores,
    train_predict,
)

REPORT = REF.CONT.BASE.load_module("exp77_attribution_report", HERE / "report.py")

import matplotlib.pyplot as plt
import numpy as np
from threadpoolctl import threadpool_limits
from hongshao.plotting import set_style


def main():
    destination = REPORT.OUTPUT / "attribution.json"
    if destination.exists() or destination.with_suffix(".npz").exists():
        raise FileExistsError(destination)
    sample, predictions, _ = REPORT.assemble()
    truth, radii = sample["target"], sample["radii"]
    controls = {
        name: np.empty_like(truth)
        for name in ("annular_fixed_map_settings", "cumulative_outer_selected_map")
    }
    choices = []
    for fold in range(5):
        record = json.loads(
            (
                REFERENCE / f"outputs/continuation/discovery_measured_fold{fold}.json"
            ).read_text()
        )
        choice = record["choices"]["measured"]["hybrid_choice"]
        with np.load(
            REFERENCE / f"outputs/continuation/discovery_measured_fold{fold}.npz"
        ) as archive:
            calibration, held = archive["calibration"], archive["evaluation"]
            baseline = archive["baseline_all"]
        features = sample[choice["feature"]]
        labels = {}
        for weight in (0, 0.25):
            labels[weight] = np.empty((len(calibration), 5, 4))
            for position, row in enumerate(calibration):
                REF.CONT.check_deadline()
                for epoch in range(5):
                    labels[weight][position, epoch] = fit_labels(
                        baseline[row, epoch], truth[row, epoch], radii, weight
                    )[0]
        candidates = []
        for degree in (1, 2):
            for penalty in (0.01, 0.1, 1, 10, 100):
                inner_predictions = np.empty_like(truth[calibration])
                for inner in (0, 1):
                    training = np.flatnonzero(np.arange(len(calibration)) % 2 != inner)
                    evaluated = np.flatnonzero(np.arange(len(calibration)) % 2 == inner)
                    inner_predictions[evaluated] = train_predict(
                        features[calibration],
                        labels[0],
                        baseline[calibration],
                        radii,
                        training,
                        evaluated,
                        degree,
                        penalty,
                    )
                candidates.append(
                    {
                        "degree": degree,
                        "penalty": penalty,
                        **scores(inner_predictions, truth[calibration], radii),
                    }
                )
        reference = next(
            value
            for value in candidates
            if value["degree"] == choice["degree"]
            and value["penalty"] == choice["penalty"]
        )
        eligible = [
            value
            for value in candidates
            if value["cog_mean_radial_rms_dex"]
            <= 1.02 * reference["cog_mean_radial_rms_dex"]
        ]
        best = min(value["outer_relative_rms"] for value in eligible)
        selected = min(
            (value for value in eligible if value["outer_relative_rms"] <= 1.01 * best),
            key=lambda value: (value["degree"], -value["penalty"]),
        )
        for name, weight, configuration in (
            ("annular_fixed_map_settings", 0.25, choice),
            ("cumulative_outer_selected_map", 0, selected),
        ):
            fitted = fit_map(
                features[calibration],
                labels[weight].reshape(len(calibration), -1),
                configuration["degree"],
                configuration["penalty"],
            )
            parameters = fitted.predict(features[held]).reshape(len(held), 5, 4)
            controls[name][held] = apply_correction(baseline[held], parameters, radii)
        choices.append(
            {"fold": fold, "original": choice, "cumulative_outer_selected": selected}
        )
    reference_outer = (
        (predictions["reference"] - truth) @ REPORT.annular_matrix(radii)
    )[..., -2:] / truth[..., -1, None]
    statistics = {}
    for name, value in controls.items():
        outer = ((value - truth) @ REPORT.annular_matrix(radii))[..., -2:] / truth[
            ..., -1, None
        ]
        rng = np.random.default_rng(77201)
        improvements = []
        for _ in range(1000):
            rows = rng.integers(0, len(truth), len(truth))
            improvements.append(
                1
                - np.sqrt(
                    np.mean(outer[rows] ** 2) / np.mean(reference_outer[rows] ** 2)
                )
            )
        statistics[name] = {
            **scores(value, truth, radii),
            "outer_improvement_fraction": float(
                1 - np.sqrt(np.mean(outer**2) / np.mean(reference_outer**2))
            ),
            "paired_outer_improvement_95_interval": np.percentile(
                improvements, [2.5, 97.5]
            ),
        }
    REPORT.DRIVER.save_archive(
        destination.with_suffix(".npz"), indices=sample["indices"], **controls
    )
    set_style()
    plt.rcParams["text.usetex"] = False
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    comparisons = [
        ("Exp75 cumulative reference", predictions["reference"], "#0072B2", "--"),
        (
            "Annular targets; original map settings",
            controls["annular_fixed_map_settings"],
            "#D55E00",
            "-",
        ),
        (
            "Cumulative targets; outer-selected map",
            controls["cumulative_outer_selected_map"],
            "#009E73",
            ":",
        ),
    ]
    for column, epoch in enumerate((0, 4)):
        for label, value, color, style in comparisons:
            annuli = ((value - truth) @ REPORT.annular_matrix(radii)) / truth[
                ..., -1, None
            ]
            axes[0, column].plot(
                np.arange(6),
                np.sqrt(np.mean(annuli[:, epoch] ** 2, axis=0)),
                marker="o",
                c=color,
                ls=style,
                label=label,
            )
        axes[0, column].set_xticks(
            np.arange(6),
            ["0–2", "2–10", "10–30", "30–50", "50–100", "100–148.22"],
            rotation=35,
        )
        axes[0, column].set(
            ylabel="RMS annular mass error / true total",
            title=f"z={REF.CONT.BASE.REDSHIFTS[epoch]}",
        )
        row = int(np.flatnonzero(sample["indices"] == 1431)[0])
        axes[1, column].plot(radii, np.log10(truth[row, epoch]), "ko", ms=3)
        for _, value, color, style in comparisons:
            axes[1, column].plot(radii, np.log10(value[row, epoch]), c=color, ls=style)
        axes[1, column].set(
            xscale="log",
            xlabel="Semi-major radius (kpc)",
            ylabel="log stellar mass within R (Msun)",
            title=f"Previously identified most-improved ID 1431; z={REF.CONT.BASE.REDSHIFTS[epoch]}",
        )
    axes[0, 0].legend(fontsize=7)
    paths = REPORT.save(
        fig,
        "attribution_controls",
        "Exp77 post-result attribution, not candidate selection\nSeparate changes to the fitted stellar-profile targets from changes to the halo-map settings",
    )
    REPORT.DRIVER.write_record(
        destination,
        {
            **REPORT.DRIVER.provenance(),
            "post_result_diagnostic_only": True,
            "scores": statistics,
            "choices": choices,
            "figures": paths,
            "archive_sha256": REF.file_hash(destination.with_suffix(".npz")),
        },
    )
    print(
        json.dumps(REF.STANDARD.json_ready({"scores": statistics, "figures": paths})),
        flush=True,
    )


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
