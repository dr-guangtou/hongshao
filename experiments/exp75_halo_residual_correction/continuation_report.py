# %%
"""Matched held-out comparisons, representation diagnostics, and standard QA."""

# ruff: noqa: E402
import argparse
import json
import time
from pathlib import Path

from prepare import file_hash, load_module

HERE = Path(__file__).resolve().parent
REPORT = load_module("exp75_continuation_report_reference", HERE / "report.py")
CONT = load_module("exp75_continuation_driver", HERE / "continuation.py")
STANDARD = load_module("exp75_continuation_standard_helpers", HERE / "standard_qa.py")

import matplotlib.pyplot as plt
import numpy as np
from threadpoolctl import threadpool_limits
from correction import apply_correction, fit_coefficients
from hongshao import qa

OUTPUT = CONT.OUTPUT
FIGURES = HERE / "figures/continuation"
DEFINITIONS = {
    "baseline": "Deposition model fitted using measured halo histories; no statistical correction",
    "hybrid": "Measured-history deposition model plus four profile corrections predicted from halo masses and concentration",
    "direct": "Five profile coordinates predicted directly from halo masses and concentration; no deposition model",
}


def assemble(stage="discovery", features="measured"):
    sample = CONT.load_sample(150 if stage == "pilot" else 0)
    predictions = {
        name: np.full_like(sample["target"], np.nan) for name in REPORT.NAMES
    }
    seen = np.zeros(len(sample["indices"]), int)
    records = []
    for fold in range(5):
        stem = OUTPUT / f"{stage}_measured_fold{fold}"
        record = json.loads(stem.with_suffix(".json").read_text())
        if not record["passed"]:
            raise ValueError("Unsettled fit is not a scientific comparison")
        if file_hash(stem.with_suffix(".npz")) != record["archive_sha256"]:
            raise ValueError("Fold archive changed")
        with np.load(stem.with_suffix(".npz")) as saved:
            np.testing.assert_array_equal(saved["indices"], sample["indices"])
            roles = CONT.BASE.split_roles(sample["folds"], fold)
            for name, rows in zip(
                ("training", "calibration", "evaluation"), roles, strict=True
            ):
                np.testing.assert_array_equal(saved[name], rows)
            held = roles[-1]
            np.testing.assert_array_equal(saved["truth"], sample["target"][held])
            seen[held] += 1
            for name in REPORT.NAMES:
                predictions[name][held] = saved[features + "_" + name]
        records.append(record)
    if not np.all(seen == 1):
        raise ValueError("Galaxies must be evaluated once")
    for prediction in predictions.values():
        if not np.isfinite(prediction).all() or np.any(prediction <= 0):
            raise ValueError("Invalid held-out prediction")
        if np.min(np.diff(prediction, axis=-1) / prediction[..., -1, None]) < -1e-12:
            raise ValueError("Non-monotone predicted CoG")
    return sample, predictions, records


def aperture(cogs, radii, radius):
    if radius == 0:
        return np.zeros(cogs.shape[:-1])
    if not radii[0] <= radius <= radii[-1]:
        raise ValueError("Aperture outside measured grid")
    return np.array(
        [np.interp(radius, radii, curve) for curve in cogs.reshape(-1, len(radii))]
    ).reshape(cogs.shape[:-1])


def annuli(cogs, radii):
    edges = [0, 2, 10, 30, 50, 100, float(radii[-1])]
    return np.diff(
        np.stack([aperture(cogs, radii, edge) for edge in edges], axis=-1), axis=-1
    )


def radial_summary(sample, predictions):
    truth, radii = sample["target"], sample["radii"]
    true_annuli = annuli(truth, radii)
    result = {}
    for name, prediction in predictions.items():
        predicted_annuli = annuli(prediction, radii)
        relative = (predicted_annuli - true_annuli) / truth[..., -1, None]
        valid = (predicted_annuli > 0) & (true_annuli > 0)
        log_error = np.full_like(relative, np.nan)
        log_error[valid] = np.log10(predicted_annuli[valid] / true_annuli[valid])
        sizes = {}
        for fraction in (0.5, 0.8, 0.9):
            true_size = np.array(
                [
                    qa.enclosed_radius(curve, radii, fraction)
                    for curve in truth.reshape(-1, len(radii))
                ]
            ).reshape(truth.shape[:2])
            pred_size = np.array(
                [
                    qa.enclosed_radius(curve, radii, fraction)
                    for curve in prediction.reshape(-1, len(radii))
                ]
            ).reshape(truth.shape[:2])
            sizes[str(fraction)] = {
                "median_log_prediction_over_data_dex_by_epoch": np.median(
                    np.log10(pred_size / true_size), axis=0
                ),
                "data_radius_below_grid_count_by_epoch": np.sum(
                    fraction * truth[..., -1] <= truth[..., 0], axis=0
                ),
                "model_radius_below_grid_count_by_epoch": np.sum(
                    fraction * prediction[..., -1] <= prediction[..., 0], axis=0
                ),
            }
        result[name] = {
            "annular_edges_kpc": [0, 2, 10, 30, 50, 100, float(radii[-1])],
            "annular_rms_residual_over_true_total_by_epoch": np.sqrt(
                np.mean(relative**2, axis=0)
            ),
            "annular_median_log_bias_dex_by_epoch": np.nanmedian(log_error, axis=0),
            "annular_log_valid_counts_by_epoch": valid.sum(0),
            "annular_data_nonpositive_counts_by_epoch": (true_annuli <= 0).sum(0),
            "outer_two_annuli_rms_residual_over_true_total": float(
                np.sqrt(np.mean(relative[..., -2:] ** 2))
            ),
            "sizes": sizes,
        }
    return result


def future_growth(sample, predictions):
    """Diagnostic partial slopes at fixed measured epoch mass, not a fit input."""
    result = {}
    values = {"data": sample["target"], **predictions}
    rng = np.random.default_rng(75109)
    for epoch in range(1, 5):
        mass = sample["measured_epoch_mass"][:, epoch]
        growth = sample["measured_epoch_mass"][:, 0] - mass
        valid = np.isfinite(mass) & np.isfinite(growth)
        mass, growth = mass[valid] - 13, growth[valid]
        design = np.column_stack((np.ones(len(mass)), mass, mass**2, growth))
        labels = np.column_stack(
            [
                np.log10(aperture(value, sample["radii"], 100)[valid, epoch])
                for value in values
            ]
        )
        slopes = np.linalg.lstsq(design, labels, rcond=None)[0][-1]
        bootstrap = []
        for _ in range(1000):
            rows = rng.integers(0, len(mass), len(mass))
            bootstrap.append(
                np.linalg.lstsq(design[rows], labels[rows], rcond=None)[0][-1]
            )
        bootstrap = np.array(bootstrap)
        result[str(CONT.BASE.REDSHIFTS[epoch])] = {
            "galaxies_with_measured_halo_mass": int(valid.sum()),
            "definition": "Partial slope of log Mstar(<100 kpc) versus log[M200c(z=.4)/M200c(epoch)], holding quadratic log M200c(epoch) fixed; dex/dex",
            "models": {
                name: {
                    "partial_slope_dex_per_dex": float(slopes[column]),
                    "partial_slope_bootstrap_95_interval": np.percentile(
                        bootstrap[:, column], [2.5, 97.5]
                    ),
                    "residual_slope_model_minus_data": float(
                        slopes[column] - slopes[0]
                    ),
                    "residual_slope_bootstrap_95_interval": np.percentile(
                        bootstrap[:, column] - bootstrap[:, 0], [2.5, 97.5]
                    ),
                }
                for column, name in enumerate(values)
            },
        }
    return result


def oracle(sample, baseline):
    path = OUTPUT / "representation.npz"
    if path.exists():
        with np.load(path) as saved:
            np.testing.assert_array_equal(saved["indices"], sample["indices"])
            return saved["oracle"], saved["direct_reconstruction"]
    coefficients = np.empty(sample["target"].shape[:2] + (4,))
    started = time.perf_counter()
    for row in range(len(coefficients)):
        CONT.check_deadline()
        for epoch in range(5):
            coefficients[row, epoch] = fit_coefficients(
                baseline[row, epoch], sample["target"][row, epoch], sample["radii"]
            )
    prediction = apply_correction(baseline, coefficients, sample["radii"])
    direct = CONT.BASE.decode(
        CONT.BASE.coordinate_targets(sample["target"], sample["radii"]), sample["radii"]
    )
    np.savez_compressed(
        path,
        indices=sample["indices"],
        oracle=prediction,
        direct_reconstruction=direct,
        coefficients=coefficients,
    )
    CONT.write_record(
        path.with_suffix(".json"),
        {
            "seconds": time.perf_counter() - started,
            "archive_sha256": file_hash(path),
            "note": "Direct fits to evaluated stellar profiles diagnose representation only; not halo predictions and never used to train a map.",
        },
    )
    return prediction, direct


def summary():
    started = time.perf_counter()
    sample, predictions, records = assemble()
    mass_sample = dict(sample)
    mass_sample["halo"] = sample["halo"].copy()
    mass_sample["halo"][:, 0] = sample["measured_epoch_mass"][:, 0]
    result = {
        "galaxies": len(sample["indices"]),
        "model_definitions": DEFINITIONS,
        "feature_comparisons": {},
    }
    for features in ("original", "measured"):
        _, values, _ = assemble(features=features)
        result["feature_comparisons"][features] = REPORT.statistics(
            mass_sample, values
        )[0]
    oracle_prediction, reconstructed = oracle(sample, predictions["baseline"])
    main = {name: predictions[name] for name in DEFINITIONS}
    main.update(oracle=oracle_prediction, direct_reconstruction=reconstructed)
    result["radial_diagnostics"] = radial_summary(sample, main)
    result["representation_mean_galaxy_radial_rms_dex"] = {
        name: float(
            np.sqrt(np.mean(np.log10(value / sample["target"]) ** 2, axis=-1)).mean()
        )
        for name, value in main.items()
    }
    result["future_growth"] = future_growth(
        sample, {name: predictions[name] for name in DEFINITIONS}
    )
    result["integration_max_dex_by_fold"] = [
        record["attempts"][-1]["integration_max_dex"] for record in records
    ]
    result["fit_seconds_by_fold"] = [record["seconds"] for record in records]
    result["historical_official_mass_bin_scores"] = REPORT.statistics(
        sample, predictions
    )[0]
    result["coordinate_audit"] = STANDARD.recover_applied_coordinates(
        predictions["baseline"], predictions["hybrid"], sample["radii"]
    )
    result["interpretation_limits"] = [
        "Point predictions, not samples from a stellar-profile distribution; narrower mass planes alone do not disqualify a mean model.",
        "Direct model uses 80% training; hybrid correction 20% calibration plus a physical baseline fitted on 60%.",
        "Only original discovery IDs; no selection or validation role. Discovery gains do not qualify production.",
        "Intervals condition on fitted folds. Full halo histories are legitimate inputs; stellar labels never enter predictions.",
        "CoG fitting treats radial residuals equally as a diagnostic loss, not independent statistical measurements or a likelihood.",
    ]
    result["seconds"] = time.perf_counter() - started
    CONT.write_record(OUTPUT / "summary.json", STANDARD.json_ready(result))
    print(
        json.dumps(STANDARD.json_ready(result["feature_comparisons"]), indent=2),
        flush=True,
    )


def standard_figures(name):
    sample, predictions, _ = assemble()
    destination = OUTPUT / f"standard_qa_{name}.json"
    if destination.exists():
        raise FileExistsError(destination)
    folder = FIGURES / name
    if folder.exists():
        raise FileExistsError(folder)
    started = time.perf_counter()
    plt.rcParams["text.usetex"] = False
    metrics = qa.evaluate(
        predictions[name],
        sample["target"],
        sample["radii"],
        CONT.BASE.REDSHIFTS,
        name=f"exp75_measured_{name}",
        figdir=folder,
        verbose=False,
        figures=True,
        bin_by=sample["measured_epoch_mass"][:, 0],
        bin_label=r"$\log M_{200c}(z=0.4)$",
        halo_mass_epochs=sample["measured_epoch_mass"],
    )
    paths = sorted(folder.glob("*.png"))
    record = {
        "definition": DEFINITIONS[name],
        "seconds": time.perf_counter() - started,
        "metrics": metrics,
        "figures": [str(path) for path in paths],
        "caption": "Held-out halo-only point predictions versus the same 842 measured CoGs at five epochs. CoG-derived masses, not direct projected apertures. See summary.json and README for quantitative interpretation. Out-of-grid apertures are NaN; no 150 kpc mass is inferred from the 148.22 kpc endpoint.",
    }
    CONT.write_record(destination, STANDARD.json_ready(record))
    print(
        json.dumps(
            {"model": name, "seconds": record["seconds"], "figures": record["figures"]},
            indent=2,
        ),
        flush=True,
    )


def overview():
    sample, predictions, _ = assemble()
    mass_sample = dict(sample)
    mass_sample["halo"] = sample["halo"].copy()
    mass_sample["halo"][:, 0] = sample["measured_epoch_mass"][:, 0]
    _, errors, rms, bins = REPORT.statistics(mass_sample, predictions)
    paths = REPORT.figures(
        mass_sample, predictions, errors, rms, bins, "continuation/measured_discovery"
    )
    CONT.write_record(OUTPUT / "overview_figures.json", {"figures": paths})
    print(json.dumps(paths, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("summary", "overview", "qa"))
    parser.add_argument("--model", choices=tuple(DEFINITIONS), default="hybrid")
    arguments = parser.parse_args()
    with threadpool_limits(limits=1):
        CONT.check_deadline()
        if arguments.stage == "summary":
            summary()
        elif arguments.stage == "overview":
            overview()
        else:
            standard_figures(arguments.model)
