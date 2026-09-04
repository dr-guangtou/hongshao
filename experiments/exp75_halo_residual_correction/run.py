# %%
"""Exp75 operational gate and checkpointed, serial discovery pilot."""

# ruff: noqa: E402 -- timing and private cache settings must precede imports.

import time

START_TIME = time.perf_counter()

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits

from correction import (
    apply_correction,
    fit_coefficients,
    fit_map,
    make_folds,
    split_roles,
)
from prepare import file_hash, load_module
from hongshao.plotting import save_fig, set_style

PHYSICAL = load_module(
    "exp75_physical", ROOT / "experiments/exp63_analytic_growth/model2.py"
)
COORDINATES = load_module(
    "exp75_coordinates", ROOT / "experiments/exp51_direct_cog_redshift/model.py"
)
OUTPUT = HERE / "outputs"
REDSHIFTS = (0.4, 0.7, 1, 1.5, 2)
STARTS = np.array(
    [
        [-3, -0.5, 0.3, 0, 12.5, 0.7, 0.4, -0.5, -0.8, -0.3, 1, 0.8],
        [-3.5, -0.8, 0.5, 0.1, 13, 1, 0.7, -0.8, -0.6, -0.5, 0.7, 1],
    ]
)


def load_inputs(limit):
    path = OUTPUT / "inputs.npz"
    record = json.loads(path.with_suffix(".json").read_text())
    if file_hash(path) != record["snapshot_sha256"]:
        raise ValueError("private input snapshot changed")
    with np.load(path) as archive:
        sample = {key: archive[key] for key in archive.files}
    if not np.isin(sample["indices"], sample["discovery_indices"]).all():
        raise ValueError("non-discovery galaxy in input")
    n_galaxy = len(sample["indices"])
    if limit:
        order = np.argsort(sample["halo"][:, 0], kind="stable")
        rows = order[
            np.linspace(0, n_galaxy - 1, min(limit, n_galaxy)).round().astype(int)
        ]
        for key in ("indices", "halo", "core", "history", "target", "epoch_mass"):
            sample[key] = sample[key][rows]
    sample["folds"] = make_folds(sample["halo"][:, 0])
    return sample


def build_curves(halo, indices):
    """The analytic mode reads only four halo parameters; no stellar fields."""
    empty = np.array([], float)
    return [
        PHYSICAL.E.HaloCurve(
            row,
            int(index),
            *parameters,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            np.array([], bool),
            np.zeros((5, 0), bool),
        )
        for row, (index, parameters) in enumerate(zip(indices, halo, strict=True))
    ]


def fit_baseline(sample, training, operational):
    spec = PHYSICAL.Spec2(compact_in_kpc=True)
    curves = build_curves(sample["halo"], sample["indices"])
    training_curves = [curves[row] for row in training]
    target = sample["target"][training]
    lower, upper = np.array(spec.bounds()).T
    results, records = [], []
    for start_number, initial in enumerate(STARTS[:1] if operational else STARTS):
        calls = 0
        started = time.perf_counter()

        def residual(theta):
            nonlocal calls
            calls += 1
            prediction = PHYSICAL.predict2(
                spec, theta, training_curves, sample["radii"], nodes=PHYSICAL.FIT_NODES
            )
            if not np.isfinite(prediction).all() or np.any(prediction <= 0):
                raise ValueError("baseline produced an unbuildable model")
            return np.log10(prediction / target).ravel()

        fitted = least_squares(
            residual,
            np.clip(initial, lower + 1e-8, upper - 1e-8),
            bounds=(lower, upper),
            x_scale=upper - lower,
            max_nfev=3 if operational else 300,
            ftol=1e-6,
            xtol=1e-6,
            gtol=1e-6,
        )
        results.append(fitted)
        record = {
            "start": start_number,
            "success": bool(fitted.success),
            "message": fitted.message,
            "nfev": fitted.nfev,
            "actual_calls": calls,
            "loss": float(np.mean(fitted.fun**2)),
            "seconds": time.perf_counter() - started,
            "theta": fitted.x.tolist(),
            "boundary_distance": np.minimum(
                (fitted.x - lower) / (upper - lower),
                (upper - fitted.x) / (upper - lower),
            ).tolist(),
        }
        records.append(record)
        print(json.dumps(record), flush=True)
    best = min(results, key=lambda result: np.mean(result.fun**2))
    prediction = PHYSICAL.predict2(
        spec, best.x, curves, sample["radii"], nodes=PHYSICAL.FULL_NODES
    )
    approximate = PHYSICAL.predict2(
        spec, best.x, curves, sample["radii"], nodes=PHYSICAL.FIT_NODES
    )
    integration_error = float(np.max(np.abs(np.log10(prediction / approximate))))
    losses = [record["loss"] for record in records]
    settled = (
        not operational
        and all(record["success"] for record in records)
        and max(losses) / min(losses) <= 1.01
        and integration_error <= 0.001
    )
    return prediction, records, integration_error, settled


def coordinate_targets(target, radii):
    return np.stack(
        [
            COORDINATES.compress_cogs(np.log10(target[:, epoch]), radii, "quantile3")
            for epoch in range(5)
        ],
        axis=1,
    )


def decode(predicted, radii):
    return np.stack(
        [
            10 ** COORDINATES.reconstruct_cogs(predicted[:, epoch], radii, "quantile3")
            for epoch in range(5)
        ],
        axis=1,
    )


def select_map(features, targets, truth, baseline, radii, direct=False):
    """Select only with inner-held-out calibration labels."""
    inner = np.arange(len(targets)) % 2
    candidates = []
    for feature_name, values in features.items():
        for degree in (1, 2):
            for penalty in (0.01, 0.1, 1, 10, 100):
                scores = []
                for fold in (0, 1):
                    training, held = inner != fold, inner == fold
                    fitted = fit_map(
                        values[training],
                        targets[training].reshape(training.sum(), -1),
                        degree,
                        penalty,
                    )
                    parameters = fitted.predict(values[held]).reshape(held.sum(), 5, -1)
                    predicted = (
                        decode(parameters, radii)
                        if direct
                        else apply_correction(baseline[held], parameters, radii)
                    )
                    scores.append(np.mean(np.log10(predicted / truth[held]) ** 2))
                candidates.append(
                    (
                        float(np.mean(scores)),
                        degree,
                        values.shape[1],
                        -penalty,
                        feature_name,
                    )
                )
    best = min(value[0] for value in candidates)
    selected = min(
        (value for value in candidates if value[0] <= best * 1.01),
        key=lambda value: value[1:],
    )
    score, degree, _, negative_penalty, feature_name = selected
    fitted = fit_map(
        features[feature_name],
        targets.reshape(len(targets), -1),
        degree,
        -negative_penalty,
    )
    return fitted, {
        "feature": feature_name,
        "degree": degree,
        "penalty": -negative_penalty,
        "inner_mse_dex2": score,
    }


def shuffled_features(values, masses, edges, seed):
    rng = np.random.default_rng(seed)
    result = values.copy()
    bins = np.searchsorted(edges[1:-1], masses)
    for mass_bin in range(len(edges) - 1):
        rows = np.flatnonzero(bins == mass_bin)
        # Keep the final mass, shuffle the other halo variables together.
        result[rows, 1:] = values[rng.permutation(rows), 1:]
    return result


def compare(sample, baseline, training, calibration, evaluation):
    radii, truth = sample["radii"], sample["target"]
    labels = np.array(
        [
            [
                fit_coefficients(baseline[row, epoch], truth[row, epoch], radii)
                for epoch in range(5)
            ]
            for row in calibration
        ]
    )
    predictions = {"baseline": baseline[evaluation]}
    intercept = labels.mean(0)
    predictions["intercept"] = apply_correction(baseline[evaluation], intercept, radii)
    features = {name: sample[name][calibration] for name in ("core", "history")}
    fitted, choice = select_map(
        features, labels, truth[calibration], baseline[calibration], radii
    )
    coefficients = fitted.predict(sample[choice["feature"]][evaluation]).reshape(
        len(evaluation), 5, 4
    )
    predictions["hybrid"] = apply_correction(baseline[evaluation], coefficients, radii)
    mass_map = fit_map(
        sample["halo"][calibration, :1],
        labels.reshape(len(labels), -1),
        1,
        choice["penalty"],
    )
    mass_parameters = mass_map.predict(sample["halo"][evaluation, :1]).reshape(
        len(evaluation), 5, 4
    )
    predictions["mass_only"] = apply_correction(
        baseline[evaluation], mass_parameters, radii
    )
    edges = np.quantile(sample["halo"][training, 0], np.linspace(0, 1, 6))
    selected_features = sample[choice["feature"]]
    shuffled_training = shuffled_features(
        selected_features[calibration], sample["halo"][calibration, 0], edges, 750
    )
    shuffled_test = shuffled_features(
        selected_features[evaluation], sample["halo"][evaluation, 0], edges, 751
    )
    null_map = fit_map(
        shuffled_training,
        labels.reshape(len(labels), -1),
        choice["degree"],
        choice["penalty"],
    )
    null_parameters = null_map.predict(shuffled_test).reshape(len(evaluation), 5, 4)
    predictions["shuffled"] = apply_correction(
        baseline[evaluation], null_parameters, radii
    )
    union = np.sort(np.r_[training, calibration])
    direct_targets = coordinate_targets(truth[union], radii)
    direct_features = {name: sample[name][union] for name in ("core", "history")}
    direct_map, direct_choice = select_map(
        direct_features, direct_targets, truth[union], None, radii, direct=True
    )
    direct_parameters = direct_map.predict(
        sample[direct_choice["feature"]][evaluation]
    ).reshape(len(evaluation), 5, 5)
    predictions["direct"] = decode(direct_parameters, radii)
    scores = {
        name: np.sqrt(np.mean(np.log10(value / truth[evaluation]) ** 2, axis=-1))
        .mean(0)
        .tolist()
        for name, value in predictions.items()
    }
    return predictions, {
        "hybrid_choice": choice,
        "direct_choice": direct_choice,
        "mean_galaxy_rms_dex_by_epoch": scores,
    }


def direct_figure(sample, evaluation, predictions, tag):
    set_style()
    plt.rcParams["text.usetex"] = False
    radii = sample["radii"]
    figure, axes = plt.subplots(2, 3, figsize=(11, 6.5), sharex=True)
    ordered = np.argsort(sample["halo"][evaluation, 0])
    choices = ordered[np.linspace(0, len(ordered) - 1, 3).round().astype(int)]
    styles = {
        "baseline": ("#0072B2", "--"),
        "hybrid": ("#D55E00", "-"),
        "direct": ("#009E73", ":"),
    }
    for column, row in enumerate(choices):
        epoch = (0, 2, 4)[column]
        actual = sample["target"][evaluation[row], epoch]
        axes[0, column].plot(radii, np.log10(actual), "ko", ms=3, label="Measured CoG")
        for name, (color, linestyle) in styles.items():
            prediction = predictions[name][row, epoch]
            axes[0, column].plot(
                radii, np.log10(prediction), color=color, ls=linestyle, label=name
            )
            axes[1, column].plot(
                radii, np.log10(prediction / actual), color=color, ls=linestyle
            )
        axes[0, column].set_title(
            f"Galaxy {sample['indices'][evaluation[row]]}; z={REDSHIFTS[epoch]}"
        )
        axes[1, column].axhline(0, color="0.5", lw=0.7)
        axes[1, column].set_xlabel("Semi-major radius (kpc)")
        for axis in axes[:, column]:
            axis.set_xscale("log")
    axes[0, 0].set_ylabel(r"$\log_{10} M_*(<R)\ [M_\odot]$")
    axes[1, 0].set_ylabel(r"$\log_{10}(\mathrm{prediction}/\mathrm{data})$ (dex)")
    axes[0, 0].legend(fontsize=8)
    caption = (
        f"Exp75 {tag}: {len(evaluation)} held-out galaxies, five epochs. "
        "Three examples selected by halo mass, not fit quality.\n"
        "Operational fits are capped: this figure checks execution, not scientific accuracy."
        if tag == "gate"
        else f"Exp75 {tag}: independent evaluation galaxies; provisional pilot fits."
    )
    figure.suptitle(caption, fontsize=9)
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    paths = save_fig(figure, HERE / "figures" / f"{tag}_individual_cogs")
    plt.close(figure)
    return [str(path) for path in paths]


def run(stage, rotation):
    operational = stage == "gate"
    if operational:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "unittest",
                "discover",
                "-s",
                str(HERE),
                "-p",
                "test_exp75.py",
            ],
            check=True,
            cwd=ROOT,
        )
    if not operational:
        gate = json.loads((OUTPUT / "gate.json").read_text())
        if not gate["passed"]:
            raise RuntimeError("operational gate did not pass")
    if stage == "discovery":
        for pilot_rotation in range(5):
            pilot = json.loads((OUTPUT / f"pilot_fold{pilot_rotation}.json").read_text())
            if not pilot["passed"]:
                raise RuntimeError("all five pilot rotations must pass first")
    tag = "gate" if operational else f"{stage}_fold{rotation}"
    destination = OUTPUT / f"{tag}.json"
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    sample = load_inputs(30 if operational else (150 if stage == "pilot" else 0))
    training, calibration, evaluation = split_roles(sample["folds"], rotation)
    baseline, starts, integration_error, settled = fit_baseline(
        sample, training, operational
    )
    predictions, scores = compare(sample, baseline, training, calibration, evaluation)
    paths = direct_figure(sample, evaluation, predictions, tag) if rotation == 0 else []
    archive = OUTPUT / f"{tag}.npz"
    np.savez_compressed(
        archive,
        indices=sample["indices"],
        training=training,
        calibration=calibration,
        evaluation=evaluation,
        baseline_all=baseline,
        truth=sample["target"][evaluation],
        **predictions,
    )
    with np.load(archive) as saved:
        np.testing.assert_array_equal(saved["hybrid"], predictions["hybrid"])
    np.testing.assert_array_equal(
        apply_correction(baseline[evaluation], np.zeros(4), sample["radii"]),
        baseline[evaluation],
    )
    elapsed = time.perf_counter() - START_TIME
    record = {
        "stage": stage,
        "rotation": rotation,
        "sample_galaxies": len(sample["indices"]),
        "role_counts": [len(training), len(calibration), len(evaluation)],
        "starts": starts,
        "integration_max_dex": integration_error,
        "baseline_settled": settled,
        "seconds": elapsed,
        "passed": elapsed < 60 if operational else settled,
        "scientific_result": False,
        "figures": paths,
        **scores,
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "input_sha256": file_hash(OUTPUT / "inputs.npz"),
        "code_sha256": {path.name: file_hash(path) for path in sorted(HERE.glob("*.py"))},
    }
    destination.write_text(json.dumps(record, indent=2) + "\n")
    (OUTPUT / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)
    if not record["passed"]:
        raise SystemExit(
            "STOP: runtime/optimization gate not passed; no scientific null claimed"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("gate", "pilot", "discovery"))
    parser.add_argument("--rotation", type=int, choices=range(5), default=0)
    arguments = parser.parse_args()
    with threadpool_limits(limits=1):
        run(arguments.stage, arguments.rotation)
