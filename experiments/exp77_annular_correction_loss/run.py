# %%
"""Isolated snapshot, operational gate, and immutable Exp77 fold checkpoints."""

# ruff: noqa: E402
import time

STARTED = time.perf_counter()

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone

from annular import (
    HERE,
    ROOT,
    REFERENCE,
    REF,
    apply_correction,
    fit_labels,
    predict_from_calibration,
    scores,
)

import matplotlib.pyplot as plt
import numpy as np
from threadpoolctl import threadpool_limits
from hongshao.plotting import save_fig, set_style

OUTPUT = HERE / "outputs"
FIGURES = HERE / "figures"


def write_record(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(REF.STANDARD.json_ready(value), stream, indent=2, allow_nan=False)
        stream.write("\n")


def save_archive(path, **values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **values)


def provenance():
    return {
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "code_sha256": {path.name: REF.file_hash(path) for path in HERE.glob("*.py")},
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_record_sha256": REF.file_hash(OUTPUT / "inputs.json"),
    }


def prepare():
    source = (
        ROOT.parent
        / "hongshao_exp75_halo_residual_correction/experiments/exp75_halo_residual_correction"
    )
    manifest_path = source / "outputs/continuation/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    names = [
        "outputs/inputs.npz",
        "outputs/inputs.json",
        "outputs/continuation/inputs.npz",
        "outputs/continuation/inputs.json",
    ]
    names += [
        f"outputs/continuation/discovery_measured_fold{fold}.{suffix}"
        for fold in range(5)
        for suffix in ("npz", "json")
    ]
    hashes = {name: REF.file_hash(source / name) for name in names}
    for name, digest in hashes.items():
        if digest != manifest["files_sha256"][name]:
            raise ValueError(f"Closed Exp75 artifact differs from its manifest: {name}")
        destination = REFERENCE / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with (source / name).open("rb") as incoming, destination.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing)
        if (
            REF.file_hash(source / name) != digest
            or REF.file_hash(destination) != digest
        ):
            raise ValueError("Source or copy changed")
    write_record(
        OUTPUT / "inputs.json",
        {
            "source": str(source),
            "source_manifest_sha256": REF.file_hash(manifest_path),
            "files_sha256": hashes,
            "selection_validation_role": "none",
        },
    )
    print(json.dumps({"copied_verified_files": len(names)}))


def load_sample(limit=0):
    record = json.loads((OUTPUT / "inputs.json").read_text())
    for name, digest in record["files_sha256"].items():
        if REF.file_hash(REFERENCE / name) != digest:
            raise ValueError("Private reference artifact changed")
    sample = REF.CONT.feature_sample(REF.CONT.load_sample(0), "measured")
    rows = np.arange(len(sample["indices"]))
    if limit:
        order = np.argsort(sample["measured_epoch_mass"][:, 0], kind="stable")
        rows = order[np.linspace(0, len(order) - 1, limit).round().astype(int)]
    return sample, rows


def comparison_figure(sample, rows, predictions, name):
    set_style()
    plt.rcParams["text.usetex"] = False
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True)
    for column, (row, epoch) in enumerate(
        zip(rows[[0, len(rows) // 2, -1]], (0, 2, 4), strict=True)
    ):
        position = int(np.flatnonzero(rows == row)[0])
        radius, truth = sample["radii"], sample["target"][row, epoch]
        axes[0, column].plot(radius, np.log10(truth), "ko", ms=3, label="Measured CoG")
        for key, color, label in (
            ("reference", "#0072B2", "Cumulative-loss correction"),
            ("selected", "#D55E00", "Calibration-selected annular loss"),
        ):
            curve = predictions[key][position, epoch]
            axes[0, column].plot(radius, np.log10(curve), c=color, label=label)
            axes[1, column].plot(radius, np.log10(curve / truth), c=color)
        axes[0, column].set_title(
            f"ID {sample['indices'][row]}; z={REF.CONT.BASE.REDSHIFTS[epoch]}"
        )
        axes[0, column].set_ylabel("log stellar mass within R (Msun)")
        axes[1, column].set_ylabel("log prediction/data (dex)")
        axes[1, column].set_xlabel("Semi-major radius (kpc)")
        axes[1, column].axhline(0, c="0.5", lw=0.7)
        for axis in axes[:, column]:
            axis.set_xscale("log")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(
        "Exp77 held-out profiles: same baseline and halo inputs, different correction loss\nOperational check only"
        if name == "gate_individual_cogs"
        else "Exp77 held-out profiles: same baseline and halo inputs, different correction loss"
    )
    fig.tight_layout()
    paths = save_fig(fig, FIGURES / name)
    plt.close(fig)
    return [str(path) for path in paths]


def run_fold(stage, fold):
    REF.CONT.check_deadline()
    path = OUTPUT / f"{stage}_fold{fold}.json"
    if path.exists() or path.with_suffix(".npz").exists():
        raise FileExistsError(path)
    if (
        stage != "gate"
        and not json.loads((OUTPUT / "gate_fold0.json").read_text())["passed"]
    ):
        raise ValueError("Operational gate must pass")
    if stage == "discovery":
        for number in range(5):
            if not json.loads((OUTPUT / f"pilot_fold{number}.json").read_text())[
                "passed"
            ]:
                raise ValueError("All pilot folds must pass")
    sample, subset = load_sample(
        30 if stage == "gate" else 150 if stage == "pilot" else 0
    )
    saved_path = REFERENCE / f"outputs/continuation/discovery_measured_fold{fold}"
    saved_record = json.loads(saved_path.with_suffix(".json").read_text())
    choice = saved_record["choices"]["measured"]["hybrid_choice"]
    with np.load(saved_path.with_suffix(".npz")) as saved:
        np.testing.assert_array_equal(saved["indices"], sample["indices"])
        calibration = saved["calibration"][np.isin(saved["calibration"], subset)]
        evaluation = saved["evaluation"][np.isin(saved["evaluation"], subset)]
        baseline = saved["baseline_all"]
        expected = saved["measured_hybrid"]
    predictions, selection = predict_from_calibration(
        sample,
        baseline,
        calibration,
        evaluation,
        choice,
        expected_reference=expected if stage == "discovery" else None,
    )
    parity = (
        float(np.max(np.abs(np.log10(predictions["reference"] / expected))))
        if stage == "discovery"
        else None
    )
    if parity is not None and parity > 1e-8:
        raise ValueError(f"Weight-zero reference changed: {parity} dex")
    for prediction in predictions.values():
        if (
            not np.isfinite(prediction).all()
            or np.any(prediction <= 0)
            or np.min(np.diff(prediction, axis=-1)) < 0
        ):
            raise ValueError("Invalid predicted CoG")
    paths = (
        comparison_figure(sample, evaluation, predictions, "gate_individual_cogs")
        if stage == "gate"
        else []
    )
    save_archive(
        path.with_suffix(".npz"),
        indices=sample["indices"],
        calibration=calibration,
        evaluation=evaluation,
        **predictions,
    )
    elapsed = time.perf_counter() - STARTED
    record = {
        **provenance(),
        "stage": stage,
        "fold": fold,
        "seconds": elapsed,
        "passed": elapsed < 60 if stage == "gate" else True,
        "reference_max_error_dex": parity,
        "calibration_count": len(calibration),
        "evaluation_count": len(evaluation),
        "selection": selection,
        "scores": {
            name: scores(value, sample["target"][evaluation], sample["radii"])
            for name, value in predictions.items()
        },
        "figures": paths,
        "archive_sha256": REF.file_hash(path.with_suffix(".npz")),
    }
    write_record(path, record)
    print(
        json.dumps(
            {
                key: record[key]
                for key in (
                    "stage",
                    "fold",
                    "seconds",
                    "passed",
                    "reference_max_error_dex",
                    "scores",
                    "figures",
                )
            }
        ),
        flush=True,
    )


def representation(stage):
    destination = OUTPUT / f"{stage}_representation.json"
    if destination.exists() or destination.with_suffix(".npz").exists():
        raise FileExistsError(destination)
    sample, rows = load_sample(150 if stage == "pilot" else 0)
    baseline = np.empty_like(sample["target"])
    for fold in range(5):
        with np.load(
            REFERENCE / f"outputs/continuation/discovery_measured_fold{fold}.npz"
        ) as archive:
            held = archive["evaluation"]
            baseline[held] = archive["baseline_all"][held]
    predictions, records = {}, {}
    for weight in (0, 0.25, 1):
        result = np.empty_like(sample["target"][rows])
        retries = 0
        for position, row in enumerate(rows):
            REF.CONT.check_deadline()
            for epoch in range(5):
                coefficients, record = fit_labels(
                    baseline[row, epoch],
                    sample["target"][row, epoch],
                    sample["radii"],
                    weight,
                )
                result[position, epoch] = apply_correction(
                    baseline[row, epoch], coefficients, sample["radii"]
                )
                retries += record["retry"]
        predictions[str(weight)] = result
        records[str(weight)] = {
            **scores(result, sample["target"][rows], sample["radii"]),
            "fit_retries": retries,
        }
        print(
            json.dumps({"diagnostic_weight": weight, **records[str(weight)]}),
            flush=True,
        )
    save_archive(
        destination.with_suffix(".npz"),
        indices=sample["indices"][rows],
        rows=rows,
        **predictions,
    )
    write_record(
        destination,
        {
            **provenance(),
            "uses_true_stellar_profiles_diagnostic_only": True,
            "scores": records,
            "seconds": time.perf_counter() - STARTED,
            "archive_sha256": REF.file_hash(destination.with_suffix(".npz")),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "prepare",
            "gate",
            "pilot",
            "discovery",
            "pilot_representation",
            "discovery_representation",
        ),
    )
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    arguments = parser.parse_args()
    with threadpool_limits(limits=1):
        if arguments.stage == "prepare":
            prepare()
        elif arguments.stage.endswith("_representation"):
            representation(arguments.stage.split("_")[0])
        else:
            run_fold(arguments.stage, arguments.fold)


if __name__ == "__main__":
    try:
        main()
    except (Exception, KeyboardInterrupt) as error:
        write_record(
            OUTPUT
            / f"stopped_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}.json",
            {"status": "stopped", "error": repr(error)},
        )
        raise
