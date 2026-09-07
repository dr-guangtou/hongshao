# %%
"""Private, checkpointed measured-history continuation of the frozen Exp75 study."""

# ruff: noqa: E402
import time

START_TIME = time.perf_counter()
import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from prepare import file_hash, load_module

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = load_module("exp75_continuation_base", HERE / "run.py")
MEASURED = load_module(
    "exp75_measured", ROOT / "experiments/exp74_c19_history_leak/measured.py"
)
import numpy as np
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits

OUTPUT = HERE / "outputs/continuation"
DEADLINE = datetime(2026, 9, 7, 22, 0, tzinfo=timezone.utc)
HISTORY_KEYS = ("logmp", "logtc", "early", "late", "logt0")


def check_deadline(now=None):
    if (now or datetime.now(timezone.utc)) >= DEADLINE:
        raise TimeoutError("Science cutoff reached; preserve checkpoints and close out")


def check_roles(training, calibration, evaluation, count):
    combined = np.concatenate((training, calibration, evaluation))
    if not np.array_equal(np.sort(combined), np.arange(count)):
        raise ValueError("Roles must partition the sample without overlap")


def scaled_nodes(original, factor):
    return {
        name: value * factor if name in ("n_early", "n_node") else value
        for name, value in original.items()
    }


def write_record(path, record):
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n")


def prepare_inputs():
    destination = OUTPUT / "inputs.npz"
    if destination.exists():
        raise FileExistsError(destination)
    source_root = Path("/Users/shuang/Dropbox/work/project/massive/hongshao")
    sources = {
        "structure": source_root
        / "experiments/exp54_unpinned_amplitude/outputs/halo_structure_history.npz",
        "fits": source_root
        / "experiments/exp74_c19_history_leak/outputs/history_curves.npz",
    }
    before = {name: file_hash(path) for name, path in sources.items()}
    sample = BASE.load_inputs(0)
    with np.load(sources["structure"]) as archive:
        index = archive["index"]
        if len(np.unique(index)) != len(index):
            raise ValueError("Duplicate structure IDs")
        lookup = {int(value): row for row, value in enumerate(index)}
        rows = np.array([lookup[int(value)] for value in sample["indices"]])
        sample["structure_rows"] = rows
        sample["snaps"] = archive["snaps"]
        for key in ("M200c", "GroupFlag"):
            sample[key] = archive[key][rows]
    with np.load(sources["fits"]) as archive:
        lookup = {int(value): row for row, value in enumerate(archive["rows"])}
        if len(lookup) != len(archive["rows"]):
            raise ValueError("Duplicate history rows")
        positions = np.array([lookup[int(row)] for row in rows])
        for key in HISTORY_KEYS:
            sample["pre_" + key] = archive[key][positions]
            if not np.isfinite(sample["pre_" + key]).all():
                raise ValueError("Missing pre-epoch fits")
    after = {name: file_hash(path) for name, path in sources.items()}
    if before != after:
        raise ValueError("Source changed while being read; no snapshot written")
    columns = [
        list(sample["snaps"]).index(snap) for snap in BASE.PHYSICAL.E.ANCHOR_SNAP
    ]
    sample["measured_epoch_mass"] = sample["M200c"][:, columns]
    valid_epoch = (
        np.isfinite(sample["measured_epoch_mass"])
        & (sample["measured_epoch_mass"] > 0)
        & (sample["GroupFlag"][:, columns] == 1)
    )
    sample["measured_epoch_mass"] = np.where(
        valid_epoch, sample["measured_epoch_mass"], np.nan
    )
    if not valid_epoch[:, 0].all():
        raise ValueError("Missing final mass prevents matched mass-only controls")
    curves, info = build_inputs(sample, "measured")
    knot_error = max(
        np.max(np.abs(curve.log_mah_table(curve.lt_knots) - curve.y_knots))
        for curve in curves[0]
    )
    minimum_growth = min(
        np.min(curve.dm_dlnt_table(times) / 10 ** curve.log_mah_table(times))
        for curve in curves[0]
        for times in [np.linspace(-2, curve.lt_knots[-1], 500)]
    )
    if knot_error > 1e-10 or minimum_growth < -1e-12:
        raise ValueError("Measured interpolation failed")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **sample)
    record = {
        "snapshot_sha256": file_hash(destination),
        "original_snapshot_sha256": file_hash(BASE.OUTPUT / "inputs.npz"),
        "sources": {
            name: {"path": str(path), "sha256": before[name]}
            for name, path in sources.items()
        },
        "galaxies": len(rows),
        "mass_unit": "log10(M200c/Msun), h-free; official DiffMAH describes SubhaloMass",
        "knot_max_error_dex": float(knot_error),
        "minimum_sampled_logarithmic_growth": float(minimum_growth),
        "logarithmic_growth_roundoff_tolerance": 1e-12,
        "valid_measured_epoch_counts": valid_epoch.sum(0).tolist(),
        "measured_info": info,
    }
    write_record(destination.with_suffix(".json"), record)
    print(json.dumps(record, indent=2), flush=True)


def load_sample(limit=0):
    path = OUTPUT / "inputs.npz"
    record = json.loads(path.with_suffix(".json").read_text())
    if file_hash(path) != record["snapshot_sha256"]:
        raise ValueError("Continuation snapshot changed")
    with np.load(path) as archive:
        sample = {key: archive[key] for key in archive.files}
    original = BASE.load_inputs(limit)
    lookup = {int(value): row for row, value in enumerate(sample["indices"])}
    rows = np.array([lookup[int(value)] for value in original["indices"]])
    count = len(sample["indices"])
    for key, value in sample.items():
        if value.ndim and value.shape[0] == count:
            sample[key] = value[rows]
    sample["folds"] = original["folds"]
    np.testing.assert_array_equal(sample["target"], original["target"])
    np.testing.assert_array_equal(sample["indices"], original["indices"])
    return sample


def build_inputs(sample, kind):
    curves = BASE.build_curves(sample["halo"], sample["indices"])
    if kind == "official":
        return {epoch: curves for epoch in range(5)}, {}
    history = {key: sample["pre_" + key] for key in HISTORY_KEYS}
    if kind == "pre_epoch":
        return {
            epoch: MEASURED.HI.curves_for_epoch(curves, history, epoch)
            for epoch in range(5)
        }, {}
    measured, info = MEASURED.build_measured(
        curves, curves, sample, hist=history, verbose=False
    )
    if info["fallback"]:
        raise ValueError("Missing measured histories; no silent fallback allowed")
    info = {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in info.items()
    }
    return {epoch: measured for epoch in range(5)}, info


def predict(theta, curves, radii, nodes, rows=None):
    check_deadline()
    spec = BASE.PHYSICAL.Spec2(compact_in_kpc=True)
    if curves[0] is curves[1]:
        selected = curves[0] if rows is None else [curves[0][row] for row in rows]
        return BASE.PHYSICAL.predict2(spec, theta, selected, radii, nodes=nodes)
    return np.concatenate(
        [
            BASE.PHYSICAL.predict2(
                spec,
                theta,
                curves[epoch] if rows is None else [curves[epoch][row] for row in rows],
                radii,
                epochs=(epoch,),
                nodes=nodes,
            )
            for epoch in range(5)
        ],
        axis=1,
    )


def fit_baseline(sample, curves, training, operational, checkpoint):
    lower, upper = np.array(BASE.PHYSICAL.Spec2(compact_in_kpc=True).bounds()).T
    attempts = []
    for refinement in range(4):
        nodes = scaled_nodes(BASE.PHYSICAL.FIT_NODES, 2**refinement)
        full_nodes = scaled_nodes(BASE.PHYSICAL.FULL_NODES, 2**refinement)
        records = []
        for number, initial in enumerate(
            BASE.STARTS[:1] if operational else BASE.STARTS
        ):
            calls = 0
            invalid_trials = []
            started = time.perf_counter()

            def residual(theta):
                nonlocal calls
                calls += 1
                prediction = predict(theta, curves, sample["radii"], nodes, training)
                if not np.isfinite(prediction).all() or np.any(prediction <= 0):
                    invalid_trials.append(theta.tolist())
                    return np.full(sample["target"][training].size, np.inf)
                return np.log10(prediction / sample["target"][training]).ravel()

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
            retried = False
            if not operational and not fitted.success:
                retried = True
                fitted = least_squares(
                    residual,
                    fitted.x,
                    bounds=(lower, upper),
                    x_scale=upper - lower,
                    max_nfev=600,
                    ftol=1e-6,
                    xtol=1e-6,
                    gtol=1e-6,
                )
            record = {
                "start": number,
                "refinement": refinement,
                "retry": retried,
                "success": bool(fitted.success),
                "message": fitted.message,
                "nfev": fitted.nfev,
                "actual_calls": calls,
                "invalid_trial_parameters": invalid_trials,
                "loss": float(np.mean(fitted.fun**2)),
                "theta": fitted.x.tolist(),
                "seconds": time.perf_counter() - started,
                "boundary_distance": np.minimum(
                    (fitted.x - lower) / (upper - lower),
                    (upper - fitted.x) / (upper - lower),
                ).tolist(),
            }
            records.append(record)
            write_record(
                OUTPUT / f"{checkpoint}_resolution{refinement}_start{number}.json",
                record,
            )
            print(json.dumps(record), flush=True)
        best = min(records, key=lambda value: value["loss"])
        full = predict(best["theta"], curves, sample["radii"], full_nodes)
        approximate = predict(best["theta"], curves, sample["radii"], nodes)
        error = float(np.max(np.abs(np.log10(full / approximate))))
        attempts.append(
            {
                "starts": records,
                "integration_max_dex": error,
                "nodes": nodes,
                "full_nodes": full_nodes,
            }
        )
        if operational or error <= 0.001:
            break
    settled = (
        all(record["success"] for record in records)
        and max(record["loss"] for record in records) <= 1.01 * best["loss"]
        and error <= 0.001
    )
    return full, attempts, settled


def feature_sample(sample, kind):
    result = dict(sample)
    if kind == "measured":
        masses = sample["measured_epoch_mass"]
        result["core"] = np.column_stack((masses, sample["core"][:, 4:]))
        result["history"] = np.column_stack((masses, sample["history"][:, 4:]))
        result["halo"] = sample["halo"].copy()
        result["halo"][:, 0] = masses[:, 0]
    return result


def frozen_swap():
    """Use only original training-fold parameters; never full-sample stellar fits."""
    check_deadline()
    path = OUTPUT / "frozen_swap.npz"
    if path.exists():
        raise FileExistsError(path)
    sample = load_sample()
    predictions = {
        name: np.full_like(sample["target"], np.nan)
        for name in ("official", "measured", "pre_epoch")
    }
    parity = 0.0
    source_hashes = {}
    for kind in predictions:
        curves, _ = build_inputs(sample, kind)
        for fold in range(5):
            old_path = BASE.OUTPUT / f"discovery_fold{fold}.json"
            old = json.loads(old_path.read_text())
            source_hashes[old_path.name] = file_hash(old_path)
            best = min(old["starts"], key=lambda item: item["loss"])
            training, calibration, held = BASE.split_roles(sample["folds"], fold)
            check_roles(training, calibration, held, len(sample["indices"]))
            value = predict(
                best["theta"], curves, sample["radii"], BASE.PHYSICAL.FULL_NODES, held
            )
            predictions[kind][held] = value
            if kind == "official":
                with np.load(BASE.OUTPUT / f"discovery_fold{fold}.npz") as saved:
                    np.testing.assert_array_equal(saved["evaluation"], held)
                    parity = max(
                        parity,
                        float(np.max(np.abs(np.log10(value / saved["baseline"])))),
                    )
    if parity > 1e-10:
        raise ValueError(f"Official reference changed: {parity} dex")
    np.savez_compressed(path, indices=sample["indices"], **predictions)
    record = {
        "official_parity_max_dex": parity,
        "mean_galaxy_radial_rms_dex_by_epoch": {
            name: np.sqrt(np.mean(np.log10(value / sample["target"]) ** 2, axis=-1))
            .mean(0)
            .tolist()
            for name, value in predictions.items()
        },
        "note": "Frozen original training-fold parameters; history input alone changes. Not a refit or production comparison.",
        "source_parameter_hashes": source_hashes,
        "archive_sha256": file_hash(path),
        "seconds": time.perf_counter() - START_TIME,
    }
    write_record(path.with_suffix(".json"), record)
    print(json.dumps(record, indent=2), flush=True)


def run(stage, kind, rotation):
    check_deadline()
    operational = stage == "gate"
    tag = f"{stage}_{kind}_fold{rotation}"
    destination = OUTPUT / f"{tag}.json"
    if destination.exists() or (OUTPUT / f"{tag}.npz").exists():
        raise FileExistsError(destination)
    if stage != "gate":
        if not json.loads((OUTPUT / "gate_measured_fold0.json").read_text())["passed"]:
            raise ValueError("Operational gate must pass")
    if stage == "discovery":
        for fold in range(5):
            if not json.loads((OUTPUT / f"pilot_{kind}_fold{fold}.json").read_text())[
                "passed"
            ]:
                raise ValueError("All five input-matched pilot folds must pass")
    sample = load_sample(30 if operational else 150 if stage == "pilot" else 0)
    training, calibration, evaluation = BASE.split_roles(sample["folds"], rotation)
    check_roles(training, calibration, evaluation, len(sample["indices"]))
    curves, info = build_inputs(sample, kind)
    baseline, attempts, settled = fit_baseline(
        sample, curves, training, operational, tag
    )
    predictions, choices = {}, {}
    if operational or settled:
        for features in ("original", "measured"):
            check_deadline()
            values, selected = BASE.compare(
                feature_sample(sample, features),
                baseline,
                training,
                calibration,
                evaluation,
            )
            predictions.update(
                {features + "_" + name: value for name, value in values.items()}
            )
            choices[features] = selected
    paths = []
    if operational:
        display_predictions = {
            name: predictions["measured_" + name]
            for name in ("baseline", "hybrid", "direct")
        }
        paths = BASE.direct_figure(
            sample, evaluation, display_predictions, "continuation/" + tag
        )
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
        np.testing.assert_array_equal(saved["baseline_all"], baseline)
    elapsed = time.perf_counter() - START_TIME
    record = {
        "stage": stage,
        "input": kind,
        "rotation": rotation,
        "galaxies": len(sample["indices"]),
        "attempts": attempts,
        "passed": elapsed < 60 if operational else settled,
        "scientific_result": False,
        "seconds": elapsed,
        "roles": [len(training), len(calibration), len(evaluation)],
        "choices": choices,
        "figures": paths,
        "measured_info": info,
        "archive_sha256": file_hash(archive),
        "input_sha256": file_hash(OUTPUT / "inputs.npz"),
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "code_sha256": {path.name: file_hash(path) for path in HERE.glob("*.py")},
    }
    write_record(destination, record)
    print(
        json.dumps(
            {
                key: value
                for key, value in record.items()
                if key not in ("measured_info", "code_sha256", "attempts")
            },
            indent=2,
        ),
        flush=True,
    )
    if not record["passed"]:
        raise RuntimeError("Numerical/runtime gate failed; not a science null")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=("prepare", "frozen", "gate", "pilot", "discovery")
    )
    parser.add_argument(
        "--input", choices=("official", "measured", "pre_epoch"), default="measured"
    )
    parser.add_argument("--rotation", type=int, choices=range(5), default=0)
    arguments = parser.parse_args()
    with threadpool_limits(limits=1):
        if arguments.stage == "prepare":
            prepare_inputs()
        elif arguments.stage == "frozen":
            frozen_swap()
        else:
            run(arguments.stage, arguments.input, arguments.rotation)
