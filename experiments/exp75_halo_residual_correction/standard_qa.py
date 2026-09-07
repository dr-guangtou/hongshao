# %%
"""Standard discovery QA from frozen held-out predictions; no refitting."""

# ruff: noqa: E402 -- load the local driver before legacy short-name imports.
import json
import time
from pathlib import Path

from prepare import file_hash, load_module

HERE = Path(__file__).resolve().parent
REPORT = load_module("exp75_discovery_report", HERE / "report.py")

import matplotlib.pyplot as plt
import numpy as np
from threadpoolctl import threadpool_limits

from correction import apply_correction, radial_basis
from hongshao import qa


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def recover_applied_coordinates(baseline, corrected, radii):
    """Invert the prediction transform for auditing, without measured targets."""
    basis = np.column_stack([np.ones(len(radii)), radial_basis(radii)])
    coefficients = np.empty(baseline.shape[:-1] + (4,))
    for index in np.ndindex(baseline.shape[:-1]):
        original = np.diff(np.r_[0, baseline[index]])
        changed = np.diff(np.r_[0, corrected[index]])
        valid = (original > baseline[index][-1] * 1e-10) & (
            changed > corrected[index][-1] * 1e-10
        )
        if np.linalg.matrix_rank(basis[valid]) != 4:
            raise ValueError("insufficient nonzero shells to recover correction")
        fitted = np.linalg.lstsq(
            basis[valid], np.log(changed[valid] / original[valid]), rcond=None
        )[0]
        coefficients[index][0] = np.log10(corrected[index][-1] / baseline[index][-1])
        coefficients[index][1:] = fitted[1:]
    reconstructed = apply_correction(baseline, coefficients, radii)
    maximum_error = float(np.max(np.abs(np.log10(reconstructed / corrected))))
    if maximum_error > 1e-6:
        raise ValueError("coordinate inversion did not reproduce stored prediction")
    return {
        "coordinate_names": [
            "log10_total_change",
            "tilt_linear",
            "tilt_quadratic",
            "tilt_cubic",
        ],
        "minimum_by_epoch": coefficients.min(0),
        "maximum_by_epoch": coefficients.max(0),
        "maximum_reconstruction_error_dex": maximum_error,
    }


def run():
    started = time.perf_counter()
    destination = HERE / "outputs/discovery_standard_qa.json"
    if destination.exists():
        raise FileExistsError("keep the recorded discovery QA immutable")
    sample, predictions, _ = REPORT.assemble("discovery")
    plt.rcParams["text.usetex"] = False
    result = {"n_galaxy": len(sample["indices"]), "models": {}}
    for name in ("baseline", "direct", "hybrid"):
        model_started = time.perf_counter()
        measured = qa.evaluate(
            predictions[name],
            sample["target"],
            sample["radii"],
            REPORT.REDSHIFTS,
            name=f"exp75_{name}",
            figdir=HERE / "figures/standard_qa" if name == "hybrid" else None,
            figures=name == "hybrid",
            verbose=False,
            bin_by=sample["halo"][:, 0],
            bin_label="log halo peak mass (Msun)",
            halo_mass_epochs=sample["epoch_mass"],
        )
        masses = {}
        for key in measured["keys"]:
            truth, model = measured["truth"][key], measured["model"][key]
            valid = (truth > 0) & (model > 0)
            residual = np.full_like(truth, np.nan)
            residual[valid] = np.log10(model[valid] / truth[valid])
            masses[key] = {
                "median_log_bias_dex": np.nanmedian(residual, axis=0),
                "rms_dex": np.sqrt(np.nanmean(residual**2, axis=0)),
                "n_valid": valid.sum(0),
            }
        result["models"][name] = {
            key: measured[key]
            for key in (
                "planes",
                "growth",
                "sizes",
                "sizes_mh",
                "size_gate_ms",
                "size_gate_mh",
                "cdfs",
            )
        }
        result["models"][name]["masses"] = masses
        result["models"][name]["seconds"] = time.perf_counter() - model_started
        print(
            f"Standard QA {name}: {result['models'][name]['seconds']:.3f} s", flush=True
        )
    result["coordinate_audit"] = recover_applied_coordinates(
        predictions["baseline"], predictions["hybrid"], sample["radii"]
    )
    result["figures"] = [
        str(path) for path in sorted((HERE / "figures/standard_qa").glob("*"))
    ]
    result["source_hashes"] = {
        str(path.relative_to(HERE)): file_hash(path)
        for path in [HERE / "standard_qa.py", HERE / "report.py"]
        + sorted((HERE / "outputs").glob("discovery_fold*.npz"))
    }
    result["seconds"] = time.perf_counter() - started
    destination.write_text(
        json.dumps(json_ready(result), indent=2, allow_nan=False) + "\n"
    )
    print(destination, flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run()
