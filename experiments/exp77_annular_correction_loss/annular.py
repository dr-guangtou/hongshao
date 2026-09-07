"""Fixed four-coordinate corrections with an explicit annular fitting term."""

# ruff: noqa: E402
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REFERENCE = ROOT / "experiments/exp75_halo_residual_correction"
sys.path.insert(0, str(REFERENCE))
specification = importlib.util.spec_from_file_location(
    "exp77_reference_report", REFERENCE / "continuation_report.py"
)
REF = importlib.util.module_from_spec(specification)
sys.modules[specification.name] = REF
specification.loader.exec_module(REF)

import numpy as np
from scipy.optimize import least_squares
from correction import apply_correction, fit_coefficients, fit_map


def annular_matrix(radii):
    return REF.annuli(np.eye(len(radii))[:, None, :], radii)[:, 0]


def annular_transform(masses, true_total):
    return np.arcsinh(masses / (0.001 * true_total[..., None])) / np.log(10)


def fit_labels(baseline, truth, radii, weight):
    if weight == 0:
        return fit_coefficients(baseline, truth, radii), {"retry": False}
    matrix = annular_matrix(radii)
    target_annuli = annular_transform(truth @ matrix, truth[-1])

    def residual(coefficients):
        prediction = apply_correction(baseline, coefficients, radii)
        cog_error = np.log10(prediction / truth) / np.sqrt(len(radii))
        annular_error = (
            annular_transform(prediction @ matrix, truth[-1]) - target_annuli
        )
        return np.r_[cog_error, np.sqrt(weight / 6) * annular_error]

    for maximum in (100, 300):
        fitted = least_squares(
            residual,
            np.zeros(4),
            max_nfev=maximum,
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-9,
        )
        if fitted.success:
            return fitted.x, {"retry": maximum > 100, "nfev": fitted.nfev}
    raise RuntimeError("Four-coordinate annular fit did not converge")


def scores(prediction, truth, radii):
    cog = np.sqrt(np.mean(np.log10(prediction / truth) ** 2, axis=-1)).mean()
    outer = ((prediction - truth) @ annular_matrix(radii))[..., -2:]
    outer = np.sqrt(np.mean((outer / truth[..., -1, None]) ** 2))
    return {"cog_mean_radial_rms_dex": float(cog), "outer_relative_rms": float(outer)}


def train_predict(features, labels, baseline, radii, training, held, degree, penalty):
    fitted = fit_map(
        features[training], labels[training].reshape(len(training), -1), degree, penalty
    )
    coefficients = fitted.predict(features[held]).reshape(len(held), 5, 4)
    return apply_correction(baseline[held], coefficients, radii)


def choose_map(features, labels, baseline, truth, radii, reference_choice):
    """Only calibration rows and their internal held-out labels enter selection."""
    candidates = []
    configurations = [(0, reference_choice["degree"], reference_choice["penalty"])]
    configurations += [
        (weight, degree, penalty)
        for weight in (0.25, 1)
        for degree in (1, 2)
        for penalty in (0.01, 0.1, 1, 10, 100)
    ]
    for weight, degree, penalty in configurations:
        predictions = np.empty_like(truth)
        for fold in (0, 1):
            training = np.flatnonzero(np.arange(len(truth)) % 2 != fold)
            held = np.flatnonzero(np.arange(len(truth)) % 2 == fold)
            predictions[held] = train_predict(
                features,
                labels[weight],
                baseline,
                radii,
                training,
                held,
                degree,
                penalty,
            )
        candidates.append(
            {
                "weight": weight,
                "degree": degree,
                "penalty": penalty,
                **scores(predictions, truth, radii),
            }
        )
    reference_score = candidates[0]["cog_mean_radial_rms_dex"]
    eligible = [
        candidate
        for candidate in candidates
        if candidate["cog_mean_radial_rms_dex"] <= 1.02 * reference_score
    ]
    best = min(candidate["outer_relative_rms"] for candidate in eligible)
    chosen = min(
        (
            candidate
            for candidate in eligible
            if candidate["outer_relative_rms"] <= 1.01 * best
        ),
        key=lambda candidate: (
            candidate["weight"],
            candidate["degree"],
            -candidate["penalty"],
        ),
    )
    return chosen, candidates


def predict_from_calibration(
    sample, baseline, calibration, evaluation, choice, expected_reference=None
):
    radii, truth = sample["radii"], sample["target"]
    labels, retries = {}, 0
    for weight in (0, 0.25, 1):
        labels[weight] = np.empty((len(calibration), 5, 4))
        for row_number, row in enumerate(calibration):
            REF.CONT.check_deadline()
            for epoch in range(5):
                values, record = fit_labels(
                    baseline[row, epoch], truth[row, epoch], radii, weight
                )
                labels[weight][row_number, epoch] = values
                retries += record["retry"]
        if weight == 0 and expected_reference is not None:
            reference_map = fit_map(
                sample[choice["feature"]][calibration],
                labels[0].reshape(len(calibration), -1),
                choice["degree"],
                choice["penalty"],
            )
            parameters = reference_map.predict(
                sample[choice["feature"]][evaluation]
            ).reshape(len(evaluation), 5, 4)
            recovered = apply_correction(baseline[evaluation], parameters, radii)
            if np.max(np.abs(np.log10(recovered / expected_reference))) > 1e-8:
                raise ValueError(
                    "Weight-zero reference changed before annular-loss fitting"
                )
    features = sample[choice["feature"]]
    selected, candidates = choose_map(
        features[calibration],
        labels,
        baseline[calibration],
        truth[calibration],
        radii,
        choice,
    )
    result = {}
    for name, configuration in (
        ("reference", {"weight": 0, **choice}),
        ("selected", selected),
    ):
        fitted = fit_map(
            features[calibration],
            labels[configuration["weight"]].reshape(len(calibration), -1),
            configuration["degree"],
            configuration["penalty"],
        )
        coefficients = fitted.predict(features[evaluation]).reshape(
            len(evaluation), 5, 4
        )
        result[name] = apply_correction(baseline[evaluation], coefficients, radii)
    return result, {
        "selected": selected,
        "inner_candidates": candidates,
        "coefficient_fit_retries": retries,
        "feature": choice["feature"],
    }
