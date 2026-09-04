"""Small halo-only maps and positive-shell residual corrections."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.special import softmax


def radial_basis(radii):
    radii = np.asarray(radii, float)
    shell_radius = np.r_[radii[0] / 2, np.sqrt(radii[:-1] * radii[1:])]
    coordinate = np.log(shell_radius)
    coordinate = 2 * (coordinate - coordinate.min()) / np.ptp(coordinate) - 1
    return np.polynomial.legendre.legvander(coordinate, 3)[:, 1:]


def apply_correction(baseline, coefficients, radii):
    """Four predicted numbers; normalization uses only the baseline mass."""
    baseline = np.asarray(baseline, float)
    coefficients = np.asarray(coefficients, float)
    if not np.isfinite(baseline).all() or not np.isfinite(coefficients).all():
        raise ValueError("prediction inputs must be finite")
    if np.any(baseline[..., 0] <= 0) or np.any(np.diff(baseline, axis=-1) < 0):
        raise ValueError("baseline must be positive and nondecreasing")
    if coefficients.shape[-1] != 4:
        raise ValueError("exactly four correction coefficients are required")
    if np.all(coefficients == 0):
        return baseline.copy()
    shells = np.diff(baseline, prepend=np.zeros_like(baseline[..., :1]), axis=-1)
    log_shells = np.full_like(shells, -np.inf)
    np.log(shells, out=log_shells, where=shells > 0)
    tilted = log_shells + coefficients[..., 1:] @ radial_basis(radii).T
    fractions = np.cumsum(softmax(tilted, axis=-1), axis=-1)
    fractions[..., -1] = 1
    return fractions * (baseline[..., -1] * 10 ** coefficients[..., 0])[..., None]


def fit_coefficients(baseline, target, radii):
    """Fit calibration labels, never call on evaluation targets."""
    result = least_squares(
        lambda value: np.log10(apply_correction(baseline, value, radii) / target),
        np.zeros(4),
        max_nfev=100,
        ftol=1e-9,
        xtol=1e-9,
        gtol=1e-9,
    )
    if not result.success:
        raise RuntimeError("calibration coefficient fit did not converge")
    return result.x


def expand_features(features, degree):
    if degree == 1:
        return features
    left, right = np.triu_indices(features.shape[1])
    return np.column_stack([features, features[:, left] * features[:, right]])


@dataclass
class HaloMap:
    impute: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    expanded_mean: np.ndarray
    expanded_scale: np.ndarray
    target_mean: np.ndarray
    target_scale: np.ndarray
    weights: np.ndarray
    degree: int

    def predict(self, features):
        features = np.asarray(features, float)
        clean = np.where(np.isfinite(features), features, self.impute)
        expanded = expand_features(
            (clean - self.feature_mean) / self.feature_scale, self.degree
        )
        standardized = (expanded - self.expanded_mean) / self.expanded_scale
        return (standardized @ self.weights) * self.target_scale + self.target_mean


def fit_map(features, targets, degree, penalty):
    features, targets = np.asarray(features, float), np.asarray(targets, float)
    finite = np.isfinite(features)
    impute = np.array(
        [
            np.median(features[finite[:, column], column])
            if finite[:, column].any()
            else 0
            for column in range(features.shape[1])
        ]
    )
    clean = np.where(finite, features, impute)
    mean, scale = clean.mean(0), np.maximum(clean.std(0), 1e-8)
    expanded = expand_features((clean - mean) / scale, degree)
    expanded_mean = expanded.mean(0)
    expanded_scale = np.maximum(expanded.std(0), 1e-8)
    design = (expanded - expanded_mean) / expanded_scale
    target_mean, target_scale = targets.mean(0), np.maximum(targets.std(0), 1e-8)
    standardized = (targets - target_mean) / target_scale
    weights = np.linalg.solve(
        design.T @ design / len(design) + penalty * np.eye(design.shape[1]),
        design.T @ standardized / len(design),
    )
    return HaloMap(
        impute,
        mean,
        scale,
        expanded_mean,
        expanded_scale,
        target_mean,
        target_scale,
        weights,
        degree,
    )


def make_folds(final_mass):
    order = np.argsort(final_mass, kind="stable")
    rng = np.random.default_rng(75)
    folds = np.empty(len(order), int)
    for group in np.array_split(order, max(1, len(order) // 5)):
        folds[group] = rng.permutation(len(group)) % 5
    return folds


def split_roles(folds, rotation):
    evaluation = np.flatnonzero(folds == rotation)
    calibration = np.flatnonzero(folds == (rotation + 1) % 5)
    training = np.flatnonzero(~np.isin(folds, [rotation, (rotation + 1) % 5]))
    return training, calibration, evaluation
