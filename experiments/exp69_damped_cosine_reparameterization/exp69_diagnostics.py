"""Degeneracy and coordinate-continuity diagnostics for Exp69."""

from __future__ import annotations

import numpy as np


def canonical_solution_index(
    objectives: np.ndarray,
    singular_minimum: np.ndarray,
    descriptors: np.ndarray,
) -> int:
    """Apply the frozen conditioned-branch rule to one multi-start fit."""
    objective_values = np.asarray(objectives, float)
    singular_values = np.asarray(singular_minimum, float)
    descriptor_values = np.asarray(descriptors, float)
    if objective_values.ndim != 1:
        raise ValueError("objectives must be one-dimensional")
    if singular_values.shape != objective_values.shape:
        raise ValueError("singular minima must align with objectives")
    if descriptor_values.shape != (len(objective_values), 6):
        raise ValueError("common descriptors must have shape (solution, 6)")
    near = np.flatnonzero(objective_values <= 1.01 * np.min(objective_values) + 1.0e-14)
    best_strength = np.max(singular_values[near])
    conditioned = near[
        np.isclose(singular_values[near], best_strength, rtol=0.0, atol=1.0e-12)
    ]
    best_objective = np.min(objective_values[conditioned])
    loss_tied = conditioned[
        np.isclose(
            objective_values[conditioned], best_objective, rtol=0.0, atol=1.0e-14
        )
    ]
    if len(loss_tied) == 1:
        return int(loss_tied[0])
    ordered = sorted(
        loss_tied.tolist(),
        key=lambda index: (
            descriptor_values[index, 4],
            descriptor_values[index, 5],
            abs(descriptor_values[index, 3]),
            descriptor_values[index, 3],
        ),
    )
    return int(ordered[0])


def weakest_direction_path(
    parameters: np.ndarray,
    direction: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    n_point: int = 21,
) -> np.ndarray:
    """Trace the available centered path along a native weakest direction."""
    center = np.asarray(parameters, float)
    vector = np.asarray(direction, float)
    lower_values = np.asarray(lower, float)
    upper_values = np.asarray(upper, float)
    if not (center.shape == vector.shape == lower_values.shape == upper_values.shape):
        raise ValueError("path inputs must share one shape")
    if n_point < 3 or n_point % 2 == 0:
        raise ValueError("n_point must be an odd integer of at least three")
    parameter_range = upper_values - lower_values
    scaled = vector * parameter_range
    norm = np.linalg.norm(scaled)
    if norm == 0.0:
        return np.repeat(center[None, :], n_point, axis=0)
    unit = scaled / norm
    positive_limits = []
    negative_limits = []
    for value, component, low, high in zip(
        center, unit, lower_values, upper_values, strict=True
    ):
        if component > 0.0:
            positive_limits.append((high - value) / component)
            negative_limits.append((value - low) / component)
        elif component < 0.0:
            positive_limits.append((value - low) / -component)
            negative_limits.append((high - value) / -component)
    positive = min(positive_limits, default=0.0)
    negative = min(negative_limits, default=0.0)
    distances = np.linspace(-0.5 * negative, 0.5 * positive, n_point)
    path = center[None, :] + distances[:, None] * unit[None, :]
    return np.clip(path, lower_values, upper_values)


def nearest_profile_pairs(truth_log: np.ndarray, epochs: np.ndarray) -> np.ndarray:
    """Return one directed nearest amplitude-pinned profile pair per profile."""
    truth = np.asarray(truth_log, float)
    epoch_values = np.asarray(epochs, int)
    if truth.ndim != 2 or len(epoch_values) != len(truth):
        raise ValueError("truth and epochs do not align")
    pinned = truth - truth[:, -1, None]
    pairs: list[tuple[int, int]] = []
    for epoch in np.unique(epoch_values):
        indices = np.flatnonzero(epoch_values == epoch)
        if len(indices) < 2:
            continue
        values = pinned[indices]
        squared = np.mean((values[:, None, :] - values[None, :, :]) ** 2, axis=2)
        np.fill_diagonal(squared, np.inf)
        nearest = np.argmin(squared, axis=1)
        pairs.extend(
            (int(index), int(indices[neighbor]))
            for index, neighbor in zip(indices, nearest, strict=True)
        )
    return np.asarray(pairs, int).reshape(-1, 2)


def _distribution(values: np.ndarray) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    return float(np.median(values)), float(np.percentile(values, 95.0))


def continuity_metrics(
    normalized_descriptors: np.ndarray,
    truth_log: np.ndarray,
    rows: np.ndarray,
    epochs: np.ndarray,
) -> dict[str, object]:
    """Measure cross-profile and adjacent-epoch coordinate continuity."""
    descriptors = np.asarray(normalized_descriptors, float)
    row_values = np.asarray(rows, int)
    epoch_values = np.asarray(epochs, int)
    if descriptors.ndim != 2 or len(descriptors) != len(row_values):
        raise ValueError("descriptors and rows do not align")
    neighbors = nearest_profile_pairs(truth_log, epoch_values)
    nearest_distance = np.linalg.norm(
        descriptors[neighbors[:, 0]] - descriptors[neighbors[:, 1]], axis=1
    )
    epoch_pairs = []
    for row in np.unique(row_values):
        indices = np.flatnonzero(row_values == row)
        order = indices[np.argsort(epoch_values[indices], kind="stable")]
        for first, second in zip(order[:-1], order[1:], strict=True):
            if epoch_values[second] - epoch_values[first] == 1:
                epoch_pairs.append((int(first), int(second)))
    epoch_pair_array = np.asarray(epoch_pairs, int).reshape(-1, 2)
    epoch_distance = (
        np.linalg.norm(
            descriptors[epoch_pair_array[:, 0]] - descriptors[epoch_pair_array[:, 1]],
            axis=1,
        )
        if len(epoch_pair_array)
        else np.asarray([], float)
    )
    nearest_median, nearest_p95 = _distribution(nearest_distance)
    epoch_median, epoch_p95 = _distribution(epoch_distance)
    return {
        "nearest_pair_count": int(len(neighbors)),
        "adjacent_epoch_pair_count": int(len(epoch_pair_array)),
        "nearest_median": nearest_median,
        "nearest_p95": nearest_p95,
        "epoch_median": epoch_median,
        "epoch_p95": epoch_p95,
        "nearest_pairs": neighbors,
        "nearest_distance": nearest_distance,
        "epoch_pairs": epoch_pair_array,
        "epoch_distance": epoch_distance,
    }
