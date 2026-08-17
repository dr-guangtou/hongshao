"""Stable enclosed-mass coordinates and a monotone CoG decoder for exp50."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.special import expit, logit

from hongshao.qa import enclosed_radius


QUANTILE_FRACTIONS = {
    "quantile3": np.array([0.2, 0.5, 0.8]),
    "quantile4": np.array([0.2, 0.5, 0.8, 0.9]),
}


def target_names(mode: str) -> list[str]:
    """Names of the transformed coordinates used by the Gaussian emulator."""
    fractions = _fractions(mode)
    names = ["log_mstar_total", "logit_f2", "log_r20"]
    names.extend(
        f"log_dlogr_{int(100 * a)}_{int(100 * b)}"
        for a, b in zip(fractions[:-1], fractions[1:])
    )
    return names


def compress_cogs(cogs_log: np.ndarray, radii: np.ndarray, mode: str) -> np.ndarray:
    """Convert log10 CoGs to stable, ordered enclosed-mass coordinates.

    The first two coordinates are the log total mass at the outer grid edge
    and logit[M(<2 kpc)/M_total]. The first radius is log10(R20/kpc); later
    radii are positive intervals in log10 radius, stored in log10 form. This
    makes R20 < R50 < R80 (< R90) exact for every prediction and draw.
    """
    cogs_log = np.asarray(cogs_log, float)
    radii = np.asarray(radii, float)
    if cogs_log.ndim != 2 or cogs_log.shape[1] != len(radii):
        raise ValueError("cogs_log must have shape (n_galaxy, n_radius)")
    if not np.all(np.diff(cogs_log, axis=1) >= -1e-10):
        raise ValueError("compress_cogs requires monotone cumulative profiles")

    fractions = _fractions(mode)
    linear = 10.0**cogs_log
    total = linear[:, -1]
    f2 = np.clip(linear[:, 0] / total, 1e-8, 1.0 - 1e-8)
    quantiles = np.array(
        [
            [enclosed_radius(cog, radii, fraction) for fraction in fractions]
            for cog in linear
        ]
    )
    log_r = np.log10(quantiles)
    gaps = np.diff(log_r, axis=1)
    if np.any(gaps <= 0.0):
        raise ValueError("enclosed-mass radii are not strictly ordered")
    return np.column_stack([cogs_log[:, -1], logit(f2), log_r[:, 0], np.log10(gaps)])


def decode_coordinates(
    parameters: np.ndarray, mode: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return central fractions and ordered quantile radii from coordinates."""
    parameters = np.atleast_2d(np.asarray(parameters, float))
    fractions = _fractions(mode)
    expected = 3 + len(fractions) - 1
    if parameters.shape[1] != expected:
        raise ValueError(
            f"{mode} needs {expected} parameters, got {parameters.shape[1]}"
        )
    f2 = np.clip(expit(parameters[:, 1]), 1e-8, 1.0 - 1e-8)
    log_r = np.empty((len(parameters), len(fractions)))
    log_r[:, 0] = parameters[:, 2]
    if len(fractions) > 1:
        gaps = 10.0 ** np.clip(parameters[:, 3:], -6.0, 3.0)
        log_r[:, 1:] = log_r[:, :1] + np.cumsum(gaps, axis=1)
    return f2, 10.0**log_r


def reconstruct_cogs(
    parameters: np.ndarray,
    radii: np.ndarray,
    mode: str,
    return_dropped: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Decode coordinates to monotone log10 CoGs on ``radii``.

    The decoder is pinned to the predicted central fraction at the first grid
    radius and unit enclosed fraction at the outer edge. A quantile knot is
    omitted only when a predicted coordinate is inconsistent with those fixed
    aperture anchors (for example, predicted R20 < 2 kpc but f2 < 0.2). The
    omission count is returned as an identifiability/consistency diagnostic.
    """
    parameters = np.atleast_2d(np.asarray(parameters, float))
    radii = np.asarray(radii, float)
    fractions = _fractions(mode)
    f2, quantiles = decode_coordinates(parameters, mode)
    output = np.empty((len(parameters), len(radii)))
    dropped = np.zeros(len(parameters), dtype=int)

    for index, (row, central_fraction, quantile_radii) in enumerate(
        zip(parameters, f2, quantiles)
    ):
        knot_r = [radii[0]]
        knot_f = [central_fraction]
        for fraction, radius in zip(fractions, quantile_radii):
            valid = radius > knot_r[-1] * (1.0 + 1e-10)
            valid &= radius < radii[-1] * (1.0 - 1e-10)
            valid &= fraction > knot_f[-1] + 1e-10
            if valid:
                knot_r.append(radius)
                knot_f.append(fraction)
            else:
                dropped[index] += 1
        knot_r.append(radii[-1])
        knot_f.append(1.0)
        enclosed = PchipInterpolator(np.log10(knot_r), knot_f)(np.log10(radii))
        enclosed = np.maximum.accumulate(np.clip(enclosed, 1e-12, 1.0))
        enclosed[-1] = 1.0
        output[index] = row[0] + np.log10(enclosed)

    if return_dropped:
        return output, dropped
    return output


def empirical_crps(truth: np.ndarray, draws: np.ndarray) -> np.ndarray:
    """Empirical CRPS for draws shaped (n_draw, ..., n_target)."""
    truth = np.asarray(truth, float)
    draws = np.asarray(draws, float)
    if draws.shape[1:] != truth.shape:
        raise ValueError("draws must have shape (n_draw, *truth.shape)")
    ordered = np.sort(draws, axis=0)
    n_draw = len(ordered)
    weights = (2 * np.arange(n_draw) - n_draw + 1).reshape(
        (n_draw,) + (1,) * truth.ndim
    )
    pair_term = np.sum(weights * ordered, axis=0) / n_draw**2
    return np.mean(np.abs(draws - truth[None]), axis=0) - pair_term


def _fractions(mode: str) -> np.ndarray:
    try:
        return QUANTILE_FRACTIONS[mode]
    except KeyError as error:
        raise ValueError(f"unknown coordinate mode {mode!r}") from error


def demo() -> None:
    """Fast mechanical checks for ordering, round-trip knots, and CRPS."""
    radii = np.geomspace(2.0, 150.0, 24)
    widths = np.array([5.0, 12.0, 30.0])
    total = np.array([10.8, 11.2, 11.7])
    cogs = np.array(
        [
            mass + np.log10(1.0 - np.exp(-((radii / width) ** 0.8)))
            for mass, width in zip(total, widths)
        ]
    )
    for mode in QUANTILE_FRACTIONS:
        parameters = compress_cogs(cogs, radii, mode)
        reconstructed, dropped = reconstruct_cogs(parameters, radii, mode, True)
        assert np.isfinite(reconstructed).all()
        assert np.all(np.diff(reconstructed, axis=1) >= 0.0)
        assert np.allclose(reconstructed[:, 0], cogs[:, 0])
        assert np.allclose(reconstructed[:, -1], cogs[:, -1])
        assert np.all(dropped <= 1), dropped
        _, radii_decoded = decode_coordinates(parameters, mode)
        for column, fraction in enumerate(_fractions(mode)):
            recovered = np.array(
                [enclosed_radius(10.0**row, radii, fraction) for row in reconstructed]
            )
            retained = radii_decoded[:, column] > radii[0]
            retained &= fraction > expit(parameters[:, 1])
            assert np.allclose(
                recovered[retained], radii_decoded[retained, column], rtol=5e-2
            )

    rng = np.random.default_rng(0)
    truth = rng.normal(size=(100, 3))
    draws = truth[None] + rng.normal(size=(200, 100, 3))
    score = empirical_crps(truth, draws)
    assert score.shape == truth.shape and np.all(score > 0.0)
    print("exp50 coordinate-model demo OK")


if __name__ == "__main__":
    demo()
