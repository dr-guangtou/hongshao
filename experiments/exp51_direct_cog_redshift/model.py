"""Central-censored CoG coordinates for exp51's high-redshift profiles.

The measured mass fraction inside 2 kpc is one explicit coordinate. The three
radii therefore describe the 20%, 50%, and 80% quantiles of the mass *outside*
2 kpc. Unlike total-mass R20/R50/R80, these radii remain resolved when a compact
galaxy already contains more than half of its mass inside the first aperture.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.special import expit, logit

from hongshao.qa import enclosed_radius

OUTER_QUANTILE_FRACTIONS = np.array([0.2, 0.5, 0.8])


def target_names(mode: str) -> list[str]:
    _check_mode(mode)
    return [
        "log_mstar_total",
        "logit_f2",
        "log_r20_outer",
        "log_dlogr_20_50_outer",
        "log_dlogr_50_80_outer",
    ]


def compress_cogs(cogs_log: np.ndarray, radii: np.ndarray, mode: str) -> np.ndarray:
    """Compress CoGs using quantiles of the stellar mass outside 2 kpc."""
    _check_mode(mode)
    cogs_log = np.asarray(cogs_log, float)
    radii = np.asarray(radii, float)
    if cogs_log.ndim != 2 or cogs_log.shape[1] != len(radii):
        raise ValueError("cogs_log must have shape (n_galaxy, n_radius)")
    if not np.all(np.diff(cogs_log, axis=1) >= -1e-10):
        raise ValueError("compress_cogs requires monotone cumulative profiles")
    linear = 10.0**cogs_log
    total = linear[:, -1]
    f2 = np.clip(linear[:, 0] / total, 1e-8, 1.0 - 1e-8)
    levels = f2[:, None] + (1.0 - f2[:, None]) * OUTER_QUANTILE_FRACTIONS
    quantiles = np.array(
        [
            [enclosed_radius(cog, radii, level) for level in row_levels]
            for cog, row_levels in zip(linear, levels)
        ]
    )
    log_radii = np.log10(quantiles)
    gaps = np.diff(log_radii, axis=1)
    if np.any(gaps <= 0.0) or np.any(quantiles[:, 0] <= radii[0]):
        raise ValueError("outer-mass quantile radii are not resolved and ordered")
    return np.column_stack(
        [cogs_log[:, -1], logit(f2), log_radii[:, 0], np.log10(gaps)]
    )


def decode_coordinates(
    parameters: np.ndarray, mode: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return central fractions, outer-mass levels, and ordered radii."""
    _check_mode(mode)
    parameters = np.atleast_2d(np.asarray(parameters, float))
    if parameters.shape[1] != 5:
        raise ValueError(f"quantile3 needs 5 parameters, got {parameters.shape[1]}")
    f2 = np.clip(expit(parameters[:, 1]), 1e-8, 1.0 - 1e-8)
    levels = f2[:, None] + (1.0 - f2[:, None]) * OUTER_QUANTILE_FRACTIONS
    log_radii = np.empty((len(parameters), 3))
    log_radii[:, 0] = parameters[:, 2]
    gaps = 10.0 ** np.clip(parameters[:, 3:], -6.0, 3.0)
    log_radii[:, 1:] = log_radii[:, :1] + np.cumsum(gaps, axis=1)
    return f2, levels, 10.0**log_radii


def reconstruct_cogs(
    parameters: np.ndarray,
    radii: np.ndarray,
    mode: str,
    return_dropped: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Decode the central-censored coordinates to monotone log10 CoGs."""
    parameters = np.atleast_2d(np.asarray(parameters, float))
    radii = np.asarray(radii, float)
    f2, levels, quantiles = decode_coordinates(parameters, mode)
    output = np.empty((len(parameters), len(radii)))
    dropped = np.zeros(len(parameters), dtype=int)
    for index, (row, central_fraction, row_levels, row_radii) in enumerate(
        zip(parameters, f2, levels, quantiles)
    ):
        knot_radii = [radii[0]]
        knot_fractions = [central_fraction]
        for fraction, radius in zip(row_levels, row_radii):
            valid = radius > knot_radii[-1] * (1.0 + 1e-10)
            valid &= radius < radii[-1] * (1.0 - 1e-10)
            if valid:
                knot_radii.append(radius)
                knot_fractions.append(fraction)
            else:
                dropped[index] += 1
        knot_radii.append(radii[-1])
        knot_fractions.append(1.0)
        enclosed = PchipInterpolator(np.log10(knot_radii), knot_fractions)(
            np.log10(radii)
        )
        enclosed = np.maximum.accumulate(np.clip(enclosed, 1e-12, 1.0))
        enclosed[-1] = 1.0
        output[index] = row[0] + np.log10(enclosed)
    if return_dropped:
        return output, dropped
    return output


def _check_mode(mode: str) -> None:
    if mode != "quantile3":
        raise ValueError(f"exp51 supports only 'quantile3', got {mode!r}")


def demo() -> None:
    """Check ordinary and centrally concentrated profiles round-trip safely."""
    radii = np.geomspace(2.0, 150.0, 24)
    total = np.array([10.8, 11.2, 11.5])
    profiles = np.array(
        [
            mass + np.log10(1.0 - np.exp(-((radii / width) ** slope)))
            for mass, width, slope in zip(total, (8.0, 2.0, 0.6), (0.8, 1.2, 0.7))
        ]
    )
    parameters = compress_cogs(profiles, radii, "quantile3")
    reconstructed, dropped = reconstruct_cogs(
        parameters, radii, "quantile3", return_dropped=True
    )
    assert np.isfinite(reconstructed).all()
    assert np.all(np.diff(reconstructed, axis=1) >= 0.0)
    assert np.allclose(reconstructed[:, 0], profiles[:, 0])
    assert np.allclose(reconstructed[:, -1], profiles[:, -1])
    assert np.all(dropped == 0)
    print("exp51 central-censored coordinate demo OK")


if __name__ == "__main__":
    demo()
