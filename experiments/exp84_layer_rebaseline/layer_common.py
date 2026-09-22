"""exp84 — the quantities every stage shares: the per-galaxy PER-EPOCH loss
(exp60 Stage 2's structure without its epoch mean), the size residual at
fixed stellar mass (the tier 2d width's own residual) and its cross-epoch
persistence, and the amplitude residual at 100 kpc."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink",
          ROOT / "experiments/exp54_unpinned_amplitude", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import fit as F                                          # noqa: E402
from hongshao import qa                                  # noqa: E402

I5 = 3                                                   # R_GRID[3] = 4.92 kpc (exp60's central aperture)
SIZE_KEYS = ("R20", "R50", "R80")
SIZE_FRACTIONS = {"R20": 0.2, "R50": 0.5, "R80": 0.8}
EPOCH_LABELS = ("z=0.4", "z=0.7", "z=1.0", "z=1.5", "z=2.0")


def gal_epoch_losses(pred, data, use):
    """(n, 5) per-galaxy PER-EPOCH loss with the population objective's
    structure — (dA / SIGMA_A)^2 at 103 kpc plus the mean squared relative
    error of the 103 kpc-normalised profile over L_F_REF^2 — NaN where the
    epoch is not admitted or the prediction is broken. exp60 Stage 2 averaged
    this over epochs; the layer's per-epoch deviations need it per epoch."""
    finite = np.isfinite(pred).all(axis=2) & (pred[:, :, F.I100] > 0)
    ok = np.asarray(use, bool) & finite & (data[:, :, F.I100] > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        dA = ((np.log10(pred[:, :, F.I100]) - np.log10(data[:, :, F.I100])) / F.SIGMA_A[None, :]) ** 2
        Fm = pred / pred[:, :, F.I100][:, :, None]
        Fd = data / data[:, :, F.I100][:, :, None]
        dF = np.mean(((Fm - Fd) / Fd) ** 2, axis=2) / F.L_F_REF ** 2
    return np.where(ok, dA + dF, np.nan)


def log_sizes(cogs, R, keys=SIZE_KEYS):
    """{key: (n, 5) log10 R_frac}, NaN where the CoG is unusable (qa's rule)."""
    return {k: np.log10(qa._safe_rhalf(cogs, R, SIZE_FRACTIONS[k])) for k in keys}


def size_residuals(log_r, log_m):
    """(n, 5) residual of log R about the OLS line on log M per epoch — the
    quantity whose scatter is the tier 2d width; NaN rows stay NaN."""
    out = np.full_like(log_r, np.nan)
    for j in range(log_r.shape[1]):
        g = np.isfinite(log_r[:, j]) & np.isfinite(log_m[:, j])
        if g.sum() < 3:
            continue
        b = np.polyfit(log_m[g, j], log_r[g, j], 1)
        out[g, j] = log_r[g, j] - np.polyval(b, log_m[g, j])
    return out


def persistence(resid):
    """(5, 5) Spearman correlation of a per-epoch residual across epoch pairs
    (pairwise-complete): how much a galaxy's size rank at fixed mass at one
    epoch persists to another."""
    nz = resid.shape[1]
    out = np.full((nz, nz), np.nan)
    for a in range(nz):
        for b in range(nz):
            g = np.isfinite(resid[:, a]) & np.isfinite(resid[:, b])
            out[a, b] = spearmanr(resid[g, a], resid[g, b]).statistic if g.sum() > 10 else np.nan
    return out


def nearest_epoch(mat):
    return float(np.nanmean(np.diag(mat, 1)))


def amplitude_residual(pred, data, use):
    """(n, 5) log10(model / truth) at 103 kpc on the admitted galaxy-epochs."""
    finite = np.isfinite(pred[:, :, F.I100]) & (pred[:, :, F.I100] > 0) & (data[:, :, F.I100] > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        a = np.log10(pred[:, :, F.I100] / data[:, :, F.I100])
    return np.where(np.asarray(use, bool) & finite, a, np.nan)


def half_width(x, axis=0):
    """Half the 16-84 range (exp60's amplitude width)."""
    return (np.nanpercentile(x, 84, axis=axis) - np.nanpercentile(x, 16, axis=axis)) / 2


def print_matrix(mat, label, indent="    "):
    print(f"{indent}{label}")
    print(f"{indent}{'':>8}" + "".join(f"{e:>8}" for e in EPOCH_LABELS))
    for j, e in enumerate(EPOCH_LABELS):
        print(f"{indent}{e:>8}" + "".join(f"{mat[j, i]:>+8.3f}" for i in range(mat.shape[1])))
