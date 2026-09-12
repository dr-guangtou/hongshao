"""exp81 measurement 4 — the FUNCTIONAL FORM of the baseline's formation-time
residual: binned medians of log10(model/truth) against the early-mass fraction
log M(2 Gyr) - log M(z_k) and against the recent growth log M(z_k) - log
M(t_k - 1 Gyr), in halo-mass terciles, from measurement 2's saved arrays.
Also the slope [dex per dex] of the residual on each variable at fixed halo
mass (linear, after removing a quadratic in log Mh), with the truth's own
slope, at the cells that matter."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments/exp54_unpinned_amplitude"))
import fit as F                                          # noqa: E402

ANCHOR_Z = (0.4, 0.7, 1.0, 1.5, 2.0)
CELLS = [(0, 4.92), (0, 103.45), (2, 103.45), (3, 4.92), (3, 103.45), (4, 4.92), (4, 32.58), (4, 103.45)]
d = np.load(HERE / "outputs/residual_features.npz")
r, lt = d["r"], d["lt_truth"]


def resid_on(y, x, lmh):
    ok = np.isfinite(y) & np.isfinite(x) & np.isfinite(lmh)
    X = np.column_stack([np.ones(ok.sum()), lmh[ok], lmh[ok] ** 2])
    ry = y[ok] - X @ np.linalg.lstsq(X, y[ok], rcond=None)[0]
    rx = x[ok] - X @ np.linalg.lstsq(X, x[ok], rcond=None)[0]
    return ry, rx


print(f"  slope [dex per dex] of the residual (model/truth) | of the truth's log M*(<R), on each variable at fixed log Mh:")
print(f"  {'cell':<16}{'early2 (M(2 Gyr)/Mh)':>24}{'recent (last Gyr)':>22}{'t50 (log Gyr)':>18}")
for k, R in CELLS:
    i = int(np.argmin(np.abs(F.R_GRID - R)))
    cells = []
    for nm in ("early2", "recent", "t50"):
        x, lmh = d[f"{k}_{nm}"], d[f"{k}_lmh"]
        ry, rx = resid_on(r[:, k, i], x, lmh); ty, tx = resid_on(lt[:, k, i], x, lmh)
        cells.append(f"{np.polyfit(rx, ry, 1)[0]:+.3f} | {np.polyfit(tx, ty, 1)[0]:+.3f}")
    print(f"  z={ANCHOR_Z[k]} R={R:<6.0f}" + "".join(f"{c:>22}" for c in cells))
print(f"\n  binned median residual [dex] vs the EARLY-MASS FRACTION log M(2 Gyr) - log M(z_k), quintiles, per halo-mass tercile (low/mid/high):")
for k, R in CELLS:
    i = int(np.argmin(np.abs(F.R_GRID - R)))
    x, lmh = d[f"{k}_early2"], d[f"{k}_lmh"]
    e = np.quantile(lmh, [0, 1 / 3, 2 / 3, 1]); q = np.quantile(x, np.linspace(0, 1, 6))
    rows = []
    for b in range(3):
        t = (lmh >= e[b]) & (lmh <= e[b + 1] + 1e-9)
        rows.append("/".join(f"{np.median(r[t & (x >= q[j]) & (x <= q[j + 1]), k, i]):+.3f}" if (t & (x >= q[j]) & (x <= q[j + 1])).sum() >= 15 else "  nan " for j in range(5)))
    print(f"  z={ANCHOR_Z[k]} R={R:<6.0f} quintile edges {'/'.join(f'{v:+.2f}' for v in q)}:  " + "   ".join(rows))
print(f"\n  the same vs RECENT GROWTH log M(z_k) - log M(t_k - 1 Gyr):")
for k, R in CELLS:
    i = int(np.argmin(np.abs(F.R_GRID - R)))
    x, lmh = d[f"{k}_recent"], d[f"{k}_lmh"]
    e = np.quantile(lmh, [0, 1 / 3, 2 / 3, 1]); q = np.quantile(x, np.linspace(0, 1, 6))
    rows = []
    for b in range(3):
        t = (lmh >= e[b]) & (lmh <= e[b + 1] + 1e-9)
        rows.append("/".join(f"{np.median(r[t & (x >= q[j]) & (x <= q[j + 1]), k, i]):+.3f}" if (t & (x >= q[j]) & (x <= q[j + 1])).sum() >= 15 else "  nan " for j in range(5)))
    print(f"  z={ANCHOR_Z[k]} R={R:<6.0f} quintile edges {'/'.join(f'{v:+.2f}' for v in q)}:  " + "   ".join(rows))
