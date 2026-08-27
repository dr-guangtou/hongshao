"""exp60 Stage 3a — the prediction cube: every galaxy's profile under every
(size-axis deviation, core-expansion strength) pair the sampler can draw.

WHY A CUBE. The layer draws two per-galaxy parameters — a deviation `dc` of
the deposit shape `c` (Stage 2's individuality axis) and a core-expansion
strength `A` (Stage 1's axis, `X3` remap at the Stage-1-selected 20 kpc
core). The evaluators apply SHARED parameters, but galaxies are independent
in the forward model, so evaluating the shared model on a (dc x A) grid
gives every galaxy's profile at every grid pair at once; a drawn galaxy's
profile is then a bilinear interpolation (in log10 of the CoG) between its
four neighbouring grid nodes. One cube serves every realization, every
calibration iteration, and every gate — no evaluator is modified and no
per-draw model call is ever made.

The interpolation is VALIDATED here, not assumed: profiles at off-grid
midpoints are computed directly and compared against the interpolation; the
worst error is printed and stored. The amplitude component needs no cube —
it is a per-galaxy-epoch scalar on the CoG, applied at draw time.

Run: HONGSHAO_DATA_DIR=... OMP_NUM_THREADS=1 PYTHONPATH=. uv run python -u \\
     experiments/exp60_stochastic_layer/stage3a_cube.py [--smoke]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink",
          ROOT / "experiments/exp54_unpinned_amplitude",
          ROOT / "experiments/exp57_expansion_term"):
    sys.path.insert(0, str(p))

import expand as X                                       # noqa: E402
import stage0_cost as S0                                 # noqa: E402

#: the size-axis grid, spanning Stage 2's measured deviation distribution
#: (5th-95th percentile -0.60 to +1.06) with room in the tails
DC_GRID = np.array([-1.0, -0.6, -0.35, -0.2, -0.1, 0.0, 0.1, 0.2,
                    0.35, 0.6, 1.0, 1.6, 2.5])
#: the core-expansion grid, Stage 1's grid at Stage 1's chosen 20 kpc core
A_GRID = np.array([-0.9, -0.6, -0.3, -0.15, 0.0, 0.15, 0.3, 0.6, 1.0,
                   1.5, 2.0, 3.0, 4.5, 6.5, 9.0, 13.0, 20.0])
LOG_RC = 1.30
OUT = HERE / "outputs" / "stage3a_cube.npz"
RULE = "=" * 96


def main(smoke=False):
    print(f"{RULE}\nexp60 STAGE 3a — the (dc x A) prediction cube\n{RULE}\n")
    recs, data, mask, lmh, base, theta, slow = S0.build(smoke)
    n = len(recs)
    sp = X.XSpec(base, "X3")
    pr = X.ExpandingProblem(sp, recs, data, mask, epochs=(0, 1, 2, 3, 4))
    jc = base.theta_names.index("c")
    lo, hi = np.array(base.bounds()).T

    def predict(dc, a):
        th = sp.nest(theta).copy()
        th[jc] = np.clip(theta[jc] + dc, lo[jc], hi[jc])
        th[-2], th[-1] = a, LOG_RC
        return pr.predict(th)

    t0 = time.time()
    cube = np.full((len(DC_GRID), len(A_GRID), n, 5, 24), np.nan,
                   dtype=np.float32)
    for i, dc in enumerate(DC_GRID):
        for j, a in enumerate(A_GRID):
            with np.errstate(invalid="ignore", divide="ignore"):
                cube[i, j] = np.log10(np.clip(predict(dc, a), 1e-30, None)
                                      ).astype(np.float32)
        print(f"  dc = {dc:+.2f} row done  [{time.time() - t0:.0f} s]",
              flush=True)
    print(f"\n  cube {cube.shape} "
          f"({cube.nbytes / 1e6:.0f} MB float32) in "
          f"{(time.time() - t0) / 60:.1f} min")

    # ---- validate the bilinear interpolation at off-grid midpoints ------- #
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(4 if smoke else 8):
        i = rng.integers(0, len(DC_GRID) - 1)
        j = rng.integers(0, len(A_GRID) - 1)
        dc = 0.5 * (DC_GRID[i] + DC_GRID[i + 1])
        a = 0.5 * (A_GRID[j] + A_GRID[j + 1])
        with np.errstate(invalid="ignore", divide="ignore"):
            direct = np.log10(np.clip(predict(dc, a), 1e-30, None))
        interp = 0.25 * (cube[i, j] + cube[i + 1, j]
                         + cube[i, j + 1] + cube[i + 1, j + 1])
        d = np.abs(direct - interp)
        e = float(np.nanpercentile(d[np.isfinite(d) & (direct > 6)], 99.9))
        worst = max(worst, e)
        print(f"  midpoint (dc {dc:+.3f}, A {a:+.2f}): 99.9th percentile "
              f"|error| {e:.5f} dex")
    print(f"\n  worst midpoint interpolation error {worst:.5f} dex "
          f"(midpoints are the grid's worst case; drawn values interpolate "
          f"from nearer nodes)")

    if not smoke:
        np.savez(OUT, cube=cube, dc_grid=DC_GRID, a_grid=A_GRID,
                 log_rc=LOG_RC, worst_interp=worst)
        print(f"  saved {OUT.relative_to(ROOT)} "
              f"({OUT.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv[1:])
