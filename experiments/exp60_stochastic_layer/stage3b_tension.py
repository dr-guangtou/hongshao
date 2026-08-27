"""exp60 Stage 3b-pre — the S1/S3 tension, measured before the sampler is
built.

THE TENSION, noticed at design time. Gate S1 wants the drawn population to
reproduce the measured central-evolution distribution (41.8% declining,
overall median +0.037 dex). Gate S3 wants the mean model untouched (drawn
median CoG within 0.010 dex of the incumbent's). But the incumbent's own
median central change is +0.135 dex — 0.10 dex above the data's — so Stage
1's empirical A pool centres at +0.98, and drawing from it would lower the
drawn population's median central CoG at z=0.4, violating S3. The two gates
can BOTH hold only if the mean model's median central change is already
near the data's — which is what the M2 mean-model candidate (running in
parallel) would do by raising the z=2 centres.

WHAT THIS MEASURES, with the cube (no model calls): for a family of shifted
copies of the Stage-1 A pool (location moved, empirical shape kept), the
drawn population's (a) declining fraction, (b) median central change, and
(c) the shift of the drawn median M*(<4.92 kpc) at z=0.4 and at z=2 relative
to the incumbent — so the exact price of hitting S1 with THIS mean model is
a number, not an argument.

Run: PYTHONPATH=. uv run python -u \\
     experiments/exp60_stochastic_layer/stage3b_tension.py
     (needs outputs/stage3a_cube.npz and stage1_core_anatomy.npz; no data dir)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

I5 = 3
SEED = 5
N_REAL = 8
RULE = "=" * 96


def interp_profiles(cube, a_grid, a_draw, i_dc0):
    """Profiles at (dc = 0, per-galaxy A), linear in A between grid nodes.
    cube is (ndc, nA, n, 5, 24) log10 CoG."""
    n = cube.shape[2]
    j = np.clip(np.searchsorted(a_grid, a_draw) - 1, 0, len(a_grid) - 2)
    w = (a_draw - a_grid[j]) / (a_grid[j + 1] - a_grid[j])
    lo = cube[i_dc0, j, np.arange(n)]
    hi = cube[i_dc0, j + 1, np.arange(n)]
    return lo + w[:, None, None] * (hi - lo)


def main():
    z = np.load(HERE / "outputs" / "stage3a_cube.npz")
    cube, dc_grid, a_grid = z["cube"], z["dc_grid"], z["a_grid"]
    a1 = np.load(HERE / "outputs" / "stage1_core_anatomy.npz")
    pool = a1["a_i"][np.isfinite(a1["a_i"])]
    i_dc0 = int(np.where(dc_grid == 0.0)[0][0])
    n = cube.shape[2]
    base = cube[i_dc0, int(np.where(a_grid == 0.0)[0][0]), np.arange(n)]

    print(f"{RULE}\nexp60 STAGE 3b-pre — what hitting S1 with this mean "
          f"model costs at gate S3\n{RULE}\n")
    print(f"  Stage-1 A pool: {len(pool)} values, median "
          f"{np.median(pool):+.3f} (the un-shifted draw)\n")
    print(f"  {'pool shift':>11} {'median A':>9} {'declining':>10} "
          f"{'med change':>11} {'S3 shift z=0.4':>15} {'S3 shift z=2':>13}")
    rng = np.random.default_rng(SEED)
    for shift in (0.0, -0.5, -1.0, -1.5, -np.median(pool)):
        fr, mc, s04, s20 = [], [], [], []
        for _ in range(N_REAL):
            a_draw = np.clip(rng.choice(pool, size=n) + shift,
                             a_grid[0], a_grid[-1])
            prof = interp_profiles(cube, a_grid, a_draw, i_dc0)
            dc = prof[:, 0, I5] - prof[:, 4, I5]
            fin = np.isfinite(dc)
            fr.append(np.mean(dc[fin] < 0))
            mc.append(np.median(dc[fin]))
            s04.append(np.median((prof[:, 0, I5] - base[:, 0, I5])[fin]))
            s20.append(np.median((prof[:, 4, I5] - base[:, 4, I5])[fin]))
        print(f"  {shift:>+11.3f} {np.median(pool) + shift:>+9.3f} "
              f"{100 * np.mean(fr):>9.1f}% {np.mean(mc):>+11.4f} "
              f"{np.mean(s04):>+15.4f} {np.mean(s20):>+13.4f}")
    print(f"\n  targets: declining 41.8%, median change +0.037; S3 demands "
          f"|median CoG shift| <= 0.010 dex.")
    print(f"  Read: whichever row hits the S1 columns, its S3 columns say "
          f"what that does to the\n  drawn population's median centre "
          f"relative to the incumbent — the tension in numbers.")


if __name__ == "__main__":
    main()
