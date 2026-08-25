"""exp57 — is G2's headline robust to the shell binning it is computed on?

**THE ANSWER (2026-08-26): yes, and by a wide margin.**

| model | as reported | coarse (5) | fine (25) | very fine (60) |
|---|---|---|---|---|
| `X1`, beyond 50 kpc | 85.0% | 85.0% | 80.4% | 80.5% |
| `X1`, beyond 148 kpc | 52.1% | 52.1% | 49.2% | 49.3% |
| `X3` small core, beyond 50 kpc | **2.2%** | 4.9% | 1.8% | 1.7% |
| `X3` small core, beyond 148 kpc | **0.2%** | 0.4% | 0.1% | 0.1% |

`X1` fails the 25% limit on every grid and `X3` passes it on every grid, with a
factor of roughly 20 to 40 between them throughout. The residual grid
dependence is real but small and in the expected direction: a coarse grid has
wide shells, so mass that moves *within* one is not counted as having crossed
anything.

Original motivation follows.

G2 -- the fraction of displaced mass delivered beyond 50 and 148 kpc -- is the
number this whole experiment turns on, and it is computed by summing signed
changes over a chosen set of shells. A coarse grid could in principle inflate or
deflate it: mass that moves from 45 to 55 kpc crosses a boundary, mass that
moves from 52 to 58 does not. So recompute on grids from very coarse to very
fine and see whether the answer moves.
"""
import sys
from pathlib import Path
import numpy as np
HERE = Path("experiments/exp57_expansion_term").resolve()
ROOT = HERE.parents[1]
for p in (ROOT, ROOT/"experiments/exp38_deposit_rethink",
          ROOT/"experiments/exp54_unpinned_amplitude", HERE):
    sys.path.insert(0, str(p))
import expand as X, gates as G, stage0_cost as S0

recs, data, mask, lmh, base, theta, slow = S0.build(False)
D = HERE/"outputs"/"stage3_fits"
GRIDS = {
    "as reported (9 shells)": G.REACH_EDGES,
    "coarse (5 shells)": (0.0, 10.0, 50.0, 148.0, 500.0, 1e7),
    "fine (25 log shells)": tuple(np.concatenate(
        [[0.0], np.geomspace(1.0, 1e4, 24), [1e7]])),
    "very fine (60 log shells)": tuple(np.concatenate(
        [[0.0], np.geomspace(0.5, 1e4, 59), [1e7]])),
}
print(f"  {'model':<22}{'grid':<28}{'>50 kpc':>10}{'>148 kpc':>10}")
for lab in ("gompertz_log-E2-S2+X1", "gompertz_log-E2-S2+X3#smallcore"):
    f = D/f"{lab}.npz"
    if not f.exists():
        continue
    z = np.load(f, allow_pickle=True)
    sp = X.XSpec(base, str(z["law"]))
    pr = X.ExpandingProblem(sp, recs, data, mask, epochs=(0,1,2,3,4))
    th = np.asarray(z["theta"], float)
    for name, edges in GRIDS.items():
        # the 50 and 148 kpc boundaries must BE grid edges or the question is
        # ill-posed; insert them if the grid does not already contain them
        e = np.unique(np.concatenate([np.array(edges), [50.0, 148.0]]))
        r = G.g2_reach(pr, th, tuple(e))
        w = max(r, key=lambda q: q["f_beyond_50"])
        print(f"  {lab.replace('gompertz_log-E2-S2+',''):<22}{name:<28}"
              f"{100*w['f_beyond_50']:>9.1f}%{100*w['f_beyond_148']:>9.1f}%")
    print()
