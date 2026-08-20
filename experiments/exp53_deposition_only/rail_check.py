"""exp53 — is any cell's answer set by a BOX rather than by the data?

exp53's whole complaint about exp48 is that every profile family was fitted
against a wall, so shipping a shootout with railed cells would repeat the
mistake. This refits every cell that converged with a parameter at a bound,
with that bound widened, and reports whether the extra freedom buys anything.

Two rails appeared in the main run, and they mean different things.

**`mu` at 5.0, on the DEPOSITION-ONLY cells** (`moffat-q0`, `sersic-q0`) --
the cells the experiment is actually about, so this one is load-bearing for
the result. Design constraint 2 requires the efficiency window to be FREE; a
railed window would mean the model was still being told when it may deposit.
There is a subtlety: `mu = 5` puts the lognormal's peak at `1 + z = e^5 = 148`,
far outside the sample's `ln(1+z) <= ~2.5`, where the window is not a peak at
all but a RAMP -- and only the ramp is identified, so many `(mu, sig)` pairs
give the same weights and a rail along a flat ridge would be cosmetic. Probed
directly (holding the ramp `ln[w(z=11)/w(z=0)]` fixed and sliding `mu`), it is
NOT a flat ridge: the loss has a real interior minimum near `mu ~ 5`, rising
to 0.16175 at `mu = 4` and 0.16253 at `mu = 8`, with the bound essentially on
the optimum. But that probe is a SLICE, and exp49 measured that slices and
profiles disagree in sign in this model -- hence this script.

**`log_R50` at 3.5, on `moffat-q1.2` and `expo-qfree`** -- a weak-tailed
deposit trying to reach the outskirts by inflating its scale, which is exactly
the exp36 Gaussian rail reproduced in a new family. Note the 300 kpc physical
ceiling absorbs part of any widening (a deposit already at the cap does not
move), so the honest question is not "does the rail move" but "does the LOSS
move".

Run: `HONGSHAO_DATA_DIR=... PYTHONPATH=. uv run python \
experiments/exp53_deposition_only/rail_check.py [--shard i/N] [--dev]`
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments/exp38_deposit_rethink"))

import fit as F                                      # noqa: E402
import kernel as K                                   # noqa: E402
import stage2_multiepoch as s2                       # noqa: E402
from hongshao.fitting import minimize_loss           # noqa: E402

#: how far to widen a hit bound, per parameter. Generous but not absurd: the
#: point is to see whether the fit ESCAPES, not to explore infinity.
WIDEN = {"mu": 12.0, "sig": 12.0, "g": 20.0, "log_R50": 5.0, "q": 6.0,
         "gam": 12.0, "n": 16.0}
#: a rail costing less than this fraction of the loss is BENIGN: the box was
#: clipping a nearly-flat optimum, not choosing the answer.
BENIGN = 0.005
#: Evaluation cap. Lower than the main run's 4000 because BOTH starts here
#: begin at or beside an already-converged solution, so the distance to travel
#: is short — and because the Sersic/expo/gauss families evaluate `gammainc`
#: and cost ~3 s per evaluation against the Moffat's ~0.8 s, which would
#: otherwise put a single cold-budget cell at 3+ hours.
MAX_EVALS = 2500


def widened_bounds(spec, theta):
    """Bounds with every hit UPPER bound widened. Returns (bounds, names)."""
    bounds, hit = list(spec.bounds), []
    for j, (lo, hi) in enumerate(spec.bounds):
        if lo is None or hi is None:
            continue
        if (hi - theta[j]) / (hi - lo) < 0.02:
            name = spec.theta_names[j]
            bounds[j] = (lo, max(WIDEN.get(name, 2.0 * hi), hi))
            hit.append(name)
        elif (theta[j] - lo) / (hi - lo) < 0.02:
            hit.append(f"{spec.theta_names[j]}(LOWER — not widened)")
    return bounds, hit


def main(dev=False, shard=(0, 1)):
    rows = (np.load(ROOT / "experiments/exp32_full_population/outputs/"
                    "population.npz")["dev100"] if dev else None)
    s2._w_init(rows)
    gals, R = s2._W["gals"], s2._W["e"].R
    store = F._store(dev=dev)
    done = F._load(store)
    todo = []
    for name, spec in F.all_cells():
        b = F.best_cell(done, name)
        if b is None:
            continue
        bounds, hit = widened_bounds(spec, np.asarray(b[0], float))
        if any("LOWER" not in h for h in hit):
            todo.append((name, spec, b, bounds, hit))
    idx, n = shard
    mine = [t for i, t in enumerate(todo) if i % n == idx]
    print(f"exp53 rail check — n={len(gals)}, {len(todo)} railed cell(s), "
          f"shard {idx}/{n} takes {len(mine)}", flush=True)
    for name, spec, (theta, loss, sname), bounds, hit in mine:
        print(f"\n  === {name} — at bound: {', '.join(hit)} ===", flush=True)
        for j, h in enumerate(hit):
            if "LOWER" in h:
                continue
            k = spec.theta_names.index(h)
            print(f"      {h}: {spec.bounds[k][1]:g} -> {bounds[k][1]:g}")
        prob = K.Problem(spec, gals, R)
        best = None
        starts = {"railed_fit": np.asarray(theta, float)}
        nudge = np.asarray(theta, float).copy()
        for h in hit:
            if "LOWER" in h:
                continue
            k = spec.theta_names.index(h)
            nudge[k] = 0.5 * (spec.bounds[k][1] + bounds[k][1])
        starts["nudged_past_wall"] = nudge
        for sn, p0 in starts.items():
            p0 = np.clip(p0,
                         [b[0] if b[0] is not None else -np.inf for b in bounds],
                         [b[1] if b[1] is not None else np.inf for b in bounds])
            t0 = time.time()
            r = minimize_loss(prob.loss, p0, method="lbfgsb", bounds=bounds,
                              max_evals=MAX_EVALS)
            esc = ", ".join(f"{h}={r.x[spec.theta_names.index(h)]:.3f}"
                            for h in hit if "LOWER" not in h)
            print(f"      [{sn:<17}] loss {r.fun:.8f}  -> {esc}  "
                  f"({r.nfev} evals, {(time.time()-t0)/60:.1f} min"
                  f"{', HIT THE EVAL CAP' if r.nfev >= MAX_EVALS else ''})",
                  flush=True)
            if best is None or r.fun < best.fun:
                best = r
        gain = (loss - best.fun) / loss
        verdict = ("BENIGN — the box was clipping a nearly flat optimum"
                   if gain < BENIGN else
                   "LOAD-BEARING — this cell's answer is set by its box, "
                   "NOT by the data")
        print(f"      bounded {loss:.8f} -> widened {best.fun:.8f}  "
              f"({100*gain:+.4f}% of the loss)\n      VERDICT: {verdict}",
              flush=True)
        np.savez(HERE / f"outputs/railcheck_{name}.npz", theta=best.x,
                 loss=np.array([best.fun]), bounded_loss=np.array([loss]),
                 gain=np.array([gain]))
    if not todo:
        print("  no cell converged at a bound — nothing to check")


if __name__ == "__main__":
    sh = (0, 1)
    if "--shard" in sys.argv:
        a, b = sys.argv[sys.argv.index("--shard") + 1].split("/")
        sh = (int(a), int(b))
    main("--dev" in sys.argv, sh)
