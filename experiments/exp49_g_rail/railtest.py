"""THE RAIL TEST — is g=4.0 a property of the data, or of the optimizer?

The adopted 1ch-mof theta has g (the exponent setting how the deposit scale
grows with time) at 4.00003 against an upper bound of 4.0, held there by the
soft box penalty. On 100 galaxies every optimizer that reached the minimum put
g at ~3.54, interior; only the ones that stalled pinned it. But the full-sample
refit still ended at 4.0 -- and it started FROM the railed point, using the
one-sided penalty whose derivative is exactly zero AT the bound, so a gradient
method can report convergence at a kink it has no information to leave.

Three candidate explanations, separated here:
  (a) the DATA wants g = 4.0          -> every start converges there
  (b) the STARTING POINT holds it     -> starts from below stay below
  (c) the PENALTY kink holds it       -> real bounds release it, penalty does not

Design: L-BFGS-B, fit scope z<=1.5, full sample, from five starts, under TWO
formulations -- real bounds with no penalty term, and the production penalty
with no bounds. Resumable: every completed cell is written immediately.

Run: HONGSHAO_DATA_DIR=... PYTHONPATH=. uv run python \
experiments/exp49_g_rail/railtest.py [--dev]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/exp38_deposit_rethink"))
import stage2_multiepoch as s2                      # noqa: E402
from hongshao.fitting import minimize_loss          # noqa: E402

OUTDIR = HERE / "outputs"
KS = [0, 1, 2, 3]
NAMES = ["log_rc", "g", "q", "mu", "sig", "gamma",
         "rc:logMh", "rc:c200c", "rc:fz2", "sig:logMh", "sig:c200c", "sig:fz2"]

# The production box, read off exp35 run.py LO/HI and stage2 GAMMA_BOX. The six
# conditioning slopes carry NO penalty in 1ch-mof, so they stay unbounded here
# too -- the two formulations must differ ONLY in how the box is imposed.
BOX_LO = [1.0, 0.0, 0.0, 0.0, 0.05, 1.05] + [None] * 6
BOX_HI = [3.0, 4.0, 3.0, 3.0, 2.0, 6.0] + [None] * 6
BOUNDS = list(zip(BOX_LO, BOX_HI))


def make_starts(th):
    """Five starts. The one that matters most is 'g_low': if the data wants
    g = 4.0, a run started at g = 2.0 must climb there."""
    rng = np.random.default_rng(0)
    s = {}
    s["adopted"] = np.clip(th, [-np.inf if x is None else x for x in BOX_LO],
                           [np.inf if x is None else x for x in BOX_HI])
    lo = th.copy()
    lo[1] = 2.0
    s["g_low"] = lo
    up = th.copy()
    up[:6] *= 1.20
    up[6:] = 0.0
    s["base_x1.2"] = np.clip(up, [-np.inf if x is None else x for x in BOX_LO],
                             [np.inf if x is None else x for x in BOX_HI])
    dn = th.copy()
    dn[:6] *= 0.80
    s["base_x0.8"] = dn
    rp = th.copy()
    rp[:6] = rp[:6] * (1.0 + rng.normal(0, 0.15, 6))
    rp[6:] = rng.normal(0, 0.03, 6)
    s["random"] = np.clip(rp, [-np.inf if x is None else x for x in BOX_LO],
                          [np.inf if x is None else x for x in BOX_HI])
    return s


def main(dev=False):
    rows = (np.load(ROOT / "experiments/exp32_full_population/outputs/"
                    "population.npz")["dev100"] if dev else None)
    s2._w_init(rows)
    gals = s2._W["gals"]
    th = np.asarray(np.load(ROOT / "experiments/exp40_epoch_objective/"
                            "outputs/latestart.npz")["theta_z15"], float)[:12]
    tag = "_dev" if dev else ""
    OUTDIR.mkdir(exist_ok=True)
    store = OUTDIR / f"railtest{tag}.npz"
    done = dict(np.load(store, allow_pickle=True)) if store.exists() else {}

    def loss_plain(p):
        return float(np.mean([s2.gal_loss(p, g, KS, "1ch-mof") for g in gals]))

    def loss_pen(p):
        return loss_plain(p) + s2.penalty(p, "1ch-mof")

    print(f"n = {len(gals)} galaxies, fit scope z<=1.5", flush=True)
    print(f"adopted g = {th[1]:.5f} (upper bound 4.0), "
          f"log_rc = {th[0]:.5f} (upper bound 3.0)\n", flush=True)

    starts = make_starts(th)
    cells = [(f, s) for f in ("bounded", "penalty") for s in starts]
    for form, sname in cells:
        key = f"{form}::{sname}"
        if f"theta::{key}" in done:
            print(f"  [{key}] cached", flush=True)
            continue
        p0 = starts[sname]
        t0 = time.time()
        if form == "bounded":
            r = minimize_loss(loss_plain, p0, method="lbfgsb", bounds=BOUNDS,
                              max_evals=4000)
        else:
            r = minimize_loss(loss_pen, p0, method="lbfgsb", max_evals=4000)
        done[f"theta::{key}"] = r.x
        done[f"loss::{key}"] = np.array([float(r.fun)])
        done[f"start::{key}"] = p0
        done[f"meta::{key}"] = np.array([json.dumps(
            dict(nfev=int(r.nfev), success=bool(r.success),
                 minutes=round((time.time() - t0) / 60, 2)))])
        np.savez(store, **done)          # after EVERY fit: this runs unattended
        print(f"  [{key:<22}] loss {r.fun:.6f}  g {p0[1]:.3f} -> "
              f"{r.x[1]:.5f}   log_rc {p0[0]:.3f} -> {r.x[0]:.5f}   "
              f"({r.nfev} evals, {(time.time()-t0)/60:.1f} min)", flush=True)

    # ---- verdict --------------------------------------------------------- #
    print("\n" + "=" * 74, flush=True)
    print("VERDICT", flush=True)
    print("=" * 74, flush=True)
    print(f"  {'formulation':<13}{'start':<13}{'loss':>11}{'g':>10}"
          f"{'log_rc':>10}  at a bound", flush=True)
    for form in ("bounded", "penalty"):
        for sname in starts:
            key = f"{form}::{sname}"
            x, lo = done[f"theta::{key}"], done[f"loss::{key}"][0]
            hits = [NAMES[i] for i in range(6)
                    if abs(x[i] - BOX_LO[i]) < 1e-4 or abs(x[i] - BOX_HI[i]) < 1e-4]
            print(f"  {form:<13}{sname:<13}{lo:11.6f}{x[1]:10.5f}"
                  f"{x[0]:10.5f}  {', '.join(hits) if hits else '-'}",
                  flush=True)
    gs = np.array([done[f"theta::bounded::{s}"][1] for s in starts])
    print(f"\n  bounded runs: g spans {gs.min():.4f} to {gs.max():.4f}", flush=True)
    print(f"  g started at 2.0 ended at "
          f"{done['theta::bounded::g_low'][1]:.5f} (bounded) / "
          f"{done['theta::penalty::g_low'][1]:.5f} (penalty)", flush=True)


if __name__ == "__main__":
    main("--dev" in sys.argv)
