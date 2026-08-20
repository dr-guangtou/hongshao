"""exp53 — is the deposition-only fit's `mu` rail load-bearing?

`moffat-q0` converged with the efficiency-window centre `mu` exactly at its
widened bound of 5.0. Design constraint 2 says the window must be FREE, so a
rail there would make the whole deposition-only result uninterpretable -- it
would mean the model was still being told when it may deposit.

Two things had to be separated.

1. **The (mu, sig) degeneracy.** `mu = 5` puts the lognormal's peak at
   `1 + z = e^5 = 148`, far outside the sample's `ln(1+z) <= ~2.5`. Out there
   the window is not a peak at all, it is a RAMP, and only the ramp is
   identified: many (mu, sig) pairs give the same weights over the observed
   range. A rail along a flat ridge is cosmetic.
   Probed directly (holding the ramp `ln[w(z=11)/w(z=0)]` fixed and sliding
   `mu`), the answer was that this is NOT a flat ridge -- the loss has a real
   interior minimum at `mu ~ 5`, rising to 0.16175 at `mu = 4` and 0.16253 at
   `mu = 8`. The bound happens to sit essentially on the optimum.

2. **Whether a genuine refit agrees.** The probe above is a SLICE, and exp49
   measured that slices and profiles disagree in sign in this model. So this
   script refits the cell properly with `mu` free to 12, from three starts,
   and reports what the extra freedom buys.

Run: `HONGSHAO_DATA_DIR=... PYTHONPATH=. uv run python \
experiments/exp53_deposition_only/mu_check.py [--dev]`
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

MU_HI = 12.0


def main(dev=False):
    rows = (np.load(ROOT / "experiments/exp32_full_population/outputs/"
                    "population.npz")["dev100"] if dev else None)
    s2._w_init(rows)
    gals, R = s2._W["gals"], s2._W["e"].R
    spec = K.Spec("moffat", q_free=False, q_fixed=0.0)
    bounds = list(spec.bounds)
    i_mu = spec.theta_names.index("mu")
    bounds[i_mu] = (K.BASE_BOX["mu"][0], MU_HI)
    prob = K.Problem(spec, gals, R)

    store = F._store(dev=dev)
    base = K.from_exp38_theta(K.adopted_theta())
    fitted = F.best_cell(F._load(store), spec.label) if store.exists() else None
    print(f"exp53 mu rail check — {spec.label}, n={len(gals)}, mu box "
          f"{K.BASE_BOX['mu'][1]:g} -> {MU_HI:g}")
    if fitted is not None:
        print(f"  bounded reference: loss {fitted[1]:.8f}, "
              f"mu = {fitted[0][i_mu]:.4f} (bound "
              f"{K.BASE_BOX['mu'][1]:g})")

    starts = dict(F.starts(spec, F.project(spec, base)))
    if fitted is not None:
        starts["bounded_fit"] = np.asarray(fitted[0], float)
        nudged = np.asarray(fitted[0], float).copy()
        nudged[i_mu] = 8.0                       # push well past the old wall
        starts["nudged_mu8"] = nudged
    best = None
    for name, p0 in starts.items():
        p0 = np.clip(p0, [b[0] if b[0] is not None else -np.inf
                          for b in bounds],
                     [b[1] if b[1] is not None else np.inf for b in bounds])
        t0 = time.time()
        r = minimize_loss(prob.loss, p0, method="lbfgsb", bounds=bounds,
                          max_evals=F.MAX_EVALS)
        print(f"  [{name:<16}] loss {r.fun:.8f}  mu {p0[i_mu]:.2f} -> "
              f"{r.x[i_mu]:.4f}  sig {r.x[i_mu+1]:.4f}  "
              f"({r.nfev} evals, {(time.time()-t0)/60:.1f} min, "
              f"conv={r.success})", flush=True)
        if best is None or r.fun < best.fun:
            best = r
    print(f"\n  best: loss {best.fun:.8f}, mu = {best.x[i_mu]:.4f}")
    if fitted is not None:
        gain = (fitted[1] - best.fun) / fitted[1]
        print(f"  the extra mu freedom buys {100*gain:+.4f}% of the loss")
        print("  VERDICT: " + (
            "the rail is BENIGN — the widened fit does not escape it "
            "materially, so the window was effectively free."
            if abs(gain) < 0.005 and best.x[i_mu] < MU_HI * 0.98 else
            "the rail is LOAD-BEARING — re-run the shootout with the wider "
            "box before reading any deposition-only result."))
    np.savez(HERE / "outputs/mu_check.npz", theta=best.x,
             loss=np.array([best.fun]), mu_hi=np.array([MU_HI]))


if __name__ == "__main__":
    main("--dev" in sys.argv)
