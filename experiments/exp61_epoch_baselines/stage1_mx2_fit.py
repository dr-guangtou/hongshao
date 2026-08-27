"""exp61 Stage 1 — the mass x time-curvature fit (`M2`), the one mean-model
candidate exp61's own measurements point at.

WHY THIS TERM AND NO OTHER. exp61's single-epoch fits showed the z=2
systematic offset is entirely the cross-epoch compromise: freed of the other
epochs, the same seven parameters wipe out the z=2 median bias (−0.046 →
+0.002 dex) and the inner-aperture deficit (−12.7% → +0.3%), with every
efficiency parameter drifting monotonically across the single-epoch fits —
in particular the E2 cross term `a_Mz` walking +0.42 → +0.86. A walking
cross term is curvature in the mass x time plane; `M2` adds exactly that one
degree of freedom, `a_mz2 * (logmh − 13.5) * ln(1+z)²` per deposit, through
exp59's exact-nesting conditioning machinery. It is NOT a new conditioning
variable, so exp59's lockstep disqualification (which was about the decliner
SPLITS, unreachable by any per-deposit reweighting) does not apply; the
splits are gated below to confirm they are not disturbed.

PRE-REGISTERED, before the fit: `a_mz2 > 0` (the drift's direction), and the
verdict comes from the frozen B gates, not the loss — B1 must pass at every
epoch (the incumbent fails only z=2, at 0.046 dex), B2/B2b must stay within
their null sizes + 0.010, B3's z=2 entry must improve with z <= 1.0 within a
point. Identifiability: four starts including the wrong sign; a railing or
sign-splitting slope is a null result.

Run: HONGSHAO_DATA_DIR=... OMP_NUM_THREADS=1 PYTHONPATH=. uv run python -u \\
     experiments/exp61_epoch_baselines/stage1_mx2_fit.py [--smoke]
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
          ROOT / "experiments/exp57_expansion_term",
          ROOT / "experiments/exp59_conditioned_efficiency"):
    sys.path.insert(0, str(p))

import b_gates as BG                                     # noqa: E402
import conditioned as C                                  # noqa: E402
import fit as F                                          # noqa: E402
import stage0_cost as S0                                 # noqa: E402
import stage35_time_law as S35                           # noqa: E402
import stage4_target as S4                               # noqa: E402

A_STARTS = (0.5, -0.5, 1.5)
OUT = HERE / "outputs" / "stage1_mx2_fit.npz"
RULE = "=" * 96


def main(smoke=False):
    print(f"{RULE}\nexp61 STAGE 1 — the mass x time-curvature term M2"
          f"\n{RULE}\n")
    recs, data, mask, lmh, base, theta, slow = S0.build(smoke)
    fell, ok = S4.decline_flag(data)
    sp = C.CSpec(base, "M2")
    pr = C.ConditionedProblem(sp, recs, data, mask, epochs=(0, 1, 2, 3, 4))

    l0 = slow.loss(theta)
    th0 = sp.nest(theta)
    lx = pr.loss(th0)
    print(f"  incumbent loss {l0:.6f}; nested start reproduces it to "
          f"{abs(lx - l0):.2e}")
    assert abs(lx - l0) < 1e-9, "nesting broken — refusing to fit"

    lo, hi = np.array(sp.bounds()).T
    starts = [th0]
    for a in A_STARTS:
        th = th0.copy()
        th[-1] = a
        starts.append(np.clip(th, lo, hi))
    if smoke:
        starts = starts[:2]

    best_th, best_l, per = None, np.inf, []
    for i, p0 in enumerate(starts):
        ts = time.time()
        th, l = F.fit(pr, starts=[p0],
                      maxiter=(300 if smoke else 600 * sp.n_theta),
                      verbose=False)
        per.append((float(l), th.copy()))
        flag = "   <- best" if l < best_l else ""
        if l < best_l:
            best_th, best_l = th, l
        print(f"    start {i} (a_mz2 {p0[-1]:+.2f}): loss {l:.6f}, "
              f"a_mz2 = {th[-1]:+.4f}  [{(time.time() - ts) / 60:.1f} min]"
              f"{flag}", flush=True)

    slopes = [t[-1] for _, t in per]
    print(f"\n  best loss {best_l:.6f} ({100 * (1 - best_l / l0):+.2f}% vs "
          f"the incumbent); fitted a_mz2 = {best_th[-1]:+.4f}")
    print(f"  slope across starts: "
          + ", ".join(f"{v:+.4f}" for v in slopes)
          + ("   <- SIGN SPLIT, a null result"
             if (max(slopes) > 0) != (min(slopes) > 0)
             and max(np.abs(slopes)) > 0.05 else ""))
    if abs(best_th[-1] - lo[-1]) < 1e-3 or abs(best_th[-1] - hi[-1]) < 1e-3:
        print(f"  AT A BOUND — report as a bound, not a value")
    sa, sf = pr.per_epoch(best_th)
    print(f"  profile-shape error {S35.shape_pct(sf):.3f}% "
          f"(incumbent 13.787%); per-epoch score_A "
          + " ".join(f"{v:.3f}" for v in sa))

    # the frozen B gates, null alongside, verdict from the gates not the loss
    null = BG.measure(slow.predict(theta), data, mask, fell, ok)
    cand = BG.measure(pr.predict(best_th), data, mask, fell, ok)
    passed = BG.report(null, cand, f"M2, a_mz2 = {best_th[-1]:+.4f}")
    print(f"\n  B-GATE VERDICT: {'PASS' if passed else 'FAIL'}")

    if not smoke:
        OUT.parent.mkdir(exist_ok=True)
        np.savez(OUT, theta=best_th, loss=best_l, loss_incumbent=l0,
                 names=np.array(sp.theta_names),
                 start_losses=np.array([l for l, _ in per]),
                 start_thetas=np.array([t for _, t in per]),
                 sA=sa, sF=sf, shape_pct=S35.shape_pct(sf),
                 b_pass=passed)
        print(f"  saved {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv[1:])
