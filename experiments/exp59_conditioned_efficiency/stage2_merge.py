"""exp59 Stage 2 — merge the two halves of the interrupted pinned-size fit.

The 9-start refit of `X3` with the size law frozen was killed externally after
starts 0–3 (they live in the `.PARTIAL.npz` checkpoint that `stage3_fit.py`
writes after every start, for exactly this situation). Starts 4–8 were then
reproduced with `--starts-only 4,5,6,7,8 --tag pinned_size_b` — the start list
is deterministic (fixed seed, same spec, same incumbent theta), so the subset
run computes exactly what the full run would have. This merges the halves into
the canonical fit file, after checking they really are halves of the same fit:
same parameter names, same frozen values, same incumbent theta and loss. The
per-epoch scores of the overall best are recomputed here rather than copied,
because each half stored scores for its own best only.

Run: HONGSHAO_DATA_DIR=... OMP_NUM_THREADS=1 PYTHONPATH=. uv run python -u \\
     experiments/exp59_conditioned_efficiency/stage2_merge.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP57 = ROOT / "experiments/exp57_expansion_term"
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink",
          ROOT / "experiments/exp54_unpinned_amplitude", EXP57):
    sys.path.insert(0, str(p))

import expand as X                                       # noqa: E402
import stage0_cost as S0                                 # noqa: E402
import stage35_time_law as S35                           # noqa: E402

FITDIR = EXP57 / "outputs" / "stage3_fits"
LABEL = "gompertz_log-E2-S2+X3+frozen_log_f0_b#pinned_size"
PART_A = FITDIR / f"{LABEL}.PARTIAL.npz"
PART_B = FITDIR / f"{LABEL}_b.npz"
OUT = FITDIR / f"{LABEL}.npz"


def main():
    a = np.load(PART_A, allow_pickle=True)
    b = np.load(PART_B, allow_pickle=True)

    # the two runs must be halves of the SAME fit, or the merge means nothing
    assert list(a["names"]) == list(b["names"]), "parameter names differ"
    assert list(a["frozen"]) == list(b["frozen"]), "frozen sets differ"
    assert np.array_equal(a["theta_incumbent"], b["theta_incumbent"]), \
        "different incumbent theta"
    assert float(a["loss_incumbent"]) == float(b["loss_incumbent"]), \
        "different incumbent loss"
    la, lb = np.asarray(a["start_losses"]), np.asarray(b["start_losses"])
    ta, tb = np.asarray(a["start_thetas"]), np.asarray(b["start_thetas"])
    assert len(la) == 4 and len(lb) == 5, \
        f"expected 4 + 5 starts, got {len(la)} + {len(lb)}"

    losses = np.concatenate([la, lb])
    thetas = np.concatenate([ta, tb])
    k = int(np.argmin(losses))
    print(f"merging {len(la)} + {len(lb)} starts of {LABEL}")
    print(f"  {'start':>6} {'loss':>10}   A        log_Rc")
    for i, (l, t) in enumerate(zip(losses, thetas)):
        print(f"  {i:>6} {l:>10.6f}   {t[-2]:+.4f}  {t[-1]:+.4f}"
              + ("   <- best" if i == k else ""))
    spread = float(losses.max() - losses.min())
    print(f"  start-to-start spread {spread:.2e}")

    # recompute the best optimum's per-epoch scores on the full sample
    recs, data, mask, lmh, base, theta, slow = S0.build(False)
    sp = X.XSpec(base, str(a["law"]))
    pr = X.ExpandingProblem(sp, recs, data, mask, epochs=(0, 1, 2, 3, 4))
    best = thetas[k]
    l_check = pr.loss(best)
    assert abs(l_check - losses[k]) < 1e-9, \
        f"stored loss {losses[k]:.9f} does not reproduce ({l_check:.9f})"
    sa, sf = pr.per_epoch(best)
    print(f"  best loss {losses[k]:.6f} reproduced on the full sample; "
          f"shape error {S35.shape_pct(sf):.3f}% (incumbent 13.787%)")

    np.savez(OUT, label=LABEL, law=a["law"], frozen=a["frozen"],
             theta=best, loss=losses[k],
             loss_incumbent=a["loss_incumbent"],
             theta_incumbent=a["theta_incumbent"], names=a["names"],
             n_starts=len(losses), start_losses=losses, start_thetas=thetas,
             sA=sa, sF=sf, shape_pct=S35.shape_pct(sf))
    PART_A.unlink()
    print(f"wrote {OUT.relative_to(ROOT)} (9 starts); removed the PARTIAL")


if __name__ == "__main__":
    main()
