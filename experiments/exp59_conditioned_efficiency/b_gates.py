"""exp59 — the bias-sighted gate battery B1/B2/B3, the referee the loss is not.

exp58 P-B measured that the loss scores z=2 as the model's BEST epoch while
the user's eyes correctly call it the worst, because the amplitude score
divides by the per-epoch benchmark scatter (0.161 dex at z=2) and an RMS
cannot tell a bias from scatter. These gates see what the loss cannot, with
thresholds frozen in `doc/plans/2026-08-26-exp58-reassessment-plan.md` BEFORE
any Stage 1+ fit:

  B1   per-epoch |median log10 of model over measured M*(<103 kpc)|, on the
       FIT sample (the mh-complete + sane mask): <= 0.020 dex at every epoch.
       The incumbent fails only z=2, at 0.046.
  B2   the decliner-minus-rest median amplitude residual at z=2, SIGNED,
       same sample: |.| <= 0.060 dex. The incumbent: -0.122.
  B2b  the no-harm guard the leverage sweep showed is load-bearing: the same
       split at z=0.4 must not exceed the null's size (+0.082) by more than
       0.010. A candidate that closes B2 by inflating this is trading one
       epoch's split for the other's.
  B3   the median relative bias of M*(<10.25 kpc, the grid point) at z=2 on
       the z=2 fit sample must IMPROVE on the null's, while the same number
       at z <= 1.0 stays within 1 percentage point of the null's.

**The null runs alongside every candidate, every time** — "run every gate at
the null first" made structural rather than remembered.

Run (null only):        HONGSHAO_DATA_DIR=... OMP_NUM_THREADS=1 PYTHONPATH=.
                        uv run python -u experiments/exp59_conditioned_efficiency/b_gates.py
Run (an exp57 fit):     ... b_gates.py --fit "gompertz_log-E2-S2+X3+frozen_log_f0_b#pinned_size" [--start K]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP57 = ROOT / "experiments/exp57_expansion_term"
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink",
          ROOT / "experiments/exp54_unpinned_amplitude", EXP57, HERE):
    sys.path.insert(0, str(p))

import expand as X                                       # noqa: E402
import fit as F                                          # noqa: E402
import stage0_cost as S0                                 # noqa: E402
import stage4_target as S4                               # noqa: E402

#: frozen thresholds (the plan, Stage 0). Never widened.
B1_LIM, B2_LIM, B2B_GUARD, B3_TOL_PP = 0.020, 0.060, 0.010, 1.0
I10 = 6                       # R_GRID[6] = 10.25 kpc, the B3 aperture
ANCHOR_Z = (0.4, 0.7, 1.0, 1.5, 2.0)
FITDIR = EXP57 / "outputs" / "stage3_fits"
RULE = "=" * 96


def amplitude_residual(pred, data, mask):
    """log10(model/measured) at the 103 kpc anchor, NaN off the fit sample."""
    finite = np.isfinite(pred).all(axis=2) & (pred[:, :, F.I100] > 0)
    use = np.asarray(mask, bool) & finite & (data[:, :, F.I100] > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        a = np.log10(pred[:, :, F.I100] / data[:, :, F.I100])
    return np.where(use, a, np.nan), use


def measure(pred, data, mask, fell, ok):
    """Every gate quantity for one model product, as a dict."""
    a, use = amplitude_residual(pred, data, mask)
    b1 = np.array([np.nanmedian(a[:, k]) for k in range(5)])
    b2 = float(np.nanmedian(a[fell & ok, 4]) - np.nanmedian(a[(~fell) & ok, 4]))
    b2b = float(np.nanmedian(a[fell & ok, 0]) - np.nanmedian(a[(~fell) & ok, 0]))
    with np.errstate(invalid="ignore", divide="ignore"):
        rel10 = pred[:, :, I10] / data[:, :, I10] - 1.0
    rel10 = np.where(use & (data[:, :, I10] > 0), rel10, np.nan)
    b3 = 100.0 * np.array([np.nanmedian(rel10[:, k]) for k in range(5)])
    return dict(b1=b1, b2=b2, b2b=b2b, b3=b3)


def report(null, cand, label):
    """Prints the battery; returns True only if every gate passes."""
    is_null = cand is None
    m = null if is_null else cand
    verdicts = []

    print(f"\n{RULE}\nB GATES — {label}\n{RULE}")
    print(f"\n  B1 — |median amplitude bias| per epoch, fit sample "
          f"(limit {B1_LIM:.3f} dex)")
    print(f"  {'':<12}" + "".join(f"{f'z={z}':>10}" for z in ANCHOR_Z))
    print(f"  {'null':<12}" + "".join(f"{v:>+10.4f}" for v in null["b1"]))
    if not is_null:
        print(f"  {'candidate':<12}" + "".join(f"{v:>+10.4f}" for v in m["b1"]))
    ok1 = bool(np.all(np.abs(m["b1"]) <= B1_LIM))
    verdicts.append(("B1", ok1, f"worst |bias| "
                     f"{np.max(np.abs(m['b1'])):.4f} dex"))

    print(f"\n  B2 — decliner-minus-rest amplitude split at z=2, signed "
          f"(limit |{B2_LIM:.3f}| dex)")
    print(f"    null {null['b2']:+.4f}"
          + ("" if is_null else f"   candidate {m['b2']:+.4f}"))
    ok2 = abs(m["b2"]) <= B2_LIM
    verdicts.append(("B2", ok2, f"{m['b2']:+.4f} dex"))

    lim2b = abs(null["b2b"]) + B2B_GUARD
    print(f"\n  B2b — the same split at z=0.4 must stay within the null's "
          f"size + {B2B_GUARD:.3f} (= {lim2b:.3f})")
    print(f"    null {null['b2b']:+.4f}"
          + ("" if is_null else f"   candidate {m['b2b']:+.4f}"))
    ok2b = abs(m["b2b"]) <= lim2b
    verdicts.append(("B2b", ok2b, f"{m['b2b']:+.4f} dex"))

    print(f"\n  B3 — median relative bias of M*(<10.25 kpc) per epoch [%]; "
          f"z=2 must improve on the null,\n  z <= 1.0 within "
          f"{B3_TOL_PP:.0f} point of the null")
    print(f"  {'':<12}" + "".join(f"{f'z={z}':>10}" for z in ANCHOR_Z))
    print(f"  {'null':<12}" + "".join(f"{v:>+9.1f}%" for v in null["b3"]))
    if not is_null:
        print(f"  {'candidate':<12}" + "".join(f"{v:>+9.1f}%" for v in m["b3"]))
    if is_null:
        verdicts.append(("B3", None, f"z=2 {m['b3'][4]:+.1f}% — the null IS "
                                     f"the reference, B3 has no verdict here"))
    else:
        ok3 = (abs(m["b3"][4]) < abs(null["b3"][4])
               and bool(np.all(np.abs(m["b3"][:3] - null["b3"][:3])
                               <= B3_TOL_PP)))
        verdicts.append(("B3", ok3, f"z=2 {m['b3'][4]:+.1f}% against the "
                                    f"null's {null['b3'][4]:+.1f}%"))

    print(f"\n  {'gate':<6}{'verdict':<8}measured")
    for g, okv, txt in verdicts:
        word = "—" if okv is None else ("PASS" if okv else "FAIL")
        print(f"  {g:<6}{word:<8}{txt}")
    return all(v for _, v, _ in verdicts if v is not None)


def main(argv):
    smoke = "--smoke" in argv
    recs, data, mask, lmh, base, theta, slow = S0.build(smoke)
    fell, ok = S4.decline_flag(data)
    print(f"exp59 B GATES — sample {len(recs)} galaxies; decliners "
          f"{int((fell & ok).sum())}")

    null = measure(slow.predict(theta), data, mask, fell, ok)
    if "--fit" not in argv:
        report(null, None, "THE NULL (the exp54 incumbent) — every number is "
                           "the state before any candidate")
        return

    label = argv[argv.index("--fit") + 1]
    z = np.load(FITDIR / f"{label}.npz", allow_pickle=True)
    sp = X.XSpec(base, str(z["law"]))
    pr = X.ExpandingProblem(sp, recs, data, mask, epochs=(0, 1, 2, 3, 4))
    th = np.asarray(z["theta"], float)
    shown = label
    if "--start" in argv:
        k = int(argv[argv.index("--start") + 1])
        th = np.asarray(z["start_thetas"], float)[k]
        shown = f"{label}#start{k}"
    print(f"  candidate: {shown}\n  parameters: "
          + ", ".join(f"{n}={v:+.4f}" for n, v in
                      zip(sp.x_names, th[base.n_theta:])))
    cand = measure(pr.predict(th), data, mask, fell, ok)
    report(null, cand, shown)


if __name__ == "__main__":
    main(sys.argv)
