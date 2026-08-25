"""exp57 Stage 8 — the trade between reach and the core-density target.

**The question this closes.** Every long-reach model passes gate G5(c) — the
model's median core surface-density growth against the measurement — and fails
G5(b); the one short-reach model does the reverse. Stated that way it is an
observation about six fits. This makes it a curve: hold everything else and
sweep the core radius `Rc` from "does nothing" to "effectively homologous",
measuring the reach gate and the core-density gap at each value.

**IT IS A CONDITIONAL SCAN AND THAT IS SAID IN THE OUTPUT.** The other eight
parameters are held fixed, so this is exactly the object exp54 Stage 3.9 spent a
day showing over-states how well determined a parameter is — there the
conditional slice exaggerated every parameter's width by 2.3x to 17.6x. It is
used here for something a conditional slice CAN do honestly: show the shape of
a trade between two measured quantities along one axis. It is not used to
locate an optimum, and the two fitted basins are marked on it so the reader can
see where the honest refits actually landed.

Two curves are drawn, one at each fitted basin's other parameters, precisely
because a single conditional curve would hide how much the answer depends on
where it is anchored. Where the two curves agree, the shape of the trade is a
property of `Rc`; where they diverge, it is not.

Run: HONGSHAO_DATA_DIR=... OMP_NUM_THREADS=1 PYTHONPATH=. uv run python -u \\
     experiments/exp57_expansion_term/stage8_tradeoff.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP54 = ROOT / "experiments/exp54_unpinned_amplitude"
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink", EXP54, HERE):
    sys.path.insert(0, str(p))

import expand as X                                       # noqa: E402
import gates as G                                        # noqa: E402
import stage0_cost as S0                                 # noqa: E402
import stage35_time_law as S35                           # noqa: E402
import stage4_target as S4                               # noqa: E402

OUT = HERE / "outputs"
FITDIR = OUT / "stage3_fits"
#: log10 Rc in kpc: 0.5 kpc (does essentially nothing) to 1000 kpc (homologous
#: on every radius anyone measures)
LOG_RC = np.linspace(-0.3, 3.0, 15)
ANCHORS = ("gompertz_log-E2-S2+X3#smallcore", "gompertz_log-E2-S2+X3#bigcore")
RULE = "=" * 100
_P = S35._PC


def main():
    print(f"{RULE}\nexp57 STAGE 8 — how the REACH trades against the "
          f"CORE-DENSITY target\n{RULE}\n")
    print(f"  A CONDITIONAL scan: `Rc` is swept with the other eight "
          f"parameters HELD FIXED.\n  exp54 Stage 3.9 measured that "
          f"conditional slices over-state a parameter's\n  determination by "
          f"2.3x to 17.6x, so this is NOT used to locate an optimum. It is\n"
          f"  used to show the SHAPE of a trade along one axis, and it is "
          f"drawn twice — once\n  at each fitted basin's other parameters — so "
          f"that dependence on the anchor is\n  visible rather than hidden.\n")

    recs, data, mask, lmh, base, theta, slow = S0.build(False)
    use = mask[:, 4] & mask[:, 0]
    d_dat, r_mid, _ = S4.core_dlogsigma(data, use)
    fell, ok_flag = S4.decline_flag(data)
    sp = X.XSpec(base, "X3")
    pr = X.ExpandingProblem(sp, recs, data, mask, epochs=(0, 1, 2, 3, 4))
    j_rc = sp.theta_names.index("log_Rc")

    print(f"  the measured core dlogSigma at {r_mid:.2f} kpc is "
          f"{d_dat:+.4f} dex; the null model is +0.1025,\n  so the null's gap "
          f"is +0.0455 and gate G5(c) asks for it to SHRINK.\n")

    results = {}
    for anchor in ANCHORS:
        f = FITDIR / f"{anchor}.npz"
        if not f.exists():
            print(f"  (no {anchor}, skipping)")
            continue
        th0 = np.asarray(np.load(f, allow_pickle=True)["theta"], float)
        tag = anchor.split("#")[-1]
        print(f"  ANCHORED AT THE {tag.upper()} FIT "
              f"(A = {th0[sp.theta_names.index('A')]:+.3f}, everything but "
              f"Rc held there)")
        print(f"    {'Rc [kpc]':>10}{'loss':>10}{'shape':>9}"
              f"{'G2 >50 kpc':>12}{'G2 >148':>10}{'core gap':>11}"
              f"{'span not':>10}")
        rows = []
        for lrc in LOG_RC:
            th = th0.copy()
            th[j_rc] = lrc
            p = pr.predict(th)
            g2 = G.g2_reach(pr, th)
            worst = max(g2, key=lambda q: q["f_beyond_50"])
            c_mod, _, _ = S4.core_dlogsigma(p, use)
            _, span_not = S4.central_span(p, data, mask, (~fell) & ok_flag)
            _, sf = pr.per_epoch(th)
            rows.append((10 ** lrc, pr.loss(th), S35.shape_pct(sf),
                         worst["f_beyond_50"], worst["f_beyond_148"],
                         c_mod - d_dat, span_not))
            flag = ""
            if worst["f_beyond_50"] <= G.G2_MAX_BEYOND_50:
                flag += "  G2 ok"
            if abs(c_mod - d_dat) < 0.0455:
                flag += "  G5c closer"
            print(f"    {10 ** lrc:>10.2f}{rows[-1][1]:>10.5f}"
                  f"{rows[-1][2]:>8.3f}{_P}"
                  f"{100 * rows[-1][3]:>11.1f}{_P}{100 * rows[-1][4]:>9.1f}{_P}"
                  f"{rows[-1][5]:>+11.4f}{rows[-1][6]:>10.3f}{flag}")
        results[tag] = np.array(rows)
        print()

    print(f"  READING IT. G2 needs the reach column below "
          f"{100 * G.G2_MAX_BEYOND_50:.0f}{_P}; G5(c) needs the core gap\n"
          f"  closer to zero than the null's +0.0455. If no `Rc` does both on "
          f"either anchor,\n  the trade is a property of this mechanism and "
          f"not of where the scan was started.")
    for tag, R in results.items():
        both = (R[:, 3] <= G.G2_MAX_BEYOND_50) & (np.abs(R[:, 5]) < 0.0455)
        if both.any():
            print(f"    {tag}: BOTH satisfied at Rc = "
                  + ", ".join(f"{v:.1f} kpc" for v in R[both, 0]))
        else:
            print(f"    {tag}: NO value of Rc satisfies both. Best reach with "
                  f"G5(c) closer than the null:\n      "
                  + (f"Rc = {R[np.abs(R[:, 5]) < 0.0455][0, 0]:.1f} kpc at "
                     f"{100 * R[np.abs(R[:, 5]) < 0.0455][0, 3]:.1f}"
                     f"{_P} beyond 50 kpc"
                     if (np.abs(R[:, 5]) < 0.0455).any()
                     else "none — G5(c) never closes on this anchor"))

    if results:
        np.savez(OUT / "stage8_tradeoff.npz",
                 log_rc=LOG_RC, core_data=d_dat, r_mid=r_mid,
                 **{f"{t}_{k}": results[t][:, i] for t in results
                    for i, k in enumerate(("rc", "loss", "shape", "b50",
                                           "b148", "core_gap", "span_not"))})
        print(f"\n  wrote stage8_tradeoff.npz")


if __name__ == "__main__":
    main()
