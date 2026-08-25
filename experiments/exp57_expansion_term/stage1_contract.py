"""exp57 Stage 1 — the contract, and the capability check that matters most.

**THE CONTRACT CHANGE.** exp54's Section 8.10 established analytically that
with every deposit non-negative and its profile fixed once made, `M*(<R,t)` is
non-decreasing in `t` at **every** radius and **every** parameter value. That
is the property expansion deliberately breaks. It was an analytic claim in the
technical note rather than a coded assertion, so this file turns it into a
two-branch MEASUREMENT:

  * with the expansion off, the model's enclosed mass must never fall — checked
    over every galaxy, every radius and every epoch pair, not argued;
  * with it on, it must be *able* to fall.

**THE CAPABILITY CHECK, which is the real content.** Admitting a decline in
principle is worthless if the achievable decline is a hundredth of the observed
one. The target is measured and preregistered: **42% of these galaxies lose
central stellar mass between redshift 2 and 0.4, by a median of 14%**
(−0.066 dex at 4.92 kpc, 10th–90th percentile −0.129 to −0.019).

So this scans the expansion strength and asks, at each value, what fraction of
galaxies the model makes decline and by how much — against those two numbers.
**If no attainable expansion reproduces a 42% / −0.066 dex decline, the
experiment is over before it is fitted**, and that is worth knowing in seconds
rather than after a day of optimisation.

It costs one prediction per grid point (about half a second), so it runs before
anything expensive.

Run: HONGSHAO_DATA_DIR=... OMP_NUM_THREADS=1 PYTHONPATH=. uv run python -u \\
     experiments/exp57_expansion_term/stage1_contract.py [--smoke]
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
import fit as F                                          # noqa: E402
import stage0_cost as S0                                 # noqa: E402

#: the aperture Stage 3.4 preregistered its decline flag on
R_FLAG = 4.92
I_FLAG = int(np.argmin(np.abs(F.R_GRID - R_FLAG)))
#: What the DATA does, from exp54 Stage 3.4 — and note WHICH POPULATION each
#: number describes, because the first draft of this file got it wrong (see
#: PROBLEMS.md P3). `TRUTH_FRAC` is over ALL galaxies; `TRUTH_DECLINER_DEX` is
#: the median over the DECLINING SUBSET ONLY, not over everyone. Comparing a
#: model median over all galaxies against it is a category error, and it is the
#: same category error this project has now recorded four times.
TRUTH_FRAC, TRUTH_DECLINER_DEX = 0.42, -0.066
#: the expansion strengths scanned. Coarse and wide: this is a capability
#: question, not a fit.
A_GRID = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0)
OUT = HERE / "outputs" / "stage1_contract.npz"
RULE = "=" * 96


def decline_stats(pred, i_r=I_FLAG, k_early=4, k_late=0):
    """The four numbers that describe how a population's centres evolve.

    Returns a dict, not a tuple, so a caller cannot silently compare the median
    over ALL galaxies against a target defined on the DECLINING SUBSET — which
    is exactly what the first version of this file did.
    """
    a, b = pred[:, k_early, i_r], pred[:, k_late, i_r]
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    d = np.log10(b[ok]) - np.log10(a[ok])
    dec = d[d < 0]
    return dict(all_median=float(np.median(d)),
                frac_declining=float((d < 0).mean()),
                decliner_median=float(np.median(dec)) if dec.size else np.nan,
                p10=float(np.percentile(d, 10)),
                p90=float(np.percentile(d, 90)),
                d=d)


def main(smoke=False):
    print(f"{RULE}\nexp57 STAGE 1 — the contract change, and whether it buys "
          f"the capability it was for\n{RULE}\n")
    recs, data, mask, lmh, base, theta, slow = S0.build(smoke)
    print(f"  sample: {len(recs)} galaxies, {int(mask.sum())} galaxy-epochs")

    # what the DATA does, recomputed here rather than quoted
    T = decline_stats(np.where(np.isfinite(data), data, np.nan))
    _P = S0.S35._PC
    print(f"\n  THE TARGET, recomputed on this sample rather than quoted, and "
          f"stated with the\n  POPULATION each number describes — the first "
          f"draft of this file compared a median\n  over all galaxies against "
          f"a median over the decliners (PROBLEMS.md P3):")
    print(f"    aperture M*(<{F.R_GRID[I_FLAG]:.2f} kpc), redshift 2 -> 0.4, "
          f"{len(T['d'])} galaxies")
    print(f"      median over ALL galaxies        {T['all_median']:+.4f} dex  "
          f"({100 * (10 ** T['all_median'] - 1):+.1f}{_P}) — the centres GROW "
          f"on the median")
    print(f"      fraction that DECLINE           "
          f"{100 * T['frac_declining']:.1f}{_P}   "
          f"(Stage 3.4 reported {100 * TRUTH_FRAC:.0f}{_P})")
    print(f"      median among the DECLINERS      "
          f"{T['decliner_median']:+.4f} dex  "
          f"({100 * (10 ** T['decliner_median'] - 1):+.1f}{_P})   "
          f"(Stage 3.4 reported {TRUTH_DECLINER_DEX:+.3f})")
    print(f"      10th to 90th percentile         {T['p10']:+.3f} to "
          f"{T['p90']:+.3f} dex  — a very WIDE distribution")
    print(f"\n    So the target is not a shift, it is a SHAPE: a population "
          f"whose centres grow by\n    {T['all_median']:+.3f} dex on the "
          f"median while {100 * T['frac_declining']:.0f}{_P} of them fall. A "
          f"single global expansion\n    strength moves the whole "
          f"distribution and cannot widen it, so whether one value\n    of A "
          f"can do both at once is the question this scan answers.")

    sp = X.XSpec(base, "X1")
    pr = X.ExpandingProblem(sp, recs, data, mask, epochs=(0, 1, 2, 3, 4))

    # ---------------------------------------------------------------- #
    print(f"\n  BRANCH 1 — with the expansion OFF, the enclosed mass must "
          f"NEVER fall.")
    print(f"    Checked over every galaxy, every radius and every adjacent "
          f"epoch pair.")
    p0 = pr.predict(sp.nest(theta))
    # epochs are ordered z = 0.4, 0.7, 1.0, 1.5, 2.0, so time runs BACKWARD
    # through the index; compare each epoch with the next-later one
    worst, n_neg, n_tot = 0.0, 0, 0
    for k in range(4, 0, -1):
        a, b = p0[:, k, :], p0[:, k - 1, :]
        ok = np.isfinite(a) & np.isfinite(b)
        d = (b - a)[ok]
        rel = d / np.maximum(np.abs(a[ok]), 1.0)
        worst = min(worst, float(rel.min()))
        n_neg += int((rel < -1e-12).sum())
        n_tot += int(ok.sum())
    print(f"    most negative relative change: {worst:.3e} over {n_tot} "
          f"galaxy-radius-pairs; {n_neg} fall by more than 1e-12")
    if n_neg:
        raise SystemExit(
            "REFUSING TO CONTINUE: the model DECREASES with time at A = 0, "
            "which contradicts exp54 Section 8.10. Either the analytic claim "
            "or this predictor is wrong; find out which.")
    print(f"    OK — exp54's monotone-in-time property holds exactly, and it "
          f"is now measured\n    rather than argued.")

    # ---------------------------------------------------------------- #
    print(f"\n  BRANCH 2 — with the expansion ON, how much decline is "
          f"REACHABLE?")
    print(f"    Everything else held at the incumbent; this is a capability "
          f"scan, not a fit.\n")
    print(f"    {'A':>6}{'oldest':>9}{'median':>10}{'declining':>11}"
          f"{'median of':>11}{'10th':>9}{'90th':>9}{'loss':>9}")
    print(f"    {'':>6}{'grows':>9}{'ALL [dex]':>10}{'fraction':>11}"
          f"{'decliners':>11}{'pct':>9}{'pct':>9}")
    rows = []
    for A in A_GRID:
        th = sp.nest(theta).copy()
        th[base.n_theta] = A
        S = decline_stats(pr.predict(th))
        loss = pr.loss(th)
        rows.append((A, S["all_median"], S["frac_declining"],
                     S["decliner_median"], S["p10"], S["p90"], loss))
        print(f"    {A:>6.2f}{1 + A:>8.2f}x{S['all_median']:>10.4f}"
              f"{100 * S['frac_declining']:>10.1f}{_P}"
              f"{S['decliner_median']:>11.4f}{S['p10']:>9.3f}"
              f"{S['p90']:>9.3f}{loss:>9.4f}")
    print(f"    {'DATA':>6}{'':>9}{T['all_median']:>10.4f}"
          f"{100 * T['frac_declining']:>10.1f}{_P}"
          f"{T['decliner_median']:>11.4f}{T['p10']:>9.3f}"
          f"{T['p90']:>9.3f}")

    R = np.array(rows)

    def a_matching(col, target):
        """The A at which a monotone column crosses its target, interpolated."""
        y = R[:, col]
        if target < min(y) or target > max(y):
            return np.nan
        o = np.argsort(y)
        return float(np.interp(target, y[o], R[o, 0]))

    a_med = a_matching(1, T["all_median"])
    a_frac = a_matching(2, T["frac_declining"])
    a_dec = a_matching(3, T["decliner_median"])
    print(f"\n  THE VERDICT OF THIS STAGE")
    print(f"    Expansion strength needed to reproduce each target, "
          f"one at a time:")
    print(f"      the median over ALL galaxies  ({T['all_median']:+.4f} dex)"
          f"   -> A = {a_med:.2f}")
    print(f"      the DECLINING fraction        "
          f"({100 * T['frac_declining']:.1f}{_P})"
          f"        -> A = {a_frac:.2f}")
    print(f"      the median among DECLINERS    "
          f"({T['decliner_median']:+.4f} dex)   -> A = {a_dec:.2f}")
    lo_a, hi_a = np.nanmin([a_med, a_frac, a_dec]), np.nanmax([a_med, a_frac,
                                                              a_dec])
    print(f"    They span A = {lo_a:.2f} to {hi_a:.2f}, a factor of "
          f"{hi_a / max(lo_a, 1e-9):.1f}.")
    if hi_a / max(lo_a, 1e-9) < 2.0:
        print(f"    That is a NARROW span: one global expansion strength can "
              f"come close to all three\n    at once, so the capability is "
              f"real and roughly self-consistent.")
    else:
        print(f"    That is a WIDE span: no single global strength does all "
              f"three, which points at\n    the population SPREAD rather "
              f"than the median (exp54 open question C4 — the\n    predicted "
              f"growth distribution is half as wide as the simulation's).")
    print(f"    The model's 10th-90th spread at A=0 is "
          f"{R[0, 4]:+.3f} to {R[0, 5]:+.3f} dex against the data's\n    "
          f"{T['p10']:+.3f} to {T['p90']:+.3f} — "
          f"{(R[0, 5] - R[0, 4]) / (T['p90'] - T['p10']):.2f}x the width. "
          f"Expansion shifts a distribution; it\n    does not widen one, so "
          f"that ratio is the ceiling on how well any single A can do.")
    print(f"\n    **That the capability exists says nothing about whether it "
          f"HELPS.** exp53 found\n    a mechanism real in a model's own "
          f"bookkeeping that was not the cause of its\n    error, with a "
          f"slice and a refit 57 points apart. Stage 3 fits it; Stages 2 and "
          f"4\n    judge it on the gates.")

    if not smoke:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        np.savez(OUT, A=R[:, 0], all_median=R[:, 1], frac=R[:, 2],
                 decliner_median=R[:, 3], p10=R[:, 4], p90=R[:, 5],
                 loss=R[:, 6],
                 truth_all_median=T["all_median"],
                 truth_frac=T["frac_declining"],
                 truth_decliner_median=T["decliner_median"],
                 truth_p10=T["p10"], truth_p90=T["p90"],
                 a_median=a_med, a_frac=a_frac, a_decliner=a_dec,
                 r_flag=F.R_GRID[I_FLAG], monotone_worst=worst)
        print(f"\n  wrote {OUT.name}")


if __name__ == "__main__":
    main("--smoke" in sys.argv)
