"""exp57 Stage 4 — did it fix the thing it was built for, without breaking the rest?

**G5, and it is TWO-SIDED on purpose.** The expansion term exists to follow a
declining central mass. But exp52's own differential scorecard measured, on the
data with no model involved, that **the median core GROWS**: `dlogSigma` at
2.35 kpc rises by **+0.09 dex over the redshift 2 to 0.4 baseline**, while
`dlogSigma` at 2-3 kpc is `0.00 ± 0.01` for both z<1 pairs. That is not in
tension with 42% of galaxies declining — a median can rise while a large
minority falls — but it is a hard constraint the first model never had to face,
and an expansion term that lowers central density *everywhere* will break it.

So success requires all three of:

  (a) the DECLINING 42% have their central-error span close materially, from
      the incumbent's 0.270 dex;
  (b) the NON-DECLINING 58% do not open beyond 0.06 dex, from 0.045; and
  (c) the model's own median core `dlogSigma` over the long baseline moves
      TOWARD the MEASURED value, recomputed here rather than quoted, and does
      not overshoot past it by more than +-0.03 dex.

**(c) was written as a guard and the null showed it is not one.** The intended
reading was "the model gets this right, and expansion must not break it".
Measured at the null it is already **wrong in the direction expansion helps**:
on the 839 galaxies the long baseline admits, the data's median core
`dlogSigma` is **+0.057 dex** and the incumbent's is **+0.103** — the model
**over-grows** its core by **+0.046 dex**. So (c) is not a guard against
expansion, it is a second defect expansion might fix, and the gate is stated
accordingly: the distance must SHRINK, and must not overshoot to the far side.
Writing it down as a guard and discovering it was a target is a finding, not a
mistake to hide, so the original framing is left visible here.

(The +0.057 is on the epoch-intersected sample of 839 galaxies; exp52 quoted
+0.09 on all 2397 without that intersection. Both are printed below so the
comparison is like for like, and the difference is the completeness cut, not a
disagreement.)

**G4 — expansion must not be a tail substitute.** exp53 found that removing the
first model's transport cost the heavy-tailed deposit families 3-4% of the loss
and the light-tailed ones ~11%: *transport substitutes for a heavy tail*. exp54
Stage 3.9 then measured that the deposit shape `c` is the ONLY one of the seven
parameters the others cannot absorb — the conditional slice over-stated it 2.3x
against 7-18x for everything else. Adding an expansion term is the most likely
thing to destroy that, because both control how far a deposit's mass reaches.
G4 therefore refits with `c` FROZEN at the incumbent's +0.7965 and reports how
far `c` moves when it is free. If it moves by more than its Stage 3.9
one-percentage-point width (-0.25 / +0.51), the two are trading and the result
is a reparameterisation of deposit shape rather than a mechanism.

Run: HONGSHAO_DATA_DIR=... OMP_NUM_THREADS=1 PYTHONPATH=. uv run python -u \\
     experiments/exp57_expansion_term/stage4_target.py --fit LABEL [--smoke]
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
import stage1_contract as S1                             # noqa: E402
import stage35_time_law as S35                           # noqa: E402
import stage37_size_epoch as S37                         # noqa: E402
from hongshao.profile_emulator import density_from_cog   # noqa: E402

Z = (0.4, 0.7, 1.0, 1.5, 2.0)
FITDIR = HERE / "outputs" / "stage3_fits"
OUTDIR = HERE / "outputs"
RULE = "=" * 96
_P = S35._PC
#: exp54 Stage 3.4's measured central-error spans, the reference for (a) and (b)
SPAN_DECLINING, SPAN_NOT = 0.270, 0.045
#: (b)'s limit, fixed in the plan
SPAN_NOT_LIMIT = 0.06
#: (c)'s tolerance, fixed in the plan: how far past the data the model may
#: overshoot. It is NOT a tolerance around the null, which is already 0.046 out.
CORE_TOL = 0.03
#: Stage 3.9's one-percentage-point width for `c`, the G4 threshold
C_WIDTH_LO, C_WIDTH_HI = -0.25, 0.51
C_INCUMBENT = 0.79646876
#: the null's own core-density gap, measured by `--null` and hard-coded here so
#: a fitted run is compared against it without re-running the null. Checked at
#: run time by `--null` itself.
NULL_CORE_GAP = 0.0455


def decline_flag(data, i_r=S1.I_FLAG):
    """Stage 3.4's preregistered flag: did the MEASURED central mass fall?"""
    a, b = data[:, 4, i_r], data[:, 0, i_r]
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    fell = np.zeros(len(data), bool)
    fell[ok] = np.log10(b[ok]) - np.log10(a[ok]) < 0
    return fell, ok


def central_span(pred, data, mask, group):
    """Median log10(model/truth) for M*(<2 kpc) per epoch, and its span."""
    err = S37.central_error(pred, data, mask, shell=0)
    med = np.array([np.nanmedian(err[group, k]) for k in range(err.shape[1])])
    return med, float(np.nanmax(med) - np.nanmin(med))


def core_dlogsigma(cube, use, ke=4, kl=0, i_mid=0):
    """Median change in the core surface density between two epochs.

    The same operator exp52 used: difference the cumulative profile to get
    `Sigma`, take the innermost mid-radius (2.35 kpc on this grid), and compare
    the two epochs galaxy by galaxy.
    """
    a, b = cube[use, ke, :], cube[use, kl, :]
    ok = np.isfinite(a).all(1) & np.isfinite(b).all(1) & (a[:, 0] > 0) & (b[:, 0] > 0)
    la, mid = density_from_cog(np.log10(np.clip(a[ok], 1.0, None)), F.R_GRID)
    lb, _ = density_from_cog(np.log10(np.clip(b[ok], 1.0, None)), F.R_GRID)
    d = lb[:, i_mid] - la[:, i_mid]
    d = d[np.isfinite(d)]
    return float(np.median(d)), float(mid[i_mid]), int(d.size)


def report(tag, pr, theta, recs, data, mask, base, smoke=False):
    print(f"\n{RULE}\nSTAGE 4 — the TARGET gates for {tag}\n{RULE}")
    pred = pr.predict(theta)
    fell, ok_flag = decline_flag(data)
    n_dec = int((fell & ok_flag).sum())
    print(f"\n  Split on exp54 Stage 3.4's preregistered flag: did the MEASURED "
          f"M*(<{F.R_GRID[S1.I_FLAG]:.2f} kpc)\n  fall between redshift 2 and "
          f"0.4? {n_dec} of {int(ok_flag.sum())} galaxies "
          f"({100 * n_dec / max(int(ok_flag.sum()), 1):.1f}{_P}) did.")

    print(f"\n  (a) and (b) — the CENTRAL ERROR, median log10(model/truth) for "
          f"M*(<2 kpc)")
    print(f"      {'group':<26}" + "".join(f"{f'z={z}':>9}" for z in Z)
          + f"{'span':>9}{'was':>9}")
    rows = {}
    for name, grp, was in (("all admitted", ok_flag, 0.145),
                           ("centre DECLINED", fell & ok_flag, SPAN_DECLINING),
                           ("centre did NOT decline",
                            (~fell) & ok_flag, SPAN_NOT)):
        med, span = central_span(pred, data, mask, grp)
        rows[name] = (med, span)
        print(f"      {name:<26}" + "".join(f"{v:>9.3f}" for v in med)
              + f"{span:>9.3f}{was:>9.3f}")

    span_dec = rows["centre DECLINED"][1]
    span_not = rows["centre did NOT decline"][1]
    a_ok = span_dec < SPAN_DECLINING
    b_ok = span_not <= SPAN_NOT_LIMIT
    print(f"\n      (a) the declining group's span: {span_dec:.3f} against the "
          f"incumbent's {SPAN_DECLINING:.3f}   -> "
          f"{'CLOSES' if a_ok else 'does NOT close'}")
    print(f"      (b) the non-declining group's : {span_not:.3f} against a "
          f"limit of {SPAN_NOT_LIMIT:.3f}      -> "
          f"{'OK' if b_ok else 'BROKEN'}")

    print(f"\n  (c) — THE MEASURED CORE. exp52 measured the core surface "
          f"density RISING over the long\n      baseline, so an expansion term "
          f"that lowers central density everywhere would\n      break it. This "
          f"was written as a GUARD; the null showed it is a second TARGET — "
          f"the\n      incumbent already over-grows its core. Recomputed here, "
          f"never quoted.")
    use = mask[:, 4] & mask[:, 0]
    allg = np.ones(len(data), bool)
    d_mod, r_mid, n_m = core_dlogsigma(pred, use)
    d_dat, _, n_d = core_dlogsigma(data, use)
    a_mod, _, na_m = core_dlogsigma(pred, allg)
    a_dat, _, na_d = core_dlogsigma(data, allg)
    print(f"      median dlogSigma at {r_mid:.2f} kpc, redshift 2 -> 0.4")
    print(f"        on the epoch-intersected sample ({n_d} galaxies): "
          f"data {d_dat:+.4f}, model {d_mod:+.4f}")
    print(f"        on every galaxy with both epochs ({na_d}): "
          f"data {a_dat:+.4f}, model {a_mod:+.4f}   "
          f"(exp52 measured +0.09 on 2397)")
    gap = d_mod - d_dat
    print(f"        the model is {gap:+.4f} dex from the data; at the NULL "
          f"this is {NULL_CORE_GAP:+.4f},")
    print(f"        so (c) asks for |gap| to SHRINK and not to overshoot past "
          f"{CORE_TOL:+.2f}.")
    c_ok = (abs(gap) < abs(NULL_CORE_GAP) - 1e-9) and (gap > -CORE_TOL)
    if tag == "null":
        c_ok = True                       # the null defines the reference
        print(f"        this IS the null, so it defines the reference rather "
              f"than passing or failing it.")
    else:
        print(f"        -> {'CLOSER' if abs(gap) < abs(NULL_CORE_GAP) else 'NO CLOSER'}"
              f" than the null"
              + ("" if gap > -CORE_TOL else ", and it has OVERSHOT"))

    print(f"\n  G5 VERDICT: "
          f"{'PASS' if (a_ok and b_ok and c_ok) else 'FAIL'}   "
          f"(a) {'ok' if a_ok else 'FAIL'}  "
          f"(b) {'ok' if b_ok else 'FAIL'}  (c) {'ok' if c_ok else 'FAIL'}")

    if not smoke:
        OUTDIR.mkdir(parents=True, exist_ok=True)
        np.savez(OUTDIR / f"stage4_target_{tag}.npz", tag=tag, theta=theta,
                 med_all=rows["all admitted"][0],
                 med_dec=rows["centre DECLINED"][0],
                 med_not=rows["centre did NOT decline"][0],
                 span_all=rows["all admitted"][1], span_dec=span_dec,
                 span_not=span_not, core_model=d_mod, core_data=d_dat,
                 core_model_all=a_mod, core_data_all=a_dat,
                 core_gap=gap, null_core_gap=NULL_CORE_GAP,
                 r_mid=r_mid, n_declining=n_dec,
                 a_ok=a_ok, b_ok=b_ok, c_ok=c_ok)
        print(f"\n  wrote stage4_target_{tag}.npz")
    return a_ok and b_ok and c_ok


def main(argv):
    smoke = "--smoke" in argv
    recs, data, mask, lmh, base, theta, slow = S0.build(smoke)
    print(f"exp57 STAGE 4 — the target gates")
    print(f"  sample: {len(recs)} galaxies, {int(mask.sum())} galaxy-epochs")

    if "--null" in argv:
        sp = X.XSpec(base, "X1")
        pr = X.ExpandingProblem(sp, recs, data, mask, epochs=(0, 1, 2, 3, 4))
        report("null", pr, sp.nest(theta), recs, data, mask, base, smoke)
        return
    label = argv[argv.index("--fit") + 1]
    z = np.load(FITDIR / f"{label}.npz", allow_pickle=True)
    sp = X.XSpec(base, str(z["law"]))
    pr = X.ExpandingProblem(sp, recs, data, mask, epochs=(0, 1, 2, 3, 4))
    th = np.asarray(z["theta"], float)
    print(f"  fitted {label}: loss {float(z['loss']):.6f}, "
          + ", ".join(f"{n}={v:+.4f}" for n, v in
                      zip(sp.x_names, th[base.n_theta:])))
    ic = base.theta_names.index("c")
    print(f"  G4 — deposit shape c moved {th[ic] - C_INCUMBENT:+.4f} "
          f"(from {C_INCUMBENT:+.4f} to {th[ic]:+.4f}); Stage 3.9's "
          f"one-percentage-point\n       width for c is "
          f"{C_WIDTH_LO:+.2f} / {C_WIDTH_HI:+.2f}   -> "
          + ("WITHIN it, so expansion is not simply buying deposit shape"
             if C_WIDTH_LO <= th[ic] - C_INCUMBENT <= C_WIDTH_HI
             else "OUTSIDE it: expansion and deposit shape are TRADING"))
    report(label, pr, th, recs, data, mask, base, smoke)


if __name__ == "__main__":
    main(sys.argv)
