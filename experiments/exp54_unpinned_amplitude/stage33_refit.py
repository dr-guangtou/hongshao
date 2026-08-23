"""exp54 Stage 3.3, RE-RUN under the corrected benchmark, on two samples.

WHY THIS EXISTS. Stage 3.3 fitted its models against `SIGMA_A`, and `SIGMA_A`
was wrong: inflated 8-18% at z >= 0.7 by one galaxy with a broken cross-match
(see `fit.py`, commit `91a26f3`). `SIGMA_A` enters `Problem.loss` through
`score_A`, so it is not merely a reporting constant -- it sets how much the
amplitude term is weighted against the shape term, epoch by epoch. Correcting
it weights the amplitude MORE at exactly the epochs where the surviving defect
lives, so the fitted parameters must be recomputed, not rescaled.

THE SECOND SAMPLE, and the question it answers. `selection.py` showed that our
high-redshift galaxies are a progenitor-selected minority below
`log10 M200c ~ 12.8-13.0`, and that the z=2 halo-mass tilt is an artifact of
that regime. The parameters were nevertheless fitted with those galaxies
included. So: **do the adopted parameters depend on the biased regime?**

The MH-COMPLETE sample is the 751 galaxies that clear their epoch's mh-complete cut at
ALL FIVE epochs. It is not a subsample of the published 1199; it is drawn from
the whole 2397, because 751 is already small and there is no reason to halve it.

**THE INTERSECTION IS ASSEMBLY-BIASED, AND THAT LIMITS WHAT IT CAN BE USED
FOR.** Requiring a halo to be above the cut at EVERY epoch selects haloes that
were already massive at z = 2, i.e. early assemblers. Measured at fixed z = 0.4
halo mass (a 13.2-13.6 slice, n = 987): the mh-complete galaxies have
`t50` = 4.14 Gyr against 5.59 for the rest -- 1.45 Gyr earlier, Mann-Whitney
p = 2e-53 -- and `fz2` = 0.350 against 0.175, twice the mass in place by z = 2,
p = 1e-83. So it removes the progenitor-selection bias and introduces an
assembly bias in its place.

What that does and does not invalidate:

  * **The robustness answer is VALID and if anything conservative.** It asks
    whether the FITTED PARAMETERS move, by scoring `theta_complete` on the full
    sample. The second sample now differs from the first in two ways rather
    than one, so parameters that still transfer have passed a harder test.
  * **The `score_A` values ON the mh-complete sample must NOT be read as "how the
    model does on a complete sample."** They are measured on an early-forming
    massive subset, which is a different population, and the model is known to
    behave differently there.

The fix is a per-epoch galaxy mask, so each epoch is scored on its own mh-complete
galaxies and no intersection is taken -- 2397+1781+1436+1145+840 galaxy-epochs
instead of 751x5, and no assembly bias. That is `stage33_perepoch.py`.

WHAT IS REPORTED. For each model on the short list, both fits, and then the
2x2 CROSS-EVALUATION: each theta scored on both samples. That last table is
the actual answer. If `theta_complete` scores the full sample about as well as
`theta_full` does, the biased regime was not driving the fit. If it does not,
the published parameters are partly a description of the selection.

Run: HONGSHAO_DATA_DIR=... PYTHONPATH=. uv run python \\
     experiments/exp54_unpinned_amplitude/stage33_refit.py [--top 3] [--smoke]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink", HERE):
    sys.path.insert(0, str(p))

import fit as F                                          # noqa: E402
import halo as H                                         # noqa: E402
import model as M                                        # noqa: E402
import stage33 as S33                                    # noqa: E402

POP = ROOT / "experiments/exp32_full_population/outputs/population.npz"
S32R = HERE / "outputs" / "stage32_rejudged.npz"
SEL = HERE / "outputs" / "selection.npz"
OUT = HERE / "outputs" / "stage33_refit.npz"
#: per-fit checkpoint; a 5-hour run must not be an all-or-nothing bet
PARTIAL = HERE / "outputs" / "stage33_refit_partial.npz"

Z = S33.Z if hasattr(S33, "Z") else (0.4, 0.7, 1.0, 1.5, 2.0)
MONO_FLOOR = S33.MONO_FLOOR
RULE = "=" * 96


def mh_complete_rows(all_rows):
    """Rows that clear their epoch's mh-complete cut at ALL five epochs."""
    import selection as S
    if not SEL.exists():
        raise SystemExit("run selection.py first — the cuts come from there")
    cuts = np.load(SEL, allow_pickle=True)["cuts"]
    m = S.sample_masses(np.load(S.HS_NPZ, allow_pickle=True))
    keep = (np.isfinite(m) & (m >= cuts[None, :])).all(axis=1)
    return all_rows[keep[all_rows]], cuts


def fit_one(sp, recs, data, tag):
    """Two starts, as Stage 3.3 used: five epochs on a large sample is ~10x the
    factorial's cost per evaluation, and Stage 3.2 showed S2 cells converge from
    any sane start."""
    pr = F.Problem(sp, recs, data, epochs=(0, 1, 2, 3, 4))
    base = M.default_theta(sp)
    rng = np.random.default_rng(1)
    j = np.clip(base + rng.normal(0, 0.12, base.shape)
                * np.maximum(np.abs(base), 0.3), *np.array(sp.bounds()).T)
    t0 = time.time()
    th, lo = F.fit(pr, starts=[base, j], maxiter=600 * sp.n_theta, verbose=False)
    print(f"    [{tag}] fitted in {(time.time() - t0) / 60:.1f} min, "
          f"loss {lo:.5f}", flush=True)
    return th, lo, pr


def score_on(sp, theta, recs, data):
    """(score_A per epoch, score_F per epoch, growth bias) of one theta on one
    galaxy set. Nothing is refitted."""
    pr = F.Problem(sp, recs, data, epochs=(0, 1, 2, 3, 4))
    sa, sf = pr.per_epoch(theta)
    pred = pr.predict(theta)
    _, gbias, gscat = S33.growth_report(pred, data, [0, 1, 2, 3, 4])
    return sa, sf, gbias, gscat, pred


def main(top=3, smoke=False):
    import stage2_multiepoch as s2
    s2._w_init(None)
    pop = np.load(POP)
    all_rows = np.array([g["row"] for g in s2._W["gals"]])

    if not S32R.exists():
        raise SystemExit("run stage32_rejudge.py first")
    s32 = np.load(S32R, allow_pickle=True)
    short = [str(s32["labels"][i])
             for i in np.argsort(s32["mean_rank"])[:top]]

    rows_full = all_rows[::2]                       # as published: every 2nd
    rows_complete, cuts = mh_complete_rows(all_rows)
    if smoke:
        rows_full, rows_complete, short = rows_full[::12], rows_complete[::8], short[:1]

    print(f"exp54 STAGE 3.3 RE-RUN — corrected SIGMA_A, two samples")
    print(f"  SIGMA_A: " + " ".join(f"{v:.4f}" for v in F.SIGMA_A)
          + "   (was 0.1047 0.1359 0.1417 0.1456 0.1744)")
    print(f"  short list from the re-judged factorial: {short}")
    print(f"  FULL sample: {len(rows_full)} galaxies (every 2nd, as published)")
    print(f"  MH-COMPLETE sample: {len(rows_complete)} galaxies — above "
          + "/".join(f"{c:.2f}" for c in cuts) + " log10 M200c at all "
          f"five epochs\n")

    recs = {}
    for tag, rr in (("full", rows_full), ("complete", rows_complete)):
        r = H.build_records(rows=rr, verbose=False)
        sel = np.array([h.row for h in r])
        recs[tag] = (r, pop["data"][sel], pop["logmh_zk_diffmah"][sel], sel)
        print(f"  built {len(r)} halo records for the {tag} sample")

    done = {}
    if PARTIAL.exists() and not smoke:
        z = np.load(PARTIAL, allow_pickle=True)
        done = {str(k): v.item() for k, v in z.items()}
        if done:
            print(f"  resuming: {len(done)} fits already complete\n")

    out = {}
    for label in short:
        sp = S33.spec_from_label(label)
        print(f"\n{RULE}\n{label}   {sp.n_theta} parameters\n{RULE}")
        theta = {}
        for tag in ("full", "mh-complete"):
            key = f"{label}::{tag}"
            if key in done:
                theta[tag] = done[key]["theta"]
                print(f"    [{tag}] loss {done[key]['loss']:.5f}  (cached)")
                continue
            r, d, _, _ = recs[tag]
            th, lo, _ = fit_one(sp, r, d, tag)
            theta[tag] = th
            if not smoke:
                done[key] = dict(theta=th, loss=lo)
                np.savez(PARTIAL, **done)

        print(f"\n    PARAMETERS — fitted on each sample")
        names = sp.theta_names
        print(f"      {'parameter':<12}{'full':>12}{'complete':>12}{'change':>12}")
        for i, nm in enumerate(names):
            a, b = theta["full"][i], theta["complete"][i]
            print(f"      {nm:<12}{a:>12.4f}{b:>12.4f}{b - a:>+12.4f}")

        print(f"\n    CROSS-EVALUATION — each theta scored on each sample; "
              f"nothing refitted")
        print(f"      {'theta':<6}{'scored on':<11}" + "".join(
            f"{f'z={z}':>9}" for z in Z) + f"{'growth':>10}")
        cross = {}
        for ta in ("full", "mh-complete"):
            for tb in ("full", "mh-complete"):
                r, d, _, _ = recs[tb]
                sa, sf, gb, gs, _ = score_on(sp, theta[ta], r, d)
                cross[f"{ta}->{tb}"] = dict(sA=sa, sF=sf, gbias=gb, gscat=gs)
                print(f"      {ta:<6}{tb:<11}" + "".join(f"{v:>9.4f}" for v in sa)
                      + f"{gb:>+10.4f}   score_A")
                print(f"      {'':<6}{'':<11}" + "".join(f"{v:>9.4f}" for v in sf)
                      + f"{'':>10}   score_F")

        # the answer, stated as one number per epoch
        pen = cross["complete->full"]["sA"] - cross["full->full"]["sA"]
        print(f"\n    THE ROBUSTNESS ANSWER: score_A cost of using the "
              f"FAIR-fitted theta\n    on the FULL sample, per epoch "
              f"(0 = the biased regime did not drive the fit):")
        print(f"      " + "".join(f"{v:>+9.4f}" for v in pen))

        r, d, lmh, _ = recs["full"]
        _, _, _, _, pred = score_on(sp, theta["full"], r, d)
        tilts = S33.tilt_report(pred, d, [0, 1, 2, 3, 4], lmh)
        print(f"\n    HALO-MASS TILT of the FULL-fitted theta on the full "
              f"sample\n    (compare `selection.py`: the z=2 column is "
              f"progenitor-selected)")
        print(f"      {'R [kpc]':>9}"
              + "".join(f"{f'z={z}':>22}" for z in Z))
        for rr, rowset in tilts.items():
            cells = [f"{v:+.3f} [{a:+.3f},{b:+.3f}]" for v, a, b in rowset]
            print(f"      {rr:>9.0f}" + "".join(f"{c:>22}" for c in cells))

        out[label] = dict(theta_full=theta["full"], theta_complete=theta["complete"],
                          cross=cross, penalty=pen)

    if smoke:
        print("\n  smoke run: writing nothing")
        return out

    np.savez_compressed(
        OUT, labels=np.array(short), rows_full=rows_full, rows_complete=rows_complete,
        sigma_a=F.SIGMA_A, cuts=cuts,
        **{f"{l}_theta_full": v["theta_full"] for l, v in out.items()},
        **{f"{l}_theta_complete": v["theta_complete"] for l, v in out.items()},
        **{f"{l}_penalty": v["penalty"] for l, v in out.items()},
        **{f"{l}_{k}_{m}": v["cross"][k][m]
           for l, v in out.items() for k in v["cross"] for m in ("sA", "sF")})
    print(f"\n  wrote {OUT}")
    PARTIAL.unlink(missing_ok=True)
    return out


if __name__ == "__main__":
    t = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 3
    main(t, smoke="--smoke" in sys.argv)
