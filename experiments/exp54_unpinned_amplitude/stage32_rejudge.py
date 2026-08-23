"""exp54 Stage 3.2 — RE-RUN after the M200c unit fix and the SIGMA_A correction.

TWO INPUTS CHANGED, so the whole 45-cell factorial is re-run rather than
patched. Nothing here is merged with the old cache; the cached values are shown
alongside only so the effect of each correction is visible.

1. **The concentration excess** ``dlogc = log10[c200c / c_Diemer19(M, z)]`` --
   how much more concentrated a halo is than typical for its mass and redshift.
   The Diemer19 reference was evaluated at a mass 2.0006 dex too low
   (commit `dc226b0`). Three model terms read it:

     * ``E3``   adds ``k * dlogc`` to the efficiency        -- 9 cells
     * ``S2a``  sets the deposit scale to ``R200c / c_eff`` -- 15 cells
     * ``S3``   adds ``dlogc`` to the size law             -- 15 cells

   overlapping in 6, so **33 of 45 cells** read it and only the twelve
   ``S2`` cells with a non-E3 efficiency do not.

   **THE FIRST VERSION OF THIS SCRIPT GOT THAT WRONG.** It classified only E3
   and S3 as affected -- 21 cells -- because `model.r50_of` reads
   ``scale = halo.r200c if spec.size == "S2" else halo.r200c / halo.c_eff``,
   and the ``else`` branch belongs to ``S2a``, which was overlooked. The merge
   control caught it within three cells: the control cell ``moffat-E5-S2a`` was
   asserted to be unaffected and came back with ``loss 1.7476`` against a
   cached ``1.7385``. **That is the whole reason a control was built**, and it
   is why the classification is now expressed as a function of what the code
   actually reads rather than as a hand-written list.

2. **`SIGMA_A`, the benchmark every `score_A` divides by**, was inflated
   8-21% at z >= 0.7 by ONE galaxy (row 181) whose measured
   ``M*(<100 kpc)`` collapses from 10^11.71 at z=0.4 to a flat ~10^8 across all
   four higher-redshift epochs -- a broken cross-match, not a progenitor. That
   galaxy is in Stage 1's 2397 but in NONE of the model-fitting subsamples
   (Stage 3.1's 300, 3.2's 240, 3.3's 1199), so the benchmark was charged for
   it while the model never was. `scoreboard.py` now rejects such histories and
   `SIGMA_A` is restated. Because `SIGMA_A` enters `Problem.loss`, the fitted
   theta of EVERY cell changes and none of the old numbers can be reused.

``stratified`` and the multi-start `fit` are both seeded, so the twelve
unaffected-by-(1) cells would have reproduced exactly under the old `SIGMA_A`;
they will not now, and that is expected rather than alarming.

Run: HONGSHAO_DATA_DIR=... PYTHONPATH=. uv run python \\
     experiments/exp54_unpinned_amplitude/stage32_rejudge.py [--n 240] [--smoke]
"""
from __future__ import annotations

import itertools
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
import stage31 as S31                                    # noqa: E402
import stage32 as S32                                    # noqa: E402

POP = ROOT / "experiments/exp32_full_population/outputs/population.npz"
CACHED = HERE / "outputs" / "stage32.npz"
OUT = HERE / "outputs" / "stage32_rejudged.npz"
#: per-cell checkpoint. A 45-cell factorial is ~2 hours; a kill at cell 11 threw
#: away 29 minutes of fits once, so every completed cell is written immediately
#: and a restart skips what is already done. Delete it to force a clean re-run.
PARTIAL = HERE / "outputs" / "stage32_rejudged_partial.npz"
RULE = "=" * 108

#: cells that read NO concentration information -- the twelve S2 cells with a
#: non-E3 efficiency. Derived from what `model.py` reads, not hand-listed.
def reads_concentration(en, sn):
    """True if this cell consumes `dlogc` or `c_eff`, and is therefore changed
    by the M200c unit fix. `model.log_eps` adds `dlogc` when `e3`;
    `model.r50_of` uses `r200c / c_eff` for every size except exactly "S2", and
    adds `dlogc` again when `conditioning` (the "+S3" suffix)."""
    return ("E3" in en) or (sn != "S2")


def cell_label(en, fam, sn):
    return M.Spec(family=fam, **dict(S32.EFFS)[en], **dict(S32.SIZES)[sn]).label





def run_cell(en, fam, sn, recs, data, lmh_s):
    """One factorial cell: fit, then the seven judge criteria."""
    sp = M.Spec(family=fam, **dict(S32.EFFS)[en], **dict(S32.SIZES)[sn])
    pr = F.Problem(sp, recs, data, epochs=(0, 4))
    th, lo = F.fit(pr, maxiter=600 * sp.n_theta, verbose=False)
    sa, sf = pr.per_epoch(th)
    J, _ = F.jacobian(pr, th)
    s = np.linalg.svd(J, compute_uv=False)
    r148, good = pr.diagnostics(th)
    return dict(label=sp.label, eff=en, fam=fam, size=sn, npar=sp.n_theta,
                loss=lo, theta=th, sA0=sa[0], sA4=sa[1], sF0=sf[0], sF4=sf[1],
                ident=float(s[-1] / s[0]), over=float(np.median(r148)),
                tilt=float(np.polyfit(lmh_s[good], r148, 1)[0]))


def main(n_total=240, smoke=False):
    cache = np.load(CACHED, allow_pickle=True) if CACHED.exists() else None
    keys = ("loss", "sA0", "sA4", "sF0", "sF4", "ident", "over", "tilt", "npar")
    old = ({str(l): {k: float(cache[k][i]) for k in keys}
            for i, l in enumerate(cache["labels"])} if cache is not None else {})
    old_pos = ({str(cache["labels"][oi]): i + 1
                for i, oi in enumerate(np.argsort(cache["mean_rank"]))}
               if cache is not None else {})

    import stage2_multiepoch as s2
    s2._w_init(None)
    pop = np.load(POP)
    all_rows = np.array([g["row"] for g in s2._W["gals"]])
    lmh = pop["logmh_zk_diffmah"][all_rows][:, 0]
    rows = S31.stratified(all_rows, lmh, n_total)
    recs = H.build_records(rows=rows)
    sel = np.array([h.row for h in recs])
    data = pop["data"][sel]
    lmh_s = pop["logmh_zk_diffmah"][sel][:, 0]

    cells = list(itertools.product([e[0] for e in S32.EFFS], S32.FAMS,
                                   [s[0] for s in S32.SIZES]))
    n_conc = sum(reads_concentration(c[0], c[2]) for c in cells)
    if smoke:
        cells = [c for c in cells if reads_concentration(c[0], c[2])][:1] + \
                [c for c in cells if not reads_concentration(c[0], c[2])][:1]

    print(f"exp54 STAGE 3.2 RE-RUN   ({len(cells)} cells, n={len(recs)} "
          f"galaxies, z=0.4 and z=2.0)")
    print(f"  SIGMA_A in use: " + " ".join(f"{v:.4f}" for v in F.SIGMA_A))
    print(f"  {n_conc} of 45 cells read the concentration excess (E3, S2a or "
          f"S3); all 45 are re-run\n  because SIGMA_A changed too, and SIGMA_A "
          f"enters the loss.\n")

    done = {}
    if PARTIAL.exists() and not smoke:
        z = np.load(PARTIAL, allow_pickle=True)
        done = {str(k): v.item() for k, v in z.items()}
        if done:
            print(f"  resuming: {len(done)} of {len(cells)} cells already "
                  f"complete in {PARTIAL.name}\n")

    results, t0 = [], time.time()
    for i, (en, fam, sn) in enumerate(cells):
        lab = cell_label(en, fam, sn)
        tag = "conc" if reads_concentration(en, sn) else "    "
        if lab in done:
            r = done[lab]
            results.append(r)
            print(f"  [{i + 1:2d}/{len(cells)}] {tag} {lab:<26} "
                  f"loss={r['loss']:7.4f}  (cached)", flush=True)
            continue
        try:
            r = run_cell(en, fam, sn, recs, data, lmh_s)
        except Exception as e:                    # a cell may be degenerate
            print(f"  [{i + 1:2d}/{len(cells)}] {tag} {lab:<26} FAILED: "
                  f"{type(e).__name__}", flush=True)
            results.append(dict(label=lab, eff=en, fam=fam, size=sn,
                                loss=np.nan, npar=M.Spec(
                                    family=fam, **dict(S32.EFFS)[en],
                                    **dict(S32.SIZES)[sn]).n_theta))
            continue
        results.append(r)
        if not smoke:
            done[lab] = r
            np.savez(PARTIAL, **done)      # checkpoint after EVERY cell
        was = old.get(lab, {}).get("loss", np.nan)
        print(f"  [{i + 1:2d}/{len(cells)}] {tag} {lab:<26} p={r['npar']:<3d} "
              f"loss={r['loss']:7.4f} (was {was:7.4f}, d={r['loss'] - was:+.5f})"
              f"  ident={r['ident']:.1e}  tilt={r['tilt']:+.4f}", flush=True)
    print(f"\n  factorial took {(time.time() - t0) / 60:.1f} min")
    if smoke:
        print("  smoke run: writing nothing")
        return None

    good, rk = S32.judge(results)
    mean_rank = rk.mean(1)
    order = np.argsort(mean_rank)
    print(f"\n{RULE}\nTHE JUDGE, RE-RUN — mean rank over seven criteria "
          f"(lower is better)\n{RULE}")
    print(f"  {'rank':>4} {'was':>4} {'model':<26}{'mean':>7}{'sA(0.4)':>9}"
          f"{'sA(2.0)':>9}{'sF(0.4)':>9}{'sF(2.0)':>9}{'ident':>9}"
          f"{'over':>8}{'tilt':>9}")
    for i, oi in enumerate(order[:15]):
        r = good[oi]
        print(f"  {i + 1:>4} {old_pos.get(r['label'], 0):>4} {r['label']:<26}"
              f"{mean_rank[oi]:>7.2f}{r['sA0']:>9.3f}{r['sA4']:>9.3f}"
              f"{r['sF0']:>9.3f}{r['sF4']:>9.3f}{r['ident']:>9.1e}"
              f"{r['over']:>+8.3f}{r['tilt']:>+9.4f}")
    print(f"\n  ... and the WORST three:")
    for oi in order[-3:]:
        print(f"       {good[oi]['label']:<26}{mean_rank[oi]:>7.2f}")

    print(f"\n  BIGGEST RANK MOVES against the pre-correction ranking "
          f"(a cell entering the top six changes what Stage 3.4 may consider):")
    moves = sorted(((old_pos.get(good[oi]["label"], 99) - (i + 1)),
                    good[oi]["label"], old_pos.get(good[oi]["label"], 99), i + 1)
                   for i, oi in enumerate(order))
    for d, lab, was, now in moves[:4] + moves[-4:]:
        print(f"       {lab:<26} {was:>3} -> {now:>3}  ({d:+d})")

    lo_order = np.argsort([r["loss"] for r in good])
    print(f"\n  If we had ranked by LOSS the top three would be:")
    for oi in lo_order[:3]:
        print(f"       {good[oi]['label']:<26} loss={good[oi]['loss']:.4f}  "
              f"judge rank {int(np.where(order == oi)[0][0]) + 1}")
    print(f"\n  SHORT LIST for Stage 3.3: {[good[i]['label'] for i in order[:4]]}")

    np.savez_compressed(
        OUT, labels=np.array([r["label"] for r in good]), mean_rank=mean_rank,
        sigma_a=F.SIGMA_A,
        reads_conc=np.array([reads_concentration(r["eff"], r["size"])
                             for r in good]),
        **{f"theta::{r['label']}": r["theta"] for r in good if "theta" in r},
        **{k: np.array([r[k] for r in good]) for k in keys})
    print(f"  wrote {OUT}")
    PARTIAL.unlink(missing_ok=True)        # the full result supersedes it
    return good, order


if __name__ == "__main__":
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 240
    main(n, smoke="--smoke" in sys.argv)
