"""exp54 Stage 3.2 — the controlled factorial over the model axes.

Every combination of

    efficiency  E1 | E2 | E1+E3 | E4 | E5
    profile     sersic | gompertz_log | moffat
    size        S2 | S2a | S2+S3

fitted on the SAME small mass-stratified sample at z=0.4 and z=2.0, then
**ranked by a judge, never by loss** (the standing project rule). A form that
wins a single epoch can fail the consistency problem, which is why every cell
is scored at both epochs separately.

The judge's seven criteria, each ranked independently and combined by mean
rank, so no single number can dominate:

    1-2  amplitude accuracy at z=0.4 and at z=2.0
    3-4  normalized-shape accuracy at z=0.4 and at z=2.0
    5    identifiability: the smallest singular value of the scaled Jacobian
    6    over-reach: |median log(model/truth)| at the measured horizon
    7    the halo-mass TILT of that residual -- a held-out slope, since exp53
         showed deposition-only models tilt with halo mass where the incumbent
         is flat, and the size law must now supply what transport used to

Run: HONGSHAO_DATA_DIR=... PYTHONPATH=. uv run python \\
     experiments/exp54_unpinned_amplitude/stage32.py [--n 240]
"""
from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink"):
    sys.path.insert(0, str(p))

import fit as F                                          # noqa: E402
import halo as H                                         # noqa: E402
import model as M                                        # noqa: E402
import stage31 as S31                                    # noqa: E402

POP = ROOT / "experiments/exp32_full_population/outputs/population.npz"
OUT = HERE / "outputs" / "stage32.npz"
RULE = "=" * 108

EFFS = [("E1", dict(eff="E1")), ("E2", dict(eff="E2")),
        ("E1+E3", dict(eff="E1", e3=True)), ("E4", dict(eff="E4")),
        ("E5", dict(eff="E5"))]
FAMS = ["sersic", "gompertz_log", "moffat"]
SIZES = [("S2", dict(size="S2")), ("S2a", dict(size="S2a")),
         ("S2+S3", dict(size="S2", conditioning=True))]


def judge(rows):
    """Mean rank over the seven criteria; lower is better. Returns (order, rk)."""
    keys = ["sA0", "sA4", "sF0", "sF4", "ident", "over", "tilt"]
    good = [r for r in rows if np.isfinite(r["loss"])]
    rk = np.zeros((len(good), len(keys)))
    for j, k in enumerate(keys):
        v = np.array([r[k] for r in good], float)
        if k == "ident":
            v = -v                                   # bigger is better
        else:
            v = np.abs(v)
        v = np.where(np.isfinite(v), v, np.inf)
        rk[:, j] = np.argsort(np.argsort(v)) + 1
    return good, rk


def main(n_total=240):
    import stage2_multiepoch as s2
    s2._w_init(None)
    pop = np.load(POP)
    all_rows = np.array([g["row"] for g in s2._W["gals"]])
    lmh = pop["logmh_zk_diffmah"][all_rows][:, 0]
    rows = S31.stratified(all_rows, lmh, n_total)
    recs = H.build_records(rows=rows)
    data = pop["data"][np.array([h.row for h in recs])]
    lmh_s = pop["logmh_zk_diffmah"][np.array([h.row for h in recs])][:, 0]

    cells = list(itertools.product(EFFS, FAMS, SIZES))
    print(f"exp54 STAGE 3.2 — the controlled factorial   "
          f"({len(cells)} cells, n={len(recs)}, z=0.4 and z=2.0)")
    print(f"  ranked by a SEVEN-CRITERION JUDGE, never by loss\n")

    results, t_all = [], time.time()
    for ci, ((en, ek), fam, (sn, sk)) in enumerate(cells):
        sp = M.Spec(family=fam, **ek, **sk)
        pr = F.Problem(sp, recs, data, epochs=(0, 4))
        npar = sp.n_theta
        try:
            th, lo = F.fit(pr, maxiter=600 * npar, verbose=False)
            sa, sf = pr.per_epoch(th)
            J, _ = F.jacobian(pr, th)
            s = np.linalg.svd(J, compute_uv=False)
            ident = float(s[-1] / s[0])
            r148, good = pr.diagnostics(th)
            over = float(np.median(r148))
            tilt = float(np.polyfit(lmh_s[good], r148, 1)[0])
        except Exception as e:                       # a cell may be degenerate
            print(f"  [{ci + 1:2d}/{len(cells)}] {sp.label:<34} FAILED: "
                  f"{type(e).__name__}", flush=True)
            results.append(dict(label=sp.label, eff=en, fam=fam, size=sn,
                                loss=np.nan, npar=npar))
            continue
        results.append(dict(label=sp.label, eff=en, fam=fam, size=sn, npar=npar,
                            loss=lo, theta=th, sA0=sa[0], sA4=sa[1],
                            sF0=sf[0], sF4=sf[1], ident=ident, over=over,
                            tilt=tilt))
        print(f"  [{ci + 1:2d}/{len(cells)}] {sp.label:<34} p={npar:<3d} "
              f"loss={lo:7.4f}  A={sa[0]:.3f}/{sa[1]:.3f}  "
              f"F={sf[0]:.3f}/{sf[1]:.3f}  ident={ident:.1e}  "
              f"over={over:+.3f}  tilt={tilt:+.4f}", flush=True)

    print(f"\n  factorial took {(time.time() - t_all) / 60:.1f} min")
    good, rk = judge(results)
    mean_rank = rk.mean(1)
    order = np.argsort(mean_rank)
    print(f"\n{RULE}\nTHE JUDGE — mean rank over seven criteria (lower is better)"
          f"\n{RULE}")
    print(f"  {'rank':>4} {'model':<34}{'mean':>7}{'sA(0.4)':>9}{'sA(2.0)':>9}"
          f"{'sF(0.4)':>9}{'sF(2.0)':>9}{'ident':>9}{'over':>8}{'tilt':>9}")
    for i, oi in enumerate(order[:15]):
        r = good[oi]
        print(f"  {i + 1:>4} {r['label']:<34}{mean_rank[oi]:>7.2f}"
              f"{r['sA0']:>9.3f}{r['sA4']:>9.3f}{r['sF0']:>9.3f}{r['sF4']:>9.3f}"
              f"{r['ident']:>9.1e}{r['over']:>+8.3f}{r['tilt']:>+9.4f}")
    print(f"\n  ... and the WORST three:")
    for oi in order[-3:]:
        r = good[oi]
        print(f"       {r['label']:<34}{mean_rank[oi]:>7.2f}")

    # what the loss ranking would have said, for contrast
    lo_order = np.argsort([r["loss"] for r in good])
    print(f"\n  If we had ranked by LOSS the top three would be:")
    for oi in lo_order[:3]:
        print(f"       {good[oi]['label']:<34} loss={good[oi]['loss']:.4f}  "
              f"judge rank {int(np.where(order == oi)[0][0]) + 1}")

    short = [good[i]["label"] for i in order[:4]]
    print(f"\n  SHORT LIST for Stage 3.3: {short}")
    np.savez_compressed(OUT, labels=np.array([r["label"] for r in good]),
                        mean_rank=mean_rank,
                        **{f"theta::{r['label']}": r["theta"] for r in good},
                        **{k: np.array([r[k] for r in good])
                           for k in ("loss", "sA0", "sA4", "sF0", "sF4",
                                     "ident", "over", "tilt", "npar")})
    print(f"  wrote {OUT}")
    return results, good, order


if __name__ == "__main__":
    n = 240
    if "--n" in sys.argv:
        n = int(sys.argv[sys.argv.index("--n") + 1])
    main(n)
