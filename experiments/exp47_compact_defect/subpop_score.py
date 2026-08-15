"""exp47 stage 3 — score COMPACT and EXTENDED sub-populations separately.

Compact galaxies (truth R50 < 5 kpc) are 6.3% of the z=0.4 sample and
85.2% of the z=2 sample. Every fix in the record was judged on an average
they barely enter at low redshift, which is a candidate explanation for a
standing puzzle: the exp38 core channel and retention floor both halved
the inner-mass deficit and then "broke the differential physics". If the
objective is dominated by the 94% that are already fine, a fix aimed at
the minority can only look like damage.

THE MANDATORY CONTROL (exp46, measured): a centered E/floor score falls
for free when n shrinks — going from n=1200 to n=200 moved a *fixed*
model from 4.4 to 1.8. The compact and extended groups differ in size by
more than an order of magnitude at low z, so their raw scores are NOT
comparable to each other or to the full-sample number. Every
sub-population score is therefore reported beside a size-matched random
subsample of the full population, and only the pair is interpretable.

No fitting. Run: PYTHONPATH=. uv run python \
experiments/exp47_compact_defect/subpop_score.py {demo|run}
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments/exp46_highz_ridge"))

OUTDIR = HERE / "outputs"
POP_NPZ = ROOT / "experiments/exp32_full_population/outputs/population.npz"
Z_GRID = (0.4, 0.7, 1.0, 1.5, 2.0)
PLANES = (("kpc:M(<30)", "kpc:M(30-50)"), ("kpc:M(<30)", "kpc:M(50-100)"))
R_COMPACT = 5.0
N_REP = 16


def matched_random(score_fn, n_total, n_keep, n_rep=N_REP, seed=0):
    """Median score over n_rep random subsamples of size n_keep."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_rep):
        m = np.zeros(n_total, bool)
        m[rng.choice(n_total, size=min(n_keep, n_total), replace=False)] = True
        vals.append(score_fn(m))
    return np.array(vals, float)


def subpop_report(truth, model, data_cogs, model_cogs, R, r50_truth,
                  label="model"):
    """{(plane, epoch, group): dict} with the size-matched control attached."""
    from hongshao import qa
    n = len(r50_truth)
    out = {}
    for kx, ky in PLANES:
        for j in range(len(Z_GRID)):
            def score(mask):
                t = np.column_stack([
                    np.log10(np.clip(truth[kx][mask, j], 1.0, None)),
                    np.log10(np.clip(truth[ky][mask, j], 1.0, None))])
                m = np.column_stack([
                    np.log10(np.clip(model[kx][mask, j], 1.0, None)),
                    np.log10(np.clip(model[ky][mask, j], 1.0, None))])
                return qa.plane_energy(t, m)["energy_ratio_centered"]

            comp = r50_truth[:, j] < R_COMPACT
            for gname, mask in (("compact", comp), ("extended", ~comp)):
                nk = int(mask.sum())
                if nk < 40:              # energy distance is meaningless here
                    out[(f"{kx}|{ky}", j, gname)] = dict(
                        n=nk, score=np.nan, ctrl=np.nan, p=np.nan)
                    continue
                e = score(mask)
                rnd = matched_random(score, n, nk, seed=17 + j)
                out[(f"{kx}|{ky}", j, gname)] = dict(
                    n=nk, score=float(e), ctrl=float(np.median(rnd)),
                    p=float(np.mean(rnd <= e)))
    return out


def cmd_run(dev=False):
    from hongshao import qa
    from unification import (_w_init, adopted_model_cogs, _W, POP_NPZ as PN)

    rows = np.load(PN)["dev100"] if dev else None
    tag = "_dev" if dev else ""
    _w_init(rows)
    gals, R = _W["gals"], _W["e"].R
    data_cogs = np.stack([g["data"] for g in gals])
    n = len(gals)
    print(f"exp47 stage 3 — compact vs extended, scored separately (n={n}"
          f"{', DEV' if dev else ''})\n", flush=True)

    model_cogs = adopted_model_cogs(data_cogs, rows, dev)
    truth, model, _, _ = qa.measure_all(model_cogs, data_cogs, R)
    r50_t = qa._safe_rhalf(data_cogs, R, 0.5)

    print("  population split by truth R50 (the exp46 boundary, 5 kpc):")
    print(f"    {'z':>5}{'compact':>10}{'extended':>10}{'% compact':>12}")
    for j, z in enumerate(Z_GRID):
        c = int((r50_t[:, j] < R_COMPACT).sum())
        print(f"    {z:5.1f}{c:10d}{n - c:10d}{100 * c / n:11.1f}%")

    rep = subpop_report(truth, model, data_cogs, model_cogs, R, r50_t)

    print("\n  centered E/floor by sub-population. 'ctrl' = the median of "
          f"{N_REP} random\n  subsamples of the SAME size drawn from the "
          "full population; p = the fraction\n  of those scoring at or "
          "below the sub-population. Read score-vs-ctrl, never\n  the raw "
          "score (it falls for free as n shrinks).")
    for kx, ky in PLANES:
        key = f"{kx}|{ky}"
        print(f"\n    {kx} vs {ky}")
        print(f"      {'z':>5} | {'compact':>21} | {'extended':>21}")
        print(f"      {'':>5} | {'n':>5}{'score':>7}{'ctrl':>6}{'p':>5} | "
              f"{'n':>5}{'score':>7}{'ctrl':>6}{'p':>5}")
        for j, z in enumerate(Z_GRID):
            cells = []
            for g in ("compact", "extended"):
                d = rep[(key, j, g)]
                if not np.isfinite(d["score"]):
                    cells.append(f"{d['n']:>5}{'  (n too small)':>18}")
                else:
                    cells.append(f"{d['n']:>5}{d['score']:>7.1f}"
                                 f"{d['ctrl']:>6.1f}{d['p']:>5.2f}")
            print(f"      {z:5.1f} | {cells[0]} | {cells[1]}")

    OUTDIR.mkdir(exist_ok=True)
    np.savez(OUTDIR / f"subpop_score{tag}.npz",
             keys=np.array([f"{k[0]}|{k[1]}|{k[2]}" for k in rep]),
             vals=np.array([[rep[k]["n"], rep[k]["score"], rep[k]["ctrl"],
                             rep[k]["p"]] for k in rep]),
             r50_truth=r50_t)
    print(f"\n  wrote {OUTDIR / f'subpop_score{tag}.npz'}")


def demo():
    from hongshao import qa
    rng = np.random.default_rng(0)
    # A model that is PERFECT on extended galaxies and under-dispersed on
    # compact ones must be caught in the compact group and cleared in the
    # extended one — the exact discrimination this stage exists to make.
    n = 2000
    r50 = np.concatenate([rng.uniform(1, 5, 300), rng.uniform(5, 20, 1700)])
    t = rng.normal(size=(n, 2)) @ np.array([[1.0, 0.9], [0.0, 0.5]])
    m = t.copy()
    comp = r50 < 5.0
    m[comp] = t[comp] * 0.3                       # under-dispersed, compact
    def score(mask):
        return qa.plane_energy(t[mask], m[mask])["energy_ratio_centered"]
    e_c, e_e = score(comp), score(~comp)
    ctrl_c = np.median(matched_random(score, n, int(comp.sum())))
    assert e_e < 1.6, f"extended should be clean, got {e_e}"
    assert e_c > ctrl_c + 1.0, (e_c, ctrl_c)
    # ... and the control really is doing work: the RAW compact score alone
    # would be compared against a full-sample number computed at 6x the n.
    e_all = score(np.ones(n, bool))
    print(f"demo OK — compact group flagged ({e_c:.1f} vs matched control "
          f"{ctrl_c:.1f}), extended cleared ({e_e:.1f}); the full-sample "
          f"score is {e_all:.1f}, which is why the control is mandatory")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if cmd == "demo":
        demo()
    elif cmd == "run":
        cmd_run(dev="--dev" in sys.argv)
    else:
        raise SystemExit(__doc__)
