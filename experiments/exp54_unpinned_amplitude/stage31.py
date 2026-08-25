"""exp54 Stage 3.1 — the minimal absolute model, on a small mass-stratified
sample, at z=0.4 AND z=2.0.

The smallest thing that can be right: efficiency E1 (3 parameters), one
finite-tail profile family, and the two candidate halo size scales

    S2   R50 = f0 * R200c(t_j)            * (1+z_j)^b     the halo BOUNDARY
    S2a  R50 = fs * R200c(t_j)/c200c(t_j) * (1+z_j)^b     the NFW SCALE RADIUS

fitted jointly against amplitude and shape with the split objective. Screening
at **two epochs, never z=0.4 alone** (review §5.5): a profile form can win a
single epoch and fail the consistency problem.

Every fit ships the identifiability report. Nothing is promoted on loss.

Run: HONGSHAO_DATA_DIR=... PYTHONPATH=. uv run python \\
     experiments/exp54_unpinned_amplitude/stage31.py [--n 300] [--figures]
"""
from __future__ import annotations

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
from hongshao.plotting import OKABE_ITO, save_fig, set_style   # noqa: E402
from hongshao.qa import _tex                             # noqa: E402

POP = ROOT / "experiments/exp32_full_population/outputs/population.npz"
OUT = HERE / "outputs" / "stage31.npz"
FIGDIR = HERE / "figures"
Z = [0.4, 0.7, 1.0, 1.5, 2.0]
RULE = "=" * 84


def incumbent_shape_loss(recs, data):
    """The exp53 pinned kernel's NORMALIZED-shape loss on this exact sample and
    these exact epochs -- the fair reference for `score_F`.

    Stage 0's `L_F_REF` was measured at z=0.4 only, so quoting `score_F`
    against it compares a two-epoch fit with a one-epoch reference. This
    recomputes it like for like.
    """
    sys.path.insert(0, str(ROOT / "experiments/exp53_deposition_only"))
    import kernel as K
    import stage2_multiepoch as s2
    from hongshao.objective import Objective
    th = K.from_exp38_theta(K.adopted_theta())
    sp = K.Spec("moffat", q_free=True)
    e = s2._W["e"]
    i100 = F.I100
    mm, dd = [], []
    by_row = {h.row: i for i, h in enumerate(recs)}
    for g in s2._W["gals"]:
        i = by_row.get(g["row"])
        if i is None:
            continue
        c = K.model_cogs(sp, th, g, [0, 4])
        if c is None:
            continue
        c = np.asarray(c)
        if np.any(c[:, i100] <= 0):
            continue
        mm.append(c / c[:, i100][:, None])
        dd.append(data[i][[0, 4]] / data[i][[0, 4], i100][:, None])
    return float(Objective()(np.array(mm), np.array(dd), F.R_GRID))


def stratified(rows, logmh, n_total, n_bins=10, seed=0):
    """Mass-stratified subsample: equal counts per halo-mass bin."""
    rng = np.random.default_rng(seed)
    order = np.argsort(logmh)
    per = max(n_total // n_bins, 1)
    pick = []
    for grp in np.array_split(order, n_bins):
        pick.extend(rng.choice(grp, size=min(per, len(grp)), replace=False))
    return rows[np.sort(np.array(pick))]


def main(n_total=300, do_figs=False):
    import stage2_multiepoch as s2
    s2._w_init(None)
    pop = np.load(POP)
    all_rows = np.array([g["row"] for g in s2._W["gals"]])
    lmh = pop["logmh_zk_diffmah"][all_rows][:, 0]
    rows = stratified(all_rows, lmh, n_total)
    recs = H.build_records(rows=rows)
    keep = np.array([np.where(all_rows == h.row)[0][0] for h in recs])
    data = pop["data"][np.array([h.row for h in recs])]
    lmh_s = pop["logmh_zk_diffmah"][np.array([h.row for h in recs])]

    print(f"exp54 STAGE 3.1 — the minimal absolute model   (n={len(recs)}, "
          f"mass-stratified, epochs z=0.4 and z=2.0)")
    ref = incumbent_shape_loss(recs, data)
    print(f"  INCUMBENT reference on THIS sample and THESE epochs: shape loss "
          f"{ref:.6f}\n  (Stage 0's L_F_REF = {F.L_F_REF:.6f} was z=0.4 only, so "
          f"score_F is quoted\n   against that constant but the fair "
          f"comparison is the number above)")
    print(f"  halo mass span {lmh_s[:, 0].min():.2f}-{lmh_s[:, 0].max():.2f}, "
          f"median {np.median(lmh_s[:, 0]):.2f}")

    results = {}
    for size in ("S2", "S2a"):
        sp = M.Spec(family="sersic", eff="E1", size=size)
        pr = F.Problem(sp, recs, data, epochs=(0, 4))
        t0 = time.time()
        print(f"\n{RULE}\n{sp.label}   params {sp.theta_names}\n{RULE}")
        th, lo = F.fit(pr, maxiter=3000)
        a, f, nb = pr.scores(th)
        print(f"    fitted in {(time.time() - t0) / 60:.1f} min, "
              f"{pr.n_eval} evaluations")
        print(f"    score_A = {a:.4f}   score_F = {f:.4f}   failed = {nb}")
        print(f"    -> amplitude is {a:.3f}x the best halo-only regression; "
              f"shape is {f * F.L_F_REF / ref:.3f}x the incumbent kernel "
              f"ON THIS SAMPLE")
        print(f"    theta: " + "  ".join(f"{n}={v:+.4f}"
                                         for n, v in zip(sp.theta_names, th)))
        rep = F.identifiability(pr, th, sp.theta_names, n_boot=10)

        # derived quantities -- often identified where parameters are not
        r50s, frac = [], []
        for h in recs:
            eff, size_p, shape = sp.unpack(th)
            r50 = M.r50_of(sp, size_p, h)
            ok = np.isfinite(h.r200c) & (h.r200c > 0)
            r50s.append(np.median((r50 / h.r200c)[ok]))
            dm = 10.0 ** M.log_eps(sp, eff, h) * h.dmh
            frac.append((dm * (h.z > 2) * ok).sum() / max((dm * ok).sum(), 1e-30))
        print(f"    DERIVED: median R50/R200c = {np.median(r50s):.4f}"
              f"  (16-84 {np.percentile(r50s, 16):.4f}-{np.percentile(r50s, 84):.4f})")
        print(f"             fraction of stellar mass deposited at z>2 = "
              f"{np.median(frac):.3f}")
        results[size] = dict(theta=th, loss=lo, score_A=a, score_F=f,
                             n_eff=rep["n_eff"], sv=rep["sv"],
                             r50_r200=np.array(r50s), pred=pr.predict(th))

    print(f"\n{RULE}\nS2 vs S2a — the two halo scales\n{RULE}")
    print(f"  {'':<26}{'loss':>10}{'score_A':>10}{'score_F':>10}{'n_eff':>8}")
    for k, v in results.items():
        lab = "R200c (boundary)" if k == "S2" else "R200c/c200c (NFW scale)"
        print(f"  {lab:<26}{v['loss']:>10.5f}{v['score_A']:>10.4f}"
              f"{v['score_F']:>10.4f}{v['n_eff']:>8}")
    win = min(results, key=lambda k: results[k]["loss"])
    d = abs(results["S2"]["loss"] - results["S2a"]["loss"])
    print(f"\n  lower loss: {win}  (by {d:.5f}); NOT a promotion -- the judge "
          f"decides at 3.2")

    np.savez_compressed(OUT, rows=np.array([h.row for h in recs]),
                        **{f"{k}_{f}": v[f] for k, v in results.items()
                           for f in ("theta", "loss", "score_A", "score_F",
                                     "n_eff", "sv", "r50_r200")})
    print(f"\n  wrote {OUT}")
    if do_figs:
        figures(results, recs, data, lmh_s)
    return results


def figures(results, recs, data, lmh_s):
    import matplotlib.pyplot as plt
    set_style()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    i100 = F.I100
    fig, ax = plt.subplots(2, 3, figsize=(14, 7.6))
    for col, (key, lab) in enumerate((("S2", "S2: R50 = f0 R200c"),
                                      ("S2a", "S2a: R50 = fs R200c/c200c"))):
        pred = results[key]["pred"]
        for row, k in enumerate((0, 1)):
            zz = Z[0] if k == 0 else Z[4]
            A_d = np.log10(data[:, 0 if k == 0 else 4, i100])
            A_m = np.log10(np.clip(pred[:, k, i100], 1.0, None))
            a = ax[row, col]
            a.scatter(A_d, A_m, s=7, alpha=0.5,
                      c=lmh_s[:, 0], cmap="viridis")
            lim = [min(A_d.min(), A_m.min()) - 0.1, max(A_d.max(), A_m.max()) + 0.1]
            a.plot(lim, lim, "k--", lw=0.8)
            a.set_xlim(lim); a.set_ylim(lim)
            a.set_xlabel(_tex(f"truth log M*(<100 kpc), z={zz}"))
            a.set_ylabel(_tex("model"))
            a.set_title(_tex(f"{lab}   z={zz}   rms="
                             f"{np.std(A_m - A_d):.3f} dex"), fontsize=9)
    for row, k in enumerate((0, 1)):
        a = ax[row, 2]
        for c, (key, lab) in enumerate((("S2", "S2"), ("S2a", "S2a"))):
            pred = results[key]["pred"]
            F_m = pred[:, k] / np.clip(pred[:, k, i100:i100 + 1], 1.0, None)
            F_d = data[:, 0 if k == 0 else 4] / data[:, 0 if k == 0 else 4,
                                                     i100:i100 + 1]
            med = np.nanmedian(np.log10(np.clip(F_m, 1e-30, None)
                                        / np.clip(F_d, 1e-30, None)), axis=0)
            a.plot(F.R_GRID, med, "-o", ms=3, color=OKABE_ITO[c], label=lab)
        a.axhline(0, color="0.6", lw=0.8)
        a.axvline(100.0, color="0.85", lw=0.8)
        a.set_xscale("log")
        a.set_xlabel(_tex("R [kpc]"))
        a.set_ylabel(_tex("median log(model/truth) SHAPE"))
        a.set_title(_tex(f"normalized shape residual, z={Z[0] if k == 0 else Z[4]}"),
                    fontsize=9)
        a.legend(fontsize=8)
        a.grid(alpha=0.3)
    fig.suptitle(_tex("exp54 Stage 3.1 -- the minimal absolute model, "
                      "E1 x sersic, two halo size scales"), fontsize=12)
    fig.tight_layout()
    print("  wrote", save_fig(fig, FIGDIR / "s31_minimal_model")[0])


if __name__ == "__main__":
    n = 300
    if "--n" in sys.argv:
        n = int(sys.argv[sys.argv.index("--n") + 1])
    main(n, "--figures" in sys.argv)
