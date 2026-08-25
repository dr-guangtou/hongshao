"""exp57 Stage 2 — run the gates, at the null first.

`--null` runs them on the incumbent with the expansion switched off. That costs
nothing and it is not optional: without it the fitted gate values have no
reference, and a reader cannot tell a number that means "expansion did this"
from a number the model already had. exp52's own headline went wrong once for
exactly the want of a reference term.

`--fit LABEL` runs the same gates on a fitted model from `outputs/stage3_fits/`.

Run: HONGSHAO_DATA_DIR=... OMP_NUM_THREADS=1 PYTHONPATH=. uv run python -u \\
     experiments/exp57_expansion_term/stage2_gates.py --null [--smoke]
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

FITDIR = HERE / "outputs" / "stage3_fits"
OUTDIR = HERE / "outputs"
RULE = "=" * 96
_P = S35._PC


def report(tag, pr, theta, data, mask, smoke=False):
    print(f"\n{RULE}\nGATES for {tag}\n{RULE}")
    rel = G.check_terms(pr, theta)
    print(f"\n  SELF-CHECK: summing the separated deposits reproduces the "
          f"model to {rel:.2e} relative.")

    g1 = G.g1_mechanism(pr, theta)
    print(f"\n  G1 — THE MECHANISM SPLIT. How much of each shell's growth is "
          f"material MOVED,\n  and how much is material newly DEPOSITED? "
          f"exp52's first model reached "
          f"{100 * G.EXP52_SHARE['z=1.0 -> 0.7']:.1f}{_P} and\n  "
          f"{100 * G.EXP52_SHARE['z=0.7 -> 0.4']:.1f}{_P} in M[50,100]; the "
          f"gate is that expansion must not out-grow new\n  deposition below "
          f"z = {G.G1_Z_LIMIT}.")
    print(f"    {'epoch pair':<18}{'shell':<14}{'expansion':>13}"
          f"{'new deposit':>13}{'share':>9}")
    for r in g1:
        flag = ""
        if r["z_late"] < G.G1_Z_LIMIT and r["share"] > 0.5 and r["total"] > 0:
            flag = "  <- FAILS"
        print(f"    {r['pair']:<18}{r['shell']:<14}{r['expansion']:>13.3e}"
              f"{r['new_deposition']:>13.3e}"
              f"{100 * r['share']:>8.1f}{_P}{flag}")

    g2 = G.g2_reach(pr, theta)
    print(f"\n  G2 — THE REACH. Of the mass expansion MOVES, where does it "
          f"come from and where\n  does it go? exp52's first model took "
          f"{100 * 0.93:.0f}{_P} from inside 5 kpc (correct) and delivered\n"
          f"  {100 * G.EXP52_REACH_50:.1f}{_P} beyond 50 kpc and "
          f"{100 * G.EXP52_REACH_148:.1f}{_P} beyond 148 kpc (the failure). "
          f"Limits: {100 * G.G2_MAX_BEYOND_50:.0f}{_P} and "
          f"{100 * G.G2_MAX_BEYOND_148:.0f}{_P}.")
    print(f"    {'epoch pair':<18}{'moved':>9}{'from':>10}{'beyond':>9}"
          f"{'beyond':>9}{'beyond':>9}{'mass':>10}")
    print(f"    {'':<18}{'of total':>9}{'<10 kpc':>10}{'50 kpc':>9}"
          f"{'148 kpc':>9}{'500 kpc':>9}{'balance':>10}")
    for r in g2:
        # every column here is a RATIO of moved mass, so all of them are
        # undefined when nothing moves. At the null `drained` is not exactly
        # zero -- it is rounding -- so the guard is relative, not `> 0`.
        if r["moved_fraction"] < 1e-10:
            print(f"    {r['pair']:<18}"
                  + f"{'no mass moved (expansion off): every ratio undefined':>58}")
            continue
        flag = ""
        if r["f_beyond_50"] > G.G2_MAX_BEYOND_50:
            flag += "  <- FAILS 50"
        if r["f_beyond_148"] > G.G2_MAX_BEYOND_148:
            flag += "  <- FAILS 148"
        print(f"    {r['pair']:<18}{100 * r['moved_fraction']:>8.1f}{_P}"
              f"{100 * r['f_source_inside_10']:>9.1f}{_P}"
              f"{100 * r['f_beyond_50']:>8.1f}{_P}"
              f"{100 * r['f_beyond_148']:>8.1f}{_P}"
              f"{100 * r['f_beyond_500']:>8.1f}{_P}"
              f"{r['residual']:>10.1e}{flag}")

    g3 = G.g3_deletion(pr, theta)
    print(f"\n  G3 — NO INVISIBLE DELETION. `score_F` normalises at "
          f"{G.R_BLIND:.0f} kpc and is blind beyond it,\n  while exp54 "
          f"over-supplies its own mass growth by 22{_P}. A fit using expansion "
          f"as an\n  amplitude regulator would have to RAISE this column.")
    print(f"    {'epoch':<10}{'deposited mass beyond ' + f'{G.R_BLIND:.0f} kpc':>34}")
    for r in g3:
        print(f"    z={r['z']:<8}{100 * r['f_beyond_blind']:>33.3f}{_P}")

    g1b = G.g1b_measured(pr, theta, data, mask)
    print(f"\n  G1b — THE MEASURED COMPANION, no threshold of mine in it. Of "
          f"the mass a galaxy gains\n  inside 148 kpc, what fraction lands "
          f"beyond 50 kpc? Same operator on the model and\n  on the measured "
          f"curves of growth; exp53 measured 0.37 on the data.")
    print(f"    {'epoch pair':<18}{'model':>10}{'data':>10}{'model-data':>13}"
          f"{'n':>8}")
    for r in g1b:
        print(f"    {r['pair']:<18}{r['model']:>10.3f}{r['data']:>10.3f}"
              f"{r['model'] - r['data']:>13.3f}{r['n']:>8}")

    g6 = G.g6_outskirt(pr, theta, data, mask)
    print(f"\n  G6 — THE OUTSKIRT STRUCTURE, reported not gated. Does the "
          f"model put the right amount\n  of mass in the 50-148 kpc annulus? "
          f"exp54 Stage 3.7 found that closing the central\n  deficit with a "
          f"shared size law cost 0.104 dex of outskirt mass at redshift 2; "
          f"this is\n  whether expansion pays the same price.")
    print(f"    {'epoch':<10}{'median dlog(model/truth)':>28}"
          f"{'population-summed':>20}{'n':>8}")
    for r in g6:
        print(f"    z={r['z']:<8}{r['median']:>28.4f}{r['summed']:>20.4f}"
              f"{r['n']:>8}")

    ok, fails = G.verdict(g1, g2, g1b)
    print(f"\n  VERDICT: {'PASS' if ok else 'FAIL'}")
    for f in fails:
        print(f"    - {f}")
    if not smoke:
        OUTDIR.mkdir(parents=True, exist_ok=True)
        np.savez(OUTDIR / f"stage2_gates_{tag}.npz",
                 tag=tag, theta=theta, passed=ok,
                 fails=np.array(fails, dtype=object),
                 g1_share=np.array([r["share"] for r in g1]),
                 g1_pair=np.array([r["pair"] for r in g1]),
                 g1_shell=np.array([r["shell"] for r in g1]),
                 g1_expansion=np.array([r["expansion"] for r in g1]),
                 g1_new=np.array([r["new_deposition"] for r in g1]),
                 g2_shells=np.array([r["shells"] for r in g2]),
                 g2_b50=np.array([r["f_beyond_50"] for r in g2]),
                 g2_b148=np.array([r["f_beyond_148"] for r in g2]),
                 g2_b500=np.array([r["f_beyond_500"] for r in g2]),
                 g2_src10=np.array([r["f_source_inside_10"] for r in g2]),
                 g2_moved=np.array([r["moved_fraction"] for r in g2]),
                 g3_beyond=np.array([r["f_beyond_blind"] for r in g3]),
                 g1b_model=np.array([r["model"] for r in g1b]),
                 g1b_data=np.array([r["data"] for r in g1b]),
                 g6_median=np.array([r["median"] for r in g6]),
                 g6_summed=np.array([r["summed"] for r in g6]),
                 reach_edges=np.array(G.REACH_EDGES))
        print(f"\n  wrote stage2_gates_{tag}.npz")
    return ok


def main(argv):
    smoke = "--smoke" in argv
    recs, data, mask, lmh, base, theta, slow = S0.build(smoke)
    print(f"exp57 STAGE 2 — the preregistered gates")
    print(f"  sample: {len(recs)} galaxies, {int(mask.sum())} galaxy-epochs")

    if "--fit" in argv:
        label = argv[argv.index("--fit") + 1]
        z = np.load(FITDIR / f"{label}.npz", allow_pickle=True)
        law = str(z["law"])
        sp = X.XSpec(base, law)
        pr = X.ExpandingProblem(sp, recs, data, mask, epochs=(0, 1, 2, 3, 4))
        th = np.asarray(z["theta"], float)
        print(f"  fitted {label}: loss {float(z['loss']):.6f}, "
              + ", ".join(f"{n}={v:+.4f}" for n, v in
                          zip(sp.x_names, th[base.n_theta:])))
        report(label, pr, th, data, mask, smoke)
    else:
        sp = X.XSpec(base, "X1")
        pr = X.ExpandingProblem(sp, recs, data, mask, epochs=(0, 1, 2, 3, 4))
        print(f"  THE NULL: the exp54 incumbent, expansion switched off "
              f"(A = 0). Every number below\n  is what the model ALREADY did "
              f"before this experiment touched it.")
        report("null", pr, sp.nest(theta), data, mask, smoke)


if __name__ == "__main__":
    main(sys.argv)
