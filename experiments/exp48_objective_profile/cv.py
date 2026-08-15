"""exp48 Step D — held-out cross-validation of the density-log objective.

Everything in exp47 and exp48 is in-sample. The candidate change (fit the
adopted 1ch-mof kernel by comparing surface-density profiles in log,
instead of fractional residuals on the curve of growth) takes the mean
centered plane score from 2.72 to 1.95 and the compact-galaxy central
deficit from -43.6% to -31.4%. Neither number means anything for adoption
until it survives held-out galaxies.

Protocol: 5 folds. For each fold, fit on the other four fifths and predict
the held-out fifth. Every galaxy therefore gets exactly ONE out-of-sample
prediction, and the assembled set goes through the same judge as
everything else — so the held-out numbers are directly comparable to the
in-sample table rather than being a different statistic.

The baseline objective is cross-validated the same way, because the
comparison that matters is held-out-vs-held-out, not held-out-vs-in-sample.

Run: PYTHONPATH=. uv run python experiments/exp48_objective_profile/cv.py \
{demo|run} [--dev] [--folds K]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments/exp47_compact_defect"))
sys.path.insert(0, str(ROOT / "experiments/exp46_highz_ridge"))

OUTDIR = HERE / "outputs"
POP_NPZ = ROOT / "experiments/exp32_full_population/outputs/population.npz"
E40 = ROOT / "experiments/exp40_epoch_objective/outputs/latestart.npz"
WINNER = "q=density|res=log|rad=rms|w=uniform|gal=mean"


def folds_of(n, k, seed=0):
    idx = np.random.default_rng(seed).permutation(n)
    return [np.sort(idx[f::k]) for f in range(k)]


def cmd_run(dev=False, k=5):
    import objective as ob
    from hongshao import qa
    from judge import verdict, print_verdict, physics_gates, PLANES

    tag = "_dev" if dev else ""
    rows_all = np.load(POP_NPZ)["dev100"] if dev else None
    # the galaxy list the harness will build, in ITS order
    ob._w_init(rows_all)
    rows_kernel = np.array([g["row"] for g in ob._W["gals"]])
    n = len(rows_kernel)
    R = ob._W["e"].R
    it = max(int(4000 * (0.05 if dev else 1.0)), 100)

    cfgs = {"baseline": dict(ob.BASELINE)}
    for st in (f"factorial{tag}.npz", "factorial.npz"):
        p = OUTDIR / st
        if p.exists():
            d = dict(np.load(p, allow_pickle=True))
            if f"cfg::{WINNER}" in d:
                cfgs["density-log"] = json.loads(str(d[f"cfg::{WINNER}"][0]))
                break
    assert "density-log" in cfgs, "winner config not found in the step-B store"

    test_folds = folds_of(n, k)
    store = OUTDIR / f"cv{tag}.npz"
    done = dict(np.load(store, allow_pickle=True)) if store.exists() else {}
    print(f"exp48 Step D — {k}-fold held-out CV (n={n}"
          f"{', DEV' if dev else ''})", flush=True)

    for oname, cfg in cfgs.items():
        for f, test in enumerate(test_folds):
            key = f"{oname}::fold{f}"
            if f"theta::{key}" in done:
                print(f"  [{key}] cached", flush=True)
                continue
            train_rows = np.setdiff1d(rows_kernel, rows_kernel[test])
            ob._w_init(train_rows)
            p0 = np.asarray(np.load(E40)["theta_z15"], float)[:12]
            t0 = time.time()
            th, lo = ob.fit(cfg, p0, it, key)
            done[f"theta::{key}"] = th
            np.savez(store, **done)
            print(f"     ({(time.time()-t0)/60:.1f} min, "
                  f"trained on {len(train_rows)})", flush=True)

    # --- assemble ONE out-of-sample prediction per galaxy ----------------
    ob._w_init(rows_all)
    gals = ob._W["gals"]
    data_cogs = np.stack([g["data"] for g in gals])
    pop = np.load(POP_NPZ)
    logms = np.array([pop["logms"][g["row"]] for g in gals])
    r50_t = qa._safe_rhalf(data_cogs, R, 0.5)
    s2 = ob._W["s2"]

    results = {}
    for oname in cfgs:
        cogs = np.full((n, 5, len(R)), np.nan)
        for f, test in enumerate(test_folds):
            th = done[f"theta::{oname}::fold{f}"]
            for i in test:                      # held-out galaxies only
                c = s2.model_cogs(th, gals[i], [0, 1, 2, 3, 4], "1ch-mof")
                if c is not None:
                    cogs[i] = c
        assert np.isfinite(cogs).any(axis=(1, 2)).all(), "a galaxy has no OOS"
        v = verdict(cogs, data_cogs, R, r50_truth=r50_t)
        gt = physics_gates(cogs, data_cogs, logms, e=ob._W["e"])
        results[oname] = (v, gt)
        print_verdict(v, f"HELD-OUT {oname}", gt)

    key = f"{PLANES[0][0]}|{PLANES[0][1]}"
    print("\n\n=== HELD-OUT COMPARISON (every galaxy predicted "
          "out-of-sample) ===")
    print(f"  {'objective':<16}{'z0.4':>7}{'mean z<=1.5':>13}"
          f"{'M5 cmp':>9}{'M5 ext':>9}{'differential':>14}")
    for oname, (v, gt) in results.items():
        pl = [v[("plane", key, j, "all")] for j in range(4)]
        print(f"  {oname:<16}{pl[0]:7.1f}{float(np.mean(pl)):13.2f}"
              f"{100*v[('inner',5,0,'compact')]:8.1f}%"
              f"{100*v[('inner',5,0,'extended')]:8.1f}%"
              f"{gt['diff_model'][0]:9.2f}/{gt['diff_model'][1]:.2f}")
    print("\n  (in-sample marks for reference: baseline 2.1 / 2.72 / "
          "-43.0%, density-log 1.3 / 1.95 / -31.4%)")


def demo():
    f = folds_of(50, 5)
    assert len(f) == 5
    allid = np.concatenate(f)
    assert len(allid) == 50 and len(np.unique(allid)) == 50, "not a partition"
    for a in range(5):
        for b in range(a + 1, 5):
            assert not np.intersect1d(f[a], f[b]).size, "folds overlap"
    # a train set must exclude exactly its own test fold
    n = 50
    for t in f:
        assert len(np.setdiff1d(np.arange(n), t)) == n - len(t)
    print(f"demo OK — {len(f)} folds partition the sample exactly, "
          f"sizes {[len(x) for x in f]}, no overlap, "
          "train = complement of test")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    kk = int(sys.argv[sys.argv.index("--folds") + 1]) if "--folds" in sys.argv else 5
    if cmd == "demo":
        demo()
    elif cmd == "run":
        cmd_run("--dev" in sys.argv, kk)
    else:
        raise SystemExit(__doc__)
