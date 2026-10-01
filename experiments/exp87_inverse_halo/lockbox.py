"""exp87 Stage 6 — the LOCKBOX, scored ONCE.

One galaxy in five was set aside at Stage 0 and never touched by a fit, a
choice of method, a hyper-parameter or a decision rule. Here every headline
method is fitted on ALL the development galaxies and predicts the lockbox.
The list of headline cells below was written before this script was first run.

For each cell the lockbox's mean CRPS is compared with what the development
folds predict for a sample of the lockbox's size: the 99% interval of the mean
of that many per-galaxy development scores (bootstrap). A lockbox score
outside its interval is a STOP condition of the plan (winner's curse or a
leak), reported as such.

The script refuses to run twice (`outputs/lockbox.json` exists) unless
`--force`, and a forced rerun is recorded in the file.

Run:  PYTHONPATH=. uv run python -u experiments/exp87_inverse_halo/lockbox.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as C                                       # noqa: E402
import generative as G                                   # noqa: E402
import harness as HN                                     # noqa: E402
import mah as MAH                                        # noqa: E402
import data as D                                         # noqa: E402
import methods as M                                      # noqa: E402

RULE = "=" * 100
OUT = C.OUTDIR / "lockbox.json"


def spec(sample, epoch, population, target, feature, extra=None):
    return dict(sample=sample, epoch=epoch, population=population, target=target, feature=feature, extra=extra)


def quad(k):
    return M.DirectLinear(f"linear+pca{k}-quad", "L2", transform=(lambda: M._LinearPlusQuadratic(k)), ridge_grid=M.RIDGE_GRID)


def headline_cells():
    cells = []
    p = ("parent", 0, "parent")
    cells += [(spec(*p, "mh", "mtot"), M.Climatology("mh")), (spec(*p, "mh", "mtot"), M.scalar_method("line")),
              (spec(*p, "mh", "M(>50)"), M.scalar_method("line")), (spec(*p, "mh", "mass_size"), M.DirectLinear("linear", "L1")),
              (spec(*p, "mh", "raw24"), M.DirectLinear("linear", "L1")), (spec(*p, "mh", "raw24"), quad(6)),
              (spec(*p, "mh", "raw24"), G.GenerativeInverse("gen-k24", k=24, epoch=0)),
              (spec(*p, "mh", "outer_shell"), M.scalar_method("line")), (spec(*p, "mh", "mtot+outer_shell"), M.DirectLinear("linear", "L1")),
              (spec(*p, "mh", "shells24"), M.DirectLinear("ridge", "L1", ridge_grid=M.RIDGE_GRID)), (spec(*p, "mh", "shells24"), quad(6))]
    for k in range(1, 5):
        for pop in ("asis", "complete"):
            c = ("curated", k, pop)
            cells += [(spec(*c, "mh", "mtot"), M.scalar_method("line")),
                      (spec(*c, "mh", "raw24"), M.DirectLinear("ridge", "L1", ridge_grid=M.RIDGE_GRID)), (spec(*c, "mh", "raw24"), quad(6)),
                      (spec(*c, "mh", "shells24"), M.DirectLinear("ridge", "L1", ridge_grid=M.RIDGE_GRID)), (spec(*c, "mh", "shells24"), quad(6))]
    for target in ("logc", "t50"):
        cells += [(spec(*p, target, "mtot"), M.Climatology(target)), (spec(*p, target, "raw24"), M.DirectLinear("linear", "L1")),
                  (spec(*p, target, "raw24"), quad(6)),
                  (spec(*p, target, "none", extra="mh"), M.DirectLinear("poly-mh", "L0", transform=M._poly_only)),
                  (spec(*p, target, "raw24", extra="mh"), quad(6))]
    return cells


def score_cell(sp, method, rng):
    s, rows, cut, X, _, y, fold = HN.cell_setup(sp)
    train, test = fold >= 0, rows & s.lockbox
    pred = method.fit_predict(X[train], y[train], X[test], cut)
    crps = pred.crps(y[test])
    err = pred.mean() - y[test]
    pit = pred.pit(y[test])
    dev_id = HN.cell_id(dict(sp, extra=sp.get("extra") or "none", method=method.name))
    dev = HN.load_cell(dev_id)
    dev_crps = dev["crps"][dev["used"]]
    boot = dev_crps[rng.integers(0, len(dev_crps), size=(4000, len(crps)))].mean(1)
    lo, hi = np.quantile(boot, [0.005, 0.995])
    inside = bool(lo <= crps.mean() <= hi)
    return dict(cell=dev_id, n_lockbox=int(test.sum()), n_dev=int(len(dev_crps)), crps_lockbox=float(crps.mean()), crps_dev=float(dev_crps.mean()),
                dev_lo=float(lo), dev_hi=float(hi), inside=inside, rmse_lockbox=float(np.sqrt(np.mean(err ** 2))),
                cover68=float(np.mean((pit > 0.16) & (pit < 0.84))), cover90=float(np.mean((pit > 0.05) & (pit < 0.95))),
                bias=float(np.mean(err)))


def history_lockbox(epoch, rng):
    """The accretion-history readings on the lockbox: the curve RMS error of each predictor."""
    s = D.load_curated(epoch, verbose=False)
    Y, taus, _ = MAH.history_targets(s)
    mh = s.targets["mh"]
    ok = np.isfinite(Y).all(1) & np.isfinite(mh)
    train, test = ok & ~s.lockbox, ok & s.lockbox
    X = s.logcog
    out = {}
    fits = {"climatology": (MAH.mean_vec, X), "mediated": (MAH.mediated_vec, np.column_stack([X, mh])), "ridge": (MAH.ridge_vec, X),
            "oracle-mh": (MAH.poly_vec, mh[:, None]), "cog+oracle": (MAH.ridge_vec, np.column_stack([X, mh]))}
    dev = np.load(C.OUTDIR / f"mah_epoch{epoch}.npz", allow_pickle=True)
    for name, (fn, design) in fits.items():
        pred = fn(design[train], Y[train], design[test])
        err = np.sqrt(np.mean((pred - Y[test]) ** 2, axis=1))
        dev_pred = dev[f"pred_{name.replace('-', '_').replace('+', '_')}"]
        used = dev["fold"] >= 0
        dev_err = np.sqrt(np.mean((dev_pred[used] - dev["Y"][used]) ** 2, axis=1))
        boot = np.sqrt(np.mean(dev_err[rng.integers(0, len(dev_err), size=(4000, len(err)))] ** 2, axis=1))
        lo, hi = np.quantile(boot, [0.005, 0.995])
        rms = float(np.sqrt(np.mean(err ** 2)))
        out[name] = dict(n_lockbox=int(test.sum()), curve_rms_lockbox=rms, curve_rms_dev=float(np.sqrt(np.mean(dev_err ** 2))),
                         dev_lo=float(lo), dev_hi=float(hi), inside=bool(lo <= rms <= hi))
    return out


def main(force=False):
    if OUT.exists() and not force:
        print(f"the lockbox has been scored ({OUT}); it is scored once. --force to redo (recorded).")
        return 1
    runs = (json.loads(OUT.read_text()).get("runs", 1) + 1) if OUT.exists() else 1
    rng = np.random.default_rng(C.SEED)
    t0 = time.time()
    print(f"{RULE}\nexp87 STAGE 6 — the lockbox, scored once (run {runs}; config {C.config_hash()})\n{RULE}")
    rows = []
    print(f"  {'cell':<74}{'n':>5}{'lockbox':>9}{'dev':>9}{'dev 99% interval':>20}{'RMSE':>8}{'cov68':>7}{'cov90':>7}")
    for sp, method in headline_cells():
        r = score_cell(sp, method, rng)
        rows.append(r)
        print(f"  {r['cell']:<74}{r['n_lockbox']:>5}{r['crps_lockbox']:>9.4f}{r['crps_dev']:>9.4f}   [{r['dev_lo']:.4f}, {r['dev_hi']:.4f}]"
              f"{r['rmse_lockbox']:>8.4f}{r['cover68']:>7.2f}{r['cover90']:>7.2f}  {'' if r['inside'] else 'OUTSIDE'}", flush=True)
    hist = {}
    for k in (0, 2, 4):
        hist[k] = history_lockbox(k, rng)
        print(f"  accretion history, z = {C.ANCHOR_Z[k]} (curve RMS, dex): "
              + "; ".join(f"{name} {v['curve_rms_lockbox']:.4f} (dev {v['curve_rms_dev']:.4f}){'' if v['inside'] else ' OUTSIDE'}" for name, v in hist[k].items()))
    n_out = sum(not r["inside"] for r in rows) + sum(not v["inside"] for h in hist.values() for v in h.values())
    n_all = len(rows) + sum(len(h) for h in hist.values())
    expected = 0.01 * n_all
    verdict = dict(n_cells=n_all, n_outside=int(n_out), expected_outside=expected, passed=bool(n_out <= max(2, int(np.ceil(3 * expected)))))
    print(f"\n  {n_out} of {n_all} lockbox scores fall outside their development 99% interval (about {expected:.1f} expected by chance) -> "
          f"{'consistent with the development scores' if verdict['passed'] else 'STOP: the lockbox contradicts the development scores'}")
    OUT.write_text(json.dumps(dict(runs=runs, config=C.config_hash(), cells=rows, history={str(k): v for k, v in hist.items()},
                                   verdict=verdict, seconds=round(time.time() - t0, 1)), indent=1))
    print(f"  wrote {OUT}")
    return 0 if verdict["passed"] else 2


if __name__ == "__main__":
    sys.exit(main(force="--force" in sys.argv))
