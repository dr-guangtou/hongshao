"""exp81 measurement 5 — the formation-time residual of a FITTED model: the
partial Spearman at fixed halo mass of log10(model/truth) with the recent
growth and the early mass (measurement 2's cells), and the out-of-fold R^2
of the residual from the full history, for a named fit file. Usage:
    ... residual_check_model.py FIT_NPZ [--delay] [--delay-exp] [--knobs q_e|none]
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink", ROOT / "experiments/exp54_unpinned_amplitude",
          ROOT / "experiments/exp57_expansion_term", ROOT / "experiments/exp63_analytic_growth",
          ROOT / "experiments/exp73_size_relative", ROOT / "experiments/exp74_c19_history_leak",
          ROOT / "experiments/exp78_size_aware_objective", ROOT / "experiments/exp80_deposit_size_law", HERE):
    sys.path.insert(0, str(p))
import fit as F                                          # noqa: E402
import engine as E                                       # noqa: E402


def _by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


S1 = _by_path("exp80_stage1_fit", ROOT / "experiments/exp80_deposit_size_law/stage1_fit.py")
ANCHOR_Z = list(E.ANCHOR_Z); EPOCHS = (0, 1, 2, 3, 4)
CELLS = [(0, 4.92), (0, 103.45), (2, 103.45), (3, 4.92), (3, 32.58), (3, 103.45), (4, 4.92), (4, 32.58), (4, 103.45)]


def partial(y, x, lmh):
    ok = np.isfinite(y) & np.isfinite(x) & np.isfinite(lmh)
    X = np.column_stack([np.ones(ok.sum()), lmh[ok], lmh[ok] ** 2])
    ry = y[ok] - X @ np.linalg.lstsq(X, y[ok], rcond=None)[0]; rx = x[ok] - X @ np.linalg.lstsq(X, x[ok], rcond=None)[0]
    return float(spearmanr(ry, rx)[0])


def oof_r2(y, X):
    ok = np.isfinite(y) & np.isfinite(X).all(1); y, X = y[ok], X[ok]; pred = np.empty(len(y))
    for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20)
        m.fit(X[tr], y[tr]); pred[te] = m.predict(X[te])
    return 1.0 - (y - pred).var() / y.var()


def main():
    a = sys.argv
    fit_file = a[1]
    kn = tuple(a[a.index("--knobs") + 1].split(",")) if "--knobs" in a else S1.DEFAULT_KNOBS
    if kn == ("none",):
        kn = ()
    recs, data, mask, lmh_dm, lmh_bins, meas, fz, spec2, th_inc, th_nested, pr, lp, bounds = S1.build(
        False, kn, "--delay" in a, "exp" if "--delay-exp" in a else "step")
    th = np.asarray(np.load(ROOT / "experiments/exp80_deposit_size_law/outputs" / fit_file, allow_pickle=True)["theta"], float)
    rows = pr.all_rows
    m = lp.predict(th, F.R_GRID, nodes=None or __import__("model2").FULL_NODES)
    truth = data[rows]; good = np.isfinite(truth).all(axis=(1, 2)) & (truth > 0).all(axis=(1, 2))
    r = np.log10(np.clip(m[good], 1, None)) - np.log10(truth[good])
    cvg = [c for c, g in zip([meas[i] for i in rows], good) if g]
    print(f"  {fit_file}: {lp.describe(th)}")
    print(f"  partial Spearman at fixed log Mh of the residual with RECENT growth | EARLY mass (2 Gyr); OOF R^2 of the residual from the full history")
    for k, R in CELLS:
        i = int(np.argmin(np.abs(F.R_GRID - R))); lt_k = np.log10(E.T_ANCHOR[k]); t_k = E.T_ANCHOR[k]
        lm_k = np.array([E.log_mah(np.array([lt_k]), hc)[0] for hc in cvg])
        lm_2 = np.array([E.log_mah(np.array([np.log10(2.0)]), hc)[0] for hc in cvg])
        lm_prev = np.array([E.log_mah(np.array([np.log10(t_k - 1.0)]), hc)[0] for hc in cvg])
        rate = np.array([E.dm_dlnt(np.array([lt_k]), hc)[0] / 10 ** E.log_mah(np.array([lt_k]), hc)[0] for hc in cvg])
        X = np.column_stack([np.array([E.log_mah(np.array([np.log10(t)]), hc)[0] for hc in cvg]) for t in np.geomspace(1.0, t_k, 12)]
                            + [np.log10(np.clip(rate, 1e-3, None)), lm_k])
        print(f"    z={ANCHOR_Z[k]} R={R:<6.0f} recent {partial(r[:, k, i], lm_k - lm_prev, lm_k):+.2f} | early {partial(r[:, k, i], lm_2 - lm_k, lm_k):+.2f}   "
              f"median {np.median(r[:, k, i]):+.3f} rms {np.sqrt(np.mean(r[:, k, i] ** 2)):.3f}  OOF R^2 {oof_r2(r[:, k, i], X):+.3f}")


if __name__ == "__main__":
    main()
