"""exp81 measurement 2 — WHICH part of the halo history carries the
information the baseline leaves in its residual (measurement 1 found R^2 of
0.2-0.35 in the outskirts at every epoch and at every radius at z >= 1.5)?

At the cells (z, R) that matter, for the baseline's residual r =
log10(model/truth) of M*(<R):
  a. partial Spearman correlations at fixed log Mh(z_k) with: the mass at
     1 Gyr (early growth), the growth rate dlnM/dlnt at the epoch (recent
     accretion), the growth over the last 1 Gyr, the half-mass time t50 of
     the history to the epoch, the mass at the epoch itself (the SMHM tilt);
  b. out-of-fold gradient-boosting R^2 with RESTRICTED feature sets: early
     history only (t <= 2 Gyr), recent only (the last 1 Gyr), the mass at the
     epoch only — which block recovers the most.
Also the same partials for the TRUTH's log M*(<R) at fixed log Mh, so the
reader sees the data's own dependence next to the model's leftover.
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
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink",
          ROOT / "experiments/exp54_unpinned_amplitude",
          ROOT / "experiments/exp57_expansion_term",
          ROOT / "experiments/exp63_analytic_growth",
          ROOT / "experiments/exp73_size_relative",
          ROOT / "experiments/exp74_c19_history_leak",
          ROOT / "experiments/exp78_size_aware_objective",
          ROOT / "experiments/exp80_deposit_size_law", HERE):
    sys.path.insert(0, str(p))

import fit as F                                          # noqa: E402
import engine as E                                       # noqa: E402
import size_law as SL                                    # noqa: E402


def _by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S0T = _by_path("exp78_stage0_terms", ROOT / "experiments/exp78_size_aware_objective/stage0_terms.py")
RB = S0T.RB
ANCHOR_Z = list(E.ANCHOR_Z)
EPOCHS = (0, 1, 2, 3, 4)
CELLS = [(0, 4.92), (0, 103.45), (2, 103.45), (3, 4.92), (3, 32.58), (3, 103.45), (4, 4.92), (4, 32.58), (4, 103.45)]
OUTDIR = HERE / "outputs"


def partial(y, x, lmh):
    ok = np.isfinite(y) & np.isfinite(x) & np.isfinite(lmh)
    X = np.column_stack([np.ones(ok.sum()), lmh[ok], lmh[ok] ** 2])
    ry = y[ok] - X @ np.linalg.lstsq(X, y[ok], rcond=None)[0]
    rx = x[ok] - X @ np.linalg.lstsq(X, x[ok], rcond=None)[0]
    return float(spearmanr(ry, rx)[0])


def oof_r2(y, X):
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    y, X = y[ok], X[ok]
    pred = np.empty(len(y))
    for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20)
        m.fit(X[tr], y[tr]); pred[te] = m.predict(X[te])
    return 1.0 - (y - pred).var() / y.var()


def main():
    recs, data, mask, lmh_dm, lmh_bins, meas, fz, spec2, th_inc, pr, Rm, truth_m, n_inner, rows_m = S0T.build(False)
    spec_b, th_b = RB.adopted_baseline()
    rows = pr.all_rows
    cv = [meas[i] for i in rows]
    pred = SL.predict_law(spec2, th_b, SL.LAW_DEFAULT, cv, F.R_GRID, epochs=EPOCHS)
    truth = data[rows]
    good = np.isfinite(truth).all(axis=(1, 2)) & (truth > 0).all(axis=(1, 2))
    r = np.log10(np.clip(pred[good], 1, None)) - np.log10(truth[good])
    lt_truth = np.log10(truth[good])
    cvg = [c for c, g in zip(cv, good) if g]
    n = len(cvg)
    # per-epoch halo variables
    feats = {}
    for k in EPOCHS:
        lt_k = np.log10(E.T_ANCHOR[k]); t_k = E.T_ANCHOR[k]
        lm_k = np.array([E.log_mah(np.array([lt_k]), hc)[0] for hc in cvg])
        lm_1 = np.array([E.log_mah(np.array([0.0]), hc)[0] for hc in cvg])                       # 1 Gyr
        lm_2 = np.array([E.log_mah(np.array([np.log10(2.0)]), hc)[0] for hc in cvg])
        lm_prev = np.array([E.log_mah(np.array([np.log10(t_k - 1.0)]), hc)[0] for hc in cvg])    # 1 Gyr before the epoch
        rate = np.array([E.dm_dlnt(np.array([lt_k]), hc)[0] / 10 ** E.log_mah(np.array([lt_k]), hc)[0] for hc in cvg])
        grid = np.log10(np.geomspace(1.0, t_k, 40))
        t50 = np.array([10 ** np.interp(lm - np.log10(2), np.sort(E.log_mah(grid, hc)), np.sort(grid)) for hc, lm in zip(cvg, lm_k)])
        feats[k] = dict(lmh=lm_k, early=lm_1 - lm_k, early2=lm_2 - lm_k, recent=lm_k - lm_prev, rate=np.log10(np.clip(rate, 1e-3, None)),
                        t50=np.log10(t50))
    print(f"  {n} galaxies. Partial Spearman at fixed log Mh(z_k) (quadratic) of the baseline's residual r | of the truth's log M*(<R):")
    names = ["early", "early2", "recent", "rate", "t50"]
    print(f"  {'cell':<14}" + "".join(f"{nm + ' (M at 1 Gyr - Mh)' if nm == 'early' else nm:>26}" for nm in names))
    for k, R in CELLS:
        i = int(np.argmin(np.abs(F.R_GRID - R)))
        cells = []
        for nm in names:
            cells.append(f"{partial(r[:, k, i], feats[k][nm], feats[k]['lmh']):+.2f} | {partial(lt_truth[:, k, i], feats[k][nm], feats[k]['lmh']):+.2f}")
        print(f"  z={ANCHOR_Z[k]} R={R:<5.0f}" + "".join(f"{c:>26}" for c in cells))
    print(f"\n  out-of-fold R^2 of the baseline's residual from RESTRICTED feature sets: mass at the epoch only | early history "
          f"(log M at 1, 1.4, 2 Gyr + Mh) | recent (last 1 Gyr: M at t_k - 1, t_k - 0.5, t_k + rate) | full 12-time history")
    for k, R in CELLS:
        i = int(np.argmin(np.abs(F.R_GRID - R)))
        t_k = E.T_ANCHOR[k]
        lm = lambda t: np.array([E.log_mah(np.array([np.log10(t)]), hc)[0] for hc in cvg])
        X_m = feats[k]["lmh"][:, None]
        X_e = np.column_stack([lm(1.0), lm(1.4), lm(2.0), feats[k]["lmh"]])
        X_r = np.column_stack([lm(max(t_k - 1.0, 0.5)), lm(max(t_k - 0.5, 0.5)), feats[k]["lmh"], feats[k]["rate"]])
        X_f = np.column_stack([lm(t) for t in np.geomspace(1.0, t_k, 12)] + [feats[k]["rate"], feats[k]["lmh"]])
        y = r[:, k, i]
        print(f"  z={ANCHOR_Z[k]} R={R:<5.0f}  {oof_r2(y, X_m):+.3f} | {oof_r2(y, X_e):+.3f} | {oof_r2(y, X_r):+.3f} | {oof_r2(y, X_f):+.3f}")
    np.savez(OUTDIR / "residual_features.npz", r=r, lt_truth=lt_truth, **{f"{k}_{q}": v for k in EPOCHS for q, v in feats[k].items()})


if __name__ == "__main__":
    main()
