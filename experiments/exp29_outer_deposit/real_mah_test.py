"""exp29 — re-test the model with the REAL (de-dipped) MAH and no inner cut.

Two corrections the user flagged:
  1. dipfree_mah used the SMOOTH DiffMAH fit curve, which erases the sudden halo
     growth from mergers. Use the ACTUAL main-branch MAH instead, de-dipped with a
     running maximum (peak_history) -- this keeps real merger bursts, removes only
     spurious dips. (The real MAH ends ~snap 71 and has gaps; we don't require snap
     72 and keep t_obs at the z=0.4 cosmic time.)
  2. The inner-3-kpc cut may discard real signal, especially at high z. Re-evaluate
     with no inner cut (all 24 radii) as well.

Factorial (independent ceiling + loose-zdep quad), max|rel| epoch-avg:
  (smooth MAH, R>3)  = the prior 4.5% baseline
  (real   MAH, R>3)  = isolates the MAH change
  (real   MAH, all R)= the full requested setup
  (smooth MAH, all R)= isolates the inner-cut change

Run: PYTHONPATH=. uv run python experiments/exp29_outer_deposit/real_mah_test.py [n]
"""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from run import dipfree_mah, ANCHOR_SNAP, ANCHOR_Z                                   # noqa: E402
from single_epoch_all import model_cog_epoch                                        # noqa: E402
import loose_zdep                                                                    # noqa: E402
from hongshao.tng_data import (load_mah, peak_history, load_cosmic_time,             # noqa: E402
                               _time_to_redshift)
from hongshao.plotting import set_style                                             # noqa: E402
from hongshao.tng_data import COG_RAD_KPC                                           # noqa: E402

set_style()
R = COG_RAD_KPC
NPZ = HERE / "outputs" / "model_compare.npz"
Z = np.array(ANCHOR_Z)
_RAW, _TS, _ZT = None, None, None


def real_mah(gi):
    """De-dipped ACTUAL main-branch MAH (running-max peak history) -> deposit dict,
    matching dipfree_mah's keys but with real merger bursts. None if coverage is
    insufficient (must span z=2 and reach near z=0.4)."""
    global _RAW, _TS, _ZT
    if _RAW is None:
        _RAW, _TS, _ZT = load_mah(), load_cosmic_time(), _time_to_redshift()
    snaps, lmp = peak_history(_RAW[gi])
    if snaps is None:
        return None
    snaps = snaps.astype(int)
    if snaps.min() > ANCHOR_SNAP[-1] or snaps.max() < 70:        # need z=2 .. ~z=0.4
        return None
    tg = _TS[snaps]; ta, za = _ZT
    return dict(snap_full=snaps, logMh_full=lmp, t_full=tg, snap=snaps[1:], t=tg[1:],
                z=np.interp(tg[1:], ta, za), dMh=np.clip(np.diff(10.0 ** lmp), 0.0, None),
                t_obs=float(_TS[72]))


def maxrel(model, data, mask):
    return np.array([np.abs((model[k][mask] - data[k][mask]) / data[k][mask]).max()
                     for k in range(5)])


def fit_independent(mah, data, mask):
    """Per-epoch free fit (the ceiling); returns CoGs (5,24) and (5,5) params."""
    cogs, pars = [], []
    for k in range(5):
        dk = data[k]

        def loss(q):
            m = model_cog_epoch(q, mah, k)
            m = m * (dk[-1] / m[-1])
            v = np.sqrt(np.mean(((m[mask] - dk[mask]) / dk[mask]) ** 2))
            return v if np.isfinite(v) and q[4] > -1.0 else 1e3

        best = None
        for zc0 in (1.0, 2.0, 3.0):
            r = minimize(loss, [np.log10(40.0), 1.5, 4.0, 1.5, zc0], method="Nelder-Mead",
                         options=dict(maxiter=6000, xatol=1e-4, fatol=1e-9))
            if best is None or r.fun < best.fun:
                best = r
        m = model_cog_epoch(best.x, mah, k); cogs.append(m * (dk[-1] / m[-1])); pars.append(best.x)
    return np.array(cogs), np.array(pars)


def fit_loose(mah, data, mask, Pg):
    """Loose z-dependent quad (15 coeffs) joint fit; returns CoGs (5,24)."""
    t_obs = mah["t_obs"]

    def loss(p):
        cogs = loose_zdep.build_cogs(p, mah, data, t_obs)
        if cogs is None:
            return 1e3
        return float(np.mean([np.sqrt(np.mean(((cogs[k][mask] - data[k][mask]) / data[k][mask]) ** 2))
                              for k in range(5)]))

    warm = np.concatenate([np.polyfit(Z, Pg[:, j], 2)[::-1] for j in range(5)])
    flat = warm.reshape(5, 3).copy(); flat[:, 1:] = 0.0
    flat[:, 0] = [np.median(Pg[:, j]) for j in range(5)]
    best = None
    for p0 in (warm, flat.ravel()):
        r = minimize(loss, p0, method="Nelder-Mead",
                     options=dict(maxiter=12000, xatol=1e-4, fatol=1e-9))
        if best is None or r.fun < best.fun:
            best = r
    return np.array(loose_zdep.build_cogs(best.x, mah, list(data), t_obs))


def run_config(idx, datas, mahfun, mask):
    """Return per-epoch median max|rel| for independent and loose over the sample."""
    ind, loo = [], []
    for i in range(len(idx)):
        mah = mahfun(int(idx[i]))
        if mah is None:
            continue
        data = [datas[i][k] for k in range(5)]
        ci, Pg = fit_independent(mah, data, mask)
        cl = fit_loose(mah, data, mask, Pg)
        ind.append(maxrel(ci, datas[i], mask)); loo.append(maxrel(cl, datas[i], mask))
    return np.array(ind), np.array(loo)


def main():
    d = np.load(NPZ)
    idx, datas = d["index"], d["data"]
    n = int(sys.argv[1]) if len(sys.argv) > 1 else len(idx)
    idx, datas = idx[:n], datas[:n]
    masks = {"R>3": R > 3.0, "all R": R > 0.0}
    configs = [("smooth", "R>3"), ("real", "R>3"), ("real", "all R"), ("smooth", "all R")]
    mahfun = {"smooth": dipfree_mah, "real": real_mah}

    print(f"exp29 — real-MAH / no-inner-cut re-test (n<= {n}), median max|rel| over the eval mask\n")
    rows = {}
    for mk, ek in configs:
        ind, loo = run_config(idx, datas, mahfun[mk], masks[ek])
        rows[(mk, ek)] = (ind, loo)
        print(f"  [{mk:>6s} MAH, {ek:>5s}]  n={len(ind):2d}  "
              f"independent epoch-avg {100*np.median([np.median(ind[:,k]) for k in range(5)]):4.1f}%   "
              f"loose-quad epoch-avg {100*np.median([np.median(loo[:,k]) for k in range(5)]):4.1f}%")

    print("\n  per-epoch loose-quad max|rel| (the current best model):")
    print(f"    {'config':>18s} | " + " | ".join(f"z={z}".rjust(6) for z in ANCHOR_Z))
    for mk, ek in configs:
        loo = rows[(mk, ek)][1]
        print(f"    {mk+' / '+ek:>18s} | " + " | ".join(f"{100*np.median(loo[:,k]):5.1f}%" for k in range(5)))

    base = 100 * np.median([np.median(rows[("smooth", "R>3")][1][:, k]) for k in range(5)])
    realc = 100 * np.median([np.median(rows[("real", "all R")][1][:, k]) for k in range(5)])
    print(f"\n[verdict] loose-quad epoch-avg max|rel|:  smooth/R>3 baseline {base:.1f}%  ->  "
          f"real-MAH/all-R {realc:.1f}%")
    print("  (real-MAH/R>3 isolates the merger-burst effect; smooth/all-R isolates the inner-cut "
          "effect)")


if __name__ == "__main__":
    sys.exit(main())
