"""exp36 phase 0 — residual anatomy pre-tests (run BEFORE building any DOF).

Three cheap measurements that decide where the new degrees of freedom go
(the exp30 lesson: the correlation pre-test costs seconds and predicts the
outcome of the build):

1. WALL CHECK — does the exp33 information wall (transport and statistical
   residuals share rho = 0.82-0.89 per radius) hold for the exp35 total-norm
   variant? The shared component is unwinnable at fixed features.
2. LEVER REGRESSION — regress the UNSHARED part of the transport residual on
   the candidate levers (c200c -> proposal B; burstiness -> E; logMh -> A;
   t50/fz2/late as controls), with shuffle controls.
3. MASSIVE-TERCILE PROBE — fit the exp35 single-width form to the massive
   logM* tercile ALONE: if it reaches the massive-end aperture fraction the
   population fit under-spreads (dlog f148 ~ 0), the two-channel split is
   about population coupling, not kernel form.

Run: PYTHONPATH=. uv run python experiments/exp36_two_channel/phase0.py
     [--skip-fit] to skip part 3 (the only slow part, ~3 min)
"""
import importlib.util
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from hongshao.profile_emulator import fit_profile                            # noqa: E402


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_EPOCHS = _load_by_path("exp33_epochs", ROOT / "experiments/exp33_single_epoch/epochs.py")
E35_DIR = ROOT / "experiments/exp35_total_norm"
POP = np.load(ROOT / "experiments/exp32_full_population/outputs/population.npz")
TN = np.load(E35_DIR / "outputs/total_norm.npz")
R = _EPOCHS.R
NFOLD = 10


def pinned_shape_res(model_cogs, data_cogs):
    """(n, R) log-space 148-pinned shape residual: log(model/data), model
    rescaled to the true total (the 16.1/19.1 convention, in dex)."""
    lm = np.log10(np.clip(model_cogs, 1.0, None))
    ld = np.log10(data_cogs)
    return (lm - lm[:, -1:]) - (ld - ld[:, -1:])


def part1_wall(X, data):
    print("\n=== phase 0.1 — the information wall, exp35 variant ===")
    n = len(X)
    prof = np.log10(data[:, 0, :-1])
    anchor = np.log10(data[:, 0, -1])
    MU = np.empty((n, 23))
    AM = np.empty(n)
    for f in range(NFOLD):
        te = np.arange(n) % NFOLD == f
        tr = ~te
        pemu = fit_profile(X[tr], prof[tr], anchor[tr], R[:-1], n_modes=3)
        MU[te] = pemu.predict(X[te])[0]
        AM[te] = pemu.emu.predict(X[te])[0][:, 0]
    cog_stat = np.column_stack([10.0 ** MU, 10.0 ** AM[:, None]])
    res_S = pinned_shape_res(cog_stat, data[:, 0])
    out = {}
    for label in ("multi-slope", "z04-slope"):
        cog_T = TN[f"cogs_{label}"][:, 0]
        ok = np.isfinite(cog_T).all(1) & (cog_T > 0).all(1)
        res_T = pinned_shape_res(cog_T[ok], data[ok, 0])
        rs = res_S[ok]
        m = (R > 5.0) & (R < R[-1])       # the pin radius is identically zero
        cors = [np.corrcoef(res_T[:, j], rs[:, j])[0, 1]
                for j in np.where(m)[0]]
        print(f"  {label:12s} (n={ok.sum()}): per-radius corr R>5 kpc "
              f"min/med/max = {np.min(cors):.2f}/{np.median(cors):.2f}/"
              f"{np.max(cors):.2f}   (exp33 record 0.82-0.89)")
        out[label] = (res_T, rs, ok)
    return out


def part2_levers(res_pair, X):
    print("\n=== phase 0.2 — what drives the UNSHARED transport residual ===")
    res_T, res_S, ok = res_pair
    m = (R > 5.0) & (R < R[-1])           # the pin radius is identically zero
    rT, rS = res_T[:, m], res_S[:, m]
    # remove the shared component per radius (OLS slope), keep the remainder
    b = (rT * rS).sum(0) / (rS * rS).sum(0)
    u = rT - b[None, :] * rS
    _, _, Vt = np.linalg.svd(u - u.mean(0), full_matrices=False)
    pcs = (u - u.mean(0)) @ Vt[:3].T
    levers = {
        "logmh": POP["logmh"][ok], "c200c": POP["c200c"][ok],
        "burst": POP["burst"][ok], "t50": POP["t50"][ok],
        "fz2": POP["fz2"][ok], "dmah_late": X[ok, 3],
    }
    rng = np.random.default_rng(0)
    print("  R^2 of each unshared-residual PC on each lever "
          "(* = above the 95th pct of 200 shuffles):")
    print("    lever      " + "  ".join(f"   PC{i+1}  " for i in range(3))
          + "  [PC var frac: "
          + "/".join(f"{v:.2f}" for v in
                     (np.linalg.svd(u - u.mean(0), compute_uv=False)[:3] ** 2
                      / (np.linalg.svd(u - u.mean(0), compute_uv=False) ** 2).sum()))
          + "]")
    for name, z in levers.items():
        good = np.isfinite(z)
        cells = []
        for i in range(3):
            y = pcs[good, i]
            zz = z[good]
            r2 = np.corrcoef(zz, y)[0, 1] ** 2
            shuf = [np.corrcoef(rng.permutation(zz), y)[0, 1] ** 2
                    for _ in range(200)]
            flag = "*" if r2 > np.percentile(shuf, 95) else " "
            cells.append(f"{r2:6.3f}{flag}")
        print(f"    {name:10s} " + "  ".join(cells))


_P3 = {}


def _p3_init(rows):
    """Spawn-safe worker init: each child loads exp35 fresh by path (the
    parent's open npz handles must NOT be shared across forks)."""
    e = _load_by_path("exp35_run", E35_DIR / "run.py")
    mh_scale = tuple(np.load(ROOT / "experiments/exp32_full_population/outputs/"
                             "um_slope_diffmah.npz")["mh_scale"])
    e._init("diffmah", mh_scale, np.asarray(rows))
    _P3["e"] = e


def _p3_chunk(args):
    p, lo, hi = args
    e = _P3["e"]
    return sum(e.gal_loss(e.theta_of(p, g, "global"), g, [0])
               for g in e._G["gals"][lo:hi])


def part3_massive(data):
    from scipy.optimize import minimize
    print("\n=== phase 0.3 — massive-tercile probe (exp35 form, refit alone) ===")
    logms = POP["logms"]
    rows = np.where(logms >= np.quantile(logms, 2.0 / 3.0))[0]
    n = len(rows)
    workers = max(os.cpu_count() - 2, 2)
    edges = np.linspace(0, n, workers + 1).astype(int)
    _p3_init(rows)                                   # parent copy for eval
    e35 = _P3["e"]
    t0 = time.time()
    with Pool(workers, initializer=_p3_init, initargs=(rows,)) as pool:
        def loss(p):
            parts = pool.map(_p3_chunk, [(p, edges[i], edges[i + 1])
                                         for i in range(workers)])
            return sum(parts) / n + e35.penalty(p)

        best = None
        for p0 in e35.STARTS:
            r = minimize(loss, p0, method="Nelder-Mead",
                         options=dict(maxiter=1500, xatol=3e-4, fatol=1e-8))
            print(f"  [massive-z04] start: loss {r.fun:.4f} ({r.nit} iters, "
                  f"{(time.time()-t0)/60:.1f} min)", flush=True)
            if best is None or r.fun < best.fun:
                best = r
    th, lo = best.x, best.fun
    gals = e35._G["gals"]
    met = np.stack([e35.gal_eval(e35.theta_of(th, g, "global"), g, [0])[0]
                    for g in gals])
    pop_met = TN["met_multi-slope"][rows, 0]
    print("  fitted theta (massive-only, z04-global): "
          + " ".join(f"{v:.2f}" for v in th)
          + f"   loss {lo:.4f}  ({(time.time()-t0)/60:.1f} min)")
    print(f"  dlog f148, massive tercile: population multi-slope fit "
          f"{np.nanmedian(pop_met[:, 3]):+.4f} dex -> massive-only refit "
          f"{np.nanmedian(met[:, 0, 3]):+.4f} dex   (0 = the data fraction)")
    print(f"  shape R>5 max|rel|: population {100*np.nanmedian(pop_met[:, 2]):.1f}% "
          f"-> massive-only {100*np.nanmedian(met[:, 0, 2]):.1f}% (in-sample)")


if __name__ == "__main__":
    X, Y, data = _EPOCHS.load()
    print(f"exp36 phase 0 — residual anatomy (n={len(X)}, z=0.4)")
    pairs = part1_wall(X, data)
    part2_levers(pairs["multi-slope"], X)
    if "--skip-fit" not in sys.argv:
        part3_massive(data)
