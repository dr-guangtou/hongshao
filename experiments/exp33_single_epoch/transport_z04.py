"""exp33 — the missing 2x2 cell: the transport kernel fitted to z=0.4 ONLY.

Every transport number so far came from a theta fitted jointly to all five
epochs; the head-to-head's ~3.5-point z=0.4 shape gap vs the statistical model
therefore conflates the kernel FORM with the multi-epoch CONSISTENCY burden.
Here the same population forms (global 7-param theta, and theta(logMh) slope)
are fitted against the z=0.4 CoGs alone (10-fold CV, same protocol), completing
{statistical, transport} x {single-epoch, multi-epoch}:

  single-epoch transport ~ statistical (15-16%)  -> the form is expressive;
                                                    the whole gap is consistency
  single-epoch transport stays ~ 19%             -> the kernel form itself is
                                                    the restrictive ingredient

Run:  PYTHONPATH=. uv run python experiments/exp33_single_epoch/transport_z04.py
Demo: ... transport_z04.py demo
"""
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP29 = ROOT / "experiments" / "exp29_outer_deposit"
EXP30 = ROOT / "experiments" / "exp30_transport_kernel"
EXP32 = ROOT / "experiments" / "exp32_full_population"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP29))
sys.path.insert(0, str(EXP30))
sys.path.insert(0, str(EXP32))
import param_emulator as pe                                                          # noqa: E402
import universal_mass as um                                                          # noqa: E402
from deposit import eff_two_epoch                                                    # noqa: E402
from run import ANCHOR_SNAP                                                          # noqa: E402

OUTDIR = HERE / "outputs"
OUT_NPZ = OUTDIR / "transport_z04.npz"
NFOLD = 10
_G = {}


def model_cog_z04(theta, mah, data):
    """Epoch-0 (z=0.4) pinned transport CoG only — ~5x cheaper than model_cogs."""
    if theta[6] <= -1.0:
        return None
    with np.errstate(over="ignore", invalid="ignore"):
        w = eff_two_epoch(mah["z"], theta[4], theta[5], theta[6])
    dM = w * mah["dMh"]
    if not np.isfinite(dM).all() or dM.sum() <= 0:
        return None
    dM = dM / dM.sum()
    mask = mah["snap"] <= ANCHOR_SNAP[0]
    B = pe.tf.basis(theta[:4], mah["t"], mah["t_obs"], pe.AT[0], "dyntrans")
    m = B @ (dM * mask)
    if not np.isfinite(m[-1]) or m[-1] <= 0:
        return None
    return m * (data[0][-1] / m[-1])


def gal_loss_z04(theta, g):
    c = model_cog_z04(theta, g["mah"], g["data"])
    if c is None:
        return 4.0
    D = g["data"][0]
    return float(np.sqrt(np.mean(((c - D) / D) ** 2)))


def gal_maxrel_z04(theta, g, rmin=0.0):
    c = model_cog_z04(theta, g["mah"], g["data"])
    if c is None:
        return np.nan
    D = g["data"][0]
    m = pe.R > rmin
    return float(np.abs((c[m] - D[m]) / D[m]).max())


def theta_of(p, g, variant):
    if variant == "global":
        return np.asarray(p[:7])
    return np.asarray(p[:7]) + np.asarray(p[7:14]) * \
        (g["logmh"] - _G["mh_scale"][0]) / _G["mh_scale"][1]


def _chunk(args):
    p, variant, lo, hi = args
    return sum(gal_loss_z04(theta_of(p, g, variant), g)
               for g in _G["gals"][lo:hi])


def fit_z04(variant, starts, maxiter, pool, nchunk):
    n = len(_G["gals"])
    edges = np.linspace(0, n, nchunk + 1).astype(int)

    def loss(p):
        parts = pool.map(_chunk, [(p, variant, edges[i], edges[i + 1])
                                  for i in range(nchunk)])
        return sum(parts) / n

    best = None
    for p0 in starts:
        t0 = time.time()
        r = minimize(loss, p0, method="Nelder-Mead",
                     options=dict(maxiter=maxiter, xatol=3e-4, fatol=1e-8))
        print(f"  [{variant}] start: loss {r.fun:.4f} ({(time.time()-t0)/60:.1f} min, "
              f"{r.nit} iters)", flush=True)
        if best is None or r.fun < best.fun:
            best = r
    return best.x, best.fun


def _init(config, mh_scale):
    _G["gals"] = um.load_gals(config)
    _G["mh_scale"] = mh_scale


def _cv_fold(args):
    variant, fold, warm, mh_scale = args
    _G["mh_scale"] = mh_scale
    gals = _G["gals"]
    n = len(gals)
    train = [i for i in range(n) if i % NFOLD != fold]
    r = minimize(lambda p: np.mean([gal_loss_z04(theta_of(p, gals[i], variant),
                                                 gals[i]) for i in train]),
                 warm, method="Nelder-Mead",
                 options=dict(maxiter=1200, xatol=3e-4, fatol=1e-8))
    out = []
    for i in range(n):
        if i % NFOLD == fold:
            th = theta_of(r.x, gals[i], variant)
            out.append((gals[i]["row"], gal_maxrel_z04(th, gals[i], 0.0),
                        gal_maxrel_z04(th, gals[i], qa_rmin())))
    return out


def qa_rmin():
    from hongshao import qa
    return qa.RMIN_KPC


def main():
    config = "diffmah"
    workers = max(os.cpu_count() - 2, 2)
    d32g = np.load(EXP32 / "outputs" / "um_global_diffmah.npz")
    d32s = np.load(EXP32 / "outputs" / "um_slope_diffmah.npz")
    mh_scale = tuple(d32s["mh_scale"])
    _init(config, mh_scale)
    n = len(_G["gals"])
    print(f"transport kernel fitted to z=0.4 ONLY (n={n}, {workers} workers)\n")

    with Pool(workers, initializer=_init, initargs=(config, mh_scale)) as pool:
        th_g, lo_g = fit_z04("global", [d32g["theta"], um.PHYS], 3000, pool, workers)
        th_s, lo_s = fit_z04("slope",
                             [np.concatenate([th_g, np.zeros(7)]),
                              d32s["p"]], 6000, pool, workers)
    print(f"  fitted: global loss {lo_g:.4f}, slope loss {lo_s:.4f}")

    res = {}
    for variant, warm in (("global", th_g), ("slope", th_s)):
        t0 = time.time()
        mr = np.full((n, 2), np.nan)
        with Pool(min(workers, NFOLD), initializer=_init,
                  initargs=(config, mh_scale)) as pool2:
            for out in pool2.imap_unordered(
                    _cv_fold, [(variant, f, warm, mh_scale) for f in range(NFOLD)]):
                for row_, m_all, m_out in out:
                    i = next(k for k, g in enumerate(_G["gals"]) if g["row"] == row_)
                    mr[i] = (m_all, m_out)
        res[variant] = mr
        print(f"  cv {variant}: z=0.4 shape max|rel| "
              f"{100*np.nanmedian(mr[:,1]):.1f}% (R>5) "
              f"{100*np.nanmedian(mr[:,0]):.1f}% (all)  "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_NPZ, theta_global=th_g, theta_slope=th_s,
             mr_global=res["global"], mr_slope=res["slope"])
    print("\n  the 2x2 (z=0.4 shape max|rel|, R>5 kpc, held-out):")
    print("    statistical single-epoch : 15.6%   (exp33 iv)")
    print(f"    transport single-epoch   : {100*np.nanmedian(res['slope'][:,1]):.1f}%")
    print("    transport multi-epoch    : 19.1%   (exp33 iv, slope)")
    print(f"wrote {OUT_NPZ}")


def demo():
    """Self-check: model_cog_z04 must equal row 0 of pe.model_cogs exactly."""
    _init("diffmah", (13.5, 0.4))
    g = _G["gals"][0]
    th = np.array([2.0, 1.5, 0.0, 0.77, 4.48, 1.88, 2.23])
    full, _ = pe.model_cogs(th, g["mah"], g["data"])
    fast = model_cog_z04(th, g["mah"], g["data"])
    assert np.allclose(fast, full[0], rtol=1e-12), "z04 fast path must match"
    l0 = gal_loss_z04(th, g)
    assert 0 < l0 < 2
    print(f"transport_z04.demo OK: fast epoch-0 path exact (loss {l0:.3f})")


if __name__ == "__main__":
    if sys.argv[1:2] == ["demo"]:
        demo()
    else:
        sys.exit(main())
