"""exp33 — the PHYSICAL reparameterization of the transport kernel (prior-side fix).

The population fits exploit the aperture-horizon degeneracy: efficiency f(z)
and deposit width multiply into one observable, so the optimizer deletes whole
epochs geometrically (widths >> aperture) instead of via efficiency — an
unphysical basin the data inside 150 kpc cannot veto. Fix the parameterization:

  theta_phys = [log_s0, g, q, mu, sig]     (5 params; was 7)
    alpha == 1 FIXED       (self-similar migration clock, confirmed 3x)
    f(z) = lognormal peak  exp(-(ln(1+z)-mu)^2 / 2 sig^2)  (no railing z_c,
                           no unconstrained early slope)
    soft box               log_s0 in [1.0, 2.4] (10-250 kpc), g in [0.5, 2.5],
                           q in [0, 2], sig in [0.1, 1.5], mu in [0.3, 2.6]
                           (the n=45 per-galaxy PHYSICAL basin)

Fits (diffmah, n=2397, 10-fold CV): z=0.4-only {global, +logMh slopes} and
multi-epoch {global}. Success = (a) held-out accuracy ~ unconstrained
(16.1/19.0/19.1%), (b) deposits VISIBLE (per-z-bin diagnostic), (c) the
z=0.4-only and multi-epoch thetas land in the SAME basin (epoch stability).

Run:  PYTHONPATH=. uv run python experiments/exp33_single_epoch/physical_theta.py
Demo: ... physical_theta.py demo
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
for sub in ("exp29_outer_deposit", "exp30_transport_kernel", "exp32_full_population"):
    sys.path.insert(0, str(ROOT / "experiments" / sub))
sys.path.insert(0, str(ROOT))
import param_emulator as pe                                                          # noqa: E402
import universal_mass as um                                                          # noqa: E402
from run import ANCHOR_SNAP                                                          # noqa: E402

OUTDIR = HERE / "outputs"
OUT_NPZ = OUTDIR / "physical_theta.npz"
NFOLD = 10
LO = np.array([1.0, 0.5, 0.0, 0.3, 0.1])
HI = np.array([2.4, 2.5, 2.0, 2.6, 1.5])
STARTS = [np.array([2.0, 1.5, 0.77, np.log(3.2), 0.5]),
          np.array([2.2, 1.0, 1.2, np.log(2.4), 0.8])]
_G = {}


def penalty(p):
    v = np.asarray(p[:5])
    return 30.0 * float(np.sum(np.clip(LO - v, 0, None) ** 2
                               + np.clip(v - HI, 0, None) ** 2))


def weights(z, mu, sig):
    return np.exp(-((np.log1p(z) - mu) ** 2) / (2.0 * sig ** 2))


def model_cogs_phys(p, mah, data, ks):
    """Pinned CoGs for epochs ks with alpha=1 and lognormal efficiency."""
    w = weights(mah["z"], p[3], p[4])
    dM = w * mah["dMh"]
    if not np.isfinite(dM).all() or dM.sum() <= 0:
        return None
    dM = dM / dM.sum()
    th4 = [p[0], p[1], 0.0, p[2]]                       # log_alpha = 0 -> alpha = 1
    out = []
    for k in ks:
        mask = mah["snap"] <= ANCHOR_SNAP[k]
        B = pe.tf.basis(th4, mah["t"], mah["t_obs"], pe.AT[k], "dyntrans")
        m = B @ (dM * mask)
        if not np.isfinite(m[-1]) or m[-1] <= 0:
            return None
        out.append(m * (data[k][-1] / m[-1]))
    return out


def gal_loss(p, g, ks):
    cogs = model_cogs_phys(p, g["mah"], g["data"], ks)
    if cogs is None:
        return 4.0
    return float(np.mean([np.sqrt(np.mean(((c - g["data"][k]) / g["data"][k]) ** 2))
                          for c, k in zip(cogs, ks)]))


def gal_maxrel(p, g, ks, rmin=5.0):
    cogs = model_cogs_phys(p, g["mah"], g["data"], ks)
    if cogs is None:
        return np.full(len(ks), np.nan)
    m = pe.R > rmin
    return np.array([np.abs((c[m] - g["data"][k][m]) / g["data"][k][m]).max()
                     for c, k in zip(cogs, ks)])


def theta_of(p, g, variant):
    if variant == "global":
        return np.asarray(p[:5])
    return np.asarray(p[:5]) + np.asarray(p[5:10]) * \
        (g["logmh"] - _G["mh_scale"][0]) / _G["mh_scale"][1]


def _chunk(args):
    p, variant, ks, lo, hi = args
    return sum(gal_loss(theta_of(p, g, variant), g, ks)
               for g in _G["gals"][lo:hi])


def fit_pop(variant, ks, starts, maxiter, pool, nchunk, label):
    n = len(_G["gals"])
    edges = np.linspace(0, n, nchunk + 1).astype(int)

    def loss(p):
        parts = pool.map(_chunk, [(p, variant, ks, edges[i], edges[i + 1])
                                  for i in range(nchunk)])
        return sum(parts) / n + penalty(p)

    best = None
    for p0 in starts:
        t0 = time.time()
        r = minimize(loss, p0, method="Nelder-Mead",
                     options=dict(maxiter=maxiter, xatol=3e-4, fatol=1e-8))
        print(f"  [{label}] start: loss {r.fun:.4f} ({(time.time()-t0)/60:.1f} min, "
              f"{r.nit} iters)", flush=True)
        if best is None or r.fun < best.fun:
            best = r
    return best.x, best.fun


def _init(config, mh_scale):
    _G["gals"] = um.load_gals(config)
    _G["mh_scale"] = mh_scale


def _cv_fold(args):
    variant, ks, fold, warm, mh_scale = args
    _G["mh_scale"] = mh_scale
    gals = _G["gals"]
    n = len(gals)
    train = [i for i in range(n) if i % NFOLD != fold]
    r = minimize(lambda p: np.mean([gal_loss(theta_of(p, gals[i], variant),
                                             gals[i], ks) for i in train])
                 + penalty(p),
                 warm, method="Nelder-Mead",
                 options=dict(maxiter=1200, xatol=3e-4, fatol=1e-8))
    return [(gals[i]["row"], gal_maxrel(theta_of(r.x, gals[i], variant), gals[i], ks))
            for i in range(n) if i % NFOLD == fold]


def physicality(p, label):
    """Per-z-bin efficiency weight + visible-within-150kpc fraction diagnostic."""
    from deposit import eff_two_epoch                                    # noqa: F401
    g = _G["gals"][3]
    mah = g["mah"]
    s0, gexp = 10.0 ** p[0], p[1]
    sig0 = np.clip(s0 * (mah["t"] / mah["t_obs"]) ** gexp, 1e-4, 1e5)
    w = weights(mah["z"], p[3], p[4])
    dM = w * mah["dMh"]
    dM = dM / dM.sum()
    vis = 1.0 - np.exp(-150.0 ** 2 / (2.0 * sig0 ** 2))
    print(f"    [{label}] peak z = {np.expm1(p[3]):.1f}, width sig = {p[4]:.2f}; "
          "per z-bin (weight | visible):")
    for zlo, zhi in ((0.4, 1), (1, 2), (2, 4), (4, 12)):
        m = (mah["z"] >= zlo) & (mah["z"] < zhi)
        if m.sum():
            print(f"      z [{zlo},{zhi}): {dM[m].sum():.3f} | "
                  f"{np.median(vis[m]):.3f}")


def main():
    config = "diffmah"
    workers = max(os.cpu_count() - 2, 2)
    d32s = np.load(ROOT / "experiments/exp32_full_population/outputs/um_slope_diffmah.npz")
    mh_scale = tuple(d32s["mh_scale"])
    _init(config, mh_scale)
    n = len(_G["gals"])
    print(f"physical 5-param transport (alpha=1, lognormal f(z), bounded widths; "
          f"n={n})\n")

    jobs = [("z04-global", "global", [0], STARTS, 2500),
            ("z04-slope", "slope", [0],
             [np.concatenate([STARTS[0], np.zeros(5)])], 5000),
            ("multi-global", "global", [0, 1, 2, 3, 4], STARTS, 2500)]
    fitted = {}
    with Pool(workers, initializer=_init, initargs=(config, mh_scale)) as pool:
        for label, variant, ks, starts, mi in jobs:
            th, lo = fit_pop(variant, ks, starts, mi, pool, workers, label)
            fitted[label] = (th, variant, ks)
            physicality(th[:5], label)

    print("\n  10-fold CV (held-out shape max|rel| R>5 kpc):")
    res = {}
    for label, (th, variant, ks) in fitted.items():
        t0 = time.time()
        mr = np.full((n, len(ks)), np.nan)
        with Pool(min(workers, NFOLD), initializer=_init,
                  initargs=(config, mh_scale)) as pool2:
            for out in pool2.imap_unordered(
                    _cv_fold, [(variant, ks, f, th, mh_scale) for f in range(NFOLD)]):
                for row_, m_ in out:
                    i = next(k for k, g in enumerate(_G["gals"]) if g["row"] == row_)
                    mr[i] = m_
        res[label] = mr
        line = " ".join(f"z{k}:{100*np.nanmedian(mr[:, j]):.1f}%"
                        for j, k in enumerate(ks))
        print(f"    {label:>13s}: {line}  ({(time.time()-t0)/60:.1f} min)", flush=True)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_NPZ, **{f"theta_{k}": v[0] for k, v in fitted.items()},
             **{f"mr_{k}": res[k] for k in res})
    print("\n  reference (unconstrained 7-param): z04 slope 16.1%, z04 global "
          "19.0%, multi slope 19.1% at z=0.4")
    print(f"wrote {OUT_NPZ}")


def demo():
    """Self-check: alpha=1 path == pe.model_cogs with log_alpha=0 and matching
    two-epoch weights; penalty active outside the box, zero inside."""
    _init("diffmah", (13.5, 0.4))
    g = _G["gals"][0]
    p = np.array([2.0, 1.5, 0.77, np.log(3.2), 0.5])
    assert penalty(p) == 0.0
    assert penalty(np.array([5.0, 1.5, 0.77, 1.0, 0.5])) > 1.0
    # cross-check the transport core: same widths/clock as pe with alpha=1
    th7 = np.array([2.0, 1.5, 0.0, 0.77, 4.48, 1.88, 2.23])
    full, _ = pe.model_cogs(th7, g["mah"], g["data"])
    from deposit import eff_two_epoch
    w_ref = eff_two_epoch(g["mah"]["z"], 4.48, 1.88, 2.23)
    ours = model_cogs_phys(p, g["mah"], g["data"], [0])
    # replace lognormal weights by the reference two-epoch weights -> exact match
    import types
    global weights
    orig = weights
    weights = lambda z, mu, sig: w_ref                                   # noqa: E731
    try:
        exact = model_cogs_phys(p, g["mah"], g["data"], [0, 3])
        assert np.allclose(exact[0], full[0], rtol=1e-10)
        assert np.allclose(exact[1], full[3], rtol=1e-10)
    finally:
        weights = orig
    assert ours is not None and np.isfinite(ours[0]).all()
    print("physical_theta.demo OK: alpha=1 core exact vs pe.model_cogs; "
          "penalty box sane; lognormal path finite")


if __name__ == "__main__":
    if sys.argv[1:2] == ["demo"]:
        demo()
    else:
        sys.exit(main())
