"""exp29 Phase 4 — population / forward emulator.

Phases 1-2 fit the deposit fraction to each galaxy's OWN profiles. A true forward
model must get the fraction from the halo. We ask:

  4a  Is the fraction ~universal? Fit the SAME parametrization (K-knot log f(t) in
      log-cosmic-time + power-law width) per galaxy; look at the parameter scatter
      and its correlation with logMh / logM*.
  4b  Does a SHARED model generalize? Fit ONE (f, width) across a training set
      (each galaxy uses its own dip-free MAH; total pinned to its z=0.4 M*), then
      evaluate the multi-epoch CoG RMS on a held-out TEST set. This is the pure
      MAH -> multi-epoch-profile forward emulator (no per-galaxy profile fit).
  4c  Does halo-conditioning help? Let the shared params carry a logMh slope.

Phenomenological: f is a deposit fraction, not an SFH. Normalization pinned to the
measured z=0.4 total, so only the SHAPE of f(t) and the width law are fit.

Run: PYTHONPATH=. uv run python experiments/exp29_outer_deposit/population.py [n]
"""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from run import dipfree_mah, ANCHOR_SNAP, ANCHOR_Z, OFFICIAL, DIFFPROF, TABLE       # noqa: E402
from cog_extrapolate import measured_cog                                           # noqa: E402
from hongshao.tng_data import COG_RAD_KPC                                          # noqa: E402
from astropy.table import Table                                                    # noqa: E402

R2 = COG_RAD_KPC ** 2 / 2.0                       # (24,) precomputed
EPOCH_SNAP = np.array(ANCHOR_SNAP)
K = 4                                             # log-f knots
TKNOTS = np.linspace(np.log10(0.35), np.log10(9.4), K)   # common log10(t) axis


def load_pop(n_max=300):
    """Per-galaxy arrays for the forward model (matched MAH + valid 5-epoch CoG)."""
    t = Table.read(TABLE); cog = np.asarray(t["logmstar_cog"], float)
    idx = np.asarray(t["index"]); use = np.asarray(t["use"])
    logmh_tab = np.asarray(t["logmh_z0p4"], float)
    r50_tab = np.nanmedian(np.asarray(t["r50_proj"], float), axis=1)   # kpc, over 3 projections
    ok = np.isfinite(cog).all(1) & use
    order = np.argsort(np.where(ok, cog[:, -1], -np.inf))[::-1]
    mz = np.load(OFFICIAL); matched = {int(g) for g, m in zip(mz["index"], mz["matched"]) if m}
    gals = []
    for i in order:
        gi = int(idx[i])
        if gi not in matched:
            continue
        mah = dipfree_mah(gi); logC = measured_cog(gi)
        if mah is None or logC is None:
            continue
        keep = mah["snap"] <= 72                  # snap/t/dMh already aligned by dipfree_mah
        snaps, tt, dMh = mah["snap"][keep], mah["t"][keep], mah["dMh"][keep]
        gate = np.array([(snaps <= sk) for sk in EPOCH_SNAP])            # (5, N)
        # halo formation time t50 and early-assembly fraction fz2 from the dip-free MAH
        mk = mah["snap_full"] <= 72
        Mh, tf = 10.0 ** mah["logMh_full"][mk], mah["t_full"][mk]
        t50 = float(np.interp(0.5 * Mh[-1], Mh, tf))
        fz2 = float(10.0 ** (np.interp(33, mah["snap_full"], mah["logMh_full"]) - np.log10(Mh[-1])))
        gals.append(dict(snaps=snaps, t=tt, lt=np.log10(tt), dMh=dMh, gate=gate,
                         t_obs=mah["t_obs"], Mstar_tot=10.0 ** logC[0, -1], logC=logC,
                         logMh=float(logmh_tab[i]), logMstar=float(logC[0, -1]),
                         logR50=float(np.log10(r50_tab[i])), t50=t50, fz2=fz2))
        if len(gals) >= n_max:
            break
    return gals


def predict(theta, gal):
    """5x24 model CoG. theta = [logf_1..logf_{K-1}, log_sigma0, g]; logf_0 ≡ 0."""
    logf_knots = np.concatenate([[0.0], theta[:K - 1]])
    logf = np.interp(gal["lt"], TKNOTS, logf_knots)
    dMstar = 10.0 ** logf * gal["dMh"]
    dMstar = dMstar * (gal["Mstar_tot"] / dMstar.sum())
    widths = np.clip((10.0 ** theta[K - 1]) * (gal["t"] / gal["t_obs"]) ** theta[K], 0.3, 500)
    kappa = 1.0 - np.exp(-R2[:, None] / widths[None, :] ** 2)            # (24, N)
    return np.array([kappa @ (dMstar * g) for g in gal["gate"]])        # (5, 24)


def gal_rms(theta, gal):
    pred = predict(theta, gal)
    return float(np.mean([np.sqrt(np.mean((np.log10(np.clip(pred[k], 1.0, None)) - gal["logC"][k]) ** 2))
                          for k in range(5)]))


def epoch_rms(theta, gal):
    pred = predict(theta, gal)
    return np.array([np.sqrt(np.mean((np.log10(np.clip(pred[k], 1.0, None)) - gal["logC"][k]) ** 2))
                     for k in range(5)])


def fit_one(gal):
    best = None
    for g0 in (0.5, 1.5):
        r = minimize(lambda th: gal_rms(th, gal), [0.0] * (K - 1) + [np.log10(40.0), g0],
                     method="Nelder-Mead", options=dict(maxiter=6000, xatol=1e-4, fatol=1e-8))
        if best is None or r.fun < best.fun:
            best = r
    return best.x, best.fun


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    gals = load_pop(n)
    rng = np.arange(len(gals)); train = rng[rng % 2 == 0]; test = rng[rng % 2 == 1]
    print(f"exp29 Phase 4 — n={len(gals)} galaxies ({len(train)} train / {len(test)} test)\n")

    # ---- 4a: per-galaxy fits -> is the fraction universal? ----
    pars = np.array([fit_one(gals[i])[0] for i in rng])
    rms_self = np.array([gal_rms(pars[i], gals[i]) for i in rng])
    logMh = np.array([g["logMh"] for g in gals]); logMs = np.array([g["logMstar"] for g in gals])
    names = [f"logf{j+1}" for j in range(K - 1)] + ["log_s0", "g"]
    print("4a per-galaxy fit (same parametrization): median param, scatter, r(logMh), r(logM*)")
    for j, nm in enumerate(names):
        p = pars[:, j]
        rh = np.corrcoef(p, logMh)[0, 1]; rs = np.corrcoef(p, logMs)[0, 1]
        print(f"    {nm:8s}  med={np.median(p):+6.2f}  [16,84]=[{np.percentile(p,16):+.2f},"
              f"{np.percentile(p,84):+.2f}]  r(logMh)={rh:+.2f}  r(logM*)={rs:+.2f}")
    print(f"    per-galaxy median 5-epoch RMS = {np.median(rms_self):.3f} dex")

    # ---- 4b: one SHARED forward model, train -> test ----
    def shared_loss(theta, ids):
        return float(np.mean([gal_rms(theta, gals[i]) for i in ids]))
    best = None
    for g0 in (0.5, 1.5):
        r = minimize(lambda th: shared_loss(th, train), [0.0] * (K - 1) + [np.log10(40.0), g0],
                     method="Nelder-Mead", options=dict(maxiter=8000, xatol=1e-4, fatol=1e-7))
        if best is None or r.fun < best.fun:
            best = r
    th = best.x
    er_tr = np.median([epoch_rms(th, gals[i]) for i in train], 0)
    er_te = np.median([epoch_rms(th, gals[i]) for i in test], 0)
    print(f"\n4b SHARED forward model (MAH -> profiles, no per-galaxy fit): params "
          f"{np.array2string(th, precision=2)}")
    print(f"    median epoch RMS  train: " + " ".join(f"{x:5.3f}" for x in er_tr) +
          f"  (mean {er_tr.mean():.3f})")
    print(f"    median epoch RMS  TEST : " + " ".join(f"{x:5.3f}" for x in er_te) +
          f"  (mean {er_te.mean():.3f})")
    print(f"    cost of universality: per-galaxy {np.median(rms_self):.3f} -> "
          f"shared(test) {np.median([gal_rms(th, gals[i]) for i in test]):.3f} dex")

    # ---- 4c: halo-conditioned (logMh slope on each shared param) ----
    mh0 = np.median(logMh)
    def cond_theta(phi, gi_logMh):
        return np.array([phi[2 * j] + phi[2 * j + 1] * (gi_logMh - mh0) for j in range(K + 1)])
    def cond_loss(phi, ids):
        return float(np.mean([gal_rms(cond_theta(phi, gals[i]["logMh"]), gals[i]) for i in ids]))
    phi0 = np.ravel([[v, 0.0] for v in th])
    rc = minimize(lambda p: cond_loss(p, train), phi0, method="Nelder-Mead",
                  options=dict(maxiter=12000, xatol=1e-4, fatol=1e-7))
    er_te_c = np.median([epoch_rms(cond_theta(rc.x, gals[i]["logMh"]), gals[i]) for i in test], 0)
    print(f"\n4c halo-conditioned (logMh slope): median epoch RMS TEST " +
          " ".join(f"{x:5.3f}" for x in er_te_c) + f"  (mean {er_te_c.mean():.3f})")
    print(f"    vs shared {er_te.mean():.3f} -> conditioning gain {er_te.mean()-er_te_c.mean():+.3f} dex")


if __name__ == "__main__":
    sys.exit(main())
