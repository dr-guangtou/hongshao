"""exp29 Phase 4e — LEGAL forward emulator: halo-only inputs.

Phase 4d conditioned on R50 (galaxy size) and pinned the amplitude to the measured
M*(z=0.4). Both are stellar/galaxy quantities, NOT available when applying the
emulator to a fresh N-body run. A legitimate forward emulator may use ONLY:
  - the halo MAH, and
  - halo-only secondaries (concentration c_200c, accretion rate, MAH shape t50/
    f_early...), all measurable from the N-body halo catalog.
The stellar amplitude M*(z=0.4) is supplied by a separate SHMR (also halo-derived),
so we report the profile-SHAPE+growth emulator given M*_tot, and separately the
SHMR scatter that the amplitude step would add.

Compare (median TEST 5-epoch CoG RMS):
  U   universal (no conditioning)
  Lc  width <- c_200c            (legal replacement for R50)
  Lf  fraction <- t50            (legal; halo formation time)
  L   both legal
  [ref] WF illegal (R50 + t50)   from Phase 4d cache

Run: PYTHONPATH=. uv run python experiments/exp29_outer_deposit/phase4e.py [n]
"""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from population import load_pop, gal_rms, epoch_rms, fit_one                        # noqa: E402
from run import ANCHOR_Z                                                            # noqa: E402

ZK = {"c200c": "c200c", "t50": "t50"}


def standardize(gals):
    for zname, src in [("z_c200c", "c200c"), ("z_t50", "t50")]:
        v = np.array([g[src] for g in gals]); mu, sd = np.nanmean(v), np.nanstd(v) + 1e-9
        for g in gals:
            g[zname] = float((g[src] - mu) / sd) if np.isfinite(g[src]) else 0.0


def theta_of(p, gal, model):
    if model == "U":
        return p[:5]
    if model == "Lc":                              # width <- c_200c
        return np.array([p[0], p[1], p[2], p[3] + p[4] * gal["z_c200c"], p[5] + p[6] * gal["z_c200c"]])
    if model == "Lf":                              # fraction <- t50
        z = gal["z_t50"]
        return np.array([p[0] + p[1] * z, p[2] + p[3] * z, p[4] + p[5] * z, p[6], p[7]])
    if model == "L":                               # both legal
        zc, zt = gal["z_c200c"], gal["z_t50"]
        return np.array([p[0] + p[1] * zt, p[2] + p[3] * zt, p[4] + p[5] * zt,
                         p[6] + p[7] * zc, p[8] + p[9] * zc])
    raise ValueError(model)


def fit_shared(gals, ids, model, p0):
    def loss(p):
        return float(np.mean([gal_rms(theta_of(p, gals[i], model), gals[i]) for i in ids]))
    best = None
    for jit in (0.0, 0.15):
        r = minimize(loss, np.array(p0) + jit, method="Nelder-Mead",
                     options=dict(maxiter=2000 * len(p0), xatol=1e-4, fatol=1e-7))
        if best is None or r.fun < best.fun:
            best = r
    return best.x


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    gals = load_pop(n); standardize(gals)
    rng = np.arange(len(gals)); train = rng[rng % 2 == 0]; test = rng[rng % 2 == 1]
    print(f"exp29 Phase 4e — LEGAL halo-only forward (n={len(gals)}, {len(train)}/{len(test)})\n")

    U = fit_shared(gals, train, "U", [0, 0, 0, np.log10(40.0), 1.5])
    models = {"U universal": (U, "U"),
              "Lc width<-c200c": (fit_shared(gals, train, "Lc", [U[0], U[1], U[2], U[3], 0, U[4], 0]), "Lc"),
              "Lf fraction<-t50": (fit_shared(gals, train, "Lf", [U[0], 0, U[1], 0, U[2], 0, U[3], U[4]]), "Lf"),
              "L both (legal)": (fit_shared(gals, train, "L", [U[0], 0, U[1], 0, U[2], 0, U[3], 0, U[4], 0]), "L")}
    print("median TEST epoch RMS (legal halo-only inputs; M*_tot given):")
    for label, (p, m) in models.items():
        er = np.median([epoch_rms(theta_of(p, gals[i], m), gals[i]) for i in test], 0)
        print(f"    {label:18s}: " + " ".join(f"{x:5.3f}" for x in er) + f"  (mean {er.mean():.3f})")

    # reference: illegal WF (R50 + t50) from Phase 4d cache
    cache = HERE / "outputs" / "p4d_cache.npz"
    if cache.exists():
        import phase4d
        z = np.load(cache, allow_pickle=True)
        phase4d.standardize(gals)
        er = np.median([phase4d.epoch_rms_wrap(z["p_WF"], gals[i]) for i in test], 0) \
            if hasattr(phase4d, "epoch_rms_wrap") else \
            np.median([epoch_rms(phase4d.theta_of(z["p_WF"], gals[i], "WF"), gals[i]) for i in test], 0)
        print(f"    [ref] WF ILLEGAL (R50+t50): " + " ".join(f"{x:5.3f}" for x in er) +
              f"  (mean {er.mean():.3f})")

    # SHMR amplitude: what the M*(z=0.4)-from-halo step would add (separable)
    logMh = np.array([g["logMh"] for g in gals]); logMs = np.array([g["logMstar"] for g in gals])
    b = np.polyfit(logMh[train], logMs[train], 1)
    resid = logMs[test] - np.polyval(b, logMh[test])
    print(f"\n  SHMR amplitude (separable, halo-derived): logM* = {b[1]:+.2f} + {b[0]:.2f} logMh; "
          f"test scatter {np.std(resid):.3f} dex")
    print("  -> the legal emulator = (SHMR amplitude) x (halo-only multi-epoch shape above).")


if __name__ == "__main__":
    sys.exit(main())
