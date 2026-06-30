"""exp29 Phase 2 — compression: how few deposit-fraction parameters approach the
Phase-1 capacity ceiling (~0.03 dex)?

Phase 1 freed the per-step fraction (NNLS, ~60 params) and fit all 5 CoGs to ~0.03
dex. Here we replace the free fraction with a smooth low-D fraction f(t) — a K-knot
piecewise-linear log-f(log t) — and a power-law width, fit jointly to all 5 CoGs.
dM*_i = 10^{logf(t_i)} dMh_i (phenomenological; no SFH). Report in-sample 5-epoch
RMS vs K, mapping the compression curve against the NNLS ceiling.

Run: PYTHONPATH=. uv run python experiments/exp29_outer_deposit/emulator_param.py
"""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from run import dipfree_mah, ANCHOR_SNAP, ANCHOR_Z                                  # noqa: E402
from cog_extrapolate import measured_cog                                           # noqa: E402
from profile_fit import pick_galaxies                                              # noqa: E402
from capacity_test import _deposits, _design, _nnls_rms                            # noqa: E402
from hongshao.tng_data import COG_RAD_KPC                                          # noqa: E402

R = COG_RAD_KPC
EPOCH_SNAP = np.array(ANCHOR_SNAP)


def _epoch_rms(pred, logC):
    return np.array([np.sqrt(np.mean((np.log10(np.clip(pred[k], 1.0, None)) - logC[k]) ** 2))
                     for k in range(5)])


def fit_param_f(gi, logC, K):
    """K-knot smooth log-f(t) + power-law width, fit jointly to all 5 CoGs."""
    snaps, t, t_obs = _deposits(gi)
    lt = np.log10(t); knots = np.linspace(lt.min(), lt.max(), K)
    dMh = np.clip(np.diff(10.0 ** dipfree_mah(gi)["logMh_full"]), 0, None)
    dMh = dMh[dipfree_mah(gi)["snap"] <= 72]
    b = (10.0 ** logC).reshape(-1)

    def predict(q):
        logf = np.interp(lt, knots, q[:K])
        widths = np.clip((10.0 ** q[K]) * (t / t_obs) ** q[K + 1], 0.3, 500)
        dMstar = 10.0 ** logf * dMh
        A = _design(snaps, widths)
        return (A @ dMstar).reshape(5, -1)

    def loss(q):
        return float(np.mean(_epoch_rms(predict(q), logC)))

    f0 = np.log10(10.0 ** logC[0, -1] / dMh.sum())               # rough mean fraction
    best = None
    for g0 in (0.5, 1.5):
        q0 = list(np.full(K, f0)) + [np.log10(40.0), g0]
        r = minimize(loss, q0, method="Nelder-Mead",
                     options=dict(maxiter=4000 + 1500 * K, xatol=1e-4, fatol=1e-8))
        if best is None or r.fun < best.fun:
            best = r
    return _epoch_rms(predict(best.x), logC)


def main():
    gxs = pick_galaxies(20)
    Ks = [2, 3, 4, 6]
    print("exp29 Phase 2 — fraction-compression: median 5-epoch in-sample CoG RMS (dex)\n")
    print(f"  model                         | " + "  ".join(f"z{z}" for z in ANCHOR_Z) + " | mean | #par")
    rows = {K: [] for K in Ks}; ceil = []
    for logm, gi in gxs:
        logC = measured_cog(gi)
        if logC is None:
            continue
        snaps, t, t_obs = _deposits(gi)
        # NNLS ceiling with the same power-law width family (refit width, free masses)
        from capacity_test import fit_powerlaw
        ceil.append(fit_powerlaw(gi, logC))
        for K in Ks:
            rows[K].append(fit_param_f(gi, logC, K))
    mc = np.median(ceil, 0)
    for K in Ks:
        m = np.median(rows[K], 0)
        print(f"  smooth f: {K}-knot + power-law width | " + " ".join(f"{x:5.3f}" for x in m) +
              f" |{m.mean():5.3f} | {K + 2}")
    print(f"  NNLS ceiling (free fraction)        | " + " ".join(f"{x:5.3f}" for x in mc) +
          f" |{mc.mean():5.3f} | ~60")
    print("\n[Phase 2] the K where smooth-f meets the NNLS ceiling = the compact emulator "
          "fraction dimension.")


if __name__ == "__main__":
    sys.exit(main())
