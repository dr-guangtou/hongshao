"""exp29 — loose redshift-dependent-parameter joint fit (feasibility ceiling).

The fixed-param multi-epoch fit fails (z=0.4 -> ~19% max-rel) because one parameter
set can't describe all five epochs. The independent single-epoch fits prove every
epoch is representable to <1% -- but as five separate fits. This script asks the
in-between question the user posed: can a SINGLE joint model whose kernel parameters
are a smooth function of the OBSERVATION epoch z_k reach single-epoch quality?

Each parameter p in (log10 sigma_0, g, b_early, b_late, z_c) becomes a polynomial in
the observation epoch z_k:
    p(z_k) = sum_d c_{p,d} * z_k^d                    (degree D -> 5*(D+1) coeffs)
At epoch z_k we build the CoG with theta(z_k), pinned to that epoch's 148-kpc
aperture. This is "loose": the deposited mass f(t_i) is allowed to depend on z_k too
(not one assembly history) -- it is a flexibility ceiling, not a forward model.

Models, SAME objective (aperture-pinned relative-RMS over R>3 kpc), per galaxy:
    constant (D0): one theta (5 params)        -- the fixed-param failure baseline
    linear   (D1): theta(z_k) linear (10)      -- simplest z-dependence
    quadratic(D2): theta(z_k) quadratic (15)   -- enough to bend through mid-epochs
    independent  : 5 free thetas (cached)       -- the ceiling
Warm-started from the cached per-epoch fits (single_epoch_params.npz).

Run: PYTHONPATH=. uv run python experiments/exp29_outer_deposit/loose_zdep.py [n]
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from deposit import deposit_cog, width_t, eff_two_epoch, deposited                  # noqa: E402
from run import dipfree_mah, ANCHOR_SNAP, ANCHOR_Z                                   # noqa: E402
from cog_extrapolate import measured_cog                                            # noqa: E402
from hongshao.tng_data import COG_RAD_KPC                                           # noqa: E402
from hongshao.plotting import set_style, save_fig, OKABE_ITO                        # noqa: E402

set_style()
FIGDIR = HERE / "figures"
NPZ = HERE / "outputs" / "single_epoch_params.npz"
R = COG_RAD_KPC
R_IN = 3.0
EVAL = R > R_IN
Z = np.array(ANCHOR_Z)


def _theta_k(p, k):
    """Parameter 5-tuple at epoch k. p is length 5*(deg+1), ascending-power coeffs
    per parameter (deg = len(p)//5 - 1)."""
    deg = len(p) // 5 - 1
    zp = Z[k] ** np.arange(deg + 1)
    s0l, g, be, bl, zc = (p.reshape(5, deg + 1) * zp[None, :]).sum(1)
    return 10.0 ** s0l, g, be, bl, zc


def build_cogs(p, mah, data, t_obs):
    """Aperture-pinned model CoG at all 5 epochs; None if any param is pathological."""
    out = []
    for k in range(5):
        s0, g, be, bl, zc = _theta_k(p, k)
        if zc <= -1.0:                                   # negative base power -> NaN
            return None
        dM = deposited(eff_two_epoch(mah["z"], be, bl, zc), mah["dMh"], 1.0)
        sig = width_t(s0, g, mah["t"], t_obs)
        inc = mah["snap"] <= ANCHOR_SNAP[k]
        m = deposit_cog(dM[inc], sig[inc], 0.0, R)
        if not np.isfinite(m[-1]) or m[-1] <= 0:
            return None
        out.append(m * (data[k][-1] / m[-1]))
    return out


def _epoch_relrms(model_k, data_k):
    return np.sqrt(np.mean(((model_k[EVAL] - data_k[EVAL]) / data_k[EVAL]) ** 2))


def fit_joint(mah, data, t_obs, Pg, deg):
    """Joint multi-epoch fit at polynomial degree ``deg``; warm-started from the
    cached per-epoch params Pg (5,5). Mean per-epoch rel-RMS objective."""
    def loss(p):
        cogs = build_cogs(p, mah, data, t_obs)
        if cogs is None:
            return 1e3
        return float(np.mean([_epoch_relrms(cogs[k], data[k]) for k in range(5)]))

    warm = np.concatenate([np.polyfit(Z, Pg[:, j], deg)[::-1] for j in range(5)])  # ascending
    flat = warm.reshape(5, deg + 1).copy(); flat[:, 1:] = 0.0                      # constant fallback
    flat[:, 0] = [np.median(Pg[:, j]) for j in range(5)]
    best = None
    for p0 in (warm, flat.ravel()):
        r = minimize(loss, p0, method="Nelder-Mead",
                     options=dict(maxiter=min(2500 * len(p0), 12000), xatol=1e-4, fatol=1e-9))
        if best is None or r.fun < best.fun:
            best = r
    return build_cogs(best.x, mah, data, t_obs)


def metrics(cogs, data):
    """Per-epoch (log-RMS, max|rel|) over R>3 kpc."""
    out = []
    for k in range(5):
        rel = np.abs((cogs[k][EVAL] - data[k][EVAL]) / data[k][EVAL])
        logr = np.sqrt(np.mean((np.log10(cogs[k][EVAL]) - np.log10(data[k][EVAL])) ** 2))
        out.append((float(logr), float(rel.max())))
    return np.array(out)                                              # (5,2)


def main():
    d = np.load(NPZ)
    idx, P, logms, ind_met = d["index"], d["params"], d["logms"], d["metrics"]
    n = int(sys.argv[1]) if len(sys.argv) > 1 else len(idx)
    sel = np.linspace(0, len(idx) - 1, min(n, len(idx))).round().astype(int)

    DEGS = [0, 1, 2]                                     # constant, linear, quadratic in z
    res = {dg: [] for dg in DEGS}; ind_m, kept_logm = [], []
    for i in sel:
        gi = int(idx[i])
        logC = measured_cog(gi)
        if logC is None:
            continue
        data = [10.0 ** logC[k] for k in range(5)]
        mah = dipfree_mah(gi); t_obs = mah["t_obs"]
        cogs = {dg: fit_joint(mah, data, t_obs, P[i], dg) for dg in DEGS}
        if any(c is None for c in cogs.values()):
            continue
        for dg in DEGS:
            res[dg].append(metrics(cogs[dg], data))
        ind_m.append(ind_met[i][:, :2]); kept_logm.append(logms[i])
    res = {dg: np.array(v) for dg, v in res.items()}
    ind_m = np.array(ind_m); kept_logm = np.array(kept_logm)

    cols = {0: "fixed (D0,5p)", 1: "linear (D1,10p)", 2: "quad (D2,15p)"}
    print(f"exp29 — joint multi-epoch fit quality (n={len(ind_m)}), max|rel| over R>3 kpc, "
          f"aperture-pinned\n")
    print(f"  {'z':>4s} | " + " | ".join(f"{cols[dg]:>15s}" for dg in DEGS) +
          f" | {'independent':>12s}")
    for k, z in enumerate(ANCHOR_Z):
        cells = " | ".join(f"{100*np.median(res[dg][:,k,1]):13.1f}%" for dg in DEGS)
        print(f"  {z:4.1f} | {cells} | {100*np.median(ind_m[:,k,1]):10.1f}%")
    print(f"\n  epoch-averaged median max|rel|: " +
          "  ".join(f"{cols[dg]} {100*np.median([np.median(res[dg][:,k,1]) for k in range(5)]):.1f}%"
                    for dg in DEGS) +
          f"  independent {100*np.median([np.median(ind_m[:,k,1]) for k in range(5)]):.1f}%")

    ceil = np.median(ind_m[:, :, 1])
    q_avg = np.median(res[2][:, :, 1])
    print(f"\n[verdict] quadratic-z worst-epoch median {100*np.median([np.median(res[2][:,k,1]) for k in range(5)]):.1f}% "
          f"vs ceiling {100*ceil:.1f}%")
    if q_avg < 1.6 * ceil:
        print("  a smooth (quadratic) obs-epoch trend in the params recovers ~single-epoch\n"
              "  quality jointly -> z-dependent parameters DO give a reasonable multi-epoch\n"
              "  fit. Next: does the constrained puff-up (mass frozen) match this with fewer,\n"
              "  more physical DOF?")
    else:
        print("  even quadratic-z does not reach the ceiling -> the per-epoch best-fit params\n"
              "  are too degenerate/scattered to lie on a low-order curve; a constrained model\n"
              "  (puff-up: fix mass + g, vary width) may fit better than free z-dependent params.")
    _figure(res, DEGS, cols, ind_m)


def _figure(res, DEGS, cols, ind_m):
    x = np.arange(5)
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    palette = {0: OKABE_ITO[5], 1: OKABE_ITO[1], 2: OKABE_ITO[2]}
    marks = {0: "s", 1: "o", 2: "D"}
    for dg in DEGS:
        ax.plot(x, [100 * np.median(res[dg][:, k, 1]) for k in range(5)], marks[dg] + "-",
                c=palette[dg], lw=2, ms=7, label=cols[dg])
    ax.plot(x, [100 * np.median(ind_m[:, k, 1]) for k in range(5)], "^--", c="0.5", lw=2, ms=7,
            label="independent (ceiling)")
    ax.axhline(2.0, c="0.7", ls=":", lw=1.2, label="single-epoch ~2%")
    ax.set_xticks(x); ax.set_xticklabels([f"{z}" for z in ANCHOR_Z])
    ax.set(xlabel="observation epoch z", ylabel="median max |rel residual| over R>3 kpc [%]",
           title="exp29 — joint multi-epoch fit: z-dependent params (const/linear/quad) vs ceiling")
    ax.legend(fontsize=9)
    fig.tight_layout()
    print("\nwrote", save_fig(fig, FIGDIR / "exp29_loose_zdep")[0])


def demo():
    """Self-check: the polynomial param eval reduces correctly."""
    p0 = np.array([np.log10(40.0), 1.5, 4.0, 1.5, 2.0])          # degree 0 -> constant in z
    assert np.allclose(_theta_k(p0, 0), _theta_k(p0, 4)), "deg-0 not constant across epochs"
    p1 = np.array([1.0, 0.0, 1.5, 0.0, 4.0, 0.0, 1.5, 0.0, 2.0, 0.0])   # deg 1, all slopes 0
    s0, g, be, bl, zc = _theta_k(p1, 3)
    assert abs(g - 1.5) < 1e-12 and abs(zc - 2.0) < 1e-12 and abs(s0 - 10.0) < 1e-9, "deg-1 eval wrong"
    print("loose_zdep.demo OK: polynomial param eval correct")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo()
    else:
        sys.exit(main())
