"""exp29 — final honest scorecard: all 4 models on the corrected standard.

One comparison of every multi-epoch model on the same honest footing:
  - real de-dipped MAH (peak_history), not the smooth DiffMAH fit
  - ALL radii (no inner-3-kpc mask)
  - both evaluation lenses: profile max|rel|, and the standard mass QA (aperture +
    outer-envelope masses, kpc and R_half bins; mass_qa.py)

Models: independent per-epoch (ceiling), loose-zdep quad, puff-ratio (consistent
history), free-mass NNLS floor (consistent history, free masses). independent + loose
are reused from integrated_check.npz; puff + floor are fit here over all radii.

Run: PYTHONPATH=. uv run python experiments/exp29_outer_deposit/final_scorecard.py [n] [--refit]
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize, nnls

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from run import ANCHOR_SNAP, ANCHOR_Z                                                # noqa: E402
from real_mah_test import real_mah                                                   # noqa: E402
from single_epoch_all import anchor_times                                            # noqa: E402
import puff_fit                                                                       # noqa: E402
import mass_qa                                                                        # noqa: E402
from hongshao.tng_data import COG_RAD_KPC                                            # noqa: E402
from hongshao.plotting import set_style, save_fig                                    # noqa: E402

set_style()
FIGDIR = HERE / "figures"
IN_NPZ = HERE / "outputs" / "integrated_check.npz"          # real MAH, all radii: data/indep/loose
OUT_NPZ = HERE / "outputs" / "final_scorecard.npz"
R = COG_RAD_KPC
ALLR = R > 0.0                                              # no inner mask
NAMES = ["independent", "loose-quad", "puff-ratio", "nnls-floor"]
COLORS = {"independent": "0.45", "loose-quad": "#0072B2", "puff-ratio": "#009E73",
          "nnls-floor": "#CC3377"}


def _relrms(cogs, data):
    return float(np.mean([np.sqrt(np.mean(((cogs[k][ALLR] - data[k][ALLR]) / data[k][ALLR]) ** 2))
                          for k in range(5)]))


def fit_puff(mah, data, at):
    """Puff-ratio consistent-history fit over all radii (6 params)."""
    t_obs = mah["t_obs"]
    base = [np.log10(60.0), 1.5, 4.0, 1.5, 2.5]

    def loss(p):
        s0, g, q = 10.0 ** p[0], p[1], max(p[2], 0.0)
        cogs = puff_fit.build_cogs(s0, g, q, p[3], p[4], p[5], mah, data, t_obs, at, "ratio")
        return _relrms(cogs, data) if cogs is not None else 1e3

    best = None
    for q0 in (0.0, 0.5, 1.0, 1.8):
        r = minimize(loss, [base[0], base[1], q0, base[2], base[3], base[4]], method="Nelder-Mead",
                     options=dict(maxiter=12000, xatol=1e-4, fatol=1e-9))
        if best is None or r.fun < best.fun:
            best = r
    x = best.x
    return np.array(puff_fit.build_cogs(10.0 ** x[0], x[1], max(x[2], 0.0), x[3], x[4], x[5],
                                        mah, data, t_obs, at, "ratio"))


def fit_nnls(mah, data):
    """Free-mass NNLS consistent-history floor over all radii."""
    t_obs = mah["t_obs"]
    use = mah["snap"] <= ANCHOR_SNAP[0]
    ti, snap_i = mah["t"][use], mah["snap"][use]
    masks = [snap_i <= sa for sa in ANCHOR_SNAP]

    def solve(par):
        sig = 10.0 ** par[0] * (ti / t_obs) ** par[1]
        basis = 1.0 - np.exp(-R[:, None] ** 2 / (2.0 * sig[None, :] ** 2))
        A = np.vstack([(basis * masks[k][None, :]) / data[k][:, None] for k in range(5)])
        x, rn = nnls(A, np.ones(5 * len(R)), maxiter=5 * basis.shape[1])
        return x, basis, rn

    best = None
    for s0 in (1.7, 2.2, 2.7):
        for g in (1.0, 1.7):
            r = minimize(lambda p: solve(p)[2], [s0, g], method="Nelder-Mead",
                         options=dict(maxiter=400, xatol=1e-3, fatol=1e-6))
            if best is None or r.fun < best.fun:
                best = r
    x, basis, _ = solve(best.x)
    return np.array([(x * masks[k]) @ basis.T for k in range(5)])


def compute(n):
    d = np.load(IN_NPZ)
    idx, logms, data = d["index"][:n], d["logms"][:n], d["data"][:n]
    indep, loose = d["independent"][:n], d["loose"][:n]
    at = anchor_times()
    puff, floor = [], []
    for i in range(len(idx)):
        mah = real_mah(int(idx[i]))
        dl = [data[i][k] for k in range(5)]
        puff.append(fit_puff(mah, dl, at)); floor.append(fit_nnls(mah, dl))
    mods = np.stack([indep, loose, np.array(puff), np.array(floor)], axis=1)   # (n,4,5,24)
    np.savez(OUT_NPZ, index=idx, logms=logms, R=R, data=data, models=mods, names=np.array(NAMES))
    print(f"wrote {OUT_NPZ}  models {mods.shape}")
    return np.load(OUT_NPZ, allow_pickle=True)


def main():
    refit = "--refit" in sys.argv
    n = next((int(a) for a in sys.argv[1:] if a.isdigit()), 45)
    d = compute(n) if (refit or not OUT_NPZ.exists()) else np.load(OUT_NPZ, allow_pickle=True)
    data, mods = d["data"], d["models"]

    prof = {nm: np.array([[np.abs((mods[i, m, k] - data[i, k]) / data[i, k]).max() for k in range(5)]
                          for i in range(len(data))]) for m, nm in enumerate(NAMES)}
    # integrated mass biases via the standard QA measurement
    mass = {}
    for m, nm in enumerate(NAMES):
        truth, model, _, _ = mass_qa.measure_all(mods[:, m], data, R)
        mass[nm] = {k: np.log10(np.clip(model[k], 1.0, None)) - np.log10(np.clip(truth[k], 1.0, None))
                    for k in ("kpc:M(<30)", "kpc:M(>50)", "Re:M(>2Re)")}

    print(f"\nexp29 FINAL SCORECARD (real MAH, all radii, n={len(data)})\n")
    print("  (1) profile max|rel| [%], median per epoch:")
    print(f"    {'model':>12s} | " + " | ".join(f"z={z}".rjust(6) for z in ANCHOR_Z) + " |  avg")
    for nm in NAMES:
        avg = np.median([np.median(prof[nm][:, k]) for k in range(5)])
        print(f"    {nm:>12s} | " + " | ".join(f"{100*np.median(prof[nm][:,k]):5.1f}%" for k in range(5)) +
              f" | {100*avg:4.1f}%")

    print("\n  (2) integrated mass bias, median dlog (model-data), z=0.4 -> z=2.0:")
    print(f"    {'model':>12s} | {'M*(<30kpc)':>22s} | {'M*(>50kpc)':>22s} | {'M*(>2Re)':>22s}")
    for nm in NAMES:
        cells = []
        for key in ("kpc:M(<30)", "kpc:M(>50)", "Re:M(>2Re)"):
            v = mass[nm][key]
            cells.append(f"{np.median(v[:,0]):+.2f}..{np.median(v[:,4]):+.2f}")
        print(f"    {nm:>12s} | " + " | ".join(f"{c:>22s}" for c in cells))

    _figure(prof, mass)
    print("\n[scorecard] Honest footing (real MAH, all radii): per-epoch shape is representable\n"
          "  (independent ceiling ~2%); every consistent model is capped well above it; cumulative\n"
          "  apertures ~0.01 dex for all; the fixed-kpc far outskirt at high z is the shared\n"
          "  failure, benign in R_half units. See figure + mass_qa figures.")


def _figure(prof, mass):
    x = np.arange(5)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.0))
    titles = ["A. Profile residual (all radii)", "B. Outskirt bias  M*(>50 kpc)  [absolute]",
              "C. Outskirt bias  M*(>2 R_half)  [relative]"]
    ylabs = ["median profile max|rel| [%]", r"median $\Delta\log M_*(>50\,{\rm kpc})$",
             r"median $\Delta\log M_*(>2R_{\rm half})$"]
    for ax, key, ttl, yl in zip(axes, (None, "kpc:M(>50)", "Re:M(>2Re)"), titles, ylabs):
        for nm in NAMES:
            if key is None:
                y = [100 * np.median(prof[nm][:, k]) for k in range(5)]
            else:
                y = [np.median(mass[nm][key][:, k]) for k in range(5)]
            ax.plot(x, y, "o-", c=COLORS[nm], lw=2, ms=6, label=nm)
        if key is not None:
            ax.axhline(0, c="0.6", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels([f"{z}" for z in ANCHOR_Z])
        ax.set(xlabel="epoch z", ylabel=yl, title=ttl)
        if key is None:
            ax.set_ylim(0, None)
    axes[0].legend(fontsize=8)
    fig.suptitle("exp29 FINAL SCORECARD — real MAH, all radii: profile + absolute/relative outskirt "
                 "mass bias, 4 models", fontsize=12)
    fig.tight_layout()
    print("wrote", save_fig(fig, FIGDIR / "exp29_final_scorecard")[0])


if __name__ == "__main__":
    sys.exit(main())
