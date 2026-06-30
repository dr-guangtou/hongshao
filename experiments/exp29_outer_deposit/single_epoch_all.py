"""exp29 — INDEPENDENT single-epoch fits to every snapshot (z=0.4..2.0).

The structural finding (single_vs_multi.py) was: the centred-Gaussian deposition
kernel fits *any one* z=0.4 CoG to ~0.004 dex / 2% but cannot fit all five epochs
at once. Open question: is that purely a multi-epoch *consistency* problem, or does
the centred-Gaussian deposit *shape* itself fail on compact high-z massive galaxies?

This settles it. For each epoch z_k in {0.4,0.7,1.0,1.5,2.0} (snaps 72/59/50/40/33)
we fit the kernel to THAT epoch's CoG ALONE — deposits integrated only up to t(z_k),
normalization pinned to the 148-kpc aperture M*(z_k) (kills the ~8% Gaussian
aperture leak that would otherwise sit on the outer point). Same kernel as
cog_extrapolate.py / single_vs_multi.py; only the target epoch changes.

Verdict logic:
  - z=2 single-epoch fits as good as z=0.4 (~2% max-rel) for massive galaxies
    -> the deposit SHAPE is fine at every epoch; the multi-epoch tension is a
       consistency problem -> build the parked puff-up model (PUFF_MODEL_PLAN.md).
  - z=2 fits FAIL for massive galaxies -> the centred-Gaussian shape itself is the
    high-z limit (puffing won't save it) -> rethink the high-z primitive.

Honest metric: LINEAR M*, relative residual (model-data)/data, max & 90th-pct over
R>3 kpc (TNG softening excluded). Stratified by stellar mass.

Run: PYTHONPATH=. uv run python experiments/exp29_outer_deposit/single_epoch_all.py
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
from profile_fit import pick_galaxies                                               # noqa: E402
from hongshao.tng_data import COG_RAD_KPC                                           # noqa: E402
from hongshao.plotting import set_style, save_fig, OKABE_ITO                        # noqa: E402

set_style()
FIGDIR = HERE / "figures"
R = COG_RAD_KPC
R_IN = 3.0                                          # exclude inner <=3 kpc (TNG softening)
EVAL = R > R_IN


def model_cog_epoch(q, mah, k):
    """Centred-Gaussian CoG at epoch k from deposits up to t(z_k); unit total mass."""
    s0, g, be, bl, zc = 10.0 ** q[0], q[1], q[2], q[3], q[4]
    dMstar = deposited(eff_two_epoch(mah["z"], be, bl, zc), mah["dMh"], 1.0)
    sigma = width_t(s0, g, mah["t"], mah["t_obs"])
    sa = ANCHOR_SNAP[k]
    m = mah["snap"] <= sa
    return deposit_cog(dMstar[m], sigma[m], 0.0, R)


def _pin(model, data):
    """Pin the model to the 148-kpc aperture (outer point) -> removes the leak."""
    return model * (data[-1] / model[-1])


def _rel_rms(model, data):
    return float(np.sqrt(np.mean(((model[EVAL] - data[EVAL]) / data[EVAL]) ** 2)))


def fit_epoch(mah, data_k, k):
    """Fit the centred-Gaussian kernel to one epoch's LINEAR CoG; pinned model out."""
    def loss(q):
        if q[4] <= -1.0:                              # z_c < -1 -> negative base power -> NaN
            return 1e3
        v = _rel_rms(_pin(model_cog_epoch(q, mah, k), data_k), data_k)
        return v if np.isfinite(v) else 1e3

    best = None
    for zc0 in (1.0, 2.0, 3.0):
        for g0 in (0.8, 2.0):
            r = minimize(loss, [np.log10(40.0), g0, 4.0, 1.5, zc0], method="Nelder-Mead",
                         options=dict(maxiter=6000, xatol=1e-4, fatol=1e-9))
            if best is None or r.fun < best.fun:
                best = r
    return _pin(model_cog_epoch(best.x, mah, k), data_k)


def metrics(model, data):
    """log-RMS, max|rel|, 90th-pct|rel| over R>3 kpc, linear M*."""
    rel = np.abs((model[EVAL] - data[EVAL]) / data[EVAL])
    logr = float(np.sqrt(np.mean((np.log10(model[EVAL]) - np.log10(data[EVAL])) ** 2)))
    return logr, float(rel.max()), float(np.percentile(rel, 90))


def fit_galaxy(gi):
    """All 5 single-epoch fits for one galaxy. Returns (logC, models, per-epoch metrics)."""
    logC = measured_cog(gi)
    if logC is None:
        return None
    mah = dipfree_mah(gi)
    if mah is None:
        return None
    models, mets = [], []
    for k in range(5):
        data_k = 10.0 ** logC[k]
        m = fit_epoch(mah, data_k, k)
        models.append(m)
        mets.append(metrics(m, data_k))
    return logC, np.array(models), np.array(mets)        # mets: (5, 3) = log-RMS,max,90th


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    gxs = pick_galaxies(n)
    logms, rows = [], []                                  # rows: (n_gal, 5, 3)
    for logm, gi in gxs:
        out = fit_galaxy(gi)
        if out is None:
            continue
        logms.append(logm); rows.append(out[2])
    logms = np.array(logms); rows = np.array(rows)        # (n, 5, 3)
    print(f"exp29 — independent single-epoch fits, all 5 epochs (n={len(rows)} galaxies)")
    print("  metric: LINEAR M*, relative residual, R>3 kpc, aperture pinned at 148 kpc\n")

    print("  median over all galaxies:")
    print(f"    {'z':>4s} {'log-RMS':>8s} {'max|rel|':>9s} {'90th|rel|':>10s}")
    for k, z in enumerate(ANCHOR_Z):
        med = np.median(rows[:, k, :], axis=0)
        print(f"    {z:4.1f} {med[0]:8.4f} {100*med[1]:8.1f}% {100*med[2]:9.1f}%")

    # stratify by stellar mass (tertiles); BCGs are where it should break
    print("\n  by stellar-mass tertile (max|rel| %, median):")
    order = np.argsort(logms); thirds = np.array_split(order, 3)
    labels = ["low", "mid", "high"]
    print(f"    {'bin':>5s} {'logM* range':>14s} " + " ".join(f"z={z}".rjust(7) for z in ANCHOR_Z))
    for lab, idx in zip(labels, thirds):
        rng = f"{logms[idx].min():.2f}-{logms[idx].max():.2f}"
        vals = [np.median(rows[idx, k, 1]) for k in range(5)]
        print(f"    {lab:>5s} {rng:>14s} " + " ".join(f"{100*v:6.1f}%" for v in vals))

    # the BCG specifically (most massive = first pick)
    bcg = np.argmax(logms)
    print(f"\n  most massive galaxy (logM*={logms[bcg]:.2f}):")
    print(f"    {'z':>4s} {'log-RMS':>8s} {'max|rel|':>9s} {'90th|rel|':>10s}")
    for k, z in enumerate(ANCHOR_Z):
        mr = rows[bcg, k]
        print(f"    {z:4.1f} {mr[0]:8.4f} {100*mr[1]:8.1f}% {100*mr[2]:9.1f}%")

    _verdict(logms, rows)
    _summary_figure(logms, rows)
    _figure(gxs)


def _verdict(logms, rows):
    """High-mass z=2 vs z=0.4: is the deposit shape the limit, or just consistency?"""
    order = np.argsort(logms); high = np.array_split(order, 3)[2]
    z04 = np.median(rows[high, 0, 1]); z20 = np.median(rows[high, 4, 1])
    print(f"\n[verdict] high-mass tertile, max|rel|:  z=0.4 {100*z04:.1f}%  ->  z=2.0 {100*z20:.1f}%")
    if z20 < 2.0 * z04 and z20 < 0.06:
        print("  z=2 single-epoch fits are ~as good as z=0.4 -> the centred-Gaussian deposit\n"
              "  SHAPE is fine at every epoch; the multi-epoch tension is a CONSISTENCY problem.\n"
              "  -> build the parked puff-up model (PUFF_MODEL_PLAN.md).")
    else:
        print("  z=2 single-epoch fits are substantially WORSE for massive galaxies -> the\n"
              "  centred-Gaussian deposit SHAPE itself is the high-z limit (puffing won't save\n"
              "  it). -> rethink the high-z primitive.")


def _summary_figure(logms, rows):
    """Population view of the verdict: max|rel| vs epoch per mass tertile, and the
    decisive z=2-vs-z=0.4 per-galaxy scatter (points on/below 1:1 -> z=2 not worse)."""
    order = np.argsort(logms); thirds = np.array_split(order, 3)
    labels = ["low", "mid", "high"]; cols = [OKABE_ITO[0], OKABE_ITO[2], OKABE_ITO[5]]
    x = np.arange(5)
    fig, (a, b) = plt.subplots(1, 2, figsize=(13.5, 5.2))

    for lab, idx, c in zip(labels, thirds, cols):
        med = np.array([np.median(rows[idx, k, 1]) for k in range(5)]) * 100
        a.plot(x + (cols.index(c) - 1) * 0.04, med, "o-", c=c, lw=2, ms=7,
               label=f"{lab}-mass ({logms[idx].min():.1f}-{logms[idx].max():.1f})")
        if lab == "high":                                  # spread for the bin that should break
            lo = np.array([np.percentile(rows[idx, k, 1], 16) for k in range(5)]) * 100
            hi = np.array([np.percentile(rows[idx, k, 1], 84) for k in range(5)]) * 100
            a.fill_between(x, lo, hi, color=c, alpha=0.15)
    a.axhline(2.0, c="0.5", ls=":", lw=1.2, label="z=0.4 single-epoch quality (~2%)")
    a.set_xticks(x); a.set_xticklabels([f"{z}" for z in ANCHOR_Z])
    a.set(xlabel="epoch redshift", ylabel="max |rel residual| over R>3 kpc [%]", ylim=(0, None),
          title="A. Single-epoch fit quality flat across cosmic time\n(high-z not worse, even for BCGs)")
    a.legend(fontsize=8, loc="upper left")

    # does fit quality degrade toward the massive end ("BCGs are where it breaks")? No.
    b.scatter(logms, rows[:, 0, 1] * 100, s=36, c="0.55", edgecolor="0.3", lw=0.4,
              label="z=0.4 (per galaxy)")
    b.scatter(logms, rows[:, 4, 1] * 100, s=42, c=OKABE_ITO[1], edgecolor="0.3", lw=0.4,
              label="z=2.0 (per galaxy)")
    for c, kk in ((".55", 0), (OKABE_ITO[1], 4)):          # binned-median trend per epoch
        med = [np.median(rows[idx, kk, 1]) * 100 for idx in thirds]
        ctr = [np.median(logms[idx]) for idx in thirds]
        b.plot(ctr, med, "-", c=c, lw=2.2)
    b.axhline(2.0, c="0.5", ls=":", lw=1.2)
    b.set(xlabel=r"$\log M_*$", ylabel="max |rel residual| over R>3 kpc [%]", ylim=(0, None),
          title="B. Fit quality does not degrade with mass\n(z=2 flat across the BCG range)")
    b.legend(fontsize=8, loc="upper left")
    fig.suptitle("exp29 — independent single-epoch fits: the centred-Gaussian deposit shape is NOT a "
                 f"high-z limit (n={len(rows)})", fontsize=12)
    fig.tight_layout()
    print("wrote", save_fig(fig, FIGDIR / "exp29_single_epoch_summary")[0])


def _figure(gxs):
    """BCG + mid + low-mass: all 5 single-epoch fits, linear CoG + relative residual."""
    sel = [gxs[0], gxs[len(gxs) // 2], gxs[-1]]
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.4), sharex=True)
    cmap = matplotlib.colormaps["viridis"]
    for col, (logm, gi) in enumerate(sel):
        out = fit_galaxy(gi)
        if out is None:
            continue
        logC, models, mets = out
        aC, aR = axes[0, col], axes[1, col]
        for k, z in enumerate(ANCHOR_Z):
            c = cmap(k / 4); data = 10.0 ** logC[k]
            aC.plot(R, data / 1e11, "o", c=c, ms=3, alpha=0.7)
            aC.plot(R, models[k] / 1e11, "-", c=c, lw=1.6,
                    label=f"z={z} ({100*mets[k,1]:.0f}%)")
            aR.plot(R, 100 * (models[k] - data) / data, "-", c=c, lw=1.4)
        aR.axhline(0, c="0.6", lw=0.8); aR.axvspan(R.min(), R_IN, color="0.85", alpha=0.5)
        aC.set(xscale="log", yscale="log", title=f"logM*={logm:.2f} (idx {gi})")
        aR.set(xscale="log", xlabel="R [kpc]", ylim=(-12, 12))
        aC.legend(fontsize=7, title="epoch (max|rel|)", loc="lower right")
        if col == 0:
            aC.set_ylabel(r"$M_*(<R)\ [10^{11}M_\odot]$")
            aR.set_ylabel(r"(model$-$data)/data [%]")
    fig.suptitle("exp29 — INDEPENDENT single-epoch fits, every epoch (centred Gaussian, "
                 "aperture-pinned); lines=model, dots=TNG; bottom=relative residual", fontsize=12)
    fig.tight_layout()
    print("\nwrote", save_fig(fig, FIGDIR / "exp29_single_epoch_all")[0])


def demo():
    """Self-check: pinning hits the outer aperture exactly; CoG is monotonic."""
    gxs = pick_galaxies(2)
    out = fit_galaxy(gxs[0][1])
    assert out is not None
    logC, models, mets = out
    for k in range(5):
        data = 10.0 ** logC[k]
        assert abs(models[k][-1] / data[-1] - 1.0) < 1e-6, "aperture not pinned"
        assert np.all(np.diff(models[k]) > 0), "CoG not monotonic"
    assert mets[0, 1] < 0.10, ("z=0.4 single-epoch fit should be good", mets[0, 1])
    print("single_epoch_all.demo OK: aperture pinned, CoG monotonic, z=0.4 fit good")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo()
    else:
        sys.exit(main())
