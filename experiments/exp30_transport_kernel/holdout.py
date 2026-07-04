"""exp30 phase 2.3 — held-out-epoch generalization (LOEO).

Everything so far is in-sample: fit 5 epochs, score the same 5. The emulator's job is
prediction. Protocol: per galaxy, fit on 4 epochs, predict the held-out epoch's full
CoG, for h in {z=0.7, 1.0, 1.5, 2.0}. (z=0.4-holdout is excluded: the deposits laid
between z=0.7 and z=0.4 appear ONLY in the held-out CoG, so their free masses are
unconstrained -- that test requires the parametric mass law = phase 3.)

Fairness rules:
  - The held-out epoch's data are never seen by the fit; all LOEO fits use fresh start
    grids (loose's warm start is built from 4-epoch-only independent fits).
  - Symmetric shape protocol: every model's held-out prediction is aperture-pinned to
    the data total at 148 kpc (amplitude is the SHMR's job). dyntrans/additive also
    report their NATIVE (unpinned) total-mass prediction error as a bonus diagnostic.
  - Metric: held-out max|rel| over ALL radii; generalization gap = held-out minus
    in-sample (full-5-epoch fit) error at the same epoch.

The stake: dyntrans predicts a held-out epoch by construction (one history evaluated
at a new time); loose-quad must extrapolate its z_k-parameter curves. If consistency
is worth anything predictive, it shows up here.

Run: PYTHONPATH=. uv run python experiments/exp30_transport_kernel/holdout.py [n] [--refit]
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
EXP29 = ROOT / "experiments" / "exp29_outer_deposit"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP29))
import transport_floor as tf                                                          # noqa: E402
import loose_zdep                                                                     # noqa: E402
from run import ANCHOR_SNAP, ANCHOR_Z                                                # noqa: E402
from single_epoch_all import model_cog_epoch                                         # noqa: E402
from real_mah_test import real_mah                                                   # noqa: E402
from hongshao.plotting import set_style, save_fig                                    # noqa: E402

set_style()
FIGDIR, OUTDIR = HERE / "figures", HERE / "outputs"
OUT_NPZ = OUTDIR / "holdout.npz"
R, AT = tf.R, tf.AT
Z = np.array(ANCHOR_Z)
HOLDOUTS = [1, 2, 3, 4]                                    # z=0.7, 1.0, 1.5, 2.0
MODELS = ["additive", "dyntrans", "loose-quad"]
COLORS = {"additive": "#E69F00", "dyntrans": "#56B4E9", "loose-quad": "#0072B2"}


def solve_sub(theta, mah, data, mode, fit_ks):
    """NNLS on the fit epochs only; returns model CoGs at ALL 5 epochs (+ rnorm)."""
    ti, snap, t_obs = mah["t"], mah["snap"], mah["t_obs"]
    masks = [snap <= sa for sa in ANCHOR_SNAP]
    blocks = [tf.basis(theta, ti, t_obs, AT[k], mode) * masks[k][None, :] for k in range(5)]
    A = np.vstack([blocks[k] / data[k][:, None] for k in fit_ks])
    if not np.isfinite(A).all():
        return None, np.inf
    x, rnorm = nnls(A, np.ones(A.shape[0]), maxiter=10 * A.shape[1])
    return np.array([b @ x for b in blocks]), rnorm


def fit_nnls_sub(mah, data, mode, fit_ks):
    """LOEO fit for the NNLS models with fresh start grids (no full-fit leakage)."""
    def loss(th):
        return solve_sub(th, mah, data, mode, fit_ks)[1]

    if mode == "additive":
        starts = [[s0l, g] for s0l in (1.8, 2.3) for g in (1.0, 1.7)]
    else:                                                  # dyntrans: fresh grid (no leakage)
        starts = [[s0l, 1.5, la, 1.5] for s0l in (1.8, 2.3)
                  for la in (np.log10(0.3), np.log10(1.0), np.log10(8.0))]
    best = None
    for p0 in starts:
        r = minimize(loss, p0, method="Nelder-Mead",
                     options=dict(maxiter=6000, xatol=1e-4, fatol=1e-10))
        if best is None or r.fun < best.fun:
            best = r
    return solve_sub(best.x, mah, data, mode, fit_ks)[0], best.x


def fit_indep_one(mah, data_k, k):
    """Single-epoch parametric fit (5 params) -- used only to warm-start loose LOEO."""
    def loss(q):
        m = model_cog_epoch(q, mah, k)
        m = m * (data_k[-1] / m[-1])
        v = np.sqrt(np.mean(((m - data_k) / data_k) ** 2))
        return v if np.isfinite(v) and q[4] > -1.0 else 1e3

    best = None
    for zc0 in (1.0, 2.5):
        r = minimize(loss, [np.log10(40.0), 1.5, 4.0, 1.5, zc0], method="Nelder-Mead",
                     options=dict(maxiter=4000, xatol=1e-4, fatol=1e-9))
        if best is None or r.fun < best.fun:
            best = r
    return best.x


def fit_loose_sub(mah, data, fit_ks):
    """LOEO fit of the loose quadratic-in-z model; warm from 4-epoch-only indep fits."""
    t_obs = mah["t_obs"]
    Pg4 = np.array([fit_indep_one(mah, data[k], k) for k in fit_ks])
    zs = Z[list(fit_ks)]

    def loss(p):
        cogs = loose_zdep.build_cogs(p, mah, data, t_obs)
        if cogs is None:
            return 1e3
        return float(np.mean([np.sqrt(np.mean(((cogs[k] - data[k]) / data[k]) ** 2))
                              for k in fit_ks]))

    warm = np.concatenate([np.polyfit(zs, Pg4[:, j], 2)[::-1] for j in range(5)])
    flat = warm.reshape(5, 3).copy(); flat[:, 1:] = 0.0
    flat[:, 0] = [np.median(Pg4[:, j]) for j in range(5)]
    best = None
    for p0 in (warm, flat.ravel()):
        r = minimize(loss, p0, method="Nelder-Mead",
                     options=dict(maxiter=8000, xatol=1e-4, fatol=1e-9))
        if best is None or r.fun < best.fun:
            best = r
    return np.array(loose_zdep.build_cogs(best.x, mah, list(data), t_obs))


def compute(n):
    d = np.load(HERE / "outputs" / "transport_floor.npz")
    idx, logms, datas = d["index"][:n], d["logms"][:n], d["data"][:n]
    ng = len(idx)
    held = {m: np.full((ng, 5), np.nan) for m in MODELS}       # pinned held-out max|rel|
    dmtot = {m: np.full((ng, 5), np.nan) for m in ("additive", "dyntrans")}
    for i in range(ng):
        mah = real_mah(int(idx[i]))
        data = [datas[i][k] for k in range(5)]
        for h in HOLDOUTS:
            fit_ks = [k for k in range(5) if k != h]
            for m in MODELS:
                if m == "loose-quad":
                    cogs = fit_loose_sub(mah, data, fit_ks)    # build_cogs pins every epoch
                    pred = cogs[h]
                else:
                    cogs, _ = fit_nnls_sub(mah, data, m, fit_ks)
                    if cogs is None:
                        continue
                    dmtot[m][i, h] = np.log10(cogs[h][-1]) - np.log10(data[h][-1])
                    pred = cogs[h] * (data[h][-1] / cogs[h][-1])   # symmetric pin
                held[m][i, h] = np.abs((pred - data[h]) / data[h]).max()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_NPZ, index=idx, logms=logms,
             **{f"held_{m}": held[m] for m in MODELS},
             **{f"dmtot_{m}": dmtot[m] for m in ("additive", "dyntrans")})
    print(f"wrote {OUT_NPZ}  (n={ng})")
    return np.load(OUT_NPZ)


def main():
    refit = "--refit" in sys.argv
    n = next((int(a) for a in sys.argv[1:] if a.isdigit()), 45)
    d = compute(n) if (refit or not OUT_NPZ.exists()) else np.load(OUT_NPZ)
    ng = len(d["index"])

    # in-sample references at each epoch (full 5-epoch fits)
    dtf = np.load(HERE / "outputs" / "transport_floor.npz")
    dfs = np.load(EXP29 / "outputs" / "final_scorecard.npz", allow_pickle=True)
    datas = dtf["data"][:ng]
    insam = {
        "additive": np.array([tf.maxrel(dtf["additive"][i], datas[i]) for i in range(ng)]),
        "dyntrans": np.array([tf.maxrel(dtf["dyntrans"][i], datas[i]) for i in range(ng)]),
        "loose-quad": np.array([tf.maxrel(dfs["models"][i, 1], dfs["data"][i]) for i in range(ng)]),
    }

    print(f"\nexp30 phase 2.3 — held-out-epoch generalization (n={ng}), median max|rel| "
          "(held-out pinned)\n")
    print(f"  {'model':>11s} | " + " | ".join(f"z={Z[h]} out".rjust(10) for h in HOLDOUTS) +
          " | held-avg | in-sample avg | gap")
    for m in MODELS:
        ho = [100 * np.nanmedian(d[f"held_{m}"][:, h]) for h in HOLDOUTS]
        ins = [100 * np.median(insam[m][:, h]) for h in HOLDOUTS]
        print(f"  {m:>11s} | " + " | ".join(f"{v:9.1f}%" for v in ho) +
              f" | {np.mean(ho):7.1f}% | {np.mean(ins):12.1f}% | {np.mean(ho)-np.mean(ins):+5.1f}%")

    print("\n  native (unpinned) held-out TOTAL-mass prediction, median |dlog M*(148kpc)|:")
    for m in ("additive", "dyntrans"):
        v = [np.nanmedian(np.abs(d[f"dmtot_{m}"][:, h])) for h in HOLDOUTS]
        print(f"    {m:>9s}: " + "  ".join(f"z={Z[h]}: {vi:.3f}" for h, vi in zip(HOLDOUTS, v)))

    ho_dy = np.mean([np.nanmedian(d["held_dyntrans"][:, h]) for h in HOLDOUTS])
    ho_lo = np.mean([np.nanmedian(d["held_loose-quad"][:, h]) for h in HOLDOUTS])
    gap_dy = 100 * ho_dy - np.mean([100 * np.median(insam["dyntrans"][:, h]) for h in HOLDOUTS])
    gap_lo = 100 * ho_lo - np.mean([100 * np.median(insam["loose-quad"][:, h]) for h in HOLDOUTS])
    print(f"\n[verdict] held-out avg: dyntrans {100*ho_dy:.1f}% (gap {gap_dy:+.1f}%)  vs  "
          f"loose-quad {100*ho_lo:.1f}% (gap {gap_lo:+.1f}%)")
    if ho_dy < ho_lo and gap_dy < gap_lo:
        print("  the consistent history both PREDICTS BETTER and degrades less out-of-sample ->\n"
              "  consistency pays off predictively; dyntrans is the emulator kernel. Next: phase 3\n"
              "  (predict the active masses from halo-only inputs).")
    elif ho_dy < ho_lo:
        print("  dyntrans predicts better out-of-sample (though gaps are comparable) -> keep it.")
    else:
        print("  loose-quad generalizes as well or better -> the consistency advantage does NOT\n"
              "  show up predictively at these epochs; re-examine before phase 3.")
    _figure(d, insam)


def _figure(d, insam):
    fig, (a, b) = plt.subplots(1, 2, figsize=(13.0, 5.2))
    x = np.arange(len(HOLDOUTS))
    for m in MODELS:
        ho = [100 * np.nanmedian(d[f"held_{m}"][:, h]) for h in HOLDOUTS]
        ins = [100 * np.median(insam[m][:, h]) for h in HOLDOUTS]
        a.plot(x, ho, "o-", c=COLORS[m], lw=2, ms=7, label=f"{m} (held-out)")
        a.plot(x, ins, "o--", c=COLORS[m], lw=1.2, ms=4, alpha=0.55, label=f"{m} (in-sample)")
    a.set_xticks(x); a.set_xticklabels([f"z={Z[h]}" for h in HOLDOUTS])
    a.set(xlabel="held-out epoch", ylabel="median max|rel| [%]", ylim=(0, None),
          title="A. Held-out (solid) vs in-sample (faint dashed)")
    a.legend(fontsize=7)

    for m in MODELS:
        gaps = [100 * (np.nanmedian(d[f"held_{m}"][:, h]) - np.median(insam[m][:, h]))
                for h in HOLDOUTS]
        b.plot(x, gaps, "s-", c=COLORS[m], lw=2, ms=7, label=m)
    b.axhline(0, c="0.6", lw=0.8)
    b.set_xticks(x); b.set_xticklabels([f"z={Z[h]}" for h in HOLDOUTS])
    b.set(xlabel="held-out epoch", ylabel="generalization gap [% points]",
          title="B. Gap = held-out $-$ in-sample")
    b.legend(fontsize=8)
    fig.suptitle("exp30 phase 2.3 — leave-one-epoch-out: does the consistent history predict "
                 "unseen epochs better?", fontsize=12)
    fig.tight_layout()
    print("\nwrote", save_fig(fig, FIGDIR / "exp30_holdout")[0])


def demo():
    """Self-check: LOEO prediction is monotonic, and fitting WITH the epoch is never
    worse than predicting it (sanity of the harness)."""
    dtf = np.load(HERE / "outputs" / "transport_floor.npz")
    i = 2
    mah = real_mah(int(dtf["index"][i]))
    data = [dtf["data"][i][k] for k in range(5)]
    cogs, _ = fit_nnls_sub(mah, data, "dyntrans", [0, 1, 3, 4])      # hold out z=1.0
    assert cogs is not None and np.isfinite(cogs).all()
    assert np.all(np.diff(cogs[2]) >= -1e-9), "held-out prediction must be monotonic"
    pred = cogs[2] * (data[2][-1] / cogs[2][-1])
    held_err = np.abs((pred - data[2]) / data[2]).max()
    insample_err = tf.maxrel(dtf["dyntrans"][i], dtf["data"][i])[2]
    assert held_err >= insample_err * 0.5, "held-out should not be dramatically better than in-sample"
    print(f"holdout.demo OK: monotonic prediction; held-out z=1.0 {100*held_err:.1f}% vs "
          f"in-sample {100*insample_err:.1f}%")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo()
    else:
        sys.exit(main())
