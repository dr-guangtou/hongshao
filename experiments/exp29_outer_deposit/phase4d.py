"""exp29 Phase 4d — can richer conditioning beat the ~0.11 dex universal floor?

Phase 4b: a universal forward model (MAH -> 5-epoch CoG) hit 0.11 dex test, and the
per-galaxy deposit fraction scattered with r(logMh)~0. Here we test whether the
per-galaxy parameters are predictable from richer halo/galaxy observables:
  - width (sigma_0, g)  <- log R50 (galaxy size; exp25 found sigma_0 ~ R50)
  - fraction f(t)       <- halo formation time t50, early-assembly fraction fz2

(1) diagnostic: R^2 of regressing each per-galaxy fitted param on the predictors;
(2) conditioned forward fits U / W(R50->width) / F(t50->fraction) / WF, test RMS;
(3) QA + summary figures.

Run: PYTHONPATH=. uv run python experiments/exp29_outer_deposit/phase4d.py [n]
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
from population import load_pop, predict, gal_rms, epoch_rms, fit_one, K, TKNOTS    # noqa: E402
from run import ANCHOR_Z                                                            # noqa: E402
from hongshao.tng_data import COG_RAD_KPC                                          # noqa: E402
from hongshao.plotting import set_style, save_fig, OKABE_ITO                        # noqa: E402

set_style()
FIGDIR = HERE / "figures"
R = COG_RAD_KPC
PREDS = ["z_logMh", "z_logR50", "z_t50", "z_fz2"]


def standardize(gals):
    for key, src in [("z_logMh", "logMh"), ("z_logR50", "logR50"), ("z_t50", "t50"), ("z_fz2", "fz2")]:
        v = np.array([g[src] for g in gals]); mu, sd = np.nanmean(v), np.nanstd(v) + 1e-9
        for g, x in zip(gals, v):
            g[key] = float((x - mu) / sd) if np.isfinite(x) else 0.0


def theta_of(p, gal, model):
    """Build the 5-vector [logf1,logf2,logf3,log_s0,g] from flat params p."""
    if model == "U":
        return p[:5]
    if model == "W":                              # width <- R50:  p=[f1,f2,f3, s0a,s0b, ga,gb]
        return np.array([p[0], p[1], p[2], p[3] + p[4] * gal["z_logR50"], p[5] + p[6] * gal["z_logR50"]])
    if model == "F":                              # fraction <- t50: p=[f1a,f1b,f2a,f2b,f3a,f3b, s0,g]
        z = gal["z_t50"]
        return np.array([p[0] + p[1] * z, p[2] + p[3] * z, p[4] + p[5] * z, p[6], p[7]])
    if model == "WF":                             # both
        zr, zt = gal["z_logR50"], gal["z_t50"]
        return np.array([p[0] + p[1] * zt, p[2] + p[3] * zt, p[4] + p[5] * zt,
                         p[6] + p[7] * zr, p[8] + p[9] * zr])
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


CACHE = HERE / "outputs" / "p4d_cache.npz"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    n = int(args[0]) if args else 200
    refit = "--refit" in sys.argv or not CACHE.exists()
    gals = load_pop(n); standardize(gals)
    rng = np.arange(len(gals)); train = rng[rng % 2 == 0]; test = rng[rng % 2 == 1]
    print(f"exp29 Phase 4d — n={len(gals)} ({len(train)} train / {len(test)} test)"
          f"{'  [cached]' if not refit else ''}\n")

    if not refit:                                        # fast path: redraw figures only
        z = np.load(CACHE, allow_pickle=True)
        pars, rms_self = z["pars"], float(z["rms_self"])
        results = {k: (z[f"p_{k}"], str(z[f"m_{k}"])) for k in ["U", "W", "F", "WF"]}
        labels = {"U": "U universal", "W": "W: R50->width", "F": "F: t50->fraction", "WF": "WF: both"}
        results = {labels[k]: v for k, v in results.items()}
        summary = {lab: np.median([epoch_rms(theta_of(p, gals[i], m), gals[i]) for i in test], 0)
                   for lab, (p, m) in results.items()}
        _report(pars, gals, rng, results, summary, rms_self)
        _figures(gals, test, pars, np.array([[gals[i][k] for k in PREDS] for i in rng]),
                 results, summary, rms_self)
        return

    # ---- per-galaxy fits + conditioned forward fits ----
    pars = np.array([fit_one(gals[i])[0] for i in rng])
    rms_self = float(np.median([gal_rms(pars[i], gals[i]) for i in rng]))
    U = fit_shared(gals, train, "U", [0, 0, 0, np.log10(40.0), 1.5])
    results = {"U universal": (U, "U"),
               "W: R50->width": (fit_shared(gals, train, "W", [U[0], U[1], U[2], U[3], 0, U[4], 0]), "W"),
               "F: t50->fraction": (fit_shared(gals, train, "F",
                                    [U[0], 0, U[1], 0, U[2], 0, U[3], U[4]]), "F"),
               "WF: both": (fit_shared(gals, train, "WF",
                            [U[0], 0, U[1], 0, U[2], 0, U[3], 0, U[4], 0]), "WF")}
    summary = {lab: np.median([epoch_rms(theta_of(p, gals[i], m), gals[i]) for i in test], 0)
               for lab, (p, m) in results.items()}
    key = {"U universal": "U", "W: R50->width": "W", "F: t50->fraction": "F", "WF: both": "WF"}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, pars=pars, rms_self=rms_self,
             **{f"p_{key[lab]}": p for lab, (p, m) in results.items()},
             **{f"m_{key[lab]}": m for lab, (p, m) in results.items()})
    _report(pars, gals, rng, results, summary, rms_self)
    _figures(gals, test, pars, np.array([[gals[i][k] for k in PREDS] for i in rng]),
             results, summary, rms_self)


def _report(pars, gals, rng, results, summary, rms_self):
    Xp = np.array([[gals[i][k] for k in PREDS] for i in rng])
    A = np.column_stack([np.ones(len(rng)), Xp])
    print("R^2 of per-galaxy param regressed on all predictors (logMh,logR50,t50,fz2):")
    for j, nm in enumerate(["logf1", "logf2", "logf3", "log_s0", "g"]):
        beta, *_ = np.linalg.lstsq(A, pars[:, j], rcond=None)
        r2 = 1 - np.var(pars[:, j] - A @ beta) / (np.var(pars[:, j]) + 1e-12)
        rr = [np.corrcoef(Xp[:, p], pars[:, j])[0, 1] for p in range(4)]
        print(f"    {nm:7s} R2={r2:+.2f}   r: logMh={rr[0]:+.2f} logR50={rr[1]:+.2f} "
              f"t50={rr[2]:+.2f} fz2={rr[3]:+.2f}")
    print("\nconditioned forward models (median TEST epoch RMS):")
    for label, er in summary.items():
        print(f"    {label:18s}: " + " ".join(f"{x:5.3f}" for x in er) + f"  (mean {er.mean():.3f})")
    print(f"    per-galaxy ceiling : {rms_self:.3f} (in-sample)")


def _figures(gals, test, pars, Xp, results, summary, rms_self):
    # Fig A — forward QA: 6 example test galaxies, measured vs best forward model (WF)
    wf_p, wf_m = results["WF: both"]
    sel = list(test[np.argsort([gals[i]["logMstar"] for i in test])[::-1]][::max(1, len(test) // 6)][:6])
    cmap = matplotlib.colormaps["viridis"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.4))
    for ax, i in zip(axes.ravel(), sel):
        g = gals[i]; pred = predict(theta_of(wf_p, g, wf_m), g)
        for k, z in enumerate(ANCHOR_Z):
            c = cmap(k / 4)
            ax.plot(R, g["logC"][k], "o", c=c, ms=3, alpha=0.75)
            ax.plot(R, np.log10(np.clip(pred[k], 1, None)), "-", c=c, lw=1.6, label=f"z={z}")
        rms = np.mean([np.sqrt(np.mean((np.log10(np.clip(pred[k], 1, None)) - g["logC"][k]) ** 2))
                       for k in range(5)])
        ax.set(xscale="log", xlabel="R [kpc]", ylabel=r"$\log M_*(<R)$",
               title=f"logM*={g['logMstar']:.2f}  logMh={g['logMh']:.2f}  RMS={rms:.3f}")
        ax.legend(fontsize=6, ncol=2)
    fig.suptitle("exp29 Phase 4 QA — conditioned forward model WF (lines) vs TNG (dots), held-out "
                 "test galaxies", fontsize=13)
    fig.tight_layout(); print("\nwrote", save_fig(fig, FIGDIR / "exp29_p4_forward_qa")[0])

    # Fig B — RMS summary across approaches
    fig2, ax = plt.subplots(figsize=(8.5, 5.2)); x = np.arange(5)
    ax.axhline(0.34, ls=":", c="0.5", lw=1); ax.text(0.05, 0.345, "frozen single-epoch (z=2)", fontsize=7, c="0.4")
    for (label, er), col in zip(summary.items(), [OKABE_ITO[0], OKABE_ITO[1], OKABE_ITO[2], OKABE_ITO[5]]):
        ax.plot(x, er, "o-", c=col, lw=1.8, ms=6, label=f"{label} ({er.mean():.3f})")
    ax.axhline(rms_self, ls="--", c="k", lw=1.2, label=f"per-galaxy ceiling ({rms_self:.3f})")
    ax.axhline(0.038, ls="-.", c="0.6", lw=1.1, label="completion ceiling (0.038)")
    ax.set_xticks(x); ax.set_xticklabels([f"z={z}" for z in ANCHOR_Z])
    ax.set(ylabel="median CoG RMS [dex]", title="exp29 — forward emulator: RMS by epoch and model")
    ax.legend(fontsize=8); fig2.tight_layout()
    print("wrote", save_fig(fig2, FIGDIR / "exp29_p4_rms_summary")[0])

    # Fig C — param vs predictor (drop degenerate railed per-galaxy fits)
    good = (pars[:, 3] > 0.5) & (pars[:, 3] < 3.6) & (pars[:, 4] > -2) & (pars[:, 4] < 6) \
        & (np.abs(pars[:, 0]) < 3) & (np.abs(pars[:, 1]) < 3) & (np.abs(pars[:, 2]) < 3)
    fig3, axes3 = plt.subplots(1, 3, figsize=(15, 4.6))
    panels = [("log_s0", 3, "z_logR50", "log R50"), ("g", 4, "z_logR50", "log R50"),
              ("logf1", 0, "z_t50", "t50 (halo formation)")]
    for ax, (nm, j, zk, xl) in zip(axes3, panels):
        xv = np.array([gals[i][zk] for i in range(len(gals))])[good]; yv = pars[good, j]
        ax.plot(xv, yv, "o", ms=4, alpha=0.55, c=OKABE_ITO[2])
        r = np.corrcoef(xv, yv)[0, 1]
        b = np.polyfit(xv, yv, 1); xs = np.array([xv.min(), xv.max()])
        ax.plot(xs, np.polyval(b, xs), "-", c=OKABE_ITO[1], lw=1.6)
        lo, hi = np.percentile(yv, [1, 99]); pad = 0.1 * (hi - lo + 1e-6)
        ax.set(xlabel=f"standardized {xl}", ylabel=nm, ylim=(lo - pad, hi + pad),
               title=f"{nm} vs {xl}:  r={r:+.2f}  (n={good.sum()})")
    fig3.suptitle("exp29 Phase 4 — per-galaxy parameters vs predictors (railed fits dropped); "
                  "weak ties only -> conditioning helps the forward fit, not param regression", fontsize=11)
    fig3.tight_layout(); print("wrote", save_fig(fig3, FIGDIR / "exp29_p4_param_predictors")[0])


if __name__ == "__main__":
    sys.exit(main())
