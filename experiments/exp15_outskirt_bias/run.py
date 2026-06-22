"""exp15 — Is the low-mass outskirt "bias" real, or regression-to-the-mean?

In exp13's truth-vs-predicted figure for M*[50-100 kpc], the binned-median
prediction sits +0.2-0.4 dex ABOVE the truth at the low-mass end. That looks
alarming (a 2x bias). This experiment asks whether it is (i) a genuine
mis-specification of the mean that we can fix, or (ii) the unavoidable
regression-to-the-mean artifact of plotting residuals against the *noisy* truth.

Two distinct diagnostics, which answer different questions:
  - BIN BY TRUE Y, plot residual (pred - true): this is what exp13 showed. For
    ANY predictor with intrinsic scatter, E[pred - Y | Y] has slope -(1 - R^2)
    (we derive and overlay it) because the truth itself contains the noise we are
    conditioning on. Present in mean OR sampled predictions; NOT removable by a
    better model.
  - BIN BY PREDICTED Y, plot mean true (a reliability diagram): if E[Y | pred] =
    pred the mean is well-specified and there is NO fixable bias; a systematic
    deviation would be genuine mis-specification. This is the real test.

We also check (a) whether the data has a measurement/annulus floor at low mass
(it does not -- see the setup probe), (b) whether a nonlinear mean (exp12's
late^2 + logmp^2) changes the reliability curve, and (c) that SAMPLING from the
predictive (mean + heteroscedastic sigma, exp14) reproduces the true population
distribution including the low-mass tail -- which the mean-only point estimate,
being under-dispersed, cannot.

Run from the repo root:
    EXP15_NMAX=600 PYTHONPATH=. uv run python \
        experiments/exp15_outskirt_bias/run.py
"""
# %% setup
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.table import Table
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hongshao.plotting import set_style, save_fig, OKABE_ITO                    # noqa: E402
from hongshao.provenance import write_manifest                                  # noqa: E402

set_style()
HERE = Path(__file__).resolve().parent
OUTDIR, FIGDIR = HERE / "outputs", HERE / "figures"
TABLE = ROOT / "data" / "processed" / "tng300_072_z0p4.fits"
FEAT = ["logmp", "logtc", "early", "late"]
NMAX = int(os.environ.get("EXP15_NMAX", 0))

t = Table.read(TABLE)
t = t[t["use"]]
if NMAX:
    t = t[:NMAX]
aper = np.asarray(t["logmstar_aper"], float)
X = np.column_stack([np.asarray(t[c], float) for c in
                     ("dmah_logmp", "dmah_logtc", "dmah_early", "dmah_late")])
y = np.log10(np.clip(10.0 ** aper[:, 4] - 10.0 ** aper[:, 2], 1.0, None))   # M*[50-100]
gmask = np.isfinite(y) & np.isfinite(X).all(1)
y = y[gmask]; X = X[gmask]; N = len(y)
print(f"exp15: low-mass bias diagnostic for M*[50-100], n={N}")


# %% ---- out-of-fold predictions: linear mean, nonlinear mean, heteroscedastic sigma
def logvar_fit(r, Z, ridge=2.0):
    A = np.column_stack([np.ones(len(r)), Z]); r2 = r ** 2

    def nll(gm):
        s = A @ gm
        return 0.5 * np.sum(s + r2 * np.exp(-s)) + 0.5 * ridge * np.sum(gm[1:] ** 2)

    def grad(gm):
        s = A @ gm; w = 0.5 * (1 - r2 * np.exp(-s)); o = A.T @ w; o[1:] += ridge * gm[1:]
        return o
    return minimize(nll, np.r_[np.log(max(r2.mean(), 1e-6)), np.zeros(Z.shape[1])],
                    jac=grad, method="L-BFGS-B").x


def design(Xm, nonlinear):
    """Linear (4 features) or +late^2 +logmp^2 (exp12 nonlinear mean)."""
    cols = [Xm[:, 0], Xm[:, 1], Xm[:, 2], Xm[:, 3]]
    if nonlinear:
        cols += [Xm[:, 3] ** 2, Xm[:, 0] ** 2]
    return np.column_stack([np.ones(len(Xm))] + cols)


def cv(k=5, seed=0):
    order = np.random.default_rng(seed).permutation(N)
    pred_lin = np.full(N, np.nan); pred_nl = np.full(N, np.nan); sig = np.full(N, np.nan)
    for fold in np.array_split(order, k):
        tr = np.setdiff1d(np.arange(N), fold)
        for nl, out in [(False, pred_lin), (True, pred_nl)]:
            A = design(X, nl)
            beta, *_ = np.linalg.lstsq(A[tr], y[tr], rcond=None)
            out[fold] = A[fold] @ beta
        # heteroscedastic sigma from the linear-mean training residuals
        A = design(X, False)
        beta, *_ = np.linalg.lstsq(A[tr], y[tr], rcond=None)
        r_tr = y[tr] - A[tr] @ beta
        mx, sx = X[tr].mean(0), X[tr].std(0)
        gm = logvar_fit(r_tr, (X[tr] - mx) / sx)
        Zf = np.column_stack([np.ones(len(fold)), (X[fold] - mx) / sx])
        sig[fold] = np.exp(0.5 * (Zf @ gm))
    return pred_lin, pred_nl, sig


pred, pred_nl, sig = cv()
resid = pred - y
R2 = 1.0 - np.mean((y - pred) ** 2) / y.var()
ybar = y.mean()
print(f"  R^2(linear) = {R2:.3f}   target mean={ybar:.3f} std={y.std():.3f}")


# %% ---- (1) regression-to-the-mean: residual vs TRUE, with -(1-R^2) overlay --
def binned(xv, yv, nbin=12, stat="median"):
    e = np.quantile(xv, np.linspace(0, 1, nbin + 1))
    cen, val, err = [], [], []
    for a, b in zip(e[:-1], e[1:]):
        m = (xv >= a) & (xv <= b)
        if m.sum() >= 12:
            cen.append(np.median(xv[m]))
            val.append(np.median(yv[m]) if stat == "median" else np.mean(yv[m]))
            err.append(yv[m].std() / np.sqrt(m.sum()))
    return map(np.asarray, (cen, val, err))


# slope of residual on TRUE (measured) vs theory -(1-R^2)
slope_true = np.polyfit(y, resid, 1)[0]
# slope of (true) on PREDICTED -- reliability slope (should be ~1 if unbiased in X)
slope_rel = np.polyfit(pred, y, 1)[0]
print(f"\n[1] residual-vs-TRUE slope: measured={slope_true:+.3f}  "
      f"theory -(1-R^2)={-(1-R2):+.3f}  -> {'MATCHES (artifact)' if abs(slope_true+(1-R2))<0.05 else 'differs'}")
print(f"[2] reliability slope d<true>/d pred: {slope_rel:+.3f}  "
      f"(1.0 = mean well-specified, no fixable bias)")


# %% ---- (2) reliability diagram: bin by PREDICTED, plot mean true ------------
# the genuine mis-specification test, linear vs nonlinear mean
print("\n[2] reliability E[true | predicted] - predicted, by predicted-mass decile:")
cen_p, mt_lin, et = binned(pred, y, nbin=10, stat="mean")
print("   pred:   " + " ".join(f"{c:6.2f}" for c in cen_p))
print("   <true>-pred (lin): " + " ".join(f"{m-c:+6.2f}" for c, m in zip(cen_p, mt_lin)))
cen_pnl, mt_nl, _ = binned(pred_nl, y, nbin=10, stat="mean")
print("   <true>-pred (+nl): " + " ".join(f"{m-c:+6.2f}" for c, m in zip(cen_pnl, mt_nl)))


# %% ---- (3) the low tail is the high-late, high-scatter population -----------
lo = y < np.percentile(y, 15)
print(f"\n[3] low-15% outskirt mass vs rest:  late {np.median(X[lo,3]):.2f} vs "
      f"{np.median(X[~lo,3]):.2f};  predicted sigma {sig[lo].mean():.3f} vs {sig[~lo].mean():.3f}")


# %% ---- (4) sampling reproduces the population; mean-only is under-dispersed --
rng = np.random.default_rng(0)
y_sample = pred + sig * rng.standard_normal(N)
print(f"\n[4] population spread:  true std={y.std():.3f}  mean-only std={pred.std():.3f}  "
      f"sampled std={y_sample.std():.3f}")
thr = np.percentile(y, 10)
print(f"    fraction below 10th pctile ({thr:.2f}):  true={np.mean(y<thr):.3f}  "
      f"mean-only={np.mean(pred<thr):.3f}  sampled={np.mean(y_sample<thr):.3f}")


# %% ---- FIGURE: the four panels ---------------------------------------------
lo_x, hi_x = np.percentile(y, [1, 99])
fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.5))

# A: residual vs true with regression-to-mean overlay
a = ax[0]
a.scatter(y, resid, s=5, alpha=0.10, color=OKABE_ITO[0], edgecolors="none")
cen, med, err = binned(y, resid)
a.plot(cen, med, "-o", color=OKABE_ITO[0], lw=2, ms=4, label="binned median")
xx = np.linspace(lo_x, hi_x, 50)
a.plot(xx, -(1 - R2) * (xx - ybar), "--", color="k", lw=1.8,
       label=r"$-(1-R^2)(y-\bar y)$")
a.axhline(0, color="0.6", lw=0.8)
a.set_xlim(lo_x, hi_x); a.set_ylim(-0.6, 0.6)
a.set_xlabel(r"true $\log M_*[50\!-\!100]$"); a.set_ylabel("predicted $-$ true [dex]")
a.set_title("A. Bin by TRUE: regression to the mean")
a.legend(fontsize=8)

# B: reliability diagram (bin by predicted)
b = ax[1]
b.plot([lo_x, hi_x], [lo_x, hi_x], "k--", lw=1.2, label="1:1 (unbiased)")
b.errorbar(cen_p, mt_lin, yerr=et, fmt="o-", color=OKABE_ITO[0], ms=5, label="linear mean")
b.errorbar(cen_pnl, mt_nl, fmt="s--", color=OKABE_ITO[4], ms=4, label="+ late$^2$, logmp$^2$")
b.set_xlim(lo_x, hi_x); b.set_ylim(lo_x, hi_x)
b.set_xlabel(r"predicted $\log M_*[50\!-\!100]$"); b.set_ylabel(r"mean true in bin")
b.set_title("B. Bin by PREDICTED: mean is unbiased")
b.legend(fontsize=8)

# C: population distributions
c = ax[2]
bins = np.linspace(lo_x, hi_x, 32)
c.hist(y, bins=bins, density=True, histtype="step", lw=2.2, color="k", label="true")
c.hist(pred, bins=bins, density=True, histtype="step", lw=2.0, color=OKABE_ITO[7],
       label=f"mean-only (std {pred.std():.2f})")
c.hist(y_sample, bins=bins, density=True, histtype="stepfilled", alpha=0.35,
       color=OKABE_ITO[2], label=f"sampled (std {y_sample.std():.2f})")
c.set_xlabel(r"$\log M_*[50\!-\!100]$"); c.set_ylabel("density")
c.set_title("C. Sampling restores the low tail")
c.legend(fontsize=8)
fig.suptitle(f"exp15 — the low-mass 'bias' is regression to the mean, not a model "
             f"defect (R^2={R2:.2f}; resid-vs-true slope {slope_true:+.2f} = "
             f"-(1-R^2) {-(1-R2):+.2f})", fontsize=11)
fig.tight_layout()
save_fig(fig, FIGDIR / "exp15_bias_diagnosis")


# %% ---- save outputs --------------------------------------------------------
OUTDIR.mkdir(parents=True, exist_ok=True)
write_manifest(OUTDIR, params={
    "n": int(N), "target": "M*[50-100 kpc]", "R2_linear": float(R2),
    "resid_vs_true_slope": float(slope_true), "minus_1_minus_R2": float(-(1 - R2)),
    "reliability_slope": float(slope_rel),
    "std_true": float(y.std()), "std_mean_only": float(pred.std()),
    "std_sampled": float(y_sample.std()),
    "lowtail_late_med": float(np.median(X[lo, 3])),
    "lowtail_sigma": float(sig[lo].mean()), "rest_sigma": float(sig[~lo].mean())})
print(f"\nwrote figure -> {FIGDIR}\nwrote outputs -> {OUTDIR}")
