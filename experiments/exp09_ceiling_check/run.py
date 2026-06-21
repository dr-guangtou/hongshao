"""exp09 — Ceiling check: is a richer-than-linear model worth pursuing?

Before committing to a model family for the Ultimate-SHMR emulator, measure how
much predictive signal a *flexible* model can extract from the same features
that the linear model uses. If a gradient-boosted-trees ceiling barely beats
linear, then linear already captures the structure and we keep the simple
closed-form equation; if the gap is large, there is nonlinear structure
(interactions / curvature) worth capturing with a richer analytic form (e.g.
symbolic regression).

Targets: the 4 aperture/annulus masses (<10, 10-30, 30-50, 50-100 kpc).
Features: M0 + MAH-PCA(4) (same as exp08). Models, all 5-fold CV:
  - linear (M0 only)            -- reference
  - linear (M0+MAH)             -- the current analytic model
  - poly-2 (M0+MAH)             -- richer *analytic* form (interactions+curvature)
  - GBM (M0+MAH)                -- flexible achievability ceiling
  - GBM (shuffled MAH)          -- control: must collapse to the M0-only level
Scored by the exp07 suite (CRPS, RMS) with a homoscedastic-Gaussian predictive.

Run from the repo root (EXP09_NMAX=400 for a sub-minute pass):
    PYTHONPATH=. uv run python experiments/exp09_ceiling_check/run.py
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
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hongshao.tng_data import (load_mah, load_cosmic_time, peak_history)       # noqa: E402
from hongshao.metrics import crps_gaussian                                     # noqa: E402
from hongshao.plotting import set_style, save_fig, OKABE_ITO                   # noqa: E402
from hongshao.provenance import write_manifest                                 # noqa: E402

set_style()
HERE = Path(__file__).resolve().parent
OUTDIR, FIGDIR = HERE / "outputs", HERE / "figures"
TABLE = ROOT / "data" / "processed" / "tng300_072_z0p4.fits"
TNAMES = ["<10", "10-30", "30-50", "50-100"]

t = Table.read(TABLE)
t = t[t["use"]]
NMAX = int(os.environ.get("EXP09_NMAX", 0))
if NMAX:
    t = t[:NMAX]
logM0 = np.asarray(t["logm0_halo"], float)
aper = np.asarray(t["logmstar_aper"], float)
idx = np.asarray(t["index"])
z50 = np.asarray(t["z50"], float)             # formation redshift, to orient the PCs

# MAH-PCA(4) features (same construction as exp06/exp08)
mah = load_mah(); tsnap = load_cosmic_time(); tgrid = np.linspace(2.2, 9.0, 18)
ms = np.full((len(t), 18), np.nan)
for r, i in enumerate(idx):
    sn, lmp = peak_history(mah[int(i)])
    if lmp is None:
        continue
    tt = tsnap[sn.astype(int)]
    if tt[0] <= tgrid[0] and tt[-1] >= tgrid[-1]:
        ms[r] = np.interp(tgrid, tt, lmp) - logM0[r]


def _annulus(a_outer, a_inner):
    return np.log10(np.clip(10.0 ** a_outer - 10.0 ** a_inner, 1.0, None))


Y = np.column_stack([aper[:, 0], _annulus(aper[:, 1], aper[:, 0]),
                     _annulus(aper[:, 2], aper[:, 1]), _annulus(aper[:, 4], aper[:, 2])])
g = np.isfinite(ms).all(1) & np.isfinite(Y).all(1) & np.isfinite(logM0) & np.isfinite(z50)
mu = ms[g].mean(0); _, _, Vt = np.linalg.svd(ms[g] - mu, full_matrices=False)
pca = (ms[g] - mu) @ Vt[:4].T
M0 = logM0[g]; Y = Y[g]; z50 = z50[g]; N = len(Y)
# orient each MAH-PC so positive = earlier formation (higher z50); sign flips do
# not change any prediction/score, only make the coefficients sign-readable.
for k in range(4):
    if np.corrcoef(pca[:, k], z50)[0, 1] < 0:
        pca[:, k] *= -1

rng = np.random.default_rng(1)
pca_shuf = pca.copy()
edges = np.quantile(M0, np.linspace(0, 1, 13))
b = np.digitize(M0, edges[1:-1])
for bi in np.unique(b):
    ii = np.where(b == bi)[0]
    pca_shuf[ii] = pca[ii][rng.permutation(len(ii))]

X_m0 = M0[:, None]
X_full = np.column_stack([M0, pca])
X_shuf = np.column_stack([M0, pca_shuf])
print(f"exp09 ceiling check: n={N}, features = M0 + MAH-PCA(4)")


def gbm():
    return HistGradientBoostingRegressor(
        max_depth=3, learning_rate=0.05, max_iter=400,
        min_samples_leaf=40, l2_regularization=1.0, random_state=0)


def poly2():
    return make_pipeline(StandardScaler(),
                         PolynomialFeatures(2, include_bias=False),
                         LinearRegression())


MODELS = {
    "linear (M0)": (LinearRegression, X_m0, OKABE_ITO[7]),
    "linear (M0+MAH)": (LinearRegression, X_full, OKABE_ITO[0]),
    "poly-2 (M0+MAH)": (poly2, X_full, OKABE_ITO[4]),
    "GBM (M0+MAH)": (gbm, X_full, OKABE_ITO[5]),
    "GBM (shuffled MAH)": (gbm, X_shuf, "0.7"),
}


def cv_score(make_model, X, k=5, seed=0):
    """Out-of-fold predictions per target + homoscedastic Gaussian sigma from
    training residuals. Returns per-target (rms, crps)."""
    order = np.random.default_rng(seed).permutation(N)
    pred = np.full_like(Y, np.nan); sig = np.full_like(Y, np.nan)
    for fold in np.array_split(order, k):
        tr = np.setdiff1d(np.arange(N), fold)
        for j in range(Y.shape[1]):
            m = make_model(); m.fit(X[tr], Y[tr, j])
            pred[fold, j] = m.predict(X[fold])
            sig[fold, j] = (Y[tr, j] - m.predict(X[tr])).std()
    rms = np.sqrt(np.mean((Y - pred) ** 2, 0))
    crps = crps_gaussian(Y, pred, sig).mean(0)
    return rms, crps, pred


results = {}
print(f"\n{'model':22s} {'RMS(mean)':>10s} {'CRPS(mean)':>11s}   per-aperture CRPS")
for name, (mk, X, _) in MODELS.items():
    rms, crps, pred = cv_score(mk, X)
    results[name] = dict(rms=rms, crps=crps, pred=pred)
    print(f"  {name:20s} {rms.mean():10.4f} {crps.mean():11.4f}   "
          + " ".join(f"{c:.3f}" for c in crps))

lin = results["linear (M0+MAH)"]["crps"].mean()
gbm_c = results["GBM (M0+MAH)"]["crps"].mean()
poly_c = results["poly-2 (M0+MAH)"]["crps"].mean()
print(f"\n[verdict] CRPS: linear={lin:.4f}  poly-2={poly_c:.4f}  GBM-ceiling={gbm_c:.4f}")
print(f"  flexible ceiling beats linear by {100*(lin-gbm_c)/lin:+.1f}%  "
      f"(poly-2 analytic: {100*(lin-poly_c)/lin:+.1f}%)")
verdict = ("linear is at the ceiling -> keep the closed-form linear model"
           if (lin - gbm_c) / lin < 0.05 else
           "nonlinear structure exists -> a richer analytic form is worth pursuing")
print(f"  => {verdict}")

# %% FIGURE
fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.0, 4.0))
x = np.arange(4); w = 0.16
for k_i, (name, (_, _, col)) in enumerate(MODELS.items()):
    axA.bar(x + (k_i - 2) * w, results[name]["crps"], w, color=col, label=name)
axA.set_xticks(x); axA.set_xticklabels(TNAMES)
axA.set_xlabel("aperture / annulus [kpc]")
axA.set_ylabel("cross-validated CRPS [dex]")
axA.set_title("Per-aperture predictive skill")
axA.legend(fontsize=7)
axA.text(-0.13, 1.04, "A", transform=axA.transAxes, fontweight="bold", fontsize=12)

names = list(MODELS)
overall = [results[n]["crps"].mean() for n in names]
cols = [MODELS[n][2] for n in names]
axB.bar(range(len(names)), overall, color=cols)
axB.axhline(lin, ls="--", color=OKABE_ITO[0], lw=1)
for i, n in enumerate(names):
    axB.annotate(f"{100*(lin-overall[i])/lin:+.0f}%", (i, overall[i]),
                 textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
axB.set_xticks(range(len(names)))
axB.set_xticklabels(["lin\nM0", "lin\nM0+MAH", "poly-2", "GBM", "GBM\nshuf"], fontsize=8)
axB.set_ylabel("overall CRPS [dex]")
axB.set_title("Linear vs flexible ceiling")
axB.set_ylim(0, max(overall) * 1.15)
axB.text(-0.13, 1.04, "B", transform=axB.transAxes, fontweight="bold", fontsize=12)
fig.suptitle(f"exp09 — ceiling check: how much does nonlinearity add? (n={N})",
             fontsize=11)
fig.tight_layout()
save_fig(fig, FIGDIR / "exp09_ceiling_check")

# %% FIGURE 2 — direct prediction check: truth vs predicted (top) and
# truth vs (predicted - truth) (bottom), per aperture, linear vs GBM ceiling.
def _binned(xv, yv, nbin=10):
    edges = np.quantile(xv, np.linspace(0, 1, nbin + 1))
    cen, med = [], []
    for a, bb in zip(edges[:-1], edges[1:]):
        m = (xv >= a) & (xv <= bb)
        if m.sum() >= 15:
            cen.append(np.median(xv[m])); med.append(np.median(yv[m]))
    return np.asarray(cen), np.asarray(med)


pred_lin = results["linear (M0+MAH)"]["pred"]
pred_gbm = results["GBM (M0+MAH)"]["pred"]
fig2, ax2 = plt.subplots(2, 4, figsize=(15.0, 6.6))
for j in range(4):
    tv = Y[:, j]
    lo, hi = np.percentile(tv, [1, 99])
    a, b = ax2[0, j], ax2[1, j]
    # top: truth vs predicted (linear scatter + both binned-median trends)
    a.scatter(tv, pred_lin[:, j], s=4, alpha=0.10, color=OKABE_ITO[0], edgecolors="none")
    a.plot([lo, hi], [lo, hi], "k--", lw=1.1)
    for pred, c, ls, lab in [(pred_lin, OKABE_ITO[0], "-", "linear"),
                             (pred_gbm, OKABE_ITO[5], "--", "GBM")]:
        cen, med = _binned(tv, pred[:, j]); a.plot(cen, med, ls, color=c, lw=1.8, label=lab)
    a.set_xlim(lo, hi); a.set_ylim(lo, hi)
    a.set_title(f"{TNAMES[j]} kpc   (RMS lin={results['linear (M0+MAH)']['rms'][j]:.3f}, "
                f"GBM={results['GBM (M0+MAH)']['rms'][j]:.3f})", fontsize=8)
    if j == 0:
        a.set_ylabel("predicted $\\log M_*$"); a.legend(fontsize=7, loc="upper left")
    # bottom: truth vs (predicted - truth)
    b.scatter(tv, pred_lin[:, j] - tv, s=4, alpha=0.10, color=OKABE_ITO[0], edgecolors="none")
    b.axhline(0, color="k", lw=1.0)
    for pred, c, ls in [(pred_lin, OKABE_ITO[0], "-"), (pred_gbm, OKABE_ITO[5], "--")]:
        cen, med = _binned(tv, pred[:, j] - tv); b.plot(cen, med, ls, color=c, lw=1.8)
    b.set_xlim(lo, hi); b.set_ylim(-0.6, 0.6)
    b.set_xlabel(f"true $\\log M_*$ ({TNAMES[j]} kpc)")
    if j == 0:
        b.set_ylabel("predicted $-$ true [dex]")
fig2.suptitle(f"exp09 — prediction check: truth vs predicted (top), residual (bottom); "
              f"linear vs GBM ceiling overlap (n={N})", fontsize=11)
fig2.tight_layout()
save_fig(fig2, FIGDIR / "exp09_prediction_check")

# %% best-fit linear model + radial trends in its coefficients ----------------
# log M*_k = b0 + b_M0*logM0 + b_PC1*PC1 + ... (PCs oriented: + = earlier-forming)
Xc = np.column_stack([np.ones(N), M0, pca])
coef = np.linalg.lstsq(Xc, Y, rcond=None)[0]            # (6, 4): feature x aperture
fstd = np.concatenate([[1.0, M0.std()], pca.std(0)])    # feature std (for standardizing)
feat = ["intercept", "logM0", "PC1", "PC2", "PC3", "PC4"]
print("\n[best-fit linear model] raw coefficients (rows=feature, cols=aperture):")
print(f"  {'feature':9s} " + " ".join(f"{n:>8s}" for n in TNAMES))
for j, f in enumerate(feat):
    print(f"  {f:9s} " + " ".join(f"{coef[j, a]:8.3f}" for a in range(4)))
print("  SHMR slope dlogM*/dlogM0 steepens outward: "
      + " -> ".join(f"{coef[1, a]:.2f}" for a in range(4)))

fig3, (axS, axP) = plt.subplots(1, 2, figsize=(9.8, 4.0))
xr = np.arange(4)
axS.plot(xr, coef[1], "-o", color=OKABE_ITO[5], ms=6)
axS.axhline(1.0, ls=":", color="0.6", lw=1)
axS.set_xticks(xr); axS.set_xticklabels(TNAMES)
axS.set_xlabel("aperture / annulus [kpc]")
axS.set_ylabel(r"SHMR slope  $d\log M_*/d\log M_0$")
axS.set_title("Halo-mass dependence steepens outward")
axS.text(-0.13, 1.04, "A", transform=axS.transAxes, fontweight="bold", fontsize=12)

for k, mk in zip(range(4), ["o", "s", "^", "D"]):
    axP.plot(xr, coef[2 + k] * fstd[2 + k], "-" + mk, color=OKABE_ITO[k], ms=5,
             label=f"PC{k+1}")
axP.axhline(0, color="k", lw=0.8)
axP.set_xticks(xr); axP.set_xticklabels(TNAMES)
axP.set_xlabel("aperture / annulus [kpc]")
axP.set_ylabel("assembly effect per +1σ  [dex]")
axP.set_title("Assembly dependence grows outward (PC2 = timing)")
axP.legend(fontsize=8)
axP.text(-0.13, 1.04, "B", transform=axP.transAxes, fontweight="bold", fontsize=12)
fig3.suptitle("exp09 — the best-fit linear model and its radial coefficient trends",
              fontsize=11)
fig3.tight_layout()
save_fig(fig3, FIGDIR / "exp09_coefficients")

# %% save outputs
OUTDIR.mkdir(parents=True, exist_ok=True)
ct = Table({"feature": feat})
for a, nm in enumerate(TNAMES):
    ct[nm] = coef[:, a]
ct.write(OUTDIR / "linear_coefficients.csv", overwrite=True)
st = Table({"model": names,
            "rms_mean": [results[n]["rms"].mean() for n in names],
            "crps_mean": overall})
st.write(OUTDIR / "ceiling_scores.csv", overwrite=True)
write_manifest(OUTDIR, params={
    "n": int(N), "features": "M0 + MAH-PCA(4)", "targets": TNAMES,
    "crps_linear": float(lin), "crps_poly2": float(poly_c), "crps_gbm": float(gbm_c),
    "gbm_gain_pct": float(100 * (lin - gbm_c) / lin)})
print(f"\nwrote figure -> {FIGDIR}\nwrote outputs -> {OUTDIR}")
