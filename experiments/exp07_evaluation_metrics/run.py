"""exp07 — What metrics should judge the Ultimate-SHMR models?

Before building the probabilistic emulator we settle on *how to score it*. Two
tracks, two model families:

TRACK 1 — predictors (regression that outputs P(target | halo)):
  (a) Evaluate recovery in **physical aperture / annulus stellar masses**
      (<10, 10-30, 30-50, 50-100, <100 kpc) instead of per-radius CoG dex —
      cumulative CoG points are nearly perfectly correlated, so per-radius dex
      double-counts; annulus masses isolate where a model actually succeeds.
  (b) **Distributional scores** (CRPS, predictive log-score) and **calibration**
      (interval coverage) on a Gaussian-scatter baseline — a point prediction
      with good RMS can still state dishonest uncertainties.
  (c) **Predicted-vs-true covariance**: the residual scatter across apertures is
      correlated, so the emulator must draw *correlated* scatter, not per-radius
      independent noise. The metric is the residual correlation matrix.

TRACK 2 — profile fits (the 5-param radial-DiffMAH CoG model):
  (d) **Reduced chi-square** using per-point CoG errors propagated from the
      isophote intensity error (see hongshao.tng_data.cog_sigma_dex).
  (e) **AIC / BIC** model comparison: single vs double sigmoid, vs Sersic, vs a
      3-mode PCA reconstruction — is the 5-parameter single sigmoid justified?
  (f) **The residual profile**: coherent (non-zero-mean) residuals vs radius
      flag a missing component.

Deliverable: the recommended suite, graduated into hongshao/metrics.py and
hongshao/tng_data.cog_sigma_dex, reused by exp08+.

Run from the repo root (EXP07_NMAX=300 for a quick sub-minute pass):
    PYTHONPATH=. uv run python experiments/exp07_evaluation_metrics/run.py
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
from scipy.optimize import least_squares
from scipy.special import expit, gammainc, gammaincinv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hongshao.tng_data import COG_RAD_KPC, SMA_KPC, cog_sigma_dex          # noqa: E402
from hongshao.profiles import fit_cog, cog_from_physical, LN10             # noqa: E402
from hongshao.metrics import (crps_gaussian, gaussian_logscore,           # noqa: E402
                              interval_coverage, aic_bic)
from hongshao.plotting import set_style, save_fig, OKABE_ITO              # noqa: E402
from hongshao.provenance import write_manifest                            # noqa: E402

set_style()
HERE = Path(__file__).resolve().parent
OUTDIR, FIGDIR = HERE / "outputs", HERE / "figures"
TABLE = ROOT / "data" / "processed" / "tng300_072_z0p4.fits"
R = np.asarray(COG_RAD_KPC, float)
U = np.log(R)

t = Table.read(TABLE)
t = t[t["use"]]
NMAX = int(os.environ.get("EXP07_NMAX", 0))
if NMAX:
    t = t[:NMAX]
logM0 = np.asarray(t["logm0_halo"], float)
cog = np.asarray(t["logmstar_cog"], float)                  # (N, 24) log10 Msun
aper = np.asarray(t["logmstar_aper"], float)                # (N, 7) at SMA_KPC
index = np.asarray(t["index"])
mah_names = ["logmpeak_z0p7", "logmpeak_z1", "logmpeak_z1p5",
             "logmpeak_z2", "z50", "z75", "z90"]
MAH = np.column_stack([np.asarray(t[c], float) for c in mah_names])
N = len(t)
print(f"analysis sample (use cut): {N} galaxies   SMA_KPC={SMA_KPC.tolist()}")


# %% =================== TRACK 1: predictor evaluation ========================
# Physical aperture / annulus masses, built from the *measured* aperture array
# (SMA_KPC = [10,30,50,75,100,120,150] kpc) so no CoG interpolation is needed.
def _annulus(a_outer, a_inner):
    return np.log10(np.clip(10.0 ** a_outer - 10.0 ** a_inner, 1.0, None))


targets = {
    "<10": aper[:, 0],
    "10-30": _annulus(aper[:, 1], aper[:, 0]),
    "30-50": _annulus(aper[:, 2], aper[:, 1]),
    "50-100": _annulus(aper[:, 4], aper[:, 2]),
    "<100": aper[:, 4],
}
tnames = list(targets)
Y = np.column_stack([targets[k] for k in tnames])           # (N, 5)
fin = (np.isfinite(Y).all(1) & np.isfinite(logM0)
       & np.isfinite(MAH).all(1))
Y, logM0f, MAHf = Y[fin], logM0[fin], MAH[fin]
Nf = len(Y)
print(f"track-1 finite sample: {Nf}   targets: {tnames}")


def cv_predict_sigma(X, Y2, k=5, seed=0):
    """5-fold out-of-fold mean and Gaussian predictive sigma (homoscedastic per
    target, sigma estimated from each fold's training residuals)."""
    n = len(Y2)
    Xd = np.column_stack([np.ones(n), X])
    p = Xd.shape[1]
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    mu = np.full_like(Y2, np.nan, dtype=float)
    sig = np.full_like(Y2, np.nan, dtype=float)
    for fold in np.array_split(order, k):
        tr = np.setdiff1d(np.arange(n), fold)
        beta, *_ = np.linalg.lstsq(Xd[tr], Y2[tr], rcond=None)
        mu[fold] = Xd[fold] @ beta
        resid_tr = Y2[tr] - Xd[tr] @ beta
        sig[fold] = resid_tr.std(0, ddof=p)                 # per-target sigma
    return mu, sig


def shuffle_within_bins(M, binvar, nbins=12, seed=1):
    rng = np.random.default_rng(seed)
    out = M.copy()
    edges = np.quantile(binvar, np.linspace(0, 1, nbins + 1))
    b = np.digitize(binvar, edges[1:-1])
    for bi in np.unique(b):
        ii = np.where(b == bi)[0]
        out[ii] = M[ii][rng.permutation(len(ii))]
    return out


X_base = logM0f[:, None]
X_full = np.column_stack([logM0f, MAHf])
X_shuf = np.column_stack([logM0f, shuffle_within_bins(MAHf, logM0f)])

mu_b, sig_b = cv_predict_sigma(X_base, Y)
mu_f, sig_f = cv_predict_sigma(X_full, Y)
mu_s, sig_s = cv_predict_sigma(X_shuf, Y)

rms_b = np.sqrt(np.mean((Y - mu_b) ** 2, 0))
rms_f = np.sqrt(np.mean((Y - mu_f) ** 2, 0))
rms_s = np.sqrt(np.mean((Y - mu_s) ** 2, 0))
print("\n[track 1] per-target recovery RMS [dex]:")
for j, nm in enumerate(tnames):
    print(f"  {nm:7s}: M0={rms_b[j]:.4f}  M0+hist={rms_f[j]:.4f}  "
          f"shuf={rms_s[j]:.4f}  -> {100*(rms_b[j]-rms_f[j])/rms_b[j]:+.1f}% "
          f"(shuf {100*(rms_b[j]-rms_s[j])/rms_b[j]:+.1f}%)")

# distributional scores (proper): mean over galaxies, per target
crps_b = crps_gaussian(Y, mu_b, sig_b).mean(0)
crps_f = crps_gaussian(Y, mu_f, sig_f).mean(0)
ls_b = gaussian_logscore(Y, mu_b, sig_b).mean(0)
ls_f = gaussian_logscore(Y, mu_f, sig_f).mean(0)
print("\n[track 1] distributional scores (lower better), overall:")
print(f"  CRPS [dex]:  M0={crps_b.mean():.4f}  M0+hist={crps_f.mean():.4f}  "
      f"({100*(crps_b.mean()-crps_f.mean())/crps_b.mean():+.1f}%)")
print(f"  log-score:   M0={ls_b.mean():.4f}  M0+hist={ls_f.mean():.4f}  "
      f"({ls_b.mean()-ls_f.mean():+.3f} nats)")

# calibration of the M0+history Gaussian baseline (pool all targets)
levels, cov_f = interval_coverage(Y, mu_f, sig_f)
print("\n[track 1] calibration (M0+hist), nominal vs empirical coverage:")
print("  " + "  ".join(f"{int(100*L)}%:{c:.2f}" for L, c in zip(levels, cov_f)))

# predicted-vs-true covariance: residual correlation across apertures
resid_f = Y - mu_f
corr_resid = np.corrcoef(resid_f.T)
offdiag = corr_resid[~np.eye(5, dtype=bool)]
print(f"\n[track 1] residual correlation across apertures: mean |off-diag| ="
      f" {np.mean(np.abs(offdiag)):.2f}  (independent-scatter model assumes 0)")
# contrast: the 24 cumulative CoG points are almost perfectly correlated
cogf = cog[fin]
corr_cog = np.corrcoef(cogf.T)
print(f"           cumulative CoG (24 pts) mean |off-diag| = "
      f"{np.mean(np.abs(corr_cog[~np.eye(24, dtype=bool)])):.3f}  "
      f"(why per-radius dex double-counts)")


# %% =================== TRACK 2: profile-fit diagnostics =====================
def _cumint(beta):
    """Cumulative trapezoid of beta over U (= ln R), from the first point."""
    return np.concatenate([[0.0], np.cumsum(0.5 * (beta[1:] + beta[:-1]) * np.diff(U))])


def model_single(y):
    """Cached-style single-sigmoid radial-DiffMAH fit. Returns (model_log, k=5)."""
    f = fit_cog(R, y)
    m = cog_from_physical(R, f["logMstar0"], f["beta_in"], f["beta_out"],
                          f["R_c"], f["Delta"])
    return np.asarray(m, float), 5


def model_double(y):
    """Double-sigmoid (two radial transitions), 8 params. Returns (model_log, 8)."""
    yln = y * LN10
    lo = [yln[0] - 5, 0.0, 0.0, U[0], 0.05, 0.0, U[0], 0.05]
    hi = [yln[0] + 5, 3.0, 6.0, U[-1], 3.0, 6.0, U[-1], 3.0]
    x0 = np.clip([yln[0], 0.1, 1.0, np.log(8.0), 0.5, 0.8, np.log(40.0), 0.5], lo, hi)

    def resid(p):
        lnM0, b_out, db1, uc1, d1, db2, uc2, d2 = p
        beta = b_out + db1 * expit((uc1 - U) / d1) + db2 * expit((uc2 - U) / d2)
        return (lnM0 + _cumint(beta) - yln) / LN10

    sol = least_squares(resid, x0, bounds=(lo, hi), method="trf", max_nfev=3000)
    return y + resid(sol.x), 8


def model_sersic(y):
    """Sersic curve of growth (incomplete-gamma), 3 params. Returns (model_log, 3)."""
    lo = [y[-1] - 1.0, np.log10(1.0), 0.5]
    hi = [y[-1] + 2.0, np.log10(300.0), 12.0]
    x0 = np.clip([y[-1] + 0.05, np.log10(20.0), 4.0], lo, hi)

    def model_log(p):
        log_mtot, log_re, n = p
        b_n = gammaincinv(2.0 * n, 0.5)
        frac = gammainc(2.0 * n, b_n * (R / 10.0 ** log_re) ** (1.0 / n))
        return log_mtot + np.log10(np.clip(frac, 1e-12, 1.0))

    sol = least_squares(lambda p: model_log(p) - y, x0, bounds=(lo, hi),
                        method="trf", max_nfev=3000)
    return model_log(sol.x), 3


# PCA reconstruction with a shared 3-mode basis (the flexible non-parametric ref)
cog_t2 = cog[np.isfinite(cog).all(1)]
mu_pca = cog_t2.mean(0)
_, _, Vt_pca = np.linalg.svd(cog_t2 - mu_pca, full_matrices=False)
KPCA = 3
Vk = Vt_pca[:KPCA]


def model_pca(y):
    """Reconstruct from the top-KPCA shared modes. Returns (model_log, KPCA)."""
    return mu_pca + (y - mu_pca) @ Vk.T @ Vk, KPCA


# fit every galaxy with the four models; collect RSS, AIC, BIC, chi2
models = {"single": model_single, "double": model_double,
          "Sersic": model_sersic, f"PCA-{KPCA}": model_pca}
n_pts = len(R)
aic = {m: np.full(N, np.nan) for m in models}        # CoG-space (24 cumulative pts)
bic = {m: np.full(N, np.nan) for m in models}
aic_ann = {m: np.full(N, np.nan) for m in models}    # annulus-space (decorrelated)
bic_ann = {m: np.full(N, np.nan) for m in models}
resid_single = np.full((N, n_pts), np.nan)
chi2_red = np.full(N, np.nan)
sigma_dex = np.full((N, n_pts), np.nan)


def _log_annuli(cog_log):
    """log10 of the 23 annulus masses from a (monotonic) log CoG; NaN if a
    model annulus is non-positive (can happen for the PCA reconstruction)."""
    ann = np.diff(10.0 ** cog_log)
    return np.log10(np.where(ann > 0, ann, np.nan))


ok2 = np.isfinite(cog).all(1)
print(f"\n[track 2] fitting {ok2.sum()} galaxies with {list(models)} ...")
for i in np.where(ok2)[0]:
    y = cog[i]
    sg = cog_sigma_dex(int(index[i]))
    sigma_dex[i] = sg
    y_ann = _log_annuli(y)
    for mname, mfun in models.items():
        m_log, k = mfun(y)
        r = y - m_log
        aic[mname][i], bic[mname][i] = aic_bic(np.sum(r ** 2), n_pts, k + 1)
        # decorrelated comparison: residuals of the (weakly correlated) annuli
        r_ann = _log_annuli(m_log) - y_ann
        good_ann = np.isfinite(r_ann)
        if good_ann.sum() > k + 2:
            aic_ann[mname][i], bic_ann[mname][i] = aic_bic(
                np.sum(r_ann[good_ann] ** 2), int(good_ann.sum()), k + 1)
        if mname == "single":
            resid_single[i] = r
            if np.all(np.isfinite(sg)) and np.all(sg > 0):
                chi2_red[i] = np.sum((r / sg) ** 2) / (n_pts - k)

print(f"[track 2] reduced chi-square (single sigmoid, naive independent pts): "
      f"median={np.nanmedian(chi2_red):.2f}  "
      f"16-84%=[{np.nanpercentile(chi2_red,16):.2f},{np.nanpercentile(chi2_red,84):.2f}]")
print(f"          per-point sigma [dex]: inner~{np.nanmedian(sigma_dex[:,0]):.4f}  "
      f"outer~{np.nanmedian(sigma_dex[:,-1]):.4f}")

# AIC/BIC model comparison: fraction of galaxies each model wins, in both the
# (correlated) CoG space and the (weakly correlated) annulus space.
def _win_fraction(score_dict):
    M = np.column_stack([score_dict[m] for m in models])
    g = np.isfinite(M).all(1)
    return np.array([np.mean(np.argmin(M[g], 1) == j) for j in range(len(models))]), g.sum()


win_aic, ng = _win_fraction(aic)
win_bic, _ = _win_fraction(bic)
win_aic_ann, ng_ann = _win_fraction(aic_ann)
win_bic_ann, _ = _win_fraction(bic_ann)
med_aic = {m: np.nanmedian(aic[m]) for m in models}
med_bic = {m: np.nanmedian(bic[m]) for m in models}
print("\n[track 2] model comparison, fraction of galaxies preferring each model:")
print(f"  {'model':8s}  AIC(CoG) BIC(CoG) | AIC(ann) BIC(ann)")
for j, m in enumerate(models):
    print(f"  {m:8s}  {100*win_aic[j]:6.0f}% {100*win_bic[j]:6.0f}% | "
          f"{100*win_aic_ann[j]:6.0f}% {100*win_bic_ann[j]:6.0f}%")
print("  (CoG: 24 cumulative pts treated as independent overcounts -> over-rewards"
      " the double sigmoid; annulus space is the honest comparison.)")

# residual profile (single sigmoid): coherent structure => missing component
mean_resid = np.nanmean(resid_single, 0)
std_resid = np.nanstd(resid_single, 0)
print(f"\n[track 2] residual profile (single): max |mean residual| = "
      f"{np.nanmax(np.abs(mean_resid)):.4f} dex at R={R[np.nanargmax(np.abs(mean_resid))]:.1f} kpc")


# %% ======================== FIGURES ========================================
# FIGURE 1 — track 1 recovery in physical aperture/annulus masses
fig1, (axA, axB) = plt.subplots(1, 2, figsize=(9.2, 4.0))
x = np.arange(len(tnames))
w = 0.27
axA.bar(x - w, rms_b, w, color=OKABE_ITO[7], label=r"$M_0$ only")
axA.bar(x, rms_f, w, color=OKABE_ITO[0], label=r"$M_0$ + history")
axA.bar(x + w, rms_s, w, color="0.7", label="shuffled history")
axA.set_xticks(x); axA.set_xticklabels(tnames)
axA.set_xlabel("aperture / annulus [kpc]")
axA.set_ylabel("cross-validated recovery RMS [dex]")
axA.set_title("Recovery in physical stellar masses")
axA.legend()
axA.text(-0.13, 1.04, "A", transform=axA.transAxes, fontweight="bold", fontsize=12)

impr = 100 * (rms_b - rms_f) / rms_b
impr_s = 100 * (rms_b - rms_s) / rms_b
axB.bar(x - w / 2, impr, w, color=OKABE_ITO[5], label="history")
axB.bar(x + w / 2, impr_s, w, color="0.7", label="shuffled")
axB.axhline(0, color="k", lw=0.8)
axB.set_xticks(x); axB.set_xticklabels(tnames)
axB.set_xlabel("aperture / annulus [kpc]")
axB.set_ylabel("scatter reduction from history [%]")
axB.set_title("Where assembly history helps")
axB.legend()
axB.text(-0.13, 1.04, "B", transform=axB.transAxes, fontweight="bold", fontsize=12)
fig1.suptitle(f"exp07 — predictor recovery in aperture/annulus masses (n={Nf})",
              fontsize=11)
fig1.tight_layout()
save_fig(fig1, FIGDIR / "exp07_track1_recovery")

# FIGURE 2 — distributional scores, calibration, covariance
fig2, (axC, axD, axE) = plt.subplots(1, 3, figsize=(13.2, 4.0))
axC.bar(x - w / 2, crps_gaussian(Y, mu_b, sig_b).mean(0), w,
        color=OKABE_ITO[7], label=r"$M_0$ only")
axC.bar(x + w / 2, crps_gaussian(Y, mu_f, sig_f).mean(0), w,
        color=OKABE_ITO[0], label=r"$M_0$ + history")
axC.set_xticks(x); axC.set_xticklabels(tnames)
axC.set_xlabel("aperture / annulus [kpc]")
axC.set_ylabel("CRPS [dex]  (lower = better)")
axC.set_title("Distributional score")
axC.legend()
axC.text(-0.18, 1.04, "C", transform=axC.transAxes, fontweight="bold", fontsize=12)

axD.plot([0, 1], [0, 1], ":", color="0.5", lw=1)
axD.plot(levels, cov_f, "-o", color=OKABE_ITO[0], ms=5, label=r"$M_0$ + history")
_, cov_b = interval_coverage(Y, mu_b, sig_b)
axD.plot(levels, cov_b, "-s", color=OKABE_ITO[7], ms=4, label=r"$M_0$ only")
axD.set_xlabel("nominal central interval")
axD.set_ylabel("empirical coverage")
axD.set_title("Calibration (Gaussian-scatter baseline)")
axD.set_xlim(0.4, 1.0); axD.set_ylim(0.4, 1.0)
axD.legend()
axD.text(-0.18, 1.04, "D", transform=axD.transAxes, fontweight="bold", fontsize=12)

im = axE.imshow(corr_resid, cmap="RdBu_r", vmin=-1, vmax=1)
axE.set_xticks(x); axE.set_xticklabels(tnames, rotation=30, ha="right", fontsize=7)
axE.set_yticks(x); axE.set_yticklabels(tnames, fontsize=7)
for ii in range(5):
    for jj in range(5):
        axE.text(jj, ii, f"{corr_resid[ii, jj]:+.2f}", ha="center", va="center",
                 fontsize=6, color="white" if abs(corr_resid[ii, jj]) > 0.5 else "k")
axE.set_title(f"Residual covariance\n(mean |off-diag|={np.mean(np.abs(offdiag)):.2f})")
fig2.colorbar(im, ax=axE, shrink=0.8, label="correlation")
axE.text(-0.25, 1.04, "E", transform=axE.transAxes, fontweight="bold", fontsize=12)
fig2.suptitle("exp07 — scoring the predictive distribution and its covariance",
              fontsize=11)
fig2.tight_layout()
save_fig(fig2, FIGDIR / "exp07_track1_distribution")

# FIGURE 3 — track 2 fit diagnostics
fig3, (axF, axG, axH) = plt.subplots(1, 3, figsize=(13.2, 4.0))
cc = chi2_red[np.isfinite(chi2_red)]
axF.hist(cc, bins=np.linspace(0, np.percentile(cc, 98), 40),
         color=OKABE_ITO[7], alpha=0.8)
axF.axvline(np.median(cc), color=OKABE_ITO[5], lw=1.5,
            label=f"median {np.median(cc):.2f}")
axF.axvline(1.0, color=OKABE_ITO[2], ls="--", lw=1.3, label=r"$\chi^2_\nu=1$")
axF.set_xlabel(r"reduced $\chi^2$ (single sigmoid)")
axF.set_ylabel("galaxies")
axF.set_title("Goodness-of-fit vs propagated noise")
axF.legend()
axF.text(-0.16, 1.04, "F", transform=axF.transAxes, fontweight="bold", fontsize=12)

axG.axhline(0, color="k", lw=0.8)
axG.fill_between(R, mean_resid - std_resid, mean_resid + std_resid,
                 color=OKABE_ITO[0], alpha=0.2, label=r"$\pm$ scatter")
axG.plot(R, mean_resid, "-o", color=OKABE_ITO[0], ms=3, label="single sigmoid")
axG.plot(R, np.nanmedian(sigma_dex, 0), ":", color=OKABE_ITO[5], lw=1.3,
         label="propagated noise")
axG.plot(R, -np.nanmedian(sigma_dex, 0), ":", color=OKABE_ITO[5], lw=1.3)
axG.set_xscale("log")
axG.set_xlabel("radius R [kpc]")
axG.set_ylabel("mean residual  data $-$ model [dex]")
axG.set_title("Residual profile (coherent = missing component)")
axG.legend(fontsize=7)
axG.text(-0.16, 1.04, "G", transform=axG.transAxes, fontweight="bold", fontsize=12)

mn = list(models)
xb = np.arange(len(mn))
axH.bar(xb - w / 2, 100 * win_bic, w, color=OKABE_ITO[5],
        label="CoG space (24 corr. pts)")
axH.bar(xb + w / 2, 100 * win_bic_ann, w, color=OKABE_ITO[0],
        label="annulus space (decorr.)")
axH.set_xticks(xb); axH.set_xticklabels(mn, rotation=20, ha="right", fontsize=8)
axH.set_ylabel("galaxies preferring model (BIC) [%]")
axH.set_title("Model comparison: correlation matters")
axH.legend(fontsize=7)
axH.text(-0.16, 1.04, "H", transform=axH.transAxes, fontweight="bold", fontsize=12)
fig3.suptitle(f"exp07 — profile-fit diagnostics (n={int(ok2.sum())})", fontsize=11)
fig3.tight_layout()
save_fig(fig3, FIGDIR / "exp07_track2_fit_diagnostics")

# %% save outputs
OUTDIR.mkdir(parents=True, exist_ok=True)
t1 = Table({"target": tnames, "rms_M0": rms_b, "rms_M0_hist": rms_f,
            "rms_shuffled": rms_s,
            "crps_M0": crps_gaussian(Y, mu_b, sig_b).mean(0),
            "crps_M0_hist": crps_f, "logscore_M0": ls_b, "logscore_M0_hist": ls_f})
t1.write(OUTDIR / "track1_predictor_scores.csv", overwrite=True)
t2 = Table({"model": list(models),
            "median_aic_cog": [med_aic[m] for m in models],
            "median_bic_cog": [med_bic[m] for m in models],
            "win_bic_cog_pct": 100 * win_bic,
            "win_bic_annulus_pct": 100 * win_bic_ann})
t2.write(OUTDIR / "track2_model_comparison.csv", overwrite=True)
np.savez(OUTDIR / "diagnostics.npz", R=R, mean_resid=mean_resid,
         std_resid=std_resid, chi2_red=chi2_red, corr_resid=corr_resid,
         levels=levels, coverage=cov_f, sigma_dex_median=np.nanmedian(sigma_dex, 0))
write_manifest(OUTDIR, params={
    "n_track1": int(Nf), "n_track2": int(ok2.sum()),
    "targets": tnames, "models": list(models),
    "median_chi2_red": float(np.nanmedian(chi2_red)),
    "crps_improvement_pct": float(100 * (crps_b.mean() - crps_f.mean()) / crps_b.mean()),
    "resid_corr_mean_offdiag": float(np.mean(np.abs(offdiag)))})
print(f"\nwrote figures -> {FIGDIR}\nwrote outputs -> {OUTDIR}")
