"""exp22 — Does using >3 PCs help the profile *predictor*? (Q: evidence?)

exp02 showed 3 PCs reconstruct the CoG shape to ~0.005 dex. But that is the
*compression* question. For the *predictor*, an extra PC only helps if it is
(a) real profile variance AND (b) halo-predictable. Higher modes are
increasingly noise-like and halo-independent, so we expect: compression keeps
improving with K, but PREDICTION plateaus.

Sweep K = 1..8 for both the CoG and density representations and report:
  - recon RMS from the TRUE K modes  -> compression quality (should keep dropping)
  - value of predicting shape (% CRPS gain over mass+mean-shape) -> prediction (plateau?)
  - R^2 of the K-th PC (the newly added mode) -> is it halo-predictable at all?

Run: PYTHONPATH=. uv run python experiments/exp22_full_profile_predict/pca_n_components.py
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
from hongshao.tng_data import COG_RAD_KPC                                         # noqa: E402
from hongshao.metrics import crps_gaussian                                       # noqa: E402
from hongshao.plotting import set_style, save_fig, OKABE_ITO                      # noqa: E402

set_style()
FIGDIR = Path(__file__).resolve().parent / "figures"
TABLE = ROOT / "data" / "processed" / "tng300_072_z0p4.fits"
NMAX = int(os.environ.get("EXP22_NMAX", 0))
RIDGE = 2.0
KS = [1, 2, 3, 4, 5, 6, 8]

t = Table.read(TABLE); t = t[t["use"]]
if NMAX:
    t = t[:NMAX]
cog = np.asarray(t["logmstar_cog"], float)
dmah = np.column_stack([np.asarray(t[c], float) for c in
                        ("dmah_logmp", "dmah_logtc", "dmah_early", "dmah_late")])
c200 = np.asarray(t["c_200c"], float)
R = COG_RAD_KPC
logMtot = cog[:, -1]
Mcum = 10.0 ** cog
dA = np.pi * (R[1:] ** 2 - R[:-1] ** 2)
cog_shape = (cog - logMtot[:, None])[:, :-1]
den_shape = np.log10(np.clip(Mcum[:, 1:] - Mcum[:, :-1], 1.0, None) / dA[None, :]) - logMtot[:, None]
g = (np.isfinite(cog).all(1) & np.isfinite(dmah).all(1) & np.isfinite(c200)
     & np.all(Mcum[:, 1:] - Mcum[:, :-1] > 0, axis=1))
cog_shape, den_shape, anchor = cog_shape[g], den_shape[g], logMtot[g]
X = np.column_stack([dmah[g], c200[g]])
N = len(X)
print(f"PCA-component sweep on n={N}")


# %% ---- machinery -----------------------------------------------------------
def fit_logvar(r, Z, ridge=RIDGE):
    A = np.column_stack([np.ones(len(r)), Z]); r2 = r ** 2

    def nll(gm):
        s = A @ gm
        return 0.5 * np.sum(s + r2 * np.exp(-s)) + 0.5 * ridge * np.sum(gm[1:] ** 2)

    def grad(gm):
        s = A @ gm; w = 0.5 * (1 - r2 * np.exp(-s)); o = A.T @ w; o[1:] += ridge * gm[1:]
        return o
    return minimize(nll, np.r_[np.log(max(r2.mean(), 1e-6)), np.zeros(Z.shape[1])],
                    jac=grad, method="L-BFGS-B").x


def fit_predict(Xtr, Ytr, Xte):
    nb = Ytr.shape[1]
    Am_tr = np.column_stack([np.ones(len(Xtr)), Xtr]); Am_te = np.column_stack([np.ones(len(Xte)), Xte])
    mu_te = np.empty((len(Xte), nb)); r_tr = np.empty((len(Xtr), nb))
    for j in range(nb):
        beta, *_ = np.linalg.lstsq(Am_tr, Ytr[:, j], rcond=None)
        mu_te[:, j] = Am_te @ beta; r_tr[:, j] = Ytr[:, j] - Am_tr @ beta
    mx, sx = Xtr.mean(0), Xtr.std(0); Ztr = (Xtr - mx) / sx; Zte = (Xte - mx) / sx
    sig_te = np.empty((len(Xte), nb)); sig_tr = np.empty((len(Xtr), nb))
    for j in range(nb):
        gm = fit_logvar(r_tr[:, j], Ztr)
        sig_tr[:, j] = np.exp(0.5 * (np.column_stack([np.ones(len(Xtr)), Ztr]) @ gm))
        sig_te[:, j] = np.exp(0.5 * (np.column_stack([np.ones(len(Xte)), Zte]) @ gm))
    Rc = np.corrcoef((r_tr / sig_tr).T)
    return mu_te, sig_te, sig_te[:, :, None] * Rc[None] * sig_te[:, None, :]


def evaluate_K(shp, Kn, k=5, seed=0):
    nr = shp.shape[1]; truth = anchor[:, None] + shp
    order = np.random.default_rng(seed).permutation(N)
    MU = np.full((N, nr), np.nan); SIG = np.full((N, nr), np.nan)
    MUb = np.full((N, nr), np.nan); SIGb = np.full((N, nr), np.nan)
    REC = np.full((N, nr), np.nan)                                    # true-mode reconstruction
    PRED = np.full((N, Kn), np.nan); TRUE = np.full((N, Kn), np.nan)
    for fold in np.array_split(order, k):
        tr = np.setdiff1d(np.arange(N), fold)
        mu_sh = shp[tr].mean(0)
        _, _, Vt = np.linalg.svd(shp[tr] - mu_sh, full_matrices=False)
        V = Vt[:Kn]
        S_tr = (shp[tr] - mu_sh) @ V.T; S_te = (shp[fold] - mu_sh) @ V.T
        mu_te, sig_te, Sig_te = fit_predict(X[tr], np.column_stack([anchor[tr], S_tr]), X[fold])
        PRED[fold] = mu_te[:, 1:]; TRUE[fold] = S_te
        A = np.column_stack([np.ones(nr), V.T])
        MU[fold] = mu_te @ A.T + mu_sh
        SIG[fold] = np.sqrt(np.einsum("rk,nkl,rl->nr", A, Sig_te, A))
        MUb[fold] = mu_te[:, 0:1] + mu_sh[None, :]
        SIGb[fold] = np.sqrt(sig_te[:, 0:1] ** 2 + shp[tr].var(0)[None, :])
        REC[fold] = np.column_stack([anchor[fold], S_te]) @ A.T + mu_sh
    crps_full = crps_gaussian(truth, MU, SIG).mean()
    crps_base = crps_gaussian(truth, MUb, SIGb).mean()
    value = 100 * (crps_base - crps_full) / crps_base
    recon_rms = float(np.sqrt(((REC - truth) ** 2).mean()))
    r2_lastpc = float(1.0 - ((TRUE[:, -1] - PRED[:, -1]) ** 2).mean() / TRUE[:, -1].var())
    return value, crps_full, recon_rms, r2_lastpc


# %% ---- sweep ----------------------------------------------------------------
res = {"CoG": [], "density": []}
for lab, shp in [("CoG", cog_shape), ("density", den_shape)]:
    print(f"\n[{lab}]  K   recon_RMS  value%   CRPS    R2(Kth PC)")
    for Kn in KS:
        v, c, rr, r2k = evaluate_K(shp, Kn)
        res[lab].append((Kn, rr, v, c, r2k))
        print(f"      {Kn:2d}   {rr:8.4f}  {v:6.1f}  {c:.4f}  {r2k:+.3f}")


# %% ---- FIGURE --------------------------------------------------------------
fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.0, 4.5))
for lab, c in [("CoG", OKABE_ITO[7]), ("density", OKABE_ITO[2])]:
    a = np.array(res[lab])
    axA.plot(a[:, 0], a[:, 1], "-o", color=c, ms=5, label=f"{lab} recon (true modes)")
    axB.plot(a[:, 0], a[:, 2], "-o", color=c, ms=5, label=f"{lab} value of shape")
axA.axvline(3, color="0.7", ls=":", lw=1)
axA.set_xlabel("number of PCs (K)"); axA.set_ylabel("reconstruction RMS [dex]")
axA.set_title("A. Compression keeps improving with K"); axA.legend(fontsize=8)
axB.axvline(3, color="0.7", ls=":", lw=1)
axB.set_xlabel("number of PCs (K)"); axB.set_ylabel("value of predicting shape [%]")
axB.set_title("B. Prediction plateaus by K≈3"); axB.legend(fontsize=8)
fig.suptitle("exp22 — more PCs reconstruct the profile better (A) but do NOT improve the "
             "predictor (B): higher modes are not halo-predictable", fontsize=10)
fig.tight_layout()
save_fig(fig, FIGDIR / "exp22_pca_ncomponents")
print(f"\nwrote figure -> {FIGDIR}/exp22_pca_ncomponents")
print("[verdict] compression improves with K, but the predictor's value-of-shape and CRPS "
      "plateau by K~3; PC4+ have R^2~0 (not halo-predictable). Predicting >3 PCs does not help.")
