"""exp22 — Predict the *whole* curve of growth, not just a few aperture masses.

The graduated emulator predicts 4 aperture/annulus masses. But exp02 showed a
massive galaxy's profile *shape* is 2-3 dimensional (PCA of the curve of growth:
2 modes -> 0.01 dex, 3 -> 0.005 dex), and exp03 gave a 5-number parametric fit.
So we can compress each CoG to a few numbers and predict *those* from the halo,
then reconstruct the entire profile.

Here: compress the CoG to [logMtot, PC1, PC2, PC3] (total mass + 3 shape modes),
predict that 4-vector from portable DiffMAH+c_200c with the same probabilistic
emulator (linear mean + heteroscedastic full covariance, exp19), and — because
the reconstruction is *linear* in the compressed vector — propagate the
predictive Gaussian analytically to a **per-radius predictive CoG**. Evaluate in
profile space: per-radius CRPS / calibration / reconstruction RMS.

Key question: how much of the profile is halo-predictable? We expect total mass
(the SHMR) to dominate; the new content is whether the halo predicts the *shape*
(the PCs) beyond a population-average profile. Baseline = "total mass + mean
shape" (predict logMtot, assume the average shape with its full scatter); the
full model adds halo-predicted PCs.

PCA is fit **inside each CV fold** (train only) to avoid leakage. Library
(`hongshao/emulator.py`) untouched — local CV machinery.

Run (fast):  EXP22_NMAX=700 PYTHONPATH=. uv run python experiments/exp22_full_profile_predict/run.py
Full:        PYTHONPATH=. uv run python experiments/exp22_full_profile_predict/run.py
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
from hongshao.metrics import crps_gaussian, interval_coverage                    # noqa: E402
from hongshao.plotting import set_style, save_fig, OKABE_ITO                      # noqa: E402
from hongshao.provenance import write_manifest                                   # noqa: E402

set_style()
HERE = Path(__file__).resolve().parent
OUTDIR, FIGDIR = HERE / "outputs", HERE / "figures"
TABLE = ROOT / "data" / "processed" / "tng300_072_z0p4.fits"
NMAX = int(os.environ.get("EXP22_NMAX", 0))
RIDGE = float(os.environ.get("EXP22_RIDGE", 2.0))
K = int(os.environ.get("EXP22_K", 3))                  # number of shape PCs

t = Table.read(TABLE); t = t[t["use"]]
if NMAX:
    t = t[:NMAX]
cog = np.asarray(t["logmstar_cog"], float)             # log10 M(<R), (N,24)
dmah = np.column_stack([np.asarray(t[c], float) for c in
                        ("dmah_logmp", "dmah_logtc", "dmah_early", "dmah_late")])
c200 = np.asarray(t["c_200c"], float)
rad = COG_RAD_KPC[:-1]                                  # 23 radii (last = anchor)
NR = len(rad)

logMtot = cog[:, -1]                                    # log M(<148 kpc) = profile anchor
shape = (cog - logMtot[:, None])[:, :-1]               # (N,23) CoG shape, total divided out
g = np.isfinite(cog).all(1) & np.isfinite(dmah).all(1) & np.isfinite(c200)
shape, cog, logMtot = shape[g], cog[g], logMtot[g]
X = np.column_stack([dmah[g], c200[g]])
N = len(X)
cog_true = cog[:, :NR]                                  # true CoG at the 23 radii
print(f"exp22: full-profile emulator on n={N}, K={K} shape modes")


# %% ---- emulator: linear mean + heteroscedastic full covariance (exp19) ------
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
    """Train the emulator on (Xtr,Ytr), predict (mu, sigma, Sigma) for Xte."""
    nb = Ytr.shape[1]
    Am_tr = np.column_stack([np.ones(len(Xtr)), Xtr])
    Am_te = np.column_stack([np.ones(len(Xte)), Xte])
    mu_te = np.empty((len(Xte), nb)); r_tr = np.empty((len(Xtr), nb))
    for j in range(nb):
        beta, *_ = np.linalg.lstsq(Am_tr, Ytr[:, j], rcond=None)
        mu_te[:, j] = Am_te @ beta
        r_tr[:, j] = Ytr[:, j] - Am_tr @ beta
    mx, sx = Xtr.mean(0), Xtr.std(0)
    Ztr = (Xtr - mx) / sx; Zte = (Xte - mx) / sx
    sig_tr = np.empty((len(Xtr), nb)); sig_te = np.empty((len(Xte), nb))
    for j in range(nb):
        gm = fit_logvar(r_tr[:, j], Ztr)
        sig_tr[:, j] = np.exp(0.5 * (np.column_stack([np.ones(len(Xtr)), Ztr]) @ gm))
        sig_te[:, j] = np.exp(0.5 * (np.column_stack([np.ones(len(Xte)), Zte]) @ gm))
    Rc = np.corrcoef((r_tr / sig_tr).T)
    Sig_te = sig_te[:, :, None] * Rc[None] * sig_te[:, None, :]
    return mu_te, sig_te, Sig_te


# %% ---- CV: PCA in-fold, predict [logMtot, PCs], reconstruct per-radius ------
def cv(k=5, seed=0):
    order = np.random.default_rng(seed).permutation(N)
    MU = np.full((N, NR), np.nan); SIG = np.full((N, NR), np.nan)         # full model
    MUb = np.full((N, NR), np.nan); SIGb = np.full((N, NR), np.nan)       # baseline
    PRED = np.full((N, K), np.nan); TRUE = np.full((N, K), np.nan)        # PC scores (oof)
    for fold in np.array_split(order, k):
        tr = np.setdiff1d(np.arange(N), fold)
        sh_tr = shape[tr]; mu_sh = sh_tr.mean(0)
        _, _, Vt = np.linalg.svd(sh_tr - mu_sh, full_matrices=False)
        V = Vt[:K]                                                        # (K,23) modes
        S_tr = (shape[tr] - mu_sh) @ V.T
        S_te = (shape[fold] - mu_sh) @ V.T
        Ytr = np.column_stack([logMtot[tr], S_tr])                        # (ntr,K+1)
        mu_te, sig_te, Sig_te = fit_predict(X[tr], Ytr, X[fold])
        PRED[fold] = mu_te[:, 1:]; TRUE[fold] = S_te
        # full reconstruction: CoG(r) = a_r . theta + mu_sh(r), a_r=[1, V[:,r]]
        A = np.column_stack([np.ones(NR), V.T])                           # (23,K+1)
        MU[fold] = mu_te @ A.T + mu_sh
        SIG[fold] = np.sqrt(np.einsum("rk,nkl,rl->nr", A, Sig_te, A))
        # baseline: predict logMtot only, assume mean shape + its full scatter
        MUb[fold] = mu_te[:, 0:1] + mu_sh[None, :]
        SIGb[fold] = np.sqrt(sig_te[:, 0:1] ** 2 + sh_tr.var(0)[None, :])
    return MU, SIG, MUb, SIGb, PRED, TRUE


MU, SIG, MUb, SIGb, PRED, TRUE = cv()
crps_full = crps_gaussian(cog_true, MU, SIG).mean(0)             # per radius (23,)
crps_base = crps_gaussian(cog_true, MUb, SIGb).mean(0)
rms_full = np.sqrt(((MU - cog_true) ** 2).mean(0))
_, cov_full = interval_coverage(cog_true.ravel(), MU.ravel(), SIG.ravel())
r2_pc = 1.0 - ((TRUE - PRED) ** 2).mean(0) / TRUE.var(0)
print(f"\n[profile prediction] mean per-radius CRPS: full {crps_full.mean():.4f}  "
      f"baseline(mass+mean shape) {crps_base.mean():.4f}  "
      f"({100*(crps_base.mean()-crps_full.mean())/crps_base.mean():+.1f}%)")
print(f"  reconstruction RMS (mean over radii) = {rms_full.mean():.4f} dex; "
      f"overall coverage 50/68/90/95 = {'/'.join(f'{c:.2f}' for c in cov_full)}")
print(f"  per-PC out-of-fold R^2 (halo-predictability of each shape mode): "
      + "  ".join(f"PC{k+1}={r2_pc[k]:+.2f}" for k in range(K)))
print(f"  -> total mass is strongly predicted; shape modes only weakly "
      f"(value of predicting shape = {100*(crps_base.mean()-crps_full.mean())/crps_base.mean():+.1f}% per-radius CRPS)")


# %% ---- FIGURE --------------------------------------------------------------
fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(14.5, 4.4))
# A: per-radius CRPS, full vs baseline
axA.plot(rad, crps_base, "-o", color=OKABE_ITO[7], ms=4, label="mass + mean shape")
axA.plot(rad, crps_full, "-s", color=OKABE_ITO[2], ms=4, label="full (halo-predicted shape)")
axA.set_xscale("log"); axA.set_xlabel("R [kpc]"); axA.set_ylabel("per-radius CV CRPS [dex]")
axA.set_title("A. Profile prediction vs radius"); axA.legend(fontsize=8)

# B: example reconstructed profiles (low/mid/high total mass) with 1-sigma band
qs = np.quantile(logMtot, [0.1, 0.5, 0.9])
ex = [int(np.argmin(np.abs(logMtot - q))) for q in qs]
for c, i in zip([OKABE_ITO[0], OKABE_ITO[4], OKABE_ITO[5]], ex):
    axB.plot(rad, cog_true[i], "o", color=c, ms=3)
    axB.plot(rad, MU[i], "-", color=c, lw=1.5,
             label=f"logMtot={logMtot[i]:.1f}")
    axB.fill_between(rad, MU[i] - SIG[i], MU[i] + SIG[i], color=c, alpha=0.18)
axB.set_xscale("log"); axB.set_xlabel("R [kpc]"); axB.set_ylabel(r"$\log M_*(<R)$")
axB.set_title("B. Reconstructed CoG (points=truth, band=1σ)"); axB.legend(fontsize=8)

# C: per-PC predictability + per-radius improvement
axC.axhline(0, color="k", lw=0.8)
axC.plot(rad, 100 * (crps_base - crps_full) / crps_base, "-o", color=OKABE_ITO[2], ms=4)
axC.set_xscale("log"); axC.set_xlabel("R [kpc]")
axC.set_ylabel("CRPS gain from halo-predicted shape [%]")
axC.set_title("C. Where does predicting shape help?")
txt = "shape-mode predictability:\n" + "\n".join(
    f"  PC{k+1} ({['concentration', 'inner/mid', 'outer'][k] if k < 3 else 'mode'}): "
    f"R²={r2_pc[k]:+.2f}" for k in range(K))
axC.text(0.04, 0.04, txt, transform=axC.transAxes, fontsize=7.5, va="bottom",
         bbox=dict(boxstyle="round", fc="white", ec="0.7"))
fig.suptitle(f"exp22 — full-profile emulator (n={N}, K={K}); per-radius CRPS "
             f"{crps_base.mean():.4f}->{crps_full.mean():.4f}, recon RMS "
             f"{rms_full.mean():.3f} dex", fontsize=10)
fig.tight_layout()
save_fig(fig, FIGDIR / "exp22_full_profile")


# %% ---- save outputs --------------------------------------------------------
OUTDIR.mkdir(parents=True, exist_ok=True)
st = Table({"R_kpc": rad, "crps_full": crps_full.tolist(),
            "crps_baseline": crps_base.tolist(), "recon_rms": rms_full.tolist()})
st.write(OUTDIR / "profile_scores.csv", overwrite=True)
write_manifest(OUTDIR, params={
    "n": int(N), "K": K, "n_radii": NR,
    "crps_full_mean": float(crps_full.mean()), "crps_base_mean": float(crps_base.mean()),
    "shape_value_pct": float(100 * (crps_base.mean() - crps_full.mean()) / crps_base.mean()),
    "recon_rms_mean": float(rms_full.mean()),
    "coverage": cov_full.tolist(), "r2_per_pc": r2_pc.tolist()})
print(f"\nwrote figure -> {FIGDIR}\nwrote outputs -> {OUTDIR}")
print(f"\n[verdict] full-profile emulator: per-radius CRPS {crps_full.mean():.4f} "
      f"(vs {crps_base.mean():.4f} for mass+mean-shape); recon RMS {rms_full.mean():.3f} dex; "
      f"PC1 R²={r2_pc[0]:+.2f} (shape is only weakly halo-predictable)")
