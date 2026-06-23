"""exp22b — PCA the 1-D stellar-mass *density* profile, not the cumulative CoG.

run.py compresses and predicts the curve of growth M(<R) (cumulative). The CoG
is an integral, so it is smooth and its shape modes are gentle. The 1-D density
profile Sigma(R) = dM/dA keeps more local structure (core, envelope transition),
and in a noiseless simulation it is reliable even in the outskirts. Question
(user): does PCA-ing the *density* profile give shape modes that are more (or
differently) halo-predictable than the CoG modes?

We derive Sigma(R) by differencing the stored CoG (M(<R) on COG_RAD_KPC), build
its shape (total mass divided out), and run the *same* predict-the-scores
machinery as run.py — for BOTH the CoG shape and the density shape on the SAME
galaxies, so the comparison is apples-to-apples:
  - value of predicting shape  = per-radius CRPS gain over a mass+mean-shape baseline
  - per-mode halo-predictability = out-of-fold R^2 of each PC from DiffMAH+c_200c

Library untouched. Run:
  EXP22_NMAX=700 PYTHONPATH=. uv run python experiments/exp22_full_profile_predict/density_profile.py
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
K = int(os.environ.get("EXP22_K", 3))

t = Table.read(TABLE); t = t[t["use"]]
if NMAX:
    t = t[:NMAX]
cog = np.asarray(t["logmstar_cog"], float)
dmah = np.column_stack([np.asarray(t[c], float) for c in
                        ("dmah_logmp", "dmah_logtc", "dmah_early", "dmah_late")])
c200 = np.asarray(t["c_200c"], float)
R = COG_RAD_KPC
logMtot = cog[:, -1]

# --- density profile by differencing the CoG: Sigma_i = dM_i / dA_i -----------
Mcum = 10.0 ** cog
dM = Mcum[:, 1:] - Mcum[:, :-1]                         # (N,23) shell mass
dA = np.pi * (R[1:] ** 2 - R[:-1] ** 2)                # (23,) shell area
Rmid = np.sqrt(R[:-1] * R[1:])                          # (23,) geometric midpoints
logSig = np.log10(np.clip(dM, 1.0, None) / dA[None, :])  # (N,23) log surface density

# the two shape representations (total mass divided out), on the same galaxies
cog_shape = (cog - logMtot[:, None])[:, :-1]            # (N,23) CoG shape
den_shape = logSig - logMtot[:, None]                  # (N,23) density shape
g = (np.isfinite(cog).all(1) & np.isfinite(dmah).all(1) & np.isfinite(c200)
     & np.all(dM > 0, axis=1))
cog_shape, den_shape, logMtot = cog_shape[g], den_shape[g], logMtot[g]
X = np.column_stack([dmah[g], c200[g]])
N = len(X)
print(f"exp22b: density-vs-CoG profile PCA on n={N}, K={K} modes")


# %% ---- machinery (same as run.py) ------------------------------------------
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
    Am_tr = np.column_stack([np.ones(len(Xtr)), Xtr])
    Am_te = np.column_stack([np.ones(len(Xte)), Xte])
    mu_te = np.empty((len(Xte), nb)); r_tr = np.empty((len(Xtr), nb))
    for j in range(nb):
        beta, *_ = np.linalg.lstsq(Am_tr, Ytr[:, j], rcond=None)
        mu_te[:, j] = Am_te @ beta; r_tr[:, j] = Ytr[:, j] - Am_tr @ beta
    mx, sx = Xtr.mean(0), Xtr.std(0)
    Ztr = (Xtr - mx) / sx; Zte = (Xte - mx) / sx
    sig_te = np.empty((len(Xte), nb)); sig_tr = np.empty((len(Xtr), nb))
    for j in range(nb):
        gm = fit_logvar(r_tr[:, j], Ztr)
        sig_tr[:, j] = np.exp(0.5 * (np.column_stack([np.ones(len(Xtr)), Ztr]) @ gm))
        sig_te[:, j] = np.exp(0.5 * (np.column_stack([np.ones(len(Xte)), Zte]) @ gm))
    Rc = np.corrcoef((r_tr / sig_tr).T)
    Sig_te = sig_te[:, :, None] * Rc[None] * sig_te[:, None, :]
    return mu_te, sig_te, Sig_te


def run_target(shp, anchor, k=5, seed=0):
    """Predict [anchor, K shape-PCs] from X; reconstruct the per-radius quantity
    (truth = anchor + shp). Returns per-radius CRPS (full + mass+mean-shape
    baseline), per-mode oof R^2, recon RMS, coverage, and the mean modes."""
    nr = shp.shape[1]
    truth = anchor[:, None] + shp
    order = np.random.default_rng(seed).permutation(N)
    MU = np.full((N, nr), np.nan); SIG = np.full((N, nr), np.nan)
    MUb = np.full((N, nr), np.nan); SIGb = np.full((N, nr), np.nan)
    PRED = np.full((N, K), np.nan); TRUE = np.full((N, K), np.nan)
    modes = np.zeros((K, nr)); mean_shape = np.zeros(nr)
    for fold in np.array_split(order, k):
        tr = np.setdiff1d(np.arange(N), fold)
        mu_sh = shp[tr].mean(0)
        _, _, Vt = np.linalg.svd(shp[tr] - mu_sh, full_matrices=False)
        V = Vt[:K]
        modes += V / k; mean_shape += mu_sh / k
        S_tr = (shp[tr] - mu_sh) @ V.T; S_te = (shp[fold] - mu_sh) @ V.T
        Ytr = np.column_stack([anchor[tr], S_tr])
        mu_te, sig_te, Sig_te = fit_predict(X[tr], Ytr, X[fold])
        PRED[fold] = mu_te[:, 1:]; TRUE[fold] = S_te
        A = np.column_stack([np.ones(nr), V.T])
        MU[fold] = mu_te @ A.T + mu_sh
        SIG[fold] = np.sqrt(np.einsum("rk,nkl,rl->nr", A, Sig_te, A))
        MUb[fold] = mu_te[:, 0:1] + mu_sh[None, :]
        SIGb[fold] = np.sqrt(sig_te[:, 0:1] ** 2 + shp[tr].var(0)[None, :])
    crps_full = crps_gaussian(truth, MU, SIG).mean(0)
    crps_base = crps_gaussian(truth, MUb, SIGb).mean(0)
    _, cov = interval_coverage(truth.ravel(), MU.ravel(), SIG.ravel())
    r2 = 1.0 - ((TRUE - PRED) ** 2).mean(0) / TRUE.var(0)
    rms = float(np.sqrt(((MU - truth) ** 2).mean()))
    return dict(crps_full=crps_full, crps_base=crps_base, r2=r2, rms=rms, cov=cov,
                modes=modes, mean_shape=mean_shape,
                value=100 * (crps_base.mean() - crps_full.mean()) / crps_base.mean())


COGR = run_target(cog_shape, logMtot)
DEN = run_target(den_shape, logMtot)
for lab, r, unit in [("CoG  M(<R)", COGR, "cumulative"), ("density Sigma(R)", DEN, "density")]:
    print(f"\n[{lab}] ({unit}) per-radius CRPS full {r['crps_full'].mean():.4f} vs "
          f"baseline {r['crps_base'].mean():.4f}  -> value of predicting shape "
          f"{r['value']:+.1f}%; recon RMS {r['rms']:.3f}; "
          f"coverage {'/'.join(f'{c:.2f}' for c in r['cov'])}")
    print("   per-mode oof R^2: " + "  ".join(f"PC{k+1}={r['r2'][k]:+.2f}" for k in range(K)))
print(f"\n[compare] value of predicting shape: CoG {COGR['value']:+.1f}%  "
      f"vs density {DEN['value']:+.1f}%   |   PC1 R^2: CoG {COGR['r2'][0]:+.2f}  "
      f"vs density {DEN['r2'][0]:+.2f}")


# %% ---- FIGURE --------------------------------------------------------------
fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(14.5, 4.4))
# A: the density shape modes (what the density PCA captures)
axA.axhline(0, color="0.7", lw=0.8)
for k in range(K):
    axA.plot(Rmid, DEN["modes"][k], "-o", ms=3, color=OKABE_ITO[[2, 5, 0][k]],
             label=f"density PC{k+1} (R²={DEN['r2'][k]:+.2f})")
axA.set_xscale("log"); axA.set_xlabel("R [kpc]"); axA.set_ylabel("mode amplitude")
axA.set_title("A. Density-profile shape modes"); axA.legend(fontsize=7)

# B: value of predicting shape per radius, density vs CoG
axB.axhline(0, color="k", lw=0.8)
axB.plot(Rmid, 100 * (DEN["crps_base"] - DEN["crps_full"]) / DEN["crps_base"],
         "-o", ms=3, color=OKABE_ITO[2], label=f"density ({DEN['value']:+.1f}%)")
axB.plot(COG_RAD_KPC[:-1], 100 * (COGR["crps_base"] - COGR["crps_full"]) / COGR["crps_base"],
         "-s", ms=3, color=OKABE_ITO[7], label=f"CoG ({COGR['value']:+.1f}%)")
axB.set_xscale("log"); axB.set_xlabel("R [kpc]")
axB.set_ylabel("CRPS gain from halo-predicted shape [%]")
axB.set_title("B. Where does predicting shape help?"); axB.legend(fontsize=8)

# C: per-mode predictability, density vs CoG
xx = np.arange(K); w = 0.38
axC.bar(xx - w / 2, COGR["r2"], w, color=OKABE_ITO[7], label="CoG modes")
axC.bar(xx + w / 2, DEN["r2"], w, color=OKABE_ITO[2], label="density modes")
axC.set_xticks(xx); axC.set_xticklabels([f"PC{k+1}" for k in range(K)])
axC.set_ylabel("out-of-fold R² (halo-predictability)")
axC.set_title("C. Are density modes more predictable?"); axC.legend(fontsize=8)
fig.suptitle(f"exp22b — density-profile PCA vs CoG PCA (n={N}); value of predicting shape "
             f"CoG {COGR['value']:+.1f}% vs density {DEN['value']:+.1f}%", fontsize=10)
fig.tight_layout()
save_fig(fig, FIGDIR / "exp22_density_vs_cog")


# %% ---- save outputs --------------------------------------------------------
OUTDIR.mkdir(parents=True, exist_ok=True)
write_manifest(OUTDIR, params={
    "n": int(N), "K": K,
    "cog_value_pct": float(COGR["value"]), "density_value_pct": float(DEN["value"]),
    "cog_r2_pc": COGR["r2"].tolist(), "density_r2_pc": DEN["r2"].tolist(),
    "cog_recon_rms": COGR["rms"], "density_recon_rms": DEN["rms"],
    "cog_coverage": COGR["cov"].tolist(), "density_coverage": DEN["cov"].tolist()})
print(f"\nwrote figure -> {FIGDIR}/exp22_density_vs_cog\n[verdict] density modes vs CoG: "
      f"PC1 R² {DEN['r2'][0]:+.2f} vs {COGR['r2'][0]:+.2f}, value {DEN['value']:+.1f}% vs "
      f"{COGR['value']:+.1f}% — "
      f"{'density adds structure' if DEN['value'] > COGR['value'] + 1 else 'similar'}")
