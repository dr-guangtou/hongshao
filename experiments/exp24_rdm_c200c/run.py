"""exp24 — Does c_200c predict the radial-DiffMAH (RDM) shape parameters?

exp03/04/05 built the 5-parameter RDM profile model (logMstar0, beta_in,
beta_out, R_c, Delta) and found that the MAH predicts the *normalization* well
but the *shape* parameters (inner/outer slope, transition radius) barely at all —
the halo->shape-parameter connection was weak. BUT those experiments predate
`c_200c` (added exp16-19), and exp22 showed concentration specifically lights up
the profile's concentration mode (PC1 R^2 0.39->0.54). RDM's beta_in / R_c ARE
that concentration direction. So the old "weak connection" verdict is stale on
the feature side. This experiment re-tests it.

Two parts:
  (A) Per-parameter predictability: CV R^2 of each RDM parameter from M0 /
      DiffMAH(4) / DiffMAH(4)+c_200c, plus the partial correlation of c_200c with
      each parameter at fixed MAH (residual-on-residual; the exp16 test). Does
      c_200c light up beta_in / R_c where the MAH could not?
  (B) Generative reconstruction: predict the joint 5-vector with the (now
      N-target) heteroscedastic emulator, sample, reconstruct the CoG via
      `profiles.cog_from_physical`, and compare per-radius CRPS / recon RMS to the
      exp22 PCA route. (Accuracy is expected to tie — same SHMR floor; the value
      is the interpretable physical statement.)

Reuses the graduated library (emulator, profiles). Run:
  EXP24_NMAX=700 PYTHONPATH=. uv run python experiments/exp24_rdm_c200c/run.py
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hongshao.emulator import fit                                                 # noqa: E402
from hongshao.profiles import cog_from_physical                                   # noqa: E402
from hongshao.tng_data import COG_RAD_KPC, COG_FIT_RMIN_KPC                        # noqa: E402
from hongshao.metrics import crps_gaussian                                        # noqa: E402
from hongshao.plotting import set_style, save_fig, OKABE_ITO                       # noqa: E402
from hongshao.provenance import write_manifest                                    # noqa: E402

set_style()
HERE = Path(__file__).resolve().parent
OUTDIR, FIGDIR = HERE / "outputs", HERE / "figures"
TABLE = ROOT / "data" / "processed" / "tng300_072_z0p4.fits"
NMAX = int(os.environ.get("EXP24_NMAX", 0))
PARAMS = ["logMstar0", "beta_in", "beta_out", "logR_c", "Delta"]
SHAPE_PARAMS = ["beta_in", "beta_out", "logR_c"]            # the pure-shape params (exp05: ~0 from MAH)

t = Table.read(TABLE); t = t[t["use"]]
if NMAX:
    t = t[:NMAX]
cog = np.asarray(t["logmstar_cog"], float)
M0 = np.asarray(t["logmh_z0p4"], float)
dmah = np.column_stack([np.asarray(t[c], float) for c in
                        ("dmah_logmp", "dmah_logtc", "dmah_early", "dmah_late")])
c200 = np.asarray(t["c_200c"], float)
# RDM params as the target vector (R_c -> log10 R_c so the regression is well-scaled)
rdm = np.column_stack([np.asarray(t["rdm_logMstar0"], float),
                       np.asarray(t["rdm_beta_in"], float),
                       np.asarray(t["rdm_beta_out"], float),
                       np.log10(np.asarray(t["rdm_R_c"], float)),
                       np.asarray(t["rdm_Delta"], float)])
rms = np.asarray(t["rdm_rms"], float)
g = (np.isfinite(cog).all(1) & np.isfinite(M0) & np.isfinite(dmah).all(1)
     & np.isfinite(c200) & np.isfinite(rdm).all(1) & (rms < 0.05))
cog, rdm, M0, dmah, c200 = cog[g], rdm[g], M0[g], dmah[g], c200[g]
N = len(rdm)
X_m0 = M0[:, None]
X_dm = dmah
X_dmc = np.column_stack([dmah, c200])
print(f"exp24: RDM params from DiffMAH+c_200c on n={N}")


# %% ---- (A) per-parameter predictability ------------------------------------
def cv_predict(Xf, Y, k=5, seed=0):
    """Out-of-fold linear prediction of each column of Y from features Xf."""
    n, nb = Y.shape
    A = np.column_stack([np.ones(n), Xf])
    order = np.random.default_rng(seed).permutation(n)
    pred = np.full((n, nb), np.nan)
    for fold in np.array_split(order, k):
        tr = np.setdiff1d(np.arange(n), fold)
        for j in range(nb):
            beta, *_ = np.linalg.lstsq(A[tr], Y[tr, j], rcond=None)
            pred[fold, j] = A[fold] @ beta
    return pred


def r2_cols(Y, pred):
    return 1.0 - ((Y - pred) ** 2).mean(0) / Y.var(0)


r2 = {name: r2_cols(rdm, cv_predict(Xf, rdm))
      for name, Xf in [("M0", X_m0), ("DiffMAH", X_dm), ("DiffMAH+c200c", X_dmc)]}

print("\n[A] per-parameter CV R^2 (variance explained):")
print(f"  {'param':10s} {'M0':>8s} {'DiffMAH':>8s} {'+c200c':>8s}  {'c200c gain':>10s}")
for j, p in enumerate(PARAMS):
    gain = r2["DiffMAH+c200c"][j] - r2["DiffMAH"][j]
    print(f"  {p:10s} {r2['M0'][j]:+8.3f} {r2['DiffMAH'][j]:+8.3f} "
          f"{r2['DiffMAH+c200c'][j]:+8.3f}  {gain:+10.3f}")


# partial correlation of c_200c with each RDM param at fixed MAH (residual-on-residual)
def resid_on(Z, y):
    A = np.column_stack([np.ones(len(y)), Z])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ beta


c200_res = resid_on(X_dm, c200)                       # c_200c with the MAH removed
print("\n[A] partial corr of c_200c with each RDM param at fixed DiffMAH(4):")
pcorr = np.zeros(5)
for j, p in enumerate(PARAMS):
    pr = resid_on(X_dm, rdm[:, j])
    pcorr[j] = np.corrcoef(c200_res, pr)[0, 1]
    print(f"  {p:10s} partial r(c200c, param | MAH) = {pcorr[j]:+.3f}")
shape_idx = [PARAMS.index(p) for p in SHAPE_PARAMS]
print(f"  -> shape params {SHAPE_PARAMS}: MAH R^2 ~"
      f"{np.mean([r2['DiffMAH'][i] for i in shape_idx]):.2f} -> "
      f"+c200c ~{np.mean([r2['DiffMAH+c200c'][i] for i in shape_idx]):.2f}; "
      f"c200c partial r up to {np.max(np.abs(pcorr[shape_idx])):.2f}")


# %% ---- (B) generative reconstruction: predict 5-vec, rebuild the CoG --------
RAD = COG_RAD_KPC
fitmask = RAD >= COG_FIT_RMIN_KPC                     # RDM was fit on R >= 5 kpc; judge there
Rfit = RAD[fitmask]
cog_true = cog[:, fitmask]


def reconstruct(theta):
    """(M, 5) transformed params -> (M, len(Rfit)) log CoG (inverse-transform R_c)."""
    return cog_from_physical(Rfit, theta[:, 0], theta[:, 1], theta[:, 2],
                             10.0 ** theta[:, 3], theta[:, 4])


def cv_reconstruct(Xf, k=5, seed=0, n_draw=200):
    """OOF mean CoG + per-radius sigma by sampling the predictive 5-vector."""
    order = np.random.default_rng(seed).permutation(N)
    MU = np.full((N, len(Rfit)), np.nan); SIG = np.full((N, len(Rfit)), np.nan)
    rng = np.random.default_rng(seed + 1)
    for fold in np.array_split(order, k):
        tr = np.setdiff1d(np.arange(N), fold)
        emu = fit(Xf[tr], rdm[tr])
        mu, _, cov = emu.predict(Xf[fold])
        MU[fold] = reconstruct(mu)                                       # mean-param profile
        L = np.linalg.cholesky(cov + 1e-9 * np.eye(5))           # (nf, 5, 5)
        z = rng.standard_normal((n_draw, len(fold), 5))
        draws = mu[None] + np.einsum("nij,snj->sni", L, z)       # mu + L z, (S, nf, 5)
        recon = np.stack([reconstruct(draws[s]) for s in range(n_draw)])  # (S, nf, R)
        SIG[fold] = recon.std(0)
    return MU, SIG


MU_r, SIG_r = cv_reconstruct(X_dmc)
crps_rdm = crps_gaussian(cog_true, MU_r, SIG_r).mean(0)
recon_rms = np.sqrt(((MU_r - cog_true) ** 2).mean(0))
print("\n[B] generative reconstruction (DiffMAH+c200c -> 5 params -> CoG, R>=5 kpc):")
print(f"  mean per-radius CRPS = {crps_rdm.mean():.4f}   recon RMS = {recon_rms.mean():.4f} dex")
print("  (exp22 PCA route, full radius: CRPS ~0.064, recon RMS ~0.118; same SHMR floor expected)")


# %% ---- FIGURE --------------------------------------------------------------
fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(14.5, 4.4))
x = np.arange(5); w = 0.27
for i, (nm, c) in enumerate([("M0", OKABE_ITO[7]), ("DiffMAH", OKABE_ITO[0]),
                             ("DiffMAH+c200c", OKABE_ITO[2])]):
    axA.bar(x + (i - 1) * w, np.clip(r2[nm], 0, None), w, color=c, label=nm)
axA.set_xticks(x); axA.set_xticklabels(PARAMS, rotation=30, fontsize=7, ha="right")
axA.set_ylabel("CV R² (variance explained)"); axA.set_title("A. Per-parameter predictability")
axA.legend(fontsize=7)

axB.axhline(0, color="k", lw=0.8)
gain = r2["DiffMAH+c200c"] - r2["DiffMAH"]
colors = [OKABE_ITO[2] if p in SHAPE_PARAMS else OKABE_ITO[7] for p in PARAMS]
axB.bar(x, gain, 0.6, color=colors)
axB.set_xticks(x); axB.set_xticklabels(PARAMS, rotation=30, fontsize=7, ha="right")
axB.set_ylabel("R² gain from c_200c"); axB.set_title("B. Where does c_200c help? (green=shape)")

axC.plot(Rfit, crps_rdm, "-o", color=OKABE_ITO[2], ms=3, label=f"RDM route ({crps_rdm.mean():.4f})")
axC.axhline(0.064, ls=":", color=OKABE_ITO[7], lw=1.2, label="exp22 PCA (~0.064)")
axC.set_xscale("log"); axC.set_xlabel("R [kpc]"); axC.set_ylabel("per-radius CV CRPS [dex]")
axC.set_title("C. Reconstructed-CoG skill"); axC.legend(fontsize=8)
fig.suptitle(f"exp24 — RDM params from DiffMAH+c_200c (n={N}); c_200c partial r(beta_in)="
             f"{pcorr[PARAMS.index('beta_in')]:+.2f}, r(logR_c)={pcorr[PARAMS.index('logR_c')]:+.2f}",
             fontsize=10)
fig.tight_layout()
save_fig(fig, FIGDIR / "exp24_rdm_c200c")


# %% ---- save outputs --------------------------------------------------------
OUTDIR.mkdir(parents=True, exist_ok=True)
st = Table({"param": PARAMS, "r2_M0": r2["M0"].tolist(), "r2_DiffMAH": r2["DiffMAH"].tolist(),
            "r2_DiffMAH_c200c": r2["DiffMAH+c200c"].tolist(),
            "c200c_partial_r_fixedMAH": pcorr.tolist()})
st.write(OUTDIR / "rdm_param_predictability.csv", overwrite=True)
write_manifest(OUTDIR, params={
    "n": int(N), "params": PARAMS,
    "r2_M0": r2["M0"].tolist(), "r2_DiffMAH": r2["DiffMAH"].tolist(),
    "r2_DiffMAH_c200c": r2["DiffMAH+c200c"].tolist(),
    "c200c_partial_r_fixedMAH": pcorr.tolist(),
    "recon_crps_mean": float(crps_rdm.mean()), "recon_rms_mean": float(recon_rms.mean())})
print(f"\nwrote figure -> {FIGDIR}\nwrote outputs -> {OUTDIR}")
print(f"\n[verdict] c_200c lifts beta_in R^2 {r2['DiffMAH'][1]:.2f}->{r2['DiffMAH+c200c'][1]:.2f}, "
      f"logR_c {r2['DiffMAH'][3]:.2f}->{r2['DiffMAH+c200c'][3]:.2f} "
      f"(partial r {pcorr[1]:+.2f}/{pcorr[3]:+.2f}); reconstruction CRPS {crps_rdm.mean():.4f}")
