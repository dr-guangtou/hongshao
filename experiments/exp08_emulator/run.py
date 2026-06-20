"""exp08 — The first probabilistic Ultimate-SHMR emulator, judged by the exp07 suite.

We model P(stellar profile | M0, assembly history) as a conditional multivariate
Gaussian: a linear mean in [M0, MAH-PCA(4)] plus a *full* residual covariance
(per-fold, so honest out-of-sample). Two target representations, compared head to
head in the SAME physical aperture-mass space:

  A (direct):  predict the 4 aperture/annulus masses (<10, 10-30, 30-50,
               50-100 kpc) directly.
  B (generative): predict the 5 radial-DiffMAH params (rdm_*, fit over R>=5 kpc),
               reconstruct the curve of growth, and read off the same 4 masses.

Both get the same probabilistic treatment and are scored with the exp07 suite
(CRPS, predictive log-score, interval calibration) plus the joint multivariate
log-score with a FULL vs DIAGONAL covariance — the test of whether modeling the
correlated scatter (exp07: residual |corr|~0.57) actually pays off. Controls:
M0-only mean and shuffled-MAH.

Run from the repo root (EXP08_NMAX=400 for a sub-minute pass):
    PYTHONPATH=. uv run python experiments/exp08_emulator/run.py
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
from hongshao.tng_data import (COG_RAD_KPC, COG_FIT_RMIN_KPC, load_mah,        # noqa: E402
                               load_cosmic_time, peak_history)
from hongshao.profiles import cog_from_physical                               # noqa: E402
from hongshao.metrics import (crps_gaussian, gaussian_logscore,               # noqa: E402
                              interval_coverage)
from hongshao.plotting import set_style, save_fig, OKABE_ITO                  # noqa: E402
from hongshao.provenance import write_manifest                               # noqa: E402

set_style()
HERE = Path(__file__).resolve().parent
OUTDIR, FIGDIR = HERE / "outputs", HERE / "figures"
TABLE = ROOT / "data" / "processed" / "tng300_072_z0p4.fits"
R = np.asarray(COG_RAD_KPC, float)
RFIT = R[R >= COG_FIT_RMIN_KPC]                    # reconstruction grid (>=5 kpc)
APER_EDGES = [10.0, 30.0, 50.0, 100.0]            # aperture boundaries (kpc)
TNAMES = ["<10", "10-30", "30-50", "50-100"]
PARAM_COLS = ["rdm_logMstar0", "rdm_beta_in", "rdm_beta_out", "rdm_R_c", "rdm_Delta"]
# physical bounds for predicted params before reconstruction (match profiles.fit_cog)
PLO = np.array([0.0, 0.0, 0.0, 1.0, 0.05])
PHI = np.array([14.0, 9.0, 3.0, 150.0, 3.0])

t = Table.read(TABLE)
t = t[t["use"]]
NMAX = int(os.environ.get("EXP08_NMAX", 0))
if NMAX:
    t = t[:NMAX]
logM0 = np.asarray(t["logm0_halo"], float)
aper = np.asarray(t["logmstar_aper"], float)
params = np.column_stack([np.asarray(t[c], float) for c in PARAM_COLS])
idx = np.asarray(t["index"])


# %% MAH-PCA(4) features (exp06 construction) -------------------------------
mah = load_mah()
t_snap = load_cosmic_time()
tgrid = np.linspace(2.2, 9.0, 18)
mah_shape = np.full((len(t), len(tgrid)), np.nan)
for row, i in enumerate(idx):
    snaps, lmp = peak_history(mah[int(i)])
    if lmp is None:
        continue
    tt = t_snap[snaps.astype(int)]
    if tt[0] <= tgrid[0] and tt[-1] >= tgrid[-1]:
        mah_shape[row] = np.interp(tgrid, tt, lmp) - logM0[row]


def _annulus(a_outer, a_inner):
    return np.log10(np.clip(10.0 ** a_outer - 10.0 ** a_inner, 1.0, None))


# truth in scoring space: the 4 measured aperture/annulus masses
Ytruth = np.column_stack([aper[:, 0], _annulus(aper[:, 1], aper[:, 0]),
                          _annulus(aper[:, 2], aper[:, 1]),
                          _annulus(aper[:, 4], aper[:, 2])])

covered = np.isfinite(mah_shape).all(1)
good = covered & np.isfinite(Ytruth).all(1) & np.isfinite(params).all(1) & np.isfinite(logM0)
mu_m = mah_shape[good].mean(0)
_, _, Vt = np.linalg.svd(mah_shape[good] - mu_m, full_matrices=False)
mah_pca = (mah_shape[good] - mu_m) @ Vt[:4].T          # top-4 MAH-PCA scores

M0 = logM0[good]
Y = Ytruth[good]
P = params[good]
N = len(Y)
print(f"exp08 sample: {N} galaxies (MAH-covered, finite targets & params)")

X_base = M0[:, None]
X_full = np.column_stack([M0, mah_pca])
rng = np.random.default_rng(1)
mah_shuf = mah_pca.copy()
# shuffle MAH-PCA within M0 bins (break the assembly link, keep marginal)
edges = np.quantile(M0, np.linspace(0, 1, 13))
binid = np.digitize(M0, edges[1:-1])
for b in np.unique(binid):
    ii = np.where(binid == b)[0]
    mah_shuf[ii] = mah_pca[ii][rng.permutation(len(ii))]
X_shuf = np.column_stack([M0, mah_shuf])


# %% emulator: out-of-fold mean (in scoring space) + per-fold covariance -----
def reconstruct(param_rows):
    """rdm params (m,5) -> 4 aperture/annulus log-masses, via the CoG model."""
    p = np.clip(param_rows, PLO, PHI)
    cog = cog_from_physical(RFIT, p[:, 0], p[:, 1], p[:, 2], p[:, 3], p[:, 4])
    cog = np.maximum.accumulate(np.atleast_2d(cog), axis=1)      # enforce monotonic
    mass = 10.0 ** cog
    out = np.empty((len(p), 4))
    for r, row in enumerate(mass):
        m10, m30, m50, m100 = np.interp(APER_EDGES, RFIT, row)
        out[r] = [np.log10(m10), np.log10(max(m30 - m10, 1.0)),
                  np.log10(max(m50 - m30, 1.0)), np.log10(max(m100 - m50, 1.0))]
    return out


def cv_emulator(X, regress_target, reconstruct_fn=None, k=5, seed=0):
    """Out-of-fold predictive mean (scoring space) + per-fold residual covariance.

    regress_target: what the linear model fits (Y directly, or the 5 params).
    reconstruct_fn: None for direct (A); for B, maps predicted params -> masses.
    Returns mu (N,4), Sigma per galaxy (N,4,4), all from training-only stats.
    """
    n = len(Y)
    Xd = np.column_stack([np.ones(n), X])
    order = np.random.default_rng(seed).permutation(n)
    mu = np.full((n, 4), np.nan)
    Sig = np.full((n, 4, 4), np.nan)
    for fold in np.array_split(order, k):
        tr = np.setdiff1d(np.arange(n), fold)
        beta, *_ = np.linalg.lstsq(Xd[tr], regress_target[tr], rcond=None)
        pred_tr, pred_te = Xd[tr] @ beta, Xd[fold] @ beta
        if reconstruct_fn is not None:
            pred_tr, pred_te = reconstruct_fn(pred_tr), reconstruct_fn(pred_te)
        mu[fold] = pred_te
        cov = np.cov((Y[tr] - pred_tr).T)              # full 4x4 residual covariance
        Sig[fold] = cov
    return mu, Sig


emulators = {
    "M0 only": cv_emulator(X_base, Y),
    "A: apertures (M0+hist)": cv_emulator(X_full, Y),
    "B: rdm params (M0+hist)": cv_emulator(X_full, P, reconstruct_fn=reconstruct),
    "A shuffled-MAH": cv_emulator(X_shuf, Y),
}


# %% scoring -----------------------------------------------------------------
def marginal_scores(mu, Sig):
    sig = np.sqrt(np.stack([np.diag(s) for s in Sig]))    # (N,4) marginal sigma
    crps = crps_gaussian(Y, mu, sig).mean(0)
    ls = gaussian_logscore(Y, mu, sig).mean(0)
    return crps, ls, sig


def joint_nll(mu, Sig, diagonal=False):
    """Mean negative multivariate-Gaussian log density (lower better)."""
    resid = Y - mu
    out = np.empty(len(Y))
    for i in range(len(Y)):
        S = np.diag(np.diag(Sig[i])) if diagonal else Sig[i]
        sign, logdet = np.linalg.slogdet(S)
        out[i] = 0.5 * (4 * np.log(2 * np.pi) + logdet
                        + resid[i] @ np.linalg.solve(S, resid[i]))
    return float(np.mean(out))


print("\n[exp08] marginal scores (mean over galaxies):")
print(f"  {'model':26s} {'RMS':>7s} {'CRPS':>7s} {'logS':>7s}  per-target CRPS")
results = {}
for name, (mu, Sig) in emulators.items():
    crps, ls, sig = marginal_scores(mu, Sig)
    rms = np.sqrt(np.mean((Y - mu) ** 2))
    results[name] = dict(mu=mu, Sig=Sig, crps=crps, ls=ls, rms=rms)
    print(f"  {name:26s} {rms:7.4f} {crps.mean():7.4f} {ls.mean():7.4f}  "
          + " ".join(f"{c:.3f}" for c in crps))

print("\n[exp08] joint multivariate log-score (NLL, lower better): full vs diagonal cov")
joint = {}
for name in ("A: apertures (M0+hist)", "B: rdm params (M0+hist)"):
    mu, Sig = emulators[name]
    nll_full = joint_nll(mu, Sig, diagonal=False)
    nll_diag = joint_nll(mu, Sig, diagonal=True)
    joint[name] = (nll_full, nll_diag)
    print(f"  {name:26s} full={nll_full:7.3f}  diagonal={nll_diag:7.3f}  "
          f"(full better by {nll_diag - nll_full:.3f} nats)")

# covariance match: predicted (model) vs empirical held-out residual correlation
def _corr(C):
    d = np.sqrt(np.diag(C))
    return C / np.outer(d, d)


emp_corr = _corr(np.cov((Y - emulators["A: apertures (M0+hist)"][0]).T))
mod_corr = _corr(np.nanmean(emulators["A: apertures (M0+hist)"][1], 0))
off = ~np.eye(4, dtype=bool)
print(f"\n[exp08] covariance match (A): mean |off-diag| empirical={np.mean(np.abs(emp_corr[off])):.2f}"
      f"  model={np.mean(np.abs(mod_corr[off])):.2f}")

# calibration of the two emulators (pool targets)
calib = {}
for name in ("A: apertures (M0+hist)", "B: rdm params (M0+hist)"):
    mu, Sig = emulators[name]
    sig = np.sqrt(np.stack([np.diag(s) for s in Sig]))
    calib[name] = interval_coverage(Y, mu, sig)
levels = calib["A: apertures (M0+hist)"][0]
print("\n[exp08] calibration (nominal -> empirical coverage):")
for name in calib:
    print(f"  {name:26s} " + "  ".join(f"{int(100*L)}%:{c:.2f}"
          for L, c in zip(levels, calib[name][1])))


# %% FIGURE 1 — per-target CRPS + calibration --------------------------------
fig1, (axA, axB) = plt.subplots(1, 2, figsize=(9.6, 4.0))
order_names = ["M0 only", "A: apertures (M0+hist)", "B: rdm params (M0+hist)",
               "A shuffled-MAH"]
cols = [OKABE_ITO[7], OKABE_ITO[0], OKABE_ITO[5], "0.7"]
xpos = np.arange(4)
w = 0.2
for k, (name, c) in enumerate(zip(order_names, cols)):
    axA.bar(xpos + (k - 1.5) * w, results[name]["crps"], w, color=c,
            label=name.replace(" (M0+hist)", ""))
axA.set_xticks(xpos); axA.set_xticklabels(TNAMES)
axA.set_xlabel("aperture / annulus [kpc]")
axA.set_ylabel("CRPS [dex]  (lower = better)")
axA.set_title("Predictive skill per aperture")
axA.legend(fontsize=7)
axA.text(-0.13, 1.04, "A", transform=axA.transAxes, fontweight="bold", fontsize=12)

axB.plot([0, 1], [0, 1], ":", color="0.5", lw=1)
for name, c, mk in [("A: apertures (M0+hist)", OKABE_ITO[0], "o"),
                    ("B: rdm params (M0+hist)", OKABE_ITO[5], "s")]:
    axB.plot(levels, calib[name][1], "-", marker=mk, color=c, ms=5,
             label=name.replace(" (M0+hist)", ""))
axB.set_xlabel("nominal central interval")
axB.set_ylabel("empirical coverage")
axB.set_title("Calibration")
axB.set_xlim(0.4, 1.0); axB.set_ylim(0.4, 1.0)
axB.legend(fontsize=8)
axB.text(-0.13, 1.04, "B", transform=axB.transAxes, fontweight="bold", fontsize=12)
fig1.suptitle(f"exp08 — probabilistic emulator: aperture (A) vs profile-param (B) "
              f"(n={N})", fontsize=11)
fig1.tight_layout()
save_fig(fig1, FIGDIR / "exp08_skill_calibration")

# %% FIGURE 2 — correlated scatter pays off (joint NLL) + covariance match ----
fig2, (axC, axD) = plt.subplots(1, 2, figsize=(9.6, 4.0))
names2 = list(joint)
xj = np.arange(len(names2))
axC.bar(xj - w, [joint[n][1] for n in names2], 2 * w, color="0.7",
        label="diagonal covariance")
axC.bar(xj + w, [joint[n][0] for n in names2], 2 * w, color=OKABE_ITO[2],
        label="full covariance")
axC.set_xticks(xj); axC.set_xticklabels([n.split(":")[0] for n in names2])
axC.set_ylabel("joint log-score (NLL, lower = better)")
axC.set_title("Modeling correlated scatter helps")
axC.legend(fontsize=8)
axC.text(-0.13, 1.04, "C", transform=axC.transAxes, fontweight="bold", fontsize=12)

im = axD.imshow(emp_corr, cmap="RdBu_r", vmin=-1, vmax=1)
axD.set_xticks(range(4)); axD.set_xticklabels(TNAMES, rotation=30, ha="right", fontsize=7)
axD.set_yticks(range(4)); axD.set_yticklabels(TNAMES, fontsize=7)
for ii in range(4):
    for jj in range(4):
        axD.text(jj, ii, f"{emp_corr[ii, jj]:+.2f}", ha="center", va="center",
                 fontsize=7, color="white" if abs(emp_corr[ii, jj]) > 0.5 else "k")
axD.set_title(f"Residual covariance the emulator draws\n"
              f"(mean |off-diag|={np.mean(np.abs(emp_corr[off])):.2f})")
fig2.colorbar(im, ax=axD, shrink=0.8, label="correlation")
axD.text(-0.18, 1.04, "D", transform=axD.transAxes, fontweight="bold", fontsize=12)
fig2.suptitle("exp08 — the emulator must draw correlated scatter", fontsize=11)
fig2.tight_layout()
save_fig(fig2, FIGDIR / "exp08_covariance")

# %% FIGURE 3 — probabilistic painting (recommended emulator A): same M0,
# different assembly history, labelled by formation redshift z50.
fig3, axE = plt.subplots(figsize=(5.6, 4.2))
sel = np.where((M0 > 13.4) & (M0 < 13.6))[0]
mu_paint, Sig_paint = emulators["A: apertures (M0+hist)"]
z50 = np.asarray(t["z50"], float)[good]
order_sel = sel[np.argsort(z50[sel])]                # ascending z50 (late -> early)
picks = [order_sel[-2], order_sel[len(order_sel) // 2], order_sel[1]]
xpos2 = np.arange(4)
for k, (pk, c, lab) in enumerate(zip(
        picks, [OKABE_ITO[4], OKABE_ITO[7], OKABE_ITO[5]],
        ["early former", "median", "late former"])):
    sig = np.sqrt(np.diag(Sig_paint[pk]))
    axE.errorbar(xpos2 + 0.04 * (k - 1), mu_paint[pk], yerr=sig,
                 fmt="o-", color=c, ms=5, capsize=3, lw=1.4,
                 label=f"{lab} ($z_{{50}}$={z50[pk]:.2f})")
    axE.plot(xpos2, Y[pk], "x", color=c, ms=8, mew=2)
axE.plot([], [], "kx", ms=8, mew=2, label="truth")
axE.set_xticks(xpos2); axE.set_xticklabels(TNAMES)
axE.set_xlabel("aperture / annulus [kpc]")
axE.set_ylabel(r"$\log_{10} M_*$ [$M_\odot$]")
axE.set_title(r"Probabilistic painting (same $M_0$, $\pm 1\sigma$)")
axE.legend(fontsize=7)
fig3.tight_layout()
save_fig(fig3, FIGDIR / "exp08_painting")

# %% save outputs ------------------------------------------------------------
OUTDIR.mkdir(parents=True, exist_ok=True)
summary = Table({
    "model": list(results),
    "rms_dex": [results[n]["rms"] for n in results],
    "crps_dex": [float(results[n]["crps"].mean()) for n in results],
    "logscore": [float(results[n]["ls"].mean()) for n in results]})
summary.write(OUTDIR / "emulator_scores.csv", overwrite=True)
write_manifest(OUTDIR, params={
    "n": int(N), "targets": TNAMES, "features": "M0 + MAH-PCA(4)",
    "crps_A": float(results["A: apertures (M0+hist)"]["crps"].mean()),
    "crps_B": float(results["B: rdm params (M0+hist)"]["crps"].mean()),
    "crps_M0": float(results["M0 only"]["crps"].mean()),
    "joint_nll_full_A": joint["A: apertures (M0+hist)"][0],
    "joint_nll_diag_A": joint["A: apertures (M0+hist)"][1]})
print(f"\nwrote figures -> {FIGDIR}\nwrote outputs -> {OUTDIR}")
