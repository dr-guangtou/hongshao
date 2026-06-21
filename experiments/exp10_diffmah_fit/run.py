"""exp10 — Fit DiffMAH to our own MAH curves (portable halo features).

The emulator's MAH features so far (MAH-PCA(4)) are defined on the TNG sample, so
they don't transfer to other simulations / observations. DiffMAH (Hearin et al.
2021) parameterizes each halo's mass accretion history with four *intrinsic*
numbers (logmp, logtc, early, late; transition speed k fixed at 3.5). We fit it
ourselves to every main-branch MAH (we have the curves; the released DiffMAH fits
are not cross-matched to our galaxies), visualize the fit quality the way we did
for the curves of growth, and check that the DiffMAH params predict the stellar
profile as well as MAH-PCA(4).

Run from the repo root (EXP10_NMAX=400 for a sub-minute pass):
    PYTHONPATH=. uv run python experiments/exp10_diffmah_fit/run.py
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
from hongshao.tng_data import load_mah, load_cosmic_time, peak_history          # noqa: E402
from hongshao.diffmah import fit_mah, log_mah, MAH_K                            # noqa: E402
from hongshao.metrics import crps_gaussian                                      # noqa: E402
from hongshao.plotting import set_style, save_fig, OKABE_ITO                    # noqa: E402
from hongshao.provenance import write_manifest                                  # noqa: E402

set_style()
HERE = Path(__file__).resolve().parent
OUTDIR, FIGDIR = HERE / "outputs", HERE / "figures"
TABLE = ROOT / "data" / "processed" / "tng300_072_z0p4.fits"

t = Table.read(TABLE)
t = t[t["use"]]
NMAX = int(os.environ.get("EXP10_NMAX", 0))
if NMAX:
    t = t[:NMAX]
logM0 = np.asarray(t["logm0_halo"], float)
aper = np.asarray(t["logmstar_aper"], float)
idx = np.asarray(t["index"])
t_snap = load_cosmic_time()
T0 = float(t_snap[72])                          # cosmic time at z=0.4 (anchor), Gyr
T_MIN = 2.0                                      # fit the resolved history (t>=2 Gyr,
#                                                 z<~3.2); earlier MAH is unresolved/noisy
#                                                 (matches the exp06 MAH-PCA grid start)
mah = load_mah()
N = len(t)
print(f"exp10: fitting DiffMAH (k={MAH_K}) to {N} MAHs; anchor t0={T0:.2f} Gyr (z=0.4), "
      f"fit t>={T_MIN} Gyr")

# %% fit DiffMAH to every MAH
keys = ["logmp", "logtc", "early", "late", "rms"]
# fit at two inner-time cuts: 1 Gyr (the DiffMAH-paper value) and 2 Gyr (ours);
# both saved for completeness. The figures/feature-check use the 2 Gyr default.
fits_by_tmin = {1.0: {k: np.full(N, np.nan) for k in keys},
                2.0: {k: np.full(N, np.nan) for k in keys}}
mah_t = [None] * N                               # cache (t, log_mpeak) for plotting
for i in range(N):
    sn, lmp = peak_history(mah[int(idx[i])])
    if lmp is None or len(sn) < 6:
        continue
    tt = t_snap[sn.astype(int)]
    mah_t[i] = (tt, lmp)
    for tmin, store in fits_by_tmin.items():
        f = fit_mah(tt, lmp, T0, t_min=tmin)
        for k in keys:
            store[k][i] = f[k]
fits = fits_by_tmin[T_MIN]                        # default fit set for figures
rms = fits["rms"]
print(f"DiffMAH fit RMS [dex]: median={np.nanmedian(rms):.4f}  "
      f"90th={np.nanpercentile(rms, 90):.4f}  frac<0.05={(rms < 0.05).mean():.2f}")
print("param ranges (16-50-84%):")
for k in ["logmp", "logtc", "early", "late"]:
    v = fits[k][np.isfinite(fits[k])]
    print(f"  {k:7s} [{np.percentile(v,16):6.2f} {np.percentile(v,50):6.2f} "
          f"{np.percentile(v,84):6.2f}]")


def model_curve(i, tgrid):
    p = [fits[k][i] for k in ("logmp", "logtc", "early", "late")]
    return log_mah(np.log10(tgrid), *p, logt0=np.log10(T0))


# %% FIGURE 1 — example fits (best/median/worst) + RMS histogram
ok = np.isfinite(rms)
order = np.where(ok)[0][np.argsort(rms[ok])]
fig1, (axA, axB) = plt.subplots(1, 2, figsize=(9.4, 4.0))
examples = [("best", order[0]), ("median", order[len(order) // 2]),
            ("worst", order[-1])]
for (lab, i), col in zip(examples, [OKABE_ITO[2], OKABE_ITO[4], OKABE_ITO[5]]):
    tt, lmp = mah_t[i]
    m = tt >= T_MIN
    tgrid = np.geomspace(T_MIN, T0, 60)
    axA.plot(tt[m], lmp[m], "o", color=col, ms=3)
    axA.plot(tgrid, model_curve(i, tgrid), "-", color=col, lw=1.5,
             label=f"{lab} (rms={rms[i]:.3f})")
axA.set_xlabel("cosmic time [Gyr]")
axA.set_ylabel(r"$\log_{10} M_{\rm peak}(t)\,[M_\odot]$")
axA.set_title("DiffMAH fits to the assembly history")
axA.legend()
axA.text(-0.13, 1.04, "A", transform=axA.transAxes, fontweight="bold", fontsize=12)

axB.hist(rms[ok], bins=np.linspace(0, np.nanpercentile(rms, 98), 40),
         color=OKABE_ITO[7], alpha=0.8)
axB.axvline(np.nanmedian(rms), color=OKABE_ITO[5], lw=1.5,
            label=f"median {np.nanmedian(rms):.3f}")
axB.set_xlabel("DiffMAH fit RMS [dex]")
axB.set_ylabel("halos")
axB.set_title(f"Fit quality (n={int(ok.sum())})")
axB.legend()
axB.text(-0.13, 1.04, "B", transform=axB.transAxes, fontweight="bold", fontsize=12)
fig1.suptitle("exp10 — DiffMAH fits to TNG300 main-branch histories", fontsize=11)
fig1.tight_layout()
save_fig(fig1, FIGDIR / "exp10_fit_quality")

# %% FIGURE 2 — individual MAHs + fits (top) and residuals (bottom), 3 mass bins
binnable = ok & np.isfinite(logM0)
members_sorted = np.where(binnable)[0][np.argsort(logM0[np.where(binnable)[0]])]
thirds = np.array_split(members_sorted, 3)
tgrid = np.geomspace(T_MIN, T0, 60)
rng = np.random.default_rng(0)
N_SHOW = 25
fig2, ax2 = plt.subplots(2, 3, figsize=(13.5, 7.0), sharex=True,
                         gridspec_kw={"height_ratios": [3, 2]})
bin_cols = [OKABE_ITO[4], OKABE_ITO[0], OKABE_ITO[5]]
for j, members in enumerate(thirds):
    axc, axr = ax2[0, j], ax2[1, j]
    lo, hi = logM0[members].min(), logM0[members].max()
    show = members if len(members) <= N_SHOW else rng.choice(members, N_SHOW, replace=False)
    res_stack = []
    for i in members:
        tt, lmp = mah_t[i]
        res_stack.append(np.interp(tgrid, tt, lmp) - model_curve(i, tgrid))
    res_stack = np.array(res_stack)
    for i in show:
        tt, lmp = mah_t[i]
        axc.plot(tt, lmp, color="0.6", lw=0.5, alpha=0.4)
        axc.plot(tgrid, model_curve(i, tgrid), color=bin_cols[j], lw=0.6, alpha=0.55)
        axr.plot(tgrid, np.interp(tgrid, tt, lmp) - model_curve(i, tgrid),
                 color=bin_cols[j], lw=0.5, alpha=0.3)
    axr.fill_between(tgrid, np.percentile(res_stack, 16, 0),
                     np.percentile(res_stack, 84, 0), color=bin_cols[j], alpha=0.18)
    axr.plot(tgrid, np.median(res_stack, 0), "-", color="k", lw=1.6, label="bin median")
    axr.axhline(0, color="k", lw=0.8, ls=":")
    axr.set_ylim(-0.15, 0.15)
    axc.set_title(rf"$\log M_0$: {lo:.2f}–{hi:.2f}" + f"\nN={len(members)} (showing {len(show)})",
                  fontsize=9)
    axr.set_xlabel("cosmic time [Gyr]")
    if j == 0:
        axc.set_ylabel(r"$\log_{10} M_{\rm peak}(t)$")
        axr.set_ylabel("data $-$ DiffMAH [dex]")
        axc.plot([], [], color="0.6", lw=1.4, label="measured MAH")
        axc.plot([], [], color=bin_cols[j], lw=1.4, label="DiffMAH fit")
        axc.legend(fontsize=7, loc="lower right"); axr.legend(fontsize=7, loc="lower left")
fig2.suptitle("exp10 — DiffMAH fits across the mass range (equal-count $M_0$ bins)",
              fontsize=11)
fig2.tight_layout()
save_fig(fig2, FIGDIR / "exp10_fits_by_mass")

# %% do the DiffMAH params work as features? (predict aperture masses) ---------
def _annulus(a_o, a_i):
    return np.log10(np.clip(10.0 ** a_o - 10.0 ** a_i, 1.0, None))


Y = np.column_stack([aper[:, 0], _annulus(aper[:, 1], aper[:, 0]),
                     _annulus(aper[:, 2], aper[:, 1]), _annulus(aper[:, 4], aper[:, 2])])
# MAH-PCA(4) on the M0-normalized log-MAH (the incumbent feature set)
tg = np.linspace(2.2, 9.0, 18)
ms = np.full((N, 18), np.nan)
for i in range(N):
    if mah_t[i] is None:
        continue
    tt, lmp = mah_t[i]
    if tt[0] <= tg[0] and tt[-1] >= tg[-1]:
        ms[i] = np.interp(tg, tt, lmp) - logM0[i]
g = (np.isfinite(Y).all(1) & np.isfinite(logM0) & np.isfinite(ms).all(1)
     & np.isfinite(fits["early"]) & np.isfinite(fits["late"]) & np.isfinite(fits["logtc"]))
mu = ms[g].mean(0); _, _, Vt = np.linalg.svd(ms[g] - mu, full_matrices=False)
pca = (ms[g] - mu) @ Vt[:4].T
diff_feats = np.column_stack([fits["logmp"][g], fits["logtc"][g],
                              fits["early"][g], fits["late"][g]])
Yg = Y[g]; M0g = logM0[g]; Ng = len(Yg)
feature_sets = {
    "M0 only": M0g[:, None],
    "M0 + MAH-PCA(4)": np.column_stack([M0g, pca]),
    "DiffMAH (4 params)": diff_feats,
}


def cv_crps(X, k=5, seed=0):
    order = np.random.default_rng(seed).permutation(Ng)
    pred = np.full_like(Yg, np.nan); sig = np.full_like(Yg, np.nan)
    Xd = np.column_stack([np.ones(Ng), X])
    for fold in np.array_split(order, k):
        tr = np.setdiff1d(np.arange(Ng), fold)
        beta, *_ = np.linalg.lstsq(Xd[tr], Yg[tr], rcond=None)
        pred[fold] = Xd[fold] @ beta
        sig[fold] = (Yg[tr] - Xd[tr] @ beta).std(0)
    return crps_gaussian(Yg, pred, sig).mean(0)


print(f"\n[feature check] aperture-mass CRPS [dex] (n={Ng}):")
crps_by = {}
for name, X in feature_sets.items():
    crps_by[name] = cv_crps(X)
    print(f"  {name:22s} mean={crps_by[name].mean():.4f}   "
          + " ".join(f"{c:.3f}" for c in crps_by[name]))

# %% FIGURE 3 — DiffMAH params match MAH-PCA(4) as features
fig3, ax3 = plt.subplots(figsize=(5.6, 4.0))
TN = ["<10", "10-30", "30-50", "50-100"]
x = np.arange(4); w = 0.26
for k_i, (name, col) in enumerate(zip(feature_sets,
                                      [OKABE_ITO[7], OKABE_ITO[0], OKABE_ITO[2]])):
    ax3.bar(x + (k_i - 1) * w, crps_by[name], w, color=col, label=name)
ax3.set_xticks(x); ax3.set_xticklabels(TN)
ax3.set_xlabel("aperture / annulus [kpc]")
ax3.set_ylabel("aperture-mass CRPS [dex]")
ax3.set_title("Portable DiffMAH params ≈ MAH-PCA(4)")
ax3.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
fig3.tight_layout()
save_fig(fig3, FIGDIR / "exp10_feature_check")

# %% DiffMAH vs PCA: how well does each *describe* the MAH curve? --------------
# Reconstruction RMS over the 2.2-9.0 Gyr grid. PCA is the optimal *linear* basis
# (minimizes MSE for given #modes); DiffMAH is a constrained 4-param parametric
# form (one of which, logmp, is the M0 normalization the PCA gets for free).
logt0 = np.log10(T0)
model_abs = log_mah(np.log10(tg), fits["logmp"][g], fits["logtc"][g],
                    fits["early"][g], fits["late"][g], logt0)        # (Ng, 18)
rms_dmah = np.sqrt(np.mean((model_abs - (ms[g] + M0g[:, None])) ** 2, axis=1))
ks = np.arange(1, 7)
rms_pca = {kk: np.sqrt(np.mean(
    (ms[g] - (mu + (ms[g] - mu) @ Vt[:kk].T @ Vt[:kk])) ** 2, axis=1)) for kk in ks}
print(f"\n[DiffMAH vs PCA] median MAH reconstruction RMS [dex], {tg[0]:.1f}-{tg[-1]:.1f} Gyr:")
print(f"  DiffMAH (4 params): {np.median(rms_dmah):.4f}")
print("  PCA modes: " + "  ".join(f"{kk}:{np.median(rms_pca[kk]):.4f}" for kk in ks))

fig4, ax4 = plt.subplots(figsize=(5.8, 4.0))
ax4.plot(ks, [np.median(rms_pca[kk]) for kk in ks], "-o", color=OKABE_ITO[0],
         ms=5, label="MAH-PCA ($k$ modes)")
ax4.axhline(np.median(rms_dmah), color=OKABE_ITO[5], ls="--", lw=1.6,
            label="DiffMAH (4 params)")
ax4.set_xlabel("number of shape components")
ax4.set_ylabel("median MAH reconstruction RMS [dex]")
ax4.set_title("DiffMAH vs PCA: MAH description quality")
ax4.legend()
fig4.tight_layout()
save_fig(fig4, FIGDIR / "exp10_vs_pca")

# %% save outputs (portable per-halo feature tables, both inner-time cuts) -----
OUTDIR.mkdir(parents=True, exist_ok=True)
for tmin, store in fits_by_tmin.items():
    pt = Table({"index": idx})
    for k in keys:
        pt[f"dmah_{k}"] = store[k]
    pt.write(OUTDIR / f"diffmah_params_tmin{tmin:g}gyr.csv", overwrite=True)
write_manifest(OUTDIR, params={
    "n": int(N), "mah_k": MAH_K, "t0_gyr": T0, "t_min_default": T_MIN,
    "median_rms_dex": float(np.nanmedian(rms)),
    "recon_rms_diffmah": float(np.median(rms_dmah)),
    "recon_rms_pca3": float(np.median(rms_pca[3])),
    "recon_rms_pca4": float(np.median(rms_pca[4])),
    "crps_diffmah": float(crps_by["DiffMAH (4 params)"].mean()),
    "crps_mahpca": float(crps_by["M0 + MAH-PCA(4)"].mean()),
    "crps_m0": float(crps_by["M0 only"].mean())})
print(f"\nwrote figures -> {FIGDIR}\nwrote outputs -> {OUTDIR}")
