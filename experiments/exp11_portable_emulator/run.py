"""exp11 — The portable Ultimate-SHMR emulator (on DiffMAH features).

exp08 built a conditional multivariate-Gaussian emulator P(aperture masses | M0,
MAH) with MAH-PCA(4) features — but that PCA basis is defined on the TNG sample,
so it doesn't transfer. exp10 showed the four portable DiffMAH params carry ~88%
of the MAH-PCA signal for the *mean*. Here we rebuild the full *probabilistic*
emulator on the cached DiffMAH params (dmah_*) and confirm it matches the
MAH-PCA version under the exp07 suite — CRPS, calibration, and the joint
covariance — so we have a portable emulator, not just portable point predictions.

Run from the repo root (EXP11_NMAX=400 for a sub-minute pass):
    PYTHONPATH=. uv run python experiments/exp11_portable_emulator/run.py
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
from hongshao.metrics import crps_gaussian, gaussian_logscore, interval_coverage  # noqa: E402
from hongshao.plotting import set_style, save_fig, OKABE_ITO                     # noqa: E402
from hongshao.provenance import write_manifest                                  # noqa: E402

set_style()
HERE = Path(__file__).resolve().parent
OUTDIR, FIGDIR = HERE / "outputs", HERE / "figures"
TABLE = ROOT / "data" / "processed" / "tng300_072_z0p4.fits"
TNAMES = ["<10", "10-30", "30-50", "50-100"]

t = Table.read(TABLE)
t = t[t["use"]]
NMAX = int(os.environ.get("EXP11_NMAX", 0))
if NMAX:
    t = t[:NMAX]
logM0 = np.asarray(t["logm0_halo"], float)
aper = np.asarray(t["logmstar_aper"], float)
z50 = np.asarray(t["z50"], float)
idx = np.asarray(t["index"])
dmah = np.column_stack([np.asarray(t[c], float) for c in
                        ("dmah_logmp", "dmah_logtc", "dmah_early", "dmah_late")])

# MAH-PCA(4) features (the exp08 incumbent), to compare against
mah = load_mah(); tsnap = load_cosmic_time(); tg = np.linspace(2.2, 9.0, 18)
ms = np.full((len(t), 18), np.nan)
for r, i in enumerate(idx):
    sn, lmp = peak_history(mah[int(i)])
    if lmp is None:
        continue
    tt = tsnap[sn.astype(int)]
    if tt[0] <= tg[0] and tt[-1] >= tg[-1]:
        ms[r] = np.interp(tg, tt, lmp) - logM0[r]


def _annulus(a_o, a_i):
    return np.log10(np.clip(10.0 ** a_o - 10.0 ** a_i, 1.0, None))


Y = np.column_stack([aper[:, 0], _annulus(aper[:, 1], aper[:, 0]),
                     _annulus(aper[:, 2], aper[:, 1]), _annulus(aper[:, 4], aper[:, 2])])
g = (np.isfinite(Y).all(1) & np.isfinite(logM0) & np.isfinite(ms).all(1)
     & np.isfinite(dmah).all(1))
mu_m = ms[g].mean(0); _, _, Vt = np.linalg.svd(ms[g] - mu_m, full_matrices=False)
pca = (ms[g] - mu_m) @ Vt[:4].T
M0 = logM0[g]; Y = Y[g]; z50 = z50[g]; dmahg = dmah[g]; N = len(Y)
print(f"exp11: portable emulator on n={N} galaxies")

FEATURES = {
    "M0 only": M0[:, None],
    "M0 + MAH-PCA(4)": np.column_stack([M0, pca]),
    "DiffMAH (portable)": dmahg,
}


# %% conditional-Gaussian emulator: out-of-fold mean + per-fold full covariance
def cv_emulator(X, k=5, seed=0):
    Xd = np.column_stack([np.ones(N), X])
    order = np.random.default_rng(seed).permutation(N)
    mu = np.full((N, 4), np.nan); Sig = np.full((N, 4, 4), np.nan)
    for fold in np.array_split(order, k):
        tr = np.setdiff1d(np.arange(N), fold)
        beta, *_ = np.linalg.lstsq(Xd[tr], Y[tr], rcond=None)
        mu[fold] = Xd[fold] @ beta
        Sig[fold] = np.cov((Y[tr] - Xd[tr] @ beta).T)
    return mu, Sig


def marginal(mu, Sig):
    sig = np.sqrt(np.stack([np.diag(s) for s in Sig]))
    return (crps_gaussian(Y, mu, sig).mean(0), gaussian_logscore(Y, mu, sig).mean(0),
            interval_coverage(Y, mu, sig))


def joint_nll(mu, Sig, diagonal=False):
    resid = Y - mu; out = np.empty(N)
    for i in range(N):
        S = np.diag(np.diag(Sig[i])) if diagonal else Sig[i]
        _, logdet = np.linalg.slogdet(S)
        out[i] = 0.5 * (4 * np.log(2 * np.pi) + logdet + resid[i] @ np.linalg.solve(S, resid[i]))
    return float(np.mean(out))


emu = {name: cv_emulator(X) for name, X in FEATURES.items()}
print(f"\n{'features':20s} {'CRPS':>7s} {'logS':>8s}   coverage 50/68/90/95")
scores = {}
for name, (mu, Sig) in emu.items():
    crps, ls, (lev, cov) = marginal(mu, Sig)
    scores[name] = dict(crps=crps, ls=ls, cov=cov, mu=mu, Sig=Sig)
    print(f"  {name:18s} {crps.mean():7.4f} {ls.mean():8.4f}   "
          + " ".join(f"{c:.2f}" for c in cov))

print("\njoint multivariate log-score (NLL): full vs diagonal covariance")
for name in ("M0 + MAH-PCA(4)", "DiffMAH (portable)"):
    mu, Sig = emu[name]
    nf, nd = joint_nll(mu, Sig), joint_nll(mu, Sig, diagonal=True)
    print(f"  {name:18s} full={nf:7.3f}  diag={nd:7.3f}  (full better by {nd-nf:.2f})")

# covariance match (DiffMAH emulator)
def _corr(C):
    d = np.sqrt(np.diag(C)); return C / np.outer(d, d)


emp_corr = _corr(np.cov((Y - emu["DiffMAH (portable)"][0]).T))
off = ~np.eye(4, dtype=bool)
print(f"\nDiffMAH emulator residual correlation: mean |off-diag| = "
      f"{np.mean(np.abs(emp_corr[off])):.2f}")


# %% FIGURE 1 — skill + calibration (DiffMAH vs MAH-PCA vs M0)
fig1, (axA, axB) = plt.subplots(1, 2, figsize=(9.8, 4.0))
x = np.arange(4); w = 0.26
cols = [OKABE_ITO[7], OKABE_ITO[0], OKABE_ITO[2]]
short = ["M0 only", "MAH-PCA(4)", "DiffMAH"]
for k_i, (name, c, lab) in enumerate(zip(FEATURES, cols, short)):
    axA.bar(x + (k_i - 1) * w, scores[name]["crps"], w, color=c, label=lab)
axA.set_xticks(x); axA.set_xticklabels(TNAMES)
axA.set_xlabel("aperture / annulus [kpc]")
axA.set_ylabel("CRPS [dex]  (lower = better)")
axA.set_title("Predictive skill")
axA.legend(fontsize=7)
axA.text(-0.13, 1.04, "A", transform=axA.transAxes, fontweight="bold", fontsize=12)

axB.plot([0, 1], [0, 1], ":", color="0.5", lw=1)
lev = np.array([0.5, 0.68, 0.9, 0.95])
for name, c, mk in [("M0 + MAH-PCA(4)", OKABE_ITO[0], "o"),
                    ("DiffMAH (portable)", OKABE_ITO[2], "s")]:
    axB.plot(lev, scores[name]["cov"], "-", marker=mk, color=c, ms=5, label=name)
axB.set_xlabel("nominal central interval"); axB.set_ylabel("empirical coverage")
axB.set_title("Calibration"); axB.set_xlim(0.4, 1.0); axB.set_ylim(0.4, 1.0)
axB.legend(fontsize=8)
axB.text(-0.13, 1.04, "B", transform=axB.transAxes, fontweight="bold", fontsize=12)
fig1.suptitle(f"exp11 — portable DiffMAH emulator matches MAH-PCA (n={N})", fontsize=11)
fig1.tight_layout()
save_fig(fig1, FIGDIR / "exp11_skill_calibration")

# %% FIGURE 2 — probabilistic painting from DiffMAH features (same M0)
fig2, axC = plt.subplots(figsize=(5.8, 4.2))
mu_d, Sig_d = emu["DiffMAH (portable)"]
sel = np.where((M0 > 13.4) & (M0 < 13.6))[0]
order_sel = sel[np.argsort(z50[sel])]
picks = [order_sel[-2], order_sel[len(order_sel) // 2], order_sel[1]]
for kk, (pk, c, lab) in enumerate(zip(
        picks, [OKABE_ITO[4], OKABE_ITO[7], OKABE_ITO[5]],
        ["early former", "median", "late former"])):
    sig = np.sqrt(np.diag(Sig_d[pk]))
    axC.errorbar(x + 0.04 * (kk - 1), mu_d[pk], yerr=sig, fmt="o-", color=c, ms=5,
                 capsize=3, lw=1.4, label=f"{lab} ($z_{{50}}$={z50[pk]:.2f})")
    axC.plot(x, Y[pk], "x", color=c, ms=8, mew=2)
axC.plot([], [], "kx", ms=8, mew=2, label="truth")
axC.set_xticks(x); axC.set_xticklabels(TNAMES)
axC.set_xlabel("aperture / annulus [kpc]"); axC.set_ylabel(r"$\log_{10} M_*$ [$M_\odot$]")
axC.set_title(r"Painting from DiffMAH (same $M_0$, $\pm1\sigma$)")
axC.legend(fontsize=7)
fig2.tight_layout()
save_fig(fig2, FIGDIR / "exp11_painting")

# %% save outputs
OUTDIR.mkdir(parents=True, exist_ok=True)
st = Table({"features": list(FEATURES),
            "crps": [float(scores[n]["crps"].mean()) for n in FEATURES],
            "logscore": [float(scores[n]["ls"].mean()) for n in FEATURES]})
st.write(OUTDIR / "portable_emulator_scores.csv", overwrite=True)
write_manifest(OUTDIR, params={
    "n": int(N), "targets": TNAMES,
    "crps_diffmah": float(scores["DiffMAH (portable)"]["crps"].mean()),
    "crps_mahpca": float(scores["M0 + MAH-PCA(4)"]["crps"].mean()),
    "crps_m0": float(scores["M0 only"]["crps"].mean())})
print(f"\nwrote figures -> {FIGDIR}\nwrote outputs -> {OUTDIR}")
