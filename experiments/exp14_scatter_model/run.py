"""exp14 — Scatter model: a heteroscedastic residual covariance for the emulator.

The mean is settled (exp09/exp12: linear) and the four DiffMAH params are at the
information ceiling for the outskirts (exp13). What remains is *scatter*. The
exp08/exp11 emulator uses a single residual covariance for every halo
(homoscedastic): P(aperture masses | DiffMAH) = N(mean(X), Sigma). But if the
predictive scatter depends on the halo (e.g. wider for low-mass or recently
accreting halos), a homoscedastic model is mis-calibrated *conditionally* even
when it looks fine *marginally* — over-confident for noisy halos, under-confident
for clean ones.

Here we (A) diagnose whether the residual scatter depends on the DiffMAH
features, (B) fit a heteroscedastic Gaussian — per-aperture log-linear standard
deviation sigma_j(X) = exp(gamma_j . [1, X]) with a fixed residual correlation R,
so Sigma(X) = D(X) R D(X) — and (C) judge it with the exp07 suite, with emphasis
on *conditional* calibration (coverage inside low/high predicted-scatter bins)
and a PIT non-Gaussianity check.

Baseline (homoscedastic, exp11): overall CRPS 0.0883; calibration 54/72/91/95.

Run from the repo root (small/fast validation pass):
    EXP14_NMAX=600 PYTHONPATH=. uv run python \
        experiments/exp14_scatter_model/run.py
Full pass:
    PYTHONPATH=. uv run python experiments/exp14_scatter_model/run.py
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
from hongshao.metrics import crps_gaussian, interval_coverage, pit              # noqa: E402
from hongshao.plotting import set_style, save_fig, OKABE_ITO                    # noqa: E402
from hongshao.provenance import write_manifest                                  # noqa: E402

set_style()
HERE = Path(__file__).resolve().parent
OUTDIR, FIGDIR = HERE / "outputs", HERE / "figures"
TABLE = ROOT / "data" / "processed" / "tng300_072_z0p4.fits"
TNAMES = ["<10", "10-30", "30-50", "50-100"]
FEAT = ["logmp", "logtc", "early", "late"]

NMAX = int(os.environ.get("EXP14_NMAX", 0))
RIDGE = float(os.environ.get("EXP14_RIDGE", 2.0))      # ridge on log-variance slopes

t = Table.read(TABLE)
t = t[t["use"]]
if NMAX:
    t = t[:NMAX]
aper = np.asarray(t["logmstar_aper"], float)
dmah = np.column_stack([np.asarray(t[c], float) for c in
                        ("dmah_logmp", "dmah_logtc", "dmah_early", "dmah_late")])


def _annulus(a_outer, a_inner):
    return np.log10(np.clip(10.0 ** a_outer - 10.0 ** a_inner, 1.0, None))


Y = np.column_stack([aper[:, 0], _annulus(aper[:, 1], aper[:, 0]),
                     _annulus(aper[:, 2], aper[:, 1]), _annulus(aper[:, 4], aper[:, 2])])
g = np.isfinite(Y).all(1) & np.isfinite(dmah).all(1)
Y = Y[g]; X = dmah[g]; N = len(Y)
print(f"exp14: scatter model on n={N} galaxies")


# %% ---- log-variance MLE: sigma^2(x) = exp(gamma . [1, x_std]) --------------
def fit_logvar(r, Z, ridge=RIDGE):
    """Max-likelihood Gaussian log-variance regression for residuals r on
    standardized features Z. Minimizes 0.5 * sum[s + r^2 exp(-s)] with s = A@g,
    plus a ridge on the slopes (not the intercept). Returns g (intercept+slopes)."""
    A = np.column_stack([np.ones(len(r)), Z])
    r2 = r ** 2

    def nll(gm):
        s = A @ gm
        return 0.5 * np.sum(s + r2 * np.exp(-s)) + 0.5 * ridge * np.sum(gm[1:] ** 2)

    def grad(gm):
        s = A @ gm
        w = 0.5 * (1.0 - r2 * np.exp(-s))
        out = A.T @ w
        out[1:] += ridge * gm[1:]
        return out

    g0 = np.r_[np.log(max(r2.mean(), 1e-6)), np.zeros(Z.shape[1])]
    return minimize(nll, g0, jac=grad, method="L-BFGS-B").x


def corr_from(C):
    d = np.sqrt(np.diag(C))
    return C / np.outer(d, d)


def joint_nll(Y_, MU, SIGFULL):
    resid = Y_ - MU
    out = np.empty(len(Y_))
    for i in range(len(Y_)):
        S = SIGFULL[i]
        _, logdet = np.linalg.slogdet(S)
        out[i] = 0.5 * (4 * np.log(2 * np.pi) + logdet + resid[i] @ np.linalg.solve(S, resid[i]))
    return out


# %% ---- 5-fold CV: shared linear mean; homoscedastic vs heteroscedastic Sigma
def cv_models(k=5, seed=0):
    order = np.random.default_rng(seed).permutation(N)
    MU = np.full((N, 4), np.nan)
    SIG_HOMO = np.full((N, 4), np.nan)             # marginal sigma per aperture
    SIG_HET = np.full((N, 4), np.nan)
    FULL_HOMO = np.full((N, 4, 4), np.nan)
    FULL_HET = np.full((N, 4, 4), np.nan)
    gammas = []
    for fold in np.array_split(order, k):
        tr = np.setdiff1d(np.arange(N), fold)
        A_tr = np.column_stack([np.ones(len(tr)), X[tr]])
        A_te = np.column_stack([np.ones(len(fold)), X[fold]])
        r_tr = np.empty((len(tr), 4))
        for j in range(4):
            beta, *_ = np.linalg.lstsq(A_tr, Y[tr, j], rcond=None)
            MU[fold, j] = A_te @ beta
            r_tr[:, j] = Y[tr, j] - A_tr @ beta
        # homoscedastic: one covariance for all halos (exp11 baseline)
        Sig0 = np.cov(r_tr.T)
        SIG_HOMO[fold] = np.sqrt(np.diag(Sig0))
        FULL_HOMO[fold] = Sig0
        # heteroscedastic: log-linear sigma_j(x); standardize features on train
        mx, sx = X[tr].mean(0), X[tr].std(0)
        Z_tr = (X[tr] - mx) / sx
        Z_te = (X[fold] - mx) / sx
        sig_tr = np.empty((len(tr), 4))
        sig_te = np.empty((len(fold), 4))
        for j in range(4):
            gm = fit_logvar(r_tr[:, j], Z_tr)
            gammas.append(gm)
            sig_tr[:, j] = np.exp(0.5 * (np.column_stack([np.ones(len(tr)), Z_tr]) @ gm))
            sig_te[:, j] = np.exp(0.5 * (np.column_stack([np.ones(len(fold)), Z_te]) @ gm))
        SIG_HET[fold] = sig_te
        # correlation from standardized residuals (scatter removed), fixed across halos
        R = np.corrcoef((r_tr / sig_tr).T)
        for n_i, i in enumerate(fold):
            D = np.diag(sig_te[n_i])
            FULL_HET[i] = D @ R @ D
    return dict(MU=MU, SIG_HOMO=SIG_HOMO, SIG_HET=SIG_HET,
                FULL_HOMO=FULL_HOMO, FULL_HET=FULL_HET, gammas=np.array(gammas))


cv = cv_models()
MU = cv["MU"]


# %% ---- (A) diagnose heteroscedasticity -------------------------------------
# full-data log-variance fit per aperture, to read which features drive scatter
resid_oof = Y - MU
Zall = (X - X.mean(0)) / X.std(0)
print("\n[A] log-variance model sigma_j^2(x) = exp(g0 + g.x_std); slopes (per +1 sigma):")
print(f"  {'aperture':8s} {'sigma0':>7s} " + " ".join(f"{f:>7s}" for f in FEAT)
      + f"  {'sig_max/min':>11s}")
gamma_full = np.zeros((4, 5))
for j in range(4):
    gm = fit_logvar(resid_oof[:, j], Zall)
    gamma_full[j] = gm
    sig = np.exp(0.5 * (np.column_stack([np.ones(N), Zall]) @ gm))
    print(f"  {TNAMES[j]:8s} {np.exp(0.5*gm[0]):7.3f} "
          + " ".join(f"{0.5*gm[1+i]:+7.3f}" for i in range(4))
          + f"  {sig.max()/sig.min():11.2f}")


# %% ---- (B,C) evaluate: marginal CRPS, calibration, joint NLL, conditional cal
def marginal_scores(SIG):
    crps = crps_gaussian(Y, MU, SIG).mean(0)
    lev, cov = interval_coverage(Y, MU, SIG)
    return crps, lev, cov


crps_homo, lev, cov_homo = marginal_scores(cv["SIG_HOMO"])
crps_het, _, cov_het = marginal_scores(cv["SIG_HET"])
nll_homo = joint_nll(Y, MU, cv["FULL_HOMO"])
nll_het = joint_nll(Y, MU, cv["FULL_HET"])

print("\n[B] marginal CRPS [dex] (lower=better):")
print(f"  homoscedastic  {crps_homo.mean():.4f}  [" + " ".join(f"{c:.3f}" for c in crps_homo) + "]")
print(f"  heteroscedastic{crps_het.mean():.4f}  [" + " ".join(f"{c:.3f}" for c in crps_het) + "]")
print(f"  marginal coverage 50/68/90/95: homo "
      + "/".join(f"{c:.2f}" for c in cov_homo) + "   het "
      + "/".join(f"{c:.2f}" for c in cov_het))
print(f"\n[B] joint negative log-score (NLL, lower=better):")
print(f"  homoscedastic  {nll_homo.mean():.4f}")
print(f"  heteroscedastic{nll_het.mean():.4f}   (better by {nll_homo.mean()-nll_het.mean():+.4f} nats)")

# conditional calibration: bin halos by the hetero-predicted sigma, per aperture,
# and report 68% interval coverage in each tercile for BOTH models. A
# homoscedastic model over-covers the clean tercile and under-covers the noisy
# one; a good heteroscedastic model is flat near 0.68.
print("\n[B] conditional calibration — 68% coverage by predicted-scatter tercile:")
print(f"  {'aperture':8s} {'tercile':7s} {'homo':>6s} {'het':>6s}")
cond = {}
for j in range(4):
    s = cv["SIG_HET"][:, j]
    edges = np.quantile(s, [0, 1/3, 2/3, 1.0])
    bins = np.clip(np.digitize(s, edges[1:-1]), 0, 2)
    for bk, blab in enumerate(["low", "mid", "high"]):
        m = bins == bk
        z = np.abs(Y[m, j] - MU[m, j])
        c_homo = np.mean(z <= cv["SIG_HOMO"][m, j])             # 1-sigma ~ 68.27%
        c_het = np.mean(z <= cv["SIG_HET"][m, j])
        cond[(j, bk)] = (c_homo, c_het)
        if bk != 1:
            print(f"  {TNAMES[j]:8s} {blab:7s} {c_homo:6.2f} {c_het:6.2f}")

# spread of conditional coverage (lower = better calibrated across noisiness)
spread_homo = np.mean([abs(cond[(j, 2)][0] - cond[(j, 0)][0]) for j in range(4)])
spread_het = np.mean([abs(cond[(j, 2)][1] - cond[(j, 0)][1]) for j in range(4)])
print(f"  => mean |high - low| coverage gap: homo {spread_homo:.3f}  het {spread_het:.3f}  "
      f"(smaller = scatter better captured)")


# %% ---- FIGURE 1: residual scatter vs each feature, per aperture ------------
def _binned_std(xv, rv, nbin=10):
    e = np.quantile(xv, np.linspace(0, 1, nbin + 1))
    cen, sd = [], []
    for a, b in zip(e[:-1], e[1:]):
        m = (xv >= a) & (xv <= b)
        if m.sum() >= 15:
            cen.append(np.median(xv[m])); sd.append(rv[m].std())
    return np.asarray(cen), np.asarray(sd)


fig1, ax1 = plt.subplots(4, 4, figsize=(13.5, 11.0), sharex="col")
for j in range(4):
    for i, fn in enumerate(FEAT):
        a = ax1[j, i]
        cen, sd = _binned_std(X[:, i], resid_oof[:, j])
        a.plot(cen, sd, "o-", color=OKABE_ITO[j], ms=4, lw=1.3)
        a.axhline(cv["SIG_HOMO"][:, j].mean(), ls="--", color="0.5", lw=1)
        a.set_ylim(0, max(0.28, sd.max() * 1.15) if len(sd) else 0.28)
        if j == 0:
            a.set_title(fn)
        if i == 0:
            a.set_ylabel(f"{TNAMES[j]} kpc\nresidual std [dex]", fontsize=9)
        if j == 3:
            a.set_xlabel(fn)
fig1.suptitle("exp14 — residual scatter vs DiffMAH feature (dashed = homoscedastic "
              "value; slope = heteroscedasticity)", fontsize=11)
fig1.tight_layout()
save_fig(fig1, FIGDIR / "exp14_scatter_structure")


# %% ---- FIGURE 2: skill, marginal + conditional calibration -----------------
fig2, (axA, axB, axC) = plt.subplots(1, 3, figsize=(13.8, 4.2))
xb = np.arange(4); w = 0.38
axA.bar(xb - w/2, crps_homo, w, color=OKABE_ITO[7], label="homoscedastic")
axA.bar(xb + w/2, crps_het, w, color=OKABE_ITO[2], label="heteroscedastic")
axA.set_xticks(xb); axA.set_xticklabels(TNAMES)
axA.set_xlabel("aperture / annulus [kpc]"); axA.set_ylabel("CV CRPS [dex]")
axA.set_title("Marginal skill"); axA.legend(fontsize=8)
axA.text(-0.13, 1.04, "A", transform=axA.transAxes, fontweight="bold", fontsize=12)

axB.plot([0, 1], [0, 1], ":", color="0.5", lw=1)
axB.plot(lev, cov_homo, "-o", color=OKABE_ITO[7], ms=5, label="homoscedastic")
axB.plot(lev, cov_het, "-s", color=OKABE_ITO[2], ms=5, label="heteroscedastic")
axB.set_xlabel("nominal central interval"); axB.set_ylabel("empirical coverage")
axB.set_title("Marginal calibration"); axB.set_xlim(0.4, 1.0); axB.set_ylim(0.4, 1.0)
axB.legend(fontsize=8)
axB.text(-0.13, 1.04, "B", transform=axB.transAxes, fontweight="bold", fontsize=12)

# conditional 68% coverage vs noisiness tercile, averaged over apertures
terc = np.arange(3)
ch = [np.mean([cond[(j, b)][0] for j in range(4)]) for b in terc]
ce = [np.mean([cond[(j, b)][1] for j in range(4)]) for b in terc]
axC.axhline(0.6827, ls=":", color="0.5", lw=1, label="nominal 68%")
axC.plot(terc, ch, "-o", color=OKABE_ITO[7], ms=6, label="homoscedastic")
axC.plot(terc, ce, "-s", color=OKABE_ITO[2], ms=6, label="heteroscedastic")
axC.set_xticks(terc); axC.set_xticklabels(["low", "mid", "high"])
axC.set_xlabel("predicted-scatter tercile"); axC.set_ylabel("68% interval coverage")
axC.set_title("Conditional calibration"); axC.legend(fontsize=8)
axC.text(-0.13, 1.04, "C", transform=axC.transAxes, fontweight="bold", fontsize=12)
fig2.suptitle(f"exp14 — heteroscedastic scatter (n={N}); joint NLL "
              f"{nll_homo.mean():.3f}->{nll_het.mean():.3f}, "
              f"cond. coverage gap {spread_homo:.2f}->{spread_het:.2f}", fontsize=11)
fig2.tight_layout()
save_fig(fig2, FIGDIR / "exp14_skill_calibration")


# %% ---- FIGURE 3: PIT (non-Gaussianity) -------------------------------------
fig3, ax3 = plt.subplots(1, 2, figsize=(9.6, 4.0))
for ax, SIG, lab in [(ax3[0], cv["SIG_HOMO"], "homoscedastic"),
                     (ax3[1], cv["SIG_HET"], "heteroscedastic")]:
    p = pit(Y, MU, SIG).ravel()
    ax.hist(p, bins=20, range=(0, 1), color=OKABE_ITO[7 if "homo" in lab else 2],
            edgecolor="white", density=True)
    ax.axhline(1.0, ls="--", color="0.4", lw=1)
    ax.set_xlabel("PIT"); ax.set_title(f"{lab}  (std={p.std():.3f}, ideal 0.289)")
    ax.set_ylim(0, 2.0)
ax3[0].set_ylabel("density")
fig3.suptitle("exp14 — PIT histograms (flat = calibrated Gaussian); "
              "U-shape = under-dispersed, dome = over-dispersed", fontsize=11)
fig3.tight_layout()
save_fig(fig3, FIGDIR / "exp14_pit")


# %% ---- save outputs --------------------------------------------------------
OUTDIR.mkdir(parents=True, exist_ok=True)
gt = Table({"aperture": TNAMES, "log_sigma0": gamma_full[:, 0].tolist()})
for i, f in enumerate(FEAT):
    gt[f"d_logsig_{f}"] = (0.5 * gamma_full[:, 1 + i]).tolist()
gt["crps_homo"] = crps_homo.tolist()
gt["crps_het"] = crps_het.tolist()
gt.write(OUTDIR / "scatter_model.csv", overwrite=True)
write_manifest(OUTDIR, params={
    "n": int(N), "ridge": RIDGE, "targets": TNAMES,
    "crps_homo": float(crps_homo.mean()), "crps_het": float(crps_het.mean()),
    "nll_homo": float(nll_homo.mean()), "nll_het": float(nll_het.mean()),
    "cond_cov_gap_homo": float(spread_homo), "cond_cov_gap_het": float(spread_het),
    "marg_cov_homo": cov_homo.tolist(), "marg_cov_het": cov_het.tolist()})
print(f"\nwrote figures -> {FIGDIR}\nwrote outputs -> {OUTDIR}")
print(f"\n[verdict] joint NLL {nll_homo.mean():.3f} -> {nll_het.mean():.3f} "
      f"({nll_homo.mean()-nll_het.mean():+.3f} nats); conditional coverage gap "
      f"{spread_homo:.3f} -> {spread_het:.3f}")
