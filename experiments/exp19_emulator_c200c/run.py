"""exp19 — Fold c_200c into the heteroscedastic emulator (the working model).

exp14 built the working Ultimate-SHMR emulator: per-aperture linear mean +
heteroscedastic full residual covariance (sigma_j(X)=exp(gamma_j.[1,X]) with a
fixed correlation R), on portable DiffMAH features. exp16-18 showed halo
concentration `c_200c` adds real, MAH-independent skill to the *mean* (+5% CRPS).

Here we fold `c_200c` into the full emulator and ask two things:
  1. the total gain of the DiffMAH+c_200c emulator over the DiffMAH one
     (mean + heteroscedastic covariance), and
  2. does `c_200c` also help the *scatter* (does it enter the log-variance and
     improve conditional calibration), or only the mean?

Decomposition (all 5-fold CV, the exp07 suite: marginal CRPS, joint NLL,
marginal + conditional calibration):
  A. mean=DiffMAH,        scatter=hetero(DiffMAH)         -- exp14 baseline
  B. mean=DiffMAH+c200c,  scatter=hetero(DiffMAH)         -- c200c in the mean only
  C. mean=DiffMAH+c200c,  scatter=hetero(DiffMAH+c200c)   -- c200c in mean + scatter
(+ homoscedastic references for the two means.)

Run (fast): EXP19_NMAX=700 PYTHONPATH=. uv run python experiments/exp19_emulator_c200c/run.py
Full:       PYTHONPATH=. uv run python experiments/exp19_emulator_c200c/run.py
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
from hongshao.metrics import crps_gaussian, interval_coverage                   # noqa: E402
from hongshao.plotting import set_style, save_fig, OKABE_ITO                     # noqa: E402
from hongshao.provenance import write_manifest                                  # noqa: E402

set_style()
HERE = Path(__file__).resolve().parent
OUTDIR, FIGDIR = HERE / "outputs", HERE / "figures"
TABLE = ROOT / "data" / "processed" / "tng300_072_z0p4.fits"
TNAMES = ["<10", "10-30", "30-50", "50-100"]
NMAX = int(os.environ.get("EXP19_NMAX", 0))
RIDGE = float(os.environ.get("EXP19_RIDGE", 2.0))

t = Table.read(TABLE); t = t[t["use"]]
if NMAX:
    t = t[:NMAX]
aper = np.asarray(t["logmstar_aper"], float)
dmah = np.column_stack([np.asarray(t[c], float) for c in
                        ("dmah_logmp", "dmah_logtc", "dmah_early", "dmah_late")])
c200 = np.asarray(t["c_200c"], float)


def _ann(a_o, a_i):
    return np.log10(np.clip(10.0 ** a_o - 10.0 ** a_i, 1.0, None))


Y = np.column_stack([aper[:, 0], _ann(aper[:, 1], aper[:, 0]),
                     _ann(aper[:, 2], aper[:, 1]), _ann(aper[:, 4], aper[:, 2])])
g = np.isfinite(Y).all(1) & np.isfinite(dmah).all(1) & np.isfinite(c200)
Y = Y[g]; dmah = dmah[g]; c200 = c200[g]; N = len(Y)
X_dm = dmah
X_dmc = np.column_stack([dmah, c200])
VAR_FEAT = ["logmp", "logtc", "early", "late", "c200c"]
print(f"exp19: emulator with c_200c on n={N}")


# %% ---- heteroscedastic emulator (exp14 machinery, parameterized) -----------
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


def cv_emulator(Xmean, Xvar, hetero, k=5, seed=0):
    order = np.random.default_rng(seed).permutation(N)
    MU = np.full((N, 4), np.nan); SIG = np.full((N, 4), np.nan)
    FULL = np.full((N, 4, 4), np.nan)
    Am = np.column_stack([np.ones(N), Xmean])
    for fold in np.array_split(order, k):
        tr = np.setdiff1d(np.arange(N), fold)
        r_tr = np.empty((len(tr), 4))
        for j in range(4):
            beta, *_ = np.linalg.lstsq(Am[tr], Y[tr, j], rcond=None)
            MU[fold, j] = Am[fold] @ beta
            r_tr[:, j] = Y[tr, j] - Am[tr] @ beta
        if not hetero:
            Sig0 = np.cov(r_tr.T)
            SIG[fold] = np.sqrt(np.diag(Sig0)); FULL[fold] = Sig0
        else:
            mx, sx = Xvar[tr].mean(0), Xvar[tr].std(0)
            Ztr = (Xvar[tr] - mx) / sx; Zte = (Xvar[fold] - mx) / sx
            sig_tr = np.empty((len(tr), 4)); sig_te = np.empty((len(fold), 4))
            for j in range(4):
                gm = fit_logvar(r_tr[:, j], Ztr)
                sig_tr[:, j] = np.exp(0.5 * (np.column_stack([np.ones(len(tr)), Ztr]) @ gm))
                sig_te[:, j] = np.exp(0.5 * (np.column_stack([np.ones(len(fold)), Zte]) @ gm))
            SIG[fold] = sig_te
            R = np.corrcoef((r_tr / sig_tr).T)
            for n_i, i in enumerate(fold):
                D = np.diag(sig_te[n_i]); FULL[i] = D @ R @ D
    return MU, SIG, FULL


def joint_nll(MU, FULL):
    resid = Y - MU; out = np.empty(N)
    for i in range(N):
        _, ld = np.linalg.slogdet(FULL[i])
        out[i] = 0.5 * (4 * np.log(2 * np.pi) + ld + resid[i] @ np.linalg.solve(FULL[i], resid[i]))
    return float(out.mean())


def cond_cov_gap(MU, SIG):
    """mean |coverage(high sigma) - coverage(low sigma)| of the 68% interval."""
    gaps = []
    for j in range(4):
        s = SIG[:, j]; edges = np.quantile(s, [0, 1/3, 2/3, 1.0])
        bins = np.clip(np.digitize(s, edges[1:-1]), 0, 2)
        cov = []
        for bk in (0, 2):
            m = bins == bk
            cov.append(np.mean(np.abs(Y[m, j] - MU[m, j]) <= SIG[m, j]))
        gaps.append(abs(cov[1] - cov[0]))
    return float(np.mean(gaps))


def evaluate(name, Xmean, Xvar, hetero):
    MU, SIG, FULL = cv_emulator(Xmean, Xvar, hetero)
    crps = crps_gaussian(Y, MU, SIG).mean(0)
    _, cov = interval_coverage(Y, MU, SIG)
    nll = joint_nll(MU, FULL)
    gap = cond_cov_gap(MU, SIG)
    print(f"  {name:34s} CRPS={crps.mean():.4f}  NLL={nll:+.3f}  "
          f"cov={'/'.join(f'{c:.2f}' for c in cov)}  condgap={gap:.3f}")
    return dict(crps=crps, cov=cov, nll=nll, gap=gap, MU=MU, SIG=SIG)


print("\n[emulator decomposition] (lower CRPS/NLL/gap = better)")
A = evaluate("A. DiffMAH | het(DiffMAH)", X_dm, X_dm, True)
B = evaluate("B. DiffMAH+c200c | het(DiffMAH)", X_dmc, X_dm, True)
C = evaluate("C. DiffMAH+c200c | het(DiffMAH+c200c)", X_dmc, X_dmc, True)
# homoscedastic references
Ah = evaluate("   DiffMAH | homosc", X_dm, X_dm, False)
Bh = evaluate("   DiffMAH+c200c | homosc", X_dmc, X_dmc, False)

print(f"\n  mean effect of c200c (A->B): CRPS {A['crps'].mean():.4f}->{B['crps'].mean():.4f} "
      f"({100*(A['crps'].mean()-B['crps'].mean())/A['crps'].mean():+.1f}%), NLL {A['nll']:+.3f}->{B['nll']:+.3f}")
print(f"  scatter effect of c200c (B->C): NLL {B['nll']:+.3f}->{C['nll']:+.3f} "
      f"({B['nll']-C['nll']:+.3f} nats), condgap {B['gap']:.3f}->{C['gap']:.3f}")


# %% ---- does c_200c enter the log-variance? (full-data fit, model C) --------
Zall = (X_dmc - X_dmc.mean(0)) / X_dmc.std(0)
MU_C = C["MU"]
resid = Y - MU_C
print("\n[scatter model C] log-sigma slopes (per +1 sigma); is c200c a scatter driver?")
print(f"  {'aperture':8s} " + " ".join(f"{f:>7s}" for f in VAR_FEAT))
slopes = np.zeros((4, 5))
for j in range(4):
    gm = fit_logvar(resid[:, j], Zall)
    slopes[j] = 0.5 * gm[1:]
    print(f"  {TNAMES[j]:8s} " + " ".join(f"{slopes[j, i]:+7.3f}" for i in range(5)))


# %% ---- FIGURE: the c_200c emulator vs the DiffMAH emulator -----------------
fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(13.8, 4.2))
x = np.arange(4); w = 0.38
axA.bar(x - w/2, A["crps"], w, color=OKABE_ITO[7], label="DiffMAH (exp14)")
axA.bar(x + w/2, C["crps"], w, color=OKABE_ITO[2], label="DiffMAH+c200c")
axA.set_xticks(x); axA.set_xticklabels(TNAMES); axA.set_ylabel("CV CRPS [dex]")
axA.set_xlabel("aperture / annulus [kpc]"); axA.set_title("Marginal skill"); axA.legend(fontsize=8)
axA.text(-0.13, 1.04, "A", transform=axA.transAxes, fontweight="bold", fontsize=12)

lev = np.array([0.5, 0.68, 0.9, 0.95])
axB.plot([0, 1], [0, 1], ":", color="0.5", lw=1)
axB.plot(lev, A["cov"], "-o", color=OKABE_ITO[7], ms=5, label="DiffMAH")
axB.plot(lev, C["cov"], "-s", color=OKABE_ITO[2], ms=5, label="DiffMAH+c200c")
axB.set_xlim(0.4, 1.0); axB.set_ylim(0.4, 1.0)
axB.set_xlabel("nominal central interval"); axB.set_ylabel("empirical coverage")
axB.set_title("Marginal calibration"); axB.legend(fontsize=8)
axB.text(-0.13, 1.04, "B", transform=axB.transAxes, fontweight="bold", fontsize=12)

# log-sigma slopes heat-ish bar for c200c column + late, to show scatter drivers
xs = np.arange(4)
axC.axhline(0, color="k", lw=0.8)
axC.plot(xs, slopes[:, 3], "-o", color=OKABE_ITO[5], ms=6, label="late")
axC.plot(xs, slopes[:, 4], "-s", color=OKABE_ITO[2], ms=6, label="c200c")
axC.plot(xs, slopes[:, 0], "-^", color=OKABE_ITO[0], ms=5, label="logmp")
axC.set_xticks(xs); axC.set_xticklabels(TNAMES)
axC.set_xlabel("aperture / annulus [kpc]"); axC.set_ylabel("log-sigma slope (per +1σ)")
axC.set_title("What drives the scatter?"); axC.legend(fontsize=8)
axC.text(-0.13, 1.04, "C", transform=axC.transAxes, fontweight="bold", fontsize=12)
fig.suptitle(f"exp19 — DiffMAH+c_200c emulator (n={N}); CRPS {A['crps'].mean():.4f}->"
             f"{C['crps'].mean():.4f}, NLL {A['nll']:+.3f}->{C['nll']:+.3f}", fontsize=11)
fig.tight_layout()
save_fig(fig, FIGDIR / "exp19_emulator_c200c")


# %% ---- save outputs --------------------------------------------------------
OUTDIR.mkdir(parents=True, exist_ok=True)
st = Table({"model": ["A_DiffMAH", "B_mean_c200c", "C_mean+scatter_c200c",
                      "DiffMAH_homosc", "DiffMAH+c200c_homosc"],
            "crps": [A["crps"].mean(), B["crps"].mean(), C["crps"].mean(),
                     Ah["crps"].mean(), Bh["crps"].mean()],
            "nll": [A["nll"], B["nll"], C["nll"], Ah["nll"], Bh["nll"]],
            "cond_gap": [A["gap"], B["gap"], C["gap"], Ah["gap"], Bh["gap"]]})
st.write(OUTDIR / "emulator_c200c_scores.csv", overwrite=True)
write_manifest(OUTDIR, params={
    "n": int(N), "targets": TNAMES,
    "crps_diffmah": float(A["crps"].mean()), "crps_dmc": float(C["crps"].mean()),
    "nll_diffmah": float(A["nll"]), "nll_dmc": float(C["nll"]),
    "condgap_diffmah": float(A["gap"]), "condgap_dmc": float(C["gap"]),
    "mean_effect_crps_pct": float(100 * (A["crps"].mean() - B["crps"].mean()) / A["crps"].mean()),
    "scatter_effect_nll_nats": float(B["nll"] - C["nll"]),
    "logsigma_c200c_slope": slopes[:, 4].tolist()})
print(f"\nwrote figure -> {FIGDIR}\nwrote outputs -> {OUTDIR}")
print(f"\n[verdict] DiffMAH+c200c emulator: CRPS {A['crps'].mean():.4f}->{C['crps'].mean():.4f}, "
      f"NLL {A['nll']:+.3f}->{C['nll']:+.3f}, condgap {A['gap']:.3f}->{C['gap']:.3f}; "
      f"c200c scatter slope (50-100) = {slopes[3,4]:+.3f}")
