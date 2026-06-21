"""exp12 — Symbolic regression for a parsimonious nonlinear SHMR mean (PySR).

exp09 showed a flexible gradient-boosted-trees ceiling does NOT beat the linear
mean on the aperture masses, so the predictable M*(annulus | MAH) relation is
essentially linear and the closed-form linear emulator (exp11) is at the
ceiling. The remaining question is *interpretive*, not about raw accuracy: is
there a single, parsimonious nonlinear term (e.g. a cross-term logmp x early)
that a human would want in the closed-form equation, and does it buy any
measurable skill *on top of the full linear model*?

We use PySR (symbolic regression), restricted to {+, -, *, square} so every
equation is a polynomial -- i.e. PySR performs sparse selection over polynomial
cross-terms (a parsimonious cousin of exp09's dense poly-2). Two angles:

  (1) DISCOVERY ON RESIDUALS (the sharp test). Fit the full 4-term linear model,
      then run PySR on its residuals vs the four DiffMAH params. OLS residuals
      are orthogonal to the linear features, so PySR can only surface *nonlinear*
      structure the linear model misses -- or nothing (a symbolic restatement of
      exp09's linearity).
  (2) DISCOVERY ON THE FULL TARGET (the literal closed form). Run PySR on M*
      itself, per aperture, for the human-readable equation + Pareto front.

EVALUATION (all 5-fold CV, the exact exp11 conditional-Gaussian emulator: linear
mean + full residual covariance, scored with the exp07 suite):
  - linear (4 DiffMAH terms)                 -- baseline
  - linear + PySR nonlinear correction       -- the parsimonious augmented model
  - poly-2 (all 14 degree-2 terms)           -- dense reference (exp09)
Every model keeps the four linear terms, so the comparison isolates the value of
the *added* nonlinear term rather than PySR's feature pruning.

Baseline to beat (exp11, DiffMAH features): overall CRPS 0.0882; per-aperture
0.069 / 0.087 / 0.097 / 0.099 dex for <10 / 10-30 / 30-50 / 50-100 kpc.

Run from the repo root (small/fast validation pass):
    EXP12_NMAX=400 EXP12_NITER=20 PYTHONPATH=. uv run python \
        experiments/exp12_symbolic_regression/run.py
Full pass (all halos, longer search):
    PYTHONPATH=. uv run python experiments/exp12_symbolic_regression/run.py
"""
# %% setup
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import sympy as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.table import Table

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hongshao.metrics import crps_gaussian, gaussian_logscore, interval_coverage  # noqa: E402
from hongshao.plotting import set_style, save_fig, OKABE_ITO                       # noqa: E402
from hongshao.provenance import write_manifest                                     # noqa: E402

set_style()
HERE = Path(__file__).resolve().parent
OUTDIR, FIGDIR = HERE / "outputs", HERE / "figures"
TABLE = ROOT / "data" / "processed" / "tng300_072_z0p4.fits"
TNAMES = ["<10", "10-30", "30-50", "50-100"]
FEAT = ["logmp", "logtc", "early", "late"]          # the four DiffMAH params (x0..x3)

NMAX = int(os.environ.get("EXP12_NMAX", 0))
NITER = int(os.environ.get("EXP12_NITER", 80))
MAXSIZE = int(os.environ.get("EXP12_MAXSIZE", 20))

t = Table.read(TABLE)
t = t[t["use"]]
if NMAX:
    t = t[:NMAX]
aper = np.asarray(t["logmstar_aper"], float)
z50 = np.asarray(t["z50"], float)
dmah = np.column_stack([np.asarray(t[c], float) for c in
                        ("dmah_logmp", "dmah_logtc", "dmah_early", "dmah_late")])


def _annulus(a_outer, a_inner):
    return np.log10(np.clip(10.0 ** a_outer - 10.0 ** a_inner, 1.0, None))


Y = np.column_stack([aper[:, 0], _annulus(aper[:, 1], aper[:, 0]),
                     _annulus(aper[:, 2], aper[:, 1]), _annulus(aper[:, 4], aper[:, 2])])
g = np.isfinite(Y).all(1) & np.isfinite(dmah).all(1) & np.isfinite(z50)
Y = Y[g]; X = dmah[g]; z50 = z50[g]; N = len(Y)
# scale features to unit std (no centering, so monomial structure is preserved);
# this only conditions the PySR search -- CRPS is invariant to it after refit.
Xs = X / X.std(0)
print(f"exp12: symbolic regression on n={N} galaxies  (niter={NITER}, maxsize={MAXSIZE})")


# %% ---- shared evaluation machinery (matches exp11) -------------------------
def cv_emulator(designs, k=5, seed=0):
    """Conditional-Gaussian emulator under 5-fold CV.

    ``designs`` is a list of 4 feature matrices (one per target), each WITHOUT
    an intercept column (added here). Per fold: per-target OLS mean + a single
    4x4 residual covariance from the training residuals (full, correlated
    scatter -- exactly the exp11 model). Returns out-of-fold (mu, Sig)."""
    order = np.random.default_rng(seed).permutation(N)
    mu = np.full((N, 4), np.nan)
    Sig = np.full((N, 4, 4), np.nan)
    Ad = [np.column_stack([np.ones(N), d]) for d in designs]
    for fold in np.array_split(order, k):
        tr = np.setdiff1d(np.arange(N), fold)
        resid_tr = np.empty((len(tr), 4))
        for j in range(4):
            beta, *_ = np.linalg.lstsq(Ad[j][tr], Y[tr, j], rcond=None)
            mu[fold, j] = Ad[j][fold] @ beta
            resid_tr[:, j] = Y[tr, j] - Ad[j][tr] @ beta
        Sig[fold] = np.cov(resid_tr.T)
    return mu, Sig


def marginal(mu, Sig):
    sig = np.sqrt(np.stack([np.diag(s) for s in Sig]))
    return (crps_gaussian(Y, mu, sig).mean(0), gaussian_logscore(Y, mu, sig).mean(0),
            interval_coverage(Y, mu, sig))


def joint_nll(mu, Sig, diagonal=False):
    resid = Y - mu
    out = np.empty(N)
    for i in range(N):
        S = np.diag(np.diag(Sig[i])) if diagonal else Sig[i]
        _, logdet = np.linalg.slogdet(S)
        out[i] = 0.5 * (4 * np.log(2 * np.pi) + logdet
                        + resid[i] @ np.linalg.solve(S, resid[i]))
    return float(np.mean(out))


def evaluate(name, designs):
    mu, Sig = cv_emulator(designs)
    crps, ls, (lev, cov) = marginal(mu, Sig)
    print(f"  {name:26s} CRPS={crps.mean():.4f}  ["
          + " ".join(f"{c:.3f}" for c in crps) + "]  "
          + "cov(50/68/90/95)=" + "/".join(f"{c:.2f}" for c in cov))
    return dict(crps=crps, ls=ls, cov=cov, mu=mu, Sig=Sig, lev=lev,
                nll=joint_nll(mu, Sig))


# %% ---- monomial helpers ----------------------------------------------------
xsym = sp.symbols("x0 x1 x2 x3")
XLOCALS = dict(zip([f"x{i}" for i in range(4)], xsym))


def to_sympy(model):
    return sp.sympify(str(model.sympy()), locals=XLOCALS)


def monomials_of(expr):
    """Distinct non-constant monomials (exponent tuples) of a polynomial expr."""
    poly = sp.Poly(sp.expand(expr), *xsym)
    return sorted({m for m in poly.monoms() if any(m)})


def design_from_monomials(monoms):
    cols = [np.prod([Xs[:, i] ** e for i, e in enumerate(m)], axis=0) for m in monoms]
    return np.column_stack(cols) if cols else np.zeros((N, 0))


def monom_label(m):
    parts = [FEAT[i] if e == 1 else f"{FEAT[i]}^{e}" for i, e in enumerate(m) if e]
    return "*".join(parts) if parts else "1"


LINEAR_MONOMS = [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]
# all degree-2 monomials (the dense poly-2 reference, exp09)
POLY2_MONOMS = sorted({tuple(np.eye(4, dtype=int)[i] + np.eye(4, dtype=int)[j])
                       for i in range(4) for j in range(4)} | set(LINEAR_MONOMS))


PYSR_TMP = tempfile.mkdtemp(prefix="pysr_exp12_")    # keep hall_of_fame out of the repo


def new_pysr(niter, maxsize):
    from pysr import PySRRegressor
    return PySRRegressor(
        niterations=niter, maxsize=maxsize,
        binary_operators=["+", "-", "*"], unary_operators=["square"],
        nested_constraints={"square": {"square": 0}},   # readable low-order polys
        model_selection="best", elementwise_loss="L2DistLoss()",
        progress=False, verbosity=0, deterministic=True, parallelism="serial",
        random_state=0, output_directory=PYSR_TMP)


# %% ---- baseline: linear emulator (raw DiffMAH features) --------------------
print("\n[baseline] linear emulator on the four DiffMAH features:")
res_lin = evaluate("linear (DiffMAH)", [design_from_monomials(LINEAR_MONOMS)] * 4)

# full-data linear residuals per aperture (target for residual-mode PySR)
A_lin = np.column_stack([np.ones(N), design_from_monomials(LINEAR_MONOMS)])
beta_lin = np.linalg.lstsq(A_lin, Y, rcond=None)[0]          # (5, 4)
resid_lin = Y - A_lin @ beta_lin
print("  linear-residual std per aperture: " + " ".join(f"{s:.3f}" for s in resid_lin.std(0)))


# %% ---- PySR discovery -------------------------------------------------------
print("\n[PySR, residual mode] nonlinear structure the linear model misses ...")
corr_monoms = {}      # aperture -> list of nonlinear monomials found in residuals
corr_expr = {}
for j, tn in enumerate(TNAMES):
    m = new_pysr(NITER, min(MAXSIZE, 15))
    m.fit(Xs, resid_lin[:, j], variable_names=[f"x{i}" for i in range(4)])
    expr = to_sympy(m)
    corr_expr[tn] = expr
    # keep only the *nonlinear* monomials (linear/const are ~0 in OLS residuals)
    corr_monoms[tn] = [mm for mm in monomials_of(expr) if mm not in LINEAR_MONOMS]
    lab = [monom_label(mm) for mm in corr_monoms[tn]] or ["(none)"]
    print(f"  {tn:>7s} kpc residual ~ {sp.expand(expr)}   -> nonlinear: {lab}")

print("\n[PySR, full-target mode] readable closed-form equation per aperture ...")
full_expr = {}
fronts = {}
for j, tn in enumerate(TNAMES):
    m = new_pysr(NITER, MAXSIZE)
    m.fit(Xs, Y[:, j], variable_names=[f"x{i}" for i in range(4)])
    full_expr[tn] = to_sympy(m)
    fronts[tn] = m.equations_[["complexity", "loss", "equation"]].copy()
    print(f"  {tn:>7s} kpc:  {sp.expand(full_expr[tn])}")


# %% ---- evaluate the augmented (linear + correction) model ------------------
# headline: keep the 4 linear terms, add the residual-discovered nonlinear terms.
aug_designs = []
for j, tn in enumerate(TNAMES):
    monoms = LINEAR_MONOMS + corr_monoms[tn]
    aug_designs.append(design_from_monomials(monoms))

extra_all = sorted({mm for tn in TNAMES for mm in corr_monoms[tn]})
print("\n[discovered nonlinear corrections] across apertures:")
if extra_all:
    for mm in extra_all:
        used = [tn for tn in TNAMES if mm in corr_monoms[tn]]
        print(f"  {monom_label(mm):20s} used in {used}")
else:
    print("  (none -- PySR finds no nonlinear structure in the linear residuals;")
    print("   a symbolic restatement of exp09: the relation is linear.)")

print("\n[evaluation] linear vs parsimonious-nonlinear vs dense poly-2:")
res_aug = evaluate("linear + PySR correction", aug_designs)
res_poly2 = evaluate("poly-2 (dense, 14 terms)", [design_from_monomials(POLY2_MONOMS)] * 4)

d_aug = 100 * (res_lin["crps"].mean() - res_aug["crps"].mean()) / res_lin["crps"].mean()
d_poly = 100 * (res_lin["crps"].mean() - res_poly2["crps"].mean()) / res_lin["crps"].mean()


# %% ---- stability: per-fold coefficient of each correction term -------------
def coef_stability(monoms, j, k=5, seed=0):
    A = np.column_stack([np.ones(N), design_from_monomials(monoms)])
    order = np.random.default_rng(seed).permutation(N)
    betas = [np.linalg.lstsq(A[np.setdiff1d(np.arange(N), fold)], Y[np.setdiff1d(np.arange(N), fold), j],
                             rcond=None)[0] for fold in np.array_split(order, k)]
    return np.array(betas)


if extra_all:
    print("\n[stability] per-fold coefficient of each correction term:")
    for j, tn in enumerate(TNAMES):
        if not corr_monoms[tn]:
            continue
        monoms = LINEAR_MONOMS + corr_monoms[tn]
        B = coef_stability(monoms, j)
        for col, mm in enumerate(monoms, start=1):
            if mm in extra_all:
                v = B[:, col]
                sign = "+" if (v > 0).all() else ("-" if (v < 0).all() else "MIXED")
                print(f"  {tn:>7s}: {monom_label(mm):16s} mean={v.mean():+.3e} "
                      f"std={v.std():.1e} signs={sign}")


# %% ---- FIGURE 1: skill + calibration ---------------------------------------
models = [("linear", res_lin, OKABE_ITO[7]),
          ("linear + PySR corr.", res_aug, OKABE_ITO[2]),
          ("poly-2 (dense)", res_poly2, OKABE_ITO[4])]
fig1, (axA, axB) = plt.subplots(1, 2, figsize=(9.8, 4.0))
x = np.arange(4); w = 0.26
for k_i, (name, r, c) in enumerate(models):
    axA.bar(x + (k_i - 1) * w, r["crps"], w, color=c, label=name)
axA.set_xticks(x); axA.set_xticklabels(TNAMES)
axA.set_xlabel("aperture / annulus [kpc]")
axA.set_ylabel("cross-validated CRPS [dex]  (lower = better)")
axA.set_title("Predictive skill")
axA.legend(fontsize=7)
axA.text(-0.13, 1.04, "A", transform=axA.transAxes, fontweight="bold", fontsize=12)

axB.plot([0, 1], [0, 1], ":", color="0.5", lw=1)
for name, r, c in models:
    axB.plot(res_lin["lev"], r["cov"], "-o", color=c, ms=4, label=name)
axB.set_xlabel("nominal central interval"); axB.set_ylabel("empirical coverage")
axB.set_title("Calibration"); axB.set_xlim(0.4, 1.0); axB.set_ylim(0.4, 1.0)
axB.legend(fontsize=7)
axB.text(-0.13, 1.04, "B", transform=axB.transAxes, fontweight="bold", fontsize=12)
fig1.suptitle(f"exp12 — parsimonious nonlinear vs linear (n={N}); CRPS "
              f"{res_aug['crps'].mean():.4f} vs {res_lin['crps'].mean():.4f} "
              f"({d_aug:+.1f}%), poly-2 {d_poly:+.1f}%", fontsize=10)
fig1.tight_layout()
save_fig(fig1, FIGDIR / "exp12_skill_calibration")


# %% ---- FIGURE 2: Pareto fronts (full-target PySR) --------------------------
fig2, ax = plt.subplots(figsize=(6.4, 4.4))
for j, tn in enumerate(TNAMES):
    fr = fronts[tn]
    ax.plot(fr["complexity"], fr["loss"], "-o", color=OKABE_ITO[j], ms=4, label=f"{tn} kpc")
ax.set_yscale("log")
ax.set_xlabel("equation complexity (nodes)")
ax.set_ylabel("training loss (MSE)")
ax.set_title("PySR Pareto front per aperture (full target)")
ax.legend(fontsize=8)
fig2.tight_layout()
save_fig(fig2, FIGDIR / "exp12_pareto")


# %% ---- FIGURE 3: visualize the linear residual vs each feature -------------
# AGENTS.md mandate: don't trust metrics alone -- show whether nonlinear
# structure actually exists. Binned-median residual vs each DiffMAH param; a
# flat line means the linear model is adequate (no missed nonlinearity).
def _binned(xv, yv, nbin=12):
    edges = np.quantile(xv, np.linspace(0, 1, nbin + 1))
    cen, med, err = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (xv >= a) & (xv <= b)
        if sel.sum() >= 12:
            cen.append(np.median(xv[sel])); med.append(np.median(yv[sel]))
            err.append(np.std(yv[sel]) / np.sqrt(sel.sum()))
    return map(np.asarray, (cen, med, err))


fig3, ax3 = plt.subplots(4, 4, figsize=(13.5, 11.0), sharex="col")
for j, tn in enumerate(TNAMES):
    for i, fn in enumerate(FEAT):
        a = ax3[j, i]
        cen, med, err = _binned(X[:, i], resid_lin[:, j])
        a.axhline(0, color="0.6", lw=0.8)
        a.errorbar(cen, med, yerr=err, fmt="o-", color=OKABE_ITO[j], ms=4, lw=1.3)
        a.set_ylim(-0.05, 0.05)
        if j == 0:
            a.set_title(fn)
        if i == 0:
            a.set_ylabel(f"{tn} kpc\nresidual [dex]", fontsize=9)
        if j == 3:
            a.set_xlabel(fn)
fig3.suptitle("exp12 — linear-model residual vs each DiffMAH feature "
              "(flat = linear adequate; curvature = missed nonlinearity)", fontsize=11)
fig3.tight_layout()
save_fig(fig3, FIGDIR / "exp12_residual_structure")


# %% ---- save outputs --------------------------------------------------------
OUTDIR.mkdir(parents=True, exist_ok=True)
eq_tbl = Table({
    "aperture": TNAMES,
    "full_equation": [str(sp.expand(full_expr[tn])) for tn in TNAMES],
    "residual_equation": [str(sp.expand(corr_expr[tn])) for tn in TNAMES],
    "correction_terms": [", ".join(monom_label(mm) for mm in corr_monoms[tn]) or "(none)"
                         for tn in TNAMES],
    "crps_linear": [float(res_lin["crps"][j]) for j in range(4)],
    "crps_augmented": [float(res_aug["crps"][j]) for j in range(4)],
    "crps_poly2": [float(res_poly2["crps"][j]) for j in range(4)],
})
eq_tbl.write(OUTDIR / "equations.csv", overwrite=True)
for tn in TNAMES:
    fronts[tn].to_csv(OUTDIR / f"pareto_{tn.replace('<', 'lt').replace('-', '_')}.csv", index=False)

write_manifest(OUTDIR, params={
    "n": int(N), "niter": NITER, "maxsize": MAXSIZE, "targets": TNAMES, "features": FEAT,
    "crps_linear": float(res_lin["crps"].mean()),
    "crps_augmented": float(res_aug["crps"].mean()),
    "crps_poly2": float(res_poly2["crps"].mean()),
    "augmented_gain_pct": float(d_aug), "poly2_gain_pct": float(d_poly),
    "correction_terms": [monom_label(mm) for mm in extra_all],
})
print(f"\nwrote figures -> {FIGDIR}\nwrote outputs -> {OUTDIR}")
print(f"\n[verdict] CRPS  linear={res_lin['crps'].mean():.4f}  "
      f"linear+correction={res_aug['crps'].mean():.4f} ({d_aug:+.1f}%)  "
      f"poly-2={res_poly2['crps'].mean():.4f} ({d_poly:+.1f}%)")
