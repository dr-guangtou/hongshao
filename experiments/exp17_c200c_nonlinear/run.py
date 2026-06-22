"""exp17 — The nonlinear limit of c_200c: how much more can a flexible model get?

exp16 added halo concentration to the *linear* mean and gained +5% CRPS on
DiffMAH(4). Here we ask the ceiling question (as exp09/exp13 did for the MAH):
does a richer-than-linear model extract *more* from `c_200c` — i.e. is there a
nonlinear interaction (e.g. concentration matters more for low-mass or
late-forming halos) worth a closed-form cross-term? We compare, on the four
CoG-derived annulus masses (the primary observable), all 5-fold CV:

  - linear              -- the exp16 model
  - poly-2 (analytic)   -- all degree-2 terms (curvature + interactions)
  - GBM (ceiling)       -- flexible gradient-boosted-trees achievability ceiling
  - PySR (parsimonious) -- polynomial symbolic regression on the linear residuals,
                           to surface any single interpretable c_200c cross-term

on the portable feature set DiffMAH(4)+c_200c (and MAH-PCA(4)+c_200c for the
richest-MAH context). If GBM barely beats linear, the +5% is essentially the
limit and concentration enters linearly; if PySR finds a stable cross-term, it
is the interpretable nonlinear form.

Run (small/fast): EXP17_NMAX=500 EXP17_NITER=20 PYTHONPATH=. uv run python \
    experiments/exp17_c200c_nonlinear/run.py
Full:             PYTHONPATH=. uv run python experiments/exp17_c200c_nonlinear/run.py
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
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hongshao.tng_data import load_mah, load_cosmic_time, peak_history          # noqa: E402
from hongshao.metrics import crps_gaussian                                      # noqa: E402
from hongshao.plotting import set_style, save_fig, OKABE_ITO                     # noqa: E402
from hongshao.provenance import write_manifest                                  # noqa: E402

set_style()
HERE = Path(__file__).resolve().parent
OUTDIR, FIGDIR = HERE / "outputs", HERE / "figures"
TABLE = ROOT / "data" / "processed" / "tng300_072_z0p4.fits"
TNAMES = ["<10", "10-30", "30-50", "50-100"]
NMAX = int(os.environ.get("EXP17_NMAX", 0))
NITER = int(os.environ.get("EXP17_NITER", 50))

t = Table.read(TABLE)
t = t[t["use"]]
if NMAX:
    t = t[:NMAX]
aper = np.asarray(t["logmstar_aper"], float)
M0 = np.asarray(t["logmh_z0p4"], float)
idx = np.asarray(t["index"])
dmah = np.column_stack([np.asarray(t[c], float) for c in
                        ("dmah_logmp", "dmah_logtc", "dmah_early", "dmah_late")])
c200 = np.asarray(t["c_200c"], float)

mah = load_mah(); tsnap = load_cosmic_time(); tg = np.linspace(2.2, 9.0, 18)
ms = np.full((len(t), 18), np.nan)
for r, i in enumerate(idx):
    sn, lmp = peak_history(mah[int(i)])
    if lmp is None:
        continue
    tt = tsnap[sn.astype(int)]
    if tt[0] <= tg[0] and tt[-1] >= tg[-1]:
        ms[r] = np.interp(tg, tt, lmp) - M0[r]


def _annulus(a_o, a_i):
    return np.log10(np.clip(10.0 ** a_o - 10.0 ** a_i, 1.0, None))


Y = np.column_stack([aper[:, 0], _annulus(aper[:, 1], aper[:, 0]),
                     _annulus(aper[:, 2], aper[:, 1]), _annulus(aper[:, 4], aper[:, 2])])
g = (np.isfinite(Y).all(1) & np.isfinite(M0) & np.isfinite(dmah).all(1)
     & np.isfinite(c200) & np.isfinite(ms).all(1))
Y = Y[g]; M0 = M0[g]; dmah = dmah[g]; c200 = c200[g]; ms = ms[g]; N = len(Y)
mu_m = ms.mean(0); _, _, Vt = np.linalg.svd(ms - mu_m, full_matrices=False)
pca = (ms - mu_m) @ Vt[:4].T
print(f"exp17: nonlinear limit of c_200c on n={N}")

FEAT_DMC = ["logmp", "logtc", "early", "late", "c200c"]
X_dm = dmah
X_dmc = np.column_stack([dmah, c200])
X_pcac = np.column_stack([M0, pca, c200])


# %% ---- CV: per-aperture mean (any sklearn model) + homoscedastic Gaussian --
def cv_score(make_model, X, k=5, seed=0):
    order = np.random.default_rng(seed).permutation(N)
    pred = np.full((N, 4), np.nan); sig = np.full((N, 4), np.nan)
    for fold in np.array_split(order, k):
        tr = np.setdiff1d(np.arange(N), fold)
        for j in range(4):
            m = make_model(); m.fit(X[tr], Y[tr, j])
            pred[fold, j] = m.predict(X[fold])
            sig[fold, j] = (Y[tr, j] - m.predict(X[tr])).std()
    return crps_gaussian(Y, pred, sig).mean(0)


def gbm():
    return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=400,
                                         min_samples_leaf=40, l2_regularization=1.0,
                                         random_state=0)


def poly2():
    return make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False),
                         LinearRegression())


print(f"\n{'model / features':28s} {'CRPS':>7s}   per-aperture")
scores = {}
runs = [("linear  (DiffMAH)", LinearRegression, X_dm),
        ("linear  (DiffMAH+c200c)", LinearRegression, X_dmc),
        ("poly-2  (DiffMAH+c200c)", poly2, X_dmc),
        ("GBM     (DiffMAH+c200c)", gbm, X_dmc),
        ("linear  (MAH-PCA+c200c)", LinearRegression, X_pcac),
        ("GBM     (MAH-PCA+c200c)", gbm, X_pcac)]
for name, mk, X in runs:
    c = cv_score(mk, X)
    scores[name] = c
    print(f"  {name:28s} {c.mean():7.4f}   [" + " ".join(f"{v:.3f}" for v in c) + "]")

lin_dmc = scores["linear  (DiffMAH+c200c)"].mean()
gbm_dmc = scores["GBM     (DiffMAH+c200c)"].mean()
print(f"\n[ceiling] DiffMAH+c200c: linear={lin_dmc:.4f}  GBM={gbm_dmc:.4f}  "
      f"flexible beats linear by {100*(lin_dmc-gbm_dmc)/lin_dmc:+.1f}%")


# %% ---- PySR: parsimonious nonlinear correction on the linear residuals ------
# (exp12 recipe: residuals of the linear DiffMAH+c200c model are orthogonal to
# the linear features, so PySR surfaces only the nonlinear structure left.)
xsym = sp.symbols("x0 x1 x2 x3 x4")
Xs = X_dmc / X_dmc.std(0)
A_lin = np.column_stack([np.ones(N), Xs])
beta_lin = np.linalg.lstsq(A_lin, Y, rcond=None)[0]
resid_lin = Y - A_lin @ beta_lin


def new_pysr(niter):
    from pysr import PySRRegressor
    return PySRRegressor(
        niterations=niter, maxsize=16,
        binary_operators=["+", "-", "*"], unary_operators=["square"],
        nested_constraints={"square": {"square": 0}}, model_selection="best",
        elementwise_loss="L2DistLoss()", progress=False, verbosity=0,
        deterministic=True, parallelism="serial", random_state=0,
        output_directory=tempfile.mkdtemp(prefix="pysr_exp17_"))


def monoms_of(expr):
    return sorted({m for m in sp.Poly(sp.expand(expr), *xsym).monoms() if any(m)})


def label(m):
    return "*".join(FEAT_DMC[i] if e == 1 else f"{FEAT_DMC[i]}^{e}"
                    for i, e in enumerate(m) if e) or "1"


LINEAR = [tuple(int(i == k) for i in range(5)) for k in range(5)]
print("\n[PySR] nonlinear correction PySR finds in the DiffMAH+c200c residuals:")
corr = {}
for j, tn in enumerate(TNAMES):
    m = new_pysr(NITER)
    m.fit(Xs, resid_lin[:, j], variable_names=[f"x{i}" for i in range(5)])
    expr = sp.sympify(str(m.sympy()), locals=dict(zip([f"x{i}" for i in range(5)], xsym)))
    corr[tn] = [mm for mm in monoms_of(expr) if mm not in LINEAR]
    print(f"  {tn:>7s}: nonlinear terms = {[label(mm) for mm in corr[tn]] or ['(none)']}")

extra = sorted({mm for tn in TNAMES for mm in corr[tn]})
involve_c = [mm for mm in extra if mm[4] > 0]
print(f"  -> distinct extra terms: {[label(mm) for mm in extra] or ['(none)']}; "
      f"involving c200c: {[label(mm) for mm in involve_c] or ['(none)']}")

# evaluate the augmented (linear + PySR correction) model
def design(monoms):
    cols = [np.prod([Xs[:, i] ** e for i, e in enumerate(mm)], axis=0) for mm in monoms]
    return np.column_stack(cols) if cols else np.zeros((N, 0))


aug = []
for j, tn in enumerate(TNAMES):
    aug.append(design(LINEAR + corr[tn]))


def cv_design(designs, k=5, seed=0):
    order = np.random.default_rng(seed).permutation(N)
    pred = np.full((N, 4), np.nan); sig = np.full((N, 4), np.nan)
    Ad = [np.column_stack([np.ones(N), d]) for d in designs]
    for fold in np.array_split(order, k):
        tr = np.setdiff1d(np.arange(N), fold)
        for j in range(4):
            beta, *_ = np.linalg.lstsq(Ad[j][tr], Y[tr, j], rcond=None)
            pred[fold, j] = Ad[j][fold] @ beta
            sig[fold, j] = (Y[tr, j] - Ad[j][tr] @ beta).std()
    return crps_gaussian(Y, pred, sig).mean(0)


crps_pysr = cv_design(aug)
scores["PySR    (DiffMAH+c200c)"] = crps_pysr
print(f"  PySR augmented CRPS = {crps_pysr.mean():.4f} "
      f"({100*(lin_dmc-crps_pysr.mean())/lin_dmc:+.1f}% vs linear DiffMAH+c200c)")


# %% ---- FIGURE: the ceiling ------------------------------------------------
names = ["linear  (DiffMAH)", "linear  (DiffMAH+c200c)", "poly-2  (DiffMAH+c200c)",
         "PySR    (DiffMAH+c200c)", "GBM     (DiffMAH+c200c)",
         "linear  (MAH-PCA+c200c)", "GBM     (MAH-PCA+c200c)"]
short = ["lin\nDiffMAH", "lin\n+c200c", "poly2\n+c200c", "PySR\n+c200c",
         "GBM\n+c200c", "lin\nPCA+c200c", "GBM\nPCA+c200c"]
cols = [OKABE_ITO[0], OKABE_ITO[2], OKABE_ITO[4], OKABE_ITO[5], OKABE_ITO[1],
        OKABE_ITO[3], OKABE_ITO[6]]
ov = [scores[n].mean() for n in names]
fig, ax = plt.subplots(figsize=(8.6, 4.4))
ax.bar(range(len(names)), ov, color=cols)
ax.axhline(scores["linear  (DiffMAH+c200c)"].mean(), ls="--", color=OKABE_ITO[2], lw=1)
for i in range(len(names)):
    ax.annotate(f"{ov[i]:.4f}", (i, ov[i]), textcoords="offset points",
                xytext=(0, 3), ha="center", fontsize=7)
ax.set_xticks(range(len(names))); ax.set_xticklabels(short, fontsize=7)
ax.set_ylabel("overall CV CRPS [dex]")
ax.set_ylim(0.07, max(ov) * 1.04)
ax.set_title(f"exp17 — nonlinear limit of c_200c (n={N}); "
             f"GBM vs linear (DiffMAH+c200c): {100*(lin_dmc-gbm_dmc)/lin_dmc:+.1f}%")
fig.tight_layout()
save_fig(fig, FIGDIR / "exp17_ceiling")


# %% ---- save outputs --------------------------------------------------------
OUTDIR.mkdir(parents=True, exist_ok=True)
st = Table({"model": names, "crps": [float(scores[n].mean()) for n in names]})
st.write(OUTDIR / "ceiling_scores.csv", overwrite=True)
write_manifest(OUTDIR, params={
    "n": int(N), "targets": TNAMES,
    "crps_linear_dmc": float(lin_dmc), "crps_gbm_dmc": float(gbm_dmc),
    "crps_poly2_dmc": float(scores["poly-2  (DiffMAH+c200c)"].mean()),
    "crps_pysr_dmc": float(crps_pysr.mean()),
    "gbm_vs_linear_pct": float(100 * (lin_dmc - gbm_dmc) / lin_dmc),
    "pysr_c200c_terms": [label(mm) for mm in involve_c]})
print(f"\nwrote figure -> {FIGDIR}\nwrote outputs -> {OUTDIR}")
ceiling = 100 * (lin_dmc - gbm_dmc) / lin_dmc
print(f"\n[verdict] c_200c enters {'≈linearly (flexible barely beats linear: %+.1f%%)' % ceiling if ceiling < 2 else 'with nonlinear structure (%+.1f%%)' % ceiling}; "
      f"limit (GBM, MAH-PCA+c200c) = {scores['GBM     (MAH-PCA+c200c)'].mean():.4f}")
