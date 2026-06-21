"""exp13 — How well can the MAH predict the 50-100 kpc outskirt mass? (limit probe)

A focused follow-up to exp12. The outer annulus (50-100 kpc) carries the most
scatter (low surface brightness, projection, ICL) and is where the mild
nonlinearity lives. Here we ask a single question: *what is the upper limit of
predictability for M*[50-100 kpc] from the halo's assembly history?* — and does
a richer MAH representation beat the four portable DiffMAH params?

This DELIBERATELY breaks the portability principle (MAH-PCA and the raw MAH
vector are defined on the TNG sample, so they don't transfer). It is a ceiling
probe, not a portable model: we want to know whether the residual outskirt
scatter is *feature-limited* (more MAH info would help) or *intrinsic* (the
scatter is projection/ICL noise we cannot remove from this dataset).

We cross feature richness x model class on the single 50-100 kpc target:
  features:  M0 only | DiffMAH(4) | M0+MAH-PCA(4) | M0+MAH-PCA(8) | raw MAH(18)
  models:    linear  | PySR (polynomial SR)       | GBM (flexible ceiling, exp09)
  control:   GBM on M0-binned-shuffled raw MAH  -> must collapse to ~M0-only.
Scored by the exp07 suite (CV CRPS) plus RMS and variance explained (R^2).

Run from the repo root (small/fast validation pass):
    EXP13_NMAX=500 EXP13_NITER=20 PYTHONPATH=. uv run python \
        experiments/exp13_outskirt_limit/run.py
Full pass:
    PYTHONPATH=. uv run python experiments/exp13_outskirt_limit/run.py
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
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hongshao.tng_data import load_mah, load_cosmic_time, peak_history          # noqa: E402
from hongshao.metrics import crps_gaussian                                      # noqa: E402
from hongshao.plotting import set_style, save_fig, OKABE_ITO                    # noqa: E402
from hongshao.provenance import write_manifest                                  # noqa: E402

set_style()
HERE = Path(__file__).resolve().parent
OUTDIR, FIGDIR = HERE / "outputs", HERE / "figures"
TABLE = ROOT / "data" / "processed" / "tng300_072_z0p4.fits"

NMAX = int(os.environ.get("EXP13_NMAX", 0))
NITER = int(os.environ.get("EXP13_NITER", 50))

t = Table.read(TABLE)
t = t[t["use"]]
if NMAX:
    t = t[:NMAX]
logM0 = np.asarray(t["logm0_halo"], float)
aper = np.asarray(t["logmstar_aper"], float)
idx = np.asarray(t["index"])
dmah = np.column_stack([np.asarray(t[c], float) for c in
                        ("dmah_logmp", "dmah_logtc", "dmah_early", "dmah_late")])


def _annulus(a_outer, a_inner):
    return np.log10(np.clip(10.0 ** a_outer - 10.0 ** a_inner, 1.0, None))


target = _annulus(aper[:, 4], aper[:, 2])      # M*[50-100 kpc]

# raw + M0-normalized log-MAH on a fixed cosmic-time grid (exp06/09/11 construction)
mah = load_mah(); tsnap = load_cosmic_time(); tg = np.linspace(2.2, 9.0, 18)
ms_raw = np.full((len(t), 18), np.nan)
for r, i in enumerate(idx):
    sn, lmp = peak_history(mah[int(i)])
    if lmp is None:
        continue
    tt = tsnap[sn.astype(int)]
    if tt[0] <= tg[0] and tt[-1] >= tg[-1]:
        ms_raw[r] = np.interp(tg, tt, lmp)      # un-normalized log M_halo(t)

g = (np.isfinite(target) & np.isfinite(logM0) & np.isfinite(dmah).all(1)
     & np.isfinite(ms_raw).all(1))
y = target[g]; M0 = logM0[g]; dmahg = dmah[g]; raw = ms_raw[g]; N = len(y)
shape = raw - M0[:, None]                       # M0-normalized MAH shape
mu_s = shape.mean(0); _, _, Vt = np.linalg.svd(shape - mu_s, full_matrices=False)
pcs = (shape - mu_s) @ Vt.T                     # all PCA modes (use first k)
var_tot = float(y.var())
print(f"exp13: M*[50-100] limit probe on n={N} galaxies (target std = {np.sqrt(var_tot):.3f} dex)")

FEATURES = {
    "M0 only": M0[:, None],
    "DiffMAH (4)": dmahg,
    "M0 + MAH-PCA(4)": np.column_stack([M0, pcs[:, :4]]),
    "M0 + MAH-PCA(8)": np.column_stack([M0, pcs[:, :8]]),
    "raw MAH (18)": raw,
}


# %% ---- CV scoring (homoscedastic Gaussian predictive; matches exp07/09) ----
def cv_score(make_model, X, k=5, seed=0):
    order = np.random.default_rng(seed).permutation(N)
    pred = np.full(N, np.nan); sig = np.full(N, np.nan)
    for fold in np.array_split(order, k):
        tr = np.setdiff1d(np.arange(N), fold)
        m = make_model(); m.fit(X[tr], y[tr])
        pred[fold] = m.predict(X[fold])
        sig[fold] = (y[tr] - m.predict(X[tr])).std()
    rms = float(np.sqrt(np.mean((y - pred) ** 2)))
    crps = float(crps_gaussian(y, pred, sig).mean())
    r2 = float(1.0 - np.mean((y - pred) ** 2) / var_tot)
    return dict(rms=rms, crps=crps, r2=r2, pred=pred)


def gbm():
    return HistGradientBoostingRegressor(
        max_depth=3, learning_rate=0.05, max_iter=400,
        min_samples_leaf=40, l2_regularization=1.0, random_state=0)


print(f"\n{'features':18s} {'model':8s} {'CRPS':>7s} {'RMS':>7s} {'R^2':>7s}")
results = {}
for name, X in FEATURES.items():
    for mlabel, mk in [("linear", LinearRegression), ("GBM", gbm)]:
        r = cv_score(mk, X)
        results[(name, mlabel)] = r
        print(f"  {name:16s} {mlabel:8s} {r['crps']:7.4f} {r['rms']:7.4f} {r['r2']:7.3f}")

# shuffle control: scramble the raw MAH within M0 bins, GBM -> should collapse
rng = np.random.default_rng(1)
edges = np.quantile(M0, np.linspace(0, 1, 13))
b = np.digitize(M0, edges[1:-1])
raw_shuf = raw.copy()
for bi in np.unique(b):
    ii = np.where(b == bi)[0]
    raw_shuf[ii] = raw[ii][rng.permutation(len(ii))]
res_shuf = cv_score(gbm, raw_shuf)
print(f"  {'raw MAH (shuffled)':16s} {'GBM':8s} {res_shuf['crps']:7.4f} "
      f"{res_shuf['rms']:7.4f} {res_shuf['r2']:7.3f}   (control: should ~ M0-only)")


# %% ---- PySR on each feature family: which equation, what skill? ------------
xall = sp.symbols(" ".join(f"x{i}" for i in range(18)))


def pysr_discover(X, niter, maxsize, label):
    """Fit PySR (polynomial ops) to y from features X; return (equation_str,
    cv_score_dict) where the score refits the discovered monomials under CV
    (same homoscedastic-Gaussian framework as the table)."""
    from pysr import PySRRegressor
    p = X.shape[1]
    Xs = X / X.std(0)
    names = [f"x{i}" for i in range(p)]
    m = PySRRegressor(
        niterations=niter, maxsize=maxsize,
        binary_operators=["+", "-", "*"], unary_operators=["square"],
        nested_constraints={"square": {"square": 0}},
        model_selection="best", elementwise_loss="L2DistLoss()",
        progress=False, verbosity=0, deterministic=True, parallelism="serial",
        random_state=0, output_directory=tempfile.mkdtemp(prefix="pysr_exp13_"))
    m.fit(Xs, y, variable_names=names)
    expr = sp.sympify(str(m.sympy()), locals=dict(zip(names, xall[:p])))
    poly = sp.Poly(sp.expand(expr), *xall[:p])
    monoms = sorted({mm for mm in poly.monoms() if any(mm)})
    design = np.column_stack([np.prod([Xs[:, i] ** e for i, e in enumerate(mm)], axis=0)
                              for mm in monoms]) if monoms else np.zeros((N, 1))
    score = cv_score(LinearRegression, design)
    return sp.expand(expr), monoms, score


def name_terms(monoms, feat_names):
    out = []
    for mm in monoms:
        parts = [feat_names[i] if e == 1 else f"{feat_names[i]}^{e}"
                 for i, e in enumerate(mm) if e]
        out.append("*".join(parts) if parts else "1")
    return out


print("\n[PySR] compact polynomial equations (deviates from portability for PCA/raw):")
pysr_runs = {
    "DiffMAH (4)": (dmahg, ["logmp", "logtc", "early", "late"], 18),
    "M0 + MAH-PCA(8)": (np.column_stack([M0, pcs[:, :8]]),
                        ["M0"] + [f"PC{i+1}" for i in range(8)], 22),
    "raw MAH (18)": (raw, [f"t{i+1}" for i in range(18)], 24),
}
pysr_results = {}
for name, (X, fnames, maxsize) in pysr_runs.items():
    expr, monoms, score = pysr_discover(X, NITER, maxsize, name)
    pysr_results[name] = dict(expr=expr, terms=name_terms(monoms, fnames), score=score)
    print(f"  {name:16s} CRPS={score['crps']:.4f} R^2={score['r2']:.3f}  "
          f"terms={pysr_results[name]['terms']}")


# %% ---- the limit, in words --------------------------------------------------
best = min(results.items(), key=lambda kv: kv[1]["crps"])
floor = results[("M0 only", "linear")]
dmah_lin = results[("DiffMAH (4)", "linear")]
print(f"\n[limit] floor (M0 only, linear):      CRPS={floor['crps']:.4f}  R^2={floor['r2']:.3f}")
print(f"        DiffMAH(4), linear:           CRPS={dmah_lin['crps']:.4f}  R^2={dmah_lin['r2']:.3f}")
print(f"        best overall ({best[0][0]}, {best[0][1]}): "
      f"CRPS={best[1]['crps']:.4f}  R^2={best[1]['r2']:.3f}")
print(f"        shuffle control:              CRPS={res_shuf['crps']:.4f}  R^2={res_shuf['r2']:.3f}")
gain_dmah = 100 * (floor["crps"] - dmah_lin["crps"]) / floor["crps"]
gain_best = 100 * (floor["crps"] - best[1]["crps"]) / floor["crps"]
print(f"        => MAH over M0: DiffMAH {gain_dmah:+.1f}%, best-possible {gain_best:+.1f}% CRPS")


# %% ---- FIGURE 1: CRPS + R^2 by feature richness x model --------------------
order = list(FEATURES)
fig1, (axA, axB) = plt.subplots(1, 2, figsize=(11.0, 4.2))
xpos = np.arange(len(order)); w = 0.38
for k_i, (mlabel, c) in enumerate([("linear", OKABE_ITO[0]), ("GBM", OKABE_ITO[5])]):
    axA.bar(xpos + (k_i - 0.5) * w, [results[(n, mlabel)]["crps"] for n in order],
            w, color=c, label=mlabel)
    axB.bar(xpos + (k_i - 0.5) * w, [results[(n, mlabel)]["r2"] for n in order],
            w, color=c, label=mlabel)
axA.axhline(floor["crps"], ls="--", color="0.5", lw=1, label="M0-only floor")
axA.axhline(res_shuf["crps"], ls=":", color=OKABE_ITO[6], lw=1.2, label="shuffle control")
axA.set_ylabel("CV CRPS [dex]  (lower = better)"); axA.set_title("Predictive skill")
axB.set_ylabel(r"variance explained $R^2$"); axB.set_title("Variance explained")
for ax in (axA, axB):
    ax.set_xticks(xpos); ax.set_xticklabels(order, rotation=20, ha="right", fontsize=8)
    ax.legend(fontsize=7)
axA.text(-0.13, 1.04, "A", transform=axA.transAxes, fontweight="bold", fontsize=12)
axB.text(-0.13, 1.04, "B", transform=axB.transAxes, fontweight="bold", fontsize=12)
fig1.suptitle(f"exp13 — limit of M*[50-100 kpc] prediction from the MAH (n={N}); "
              f"DiffMAH {gain_dmah:+.0f}%, ceiling {gain_best:+.0f}% over M0", fontsize=11)
fig1.tight_layout()
save_fig(fig1, FIGDIR / "exp13_limit_skill")


# %% ---- FIGURE 2: truth vs predicted, DiffMAH-linear vs the flexible ceiling -
def _binned(xv, yv, nbin=12):
    e = np.quantile(xv, np.linspace(0, 1, nbin + 1))
    cen, med = [], []
    for a, bb in zip(e[:-1], e[1:]):
        s = (xv >= a) & (xv <= bb)
        if s.sum() >= 12:
            cen.append(np.median(xv[s])); med.append(np.median(yv[s]))
    return np.asarray(cen), np.asarray(med)


lo, hi = np.percentile(y, [1, 99])
pred_dmah = results[("DiffMAH (4)", "linear")]["pred"]
pred_best = best[1]["pred"]
fig2, (a, b) = plt.subplots(1, 2, figsize=(10.4, 4.6))
a.plot([lo, hi], [lo, hi], "k--", lw=1.1)
for pred, c, lab in [(pred_dmah, OKABE_ITO[0], "DiffMAH (4), linear"),
                     (pred_best, OKABE_ITO[5], f"{best[0][0]}, {best[0][1]} (ceiling)")]:
    a.scatter(y, pred, s=5, alpha=0.10, color=c, edgecolors="none")
    cen, med = _binned(y, pred); a.plot(cen, med, "-", color=c, lw=2, label=lab)
a.set_xlim(lo, hi); a.set_ylim(lo, hi)
a.set_xlabel(r"true $\log M_*[50\!-\!100\,{\rm kpc}]$")
a.set_ylabel("predicted"); a.set_title("Truth vs predicted"); a.legend(fontsize=8)
a.text(-0.13, 1.04, "A", transform=a.transAxes, fontweight="bold", fontsize=12)
# residual vs truth
b.axhline(0, color="k", lw=1)
for pred, c in [(pred_dmah, OKABE_ITO[0]), (pred_best, OKABE_ITO[5])]:
    cen, med = _binned(y, pred - y); b.plot(cen, med, "-", color=c, lw=2)
    b.scatter(y, pred - y, s=5, alpha=0.07, color=c, edgecolors="none")
b.set_xlim(lo, hi); b.set_ylim(-0.6, 0.6)
b.set_xlabel(r"true $\log M_*[50\!-\!100\,{\rm kpc}]$")
b.set_ylabel("predicted $-$ true [dex]"); b.set_title("Residual")
b.text(-0.13, 1.04, "B", transform=b.transAxes, fontweight="bold", fontsize=12)
fig2.suptitle(f"exp13 — even the flexible ceiling barely tightens the outskirt "
              f"relation (RMS {results[('DiffMAH (4)','linear')]['rms']:.3f} -> "
              f"{best[1]['rms']:.3f} dex)", fontsize=11)
fig2.tight_layout()
save_fig(fig2, FIGDIR / "exp13_truth_vs_pred")


# %% ---- save outputs --------------------------------------------------------
OUTDIR.mkdir(parents=True, exist_ok=True)
rows = []
for name in order:
    for mlabel in ("linear", "GBM"):
        r = results[(name, mlabel)]
        rows.append((name, mlabel, r["crps"], r["rms"], r["r2"]))
rows.append(("raw MAH (shuffled)", "GBM", res_shuf["crps"], res_shuf["rms"], res_shuf["r2"]))
st = Table(rows=rows, names=("features", "model", "crps", "rms", "r2"))
st.write(OUTDIR / "limit_scores.csv", overwrite=True)
write_manifest(OUTDIR, params={
    "n": int(N), "niter": NITER, "target": "M*[50-100 kpc]",
    "target_std": float(np.sqrt(var_tot)),
    "crps_m0": floor["crps"], "crps_diffmah": dmah_lin["crps"],
    "crps_best": best[1]["crps"], "best_model": f"{best[0][0]} / {best[0][1]}",
    "crps_shuffle": res_shuf["crps"],
    "gain_diffmah_pct": float(gain_dmah), "gain_ceiling_pct": float(gain_best),
    "pysr_diffmah_terms": pysr_results["DiffMAH (4)"]["terms"]})
print(f"\nwrote figures -> {FIGDIR}\nwrote outputs -> {OUTDIR}")
