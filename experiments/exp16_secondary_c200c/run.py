"""exp16 — Secondary-property test: does halo concentration help beyond the MAH?

The roadmap's standing question (AGENTS.md: "test, don't assume"): do secondary
halo properties reduce the residual stellar-mass scatter once we already
condition on the assembly history? We start with halo concentration `c_200c`
(now in the dataset from the aperture table). The prior, from exp06 and the
literature, is that concentration is largely *determined* by the MAH (it tracks
formation time), so once the portable DiffMAH params are in the model it should
add little — but we measure it.

Target: the CoG-derived aperture/annulus masses (<10, 10-30, 30-50, 50-100 kpc)
— the primary observable (AGENTS.md: model the CoG masses, not `*_aper_proj`).
Feature sets, all 5-fold CV with a per-aperture linear mean + homoscedastic
Gaussian (exp07 suite: CRPS, R^2, calibration):
  - M0 only (logmh_z0p4, the exact z=0.4 mass)        -- floor
  - DiffMAH(4)                                         -- portable baseline
  - DiffMAH(4) + c_200c                                -- does concentration add?
  - M0 + c_200c                                        -- concentration's standalone power
  - DiffMAH(4) + shuffled c_200c                       -- control (must match DiffMAH)
Plus: how well DiffMAH predicts c_200c (is it MAH-determined?), and the partial
correlation of c_200c with each annulus at fixed DiffMAH (leftover info).

Run from the repo root (small/fast):
    EXP16_NMAX=600 PYTHONPATH=. uv run python experiments/exp16_secondary_c200c/run.py
Full pass:
    PYTHONPATH=. uv run python experiments/exp16_secondary_c200c/run.py
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
from hongshao.metrics import crps_gaussian, interval_coverage                   # noqa: E402
from hongshao.plotting import set_style, save_fig, OKABE_ITO                     # noqa: E402
from hongshao.provenance import write_manifest                                  # noqa: E402

set_style()
HERE = Path(__file__).resolve().parent
OUTDIR, FIGDIR = HERE / "outputs", HERE / "figures"
TABLE = ROOT / "data" / "processed" / "tng300_072_z0p4.fits"
TNAMES = ["<10", "10-30", "30-50", "50-100"]
NMAX = int(os.environ.get("EXP16_NMAX", 0))

t = Table.read(TABLE)
t = t[t["use"]]
if NMAX:
    t = t[:NMAX]
aper = np.asarray(t["logmstar_aper"], float)        # CoG-derived (xy), the primary observable
M0 = np.asarray(t["logmh_z0p4"], float)             # exact z=0.4 halo mass
idx = np.asarray(t["index"])
dmah = np.column_stack([np.asarray(t[c], float) for c in
                        ("dmah_logmp", "dmah_logtc", "dmah_early", "dmah_late")])
c200 = np.asarray(t["c_200c"], float)

# MAH-PCA(4): the richer, sample-specific MAH representation (exp11 construction).
# Used to ask whether c_200c helps even on top of the *full* MAH, not just the
# smoothed 4-param DiffMAH (exp10: DiffMAH carries ~88% of the MAH-PCA signal).
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
pca = (ms - mu_m) @ Vt[:4].T                        # MAH-PCA(4)
print(f"exp16: secondary-property test (c_200c) on n={N}")

# shuffle c_200c within M0 bins (control: destroys real concentration info)
rng = np.random.default_rng(1)
edges = np.quantile(M0, np.linspace(0, 1, 13))
b = np.digitize(M0, edges[1:-1])
c200_shuf = c200.copy()
for bi in np.unique(b):
    ii = np.where(b == bi)[0]
    c200_shuf[ii] = c200[ii][rng.permutation(len(ii))]

FEATURES = {
    "M0 only": M0[:, None],
    "M0 + c200c": np.column_stack([M0, c200]),
    "DiffMAH(4)": dmah,
    "DiffMAH + c200c": np.column_stack([dmah, c200]),
    "DiffMAH + shuf c200c": np.column_stack([dmah, c200_shuf]),
    "M0 + MAH-PCA(4)": np.column_stack([M0, pca]),
    "MAH-PCA(4) + c200c": np.column_stack([M0, pca, c200]),
}
var_tot = Y.var(0)


# %% ---- CV: per-aperture linear mean + homoscedastic Gaussian ---------------
def cv_score(X, k=5, seed=0):
    Xd = np.column_stack([np.ones(N), X])
    order = np.random.default_rng(seed).permutation(N)
    pred = np.full((N, 4), np.nan); sig = np.full((N, 4), np.nan)
    for fold in np.array_split(order, k):
        tr = np.setdiff1d(np.arange(N), fold)
        for j in range(4):
            beta, *_ = np.linalg.lstsq(Xd[tr], Y[tr, j], rcond=None)
            pred[fold, j] = Xd[fold] @ beta
            sig[fold, j] = (Y[tr, j] - Xd[tr] @ beta).std()
    crps = crps_gaussian(Y, pred, sig).mean(0)
    r2 = 1.0 - np.mean((Y - pred) ** 2, 0) / var_tot
    _, cov = interval_coverage(Y, pred, sig)
    return dict(crps=crps, r2=r2, cov=cov, pred=pred)


print(f"\n{'features':22s} {'CRPS(all)':>9s}  per-aperture CRPS            R^2(50-100)")
res = {}
for name, X in FEATURES.items():
    r = cv_score(X)
    res[name] = r
    print(f"  {name:20s} {r['crps'].mean():9.4f}  [" + " ".join(f"{c:.3f}" for c in r["crps"])
          + f"]   {r['r2'][3]:.3f}")

base = res["DiffMAH(4)"]
add = res["DiffMAH + c200c"]
gain = 100 * (base["crps"] - add["crps"]) / base["crps"]
print(f"\n[c200c on top of DiffMAH]    per-aperture CRPS gain %: "
      + " ".join(f"{x:+.1f}" for x in gain) + f"   overall {gain.mean():+.2f}%")
# the decisive comparison: does c200c still help on top of the richer MAH-PCA(4)?
base_pca = res["M0 + MAH-PCA(4)"]
add_pca = res["MAH-PCA(4) + c200c"]
gain_pca = 100 * (base_pca["crps"] - add_pca["crps"]) / base_pca["crps"]
print(f"[c200c on top of MAH-PCA(4)] per-aperture CRPS gain %: "
      + " ".join(f"{x:+.1f}" for x in gain_pca) + f"   overall {gain_pca.mean():+.2f}%")
print("  -> if it still helps on top of MAH-PCA(4), the info is genuinely "
      "independent of the MAH (not just DiffMAH smoothing loss).")


# %% ---- diagnostics: is c_200c MAH-determined? leftover info? ----------------
def cv_predict(X, y, k=5, seed=0):
    Xd = np.column_stack([np.ones(N), X]); order = np.random.default_rng(seed).permutation(N)
    p = np.full(N, np.nan)
    for fold in np.array_split(order, k):
        tr = np.setdiff1d(np.arange(N), fold)
        beta, *_ = np.linalg.lstsq(Xd[tr], y[tr], rcond=None)
        p[fold] = Xd[fold] @ beta
    return p


c_pred = cv_predict(dmah, c200)
r2_c = 1.0 - np.mean((c200 - c_pred) ** 2) / c200.var()
print(f"\n[is c_200c MAH-determined?] R^2(c_200c | DiffMAH, linear CV) = {r2_c:.3f}")

# partial correlation: residualize c_200c and each annulus vs DiffMAH, correlate
c_resid = c200 - cv_predict(dmah, c200)
print("[leftover] partial corr( c_200c , annulus | DiffMAH ):")
pcorr = []
for j in range(4):
    y_resid = Y[:, j] - cv_predict(dmah, Y[:, j])
    pc = np.corrcoef(c_resid, y_resid)[0, 1]
    pcorr.append(pc)
    print(f"    {TNAMES[j]:>7s}: r = {pc:+.3f}")


# %% ---- FIGURE 1: skill by feature set --------------------------------------
order = list(FEATURES)
short = ["M0", "M0+c200c", "DiffMAH", "DiffMAH\n+c200c", "DiffMAH\n+shuf",
         "MAH-PCA", "MAH-PCA\n+c200c"]
cols = [OKABE_ITO[7], OKABE_ITO[6], OKABE_ITO[0], OKABE_ITO[2], "0.7",
        OKABE_ITO[1], OKABE_ITO[4]]
fig1, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 4.4))
x = np.arange(4); w = 0.12
for k_i, name in enumerate(order):
    axA.bar(x + (k_i - 3) * w, res[name]["crps"], w, color=cols[k_i], label=name)
axA.set_xticks(x); axA.set_xticklabels(TNAMES)
axA.set_xlabel("aperture / annulus [kpc]"); axA.set_ylabel("CV CRPS [dex]")
axA.set_title("Predictive skill"); axA.legend(fontsize=6, ncol=2)
axA.text(-0.13, 1.04, "A", transform=axA.transAxes, fontweight="bold", fontsize=12)
# overall CRPS bars
ov = [res[n]["crps"].mean() for n in order]
axB.bar(range(len(order)), ov, color=cols)
axB.axhline(base["crps"].mean(), ls="--", color=OKABE_ITO[0], lw=1)
for i, n in enumerate(order):
    axB.annotate(f"{ov[i]:.4f}", (i, ov[i]), textcoords="offset points",
                 xytext=(0, 3), ha="center", fontsize=6.5)
axB.set_xticks(range(len(order)))
axB.set_xticklabels(short, fontsize=6.5)
axB.set_ylabel("overall CV CRPS [dex]"); axB.set_title("Overall")
axB.set_ylim(0, max(ov) * 1.15)
axB.text(-0.13, 1.04, "B", transform=axB.transAxes, fontweight="bold", fontsize=12)
fig1.suptitle(f"exp16 — does c_200c help beyond the MAH? (n={N}; "
              f"+c200c gain {gain.mean():+.2f}% CRPS)", fontsize=11)
fig1.tight_layout()
save_fig(fig1, FIGDIR / "exp16_skill")


# %% ---- FIGURE 2: the decisive check -- DiffMAH residual vs c_200c -----------
def binned(xv, yv, nbin=12):
    e = np.quantile(xv, np.linspace(0, 1, nbin + 1)); cen, med, err = [], [], []
    for a, bb in zip(e[:-1], e[1:]):
        m = (xv >= a) & (xv <= bb)
        if m.sum() >= 12:
            cen.append(np.median(xv[m])); med.append(np.median(yv[m]))
            err.append(yv[m].std() / np.sqrt(m.sum()))
    return map(np.asarray, (cen, med, err))


fig2, ax2 = plt.subplots(1, 2, figsize=(10.4, 4.3))
# left: c_200c vs its DiffMAH prediction (how MAH-determined)
a = ax2[0]
a.scatter(c_pred, c200, s=5, alpha=0.12, color=OKABE_ITO[0], edgecolors="none")
lo, hi = np.percentile(c200, [1, 99])
a.plot([lo, hi], [lo, hi], "k--", lw=1.1)
a.set_xlim(lo, hi); a.set_ylim(lo, hi)
a.set_xlabel("c_200c predicted from DiffMAH"); a.set_ylabel("true c_200c")
a.set_title(f"c_200c only weakly MAH-determined (R²={r2_c:.2f})")
a.text(-0.13, 1.04, "A", transform=a.transAxes, fontweight="bold", fontsize=12)
# right: DiffMAH-model residual of the 50-100 annulus vs c_200c (flat = no leftover)
b = ax2[1]
b.axhline(0, color="0.6", lw=0.8)
for j, c in [(3, OKABE_ITO[5]), (0, OKABE_ITO[2])]:
    yr = Y[:, j] - cv_predict(dmah, Y[:, j])
    cen, med, err = binned(c200, yr)
    b.errorbar(cen, med, yerr=err, fmt="o-", color=c, ms=4, label=f"{TNAMES[j]} kpc (r={pcorr[j]:+.2f})")
b.set_xlabel("c_200c"); b.set_ylabel("DiffMAH residual [dex]")
b.set_ylim(-0.12, 0.12)
b.set_title("Leftover annulus mass vs c_200c (slope = independent info)")
b.legend(fontsize=8)
b.text(-0.13, 1.04, "B", transform=b.transAxes, fontweight="bold", fontsize=12)
fig2.suptitle("exp16 — at fixed MAH, more concentrated halos host more stellar mass "
              "(c_200c carries assembly info the MAH misses)", fontsize=11)
fig2.tight_layout()
save_fig(fig2, FIGDIR / "exp16_residual_vs_c200c")


# %% ---- save outputs --------------------------------------------------------
OUTDIR.mkdir(parents=True, exist_ok=True)
st = Table({"features": order,
            "crps": [float(res[n]["crps"].mean()) for n in order],
            "r2_50_100": [float(res[n]["r2"][3]) for n in order]})
st.write(OUTDIR / "c200c_scores.csv", overwrite=True)
write_manifest(OUTDIR, params={
    "n": int(N), "targets": TNAMES,
    "crps_diffmah": float(base["crps"].mean()),
    "crps_diffmah_c200c": float(add["crps"].mean()),
    "c200c_gain_on_diffmah_pct": float(gain.mean()),
    "c200c_gain_on_mahpca_pct": float(gain_pca.mean()),
    "c200c_gain_per_aper_pct": gain.tolist(),
    "r2_c200c_given_diffmah": float(r2_c),
    "partial_corr_c200c_annulus": [float(p) for p in pcorr]})
print(f"\nwrote figures -> {FIGDIR}\nwrote outputs -> {OUTDIR}")
indep = abs(gain_pca.mean()) > 1.0 and abs(pcorr[3]) > 0.1
verdict = ("adds information INDEPENDENT of the MAH" if indep else
           "is largely redundant with the MAH")
print(f"\n[verdict] c_200c {verdict}: gain on DiffMAH {gain.mean():+.2f}%, "
      f"gain on MAH-PCA(4) {gain_pca.mean():+.2f}%, partial corr(50-100)={pcorr[3]:+.3f}, "
      f"R²(c|MAH)={r2_c:.2f}")
