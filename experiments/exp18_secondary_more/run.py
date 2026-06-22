"""exp18 — More secondary properties: accretion rate and 3D halo shape.

exp16 showed halo concentration `c_200c` adds real, MAH-independent information
(+5% on DiffMAH, +2.7% on MAH-PCA(4)). Here we apply the *same* test to the
other secondary halo properties in the aperture table — accretion rate
`acc_rate` and 3D halo shape (`c_to_a_3d`, `b_to_a_3d`) — and measure the
combined limit of all secondary properties together.

For each property P (and the combined set), 5-fold CV on the four CoG-derived
annulus masses (primary observable):
  - CRPS gain of DiffMAH(4)+P over DiffMAH(4)         -- does it help the portable model?
  - CRPS gain of MAH-PCA(4)+P over MAH-PCA(4)         -- independent of the full MAH?
  - shuffle control (DiffMAH + shuffled P)            -- must collapse to DiffMAH
  - R^2(P | DiffMAH) and partial corr(P, annulus | DiffMAH)  -- how MAH-determined / leftover info
  - DiffMAH + ALL secondaries                         -- the combined ceiling

Run (fast): EXP18_NMAX=600 PYTHONPATH=. uv run python experiments/exp18_secondary_more/run.py
Full:       PYTHONPATH=. uv run python experiments/exp18_secondary_more/run.py
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
from hongshao.metrics import crps_gaussian                                      # noqa: E402
from hongshao.plotting import set_style, save_fig, OKABE_ITO                     # noqa: E402
from hongshao.provenance import write_manifest                                  # noqa: E402

set_style()
HERE = Path(__file__).resolve().parent
OUTDIR, FIGDIR = HERE / "outputs", HERE / "figures"
TABLE = ROOT / "data" / "processed" / "tng300_072_z0p4.fits"
TNAMES = ["<10", "10-30", "30-50", "50-100"]
NMAX = int(os.environ.get("EXP18_NMAX", 0))
PROPS = ["c_200c", "acc_rate", "c_to_a_3d", "b_to_a_3d"]

t = Table.read(TABLE)
t = t[t["use"]]
if NMAX:
    t = t[:NMAX]
aper = np.asarray(t["logmstar_aper"], float)
M0 = np.asarray(t["logmh_z0p4"], float)
idx = np.asarray(t["index"])
dmah = np.column_stack([np.asarray(t[c], float) for c in
                        ("dmah_logmp", "dmah_logtc", "dmah_early", "dmah_late")])
P = {p: np.asarray(t[p], float) for p in PROPS}

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
     & np.isfinite(ms).all(1) & np.all([np.isfinite(P[p]) for p in PROPS], axis=0))
Y = Y[g]; M0 = M0[g]; dmah = dmah[g]; ms = ms[g]
P = {p: P[p][g] for p in PROPS}
N = len(Y)
mu_m = ms.mean(0); _, _, Vt = np.linalg.svd(ms - mu_m, full_matrices=False)
pca = (ms - mu_m) @ Vt[:4].T
print(f"exp18: secondary properties {PROPS} on n={N}")

# raw correlation of each property with formation redshift (context)
z50 = np.asarray(t["z50"], float)[g]
print("  raw corr with z50:  " + "  ".join(f"{p}={np.corrcoef(P[p], z50)[0,1]:+.2f}" for p in PROPS))


# %% ---- CV scorer (per-aperture linear mean + homoscedastic Gaussian) -------
var_tot = Y.var(0)


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
    return crps_gaussian(Y, pred, sig).mean(0)


def cv_predict(X, y, k=5, seed=0):
    Xd = np.column_stack([np.ones(N), X]); order = np.random.default_rng(seed).permutation(N)
    p = np.full(N, np.nan)
    for fold in np.array_split(order, k):
        tr = np.setdiff1d(np.arange(N), fold)
        beta, *_ = np.linalg.lstsq(Xd[tr], y[tr], rcond=None)
        p[fold] = Xd[fold] @ beta
    return p


rng = np.random.default_rng(1)
edges = np.quantile(M0, np.linspace(0, 1, 13)); mbin = np.digitize(M0, edges[1:-1])


def shuffle_in_mass(v):
    out = v.copy()
    for bi in np.unique(mbin):
        ii = np.where(mbin == bi)[0]
        out[ii] = v[ii][rng.permutation(len(ii))]
    return out


# %% ---- per-property test ----------------------------------------------------
crps_dm = cv_score(dmah)
crps_pca = cv_score(np.column_stack([M0, pca]))
print(f"\nbaselines: DiffMAH(4) CRPS={crps_dm.mean():.4f}   MAH-PCA(4) CRPS={crps_pca.mean():.4f}")
print(f"\n{'property':10s} {'gain/DiffMAH':>12s} {'gain/MAH-PCA':>12s} {'shuf':>7s} "
      f"{'R2(P|MAH)':>9s} {'pcorr(50-100)':>13s}")
rows = {}
for p in PROPS:
    v = P[p]
    g_dm = 100 * (crps_dm.mean() - cv_score(np.column_stack([dmah, v])).mean()) / crps_dm.mean()
    g_pca = 100 * (crps_pca.mean() - cv_score(np.column_stack([M0, pca, v])).mean()) / crps_pca.mean()
    g_shuf = 100 * (crps_dm.mean() - cv_score(np.column_stack([dmah, shuffle_in_mass(v)])).mean()) / crps_dm.mean()
    r2 = 1.0 - np.mean((v - cv_predict(dmah, v)) ** 2) / v.var()
    v_res = v - cv_predict(dmah, v)
    y_res = Y[:, 3] - cv_predict(dmah, Y[:, 3])
    pc = np.corrcoef(v_res, y_res)[0, 1]
    rows[p] = dict(g_dm=g_dm, g_pca=g_pca, g_shuf=g_shuf, r2=r2, pcorr=pc)
    print(f"  {p:10s} {g_dm:+11.2f}% {g_pca:+11.2f}% {g_shuf:+6.2f}% {r2:9.2f} {pc:+13.3f}")

# combined: DiffMAH + all secondaries; MAH-PCA + all
allP = np.column_stack([P[p] for p in PROPS])
crps_dm_all = cv_score(np.column_stack([dmah, allP]))
crps_pca_all = cv_score(np.column_stack([M0, pca, allP]))
g_dm_all = 100 * (crps_dm.mean() - crps_dm_all.mean()) / crps_dm.mean()
g_pca_all = 100 * (crps_pca.mean() - crps_pca_all.mean()) / crps_pca.mean()
print(f"\n  {'ALL 4':10s} {g_dm_all:+11.2f}% {g_pca_all:+11.2f}%   "
      f"(DiffMAH+all={crps_dm_all.mean():.4f}, MAH-PCA+all={crps_pca_all.mean():.4f})")


# %% ---- FIGURE: gains per property ------------------------------------------
fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.5, 4.3))
labels = PROPS + ["ALL 4"]
gdm = [rows[p]["g_dm"] for p in PROPS] + [g_dm_all]
gpca = [rows[p]["g_pca"] for p in PROPS] + [g_pca_all]
gshuf = [rows[p]["g_shuf"] for p in PROPS] + [0.0]
x = np.arange(len(labels)); w = 0.27
axA.bar(x - w, gdm, w, color=OKABE_ITO[2], label="on DiffMAH(4)")
axA.bar(x, gpca, w, color=OKABE_ITO[4], label="on MAH-PCA(4)")
axA.bar(x + w, gshuf, w, color="0.7", label="shuffled (control)")
axA.axhline(0, color="k", lw=0.8)
axA.set_xticks(x); axA.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
axA.set_ylabel("CRPS gain [%]"); axA.set_title("Does each property help beyond the MAH?")
axA.legend(fontsize=7)
axA.text(-0.13, 1.04, "A", transform=axA.transAxes, fontweight="bold", fontsize=12)
# partial corr (50-100) and R2(P|MAH)
pc = [rows[p]["pcorr"] for p in PROPS]
r2 = [rows[p]["r2"] for p in PROPS]
axB.bar(np.arange(len(PROPS)) - 0.2, pc, 0.4, color=OKABE_ITO[5], label="partial corr (50-100 | DiffMAH)")
axB.bar(np.arange(len(PROPS)) + 0.2, r2, 0.4, color=OKABE_ITO[0], label="R²(P | DiffMAH)")
axB.axhline(0, color="k", lw=0.8)
axB.set_xticks(np.arange(len(PROPS))); axB.set_xticklabels(PROPS, rotation=20, ha="right", fontsize=8)
axB.set_title("Independent info vs MAH-determined"); axB.legend(fontsize=7)
axB.text(-0.13, 1.04, "B", transform=axB.transAxes, fontweight="bold", fontsize=12)
fig.suptitle(f"exp18 — secondary halo properties beyond the MAH (n={N})", fontsize=11)
fig.tight_layout()
save_fig(fig, FIGDIR / "exp18_secondary_gains")


# %% ---- save outputs --------------------------------------------------------
OUTDIR.mkdir(parents=True, exist_ok=True)
st = Table({"property": PROPS,
            "gain_diffmah_pct": [rows[p]["g_dm"] for p in PROPS],
            "gain_mahpca_pct": [rows[p]["g_pca"] for p in PROPS],
            "gain_shuffled_pct": [rows[p]["g_shuf"] for p in PROPS],
            "r2_given_diffmah": [rows[p]["r2"] for p in PROPS],
            "pcorr_50_100": [rows[p]["pcorr"] for p in PROPS]})
st.write(OUTDIR / "secondary_gains.csv", overwrite=True)
write_manifest(OUTDIR, params={
    "n": int(N), "props": PROPS,
    "crps_diffmah": float(crps_dm.mean()), "crps_mahpca": float(crps_pca.mean()),
    "gain_diffmah_pct": {p: float(rows[p]["g_dm"]) for p in PROPS},
    "gain_mahpca_pct": {p: float(rows[p]["g_pca"]) for p in PROPS},
    "gain_all_on_diffmah_pct": float(g_dm_all),
    "gain_all_on_mahpca_pct": float(g_pca_all),
    "crps_diffmah_all": float(crps_dm_all.mean())})
print(f"\nwrote figure -> {FIGDIR}\nwrote outputs -> {OUTDIR}")
