"""exp06 — Use the *full* assembly history via PCA (no DiffMAH), connect the
halo and galaxy principal components, and compare halo-side representations.

Three questions:
1. How many numbers does the MAH actually need (PCA of the full log-MAH curve)?
2. Do the MAH principal components connect to the stellar-profile principal
   components (at fixed final halo mass)?
3. Does the data-driven MAH-PCA predict the profile better than the hand-picked
   summaries used in exp04/05?

Also a side check the user asked about: is the curve of growth as compressible as
exp02 found, and how does that compare to the (differential) surface-density
profile?

Run from the repo root:
    PYTHONPATH=. uv run python experiments/exp06_mah_pca/run.py
"""
# %% setup
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.table import Table

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hongshao.tng_data import (COG_RAD_KPC, load_mah, load_cosmic_time,        # noqa: E402
                               peak_history, TNG_COSMO)
from hongshao.stats import partial_spearman                                    # noqa: E402
from hongshao.plotting import set_style, save_fig, OKABE_ITO                   # noqa: E402
from hongshao.provenance import write_manifest                                 # noqa: E402

set_style()
HERE = Path(__file__).resolve().parent
OUTDIR, FIGDIR = HERE / "outputs", HERE / "figures"
TABLE = ROOT / "data" / "processed" / "tng300_072_z0p4.fits"
R = np.asarray(COG_RAD_KPC, float)

t = Table.read(TABLE)
t = t[t["use"]]
logM0 = np.asarray(t["logm0_halo"], float)
cog = np.asarray(t["logmstar_cog"], float)
idx = np.asarray(t["index"])


# %% build the MAH curves on a common cosmic-time grid, normalized by M0
mah = load_mah()
t_snap = load_cosmic_time()
tgrid = np.linspace(2.2, 9.0, 18)                 # Gyr; ~z=2.9 -> z=0.46 (well covered)
zgrid = np.array([TNG_COSMO.age(0).value])        # placeholder; label via redshift below
mah_shape = np.full((len(t), len(tgrid)), np.nan)
for row, i in enumerate(idx):
    snaps, lmp = peak_history(mah[i])
    tt = t_snap[snaps.astype(int)]                # ascending cosmic time
    if tt[0] <= tgrid[0] and tt[-1] >= tgrid[-1]:
        mah_shape[row] = np.interp(tgrid, tt, lmp) - logM0[row]   # log[Mpeak(t)/M0]
covered = np.isfinite(mah_shape).all(1)
print(f"MAH coverage: {covered.sum()}/{len(t)} halos span the grid")


def pca(X):
    """Centered covariance PCA. Returns (mean, components Vt, evr, scores)."""
    mu = X.mean(0)
    Xc = X - mu
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return mu, Vt, S**2 / np.sum(S**2), Xc @ Vt.T


mu_mah, V_mah, evr_mah, sc_mah = pca(mah_shape[covered])
# orient so positive PC1 = earlier-assembled (more mass at early times)
if np.corrcoef(sc_mah[:, 0], mah_shape[covered][:, 0])[0, 1] < 0:
    V_mah[0] *= -1; sc_mah[:, 0] *= -1
print("MAH variance: " + ", ".join(f"PC{k+1}={evr_mah[k]:.1%}" for k in range(4))
      + f"  (PC1-3 {evr_mah[:3].sum():.1%})")

# %% CoG-shape PCA (exp02, recomputed) and a Sigma-like PCA, on the same halos
cog_c = cog[covered]
cog_shape = (cog_c - cog_c[:, -1:])[:, :-1]
mu_cog, V_cog, evr_cog, sc_cog = pca(cog_shape)
for k in range(3):
    if np.corrcoef(sc_cog[:, k], cog_shape[:, 0])[0, 1] < 0:
        V_cog[k] *= -1; sc_cog[:, k] *= -1

# differential surface-density-like profile from the CoG (annulus mass / area)
M = 10.0 ** cog_c
annulus = np.diff(M, axis=1)
area = np.pi * (R[1:] ** 2 - R[:-1] ** 2)
with np.errstate(divide="ignore", invalid="ignore"):
    log_sigma = np.log10(np.where(annulus > 0, annulus, np.nan) / area)
ok = np.isfinite(log_sigma).all(1)
_, _, evr_sig, _ = pca(log_sigma[ok])
print(f"compressibility (cumulative variance in 3 modes): "
      f"CoG={evr_cog[:3].sum():.1%}, Sigma-like={evr_sig[:3].sum():.1%}, "
      f"MAH={evr_mah[:3].sum():.1%}")

# %% connect MAH PCs to CoG PCs (partial corr, controlling for M0)
m0c = logM0[covered]
nmah, ncog = 3, 3
conn = np.full((nmah, ncog), np.nan)
for i in range(nmah):
    for j in range(ncog):
        conn[i, j], _, _ = partial_spearman(sc_mah[:, i], sc_cog[:, j], m0c)
print("\nMAH-PC x CoG-PC partial corr (fixed M0):")
for i in range(nmah):
    print(f"  MAH-PC{i+1}: " + ", ".join(f"CoG-PC{j+1}={conn[i, j]:+.2f}"
                                         for j in range(ncog)))


# %% head-to-head: which halo representation predicts the profile best?
def cv_predict(X, Y, k=5, seed=0):
    n = len(Y)
    Xd = np.column_stack([np.ones(n), X])
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    pred = np.full_like(Y, np.nan)
    for fold in np.array_split(order, k):
        tr = np.setdiff1d(np.arange(n), fold)
        beta, *_ = np.linalg.lstsq(Xd[tr], Y[tr], rcond=None)
        pred[fold] = Xd[fold] @ beta
    return pred


def shuffle_within_bins(MX, binvar, nbins=12, seed=1):
    rng = np.random.default_rng(seed)
    out = MX.copy()
    edges = np.quantile(binvar, np.linspace(0, 1, nbins + 1))
    b = np.digitize(binvar, edges[1:-1])
    for bi in np.unique(b):
        ii = np.where(b == bi)[0]
        out[ii] = MX[ii][rng.permutation(len(ii))]
    return out


hp_names = ["logmpeak_z0p7", "logmpeak_z1", "logmpeak_z1p5", "logmpeak_z2",
            "z50", "z75", "z90"]
HP = np.column_stack([np.asarray(t[c], float)[covered] for c in hp_names])
KPC = 4                                            # MAH-PCA components to use
Ymat = cog_c                                       # predict the full CoG
fmask = np.isfinite(HP).all(1)
y, m0f = Ymat[fmask], m0c[fmask]
reps = {
    "M0 only": m0f[:, None],
    "hand-picked summaries": np.column_stack([m0f, HP[fmask]]),
    f"MAH-PCA (top {KPC})": np.column_stack([m0f, sc_mah[fmask, :KPC]]),
    "MAH-PCA shuffled": np.column_stack(
        [m0f, shuffle_within_bins(sc_mah[fmask, :KPC], m0f)]),
}
base = None
rms = {}
for name, X in reps.items():
    rms[name] = float(np.sqrt(np.mean((y - cv_predict(X, y)) ** 2)))
    if base is None:
        base = rms[name]
print("\nfull-CoG prediction RMS [dex] (improvement vs M0-only):")
for name, v in rms.items():
    print(f"  {name:24s} {v:.4f}   {100*(base-v)/base:+.1f}%")

# %% FIGURE 1: MAH-PCA
fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.2, 4.0))
nshow = 8
axA.bar(range(1, nshow + 1), evr_mah[:nshow], color=OKABE_ITO[7], alpha=0.7,
        label="per mode")
axA.plot(range(1, nshow + 1), np.cumsum(evr_mah[:nshow]), "-o", color=OKABE_ITO[5],
         ms=4, label="cumulative")
axA.set_xlabel("MAH principal component")
axA.set_ylabel("fraction of assembly-history variance")
axA.set_title("How many numbers the MAH needs")
axA.legend(loc="center right")
axA.set_ylim(0, 1.05)
axA.text(-0.13, 1.04, "A", transform=axA.transAxes, fontweight="bold", fontsize=12)

zlab = np.array([TNG_COSMO.age(0).value])  # noqa: F841
axB.axhline(0, ls=":", color="0.6", lw=1)
for k, (mk, ls) in enumerate([("o", "-"), ("s", "--"), ("^", "-.")]):
    axB.plot(tgrid, V_mah[k], color=OKABE_ITO[k], marker=mk, ls=ls, ms=4,
             label=f"MAH-PC{k+1} ({evr_mah[k]:.0%})")
axB.set_xlabel("cosmic time [Gyr]")
axB.set_ylabel(r"mode amplitude in $\log[M_{\rm peak}(t)/M_0]$")
axB.set_title("What each assembly mode looks like")
axB.legend()
axB.text(-0.13, 1.04, "B", transform=axB.transAxes, fontweight="bold", fontsize=12)
fig.suptitle(f"exp06 — full assembly history via PCA (n={covered.sum()})", fontsize=11)
fig.tight_layout()
save_fig(fig, FIGDIR / "exp06_mah_pca")

# %% FIGURE 2: connection + head-to-head
fig2, (axC, axD) = plt.subplots(1, 2, figsize=(9.2, 4.0))
im = axC.imshow(conn, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
axC.set_xticks(range(ncog)); axC.set_xticklabels([f"CoG-PC{j+1}" for j in range(ncog)])
axC.set_yticks(range(nmah)); axC.set_yticklabels([f"MAH-PC{i+1}" for i in range(nmah)])
for i in range(nmah):
    for j in range(ncog):
        axC.text(j, i, f"{conn[i, j]:+.2f}", ha="center", va="center", fontsize=8,
                 color="white" if abs(conn[i, j]) > 0.3 else "k")
axC.set_title("Assembly modes vs profile modes\n(partial corr, fixed $M_0$)")
fig2.colorbar(im, ax=axC, label="partial Spearman r", shrink=0.8)
axC.text(-0.2, 1.04, "C", transform=axC.transAxes, fontweight="bold", fontsize=12)

names = list(rms)
vals = [rms[n] for n in names]
colors = [OKABE_ITO[7], OKABE_ITO[4], OKABE_ITO[0], "0.7"]
axD.bar(range(len(names)), vals, color=colors)
for i, n in enumerate(names):
    axD.annotate(f"{100*(base-vals[i])/base:+.0f}%", (i, vals[i]),
                 textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
axD.set_xticks(range(len(names)))
axD.set_xticklabels(["M0", "hand-picked", f"MAH-PCA{KPC}", "MAH-PCA\nshuffled"],
                    fontsize=8)
axD.set_ylabel("full-CoG prediction RMS [dex]")
axD.set_title("Which halo representation predicts best")
axD.set_ylim(0, max(vals) * 1.15)
axD.text(-0.13, 1.04, "D", transform=axD.transAxes, fontweight="bold", fontsize=12)
fig2.suptitle("exp06 — connecting and comparing halo representations", fontsize=11)
fig2.tight_layout()
save_fig(fig2, FIGDIR / "exp06_connection")

# %% FIGURE 3: compressibility comparison (the user's CoG-vs-Sigma question)
fig3, ax = plt.subplots(figsize=(5.2, 4.0))
for evr, lab, col, mk in [(evr_cog, "curve of growth", OKABE_ITO[0], "o"),
                          (evr_sig, "surface density (Σ-like)", OKABE_ITO[5], "s"),
                          (evr_mah, "assembly history", OKABE_ITO[2], "^")]:
    ax.plot(range(1, 9), np.cumsum(evr[:8]), "-", marker=mk, color=col, ms=4, label=lab)
ax.axhline(0.99, ls=":", color="0.6", lw=1)
ax.set_xlabel("number of principal components")
ax.set_ylabel("cumulative variance explained")
ax.set_title("How compressible is each object?")
ax.legend()
ax.set_ylim(0.5, 1.02)
fig3.tight_layout()
save_fig(fig3, FIGDIR / "exp06_compressibility")

# %% save outputs
OUTDIR.mkdir(parents=True, exist_ok=True)
np.savez(OUTDIR / "mah_pca.npz", tgrid=tgrid, mean=mu_mah, components=V_mah,
         evr=evr_mah, scores=sc_mah[:, :5], connection=conn)
ct = Table({"representation": names, "rms_dex": vals,
            "improvement_pct": [100 * (base - v) / base for v in vals]})
ct.write(OUTDIR / "representation_comparison.csv", overwrite=True)
write_manifest(OUTDIR, params={
    "n": int(covered.sum()), "tgrid_Gyr": [float(tgrid[0]), float(tgrid[-1])],
    "evr_mah_pc1_3": float(evr_mah[:3].sum()),
    "evr_cog_pc1_3": float(evr_cog[:3].sum()),
    "evr_sigma_pc1_3": float(evr_sig[:3].sum())})
print(f"\nwrote figures -> {FIGDIR}\nwrote outputs -> {OUTDIR}")
