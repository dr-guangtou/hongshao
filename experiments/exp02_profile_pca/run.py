"""exp02 — Compress the curves of growth, and test which profile-shape modes
carry the halo-assembly signal found in exp01.

We PCA the *shape* of each curve of growth (log M*(<R) normalized to its value
at the largest radius, so total mass is divided out). This asks: how few numbers
describe a massive galaxy's light profile, and — at fixed final halo mass M0 —
do the shape modes correlate with how the halo assembled?

Run from the repo root:
    PYTHONPATH=. uv run python experiments/exp02_profile_pca/run.py
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
from hongshao.tng_data import COG_RAD_KPC           # noqa: E402
from hongshao.stats import partial_spearman         # noqa: E402
from hongshao.plotting import set_style, save_fig, OKABE_ITO  # noqa: E402
from hongshao.provenance import write_manifest      # noqa: E402

set_style()
HERE = Path(__file__).resolve().parent
OUTDIR, FIGDIR = HERE / "outputs", HERE / "figures"
TABLE = ROOT / "data" / "processed" / "tng300_072_z0p4.fits"

t = Table.read(TABLE)
t = t[t["use"]]
N = len(t)
print(f"analysis sample (use cut): {N} galaxies")

logM0 = np.asarray(t["logm0_halo"], float)
cog = np.asarray(t["logmstar_cog"], float)          # (N, 24) log M*(<R)
logMtot = cog[:, -1]                                # ~ log M*(<148 kpc)

# %% PCA on the curve-of-growth SHAPE (total mass divided out)
# shape_i(R) = log[ M*(<R) / M*(<Rmax) ]; last radius is identically 0 -> drop it
shape = (cog - logMtot[:, None])[:, :-1]            # (N, 23)
rad = COG_RAD_KPC[:-1]
mu = shape.mean(axis=0)
Xc = shape - mu
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
evr = S**2 / np.sum(S**2)                           # explained-variance ratio
scores = Xc @ Vt.T                                  # (N, 23) PC coefficients

# sign convention: positive score correlates with a higher inner-shape value,
# i.e. a MORE centrally concentrated profile (more mass already within ~2 kpc)
for k in range(3):
    if np.corrcoef(scores[:, k], shape[:, 0])[0, 1] < 0:
        Vt[k] *= -1
        scores[:, k] *= -1

print("\nexplained variance: "
      + ", ".join(f"PC{k+1}={evr[k]:.1%}" for k in range(4))
      + f"  (PC1-3 cumulative {evr[:3].sum():.1%})")

# reconstruction accuracy of log M*(<R) using K modes
recon_rms = []
for K in range(1, 5):
    rec = mu + scores[:, :K] @ Vt[:K]
    recon_rms.append(float(np.sqrt(np.mean((rec - shape) ** 2))))
print("reconstruction RMS [dex] with K modes:",
      {K: round(r, 4) for K, r in zip(range(1, 5), recon_rms)})

# %% which PCs are "mass/size" axes, which carry assembly info?
mah_feats = {
    "Mpeak(z=1)": np.asarray(t["logmpeak_z1"], float),
    "Mpeak(z=2)": np.asarray(t["logmpeak_z2"], float),
    "z50": np.asarray(t["z50"], float),
    "z90": np.asarray(t["z90"], float),
}
# context: how does each PC relate to total stellar mass and final halo mass?
print("\nPC vs (raw Spearman):  total M*    |  halo M0")
from scipy.stats import spearmanr
for k in range(3):
    print(f"  PC{k+1}: {spearmanr(scores[:, k], logMtot)[0]:+.2f}      "
          f"{spearmanr(scores[:, k], logM0)[0]:+.2f}")

# headline: partial corr of each PC with assembly, controlling for M0
R_assembly = np.full((3, len(mah_feats)), np.nan)
P_assembly = np.full_like(R_assembly, np.nan)
for k in range(3):
    for j, f in enumerate(mah_feats.values()):
        R_assembly[k, j], P_assembly[k, j], _ = partial_spearman(
            scores[:, k], f, logM0)
print("\npartial corr of PC_k with assembly at fixed M0:")
for k in range(3):
    print(f"  PC{k+1}: " + ", ".join(
        f"{name}={R_assembly[k, j]:+.2f}" for j, name in enumerate(mah_feats)))

# %% FIGURE 1: compression — variance explained + the mode shapes
fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.2, 4.0))
nshow = 8
axA.bar(np.arange(1, nshow + 1), evr[:nshow], color=OKABE_ITO[7], alpha=0.7,
        label="per mode")
axA.plot(np.arange(1, nshow + 1), np.cumsum(evr[:nshow]), color=OKABE_ITO[5],
         marker="o", ms=5, label="cumulative")
axA.axhline(1.0, ls=":", color="0.6", lw=1)
axA.set_xlabel("principal component")
axA.set_ylabel("fraction of shape variance")
axA.set_title("Profiles are low-dimensional")
axA.legend(loc="center right")
axA.set_ylim(0, 1.05)
axA.text(-0.13, 1.04, "A", transform=axA.transAxes, fontweight="bold", fontsize=12)

axB.axhline(0, ls=":", color="0.6", lw=1)
for k, (mk, ls) in enumerate([("o", "-"), ("s", "--"), ("^", "-.")]):
    axB.plot(rad, Vt[k], color=OKABE_ITO[k], marker=mk, ls=ls, ms=4, lw=1.5,
             label=f"PC{k+1} ({evr[k]:.0%})")
axB.set_xscale("log")
axB.set_xlabel("radius R [kpc]")
axB.set_ylabel("mode amplitude  in $\\log M_*(<R)$")
axB.set_title("What each mode does to the profile")
axB.legend()
axB.text(-0.13, 1.04, "B", transform=axB.transAxes, fontweight="bold", fontsize=12)
fig.suptitle(f"exp02 — curve-of-growth shape PCA, TNG300 z=0.4 (n={N})", fontsize=11)
fig.tight_layout()
save_fig(fig, FIGDIR / "exp02_modes")

# %% FIGURE 2: do the shape modes carry assembly information at fixed M0?
fig2, (axC, axD) = plt.subplots(1, 2, figsize=(9.2, 4.0))
x = np.arange(len(mah_feats))
w = 0.26
for k in range(3):
    axC.bar(x + (k - 1) * w, R_assembly[k], w, color=OKABE_ITO[k],
            label=f"PC{k+1}")
axC.axhline(0, color="k", lw=0.8)
axC.set_xticks(x)
axC.set_xticklabels(list(mah_feats), rotation=25, ha="right")
axC.set_ylabel(r"partial Spearman $r$  (at fixed $M_0$)")
axC.set_title("Shape modes vs. halo assembly")
axC.legend()
axC.text(-0.13, 1.04, "C", transform=axC.transAxes, fontweight="bold", fontsize=12)

# scatter of the strongest PC-vs-assembly cell (residuals of both vs M0)
kbest, jbest = np.unravel_index(np.nanargmax(np.abs(R_assembly)), R_assembly.shape)
fname = list(mah_feats)[jbest]
fvals = list(mah_feats.values())[jbest]
m = np.isfinite(scores[:, kbest]) & np.isfinite(fvals) & np.isfinite(logM0)
# residuals vs M0 (quadratic) to visualize the partial relation
def resid(v):
    c = np.polyfit(logM0[m], v[m], 2)
    return v[m] - np.polyval(c, logM0[m])
axD.scatter(resid(fvals), resid(scores[:, kbest]), s=5, alpha=0.25,
            color=OKABE_ITO[kbest])
axD.set_xlabel(f"{fname} residual  (at fixed $M_0$)")
axD.set_ylabel(f"PC{kbest+1} residual  (at fixed $M_0$)")
axD.set_title(f"Strongest link: PC{kbest+1} vs {fname} "
              f"(r={R_assembly[kbest, jbest]:+.2f})")
axD.text(-0.13, 1.04, "D", transform=axD.transAxes, fontweight="bold", fontsize=12)
fig2.suptitle("exp02 — assembly information in profile shape (beyond total mass)",
              fontsize=11)
fig2.tight_layout()
save_fig(fig2, FIGDIR / "exp02_assembly")

# %% save outputs + provenance
OUTDIR.mkdir(parents=True, exist_ok=True)
np.savez(OUTDIR / "pca.npz", mean_shape=mu, components=Vt, evr=evr,
         radii=rad, scores=scores[:, :5], recon_rms=recon_rms)
at = Table({"PC": [f"PC{k+1}" for k in range(3)]})
for j, name in enumerate(mah_feats):
    at[name] = R_assembly[:, j]
at.write(OUTDIR / "pc_assembly_partial_corr.csv", overwrite=True)
write_manifest(OUTDIR, params={
    "n_use": N, "method": "covariance PCA on normalized log-CoG shape",
    "control": "partial Spearman vs M0", "evr_pc1_3": evr[:3].tolist()})
print(f"\nwrote figures -> {FIGDIR}\nwrote outputs -> {OUTDIR}")
