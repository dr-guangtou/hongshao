"""exp03 — Fit the interpretable radial-DiffMAH profile model to every curve of
growth, and ask which physical shape parameters carry the assembly signal.

Each galaxy's curve of growth is reduced to 5 meaningful numbers: normalization,
inner slope, outer slope, transition radius R_c, transition width Delta. We then
(1) check the fit quality vs the PCA benchmark from exp02, (2) map the parameters
onto the PCA modes, and (3) test which parameters correlate with halo assembly at
fixed final halo mass M0.

Run from the repo root:
    PYTHONPATH=. uv run python experiments/exp03_radial_diffmah/run.py
"""
# %% setup
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.table import Table
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hongshao.tng_data import COG_RAD_KPC                       # noqa: E402
from hongshao.profiles import fit_cog, beta_of_R, LN10         # noqa: E402
from hongshao.stats import partial_spearman                    # noqa: E402
from hongshao.plotting import set_style, save_fig, OKABE_ITO   # noqa: E402
from hongshao.provenance import write_manifest                 # noqa: E402

set_style()
HERE = Path(__file__).resolve().parent
OUTDIR, FIGDIR = HERE / "outputs", HERE / "figures"
TABLE = ROOT / "data" / "processed" / "tng300_072_z0p4.fits"
R = np.asarray(COG_RAD_KPC, float)

t = Table.read(TABLE)
t = t[t["use"]]
N = len(t)
logM0 = np.asarray(t["logm0_halo"], float)
cog = np.asarray(t["logmstar_cog"], float)
print(f"analysis sample (use cut): {N} galaxies")


def model_cog(params):
    """log10 M*(<R) from physical radial-DiffMAH params (for overlays)."""
    beta = beta_of_R(R, params["beta_in"], params["beta_out"],
                     params["R_c"], params["Delta"])
    u = np.log(R)
    integ = np.concatenate([[0.0], np.cumsum(0.5 * (beta[1:] + beta[:-1]) * np.diff(u))])
    return params["logMstar0"] + integ / LN10


# %% fit every galaxy
keys = ["logMstar0", "beta_in", "beta_out", "R_c", "Delta", "rms"]
fits = {k: np.full(N, np.nan) for k in keys}
for i in range(N):
    f = fit_cog(R, cog[i])
    for k in keys:
        fits[k][i] = f[k]
rms = fits["rms"]
print(f"fit RMS [dex]: median={np.median(rms):.4f}  90th={np.percentile(rms, 90):.4f}  "
      f"frac<0.02={(rms < 0.02).mean():.2f}")
print("PCA benchmark (exp02): 2 modes 0.010 dex, 3 modes 0.005 dex")

# %% recompute exp02 shape-PCA scores here (self-contained) for param<->PC mapping
shape = (cog - cog[:, -1:])[:, :-1]
Xc = shape - shape.mean(0)
_, _, Vt = np.linalg.svd(Xc, full_matrices=False)
scores = Xc @ Vt.T
for k in range(3):                                  # match exp02 orientation
    if np.corrcoef(scores[:, k], shape[:, 0])[0, 1] < 0:
        scores[:, k] *= -1
PCs = {f"PC{k+1}": scores[:, k] for k in range(3)}

params = {k: fits[k] for k in ["logMstar0", "beta_in", "beta_out", "R_c", "Delta"]}
print("\nparam <-> PCA mode (Spearman):")
corr_pc = np.full((len(params), 3), np.nan)
for ip, (pn, pv) in enumerate(params.items()):
    for k in range(3):
        m = np.isfinite(pv)
        corr_pc[ip, k] = spearmanr(pv[m], scores[m, k])[0]
    print(f"  {pn:10s}: " + ", ".join(f"PC{k+1}={corr_pc[ip, k]:+.2f}" for k in range(3)))

# %% which parameters carry assembly info at fixed M0?
mah = {"Mpeak(z=1)": np.asarray(t["logmpeak_z1"], float),
       "Mpeak(z=2)": np.asarray(t["logmpeak_z2"], float),
       "z50": np.asarray(t["z50"], float),
       "z90": np.asarray(t["z90"], float)}
assembly_params = ["beta_in", "beta_out", "R_c", "Delta"]
R_assembly = np.full((len(assembly_params), len(mah)), np.nan)
for ip, pn in enumerate(assembly_params):
    pv = np.log10(params[pn]) if pn == "R_c" else params[pn]   # R_c in dex
    for j, fv in enumerate(mah.values()):
        R_assembly[ip, j], _, _ = partial_spearman(pv, fv, logM0)
print("\npartial corr of profile param with assembly at fixed M0:")
for ip, pn in enumerate(assembly_params):
    print(f"  {pn:9s}: " + ", ".join(
        f"{n}={R_assembly[ip, j]:+.2f}" for j, n in enumerate(mah)))

# %% FIGURE 1: fit quality
fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.2, 4.0))
order = np.argsort(rms)
examples = [("best", order[0]), ("median", order[len(order) // 2]),
            ("worst", order[-1])]
for (lab, idx), col in zip(examples, [OKABE_ITO[2], OKABE_ITO[4], OKABE_ITO[5]]):
    axA.plot(R, cog[idx], "o", color=col, ms=4)
    p = {k: fits[k][idx] for k in params}
    axA.plot(R, model_cog(p), "-", color=col, lw=1.5,
             label=f"{lab} (rms={rms[idx]:.3f})")
axA.set_xscale("log")
axA.set_xlabel("radius R [kpc]")
axA.set_ylabel(r"$\log_{10} M_*(<R)\,[M_\odot]$")
axA.set_title("radial-DiffMAH fits to the curve of growth")
axA.legend()
axA.text(-0.13, 1.04, "A", transform=axA.transAxes, fontweight="bold", fontsize=12)

axB.hist(rms, bins=np.linspace(0, 0.05, 40), color=OKABE_ITO[7], alpha=0.8)
axB.axvline(np.median(rms), color=OKABE_ITO[5], lw=1.5,
            label=f"median {np.median(rms):.3f}")
axB.axvline(0.010, color=OKABE_ITO[2], ls="--", lw=1.3, label="PCA 2-mode")
axB.set_xlabel("fit RMS [dex]")
axB.set_ylabel("galaxies")
axB.set_title(f"5-parameter fit quality (n={N})")
axB.legend()
axB.text(-0.13, 1.04, "B", transform=axB.transAxes, fontweight="bold", fontsize=12)
fig.suptitle("exp03 — radial-DiffMAH profile fits, TNG300 z=0.4", fontsize=11)
fig.tight_layout()
save_fig(fig, FIGDIR / "exp03_fit_quality")

# %% FIGURE 2: param<->PC mapping + assembly correlations
fig2, (axC, axD) = plt.subplots(1, 2, figsize=(9.2, 4.0))
im = axC.imshow(corr_pc, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
axC.set_xticks(range(3)); axC.set_xticklabels(list(PCs))
axC.set_yticks(range(len(params))); axC.set_yticklabels(list(params))
for i in range(len(params)):
    for k in range(3):
        axC.text(k, i, f"{corr_pc[i, k]:+.2f}", ha="center", va="center",
                 fontsize=7, color="white" if abs(corr_pc[i, k]) > 0.5 else "k")
axC.set_title("Physical params vs PCA modes")
fig2.colorbar(im, ax=axC, label="Spearman r", shrink=0.8)
axC.text(-0.2, 1.04, "C", transform=axC.transAxes, fontweight="bold", fontsize=12)

x = np.arange(len(mah)); w = 0.2
labels = {"beta_in": r"$\beta_{\rm in}$", "beta_out": r"$\beta_{\rm out}$",
          "R_c": r"$R_c$", "Delta": r"$\Delta$"}
for ip, pn in enumerate(assembly_params):
    axD.bar(x + (ip - 1.5) * w, R_assembly[ip], w, color=OKABE_ITO[ip],
            label=labels[pn])
axD.axhline(0, color="k", lw=0.8)
axD.set_xticks(x); axD.set_xticklabels(list(mah), rotation=25, ha="right")
axD.set_ylabel(r"partial Spearman $r$ (fixed $M_0$)")
axD.set_title("Profile params vs halo assembly")
axD.legend(ncol=2, fontsize=7)
axD.text(-0.13, 1.04, "D", transform=axD.transAxes, fontweight="bold", fontsize=12)
fig2.suptitle("exp03 — interpretable shape parameters and their assembly links",
              fontsize=11)
fig2.tight_layout()
save_fig(fig2, FIGDIR / "exp03_params")

# %% save outputs
OUTDIR.mkdir(parents=True, exist_ok=True)
pt = Table({k: fits[k] for k in keys})
pt["index"] = np.asarray(t["index"])
pt.write(OUTDIR / "radial_diffmah_params.csv", overwrite=True)
at = Table({"param": assembly_params})
for j, n in enumerate(mah):
    at[n] = R_assembly[:, j]
at.write(OUTDIR / "param_assembly_partial_corr.csv", overwrite=True)
write_manifest(OUTDIR, params={"n_use": N, "model": "radial-DiffMAH (5 params)",
                               "median_rms_dex": float(np.median(rms))})
print(f"\nwrote figures -> {FIGDIR}\nwrote outputs -> {OUTDIR}")
