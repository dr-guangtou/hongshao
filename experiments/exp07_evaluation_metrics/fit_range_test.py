"""exp07 robustness check — does the single-sigmoid (radial-DiffMAH) residual
S-shape depend on the CoG *fitting range*?

The track-2 residual profile shows a coherent ~0.01 dex S-shape (under-fit at
2 kpc, dip ~15 kpc, bump ~50 kpc). Is that intrinsic two-component structure, or
an artifact of anchoring the fit at the innermost (2 kpc) point and extending to
148 kpc? We refit every galaxy over four radial ranges and overlay the mean
residual profiles. If the S-shape survives the inner (5 kpc) and outer (100 kpc)
cuts at the same physical radii, it is real, not a boundary effect.

Run from the repo root (EXP07_NMAX=300 for a sub-minute pass):
    PYTHONPATH=. uv run python experiments/exp07_evaluation_metrics/fit_range_test.py
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
from hongshao.tng_data import COG_RAD_KPC, cog_sigma_dex                  # noqa: E402
from hongshao.profiles import fit_cog, cog_from_physical                 # noqa: E402
from hongshao.plotting import set_style, save_fig, OKABE_ITO             # noqa: E402

set_style()
HERE = Path(__file__).resolve().parent
FIGDIR = HERE / "figures"
TABLE = ROOT / "data" / "processed" / "tng300_072_z0p4.fits"
R = np.asarray(COG_RAD_KPC, float)

t = Table.read(TABLE)
t = t[t["use"]]
NMAX = int(os.environ.get("EXP07_NMAX", 0))
if NMAX:
    t = t[:NMAX]
cog = np.asarray(t["logmstar_cog"], float)
index = np.asarray(t["index"])
ok = np.isfinite(cog).all(1)
N = len(t)

# four fitting ranges (same 24-point grid, different radial windows)
ranges = {
    "2-148 (full)": np.ones_like(R, bool),
    "5-148": R >= 5.0,
    "2-100": R <= 100.0,
    "5-100": (R >= 5.0) & (R <= 100.0),
}

resid_grid = {k: np.full((N, 24), np.nan) for k in ranges}   # residual at each radius
rms = {k: np.full(N, np.nan) for k in ranges}
chi2 = {k: np.full(N, np.nan) for k in ranges}

print(f"refitting {ok.sum()} galaxies over {list(ranges)} ...")
for i in np.where(ok)[0]:
    y = cog[i]
    sig = cog_sigma_dex(int(index[i]))
    for name, m in ranges.items():
        f = fit_cog(R[m], y[m])
        model = cog_from_physical(R[m], f["logMstar0"], f["beta_in"],
                                  f["beta_out"], f["R_c"], f["Delta"])
        r = y[m] - np.asarray(model, float)
        resid_grid[name][i, m] = r
        rms[name][i] = np.sqrt(np.mean(r ** 2))
        if np.all(np.isfinite(sig[m])) and np.all(sig[m] > 0):
            chi2[name][i] = np.sum((r / sig[m]) ** 2) / (int(m.sum()) - 5)

print("\nfit-range comparison (radial-DiffMAH single sigmoid):")
print(f"  {'range':14s}  n_pts  median RMS [dex]  median reduced chi2")
for name, m in ranges.items():
    print(f"  {name:14s}  {int(m.sum()):4d}   {np.nanmedian(rms[name]):.4f}"
          f"            {np.nanmedian(chi2[name]):.2f}")

# %% FIGURE — overlaid mean residual profiles + per-range fit quality
fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.2, 4.2))
colors = [OKABE_ITO[7], OKABE_ITO[0], OKABE_ITO[5], OKABE_ITO[2]]
markers = ["o", "s", "^", "D"]
axA.axhline(0, color="k", lw=0.8)
# baseline scatter band for visual scale
base_std = np.nanstd(resid_grid["2-148 (full)"], 0)
axA.fill_between(R, -base_std, base_std, color="0.85",
                 label="full-range $\\pm$ scatter")
for (name, m), c, mk in zip(ranges.items(), colors, markers):
    mr = np.nanmean(resid_grid[name][:, m], 0)         # only over fitted radii
    axA.plot(R[m], mr, "-", marker=mk, color=c, ms=3, lw=1.4, label=name)
axA.set_xscale("log")
axA.set_xlabel("radius R [kpc]")
axA.set_ylabel("mean residual  data $-$ model [dex]")
axA.set_title("Residual S-shape vs fitting range")
axA.legend(fontsize=7)
axA.text(-0.13, 1.04, "A", transform=axA.transAxes, fontweight="bold", fontsize=12)

names = list(ranges)
xb = np.arange(len(names))
medrms = [np.nanmedian(rms[k]) for k in names]
medchi = [np.nanmedian(chi2[k]) for k in names]
bars = axB.bar(xb, medrms, color=colors)
for xi, (rv, cv) in enumerate(zip(medrms, medchi)):
    axB.annotate(f"RMS {rv:.4f}\n$\\chi^2_\\nu$ {cv:.2f}", (xi, rv),
                 textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7)
axB.set_xticks(xb); axB.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
axB.set_ylabel("median fit RMS [dex]")
axB.set_title("Fit quality is range-insensitive")
axB.set_ylim(0, max(medrms) * 1.25)
axB.text(-0.13, 1.04, "B", transform=axB.transAxes, fontweight="bold", fontsize=12)
fig.suptitle(f"exp07 — single-sigmoid residual vs CoG fitting range (n={int(ok.sum())})",
             fontsize=11)
fig.tight_layout()
save_fig(fig, FIGDIR / "exp07_fit_range_test")
print(f"\nwrote figure -> {FIGDIR / 'exp07_fit_range_test.png'}")
