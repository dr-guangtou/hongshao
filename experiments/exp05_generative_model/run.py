"""exp05 — Generative profile model: predict the 5 radial-DiffMAH parameters
from halo features, then reconstruct guaranteed-valid (monotonic) profiles.

Unlike exp04 (which predicted the 24 curve-of-growth points directly), here we
predict the compact parameter vector theta_prof = (logMstar0, beta_in, beta_out,
log R_c, Delta) and rebuild the profile from it. This gives a true generative
"painting" model. We check: (1) is generating-via-parameters as accurate as
direct point prediction? (2) which parameters does assembly history help predict?
(3) does the model paint realistic profile diversity at fixed halo mass?

Run from the repo root:
    PYTHONPATH=. uv run python experiments/exp05_generative_model/run.py
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
from hongshao.tng_data import COG_RAD_KPC                       # noqa: E402
from hongshao.profiles import cog_from_physical                # noqa: E402
from hongshao.plotting import set_style, save_fig, OKABE_ITO   # noqa: E402
from hongshao.provenance import write_manifest                 # noqa: E402

set_style()
HERE = Path(__file__).resolve().parent
OUTDIR, FIGDIR = HERE / "outputs", HERE / "figures"
TABLE = ROOT / "data" / "processed" / "tng300_072_z0p4.fits"
R = np.asarray(COG_RAD_KPC, float)

t = Table.read(TABLE)
if "rdm_Delta" not in t.colnames:
    raise SystemExit("Table lacks rdm_* columns; rebuild with "
                     "`uv run python -m hongshao.tng_data --full --out ...`")
t = t[t["use"]]

# profile parameter vector (R_c in dex for a well-scaled regression)
theta_names = ["rdm_logMstar0", "rdm_beta_in", "rdm_beta_out", "rdm_R_c", "rdm_Delta"]
theta = np.column_stack([np.asarray(t[c], float) for c in theta_names])
theta[:, 3] = np.log10(theta[:, 3])                            # log R_c
cog = np.asarray(t["logmstar_cog"], float)
logM0 = np.asarray(t["logm0_halo"], float)
mah_names = ["logmpeak_z0p7", "logmpeak_z1", "logmpeak_z1p5",
             "logmpeak_z2", "z50", "z75", "z90"]
MAH = np.column_stack([np.asarray(t[c], float) for c in mah_names])

mask = (np.isfinite(theta).all(1) & np.isfinite(cog).all(1)
        & np.isfinite(logM0) & np.isfinite(MAH).all(1))
theta, cog, logM0, MAH = theta[mask], cog[mask], logM0[mask], MAH[mask]
z50 = np.asarray(t["z50"], float)[mask]
N = len(theta)
print(f"analysis sample: {N} galaxies; theta = {theta_names}")


# %% CV machinery
def cv_predict(X, Y, k=5, seed=0):
    n = len(Y)
    Xd = np.column_stack([np.ones(n), X])
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    pred = np.full(np.atleast_2d(Y).shape if Y.ndim > 1 else (n,), np.nan, float)
    for fold in np.array_split(idx, k):
        tr = np.setdiff1d(np.arange(n), fold)
        beta, *_ = np.linalg.lstsq(Xd[tr], Y[tr], rcond=None)
        pred[fold] = Xd[fold] @ beta
    return pred


def shuffle_within_bins(M, binvar, nbins=12, seed=1):
    rng = np.random.default_rng(seed)
    out = M.copy()
    edges = np.quantile(binvar, np.linspace(0, 1, nbins + 1))
    b = np.digitize(binvar, edges[1:-1])
    for bi in np.unique(b):
        idx = np.where(b == bi)[0]
        out[idx] = M[idx][rng.permutation(len(idx))]
    return out


def gen_cog(theta_pred):
    """Reconstruct log10 M*(<R) from predicted theta (R_c stored as dex)."""
    return cog_from_physical(R, theta_pred[:, 0], theta_pred[:, 1],
                             theta_pred[:, 2], 10 ** theta_pred[:, 3], theta_pred[:, 4])


X_base = logM0[:, None]
X_full = np.column_stack([logM0, MAH])
X_shuf = np.column_stack([logM0, shuffle_within_bins(MAH, logM0)])

# %% predict theta, reconstruct CoG, and (for reference) predict CoG directly
theta_base, theta_full = cv_predict(X_base, theta), cv_predict(X_full, theta)
theta_shuf = cv_predict(X_shuf, theta)
gen_base, gen_full, gen_shuf = gen_cog(theta_base), gen_cog(theta_full), gen_cog(theta_shuf)
direct_full = cv_predict(X_full, cog)                          # exp04-style baseline


def overall(pred):
    return float(np.sqrt(np.mean((cog - pred) ** 2)))


print(f"\nreconstructed-CoG RMS [dex]:  generative M0={overall(gen_base):.4f}  "
      f"generative M0+MAH={overall(gen_full):.4f}  shuffled={overall(gen_shuf):.4f}")
print(f"   improvement {100*(overall(gen_base)-overall(gen_full))/overall(gen_base):.1f}%  "
      f"(direct point-prediction M0+MAH = {overall(direct_full):.4f})")

# per-parameter prediction quality (CV RMS), M0 vs M0+MAH
print("\nper-parameter CV RMS:   M0      M0+MAH    improvement")
param_impr = []
for j, nm in enumerate(theta_names):
    rb = np.sqrt(np.mean((theta[:, j] - theta_base[:, j]) ** 2))
    rf = np.sqrt(np.mean((theta[:, j] - theta_full[:, j]) ** 2))
    param_impr.append(100 * (rb - rf) / rb)
    print(f"  {nm:16s} {rb:.4f}  {rf:.4f}   {param_impr[-1]:+.1f}%")

# %% FIGURE 1: generative accuracy
fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.2, 4.0))
rms_r = lambda p: np.sqrt(np.mean((cog - p) ** 2, axis=0))
axA.plot(R, rms_r(gen_base), "-o", color=OKABE_ITO[7], ms=3, label=r"generative $M_0$")
axA.plot(R, rms_r(gen_full), "-s", color=OKABE_ITO[0], ms=3, label=r"generative $M_0$+history")
axA.plot(R, rms_r(direct_full), ":", color=OKABE_ITO[5], lw=1.6,
         label="direct points (exp04)")
axA.plot(R, rms_r(gen_shuf), "--^", color="0.6", ms=3, label="shuffled")
axA.set_xscale("log")
axA.set_xlabel("radius R [kpc]")
axA.set_ylabel("reconstructed-CoG RMS [dex]")
axA.set_title("Generating via 5 params ≈ direct prediction")
axA.legend(fontsize=7)
axA.text(-0.13, 1.04, "A", transform=axA.transAxes, fontweight="bold", fontsize=12)

short = [n.replace("rdm_", "") for n in theta_names]
short[3] = "log R_c"
axB.bar(range(len(short)), param_impr, color=OKABE_ITO[2])
axB.axhline(0, color="k", lw=0.8)
axB.set_xticks(range(len(short)))
axB.set_xticklabels(short, rotation=30, ha="right")
axB.set_ylabel("scatter reduction from history [%]")
axB.set_title("Which parameters history helps predict")
axB.text(-0.13, 1.04, "B", transform=axB.transAxes, fontweight="bold", fontsize=12)
fig.suptitle(f"exp05 — generative profile model (n={N})", fontsize=11)
fig.tight_layout()
save_fig(fig, FIGDIR / "exp05_accuracy")

# %% FIGURE 2: painting + diversity
fig2, (axC, axD) = plt.subplots(1, 2, figsize=(9.2, 4.0))
sel = np.where((logM0 > 13.4) & (logM0 < 13.6))[0]
order = sel[np.argsort(z50[sel])]
picks = [order[0], order[len(order) // 2], order[-1]]
labels = ["late former", "median", "early former"]
for idx, lab, col in zip(picks, labels, [OKABE_ITO[4], OKABE_ITO[7], OKABE_ITO[5]]):
    axC.plot(R, cog[idx], "o", color=col, ms=4)
    axC.plot(R, gen_full[idx], "-", color=col, lw=1.5,
             label=f"{lab} (z50={z50[idx]:.2f})")
axC.plot(R, gen_base[picks[0]], ":", color="k", lw=1.4, label=r"$M_0$-only (all alike)")
axC.set_xscale("log")
axC.set_xlabel("radius R [kpc]")
axC.set_ylabel(r"$\log_{10} M_*(<R)$  (points=truth, lines=generated)")
axC.set_title("Generated profiles: same $M_0$, different history")
axC.legend(fontsize=7)
axC.text(-0.13, 1.04, "C", transform=axC.transAxes, fontweight="bold", fontsize=12)

# diversity: per-radius scatter of profiles within the narrow M0 bin
sh = lambda Y: (Y - Y[:, -1:])[sel]                # shape, bin only
std_true = sh(cog).std(0)
std_full = sh(gen_full).std(0)
std_base = sh(gen_base).std(0)
axD.plot(R, std_true, "-o", color=OKABE_ITO[5], ms=3, label="true diversity")
axD.plot(R, std_full, "-s", color=OKABE_ITO[0], ms=3, label=r"generated ($M_0$+history)")
axD.plot(R, std_base, "--^", color=OKABE_ITO[7], ms=3, label=r"generated ($M_0$ only)")
axD.set_xscale("log")
axD.set_xlabel("radius R [kpc]")
axD.set_ylabel("profile-shape scatter in M0 bin [dex]")
axD.set_title("Does it paint realistic diversity?")
axD.legend(fontsize=7)
axD.text(-0.13, 1.04, "D", transform=axD.transAxes, fontweight="bold", fontsize=12)
fig2.suptitle("exp05 — profile painting and diversity at fixed halo mass", fontsize=11)
fig2.tight_layout()
save_fig(fig2, FIGDIR / "exp05_painting")

# %% save outputs
OUTDIR.mkdir(parents=True, exist_ok=True)
res = Table({"param": theta_names, "improvement_pct": param_impr})
res.write(OUTDIR / "param_prediction.csv", overwrite=True)
write_manifest(OUTDIR, params={
    "n": N, "model": "predict radial-DiffMAH params -> reconstruct CoG",
    "gen_rms_full": overall(gen_full), "gen_rms_base": overall(gen_base),
    "direct_rms_full": overall(direct_full)})
print(f"\nwrote figures -> {FIGDIR}\nwrote outputs -> {OUTDIR}")
