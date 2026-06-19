"""exp04 — First halo→profile model: P(profile | M0, assembly history).

We predict each galaxy's full curve of growth from halo features using
cross-validated linear regression, and ask how much halo assembly history adds
over final halo mass M0 alone — radius by radius, with a shuffle control. We do
this both for the absolute curve of growth and for its (mass-normalized) shape,
and finish with a "profile painting" demo: same-M0 halos with different
histories get different predicted profiles.

Run from the repo root:
    PYTHONPATH=. uv run python experiments/exp04_conditional_model/run.py
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
from hongshao.plotting import set_style, save_fig, OKABE_ITO   # noqa: E402
from hongshao.provenance import write_manifest                 # noqa: E402

set_style()
HERE = Path(__file__).resolve().parent
OUTDIR, FIGDIR = HERE / "outputs", HERE / "figures"
TABLE = ROOT / "data" / "processed" / "tng300_072_z0p4.fits"
R = np.asarray(COG_RAD_KPC, float)

t = Table.read(TABLE)
t = t[t["use"]]
cog = np.asarray(t["logmstar_cog"], float)                     # (N, 24)
logM0 = np.asarray(t["logm0_halo"], float)
mah_names = ["logmpeak_z0p7", "logmpeak_z1", "logmpeak_z1p5",
             "logmpeak_z2", "z50", "z75", "z90"]
MAH = np.column_stack([np.asarray(t[c], float) for c in mah_names])

mask = np.isfinite(logM0) & np.isfinite(cog).all(1) & np.isfinite(MAH).all(1)
cog, logM0, MAH = cog[mask], logM0[mask], MAH[mask]
z50 = np.asarray(t["z50"], float)[mask]
N = len(cog)
print(f"analysis sample: {N} galaxies; MAH features: {mah_names}")

shape = cog - cog[:, -1:]                                       # mass-normalized


# %% cross-validated multi-output linear prediction
def cv_predict(X, Y, k=5, seed=0):
    """Out-of-fold linear predictions of Y (N,m) from features X (N,p)."""
    n = len(Y)
    Xd = np.column_stack([np.ones(n), X])
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    pred = np.full_like(np.atleast_2d(Y), np.nan, dtype=float)
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


def rms_per_radius(Y, pred):
    return np.sqrt(np.mean((Y - pred) ** 2, axis=0))


X_base = logM0[:, None]
X_full = np.column_stack([logM0, MAH])
X_shuf = np.column_stack([logM0, shuffle_within_bins(MAH, logM0)])

results = {}
for tname, Y in [("abs", cog), ("shape", shape)]:
    pb = cv_predict(X_base, Y)
    pf = cv_predict(X_full, Y)
    ps = cv_predict(X_shuf, Y)
    results[tname] = {
        "rms_base": rms_per_radius(Y, pb), "rms_full": rms_per_radius(Y, pf),
        "rms_shuf": rms_per_radius(Y, ps), "pred_full": pf, "pred_base": pb}
    ob = float(np.sqrt(np.mean((Y - pb) ** 2)))
    of = float(np.sqrt(np.mean((Y - pf) ** 2)))
    osf = float(np.sqrt(np.mean((Y - ps) ** 2)))
    print(f"\n[{tname}] overall RMS [dex]: M0={ob:.4f}  M0+MAH={of:.4f}  "
          f"shuffled={osf:.4f}  -> improvement {100*(ob-of)/ob:.1f}% "
          f"(shuffled {100*(ob-osf)/ob:.1f}%)")

# %% FIGURE 1: per-radius prediction error, M0 vs M0+MAH vs shuffled
fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
for ax, tname, title in [(axes[0], "abs", "Full curve of growth"),
                         (axes[1], "shape", "Profile shape (mass divided out)")]:
    r = results[tname]
    ax.plot(R, r["rms_base"], "-o", color=OKABE_ITO[7], ms=3, label=r"$M_0$ only")
    ax.plot(R, r["rms_full"], "-s", color=OKABE_ITO[0], ms=3,
            label=r"$M_0$ + history")
    ax.plot(R, r["rms_shuf"], "--^", color="0.6", ms=3, label="shuffled")
    ax.set_xscale("log")
    ax.set_xlabel("radius R [kpc]")
    ax.set_ylabel("cross-validated prediction RMS [dex]")
    ax.set_title(title)
    ax.legend()
axes[0].text(-0.13, 1.04, "A", transform=axes[0].transAxes, fontweight="bold",
             fontsize=12)
axes[1].text(-0.13, 1.04, "B", transform=axes[1].transAxes, fontweight="bold",
             fontsize=12)
fig.suptitle(f"exp04 — predicting the profile from halo mass + history (n={N})",
             fontsize=11)
fig.tight_layout()
save_fig(fig, FIGDIR / "exp04_prediction_error")

# %% FIGURE 2: per-radius improvement + profile-painting demo
fig2, (axC, axD) = plt.subplots(1, 2, figsize=(9.2, 4.0))
for tname, color, lab in [("abs", OKABE_ITO[5], "full CoG"),
                          ("shape", OKABE_ITO[2], "shape")]:
    r = results[tname]
    impr = np.where(r["rms_base"] > 1e-6,
                    100 * (r["rms_base"] - r["rms_full"]) / r["rms_base"], np.nan)
    axC.plot(R, impr, "-o", color=color, ms=3, label=lab)
axC.axhline(0, color="k", lw=0.8)
axC.set_xscale("log")
axC.set_xlabel("radius R [kpc]")
axC.set_ylabel("scatter reduction from history [%]")
axC.set_title("Where assembly history helps")
axC.legend()
axC.text(-0.13, 1.04, "C", transform=axC.transAxes, fontweight="bold", fontsize=12)

# painting demo: a narrow M0 bin, 3 galaxies spanning formation time z50
sel = np.where((logM0 > 13.4) & (logM0 < 13.6))[0]
order = sel[np.argsort(z50[sel])]
picks = [order[0], order[len(order) // 2], order[-1]]   # late, mid, early former
pf = results["shape"]["pred_full"]
pb = results["shape"]["pred_base"]
labels = ["late former (low z50)", "median", "early former (high z50)"]
for idx, lab, col in zip(picks, labels, [OKABE_ITO[4], OKABE_ITO[7], OKABE_ITO[5]]):
    axD.plot(R, shape[idx], "o", color=col, ms=4)
    axD.plot(R, pf[idx], "-", color=col, lw=1.5, label=f"{lab} (z50={z50[idx]:.2f})")
axD.plot(R, pb[picks[0]], ":", color="k", lw=1.4, label=r"$M_0$-only (same for all)")
axD.set_xscale("log")
axD.set_xlabel("radius R [kpc]")
axD.set_ylabel(r"shape  $\log[M_*(<R)/M_*(<R_{\max})]$")
axD.set_title("Painting: same $M_0$, different history")
axD.legend(fontsize=7)
axD.text(-0.13, 1.04, "D", transform=axD.transAxes, fontweight="bold", fontsize=12)
fig2.suptitle("exp04 — assembly history shifts the predicted profile", fontsize=11)
fig2.tight_layout()
save_fig(fig2, FIGDIR / "exp04_painting")

# %% save outputs
OUTDIR.mkdir(parents=True, exist_ok=True)
out = Table({"R_kpc": R})
for tname in ("abs", "shape"):
    for key in ("rms_base", "rms_full", "rms_shuf"):
        out[f"{tname}_{key}"] = results[tname][key]
out.write(OUTDIR / "prediction_rms_per_radius.csv", overwrite=True)
write_manifest(OUTDIR, params={"n": N, "mah_features": mah_names,
                               "model": "5-fold CV multi-output linear"})
print(f"\nwrote figures -> {FIGDIR}\nwrote outputs -> {OUTDIR}")
