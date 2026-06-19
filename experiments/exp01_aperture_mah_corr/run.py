"""exp01 — Which radial zones of the central galaxy remember which epochs of
halo growth?

At fixed final halo mass M0 = Mpeak(z=0.4), we ask whether the stellar mass in
each radial zone correlates with how much halo mass was already in place at
earlier epochs (and with halo formation redshift). The control for M0 is a
partial (rank) correlation; we then test whether adding halo-history information
measurably reduces the scatter in predicting inner vs outer stellar mass, with a
shuffle test to confirm any gain is real.

Run from the repo root:
    PYTHONPATH=. uv run python experiments/exp01_aperture_mah_corr/run.py
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
from scipy.stats import t as tdist

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hongshao.tng_data import SMA_KPC, COG_RAD_KPC  # noqa: E402
from hongshao.provenance import write_manifest       # noqa: E402

HERE = Path(__file__).resolve().parent
OUTDIR = HERE / "outputs"
FIGDIR = HERE / "figures"
TABLE = ROOT / "data" / "processed" / "tng300_072_z0p4.fits"

t = Table.read(TABLE)
t = t[t["use"]]
print(f"analysis sample (use cut): {len(t)} galaxies")

logM0 = np.asarray(t["logm0_halo"], float)

# %% galaxy-side: differential radial zones (log shell mass, Msun)
aper_lin = 10.0 ** np.asarray(t["logmstar_aper"], float)   # (N,7) cumulative
zone_edges = [(0, SMA_KPC[0])] + [(SMA_KPC[i], SMA_KPC[i + 1])
                                  for i in range(len(SMA_KPC) - 1)]
zone_labels = [f"<{int(SMA_KPC[0])}"] + [f"{int(a)}-{int(b)}"
                                         for a, b in zone_edges[1:]]
shell = np.empty_like(aper_lin)
shell[:, 0] = aper_lin[:, 0]
shell[:, 1:] = np.diff(aper_lin, axis=1)
with np.errstate(divide="ignore", invalid="ignore"):
    log_shell = np.log10(np.where(shell > 0, shell, np.nan))   # (N, 7 zones)

# galaxy-side: cumulative CoG (log Mstar(<R)), 24 radii
log_cog = np.asarray(t["logmstar_cog"], float)

# %% halo-side MAH features
mah_features = {
    "Mpeak(z=0.7)": np.asarray(t["logmpeak_z0p7"], float),
    "Mpeak(z=1)": np.asarray(t["logmpeak_z1"], float),
    "Mpeak(z=1.5)": np.asarray(t["logmpeak_z1p5"], float),
    "Mpeak(z=2)": np.asarray(t["logmpeak_z2"], float),
    "z50": np.asarray(t["z50"], float),
    "z75": np.asarray(t["z75"], float),
    "z90": np.asarray(t["z90"], float),
}
epoch_features = {k: mah_features[k] for k in
                  ["Mpeak(z=0.7)", "Mpeak(z=1)", "Mpeak(z=1.5)", "Mpeak(z=2)"]}


# %% partial Spearman correlation controlling for Z (= log M0)
def partial_spearman(x, y, z):
    """Rank partial correlation of x,y controlling for z. Returns (r, p, n)."""
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[m], y[m], z[m]
    n = len(x)
    if n < 20:
        return np.nan, np.nan, n
    rxy = spearmanr(x, y)[0]
    rxz = spearmanr(x, z)[0]
    ryz = spearmanr(y, z)[0]
    denom = np.sqrt((1 - rxz**2) * (1 - ryz**2))
    if denom == 0:
        return np.nan, np.nan, n
    r = np.clip((rxy - rxz * ryz) / denom, -1, 1)
    df = n - 3
    if df <= 0 or abs(r) >= 1:
        return r, np.nan, n
    tstat = r * np.sqrt(df / (1 - r**2))
    p = float(2 * tdist.sf(abs(tstat), df))
    return float(r), p, n


def corr_matrix(rows, row_vals, cols, col_vals, z):
    R = np.full((len(rows), len(cols)), np.nan)
    P = np.full_like(R, np.nan)
    for i, rv in enumerate(row_vals):
        for j, cv in enumerate(col_vals):
            R[i, j], P[i, j], _ = partial_spearman(rv, cv, z)
    return R, P


zone_vals = [log_shell[:, k] for k in range(log_shell.shape[1])]
R_zone, P_zone = corr_matrix(zone_labels, zone_vals,
                             list(mah_features), list(mah_features.values()), logM0)

cog_vals = [log_cog[:, k] for k in range(log_cog.shape[1])]
R_cog, P_cog = corr_matrix([f"{r:.0f}" for r in COG_RAD_KPC], cog_vals,
                           list(epoch_features), list(epoch_features.values()), logM0)


# %% heatmap helper
def heatmap(ax, M, P, row_labels, col_labels, title, ylabel):
    vmax = np.nanmax(np.abs(M))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                star = "*" if (np.isfinite(P[i, j]) and P[i, j] < 0.001) else ""
                ax.text(j, i, f"{M[i, j]:+.2f}{star}", ha="center", va="center",
                        fontsize=6, color="k")
    return im


# headline figure: radial zones x MAH features
fig, ax = plt.subplots(figsize=(7.5, 4.8))
im = heatmap(ax, R_zone, P_zone, zone_labels, list(mah_features),
             "Partial corr. of zone stellar mass with halo history\n"
             "(at fixed final halo mass M0; * = p<0.001)",
             "radial zone [kpc]")
fig.colorbar(im, ax=ax, label="partial Spearman r")
ax.set_xlabel("halo assembly feature")
fig.tight_layout()
FIGDIR.mkdir(parents=True, exist_ok=True)
fig.savefig(FIGDIR / "exp01_radius_epoch_map.png", dpi=140)

# companion: cumulative CoG x epoch
fig2, ax2 = plt.subplots(figsize=(6.5, 7))
im2 = heatmap(ax2, R_cog, P_cog, [f"{r:.0f}" for r in COG_RAD_KPC],
              list(epoch_features),
              "Partial corr. of M*(<R) with Mpeak(z)\n(at fixed M0)",
              "aperture radius R [kpc]")
fig2.colorbar(im2, ax=ax2, label="partial Spearman r")
ax2.set_xlabel("halo mass epoch")
fig2.tight_layout()
fig2.savefig(FIGDIR / "exp01_cog_epoch_map.png", dpi=140)


# %% does halo history reduce scatter at fixed M0?  (5-fold cross-validation)
def cv_rmse(feature_cols, y, k=5, seed=0):
    """Cross-validated RMSE of a linear fit. All inputs already finite."""
    X = np.column_stack([np.ones(len(y))] + list(feature_cols))
    n = len(y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    pred = np.full(n, np.nan)
    for fold in np.array_split(idx, k):
        tr = np.setdiff1d(np.arange(n), fold)
        beta, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
        pred[fold] = X[fold] @ beta
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def shuffle_within_bins(values, binvar, nbins=12, seed=1):
    """Permute values among halos in the same M0 bin (breaks the MAH link)."""
    rng = np.random.default_rng(seed)
    out = values.copy()
    edges = np.quantile(binvar, np.linspace(0, 1, nbins + 1))
    b = np.digitize(binvar, edges[1:-1])
    for bi in np.unique(b):
        idx = np.where(b == bi)[0]
        out[idx] = values[idx][rng.permutation(len(idx))]
    return out


inner = np.asarray(t["logmstar_aper"], float)[:, 0]                  # log M*(<10 kpc)
with np.errstate(divide="ignore", invalid="ignore"):
    outer = np.log10(aper_lin[:, 4] - aper_lin[:, 2])                # log M*(50-100 kpc)

summaries = {  # MAH info added on top of M0
    "Mpeak(z=1)": mah_features["Mpeak(z=1)"],
    "Mpeak(z=2)": mah_features["Mpeak(z=2)"],
    "z50": mah_features["z50"],
    "z90": mah_features["z90"],
}

scatter_rows = []
for tname, y_full in [("M*(<10kpc) inner", inner), ("M*(50-100kpc) outer", outer)]:
    mask = np.isfinite(y_full) & np.isfinite(logM0)
    for s in summaries.values():
        mask &= np.isfinite(s)
    y = y_full[mask]
    m0 = logM0[mask]
    sm = {k: v[mask] for k, v in summaries.items()}       # masked, all finite
    base = cv_rmse([m0], y)
    allf = cv_rmse([m0] + list(sm.values()), y)
    shuf = [shuffle_within_bins(s, m0) for s in sm.values()]   # control
    allf_shuf = cv_rmse([m0] + shuf, y)
    row = {"target": tname, "n": int(mask.sum()), "rmse_M0": base,
           "rmse_M0+MAH": allf, "rmse_M0+MAH_shuffled": allf_shuf,
           "pct_reduction": 100 * (base - allf) / base,
           "pct_reduction_shuffled": 100 * (base - allf_shuf) / base}
    for sname, s in sm.items():
        row[f"rmse_M0+{sname}"] = cv_rmse([m0, s], y)
    scatter_rows.append(row)

scatter = Table(scatter_rows)
print("\n=== scatter reduction (cross-validated RMSE in dex) ===")
scatter[["target", "n", "rmse_M0", "rmse_M0+MAH",
         "rmse_M0+MAH_shuffled", "pct_reduction",
         "pct_reduction_shuffled"]].pprint(max_width=-1)


# %% scatter-reduction figure
fig3, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, row in zip(axes, scatter_rows):
    labels = ["M0", "+Mpeak(z=1)", "+Mpeak(z=2)", "+z50", "+z90",
              "+all", "+all (shuffled)"]
    vals = [row["rmse_M0"], row["rmse_M0+Mpeak(z=1)"], row["rmse_M0+Mpeak(z=2)"],
            row["rmse_M0+z50"], row["rmse_M0+z90"], row["rmse_M0+MAH"],
            row["rmse_M0+MAH_shuffled"]]
    colors = ["0.5", "C0", "C0", "C0", "C0", "C2", "C3"]
    ax.bar(range(len(vals)), vals, color=colors)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("cross-validated RMSE [dex]", fontsize=9)
    ax.set_title(row["target"], fontsize=10)
    ax.set_ylim(0, max(vals) * 1.15)
fig3.suptitle("Predicting stellar mass: halo mass alone vs. + assembly history",
              fontsize=11)
fig3.tight_layout()
fig3.savefig(FIGDIR / "exp01_scatter_reduction.png", dpi=140)


# %% save outputs + provenance
OUTDIR.mkdir(parents=True, exist_ok=True)
np.savez(OUTDIR / "partial_corr.npz",
         R_zone=R_zone, P_zone=P_zone, zone_labels=zone_labels,
         mah_features=list(mah_features),
         R_cog=R_cog, P_cog=P_cog, cog_radii=COG_RAD_KPC,
         epoch_features=list(epoch_features))
scatter.write(OUTDIR / "scatter_reduction.csv", overwrite=True)

# zone x feature correlation table for easy reading
zt = Table({"zone_kpc": zone_labels})
for j, name in enumerate(mah_features):
    zt[name] = R_zone[:, j]
zt.write(OUTDIR / "partial_corr_zones.csv", overwrite=True)

write_manifest(OUTDIR, params={
    "n_use": int(len(t)), "table": str(TABLE.relative_to(ROOT)),
    "control": "log M0 = logmpeak_z0p4", "method": "partial Spearman; 5-fold CV",
})
print(f"\nwrote figures -> {FIGDIR}")
print(f"wrote outputs -> {OUTDIR}")
