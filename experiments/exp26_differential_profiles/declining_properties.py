"""exp26 — are the declining-MAH galaxies a BIASED subset, or just fewer?

Before dropping the 695 `mah_declined` galaxies from the deposition-kernel sample,
check whether they are special in their z=0.4 properties (halo mass, concentration,
SHMR, accretion, assembly, profile shape). If they only differ in ways explained
by their (slightly lower) halo mass, removing them costs statistics, not bias; if
they occupy a distinct region at FIXED mass, removing them biases the sample.

Compares declined vs non-declined within the profile-valid sample (flag & finite
CoG), reports medians + KS p-values, and the key properties controlled for halo
mass (binned medians).

Run: PYTHONPATH=. uv run python experiments/exp26_differential_profiles/declining_properties.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.table import Table
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from hongshao.plotting import set_style, save_fig, OKABE_ITO                  # noqa: E402

set_style()
FIGDIR = HERE / "figures"
TABLE = ROOT / "data" / "processed" / "tng300_072_z0p4.fits"

# (column, label, derived?) — properties to compare at z=0.4
PROPS = [
    ("logmh_z0p4", r"$\log M_h$"),
    ("logmstar", r"$\log M_*$"),
    ("shmr", r"$\log M_*/M_h$"),
    ("c_200c", r"$c_{200c}$"),
    ("acc_rate", "acc_rate"),
    ("z50", r"$z_{50}$ (assembly)"),
    ("c_to_a_3d", "c/a (3D shape)"),
    ("r50", r"$R_{50}$ [kpc]"),
    ("rdm_beta_out", r"$\beta_{\rm out}$ (profile)"),
    ("rdm_R_c", r"$R_c$ (profile)"),
]


def main():
    t = Table.read(TABLE)
    valid = np.asarray(t["flag"]) & np.isfinite(np.asarray(t["logmstar_cog"], float)).all(axis=1)
    t = t[valid]
    decl = np.asarray(t["mah_declined"])
    cols = {
        "logmh_z0p4": np.asarray(t["logmh_z0p4"], float),
        "logmstar": np.asarray(t["logmstar_cog"], float)[:, -1],
        "c_200c": np.asarray(t["c_200c"], float),
        "acc_rate": np.asarray(t["acc_rate"], float),
        "z50": np.asarray(t["z50"], float),
        "c_to_a_3d": np.asarray(t["c_to_a_3d"], float),
        "r50": np.nanmedian(np.asarray(t["r50_proj"], float), axis=1),
        "rdm_beta_out": np.asarray(t["rdm_beta_out"], float),
        "rdm_R_c": np.asarray(t["rdm_R_c"], float),
    }
    cols["shmr"] = cols["logmstar"] - cols["logmh_z0p4"]
    logmh = cols["logmh_z0p4"]

    print(f"profile-valid sample: {len(t)}   declined: {int(decl.sum())}   "
          f"non-declined: {int((~decl).sum())}\n")
    print(f"  {'property':22s} {'med(decl)':>10s} {'med(keep)':>10s} {'Delta':>8s} "
          f"{'KS p':>9s} {'Delta@fixed-Mh':>15s}")
    for key, lab in PROPS:
        v = cols[key]
        a, b = v[decl], v[~decl]
        ok_a, ok_b = np.isfinite(a), np.isfinite(b)
        ma, mb = np.nanmedian(a), np.nanmedian(b)
        ks = ks_2samp(a[ok_a], b[ok_b]).pvalue
        # mass-controlled: median property in matched logMh bins, then average the
        # per-bin declined-minus-kept difference (removes the slight mass offset).
        bins = np.percentile(logmh[np.isfinite(logmh)], np.linspace(0, 100, 9))
        diffs = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (logmh >= lo) & (logmh < hi)
            da, db = v[m & decl], v[m & ~decl]
            if np.isfinite(da).sum() >= 5 and np.isfinite(db).sum() >= 5:
                diffs.append(np.nanmedian(da) - np.nanmedian(db))
        d_fixed = np.mean(diffs) if diffs else np.nan
        star = " *" if ks < 1e-3 else ""
        print(f"  {lab:22s} {ma:>10.3f} {mb:>10.3f} {ma-mb:>+8.3f} {ks:>9.1e} "
              f"{d_fixed:>+15.3f}{star}")

    make_figure(cols, decl, logmh)
    print(f"\nwrote figure -> {FIGDIR}/exp26_declining_properties.png")
    print("\n(* = KS p<1e-3; 'Delta@fixed-Mh' is the bias that survives matching halo mass.)")


def make_figure(cols, decl, logmh):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for ax, (key, lab) in zip(axes.flat, PROPS):
        v = cols[key]
        a, b = v[decl & np.isfinite(v)], v[~decl & np.isfinite(v)]
        lo, hi = np.percentile(np.concatenate([a, b]), [1, 99])
        bins = np.linspace(lo, hi, 30)
        ax.hist(b, bins=bins, density=True, histtype="stepfilled", alpha=0.4,
                color=OKABE_ITO[0], label="non-declined")
        ax.hist(a, bins=bins, density=True, histtype="step", lw=2,
                color=OKABE_ITO[1], label="declined")
        ax.axvline(np.median(b), color=OKABE_ITO[0], ls="--", lw=1)
        ax.axvline(np.median(a), color=OKABE_ITO[1], ls="--", lw=1)
        ax.set_xlabel(lab); ax.set_yticks([])
        if key == "logmh_z0p4":
            ax.legend(fontsize=8)
    fig.suptitle("exp26 — declining-MAH (orange) vs non-declined (blue) z=0.4 properties: "
                 "where is the dropped sub-sample special?", fontsize=12)
    fig.tight_layout(); save_fig(fig, FIGDIR / "exp26_declining_properties")


if __name__ == "__main__":
    main()
