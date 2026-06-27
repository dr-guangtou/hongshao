"""exp27 step 4 — reconcile the TWO DiffMAH flavours and build one combined,
unambiguous catalog.

Two DiffMAH fits exist for these galaxies, same model (Hearin+2021 rolling power
law, k=3.5 fixed) but different anchor time t0:

  OWN  (hongshao.diffmah, data/processed/tng300_072_z0p4.fits, cols dmah_*)
       fit to the running-max main-branch Mpeak up to z=0.4, anchored at
       t0 = t(snap 72) = 9.39 Gyr -> `logmp` is the peak mass AT z=0.4.
       Covers 3375/3388 (13 have too few MAH points).
  OFFICIAL (diffmah_tng.h5, attached via crossmatch.fits, cols logmp_fit/...)
       fit to the full main branch to z=0, anchored at t0 = 13.80 Gyr ->
       `logmp_fit` is the peak mass AT z=0. Covers the 3154 matched galaxies.

The shape params (logtc, early, late) are anchor-independent and compare directly.
`logmp` is NOT comparable across anchors, so for the consistency test and the
merged normalization we re-anchor the official fit to z=0.4 by reading its own
reconstructed curve at snap 72 (`diffmah_log_mah_fit[:,72]`).

This script (a) quantifies own-vs-official consistency on the 3154 common galaxies
(curve RMS over z>=0.4 + per-param scatter), saves a figure, and (b) writes
`outputs/diffmah_combined.fits` with `own_*` / `official_*` columns plus a
`diffmah_source` ('official'|'own'|'none') and recommended anchor-independent
shape params + a z=0.4-anchored normalization (official preferred, own fallback).

Run: PYTHONPATH=. uv run python experiments/exp27_tng_api_crossmatch/diffmah_consistency.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
from astropy.table import Table

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from hongshao.diffmah import log_mah, MAH_K  # noqa: E402
from hongshao.plotting import set_style, save_fig, OKABE_ITO  # noqa: E402

set_style()
OUT, FIG = HERE / "outputs", HERE / "figures"
PROC = ROOT / "data" / "processed" / "tng300_072_z0p4.fits"
SNAP_Z04 = 72
FIT_TMIN_GYR = 2.0          # same inner cut our fit used


def main() -> None:
    proc = Table.read(PROC)
    cm = Table.read(OUT / "crossmatch.fits")
    mah = np.load(OUT / "official_mah.npz")
    tsnap = mah["cosmic_time_gyr"]                 # Gyr, index == snapshot
    assert tsnap.size == 100
    n = len(proc)

    own_ok = np.isfinite(np.asarray(proc["dmah_logmp"]))
    matched = np.asarray(cm["matched"])

    # --- reconstruct the OWN MAH curve on the snapshot grid ------------------
    logt = np.log10(np.where(tsnap > 0, tsnap, np.nan))
    logt0 = np.log10(tsnap[SNAP_Z04])
    own_curve = np.full((n, 100), np.nan)
    okp = own_ok
    own_curve[okp] = log_mah(
        logt, np.asarray(proc["dmah_logmp"])[okp], np.asarray(proc["dmah_logtc"])[okp],
        np.asarray(proc["dmah_early"])[okp], np.asarray(proc["dmah_late"])[okp],
        logt0=logt0, k=MAH_K)
    off_curve = mah["diffmah_log_mah_fit"]         # NaN where unmatched

    # --- consistency on the common set, over the fitted range z>=0.4 ---------
    common = own_ok & matched
    snaps = np.arange(100)
    rng = (tsnap >= FIT_TMIN_GYR) & (snaps <= SNAP_Z04)    # z>=0.4, resolved
    diff = (own_curve - off_curve)[:, rng]
    rms = np.sqrt(np.nanmean(diff[common] ** 2, axis=1))   # per-galaxy dex
    print(f"common compare set: {common.sum()} galaxies")
    print(f"curve RMS over z>=0.4 (dex): median {np.nanmedian(rms):.3f}, "
          f"90th {np.nanpercentile(rms, 90):.3f}, "
          f"frac<0.05dex {np.mean(rms < 0.05):.2f}, frac<0.10dex {np.mean(rms < 0.10):.2f}")

    # official re-anchored to z=0.4 (its own curve at snap 72)
    off_logmp_z04 = off_curve[:, SNAP_Z04]
    pairs = {  # label: (own col, official value)  -- shape params compare directly
        "logmp (z=0.4)": (np.asarray(proc["dmah_logmp"]), off_logmp_z04),
        "logtc": (np.asarray(proc["dmah_logtc"]), np.asarray(cm["mah_logtc"])),
        "early_index": (np.asarray(proc["dmah_early"]), np.asarray(cm["early_index"])),
        "late_index": (np.asarray(proc["dmah_late"]), np.asarray(cm["late_index"])),
    }
    print("\nparam own-vs-official (common set): median(off-own), scatter(MAD*1.4826)")
    for lab, (a, b) in pairs.items():
        d = (b - a)[common]
        d = d[np.isfinite(d)]
        print(f"  {lab:14s} dmed {np.median(d):+.3f}  scatter {1.4826*np.median(np.abs(d-np.median(d))):.3f}")

    # --- figure: 4 param scatters + curve-RMS hist + example overlays --------
    fig, ax = plt.subplots(2, 3, figsize=(13, 8))
    for k, (lab, (a, b)) in enumerate(pairs.items()):
        axx = ax.flat[k]
        x, y = a[common], b[common]
        axx.scatter(x, y, s=3, alpha=0.25, color=OKABE_ITO[4], edgecolors="none")
        lim = [np.nanpercentile(np.r_[x, y], 1), np.nanpercentile(np.r_[x, y], 99)]
        axx.plot(lim, lim, "k--", lw=1)
        axx.set(xlabel=f"own  {lab}", ylabel=f"official  {lab}", xlim=lim, ylim=lim)
    # curve RMS histogram
    ax.flat[4].hist(rms[np.isfinite(rms)], bins=40, color=OKABE_ITO[5])
    ax.flat[4].axvline(0.05, ls="--", c="k", lw=1)
    ax.flat[4].set(xlabel="own-vs-official MAH curve RMS [dex], z>=0.4",
                   ylabel="galaxies", title=f"median {np.nanmedian(rms):.3f} dex")
    # a few example MAH overlays
    axx = ax.flat[5]
    ex = np.where(common)[0][::common.sum() // 5][:5]
    for j, i in enumerate(ex):
        c = OKABE_ITO[j]
        axx.plot(tsnap, own_curve[i], "-", c=c, lw=1.3)
        axx.plot(tsnap, off_curve[i], "--", c=c, lw=1.1)
    axx.axvline(tsnap[SNAP_Z04], ls=":", c="grey", lw=1)
    axx.plot([], [], "k-", label="own (to z=0.4)")
    axx.plot([], [], "k--", label="official (to z=0)")
    axx.set(xlabel="cosmic time [Gyr]", ylabel=r"$\log M\ [M_\odot]$",
            title="5 example MAHs", ylim=(11, 15.4))
    axx.legend(fontsize=7)
    fig.suptitle("DiffMAH: our z=0.4 fit vs official z=0 fit (3154 common)", y=1.0)
    fig.tight_layout()
    print("\nwrote", save_fig(fig, FIG / "diffmah_own_vs_official")[0])

    # --- combined catalog ----------------------------------------------------
    halo_id = np.asarray(cm["halo_id"])
    t = Table()
    t["index"] = proc["index"]
    t["subhalo_id_snap72"] = cm["subhalo_id_snap72"]
    # own fit (z=0.4 anchored)
    t["own_ok"] = own_ok
    t["own_logmp_z0p4"] = proc["dmah_logmp"]
    t["own_logtc"] = proc["dmah_logtc"]
    t["own_early"] = proc["dmah_early"]
    t["own_late"] = proc["dmah_late"]
    t["own_rms"] = proc["dmah_rms"]
    # official fit (native z=0 anchored) + re-anchored z=0.4 normalization
    t["official_matched"] = matched
    t["official_halo_id"] = halo_id
    t["official_logmp_z0"] = cm["logmp_fit"]
    t["official_logmp_z0p4"] = off_logmp_z04.astype(np.float32)
    t["official_logtc"] = cm["mah_logtc"]
    t["official_early"] = cm["early_index"]
    t["official_late"] = cm["late_index"]
    t["official_loss"] = cm["diffmah_loss"]
    # recommended: official preferred, own fallback. shape params are anchor-free;
    # the normalization is the z=0.4-anchored peak mass (apples-to-apples).
    src = np.where(matched, "official", np.where(own_ok, "own", "none"))
    t["diffmah_source"] = src
    pick_off = matched

    def merge(off, own):
        out = np.where(pick_off, off, np.where(own_ok, own, np.nan))
        return out

    t["diffmah_logmp_z0p4"] = merge(off_logmp_z04, np.asarray(proc["dmah_logmp"]))
    t["diffmah_logtc"] = merge(np.asarray(cm["mah_logtc"]), np.asarray(proc["dmah_logtc"]))
    t["diffmah_early"] = merge(np.asarray(cm["early_index"]), np.asarray(proc["dmah_early"]))
    t["diffmah_late"] = merge(np.asarray(cm["late_index"]), np.asarray(proc["dmah_late"]))
    t.meta["MAH_K"] = MAH_K
    t.meta["comment"] = ("DiffMAH combined. own_*: hongshao z=0.4-anchored fit (Mpeak"
                         " to z=0.4). official_*: diffmah_tng.h5 z=0-anchored fit."
                         " diffmah_*: recommended (official if matched else own);"
                         " shape params anchor-free, logmp at z=0.4 anchor.")
    t.write(OUT / "diffmah_combined.fits", overwrite=True)
    nsrc = {s: int((src == s).sum()) for s in ("official", "own", "none")}
    print(f"wrote {OUT/'diffmah_combined.fits'}  sources: {nsrc}")


if __name__ == "__main__":
    sys.exit(main())
