"""exp28 — all MAH flavours from a full SubLink tree, in one place, plus the
DiffMAH model, for direct comparison and reuse.

`mah_flavours(tree_path)` returns every flavour snapshot-aligned (0..72) in h-free
Msun. All tree flavours use **SubhaloMass** (the total bound mass of each SUBFIND
subhalo) — the same quantity the official DiffMAH was fit to (verified: the catalog
`log_mah_sim` equals the tree main-branch SubhaloMass Mpeak to <0.005 dex), so the
tree flavours and DiffMAH are on one mass definition. `main_m200c` is carried too
(the FoF spherical-overdensity halo mass) for reference.

Flavours (each a length-73 array, NaN where undefined):
  main_raw        main-branch SubhaloMass (FirstProgenitor chain) — has tree
                  dropouts / switching spikes
  main_mpeak      running-max of main_raw (== DiffMAH's input `log_mah_sim`)
  max_prog        most-massive single progenitor per snapshot (defect-robust main
                  branch: picks the real big object where main_raw drops out)
  max_prog_mpeak  running-max of max_prog (a clean "repaired central-halo" MAH)
  summed          Σ SubhaloMass over ALL progenitors per snapshot (total resolved
                  collapsed mass; leads in time; merger-accreted only)
  infall_peak     cumulative Σ of each merged satellite's pre-infall peak mass,
                  added at its merger snapshot (monotonic merger-delivered budget;
                  lags in time; excludes smooth accretion)
  main_m200c      main-branch Group_M_Crit200 (reference halo mass)
  n_prog          progenitor count per snapshot

See README and `doc/summed_accreted_mah.md` for the physical meaning of each.

Run: PYTHONPATH=. uv run python experiments/exp28_summed_accreted_mah/mah_flavours.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from hongshao.diffmah import log_mah, MAH_K  # noqa: E402
from hongshao.plotting import set_style, save_fig, OKABE_ITO  # noqa: E402

set_style()
OUT, FIG = HERE / "outputs", HERE / "figures"
TREE_DIR = Path(os.environ.get("HONGSHAO_TNG_WORK", "/Users/mac/work/tng")) / "exp28_full_trees"
COSMIC_TIME = ROOT / "data" / "external" / "tng_cosmic_time.txt"
PROC = ROOT / "data" / "processed" / "tng300_072_z0p4.fits"
OFFICIAL_MAH = ROOT / "experiments/exp27_tng_api_crossmatch/outputs/official_mah.npz"
H = 0.6774
U = 1e10 / H                         # raw (1e10 Msun/h) -> Msun
ROOT_SNAP = 72
N = ROOT_SNAP + 1

# example halos (logMh ~ 13.5, NOT the most massive): label -> (subhalo_id, index)
EXAMPLES = {"clean": (283371, 301), "declining": (293978, 323)}


def _rmax(a: np.ndarray) -> np.ndarray:
    """running maximum over time (snapshot index), preserving NaN gaps."""
    out = np.full_like(a, np.nan)
    m = np.isfinite(a)
    if m.any():
        out[m] = np.maximum.accumulate(np.where(m, a, -np.inf))[m]
    return out


def mah_flavours(tree_path: Path) -> dict:
    """All MAH flavours (Msun, snapshot-aligned 0..72) for one full SubLink tree."""
    with h5py.File(tree_path, "r") as f:
        snap = f["SnapNum"][:]
        sub = f["SubhaloMass"][:] * U
        m200 = f["Group_M_Crit200"][:] * U
        sid = f["SubhaloID"][:]
        mleaf = f["MainLeafProgenitorID"][:]
        firstprog = f["FirstProgenitorID"][:]
        nextprog = f["NextProgenitorID"][:]
    assert snap[0] == ROOT_SNAP and (snap == ROOT_SNAP).sum() == 1
    assert np.all(np.diff(sid) == 1), "SubLink IDs not contiguous (depth-first)"
    off = sid[0]                                  # row == SubhaloID - off

    main_raw = np.full(N, np.nan)
    main_m200 = np.full(N, np.nan)
    summed = np.full(N, np.nan)
    max_prog = np.full(N, np.nan)
    n_prog = np.zeros(N, int)
    on_main = (sid >= sid[0]) & (sid <= mleaf[0])
    main_raw[snap[on_main]] = sub[on_main]
    main_m200[snap[on_main]] = m200[on_main]
    for s in range(N):
        at = snap == s
        if at.any():
            summed[s] = sub[at].sum()
            max_prog[s] = sub[at].max()
            n_prog[s] = int(at.sum())

    # cumulative infall-peak: each merging satellite counted once at its pre-infall
    # peak mass, added at the snapshot where it joins the main branch.
    deliver = np.zeros(N)
    seed_snap, seed_mass = int(snap[on_main].min()), float(main_raw[snap[on_main].min()])
    for r in np.where(on_main)[0]:
        fp = firstprog[r]
        if fp < 0:
            continue
        c = nextprog[int(fp - off)]               # co-progenitors = satellites merging here
        while c >= 0:
            rc = int(c - off)
            peak = sub[rc: int(mleaf[rc] - off) + 1].max()   # satellite's own pre-merge peak
            deliver[snap[r]] += peak
            c = nextprog[rc]
    infall_peak = np.full(N, np.nan)
    infall_peak[seed_snap:] = seed_mass + np.cumsum(deliver[seed_snap:])

    return dict(
        main_raw=main_raw, main_mpeak=_rmax(main_raw),
        max_prog=max_prog, max_prog_mpeak=_rmax(max_prog),
        summed=summed, infall_peak=infall_peak,
        main_m200c=main_m200, n_prog=n_prog,
    )


def diffmah_models(index: int, tsnap: np.ndarray):
    """(official_fit, own_fit) DiffMAH model curves [log10 Msun] for one galaxy."""
    from astropy.table import Table
    mah = np.load(OFFICIAL_MAH)
    official = mah["diffmah_log_mah_fit"][index]            # already h-free Msun
    proc = Table.read(PROC)
    logt = np.log10(np.where(tsnap > 0, tsnap, np.nan))
    own = log_mah(logt, float(proc["dmah_logmp"][index]), float(proc["dmah_logtc"][index]),
                  float(proc["dmah_early"][index]), float(proc["dmah_late"][index]),
                  logt0=np.log10(tsnap[ROOT_SNAP]), k=MAH_K)
    return official[:N], own[:N]                            # to z=0.4 (tree horizon)


def main() -> None:
    tsnap = np.loadtxt(COSMIC_TIME)
    res = {lab: mah_flavours(TREE_DIR / f"full_{sid}.hdf5") for lab, (sid, _) in EXAMPLES.items()}

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    t = tsnap[:N]
    save = {"cosmic_time_gyr": tsnap, "snaps": np.arange(N)}
    for row, (lab, (sid, idx)) in enumerate(EXAMPLES.items()):
        r = res[lab]
        off_fit, own_fit = diffmah_models(idx, tsnap)
        lg = {k: np.log10(v) for k, v in r.items() if k != "n_prog"}
        for k, v in r.items():
            save[f"{lab}_{k}"] = v
        save[f"{lab}_diffmah_official"] = off_fit
        save[f"{lab}_diffmah_own"] = own_fit

        # col 0 — main branch + DiffMAH models (the "central halo" view)
        a = axes[row, 0]
        a.plot(t, lg["main_raw"], ":", c="0.6", lw=1.2, label="main-branch raw")
        a.plot(t, lg["main_mpeak"], "-", c="k", lw=1.6, label="main-branch Mpeak (=DiffMAH data)")
        a.plot(t, lg["max_prog_mpeak"], "-", c=OKABE_ITO[2], lw=1.4, label="max-prog Mpeak (repaired)")
        a.plot(t, off_fit, "--", c=OKABE_ITO[5], lw=1.8, label="DiffMAH official (model)")
        a.plot(t, own_fit, "--", c=OKABE_ITO[0], lw=1.2, label="DiffMAH own (model)")
        a.set(ylabel=r"$\log M\ [M_\odot]$",
              title=f"{lab} (sid {sid}) — central-halo view")

        # col 1 — assembly flavours (the "how mass arrives" view)
        b = axes[row, 1]
        b.plot(t, lg["main_mpeak"], "-", c="k", lw=1.2, alpha=0.6, label="main-branch Mpeak")
        b.plot(t, lg["summed"], "-", c=OKABE_ITO[2], lw=2, label="summed bound")
        b.plot(t, lg["max_prog"], "-", c=OKABE_ITO[1], lw=1, label="max progenitor")
        b.plot(t, lg["infall_peak"], "-", c=OKABE_ITO[6], lw=2, label="infall-peak cumul")
        b.set(title=f"{lab} — assembly view")
        for ax in (a, b):
            ax.axvline(tsnap[ROOT_SNAP], ls=":", c="grey", lw=0.8)
            ax.set_ylim(10.8, 13.9)
    for ax in axes[1]:
        ax.set_xlabel("cosmic time [Gyr]")
    axes[0, 0].legend(fontsize=7, loc="lower right")
    axes[0, 1].legend(fontsize=7, loc="lower right")
    fig.suptitle("All MAH flavours + DiffMAH, two example halos (SubhaloMass, h-free)", y=1.0)
    fig.tight_layout()
    print("wrote", save_fig(fig, FIG / "all_mah_flavours")[0])

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "all_mah_flavours.npz", **save)
    print("wrote", OUT / "all_mah_flavours.npz")

    # headline table
    print("\nlog10 M [Msun] at key snaps:")
    for lab in EXAMPLES:
        r = res[lab]
        print(f"\n{lab}:  snap  main_mpeak  summed  infall_peak  max_prog  DiffMAH_off")
        offf, _ = diffmah_models(EXAMPLES[lab][1], tsnap)
        for s, z in [(72, 0.4), (50, 1.0), (33, 2.0)]:
            print(f"        {s:3d}    {np.log10(r['main_mpeak'][s]):.3f}    "
                  f"{np.log10(r['summed'][s]):.3f}   {np.log10(r['infall_peak'][s]):.3f}      "
                  f"{np.log10(r['max_prog'][s]):.3f}    {offf[s]:.3f}")


if __name__ == "__main__":
    sys.exit(main())
