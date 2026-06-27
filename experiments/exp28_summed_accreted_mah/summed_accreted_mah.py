"""exp28 step 1 — summed-accreted-mass MAH vs the main-branch MAH, on example
halos, from the full SubLink merger tree.

Motivation (exp26/exp27): the DiffMAH MAH is the *main-branch* peak mass — follow
the single most-massive progenitor back in time. That branch suffers tree defects
(single-snapshot dropouts) and central-satellite switching (transient mass spikes
that the running-max Mpeak then locks in), which corrupt the late history of the
~20% "declining-MAH" galaxies. The *summed-accreted-mass* MAH sidesteps the
single-branch ambiguity: at each snapshot it sums the bound mass of ALL progenitor
subhalos in the tree (everything that has already coalesced into the final object),
so it is immune to which branch is tagged "main".

We compute, on the full tree of one halo (root = snap-72 subhalo), all on the TNG
snapshot grid 0..72, in h-free Msun:
  main_raw   : main-branch SubhaloMass (SubLink first-progenitor chain)
  main_mpeak : running-max of main_raw (the DiffMAH input; removes downward dips)
  summed     : sum of SubhaloMass over all tree nodes at each snapshot
  max_prog   : the single most-massive progenitor at each snapshot
  n_prog     : number of progenitor subhalos at each snapshot
At snap 72 only the root exists, so summed == main_raw there (a built-in check).

SubLink layout: the tree is depth-first ordered; a node's main branch is the
contiguous SubhaloID block [node_id, MainLeafProgenitorID]. SUBFIND SubhaloMass is
exclusive (no particle double-counted), so summing it across nodes is well defined.

Raw trees live OUTSIDE the repo (big): $HONGSHAO_TNG_WORK or /Users/mac/work/tng.
Only the small derived curves + figure land in the experiment dir.

Run: PYTHONPATH=. uv run python experiments/exp28_summed_accreted_mah/summed_accreted_mah.py
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
from hongshao.plotting import set_style, save_fig, OKABE_ITO  # noqa: E402

set_style()
OUT, FIG = HERE / "outputs", HERE / "figures"
TREE_DIR = Path(os.environ.get("HONGSHAO_TNG_WORK", "/Users/mac/work/tng")) / "exp28_full_trees"
COSMIC_TIME = ROOT / "data" / "external" / "tng_cosmic_time.txt"
H = 0.6774
MASS_UNIT = 1e10 / H                 # raw (1e10 Msun/h) -> Msun
ROOT_SNAP = 72

# example halos (logMh ~ 13.5, NOT the most massive): see crossmatch.fits
EXAMPLES = {"clean": 283371, "declining": 293978}


def mah_from_tree(path: Path) -> dict:
    """Compute the MAH flavours (Msun, snapshot-aligned 0..72) for one full tree."""
    with h5py.File(path, "r") as f:
        snap = f["SnapNum"][:]
        mass = f["SubhaloMass"][:] * MASS_UNIT
        sid = f["SubhaloID"][:]
        mainleaf = f["MainLeafProgenitorID"][:]
        subfind0 = int(f["SubfindID"][0])
    assert snap[0] == ROOT_SNAP, f"root snap {snap[0]} != {ROOT_SNAP}"
    assert (snap <= ROOT_SNAP).all() and (snap == ROOT_SNAP).sum() == 1

    n = ROOT_SNAP + 1
    main_raw = np.full(n, np.nan)
    summed = np.full(n, np.nan)
    max_prog = np.full(n, np.nan)
    n_prog = np.zeros(n, int)

    on_main = (sid >= sid[0]) & (sid <= mainleaf[0])     # depth-first main block
    main_raw[snap[on_main]] = mass[on_main]
    for s in range(n):
        at = snap == s
        if at.any():
            summed[s] = mass[at].sum()
            max_prog[s] = mass[at].max()
            n_prog[s] = int(at.sum())

    # running-max Mpeak of the main branch (time increases with snapshot index)
    main_mpeak = np.full(n, np.nan)
    valid = np.isfinite(main_raw)
    vals = np.where(valid, main_raw, -np.inf)
    rmax = np.maximum.accumulate(vals)
    main_mpeak[valid] = rmax[valid]

    assert np.isclose(summed[ROOT_SNAP], main_raw[ROOT_SNAP]), "summed!=main at root"
    return dict(subfind=subfind0, main_raw=main_raw, main_mpeak=main_mpeak,
                summed=summed, max_prog=max_prog, n_prog=n_prog)


def main() -> None:
    tsnap = np.loadtxt(COSMIC_TIME)                      # Gyr, index == snapshot
    res = {}
    for label, sid in EXAMPLES.items():
        p = TREE_DIR / f"full_{sid}.hdf5"
        if not p.exists():
            raise SystemExit(f"missing tree {p} -- fetch it to {TREE_DIR} first")
        res[label] = mah_from_tree(p)
        r = res[label]
        # headline: main-branch vs summed at z=0.4 and at z=1/z=2
        for s, z in [(72, 0.4), (50, 1.0), (33, 2.0)]:
            print(f"{label} sid={sid} snap{s}(z~{z}): "
                  f"main_raw {np.log10(r['main_raw'][s]):.2f}  "
                  f"Mpeak {np.log10(r['main_mpeak'][s]):.2f}  "
                  f"summed {np.log10(r['summed'][s]):.2f}  "
                  f"max_prog {np.log10(r['max_prog'][s]):.2f}  n={r['n_prog'][s]}")

    # --- figure: per example, log M(t) for each flavour ----------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, (label, r) in zip(axes, res.items()):
        t = tsnap[: ROOT_SNAP + 1]
        ax.plot(t, np.log10(r["summed"]), "-", c=OKABE_ITO[2], lw=2, label="summed-accreted")
        ax.plot(t, np.log10(r["max_prog"]), "-", c=OKABE_ITO[1], lw=1, alpha=0.8, label="max progenitor")
        ax.plot(t, np.log10(r["main_mpeak"]), "--", c=OKABE_ITO[5], lw=1.4, label="main-branch Mpeak (DiffMAH)")
        ax.plot(t, np.log10(r["main_raw"]), ":", c=OKABE_ITO[7], lw=1.4, label="main-branch raw")
        ax.set(xlabel="cosmic time [Gyr]",
               title=f"{label}  (sid {EXAMPLES[label]}, logMh$_{{z=0.4}}$={np.log10(r['summed'][72]):.2f})")
        ax.axvline(tsnap[ROOT_SNAP], ls=":", c="grey", lw=0.8)
    axes[0].set(ylabel=r"$\log M\ [M_\odot]$", ylim=(10.5, 13.8))
    axes[0].legend(fontsize=7, loc="lower right")
    fig.suptitle("Summed-accreted vs main-branch MAH (full SubLink tree)", y=1.0)
    fig.tight_layout()
    print("\nwrote", save_fig(fig, FIG / "summed_vs_mainbranch")[0])

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / "example_mah_curves.npz",
        cosmic_time_gyr=tsnap, snaps=np.arange(ROOT_SNAP + 1),
        **{f"{lab}_{k}": v for lab, r in res.items()
           for k, v in r.items() if k != "subfind"})
    print("wrote", OUT / "example_mah_curves.npz")


if __name__ == "__main__":
    sys.exit(main())
