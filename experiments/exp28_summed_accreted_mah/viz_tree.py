"""exp28 — merger-tree diagram from a flat SubLink tree (custom matplotlib).

ytree does not read SubLink HDF5 natively, so we lay the tree out directly. Per the
standard convention: vertical axis = cosmic time (root/z=0.4 at top, progenitors
below), horizontal = a crossing-free layout coordinate (no physical meaning), node
size ∝ log SubhaloMass, the **main branch highlighted red**. A ~13k-node tree is
unreadable in full, so we prune to branches whose peak mass exceeds a fraction of
the root mass (the main branch is always kept, so the dropout stays visible).

x-layout: post-order depth-first (Reingold–Tilford "centred parent") — each leaf
gets the next slot, each internal node sits at the mean of its kept children, which
guarantees no crossings within a subtree. The SubLink depth-first array makes the
progenitor walk O(1) (row = SubhaloID − SubhaloID[0]).

Run: PYTHONPATH=. uv run python experiments/exp28_summed_accreted_mah/viz_tree.py
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
from hongshao.plotting import set_style, save_fig  # noqa: E402

set_style()
FIG = HERE / "figures"
TREE_DIR = Path(os.environ.get("HONGSHAO_TNG_WORK", "/Users/mac/work/tng")) / "exp28_full_trees"
COSMIC_TIME = ROOT / "data" / "external" / "tng_cosmic_time.txt"
H = 0.6774
U = 1e10 / H
KEEP_RATIO = 0.02            # draw branches whose peak mass > 2% of the root mass


def plot_tree(sid: int, label: str):
    with h5py.File(TREE_DIR / f"full_{sid}.hdf5", "r") as f:
        snap = f["SnapNum"][:]
        mass = f["SubhaloMass"][:] * U
        idd = f["SubhaloID"][:]
        mleaf = f["MainLeafProgenitorID"][:]
        firstprog = f["FirstProgenitorID"][:]
        nextprog = f["NextProgenitorID"][:]
    off = idd[0]
    def row(i): return int(i - off)
    tsnap = np.loadtxt(COSMIC_TIME)
    y = tsnap[snap]                                       # cosmic time (root largest -> top)
    root_mass = mass[0]
    thresh = KEEP_RATIO * root_mass
    on_main = (idd >= idd[0]) & (idd <= mleaf[0])
    # peak mass along each node's own main-progenitor branch [id, mainleaf]
    branch_peak = np.array([mass[r:row(mleaf[r]) + 1].max() for r in range(len(idd))])

    x = {}
    leaf = [0]
    sys.setrecursionlimit(10000)
    def assign_x(r):
        kids = []
        c = firstprog[r]
        while c >= 0:
            rc = row(c)
            if on_main[rc] or branch_peak[rc] >= thresh:
                kids.append(rc)
            c = nextprog[rc]
        if not kids:
            x[r] = leaf[0]; leaf[0] += 1
        else:
            for rc in kids:
                assign_x(rc)
            x[r] = float(np.mean([x[rc] for rc in kids]))
    assign_x(0)
    kept = np.array(sorted(x))
    print(f"{label}: {len(idd)} nodes -> {len(kept)} drawn (>{KEEP_RATIO:.0%} of root)")

    fig, ax = plt.subplots(figsize=(7, 9))
    # edges: each kept node to its first-progenitor (drawn from descendant side)
    for r in kept:
        c = firstprog[r]
        while c >= 0:
            rc = row(c)
            if rc in x:
                is_main = on_main[r] and on_main[rc]
                ax.plot([x[r], x[rc]], [y[r], y[rc]],
                        c="#D55E00" if is_main else "0.7",
                        lw=2.2 if is_main else 0.7, zorder=2 if is_main else 1)
            c = nextprog[rc]
    xs = np.array([x[r] for r in kept]); ys = y[kept]
    sz = np.clip(6 + 60 * (np.log10(mass[kept]) - 10) / (np.log10(root_mass) - 10), 4, None)
    mm = on_main[kept]
    ax.scatter(xs[~mm], ys[~mm], s=sz[~mm], c="#0072B2", edgecolors="none", alpha=0.7, zorder=3)
    ax.scatter(xs[mm], ys[mm], s=sz[mm], c="#D55E00", edgecolors="k", linewidths=0.3, zorder=4)
    ax.set(xticks=[], xlabel="layout (no physical meaning)",
           ylabel="cosmic time [Gyr]  (z=0.4 at top)",
           title=f"{label}  (sid {sid}, logMh={np.log10(root_mass):.2f}) — main branch red")
    # redshift ticks on the right
    z_at = {0.4: tsnap[72], 1.0: tsnap[50], 2.0: tsnap[33], 4.0: tsnap[21]}
    ax2 = ax.twinx(); ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(list(z_at.values())); ax2.set_yticklabels([f"z={z}" for z in z_at])
    fig.tight_layout()
    print("wrote", save_fig(fig, FIG / f"merger_tree_{label}")[0])


if __name__ == "__main__":
    plot_tree(293978, "declining")
    plot_tree(283371, "clean")
