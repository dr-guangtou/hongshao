# TNG SubLink merger trees — format & contents (quick reference)

*From the exp27/exp28 API pulls. TNG docs: tng-project.org/data/docs/specifications
(Sec. "Merger Trees") and /data/docs/api. Mass units in the raw files are
**1e10 Msun/h**, positions **ckpc/h** — divide by h=0.6774 for our h-free Msun.*

## Two tree algorithms
TNG ships **SubLink** (and `SubLink_gal`, baryonic) and **LHaloTree**. We use
**SubLink** (Rodriguez-Gomez+2015, arXiv:1502.01339). It tracks **SUBFIND
subhalos** (not FoF groups). The "main progenitor" is the branch with the
**most-massive history** (Springel+2005), not the instantaneous most-massive one —
more stable to noise when two progenitors have similar mass.

## What a tree is: a flat, depth-first array
A pulled tree (`…/sublink/full.hdf5`) is a set of equal-length 1-D arrays, one
**row per (subhalo, snapshot) node**. Rows are in **depth-first order**, and
`SubhaloID` is the contiguous depth-first index (so `row = SubhaloID − SubhaloID[0]`
within one tree). This ordering is what makes subtree operations O(1)-sliceable.

### Linkage fields (how nodes connect)
| field | meaning |
|---|---|
| `SubhaloID` | unique depth-first index within the tree (the node id) |
| `SubfindID` | the subhalo's id in the SUBFIND catalog at its snapshot |
| `SnapNum` | snapshot (0..99; 72 = z=0.4, our root) |
| `DescendantID` | the node this one merges into next snapshot (−1 if none) |
| `FirstProgenitorID` | the main progenitor (largest-history) one snapshot back |
| `NextProgenitorID` | sibling progenitor sharing the same descendant (the merger chain) |
| `MainLeafProgenitorID` | earliest node of this node's **main branch** |
| `LastProgenitorID` | last node of this node's **entire subtree** (all branches) |
| `RootDescendantID` | the z=0 endpoint of the whole tree |

### The two depth-first identities that do all the work
For any node with id `i`:
- **main branch** (follow FirstProgenitor) = contiguous rows `[i, MainLeafProgenitorID[i]]`.
- **full progenitor subtree** (all branches) = contiguous rows `[i, LastProgenitorID[i]]`.

So from the root (row 0, snap 72): its main branch is `[0, MainLeafProgenitorID[0]]`,
and the whole file is its subtree.

## Per-node data (≈91 fields)
Every node carries the **full SUBFIND subhalo record** *and* its **FoF host group**
record at that snapshot:
- **subhalo mass:** `SubhaloMass` (total bound — what DiffMAH is fit to),
  `SubhaloMassType` (gas/DM/_/_/stars/BH; stars = index 4), `SubhaloMassInRadType`,
  `SubhaloMassInHalfRadType`, `SubhaloMassInMaxRadType`.
- **subhalo kinematics/size:** `SubhaloPos` (ckpc/h), `SubhaloVel`, `SubhaloVmax`,
  `SubhaloVelDisp`, `SubhaloHalfmassRad(Type)`, `SubhaloSpin`.
- **subhalo baryonic:** `SubhaloSFR(inRad/HalfRad)`, `SubhaloStarMetallicity`,
  `SubhaloGasMetallicity`, `SubhaloBHMass`, `SubhaloBHMdot`, `SubhaloStellarPhotometrics`.
- **FoF group (host):** `Group_M_Crit200/Crit500/Mean200/TopHat200`, `Group_R_*`,
  `GroupPos`, `GroupMass`, `GroupNsubs`, `GroupFirstSub`, `GroupSFR`. (A node is a
  **central** iff `GroupFirstSub == SubfindID`.)

## API endpoints
- `…/snapshots/72/subhalos/{id}/sublink/mpb.hdf5` — **main progenitor branch only**
  (one row per snapshot; cheap, ~130 KB). Used in exp27 for the position match + MAH.
- `…/sublink/full.hdf5` — **the entire tree** (all branches). Used in exp28. Big:
  ~15 MB for logMh 13.5, ~300 MB for a 1e15 cluster.
- `…/sublink/simple.json` — small JSON summary.

## Recipes (used in exp28 `mah_flavours.py`)
- **main-branch MAH:** `mass[on_main]` where `on_main = (SubhaloID ∈ [id0, MainLeaf0])`.
- **all progenitors at snapshot s:** `mass[SnapNum == s].sum()` (summed) or `.max()`
  (max-progenitor) — the full tree already holds every progenitor.
- **mergers onto the main branch:** for each main-branch node, take its
  `FirstProgenitorID`, then walk the `NextProgenitorID` chain off it — those siblings
  are the satellites merging in at that node's snapshot. Each satellite's pre-infall
  peak = `max(SubhaloMass over [sat_id, MainLeaf[sat_id]])` (the infall-peak MAH).

See `doc/summed_accreted_mah.md` for the physics of each MAH flavour and
`experiments/exp28_summed_accreted_mah/` for the code + figures.
