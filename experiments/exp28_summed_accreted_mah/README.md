# exp28 — MAH flavours from the SubLink merger tree (+ DiffMAH), for massive-galaxy growth

## Goal
For a model of **massive central-galaxy growth** we want a MAH that reflects the
halo growth the central actually experiences. The DiffMAH model (exp27) fits the
**main-branch peak mass**, which has tree defects (dropouts) and switching spikes
on the ~20% "declining-MAH" galaxies (exp26). This experiment builds **every MAH
flavour** from the full SubLink tree, compares them against each other and the
DiffMAH model, organises them for reuse, and documents the tree format + how to
visualise it.

## The MAH flavours (`mah_flavours.py`, all h-free Msun, SubhaloMass)
`mah_flavours(tree_path)` returns each flavour snapshot-aligned (0..72). All use
**SubhaloMass** — verified to be the quantity DiffMAH was fit to (the catalog
`log_mah_sim` = the tree main-branch SubhaloMass Mpeak to <0.005 dex; *not* M200c,
which differs by ~0.05–0.07 dex per halo).

| flavour | one-line meaning | for the central galaxy |
|---|---|---|
| `main_raw` | main-branch SubhaloMass (FirstProgenitor chain) | the central's halo — but with tree dropouts/spikes |
| `main_mpeak` | running-max of `main_raw` (= DiffMAH's data) | de-dipped central halo (downward only) |
| `max_prog` | most-massive single progenitor per snapshot | **defect-robust** central halo (recovers dropouts) |
| `max_prog_mpeak` | running-max of `max_prog` | cleanest "repaired central-halo" MAH |
| `summed` | Σ SubhaloMass over all progenitors per snapshot | total collapsed mass — **over-counts** (includes future satellites' halos) |
| `infall_peak` | cumulative Σ of each satellite's pre-infall peak, at its merger snap | merger-delivered budget — right **timing** for ex-situ, monotonic |
| `main_m200c` | main-branch Group_M_Crit200 | reference FoF halo mass |

Physical reading (see `doc/summed_accreted_mah.md`): `max-progenitor` ≈ the central's
halo (repaired); `summed` leads in time and over-counts (the extra mass is in *other*
galaxies' halos not yet accreted); `infall_peak` lags (counts a piece only when it
arrives) and undershoots by the ~smooth-accretion fraction.

## Results (two example halos, logMh≈13.5: clean `283371`, declining `293978`)
`figures/all_mah_flavours.png` (2×2: central-halo view | assembly view, per halo).

log₁₀ M [M⊙]:
| halo | snap (z) | main_mpeak | summed | infall_peak | max_prog | DiffMAH(model) |
|---|---|---|---|---|---|---|
| clean | 50 (z1) | 13.33 | 13.40 | 13.07 | 13.33 | 13.27 |
| clean | 33 (z2) | 12.87 | 13.14 | 12.45 | 12.81 | 12.68 |
| declining | 50 (z1) | 13.10 | 13.23 | 12.47 | 13.21 | 13.20 |
| declining | 33 (z2) | 12.60 | 12.84 | 12.36 | 12.61 | 12.64 |

Findings:
- **`max_prog` repairs the main branch** without `summed`'s over-counting: declining
  halo at z=1 has `main_raw` dropping to 10.98 (a fragment) while `max_prog` = 13.21
  is the real progenitor. → strongest candidate for a clean central-halo MAH.
- **DiffMAH (the model) has a real ~0.1–0.17 dex fit residual** vs its own data at
  z=0.4: anchored at z=0, the rolling power law overshoots the actual main-branch
  Mpeak near z=0.4 (clean: model 13.74 vs data 13.57 at z=0.4). Worth remembering
  when DiffMAH params feed a model.
- `summed` leads, `infall_peak` lags, both converge toward the main branch near z=0.4
  (`infall_peak` stays ~0.1 dex low = the smooth-accretion deficit).

## Merger-tree format & visualisation
- **Format:** `doc/tng_merger_trees.md` — SubLink's flat depth-first array, the
  linkage fields, the `[id, MainLeafProgenitorID]` / `[id, LastProgenitorID]`
  contiguous-block trick, the ≈91 per-node fields, and extraction recipes.
- **Visualisation (`viz_tree.py`):** ytree does **not** read SubLink HDF5, so we lay
  the tree out in matplotlib: vertical axis = cosmic time (z=0.4 root at top),
  horizontal = a crossing-free post-order layout, node size ∝ log mass, **main
  branch red**, pruned to branches >2% of the root mass (12.8k → 134 nodes).
  `figures/merger_tree_{declining,clean}.png`. (For interactive/very large trees:
  Plotly with `Scattergl`, or graphviz→SVG; see the viz research notes.)

## Caveats
- Tree flavours are **SubhaloMass** (bound subhalo mass), matching official DiffMAH.
  `summed`/`infall_peak` are **merger-accreted only** (exclude ~40% smooth accretion,
  Genel+2010) and resolution-dependent → not drop-in `M200c`.
- `summed`/`infall_peak` are **not standard named quantities** (lit review,
  `doc/summed_accreted_mah.md`); validate, don't assume a baseline.

## Outputs
- `outputs/all_mah_flavours.npz` — every flavour + both DiffMAH model curves, for
  both examples (snapshot-aligned, h-free Msun). The reusable computation is
  `mah_flavours()`.
- `figures/all_mah_flavours.{png,pdf}`, `figures/merger_tree_{declining,clean}.{png,pdf}`.
- Raw trees: `/Users/mac/work/tng/exp28_full_trees/` (outside the repo).

## Next
- Pick the central-halo MAH (recommend `max_prog_mpeak`) and a delivered-mass clock
  (`infall_peak`), then **scale to the matched subset** (stream-walk full trees,
  don't cache all to disk). Biggest full tree ≈300 MB / 290k rows.
- Optional: fit the DiffMAH *form* to the chosen curve (label it "DiffMAH-form fit",
  never "the DiffMAH fit").

## Reproduce
```
PYTHONPATH=. uv run python experiments/exp28_summed_accreted_mah/mah_flavours.py
PYTHONPATH=. uv run python experiments/exp28_summed_accreted_mah/viz_tree.py
```
Needs the full trees in `/Users/mac/work/tng/exp28_full_trees/` (TNG API key at
`~/.tng_api_key`).
