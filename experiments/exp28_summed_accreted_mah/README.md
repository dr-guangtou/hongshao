# exp28 — summed-accreted-mass MAH vs the main-branch MAH

## Question
DiffMAH (own + official, exp27) fits the **main-branch peak mass** — follow the
single most-massive progenitor back in time. That branch carries two defects that
corrupt the ~20% "declining-MAH" galaxies (exp26): **tree dropouts** (the
first-progenitor link jumps to a tiny fragment for one snapshot) and
**central–satellite switching** (a transient mass spike the running-max Mpeak then
locks in). The **summed-accreted-mass MAH** avoids picking a single branch: at each
snapshot it sums the bound mass of *all* progenitor subhalos in the tree. Does it
actually de-bias the history, and by how much? This experiment builds it from the
full SubLink tree for example halos and compares.

## Method
For a halo's full SubLink tree (root = its snap-72 subhalo; raw trees pulled to
`/Users/mac/work/tng`, NOT the repo), all on the TNG snapshot grid 0..72 in **h-free
Msun**:
- `main_raw` — main-branch `SubhaloMass` (depth-first block `[id, MainLeafProgenitorID]`).
- `main_mpeak` — running-max of `main_raw` (the DiffMAH input; removes downward dips).
- `summed` — Σ `SubhaloMass` over all tree nodes at each snapshot. SUBFIND mass is
  exclusive, so this never double-counts. (At snap 72 only the root exists →
  `summed == main_raw`, a built-in check.)
- `max_prog`, `n_prog` — most-massive single progenitor and progenitor count per snap.

Two example halos at logMh(z=0.4) ≈ 13.5 (deliberately **not** the most massive):
a **clean** one (`sid 283371`) and a **declining-MAH** one (`sid 293978`).

## Result (examples)
The summed-accreted MAH is **smooth and sits ~0.07–0.27 dex above the main-branch
Mpeak**, converging to the final mass at z=0.4 (it never exceeds it by >0.6%, a
mass-conservation sanity check). The raw main branch has catastrophic
single-snapshot dropouts the Mpeak only partly hides. It is *near*-monotonic, not
strictly: it dips by ≤0.12 dex where progenitors are tidally stripped just before
merging — so the bound-mass sum is smoother than the main branch but still not
spike-free. A true **infall-peak sum** (each progenitor's peak mass before infall)
would be monotonic by construction; that's the natural next variant.

| halo | snap (z) | main_raw | main_mpeak | summed | max_prog | n_prog |
|---|---|---|---|---|---|---|
| clean 283371 | 50 (z≈1) | 12.49 | 13.33 | **13.40** | 13.33 | 93 |
| clean 283371 | 33 (z≈2) | 11.84 | 12.87 | **13.14** | 12.81 | 218 |
| declining 293978 | 50 (z≈1) | **10.98** | 13.10 | **13.23** | 13.21 | 162 |
| declining 293978 | 33 (z≈2) | 12.07 | 12.60 | **12.84** | 12.61 | 227 |

The declining halo at snap 50 is the textbook case: the SubLink main branch links to
a 10.98-dex fragment while the real main progenitor (`max_prog` = 13.21) sits
off-branch; `summed` (13.23) recovers it and the Mpeak (13.10) lands ~0.1 dex low.
See `figures/summed_vs_mainbranch.png`.

## Caveats (literature-checked — see `doc/summed_accreted_mah.md`)
- **`summed` is the merger-accreted component only.** It excludes *smooth*
  accretion (mass never in a resolved halo) = ~40% of total halo growth
  (Genel+2010), and it is **resolution-dependent**. So it is the resolved-collapsed
  fraction of the final mass, **not** `M200c(z)` — do not treat it as a drop-in halo
  mass. Our tree is rooted at snap 72, so `summed` is normalised to the central's
  z=0.4 `SubhaloMass` (one node there), not the FoF mass.
- **`M_sum(z)` is not a standard named quantity.** No surveyed paper publishes it
  as a MAH time series; the closest are the EPS collapsed fraction and the
  USMF/infall-mass sum (which is also merger-accreted-only). Novelty + caution: no
  established baseline → validate against the main branch and resolution.
- `summed` is dominated by the largest progenitor when one exists
  (`summed ≈ max_prog`); the early rise sums a tail of near-resolution halos → a
  mass-threshold variant is the obvious robustness check.
- Main-branch terminology: our SubLink trees use the **most-massive-*history*
  (MMH)** branch (Springel+2005), not the instantaneous most-massive progenitor.

## Definition & prior work
`doc/summed_accreted_mah.md`: definitions (main-branch MMP/MMH, Mpeak,
summed-progenitor `M_sum(z)`, USMF infall-sum), canonical references, and the
finding that no prior work uses a summed-progenitor-mass MAH as a named time series.

## Outputs
- `outputs/example_mah_curves.npz` — the four MAH flavours + n_prog per example
  (snapshot-aligned, Msun). Raw trees stay in `/Users/mac/work/tng/exp28_full_trees/`.
- `figures/summed_vs_mainbranch.{png,pdf}`.

## Next
- Resolution-threshold variant + the infall-peak-sum variant (UniverseMachine style).
- Scale to the matched subset (stream-walk full trees, don't cache all to disk),
  and fit the DiffMAH form to the summed-accreted curve (labelled as such, never
  "the DiffMAH fit").

## Reproduce
```
# fetch one tree (needs ~/.tng_api_key); saves to /Users/mac/work/tng/exp28_full_trees/
PYTHONPATH=. uv run python experiments/exp28_summed_accreted_mah/summed_accreted_mah.py
```
