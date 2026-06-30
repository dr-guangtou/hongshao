# exp29 — the "puff-up" deposition model (UN-PARKED — build it)

> **Status: UN-PARKED (2026-06-30).** The NEXT-SESSION diagnostic (below) ran:
> `single_epoch_all.py` fit the centred-Gaussian kernel to each epoch's CoG
> independently. **The deposit shape is NOT the limit** — every epoch, every
> mass tertile fits to ≤1.4% max-rel; high-mass z=2 (0.9%) ≈ z=0.4 (0.7%); the
> BCG's z=2 is its *best* fit (0.3% max-rel). So the multi-epoch failure is a
> **consistency** problem (a fixed-width additive deposit can't redistribute
> early-compact mass), exactly what puffing addresses. **Build this model.** The
> design below is ready; start at "Test plan".

## Why this exists — the structural finding it would solve
`single_vs_multi.py` showed the current cumulative-**additive, fixed-width**
deposition kernel fits **any one** CoG to ~0.004 dex / 2% but **cannot fit all five
epochs at once** (z=0.4 degrades to 0.034 dex / 19%; the multi-epoch fit falls
15–20% below the z=0.4 outskirts). This persists with the free-fraction NNLS +
flexible widths (capacity z=0.4 ≈ 0.030), so it is **structural**: the model can only
*add* mass at a fixed width and never moves it again. Real massive galaxies puff
their early-compact mass outward (mergers / violent relaxation; exp25/exp26 saw inner
density even drop). To build z=0.4's extended envelope the early (z=2-compact) mass
must migrate outward — impossible for a pure-addition model without ruining high-z.

## The model
Each deposit `i` is laid at time `t_i` with mass `dM*_i` and an **initial** width
`σ_{i,0}`. After deposition its width **grows**, so observed at epoch `t_k` it has
width `σ_i(t_k) ≥ σ_{i,0}` (narrow at high z, puffed by z=0.4). Closed-form CoG is
preserved (centred Gaussian):
```
M*(<R, z_k) = Σ_{i: t_i ≤ t_k} dM*_i · [ 1 − exp( −R² / 2 σ_i(t_k)² ) ]
dM*_i       = f(t_i) · dM_halo,i                       (the phenomenological fraction)
```
`q = 0` (no puffing) recovers today's model exactly, so puffing is a strict superset.

### Puffing-law candidates (one new ingredient)
1. **Time-ratio power law:** `σ_i(t_k) = σ_{i,0} · (t_k / t_i)^q`, `q ≥ 0`. Simplest; 1 global param.
2. **Diffusion / additive:** `σ_i(t_k)² = σ_{i,0}² + κ·(t_k − t_i)`. 1 param `κ`.
3. **Halo-growth driven:** `σ_i(t_k) = σ_{i,0} · (M_h(t_k)/M_h(t_i))^p`. Ties puffing to
   accretion/mergers (legal halo input); 1 param `p`.
Initial width `σ_{i,0}`: keep simple (constant, or small ∝ halo size at `t_i`).

## Why it should break the ceiling
The capacity test fixed each deposit's width at its deposition time. Puffing makes the
width depend on the **observation** time `t_k`, a structurally new freedom never tested.
The SAME early deposits then supply the compact high-z profile (small `σ_i(t_high)`)
**and** the extended z=0.4 envelope (large `σ_i(t_obs)`), dissolving the tension.

## Calibration from the single-epoch param trends (`param_trends.py`)
The independent per-epoch fits already tell us what puffing must deliver:
- **`g ≈ 1.7` is epoch-stable** (every epoch; matches exp25's per-galaxy `g≈1.67`).
  The width-growth-with-cosmic-time *shape* is a shared invariant — keep it; puffing
  is an extra DOF on top, not a replacement.
- **The early-formed mass de-concentrates with time (stated in fixed physical-kpc
  apertures, not R50 — R50 is observationally fragile).** The fraction of the pre-z=2
  stellar mass inside a fixed **5 kpc** aperture falls **0.64 → 0.44** from z=2 to
  z=0.4 (inside **10 kpc**: 0.76 → 0.66); mass beyond **30 kpc is unchanged (~0.92)**.
  Massive galaxies de-concentrate more (inside 10 kpc: **0.71 → 0.56**). This is
  robust — at z=2 the pre-z=2 mass is 100% of the galaxy, so the z=2 value equals the
  measured CoG. The redistribution is entirely in the inner ≲10 kpc; that fixed-kpc
  inner-fraction drop is the target the puffing law must reproduce (more for BCGs).
- **Single-epoch fits fake it via the efficiency, not the width.** Individual pre-z=2
  deposit widths barely move (mass-wtd 7.5→8.4 kpc); the `R50` growth comes from the
  efficiency re-spreading early mass onto later/wider pre-z=2 deposits (`b_early`
  drops 5.8→3.2 from z=2 to z=0.4). A *joint* fit can't do this — `f(t_i)` is one
  fixed function across epochs — which is exactly why multi-epoch fails and exactly
  the freedom puffing restores: let `σ_i(t_k)` grow so the *same* early mass is
  compact at z=2 and extended at z=0.4.

## Tractability — the convex inner solve survives
Given the puffing-law params (and `σ_{i,0}`), every `σ_i(t_k)` is fixed, so the CoG is
still **linear in the masses** `dM*_i` → the joint multi-epoch fit stays a non-negative
least squares (NNLS, `capacity_test.py` machinery). Only the puffing-law + initial-width
params are the (low-D) nonlinear outer loop. So the exploration cost is the same shape as
Phase 1.

## Test plan (when un-parked)
1. **Capacity with puffing:** re-run `capacity_test.py` with `σ_i(t_k)` puffing; does the
   joint 5-epoch z=0.4 RMS fall from ~0.030 toward the single-epoch ~0.004?
2. **single_vs_multi:** does multi-epoch z=0.4 recover to single-epoch quality (2%)?
3. **Residual map:** does the high-z S-shape (under-centre / over-mid / under-outer) flatten?
4. Then re-do the population forward emulator (Phase 4e, legal halo-only) with puffing.

## Risks / why deferred
- One more knob → degeneracy among `σ_{i,0}`, the puffing rate, and `f`. Needs the
  honest residual metric (radius×epoch relative residuals + percentiles) to judge.
- Puffing ≈ "mergers redistribute stars" — appealing, but keep it **phenomenological**
  (a flexible, fast emulator), not a physical claim.
- May still not fix high-z if the limit is the *deposit shape* (Gaussian) rather than
  multi-epoch consistency — which is exactly what the NEXT-SESSION diagnostic settles.

---

# NEXT SESSION — do this FIRST (before puffing)
**Independent single-epoch fits to every snapshot, especially z=2, to test for a
fundamental limit of the (oversimplified) deposition model on high-z massive galaxies.**

- For each epoch `z_k ∈ {0.4, 0.7, 1.0, 1.5, 2.0}` (snaps 72/59/50/40/33),
  fit the deposition kernel to **that epoch's CoG ALONE**, per galaxy: deposits up to
  `t(z_k)`, normalization pinned to that epoch's measured total `M*(z_k)`. (Reuse
  `cog_extrapolate.py` / `single_vs_multi.py` machinery; just change the target epoch.)
- **Question:** is the single-epoch fit at z=2 as good as at z=0.4 (~0.004 dex / 2%)?
  - **If yes** → the deposit *shape* is fine at every epoch; the only problem is
    multi-epoch *consistency* → **puffing is the right fix** (build it).
  - **If no** (z=2 fits fail for massive galaxies) → the centred-Gaussian deposit
    cannot describe compact high-z massive profiles → the *deposit shape itself* is
    the fundamental limit (puffing won't save it); rethink the high-z primitive.
- **Evaluate honestly:** linear `M*`, relative residual `(model−data)/data`, report
  **max / 90th-percentile** relative error (not just mean log-RMS), exclude inner ≤3 kpc
  (TNG softening). Stratify by stellar/halo mass — the BCGs are where it breaks.
- Watch the **aperture leak** (model puts ~8% mass beyond 148 kpc → ~0.035 dex low at
  the outer radius); consider pinning the normalization to the 148-kpc aperture.

## Metric lesson to carry forward
The averaged log-RMS hides structured errors: even the 5 best per-galaxy fits had
21–38% max relative errors. Always pair the mean RMS with the radius×epoch relative
residual map (`best5_qa.py`, `exp29_residual_map`).
