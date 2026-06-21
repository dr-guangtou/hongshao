# exp13 — Limit of predicting the 50–100 kpc outskirt mass from the MAH

## Question
The 50–100 kpc annulus carries the most scatter and is where the mild
nonlinearity lives (exp12). A focused, exploratory question: **what is the
upper limit of predictability for `M*[50–100 kpc]` from the halo's assembly
history**, and does a richer MAH representation beat the four portable DiffMAH
params? Is the residual outskirt scatter *feature-limited* (more MAH info would
help) or *intrinsic* (projection / ICL / low-SB noise we cannot remove)?

This **deliberately breaks the portability principle**: MAH-PCA and the raw MAH
vector are defined on the TNG sample and do not transfer. It is a ceiling probe,
not a model to ship.

## Method
Single target `M*[50–100 kpc]`. Cross feature richness × model class, all
5-fold CV scored by the exp07 suite (CRPS) plus RMS and variance explained R²:
- **features:** M0 only · DiffMAH(4) · M0+MAH-PCA(4) · M0+MAH-PCA(8) · raw MAH(18)
- **models:** linear · PySR (polynomial SR, exp12) · GBM (flexible ceiling, exp09)
- **control:** GBM on the raw MAH shuffled within M0 bins → must collapse to the
  M0-only level if the signal is real.

The raw MAH is the un-normalized `log M_halo(t)` on an 18-point cosmic-time grid
(2.2–9.0 Gyr); MAH-PCA is its M0-normalized-shape PCA + M0. Inputs:
`data/processed/tng300_072_z0p4.fits` (use sample, n=2533), `hongshao.tng_data`
(MAH loaders), `hongshao.metrics`.

## Key result
**The MAH predicts the outskirts well, the four DiffMAH params already reach the
ceiling, and the remaining scatter is intrinsic — not extractable by any MAH
representation or model we tried.** (target std = 0.411 dex)

| features | linear CRPS | RMS [dex] | R² | vs M0 |
|---|---|---|---|---|
| M0 only | 0.1275 | 0.230 | 0.687 | — |
| **DiffMAH (4)** | **0.0990** | **0.179** | **0.810** | **+22.4%** |
| M0 + MAH-PCA(4) | 0.0963 | 0.175 | 0.819 | +24.5% |
| M0 + MAH-PCA(8) | 0.0964 | 0.175 | 0.819 | +24.4% |
| raw MAH (18) | 0.0966 | 0.175 | 0.818 | +24.3% |
| raw MAH *shuffled* | 0.1367 | 0.245 | 0.647 | −7.2% (control) |

- **The MAH matters a lot for the outskirts.** Adding assembly history to M0
  lifts R² from 0.69 → 0.82 and cuts CRPS by ~24% — a larger relative gain than
  the inner apertures get. The outer envelope is genuinely assembly-driven.
- **DiffMAH(4) is essentially at the ceiling.** The richest representation
  (MAH-PCA or the full 18-point raw MAH vector) beats DiffMAH(4) by only **+2.7%
  CRPS** (R² 0.810 → 0.819). MAH-PCA(8) and raw-MAH(18) do **not** beat
  MAH-PCA(4) — diminishing returns set in immediately. So the four portable
  params capture ~90% of *all* the MAH information available for the outskirts
  (consistent with exp10's aggregate ~88%, now confirmed on the hardest annulus).
- **No nonlinear structure to exploit.** GBM ≈ linear for every feature set
  (e.g. raw-MAH GBM 0.0975 vs linear 0.0966) — a flexible model finds nothing
  the linear one misses. The relation is linear here too.
- **The limit is R² ≈ 0.82, RMS ≈ 0.175 dex, CRPS ≈ 0.096.** The residual ~0.175
  dex (~43% of the original 0.411 std) is **irreducible from the MAH**: no
  representation or model closes it. This is the intrinsic floor — single 2-D
  projections of triaxial galaxies, ICL, and low-SB measurement noise (per
  AGENTS.md), not missing halo features.
- **Signal is real.** The shuffled-MAH control falls *below* the M0-only floor
  (0.137 > 0.128), so the +24% gain is genuine assembly information, not
  overfitting.

### A parsimony nugget (PySR)
- On the **raw MAH vector**, PySR's best equation keeps a **single epoch**,
  `t13` ≈ the halo mass at cosmic time ~7 Gyr (z ≈ 0.7, ~2.5 Gyr before the z=0.4
  observation), and already reaches R² = 0.800 — nearly the full-vector 0.818.
  The outskirt mass is set almost entirely by the *recent* halo mass.
- On **DiffMAH**, PySR keeps `logmp` + `late` (normalization + late-time
  accretion index) — the same `late`-driven story as exp12.

## Decision
- **Keep the four portable DiffMAH params.** For the outskirts they reach ~98%
  of the best-possible CRPS; the non-portable MAH-PCA / raw-MAH representations
  buy almost nothing (+2.7%), so there is no reason to abandon portability.
- **The outskirt residual is intrinsic, not feature-limited.** Effort to improve
  50–100 kpc predictions should target the *scatter model* (the irreducible
  ~0.175 dex) and/or new observables (projection-robust measures), not richer
  halo features. This sharpens the exp11/exp12 "the lever is scatter, not the
  mean" conclusion with a concrete ceiling.
