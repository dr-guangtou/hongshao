# HongShao Roadmap

Cross-experiment plan. Mirrors the phase sequence in
`ultimate_shmr_possible_directions.md`. Per-experiment results live in each
`experiments/expNN_*/README.md`; this file tracks the arc.

## Status

- [x] **Data layer** — TNG300 z=0.4 loaders, dataset builder, QC figure,
  cosmic-time mapping, decline cut. Clean sample: 2545/3388. (`hongshao/tng_data.py`)

## Phase 1 — exploratory diagnostics
- [x] **exp01_aperture_mah_corr** — partial-correlation map of stellar aperture
  mass vs `Mpeak(z)` at fixed `M0` (directions A + F). **Result:** at fixed M0,
  halo history cuts inner & outer stellar-mass scatter by ~20% (shuffle-confirmed
  real); inner <10 kpc uniquely tracks early mass (Mpeak(z=2), r=0.46, falling
  ~2× outward), recent growth (z≲1) dominates all radii. Positive → proceed.

## Phase 2 — profile compression
- [ ] PCA of curves of growth (direction D): intrinsic dimensionality.
- [ ] radial-DiffMAH CoG fits (direction C): compress each profile to ~5 params.
- [ ] decide single- vs two-component representation.

## Phase 3 — MAH compression
- [ ] compare raw MAH summaries vs PCA-MAH (DiffMAH fits deferred; no halo_id
  cross-match yet).

## Phase 4 — conditional emulator
- [ ] `P(theta_prof | M0)` baseline vs `P(theta_prof | M0, theta_MAH)`; null /
  shuffled controls; quantify information gain.

## Phase 5 — mock profile painting
- [ ] apply emulator to an N-body catalog; compare profile distributions.

## Open data gaps (not blocking Phase 1)
- [ ] exact snap-72 halo mass (`mass_halo`) to define `M0` precisely.
- [ ] `halo_id` cross-match to DiffMAH/Diffstar HDF5 files.
