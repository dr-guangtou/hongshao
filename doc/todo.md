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
- [x] **exp02_profile_pca** — PCA of curves of growth (direction D). **Result:**
  profile shape is ~2-3 dimensional (PC1=93%, +PC2=99%); shape modes carry a
  real assembly signal at fixed M0 (PC2 ↔ Mpeak(z=2) r=+0.28, PC3 ↔ z50 r=+0.23),
  while PC1 = concentration tracks total mass. Confirms exp01 in a shape basis.
- [x] **exp03_radial_diffmah** — radial-DiffMAH CoG fits (direction C). **Result:**
  5 interpretable params fit every CoG to ~0.005 dex (matches PCA-3); a single
  sigmoid suffices. Assembly signal at fixed M0 lives mainly in transition width
  Delta (r≈0.21 with Mpeak(z=1)/z50), slightly weaker than PCA's best direction.
- [x] decide single- vs two-component representation → **single is enough** at z=0.4.

## Phase 3 — MAH compression
- [ ] compare raw MAH summaries vs PCA-MAH (DiffMAH fits deferred; no halo_id
  cross-match yet).

## Phase 4 — conditional emulator
- [x] **exp04_conditional_model** — `P(profile | M0)` vs `P(profile | M0, MAH)`,
  5-fold CV linear, shuffle control. **Result:** assembly history improves
  full-CoG prediction by 22.6% (0.152→0.118 dex; shuffle ~0%), shape by 7.3%;
  gain grows with radius for absolute mass; profile painting works (early formers
  more extended). First Ultimate-SHMR prototype.

## Phase 5 — refinements & mock painting (choose next)
- [ ] predict radial-DiffMAH params (generative, monotonic) instead of raw CoG.
- [ ] add secondary halo properties (concentration / tidal field).
- [ ] probabilistic emulator (scatter / GP / normalizing flow).
- [ ] apply emulator to an N-body catalog; compare profile distributions.
- [ ] redshift evolution once other-z profiles available.

## Open data gaps (not blocking Phase 1)
- [ ] exact snap-72 halo mass (`mass_halo`) to define `M0` precisely.
- [ ] `halo_id` cross-match to DiffMAH/Diffstar HDF5 files.
