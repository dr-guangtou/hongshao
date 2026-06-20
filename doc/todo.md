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
- [x] **exp06_mah_pca** — PCA of the full M0-normalized log-MAH (no DiffMAH).
  **Result:** MAH is ~3-4 dimensional (PC1=73%, PC1-3=93%); MAH-PCA(4) matches
  hand-picked summaries for profile prediction (+22.9% vs +22.6%, shuffle ~0%);
  halo and galaxy PCs connect (MAH-PC2 timing → CoG-PC1 concentration r=0.46).
  Compressibility: CoG 99.7% > Σ 97.4% > MAH 93.0% in 3 modes. Adopt MAH-PCA as
  the principled halo representation.

## Phase 4 — conditional emulator
- [x] **exp04_conditional_model** — `P(profile | M0)` vs `P(profile | M0, MAH)`,
  5-fold CV linear, shuffle control. **Result:** assembly history improves
  full-CoG prediction by 22.6% (0.152→0.118 dex; shuffle ~0%), shape by 7.3%;
  gain grows with radius for absolute mass; profile painting works (early formers
  more extended). First Ultimate-SHMR prototype.

## Phase 5 — refinements & mock painting

- [x] **exp05_generative_model** — predict radial-DiffMAH params → reconstruct
  valid profiles. **Result:** generating via 5 params reaches 0.128 dex (vs 0.118
  direct), history improves it 19.8% (shuffle ~0%); helps normalization most
  (+13%). Paints ~45% of the true profile-shape diversity at fixed M0. Also
  bounded the fit (identifiable params) and cached rdm_* in the dataset.

### exp07 (NEXT) — evaluation metrics: how to judge fit/recovery quality
Investigate the *best metrics* to evaluate model fitting and recovery, before
building the probabilistic emulator. Two tracks:
- **Predictors (PCA / regression recovery):** evaluate in **stellar mass within
  apertures and annuli** (e.g. <10, 10–30, 30–50, 50–100, <100 kpc) — physical,
  weakly correlated quantities — not just per-radius CoG dex. Add distributional
  scores (CRPS / predictive log-likelihood, interval calibration) and
  predicted-vs-true **covariance** match.
- **Profile-model FITTING (the 5-param radial-DiffMAH to the CoG):** careful fit
  diagnostics — **reduced chi² (needs per-point CoG uncertainties — flag),
  AIC/BIC to compare models** (single vs double sigmoid; radial-DiffMAH vs
  Einasto vs spline/PCA), and **the residual profile itself** (look for coherent
  radial residuals indicating a missing component).
- Deliverable: a recommended evaluation suite reused by exp08+.

### Later
- [ ] probabilistic emulator (scatter / GP / normalizing flow), judged by the
  exp07 metrics.
- [ ] secondary halo properties — *test*, don't assume: MAH-derived ones
  (concentration, accretion rate) likely redundant with the MAH (exp06); only
  initial conditions / environment are independent (hard to get here).
- [ ] apply emulator to an N-body catalog; compare profile distributions.
- [ ] redshift evolution once other-z profiles available.

## Open data gaps (not blocking Phase 1)
- [ ] exact snap-72 halo mass (`mass_halo`) to define `M0` precisely.
- [ ] `halo_id` cross-match to DiffMAH/Diffstar HDF5 files.
