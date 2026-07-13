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

## Phase 6 — evaluation metrics
- [x] **exp07_evaluation_metrics** — the metrics that judge fit/recovery quality.
  **Result:** (1) score predictors in **aperture/annulus masses** not per-radius
  CoG dex (the 24 CoG points are 93% correlated → double-counting); history cuts
  scatter 19–30% across apertures (shuffle ~0%). (2) Use **CRPS + log-score +
  interval calibration**, not RMS alone: history improves CRPS +24%, and the
  Gaussian-scatter baseline is well-calibrated (90%→0.91) — the bar exp08 must
  beat. (3) Residual scatter is **correlated across apertures** (mean |off-diag|
  = 0.57) → the emulator must draw *correlated* scatter. (4) `cog_sigma_dex`
  propagates `intens_err` → single-sigmoid **reduced chi² median = 1.00**. (5)
  **AIC/BIC must be computed in decorrelated annulus space** (raw CoG over-rewards
  complexity: double-sigmoid preference 94%→65%); single sigmoid is adequate to
  the noise, a 2nd transition is mildly favored/optional (coherent residual
  ≲0.011 dex). Suite graduated to `hongshao/metrics.py` + `tng_data.cog_sigma_dex`.

### Decisions adopted from exp07
- [x] **5 kpc inner cut** is the default radial-DiffMAH CoG fit (`COG_FIT_RMIN_KPC`);
  the inner 2–5 kpc is marginally resolved and was the sole source of the coherent
  residual. Dataset rebuilt: cached `rdm_rms` median 0.0060 → 0.0013 dex.

## Phase 7 — probabilistic emulator
- [x] **exp08_emulator** — conditional multivariate-Gaussian `P(profile | M0,
  MAH-PCA(4))`, mean + full residual covariance, judged by the exp07 suite.
  **Result:** (1) the emulator is well-calibrated (90%→0.91) and cuts CRPS +24%
  vs M0-only (shuffle ~0) — a plain conditional Gaussian suffices, no GP/flow yet.
  (2) **Predict the observable directly:** target A (aperture masses) ≫ target B
  (predict rdm_* params → reconstruct); B only recovers the inner normalization
  and is no better than M0-only in the outskirts. NB (corrected): this is a
  *coordinate-system* problem, not a lack of shape signal — the MAH explains ~24%
  of the at-fixed-M0 variance in concentration (r≈0.45), but the degenerate
  radial-DiffMAH params bury it (fixing Δ DiffMAH-style doesn't recover it).
  Predict the observable masses (or an aligned basis: concentration, PCA modes). (3) **Full covariance beats diagonal** by 1.4–2.0 nats (joint
  log-score); the emulator reproduces the residual correlation (0.50) out of
  sample. Adopt the direct aperture-mass full-covariance emulator (A) as baseline.

## Phase 8 — model-family decision
- [x] **exp09_ceiling_check** — is a richer-than-linear model worth pursuing?
  Compared linear vs poly-2 (analytic) vs a gradient-boosted-trees ceiling on the
  aperture masses (M0+MAH-PCA(4)), CV-CRPS. **Result:** the flexible GBM ceiling
  does **not** beat linear (−0.6%, within noise; shuffle control collapses to
  M0-only, so GBM is working); poly-2 adds a marginal +3.3%. The predictable
  `M*(annulus | M0, MAH)` relation is essentially **linear** — keep the
  closed-form equation; a nonlinear / symbolic-regression form is not justified
  by accuracy. What remains is *scatter*, not a missing mean-shape term. (Adds
  scikit-learn dependency, used only as the achievability ceiling.)

## Phase 9 — portable MAH parameterization
- [x] **exp10_diffmah_fit** — fit DiffMAH (arXiv:2105.05859, k≡3.5) to our own
  MAH curves; four intrinsic per-halo params (logmp, logtc, early, late), anchored
  at z=0.4, fit over t≥2 Gyr (the MAH analog of the 5-kpc CoG cut; early history
  unresolved). **Result:** median fit RMS 0.063 dex (MAH is wigglier than the CoG
  — mergers/bursts a smooth model can't capture; cf. exp06 93% vs 99.7%
  compressibility). The portable DiffMAH params carry **~88% of the MAH-PCA(4)
  assembly signal** (aperture CRPS 0.0882 vs 0.0849; M0-only 0.1117) — the 4% gap
  is the price of smoothing + portability. Adopt DiffMAH params as the portable
  features. Library: `hongshao/diffmah.py`; params in
  `exp10_diffmah_fit/outputs/diffmah_params.csv`.

## Phase 10 — portable emulator
- [x] **exp11_portable_emulator** — rebuilt the exp08 conditional-Gaussian
  emulator (linear mean + full residual covariance) on the cached portable
  DiffMAH params (`dmah_*`, now in the dataset). **Result:** it **matches the
  MAH-PCA(4) version** under the exp07 suite — identical calibration
  (54/72/91/95), full-covariance gain +1.43 nats, reproduces the residual
  correlation (0.52); CRPS 0.0882 vs 0.0849 (the same ~4% portability cost as
  exp10). Probabilistic painting works from DiffMAH features. Adopt the portable
  DiffMAH emulator as the working Ultimate-SHMR model.

## Phase 11 — symbolic regression for the mean (interpretability)
- [x] **exp12_symbolic_regression** — PySR search (polynomial ops only) for a
  parsimonious nonlinear term in the emulator mean. Discovery on the **linear
  residuals** (sharp test: residuals are orthogonal to the linear features, so
  PySR can only find missed nonlinearity) + on the full target (readable form).
  **Result:** the relation is essentially linear (confirms exp09 symbolically).
  The residual nonlinearity is small and lives in the **late-time accretion
  index** `late`: PySR independently picks **`late²`** for the outskirts
  (10–100 kpc) and **`logtc·late`** for the core (<10 kpc), coefficients
  positive + stable across all folds. One term/aperture: CRPS 0.0883 → 0.0865
  (**+2.1%**); the dense 14-term poly-2 ceiling is only +4.6%; calibration
  unchanged. Pareto knee at complexity ~3–5. **Decision:** keep the linear
  closed form; `late²` is optional interpretive polish, not a structural change.
  The lever for real gains is *scatter*, not the mean shape.
- [x] **exp13_outskirt_limit** — ceiling probe for the hardest annulus
  `M*[50-100 kpc]`: feature richness (M0 / DiffMAH(4) / MAH-PCA(4,8) / raw
  MAH(18)) × model (linear / PySR / GBM), CV CRPS+RMS+R². *Deliberately breaks
  portability* to find the limit. **Result:** the MAH lifts R² 0.69→0.81 (CRPS
  −22%) for the outskirts (bigger relative gain than the inner apertures), but
  **DiffMAH(4) is already at the ceiling** — MAH-PCA / full raw-MAH beat it by
  only +2.7% CRPS (R²→0.819) and GBM≈linear (no nonlinear structure). The limit
  is R²≈0.82 / RMS≈0.175 dex; the residual ~43% of the variance is **intrinsic**
  (projection/ICL/low-SB), not feature-limited. Shuffle control collapses below
  the M0 floor (signal real). PySR on raw MAH keeps a *single epoch* (~7 Gyr,
  z≈0.7) at R²=0.80 — the outskirts track recent halo mass. **Decision:** keep
  the portable params; the outskirt lever is the scatter model, not richer
  features.

## Phase 12 — scatter model
- [x] **exp14_scatter_model** — heteroscedastic residual covariance for the
  emulator: per-aperture log-linear sigma_j(X)=exp(gamma_j.[1,X]) (Gaussian MLE,
  ridge on slopes) + fixed correlation R, so Sigma(X)=D(X) R D(X). **Result:**
  the scatter is strongly heteroscedastic, driven by **`late`** (recent
  accretion; +0.16..+0.22 per sigma for the outer annuli; predicted sigma spans
  3.5-5.8x). Modeling it barely changes marginal CRPS (0.0883->0.0873) but
  improves the **joint NLL +0.24 nats** and fixes **conditional calibration**:
  the homoscedastic model over-covers clean halos (0.78-0.82) and under-covers
  noisy ones (0.60-0.64); heteroscedastic is flat ~0.68 (coverage gap
  0.19->0.02, ~10x). PIT shows the Gaussian is adequate once the variance is
  halo-dependent (no heavy tails -> no Student-t needed). `late` is *both* where
  the mean nonlinearity lives (exp12 late^2) and where the scatter concentrates.
  **Decision:** adopt Sigma(X)=D(X) R D(X). The Ultimate-SHMR emulator is now
  specified: linear mean + heteroscedastic full covariance on portable DiffMAH.

## Phase 13 — outskirt-bias diagnostic
- [x] **exp15_outskirt_bias** — investigate the +0.2-0.4 dex over-prediction at
  the low-mass end of M*[50-100] seen in exp13's truth-vs-pred figure.
  **Result:** it is **regression to the mean, not a defect.** The residual-vs-true
  slope is exactly -(1-R^2) (measured -0.190 vs theory -0.190); binned by
  *predicted* value the reliability slope is +0.999 with <0.06 dex deviation, so
  the mean is unbiased in feature space (a nonlinear mean doesn't change it). The
  data is clean (no annulus floor). The low tail is the high-late, high-scatter
  population (exp14). Mean-only predictions are under-dispersed (std 0.371 vs
  0.412) and recover only 5% of the bottom-decile tail; **sampling from the
  predictive recovers it (9% vs true 10%).** **Decision:** no model change; use
  the emulator generatively (sample N(mu,Sigma)), report reliability diagrams.
  Graduation unblocked.

### Adopted model (decision)
- **Default mean = linear on `DiffMAH(4) + c_200c`** (exp16/17/18). `c_200c` is
  portable and adds +5% CRPS; `acc_rate`/3D shape add nothing. The degree-2
  nonlinearity is real but diffuse (exp17).
- **Optional richer mean = the BIC-preferred 7-extra-term degree-2 poly**
  (`logmp²`, `logtc·late`, `early²`, `late·c200c`, `late²`, `logmp·early`,
  `logmp·late` on top of the linear terms; exp17 `poly2_check`) — expose as an
  option at graduation, not the default.
- Scatter = heteroscedastic full covariance (exp14).

### Next
- [x] **exp19_emulator_c200c** — folded `c_200c` into the heteroscedastic
  emulator. It improves the **mean** (CRPS 0.0873→0.0832, +4.7%; joint NLL
  −3.349→−3.432, +0.08 nats) but **not the scatter** (adding it to the
  log-variance = +0.001 nats; `late` stays the scatter driver, +0.16..+0.22/σ vs
  c200c ~+0.03). Conditional calibration 0.018→0.010, marginal unchanged. Working
  emulator = linear mean on DiffMAH+c_200c + heteroscedastic full covariance.
- [x] **exp20 — graduate the emulator into `hongshao/emulator.py`** — one
  `fit`/`predict`/`sample` module: linear mean on `DiffMAH + c_200c` (default,
  54 free params) OR the 7-term degree-2 poly (option), + heteroscedastic full
  covariance, generative `sample()`. Self-check reproduces exp19 (CRPS 0.0832,
  joint NLL −3.432, cond gap 0.010; poly2 mean 0.0800).
- [x] **exp20 — deformation layer (`hongshao/forward.py`)** — a 5-knob,
  physically-labeled deformation of the frozen emulator for *external* inference
  (`d0`, `d_slope`, `d_out`, `f_ab`, `s`); `Deform()` = identity = TNG baseline.
  Predicts deformed stellar masses/profiles with no re-fit. **Scope decision:**
  HongShao stops here — it predicts masses/profiles only; lensing/clustering/
  likelihood/sampler machinery lives in a *separate* inference repo (see AGENTS.md
  "Scope"). The deformer is the hand-off boundary.
- [ ] **portability test (useful future work; skip for now).** Apply the emulator
  to an external N-body halo catalog and compare predicted profile distributions.
  Gated: the one external sim we have provides DiffMAH params but **no `c_200c`**
  (and no stellar profiles — it is gravity-only, so there are no galaxies to
  compare against). Needs a sim with `c_200c` measured + a hydro/painted galaxy
  reference, or the blocked DiffMAH cross-match resolved.

### Later
- [x] secondary halo properties — *test*, don't assume. **Done (exp16/17/18):**
  ceiling is `DiffMAH + c_200c`; `c_200c` is only ~25% MAH-determined and adds
  real independent skill, `acc_rate` is MAH-redundant, 3D shape carries no M* info.
- [ ] redshift evolution once other-z profiles available.

## Data layer updates
- [x] **aperture-mass table merged** (`galaxies_tng300_072_hmc_13_aperture_mass.txt`,
  via `gal_num`==`index`; `tng_data.load_aperture_extras`). Adds the **exact z=0.4
  halo mass** (`logmh_z0p4`; matches the old proxy to +0.004 dex median),
  **secondary halo properties** (`c_200c`, 3D shape, `acc_rate`), and **3 sky
  projections** of aperture masses/galaxy shapes (`logmstar_aper_proj`, `*_proj`).
  Projection scatter is small (≈0.007 dex @100 kpc cumulative, ≈0.037 dex @50–100
  annulus, vs ≈0.17 dex total) → the outskirt residual is mostly genuine
  halo-to-halo variation, not projection. `catgrp_id` (FoF GroupID) stored.
- [x] exact snap-72 halo mass — **resolved** (`logmh_z0p4`).
- [ ] **DiffMAH/Diffstar cross-match**: blocked on an id-system mismatch —
  `catgrp_id` (GroupID) ≠ catalog `halo_id` (SubhaloID). Need `GroupFirstSub`,
  per-galaxy SubhaloID, or 3D positions. Use our own `dmah_*` fits meanwhile.

## New analyses unlocked by the aperture table
- [x] **exp16_secondary_c200c** — does halo concentration help beyond the MAH?
  **Yes (overturns the prior).** `c_200c` improves CV CRPS +5.0% on portable
  DiffMAH(4) (0.0882→0.0839) and **+2.7% even on top of MAH-PCA(4)** (0.0850→
  0.0827); shuffle control collapses (real). It is only ~25% MAH-determined
  (R²(c|DiffMAH)=0.25), partial corr +0.29..+0.36 with the annuli at fixed MAH
  (positive at all radii, strongest in the core). `c_200c` is portable (N-body
  available), so **adopt DiffMAH + c_200c** as the portable feature set.
- [x] **exp17_c200c_nonlinear** — nonlinear limit of `c_200c` (linear/poly-2/GBM/
  PySR on DiffMAH+c_200c). It enters **mostly linearly** (GBM ≈ linear, +0.3%);
  a modest degree-2 bonus exists (poly-2 best, 0.0808 = +3.7% beyond linear+c200c,
  +8.4% over DiffMAH-only; trees can't beat it). PySR names the interpretable
  terms: `c_200c²` (10–50 kpc) and `late·c_200c` (50–100 kpc). Limit of `c_200c`
  ≈ +8% over DiffMAH; nothing beyond degree-2.
- [x] **exp18_secondary_more** — `acc_rate` + 3D shape, same test. **Only `c_200c`
  helps.** `acc_rate` is MAH-redundant (+0.9% on DiffMAH, ~0 on MAH-PCA; most
  MAH-determined, R²=0.35); 3D shape is MAH-independent (R²≈0) but carries no
  stellar-mass info (gains ~0). "All 4" ≈ `c_200c` alone. Secondary-property axis
  exhausted: ceiling is DiffMAH + `c_200c`.
- [x] does `c_200c` also help the **scatter** model? **Answered (exp19): no.** It
  improves the mean (+4.7% CRPS) but not the scatter (+0.001 nats; `late` stays the
  driver). The mean's higher R² modestly shrinks exp15's regression-to-mean slope
  (slope = −(1−R²)).
- [ ] **projection-scatter budget** — quantify projection vs intrinsic halo-to-halo
  scatter per annulus from the 3 projections (refines exp13/14/15's "intrinsic" floor).

## Model-target extensions (post-graduation; independent of the frozen emulator)
- [x] **exp21_re_based_bins** — aperture/outskirt masses in **Re units** instead
  of fixed kpc. Re = half-mass radius within 120 kpc from the CoG; six bins
  `<0.5/0.5-1/1-2/2-4/4-6/6-9 Re` (median Re≈10 kpc). Same emulator approach
  (local n-bin CV; library untouched). **Results:** (1) the 6-bin Re emulator is
  well-calibrated (mean CRPS 0.0701) — but `6-9 Re` costs a **19% mass-correlated
  selection** (9 Re > 148 kpc CoG limit for large galaxies) and probes past the
  120 kpc observational limit, so keep Re-binning as an exploration, not the
  default. (2) **A richer MAH does NOT help the outskirts:** MAH-PCA(8)+c200c
  +1.0%, raw-MAH(18)+c200c +0.6% over DiffMAH(4)+c200c, flat-to-core-weighted
  (NOT rising outward; `6-9Re` gains ~0). Confirms/extends exp13 to the Re frame
  — DiffMAH(4) is at the MAH ceiling at every radius; the outskirt residual is
  intrinsic, not feature-limited. **Keep DiffMAH+c_200c** (portable AND at the
  ceiling).
- [x] **exp22_full_profile_predict** — predict the *whole* CoG (Option 1). Compress
  each CoG to `[logMtot, PC1, PC2, PC3]` (exp02 shape PCA, fit in-fold), predict the
  4-vector from `DiffMAH+c_200c` with the heteroscedastic emulator, reconstruct the
  per-radius CoG **analytically** (linear → Gaussian per radius). **Result:** the
  full profile is predictable and well-calibrated (recon RMS 0.116 dex — dominated
  by the total-mass SHMR scatter; coverage on the line). Predicting *shape* adds
  **+10% per-radius CRPS** over a mass+mean-shape baseline, peaking ~17% at ~10-20
  kpc; almost all of it is PC1=concentration (R²=0.39, via `c_200c`), PC2 0.33,
  PC3 0.05. Beyond total mass + concentration the shape is largely intrinsic.
  - **exp22b (`density_profile.py`)** — PCA the 1-D **density** profile `Σ(R)`
    (differenced from the CoG) instead of the cumulative CoG. **The density is
    more halo-predictable:** value of predicting shape +15.7% vs +10.0%, PC1
    R²=0.54 vs 0.39, and — unlike the CoG — it keeps ~30% shape value in the
    **outskirts** (>50 kpc) where the cumulative mass carries none. For
    envelope/ICL modeling, predict `Σ(R)`, not `M(<R)`.
  - **generative demo (`exp22_generative.png`)** — the derived-aperture "bias" is
    regression to the mean (slope −(1−R²)), the mean is unbiased in feature space,
    and sampling the predictive restores the population (std 0.41=0.41); the
    point estimate is under-dispersed by construction.
- [x] **exp23 — graduate all four prediction modes into the library.** Generalized
  `hongshao/emulator.py` to **N targets** (was hardcoded 4; exp19 still reproduced
  exactly) and added `hongshao/profile_emulator.py`: `ProfileEmulator`
  (fit/predict/sample of a full 1-D profile via PCA→core emulator→linear
  reconstruction) + target builders `aperture_targets` (kpc), `re_targets` (Re),
  `density_from_cog`, and `integrate_density` (stable outward CoG integration). One
  self-check reproduces all four: (1) kpc CRPS 0.0832, (2) Re 6-bin CRPS 0.0703
  calibrated, (3) CoG profile recon RMS 0.118 / PC1 R²=0.39, (4) density PC1
  R²=0.54. All four share the one heteroscedastic core; library only.
- [x] **exp23 — generalize the deformer (`hongshao/forward.py`).** The 5-knob
  deformation now spans all four modes: `forward`/`sample` are target-agnostic
  (`norm_weight`/`outer_weight` default to all-bins / last-bin, so any-T aperture
  emulator works), and `forward_profile` deforms a `ProfileEmulator` — `d0` a
  uniform profile shift, `d_out` an inner↔outer redistribution (pure shape, built
  by projecting an outer-radius weight onto the PCA modes), `f_ab` the
  assembly-bias amplitude, `s` the scatter. `Deform()` reproduces every mode
  exactly. Self-check covers T=4, T=6, and the profile path.
- [x] **exp24_rdm_c200c** — does `c_200c` predict the RDM shape parameters the
  MAH couldn't? **Hypothesis refuted.** `c_200c` helps only the normalization
  (`logMstar0` R² 0.61→0.65, partial r +0.33 — the total-mass effect); the shape
  params (`beta_in`, `beta_out`, `R_c`) stay weakly predictable and gain ≈0 from
  `c_200c` (≤0.006 R², partial r ≤0.08). The exp03/04/05 "weak halo→shape-param
  connection" verdict stands, now extended to DiffMAH+c_200c. Reconstruction ties
  the PCA route (CRPS 0.0605, recon RMS 0.108 dex on R≥5 kpc) because the profile
  is normalization-dominated and the shape signal hides in a degenerate
  combination. Concentration sets *how much* mass, not *how it's distributed*.
- [~] **2-Sérsic on the density profile (prototyped, not pursued).** A 6-param
  2-Sérsic (cumulative-to-CoG, multi-start, ordered Re1<Re2) fits the profiles
  excellently (median 0.0008 dex, better than RDM/PCA) but the parameters are
  **degenerate / non-identifiable**: `n` hits bounds in ~¼ of galaxies, components
  merge or swap, masses jump between similar galaxies. 6 params is too many for an
  intrinsically ~2–3D profile (exp02), so a low halo→param correlation would be
  confounded by fit non-uniqueness, not physics. **Decision: not worth pursuing**
  (consistent with exp24 — more/finer profile parameters don't buy a stronger halo
  connection; the shape residual is intrinsic). If revisited, use *identifiable*
  derived descriptors (R50, R80/R20, outer mass fraction), not raw Sérsic params.
- [x] **exp25_deposition_kernel** — physics-inspired forward toy: build a galaxy's
  1-D profile directly from its *actual* MAH (no in-situ/ex-situ labels). Each
  halo-mass increment deposits `ε(z)·dM_h` of stars as a centred, mass-normalized
  2-D Gaussian of width `σ(t)` (amplitude not free; closed-form CoG). On TNG300's
  most massive galaxy: width `σ(t)=σ_0(t/t_obs)^g` (tested vs `R_200c`-tied —
  equally good, so R_200c dropped) + **two-epoch quenching efficiency** (steep
  early `b_early≈8.7` until `z_c≈5`, then shallow `b_late≈1.1`) reproduces the BCG
  to **0.008 dex** (vs 0.028 for a single power-law). Caveat: ε>1 at the 2 earliest
  epochs (2.8% of M*; capping at f_b keeps 0.007).
- [x] **exp25 phase 2 — population fit** (`population_fit.py`, n=2540). The forward
  map generalizes: the toy reconstructs *every* clean galaxy's CoG to median
  **0.0045 dex** (5-param), and **0.022 dex** with a reduced model (shape frozen at
  the population median `g=1.67, b_early=3.62, b_late=1.15`; only `σ_0, z_c` free).
  The shape params are **~universal** (|r(logMₕ)|≤0.08 for all five). **Headline:
  `z_c` does NOT scale with halo mass** — slope −0.55/dex but **r=−0.09** (r²<0.01,
  opposite the predicted sign); also uncorrelated with `M*` (r=0.01) and `z50`
  (r=0.07). The quenching-mass prediction is **refuted**; the phase-1 single-galaxy
  `z_c≈5` was a degenerate steep-burst solution (relaxes to ~1.8 under the ε≤1 cap).
  Method: cap ε≤1 (NOT f_b — f_b distorts low-mass halos differentially and would
  fake a trend). Toy's value = interpretable forward MAH→profile map, not a new
  predictor or quenching clock.
- [x] **exp25 phase 3 — TRUE population fit** (one *shared* kernel for all 2540
  galaxies, vectorized; Phase 2 was 2540 independent fits). A single universal
  parameter set reconstructs every profile to median **~0.08 dex** (global A 5p
  0.083; B with σ₀=f(R50) 0.080), below the emulator's 0.116. Cost-of-universality
  ladder: free 0.0045 → reduced 0.022 → global 0.080 (≈×4/step). **(1)** A
  population-level `z_c(logMₕ)` slope changes RMS by −0.0003 → quenching-mass trend
  refuted in its strongest form. **(2)** σ₀–R50 is shallow (slope 0.33): width set
  by cosmic time, not final size. **(3)** The shared optimum **inverts** the
  per-galaxy steep-early shape to steep-late (b_early≈0.2<b_late≈4, g≈2.4; robust,
  3/4 starts) — a `(g,b_early,b_late)` degeneracy, so reconstruction accuracy & the
  null z_c trend are robust but the efficiency-slope *interpretation* is not.
  Suggested follow-on (parked): hierarchical/mixed-effects fit (partial pooling) to
  put proper uncertainties on the population mean/scatter/mass-slope of each param.
- [~] **exp26 — differential stellar-density profiles (in progress, branch
  `exp26_differential_profiles`).** Tests the exp25 centred-Gaussian deposition
  assumption against the *measured* z-evolution: the drop has surface-density
  profiles at 5 epochs (z=0.4/0.7/1.0/1.5/2.0 = snaps 72/59/50/40/33; flag=True for
  3380/3388 at every z). Built & stored `ΔΣ(R)=Σ(later)−Σ(earlier)` for the 4
  adjacent pairs (n=2545). **Result: the added mass is NOT a centred Gaussian** —
  growth is **inside-out and multiplicative**, `Σ_low/Σ_high ∝ R^b` with the
  fractional growth `ΔlogΣ` rising ~linearly in `logR` (long-baseline z=2→0.4
  b=+0.85; outer 60 kpc grows ~22× vs inner 8 kpc ~3.8×; only ~3% show a central-
  density drop). A Gaussian piles mass at R=0; reality flattens/extends the profile.
  Implication: replace the centred-Gaussian primitive with a multiplicative power-
  law amplification (or an outer-weighted/shell deposit). Caveats: 5 epochs only,
  z≤2, adjacent-pair ΔΣ noisy (use stacked median / long baseline), ellipticity
  normalization, inner <6 kpc marginally resolved.
- [x] **exp27 — TNG-API cross-match to DiffMAH/DiffStar (branch
  `exp27_tng_api_crossmatch`).** Bridge is **position, not group ID**: the
  `diffmah_tng.h5` `x/y/z` arrays are the main-progenitor positions with **column
  index == snapshot number**, so column 72 (cMpc/h) == our snap-72 `SubhaloPos`
  (ckpc/h ÷1000) *exactly* when our subhalo is a z=0 main progenitor. Pulled all
  3388 MPBs from the API (30 min, 0 fail), KDTree-matched at 1 ckpc/h:
  **3154/3388 matched (93.1%)**, 234 off-main-branch (no z=0 descendant → no
  DiffMAH row). Input `logMh(z=0.4)`≡official M200c snap72 to +0.000 dex.
  Outputs `crossmatch.fits` (+DiffMAH/DiffStar params) and `official_mah.npz`.
- [~] **exp28 — summed-accreted-mass MAH (branch `exp28_summed_accreted_mah`).**
  Built `M_sum(z)` = Σ exclusive `SubhaloMass` over all SubLink-tree progenitors
  per snapshot, from `…/sublink/full.hdf5` (trees in `/Users/mac/work/tng`, NOT
  Dropbox). Two example halos (logMh≈13.5, clean + declining): summed-accreted is
  smooth, +0.07–0.27 dex above the main-branch Mpeak, recovers the real progenitor
  where the main branch drops out. **Lit review (`doc/summed_accreted_mah.md`):
  `M_sum(z)` is NOT a standard named quantity, captures merger-accreted-only
  (excludes ~40% smooth accretion, Genel+2010), is resolution-dependent → not a
  drop-in `M200c`.** Next: (a) infall-peak-sum variant (monotonic, USMF-grounded);
  (b) mass-threshold robustness; (c) scale to the matched subset — biggest full
  tree ≈300 MB/290k rows → stream-walk, don't cache every tree.
- [~] **Full-tree fetch campaign (`scripts/fetch_full_trees.py`).** Cost measured
  (exp28): the full-tree API generates each tree server-side, ~30–100 s, **serial
  only** (concurrency → 503), so all 3154 ≈ **2–3 days**; disk ~40 GB if kept, ~0 if
  streamed; bulk download is 1.5 TB (off the table). Utility fetches in resumable
  chunks (most-massive-first) to `/Users/mac/work/tng/full_trees/` with a worklist +
  `PROGRESS.md` + `fetch_log.csv`. **Chunk 1 = the 10 most massive.** To continue:
  `uv run python scripts/fetch_full_trees.py --next N` ("grab the next chunk").
  Parked for now (laptop / travel).
- [x] **exp29 — deposition kernel on a dip-free MAH; primitive question answered
  (branch `exp29-outer-deposit-kernel`).** (1) Dip-free MAH = official **DiffMAH fit
  curve** (`official_mah.npz`, monotonic, all galaxies) replaces dippy
  `peak_history`. (2) Generalized deposit `deposit.py` (`p`: 0=Gaussian, >0=shell).
  Tested on the **stacked multi-epoch differential** `b(z)` (n=2399, exp26 cache,
  fit over valid radii). **Result: the CENTRED Gaussian (p=0) with σ(t)=σ₀(t/t_obs)^
  0.55 reproduces the whole inside-out trend** (data b 0.13/0.14/0.27/0.33/0.82 →
  model 0.11/0.13/0.28/0.31/0.82, mean |Δb|=0.012); **outer-weighting p>0
  monotonically undershoots** (long-baseline 0.82→0.53→0.31). σ(t), not deposit
  off-centredness, is the inside-out mechanism — exp26's "not a Gaussian" was about a
  *single* deposit. The multi-epoch data pin g≈0.55 (the z=0.4 CoG alone couldn't).
- [x] **exp29 — independent single-epoch fits to every snapshot
  (`single_epoch_all.py`, branch `exp29-single-epoch-highz`).** Settles whether the
  centred-Gaussian deposit *shape* has a fundamental high-z limit. Per galaxy, per
  epoch z_k∈{0.4,0.7,1.0,1.5,2.0}: fit the kernel to **that epoch's CoG alone**
  (deposits up to t(z_k), normalization pinned to the 148-kpc aperture). Honest
  metric: linear M*, relative residual, max/90th-pct over R>3 kpc. **Result (n=60):
  the Gaussian shape is NOT the limit** — every epoch, every mass tertile fits to
  ≤1.4% max-rel; high-mass z=2 (0.9%) ≈ z=0.4 (0.7%); the BCG's z=2 is its *best*
  fit (0.3%). → The multi-epoch tension is a **consistency** problem, not a shape
  one → **build the puff-up model** (PUFF_MODEL_PLAN.md, now un-parked).
- [x] **exp29 — single-epoch best-fit parameter trends (`param_trends.py`).** Mined
  the cached per-epoch fits for patterns. **(1)** `g≈1.7` is epoch-stable (matches
  exp25) — the spatial-kernel shape is a shared invariant. **(2)** The efficiency
  rotates: `b_early` 3.2→5.8 toward high z. **(3)** Robust puff-up calibration: the
  `R50` of the pre-z=2 mass (anchored — at z=2 it equals the data `R50`) grows
  **3.0→6.1 kpc (all), 3.0→8.2 kpc (high-mass)** from z=2 to z=0.4 → the same early
  mass must extend **~1.8× (≈2.7× for BCGs)**. The single-epoch fits realize this via
  the efficiency (a per-epoch freedom the joint fit lacks) → confirms the joint model
  needs an explicit extra DOF, and `R50`-doubling sets the puff-law magnitude.
- [x] **exp29 — loose redshift-dependent-parameter joint fit (`loose_zdep.py`).**
  Each kernel param a polynomial in observation z (const/linear/quad), fit jointly,
  vs the independent ceiling. **Result (n=60): epoch-avg max|rel| fixed 10.2% →
  linear 4.8% → quad 4.5%, ceiling 0.7%.** z-dependence ~halves the error (reasonable
  multi-epoch fit) but plateaus ~6× above the ceiling — quad≈linear, middle epochs
  (z=0.7/1.0) hardest — because the degenerate per-epoch params don't lie on a
  low-order z-curve. So generic z-floating ≠ single-epoch quality; need *structured*
  DOF. **Benchmark for the puff-up model to beat: ~4.5% epoch-avg.**
- [x] **exp29 — built & tested the puff-up deposition model (`puff_fit.py`).** One
  consistent history (mass frozen), only widths migrate post-deposition: ratio law
  `σ₀(t_i/t_obs)^g (t_k/t_i)^q` and diffusion law `√(σ_{i,0}²+κ(t_k−t_i))`, fit jointly
  (6 params) vs no-puff / loose-zdep / ceiling. **Result (n=60, epoch-avg max|rel|):
  no-puff 9.1% → ratio 7.1% → diff 7.7%, loose-zdep ~4.5%, ceiling 0.7%.** Puffing
  helps but does NOT clear the looser z-dependent fit; diffusion nearly inert (κ→0).
  → with mass frozen, width migration is a weaker lever than epoch-dependent mass
  distribution. (Matches param-trends: single-epoch fits de-concentrate early mass via
  the efficiency, not the width.)
- [x] **exp29 — free-mass NNLS floor (`nnls_floor.py`).** Gave every deposit a free
  non-negative mass (convex NNLS), one shared mass vector = one consistent history.
  **Decisive (n=60, max|rel|): free masses fit each epoch ALONE to 0.2%, but the
  consistent JOINT fit caps at 12% (~60×).** Free masses do NOT relieve the multi-epoch
  tension; the binding limit is the single consistent additive Gaussian-sum history
  itself, not the mass parameterization. Parametric joint models (loose 4.5%, puff 7%)
  do better only by relaxing consistency or adding width freedom. Reaching the 0.7%
  ceiling would need a NON-additive primitive (mass that moves, not just adds).
- [x] **exp29 — real-MAH / no-inner-cut re-test (`real_mah_test.py`).** `dipfree_mah`
  used the SMOOTH DiffMAH fit (no merger bursts); re-tested with the real de-dipped
  main-branch MAH (`peak_history`) and no inner cut. **Result (n=50, loose-quad
  epoch-avg max|rel|): smooth/R>3 4.4% → real/R>3 6.1% → real/all-R 8.9%** (smooth/all-R
  7.0%). Both changes make the fit worse and ~add; per-epoch ceiling stays ~2% (shape
  still fine). Smooth curve flattered the model. Real MAH is the honest input AND carries
  the merger events needed for an event-driven width model.
- [x] **exp29 — corrected multi-epoch evaluation (`integrated_check.py`): real MAH,
  ALL radii, + integrated aperture/outskirt mass checks.** No inner mask (high-z Re<3
  kpc → inner holds most of the mass). **Result (n=45): loose-quad profile max|rel|
  epoch-avg ~10% (ceiling ~2%); cumulative aperture masses M*(<10..<100) reproduced to
  ~0.01 dex; but outskirt M*(>50 kpc) under-predicted up to 0.31 dex (~2x) at z=2, worst
  for massive galaxies** — the centred-Gaussian sum can't build the extended high-z
  outskirt. Integrated outskirt mass is the sensitive diagnostic.
- [x] **exp29 — standardized mass QA (`mass_qa.py`).** Reusable `evaluate()` with two
  bin sets (kpc + R_half) x {aperture M*(<R), envelope M*(>R)} and the two QA figure
  types (truth-vs-model values; truth-vs-relerror), colored by epoch. Insight: loose
  model reproduces kpc apertures ~1% and ALL R_half quantities ~few% at every epoch, but
  under-fills the fixed-kpc far outskirt at high z (M*(>100 kpc) -88% at z=2) -- a
  far-tail/absolute-radius effect, NOT a shape error (R_half envelopes are fine).
- **STANDARD GOING FORWARD**: after every fit run (1) profile max|rel| over ALL radii
  and (2) `mass_qa.evaluate()` (kpc + R_half aperture/envelope masses). Honest
  best-model profile number is ~10% (not the inner-masked 4.5%).
- [ ] **(consider) switch the model's default MAH to the real de-dipped `peak_history`**
  (currently `dipfree_mah` = smooth DiffMAH fit). Would change every exp29 number.
  Decide before any final emulator numbers.
- [x] **exp29 — honest final scorecard (`final_scorecard.py`).** All 4 models on the
  corrected standard (real MAH, all radii, profile + mass QA). n=45 profile max|rel|
  epoch-avg: **independent 1.9% (ceiling), loose-quad 9.9%, puff 10.9%, free-mass floor
  18.5%.** Free-mass floor is worst in max|rel| (L2 objective spikes the worst radius).
  All nail cumulative apertures (~0.01 dex) + relative outskirt M*(>2Re) (~0.02 dex);
  only fixed-kpc far outskirt at high z fails (loose −0.31 dex at z=2).
- [x] **exp30 — transport-kernel feasibility gate PASSED (`transport_floor.py`, branch
  `exp30-transport-kernel`).** Core-retaining redistribution: deposit splits into a
  retained core + migrated envelope, mass-conserving, dt-only observation dependence,
  CoG linear in masses → NNLS inner + 4-5 param outer. **n=45 (real MAH, all radii,
  median max|rel| epoch-avg): additive floor 18.5% → transport 9.1% (envelope 11.3%),
  loose-zdep 9.9%, ceiling 0.2%.** A consistent history now BEATS the inconsistent
  loose fit — redistribution is the missing freedom. Remaining gaps: still ~9% vs the
  0.2% ceiling; fixed-kpc far outskirt at high z unfixed by the global clock (the
  dynamical-clock envelope variant fixes z=2 on BCGs); n=5→n=45 flip suggests
  mass-dependent preference between the two clocks.
- [x] **exp30 phase 2.1 — combined clock: 2×2 completed, winner `dyntrans` 7.5%.**
  The intuitive combination (two-param clock τ₀+α·tᵢ + shared obs-epoch width, 7p)
  collapsed onto envelope (11.7%) — the shared width was the flaw, not the clock.
  Completing the {clock}×{width form} factorial exposed the untested cell: **dyntrans
  = dynamical clock + multi-scale per-deposit width, 4 params, 7.5% epoch-avg max|rel|
  (n=45, real MAH, all radii)** — best at every epoch among consistent models, beats
  loose-zdep (9.9%) and every IC (k_eff=14, rel-RMS 3.4%, ΔAICc 495 vs 600). Fitted
  clock is self-similar: α≈1.04 → migration timescale = cosmic time at deposition;
  q≈1.6. Mass QA: apertures ≤1%; R_half envelopes ≤1.8% (M>2Re) at every epoch; only
  the z=2 fixed-kpc far tail remains (M(>50) −86%).
- [x] **exp30 phase 2.2 — event-triggered kicks: clean NEGATIVE (`event_kicks.py`).**
  Pre-test null (fitted α vs MAH burstiness, ρ≈0); event-clock model underperforms the
  smooth self-similar clock at every threshold (10.3–12.1% vs dyntrans 7.5%, n=45),
  monotonically worse with fewer events; no per-galaxy scatter reduction. Halo-MAH-step
  timing ≠ stellar redistribution timing (dynamical-friction delays; continuous
  relaxation). **Keep dyntrans (τ≈tᵢ).**
- [x] **exp30 phase 2.3 — LOEO generalization: the in-sample ranking INVERTS
  (`holdout.py`).** n=45, held-avg max|rel|: additive 30.9% (gap +11), loose-quad 35.3%
  (+26), dyntrans 53.7% (+46) — the best in-sample model is the worst predictor. The
  discriminator is the mass parameterization: free NNLS masses absorb epoch-specific
  information; parametric-mass loose degrades less; rigid additive least. Totals
  predict fine (dyntrans |dlog M*| 0.06–0.16); the SHAPE overfits. No current model
  predicts acceptably (all ≥30%).
- [x] **exp30 phase 2.2b — lagged event kicks (`event_kicks.py lagged`).** Coalescence
  delay t'=(1+β)t_j, β free. Lag genuinely helps events (10.3→9.5%) with a coherent
  physical delay (median β=0.37, IQR 0.30–0.72 ≈ dynamical friction; echoes α≈1), but
  still trails the smooth clock (7.5%). **τ≈tᵢ is the delay-averaged merger clock**;
  discreteness adds nothing in-sample. Ex-situ channel (v2) now rests on the
  dual-region merger deposit, gated on the v1 burstiness-residual diagnostic.
- [x] **exp30 phase 3 v1 — parametric-mass transport emulator: IT PREDICTS
  (`param_emulator.py`).** 7 params (4 transport + 3 efficiency), zero free masses.
  **In-sample 9.7%** (only +2.2 over free-mass dyntrans); **LOEO held-avg 24.0%, gap
  +14.3, including the z=0.4 forward holdout (31.4%)** — beats every 2.3 model
  (additive 30.9, loose 35.3, dyntrans-free 53.7). Fitted params physical (α=1.01
  self-similar again; b_early 4.5, b_late 1.9, z_c 2.2). Native mass growth 0.075→0.31
  dex (z=0.7→2). **v2 gate CLOSED** (residual-burstiness ρ≈0) — no evidence for the
  dual-region ex-situ channel at current precision. Mass QA: apertures ≤3%, R_half
  envelopes ≤5.5%, known z≥1.5 far-kpc tail.
- [x] **exp30 phase 4 — the population/forward step (`pop_forward.py`).** LOGO
  (leave-galaxy-out = the new-halo number), n=45, median profile max|rel| all radii,
  vs the 10.2% per-galaxy floor: **(i) universal θ 33.6% real / 30.6% DiffMAH-input**,
  in-sample ≈ LOGO (gap ~1) → a CAPACITY limit, not overfitting — the MAH through
  the transport model + one shared θ leaves a +20-point individuality gap. Median of
  per-galaxy θ is useless (55–82%): the per-galaxy fits are degenerate (b_early 3–44,
  z_c 1.5–48), so population θ must be refit jointly through the data. **(ii)
  halo-conditioning**: no θ–halo correlation passes p<0.01 (best log_s0←c200c ρ=+0.36
  p=0.014); width←c200c (4e's Lc, p<0.05 selection, LOGO promotion) gives −1.6 real
  but +1.6 diffmah — marginal, not robust. **(iii) end-to-end** with per-epoch
  MAH-derived SHMR (0.10–0.18 dex): 44.9% real / 39.8% diffmah. **The
  DiffMAH-parameter input is validated and FREE at the population level** (30.6 vs
  33.6 — the ~2% per-galaxy penalty vanishes for a shared θ) → the differentiable
  configuration is the product recommendation. Bounded f(z) (population-informed box;
  the unbounded fit rails z_c) restores identifiability but does NOT improve LOGO —
  rejected. Integrated aperture masses stay few-% (mass QA); the 30% is the
  worst-radius shape metric.
- [x] **exp31 — standardized tiered QA harness + the honest scoreboard.** Part A:
  `hongshao/qa.py` graduated (apertures, annuli, envelopes, observational planes,
  profile max|rel| all-R AND R>5 kpc, mass-tercile + best/worst visual QA; synthetic
  demo self-check). Part B (LOGO, n=45, halo-only): **tier 1 is feature-limited**
  (all four models within ~0.02 dex on apertures; direct regression 0.083 at z=0.4);
  **transport-diffmah wins tier 2 in FIXED KPC** (annuli/envelopes 2–5× tighter at
  z≥1.5) but in **Re units tier 2 is feature-limited too** (all models ~0.19–0.27
  dex at z≥1.5; regressions marginally ahead on envelopes) — report both units;
  the unit-independent win is **tier 2b** (plane fidelity |Δscatter| 0.137 vs
  direct 0.398 — regression-to-the-mean shrinks the predicted planes; the forward
  model propagates MAH diversity);
  tier-3 medians tie (31.6% vs 32.0% over R>5 kpc). **Product configuration
  confirmed: transport-diffmah universal-θ.** Caveat: even its planes are tighter
  than truth (slope 1.88→1.15, scatter 0.206→0.108 at z=0.4) — the individuality
  gap and the plane shortfall are the same open problem.
  HongShao's core = a forward-model engine (real-MAH or DiffMAH input); basic goal =
  aperture/outskirt masses per epoch, ambitious goal = full profile evolution. The QA
  must score every approach identically across that ladder. **Part A — graduate the
  evaluation stack to `hongshao/qa.py`**, one entry point (predicted 5-epoch CoGs,
  truth) -> (1) aperture masses, kpc {<10,<30,<50,<100} + Re {<1,<2,<4}: per-epoch
  bias + dex scatter; (2) ANNULUS masses [10,30],[30,50],[50,100],[100,150] kpc +
  Re analogs [1,2],[2,4] + outskirts >50,>100 kpc / >2,>4 Re; (2b) OBSERVATIONAL
  PLANES — joint 2-D distributions, e.g. M*(<30 kpc) vs M*[50,100 kpc] (and the Re
  analog): truth-vs-prediction overlay + the relation's slope/scatter in both samples
  (this plane is what real observations use — the predicted DISTRIBUTION must match,
  not just per-galaxy residuals); (3) profile max|rel| quoted over ALL radii AND
  R>5 kpc (inner 2-5 kpc marginally resolved, exp07) + the two visual products
  (median CoG by mass tercile with residuals; best/worst gallery). **Part B — LOGO
  scoreboard** on the same n=45: transport universal-theta end-to-end (real +
  DiffMAH, SHMR amplitude) vs (i) logMh-only per-quantity regression baseline and
  (ii) exp08-pattern direct statistical emulator (halo-only features -> each mass
  directly; no profile, no consistency — the tier-1/2 ceiling). Decides whether the
  tier-3 individuality push is needed or the basic goal is already served.
- [x] **exp32 steps 1–3 — the full-population emulator (n=2397, logM* 10.66–12.36).**
  (1) Cache + per-galaxy θ atlas both configs (~10% floor holds over the whole
  mass range, rising 9.7→13.7% with mass; the historical n=45 was a stratified
  every-41st subsample, not the top-45). (2) Universal θ: held-out 30.4% with
  ZERO CV gap (capacity confirmed at n=2397); mass-conditioning +1 point
  (29.4%), continuous θ(logMh) = binning with half the params → adopted;
  DiffMAH stays ~5 points ahead of real-MAH; ANATOMY: individuality is NOT
  log_s0 (0.2% gap closure) — g/q/b_early/b_late/z_c each close 35–40% → a
  degenerate width-growth × efficiency-shape subspace. (3) Scoreboard vs mass:
  **epoch-matched history features (direct-epoch) are the best per-quantity
  regression at every tier/quartile (apertures 0.139 dex, R>5 max|rel| 26.9%) —
  the exp31 "MAH decays with z" was feature misalignment**; but the better the
  regression, the worse its plane fidelity (direct-epoch 0.542 vs transport
  0.19–0.21) — the per-quantity/distribution trade-off is now sharp and no
  model wins both.
- [ ] **(PARKED, user decision 2026-07-11) exp32 step 4 — the stochastic layer**
  (correlated θ-deviations in the anatomy subspace; target centered plane
  energy → ~1). Step 5 (graduation) explicitly NOT reached: the multi-epoch
  model is not good enough to graduate; the fundamental (mean-model) side needs
  work first. Revisit after the single-epoch consolidation below.
- [ ] **(next, data-side fix for the aperture-horizon degeneracy) asymptotic
  total stellar mass via CoG extrapolation.** Raw drop audited: NOTHING beyond
  ~160 kpc (profiles to 159.7, apertures to 150). User decision: prefer a
  methodology-consistent total from extrapolating our own CoG (careful choice
  of functional form + radial fitting range; exp29 `cog_extrapolate.py` is the
  prototype) over TNG SubhaloMassType (different definition). Then refit the
  transport model normalized to the TOTAL, with the aperture fraction
  M*(<150)/M*_total as a fitted datum per epoch — the geometric-deletion basin
  becomes falsified by data, and the emulator gains an honest native
  mass-growth target. Easiest/most binding at z<=1.0.
- [x] **exp33 — single-epoch consolidation COMPLETE (verdict in the README).**
  (i–ii) frozen spec reproduces its record; generative sample() PASSES the 2-D
  planes at 0.8–1.0x floor in native targets (first models to do so); PCA-3
  fails only in Re coordinates. (iii) features at the limit (+2.1% ceiling;
  burst real but not worth breaking portability; DiffMAH encoding beats raw
  summaries by 8%). (iv) statistical and transport residuals correlate at
  rho=0.82–0.89 — one shared information wall; consistency tax ~3 points,
  form ~0.5 (the 2x2 completed via the z=0.4-only transport fit). (repr)
  size-aware/core-split/density representations: clean negative. (vi) epoch
  connection: coefficient-interpolation closure +-4%; cross-epoch residuals
  are AR(1) with rho=0.67. DISCOVERY: the aperture-horizon degeneracy (fits
  delete epochs geometrically; per-epoch pinning hides it) -> physical 5-param
  refit (+3–4 points, g rails) -> exp34 asymptotic totals: f_out is REAL
  (12% median, 26% massive quartile at z=0.4); differential deposition
  measured (37%/11% of late growth beyond 50/100 kpc, massive). Graduation:
  the stack stays as-is with its documented error budget; nothing new
  graduates; headroom belongs to new inputs only.
- [ ] **(next A, product path) the multi-epoch statistical emulator from the
  exp33-vi blueprint**: continuous-z coefficient interpolation + AR(1)-in-epoch
  latent (rho=0.67) + generative sampling; judged by the full QA incl. planes
  per epoch and cross-epoch coherence.
- [x] **exp35 — the TOTAL-NORMALIZED transport refit COMPLETE (verdict in the
  README; branch exp35-total-norm).** M(<500) datum from exp34 power tails
  (form-sys 0.011 dex); aperture fraction fitted, not pinned. Physicality tax
  shrinks +3–4 -> +1.4–2.3 points (multi-slope held-out shape 20.5% vs the
  19.1% mark) with every z>=1 bin visible and f148 reproduced to 0.016–0.004
  dex; all four fits in ONE basin. **The differential-deposition test PASSES**
  (massive tercile z0.7->0.4: measured 0.37/0.11 vs model 0.40/0.12, mass
  trend reproduced) — the transport family's unique physics claim now has a
  passed out-of-model test. Consistency tax vs the single-epoch statistical
  emulator persists (~5 points, as expected). Remaining tension: log_s0/g rail
  at the loose bounds and the massive-end f148 is under-spread (model 0.875 vs
  data 0.83) — the data want more outward transport at the massive end.
  Decision input: Path A stays the accuracy product; the transport kernel is
  the physics companion.
- [ ] **(superseded by the exp33 verdict) exp33 original step list.** Review
  finding: the graduated stack (`hongshao/emulator.py` heteroscedastic
  conditional Gaussian on [DiffMAH(4) + c200c]; `profile_emulator.py` modes
  1–4 = kpc apertures / Re apertures / CoG / density profile; `forward.py`
  deformation knobs) predates the standardized QA and has NEVER been scored on
  it — its record is CRPS/NLL/coverage (exp19: CV CRPS ~0.083; exp22: +10%
  per-radius CRPS from shape, mostly c200c via PC1). Plan: **(i)** refit the
  frozen spec on the current z=0.4 sample (n~2545), 5-fold CV; **(ii)** run
  `hongshao/qa.py` on every mode — point-prediction tiers (apertures/annuli/
  outskirts kpc+Re, bias + dex scatter) AND the GENERATIVE test the stack has
  never had: score its `sample()` draws on the observational planes
  (energy/floor full + centered) — "generative and calibrated" was claimed
  from 1-D coverage only; the 2-D plane test is the honest version. Profile
  modes (3)/(4) additionally get tier 3 (max|rel| all-R and R>5 kpc) + the
  visual QA (mass-tercile medians, best/worst gallery); **(iii)** feature
  increments at z=0.4 under the new harness with shuffle controls: DiffMAH+
  c200c (portable baseline) vs + burstiness (never tested as a feature), real-
  MAH t50/fz2 vs the smooth DiffMAH params (exp29 lesson: the smooth curve can
  flatter), acc_rate; **(iv)** physical-vs-statistical CoG head-to-head at
  z=0.4: exp32's universal-θ transport CoG (pinned) vs mode (3)/(4), same QA;
  **(v)** verdict: the single-epoch error budget, whether the generative layer
  passes the plane test (if yes → template for the multi-epoch stochastic
  layer; if no → THE fundamental problem, fix here first), and what transfers
  to the multi-epoch design. **(vi, user 2026-07-11) after z=0.4: repeat the
  single-epoch fit at the higher-z snapshots (z=0.7/1.0/1.5/2.0)** — how do the
  fitted single-epoch models RELATE to the z=0.4 one (coefficients, scatter,
  feature importances vs z)? A smooth relation is itself a path to the
  multi-epoch model (connect independently-fitted epochs rather than force one
  consistent history). QA figure improvement (also user): split qa_mass_* into
  groups; stack each quantity's truth-vs-prediction and residual panels
  vertically with a shared x-axis; per-quantity x-ranges.
  Decision: the emulator must serve the ENTIRE mass range; full sample measured
  feasible (per-galaxy fits 2 s/gal → 1.6 h; universal fit ~2 h; 10-fold CV ~1.7 h,
  single core; 100% of 2545 have valid 5-epoch CoGs + real MAHs). Steps, in order:
  **(1) foundation** — full-sample loader + cached npz (both MAH configs, CoGs,
  halo props; validate small first), then the per-galaxy 7-param θ atlas for all
  2545; **(2) mass structure of θ** — mass-binned universal θ vs global vs
  logMh-slope conditioning (10-fold CV); θ-anatomy (free ONE component per galaxy:
  which direction carries the individuality, constant with mass?); condition that
  component on halo props with real power; **(3) scoreboard vs mass** — exp31
  rerun at n=2545 in mass bins (+ epoch-matched-features regression to settle the
  MAH-decay question); does the plane-fidelity win and the DiffMAH-config choice
  hold down-mass?; **(4) stochastic layer** — fit the distribution of per-galaxy
  θ deviations, emulator = mean + correlated scatter, judged by tier-2b plane
  fidelity + exp07 CRPS/calibration; **(5) graduate** to the library with the
  error budget vs mass and epoch. Protocol: n=45 + a stratified ~100 as dev
  subsamples; long runs in background; DiffMAH input primary unless step 3 objects.
- [ ] **(superseded by exp32 — the massive-end-only follow-up) phase 4 follow-up —
  close or accept the +20-point individuality gap.**
  Options, in order of information gained per cost: (a) **n=200 re-test of the
  conditioning step** (4e detected c200c structure at n=200; n=45 may simply lack
  power — requires fitting per-galaxy θ for the larger sample first); (b) **θ-residual
  anatomy**: which θ directions carry the individuality (fit per-galaxy θ warm-started
  from the universal set with the other 6 frozen, one dimension at a time) — tells
  whether the gap is one interpretable knob (e.g. width normalization = size) or
  irreducibly multi-dimensional; (c) **accept ~30% shape / few-% integrated masses**
  and graduate the DiffMAH-input universal-θ emulator as the profile-growth module.
- [ ] **(superseded by exp30 gate — kept for the record) the multi-epoch ceiling is
  unreachable by any consistent additive Gaussian history — HONEST numbers.** Practical floor is now ~10% profile
  max|rel| (loose-quad, real MAH, all radii), ceiling ~2%. Either (a) **accept ~10% and
  build the forward emulator** (halo-only → 5-epoch CoG); note cumulative-aperture and
  relative-outskirt masses are excellent (~0.01-0.02 dex), so the emulator is far better
  on integrated/size-relative quantities than the ~10% profile number implies. Or
  (b) **rethink the primitive** (non-additive / transport, stars migrate) to reach the
  ceiling. Or (c) **event-driven width** (real MAH now carries the merger events) —
  the one untried lever that could exploit the bursts. Recommend (a) for the product,
  (c) as the highest-upside modeling try.
- [ ] **(next) shared-kernel population CoG fit on the dip-free MAH** (redo exp25
  Phase 3 with the DiffMAH curve, g anchored near 0.55, centred Gaussian).
- [ ] **(next) width set by the accretion *event*** (merger mass-ratio /
  smooth-vs-clumpy), the only way to capture late-merger core rebuilding that a pure
  σ(t) time→radius rule cannot — exp25's flagged structural limitation, untouched by `p`.
- [ ] (optional follow-ons) the **density profile in Re units**; feed the
  predictive profile uncertainty to the forward model.
