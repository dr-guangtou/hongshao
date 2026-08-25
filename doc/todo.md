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
- [x] **(A, product path) exp37 — the multi-epoch statistical emulator
  COMPLETE and GRADUATED (2026-07-14, branch exp37-multi-epoch-emulator ->
  `hongshao/multi_epoch.py`).** Continuous-z coefficient interpolation
  (closure <=0.9 max|rel| points) + AR(1)-in-epoch latent (rho=0.622 measured,
  poly2 cores) + generative sampling. Three profile representations built and
  judged head-to-head (README evidence table); the BLOCK-PINNED product wins
  (draw planes 0.4-0.9x floor through z=1, monotone by construction,
  uncertainty-weighted deficit allocation) and is the graduated DEFAULT; the
  K=3 log-CoG product graduates alongside it (user decision: conservative
  pairing — best z=0.4 inner-region mean, +1.8% vs +3.5%); the log-density
  variant stays in the experiment as `--profile-mode dens`. n=2395 (2
  broken-progenitor galaxies masked); kpc CRPS 0.0784->0.1523 beats the
  linear exp33-vi ceiling at every epoch; growth-plane draws 1.0-1.3x floor.
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
- [x] **exp36 multi-epoch round COMPLETE (2026-07-14, branch
  exp36-multi-epoch; verdicts in the README).** (1) Joint 5-epoch fit:
  multi 2ch-fa (16 params) held-out pinned shape 17.7/17.1/16.3/16.0/14.7%
  (epoch avg 16.4 = the z04-only mark; exp35 marks 20.5/19.1) — the
  consistency tax shrinks to +1.3 points. (2) The differential-deposition
  test SURVIVES via fa (massive 0.40/0.14 vs data 0.37/0.11; prune alone
  strains, 0.44/0.16). (3) Bounds-stress: log_s0_ex rails at 3.5 and spends
  the freedom past the 500-kpc horizon (z<1 ex visibility 0.39 -> 0.06) —
  the box stays at 3.0; fa is the structural fix. (4) The low-mass outskirt
  overshoot FALLS (+0.117/+0.127 -> +0.029/+0.070 dex) — but only under
  the JOINT constraint (the z04-only fa fit cannot see it; lesson logged).
  (5) P4 peak drift halved (3.5 -> 4.1); 2ch-fa takes log_s0_ex off the
  rail for the first time. **(C) built and judged: the flat-spine ablation
  guard FAILS** — the dressing reaches the statistical wall from either
  spine (kernel 15.6-10.8% vs flat 15.2-11.2%, a tie), so the spine earns
  nothing on held-out accuracy; the two-model program verdict stands with
  the hybrid now tested (statistical = product, kernel = physics
  companion). Log-CoG residual draws reproduce exp37's high-z failure
  mode; the block-pinned product remains the generative layer.
- [x] **exp38 — rethink the deposition primitive: COMPLETE (2026-07-16,
  branch exp38-deposit-rethink; full record in the README).** The Gaussian
  rail explained by measurement (the added light has Sersic n ~ 2-3 wings
  at 15-47 kpc; a Gaussian fakes them only by scale inflation). **ADOPTED:
  the single-channel power-law-tail kernel (1ch-mof, 12 params)** — the
  first bound-free joint fit in the family, held-out shape 18.5-14.2% by
  epoch, the program's best differential-deposition pass (0.39/0.12 vs
  data 0.37/0.11), flat outskirt terciles. 2ch-exp (16p) recorded as the
  accuracy alternative (avg 16.3%, finite wide-channel optimum at 3.19 —
  the exp35/36 rail dissolved by the tail). Profile-family branch: the
  6-coefficient evolving-Re template conditioned on [DiffMAH, c200c]
  reaches 16.3-18.8% (matches the kernel at z<=1; one shape + two
  halo-predictable growth tracks); the 15-coefficient sigmoid family has
  the capacity (0.6% per epoch) but is not feature-reachable. Stage 3:
  the inner-mass deficit decomposed — mostly a population-sharing limit
  (-5 to -8% in ANY shared single-epoch fit; ~0% per-galaxy) plus a
  ~3-4-point multi-epoch transport tax (q = 0 fits z=0.4 alone; the
  joint fit needs q ~ 0.9, which drains the core); the core-channel fix
  works for the inner masses but failed the physics tests — parked with
  the post-mortem (the inherited power-law tail leaks 8% of core mass
  beyond 30 kpc; see the PARKED entry below).
- [x] **exp39 — the core channel revisited: all three leads MEASURED
  (2026-07-17, branch exp39-core-revisit, full n=2397).** (1) The
  core-form shootout (mof/gauss/exp/ser2/ser3/ser4, all at a common
  half-mass-radius parameterization) FALSIFIED the tail-leakage
  mechanism: every form fits to the same loss (0.2540-0.2543) and every
  form — including the zero-leakage Gaussian — fails the differential
  test identically (0.53-0.56/0.18-0.19 vs data 0.37/0.11). The
  outer-kernel re-balancing is THE breaker; no core form rescues the
  free-kernel fit. The decisive follow-up: FREEZE the kernel at the
  adopted stage-2 theta and fit only the 3 core parameters — a small
  zero-wing Gaussian core (median f_core ~0.02-0.03, strongly
  halo-dependent) is then a STRICT improvement on the adopted kernel:
  held-out shape avg 16.1-16.5% vs 16.6%, held-out M(<5) -11 -> -4.4%
  and M(<10) -5.4 -> -1.0% at z=0.4, physics untouched by construction
  (the kernel's 12 parameters unchanged), nothing at a bound. The z=2
  inner deficit (-6.2%) stays — deposits born too wide at high z is an
  efficiency-side problem, not a retention one. (2) z=2-halo-mass
  conditioning helps exactly where the core is usable: frozen point
  0.2737 -> 0.2719 with better inner bias at both epochs (held-out avg
  16.1%); free point 0.2546 vs 0.2540 + a mu rail — no gain. (3)
  Per-galaxy core diversity is large (f_core 0.00-0.20 at 16-84 pct,
  1.45 dex logit scatter; freeing it closes the median M(<5) to +2.1%)
  but only weakly feature-predictable (best |Spearman rho| = 0.31,
  logms) — the population-sharing limit is per-object information;
  treat it as a statistical scatter layer in the emulator draws, not
  as more conditioning.
- [x] **(user decision 2026-07-17) the core channel stays PARKED;
  nothing graduates from exp39.** The frozen-kernel + gauss-core point
  was rejected: pinning the kernel is an ad-hoc restriction with no
  physical inspiration — the same structure fit jointly breaks the
  physics, so the "improvement" exists only under the artificial
  freeze. The durable exp39 outputs are mechanism knowledge: (a) the
  physics break is kernel re-balancing, NOT core-tail leakage
  (falsified by the six-form shootout); (b) z2-halo conditioning is
  the better core regressor where a core is usable; (c) the remaining
  inner deficit is per-object information (1.45 dex f_core diversity,
  |rho| <= 0.31) — only a statistical scatter layer could represent
  it. The adopted kernel remains exp38 1ch-mof unchanged; inner masses
  remain the statistical emulator's product.
- [x] **(exp39 final task, user-approved 2026-07-17) the retention x
  inner-aware cell — the exp38 factorial hole, now closed with a clean
  negative.** The retention floor (mass-conserving redistribution
  within the kernel's own deposits, targeting the measured clock-drain
  cause) under the inner-aware objective: f_ret = 0.564, no bound,
  best held-out shape in the program (avg 14.8%; parked core 15.0) and
  the only model that also fixes z=2 (M(<5) -2.3% held-out) — yet the
  SAME differential failure (0.53/0.17 vs data 0.37/0.11; outskirt T1
  +0.065/+0.099). The boundness-conditioned floor (rc0/R200 from the
  measured mass history) is a null (b = 0.10) that rails mu. With
  {6 core forms + 2 retention modes} x inner-aware ALL failing in the
  same band, the re-balancing is OBJECTIVE-driven: the inner-mass /
  outskirt-physics tension is structural to the kernel family, not an
  allocation artifact. Any future attack means a different kernel
  family or consistency constraint, not another inner-mass mechanism.
- [x] **exp40 — the two scope tests (user plan 2026-07-18) COMPLETE;
  verdicts in the exp40 README.** (1) LATE-START fits: the transport
  tax lives in the dissipative era — M(<5) at z=0.4 shrinks -11.7%
  (z<=2 fit) -> -9.5% (z<=1.5, held-out) -> -8.2% (z<=1.0, held-out)
  toward the -7.6% population-sharing wall; the two late starts agree
  on everything; physics IMPROVES (overshoot halved, differential
  passes, and the late fits predict the unfitted z2.0->1.5 pair
  better than the z=2-anchored fit trained on it); held-out accuracy
  unchanged on fitted epochs; the shared g=4.0 rail is benign
  (stress: interior 4.37 at 0.1% gain, physics unchanged — the SAME
  railed value that marked the pathological 5-epoch basin; a rail's
  meaning is scope-dependent). (2) The RE-AIMED FIDUCIAL (inner-aware
  objective on the 12-param kernel, exp38 capacity theta, now fully
  judged): best-tier held-out accuracy (14.9% avg) and the deficit
  halved (M(<5) -6.4%) — but differential FAILS (0.48/0.15) with mu
  railed and the overshoot back: the exp39 objective-driven
  conclusion confirmed with zero added components; NOT adoptable.
- [x] **(ADOPTED, user 2026-07-18) z<=1.5 is the kernel's OFFICIAL fit
  scope; the z<=2.0 five-epoch fit stays available as the comparison
  option.** Rationale: same held-out accuracy on fitted epochs, better
  physics (overshoot T1 +0.025/+0.028; the extrapolated z2.0->1.5
  differential pair at 0.21-0.23 vs data 0.23), -1.5 points of inner
  deficit (held-out M(<5) -9.5% vs ~-11%), z=2 kept as an honest
  extrapolation diagnostic (+1.2 shape points). The physical reading:
  z>=1.5 mergers are dissipative and outside the additive collisionless
  model's domain — the fit scope IS the physical statement. z<=1.0 is
  too short a lever arm (z>=1.5 extrapolation degrades: shape
  17.9/17.7, M(<5) -14/-16). Thetas: OFFICIAL =
  exp40 `outputs/latestart.npz[theta_z15]` (params [2.74, 4.00, 0.91,
  1.52, 0.24, 1.38]; the g=4.0 rail is stress-verified benign);
  comparison = exp38 `outputs/stage2_multiepoch.npz[theta_1ch-mof]`.
- [x] **exp41 — the stochastic layer (exp32 step 4) COMPLETE and
  ADOPTED (user 2026-07-18, branch exp41-stochastic-layer).** Stage 0:
  the kernel's per-galaxy individuality is ONE latent axis (all six
  single-component deltas rank-correlate at |rho| >= 0.92), it IS the
  exp39 core-diversity axis (|rho| = 0.78-0.87 vs per-galaxy f_core),
  and it is feature-orthogonal (|rho| <= 0.20) — pure per-object
  information; a real second axis exists for ~a fifth of galaxies but
  adds nothing on the planes and strains the overshoot. Stage 1: the
  sig-axis deviations are heavy-tailed (robust sigma 0.08, t-dof 4-9).
  Stage 2 (held-out, symmetrized split): the 1-D empirical layer takes
  the kpc planes from the kernel's 1.6-2.1x split-half floor to
  1.0-1.4x at z<=0.7 (target ~1 MET at z=0.4 — the program's first
  kernel-based drawn population at the floor), the Re plane scored
  SELF-CONSISTENTLY is 1.0-1.4x at every epoch (the paired-aperture
  "failure" was a metric artifact — resolves the exp33 Re-coordinate
  puzzle; lesson recorded), physics in the adopted band, mean model
  untouched. The Re-preserving refinement was built, measured, and
  REJECTED (it trades the kpc gains for nothing; its failure exposed
  that the layer's diversity is size diversity — the exp38
  self-similarity from the model side). Known limits (documented, the
  mean model's): kpc planes at z>=1.5 stay at 2.0-2.8x (ridge error).
  ADOPTED operating point: official kernel + empirical d_sig resample
  (exp41 `outputs/stage1_dist.npz`); QA figure set in exp41 figures/.
- [x] **tech notes — journal-style documentation of the emulators
  (user direction 2026-07-18; branch tech-note-emulators).** New
  `doc/tech_note/`: 01 = the statistical emulator (heteroscedastic
  conditional Gaussian core, PCA profile modes, exp37 multi-epoch
  AR(1) block-pinned product), 02 = the adopted 1ch-mof transport
  kernel at the official z<=1.5 scope + the exp41 stochastic layer,
  03 = the 2ch-exp two-channel alternative and why the heavy tail
  superseded the split. Markdown + LaTeX math; 12 pedagogical figures
  + 2 GIF animations (deposition-then-migration assembly, PCA-mode
  sweep) generated by `doc/tech_note/scripts/` (self-contained toy
  kernel mirroring exp38 `basis_mof`, self-checked; usetex-safe
  labels, Okabe-Ito epoch colors). Audience: knows the motivation,
  not the experiments.
- [ ] **THE OPTION MENU for improving 1ch/2ch (reviewed 2026-07-21,
  user asked for options; recorded for later reference).** Where the
  model stands: adopted 1ch-mof z<=1.5 (held-out shape
  18.2/17.4/16.6/16.4, z=2.0* 15.4; M(<5) -9.5%; differential
  0.40/0.13 vs data 0.37/0.11; no bound but the benign g rail). THREE
  MEASURED WALLS bound any further work: (i) the population-sharing
  wall (M(<5) ~ -8% in every shared-theta fit, even single-epoch —
  only per-object freedom crosses it); (ii) the accuracy wall (15.6%
  statistical vs kernel 18.2 at z=0.4 — but exp36 showed that gap is
  feature-reachable from a FLAT spine, so it is not the kernel's to
  win); (iii) the inner-mass vs differential-physics tension (exp39,
  structural to the family). THREE UNTESTED INHERITANCES: the
  conditioning rows (chosen by the exp36 TWO-channel ablation,
  inherited wholesale by 1ch-mof), the lognormal efficiency window
  (exp36's proposal (D) — a monotone 3-4-knot spline — was never
  built), and the single global tail index gamma.
  - **(A) "Calibrate low, predict high" — per-object low-z anchoring.**
    Fit each galaxy's individuality parameter (the exp41 axis, ONE
    number) against its OBSERVED low-z profile, freeze everything else,
    predict z >= 1.0. Neither fully shared (hits the sharing wall) nor
    mean-zero scatter (fixes distributions, not per-object accuracy).
    Grounding: exp41 (one freed component buys 18.5-22.3% median
    per-galaxy improvement, feature-orthogonal = genuinely per-object)
    + exp43 (the transport clock locks in from two epochs). Untested
    question: does per-galaxy information learned at low z TRANSFER
    upward in z? Cheap (exp41 stage-0 anatomy ran the full sample in
    0.6 min). -> STARTED as exp44.
  - **(B) Re-derive the conditioning structure for 1ch-mof.** Only
    log_rc and sig carry halo slopes, on rows selected for 2ch-exp;
    exp42's lesson is that such verdicts are conditional on the
    deposit form. Test slopes on q and gamma, drop rows that do not
    earn their slot, try epoch-matched features. Cheap; decides
    whether the current twelve parameters are the right twelve.
  - **(C) Free the efficiency window** (exp36's parked (D), never
    built). Two numbers currently encode the whole efficiency history.
    Load-bearing evidence: exp42's real-MAH refit DOUBLED sig
    (0.24 -> 0.52) and moved the peak; mild scope drift along the
    exp43 ladder. Monotone 3-4-knot spline or a skewed form. Medium
    cost; bounds-stress any new freedom before reading it.
  - **(D) Mass-dependent deposit shape.** gamma is one global number,
    yet exp38 stage 0.2 measured the added-light kernel half-mass
    radius growing 15 -> 47 kpc with mass AND cosmic time and Sersic
    index 2.1-3.1 across mass/epoch cells. Honest counter-evidence:
    stage-1 per-epoch fits found gamma ~ 1.3 FLAT IN Z — so MASS, not
    redshift, is the better-motivated axis. Small (1-3 params);
    targets the residual outskirt structure.
  - **(E) JAX port — the enabler** (tech-note-2 section 7 documents
    feasibility). Derivative-free Nelder-Mead is the binding
    constraint on model richness (minutes per start, 40-60 min per
    CV); gradients make (A)-(D) jointly explorable and unlock HMC
    posteriors over the recurring degeneracies (q x g, gamma x
    log_rc). Not an improvement itself — it changes what is reachable.
  - Maintenance items still open from exp43: CV the z07/z10 rungs if a
    production fit-scope decision is made on them; two-start refit of
    the z04 rungs if any z04 conclusion becomes load-bearing.
- [x] **exp44 — option (A), per-object low-z anchoring: COMPLETE
  (2026-07-21/08-12, branch exp44-lowz-anchoring), three stages.**
  - **Stage 1 (anchoring):** fitting ONE per-galaxy coordinate against
    observed low-z epochs and predicting the rest. The ORACLE (same
    coordinate fitted on all five epochs) is worth 3.0-7.6 shape points
    and halves the inner deficit — the sharing wall IS a per-object
    information limit. Anchoring realizes that at its own epoch (z=0.4
    shape 18.2 -> 8.4%) and recovers ~1/3 of it at the ADJACENT
    unanchored epoch (shuffle margin +6 points), decaying to ~0 beyond
    (+0.3). The apparent inner-mass gain at distant epochs is an
    asymmetric-pool MEAN SHIFT, not transfer — the shuffle control
    caught it. Aperture-only anchoring (the observationally honest arm)
    works at ~2/3 strength; the axis choice is irrelevant.
  - **Stage 2 (decorrelation):** the axis fitted INDEPENDENTLY per epoch
    decorrelates as AR(1), rho = 0.56 (sig) / 0.66 (log_rc); split-half
    reliability is 0.96-1.00 so disattenuation is a no-op and the decay
    is REAL, not noisy estimation — the stage-1 reach limit is
    fundamental, richer low-z data cannot extend it. KEYSTONE: exp37's
    statistical emulator measures rho = 0.62 with the identical
    estimator despite having NO consistency constraint, so rho ~ 0.6 is
    a property of the GALAXIES, not of the kernel's one-history
    structure.
  - **Stage 3 (the AR(1) layer, built as an OPTION; default
    unchanged):** Gaussian AR(1) rank-mapped onto the measured
    empirical pool, so every epoch's marginal is identical to the
    adopted layer's. Every qa tier-2b plane is SINGLE-EPOCH and
    therefore structurally blind to rho (identical scores) — which
    FALSIFIED the stage-2 guess that over-persistence contributes to
    the high-z plane stall. The cross-epoch size-growth plane does see
    it: the adopted layer holds drawn galaxies at size coherence +0.97
    vs a TRUE +0.33; ar1 tracks the truth closely (0.74/0.63/0.54/0.48
    vs 0.82/0.68/0.50/0.33). Cost: the flagship differential pair
    degrades 0.41 -> 0.29 (data 0.37). VERDICT: keep `persistent`
    fiducial (single-epoch applications are provably rho-blind); ship
    `ar1` for lightcone / progenitor-linked mocks.
  - **GRADUATED: `hongshao.qa` tier 2c** (`growth_planes`, reported by
    `evaluate`, figure `qa_growth_<name>`) — the standard set contained
    NO cross-epoch statistic, which is how a 0.97-vs-0.33 defect
    survived exp41's adoption. First run on the DETERMINISTIC exp43
    kernel: size coherence +1.00/+1.00/+1.00/+0.96 vs truth
    +0.82/+0.68/+0.50/+0.33 — **the mean model, not the scatter layer,
    is the dominant source of over-coherence.** New, previously
    unmeasurable defect; a natural selection metric for options B/C/D.
  - Closing note: the ~15% negative epoch-to-epoch growth rate is a
    DATA property (truth 13.78%), faithfully inherited, not an artifact.
- [x] **exp49 — is the railed `g` measurable? CLOSED 2026-08-20 with unfinished
  business (branch exp49-identifiability).** Answered for the production
  objective: **`g` = 4.35222 +/- 0.00320**, 4/5 starts agreeing to 1.17e-07, no
  parameter within 2% of a bound. The rail was real (nine of ten cells, five
  starts x two box formulations), but `g` is one coordinate of a degeneracy in
  which all six BASE parameters correlate at |rho| = 0.834-0.995 (effective rank
  **7 of 12**), it is worth **0.064%** of the loss, and **raising it makes the
  compact-galaxy central deficit WORSE** (-35.7% at g=3.0 to -40.5% at g=5.5)
  while the loss minimizes in between. **Do not widen the box in production.**
  A 300 kpc deposit-radius ceiling costs 1.5e-5 refitted — essentially free.
  - **INVALID, do not quote bare**: the density-log widened fit is PINNED at the
    `log_rc` bound, so its `g` = 5.139 is a LOWER BOUND (the qualitative finding
    holds: the objective shifts `g` by >= 0.8). Production profile points
    `g` = 5.25 and 5.50 are contaminated by the same wall.
  - **UNFINISHED, deferred and resumable** (stores cached and keyed): the
    density-log widened fit with `log_rc` widened to ~4.5 and its profile; the
    +/-0.30 refinement grids; extending the grid below `g` = 3.0 to settle the
    two-basin question; and **stage 3 entirely** (interior Hessian at the widened
    solution, weak-subspace principal angles, sandwich covariance, bootstrap).
  - **Why closed**: exp52 showed the deposition mechanism is misspecified.
    Identifiability results for a misspecified model may not survive fixing it,
    so the remaining stages of
    `doc/plans/2026-08-16-kernel-identifiability.md` are on hold, not abandoned.
- [x] **exp53 — the DEPOSITION-ONLY boundary test: COMPLETE (2026-08-21,
  branch exp53-deposition-only).** Ran graded (7-point profiled `q` grid plus
  the free fit), efficiency window free, judged against the MEASURED
  added-light kernel, Sersic reinstated behind an R50 reparameterization.
  14 cells x 3 starts, n=2397, full sample. Plan:
  `doc/plans/2026-08-20-deposition-only-test.md`.
  - **THE PREDICTION FAILS UNDER PROFILING.** exp52 predicted that removing
    transport would improve central masses. On the SLICE it does, hugely
    (compact `M(<5)` bias -43.6% -> +13.6%). Under profiling the compact
    deficit is **-39% to -47% at EVERY `q`** — removing transport entirely
    buys 4 points out of 43. **The compact-galaxy central deficit is not
    caused by transport**; exp52's mechanism claim survives in its narrow form
    (late outskirt growth IS redistribution) and fails in its causal form. The
    exp47/exp48 defect is unexplained again. Textbook slice-vs-profile, on the
    central question.
  - **ACTIONABLE: trim `q` from 0.91 to ~0.6-0.8, do not remove it.** Judged
    against the measured kernel, `q`=0.8 costs **0.27%** of the loss and
    improves the kernel distance 0.081 -> 0.068, the `M(<10)` growth ratio
    0.722 -> **0.991**, sizes +0.026 -> +0.012 and the overshoot gate
    +0.023 -> +0.006; `q`=0.6 costs 2.2% and matches the differential gate
    exactly (0.37 vs 0.37) with the best sizes (-0.009). The loss alone would
    never point there: it minimizes at 0.91.
  - **The profiled loss curve is a clean single minimum at `q` ~ 0.9**
    (0.160001 at `q`=0 rising from 0.153577), so under the production
    objective the data genuinely wants transport. Removing it costs 4.2%.
  - **Two branches, switching at `q` ~ 0.5**: below, a wide efficiency window
    (sig up to 2.68), small deposits (log R50 1.73), weak growth (g 1.7);
    above, a narrow early window (sig ~0.22), huge deposits (log R50 3.3-3.5),
    steep growth (g 4.3-4.7). They trade epochs — the low branch is worse at
    z=0.4 (plane 2.7 vs 1.9-2.1) and better at z=2.0 (4.1 vs 4.4-4.9), which
    is exp40's dissipative-era finding from a new direction.
  - **Sersic: a CONFIRMATION of exp48, not a new result.** It fits at
    **n = 2.29** unrailed, inside the measured kernel's 2.1-3.1, and still
    loses to the Moffat on both loss (0.1567 vs 0.1536) and kernel (0.118 vs
    0.081). **The 2026-08-20 handover's framing was wrong**: it said Sersic
    "was eliminated by a numerical artifact" for exp53 to fix, but **exp48
    step C had already built `sersic_r50` and already re-run it in the full
    multi-epoch kernel**, where it came LAST on the judge (2.44 vs Moffat
    1.95). exp53 reproduces that under a different objective, which
    strengthens exp48. The new part is narrow: exp48 left the Moffat in raw
    `rc`, so exp53 is the first run where `log_R50` means the same physical
    quantity for every family. (The degeneracy is now quantified though: at
    fixed raw scale a=1, n from 0.5 -> 8 moves R50 by **1.2e6x**.)
  - **SCOPE LIMIT: one objective only.** Every exp53 fit used the PRODUCTION
    objective (CoG, fractional residuals), NOT log-density. exp48 measured
    that the objective REVERSES the profile ranking (`sersic_r50` worst under
    production, tied-best under exp48's density-log winner), so exp53's family
    ordering is conditional. **`gompertz_log` was not tested and should have
    been**: exp48 ranked it 2nd of six on the judge with the BEST compact
    central mass of any candidate (-27.7% vs Moffat -31.4%) — the one family
    that has ever moved the number exp53 found stuck at -39% to -47%.
  - **Transport SUBSTITUTES for a heavy tail.** Removing it costs the
    heavy-tailed families 3-4% and the light-tailed ones ~11%. gauss/expo
    *need* transport because they have no other way to reach the outskirts.
    The falsification control behaves: `gauss-q0` is worst on the kernel by 4x
    (0.339), with kernel n = 1.7 against a measured 2.7.
  - **The deposit scale is UNIDENTIFIED for every family except the Moffat.**
    `rail_check.py`: sersic/expo/gauss escape `log_R50` = 3.5 and immediately
    rail at 5.0; the Moffat is interior at 3.266 and its one railed cell
    escapes to a finite 3.626. Widened, the three converge to
    0.15613/0.15616/0.15621 — indistinguishable — because "every deposit at
    the ceiling" is the same model whatever shape is hung on it.
    **A mechanism for exp48's "the profile does not matter"**, and it means a
    family shootout is only meaningful with the deposit scale pinned. The `mu`
    rail on the deposition-only cells is BENIGN (0.02-0.05%), so design
    constraint 2 held.
  - **Reach is a FAMILY property, not only a transport one**: the Moffat puts
    4.9-6.8% of deposited mass beyond 500 kpc where the normalization discards
    it; every lighter-tailed family puts 0.5-1.4%. In tension with the Moffat
    also matching the measured kernel best. **A lower, physically motivated
    deposit ceiling is the obvious next lever** (exp49 measured 300 kpc at
    1.5e-5).
  - **CAVEAT: exp53's fits are IN-SAMPLE** (10-fold CV over 14 cells x 3
    starts was unaffordable). Parameter counts differ by <= 2 of ~11 against
    ~220k residuals, so optimism cannot reorder families separated by more
    than a fraction of a per cent — but `q`=0.8 vs `q` free (0.27%) is inside
    that margin ON THE LOSS. It is not close on the judged observables.
  - **The generalized-sigmoid family, run 2026-08-21.** `richards` fits
    **nu = 0.345** (nu->0 = gompertz_log, nu=1 = loglogistic), so the data want
    an asymmetric sigmoid that is neither exactly; it beats both families it
    nests. `gompertz_log-q0` posts the best added-light kernel of all eleven
    candidates (0.072 vs the fiducial's 0.081) and beats the fiducial on 10 of
    13 metrics at EQUAL parameter count to `moffat-q0`. Confirms exp48's
    2nd-of-six ranking across a different objective and harness. All `mu` rails
    benign. **The sigmoid loss surface is much rougher** — start-to-start
    spreads 1.7e-3 to 2.5e-3 against 1e-8 to 1e-10 for the other families, with
    one start in three landing in a worse basin, so multi-start is mandatory
    for this family.
  - **STANDING VERDICT (user, 2026-08-21), overriding the loss and kernel
    tables**: no deposition-only model is significantly better than the
    fiducial. The sigmoid family is not bad but still has issues at the centre
    and at high redshift; **it cannot solve the problem alone but is worth
    carrying into the next phase.**
  - **The defect that decides it, quantified**: `M(<148)` is under-estimated by
    3% in the low HALO-mass tercile and over-estimated by 6% in the high one, a
    **9-point tilt where the fiducial is FLAT at +3%**. Not a mass-budget
    error — the `M[50,148]` annulus tilts the OTHER way (+17% low, -9% high),
    so the deposition-only models carry a **RADIAL DISTRIBUTION error that
    flips sign with halo mass**: too extended in low-mass haloes, too
    concentrated in high-mass ones. The fiducial's transport term is what
    supplies that halo-mass-dependent concentration, which is the sharpest
    statement exp53 produced about what transport is actually FOR.
  - Untested: the `gausswing` family; a lower deposit ceiling; CV; the sigmoid
    family with `q` free.
- [x] **exp52 — the kernel grows galaxies by REDISTRIBUTION, not accretion:
  DIAGNOSIS COMPLETE (2026-08-20, branch exp49-identifiability).** Opened after
  the user challenged an offhand claim that late outskirt growth comes from
  migration. It does, and that is a structural problem. Full sample n=2397.
  - **Transport share of the model's own M\*[50,100] growth**: 18.0%
    (z=2.0-1.5), 38.4% (1.5-1.0), 76.9% (1.0-0.7), **96.5% (0.7-0.4)**; 87.5%
    over z=1.0-0.4. Crossover at z~1.2, exactly where minor-merger accretion
    should take over.
  - **The kernel's own deposition supplies ~11% of the mass growth** from z=2 to
    z=0.4 (x1.115 vs measured x2.645); the other 89% is the M(<500) pinning.
    After z~1.5 deposition is off and the un-normalized mass inside 500 kpc
    FALLS (x0.992).
  - **Explains the exp47/exp48 compact-galaxy central deficit**: with deposition
    off, filling an outskirt requires emptying the centre. z=1.0-0.4 model/truth
    growth M(<10) = **57.3%** against M[50,100] = 98.9%.
    **CORRECTED 2026-08-21 (found in exp53): M(<10) is 0.726, not 0.573.**
    `decompose.py` divided the model's APERTURE growth by the truth's ANNULUS
    growth (`np.interp(0.0, R, cog)` clamps to the CoG at 2 kpc, not zero).
    The finding survives at 27% rather than 43%; M[50,100] and every transport
    share are unaffected.
  - **RIGHT SOURCE, ~10x TOO MUCH REACH** (user's framing, and the actionable
    part). Adiabatic expansion from AGN feedback is well grounded but FLATTENS
    the inner profile; it does not build the stellar halo. Measured: 93.1% of
    the drained mass comes from 0-5 kpc (correct region) but 58.5% is delivered
    beyond 50 kpc, 29.4% beyond 148 kpc, 12.0% beyond 500 kpc where the
    normalization discards it. **Target the radial extent, not the existence, of
    the transport term.**
  - **NEXT, unrun**: TNG's own in-situ/ex-situ decomposition is the direct
    falsification test. NOT in the local drop — historical profiles carry
    isophote quantities only, and `map_tng100_hist_stellar.hdf5` is 852 MB of
    ZEROS (failed download, needs re-fetching). TNG's public Stellar Assembly
    catalogs give the per-subhalo split; the RADIAL split is a further step,
    availability unverified.
- [x] **the objective module — a REFACTOR, not an experiment: COMPLETE
  (2026-08-15, branch objective-module).** There was no loss/objective
  code in `hongshao/` at all; the production objective was six hardcoded
  lines in `exp38/stage2_multiepoch.py::gal_loss`, with a flexible copy
  in `exp48/objective.py` and a third partial copy in
  `exp48/shootout.py::score_cogs`. Now `hongshao/objective.py`: a frozen
  dataclass with six selectable axes (quantity cog/density, residual
  fractional/absolute/log, radial rms/mean-abs/worst-radius, weight
  uniform/per-dex/inner, epoch mean/weights, galaxy
  mean/median/worst-20%). `Objective()` with no arguments IS the
  production objective — verified against the previous implementation on
  all 2397 galaxies at every epoch to 4.4e-16, and against
  `exp48/shootout.py::score_cogs` across 182 configurations (every cell
  exp48 ran plus the exhaustive six-axis product) to 5.9e-16 relative.
  Adopted theta and adopted model UNCHANGED; the port costs no time
  (1.00x on a full pass). Design: `doc/plans/2026-08-15-objective-module-design.md`.
  - **NEW AND UNTESTED: the epoch axis.** Both previous versions took a
    plain arithmetic mean over the fitted epochs, hardcoded — an
    aggregation nobody had ever varied. It is now selectable via an
    explicit per-epoch weight vector, default unchanged. **This is the
    obvious untested lever on the one defect exp48 left open**: on the
    test of how fast total stellar mass declines with redshift, both the
    adopted model and exp48's winner decline too steeply where the truth
    is nearly flat between the first two epoch pairs (truth 0.372 ->
    0.361, models 0.403 -> 0.315). Note the differential gate checks only
    ONE epoch pair (z=0.7 -> 0.4), which is what let this hide.
- [x] **the optimizer question: `hongshao/fitting.py::minimize_loss`
  (2026-08-15, branch objective-module -> fitting-optimizers).** Seven
  optimizers benchmarked on the real 1ch-mof loss. **L-BFGS-B on
  finite-difference gradients reached 0.152764 in 1170 evaluations / 38 s;
  the plain Nelder-Mead in use exhausted 4000 evaluations at 0.159980 — 4.7%
  worse — in 131 s.** BFGS, trust-constr and `Nelder-Mead(adaptive=True)`
  agreed on 0.152764 to six decimals. Gradient methods apply because the
  loss is measurably smooth (none of the kernel's clip/max guards active at
  the solution; central differences stable from a step of 1e-2 to 1e-7).
  `minimize_loss` exposes lbfgsb (default) / bfgs / neldermead / powell,
  applies `adaptive=True` and `initial_simplex` automatically for
  Nelder-Mead, takes real bounds, and refuses both a finite-difference step
  coarser than 1e-4 and a start point outside the bounds. Nothing calls it;
  production fits and the adopted theta are untouched.
  - **OPEN, and it may matter for interpretation: is "the rail" real?**
    Every method that reached the minimum put `g` at ~3.54 and `log_rc` at
    ~2.47, both interior; every method that failed pinned one or the other
    at its bound (3.9999 / 2.9999). The bounded run with the penalty removed
    agrees — no parameter at a bound. But the FULL-sample refit still ends
    with `g` at 4.0, having started from the already-railed adopted theta and
    used the one-sided box penalty, whose derivative is exactly zero at the
    bound, so a gradient method can report convergence at that kink.
    **Decisive test: full sample, real bounds instead of the penalty, several
    displaced starts.** If `g` comes off the rail, the adopted fit's railed
    parameter is an optimizer artifact and its interpretation needs revisiting.
  - Also measured, and NOT acted on: re-running the adopted fit with L-BFGS-B
    moves the loss only 0.119% (0.153795 -> 0.153612, n=2397) but drops the
    gradient norm 39-fold (0.0196 -> 0.0005), and the six conditioning slopes
    move a lot in relative terms — the deposit scale's formation-time slope
    flips sign, +0.00016 -> -0.01570. The base kernel is well determined;
    those slopes are not. Nobody should read the adopted slope values without
    re-fitting and re-judging first.
  - **Deferred (user, 2026-08-15): a JAX prototype of the kernel.** Automatic
    differentiation would replace 13 evaluations per gradient with one
    backward pass and could vectorize the per-galaxy Python loop; DiffMAH is
    itself a JAX library, so the dependency has precedent. Speedup UNMEASURED
    — the honest first step is a single-galaxy prototype checked against the
    current output and timed. Treated as a new direction, not this refactor.
  - **Also new: `hongshao/fitting.py::initial_simplex`.** scipy's
    Nelder-Mead spans a parameter started at exactly 0 by only 0.00025,
    and any parameter by only 5% of its own value, so parameters at or
    near zero come back where they started and look measured. This
    produced a false null in exp47. Nothing calls it yet — every existing
    fit keeps scipy's default, so no published number moves — but any NEW
    fit nesting a component at zero must use it. See `doc/lessons.md`.
- [x] **exp46 stage 1 — "the high-z ridge" is the model failing on
  COMPACT galaxies, not on high redshift (2026-08-12, branch
  exp46-highz-ridge).** Opened after the 2026-08-12 discussion, which
  split the phenomenon in three — (A) the plane residual, an
  *under-dispersion* failure (truth scatter 0.170 -> 0.338 dex with z
  while the model's falls 0.077 -> 0.067 and its slope collapses
  +1.41 -> +1.07); (B) per-object accuracy, which does NOT degrade at
  high z; (C) the size story, an R50-only offset at z<=1.5 plus a
  separate outer over-extension at z=2 — and **falsified the fit-scope
  explanation for (A) twice from logs already on disk**: the z<=2.0 fit,
  with z=2 IN the loss, scores centered E/floor 4.8 at z=2, identical to
  the z<=1.5 fit extrapolating there (slope +1.07 both), and the exp37
  emulator fits every epoch separately and still breaks at exactly those
  epochs. exp40's dissipative-era finding is about the mean profile's
  INNER MASS and does not address (A).
  - **Design correction recorded**: measuring "how much of the high-z
    diversity is predictable from halo features" is impossible with this
    metric — `plane_energy` compares POPULATIONS and is blind to which
    halo produced which galaxy, so any feature-conditioned resampler
    scores at the floor by construction (now a self-check assertion).
    The question sharpens to **"is the spread REAL"**, since a model can
    always match a wide target by injecting scatter.
  - **Every "not real" candidate died**: stellar shot noise at z=2 is
    0.027 dex (95th pct 0.094) against 0.338 measured; the excess sits
    in EVERY inner-mass quartile (best quartile 0.288 at z=2 vs 0.163 at
    z=0.4), not just a tail; near-empty progenitors are 0.1%.
  - **The result, which none of the three pre-registered outcomes
    covered**: trimming to truth R50 >= 5 kpc is significant at
    **p = 0.00 at EVERY epoch** (24 random same-n trims as the control),
    including z=0.4 where it drops just 6.3% of galaxies and still moves
    2.2 -> 1.5. The apparent high-z localization is pure demographics —
    the R50 < 5 kpc fraction runs 6.3 / 14.0 / 27.4 / 60.7 / 85.2%.
    The failure is about SIZE, not MASS: the top inner-mass quartile is
    *worse* than random at z>=1.5 (p = 0.96-1.00), consistent with the
    truth's mass-size slope collapsing to +0.08 by z=2.
  - **One genuine artifact, confined to tier 2d**: the 13.0% of z=2
    galaxies with R50 below the 2.0 kpc CoG grid start (size by log-log
    extrapolation) carry 5.0 -> 3.5 of the mass-size plane (p = 0.04)
    and NOTHING of tier 2b (p = 0.21/0.17). ~1.5 of tier 2d's z=2 score
    should not be chased; none of tier 2b's 4.8 is an artifact.
  - **Metric hygiene**: removing ONE galaxy moved z=1.5 from 3.5 to 4.1
    (worse than all 24 random one-galaxy removals) — `plane_energy`
    standardizes by the truth's per-axis std and estimates its floor
    from 8 half-splits, so single objects move it. Report p, not deltas.
  - **Consequence for the option menu**: the efficiency-window
    hypothesis predicts a high-z-SPECIFIC defect and so cannot alone
    explain an equally significant z=0.4 cut. Leading unified
    hypothesis for stage 2: the compact-galaxy plane failure and the
    known inner-mass deficit are ONE defect (exp45 measured per-galaxy
    Spearman(dlog R50, M(<5) bias) = -0.89 to -0.97 at every epoch),
    which would partially rehabilitate exp40 — as a statement about
    compact galaxies, not about redshift.
  - Stage 2 (not started): (1) test that unification per galaxy;
    (2) re-run the exp45 sigma sweep under trimmed scoring; (3) score
    the z<=2.0 and z=2.0-only thetas on tier 2d — still the one open
    fit-scope question, since tier 2d postdates exp38.
- [x] **exp45 — the cross-epoch coherence scoreboard + two new STANDARD
  QA tiers: COMPLETE (2026-08-12, branch exp45-coherence-scoreboard).**
  Retro-scored 13 recorded thetas + both layer modes + the exp37
  emulator's saved draws on tier 2c; no new fitting.
  - **The over-coherence defect is UNIVERSAL to the kernel family.**
    Eleven deterministic variants spanning both channel counts and all
    five fit scopes are indistinguishable (excess +0.49 to +0.66, energy
    8.7-12.3x floor). Two partial exceptions land at the SAME excess
    +0.45 (the inner-aware objective, 1ch only; the real-MAH input),
    both previously rejected on other grounds. exp44's `ar1` beats every
    structural choice (+0.15, 4.7x). **The exp37 statistical emulator
    does not have the defect at all** (-0.16, 2.1x): it carries an AR(1)
    latent by design, where a kernel galaxy is one theta plus one
    history and so is near-deterministic. -> the fix cannot come from
    re-parameterizing the kernel; the exp44 ar1 trade is a LIVE
    decision, not settled.
  - **Follow-up 1: the lever is the EFFICIENCY WINDOW WIDTH.** Sweeping
    one coordinate at a time, only `sig` moves coherence (+0.96 ->
    +0.67 over 0.24 -> 1.20); the migration exponent q is COMPLETELY
    INERT, as is log_rc — coherence is set by WHEN stars arrive, not by
    how they are transported. But it saturates at ~+0.67 vs truth
    +0.33, so menu option (C) is now targeted yet cannot close the gap.
  - **Follow-up 2: the emulator does NOT take over the physics** — its
    draws undershoot the differential test at every pair (0.32/0.09 vs
    data 0.37/0.11 at the flagship; 0.15/0.04 vs 0.23/0.06 at the
    highest, where the kernel is 0.40/0.13 and 0.21/0.06). The
    two-model division of labour survives with an independent 2nd leg.
  - **GRADUATED: `qa` tier 2c** (`growth_planes`) and **tier 2d**
    (`size_planes`, R50/R80/R90 via the new `enclosed_radius`), both in
    `evaluate()` with figures `qa_growth_*` / `qa_size_*`. Tech-note
    README documents them as standard.
  - **Follow-up 4 (user hypothesis, CONFIRMED): the mass-size offset IS
    the inner-mass deficit re-expressed.** Per-galaxy Spearman between
    dlog R50 and the M(<5) bias is -0.89 to -0.97 at EVERY epoch. At
    z<=1.5 R50 is biased (+0.024 to +0.084) while R80/R90 sit near zero
    — and that is conservative, since R90 AMPLIFIES outer errors (the
    CoG is flat there; documented + self-checked). **At z=2.0 the
    ordering FLIPS** (R80/R90 +0.116/+0.112 vs R50 +0.087): a genuine,
    SEPARATE outer over-extension. So the high-z size story is TWO
    effects, which a half-mass radius alone conflates.
  - Standing gap by the user's own criterion: the adopted product does
    NOT pass the mass-size plane (E/floor 3.1 at best, 7.8 at z=2, vs
    1.0-1.1 on the mass planes); the relation is too FLAT at every epoch
    (+0.34 vs +0.50 at z=0.4) and the compact tail is missing.
- [ ] **PINNED (user, 2026-07-21) — validate the stochastic layer as a
  PREDICTIVE DISTRIBUTION, not just a population generator.** Raised
  while discussing "predict a distribution, not a profile": that IS the
  adopted exp41 design (per-galaxy best-fit deviations, empirical +
  heavy-tailed, resampled at draw time; planes at 1.0-1.1x the
  split-half floor at z=0.4). But three things it does NOT do, each
  verified against the code, are unbuilt:
  - **(a) It has never been scored as a calibrated predictive
    distribution.** exp41 judged population fidelity only (2-D plane
    energy vs floor); the per-object question — does the truth land
    inside the predicted 68% interval 68% of the time — was never
    asked. CRPS/PIT/interval-coverage machinery has only ever been run
    for exp37 (the statistical emulator), never for any kernel
    experiment. REQUIRED if P(profile | MAH) is to be used as a
    LIKELIHOOD in forward modeling. Highest value; machinery already
    exists in `hongshao.metrics`.
  - **(b) The deviation spread is halo-INDEPENDENT.** The base theta is
    conditioned, so the distribution's LOCATION moves with the halo,
    but one global pool is drawn for a 1e13 and a 1e15 halo. The
    decision to skip feature-dependent width rested on exp41 stage 0(d),
    which tested whether the deviation VALUE correlates with features
    (|Spearman rho| <= 0.20) — a test of the MEAN, not the SPREAD. A
    quantity can be mean-uncorrelated with X while its VARIANCE depends
    on X (exp14 precedent: heteroscedasticity barely moved marginal
    CRPS but bought +0.24 nats and a ~10x smaller conditional-coverage
    gap). CAUTION if built: the pool comes from per-galaxy BEST FITS,
    so its width is an upper bound on the intrinsic scatter —
    differential fitting noise across halo mass could manufacture a
    spurious width trend; needs a noise control / deconvolution, not a
    raw binned spread.
  - **(c) The deviations are epoch-INDEPENDENT** — one delta per galaxy
    at every epoch — while exp44 measured the individuality to be
    epoch-local. -> BEING STARTED NOW as exp44 stage 2 (the
    cross-epoch decorrelation measurement).
  - Honest scope limit for the whole direction: a distribution cannot
    repair a biased mean. exp41's own diagnosis of the z >= 1.5 plane
    failure (2.0-2.8x floor) is that the MEAN model's ridge error
    dominates there — a scatter layer cannot fix a ridge. Options
    (B)/(C)/(D) are the levers for that.
- [x] **exp43 — the extrapolation ladder COMPLETE (2026-07-19/21,
  branch exp43-extrapolation-ladder).** Both kernels x five low-z fit
  scopes, everything unfitted judged (the observational
  forward-modeling framing). TWO EPOCHS is the minimum viable scope:
  z04-only fails both ways (1ch collapses honestly, q unidentifiable
  -> 44%/-47% at z=2; 2ch posts the table's best extrapolated shape,
  12.7%*, with SIGN-FLIPPED inner masses and a broken 0.54/0.20
  differential — flexibility extrapolates the metric, not the
  physics). The second epoch locks the transport clock (q 0->0.66,
  then 0.79/0.91 monotone) and the 1ch z07 fit predicts the unfitted
  high-z differential pairs at data precision (z2.0->1.5* 0.23/0.06
  EXACT — better than the z20 fit trained there); the third epoch buys
  most of the remaining shape (z=2.0* 17.2 vs 14.1). 2ch-exp's
  z2.0->1.5 differential overshoots (~0.32 vs 0.23) at EVERY scope —
  its high-z physics does not extrapolate. Adds an extrapolation
  argument to the 1ch adoption. Demonstration figures + standard QA
  for the z07 rung in exp43 figures/. Follow-up if a scope decision is
  made on z07/z10: CV those rungs.
- [x] **exp42 — the real-MAH head-to-head for the adopted kernel
  COMPLETE (2026-07-19, branch exp42-real-mah-kernel).** The same
  12-param 1ch-mof + z<=1.5 protocol with config="real": the
  Gaussian-era 3-5-point real-MAH penalty does NOT survive the Moffat
  tail — held-out shape ties at z<=1.0 and wins at z>=1.5
  (18.1/17.1/16.7/15.5, z=2.0* 13.6 vs DiffMAH 15.4), from a bound-free
  basin (log_rc 1.56, g 2.43 interior — the g=4.0 rail was a property
  of the smooth basis; window peak z=4.9, sig 0.52). Physics mixed: the
  z0.7->0.4 differential is an EXACT 0.37/0.11 (best-ever) but earlier
  pairs undershoot and the low-mass 60-148 kpc overshoot hits +0.080
  dex (out of band). Recommendation recorded (adoption unchanged
  pending user review): DiffMAH stays the operating point (physics
  band + differentiability + the exp41 layer's calibration); the
  real input is now an accuracy peer, not a fallback. Tech-note-2 §7
  caveat closed with the measured result.
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

## exp54 — deferred tests (2026-08-23)

- [ ] **Hold z=2.0 out of the fit.** Every five-epoch fit currently uses all
  five epochs simultaneously, so z=2 is a constraint, not a prediction. Fit on
  z <= 1.5 only and score z=2.0 as a genuine EXTRAPOLATION IN REDSHIFT. A law
  calibrated over 0.4 <= z <= 1.5 that also predicts z=2 demonstrates something
  a fitted curve does not, and it is the honest way to advertise the model's
  capability rather than its flexibility. Cost: one extra fit per model
  (~7 min for an E2 model, ~30 min for E5). Note the counterpart risk: the
  Stage 3.2 factorial fits only the two ENDPOINTS (z=0.4 and z=2.0), so a model
  selected there has already seen z=2 — the hold-out must therefore re-select
  as well as re-fit, or be reported as a hold-out test of an already-chosen
  model rather than a clean one.

### exp54 Stage 3.4 — the representational ceiling, v2 (2026-08-24)

- [x] **The ceiling, with the mask INSIDE the solve** (`stage34_ceiling.py`).
  Three sample definitions; the ladder's ordering is identical in all three.
  The basis draws any single epoch's profile to 0.9-2.2 per cent, so the
  profile family and size law are not the limitation; one causal weight vector
  for five epochs is what costs.
- [x] **The EXACT production-objective bounds** (`stage34_objective.py`).
  `score_F` and `score_A^2` are means over galaxies, so each is minimised galaxy
  by galaxy exactly. The exact shape ceiling is **0.282**, not the surrogate's
  0.538, against a model at 0.898 — and the permuted control under the same
  objective is **0.319**, so seven eighths of that gap is dictionary
  overcompleteness.
- [x] **The increment test, solved in the units it reports.** The truth's own
  decline costs more than the basis at every interval, reversing v1.
- [x] **The temporal-resolution ladder.** K=1,2 are an identity at z=2; most of
  the reachable improvement arrives by K=3-8.
- [x] **The central-decline split.** 53 per cent of cohort galaxies decline at
  4.92 kpc; the model is -37.4 per cent there and only -11.5 per cent among the
  rest.
- [x] **`c = c0 + c_z ln(1+z)` rejected at the bound**, at two freedom levels
  and confirmed on all 2397 galaxies and on the 840 cohort. Not built.
- [~] ~~The compact second deposit channel~~ — **deprioritised, not retired.**
  It cannot address a temporal decline, which is the larger half of the z=2
  central deficit. It remains a conditional hypothesis for the non-declining
  galaxies, whose positive increments are the ones the current basis fits worst
  (4.65 per cent against 2.11 per cent).

### exp54 — what the ceiling now points at, in order

- [x] **Enrich the efficiency law's TIME dependence — DONE, and NEGATIVE**
  (Stage 3.5, `stage35_time_law.py`). Five nested shared forms — Legendre
  polynomials of degree 2-4 in `ln(1+z)`, and hinges at z = 1, 3, 6 — moved the
  profile-shape error from 13.80% to at best **13.62%**, against a 10-11%
  target. A FREE shared curve (`E8`) reaches 13.64%, so **0.2 percentage points
  is the whole budget for a shared time law and two parameters already collect
  it**. The new coefficients are not determined: conditional loss rise 0.0035
  to 0.047 against 0.14 to 39.4 for the parameters E2 already had, bootstrap
  spread 10.6% to 201% of their own value against 0.9% to 8.1%, and three are
  outright unstable. Adding them also doubles the uncertainty on `a_z`. The
  judge ranks the incumbent SECOND, ahead of three of five enrichments.
  **Nothing adopted; `gompertz_log-E2-S2` stands.**
  - The premise was wrong and is corrected in the record: the K-interval ladder
    that motivated this is solved **per galaxy**, so its 10.5% at K=3 was never
    a target for a law with no per-galaxy freedom. `bins1` is identical to the
    fitted model, which was the tell.
  - The correction the extra parameters buy is spent above z = 4, where only
    7.3% of the deposited stellar mass is made (1.0% above z = 7).
- [x] **Repair the objective — DONE 2026-08-25, NEGATIVE** (Stage 3.6,
  `stage36_objective.py`). The loss IS blind past its 100 kpc pin: an error in
  the 132-148 kpc shell must be **175,000x larger** than one inside 2 kpc to
  cost the same. Five objectives fitted, varying the pin and the compared
  quantity one at a time. The pin is not a cause of anything (0.005 points).
  The quantity shifts the central error by a **constant** across epochs
  (varying 0.006 dex) and never changes its epoch dependence. And the repair
  made the outskirts WORSE. **exp48's winner is disqualified**: a surface
  density drops the mass inside the innermost radius entirely — verified to a
  relative 1e-13 and now asserted in `hongshao/objective.py`. New
  `quantity="shells"` added, bijective with the curve of growth. Nothing
  adopted.
- [x] **The size law's dependence on epoch — DONE 2026-08-25** (Stage 3.7,
  `stage37_size_epoch.py`). The central error's 0.145 dex swing between
  redshift 0.4 and 2 is real at fixed halo mass (0.175-0.192 in mass bins,
  0.131 on a fixed cohort). Four nested enrichments reach at best 0.132 against
  a preregistered target of 0.10. **But a free shared size law fitted to reduce
  the swing DIRECTLY reaches 0.061 dex, a 58% reduction** — the defect is
  reachable and the loss simply never asks for it. **The cost is the finding**:
  closing 0.078 dex of central deficit at redshift 2 costs 0.104 dex of
  outskirt mass at the same epoch. With one radial scale per deposit the centre
  and the outskirts cannot both be right. Nothing adopted.
- [x] **The high-mass amplitude failure — WITHDRAWN, it was one galaxy.** See
  the row-181 entry below and tech note Section 5.3.1. The production loss
  cannot see the outskirts at high redshift (at z=2 only 6.3 per cent of the
  mass lies beyond 52 kpc) and is nearly blind to monotonicity once each epoch
  is renormalised. Re-examine exp48's density-plus-log-residual objective.
- [x] ~~**The high-mass amplitude failure.** `score_A` 1.278 on the cohort.~~
  **WITHDRAWN 2026-08-25**: one broken cross-match (row 181) inside the fitting
  sample carried 17.8% of the loss. Corrected, the model is 3-11% worse than a
  plain regression and **6% better at redshift 2**, and the completeness
  restriction improves it rather than degrading it.
- [x] **A compact second channel — BUILT, FITTED, AND REJECTED 2026-08-25**
  (Stage 3.8, `stage38_two_channel.py`). Each deposit became a mixture of an
  extended blob and a compact exponential at a fraction of its radius, with the
  mixing weight allowed to be constant or to rise toward early times. **The
  weight went to zero in all five fits**, both under the production loss (which
  returned the incumbent's value, 1.874253, to six decimals, with every
  diagnostic numerically identical) and when asked to minimise the central swing
  directly. It cannot help: a component that ADDS central mass cannot reproduce
  a central mass that FALLS, and the galaxies whose centres do not fall are
  already fitted to 0.045 dex against a per-galaxy scatter of 0.16-0.20 dex.
- [x] **THE LIMIT IS THE MODEL CLASS.** With every deposit non-negative and its
  profile fixed once made, `M*(<R,t)` is non-decreasing in `t` at every radius,
  at every parameter value. In **42%** of these galaxies the measured central
  mass falls, by a median of **14%** from z=2 to z=0.4. Four enrichments
  (Stages 3.5, 3.6, 3.7, 3.8) failed against the same defect because it lies
  outside the reachable set. **User's decision: report this honestly; do NOT
  refit on the non-declining subset.**
- [ ] **NEXT, user-proposed: an empirical expansion term.** Observationally the
  central density of massive galaxies appears to decrease with time — their
  profiles "expand" near the centre — plausibly tied to black-hole evolution,
  so the link to halo assembly is secondary at best; an empirical description
  is all that is wanted. The natural form: let each deposit expand homologously
  after it is laid down, both radii scaled by `g = (t_k/t_j)^alpha`. Mass is
  conserved, nothing goes negative, no stellar information enters, `alpha = 0`
  nests the current model — and `M*(<R)` at fixed small R can then decrease.
  **Changes the model class**: Stage 0's monotone-in-time assertion becomes
  conditional, and the deposit basis starts depending on the viewing epoch
  (measure the cost, do not estimate it). NOT started.
- [ ] ~~A compact channel, IF it earns it~~ — superseded by the above.
  Original wording follows:
- [ ] **A compact channel, IF it earns it**, following the reviewer's nested
  sequence: map the inner residual against epoch-local halo variables with a
  shuffled-secondary null; then a fixed compact fraction; then mass/redshift
  dependence; then formation history only if it beats its shuffle. Test on the
  non-declining subset, where it is the only remaining candidate.
- [ ] **A mass-dependent size-evolution exponent** (`b_M m log10(1+z)`), which
  `S3` never tested — it varied the size NORMALISATION with halo properties,
  not the redshift EXPONENT.

### exp54 — corrections owed to the record (from the 2026-08-23 review)

- [x] The curve of growth is a PROJECTED elliptical-aperture mass, not a
  spherical enclosed mass (verified in `hongshao.tng_data`: it is a cumulative
  integral of the X-Y isophote surface density).
- [x] The sample is selected on PEAK HALO MASS at z=0.4, not on stellar mass.
- [x] `C = 3` is an empirical convention, not a scanned optimum.
- [x] `fit.identifiability`'s per-parameter curves are CONDITIONAL SLICES, not
  profile scans — the other parameters are held fixed.
- [x] The 0.03 dex growth figure is a median bias. Re-measured here: per-galaxy
  sd **0.208 dex**, NMAD 0.179, correlation 0.71, and a predicted growth
  distribution **half as wide** as TNG's (0.145 against 0.284 dex).
- [ ] **Re-optimised parameter profiles for E2-S2** — fix one parameter on a
  9-15 point grid and re-optimise the other six with multiple starts. Not run;
  this is the missing piece of the identifiability claim.
- [ ] **A truncation-factor sensitivity test**: refit at `C = 1, 2, 3, 5` and
  report amplitude and shape scores, mass beyond the last aperture, and the
  change in the fitted efficiency.
- [ ] **Completeness-threshold sensitivity**: repeat the Section 6 conclusions
  over the 50/60/70/80 per cent fraction, the mass-bin width and the plateau
  rule.
- [ ] **Persist the recalibrated per-epoch regression benchmarks** so the
  mh-complete `score_A` table is reproducible from one output file.
- [ ] **A stochastic residual layer.** The mean law cannot generate TNG's
  growth diversity by construction; this is scope, not a defect.

## exp50 — direct DiffMAH-to-CoG conversion

- [x] Test whether a stable, interpretable coordinate vector consisting of
  total stellar mass, the 2-kpc central fraction, and ordered R20/R50/R80 can
  replace the PCA CoG coordinates at z=0.4 without measurable predictive loss.
  **Result:** on 2,539 galaxies, the readable degree-2 map has held-out profile
  CRPS 0.06327 dex versus 0.06317 dex for the like-for-like PCA-3 emulator. The
  readable map is worse by 0.00010 dex, less than the PCA score's measured
  0.00024 dex fold-to-fold standard deviation, so it passes the predeclared
  parity test. The map remains probabilistic: total stellar mass has held-out
  R2=0.87, while the four shape coordinates have R2=0.15--0.36, and correlated
  draws—not the conditional mean—recover the population width. Mass-conditioned
  shuffle controls confirm distinct information from MAH shape and z=0.4 halo
  concentration. Residual-only symbolic regression recovers 62% of the dense
  degree-2 improvement with three stable nonlinear terms. In the 2,366-galaxy
  overlap sample, five-epoch concentration history improves profile CRPS by
  2.89% relative to current concentration, while shuffled histories make it
  0.12% worse. **Decision:** retain the readable conversion as a successful
  experimental alternative to PCA; test concentration history across epochs
  before considering library graduation.

## exp51 — redshift evolution of the readable CoG conversion

- [x] Repeat the direct DiffMAH-to-CoG mapping at z=0.7, 1.0, 1.5, and 2.0 on
  the same progenitor sample and held-out folds. Separate descendant-conditioned
  prediction, which may use the final halo's complete DiffMAH fit, from
  epoch-local prediction using only information available by that epoch.
  Measure representation stability, parity with PCA, coefficient interpolation
  across redshift, cross-epoch residual coherence, and the incremental value of
  concentration measured at the correct epoch. Require direct profile figures
  and mass-conditioned shuffled controls before interpreting an assembly
  signal. **Result:** on 2,395 galaxies, the five readable profile coordinates
  pass the matched PCA parity test at every snapshot from z=0.4 to z=2; their
  profile CRPS is worse by only 0.00018--0.00070 dex, below the PCA score's
  measured fold-to-fold variation. Interpolating descendant-conditioned model
  coefficients worsens held-epoch CRPS by only 0.43--1.23%, within the 5% gate.
  More importantly, fitting DiffMAH only to halo history available at each
  prediction epoch and adding concurrent concentration lowers profile CRPS by
  20.79--25.41% relative to current halo mass alone. At z=1.5 and z=2, this
  portable model is 8.28% and 8.17% better than using the final descendant's
  complete DiffMAH parameters. Concurrent concentration improves on z=0.4
  concentration by 2.23--4.66% at z>0.4, and mass-conditioned shuffles remove
  the gain. Correlated stochastic draws substantially repair the conditional
  mean's excessive cross-epoch size-rank persistence, though the z=0.4-to-2
  correlation is slightly too weak. A nested z=2 symbolic search improves the
  linear map by only 1.00%, versus 5.01% for dense degree-2 terms, so no compact
  redshift-independent equation is adopted. **Decision:** retain the readable
  degree-2 conversion as the working direct-map candidate. Before library
  graduation, test held-epoch coefficient closure for the epoch-local inputs
  under one common physical scaling convention and refine the temporal
  residual model.
  **Documentation:** a 35-page mathematical and methodological report covering
  Exp50 and Exp51 is maintained in
  `experiments/exp51_direct_cog_redshift/doc/`. It includes exact coordinate
  definitions, the inverse monotone decoder, the probabilistic model, direct
  profile figures, control logic, claim qualifications, and a reproducibility
  map. The source compiles without unresolved references or overflowing text;
  the generated PDF and staged figure copies are gitignored.

## exp55 — analytic CoG representation and identifiability

- [x] Test whether a continuous three-to-five-parameter CoG function can match
  the low-dimensional PCA representation at redshift 0.4 while retaining
  parameters stable enough for a later DiffMAH-to-profile map. Compare PCA-3/5,
  the Exp50 five-coordinate decoder, one-component analytic families, and a
  deliberately restricted compact-plus-extended Sérsic mixture. Treat
  reconstruction quality and parameter identifiability as equal gates. Use
  multiple starts, re-optimized profile scans, residual-Jacobian singular
  values, synthetic recovery, boundary audits, and direct CoG plus density
  figures. Do not fit the halo relation until a profile representation passes.
  **Result:** on all 2,545 usable redshift-0.4 CoGs, a four-parameter compact
  exponential plus extended Moffat profile reaches 0.00441 dex median
  cumulative-profile RMS under fold-clean selection of the two global shape
  indices. Its paired median difference from Exp50's five sampled coordinates
  is -0.000061 dex, with a bootstrap 16th--84th percentile interval spanning
  zero, while PCA-3 remains better by 0.000975 dex. The selected analytic
  family has a median scaled-Jacobian singular-value ratio of `10^-1.70`, only
  2.55% boundary incidence, exact synthetic recovery, and median parameter
  changes below 0.008 dex (or 0.004 in compact fraction) when alternating radii
  are removed. **Decision:** adopt this function as the profile target for a
  separate held-out DiffMAH mapping experiment, with per-object conditioning
  flags and no in-situ/ex-situ interpretation. The subsequent standard QA
  confirms 0.003--0.006 dex scatter in cumulative apertures but finds the
  expected derivative limitation: the mass beyond 100 kpc has +12.1% median
  bias and 0.161 dex scatter, and the median absolute annular-density error
  reaches 0.132 dex near 140 kpc. The halo-mapping experiment must therefore
  judge differential profiles as well as CoGs.
- [x] **exp55 halo-mapping stage.** Predict the four selected
  analytic CoG coordinates from `[DiffMAH(4), c_200c]` on the same five
  held-out folds as exp50. Require final-mass-only, mass-conditioned shuffled
  MAH-shape, and shuffled-concentration controls; predict the joint residual
  distribution; reconstruct full CoGs; compare directly with exp50 on common
  galaxies; and run the full standard QA battery on both the conditional mean
  and correlated draws before deciding whether these analytic coordinates are
  suitable Ultimate-SHMR targets. **Result:** on 2,539 held-out galaxies, the
  analytic map has 0.08519 dex median cumulative-profile RMS error, effectively
  tied with the Exp50 sampled-coordinate map, and its mean profile CRPS is
  0.00094 dex worse. Real MAH shape lowers CRPS by 13.20% relative to its
  mass-conditioned shuffle, and real concentration lowers it by 4.44% relative
  to its shuffle. Stellar mass within 148.2 kpc is predictable (held-out
  R2=0.865), but compact fraction and compact radius are mostly stochastic
  (R2=0.060 and 0.068). Correlated draws have accurate marginal 68% profile
  coverage (68.3%) but broaden the tight TNG `<2Re` versus `2--4Re` plane to
  0.188 dex scatter from the TNG value of 0.072 dex when every generated galaxy
  is correctly measured at its own `Re`. The initially reported 0.213 dex used
  the TNG counterpart's true `Re` and is retired. **Decision:** retain the
  analytic CoG as a viable continuous halo-map target, but revise the residual
  coordinates and joint stochastic model before adding epochs or freeing the
  globally fixed component shapes.
- [x] **exp55 halo-map refinement, Stages 1--4.** Decompose the
  halo-mass-dependent radial residuals into representation, mapping, and
  parameter-derivative contributions; compare a predeclared small set of
  fold-clean analytic mean expansions; rotate the residual coordinates using
  the analytic profile metric; and test simple stochastic alternatives against
  both pointwise profile coverage and the standard population planes. Do not
  add epochs or free the globally fixed profile indices in this work block.
  **Result:** the halo-to-profile map contributes 0.08494 dex median profile
  RMS, about 19 times the analytic function's 0.00437 dex representation error.
  None of three broader polynomial means passed the predeclared combined gate:
  the complete degree-2 basis lowered profile CRPS by only 0.45% while
  increasing the worst mass-bin shape residual from 5.45% to 6.49%. A
  profile-metric rotation separates four residual modes containing 62.5%,
  24.4%, 10.2%, and 2.9% of residual profile variance, but is not claimed as a
  mean-model improvement. Replacing raw Gaussian residuals with fold-local
  nearest-neighbour residuals, inflated by a single scale selected on inner
  held-out galaxies, lowers profile CRPS from 0.06429 to 0.06368 dex and gives
  68.4% coverage for the nominal 68% interval. It reduces the generated
  self-consistent `<2Re` versus `2--4Re` scatter from 0.188 to 0.084 dex,
  compared with 0.072 dex in TNG. **Decision:** retain the sparse mean and use
  the nested-calibrated empirical residual generator experimentally. The main
  unresolved defect is the R90 population relation: its energy distance is
  3.63 times the TNG split-half sampling floor because its mean mass trend is
  too shallow, despite realistic scatter.
- [x] **exp55 outer-envelope diagnosis, Stages 5--9.** Establish stochastic
  robustness; separate R50/R80/R90 errors at the analytic-representation,
  held-out mean, and generated-population layers; test whether actual-minus-
  DiffMAH history residuals explain the failure; compare the existing fixed
  outer-slope candidates under fold-clean multi-metric gates; audit a small set
  of added halo features with mass-conditioned shuffles; and test one
  predeclared mass-dependent boundary. **Result:** the original fixed
  `gamma=1.25` representation itself changes the TNG mass--R90 slope from
  0.291 to 0.251 despite only 0.00437 dex median CoG RMS. DiffMAH history misses
  correlate only -0.030 to +0.045 with the R90 residual and do not beat their
  shuffled controls. Fixed `gamma=1.4` improves density RMS from 0.06601 to
  0.06093 dex and the generated R90 energy-distance ratio from 3.40 to 2.20 at
  negligible CoG and CRPS cost. A fold-local boundary using `gamma=1.4` below
  the training 2/3 halo-mass quantile and `gamma=1.25` above it passes every
  declared gate: 68.9% nominal-68% coverage, 0.06377 dex profile CRPS, and R90
  energy-distance ratio 0.96 relative to the TNG sampling floor. Its complete-
  population R90 slope is 0.302 versus 0.291 in TNG. **Decision:** retain
  `gamma=1.4` as the preferred global baseline and the hard boundary as a
  diagnostic ceiling. The result establishes mass dependence of the required
  outer slope, not a physical discontinuity.
- [x] **exp55 smooth outer-slope law.** Replace the successful hard boundary
  with one low-complexity continuous `gamma(log M_peak)` relation. Select any
  transition scale and width only inside each outer training fold. Refit the
  profile representation at the resulting per-halo slopes rather than
  interpolating final predictions, and repeat identifiability, density, radial-
  residual, null-control, coverage, and complete-population gates. Compare
  against both fixed `gamma=1.4` and the two-regime diagnostic. Do not free
  `gamma` independently for every galaxy, and do not add epochs until this
  single-epoch coordinate system is stable. **Result:** a logistic law between
  the tested slopes 1.4 and 1.25 selects the widest tested transition, 0.20 dex,
  in all five training folds. On 2,539 held-out galaxies it has 0.06391 dex
  profile CRPS, 69.5% coverage for a nominal 68% interval, and a generated
  stellar-mass--R90 slope of 0.2895 versus 0.2909 in TNG. Its R90
  energy-distance ratio is 0.93 relative to the TNG sampling floor, and both
  fixed-kpc population planes pass. It formally fails replacement because the
  self-consistent size-scaled ratio is 10.0034% above the hard boundary, barely
  outside its 10% gate, and the largest mass-bin profile residual is 3.34%
  versus 2.74%. **Decision:** keep the formal non-selection, but regard the
  continuous law as a successful feasibility demonstration. At this stage the
  0.20-dex width was the widest tested value; the separately declared closure
  test below establishes it as an interior coarse-grid choice, not a
  high-precision physical scale.
- [x] **exp55 smooth-law closure.** In a newly declared test, extend the
  transition-width grid above 0.20 dex and diagnose the smooth model's 3.34%
  intermediate-radius mass-bin residual. Retain the same outer folds and final
  gates; do not relax the just-missed size-scaled threshold after seeing it.
  Stop if the wider law merely moves to a new boundary or if the radial
  residual does not improve. Only a candidate with an interior width and stable
  geometry should be considered for multi-epoch testing. **Result:** after
  adding widths 0.30, 0.50, and 0.80 dex, every fold still chooses 0.20 dex.
  The median training stellar-mass--R90 slope error decreases from 0.0107 at
  width 0.10 to 0.0020 at 0.20, then rises to 0.0051, 0.0134, and 0.0196. Thus
  0.20 dex is an interior, reproducible grid choice. The held-out result is
  unchanged and still fails the same radial-shape and size-scaled gates.
  **Decision:** close the width scan; diagnose the 5--30 kpc radial residual
  before considering an epoch extension.
- [ ] **exp55 population-QA performance.** Cache the truth measurements and
  fixed TNG split-half sampling floors, then vectorize repeated energy-distance
  calculations across complete draws. The Stage 10--11 all-32-draw comparison
  currently takes about ten minutes and should remain a final robustness step
  until this mechanical optimization is verified to reproduce every metric.

## exp56 — compact-shape closure and radial halo–CoG relation

- [x] **Stage 0: freeze the references and predeclare the test.** Retain fixed
  `gamma=1.4` as the simple baseline, the hard halo-mass boundary as the
  diagnostic ceiling, and the smooth `gamma(log M_peak)` law with width
  0.20 dex as the continuous candidate. Do not retune the outer-slope
  endpoints, transition, or width while testing the compact component.
  **Predeclared in the Exp56 README before writing the driver.**
- [x] **Stage 1: fold-clean compact Sérsic-index grid.** Compare globally fixed
  `n = 0.5, 0.75, 1.0` under both fixed `gamma=1.4` and the frozen smooth
  outer-slope law. Refit the same four galaxy coordinates for every cell.
  Select on training galaxies only, require a bootstrap-resolved reduction of
  the 5--30 kpc residual, and preserve cumulative-profile accuracy, density
  accuracy, R50/R80/R90, parameter conditioning, boundary incidence, and
  radial-jackknife stability. Do not free `n` per galaxy.
  **Result:** the complete mass-stratified 90-galaxy path passed in 31.11
  seconds, then the declared full grid fitted 2,539 galaxies in 478.88 seconds.
  `n=1` is the numerical winner in all five outer training folds under both
  outer laws. Under the frozen smooth law its median 5--30 kpc CoG RMS is
  0.00495 dex against the TNG CoG, better than the worse 0.00625 dex for
  `n=0.75` and 0.00947 dex for `n=0.5`; its density RMS is likewise better at
  0.02847 dex versus 0.03591 and 0.04812 dex. Lower indices also move R80/R90
  inward, and `n=0.75` has one large radial-jackknife coordinate excursion.
  **Decision:** retain `n=1`; a lower compact index does not repair the
  5--30 kpc residual.
- [x] **Stage 2: component and degeneracy diagnosis not triggered.** Plot the
  compact and extended CoGs and density profiles separately. Use profiled scans
  that fix one coordinate while re-optimizing the other three to determine
  whether `n<1` removes compact-component leakage at 5--30 kpc or merely
  transfers mass among compact fraction and the two radii. The component
  figure was completed, but no lower index survived Stage 1, so the
  conditional profiled scans were not run.
- [ ] **Stage 3: refit the held-out halo–CoG relation.** Only the representation
  selected inside the training folds proceeds. Refit the conditional mean and
  stochastic residual distribution from `[DiffMAH(4), c_200c]`; repeat the
  final-mass-only, mass-conditioned shuffled-MAH, and shuffled-concentration
  controls and the complete standard QA battery.
- [ ] **Stage 4: measure radial halo-information content without imposing the
  expected answer.** Compare nested held-out relations using final mass,
  recent DiffMAH growth, early assembled fraction, the complete DiffMAH
  history, and concurrent concentration. Use mass-conditioned shuffles and
  report incremental CRPS and RMS as functions of radius and in differential
  annuli. Treat observational motivation as a reason to run the test, not as
  a constraint on the result.
- [x] **Stop conditions.** Stop after the declared Sérsic grid. If `n=0.5`
  is selected, report a one-sided result rather than extending automatically.
  Do not add epochs until the redshift-0.4 representation, identifiability,
  stochastic population geometry, and 5--30 kpc residual are stable. The
  declared grid is closed without extending `n` or launching a joint global
  `(n, gamma)` fit.
- [x] **Stage 1b: jointly fit one global compact index and one global Moffat
  index.** On a predeclared focused 5-by-5 grid, profile out the same four
  galaxy coordinates, select grid cells on the five outer training folds, and
  attempt one local quadratic refinement only if the winner is interior and
  fold-stable. Compare with fixed `(n=1, gamma=1.4)` and the frozen smooth
  outer law using 5--30 kpc residuals, full CoG and density accuracy,
  R50/R80/R90, conditioning, boundaries, exact recovery, four-start agreement,
  radial jackknifes, and direct profile figures. The complete 90-galaxy path
  must remain below one minute before a full run. **Predeclared in the Exp56
  README before writing the Stage 1b driver. Result:** the complete demo passed
  in 39.04 seconds and the 2,539-galaxy grid finished in 1,236.58 seconds. All
  five training folds and 99.7% of galaxy bootstraps choose the declared
  upper-bound cell `(n=1.25, gamma=1.4)`, so the conditional continuous
  refinement is not attempted. The cell improves the paired 5--30 kpc CoG and
  density errors relative to fixed `(n=1, gamma=1.4)`, but worsens full-range
  CoG RMS from 0.00440 to 0.00459 dex and worsens local conditioning. Its
  stellar-mass--R90 slope is 0.2732 versus 0.2909 in TNG and 0.2924 for the
  frozen smooth law. Exact recovery, multi-start agreement, boundaries, and
  radial jackknifes pass, so this is a real profile trade rather than a failed
  optimizer. **Decision:** do not adopt or extend the boundary pair; retain
  the fixed single-slope baseline and the frozen smooth outer-shape candidate.
