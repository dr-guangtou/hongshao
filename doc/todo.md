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
- [ ] (optional follow-ons) the **density profile in Re units**; feed the
  predictive profile uncertainty to the forward model.
