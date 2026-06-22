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
- [ ] **graduate the emulator into `hongshao/`** — a single fit/predict/sample
  module: linear mean on `DiffMAH + c_200c` (default) OR the 7-term degree-2 poly
  (option), + heteroscedastic full covariance, validated, with a self-check.
  Expose a `sample()` path (exp15: the model must be used generatively).
- [ ] apply the emulator to an N-body / other-sim halo catalog with DiffMAH fits;
  compare predicted profile distributions (the portability test).

### Later
- [ ] secondary halo properties — *test*, don't assume: MAH-derived ones
  (concentration, accretion rate) likely redundant with the MAH (exp06); only
  initial conditions / environment are independent (hard to get here).
- [ ] apply emulator to an N-body catalog; compare profile distributions.
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
- [ ] exact snap-72 halo mass — **resolved** (`logmh_z0p4`).
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
- [ ] does `c_200c` also help the **scatter** model (exp14) / shrink exp15's
  regression-to-mean by raising R²?
- [ ] **projection-scatter budget** — quantify projection vs intrinsic halo-to-halo
  scatter per annulus from the 3 projections (refines exp13/14/15's "intrinsic" floor).
