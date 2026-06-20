# exp07 — How to evaluate model fitting and recovery

A methods experiment: decide the **metrics** that judge the Ultimate-SHMR models
before we build the probabilistic emulator (exp08). The recommended suite is
graduated into the library (`hongshao/metrics.py`,
`hongshao/tng_data.cog_sigma_dex`); this experiment demonstrates it and shows
*why* the naive choices (per-radius CoG dex, RMS alone, chi-square on the raw
curve of growth) are misleading.

## Questions

1. **Predictors** (regression that outputs a distribution `P(target | halo)`):
   in what quantities should we measure recovery, and how do we score the
   *distribution* — not just the point prediction?
2. **Profile fits** (the 5-param radial-DiffMAH CoG model): how do we judge fit
   quality and compare models, given that per-point CoG errors are not stored?

## Method

- **Aperture / annulus masses.** Recovery is measured in five physical stellar
  masses built from the *measured* aperture array (SMA = 10,30,50,75,100 kpc):
  `<10`, `10-30`, `30-50`, `50-100`, `<100` kpc — not per-radius CoG dex.
- **Distributional scores.** A Gaussian-scatter baseline predictor (5-fold CV
  linear mean + homoscedastic per-target sigma from training residuals) is
  scored with **CRPS**, **predictive log-score**, and **interval calibration**
  (empirical vs nominal coverage). M0-only vs M0+history, with a shuffle control.
- **Residual covariance.** The 5×5 correlation matrix of the prediction
  residuals across apertures — what an emulator's scatter must reproduce.
- **Per-point CoG noise.** `cog_sigma_dex` propagates the isophote intensity
  error (`intens_err`) through the elliptical-annulus cumulative integral and
  returns a fractional error in dex on the CoG grid (resolves the chi-square
  blocker; caveats below).
- **Fit diagnostics.** Reduced chi-square of the single sigmoid; **AIC/BIC**
  model comparison (single vs double sigmoid, vs Sersic, vs a 3-mode PCA
  reconstruction) computed **both** on the raw CoG and in the decorrelated
  annulus space; and the **mean residual profile** vs radius.

Driver: `run.py` (`EXP07_NMAX=300` for a sub-minute pass). Figures:
`exp07_track1_recovery`, `exp07_track1_distribution`,
`exp07_track2_fit_diagnostics`, and two per-object QA figures —
`exp07_track1_residual_vs_true` (residual vs true mass with marginal
histograms, per aperture) and `exp07_track2_cog_fits_by_mass` (measured vs
fitted CoG and residual profile in 3 equal-count mass bins). Sample: n = 2545
(`use` cut); 2538 with finite aperture masses.

## Key results

**1. Aperture/annulus masses are the right target, and history helps
everywhere.** Cross-validated recovery RMS (M0 → M0+history), shuffle ~0%:

| aperture/annulus | M0 only | M0+history | improvement |
|---|---|---|---|
| `<10` kpc | 0.143 | 0.115 | +19.5% |
| `10-30` | 0.202 | 0.160 | +21.0% |
| `30-50` | 0.232 | 0.177 | +23.8% |
| `50-100` | 0.230 | 0.178 | +22.9% |
| `<100` kpc | 0.146 | 0.102 | +29.7% |

The scatter is largest in the intermediate annuli (~0.23 dex) and history gives
its biggest *fractional* gain on the total `<100` kpc mass. Per-radius CoG dex
hides this: the 24 cumulative CoG points are **93% correlated** (mean
|off-diagonal|), so a per-radius dex curve double-counts the same information.

**2. Score the distribution, not just the mean.** Adding history improves the
proper scores as much as the RMS, and the Gaussian-scatter baseline is already
**well-calibrated**:

- CRPS: 0.106 → 0.080 dex (**+24%**); log-score: −0.26 → −0.53 nats.
- Coverage (nominal → empirical): 50%→0.54, 68%→0.71, 90%→0.91, 95%→0.95
  (very mild conservatism). This validates the homoscedastic-Gaussian model as
  the **baseline the exp08 emulator must beat** on CRPS/log-score, not RMS.

**3. The residual scatter is correlated → the emulator must draw correlated
scatter.** The residual correlation across apertures has **mean |off-diagonal|
= 0.57** (adjacent annuli up to 0.85). A model that draws independent
per-aperture (or per-radius) noise would get the joint distribution wrong even
with perfect marginals. The metric to track is the residual covariance matrix.

**4. The propagated isophote noise is a sensible CoG error.** With `cog_sigma_dex`
(~0.014 dex at 2 kpc falling to ~0.004 dex by 150 kpc), the single-sigmoid
**reduced chi-square has median = 1.00** (16–84% [0.28, 3.21]). So the
radial-DiffMAH fit reaches the isophote-noise floor on average.

**5. Model comparison: correlation handling changes the verdict.** Fraction of
galaxies preferring each model by BIC:

| model (params) | BIC on raw CoG | BIC in annulus space |
|---|---|---|
| single sigmoid (5) | 4% | 9% |
| **double sigmoid (8)** | **94%** | **65%** |
| Sersic (3) | 0% | 6% |
| PCA-3 | 2% | 20% |

Treating the 24 cumulative CoG points as independent **over-rewards complexity**
(double sigmoid 94%); in the weakly-correlated annulus space the preference
falls to 65%. The honest reading: a second radial transition is statistically
favored for a *majority* of galaxies, but the gain is small — the single
sigmoid's **mean residual profile is a coherent S-shape of only ≲0.011 dex**
(under-fits the 2 kpc center, dips near ~15 kpc, bumps near ~50 kpc). That
coherent (not random) residual is exactly the missing-component signature, and
reconciles chi²≈1 (residuals are noise-*sized*) with BIC favoring two
transitions (residuals are noise-sized but *structured*). The standard Sersic
CoG is essentially never preferred.

**6. Per-object QA (the two extra figures).** `residual_vs_true` shows the
M0+history predictor's residual sloping *down* with true mass in every aperture
(bias ≈ 0, scatter 0.12–0.14 dex): the conditional-mean predictor **regresses to
the mean**, so its output is under-dispersed — a second reason the emulator must
add back (correlated) scatter rather than trusting the point prediction.
`cog_fits_by_mass` confirms the radial-DiffMAH fit tracks the measured CoG across
the full mass range, with the same coherent ≲0.01 dex residual S-shape in all
three mass bins (not a feature of any single mass scale).

## Interpretation & caveats

- `cog_sigma_dex` is isophote-modeling / azimuthal scatter, **not** Poisson
  measurement noise; a CoG reconstructed from the isophotes does not exactly
  match the stored CoG (different pipeline/grid), so only the *fractional* error
  is used. Cumulative CoG points are strongly correlated, so any chi-square /
  AIC / BIC on the raw curve overcounts the degrees of freedom — always
  decorrelate (annulus space, or use the full covariance).
- The Gaussian-scatter baseline is homoscedastic per target; the real scatter is
  mildly mass-dependent, so exp08 should test a heteroscedastic / full-covariance
  predictive and re-check calibration.

## Decision — the recommended evaluation suite (reused by exp08+)

- **Predictors:** report recovery in **aperture/annulus masses** (not per-radius
  CoG dex); score the predictive distribution with **CRPS + log-score**, check
  **interval calibration**, and always inspect the **residual covariance**
  (correlated → emulator must draw correlated scatter). RMS alone is insufficient.
- **Profile fits:** report **reduced chi-square** with `cog_sigma_dex`, the
  **mean residual profile** (coherent structure flags a missing component), and
  **AIC/BIC for model selection — computed in annulus space, never naively on
  the raw CoG.** The single sigmoid is adequate to the noise; a second transition
  is mildly favored and optional (5 interpretable params vs ~0.005 dex).
- Helpers live in `hongshao/metrics.py` (`crps_gaussian`, `gaussian_logscore`,
  `pit`, `interval_coverage`, `aic_bic`) and `hongshao/tng_data.cog_sigma_dex`.
