# exp14 — Scatter model: a heteroscedastic residual covariance

## Question
The mean is settled (exp09/exp12: linear) and the four DiffMAH params are at the
information ceiling for the outskirts (exp13). What remains is **scatter**. The
exp08/exp11 emulator uses one residual covariance for every halo
(homoscedastic): `P(aperture masses | DiffMAH) = N(mean(X), Σ)`. If the
predictive scatter actually depends on the halo, this model is mis-calibrated
*conditionally* even when it looks fine *marginally* — over-confident for noisy
halos, under-confident for clean ones. Does a feature-dependent scatter fix
that, and is the Gaussian shape adequate?

## Method
(A) **Diagnose** whether the residual scatter depends on the DiffMAH features
(bin residual std by feature; fit a log-variance model). (B) **Model** it with a
heteroscedastic Gaussian: per-aperture log-linear standard deviation
`σ_j(X) = exp(γ_j · [1, X_std])` (positivity automatic; fit by Gaussian maximum
likelihood with a ridge on the slopes), and a fixed residual correlation `R` from
the standardized residuals, so `Σ(X) = D(X) R D(X)`. (C) **Judge** with the exp07
suite — marginal CRPS, marginal *and conditional* calibration, joint log-score —
plus a PIT non-Gaussianity check.

The mean is the same per-aperture OLS as exp11; only the covariance changes.
Everything is 5-fold CV, out-of-fold. Inputs: `data/processed/tng300_072_z0p4.fits`
(use sample, n=2539), DiffMAH params, `hongshao.metrics`. Runs in ~2 s.

## Key result
**The scatter is strongly heteroscedastic and driven by `late` (recent
accretion); modeling it barely changes marginal sharpness but makes the per-halo
uncertainties honest — conditional calibration improves ~10×.**

| metric | homoscedastic (exp11) | heteroscedastic | change |
|---|---|---|---|
| marginal CRPS | 0.0883 | 0.0873 | +1.1% |
| joint NLL [nats] | −3.109 | −3.349 | **+0.240** |
| marginal coverage 50/68/90/95 | 54/72/91/95 | 52/70/90/95 | both ok |
| **conditional coverage gap** (high−low σ tercile) | **0.189** | **0.018** | **~10× better** |

- **What drives the scatter** (log-σ slopes, per +1σ feature; predicted σ spans a
  factor 3.5–5.8 across halos):
  - **`late` (late-time accretion index): +0.16 to +0.22** for the three outer
    annuli — halos with vigorous recent accretion have much noisier outskirts.
  - `early`: −0.07 to −0.09 (early-formers are tighter).
  - `logmp`: −0.08 outer (massive halos tighter, more particles) but **+0.10 for
    the core** (massive BCG cores are more varied).
  - `logtc`: ≈0 (transition time alone does not set the scatter).
- **`late` is where everything happens.** It is *both* where the mean
  nonlinearity lives (exp12's `late²`) *and* where the scatter concentrates:
  recent accretion both boosts the outer envelope super-linearly and destabilizes
  it (stochastic merger timing/projection). A single physical axis governs the
  outskirts' mean curvature and its variance.
- **Marginal vs conditional.** The homoscedastic model is already fine
  *marginally* (its average interval coverage matches nominal), which is why
  marginal CRPS hardly moves. But split by predicted noisiness it is badly off:
  it covers 78–82% of "clean" halos inside the 68% interval (over-confident
  intervals... too wide) and only 60–64% of "noisy" halos (too narrow). The
  heteroscedastic model is flat at ≈0.68–0.71 across terciles. This is the whole
  point of a *conditional* density model and the headline of the experiment.
- **Gaussian is adequate.** The PIT histogram for the homoscedastic model is a
  mild dome (std 0.278 vs ideal 0.289); the heteroscedastic PIT is flatter (std
  0.284) with no U-shape or heavy tails. The residuals are adequately Gaussian
  **once the variance is made halo-dependent** — the defect was heteroscedasticity,
  not tail shape, so no Student-t / flow is warranted.

## Decision
- **Adopt the heteroscedastic covariance `Σ(X) = D(X) R D(X)` as the emulator's
  scatter model.** It keeps the linear mean and the portable DiffMAH features,
  costs four extra parameters per aperture, and makes the per-halo predictive
  uncertainties honest (+0.24 nats joint score; conditional calibration gap
  0.19 → 0.02) — exactly the property a `P(profile | halo)` model needs.
- **No richer likelihood needed.** A Gaussian predictive with feature-dependent
  variance is sufficient; non-Gaussian tails are not the limiting factor.
- This completes the Ultimate-SHMR emulator specification: **linear mean +
  heteroscedastic full covariance on portable DiffMAH features.** Natural next
  step: graduate it into `hongshao/` as a single fit/predict module, then run the
  portability test on an N-body / other-sim catalog.
