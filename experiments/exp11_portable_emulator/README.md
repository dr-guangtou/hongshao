# exp11 — The portable Ultimate-SHMR emulator

exp08 built a conditional multivariate-Gaussian emulator
`P(aperture masses | M0, MAH)` but on MAH-PCA(4) features, whose basis is defined
on the TNG sample and so doesn't transfer. exp10 fit DiffMAH to our own MAHs and
showed the four intrinsic params carry ~88% of the MAH-PCA signal for the *mean*.
Here we rebuild the full *probabilistic* emulator on the cached DiffMAH params and
confirm it matches the MAH-PCA version under the exp07 suite — so the emulator is
portable end to end, not just its point predictions.

## Question

Does the conditional-Gaussian emulator built on the four portable DiffMAH
parameters (`dmah_logmp, logtc, early, late`) match the MAH-PCA(4) version on
CRPS, calibration, and the joint residual covariance?

## Method

- Target: the 4 aperture/annulus masses (`<10, 10-30, 30-50, 50-100 kpc`).
- Emulator (as exp08): 5-fold CV, out-of-fold linear mean + per-fold full 4×4
  residual covariance → a Gaussian predictive per galaxy.
- Three feature sets compared under the same scoring: `M0 only`, `M0 + MAH-PCA(4)`
  (the incumbent), and `DiffMAH (portable)` — the cached `dmah_*` params.
- Scores (exp07 suite): per-target CRPS, predictive log-score, interval
  calibration, joint multivariate log-score (full vs diagonal covariance),
  residual-covariance match. Plus a DiffMAH-driven probabilistic painting.

The DiffMAH params are cached in the dataset (`hongshao/tng_data.py`,
`DMAH_FIT_TMIN_GYR`), symmetric with the `rdm_*` profile params. Driver: `run.py`
(`EXP11_NMAX=400` for a sub-minute pass). Figures: `exp11_skill_calibration`,
`exp11_painting`. Sample: n = 2533.

## Key results

**The portable DiffMAH emulator matches the MAH-PCA version.**

| features | CRPS [dex] | coverage 50/68/90/95 | joint NLL full−diag |
|---|---|---|---|
| M0 only | 0.1117 | 0.53/0.71/0.91/0.95 | — |
| M0 + MAH-PCA(4) | 0.0849 | 0.54/0.72/0.91/0.95 | −1.38 |
| **DiffMAH (portable)** | **0.0882** | **0.54/0.72/0.91/0.95** | **−1.43** |

- **Calibration is identical** — both well-calibrated (slightly conservative at
  50%, excellent at 90/95%).
- **CRPS is 4% higher** than MAH-PCA (0.0882 vs 0.0849), the same small gap exp10
  found — the price of a portable, smooth 4-param MAH summary. Both cut CRPS ~21%
  vs M0-only.
- **Full covariance still pays off** (+1.43 nats over diagonal), and the emulator
  **reproduces the correlated scatter** out of sample (residual mean |off-diag|
  0.52, matching exp08's 0.5–0.57).
- **Probabilistic painting works from DiffMAH features**: at fixed M0, early
  formers (high z50) are painted above late formers, the separation widening
  outward, with calibrated ±1σ bracketing the truth.

## Decision

Adopt the **portable DiffMAH conditional-Gaussian emulator** as the working
Ultimate-SHMR model. It is calibrated, reproduces the joint scatter, and costs
only ~4% CRPS vs the TNG-specific MAH-PCA version — in exchange for running on
intrinsic per-halo parameters that exist for any simulated (or, in principle,
observed) halo. The mean form is settled (linear, exp09) and the features are now
portable; the remaining lever is the **scatter model** (heteroscedasticity /
non-Gaussianity) under the exp07 suite.
