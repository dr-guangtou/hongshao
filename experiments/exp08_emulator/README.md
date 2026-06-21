# exp08 — The first probabilistic Ultimate-SHMR emulator

`P(stellar profile | M0, assembly history)` as a conditional multivariate
Gaussian (linear mean + full residual covariance), judged by the exp07 suite.
Two target representations compared head-to-head, decided in the previous
session: **A** predicts the physical aperture/annulus masses directly; **B**
predicts the 5 radial-DiffMAH params and reconstructs them to the same masses.

## Questions

1. Does a simple conditional-Gaussian emulator capture the assembly signal, and
   is it *calibrated* (honest uncertainties), scored properly (CRPS / log-score),
   not just by RMS?
2. Which target wins — predicting the **observable masses** directly (A) or going
   through the **profile-parameter** bottleneck (B)?
3. Does modeling the **correlated** scatter (exp07: residual |corr| ~0.5) pay off
   over independent per-aperture noise?

## Method

- **Features:** `M0 + MAH-PCA(4)` — the principled halo representation from exp06
  (top-4 PCs of the M0-normalized log-MAH on a 2.2–9.0 Gyr grid).
- **Targets (scoring space):** 4 aperture/annulus masses `<10, 10-30, 30-50,
  50-100 kpc` from the measured aperture array.
- **Emulator:** 5-fold CV. Linear mean (out-of-fold) + a **per-fold full 4×4
  residual covariance** estimated on training only → a proper Gaussian predictive
  `N(mu_i, Sigma_fold)` for each held-out galaxy.
  - **A (direct):** regress the 4 masses on the features.
  - **B (generative):** regress the 5 `rdm_*` params, reconstruct the CoG
    (R ≥ 5 kpc), read off the same 4 masses; covariance from B's mass-space
    residuals — so both are scored identically.
- **Scores (exp07 suite):** per-target CRPS, predictive log-score, interval
  calibration; the **joint** multivariate log-score with **full vs diagonal**
  covariance; the residual covariance match. Controls: M0-only, shuffled-MAH.

Driver: `run.py` (`EXP08_NMAX=400` for a sub-minute pass). Figures:
`exp08_skill_calibration`, `exp08_covariance`, `exp08_painting`, and
`exp08_param_predictability` (predicted-vs-true and residual-vs-true for each
radial-DiffMAH param — why B fails). Sample: n = 2534 (MAH-covered, finite
targets & params).

## Key results

**1. The conditional-Gaussian emulator works and is calibrated.** Direct
emulator A cuts the CRPS from 0.112 (M0-only) to **0.085 (+24%)**; shuffled-MAH
returns to 0.112 (the assembly signal is real). Its intervals are honest —
empirical coverage 0.54 / 0.72 / 0.91 / 0.95 at nominal 50 / 68 / 90 / 95%
(slightly conservative at low levels, excellent at high). So a plain conditional
Gaussian is already a well-calibrated probabilistic emulator — no GP / flow
needed yet.

**2. Predict the observable directly (A ≫ B).** Going through the profile-param
bottleneck loses most of the signal:

| target | CRPS `<10` | `10-30` | `30-50` | `50-100` | overall |
|---|---|---|---|---|---|
| M0 only | 0.081 | 0.111 | 0.128 | 0.128 | 0.112 |
| **A: apertures** | 0.064 | **0.085** | **0.095** | **0.097** | **0.085** |
| B: rdm params | 0.065 | 0.114 | 0.129 | 0.133 | 0.111 |

B matches A only for the innermost aperture (set by the normalization
`rdm_logMstar0`); for every outer annulus it is **no better than M0 alone**. The
radial-DiffMAH *shape* parameters (`beta_out`, `R_c`, `Delta`) are far less
predictable from halo features than the aperture masses themselves, so the
assembly-driven outer-profile variation does not survive the param route. B is
also mildly **over-confident** (coverage 0.45 / 0.62 / 0.84 / 0.90). The profile
parameters are the right tool for *compression and generating valid profiles*
(exp05), not for *halo → profile prediction*.

**3. Correlated scatter pays off.** The **full** covariance beats the
**diagonal** on the joint log-score by **1.4 nats (A)** and **2.0 nats (B)** —
a large gain from modeling that the aperture residuals move together (adjacent
annuli correlate at ~0.84; mean |off-diagonal| 0.50). The emulator reproduces
this structure out of sample (empirical = model = 0.50). This operationalizes
exp07's covariance finding: the emulator must draw correlated scatter.

**4. Probabilistic painting.** At fixed M0 (13.4–13.6), the emulator paints
**early formers (high z50) above late formers** at every radius, the separation
widening into the outskirts, with calibrated ±1σ bands that bracket the truth —
the conditional Ultimate-SHMR realized.

## Interpretation & caveats

- **Why B fails, visually** (`exp08_param_predictability`): only the
  normalization `rdm_logMstar0` tracks the 1:1 line (R²=0.67); the four shape
  params collapse toward their mean (R² ≤ 0.22, residual-vs-true slope ≈ −1 =
  pure regression to the mean). The radial-DiffMAH shape parameters are an
  excellent *descriptive* basis but a poor *predictive* target: `R_c`/`Delta`
  are partially degenerate, so each galaxy's value carries fitting noise
  uncorrelated with anything physical, burying the (already modest) MAH→shape
  signal. A stable orthogonal basis reflects it better — exp06's CoG-PC1
  (concentration) ↔ MAH-PC2 (timing) at r=0.46, vs the best single radial-DiffMAH
  shape param at r≈0.21 (exp03). So most of the MAH's influence is on profile
  *amplitude/concentration* (captured directly by integrated masses), not on the
  fine shape.
- A vs B is a like-for-like comparison **within the linear-mean class**; a
  nonlinear param predictor might narrow B's gap, but A uses the same class, so
  the verdict (predict the observable directly) holds at this level.
- The predictive covariance is homoscedastic (per-fold, shared). A's calibration
  is already good, so heteroscedasticity is not needed yet; the slight
  conservatism at the 50% interval is the only residual miscalibration.
- MAH-PCA(4) covers z ≈ 2.9 → 0.46 (exp06); the earliest assembly is not
  represented. Profiles are single 2-D projections (irreducible scatter).

## Decision

Adopt the **direct aperture/annulus-mass conditional-Gaussian emulator with full
covariance (A)** as the baseline Ultimate-SHMR emulator: well-calibrated, proper
scores beat M0-only by ~24% CRPS, and it reproduces the joint scatter. The
profile-parameter route (B) is retained for compression / valid-profile
generation only. Next candidates — only if a target metric demands it: a
heteroscedastic (mass-dependent) covariance, and a non-Gaussian predictive
(normalizing flow) if calibration breaks on a harder target or sample.
