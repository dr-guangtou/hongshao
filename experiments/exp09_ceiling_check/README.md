# exp09 — Ceiling check: is a richer-than-linear model worth pursuing?

Before committing the Ultimate-SHMR emulator to a model family, measure how much
predictive signal a *flexible* model extracts from the same features the linear
model uses. The goal is a closed-form, observation-constrainable equation, so the
question is narrow: **does the linear form leave accuracy on the table?**

## Question

For `M*(annulus) = f(M0, MAH)`, does a flexible nonlinear model (gradient-boosted
trees) beat the linear model — i.e. is there interaction/curvature structure that
a richer analytic form (e.g. found by symbolic regression) should capture?

## Method

- Targets: the 4 aperture/annulus masses (`<10, 10-30, 30-50, 50-100 kpc`).
- Features: `M0 + MAH-PCA(4)` (same as exp08).
- Models, 5-fold CV, scored by CRPS (exp07 suite) with a homoscedastic-Gaussian
  predictive:
  - **linear (M0)** — reference.
  - **linear (M0+MAH)** — the current closed-form model.
  - **poly-2 (M0+MAH)** — richer *analytic* form (all degree-2 interactions +
    curvature; standardized).
  - **GBM (M0+MAH)** — `HistGradientBoostingRegressor`, the flexible
    achievability ceiling.
  - **GBM (shuffled MAH)** — control; must collapse to the M0-only level.

Driver: `run.py` (`EXP09_NMAX=400` for a sub-minute pass). Figure:
`exp09_ceiling_check`. Sample: n = 2534.

## Key result

**The predictable relation is essentially linear — the flexible ceiling adds
nothing.**

| model | overall CRPS [dex] | vs linear(M0+MAH) |
|---|---|---|
| linear (M0) | 0.1118 | −31% |
| **linear (M0+MAH)** | **0.0851** | — |
| poly-2 (M0+MAH) | 0.0822 | **+3.3%** |
| GBM ceiling (M0+MAH) | 0.0855 | **−0.6%** |
| GBM (shuffled MAH) | 0.1141 | −34% (control ✓) |

- The GBM is **not** beating linear (−0.6%, within noise), and the per-aperture
  CRPS curves of linear / poly-2 / GBM are visually identical.
- The GBM is working correctly, not broken or under-powered: it uses the MAH
  (its real-MAH score matches linear) and the **shuffled-MAH GBM collapses back
  to the M0-only level** (0.114) without over-fitting. So the flat result is
  genuine — there is simply no nonlinear structure to exploit.
- The richer *analytic* form (poly-2) buys a marginal **+3.3%** — a whiff of
  low-order curvature/interaction, but small and not corroborated by the GBM.

## Interpretation & caveats

- At the current signal-to-noise, `M*(annulus | M0, MAH-PCA)` is **linear in
  these features**. The accuracy ceiling is ~0.085 dex CRPS, and a fancier
  functional form will not move it. What remains is **scatter**, not a missing
  mean-shape term — the modeling effort belongs there, not in the functional form.
- "Linear in these features" is the honest scope: a different MAH
  parameterization (e.g. DiffMAH params) could expose mild nonlinearity, but it
  cannot raise the achievable ceiling measured here.
- The marginal poly-2 gain (+3.3%) is available cheaply and analytically if ever
  wanted, but is not worth the added terms now.

## Decision

**Keep the linear closed-form equation.** A flexible/nonlinear or
symbolic-regression model is not justified by accuracy — linear is already at the
ceiling. Direct the next effort to the two things that *do* matter:

1. **Portable MAH features** — fit DiffMAH to our own MAH curves so the model
   takes intrinsic per-halo parameters (applicable to N-body / observation),
   replacing the TNG-sample-defined MAH-PCA basis; verify it matches MAH-PCA(4).
2. **Scatter / probabilistic modeling** — the residual covariance (exp08) is the
   dominant remaining term; refine it (heteroscedasticity, non-Gaussianity) under
   the exp07 suite.
