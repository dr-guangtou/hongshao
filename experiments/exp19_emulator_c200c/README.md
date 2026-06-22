# exp19 — Fold c_200c into the heteroscedastic emulator (the working model)

## Question
exp14 is the working Ultimate-SHMR emulator (per-aperture linear mean +
heteroscedastic full residual covariance, on portable DiffMAH features).
exp16–18 showed `c_200c` adds real skill to the *mean*. Fold it into the full
emulator: what is the total gain, and does `c_200c` also help the *scatter*
(enter the log-variance / improve conditional calibration), or only the mean?

## Method
5-fold CV decomposition (exp07 suite: marginal CRPS, joint NLL, marginal +
conditional calibration), reusing the exp14 machinery:
- **A** mean=DiffMAH, scatter=hetero(DiffMAH) — exp14 baseline
- **B** mean=DiffMAH+c200c, scatter=hetero(DiffMAH) — `c_200c` in the mean only
- **C** mean=DiffMAH+c200c, scatter=hetero(DiffMAH+c200c) — `c_200c` in mean + scatter

A→B isolates the mean effect, B→C the scatter effect. (Homoscedastic references
included.)

## Key result
**`c_200c` improves the mean (+4.7% CRPS, +0.08 nats) but not the scatter; the
scatter is still driven by `late`. Conditional calibration stays excellent.**

| emulator (n=2539) | CRPS | joint NLL | cond. cov. gap |
|---|---|---|---|
| A. DiffMAH \| het(DiffMAH) (exp14) | 0.0873 | −3.349 | 0.018 |
| B. DiffMAH+c200c \| het(DiffMAH) | 0.0832 | −3.431 | 0.014 |
| **C. DiffMAH+c200c \| het(DiffMAH+c200c)** | **0.0832** | **−3.432** | **0.010** |

- **Mean effect (A→B):** CRPS 0.0873 → 0.0832 (**+4.7%**), joint NLL −3.349 →
  −3.431 (**+0.082 nats** — each halo's profile ~9% more probable). Marginal
  calibration unchanged (still on the 1:1 line).
- **Scatter effect (B→C): negligible.** Adding `c_200c` to the log-variance model
  changes the joint NLL by +0.001 nats and only nudges the conditional-coverage
  gap (0.014 → 0.010, already excellent). The log-σ slopes confirm it: **`late`
  remains the scatter driver** (+0.16 to +0.22 per σ in the outer annuli), while
  `c_200c`'s slopes are small (~+0.03 to +0.06) and not sign-stable across
  subsamples. Concentration sets *where the mean sits*, not *how noisy* a halo is.
- The two physical axes stay distinct: **`c_200c` → mean, `late` → scatter** (and
  `late` also carried the mean's `late²` curvature, exp12). Recent accretion
  destabilizes the outskirts; concentration shifts the central galaxy's mass.

## Decision
- **Working emulator = linear mean on `DiffMAH(4) + c_200c` + heteroscedastic full
  covariance.** The scatter model can keep its DiffMAH features (adding `c_200c`
  to the variance is harmless but unnecessary, +0.001 nats); for a clean default,
  drive the variance with the same feature vector as the mean — it costs nothing.
- Total gain over the exp14 (DiffMAH-only) emulator: **+4.7% CRPS, +0.08 nats**,
  conditional gap 0.018 → 0.010, marginal calibration unchanged.
- This is the model to **graduate into `hongshao/`** (default mean = linear
  DiffMAH+c_200c; optional 7-term degree-2 poly mean per exp17; heteroscedastic
  covariance; a `sample()` path per exp15).
