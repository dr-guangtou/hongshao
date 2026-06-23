# exp22 — Predict the whole curve of growth, not just a few aperture masses

## Question
The graduated emulator predicts four aperture/annulus masses. Can we instead
predict the **entire profile** (the curve of growth, CoG), and how much of it is
halo-predictable — in particular, does the halo predict the profile *shape*
beyond just the total mass (the SHMR)?

## Method
- **Compress** each CoG to `[logMtot, PC1, PC2, PC3]`: total mass (anchor) +
  3 shape modes (exp02's CoG-shape PCA; 3 modes reconstruct the shape to
  ~0.005 dex). **PCA is fit inside each CV fold** (train only) — no leakage.
- **Predict** that 4-vector from portable `DiffMAH + c_200c` with the same
  probabilistic emulator (linear mean + heteroscedastic full covariance, exp19;
  local copy, `hongshao/emulator.py` untouched).
- **Reconstruct analytically.** The CoG is *linear* in the compressed vector
  (`CoG(R) = logMtot + mean_shape(R) + Σ_k PC_k·mode_k(R)`), so the predictive
  Gaussian on `[logMtot, PCs]` propagates to a **per-radius Gaussian CoG** with
  no sampling: mean `a_R·μ_θ + mean_shape(R)`, variance `a_Rᵀ Σ_θ a_R`.
- **Evaluate in profile space:** per-radius CRPS / calibration / reconstruction
  RMS, vs a baseline = "total mass + mean shape" (predict `logMtot`, assume the
  population-average shape with its full scatter). The baseline→full gap isolates
  the value of predicting *shape* from the halo.

## Key result
**The whole profile is predictable and well-calibrated; predicting the shape
from the halo adds +10% per-radius CRPS, almost all of it the concentration
mode (PC1, via `c_200c`). Beyond total mass + concentration, the profile shape
is largely intrinsic.** (n=2539, K=3)

| | mean per-radius CRPS | recon RMS | coverage 50/68/90/95 |
|---|---|---|---|
| mass + mean shape | 0.0715 | — | — |
| **full (halo-predicted shape)** | **0.0643** (+10.0%) | 0.116 dex | 0.51/0.69/0.90/0.95 |

- **Reconstruction RMS 0.116 dex** is dominated by the *total-mass* SHMR scatter
  (~0.13 dex), not the shape — the CoG is anchored by `logMtot`. The model is
  well-calibrated across all 23 radii.
- **Predicting shape buys +10% CRPS overall, peaking ~17% at intermediate radii
  (~10–20 kpc)** — exactly where the concentration mode redistributes mass
  between the inner and outer profile. The gain shrinks at the center and the
  outskirts (where the CoG is pinned by total mass).
- **Per-mode halo-predictability:** PC1 (concentration) **R²=0.39**, PC2 0.33,
  PC3 0.05. The dominant shape mode is moderately predictable — naturally,
  since PC1 ≈ profile concentration and `c_200c` ≈ halo concentration. The
  higher modes are essentially intrinsic.

## Derived-aperture bias check (does the reconstruction bias the graduated targets?)
Deriving the four graduated targets (`<10`, `10–30`, `30–50`, `50–100` kpc) from
the reconstructed CoG and comparing to truth (CoG-derived apertures match the
real `logmstar_aper` to **0.003 dex**, so the derivation is sound):

| aperture | repr bias | pred bias | reg-to-mean slope (profile) | (direct) | −(1−R²) |
|---|---|---|---|---|---|
| <10 | +0.000 | +0.000 | −0.289 | −0.291 | −0.292 |
| 10–30 | −0.000 | +0.018 | −0.281 | −0.262 | −0.264 |
| 30–50 | −0.002 | +0.020 | −0.225 | −0.210 | −0.206 |
| 50–100 | +0.002 | +0.022 | −0.180 | −0.174 | −0.170 |

- **The mass-dependent tilt is regression to the mean, not a reconstruction
  defect.** The residual-vs-true slope ≈ −(1−R²) (exp15), and is **steepest in
  the core (−0.29)** — exactly the inner-region deviation visible for the extreme
  example galaxies in `exp22_full_profile.png` Panel B. The core regresses most
  because its mass is the *least* tightly halo-determined relative to its
  variance (R²≈0.71 vs 0.83 in the outskirts). It is **identical to the direct
  (graduated) emulator** (slopes match to ~0.02), so the profile route adds no
  bias of its own here.
- **The PCA compression is unbiased.** Reconstructing from the *true* compressed
  vector gives ≤0.002 dex aperture bias at every radius — the K=3 representation
  does not distort the derived masses.
- **One real, small artifact of the profile route:** a **+0.02 dex offset in the
  *annuli*** (not the `<10` cumulative), from differencing a predicted
  *cumulative* log-profile (a nonlinear `10^outer − 10^inner` step). The direct
  emulator, which predicts each annulus directly, avoids it. So: if annular /
  outskirt masses are the quantity of interest, predict them directly (the
  graduated emulator); the profile route is for the cumulative CoG.
- **Cure for the tilt:** it is not fixable in the mean — use the model
  generatively (sample the predictive CoG), per exp15. The sampled population is
  unbiased; the point estimate is a conditional mean and *must* regress.

## Generative demonstration (why the bias is not a problem)
`exp22_generative.png` (the 50–100 kpc outskirt) shows the three facts together:
- **A — bin by TRUE:** the residual-vs-true slope follows −(1−R²). This appears
  for *any* predictor with scatter (you are conditioning on the noisy truth) — it
  is regression to the mean, not a model defect.
- **B — bin by PREDICTED:** `E[true | predicted]` sits on the 1:1 line — the mean
  is **unbiased in feature space**, so there is nothing to "correct".
- **C — population:** the conditional mean is **under-dispersed** (std 0.37 vs the
  true 0.41); drawing one sample per galaxy from the predictive
  `N(mean, σ(X))` **restores the population exactly** (sampled std 0.41, matching
  tails). For any population-level use (SMF, lensing-selected stacks) the
  generative model is unbiased.
- **Caveat surfaced here:** sample the **direct** outskirt predictor, not a
  differenced cumulative profile — sampling the CoG and forming `10^M_out −
  10^M_in` over-disperses the annulus (std 0.80 vs 0.41), the same nonlinearity
  behind the +0.02 dex annulus offset above. Cumulative quantities sample
  cleanly; annuli should be predicted (and sampled) directly.

## Decision
- **The PCA-route full-profile emulator works** — a viable richer target than a
  few apertures, well-calibrated, with analytic per-radius uncertainties. Keep
  it as an alternative output of the SHMR (the aperture emulator stays the
  default for directly-observable masses).
- **Most of the profile = total mass + a population-average shape;** the halo
  refines the shape only modestly (+10%), almost entirely through *concentration*
  (PC1 via `c_200c`). This is consistent with exp02 (shape modes carry only a
  modest, ≤0.28 assembly partial-correlation) and exp13/exp21 (the residual is
  largely intrinsic) — there is no large hidden halo-predictable signal in the
  profile shape.
- Independent experiment; `hongshao/emulator.py` and `forward.py` unchanged.
- Next options: the **parametric (radial-DiffMAH `rdm_*`, exp03) route** as a
  physical-parameter alternative to PCA; predicting the profile in **Re units**
  (exp21); or feeding the reconstructed CoG's per-radius uncertainty into the
  forward model.
