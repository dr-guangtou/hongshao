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
