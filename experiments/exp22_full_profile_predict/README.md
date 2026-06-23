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

## Density profile vs CoG (Q2 — `density_profile.py`)
The CoG `M(<R)` is an integral, so it is smooth and its large-R value is just the
total mass. The 1-D **density** profile `Σ(R) = dM/dA` (here derived by
differencing the noiseless-sim CoG) keeps local structure. PCA-ing the density
shape vs the CoG shape on the *same* galaxies:

| representation | value of predicting shape | PC1 R² | PC2 R² | recon (own units) |
|---|---|---|---|---|
| CoG `M(<R)` | +10.0% | 0.39 | 0.33 | 0.118 dex |
| **density `Σ(R)`** | **+15.7%** | **0.54** | 0.13 | 0.166 dex |

- **The density profile is more halo-predictable.** Predicting its shape buys
  +15.7% (vs +10.0% for the CoG), and its dominant mode is much more
  halo-tied (**PC1 R²=0.54 vs 0.39**). The density concentrates the predictable
  signal in one mode (the inner↔outer concentration contrast, naturally linked to
  `c_200c`); the CoG smears it across PC1+PC2.
- **The big difference is in the outskirts.** Per-radius (figure Panel B), the
  CoG's shape value falls to ~0 beyond ~50 kpc — the cumulative mass there is
  pinned to the total, so it carries no extra shape info. The **density keeps
  rising to ~30%** at large R: the *local outskirt density* is meaningfully
  halo-predictable in a way the enclosed mass completely hides. For
  envelope/ICL/outskirt science, model the **density** profile, not the CoG.
- **Caveat:** the density's absolute per-radius CRPS is larger (it is a
  derivative — more dynamic range / noisier), so only the *relative* gain and the
  per-mode R² are comparable across the two. Deriving `Σ` by differencing only
  works because the simulation is noiseless; on real, noisy outskirt data this
  differencing would amplify noise (exactly the user's premise that the sim
  profiles are reliable).

### Is the density predictor "better" than the graduated emulator? Depends on the target.
Scored on the **graduated emulator's own metric** (the 4 aperture/annulus masses),
RMS [dex]. The density route integrates **outward from R=0**, starting from a
predicted central mass `M(<2)` — the resolution-limited center is still real and
counts toward the larger apertures:

| aperture | direct emulator | from CoG | from density |
|---|---|---|---|
| <10 | 0.116 | 0.116 | 0.117 |
| 10–30 | 0.155 | 0.156 | 0.155 |
| 30–50 | 0.171 | 0.169 | 0.171 |
| 50–100 | 0.172 | 0.170 | 0.172 |

- **On aperture masses all three routes are equivalent** (once the central mass
  is counted) — aperture masses are **total-mass-dominated**, so the density's
  shape advantage neither helps nor hurts here. The density predictor is *not*
  better on apertures, but it is not worse either.
- Only the direct emulator gives clean, *calibrated* annulus **uncertainties** —
  the Q1 differencing instability bites when you *sample* an annulus from a
  cumulative profile (not for the mean), so for annulus error bars predict the
  annulus directly.
- **The density predictor wins only on its own product** — the density profile,
  where its shape is markedly more halo-predictable (PC1 R² 0.54 vs 0.39) and it
  uniquely captures outskirt-density structure. The two are **complementary**:
  graduated emulator → clean aperture masses + uncertainties; density predictor →
  the full profile and the outskirt density. "Better" is target-dependent.
- **Integration direction matters (the mirror of Q1):** build the cumulative by
  integrating the density *outward from the center* (stable); both the inward
  integration from the total and the outward-from-2-kpc-with-no-center are
  unstable/biased for the small inner cumulative — the same near-cancellation as
  differencing.

### Aside — does DiffMAH suffer the same issue? (`diffmah_rate_check.py`)
DiffMAH fits the **cumulative** peak-mass history `log M(t)`, so getting the
accretion rate means differentiating it — the same operation that blew up the
annuli. **It does not blow up**, because DiffMAH is a smooth 4-parameter analytic
fit: the noise is removed at the *fit* step, so its derivative is clean
(`exp22_diffmah_rate.png`, Panel B). Finite-differencing the *raw* peak-mass
history is jumpy and plunges to zero whenever growth is flat — the exact
near-cancellation of Q1. The unifying principle: **model whichever quantity is
reliable, get the other by the stable operation.** Stellar profile: the density
is reliable → predict `Σ`, *integrate* to the CoG. Halo MAH: the cumulative is
reliable (the rate is merger-noisy) → fit `M(t)`, *differentiate the smooth
model*. DiffMAH's `late` parameter already is a denoised recent-accretion rate,
so we never differentiate raw data.

**Refinement — is the *recent* accretion poorly constrained? (`diffmah_recent_uncertainty.py`).**
Yes, and it is the **time-domain twin of the galaxy outskirt**: the recently
accreted mass `M(t0)−M(t0−Δt)` is a small increment on a large cumulative — the
same near-cancellation. Under a fixed 0.02-dex cumulative perturbation, the rate
slopes (`early`/`late`) bootstrap **~4× softer than the final mass `logmp`**, and
the fractional uncertainty of `dM/dt` rises **~3× from early to recent times**.
So fitting the cumulative does leave the recent rate the least-constrained piece.
Two honest caveats: (1) the effect is *moderate* (3–4×, not the ~10× of the
stellar annulus) because these halos are still accreting at z=0.4 — the recent
rate is not as suppressed as the outskirt stellar density; (2) **this softening
does not limit the outskirt prediction** — exp13 showed the raw MAH (which keeps
the full recent history) does *not* beat DiffMAH for `M*[50–100]`, so the outskirt
residual is intrinsic, and DiffMAH's `late` already carries the part that matters
(it is the outskirt *scatter* driver, exp19). The user's intuition is right about
the mechanism; the data says it is not the bottleneck.

## How many PCs? Does >3 help the predictor? (`pca_n_components.py`)
**No — the predictor plateaus by K≈3, with direct evidence.** Sweeping K=1..8
separates two different curves:

| K | CoG recon RMS | CoG value% | R²(Kth PC) | density recon RMS | density value% | R²(Kth PC) |
|---|---|---|---|---|---|---|
| 1 | 0.027 | 9.4 | 0.39 | 0.094 | 13.7 | 0.54 |
| 2 | 0.010 | 10.0 | 0.33 | 0.063 | 15.6 | 0.13 |
| **3** | 0.005 | **10.0** | 0.05 | 0.049 | **15.7** | 0.06 |
| 4 | 0.003 | 10.0 | 0.02 | 0.039 | 15.7 | 0.04 |
| 6 | 0.0015 | 10.0 | −0.00 | 0.028 | 15.8 | 0.02 |
| 8 | 0.0009 | 10.0 | 0.00 | 0.021 | 15.8 | 0.00 |

- **Compression keeps improving with K** (recon RMS falls monotonically — more
  modes describe the profile better, as exp02 found). But the **predictor's
  value-of-shape and CRPS flatten by K≈2–3.**
- The reason is in the last column: **PC4 and beyond have R²≈0** — they are not
  halo-predictable. Adding them to the target just means predicting their
  population mean (≈0), which contributes nothing. All the halo-predictable
  profile-shape information lives in **2–3 modes** (concentration + one or two
  redistribution modes); the rest is galaxy-by-galaxy noise the halo cannot
  reach. K=3 is the right default.

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
- **Prefer the density profile `Σ(R)` over the CoG for shape modeling** (Q2): it
  is more halo-predictable (PC1 R² 0.54 vs 0.39) and, unlike the cumulative CoG,
  keeps real halo-predictable structure in the outskirts (~30% shape value at
  large R vs ~0 for the CoG). Use the CoG only when the cumulative mass itself is
  the observable.
- Next options: the **parametric (radial-DiffMAH `rdm_*`, exp03) route** as a
  physical-parameter alternative to PCA; the density profile in **Re units**
  (exp21); or feeding the predictive profile uncertainty into the forward model.
