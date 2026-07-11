# exp33 — single-epoch (z=0.4) consolidation under the standardized QA

> **RESULT (2026-07-11, steps i–ii).** The graduated stack's frozen spec
> reproduces its record numbers on the current sample (mode-1 CRPS 0.0832 =
> exp19/20), and — the test it had never faced — **its generative `sample()`
> draws PASS the 2-D observational-plane test in the emulator's native target
> space**: energy/floor 0.8–1.0 (statistically indistinguishable from the real
> population) where the mean prediction sits at 2.7–4.0. The only failure is
> the profile mode's PCA-3 compression in Re coordinates. The multi-epoch
> stochastic layer therefore has a PROVEN template; the exp32 plane failure
> was the missing generative layer, not a defect of conditional-Gaussian
> modeling.

## Setup
n=2539 (`use` + finite CoG/features), portable features [DiffMAH(4), c_200c],
5-fold out-of-fold everywhere; N=10 sampled populations per generative test;
plane metric = 2-D energy distance / truth split-half floor (full | centered).

## Results

| mode | point prediction | plane: mean | plane: `sample()` draws |
|---|---|---|---|
| 1 kpc masses (4 targets) | CRPS 0.0832; 0.116–0.172 dex | 3.8 \| 4.0 | **0.9 \| 0.9** |
| 2 Re masses (6 targets, n=2056) | CRPS 0.0703; 0.106–0.138 dex | 2.8 \| 2.9 | **0.8 \| 0.8** |
| 3 CoG (PCA-3), kpc planes | apertures 0.100–0.116 dex; max\|rel\| 34.1% all-R / 24.7% R>5 kpc | 3.0–3.3 | **0.7–1.0** |
| 3 CoG, Re plane M(<2Re) vs M(2–4Re) | — | 2.7 \| 2.6 | **2.8–3.0 (FAILS)** |

1. **The generative layer works where the emulator predicts the plane's
   quantities directly** (modes 1/2, and mode 3 evaluated in kpc bins): full
   covariance + heteroscedastic sigma restore the population to within
   sampling noise of the truth. These are the first models in the project to
   pass the plane test.
2. **The failure is localized to the PCA-3 profile compression in Re
   coordinates**: mode-3 draws do not improve the Re-relative plane (its mean
   is already mildly OVER-dispersed there, truth scatter 0.072 -> model 0.100,
   slope 1.12 -> 1.01) — the 3-mode shape basis + reconstruction error
   misplaces mass relative to each galaxy's own size. Candidate fixes for the
   consolidation: more/Re-native modes, or predict Re-binned targets directly
   (mode 2 passes) and reconstruct.
3. **Single-epoch context for the multi-epoch program**: the z=0.4 statistical
   CoG emulator reaches 24.7% (R>5 kpc) worst-radius error vs ~31% for the
   multi-epoch transport model evaluated at z=0.4 — the price of one
   consistent multi-epoch history at a single epoch is ~6 points. And the
   exp32 conclusion "no model reproduces the planes" is now sharpened: no
   DETERMINISTIC model does; the graduated generative machinery does, and
   porting it across epochs (with the anatomy subspace as the correlated
   dimensions) is the concrete path.

## Mean-CoG recovery decomposed (user question on the bins figure)
The apparent systematic failure in truth-M* terciles (low bin over-predicted at
all radii, high bin under-predicted in the outskirts) is ~100% AMPLITUDE
regression-to-the-mean, not shape error: total-mass residual +0.044/-0.021/
-0.035 dex by truth tercile, while the amplitude-pinned shape residual is
<=0.013 dex at 8 kpc and <=0.003 dex beyond 45 kpc in every bin; binned by
HALO mass (what the model knows) the bias is ~0. The 0.099 dex amplitude
scatter is the SHMR information limit at fixed [DiffMAH+c200c] (exp13/16), and
truth-binning converts it into apparent systematics — unavoidable for ANY
conditional mean. The bins figure now shows raw AND amplitude-pinned residuals
so the two cannot be conflated again.

## Step iii — feature increments (`features.py`, n=2484, mode-1 targets)
| feature set | CRPS | vs baseline |
|---|---|---|
| baseline [DiffMAH(4), c200c] | 0.0824 | — |
| + burst (real-MAH burstiness) | 0.0813 | +1.3% (shuffle: 0.0%) |
| + t50 + fz2 (real-MAH summaries) | 0.0819 | +0.5% |
| + acc_rate | 0.0821 | +0.3% |
| + all four | 0.0806 | +2.1% (shuffle: 0.0%) |
| REPLACE shape params: [logmh, t50, fz2, c200c] | 0.0890 | **−8.0%** |

**Verdict: the frozen feature set stands.** (1) Burstiness carries a real
(shuffle-controlled) but small +1.3% signal — the first detection of merger
content in mass prediction (exp30's residual test saw none) — yet it requires
the raw MAH, so adopting it would break the portable/differentiable
DiffMAH-input configuration for a ~1% gain: not worth it. (2) The alternative
raw-summary parameterization is 8% WORSE — the smooth DiffMAH fit is a
better-conditioned encoding of the MAH than [logmh, t50, fz2], confirming
exp10 from the opposite direction. The exp29 "smooth curve flatters" concern
does not apply to the statistical single-epoch emulator.

## Step iv — physical vs statistical CoG head-to-head (`head2head.py`, common n=2397)
Held-out z=0.4 CoGs: mode-3 statistical (all freedom at one epoch, PCA-3 on 5
numbers) vs the exp32 transport theta(logMh) CV prediction (one history through
five epochs, full DiffMAH deposit input). Median max|rel|:

| | R>5 kpc | all R |
|---|---|---|
| statistical e2e / shape | 24.4% / 15.6% | 33.6% / 28.0% |
| physical e2e / shape | 29.7% / 19.1% | 39.2% / 31.1% |

1. **The multi-epoch consistency tax at a single epoch is ~3–5 points**
   (paired median +3.1; the statistical model is better for only 63% of
   galaxies) — the physical model is competitive despite fitting five epochs
   with one history and just 14 population parameters.
2. **THE finding: both model classes fail on the SAME galaxies** — shape
   residual correlation rho = +0.89 / +0.87 / +0.82 at 5/30/100 kpc, +0.72 in
   amplitude, +0.60 on per-galaxy max|rel|. Two maximally different
   architectures (empirical PCA-Gaussian vs physical transport kernel) hit one
   shared information wall: the residual is information MISSING from the halo
   (intrinsic/baryonic stochasticity + the single-projection noise flagged in
   AGENTS), not a model-class deficiency. Corollaries: (a) hybrid/ensemble
   headroom is minimal (errors correlated); (b) further mean-model sophistication
   at fixed inputs is not the path — the generative layer over this wall is;
   (c) exp13's "remaining scatter is intrinsic" now has a two-independent-model
   proof.
3. Both error profiles are radius-identical in form (26% at 2 kpc falling to
   ~1% near the pin), statistical uniformly ~5–10% lower; the gap grows toward
   high halo mass (panel C) — the massive end is where one consistent history
   costs most, echoing exp32.

## Representation repairs (`repr_fix.py`): a clean NEGATIVE — the wall holds
Same folds/inputs/metrics, only the target representation changed (OOF mean,
shape pinned; Re plane = the known defect):

| architecture | max\|rel\| R>5 / all R | Re-plane scatter / E centered |
|---|---|---|
| A baseline (kpc PCA-3) | 15.7% / 28.1% | 0.072→0.100 / 2.6 |
| B size-aware (R/R50 basis + predicted R50) | 18.0% / 39.6% | 0.072→0.117 / 2.6 |
| C core-split (M(<5kpc) + R>5 basis) | 15.7% / 28.2% | 0.072→0.100 / 2.6 |
| D density-integrated | 21.1% / 51.4% | 0.072→0.096 / 2.8 |

- **No representation beats the baseline** — exactly what the rho=0.87
  head-to-head predicted: the residual is missing information, so re-encoding
  the same information cannot help. B fails BECAUSE the predicted R50 carries
  the same ~0.04+ dex unpredictability (reconstructing at a wrong size is a
  new error source, and the Re plane is measured against the TRUE Re — it is
  not fixable by coordinates alone). C is a wash (the core neither helped nor
  contaminated). D adds a structural handicap discovered here: the
  shell-integration CoG path is biased ~0.11 dex at the steep inner edge and
  accumulates ~0.05 dex outward (exp22's "stable" meant monotonic, not exact)
  — mode 4's higher density-space predictability cannot survive the transform.
- The Re-plane defect is therefore ALSO an information problem (each galaxy's
  size at fixed halo), not a basis problem: closing it needs either new
  size-carrying inputs (halo spin — direction 1) or the generative layer in
  Re-native targets (mode 2 already passes its plane).

## Remaining (per todo)
(v) verdict + what graduates — the mean-model consolidation is now COMPLETE
(features at limit, representations at limit, wall proven twice); (vi)
higher-z single-epoch fits and their relation to the z=0.4 model.

## Files
- `run.py` — modes 1/2/3 OOF + generative plane tests (`demo`: synthetic
  under-dispersed-mean vs restored-draws check).
