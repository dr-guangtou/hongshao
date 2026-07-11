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

## Remaining (per todo)
(iii) feature increments (burstiness, real-MAH t50/fz2 vs DiffMAH, acc_rate,
shuffle controls); (iv) physical-vs-statistical CoG head-to-head at z=0.4;
(v) verdict + what graduates; (vi) higher-z single-epoch fits and their
relation to the z=0.4 model (the alternative path to a multi-epoch model).

## Files
- `run.py` — modes 1/2/3 OOF + generative plane tests (`demo`: synthetic
  under-dispersed-mean vs restored-draws check).
