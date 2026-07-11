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

## Step vi — the epoch connection (`epochs.py`, n=2397)
Five independent single-epoch emulators (shared [DiffMAH(4)+c200c], same
folds), z=0.4..2.0:
1. **Performance vs z**: CRPS 0.081 -> 0.200, CoG shape max|rel| 24.3% ->
   33.6% — the independent-single-epoch ceiling per epoch. The multi-epoch
   transport (~29-30% every epoch) sits close to this ceiling at z>=1 and
   pays its consistency tax mostly at low z.
2. **The residual process is MARKOVIAN in epoch.** Cross-epoch OOF residual
   correlation: adjacent epochs 0.64-0.70, decaying with separation almost
   exactly as AR(1) with rho=0.67: predicted 0.67/0.45/0.30/0.20 vs observed
   0.64-0.70/0.43-0.50/0.27-0.34/0.17. A per-galaxy latent persists but
   decorrelates over ~2 epoch steps — the multi-epoch stochastic layer should
   draw an AR(1)-in-epoch latent, NOT a single static per-galaxy offset.
3. **The closure test PASSES**: hold an epoch out entirely, interpolate the
   other epochs' fitted coefficients quadratically in z, predict the held
   epoch: CRPS within -3.2%/+0.7%/+3.9% of the direct fit at z=0.7/1.0/1.5
   (and 12-22% better than transplanting the nearest epoch's model). The
   coefficients evolve smoothly (panel C) — mostly a slow growth of the
   logmp weight and decline of `early`.

**The multi-epoch blueprint that falls out**: continuous-z mean model by
coefficient interpolation + AR(1)-in-epoch correlated latent + generative
sampling (which passes the 2-D planes). Every ingredient now has a measured
justification. This is the alternative architecture to the one-consistent-
history transport kernel — smoother, fully statistical, differentiable; what
it gives up is physical interpretability and mass conservation.

## The completed 2x2 (`transport_z04.py`): the head-to-head gap decomposed
Transport kernel fitted to z=0.4 ONLY (same population forms, 10-fold CV;
z=0.4 shape max|rel| R>5 kpc, held-out):

| | single-epoch fit | multi-epoch fit |
|---|---|---|
| statistical (PCA-Gaussian, ~90 params) | 15.6% | — |
| transport theta(logMh) (14 params) | **16.1%** | 19.1% |
| transport global (7 params) | 19.0% | ~19% |

**Decomposition of the head-to-head gap: form costs ~0.5 points, consistency
costs ~3.0.** The mass-conditioned transport kernel fitted to one epoch
matches the statistical emulator to within 0.5 points using 6x fewer
parameters — the kernel FORM is essentially as expressive as PCA-Gaussian at
a single epoch, and physics compresses extremely well. The multi-epoch
constraint is the real (and known) price, bought back as cross-epoch
structure, mass conservation, and interpretability. Note mass-conditioning
matters far more for the single-epoch kernel (19.0 -> 16.1) than it did for
the multi-epoch fit (exp32: ~1 point): with five epochs, the epoch lever arm
partially substitutes for the mass lever. The z=0.4-only theta also lands in
a different parameter regime (b_late 4.1 vs 6.5) — the theta components are
epoch-degenerate, consistent with the exp32 anatomy.

## The aperture-horizon degeneracy + the physical refit (`physical_theta.py`)
Discovery (user-prompted): the population transport fits delete whole epochs
GEOMETRICALLY — z=1-2 deposits carrying 14% of the star budget are placed at
widths ~550 kpc (4% visible inside 150 kpc), observationally identical to zero
efficiency because each epoch is renormalized to the 150-kpc mass. Physically
absurd; performance numbers unaffected (the basins are observationally
equivalent); all population-theta VALUES uninterpretable. The raw drop has
NOTHING beyond ~160 kpc (audited), so the fix is prior-side now, data-side
next (asymptotic total via CoG extrapolation — user prefers methodology-
consistent totals over SubhaloMassType).

Physical 5-param refit (alpha=1 fixed, lognormal f(z), widths boxed to the
per-galaxy basin; held-out R>5 kpc):

| | unconstrained 7p | physical 5p |
|---|---|---|
| z=0.4-only global | 19.0% | 22.4% |
| z=0.4-only +logMh slope | 16.1% | 20.5% |
| multi-epoch global (epoch-avg) | 18.7% | 21.7% |

1. **Physicality restored at a +3-4.4 point price**: every z-bin now
   contributes visibly (0.49-1.00), the efficiency is a broad peak (z~4.2
   single-epoch, z~6.8 multi-epoch), no railing z_c/b_early.
2. **Epoch stability improved but incomplete**: both fits land in the same
   physical basin (log_s0 2.35 vs 2.20, g both at bound) yet mu and q still
   differ — and **g rails at its 2.5 bound in every fit**: the data actively
   want part of the geometric channel.
3. **The synthesis: some of the "deleted" mass is probably REAL.** Massive
   galaxies genuinely deposit accreted stars beyond 150 kpc (ICL); the
   unconstrained basin exaggerates a true effect that the physical box now
   forbids entirely — hence the accuracy price and the railing. Neither basin
   is right: the total-normalized refit (aperture fraction as a fitted datum,
   from the extrapolated asymptotic CoG) is not just hygiene, it measures the
   real beyond-aperture mass budget. That is the decisive next experiment for
   the transport family.

## Remaining (per todo)
(v) verdict + what graduates — all evidence is in; write the consolidation
verdict and decide the graduation set. Transport-side: the total-normalized
refit (extrapolated asymptotic totals) supersedes further basin arguments.

## Files
- `run.py` — modes 1/2/3 OOF + generative plane tests (`demo`: synthetic
  under-dispersed-mean vs restored-draws check).
