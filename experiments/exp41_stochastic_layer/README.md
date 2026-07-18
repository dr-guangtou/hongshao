# exp41 — the stochastic layer (exp32 step 4, resumed)

The kernel is deterministic given the halo: two galaxies with the same
halo features get the same profile. The measured population is not:
the model populations sit 3.5-4.8x the split-half floor on the
centered (shape-only) 2-D observational planes (exp32; the floor = the
energy distance between two random halves of the truth, so 1x floor =
indistinguishable from real), per-quantity predictors are badly
under-dispersed (exp31: the M(<30) vs M(50-100) plane has truth slope
1.88 / scatter 0.206 dex; regression predicts ~1.05/0.05), and exp39
lead 3 measured a named physical axis of the missing diversity (the
per-galaxy core fraction: 1.45 dex logit scatter, best feature
correlation |rho| = 0.31 — per-object information).

The layer: per-galaxy deviations delta-theta around the conditioned
population theta, drawn from a fitted, correlated, LOW-dimensional
distribution living in the anatomy subspace (the directions that carry
per-galaxy individuality), mean-zero so the mean model is untouched.
Fitted against population-distribution statistics, NOT the per-object
loss (which it cannot and must not improve).

Base model: the ADOPTED kernel at its OFFICIAL scope (exp40: 1ch-mof,
z<=1.5 fit, theta = exp40 `outputs/latestart.npz[theta_z15]`).

## Pre-registered criteria (stage 2)

1. Drawn populations reach centered plane energy ~1x the split-half
   floor on the qa tier-2b observational planes, scored HELD-OUT
   (calibrate on one half, score on the other — never calibrate and
   judge on the same split).
2. The mean model is untouched: deviations are mean-zero per
   direction; the differential-deposition and outskirt tests on the
   DRAWN population stay within the adopted kernel's band.
3. No parameter of the deviation distribution at a bound; the layer's
   dimensionality is the smallest that meets criterion 1.

## The staged ladder (checkpoints after stage 0 and before adoption)

**Stage 0 — anatomy on the current kernel + alignment
(`stage0_anatomy.py`).** The exp32 anatomy protocol on the official
theta: per galaxy, free ONE base component at a time (bounded scalar
refit inside the physical box, official z<=1.5 scope, plain loss) and
record the freed loss and the fitted per-galaxy value. Deliverables:
(a) which components carry the individuality (median loss improvement
per component); (b) the per-galaxy delta distributions (spread,
mutual correlations); (c) ALIGNMENT: do the kernel's own leading
deltas correlate with the exp39 per-galaxy core fractions
(percore_gauss_z2.npz) — is the core-diversity signal reachable
inside the kernel's parameter space? (d) the feature-predictable part
(Spearman vs logmh/c200c/fz2/logms): what goes to conditioning vs
what remains stochastic.

**Stage 1 — the deviation distribution.** Joint per-galaxy fits in
the top anatomy directions (2-3 params per galaxy), then fit the
correlated deviation distribution (location-free: mean forced zero;
covariance + tails measured, feature-dependent width if stage 0's
(d) says so).

**Stage 2 — generative draws, judged.** Sample delta-theta per
galaxy, build drawn populations, score the pre-registered criteria
(centered plane energy vs floor held-out, physics band, bounds).

**Stage 3 — graduation decision (user).**

Run: `PYTHONPATH=. uv run python
experiments/exp41_stochastic_layer/stage0_anatomy.py {demo|run|report}
[--dev]`. Always `export
HONGSHAO_DATA_DIR=/Users/shuang/Desktop/tng300_mah_mprof`.

## Results

(stage 0 pending)
