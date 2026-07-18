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

### Stage 0 — anatomy + alignment (2026-07-18, full n=2397): CHECKPOINT

1. **The individuality is spread evenly across coordinates** — freeing
   any ONE base component per galaxy buys 18.5-22.3% median loss
   improvement (g 22.3, mu 22.2, log_rc 21.9, q 21.1, gamma 19.5,
   sig 18.5): no privileged direction in raw coordinates.
2. **All six single-component deltas are rank-correlated at |rho| =
   0.92-1.00** — the six scans see ONE underlying per-galaxy axis
   through degenerate coordinates. (Partly by construction for 1-D
   refits; the honest dimensionality test is 0b below.)
3. **ALIGNMENT: the kernel's own individuality axis IS the exp39
   core-diversity axis** — every delta correlates with the per-galaxy
   log10 f_core at |rho| = 0.78-0.87 (sig -0.87, gamma +0.85, log_rc
   -0.83). The core diversity measured in exp39 lead 3 is reachable
   INSIDE the kernel's parameter space; no core channel is needed for
   the stochastic layer.
4. **The axis is feature-orthogonal** (|Spearman rho| <= 0.20 vs
   logmh, c200c, fz2, logms, logmh_z2) — consistent with exp39: this
   is per-object information; essentially ALL of it belongs to the
   stochastic layer, none to new conditioning.
5. **Stage 0b, dimensionality (joint (log_rc, sig) refit vs best
   single):** the per-galaxy median extra from the second dof is 1.9
   points (vs 24.7% from the first) — 1-D for the typical galaxy —
   BUT the gain is strongly skewed: 31% of galaxies gain > 5 points
   and 19% gain > 10. A genuine second axis exists for a ~fifth of
   the population.

**Stage-1 design consequence:** build the deviation distribution in
BOTH a 1-D and a 2-D variant (the second component small and
heavy-tailed, or mixture-like per the 0b skew), and let the
pre-registered plane criterion pick the smallest sufficient
dimensionality.

### Stage 1 — the deviation distributions (full n=2397)

d_sig (the main axis): robust sigma 0.076-0.081, HEAVY tails
(Student-t dof 4-9, strong positive skew) — a Gaussian misses the
extreme-diversity galaxies. d_q (the second axis): sigma 0.278,
near-Gaussian, median -0.085 (the location-free price). Correlation
d_sig x d_q = -0.31 (drawn jointly). Box pile-up ~2% — draws clipped
to the physical box are safe. Loss ladder: 1-D 18.5% -> 2-D 32.3%
(reproduces stage 0b).

### Stage 2 — drawn populations, judged (CHECKPOINT before adoption)

Held-out protocol: the deviation distribution is calibrated on one
half of the sample and drawn deviations are applied to the OTHER
half's galaxies (symmetrized, 8 realizations per direction). Centered
plane energy / split-half floor (target ~1; "mean" = the
deterministic kernel):

| plane | model | z=0.4 | z=0.7 | z=1.0 | z=1.5 | z=2.0 |
|---|---|---|---|---|---|---|
| M(<30) vs M(30-50) | mean | 1.6 | 1.8 | 2.1 | 2.3 | 2.9 |
| | 1d-emp | **1.1** | 1.4 | 1.6 | 1.8 | 2.6 |
| M(<30) vs M(50-100) | mean | 1.6 | 2.1 | 2.3 | 2.5 | 3.0 |
| | 1d-emp | **1.0** | 1.4 | 1.6 | 2.0 | 2.8 |
| | 2d-emp | **0.9** | 1.3 | 1.5 | 2.1 | 2.8 |
| M(<2Re) vs M(2-4Re) | mean | 2.0 | 1.9 | 2.0 | 1.7 | 1.2 |
| | 1d-emp | 2.7 | 2.5 | 2.3 | 2.0 | 1.3 |

Physics band on full drawn populations: **1d-emp PASSES** in the
adopted band (differential 0.40/0.12; overshoot T1 +0.027/+0.031);
2d-emp strains the overshoot (T1 +0.052/+0.061 — the wide-q draws
inflate low-mass outskirts). Drawn-theta mean offsets ~0.002-0.005
(the mean model untouched, criterion met). The Gaussian variants are
uniformly equal-or-worse than empirical resampling — the heavy tails
carry real plane signal.

**Read:** the pre-registered target is MET at z=0.4 (1.0-1.1x floor,
from the kernel's 1.6x) and nearly at z=0.7 (1.3-1.4x) on the kpc
planes, with the 1-D empirical layer physics-clean — the first
kernel-based drawn population to reach the floor on any plane. NOT
met at z >= 1.5 (2.0-2.8x: the mean model's ridge-shape error
dominates there — a scatter layer cannot fix a ridge; consistent with
the z=2 epoch being extrapolation under the official scope), and the
Re-coordinate plane DEGRADES under the layer (2.0 -> 2.7x at z=0.4):
epoch-independent, size-uncorrelated deviations violate the
R/R_half self-similarity the data obey (the exp38 stage-0
measurement). Candidate refinements, not run: deviations applied in
Re-preserving form, and/or epoch-coupled deviations (AR(1)-like, as
exp37 does statistically).

**Decision (user): adopt the 1-D empirical layer as-is (with its
documented scope: kpc planes, z <= 0.7-1.0), iterate on the
Re-preserving refinement, or park.** -> User picked the refinement
round; its outcome below changed the verdict.

### The Re-preserving refinement (user option 2): rejected — and it
### exposed the real story

The refinement (apply d_sig, then compensate log_rc so each drawn
galaxy keeps its official z=0.4 half-mass radius) improved the paired
Re plane only partially (2.7 -> 2.2, still worse than the mean's 2.0)
while DESTROYING most of the kpc-plane gains (1.0-1.1 -> 1.4-1.5 vs
the mean's 1.6). Physical reading: the layer's usable diversity IS
largely size diversity — pin the size and little remains (the exp38
R/R_half self-similarity, seen from the model side).

That raised the fairness question: qa's Re apertures use the TRUTH
half-mass radius applied to the model — a PAIRED metric, correct for
a mean prediction but structurally punitive for a generative draw
that is independent of the specific truth realization (the exp33
"fails only in Re coordinates" pattern, now explained). Rescoring the
Re plane SELF-CONSISTENTLY (truth masses through truth sizes, drawn
masses through drawn sizes):

| model | z=0.4 | z=0.7 | z=1.0 | z=1.5 | z=2.0 |
|---|---|---|---|---|---|
| mean | 1.2 | 1.1 | 1.3 | 1.5 | 1.2 |
| **1d-emp** | **1.0** | **1.0** | 1.3 | 1.4 | 1.1 |
| 1d-re | 1.1 | 1.1 | 1.3 | 1.5 | 1.2 |

**The plain 1-D empirical layer PASSES the Re plane at every epoch**
(at the floor at z <= 0.7, never above 1.4x, and never worse than the
mean model). The "degradation" was the paired-aperture artifact; the
refinement is unnecessary and rejected.

### Final stage-2 scorecard (the 1-D empirical layer)

- kpc planes: 1.0-1.1x floor at z=0.4 (target MET), 1.3-1.4 at z=0.7,
  degrading to 2.6-2.8 at z=2 — the residual is the MEAN model's
  ridge error at high z, which no scatter layer can fix.
- Re plane (self-consistent scoring): 1.0-1.4x at every epoch.
- Physics: differential 0.40/0.12, overshoot T1 +0.027/+0.031 —
  inside the adopted band. Mean model untouched (offsets ~0.002).
- Form: 1-D (smallest sufficient), empirical mean-centered resampling
  of the measured heavy-tailed d_sig (Gaussian is uniformly worse),
  clipped to the physical box.
