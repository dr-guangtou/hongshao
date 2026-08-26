# exp60 — the stochastic layer on the deposition-only model

*Written 2026-08-26, before any code, at the user's direction after the exp59
verdict: "the stochastic layer is worth trying, and you may lead with this as
the next step." Companion directions (single-epoch baselines and the z ≤ 1.0
scope, exp61) run alongside and feed this plan's targets.*

## 1. What the layer is, and why now

The deposition-only model is deterministic: two galaxies with the same halo
history get the same profile. The measured population is not, and the
programme has now measured — three ways — that the missing galaxy-to-galaxy
variation is not recoverable from the halo record: the model's central-growth
spread is 0.33 times the data's (exp57 Stage 1), every efficiency-law
conditioning moves the decliner splits in lockstep (exp59 Stage 0b), and the
conditioned-amplitude ceiling recovers about a fifth of the z=2 gap (exp58
P-C). The user's physical reading says the same thing from the physics side:
the declining centres mix PROJECTION/ORIENTATION variation (the profiles are
2-D along one box axis) with SMBH-FEEDBACK-driven expansion, neither of which
the halo assembly history carries. Both are naturally modelled as a RANDOM
PROCESS around the mean model — a stochastic layer — and neither should ever
be chased by a shared law again.

The precedent is exp41, which built exactly this on the OLD kernel and
graduated three lessons this plan inherits: (1) a 1-D empirical
(resampled, heavy-tailed) deviation layer reached ~1.0–1.4× the split-half
floor on the kpc observational planes at z ≤ 0.7–1.0 — the first drawn
population to reach the floor; (2) **a scatter layer cannot fix a mean-model
ridge** — at z ≥ 1.5 the mean error dominates and no draw helps, which is
also the measured motivation for the user's z ≤ 1.0 retreat; (3) paired
(per-galaxy) metrics are structurally punitive for draws — a draw is a
realization, not a prediction — so draws are judged on DISTRIBUTION
statistics, held out, and never on paired errors.

## 2. The three deviation components

1. **Amplitude** — a per-galaxy, cross-epoch-correlated deviation of the
   total-mass scale. The machinery exists (`exp54/rebaseline.py`
   `draw_amplitudes`: per-epoch residual sigmas with the measured cross-epoch
   correlation). Carries the M*(<100 kpc) scatter and the growth tiers.
2. **Size/shape** — exp41's anatomy protocol rebuilt on the incumbent: free
   ONE parameter per galaxy (which one is Stage 2's first measurement, on a
   subsample), fit the per-galaxy deviation distribution empirically,
   mean-zero. Carries the tier-2b kpc planes.
3. **Core evolution — the NEW component.** A per-galaxy stochastic core-only
   expansion: `A_i` drawn from a mixture (most galaxies near zero; a
   decliner-scale tail), applied through the GATED small-core remap (`X3`,
   `Rc` frozen at ~2 kpc — reach-safe by construction, 0.5–1.5% of moved
   mass beyond 50 kpc at the shared optimum). Mechanically this is "SMBH
   episodes and orientation flips as noise", the user's two mechanisms.
   Carries the central-evolution distribution, which no previous layer ever
   targeted.

## 3. Pre-registered targets and gates (thresholds frozen here)

All scored HELD OUT (calibrate on one random half, score on the other), at
z ≤ 1.0 as the primary scope; z ≥ 1.5 is reported, not gated (exp41 lesson 2
plus the user's retreat). Controls: every distribution fit gets a
permutation control (deviations assigned to shuffled galaxies), and the
core component is fitted twice — with and without an `f_form` tilt on the
mixture weight — so "conditioning helps the conditional statistics" is a
measurement, not an assumption.

| gate | quantity (all on drawn populations) | threshold |
|---|---|---|
| S1 | central-evolution distribution at 4.92 kpc, z=2→0.4: declining fraction / decliner median / overall median (measured 41.8% / −0.066 / +0.037) | within ±3 points / ±0.010 dex / ±0.010 dex |
| S1b | the same 10th-to-90th width (measured −0.099 to +0.438) | width within 15% |
| S2 | tier-2b kpc-plane energy vs the truth's split-half floor, z ≤ 1.0, held out | ≤ 1.3× floor |
| S3 | the mean untouched: drawn-population median CoG vs the mean model | within 0.010 dex at every radius, z ≤ 1.0 |
| S4 | physics band: G1b outskirt share and the M[50,148] annulus on the drawn population | within the incumbent's band |
| S5 | amplitude scatter and cross-epoch correlation reproduced | scatter within 10%, correlations within 0.05 |

Never quoted: paired per-galaxy errors of draws, except to restate that a
draw inflates them by construction (~√2, exp54 rebaseline).

## 4. Stages

- **Stage 0 — freeze the targets** (half a day). One script recomputes every
  S1/S5 target number on the exp57 sample, per `f_form` quartile as well
  (for the tilt test), and stores them; the gates never read the data again.
- **Stage 1 — the core component's anatomy** (~half a day of compute). Per
  galaxy, the 1-D bounded `A_i` that matches its measured central change at
  4.92 kpc (bisection on the X3 evaluator, `Rc` frozen); deliverables: the
  empirical `A_i` distribution, its `f_form`/`dlogc` tilt, and the fraction
  of galaxies whose measured change is unreachable even with per-galaxy
  freedom (the layer's own ceiling, stated before it is fitted).
- **Stage 2 — the size/shape anatomy on the incumbent** (exp41 Stage 0
  protocol, subsample first). Which single coordinate carries the
  individuality now, its per-galaxy distribution, and its feature
  correlations (what would belong to conditioning — expected ~none, per
  exp41's finding 4, but measured not assumed).
- **Stage 3 — compose and judge.** Draw all three components (independent
  first; couple only if a gate demands it), build drawn populations, run the
  gates and the standard battery with `draw_cogs`, held out. Decision on
  adoption: the user's.

## 5. What this plan does NOT do

- No refit of the mean model. The incumbent stays exactly as adopted; every
  deviation is around it (S3 enforces this).
- No per-galaxy freedom in the delivered model: the layer is a DISTRIBUTION;
  per-galaxy values are drawn, never fitted to the target galaxy.
- No z ≥ 1.5 promises. If exp61 confirms the mean model's z ≥ 1.5 gap is
  structural, the layer's scope statement is "z ≤ 1.0", and the z ≥ 1.5
  record points at additional data (the real merger tree, 3-D profiles) —
  the user's stated read.
