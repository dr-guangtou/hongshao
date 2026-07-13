# exp37 — the multi-epoch statistical emulator (Path A, the product)

Status: first full results 2026-07-13 (branch `exp37-multi-epoch-emulator`);
see Results below. Open item: the high-z kpc-annuli plane mismatch of the
draws.

## Results (2026-07-13, n=2397, OOF 5-fold, `outputs/full_*.log`)

1. **The multi-epoch product costs ~nothing vs independent per-epoch fits.**
   kpc CRPS per epoch 0.0813/0.1028/0.1294/0.1713/0.2001 — AT the exp33-vi
   independent-single-epoch ceiling (0.081 -> 0.200). Profile tier 3 median
   max|rel| (all R | R>5 kpc): 34.0–42.6% | 24.4–33.6%, matching the exp33-vi
   per-epoch ceiling (24.3% -> 33.6%) — the ONE shared PCA basis is free.
2. **rho = 0.664 measured** (exp33-vi reference 0.67, adjacent C 0.64–0.70);
   the AR(1) draws reproduce the full measured Markov decay matrix
   (draw-rho 0.674, every element within ~0.06).
3. **The multi-epoch generative claim is TRUE with the AR(1) latent**: the
   growth plane logMtot(z=2) vs logMtot(z=0.4) sits at 3.0x the sampling
   floor for the mean prediction (regression to the mean) but 1.0–1.1x for
   the drawn populations — statistically indistinguishable from TNG.
   Re-native planes pass at every epoch (0.8–1.4x floor, full and centered).
4. **Continuous-z closure holds in the shared-basis profile space**: held
   epochs z=0.7/1.0/1.5 lose <=0.5 points max|rel| (e.g. 35.2% -> 35.5% all-R
   at z=0.7) when the cores are coefficient-interpolated instead of fitted.
5. **z=2 low-halo-mass bias — MEASURED cause (2026-07-13 review)**: the model
   under-predicts the lowest-logmp tercile at z=2 by −5.2% median (−0.023
   dex, flat in R; the visible dashed–solid gap in the bins figure is −0.05
   dex because comparing medians of log in an under-dispersed, skewed
   population roughly doubles the visual gap). It is a LINEAR-MEAN
   limitation: poly2 cores cut the tercile bias to −1.9% and the z=2 anchor
   OOF rms from 0.221 to 0.166 dex. Candidate change (user decision):
   switch the cores to `mean="poly2"` — costs the exp33-vi linear-mark
   comparability, forbidden only for the forward-deformation baseline (which
   this product is not).
6. **Best/worst gallery reading (2026-07-13 review)**: under minimax (worst-
   epoch) ranking the best galaxies still carry smooth ±10–15% shape
   residuals — the per-galaxy information limit (population medians are ±5%),
   not a defect of the gallery. The "worst" cases are dominated by
   near-empty TRUTH progenitor CoGs (flat at ~1e8 Msun over 4 epochs while
   z=0.4 is ~10^11.8, e.g. idx 181, 2351) — a DATA-quality flag for the
   exp32 population table (broken progenitor match?), not model failure;
   the relative-to-truth metric explodes there by construction.
7. **RESOLVED 2026-07-14 — monotonicity-preserving draws (the density-space
   product)**. The mode-3 per-radius log-CoG draws could cross themselves
   where sigma is large (drawn M(<100) < M(<50) -> annulus floored to
   log M = 0, a spurious tail that blew the kpc planes to 2–12x floor at
   z>=1). The stochastic layer now lives in LOG-DENSITY space (mode 4,
   `fit_density_profile`/`MultiEpochDensity`): any real draw is a positive
   shell mass summed outward (the exact `integrate_density` identity,
   round-trip asserted at 1e-9 in the demo), total pinned to the drawn
   anchor — every drawn CoG is monotone BY CONSTRUCTION (runtime-asserted
   in `fit`). Compressed vector: [anchor, K=6 pooled density-shape scores,
   central fraction]; K scanned 3/6/8 — K=6 adopted (low-z outskirt mean
   bias +9.9% at K=3 -> +2.7%; K=8 marginal further gain).
   Results (n=2395, poly2, K=6): draw kpc planes 0.9–1.3x floor through
   z=1.0 (were 3–4x), growth plane draws 1.2–1.3x, rho in draws 0.632 vs
   0.622 measured; tier-3 R>5 kpc avg improves 29.2 -> 26.9%. Two honest
   caveats: (a) the MEAN reconstruction under-predicts the far kpc outskirts
   at z>=1.5 (M(50-100) -22%, M(>100) -59% at z=2, the 15–30 R_half
   near-empty regime; the Re-relative view — the honest one there, exp29
   lesson — is <=2% biased in the envelopes at z=2); (b) the remaining z=2
   kpc-plane residual (~7x floor) is now the OPPOSITE tail: the truth has
   floored zero annuli that monotone lognormal shells can never produce —
   irreducible in this model class and plausibly a resolution artifact of
   the truth.

## Goal

Build the product-path multi-epoch emulator from the exp33-vi blueprint, in
which every ingredient has a measured justification (exp33 README step vi,
n=2397, z=0.4/0.7/1.0/1.5/2.0):

1. **Continuous-z mean/scatter by coefficient interpolation** — five
   independent per-epoch heteroscedastic cores (`hongshao.emulator.fit`,
   shared portable features [DiffMAH(4), c200c], shared folds), coefficients
   interpolated quadratically in z. Measured basis: the held-epoch closure
   passes to −3.2%/+0.7%/+3.9% of the direct fit at z=0.7/1.0/1.5.
2. **AR(1)-in-epoch latent** — the cross-epoch OOF residual correlation is
   Markovian (adjacent 0.64–0.70, decaying as rho^sep with rho=0.67), so the
   stochastic layer draws one AR(1)-in-epoch latent per galaxy, NOT a static
   offset. rho is MEASURED in-fit from the OOF residuals (recorded reference
   0.67), never hardcoded.
3. **Generative sampling** — the mean alone is under-dispersed (exp15);
   draws must restore the population. The single-epoch generative layer
   passes the 2-D plane test (exp33 step v); the multi-epoch version must
   pass it per epoch AND be coherent across epochs.

## Re-baseline 2026-07-13 (user decisions: poly2 cores DEFAULT + broken-progenitor mask)

n=2395 (idx 181, 2372 masked: progenitor totals >2 dex below the z=0.4
descendant — the population's max drop otherwise ends at 1.68 dex; criterion
in `progenitor_quality_mask`, demo-asserted). `mean="poly2"` (exp17's 7
degree-2 terms) is now the core default (`--linear-mean` restores the old
behavior); note poly2 changes the MODEL (the mean function), so every number
below supersedes the linear Results section, which stays as reference.

- kpc CRPS per epoch **0.0784/0.0888/0.1027/0.1285/0.1523** — now BEATS the
  linear exp33-vi ceiling at every epoch (−4% at z=0.4, −24% at z=2).
- Cumulative-aperture biases <=1.4% at every epoch (the z=2 low-mass −5%
  linear-mean bias is gone); M(<100) dex scatter at z=2: 0.220 -> 0.153.
- rho = 0.622 (was 0.664): part of the "persistent per-galaxy latent" was
  linear-mean misfit, now absorbed by the mean.
- Closure unchanged (<=0.7 points at every held epoch); tier 3 all-R direct
  improves 35.2/38.0/40.8 -> 33.6/35.1/36.6 at z=0.7/1.0/1.5.
- Growth plane: mean 1.7x floor (was 3.0), draws 1.0–1.3x. Re-planes pass at
  every epoch. The kpc-annuli draw planes at z>=1.5 (4–13x floor) remain
  gated by the non-monotonic-draw open item below.

## Option (b) verdict — the BLOCK-PINNED product wins (2026-07-14)

Three profile representations are now separately selectable
(`--profile-mode cog|dens|block`, per-mode outputs/figures; user decision:
keep all three available, evidence decides the default). Head-to-head, same
folds and poly2 cores (n=2395):

| metric | cog K=3 | dens K=6 | block K=6 |
|---|---|---|---|
| R<10 kpc bias, z=0.4 -> 2 | +1.8 -> -0.1% | +7.2 -> +2.8% | +3.5 -> -1.4% |
| M(10-30) bias z=0.4 / z=2 | +1.2 / +13.4% | -1.4 / +2.4% | +0.9 / +2.1% |
| M(50-100) bias z=0.4 / z=2 | +4.3 / +13.6% | +2.7 / -22.4% | +5.3 / +7.9% |
| tier-3 R>5 kpc (avg) | 26.5% | 26.9% | 26.5% |
| draw kpc planes z<=1 / z=2 | 0.8-4x / 7-13x | 1.0-1.3x / 7-8x | **0.4-0.9x / 1.8-3.5x** |
| monotone draws | NO | yes | yes |
| closure (held z, all-R cost) | <=0.3 pts | <=1.1 pts | <=0.9 pts |

`MultiEpochBlock` predicts the log masses of the kpc blocks (2-10/10-30/
30-50/50-100/100-148, snapped to grid radii) DIRECTLY, plus the central
fraction and K=6 pooled density-shape scores that only distribute mass
WITHIN blocks — monotone by construction, full radial resolution. Two
mechanisms behind the win: (1) drawn block masses are lognormal per block,
so draws CAN produce the truth's near-empty high-z annuli while staying
monotone — the z=2 planes drop from ~7-12x floor to 1.8-3.5x (growth plane
1.0-1.3x); (2) for the MEAN curve, the gap between the predicted total and
the sum of median-predicted parts (a lognormal median-vs-mean effect,
measured +2.8 -> +6.6% per epoch) is allocated in proportion to each
block's expected gap B_j (e^(sigma_ln^2/2)-1) instead of uniformly — tight
blocks stay at their direct predictions (demo-asserted), and the residual
inner bias drops +5.6 -> +3.5% (z=0.4), beating cog at z>=1. Remaining
edge for cog: the z=0.4 inner bias (+1.8 vs +3.5%). Default: block.

## Design

- `run.py` subcommands: `demo` (synthetic self-checks, no data), `fit`
  (per-epoch cores + shared-basis profile cores + rho measurement ->
  `outputs/multi_epoch.npz`), `qa` (OOF model CoGs -> `hongshao.qa.evaluate`
  per epoch, figures), `closure` (held-epoch closure re-run in the
  shared-basis profile space), `report` (verdict tables).
- **Targets**: mode-1 kpc masses [M(<10), M(10-30), M(30-50), M(50-100)] per
  epoch (the exp33-vi setup) for the scoreboard; a PROFILE mode for tier-3 QA.
- **Shared PCA basis for the profile mode**: per-epoch `fit_profile` bases
  differ per epoch, which blocks coefficient interpolation; exp37 pools the
  amplitude-removed shapes of ALL epochs into ONE PCA basis, then fits
  per-epoch cores on [anchor, shared-PC scores]. The closure test is re-run
  in this space — if it fails here, continuous-z holds only for the kpc
  masses and the README must say so.
- **Cross-epoch sampling construction**: eps = chol(A(rho)) @ Z per target
  (A = AR(1) in epoch-index separation), then within-epoch eps_k @
  chol(R_k).T. Within-epoch correlation is exactly R_k; cross-epoch is
  A_kj-modulated (exact when R_k == R_j); PSD by construction. The demo
  asserts both empirically.
- Reuses exp33's validated `load()`, `folds_of`, `interp_emulator` via the
  importlib `_load_by_path` pattern (`import run` collides across experiment
  dirs).

## Judged by (fixed before fitting)

- `hongshao.qa.evaluate` per epoch on OOF mean CoGs: all tiers, max|rel|
  quoted all-R AND R>5 kpc, planes per epoch (energy/floor, full + centered).
- Generative: per-epoch 2-D planes on draws (the exp33 step-v standard), plus
  cross-epoch coherence — the growth plane M*(z=2) vs M*(z=0.4) (truth vs
  draws, energy/floor) and draw rank-persistence across epochs vs the truth's
  (the AR(1) structure must reproduce the measured Markov decay, not just
  adjacent-epoch correlation).
- Reference marks: independent-single-epoch ceiling CRPS 0.081 -> 0.200 and
  CoG shape max|rel| R>5 kpc 24.3% -> 33.6% across z=0.4 -> 2.0 (exp33-vi);
  closure within +-4% of direct.
