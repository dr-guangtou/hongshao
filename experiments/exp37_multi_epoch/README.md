# exp37 — the multi-epoch statistical emulator (Path A, the product)

Status: IN PROGRESS (branch `exp37-multi-epoch-emulator`, started 2026-07-13).

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
