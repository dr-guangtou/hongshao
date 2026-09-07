# Handover — 2026-09-08 overnight run (adoption + re-baseline; exp76)

Two branches, neither merged (the user's call):

- **`adopt-measured-input`** (this working directory): the measured history
  adopted as the application default; the incumbent and exp63's model
  refitted on it under references rebuilt at the nested incumbent on the
  measured curves, halo-mass bins by the measured mass. Code
  `experiments/exp74_c19_history_leak/rebaseline.py` and `rebaseline_eval.py`;
  the README section "Adoption and the re-baseline"; figures
  `experiments/exp74_c19_history_leak/figures/qa/*exp74_adopt_*`.
- **`exp76-growth-rate-split`** (worktree `hongshao_exp76_growth_rate_split`,
  its `experiments/*/outputs` symlinked to this checkout's outputs): the
  deposit's compact-or-extended choice allowed to depend on the halo's growth
  rate at the deposit. `experiments/exp76_growth_rate_split/README.md`;
  figures `.../figures/exp76_stage0_sweep` and `.../figures/qa/qa_bins_exp76_*`.
- A parallel Codex agent owns `hongshao_exp75_halo_residual_correction`;
  untouched. Check `git worktree list` before numbering anything.

## The results, one paragraph each

**Adoption.** The incumbent re-baselines cleanly: three starts agree at
16.14 (null 20.85), a single basin, with a softer mass dependence in the
efficiency and zero dependence on the halo's future. exp63's model does NOT
re-baseline cleanly: the adopted references open a basin 6 per cent lower in
loss (14.63) than exp74's measured optimum (15.56, which is a stationary
point), and it is a different model — a 25 kpc "compact" channel holding 44
per cent of the deposits — that every gate ranks worse (size-gate offset
10/15 against 12/15, R50 +21 against +13 per cent at z = 2, the z = 0.4
centre +9.0 against +5.6 per cent, the z = 2 massive progenitors +3.8
against −1.0 per cent). This is the fifth recorded case of the loss ranking
basins against the gates (C23). **Recommendation: adopt exp74's measured
optimum as the baseline mean under the adopted references, and record the
14.63 basin as loss-preferred and gate-rejected.** The v1 stochastic layer is
NOT re-baselined: it is exp57's X3 expansion problem on the old step engine
reading DiffMAH-generated snapshot masses; re-running exp60's stages 1–3 on a
measured-history predictor is about half a day and should follow the
baseline decision.

**exp76.** Stage 0 (no fit): on the measured input the g = 0 model's compact
share rises with recent growth and late formation at +0.3 where the data
show 0.0, and gets the DiffMAH-variable signs backwards (P9); g ≈ −1.5 to −2
brings all eight assembly correlations within about 0.1 of the data's, and on
the fixed slice halves the high-redshift central deficit. Stage 1 (the fit):
the objective rejects the split — from the baseline, g stays at exactly 0;
starts at −1 and −2 walk back to −0.27 at 2.5 per cent worse loss. With the
split on (g = −0.27) the size diversity at fixed mass more than doubles in
the outskirts at z = 2 (R80 width ratio 0.22 → 0.48), both centres improve,
high-z sizes fall and the spurious growth dependence halves; low-z sizes
overshoot the other way and the z = 1.5 massive progenitors get heavier
inside 10 kpc. **Not adopted, not closed**: a lever the gates want and the
loss cannot see. **C22, found on the way**: the data's compact share tracks
DiffMAH's `late` and `f_form`, not the measured formation time or recent
growth — P9's target may be partly the parametrisation.

## Decisions owed

1. The baseline mean: exp74's point (by the gates) or the 14.63 basin (by
   the loss). Recommendation: exp74's point.
2. Whether exp63's objective grows a size term (C23; exp76's option 1),
   which would let a fit keep the growth-rate split if it is real.
3. exp76's fate: a size term then refit with g free; or g fixed by physics
   (−0.3 to −2) with the layer owning the rest of the width; or closed.
4. C22: what measured quantity, if any, does the compact share follow.
5. Merge `adopt-measured-input`; merge or close `exp76-growth-rate-split`.
6. The v1 layer's re-baseline, after 1.

## Ops notes

- `rebaseline_eval.py` imports `rebaseline.py` by file path: exp54 has a
  module of the same name that shadowed it.
- Near starts (5 per cent of the bound range) rail into the failure penalty
  under this objective on the first step; a capped solution is settled with
  a continuation start from its own parameters.
- Seven fits and two judges ran in parallel at 0.4 GB each with
  `caffeinate -i -w` on every process; the machine stayed awake.
- A foreground shell call is capped at ten minutes; every fit and judge ran
  through nohup.
