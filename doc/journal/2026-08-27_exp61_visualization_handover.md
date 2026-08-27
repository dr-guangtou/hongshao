# Session Handover — 2026-08-27 (exp61 visualization)

## Goal

The previous handover's owed artifact: **add QA figures and visualization to
the exp61 results**. Delivered, and then extended twice at the user's
direction — first with the project's STANDARD QA battery at every epoch, then
with a diagnostic answering the user's question about what the battery showed.
Six figures of exp61's own plus the full 3-model battery.

## Completed This Session

Four new stages in `experiments/exp61_epoch_baselines/`, all committed
(`c7f926b`, `322def8`, `9bfcf95`, plus this session's final bookkeeping
commit). Read the exp61 README top to bottom — it now carries a script map
and one section per stage.

1. **Stage 2 — the tables as figures.** `stage2_figure_data.py` re-evaluates
   the seven frozen thetas (one build + 0.12 s per loss) to recover what the
   Stage 0/1 npz files never saved — the cross-epoch loss matrix, the B1/B3
   bias currencies for every parameter set including the M2 fit, the bounds —
   and REFUSES to write unless every number reproduces the fit's own record
   (it did, 0.00e+00 on all five single-epoch losses and both scores).
   `stage2_figures.py` then draws `exp61_baselines` (the table in four
   currencies), `exp61_transfer` (the cross-epoch matrix), `exp61_drift`
   (seven parameter panels + the degeneracy caveat), `exp61_m2_null`.
2. **Stage 3 — the STANDARD battery.** `stage3_qa.py`, `hongshao.qa.evaluate`
   on three models over the same 2397 galaxies: the shared incumbent, the
   per-epoch ceiling (each epoch's own fit, stitched), the z ≤ 1.0 refit.
   Eleven figure families per model in `figures/qa/`, plus `exp61_qa_compare`.
   Every bias printed on BOTH samples.
3. **Stage 4 — the user's question, answered by measurement.**
   `stage4_cdf_anatomy.py` decomposes each tier 2e distance into a median
   SHIFT and a WIDTH error by centring each distribution on its own median.
   Gated on reproducing the battery's own tier 2e table.

## The three results worth carrying forward

- **The shared law is a CLOSE compromise, not a poor one** (Stage 2). Its row
  of the transfer matrix is 4.5 / 2.3 / 1.8 / 5.2 / 15.9 per cent, while
  carrying any single epoch's own optimum elsewhere costs up to 82.9 per cent.
  Memory: `shared-law-is-a-close-compromise`.
- **The per-epoch ceiling does NOT transfer** (Stage 3). It zeroes the z=2
  bias on the mh-complete fit mask (839 of 2397 galaxies at z=2) and
  OVERSHOOTS the whole population, +5.5 per cent inside 10.25 kpc and +12.4
  inside 100 kpc. **This corrected a Stage 2 claim already written into the
  README and reported to the user** — logged as exp61 PROBLEMS P3. Memory:
  `per-epoch-ceiling-does-not-transfer`.
- **Fitting harder NARROWS the population** (Stage 4). Conditional-mean
  under-dispersion grows with fitting effort: width over the truth's at
  `kpc:M(50-100)`/z=2 is 0.54 for the shared law and 0.46 for the per-epoch
  fits, and the z ≤ 1.0 refit is the control (at z ≥ 1.5, unfitted, its width
  returns to the shared law's). Memory:
  `fitting-harder-narrows-the-population`.

## In Progress (Not Finished)

Nothing running, nothing uncommitted. Every figure on disk was regenerated
from the committed scripts and the Stage 3 log was reproduced end to end
(tables identical).

## Problems / Blockers

- **NOT PUSHED.** Three (now four) commits sit ahead of `origin/master` on
  `exp60-stochastic-layer`. The user has not been asked for the push yet.
- **Open question C16, filed this session** (`doc/open_questions.md`):
  `qa.evaluate`'s `draw_cogs` overlays draws on the PLANE figure only — every
  tier, 2e included, is computed from the MEAN. exp60's adopted-layer tier 2e
  table is identical digit for digit to the bare incumbent's. So every tier 2e
  number quoted for hongshao v1 in this programme describes the mean alone,
  and the stochastic layer is precisely the component that supplies the width
  tier 2e says is missing. Cheap to answer; changes no adopted decision.
  Tracked as exp61 PROBLEMS P4 (open).
- Open question **C13** (which z=2 population the model owes its median to) is
  promoted from optional to load-bearing by Stage 3 — no z=2 attempt should
  start before the user answers it.
- Local `master` ref still STALE at `a5e4f6a`, held by the exp56 worktree
  (another agent's — do not touch). `exp62` still taken.

## Key Decisions

None taken this session — no model was fitted and nothing was adopted. Stages
2 and 4 evaluate frozen thetas; Stage 3 runs the standard battery. The one
substantive change is to what may be CLAIMED from exp61's z=2 result, which
the README, PROBLEMS.md and memory now all state.

## Files Modified This Session

New: `experiments/exp61_epoch_baselines/{stage2_figure_data,stage2_figures,
stage3_qa,stage4_cdf_anatomy}.py` and `PROBLEMS.md`. Extended: that
experiment's `README.md` (script map + Stage 2/3/4 sections + the Stage 2
qualification), `doc/lessons.md` (six new entries under Figures / QA
presentation), `doc/todo.md` (three stages with review sections),
`doc/open_questions.md` (C16). Memory: three new files plus the index.

---
*Wrap-up requested by the user after Stage 4.*
