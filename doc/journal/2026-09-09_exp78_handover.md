# Session Handover — 2026-09-09 (exp78, the size-aware objective)

## Goal

The user's ask: exp78, a SIZE TERM for the objective, designed and vetted
before any fit (Stage 0), then the baseline refitted under it (Stage 1) and
with the growth-rate split free (Stage 2), judged by the gates, reported in
plain language with positives and negatives. Rules: nothing per galaxy or per
epoch from the simulation enters the model; every candidate states its
parameter count; the gates decide, the loss does not.

## Completed

- Branch `exp78-size-aware-objective` from master `d9bad92`; code in
  `experiments/exp78_size_aware_objective/` (`size_terms.py`,
  `stage0_terms.py`, `stage1_fit.py`, `stage1_eval.py`, `README.md`).
- **Stage 0** (`outputs/stage0_terms.log`): the radius term (tercile-median
  log R20/R50/R80 offsets on the merged 0.673–148 kpc grid) passes the
  blind-spot probes and ranks the five known models as the gates do
  (baseline 0.550 < exp76 split 0.589 < exp63 0.609 < 14.63 basin 0.642 <
  incumbent 0.946). exp77's annular term (both normalisations) ranks the
  gate-rejected basin BEST and is exactly blind to a central point mass —
  it did not enter the loss. Also recorded: at weight 1, Z² does not flip
  the loss's ordering of the two basins (17.07 vs 16.69).
- **Stage 1** (four starts + a continuation, `stage1_fit.npz`): optimum
  15.82 in the basin's family, the compact channel's size railed at 31.6 kpc
  (b_c −1.83); a second basin at 16.31 from the baseline start. Judge
  (`outputs/stage1_eval.log`, figures `figures/qa/*exp78_*`): offset gate
  11 of 15 vs the baseline's 12; R50 at z = 1.5/2 worse (+0.059/+0.068 vs
  +0.037/+0.054); centre at z = 0.4 +8.4 vs +5.6 per cent; the 50–100 kpc
  shell at z = 2 +9.8 vs +29.6; widths up at every entry; the
  future-dependence gate unchanged. **Not adopted.** The term improved only
  at z ≤ 1, where the gate already passed.
- **Stage 2** (`stage2_fit_growth.npz`): g → −0.005 from −1; g = −2 stops at
  −0.08 at a higher loss. The size-aware loss rejects the split as exp63's did.
- Docs: README verdict; `doc/lessons.md` (7 lessons), `doc/todo.md`,
  `doc/open_questions.md` (C24; C23 annotated), `CLAUDE.md` (exp78 taken);
  memory `size-aware-objective-verdict.md`.

## Not finished / owed

- The extended deposit's r200 scaling under the measured input (the high-z
  R50 excess is in the top halo-mass tercile and grows with stellar mass,
  mass–size slope 0.37 vs 0.11 at z = 2): a MODEL change, the next step.
- The compact channel's size bound (1.5 dex = 31.6 kpc) binds under every
  size-aware start; widen or re-parametrise only if the r200 re-tune leaves
  it binding.
- If a size term stays: a gated weight sweep; a width term is the layer's.
- exp76: option 2 (g fixed by physics) remains; option 1 (a size term) did
  not rescue it.
- The v1 stochastic layer's re-baseline (still owed since 2026-09-08).
- Merge of `exp78-size-aware-objective`: the user's call. Not pushed.

## Ops notes

- Foreground calls capped at ten minutes: every fit and judge ran under
  nohup with `caffeinate -i -w <pid>`; five processes in parallel took
  26–27 min per 3000 evaluations (20 min alone).
- `stage1_fit` and `rebaseline` are shadowed by other experiments' modules;
  both are imported by file path.
- The judge first looked for `stage1_fit_growth.npz` while the merge wrote
  `stage2_fit_growth.npz`; fixed, judge re-run (the first log is kept as
  `stage1_eval_without_stage2.log`).
- The polling loop's `grep "^wrote"` matched figure lines; poll on the final
  npz name instead.

## Branch state

`exp78-size-aware-objective`, all code committed; outputs gitignored under
`experiments/exp78_size_aware_objective/outputs` and `figures`. Nothing running.
