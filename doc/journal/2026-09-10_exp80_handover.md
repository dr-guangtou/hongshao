# Handover — 2026-09-10 (exp80, the deposit size law: DONE on its branch, not merged)

## Goal

exp80: read from the data, epoch by epoch, where the stars are deposited
(the deposit-size distribution), compare with the adopted baseline's size
law, choose the law change, test it at frozen amplitude, then fit it under
the standard objective and judge by the gates. Plan
`doc/plans/2026-09-09-exp80-deposit-size-law.md`; record
`experiments/exp80_deposit_size_law/README.md` (read its four sections in
order — they are the result).

## Completed

- Stage 0 A/B (`stage0_deconvolve.py`, 1 min): exp63's NNLS generalised to
  every epoch on the merged grid, applied to the data AND the baseline's own
  curves, with the baseline's true deposits read analytically. The two
  modes are partly the operator's (a continuum reads as two spikes). The
  data's compact mode is a fixed physical size at every halo mass (7.0 →
  4.1 kpc, z = 0.4 → 2); the baseline's grows with halo mass at z ≥ 1 (top
  tercile 0.06–0.10 dex too large, 9–15 s.e.) = the high-z size failure.
  The data's extended mode grows with halo mass and holds at 0.13 R200c of
  the CURRENT halo; the baseline's falls and shrinks. Shares agree.
- Stage 0 C (`size_law.py`, `stage0_candidates.py`, 8 candidates in
  parallel, 2–10 min each): the plan's gate (> 0.02 dex of z = 2 R50
  recovered) is passed by the re-tune control alone (0.026 dex, at +0.30
  loss and a doubled z = 0.4 centre); g_e, a break, early-fixed-kpc all tune
  back to the control; q_e = 1 empties the centre; q_e ≈ 0.13 with re-tuned
  constants: 15/15 offsets, R50 z = 2 +0.054 → +0.020, centre better at
  every epoch, loss +0.07. Chosen: q_e, 13 parameters.
- Stage 1 (`stage1_fit.py`, 5 starts + 3 continuations, ~1 h each;
  `stage1_eval.py`, 15 min): one new basin at 14.66 with q_e = 0.33–0.44
  (n_c railed at 0.5); the 14.63 basin sheds q_e. Judge: offset gate 12 of
  15 (= baseline), R50 at z = 2 unchanged (+0.054), slope worse (0.40), widths
  narrower, z = 0.4 outskirts −4.7, planes overshoot. NOT ADOPTED (seventh
  loss-vs-gates case). The frozen Stage 0 C point in the same judge: 15/15,
  R50 z = 1.5 / 2 +0.013 / +0.020, widths UP, centres at z ≥ 1.5 halved,
  loss 15.63.
- Docs: README (four sections), `doc/lessons.md` (exp80 section, 9
  lessons), `doc/todo.md` (exp80 section + review), `doc/open_questions.md`
  (C24 annotated, C25 new), `CLAUDE.md` (exp80 taken, exp81 next), the plan
  marked executed. Memory: `deposit-size-law-stage0-verdict.md`,
  `deposit-size-law-stage1-verdict.md`, `deposit-size-distribution-is-bimodal.md`
  qualified.

## Decisions owed (the user)

1. Adopt the frozen point (q_e = 0.127, constants set by the sizes; not a
   fit of the standard objective) as the baseline mean by the gates — or
   first run the frozen-q_e fit (q_e fixed at 0.127, 12 free) under the
   standard objective (~1 h, `stage1_fit.py` needs a `--fix q_e=0.127`
   option: bounds (v, v) as exp74's `INCUMBENT_FROZEN` does), and adopt if
   the sizes survive; if not, a gated q_e sweep (0.05–0.25).
2. Merge `exp80-deposit-size-law` (outputs and figures are gitignored under
   `experiments/exp80_deposit_size_law/{outputs,figures}`; preserve them).
3. Afterwards: re-baseline the v1 stochastic layer on the adopted mean.

## Branch state

- `exp80-deposit-size-law` from master `e2dcd09`; committed at the end of
  this session; NOT merged, NOT pushed unless the user says so.
- Worktrees unchanged (exp79 is Codex's; a `codex_direction_review_20260909`
  worktree also exists). Nothing running.

## Ops notes

- `size_law.predict_law` with `LAW_DEFAULT` == `model2.predict2` bit for bit
  (`python size_law.py` asserts it); q_e ≠ 0 evaluates the extended kernel
  per epoch (fits take ~55 min per 3000 evaluations, five in parallel).
- `LawProblem.loss` caps the loss at the failure penalty: L-BFGS-B from a
  start with a real gradient otherwise jumps to the box corner (loss 10^24)
  and aborts after 43 evaluations with the loss unchanged.
- The merge's loss-best is the 14.63 basin; judge with `--best cont_tuned`.
- Poll on npz names; a `^wrote` grep matches the figure lines.
- The background wait loop was killed once for low memory (five fits + a
  judge at ~750 MB each); the fits themselves survived.
