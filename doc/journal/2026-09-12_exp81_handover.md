# Handover — 2026-09-12 (exp81, the evidence-based rethink: a solution with a clear improvement)

## Goal

The user's ask (2026-09-12 00:20): pause the size-law work, measure where
and why the baseline fails, review the record, and deliver a roadmap of
strategies (motivation, evidence, feasibility, plan); stop on a clear
improvement, the 5-hour limit, or Sep 12 08:00. Stop criterion 1 was met.

## The deliverable

`doc/plans/2026-09-12-roadmap-evidence-rethink.md` — read §0 (the answer
in a paragraph), §1 (evidence E1–E8), §3 (strategies S1–S6 ranked). The
record of the measurements is `experiments/exp81_evidence_rethink/README.md`
(logs under its `outputs/`); the controlled fits live in exp80's `outputs/`
(`stage1_fit_*delay*`, `stage1_eval_delay_*.log`, `stage1_eval_fix_q*.log`).

## What was found (numbers in the roadmap)

- The baseline's error is per-galaxy scatter (systematic ≤ 7 per cent);
  the centre at z ≤ 1 is at the halo-information ceiling (R² ≤ 0.10); the
  outskirts and everything at z ≥ 1.5 have 0.22–0.35 of the residual
  variance predictable from the PAST history.
- The missing information is formation time: the truth holds more stars in
  early-formed haloes at fixed mass; the model deposits stars from accreted
  mass instantly (residual +0.45 dex per dex of recent growth; truth −0.6).
- A deposition delay (exp63's tau_d, rejected there on a z = 0.4-only fit
  of the leaky input) fixes it: fitted jointly with q_e free, loss 12.72
  (baseline 15.56), 14/15 offsets, 2 widths (first ever), centres better
  at every epoch, z = 2 slope 0.20 (0.33 → toward 0.11); q_e settles at
  0.125 by itself. Costs: the mh-complete 50–100 kpc shell at z ≥ 1.5
  −14 to −19 per cent (the cumulative there improves), the leak gate at
  z = 0.7 / 2 −0.04 / −0.03 dex per dex, a new early-mass residual at
  z = 0.4 (+0.38 at 5 kpc) — the next lever (S3).
- Fixing q_e alone (0.08 / 0.127 / 0.20) fails: the loss reorganises the
  twelve into the gate-rejected family (10 of 15).

## Decisions owed (the user)

1. Adopt the delay + q_e model as the baseline mean (by the gates; the
   loss agrees), in its step form (tau_d = 0.15 held) or the exponential
   form (fitted tau_d; see the roadmap S1's last line).
2. Then: S3 (early-mass conditioning of the centre), S4 (the layer's
   re-baseline with an outer size component), S5 (gate-consistent
   selection), S6 (B2, C22, the jumper check).
3. Merge `exp81-evidence-rethink` (it contains the exp80 branch); preserve
   the gitignored outputs.

## Branch and ops state

- Branch `exp81-evidence-rethink` from `exp80-deposit-size-law`; committed;
  NOT merged, NOT pushed.
- exp80's `outputs/stage1_eval.npz` was overwritten by a mis-parsed judge
  run and regenerated at the session's end (`stage1_eval_regen.log`); the
  exp80 log `stage1_eval.log` was never touched.
- New options: `stage1_fit.py --fix NAME=VALUE`, `--delay TAU0`,
  `--delay-exp`, `--start-from FILE`, `--knobs none`; `stage1_eval.py
  --fit-tag TAG --delay [--delay-exp] --best NAME`. model2's step arrival
  has no gradient in tau_d; `size_law.arrival_weights(smooth_delay=True)`
  is the exponential arrival (nests at 0). `judge_all.sh` must be bash.
- Memory: `deposition-delay-is-the-lever.md`; MEMORY.md Current State.
