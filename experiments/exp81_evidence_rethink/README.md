# exp81 — the evidence-based rethink (2026-09-12)

The user's ask: before more size-law tests, measure where the baseline
fails and why, review the record, and produce a roadmap
(`doc/plans/2026-09-12-roadmap-evidence-rethink.md` — the deliverable; this
README is the record of the measurements).

## Measurements (all on the fitting sample, the adopted baseline on the measured history)

1. `residual_budget.py` → `outputs/residual_budget.{log,npz}`: the error
   budget of log10(model/truth) per epoch and radius (systematic vs
   per-galaxy), the halo-predictability ceiling of the residual (5-fold
   out-of-fold gradient boosting from the measured history, past-only and
   full), and the centre split by decline, for the baseline, exp80's frozen
   q_e point and the loss's q_e basin.
2. `residual_features.py` → `outputs/residual_features.{log,npz}`: which
   part of the history carries the unused information — partial Spearman
   at fixed halo mass with the early mass, recent growth, growth rate and
   t50; restricted-feature R².
4. `residual_shape.py` → `outputs/residual_shape.log`: the functional form
   (binned medians and slopes at fixed halo mass) of the residual against
   the early-mass fraction and the recent growth.
3. `delay_probe.py` → `outputs/delay_probe.{log,npz}`: exp63's deposition
   delay tau_d applied to the baseline at frozen theta.

Findings are written up in the roadmap §1 (E1–E7).

## Controlled fits (exp80's scripts with `--fix` and `--delay`, outputs under exp80's `outputs/`)

- q_e fixed at 0.08 / 0.127 / 0.20 (12 free): `stage1_fit_fix-q_e*_start_tuned.npz`.
- the delay alone (13), the delay with q_e fixed at 0.127 (13 free of 14),
  the delay with q_e free (14): `stage1_fit_noknob_delay0.15_*`,
  `stage1_fit_delay0.15_fix-q_e0.127_*`, `stage1_fit_delay0.15_*`.
- Judges: `stage1_eval.py --fit-tag TAG [--delay] --best tuned`.

## The record survey

A subagent's survey of open questions, roads not taken, standing failures
and the SPEC is summarised in the roadmap's E8; the full text is in the
session transcript (2026-09-12).
