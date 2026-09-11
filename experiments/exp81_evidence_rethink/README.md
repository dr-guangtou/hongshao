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

## Results of the controlled fits (2026-09-12; the judge logs are `exp80/outputs/stage1_eval_fix_q*.log`, `stage1_eval_delay_*.log`)

**q_e fixed (0.08 / 0.127 / 0.20), 12 free.** Losses 14.79 / 14.71 / 14.69
(baseline 15.56); in every case the twelve go to the 14.63 basin's family
(compact size 18–19 kpc, index railed at 0.5, d_split ≈ 1.0); the offset
gate falls to 10 of 15 (baseline 12); the z = 2 centre −12 per cent. Holding
q_e alone does not hold the sizes — exp80's frozen point held the two size
constants as well. S2 of the roadmap is reachable only through the two-block
selection (S5) or through the delay (below).

**The deposition delay, tau_d held at 0.15 Hubble times (model2's step
arrival has no gradient in tau_d, so the fits could not move it).**

| model | params | loss | offset / width of 15 | R50 z = 1.5 / 2 | slope z = 2 | M(<2) z = 0.4 / 1.5 / 2 | leak z = 0.7 / 1 / 1.5 / 2 [dex per dex] |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| baseline | 12 | 15.56 | 12 / 0 | +0.037 / +0.054 | 0.33 | +5.6 / −9.8 / −12.6 | −0.013 / −0.061 / −0.038 / −0.007 |
| delay alone | 13 | 12.49 | 13 / 0 | +0.045 / +0.046 | 0.33 | +9.5 / −9.3 / −11.6 | −0.042 / −0.048 / −0.028 / −0.029 |
| delay + q_e = 0.127 fixed | 13 | 12.47 | 13 / 0 | +0.045 / +0.046 | 0.32 | +8.6 / −9.0 / −11.9 | −0.042 / −0.049 / −0.029 / −0.031 |
| **delay + q_e free (→ 0.125)** | 14 | 12.72 | **14 / 2** | **+0.018 / +0.030** | **0.20** | **+0.9 / −6.7 / −7.3** | −0.043 / −0.039 / −0.015 / −0.027 |

(the truth's own leak row: −0.005 / +0.061 / +0.033 / +0.004.) The delay is
the first change the loss and the gates both prefer; with it the loss
settles q_e at the gate-chosen value instead of abusing it. Its price is
the mh-complete progenitors' 50–100 kpc shell at z ≥ 1.5 (−14 to −19 per
cent; the cumulative M(<103) there improves, +8.6 → +2.9 at z = 1.5) and a
larger future-growth dependence at z = 0.7 and z = 2 (0.03 dex per dex).

**The exponential arrival** (`size_law.arrival_weights`, smooth in tau_d):
fits from the step solutions — see the roadmap's S1 for the fitted tau_d
(added when they finish).
