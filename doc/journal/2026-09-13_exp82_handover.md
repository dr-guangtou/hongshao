# Handover — 2026-09-13 (exp82: the delay + expansion exponent, gate-chosen; adoption owed)

## Goal
exp82 (roadmap S1 + S2): fit the deposition delay and the expansion
exponent together under the standard objective, tau_d held on a grid and
chosen by the gates, the full protocol; adopt by the gates.

## Completed
- Twelve grid fits (tau_d 0.10 / 0.15 / 0.20 / 0.30 × starts tuned /
  baseline / far; q_e free; step arrival), judged per grid point with every
  basin named; the full judge with figures on the chosen model; the
  residual check and decliner split. Record:
  `experiments/exp82_delay_expansion/README.md` (grid table, the decision,
  the checks, the verdict). Roadmap §6, todo, lessons, C26 updated.
- **The gate-chosen mean: tau_d = 0.15, the tuned-start basin** —
  `experiments/exp82_delay_expansion/outputs/stage1_fit_delay0.15_fix-tau_d0.15_start_tuned.npz`
  (14 in the model, 13 fitted): loss 12.70 (baseline 15.56), 14/15 offsets
  + 1 width, centre +0.5 / −6.2 / −6.9, R50 z = 1.5 / 2 +0.016 / +0.027,
  z = 2 slope 0.20, formation-time residual gone. Costs: mh-complete
  50–100 kpc shell at z ≥ 1.5 −12 / −18 per cent; leak gate +0.03 at
  z = 0.7 / 2; the z = 0.4 early-mass residual (exp83's target).
- Round 3 (the exponential arrival held at 0.15, three starts) was running
  at the end of the session; its judge line is appended to the README when
  done (`judge.sh 0.15 --delay-exp --best tuned --also baseline,far`).

## Decisions owed (the user)
1. Adopt the tau_d = 0.15 (step) model — or the exponential form if its
   judge is at least as good — as the baseline mean: re-point
   `exp74/rebaseline.py::adopted_baseline()` to the exp82 file with
   `size_law.predict_law` as the engine (a code change: the baseline's
   engine becomes predict_law with a delay spec), update the CLAUDE.md
   rule; merge `exp82-delay-expansion`; push.
2. exp83 (S3): the early-mass conditioning of the centre, read from the
   adopted model's residual (`exp81/residual_check_model.py`); frozen
   probe first. exp84 (S4): the layer on the adopted mean (Codex
   candidate). S6 items for Codex.

## Ops (all in doc/lessons.md)
- Never more than TWO of these fits at once on this machine (11.5 GB each;
  twelve froze it twice). `queue.sh MAXJOBS` runs a grid gently;
  `launch.py` gives a job its own session (nohup alone dies with the
  Claude Code process).
- `judge.sh TAU [--delay-exp] [--figures] [--best NAME] [--also NAMES]`;
  the fit files are named with tau_d formatted %g (0.10 → 0.1).
- Branch `exp82-delay-expansion` from master `ed28a45`; committed; NOT
  merged, NOT pushed.
