# Session Handover — 2026-10-02 (exp87)

## Goal
An independent side study the user asked for: reverse the direction of
prediction. Knowing the stellar curve of growth (stellar mass included),
how well can halo mass, concentration and the accretion history be
predicted; a ladder of methods from linear regression to symbolic
regression; a scoring metric that is rigorous under the sample's halo-mass
selection; run in autopilot (`doc/plans/2026-10-02-exp87-inverse-halo.md`,
approved 2026-10-02).

## Completed This Session
- `experiments/exp87_inverse_halo/`: config, data (the complete parent at
  z = 0.4; the curated five-epoch sample with input-only cuts; the frozen
  lockbox and folds; the box prior), scoring (truncated-support predictives
  and proper scores), heads (truncated likelihood), methods (L0–L3 and the
  augmented methods), generative (L5), harness, synthetic (the mechanics
  gate), stage0/2/3 scripts, report, sr + stage4_driver (PySR), mah,
  lockbox, figures, README.
- All seven stages run; every decision in `outputs/decisions.jsonl`.
- **Results** (README, plain-language section first; six figures in
  `figures/qa/`): halo mass at z = 0.4 on the complete parent — stellar
  mass 0.0875, the outer shell alone 0.0784, mass + outer shell 0.0763,
  24 shell masses 0.0744 dex CRPS (RMSE 0.137); the truncated likelihood
  worth 11–17%; the relation linear in log shell masses, no method or
  symbolic formula beats it; concentration / formation time / history weak
  from the profile alone (6–10%), strong with the true halo mass (21–26%);
  the lockbox agrees (1 of 76 outside, 0.8 expected).
- Eight lessons, open question C30, todo with review, CLAUDE.md ids.

## In Progress (Not Finished)
- Nothing running. Branch committed, NOT merged (the user decides).

## Problems / Blockers
- C30: what the outer shell is (smooth light or satellites), whether it
  survives observational depth, where the information saturates beyond
  148 kpc, the as-is progenitors' residual curvature, how good an external
  mass must be for the oracle result to matter.
- The parent = box identity bears on the Paper-1 sample-provenance
  question; exp87 did not edit `doc/paper1_statistical/`.
- `master` is 7 commits ahead of `origin/master` (Codex record merges from
  2026-09-28, not pushed); exp87 branched from that master.

## Key Decisions
- The user's: headline on the complete parent at z = 0.4 (an exception to
  the fitting-sample rule, scoped to exp87); two labelled high-z
  populations; full autopilot with the oracle extension.
- On the way (all recorded): L2/L3 methods that nest the linear model
  added after the first ladder; the shell-mass feature sets added after
  the radius reading; symbolic regression escalated per cell where N was
  significant (exploratory beyond the z = 0.4 entry rule) and a shell-aware
  pass replacing two raw-baseline cells; the lockbox pass rule fixed before
  scoring. No gate or threshold was changed after a result.

## Branch State
- `exp87-inverse-halo` from master `ba7d8ab`; committed; not merged; not
  pushed. Gitignored outputs in `experiments/exp87_inverse_halo/{outputs,figures}/`.
- exp88 is the next free id (check `git worktree list`, `git branch -a`,
  `ls experiments`).

## Files Modified This Session
- `experiments/exp87_inverse_halo/*` (new), `doc/plans/2026-10-02-exp87-inverse-halo.md`
  (new), `doc/todo.md`, `doc/lessons.md`, `doc/open_questions.md`,
  `CLAUDE.md`, this file. Nothing outside exp87 was changed in code.

## Addendum (same day): Stage 7, the cross-epoch reading
The user pointed out that the first report covered only same-epoch
predictions at high redshift. `stage7_cross_epoch.py` (+ `--lag`): the
z = 0.4 profile predicts the progenitor's halo mass at z = 0.7 better than
the z = 0.7 profile (0.0644 vs 0.0737 dex CRPS), beats the true z = 0.4
halo mass as a predictor at every earlier epoch, and its skill peaks
2.5 Gyr before z = 0.4 (0.66 vs 0.59): the stars lag the halo. README
section "Across epochs", figure `exp87_cross_epoch`, one lesson, C30(g).
Development folds only; the lockbox was not touched again.

## Addendum 2 (same day): Stage 8, exploratory symbolic regression across epochs
The user asked for a further symbolic-regression pass with the robustness
rule relaxed: the z = 0.4 stellar mass distribution to the halo mass at
each of the five epochs, three best formulas per epoch, the distribution
written (1) as a central aperture plus annuli and (2) as the parameters of
a fitted curve of growth; longer runs and richer forms allowed. Plan
`doc/plans/2026-10-02-exp87-stage8-exploratory-sr.md` (with the changes
made on the way). Code `cogparams.py`, `stage8_explore.py`
(`bench` / `search` / `annulus` / `report`), figures `exp87_explore` and
`exp87_cog_families`; README section "Exploratory formulas across epochs"
holds the formulas. Outputs in `outputs/stage8/`, `outputs/stage8_report.log`,
`outputs/stage8_explore.json`.

- 35 searches, 900 s on four threads each. The best formulas match the
  linear + quadratic model on the same inputs and never beat the 24-shell
  reference (CRPS refit, best formula on the annuli: 0.0745 / 0.0644 /
  0.0721 / 0.1051 / 0.1298 at z = 0.4 / 0.7 / 1.0 / 1.5 / 2.0; on a fitted
  curve 0.0793 / 0.0671 / 0.0744 / 0.1054 / 0.1304). A 3600 s search
  changes nothing held out.
- The annuli beat the fitted curves at z <= 1; a single Sersic fit is
  barely better than total stellar mass at z = 0.4.
- The best single annulus moves inward with the epoch asked about:
  132-148 kpc (z = 0.4, 0.7), 52-80 kpc (z = 1.0, 1.5), inside 5 kpc
  (z = 2). C30(h).
- Ops: PySR's sympy export crashed two jobs and hung one after the search
  had finished; the export is now skipped and the hall-of-fame file read
  directly. Nothing is running. Development folds only; the lockbox was not
  touched. Committed on `exp87-inverse-halo`, not merged.
