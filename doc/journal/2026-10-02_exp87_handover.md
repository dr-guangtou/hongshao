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
