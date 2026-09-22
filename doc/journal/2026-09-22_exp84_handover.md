# Session Handover — 2026-09-22 (exp84)

## Goal
Roadmap S4: the stochastic layer re-baselined on the adopted exp82 mean
with an outer size component (`doc/plans/2026-09-22-exp84-layer-rebaseline.md`,
approved by the user at the start of the session).

## Completed This Session
- Design decisions (the user): components = amplitude + compact size +
  extended size (X3 dropped; decline targets reported, not gated); the size
  deviations drawn PER GALAXY AND PER EPOCH with a fitted cross-epoch
  correlation.
- Engine: `exp80/size_law.py::predict_law(size_dev=)` — per-galaxy,
  per-epoch deviations of the two deposit sizes at the observed epoch's
  evaluation; a zero deviation nests bit for bit (`selfcheck_size_dev`).
- QA: `hongshao/qa.py::evaluate_draws` (tiers 2d and 2e on drawn
  populations, the same code path as a mean), `print_draw_gate`,
  `print_draw_cdfs`; demo checks identity, permutation, too-tight.
- `experiments/exp84_layer_rebaseline/`: `predictor.py` (the adopted mean
  on the measured history; one call 1.2–2.4 s, 11 GB from the build),
  `stage0_targets.py`, `stage1_anatomy.py` (smallest-sufficient rule),
  `sampler.py`, `stage2_judge.py` (held out both ways; variants gauss /
  gauss-scaled / gauss-2scale / gauss-centred / gauss-2sc-cent / rows /
  independent / persistent; `--variants`), `stage3_adopt.py`, `README.md`.
- **Verdict** (README): the two-scale layer — the compact axis calibrated
  on R20's width goes to scale 0.05 (off), the extended on R80's stays 1.0
  — passes offsets 15/15 at both conditionings, widths 12/15 (M*) and 15/15
  (Mh) from the mean's 1/15 and 0/15 (v1 3/15); S5 exact; S4 +0.003;
  persistence 0.67/0.61/0.57 vs the truth's 0.68/0.71/0.59 (identity 0.19,
  persistent 0.95). S3 0.012 (z ≤ 1) / 0.031 (all) against 0.010; centring
  at two radii is harmful (0.078, offsets 9/15). Logged as C27.
- Artifact `outputs/hongshao_v2_layer.npz` (gauss-2scale, full-sample
  calibration), figures `figures/qa/qa_*_exp84_v2_layer.*`.
- Five lessons in `doc/lessons.md`, `doc/open_questions.md` C27,
  `doc/todo.md` with review, CLAUDE.md id list (exp84 taken, exp85 next).

## In Progress (Not Finished)
- Nothing running at handover (Stage 3's tables complete; see its log).
- Not merged, not pushed: `exp84-layer-rebaseline` awaits the user's
  adoption decision.

## Problems / Blockers
- C27: the median-profile shift of a symmetric log-size draw (S3); the
  candidate levers (a profile-symmetric draw; a narrower extended width at
  z ≥ 1.5) untried. R20/R50 at z ≤ 1 over-dispersed 1.20 at fixed M* with
  the compact draw already off.
- The decline statistics (S1) are v1's walls unchanged: an exchangeable
  draw cannot put the right depth on the right galaxy.
- The centre's mechanism (mass outward in early-formed galaxies z = 2 →
  0.4) remains its own experiment (exp85 candidate).

## Key Decisions
- The draws are CENTRED in the parameter (the anatomy's column medians —
  δ_c +0.14 at z = 0.4 — are NOT applied): a mean-model correction through
  the layer is exp60's wall 2.
- Every axis's scale is calibrated against the gate that measures it
  before its anatomy width is believed (the compact axis's 0.4–0.6 dex was
  a flat valley).

## Branch State
- Branch `exp84-layer-rebaseline` from master `17f2127`; committed; not
  merged; not pushed. Gitignored outputs in
  `experiments/exp84_layer_rebaseline/outputs/` and `figures/qa/*exp84_*`.
- exp79 is Codex's; exp85 is the next free id (check `git worktree list`,
  `git branch -a`, `ls experiments`).

## Files Modified This Session
- `experiments/exp80_deposit_size_law/size_law.py`, `hongshao/qa.py`,
  `experiments/exp84_layer_rebaseline/*` (new), `doc/plans/2026-09-22-exp84-layer-rebaseline.md`
  (new), `doc/todo.md`, `doc/lessons.md`, `doc/open_questions.md`,
  `CLAUDE.md`, this file.
