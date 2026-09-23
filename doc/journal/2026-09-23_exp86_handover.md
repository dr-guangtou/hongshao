# Session Handover — 2026-09-23 (exp86)

## Goal
Where the delayed mass lands (C26(b); exp82's unrun arrival-radius check;
chosen by the user from the 2026-09-23 review of the record): the delayed
extended deposits sized by the halo at arrival, probed frozen on the
adopted mean (`doc/plans/2026-09-23-exp86-arrival-radius.md`).

## Completed This Session
- Engine: `size_law` knob `w_arr` (`arrival_time`, `arrival_references`,
  `sizes_at_nodes(arrival=)`), nesting at zero; `selfcheck_arrival` passes;
  bounds in `stage0_candidates`.
- `experiments/exp86_arrival_radius/`: `stage0_probe.py` (the shell's
  anatomy by channel / sample / arrival factor, then the probe with
  (log_f_e, b_e) re-tuned on the radius term against the re-tuned
  control), `stage0_figures.py` (`figures/qa/exp86_probe_summary`),
  `queue.sh` / `judge.sh` (prepared, unused), `README.md`.
- **Verdict (README)**: gate failed at every strength; the complete
  progenitors' shell −12 / −19 per cent and the fitting sample's +5 do not
  move (the re-tune absorbs the enlargement); the arrival factor differs
  between the samples by 0.013 dex against a 0.20 dex gap. **Stage 1 not
  run; not adopted; C26(b) closed — the split is the selection (C18).**
  New lead C29 (the outskirts under-respond to the halo's growth rate;
  exp76's g on the adopted mean). Three lessons, roadmap §10, todo review.

## In Progress (Not Finished)
- Nothing running. Branch committed, NOT merged (the user decides).

## Problems / Blockers
- The z ≥ 1.5 shell cost of the adopted mean is now understood as a
  selection cost; no size-law change should be aimed at it again.
- C29 is untested: a growth-rate term on the extended deposit (exp76's g
  frozen-probed on the adopted mean, gate-chosen) is the one lead the
  anatomy left.

## Key Decisions
- Anatomy before probe (exp85's rule) — the expected null was stated
  before the grid; the probe confirmed it.
- Fit only on a passing probe — no fit launched.

## Branch State
- `exp86-arrival-radius` from master `7215df2`; committed; not merged; not
  pushed. Gitignored outputs in `experiments/exp86_arrival_radius/{outputs,figures}/`.
- exp87 is the next free id (check `git worktree list`, `git branch -a`,
  `ls experiments`); exp79 is Codex's.

## Files Modified This Session
- `experiments/exp80_deposit_size_law/{size_law.py,stage0_candidates.py}`,
  `experiments/exp86_arrival_radius/*` (new), `doc/plans/2026-09-23-exp86-arrival-radius.md`
  (new), `doc/plans/2026-09-12-roadmap-evidence-rethink.md` (§10),
  `doc/todo.md`, `doc/lessons.md`, `doc/open_questions.md`, `CLAUDE.md`, this file.
