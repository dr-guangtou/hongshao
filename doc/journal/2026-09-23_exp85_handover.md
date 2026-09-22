# Session Handover — 2026-09-23 (exp85)

## Goal
The centre's mechanism (roadmap §8 "next", exp83's open question): an
age-driven expansion of the compact channel, probed frozen on the adopted
mean (`doc/plans/2026-09-23-exp85-compact-expansion.md`; the user asked to
proceed on the existing plan at the start of the session).

## Completed This Session
- Engine: `size_law` knobs `q_c` (s_c × (t_obs / t')^q_c) and `q_ch`
  (× (R200c(t_obs) / R200c(t'))^q_ch), nesting at zero; the compact kernel
  per epoch when on; exp84's `size_dev` composes; `selfcheck` +
  `selfcheck_compact_expansion` pass; bounds in `stage0_candidates`.
- `experiments/exp85_compact_expansion/`: `stage0_probe.py` (the frozen
  probe, (log_f_c, b_c) re-tuned to the adopted concentration at z = 0.4
  and z = 2 by a damped Newton, a0 re-centred; nine points),
  `stage0_anatomy.py` (channels and deposit ages inside 4.9 kpc by the
  decline), `stage0_figures.py` (`figures/qa/exp85_probe_summary`),
  `queue.sh` / `judge.sh` (prepared, unused), `README.md`.
- **Verdict (README)**: gate failed at every point — the decliner /
  non-decliner split of the central change 0.054–0.066 vs the adopted
  0.063 and the truth's 0.227; the response has the wrong sign (−0.42)
  because the decliners' centres are 73 per cent extended-channel mass
  (compact share 0.27 vs 0.33). Both forms identical. **Stage 1 not run;
  not adopted; the adopted mean stands.** Three lessons, C28, todo review,
  plan status, roadmap §9.

## In Progress (Not Finished)
- Nothing running. Branch committed, NOT merged (the user decides).

## Problems / Blockers
- C28: the model's in-situ share of the centre (27–33 per cent at 4.9
  kpc) has never been checked against TNG's; if it is too low, the channel
  split is the defect and a compact expansion would regain its leverage;
  if it is right, the decline lives in the early-accreted deposits and a
  mechanism must act on the extended channel by deposit age.
- The 4.9 kpc gate cannot see a 2.2 kpc deposit expand by ×1.5; a 2 kpc
  reading of the decline split is owed next to it.

## Key Decisions
- Fit only on a passing probe (the exp83 rule) — no fit was launched.
- The re-tune is the probe's control (holds the population's concentration
  at two epochs); a railed re-tune is a failed probe, not a pass.

## Branch State
- `exp85-compact-expansion` from master `aae792d`; committed; not merged;
  not pushed. Gitignored outputs in `experiments/exp85_compact_expansion/{outputs,figures}/`.
- exp86 is the next free id (check `git worktree list`, `git branch -a`,
  `ls experiments`); exp79 is Codex's.

## Files Modified This Session
- `experiments/exp80_deposit_size_law/{size_law.py,stage0_candidates.py}`,
  `experiments/exp85_compact_expansion/*` (new), `doc/plans/2026-09-23-exp85-compact-expansion.md`
  (new), `doc/plans/2026-09-12-roadmap-evidence-rethink.md` (§9),
  `doc/todo.md`, `doc/lessons.md`, `doc/open_questions.md`, `CLAUDE.md`, this file.
