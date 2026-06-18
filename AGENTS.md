# AGENTS.md — HongShao

Orientation for AI agents (and humans) working in this repo.

## What this is

HongShao is a **research project**, not a software product. The goal is the
"Ultimate SHMR": an assembly-resolved, profile-level extension of the
stellar–halo mass relation for massive central galaxies. See `README.md` for
the goal and `doc/` for the scientific source of truth:

- `doc/ultimate_shmr_context.md` — background, motivation, references.
- `doc/ultimate_shmr_possible_directions.md` — analysis directions + suggested sequence.

Read those two before proposing analyses.

## How to work here

- **Research mindset.** Expect exploration, false starts, and throwaway code.
  Favor fast, small-scale validation of an idea over polished infrastructure.
  Don't build frameworks for analyses that may not survive the week.
- **Measure, don't guess.** Never estimate numbers (scatter, variance
  explained, fit quality) — benchmark on the real data. Validate on a small
  subsample (sub-minute) before running on all 3388 halos.
- **Probabilistic framing.** Models are `P(theta_prof | M0, theta_MAH)`, not
  deterministic fits. Residual scatter is a result, not a bug.
- **Null models matter.** Compare against final-mass-only and shuffled-MAH
  controls before claiming an assembly signal.

## Conventions

- English only — in code, comments, commits, and docs.
- Python: `snake_case` everywhere, never camelCase. Use `uv` for dependencies,
  `ruff` for lint/format.
- Never work on `master`/`main` directly — use feature branches; don't merge
  without permission.
- Record mistakes and rationale in `doc/lessons.md`; track tasks in
  `doc/todo.md` (create each when first needed).
