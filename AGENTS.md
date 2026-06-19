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

## Repo layout

```
hongshao/        # LIBRARY: stable, reusable, importable code (tng_data.py, …)
experiments/     # one self-contained folder per experiment (created on demand)
  expNN_slug/
    README.md    # question / method / inputs / key result / decision  (committed)
    run.py       # driver, written as # %% cell script                 (committed)
    figures/     # gitignored
    outputs/     # gitignored (tables, .npz/.fits) + manifest.json
data/
  external/      # vendored third-party inputs (committed, small)
  processed/     # shared derived datasets (gitignored, regenerable)
  raw/           # gitignored; raw drop lives outside the repo (see below)
scripts/         # cross-experiment tools (build dataset, QC)
doc/             # science context, data reference, todo.md, lessons.md
```

- **The experiment is the unit of organization.** Each gets a numbered,
  co-located folder; create it when the experiment starts, not upfront.
- **Library vs. experiment.** Reusable, validated code graduates into
  `hongshao/`; exploratory one-off analysis stays in the experiment folder.
- **Artifacts are regenerable, not committed.** Figures/datasets are gitignored
  and rebuilt from committed code + committed/vendored inputs. No DVC/Snakemake
  until a real shared pipeline needs it. Each `run.py` stamps the git SHA + key
  params into `outputs/manifest.json` for traceability.
- **Records:** per-experiment `README.md` = scientific source of truth;
  `doc/todo.md` = cross-experiment roadmap; the Obsidian journal = chronological
  session log. No overlap.

## Communication

- The user is a professional astronomer, not a software engineer or project
  manager. Explain plans, decisions, and trade-offs in plain language.
- Avoid software/PM jargon and tool names the user wouldn't know (e.g. YAGNI,
  CI/CD, DVC, Snakemake). If a term is genuinely important, define it in one
  plain sentence instead of assuming it.

## Conventions

- English only — in code, comments, commits, and docs.
- Python: `snake_case` everywhere, never camelCase. Use `uv` for dependencies,
  `ruff` for lint/format.
- Exploration as `# %%` cell scripts (`.py`), not committed `.ipynb`.
- Raw data path via the `HONGSHAO_DATA_DIR` env var (defaults to the local drop).
- Never work on `master`/`main` directly — use feature branches; don't merge
  without permission.
- Record mistakes and rationale in `doc/lessons.md`; track the roadmap in
  `doc/todo.md`.
