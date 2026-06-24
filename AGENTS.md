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

## Scope — what HongShao is and isn't

HongShao predicts **central-galaxy stellar masses and profiles from halos** (the
Ultimate SHMR), and nothing downstream of that. In scope: the halo→galaxy map —
features (DiffMAH + `c_200c`), the N-target mean/scatter emulator
(`hongshao/emulator.py`) with generative sampling, the **profile/target layer**
(`hongshao/profile_emulator.py`) that graduates all four prediction modes (kpc
apertures, Re apertures, the cumulative CoG, and the 1-D density profile) through
that one core, and a thin, physically-labeled **deformation layer**
(`hongshao/forward.py`) that lets an external analysis tune the relation. The
deformation layer is the **hand-off boundary**: it outputs (deformed) stellar
masses/profiles for a halo catalog, full stop.

Out of scope (do NOT build here): weak-lensing or clustering predictions,
summary-statistic estimators, likelihoods, or samplers. Those need particle
data / pre-computed catalogs and a separate emulator each, and belong in a
distinct inference repo that *consumes* HongShao's predictions. Keep HongShao a
clean, portable SHMR library — don't let it grow into an inference framework.

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
- **Expect residual scatter; don't over-explain it — but test, don't assume.**
  Do not assume secondary halo properties will reduce it without evidence, and do
  not assume they *won't* either. We expected concentration to be redundant with
  the MAH; exp16 showed otherwise — `c_200c` is only ~25% MAH-determined and adds
  a real, independent +2.7% CRPS even on top of the full MAH-PCA(4) (+5% on the
  portable DiffMAH), and it is itself portable. So: measure each secondary
  property's incremental value (with a shuffle control), don't reason about it
  from "it correlates with formation time." Genuinely independent information
  (concentration, and possibly initial conditions / environment) can help. The stellar profiles are also single random 2-D projections of
  triaxial galaxies, adding noise we cannot remove from this dataset. Goal: a
  phenomenological model capturing the assembly-driven part, not zero scatter.

## Figures

- **Every experiment must produce at least one figure** demonstrating its
  result, whenever feasible. Use publication-quality styling via
  `hongshao.plotting.set_style()` (Okabe-Ito colorblind palette; `cividis` for
  sequential heatmaps; sequential, not diverging, when all values share a sign).
  Save PNG + PDF with `hongshao.plotting.save_fig()`. Follow the
  scientific-visualization skill for presentation.
- **Always surface new figures to the user for review** — point to the file
  paths and display them; don't just save them silently.
- **Always visualize fits and evaluations directly, not just summary metrics.**
  For any model fit or prediction check, produce an intuitive per-object
  visualization (e.g. measured-vs-model curves and residual profiles, by mass
  bin; truth-vs-predicted and residual-vs-truth) in addition to aggregate scores.
  Median RMS / CRPS hide problems that the eye catches instantly — e.g. the
  DiffMAH early-growth coverage issue (exp10) was obvious in the by-mass fit
  figure but invisible in the median RMS. Visual inspection is a first-class
  evaluation step, not an afterthought.

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

- **Primary observable = CoG-derived masses.** Always model the
  aperture/annulus/outskirt stellar masses derived from the 1-D curve of growth
  (`logmstar_cog` / `logmstar_aper`, from X–Y isophote analysis). This is the
  observation-relevant quantity, and the end goal is to reproduce the CoG. The
  direct 2-D `logmstar_aper_proj` masses are physically more accurate (no
  perfect-ellipse assumption, matters during mergers) but are for
  **cross-checks only** — the difference is small. Do not switch the modeling
  target to `*_aper_proj`.
- English only — in code, comments, commits, and docs.
- Python: `snake_case` everywhere, never camelCase. Use `uv` for dependencies,
  `ruff` for lint/format.
- Exploration as `# %%` cell scripts (`.py`), not committed `.ipynb`.
- Raw data path via the `HONGSHAO_DATA_DIR` env var (defaults to the local drop).
- Never work on `master`/`main` directly — use feature branches; don't merge
  without permission.
- Record mistakes and rationale in `doc/lessons.md`; track the roadmap in
  `doc/todo.md`.
